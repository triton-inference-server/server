// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/core/ensemble_scheduler.h"

#include <mutex>
#include "src/core/api.pb.h"
#include "src/core/backend.h"
#include "src/core/logging.h"
#include "src/core/provider_utils.h"
#include "src/core/server.h"
#include "src/core/server_status.h"

namespace nvidia { namespace inferenceserver {

namespace {

// Step specifies the backend, providers and status objects used for
// the internal infer request
struct Step {
  Step(size_t step_idx) : step_idx_(step_idx) {}

  std::shared_ptr<InferenceBackend> backend_;
  std::shared_ptr<InferRequestProvider> request_provider_;
  std::shared_ptr<DelegatingInferResponseProvider> response_provider_;
  std::unordered_map<std::string, std::shared_ptr<AllocatedSystemMemory>>
      output_map_;
  Status infer_status_;

  size_t step_idx_;
};

// EnsembleContext maintains the state of the ensemble request
//
// Using static functions to take advantage of shared_ptr, a copy of the
// shared_ptr will be made when a step is scheduled and it will go out of
// scope after the step's callback is finished. The step's callback will
// schedule new steps if available and the last step will finish the ensemble
// request.
// So we don't have to maintian the context in scheduler as the shared_ptr
// will destroy the context for us if there are no "in-flight" steps.
class EnsembleContext {
 public:
  EnsembleContext(
      InferenceServer* is, EnsembleInfo* info,
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(const Status&)> OnComplete);

  // Perform transition on 'context' state given the information of
  // 'completed_step'
  static void Proceed(
      const std::shared_ptr<EnsembleContext>& context,
      const std::shared_ptr<Step>& completed_step = nullptr);

 private:
  static TRTSERVER_Error* ResponseAlloc(
      TRTSERVER_ResponseAllocator* allocator, void** buffer,
      void** buffer_userp, const char* tensor_name, size_t byte_size,
      TRTSERVER_Memory_Type memory_type, int64_t memory_type_id, void* userp);
  static TRTSERVER_Error* ResponseRelease(
      TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
      size_t byte_size, TRTSERVER_Memory_Type memory_type,
      int64_t memory_type_id);

  using StepList = std::vector<std::shared_ptr<Step>>;
  using VersionMap =
      std::unordered_map<int64_t, std::shared_ptr<InferenceBackend>>;
  // Storing each tensor's meta data in 1st element, batch size in 2nd
  // (0 for non-batchable), and the raw data in 3rd.
  using TensorData = std::tuple<
      InferRequestHeader::Input, size_t, std::shared_ptr<SystemMemory>>;

  // Return the list of step that becomes ready due to tensor update
  // from 'completed_step'
  Status PrepareSteps(
      const std::shared_ptr<Step>& completed_step, StepList& steps);

  // Prepare infer stats and call the inference server's function to process
  // the infer requests specified in 'steps'
  static void ScheduleSteps(
      const std::shared_ptr<EnsembleContext>& context, const StepList& steps);

  // Helper function that updates ensemble state given 'completed_step' and
  // returns the list of updated tensors in 'updated_tensors'
  Status UpdateEnsembleState(
      const std::shared_ptr<Step>& completed_step,
      std::vector<std::string>& updated_tensors);

  // Helper function that returns a list of 'steps' that should be run under
  // current ensemble state. 'updated_tensors' is used so that we don't need to
  // iterate all the tensors to determine which step can be run.
  Status GetNextSteps(
      const std::vector<std::string>& updated_tensors, StepList& steps);

  // Helper function that completes the response of the ensemble request
  Status FinishEnsemble();

  // Helper function that initialize the 'step' given the info at 'step_idx'.
  // The 'step' will have proper request / response provider for the model
  Status InitStep(size_t step_idx, std::shared_ptr<Step>* step);

  // Helper function that set the output of the ensemble request if it is ready
  // and valid.
  // Return error if some of the required outputs are not set (deadlock)
  Status CheckAndSetEnsembleOutput();

  // Helper function to reshape the given tensor according to the
  // config shape and batching info and its actual shape and batching info.
  // Returns the batch size to be used after the reshape.
  size_t ReshapeTensorDims(
      const DimsList& config_dims, const bool allow_batching,
      const size_t tensor_batch_size, DimsList* mutable_dims);

  InferenceServer* is_;

  EnsembleInfo* info_;

  // Mutex to avoid concurrent call on 'PrepareSteps' where ensemble state
  // are being modified
  std::mutex mutex_;

  size_t inflight_step_counter_;

  std::unordered_map<std::string, TensorData> tensor_data_;

  // Handle to all backend that may be used in the ensemble
  std::unordered_map<std::string, VersionMap> handles_;

  // Request specific information that obtained from ensemble request and
  // should be applied to all internal requests
  uint32_t flags_;
  uint64_t correlation_id_;
  uint32_t batch_size_;

  // Objects related to the ensemble infer request
  Status ensemble_status_;
  std::shared_ptr<ModelInferStats> stats_;
  std::shared_ptr<InferRequestProvider> request_provider_;
  std::shared_ptr<InferResponseProvider> response_provider_;
  std::function<void(const Status&)> OnComplete_;

  // Output tensors whose labels are not provided by the ensemble
  std::set<std::string> no_label_tensors_;

  // The allocator that will be used to allocate buffers for the
  // inference result tensors.
  std::unique_ptr<
      TRTSERVER_ResponseAllocator, decltype(&TRTSERVER_ResponseAllocatorDelete)>
      allocator_;
};

EnsembleContext::EnsembleContext(
    InferenceServer* is, EnsembleInfo* info,
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(const Status&)> OnComplete)
    : is_(is), info_(info), inflight_step_counter_(0), stats_(stats),
      request_provider_(request_provider),
      response_provider_(response_provider), OnComplete_(OnComplete),
      allocator_(nullptr, TRTSERVER_ResponseAllocatorDelete)
{
  // Obtain backend handles of all models in ensemble request such that
  // they have the same lifetime as the ensemble request to avoid unloading
  // while the ensemble is executing.
  for (const auto& step_info : info_->steps_) {
    auto it = handles_.find(step_info.model_name_);
    if (it == handles_.end()) {
      it = handles_.emplace(std::make_pair(step_info.model_name_, VersionMap()))
               .first;
    }
    auto ver_it = it->second.find(step_info.model_version_);
    if (ver_it == it->second.end()) {
      std::shared_ptr<InferenceBackend> backend = nullptr;
      ensemble_status_ = is_->GetInferenceBackend(
          step_info.model_name_, step_info.model_version_, &backend);
      if (!ensemble_status_.IsOk()) {
        break;
      }

      it->second.emplace(std::make_pair(step_info.model_version_, backend));
    }
  }

  for (const auto& pair : info_->tensor_to_step_) {
    tensor_data_.emplace(pair.first, TensorData());
  }

  if (ensemble_status_.IsOk()) {
    const auto& request_header = request_provider_->RequestHeader();

    batch_size_ = request_header.batch_size();
    correlation_id_ = request_header.correlation_id();
    flags_ = request_header.flags();

    for (const auto& input : request_header.input()) {
      auto it = tensor_data_.find(input.name());
      if (it != tensor_data_.end()) {
        auto& tensor_data = it->second;
        std::get<0>(tensor_data) = input;
        std::get<1>(tensor_data) = (info_->allow_batching_ ? batch_size_ : 0);
        request_provider_->GetSystemMemory(
            it->first, &(std::get<2>(tensor_data)));
      } else {
        ensemble_status_ = Status(
            RequestStatusCode::INVALID_ARG,
            "unexpected input '" + input.name() +
                "' in request header that does not map to any ensemble inputs");
      }
    }
  }

  if (ensemble_status_.IsOk()) {
    const std::shared_ptr<LabelProvider>& label_provider =
        response_provider_->GetLabelProvider();
    for (const auto& pair : info_->ensemble_output_shape_) {
      const auto& label = label_provider->GetLabel(pair.first, 0);
      if (label == "") {
        no_label_tensors_.emplace(pair.first);
      }
    }
  }

  TRTSERVER_ResponseAllocator* allocator;
  TRTSERVER_Error* err = TRTSERVER_ResponseAllocatorNew(
      &allocator, ResponseAlloc, ResponseRelease);
  if (err != nullptr) {
    ensemble_status_ = Status(
        TrtServerCodeToRequestStatus(TRTSERVER_ErrorCode(err)),
        TRTSERVER_ErrorMessage(err));
    TRTSERVER_ErrorDelete(err);
  } else {
    allocator_.reset(allocator);
  }
}

TRTSERVER_Error*
EnsembleContext::ResponseAlloc(
    TRTSERVER_ResponseAllocator* allocator, void** buffer, void** buffer_userp,
    const char* tensor_name, size_t byte_size,
    TRTSERVER_Memory_Type memory_type, int64_t memory_type_id, void* userp)
{
  auto tensor_data_map = reinterpret_cast<
      std::unordered_map<std::string, std::shared_ptr<AllocatedSystemMemory>>*>(
      userp);

  *buffer = nullptr;
  *buffer_userp = nullptr;

  // [TODO] Response provider should be able to take hints (i.e. CPU / GPU
  // memory) from alloc caller (i.e. backend). Can't allocate for any memory
  // type other than CPU.
  if (memory_type == TRTSERVER_MEMORY_GPU) {
    // [TODO] support it with cudaMalloc
    LOG_VERBOSE(1) << "Internal response allocation not supported for type "
                   << memory_type << " for " << tensor_name;
    return nullptr;  // Success
  } else {
    auto it =
        tensor_data_map
            ->emplace(
                tensor_name, std::make_shared<AllocatedSystemMemory>(byte_size))
            .first;
    if (byte_size > 0) {
      *buffer = static_cast<void*>(it->second->MutableBuffer());
    }
  }

  LOG_VERBOSE(1) << "Internal response allocation: " << tensor_name << ", size "
                 << byte_size << ", addr " << *buffer;

  return nullptr;  // Success
}

TRTSERVER_Error*
EnsembleContext::ResponseRelease(
    TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRTSERVER_Memory_Type memory_type, int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "Internal response release: "
                 << "size " << byte_size << ", addr " << buffer;

  // Don't do anything when releasing a buffer since ResponseAlloc
  // passes the ownership of the data to ensemble context.
  return nullptr;  // Success
}

void
EnsembleContext::Proceed(
    const std::shared_ptr<EnsembleContext>& context,
    const std::shared_ptr<Step>& completed_step)
{
  StepList ready_steps;
  Status status = context->PrepareSteps(completed_step, ready_steps);
  if (status.IsOk()) {
    ScheduleSteps(context, ready_steps);
  }
}

Status
EnsembleContext::PrepareSteps(
    const std::shared_ptr<Step>& completed_step, StepList& ready_steps)
{
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Initialization error, ensemble status will be not ok since the beginning
    if (completed_step == nullptr && !ensemble_status_.IsOk()) {
      ensemble_status_ = FinishEnsemble();
    }

    if (ensemble_status_.IsOk()) {
      StepList res;
      std::vector<std::string> updated_tensors;
      ensemble_status_ = UpdateEnsembleState(completed_step, updated_tensors);
      if (ensemble_status_.IsOk()) {
        ensemble_status_ = GetNextSteps(updated_tensors, res);
      }
      // Error or no more progress (completed or deadlock)
      // in either case, FinishEnsemble() won't be called again
      if ((!ensemble_status_.IsOk()) || (inflight_step_counter_ == 0)) {
        ensemble_status_ = FinishEnsemble();
      } else {
        ready_steps.swap(res);
      }
    }
    return ensemble_status_;
  }
}

Status
EnsembleContext::UpdateEnsembleState(
    const std::shared_ptr<Step>& completed_step,
    std::vector<std::string>& updated_tensors)
{
  updated_tensors.clear();
  if (completed_step == nullptr) {
    for (const auto& pair : tensor_data_) {
      if (std::get<2>(pair.second) != nullptr) {
        updated_tensors.push_back(pair.first);
      }
    }
  } else {
    inflight_step_counter_--;
    RETURN_IF_ERROR(completed_step->infer_status_);

    auto step_idx = completed_step->step_idx_;
    RETURN_IF_ERROR(completed_step->response_provider_->FinalizeResponse(
        *(completed_step->backend_)));
    const auto& response_header =
        completed_step->response_provider_->ResponseHeader();
    const bool allow_batching =
        (completed_step->backend_->Config().max_batch_size() > 0);
    const size_t batch_size =
        (allow_batching ? response_header.batch_size() : 0);
    for (const auto& output : response_header.output()) {
      if (output.has_raw()) {
        auto it = info_->steps_[step_idx].output_to_tensor_.find(output.name());
        if (it != info_->steps_[step_idx].output_to_tensor_.end()) {
          auto& tensor_data = tensor_data_[it->second];
          auto& meta_data = std::get<0>(tensor_data);
          *(meta_data.mutable_dims()) = output.raw().dims();
          meta_data.set_batch_byte_size(output.raw().batch_byte_size());

          std::get<1>(tensor_data) = batch_size;

          std::get<2>(tensor_data) =
              std::move(completed_step->output_map_[it->first]);
          updated_tensors.push_back(it->second);

          auto tensor_it = no_label_tensors_.find(it->second);
          if (tensor_it != no_label_tensors_.end()) {
            // Check the inner model's lookup map first in case it is also an
            // ensemble model. In that case, the label of the inner model may
            // come from another model.
            InferResponseProvider::SecondaryLabelProvider provider;
            if (completed_step->response_provider_->GetSecondaryLabelProvider(
                    it->first, &provider)) {
              response_provider_->SetSecondaryLabelProvider(
                  *tensor_it, provider);
            } else {
              const std::shared_ptr<LabelProvider>& label_provider =
                  completed_step->response_provider_->GetLabelProvider();
              response_provider_->SetSecondaryLabelProvider(
                  *tensor_it, std::make_pair(it->first, label_provider));
            }
            no_label_tensors_.erase(tensor_it);
          }
        } else {
          return Status(
              RequestStatusCode::INTERNAL,
              "internal response header specified output '" + output.name() +
                  "' that does not map to any ensemble tensors");
        }
      } else {
        return Status(
            RequestStatusCode::INTERNAL,
            "internal response header should return output '" + output.name() +
                "' as raw data instead of classification result");
      }
    }
  }
  return Status::Success;
}

Status
EnsembleContext::GetNextSteps(
    const std::vector<std::string>& updated_tensors, StepList& steps)
{
  steps.clear();

  std::set<size_t> next_step_idx;
  // Get steps whose tensors used for input are set
  for (const auto tensor_name : updated_tensors) {
    const auto& step_idx = info_->tensor_to_step_[tensor_name];
    for (const auto& idx : step_idx) {
      bool ready = true;
      for (const auto& input_pair : info_->steps_[idx].input_to_tensor_) {
        if (std::get<2>(tensor_data_[input_pair.second]) == nullptr) {
          ready = false;
          break;
        }
      }
      if (ready) {
        next_step_idx.insert(idx);
      }
    }
  }

  for (const auto& idx : next_step_idx) {
    steps.emplace_back();
    RETURN_IF_ERROR(InitStep(idx, &(steps.back())));
  }
  inflight_step_counter_ += steps.size();

  return Status::Success;
}

Status
EnsembleContext::InitStep(size_t step_idx, std::shared_ptr<Step>* step)
{
  std::unordered_map<std::string, std::shared_ptr<SystemMemory>> input_map;
  InferRequestHeader request_header;
  auto& version_map = handles_[info_->steps_[step_idx].model_name_];
  auto& backend = version_map[info_->steps_[step_idx].model_version_];

  const bool allow_batching = (backend->Config().max_batch_size() > 0);
  size_t batch_size = (allow_batching ? batch_size_ : 0);

  // Set inputs in request header and prepare input map
  for (const auto& pair : info_->steps_[step_idx].input_to_tensor_) {
    auto input = request_header.add_input();
    *input = std::get<0>(tensor_data_[pair.second]);
    input->set_name(pair.first);

    // If the actual shape and config shape agree with each other without
    // considering batch size, non-batch / batch conversion are not required
    const ModelInput* input_config;
    backend->GetInput(pair.first, &input_config);
    batch_size = ReshapeTensorDims(
        input_config->dims(), allow_batching,
        std::get<1>(tensor_data_[pair.second]), input->mutable_dims());

    input_map[pair.first] = std::get<2>(tensor_data_[pair.second]);
  }

  // Set requested outputs in request header
  for (const auto& pair : info_->steps_[step_idx].output_to_tensor_) {
    request_header.add_output()->set_name(pair.first);
  }

  request_header.set_correlation_id(correlation_id_);
  request_header.set_flags(flags_);
  request_header.set_batch_size((batch_size == 0 ? 1 : batch_size));
  RETURN_IF_ERROR(NormalizeRequestHeader(*backend, request_header));

  step->reset(new Step(step_idx));
  (*step)->backend_ = backend;
  RETURN_IF_ERROR(InferRequestProvider::Create(
      info_->steps_[step_idx].model_name_,
      info_->steps_[step_idx].model_version_, request_header, input_map,
      &((*step)->request_provider_)));
  // Request header is stored in response provider as reference, so use
  // header from request provider as the providers have same lifetime
  RETURN_IF_ERROR(DelegatingInferResponseProvider::Create(
      (*step)->request_provider_->RequestHeader(),
      (*step)->backend_->GetLabelProvider(), allocator_.get(), ResponseAlloc,
      &((*step)->output_map_), ResponseRelease,
      &((*step)->response_provider_)));

  return Status::Success;
}

size_t
EnsembleContext::ReshapeTensorDims(
    const DimsList& config_dims, const bool allow_batching,
    const size_t tensor_batch_size, DimsList* mutable_dims)
{
  size_t batch_size = tensor_batch_size;
  // If the actual shape and config shape agree with each other without
  // considering batch size, non-batch / batch conversion are not required.
  if (!CompareDimsWithWildcard(*mutable_dims, config_dims)) {
    // Only reshape if one setting is batchable while the other is not,
    // otherwise the shape mismatch can not be recovered with reshape
    // and should have caused error during validation.
    if (allow_batching != (tensor_batch_size != 0)) {
      if (allow_batching) {
        // assume first dim is batch dim and extract it.
        auto bit = mutable_dims->begin();
        batch_size = *bit;
        mutable_dims->erase(bit);
      } else {
        // insert batch size as first dim
        mutable_dims->Add(tensor_batch_size);
        mutable_dims->SwapElements(0, (mutable_dims->size() - 1));
        batch_size = 0;
      }
    }
  }
  return batch_size;
}

Status
EnsembleContext::FinishEnsemble()
{
  stats_->SetModelExecutionCount(1);
  if (ensemble_status_.IsOk()) {
    ensemble_status_ = CheckAndSetEnsembleOutput();
  }
  // Add ensemble name to make error message more trackable
  if (!ensemble_status_.IsOk()) {
    ensemble_status_ = Status(
        ensemble_status_.Code(), "in ensemble '" + info_->ensemble_name_ +
                                     "', " + ensemble_status_.Message());
  }
  OnComplete_(ensemble_status_);

  // Reset stats_ to make sure the timers are stopped even though
  // there may be other internal requests (i.e. invoke FinishEnsemble()
  // because of failure in one of the internal requests)
  stats_.reset();
  return ensemble_status_;
}

Status
EnsembleContext::CheckAndSetEnsembleOutput()
{
  for (const auto& output_pair : info_->ensemble_output_shape_) {
    if (!response_provider_->RequiresOutput(output_pair.first)) {
      continue;
    }
    // Check if output is ready
    const auto& tensor_data = tensor_data_[output_pair.first];
    const auto& meta_data = std::get<0>(tensor_data);
    const auto& memory_block = std::get<2>(tensor_data);
    if (memory_block == nullptr) {
      return Status(
          RequestStatusCode::INVALID_ARG,
          "unexpected deadlock, output '" + output_pair.first +
              "' is not set while no more ensemble steps can be made");
    } else if (meta_data.batch_byte_size() != memory_block->TotalByteSize()) {
      return Status(
          RequestStatusCode::INTERNAL,
          "unexpected size for output '" + output_pair.first + "', byte-size " +
              std::to_string(meta_data.batch_byte_size()) + " does not equal " +
              std::to_string(memory_block->TotalByteSize()));
    }

    // copy data to ensemble response provider
    size_t expected_byte_size = meta_data.batch_byte_size();
    DimsList output_dims = meta_data.dims();

    ReshapeTensorDims(
        output_pair.second, info_->allow_batching_, std::get<1>(tensor_data),
        &output_dims);

    std::vector<int64_t> shape;
    if (info_->allow_batching_) {
      shape.push_back(batch_size_);
    }
    for (const auto& dim : output_dims) {
      shape.push_back(dim);
    }

    void* buffer;
    RETURN_IF_ERROR(response_provider_->AllocateOutputBuffer(
        output_pair.first, &buffer, expected_byte_size, shape));

    size_t content_offset = 0;
    size_t content_idx = 0;
    size_t content_size;
    const char* content = memory_block->BufferAt(content_idx, &content_size);
    while (content != nullptr) {
      memcpy(((char*)buffer) + content_offset, content, content_size);
      content_offset += content_size;
      content_idx++;
      content = memory_block->BufferAt(content_idx, &content_size);
    }
  }
  return Status::Success;
}

void
EnsembleContext::ScheduleSteps(
    const std::shared_ptr<EnsembleContext>& context, const StepList& steps)
{
  for (const auto& step : steps) {
    auto infer_stats = std::make_shared<ModelInferStats>(
        context->is_->StatusManager(), step->backend_->Name());
    auto timer = std::make_shared<ModelInferStats::ScopedTimer>();
    infer_stats->StartRequestTimer(timer.get());
    infer_stats->SetRequestedVersion(step->backend_->Version());
    infer_stats->SetMetricReporter(step->backend_->MetricReporter());
    infer_stats->SetBatchSize(
        step->request_provider_->RequestHeader().batch_size());
    infer_stats->SetFailed(true);

    context->is_->Infer(
        step->backend_, step->request_provider_, step->response_provider_,
        infer_stats,
        [context, step, infer_stats, timer](const Status& status) mutable {
          infer_stats->SetFailed(!status.IsOk());
          if (!status.IsOk()) {
            LOG_VERBOSE(1) << "Ensemble infer failed: " << status.Message();
          }

          timer.reset();
          // Accumulate the queue and compute durations from this composing
          // model
          context->stats_->IncrementQueueDuration(
              infer_stats->GetQueueDuration());
          context->stats_->IncrementComputeDuration(
              infer_stats->GetComputeDuration());
          infer_stats.reset();
          step->infer_status_ = status;
          Proceed(context, step);
        });
  }
}

}  // namespace

Status
EnsembleScheduler::Create(
    const ModelConfig& config, std::unique_ptr<Scheduler>* scheduler)
{
  scheduler->reset(new EnsembleScheduler(config));
  return Status::Success;
}

void
EnsembleScheduler::Enqueue(
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(const Status&)> OnComplete)
{
  std::shared_ptr<EnsembleContext> context(new EnsembleContext(
      is_, info_.get(), stats, request_provider, response_provider,
      OnComplete));
  EnsembleContext::Proceed(context);
}

EnsembleScheduler::EnsembleScheduler(const ModelConfig& config)
{
  // Set 'info_' based on 'config'
  info_.reset(new EnsembleInfo());

  info_->ensemble_name_ = config.name();
  info_->allow_batching_ = (config.max_batch_size() != 0);

  for (const auto& input : config.input()) {
    info_->tensor_to_step_.emplace(input.name(), std::set<size_t>());
  }
  for (const auto& output : config.output()) {
    info_->tensor_to_step_.emplace(output.name(), std::set<size_t>());

    if (output.has_reshape()) {
      info_->ensemble_output_shape_[output.name()] = output.reshape().shape();
    } else {
      info_->ensemble_output_shape_[output.name()] = output.dims();
    }
  }

  for (const auto& element : config.ensemble_scheduling().step()) {
    size_t step_idx = info_->steps_.size();
    info_->steps_.emplace_back(element.model_name(), element.model_version());
    for (const auto& pair : element.input_map()) {
      auto it = info_->tensor_to_step_.find(pair.second);
      if (it == info_->tensor_to_step_.end()) {
        it = info_->tensor_to_step_.emplace(pair.second, std::set<size_t>())
                 .first;
      }
      it->second.insert(step_idx);
      info_->steps_[step_idx].input_to_tensor_.emplace(
          std::make_pair(pair.first, pair.second));
    }

    for (const auto& pair : element.output_map()) {
      auto it = info_->tensor_to_step_.find(pair.second);
      if (it == info_->tensor_to_step_.end()) {
        it = info_->tensor_to_step_.emplace(pair.second, std::set<size_t>())
                 .first;
      }
      info_->steps_[step_idx].output_to_tensor_.emplace(
          std::make_pair(pair.first, pair.second));
    }
  }
}

}}  // namespace nvidia::inferenceserver
