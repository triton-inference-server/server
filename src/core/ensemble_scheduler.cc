// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifdef TRITON_ENABLE_ENSEMBLE

#include "src/core/ensemble_scheduler.h"

#include <mutex>
#include "src/core/backend.h"
#include "src/core/cuda_utils.h"
#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/server.h"

namespace nvidia { namespace inferenceserver {

namespace {

class EnsembleContext;

using IterationCount = size_t;

// Request tracker is passed as 'userp' in RequestRelease function and used
// to manage the lifecycle of the ensemble request
class RequestTracker {
 public:
  explicit RequestTracker(
      std::unique_ptr<InferenceRequest>&& request, uint64_t compute_start_ns,
      MetricModelReporter* metric_reporter,
      InferenceStatsAggregator* stats_aggregator)
      : inflight_request_counter_(1), request_(std::move(request)),
        compute_start_ns_(compute_start_ns), metric_reporter_(metric_reporter),
        stats_aggregator_(stats_aggregator), status_(Status::Success)
  {
  }

  std::unique_ptr<InferenceRequest>& Request() { return request_; }

  InferenceStatsAggregator& ContextStatsAggregator()
  {
    return context_stats_aggregator_;
  }

  void IncrementCounter()
  {
    std::lock_guard<std::mutex> lk(mtx_);
    inflight_request_counter_++;
  }

  bool DecrementCounter()
  {
    std::lock_guard<std::mutex> lk(mtx_);
    inflight_request_counter_--;
    if (inflight_request_counter_ == 0) {
#ifdef TRITON_ENABLE_STATS
      const auto& infer_stats = context_stats_aggregator_.ImmutableInferStats();
      request_->ReportStatisticsWithDuration(
          metric_reporter_, status_.IsOk(), compute_start_ns_,
          infer_stats.compute_input_duration_ns_,
          infer_stats.compute_infer_duration_ns_,
          infer_stats.compute_output_duration_ns_);
      if (status_.IsOk()) {
        stats_aggregator_->UpdateInferBatchStatsWithDuration(
            metric_reporter_, std::max(1U, request_->BatchSize()),
            infer_stats.compute_input_duration_ns_,
            infer_stats.compute_infer_duration_ns_,
            infer_stats.compute_output_duration_ns_);
      }
#endif
      InferenceRequest::Release(
          std::move(request_), TRITONSERVER_REQUEST_RELEASE_ALL);
    }
    return (inflight_request_counter_ == 0);
  }

  void SetStatus(const Status& status)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    status_ = status;
  }

 private:
  std::mutex mtx_;
  uint32_t inflight_request_counter_;
  std::unique_ptr<InferenceRequest> request_;
  uint64_t compute_start_ns_;
  MetricModelReporter* metric_reporter_;
  InferenceStatsAggregator* stats_aggregator_;
  InferenceStatsAggregator context_stats_aggregator_;
  Status status_;
};

// Step is used as 'userp' and keeps ensemble context alive
// until no more internal requests are inflight.
// Step contains metadata, and status for the
// internal infer request
struct Step {
  Step(
      size_t step_idx, const InferenceRequest::SequenceId& correlation_id,
      uint32_t flags)
      : correlation_id_(correlation_id), flags_(flags), response_flags_(0),
        infer_status_(nullptr), step_idx_(step_idx)
  {
  }

  std::shared_ptr<EnsembleContext> ctx_;
  std::unique_ptr<InferenceRequest> request_;
  InferenceRequest::SequenceId correlation_id_;
  uint32_t flags_;

  std::mutex output_mtx_;
  // Different output map to avoid address conflict from different memory types
  std::unordered_map<uintptr_t, std::shared_ptr<AllocatedMemory>>
      cpu_output_map_;
  std::unordered_map<
      int64_t, std::unordered_map<uintptr_t, std::shared_ptr<AllocatedMemory>>>
      gpu_output_map_;
  std::set<std::pair<std::string, IterationCount>> updated_tensors_;
  uint32_t response_flags_;
  TRITONSERVER_Error* infer_status_;

  size_t step_idx_;
};

struct TensorData {
  struct Metadata {
    Metadata() = default;
    Metadata(
        std::unique_ptr<InferenceRequest::Input>&& data, size_t reference_count)
        : data_(std::move(data)), remaining_reference_count_(reference_count),
          parameter_override_(false)
    {
    }
    Metadata(
        std::unique_ptr<InferenceRequest::Input>&& data, size_t reference_count,
        const InferenceRequest::SequenceId& correlation_id, uint32_t flags)
        : data_(std::move(data)), remaining_reference_count_(reference_count),
          parameter_override_(true), correlation_id_(correlation_id),
          flags_(flags)
    {
    }
    std::unique_ptr<InferenceRequest::Input> data_;
    size_t remaining_reference_count_;
    bool parameter_override_;
    InferenceRequest::SequenceId correlation_id_;
    uint32_t flags_;
  };
  TensorData() = default;
  TensorData(size_t outgoing_steps_count)
      : current_iteration_(0), outgoing_steps_count_(outgoing_steps_count),
        batch_size_(0)
  {
  }

  IterationCount AddTensor(std::unique_ptr<InferenceRequest::Input>&& tensor)
  {
    tensor_.emplace(
        current_iteration_, Metadata(std::move(tensor), outgoing_steps_count_));
    return current_iteration_++;
  }

  IterationCount AddTensor(
      std::unique_ptr<InferenceRequest::Input>&& tensor,
      const InferenceRequest::SequenceId& correlation_id, uint32_t flags)
  {
    tensor_.emplace(
        current_iteration_,
        Metadata(
            std::move(tensor), outgoing_steps_count_, correlation_id, flags));
    return current_iteration_++;
  }

  // Tensors associated with the particular ensemble tensor.
  // A container is used to handle the decoupled case
  // where variable number of tensors will be produced.
  // map 'iteration count' to pair of <tensor, remaining outgoing count>
  std::unordered_map<IterationCount, Metadata> tensor_;
  size_t current_iteration_;
  size_t outgoing_steps_count_;
  size_t batch_size_;
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
      MetricModelReporter* metric_reporter,
      InferenceStatsAggregator* stats_aggregator, InferenceServer* is,
      EnsembleInfo* info, std::unique_ptr<InferenceRequest>& request,
      cudaStream_t stream);

  // Perform transition on 'context' state given the information of
  // 'completed_step'
  static void Proceed(
      const std::shared_ptr<EnsembleContext>& context,
      const std::unique_ptr<Step>& completed_step = nullptr);

 private:
  static TRITONSERVER_Error* ResponseAlloc(
      TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
      size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
      int64_t preferred_memory_type_id, void* userp, void** buffer,
      void** buffer_userp, TRITONSERVER_MemoryType* allocated_memory_type,
      int64_t* allocated_memory_type_id);
  static TRITONSERVER_Error* ResponseRelease(
      TRITONSERVER_ResponseAllocator* allocator, void* buffer,
      void* buffer_userp, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id);
  static void RequestComplete(
      TRITONSERVER_InferenceRequest* request, const uint32_t flags,
      void* userp);
  static void ResponseComplete(
      TRITONSERVER_InferenceResponse* response, const uint32_t flags,
      void* userp);

  using StepList = std::vector<std::unique_ptr<Step>>;
  using VersionMap =
      std::unordered_map<int64_t, std::shared_ptr<InferenceBackend>>;

  // Helper function to reshape the given tensor according to the
  // config shape and batching info and its actual shape and batching info.
  // Note that 'dims' will be in full shape as opposed to 'config_dims'.
  // Return the dims after reshape.
  std::vector<int64_t> ReshapeTensorDims(
      const DimsList& config_dims, const bool config_allow_batching,
      const size_t tensor_batch_size, const std::vector<int64_t>& dims);

  // Return the list of step that becomes ready due to tensor update
  // from 'completed_step'
  Status PrepareSteps(
      const std::unique_ptr<Step>& completed_step, StepList* steps);

  // Prepare infer stats and call the inference server's function to process
  // the infer requests specified in 'steps'
  static void ScheduleSteps(
      const std::shared_ptr<EnsembleContext>& context, StepList&& steps);

  // Helper function that updates ensemble state given 'completed_step' and
  // returns the list of updated tensors in 'updated_tensors'
  Status UpdateEnsembleState(
      const std::unique_ptr<Step>& completed_step,
      std::set<std::pair<std::string, IterationCount>>* updated_tensors);

  // Helper function that returns a list of 'steps' that should be run under
  // current ensemble state. 'updated_tensors' is used so that we don't need to
  // iterate all the tensors to determine which step can be run.
  Status GetNextSteps(
      const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
      StepList* steps);

  // Helper function that completes the response of the ensemble request
  Status FinishEnsemble(
      std::unique_ptr<InferenceResponse>&& response = nullptr);

  // Helper function that initialize the 'step' given the info at 'step_idx'.
  // The 'step' will have proper request / response provider for the model
  Status InitStep(
      const size_t step_idx, const IterationCount iteration_count,
      std::unique_ptr<Step>* step);

  // Helper function that set the output of the ensemble request if it is ready
  // and valid.
  Status CheckAndSetEnsembleOutput(
      const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
      std::unique_ptr<InferenceResponse>* response);

  InferenceServer* is_;

  EnsembleInfo* info_;

  // All EnsembleContext will use the same CUDA stream managed by
  // the ensemble scheduler
  cudaStream_t stream_;

  // Mutex to avoid concurrent call on 'PrepareSteps' where ensemble state
  // are being modified
  std::mutex mutex_;

  size_t inflight_step_counter_;

  // pointer that either points to 'pruned_tensor_to_step_' or to
  // 'info_->tensor_to_step_' if all ensemble outputs are requested
  std::unordered_map<std::string, std::set<size_t>>* tensor_to_step_;

  std::unordered_map<std::string, std::set<size_t>> pruned_tensor_to_step_;
  std::unordered_map<std::string, TensorData> tensor_data_;

  // Handle to all backend that may be used in the ensemble
  std::unordered_map<std::string, VersionMap> handles_;

  // Request specific information that obtained from ensemble request and
  // should be applied to all internal requests
  uint32_t flags_;
  std::string request_id_;
  InferenceRequest::SequenceId correlation_id_;
  uint32_t priority_;
  uint64_t timeout_;

  // Objects related to the ensemble infer request
  Status ensemble_status_;
  RequestTracker* request_tracker_;

  // The allocator that will be used to allocate buffers for the
  // inference result tensors.
  std::unique_ptr<
      TRITONSERVER_ResponseAllocator,
      decltype(&TRITONSERVER_ResponseAllocatorDelete)>
      allocator_;
};

EnsembleContext::EnsembleContext(
    MetricModelReporter* metric_reporter,
    InferenceStatsAggregator* stats_aggregator, InferenceServer* is,
    EnsembleInfo* info, std::unique_ptr<InferenceRequest>& request,
    cudaStream_t stream)
    : is_(is), info_(info), stream_(stream), inflight_step_counter_(0),
      allocator_(nullptr, TRITONSERVER_ResponseAllocatorDelete)
{
  uint64_t compute_start_ns = 0;
  INFER_STATS_SET_TIMESTAMP(compute_start_ns);
  request_tracker_ = new RequestTracker(
      std::move(request), compute_start_ns, metric_reporter, stats_aggregator);

  auto& lrequest = request_tracker_->Request();

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

  // Prune ensemble first if not all outputs are requested
  std::set<std::string> ignored_tensor;
  for (const auto& ensemble_output : info_->ensemble_output_shape_) {
    ignored_tensor.insert(ensemble_output.first);
  }
  for (const auto& requested_output : lrequest->ImmutableRequestedOutputs()) {
    ignored_tensor.erase(requested_output);
  }
  if (ignored_tensor.empty()) {
    tensor_to_step_ = &(info_->tensor_to_step_);
  } else {
    pruned_tensor_to_step_ = info_->tensor_to_step_;
    tensor_to_step_ = &pruned_tensor_to_step_;
    // Backward traversal
    std::unordered_map<size_t, size_t> step_requested_output_count;
    while (!ignored_tensor.empty()) {
      std::set<std::string> new_ignored_tensor;
      for (const auto& output : ignored_tensor) {
        auto step_idx = info_->tensor_to_prev_step_[output];
        auto& step = info_->steps_[step_idx];
        auto it = step_requested_output_count.find(step_idx);
        if (it == step_requested_output_count.end()) {
          auto output_count = step.output_to_tensor_.size();
          it =
              step_requested_output_count.emplace(step_idx, output_count).first;
        }
        // If none of the outputs of the step is requested,
        // then the step can be pruned
        if (--it->second == 0) {
          for (const auto& input : step.input_to_tensor_) {
            auto& step_set = pruned_tensor_to_step_[input.second];
            step_set.erase(step_idx);
            // If all steps depend on a tensor are pruned,
            // then the tensor can be ignored.
            if (step_set.empty()) {
              new_ignored_tensor.insert(input.second);
            }
          }
        }
      }
      ignored_tensor.swap(new_ignored_tensor);
    }
  }

  for (const auto& pair : *tensor_to_step_) {
    const auto& requested_outputs = lrequest->ImmutableRequestedOutputs();
    // For requested outputs, add 1 to outgoing count as the ensemble itself
    // isn't counted as step.
    if (requested_outputs.find(pair.first) != requested_outputs.end()) {
      tensor_data_.emplace(pair.first, TensorData(pair.second.size() + 1));
    } else {
      tensor_data_.emplace(pair.first, TensorData(pair.second.size()));
    }
  }

  if (ensemble_status_.IsOk()) {
    request_id_ = lrequest->Id();
    correlation_id_ = lrequest->CorrelationId();
    flags_ = lrequest->Flags();
    priority_ = lrequest->Priority();
    timeout_ = lrequest->TimeoutMicroseconds();

    for (const auto& pr : lrequest->ImmutableInputs()) {
      const InferenceRequest::Input* input = pr.second;
      auto it = tensor_data_.find(input->Name());
      if (it != tensor_data_.end()) {
        auto& tensor_data = it->second;
        // Shape() represents reshaped value without batch dimension,
        // thus need to fill it if necessary.
        std::unique_ptr<InferenceRequest::Input> tensor;
        if (lrequest->BatchSize() != 0) {
          std::vector<int64_t> shape{lrequest->BatchSize()};
          shape.insert(
              shape.end(), input->Shape().begin(), input->Shape().end());
          tensor.reset(new InferenceRequest::Input(
              input->Name(), input->DType(), shape));
        } else {
          tensor.reset(new InferenceRequest::Input(
              input->Name(), input->DType(), input->Shape()));
        }
        tensor->SetData(input->Data());
        for (const auto& host_policy_data : input->HostPolicyData()) {
          tensor->SetData(host_policy_data.first, host_policy_data.second);
        }
        tensor_data.AddTensor(std::move(tensor));
        tensor_data.batch_size_ = lrequest->BatchSize();
      } else {
        ensemble_status_ = Status(
            Status::Code::INVALID_ARG,
            "unexpected input '" + input->Name() +
                "' in request header that does not map to any ensemble inputs");
      }
    }
  }

  TRITONSERVER_ResponseAllocator* allocator;
  TRITONSERVER_Error* err = TRITONSERVER_ResponseAllocatorNew(
      &allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */);
  if (err != nullptr) {
    ensemble_status_ = Status(
        TritonCodeToStatusCode(TRITONSERVER_ErrorCode(err)),
        TRITONSERVER_ErrorMessage(err));
    TRITONSERVER_ErrorDelete(err);
  } else {
    allocator_.reset(allocator);
  }
}

TRITONSERVER_Error*
EnsembleContext::ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* allocated_memory_type,
    int64_t* allocated_memory_type_id)
{
  *buffer = nullptr;
  *buffer_userp = nullptr;

  auto allocated_buffer = std::make_shared<AllocatedMemory>(
      byte_size, preferred_memory_type, preferred_memory_type_id);

  auto mutable_buffer = allocated_buffer->MutableBuffer(
      allocated_memory_type, allocated_memory_type_id);
  if ((mutable_buffer != nullptr) || (byte_size == 0)) {
    if (byte_size != 0) {
      *buffer = static_cast<void*>(mutable_buffer);
      auto step = reinterpret_cast<Step*>(userp);
      std::lock_guard<std::mutex> lk(step->output_mtx_);
      if (*allocated_memory_type == TRITONSERVER_MEMORY_GPU) {
        step->gpu_output_map_[*allocated_memory_type_id].emplace(
            reinterpret_cast<uintptr_t>(*buffer), std::move(allocated_buffer));
      } else {
        step->cpu_output_map_.emplace(
            reinterpret_cast<uintptr_t>(*buffer), std::move(allocated_buffer));
      }
    }
    LOG_VERBOSE(1) << "Internal response allocation: " << tensor_name
                   << ", size " << byte_size << ", addr " << *buffer
                   << ", memory type " << *allocated_memory_type << ", type id "
                   << *allocated_memory_type_id;
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
EnsembleContext::ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "Internal response release: "
                 << "size " << byte_size << ", addr " << buffer;

  // Don't do anything when releasing a buffer since ResponseAlloc
  // passes the ownership of the data to ensemble context.
  return nullptr;  // Success
}

void
EnsembleContext::RequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "deleting ensemble inference request");
    auto request_tracker = reinterpret_cast<RequestTracker*>(userp);
    if (request_tracker->DecrementCounter()) {
      delete request_tracker;
    }
  }
}

void
EnsembleContext::ResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  auto step_ptr = std::unique_ptr<Step>(reinterpret_cast<Step*>(userp));
  step_ptr->response_flags_ = flags;

  if (response != nullptr) {
    auto err = TRITONSERVER_InferenceResponseError(response);
    uint32_t count;
    bool parameter_override = false;
    InferenceRequest::SequenceId correlation_id{0};
    uint32_t flags = 0;
    if (err == nullptr) {
      err = TRITONSERVER_InferenceResponseParameterCount(response, &count);
      if (err == nullptr) {
        for (uint32_t idx = 0; idx < count; idx++) {
          const char* name;
          TRITONSERVER_ParameterType type;
          const void* vvalue;
          err = TRITONSERVER_InferenceResponseParameter(
              response, idx, &name, &type, &vvalue);
          if (err == nullptr) {
            if (!strcmp(name, "sequence_id")) {
              switch (type) {
                case TRITONSERVER_PARAMETER_INT:
                  correlation_id = InferenceRequest::SequenceId(
                      *reinterpret_cast<const uint64_t*>(vvalue));
                  parameter_override = true;
                  break;
                case TRITONSERVER_PARAMETER_STRING:
                  correlation_id = InferenceRequest::SequenceId(std::string(
                      *reinterpret_cast<const char* const*>(vvalue)));
                  parameter_override = true;
                  break;
                default:
                  err = TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INVALID_ARG,
                      "expected parameter 'sequence_id' to be "
                      "TRITONSERVER_PARAMETER_INT or "
                      "TRITONSERVER_PARAMETER_STRING");
              }
            } else if (!strcmp(name, "sequence_start")) {
              if (type != TRITONSERVER_PARAMETER_BOOL) {
                err = TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    "expect paremeter 'sequence_start' to be "
                    "TRITONSERVER_PARAMETER_BOOL");
              } else {
                if (*reinterpret_cast<const bool*>(vvalue)) {
                  flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
                }
                parameter_override = true;
              }
            } else if (!strcmp(name, "sequence_end")) {
              if (type != TRITONSERVER_PARAMETER_BOOL) {
                err = TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    "expect paremeter 'sequence_end' to be "
                    "TRITONSERVER_PARAMETER_BOOL");
              } else {
                if (*reinterpret_cast<const bool*>(vvalue)) {
                  flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
                }
                parameter_override = true;
              }
            }
          }
        }
      }
    }
    if (err == nullptr) {
      err = TRITONSERVER_InferenceResponseOutputCount(response, &count);
      if (err == nullptr) {
        std::lock_guard<std::mutex> lock(step_ptr->ctx_->mutex_);
        auto& output_to_tensor =
            step_ptr->ctx_->info_->steps_[step_ptr->step_idx_]
                .output_to_tensor_;
        for (uint32_t idx = 0; idx < count; idx++) {
          const char* name;
          TRITONSERVER_DataType datatype;
          const int64_t* shape;
          uint64_t dim_count;
          const void* base;
          size_t byte_size;
          TRITONSERVER_MemoryType memory_type;
          int64_t memory_type_id;
          void* userp;
          err = TRITONSERVER_InferenceResponseOutput(
              response, idx, &name, &datatype, &shape, &dim_count, &base,
              &byte_size, &memory_type, &memory_type_id, &userp);
          if (err == nullptr) {
            auto it = output_to_tensor.find(name);
            if (it != output_to_tensor.end()) {
              std::unique_ptr<InferenceRequest::Input> tensor(
                  new InferenceRequest::Input(
                      it->second, TritonToDataType(datatype), shape,
                      dim_count));

              if (byte_size != 0) {
                std::lock_guard<std::mutex> output_lk(step_ptr->output_mtx_);
                if (memory_type == TRITONSERVER_MEMORY_GPU) {
                  auto& gpu_output_map =
                      step_ptr->gpu_output_map_[memory_type_id];
                  auto it =
                      gpu_output_map.find(reinterpret_cast<uintptr_t>(base));
                  tensor->SetData(std::move(it->second));
                  gpu_output_map.erase(it);
                } else {
                  auto it = step_ptr->cpu_output_map_.find(
                      reinterpret_cast<uintptr_t>(base));
                  tensor->SetData(std::move(it->second));
                  step_ptr->cpu_output_map_.erase(it);
                }
              }

              auto& tensor_data = step_ptr->ctx_->tensor_data_[it->second];
              if (parameter_override) {
                step_ptr->updated_tensors_.emplace(
                    it->second, tensor_data.AddTensor(
                                    std::move(tensor), correlation_id, flags));
              } else {
                step_ptr->updated_tensors_.emplace(
                    it->second,
                    tensor_data.AddTensor(
                        std::move(tensor), step_ptr->correlation_id_,
                        step_ptr->flags_));
              }
            } else {
              LOG_VERBOSE(1)
                  << "in ensemble, an internal response header specified "
                     "output '"
                  << name << "' that does not map to any ensemble tensors";
            }
          }
          if (err != nullptr) {
            break;
          }
        }
      }
    }

    if (err != nullptr) {
      step_ptr->infer_status_ = err;
    }
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseDelete(response),
        "deleting inference response");
  }

  EnsembleContext::Proceed(step_ptr->ctx_, step_ptr);
  // Expecting more responses
  if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
    step_ptr.release();
  }
}

void
EnsembleContext::Proceed(
    const std::shared_ptr<EnsembleContext>& context,
    const std::unique_ptr<Step>& completed_step)
{
  StepList ready_steps;
  Status status = context->PrepareSteps(completed_step, &ready_steps);
  if (status.IsOk()) {
    ScheduleSteps(context, std::move(ready_steps));
  }
}

Status
EnsembleContext::PrepareSteps(
    const std::unique_ptr<Step>& completed_step, StepList* ready_steps)
{
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Initialization error, ensemble status will be not ok since the beginning
    if (completed_step == nullptr && !ensemble_status_.IsOk()) {
      ensemble_status_ = FinishEnsemble();
    }

    if (ensemble_status_.IsOk()) {
      StepList res;
      std::set<std::pair<std::string, IterationCount>> updated_tensors;
      ensemble_status_ = UpdateEnsembleState(completed_step, &updated_tensors);
      if (ensemble_status_.IsOk()) {
        ensemble_status_ = GetNextSteps(updated_tensors, ready_steps);
      }

      // Check and send ensemble response
      if ((!ensemble_status_.IsOk()) || (inflight_step_counter_ == 0) ||
          info_->is_decoupled_) {
        std::unique_ptr<InferenceResponse> response;
        if (ensemble_status_.IsOk()) {
          ensemble_status_ =
              CheckAndSetEnsembleOutput(updated_tensors, &response);
        }
        ensemble_status_ = FinishEnsemble(std::move(response));
      }
    }
    return ensemble_status_;
  }
}

Status
EnsembleContext::UpdateEnsembleState(
    const std::unique_ptr<Step>& completed_step,
    std::set<std::pair<std::string, IterationCount>>* updated_tensors)
{
  updated_tensors->clear();
  if (completed_step == nullptr) {
    for (const auto& tensor_data : tensor_data_) {
      if (!tensor_data.second.tensor_.empty()) {
        updated_tensors->emplace(tensor_data.first, 0);
      }
    }
  } else {
    if (completed_step->response_flags_ &
        TRITONSERVER_RESPONSE_COMPLETE_FINAL) {
      inflight_step_counter_--;
    }
    RETURN_IF_TRITONSERVER_ERROR(completed_step->infer_status_);
    updated_tensors->swap(completed_step->updated_tensors_);
  }
  return Status::Success;
}

Status
EnsembleContext::GetNextSteps(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    StepList* steps)
{
  steps->clear();

  std::set<std::pair<size_t, IterationCount>> next_step_idx;
  // Get steps whose tensors used for input are set
  for (const auto updated_tensor : updated_tensors) {
    const auto& step_idx = (*tensor_to_step_)[updated_tensor.first];
    for (const auto& idx : step_idx) {
      bool ready = true;
      for (const auto& input_pair : info_->steps_[idx].input_to_tensor_) {
        auto& tensor = tensor_data_[input_pair.second].tensor_;
        if (tensor.empty()) {
          ready = false;
          break;
        } else {
          // Check if other inputs have tensor with corresponding iteration
          // count
          if (tensor.find(updated_tensor.second) == tensor.end()) {
            ready = false;
            break;
          }
        }
      }
      if (ready) {
        next_step_idx.emplace(idx, updated_tensor.second);
      }
    }
  }

  for (const auto& idx : next_step_idx) {
    steps->emplace_back();
    RETURN_IF_ERROR(InitStep(idx.first, idx.second, &(steps->back())));
  }
  inflight_step_counter_ += steps->size();

  return Status::Success;
}

Status
EnsembleContext::InitStep(
    const size_t step_idx, const IterationCount iteration_count,
    std::unique_ptr<Step>* step)
{
  const auto& istep = info_->steps_[step_idx];
  auto& version_map = handles_[istep.model_name_];
  auto& backend = version_map[istep.model_version_];

  const bool allow_batching = (backend->Config().max_batch_size() > 0);

  auto irequest = std::unique_ptr<InferenceRequest>(
      new InferenceRequest(backend, istep.model_version_));

  // Store the pointers to tensors used so that we can prune them afterward.
  // Can't prune the tensor in the input loop below as it may be used by
  // multiple inputs in the same step.
  std::map<TensorData*, size_t*> releasing_tensors;

  // Set inputs in request, prepare input map,
  // and set overridden parameter if any.
  auto correlation_id = correlation_id_;
  auto flags = flags_;
  bool parameter_set = false;
  for (const auto& pair : istep.input_to_tensor_) {
    auto& tensor_data = tensor_data_[pair.second];
    auto& tensor = tensor_data.tensor_[iteration_count];

    // If the actual shape and config shape agree with each other without
    // considering batch size, non-batch / batch conversion are not required.
    const inference::ModelInput* input_config;
    backend->GetInput(pair.first, &input_config);
    auto shape = ReshapeTensorDims(
        input_config->dims(), allow_batching, tensor_data.batch_size_,
        tensor.data_->OriginalShape());

    InferenceRequest::Input* input;
    RETURN_IF_ERROR(irequest->AddOriginalInput(
        pair.first, tensor.data_->DType(), shape, &input));
    RETURN_IF_ERROR(input->SetData(tensor.data_->Data()));
    for (const auto& host_policy_data : tensor.data_->HostPolicyData()) {
      RETURN_IF_ERROR(
          input->SetData(host_policy_data.first, host_policy_data.second));
    }

    releasing_tensors.emplace(&tensor_data, &tensor.remaining_reference_count_);

    if (tensor.parameter_override_) {
      if (parameter_set && ((correlation_id != tensor.correlation_id_) ||
                            (flags != tensor.flags_))) {
        LOG_ERROR << "Different set of response parameters are set for '"
                  << istep.model_name_ << "'. Parameter correlation ID "
                  << correlation_id << ", flags " << flags << " is used.";
        continue;
      }
      correlation_id = tensor.correlation_id_;
      flags = tensor.flags_;
      parameter_set = true;
    }
  }

  // Prune the tensor if it is not needed by other steps
  for (auto& releasing_pair : releasing_tensors) {
    if ((--(*releasing_pair.second)) == 0) {
      releasing_pair.first->tensor_.erase(iteration_count);
    }
  }

  // Set requested outputs in request header
  for (const auto& pair : istep.output_to_tensor_) {
    irequest->AddOriginalRequestedOutput(pair.first);
  }

  step->reset(new Step(step_idx, correlation_id, flags));

  irequest->SetId(request_id_);
  irequest->SetCorrelationId(correlation_id);
  irequest->SetFlags(flags);
  irequest->SetPriority(priority_);
  irequest->SetTimeoutMicroseconds(timeout_);
#ifdef TRITON_ENABLE_STATS
  irequest->SetSecondaryStatsAggregator(
      &request_tracker_->ContextStatsAggregator());
#endif
  irequest->SetResponseCallback(
      reinterpret_cast<ResponseAllocator*>(allocator_.get()), step->get(),
      ResponseComplete, step->get());
  irequest->SetReleaseCallback(RequestComplete, request_tracker_);

  RETURN_IF_ERROR(irequest->PrepareForInference());

#ifdef TRITON_ENABLE_TRACING
  auto& parent_trace = request_tracker_->Request()->Trace();
  if (parent_trace != nullptr) {
    irequest->SetTrace(std::move(parent_trace->SpawnChildTrace()));
    irequest->Trace()->SetModelName(irequest->ModelName());
    irequest->Trace()->SetModelVersion(irequest->ActualModelVersion());
  }
#endif

  // Record the batch size of output in advance as
  // there is no other way to access it later on.
  for (const auto& pair : istep.output_to_tensor_) {
    auto& output_data_ = tensor_data_[pair.second];
    output_data_.batch_size_ = irequest->BatchSize();
  }

  (*step)->request_ = std::move(irequest);

  return Status::Success;
}

std::vector<int64_t>
EnsembleContext::ReshapeTensorDims(
    const DimsList& config_dims, const bool config_allow_batching,
    const size_t tensor_batch_size, const std::vector<int64_t>& dims)
{
  bool reshaped = false;
  std::vector<int64_t> res;

  // Only attempt to reshape if one setting is batchable while the other is not,
  // the case of two mismatched batchable shapes is not considered.
  // If the actual shape and config shape agree with each other without
  // considering batch size, non-batch / batch conversion are not required.
  if (config_allow_batching != (tensor_batch_size != 0)) {
    // expect batching but the tensor is generated from nobatching model
    if (config_allow_batching) {
      if (CompareDimsWithWildcard(config_dims, dims)) {
        // If 'dims' already matches 'config_dims', prepend with batch size 1
        res.push_back(1);
        res.insert(res.end(), dims.begin(), dims.end());
        reshaped = true;
      }
      // Otherwise, assuming the tensor is already in the batch expected
      // by the model and do nothing
    } else {
      // Check if the batched tensor can be sent to the non-batching
      // model as one tensor. If not, strip the batch dimension if
      // it is batch size 1
      if (!CompareDimsWithWildcard(config_dims, dims) &&
          (tensor_batch_size == 1)) {
        res.assign(dims.begin() + 1, dims.end());
        reshaped = true;
      }
    }
  }

  if (!reshaped) {
    res = dims;
  }
  return res;
}

Status
EnsembleContext::FinishEnsemble(std::unique_ptr<InferenceResponse>&& response)
{
  // Do nothing if the ensemble is finished
  if (request_tracker_ == nullptr) {
    return ensemble_status_;
  }

  // Add ensemble name to make error message more trackable
  if (!ensemble_status_.IsOk()) {
    ensemble_status_ = Status(
        ensemble_status_.StatusCode(), "in ensemble '" + info_->ensemble_name_ +
                                           "', " + ensemble_status_.Message());
  }

  if (ensemble_status_.IsOk()) {
    if (info_->is_decoupled_) {
      if (response != nullptr) {
        InferenceResponse::Send(std::move(response), 0 /* flags */);
      }
      if (inflight_step_counter_ != 0) {
        return ensemble_status_;
      }
      request_tracker_->Request()->ResponseFactory().SendFlags(
          TRITONSERVER_RESPONSE_COMPLETE_FINAL);
    } else {
      InferenceResponse::Send(
          std::move(response), TRITONSERVER_RESPONSE_COMPLETE_FINAL);
    }
  } else {
    if (response != nullptr) {
      InferenceResponse::SendWithStatus(
          std::move(response), TRITONSERVER_RESPONSE_COMPLETE_FINAL,
          ensemble_status_);
    } else {
      InferenceRequest::RespondIfError(
          request_tracker_->Request(), ensemble_status_);
    }
  }

  // Reach here when the ensemble execution comes to the end, 'ensemble_status_'
  // at this point is representative.
  request_tracker_->SetStatus(ensemble_status_);
  if (request_tracker_->DecrementCounter()) {
    delete request_tracker_;
  }
  request_tracker_ = nullptr;
  return ensemble_status_;
}

Status
EnsembleContext::CheckAndSetEnsembleOutput(
    const std::set<std::pair<std::string, IterationCount>>& updated_tensors,
    std::unique_ptr<InferenceResponse>* response)
{
  IterationCount iteration_count = 0;
  // Check if updated tensor is one of the ensemble output and if all outputs
  // have tensor of the same iteration count
  bool ready = false;
  auto& lrequest = request_tracker_->Request();
  const auto& requested_outputs = lrequest->ImmutableRequestedOutputs();
  for (const auto updated_tensor : updated_tensors) {
    if (requested_outputs.find(updated_tensor.first) ==
        requested_outputs.end()) {
      continue;
    }

    ready = true;
    iteration_count = updated_tensor.second;
    for (const auto& output : requested_outputs) {
      auto& tensor = tensor_data_[output].tensor_;
      if (tensor.empty()) {
        ready = false;
        break;
      } else {
        // Check if other outputs have tensor with corresponding iteration count
        if (tensor.find(iteration_count) == tensor.end()) {
          ready = false;
          break;
        }
      }
    }
  }
  if (!ready) {
    if (info_->is_decoupled_) {
      return Status::Success;
    }
    return Status(
        Status::Code::INVALID_ARG,
        "unexpected deadlock, at least one output is not set while no more "
        "ensemble steps can be made");
  }

  RETURN_IF_ERROR(lrequest->ResponseFactory().CreateResponse(response));

  bool cuda_async_copy = false;
  std::map<TensorData*, size_t*> releasing_tensors;
  for (const auto& output_pair : info_->ensemble_output_shape_) {
    if (requested_outputs.find(output_pair.first) == requested_outputs.end()) {
      continue;
    }
    // Check if output is ready
    auto& tensor_data = tensor_data_[output_pair.first];
    auto& tensor = tensor_data.tensor_[iteration_count];

    auto shape = ReshapeTensorDims(
        output_pair.second, (lrequest->BatchSize() != 0),
        tensor_data.batch_size_, tensor.data_->OriginalShape());

    InferenceResponse::Output* output;
    RETURN_IF_ERROR((*response)->AddOutput(
        output_pair.first, tensor.data_->DType(), shape, &output));

    // Use the memory type of the memory block as preferred memory type
    TRITONSERVER_MemoryType dst_memory_type;
    int64_t dst_memory_type_id;
    size_t content_size;
    tensor.data_->Data()->BufferAt(
        0, &content_size, &dst_memory_type, &dst_memory_type_id);

    void* buffer;
    RETURN_IF_ERROR(output->AllocateDataBuffer(
        &buffer, content_size, &dst_memory_type, &dst_memory_type_id));

    // Done with this output if 'expected_byte_size' is 0
    if (content_size == 0) {
      continue;
    } else if (buffer == nullptr) {
      return Status(
          Status::Code::INTERNAL,
          "failed to allocate buffer for output '" + output_pair.first + "'");
    }

    size_t content_offset = 0;
    size_t content_idx = 0;
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;

    const char* content = tensor.data_->Data()->BufferAt(
        content_idx, &content_size, &src_memory_type, &src_memory_type_id);
    bool cuda_used = false;
    while (content != nullptr) {
      RETURN_IF_ERROR(CopyBuffer(
          output_pair.first, src_memory_type, src_memory_type_id,
          dst_memory_type, dst_memory_type_id, content_size, content,
          ((char*)buffer) + content_offset, stream_, &cuda_used));
      cuda_async_copy |= cuda_used;

      content_offset += content_size;
      content_idx++;
      content = tensor.data_->Data()->BufferAt(
          content_idx, &content_size, &src_memory_type, &src_memory_type_id);
    }

    releasing_tensors.emplace(&tensor_data, &tensor.remaining_reference_count_);

    if (tensor.parameter_override_) {
      switch (lrequest->CorrelationId().Type()) {
        case InferenceRequest::SequenceId::DataType::STRING:
          (*response)->AddParameter(
              "sequence_id", tensor.correlation_id_.StringValue().c_str());
          break;
        case InferenceRequest::SequenceId::DataType::UINT64:
          (*response)->AddParameter(
              "sequence_id",
              (int64_t)tensor.correlation_id_.UnsignedIntValue());
          break;
        default:
          (*response)->AddParameter(
              "sequence_id",
              (int64_t)tensor.correlation_id_.UnsignedIntValue());
          break;
      }
      (*response)->AddParameter(
          "sequence_start",
          (tensor.flags_ & TRITONSERVER_REQUEST_FLAG_SEQUENCE_START) != 0);
      (*response)->AddParameter(
          "sequence_end",
          (tensor.flags_ & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END) != 0);
    }
  }

  if (cuda_async_copy) {
#ifdef TRITON_ENABLE_GPU
    cudaStreamSynchronize(stream_);
#else
    return Status(
        Status::Code::INTERNAL,
        "unexpected CUDA copy flag set while GPU is not supported");
#endif  // TRITON_ENABLE_GPU
  }

  // Prune the tensor if it is not needed by other steps
  for (auto& releasing_pair : releasing_tensors) {
    if ((--(*releasing_pair.second)) == 0) {
      releasing_pair.first->tensor_.erase(iteration_count);
    }
  }

  return Status::Success;
}

void
EnsembleContext::ScheduleSteps(
    const std::shared_ptr<EnsembleContext>& context, StepList&& steps)
{
  for (auto& step : steps) {
    step->ctx_ = context;
    {
      std::lock_guard<std::mutex> lock(context->mutex_);

      // Need to check the ensemble_status_ to ensure the FinishEnsemble()
      // is called only once.
      if (context->ensemble_status_.IsOk()) {
        context->request_tracker_->IncrementCounter();
        context->ensemble_status_ = context->is_->InferAsync(step->request_);
        if (!context->ensemble_status_.IsOk()) {
          // The request is not sent to server properly, shouldn't expect its
          // release function get called.
          context->request_tracker_->DecrementCounter();
          context->ensemble_status_ = context->FinishEnsemble();
          break;
        }
      }
      step.release();
    }
  }
}

}  // namespace

Status
EnsembleScheduler::Create(
    InferenceStatsAggregator* const stats_aggregator,
    InferenceServer* const server, const inference::ModelConfig& config,
    std::unique_ptr<Scheduler>* scheduler)
{
  scheduler->reset(new EnsembleScheduler(stats_aggregator, server, config));
  return Status::Success;
}

Status
EnsembleScheduler::Enqueue(std::unique_ptr<InferenceRequest>& request)
{
  // Queue timer starts at the beginning of the queueing and
  // scheduling process
  request->CaptureQueueStartNs();
  INFER_TRACE_ACTIVITY(
      request->Trace(), TRITONSERVER_TRACE_QUEUE_START,
      request->QueueStartNs());
  std::shared_ptr<EnsembleContext> context(new EnsembleContext(
      metric_reporter_.get(), stats_aggregator_, is_, info_.get(), request,
      stream_));
  EnsembleContext::Proceed(context);
  return Status::Success;
}

EnsembleScheduler::EnsembleScheduler(
    InferenceStatsAggregator* const stats_aggregator,
    InferenceServer* const server, const inference::ModelConfig& config)
    : stats_aggregator_(stats_aggregator), is_(server), stream_(nullptr)
{
#ifdef TRITON_ENABLE_GPU
  // create CUDA stream
  auto cuerr = cudaStreamCreate(&stream_);
  if (cuerr != cudaSuccess) {
    stream_ = nullptr;
    LOG_ERROR << "unable to create stream for " << config.name() << ": "
              << cudaGetErrorString(cuerr);
  }
#endif  // TRITON_ENABLE_GPU

#ifdef TRITON_ENABLE_METRICS
  if (Metrics::Enabled()) {
    MetricModelReporter::Create(
        config.name(), 1, -1, config.metric_tags(), &metric_reporter_);
  }
#endif  // TRITON_ENABLE_METRICS

  // Set 'info_' based on 'config'
  info_.reset(new EnsembleInfo());

  info_->ensemble_name_ = config.name();

  // This config field is filled internally for ensemble models
  info_->is_decoupled_ = config.model_transaction_policy().decoupled();

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

      info_->tensor_to_prev_step_.emplace(pair.second, step_idx);
    }
  }
}

EnsembleScheduler::~EnsembleScheduler()
{
#ifdef TRITON_ENABLE_GPU
  if (stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda stream: " << cudaGetErrorString(err);
    }
  }
#endif  // TRITON_ENABLE_GPU
}

}}  // namespace nvidia::inferenceserver

#endif  // TRITON_ENABLE_ENSEMBLE
