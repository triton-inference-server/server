// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/backend.h"

#include <chrono>
#include <future>
#include "src/core/constants.h"
#include "src/core/dynamic_batch_scheduler.h"
#include "src/core/filesystem.h"
#include "src/core/infer_request.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"
#include "src/core/sequence_batch_scheduler.h"

namespace nvidia { namespace inferenceserver {

namespace {

// Utilities for warmup feature
TRITONSERVER_Error*
WarmupResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  *buffer = malloc(byte_size);
  if (*buffer != nullptr) {
    *actual_memory_type = TRITONSERVER_MEMORY_CPU;
    *actual_memory_type_id = 0;
    return nullptr;
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL,
      "failed to allocate output buffer for warmup.");
}

TRITONSERVER_Error*
WarmupResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  free(buffer);
  return nullptr;
}

ResponseAllocator warmup_allocator = ResponseAllocator(
    WarmupResponseAlloc, WarmupResponseRelease, nullptr /* start_fn */);

void
WarmupResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
    void* userp)
{
  if (iresponse != nullptr) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseError(iresponse), "warmup error");
    // Just delete the response, warmup doesn't check for correctness
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseDelete(iresponse),
        "deleting warmup response");
  }
}

void
WarmupRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    TRITONSERVER_InferenceRequestDelete(request);
    if (userp != nullptr) {
      auto warmup_promise = reinterpret_cast<std::promise<void>*>(userp);
      warmup_promise->set_value();
    }
  }
}

}  // namespace

Status
InferenceBackend::GetInput(
    const std::string& name, const ModelInput** input) const
{
  const auto itr = input_map_.find(name);
  if (itr == input_map_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "unexpected inference input '" + name + "' for model '" + Name() + "'");
  }

  *input = &itr->second;
  return Status::Success;
}

Status
InferenceBackend::GetOutput(
    const std::string& name, const ModelOutput** output) const
{
  const auto itr = output_map_.find(name);
  if (itr == output_map_.end()) {
    return Status(
        Status::Code::INVALID_ARG, "unexpected inference output '" + name +
                                       "' for model '" + Name() + "'");
  }

  *output = &itr->second;
  return Status::Success;
}

Status
InferenceBackend::SetModelConfig(
    const std::string& path, const ModelConfig& config)
{
  config_ = config;
  RETURN_IF_ERROR(GetModelVersionFromPath(path, &version_));

  // Initialize the input map
  for (const auto& io : config.input()) {
    input_map_.insert(std::make_pair(io.name(), io));
  }

  // Initialize the output map and label provider for each output
  label_provider_ = std::make_shared<LabelProvider>();
  model_dir_ = DirName(path);
  for (const auto& io : config.output()) {
    output_map_.insert(std::make_pair(io.name(), io));

    if (!io.label_filename().empty()) {
      const auto label_path = JoinPath({model_dir_, io.label_filename()});
      RETURN_IF_ERROR(label_provider_->AddLabels(io.name(), label_path));
    }
  }

  if (config_.has_dynamic_batching()) {
    default_priority_level_ =
        config_.dynamic_batching().default_priority_level();
    max_priority_level_ = config_.dynamic_batching().priority_levels();
  } else {
    default_priority_level_ = 0;
    max_priority_level_ = 0;
  }

  return Status::Success;
}

Status
InferenceBackend::SetScheduler(std::unique_ptr<Scheduler> scheduler)
{
  if (scheduler_ != nullptr) {
    return Status(
        Status::Code::INTERNAL, "Attempt to change scheduler not allowed");
  }

  scheduler_ = std::move(scheduler);
  return Status::Success;
}

Status
InferenceBackend::SetConfiguredScheduler(
    const uint32_t runner_cnt, const Scheduler::StandardInitFunc& OnInit,
    const Scheduler::StandardRunFunc& OnRun)
{
  std::unique_ptr<Scheduler> scheduler;

  // Create a warmup function for the scheduler thread to run the contexts
  // in corresponding threads. Currently the warmup function can't be run
  // asynchronously with respect to Scheduler::Create() as there is no way to
  // change ModelReadyState, which is controlled by model manager, from within
  // the scheduler.
  // But running warmup synchronously allows us to use one set of warmup data
  // for all contexts.
  std::vector<std::vector<WarmupData>> samples{runner_cnt};
  if (Config().model_warmup_size() != 0) {
    // The ownership of InferenceRequest will be transferred on model execution,
    // so must have unique set of samples for each runner
    for (auto& runner_samples : samples) {
      RETURN_IF_ERROR(GenerateWarmupData(&runner_samples));
    }
  }

  auto OnWarmup = [this, &samples](uint32_t runner_idx) -> Status {
    auto& runner_samples = samples[runner_idx];
    for (auto& sample : runner_samples) {
      LOG_VERBOSE(1) << "model '" << sample.requests_.back()->ModelName()
                     << "' instance " << std::to_string(runner_idx)
                     << " is running warmup sample '" << sample.sample_name_
                     << "'";

      std::promise<void> warmup_promise;
      // only now we can set the proper request complete callback,
      // only one request in the batch can set the promise.
      sample.requests_[0]->SetReleaseCallback(
          WarmupRequestComplete, &warmup_promise);
      for (size_t idx = 1; idx < sample.requests_.size(); idx++) {
        sample.requests_[idx]->SetReleaseCallback(
            WarmupRequestComplete, nullptr);
      }
      WarmUp(runner_idx, sample);
      warmup_promise.get_future().get();
    }

    return Status::Success;
  };

  // Need to enforce equal shape batches (i.e. non-ragged batches) if
  // the model 1) allows one or more variable-size input tensors that
  // are not marked as 'allow_ragged_batch' or 2) has one or more
  // shape-tensor inputs. This is not needed if all input shapes are
  // non-variable and if there are no shape tensors... so we don't
  // enable it in that case for efficiency reasons.
  std::unordered_map<std::string, bool> enforce_equal_shape_tensors;
  for (const auto input : config_.input()) {
    if (input.is_shape_tensor()) {
      enforce_equal_shape_tensors.insert({input.name(), true});
    } else if (!input.allow_ragged_batch() && (GetElementCount(input) == -1)) {
      enforce_equal_shape_tensors.insert({input.name(), false});
    }
  }

  // If 'sequence_batching' is configured use the SequenceBatchScheduler,
  // otherwise use the default DynamicBatchScheduler.
  if (config_.has_sequence_batching()) {
    // Sequence batcher
    RETURN_IF_ERROR(SequenceBatchScheduler::Create(
        config_, runner_cnt, OnInit, OnWarmup, OnRun,
        enforce_equal_shape_tensors, &scheduler));
  } else if (config_.has_dynamic_batching()) {
    // Dynamic batcher
    std::set<int32_t> preferred_batch_sizes;
    for (const auto size : config_.dynamic_batching().preferred_batch_size()) {
      preferred_batch_sizes.insert(size);
    }

    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
        0 /* runner_id_start */, runner_cnt, GetCpuNiceLevel(config_), OnInit,
        OnWarmup, OnRun, true /* dynamic_batching_enabled */,
        enforce_equal_shape_tensors,
        config_.dynamic_batching().preserve_ordering(), preferred_batch_sizes,
        config_.dynamic_batching().max_queue_delay_microseconds(),
        config_.dynamic_batching().default_queue_policy(),
        config_.dynamic_batching().priority_levels(),
        config_.dynamic_batching().priority_queue_policy(), &scheduler));
  } else {
    // Default scheduler. Use dynamic batch scheduler (with batching
    // disabled) as the default scheduler.
    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
        0 /* runner_id_start */, runner_cnt, GetCpuNiceLevel(config_), OnInit,
        OnWarmup, OnRun, false /* dynamic_batching_enabled */,
        std::unordered_map<
            std::string, bool>() /* enforce_equal_shape_tensors */,
        false /* preserve_ordering */,
        std::set<int32_t>() /* preferred_batch_sizes */,
        0 /* max_queue_delay_microseconds */, &scheduler));
  }

  return SetScheduler(std::move(scheduler));
}

Status
InferenceBackend::Init(
    const std::string& path, const ModelConfig& config,
    const std::string& platform)
{
  RETURN_IF_ERROR(
      ValidateModelConfig(config, platform, min_compute_capability_));
  RETURN_IF_ERROR(SetModelConfig(path, config));

  return Status::Success;
}

void
InferenceBackend::Run(
    uint32_t runner_idx,
    std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  // Each runner executes using the corresponding context...  If the
  // runner_idx is invalid then the scheduler has done something badly
  // wrong so fail and release all requests.
  if (runner_idx >= contexts_.size()) {
    InferenceRequest::RespondIfError(
        requests,
        Status(
            Status::Code::INTERNAL,
            "unexpected runner index" + std::to_string(runner_idx) +
                ", max allowed " + std::to_string(contexts_.size())),
        true /*release_requests */);
    return;
  }

  contexts_[runner_idx]->Run(this, std::move(requests));
}

void
InferenceBackend::WarmUp(uint32_t runner_idx, WarmupData& sample)
{
  // Capture timestamp before run to avoid incorrect accumulation from
  // sequential warmup runs
  for (auto& request : sample.requests_) {
#ifdef TRITON_ENABLE_STATS
    request->CaptureRequestStartNs();
#endif  // TRITON_ENABLE_STATS
    request->CaptureQueueStartNs();
  }
  Run(runner_idx, std::move(sample.requests_));
}

Status
InferenceBackend::GenerateWarmupData(std::vector<WarmupData>* samples)
{
  samples->clear();
  for (const auto& warmup_setting : config_.model_warmup()) {
    if (warmup_setting.batch_size() == 0) {
      LOG_VERBOSE(1) << "Skipping batch 0 warmup sample '"
                     << warmup_setting.name() << "'";
      continue;
    }
    LOG_VERBOSE(1) << "Generating warmup sample data for '"
                   << warmup_setting.name() << "'";

    // Two passes. First pass to get max byte size for synthetic
    // data. Second pass to add original inputs and override inputs
    // for control inputs.
    int64_t max_zero_byte_size = 0;
    int64_t max_random_byte_size = 0;
    for (const auto& input_meta : warmup_setting.inputs()) {
      auto element_count = GetElementCount(input_meta.second.dims());
      if (element_count == -1) {
        return Status(
            Status::Code::INVALID_ARG,
            "warmup setting expects all variable-size dimensions are specified "
            "for input '" +
                input_meta.first + "'");
      }

      int64_t batch_byte_size =
          element_count * GetDataTypeByteSize(input_meta.second.data_type());
      if (batch_byte_size == 0) {
        batch_byte_size = element_count * sizeof(int32_t);
      }

      switch (input_meta.second.input_data_type_case()) {
        case ModelWarmup_Input::InputDataTypeCase::kZeroData:
          max_zero_byte_size = std::max(batch_byte_size, max_zero_byte_size);
          break;
        case ModelWarmup_Input::InputDataTypeCase::kRandomData: {
          if (input_meta.second.data_type() == DataType::TYPE_STRING) {
            max_zero_byte_size = std::max(batch_byte_size, max_zero_byte_size);
          } else {
            max_random_byte_size =
                std::max(batch_byte_size, max_random_byte_size);
          }
          break;
        }
        default:
          break;
      }
    }

    samples->emplace_back(warmup_setting.name());
    auto& warmup_data = samples->back();
    // Create buffers for synthetic data
    TRITONSERVER_MemoryType type;
    int64_t type_id;
    warmup_data.zero_data_.reset(new AllocatedMemory(
        max_zero_byte_size, TRITONSERVER_MEMORY_CPU_PINNED /* memory_type */,
        0 /* memory_type_id */));
    char* zero_buffer = warmup_data.zero_data_->MutableBuffer(&type, &type_id);
    memset(zero_buffer, 0, max_zero_byte_size);

    warmup_data.random_data_.reset(new AllocatedMemory(
        max_random_byte_size, TRITONSERVER_MEMORY_CPU_PINNED /* memory_type */,
        0 /* memory_type_id */));
    char* random_buffer =
        warmup_data.random_data_->MutableBuffer(&type, &type_id);
    for (int64_t offset = 0; offset < max_random_byte_size; offset++) {
      random_buffer[offset] = rand();
    }

    // Prepare the inference request for the specified sample.
    for (size_t cnt = 0; cnt < warmup_setting.batch_size(); cnt++) {
      warmup_data.requests_.emplace_back(new InferenceRequest(this, Version()));
      auto& lrequest = warmup_data.requests_.back();

      // Second pass to prepare original inputs.
      std::vector<std::shared_ptr<InferenceRequest::Input>> input_sps;
      for (const auto& input_meta : warmup_setting.inputs()) {
        auto batch1_element_count = GetElementCount(input_meta.second.dims());
        auto batch_byte_size =
            batch1_element_count *
            GetDataTypeByteSize(input_meta.second.data_type());
        if (batch_byte_size == 0) {
          batch_byte_size = batch1_element_count * sizeof(int32_t);
        }

        const char* allocated_ptr;
        switch (input_meta.second.input_data_type_case()) {
          case ModelWarmup_Input::InputDataTypeCase::kZeroData:
            allocated_ptr = zero_buffer;
            break;
          case ModelWarmup_Input::InputDataTypeCase::kRandomData: {
            if (input_meta.second.data_type() == DataType::TYPE_STRING) {
              allocated_ptr = zero_buffer;
            } else {
              allocated_ptr = random_buffer;
            }
            break;
          }
          case ModelWarmup_Input::InputDataTypeCase::kInputDataFile: {
            // For data provided from file, we can set buffer in first pass
            warmup_data.provided_data_.emplace_back(new std::string());
            auto input_data = warmup_data.provided_data_.back().get();
            RETURN_IF_ERROR(ReadTextFile(
                JoinPath({model_dir_, kWarmupDataFolder,
                          input_meta.second.input_data_file()}),
                input_data));
            if (input_meta.second.data_type() == DataType::TYPE_STRING) {
              batch_byte_size = input_data->size();
            } else if (((size_t)batch_byte_size) > input_data->size()) {
              return Status(
                  Status::Code::INVALID_ARG,
                  "warmup setting expects " + std::to_string(batch_byte_size) +
                      " bytes, but the data "
                      "provided from " +
                      input_meta.second.input_data_file() + "only has " +
                      std::to_string(input_data->size()) + " bytes");
            }
            allocated_ptr = input_data->data();
            break;
          }
          default:
            return Status(
                Status::Code::INVALID_ARG, "warmup setting expects input '" +
                                               input_meta.first +
                                               "' to have input_data_type set");
        }

        const ModelInput* input_config;
        bool is_original_input =
            GetInput(input_meta.first, &input_config).IsOk();
        InferenceRequest::Input* input = nullptr;
        std::vector<int64_t> input_meta_shape;
        // Append batch size only if the model supports batching
        // and not control inpt.
        if ((config_.max_batch_size() != 0) && is_original_input) {
          input_meta_shape.push_back(1);
        }
        for (auto d : input_meta.second.dims()) {
          input_meta_shape.push_back(d);
        }
        if (is_original_input) {
          RETURN_IF_ERROR(lrequest->AddOriginalInput(
              input_meta.first, input_meta.second.data_type(), input_meta_shape,
              &input));
        } else {
          input_sps.emplace_back();
          RETURN_IF_ERROR(lrequest->AddOverrideInput(
              input_meta.first, input_meta.second.data_type(), input_meta_shape,
              &input_sps.back()));
          input = input_sps.back().get();
        }
        RETURN_IF_ERROR(input->AppendData(
            allocated_ptr, batch_byte_size,
            TRITONSERVER_MEMORY_CPU /* memory_type */, 0 /* memory_type_id */));
      }

      RETURN_IF_ERROR(lrequest->PrepareForInference());
      // Override inputs must be added after PrepareForInference() is called
      for (const auto& sp : input_sps) {
        RETURN_IF_ERROR(lrequest->AddOverrideInput(sp));
      }

      RETURN_IF_ERROR(lrequest->SetResponseCallback(
          &warmup_allocator, nullptr, WarmupResponseComplete, nullptr));
    }
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
