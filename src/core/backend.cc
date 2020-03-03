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
#include "src/core/logging.h"
#include "src/core/metric_model_reporter.h"
#include "src/core/model_config_utils.h"
#include "src/core/sequence_batch_scheduler.h"
#include "src/core/trtserver.h"

namespace nvidia { namespace inferenceserver {

Status
InferenceBackend::GetInput(
    const std::string& name, const ModelInput** input) const
{
  const auto itr = input_map_.find(name);
  if (itr == input_map_.end()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
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
        RequestStatusCode::INVALID_ARG, "unexpected inference output '" + name +
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

  // Create the metric reporter for this backend.
  metric_reporter_ = std::make_shared<MetricModelReporter>(
      Name(), version_, config_.metric_tags());

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
        RequestStatusCode::INTERNAL, "Attempt to change scheduler not allowed");
  }

  scheduler_ = std::move(scheduler);
  return Status::Success;
}

Status
InferenceBackend::SetConfiguredScheduler(
    const uint32_t runner_cnt, const Scheduler::StandardInitFunc& OnInit,
    const Scheduler::StandardRunFunc& OnRun,
    const Scheduler::StandardShapeTensorPeekFunc& OnPeek)
{
  std::unique_ptr<Scheduler> scheduler;

  // Create a warmup function for the scheduler thread to run the contexts
  // in corresponding threads. Currently the warmup function can't be run
  // asynchronously with respect to Scheduler::Create() as there is no way to
  // change ModelReadyState, which is controlled by model manager, from within
  // the scheduler.
  // But running warmup synchronously allows us to use one set of warmup data
  // for all contexts.
  std::vector<WarmupData> samples;
  if (Config().model_warmup_size() != 0) {
    RETURN_IF_ERROR(GenerateWarmupData(&samples));
  }

  // [DLIS-1001] Assumption on runner tied to instance is not longer valid
  auto OnWarmup = [this, &samples](uint32_t runner_idx) -> Status {
    for (const auto& sample : samples) {
      LOG_VERBOSE(1) << "model '" << sample.irequest_->ModelName()
                     << "' instance " << std::to_string(runner_idx)
                     << " is running warmup sample '" << sample.sample_name_
                     << "'";
      std::vector<Scheduler::Payload> payloads;
      // Duplicate payloads to match batch size requirement.
      for (size_t idx = 0; idx < sample.batch_size_; idx++) {
        std::shared_ptr<InferRequestProvider> request_provider;
        RETURN_IF_ERROR(
            InferRequestProvider::Create(sample.irequest_, &request_provider));
        RETURN_IF_ERROR(
            request_provider->AddInputOverrides(sample.input_override_));
        payloads.emplace_back(nullptr, request_provider, nullptr, nullptr);
      }

      std::promise<Status> warmup_promise;
      auto warmup_future = warmup_promise.get_future();
      Run(runner_idx, &payloads, [&warmup_promise](Status status) {
        warmup_promise.set_value(status);
      });
      RETURN_IF_ERROR(warmup_future.get());
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
        config_, runner_cnt, OnInit, OnWarmup, OnRun, OnPeek,
        enforce_equal_shape_tensors, &scheduler));
  } else if (config_.has_dynamic_batching()) {
    // Dynamic batcher
    std::set<int32_t> preferred_batch_sizes;
    for (const auto size : config_.dynamic_batching().preferred_batch_size()) {
      preferred_batch_sizes.insert(size);
    }

    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
        0 /* runner_id_start */, runner_cnt, GetCpuNiceLevel(config_), OnInit,
        OnWarmup, OnRun, OnPeek, true /* dynamic_batching_enabled */,
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
        OnWarmup, OnRun, OnPeek, false /* dynamic_batching_enabled */,
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
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(const Status&)> OnCompleteHandleInfer)
{
  scheduler_->Enqueue(
      stats, request_provider, response_provider, OnCompleteHandleInfer);
}

void
InferenceBackend::Run(
    uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
    std::function<void(Status)> OnCompleteQueuedPayloads)
{
  // Each runner executes using the corresponding context...
  if (runner_idx >= contexts_.size()) {
    OnCompleteQueuedPayloads(Status(
        RequestStatusCode::INTERNAL,
        "unexpected runner index" + std::to_string(runner_idx) +
            ", max allowed " + std::to_string(contexts_.size())));
    return;
  }

#ifdef TRTIS_ENABLE_STATS
  // Stop queue timer and start compute timer when the payload is
  // scheduled to run
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeStart);
      payload.stats_->SetGPUDevice(contexts_[runner_idx]->gpu_device_);
    }
  }
#endif  // TRTIS_ENABLE_STATS

  Status status = contexts_[runner_idx]->Run(this, payloads);

#ifdef TRTIS_ENABLE_STATS
  // Stop compute timers.
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeEnd);
    }
  }
#endif  // TRTIS_ENABLE_STATS

  OnCompleteQueuedPayloads(status);
}

Status
InferenceBackend::GenerateWarmupData(std::vector<WarmupData>* samples)
{
  samples->clear();
  for (const auto& warmup_setting : config_.model_warmup()) {
    LOG_VERBOSE(1) << "Generating warmup sample data for '"
                   << warmup_setting.name() << "'";

    // Two passes. First pass to get max byte size for synthetic data
    int64_t max_zero_byte_size = 0;
    int64_t max_random_byte_size = 0;
    for (const auto& input_meta : warmup_setting.inputs()) {
      auto element_count = GetElementCount(input_meta.second.dims());
      if (element_count == -1) {
        return Status(
            RequestStatusCode::INVALID_ARG,
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

    samples->emplace_back(warmup_setting.name(), warmup_setting.batch_size());
    auto& warmup_data = samples->back();
    // Create buffers for synthetic data
    TRTSERVER_Memory_Type type;
    int64_t type_id;
    warmup_data.zero_data_.reset(new AllocatedMemory(
        max_zero_byte_size, TRTSERVER_MEMORY_CPU_PINNED /* memory_type */,
        0 /* memory_type_id */));
    char* zero_buffer = warmup_data.zero_data_->MutableBuffer(&type, &type_id);
    memset(zero_buffer, 0, max_zero_byte_size);

    warmup_data.random_data_.reset(new AllocatedMemory(
        max_random_byte_size, TRTSERVER_MEMORY_CPU_PINNED /* memory_type */,
        0 /* memory_type_id */));
    char* random_buffer =
        warmup_data.random_data_->MutableBuffer(&type, &type_id);
    for (int64_t offset = 0; offset < max_random_byte_size; offset++) {
      random_buffer[offset] = rand();
    }

    // Use batch-1 for every request, batch size is simulated by populating
    // requests for single run.
    warmup_data.irequest_ = std::make_shared<InferenceRequest>();
    warmup_data.irequest_->SetModelName(Name());
    warmup_data.irequest_->SetRequestedModelVersion(Version());
    warmup_data.irequest_->SetBatchSize(1);
    warmup_data.irequest_->SetPriority(0);
    warmup_data.irequest_->SetTimeoutMicroseconds(0);

    // Request all outputs
    for (const auto& io : Config().output()) {
      RETURN_IF_ERROR(warmup_data.irequest_->RequestOutput(
          io.name(), 0 /* classification_cnt */));
    }

    // Second pass to prepare input buffer and request header
    for (const auto& input_meta : warmup_setting.inputs()) {
      std::vector<int64_t> input_meta_shape;
      for (auto d : input_meta.second.dims()) {
        input_meta_shape.push_back(d);
      }

      // If not found in model inputs, then treat it as control tensor
      const ModelInput* input_config;
      if (!GetInput(input_meta.first, &input_config).IsOk()) {
        if (warmup_data.input_override_ == nullptr) {
          warmup_data.input_override_ =
              std::make_shared<InferRequestProvider::InputOverrideMap>();
        }

        auto element_count = GetElementCount(input_meta_shape);
        auto batch_byte_size =
            element_count * GetDataTypeByteSize(input_meta.second.data_type());
        if (batch_byte_size == 0) {
          batch_byte_size = element_count * sizeof(int32_t);
        }

        InferRequestProvider::InputOverride value_override;
        value_override.dims_ = input_meta_shape;
        value_override.datatype_ = input_meta.second.data_type();
        switch (input_meta.second.input_data_type_case()) {
          case ModelWarmup_Input::InputDataTypeCase::kZeroData:
            value_override.content_.assign(
                zero_buffer, zero_buffer + batch_byte_size);
            break;
          case ModelWarmup_Input::InputDataTypeCase::kRandomData: {
            if (input_meta.second.data_type() == DataType::TYPE_STRING) {
              value_override.content_.assign(
                  zero_buffer, zero_buffer + batch_byte_size);
            } else {
              value_override.content_.assign(
                  random_buffer, random_buffer + batch_byte_size);
            }
            break;
          }
          case ModelWarmup_Input::InputDataTypeCase::kInputDataFile: {
            std::string input_data;
            RETURN_IF_ERROR(ReadTextFile(
                JoinPath({model_dir_, kWarmupDataFolder,
                          input_meta.second.input_data_file()}),
                &input_data));

            if (input_meta.second.data_type() == DataType::TYPE_STRING) {
              batch_byte_size = input_data.size();
            } else if (((size_t)batch_byte_size) > input_data.size()) {
              return Status(
                  RequestStatusCode::INVALID_ARG,
                  "warmup setting expects " + std::to_string(batch_byte_size) +
                      " bytes, but the data "
                      "provided from " +
                      input_meta.second.input_data_file() + "only has " +
                      std::to_string(input_data.size()) + " bytes");
            }

            value_override.content_.assign(
                input_data.data(), input_data.data() + input_data.size());
            break;
          }
          default:
            return Status(
                RequestStatusCode::INVALID_ARG,
                "warmup setting expects input '" + input_meta.first +
                    "' to have input_data_type set");
        }
        warmup_data.input_override_->emplace(input_meta.first, value_override);
        continue;
      }

      auto element_count = GetElementCount(input_meta_shape);
      auto batch_byte_size =
          element_count * GetDataTypeByteSize(input_meta.second.data_type());
      if (batch_byte_size == 0) {
        batch_byte_size = element_count * sizeof(int32_t);
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
          warmup_data.provided_data_.emplace_back();
          auto& input_data = warmup_data.provided_data_.back();
          RETURN_IF_ERROR(ReadTextFile(
              JoinPath({model_dir_, kWarmupDataFolder,
                        input_meta.second.input_data_file()}),
              &input_data));

          if (input_meta.second.data_type() == DataType::TYPE_STRING) {
            batch_byte_size = input_data.size();
          } else if (((size_t)batch_byte_size) > input_data.size()) {
            return Status(
                RequestStatusCode::INVALID_ARG,
                "warmup setting expects " + std::to_string(batch_byte_size) +
                    " bytes, but the data "
                    "provided from " +
                    input_meta.second.input_data_file() + "only has " +
                    std::to_string(input_data.size()) + " bytes");
          }
          allocated_ptr = input_data.data();
          break;
        }
        default:
          return Status(
              RequestStatusCode::INVALID_ARG,
              "warmup setting expects input '" + input_meta.first +
                  "' to have input_data_type set");
      }

      InferenceRequest::Input* input = nullptr;
      RETURN_IF_ERROR(warmup_data.irequest_->AddInput(
          input_meta.first, input_meta_shape, batch_byte_size, &input));
      RETURN_IF_ERROR(input->AppendData(
          allocated_ptr, batch_byte_size,
          TRTSERVER_MEMORY_CPU /* memory_type */, 0 /* memory_type_id */));
    }

    RETURN_IF_ERROR(warmup_data.irequest_->Normalize(*this));
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
