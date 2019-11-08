// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
#include "src/core/provider_utils.h"
#include "src/core/sequence_batch_scheduler.h"
#include "src/core/trtserver.h"

namespace nvidia { namespace inferenceserver {

namespace {

enum OverrideType { TYPE_STARTEND, TYPE_START, TYPE_END, TYPE_CONTINUE };

Status
CreateControlTensors(
    const ModelConfig& config, const InferRequestHeader& request_header,
    std::shared_ptr<InferRequestProvider::InputOverrideMap>* input_overrides)
{
  *input_overrides = std::make_shared<InferRequestProvider::InputOverrideMap>();

  // Create InputOverrideMap according to control flags in request header
  OverrideType override_type;
  if ((request_header.flags() & (InferRequestHeader::FLAG_SEQUENCE_START |
                                 InferRequestHeader::FLAG_SEQUENCE_END)) ==
      (InferRequestHeader::FLAG_SEQUENCE_START |
       InferRequestHeader::FLAG_SEQUENCE_END)) {
    override_type = OverrideType::TYPE_STARTEND;
  } else if (
      (request_header.flags() & InferRequestHeader::FLAG_SEQUENCE_START) != 0) {
    override_type = OverrideType::TYPE_START;
  } else if (
      (request_header.flags() & InferRequestHeader::FLAG_SEQUENCE_END) != 0) {
    override_type = OverrideType::TYPE_END;
  } else {
    override_type = OverrideType::TYPE_CONTINUE;
  }

  std::string tensor_name;
  DataType tensor_datatype;
  int32_t int32_false_value, int32_true_value;
  float fp32_false_value, fp32_true_value;

  // START
  {
    RETURN_IF_ERROR(GetSequenceControlProperties(
        config.sequence_batching(), config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        true /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value));

    auto value_override =
        std::make_shared<InferRequestProvider::InputOverride>();
    uint8_t* value_p = nullptr;
    switch (override_type) {
      // True
      case OverrideType::TYPE_START:
      case OverrideType::TYPE_STARTEND:
        value_p =
            ((tensor_datatype == DataType::TYPE_INT32)
                 ? reinterpret_cast<uint8_t*>(&int32_true_value)
                 : reinterpret_cast<uint8_t*>(&fp32_true_value));
        break;
      // False
      default:
        value_p =
            ((tensor_datatype == DataType::TYPE_INT32)
                 ? reinterpret_cast<uint8_t*>(&int32_false_value)
                 : reinterpret_cast<uint8_t*>(&fp32_false_value));
        break;
    }

    value_override->content_.assign(value_p, value_p + sizeof(float));
    value_override->dims_.Add(1);
    value_override->datatype_ = tensor_datatype;
    (*input_overrides)->insert(std::make_pair(tensor_name, value_override));
  }

  // END, optional
  {
    RETURN_IF_ERROR(GetSequenceControlProperties(
        config.sequence_batching(), config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        false /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value));
    if (!tensor_name.empty()) {
      auto value_override =
          std::make_shared<InferRequestProvider::InputOverride>();
      uint8_t* value_p = nullptr;
      switch (override_type) {
        // True
        case OverrideType::TYPE_END:
        case OverrideType::TYPE_STARTEND:
          value_p =
              ((tensor_datatype == DataType::TYPE_INT32)
                   ? reinterpret_cast<uint8_t*>(&int32_true_value)
                   : reinterpret_cast<uint8_t*>(&fp32_true_value));
          break;
        // False
        default:
          value_p =
              ((tensor_datatype == DataType::TYPE_INT32)
                   ? reinterpret_cast<uint8_t*>(&int32_false_value)
                   : reinterpret_cast<uint8_t*>(&fp32_false_value));
          break;
      }

      value_override->content_.assign(value_p, value_p + sizeof(float));
      value_override->dims_.Add(1);
      value_override->datatype_ = tensor_datatype;
      (*input_overrides)->insert(std::make_pair(tensor_name, value_override));
    }
  }

  // READY
  {
    RETURN_IF_ERROR(GetSequenceControlProperties(
        config.sequence_batching(), config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY,
        true /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value));

    auto value_override =
        std::make_shared<InferRequestProvider::InputOverride>();
    uint8_t* value_p =
        ((tensor_datatype == DataType::TYPE_INT32)
             ? reinterpret_cast<uint8_t*>(&int32_true_value)
             : reinterpret_cast<uint8_t*>(&fp32_true_value));

    value_override->content_.assign(value_p, value_p + sizeof(float));
    value_override->dims_.Add(1);
    value_override->datatype_ = tensor_datatype;
    (*input_overrides)->insert(std::make_pair(tensor_name, value_override));
  }

  return Status::Success;
}

}  // namespace

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
    const uint32_t runner_cnt, Scheduler::StandardInitFunc OnInit,
    Scheduler::StandardRunFunc OnRun)
{
  std::unique_ptr<Scheduler> scheduler;

  // If 'sequence_batching' is configured use the SequenceBatchScheduler,
  // otherwise use the default DynamicBatchScheduler.
  if (config_.has_sequence_batching()) {
    RETURN_IF_ERROR(SequenceBatchScheduler::Create(
        config_, runner_cnt, OnInit, OnRun, &scheduler));
  } else {
    RETURN_IF_ERROR(DynamicBatchScheduler::Create(
        config_, runner_cnt, OnInit, OnRun, &scheduler));
  }

  return SetScheduler(std::move(scheduler));
}

Status
InferenceBackend::Init(
    const std::string& path, const ModelConfig& config,
    const std::string& platform)
{
  RETURN_IF_ERROR(ValidateModelConfig(config, platform));
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

  // Stop queue timer and start compute timer when the payload is
  // scheduled to run
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeStart);
      payload.stats_->SetGPUDevice(contexts_[runner_idx]->gpu_device_);
    }
  }

  Status status = contexts_[runner_idx]->Run(this, payloads);

  // Stop compute timers.
  for (auto& payload : *payloads) {
    if (payload.stats_ != nullptr) {
      payload.stats_->CaptureTimestamp(
          ModelInferStats::TimestampKind::kComputeEnd);
    }
  }

  OnCompleteQueuedPayloads(status);
}

Status
InferenceBackend::WarmUp()
{
  LOG_VERBOSE(1) << "warming up model '" << Name() << "' with sample request";

  static std::string warmup_data_folder = "sample";

  // Request header and input data can be produced for all contexts,
  // only providers need to be unique for every context.
  auto request_header = config_.model_warm_up().request_header();
  RETURN_IF_ERROR(NormalizeRequestHeader(*this, request_header));

  std::unordered_map<std::string, std::shared_ptr<Memory>> input_buffer;

  // Placeholder for input data
  std::unique_ptr<AllocatedSystemMemory> zero_buffer;
  std::vector<std::string> provided_data;

  if (config_.model_warm_up().use_zero_value()) {
    // Allocate large enough zero buffer, and used by all inputs.
    size_t max_byte_size = 0;
    for (const auto& input : request_header.input()) {
      max_byte_size = std::max(input.batch_byte_size(), max_byte_size);
    }
    zero_buffer.reset(new AllocatedSystemMemory(
        max_byte_size, TRTSERVER_MEMORY_CPU /* memory_type */,
        0 /* memory_type_id */));
    TRTSERVER_Memory_Type type;
    int64_t type_id;
    char* allocated_buffer =
        static_cast<AllocatedSystemMemory*>(zero_buffer.get())
            ->MutableBuffer(&type, &type_id);
    memset(allocated_buffer, 0, max_byte_size);

    for (const auto& input : request_header.input()) {
      auto pr = input_buffer.emplace(input.name(), nullptr);
      pr.first->second.reset(new MemoryReference());
      static_cast<MemoryReference*>(pr.first->second.get())
          ->AddBuffer(
              allocated_buffer, input.batch_byte_size(),
              TRTSERVER_MEMORY_CPU /* memory_type */, 0 /* memory_type_id */);
    }
  } else {
    for (const auto& input : request_header.input()) {
      provided_data.emplace_back();
      auto& input_data = provided_data.back();
      RETURN_IF_ERROR(ReadTextFile(
          JoinPath({model_dir_, warmup_data_folder, input.name()}),
          &input_data));

      auto pr = input_buffer.emplace(input.name(), nullptr);
      pr.first->second.reset(new MemoryReference());
      static_cast<MemoryReference*>(pr.first->second.get())
          ->AddBuffer(
              input_data.data(), input_data.size(),
              TRTSERVER_MEMORY_CPU /* memory_type */, 0 /* memory_type_id */);
    }
  }

  // Input override for model that uses sequence batcher
  std::shared_ptr<InferRequestProvider::InputOverrideMap> input_overrides;
  if (config_.has_sequence_batching()) {
    RETURN_IF_ERROR(
        CreateControlTensors(config_, request_header, &input_overrides));
  }

  std::vector<std::future<Status>> warmup_res;
  for (auto& context : contexts_) {
    warmup_res.emplace_back(std::async(
        std::launch::async,
        [this, &request_header, &input_buffer,
         &input_overrides](BackendContext* context) {
          std::shared_ptr<InferRequestProvider> request_provider;
          RETURN_IF_ERROR(InferRequestProvider::Create(
              config_.name(), version_, request_header, input_buffer,
              &request_provider));
          RETURN_IF_ERROR(request_provider->SetInputOverride(input_overrides));

          std::vector<Scheduler::Payload> payloads;
          // Only request provider is required for triggering model run
          payloads.emplace_back(nullptr, request_provider, nullptr, nullptr);

          RETURN_IF_ERROR(context->Run(this, &payloads));
          return Status::Success;
        },
        context.get()));
  }

  for (auto& res_future : warmup_res) {
    auto status = res_future.get();
    if (!status.IsOk()) {
      return Status(
          RequestStatusCode::INTERNAL,
          "failed to warm up model '" + Name() + "': " + status.AsString());
    }
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
