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
#include "src/core/constants.h"
#include "src/core/dynamic_batch_scheduler.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/metric_model_reporter.h"
#include "src/core/model_config_utils.h"
#include "src/core/sequence_batch_scheduler.h"
#include "tensorflow/core/lib/io/path.h"

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
InferenceBackend::SetInferenceServer(void* inference_server)
{
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
  const auto model_dir = DirName(path);
  for (const auto& io : config.output()) {
    output_map_.insert(std::make_pair(io.name(), io));

    if (!io.label_filename().empty()) {
      const auto label_path = JoinPath({model_dir, io.label_filename()});
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

void
InferenceBackend::Run(
    std::shared_ptr<ModelInferStats> stats,
    std::shared_ptr<InferRequestProvider> request_provider,
    std::shared_ptr<InferResponseProvider> response_provider,
    std::function<void(Status)> OnCompleteHandleInfer)
{
  scheduler_->Enqueue(
      stats, request_provider, response_provider, OnCompleteHandleInfer);
}

}}  // namespace nvidia::inferenceserver
