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
#include "src/core/logging.h"
#include "src/core/sequence_batch_scheduler.h"
#include "src/core/utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

void
InferenceBackend::GetMetricLabels(
    std::map<std::string, std::string>* labels, const int gpu_device) const
{
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelName), Name()));
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelVersion), std::to_string(Version())));
  for (const auto& tag : Tags()) {
    labels->insert(std::map<std::string, std::string>::value_type(
        "_" + tag.first, tag.second));
  }

  // 'gpu_device' can be -1 to indicate that the GPU is not known. In
  // that case use a metric that doesn't have the gpu_uuid label.
  if (gpu_device >= 0) {
    std::string uuid;
    if (Metrics::UUIDForCudaDevice(gpu_device, &uuid)) {
      labels->insert(std::map<std::string, std::string>::value_type(
          std::string(kMetricsLabelGpuUuid), uuid));
    }
  }
}

prometheus::Counter&
InferenceBackend::GetCounterMetric(
    std::map<int, prometheus::Counter*>& metrics,
    prometheus::Family<prometheus::Counter>& family, const int gpu_device) const
{
  const auto itr = metrics.find(gpu_device);
  if (itr != metrics.end()) {
    return *(itr->second);
  }

  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, gpu_device);

  prometheus::Counter& counter = family.Add(labels);
  metrics.insert(
      std::map<int, prometheus::Counter*>::value_type(gpu_device, &counter));
  return counter;
}

prometheus::Counter&
InferenceBackend::MetricInferenceSuccess(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_success_, Metrics::FamilyInferenceSuccess(), gpu_device);
}

prometheus::Counter&
InferenceBackend::MetricInferenceFailure(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_failure_, Metrics::FamilyInferenceFailure(), gpu_device);
}

prometheus::Counter&
InferenceBackend::MetricInferenceCount(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_count_, Metrics::FamilyInferenceCount(), gpu_device);
}

prometheus::Counter&
InferenceBackend::MetricInferenceExecutionCount(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_exec_count_, Metrics::FamilyInferenceExecutionCount(),
      gpu_device);
}

prometheus::Counter&
InferenceBackend::MetricInferenceRequestDuration(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_request_duration_us_,
      Metrics::FamilyInferenceRequestDuration(), gpu_device);
}

prometheus::Counter&
InferenceBackend::MetricInferenceComputeDuration(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_compute_duration_us_,
      Metrics::FamilyInferenceComputeDuration(), gpu_device);
}

prometheus::Counter&
InferenceBackend::MetricInferenceQueueDuration(int gpu_device) const
{
  return GetCounterMetric(
      metric_inf_queue_duration_us_, Metrics::FamilyInferenceQueueDuration(),
      gpu_device);
}

prometheus::Histogram&
InferenceBackend::MetricInferenceLoadRatio(int gpu_device) const
{
  const auto itr = metric_inf_load_ratio_.find(gpu_device);
  if (itr != metric_inf_load_ratio_.end()) {
    return *(itr->second);
  }

  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, gpu_device);

  prometheus::Histogram& hist = Metrics::FamilyInferenceLoadRatio().Add(
      labels, std::vector<double>{1.05, 1.10, 1.25, 1.5, 2.0, 10.0, 50.0});
  metric_inf_load_ratio_.insert(
      std::map<int, prometheus::Histogram*>::value_type(gpu_device, &hist));
  return hist;
}

tensorflow::Status
InferenceBackend::GetInput(
    const std::string& name, const ModelInput** input) const
{
  const auto itr = input_map_.find(name);
  if (itr == input_map_.end()) {
    return tensorflow::errors::InvalidArgument(
        "unexpected inference input '", name, "' for model '", Name(), "'");
  }

  *input = &itr->second;
  return tensorflow::Status::OK();
}

tensorflow::Status
InferenceBackend::GetOutput(
    const std::string& name, const ModelOutput** output) const
{
  const auto itr = output_map_.find(name);
  if (itr == output_map_.end()) {
    return tensorflow::errors::InvalidArgument(
        "unexpected inference output '", name, "' for model '", Name(), "'");
  }

  *output = &itr->second;
  return tensorflow::Status::OK();
}

tensorflow::Status
InferenceBackend::SetModelConfig(
    const tensorflow::StringPiece& path, const ModelConfig& config)
{
  config_ = config;
  TF_RETURN_IF_ERROR(GetModelVersionFromPath(path, &version_));
  for (const auto& tag : config_.metric_tags()) {
    tags_.insert(
        std::map<std::string, std::string>::value_type(tag.first, tag.second));
  }

  // Initialize the input map
  for (const auto& io : config.input()) {
    input_map_.insert(std::make_pair(io.name(), io));
  }

  // Initialize the output map and label provider for each output
  const auto model_dir = tensorflow::io::Dirname(path);
  for (const auto& io : config.output()) {
    output_map_.insert(std::make_pair(io.name(), io));

    if (!io.label_filename().empty()) {
      const auto label_path =
          tensorflow::io::JoinPath(model_dir, io.label_filename());
      TF_RETURN_IF_ERROR(label_provider_.AddLabels(io.name(), label_path));
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
InferenceBackend::SetScheduler(std::unique_ptr<Scheduler> scheduler)
{
  if (scheduler_ != nullptr) {
    return tensorflow::errors::Internal(
        "Attempt to change scheduler not allowed");
  }

  scheduler_ = std::move(scheduler);
  return tensorflow::Status::OK();
}

tensorflow::Status
InferenceBackend::SetConfiguredScheduler(
    const uint32_t runner_cnt, Scheduler::StandardRunFunc OnRun)
{
  std::unique_ptr<Scheduler> scheduler;

  // If 'sequence_batching' is configured use the SequenceBatchScheduler,
  // otherwise use the default DynamicBatchScheduler.
  if (config_.has_sequence_batching()) {
    TF_RETURN_IF_ERROR(
        SequenceBatchScheduler::Create(config_, runner_cnt, OnRun, &scheduler));
  } else {
    TF_RETURN_IF_ERROR(
        DynamicBatchScheduler::Create(config_, runner_cnt, OnRun, &scheduler));
  }

  return SetScheduler(std::move(scheduler));
}

void
InferenceBackend::AsyncRun(
    std::shared_ptr<ModelInferStats> stats,
    std::shared_ptr<InferRequestProvider> request_provider,
    std::shared_ptr<InferResponseProvider> response_provider,
    std::function<void(tensorflow::Status)> OnCompleteHandleInfer)
{
  scheduler_->Enqueue(
      stats, request_provider, response_provider, OnCompleteHandleInfer);
}

// Since callers are expecting synchronous behavior, this function
// must wait until the request is processed and the response is
// returned. This function can be simplified significantly once we
// have [DLIS-124].
void
InferenceBackend::Run(
    std::shared_ptr<ModelInferStats> stats,
    std::shared_ptr<InferRequestProvider> request_provider,
    std::shared_ptr<InferResponseProvider> response_provider,
    std::function<void(tensorflow::Status)> OnCompleteHandleInfer)
{
  std::mutex lmu;
  std::condition_variable lcv;
  tensorflow::Status run_status;
  bool run_completed = false;

  // Add request to queue...
  {
    scheduler_->Enqueue(
        stats, request_provider, response_provider,
        [&lmu, &lcv, &run_status, &run_completed](tensorflow::Status status) {
          // signal complete and propagate status
          {
            std::lock_guard<std::mutex> lk(lmu);
            run_status = status;
            run_completed = true;
          }
          lcv.notify_one();
        });
  }

  // [DLIS-124] must wait for request to indicate complete...
  {
    std::chrono::seconds wait_timeout(1);
    std::unique_lock<std::mutex> lk(lmu);
    while (!run_completed) {
      lcv.wait_for(lk, wait_timeout);
    }
  }

  OnCompleteHandleInfer(run_status);
}

}}  // namespace nvidia::inferenceserver
