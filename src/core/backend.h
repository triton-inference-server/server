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
#pragma once

#include "src/core/label_provider.h"
#include "src/core/metrics.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

class InferRequestProvider;
class InferResponseProvider;

//
// Interface for backends that handle inference requests.
//
class InferenceBackend {
 public:
  InferenceBackend() = default;
  virtual ~InferenceBackend() {}

  // Get the name of model being served.
  const std::string& Name() const { return config_.name(); }

  // Get the version of model being served.
  int64_t Version() const { return version_; }

  // Get the configuration of model being served.
  const ModelConfig& Config() const { return config_; }

  // Get the model configuration for a named input.
  tensorflow::Status GetInput(
      const std::string& name, const ModelInput** input) const;

  // Get the model configuration for a named output.
  tensorflow::Status GetOutput(
      const std::string& name, const ModelOutput** output) const;

  // Get a label provider for the model.
  const LabelProvider& GetLabelProvider() const { return label_provider_; }

  // Get the tags of model being served.
  const std::map<std::string, std::string>& Tags() const { return tags_; }

  // Run inference using the provided request to produce outputs in
  // the provide response. This method should be called by synchronous
  // frontends.
  void Run(
      std::shared_ptr<ModelInferStats> stats,
      std::shared_ptr<InferRequestProvider> request_provider,
      std::shared_ptr<InferResponseProvider> response_provider,
      std::function<void(tensorflow::Status)> OnCompleteHandleInfer);

  // Run inference using the provided request to produce outputs in
  // the provide response. This method should be called by
  // asynchronous frontends.
  void AsyncRun(
      std::shared_ptr<ModelInferStats> stats,
      std::shared_ptr<InferRequestProvider> request_provider,
      std::shared_ptr<InferResponseProvider> response_provider,
      std::function<void(tensorflow::Status)> OnCompleteHandleInfer);

  // Get a metric for the servable specialized for the given GPU index
  // (if -1 then return non-specialized version of the metric).
  prometheus::Counter& MetricInferenceSuccess(int gpu_device) const;
  prometheus::Counter& MetricInferenceFailure(int gpu_device) const;
  prometheus::Counter& MetricInferenceCount(int gpu_device) const;
  prometheus::Counter& MetricInferenceExecutionCount(int gpu_device) const;
  prometheus::Counter& MetricInferenceRequestDuration(int gpu_device) const;
  prometheus::Counter& MetricInferenceComputeDuration(int gpu_device) const;
  prometheus::Counter& MetricInferenceQueueDuration(int gpu_device) const;
  prometheus::Histogram& MetricInferenceLoadRatio(int gpu_device) const;

 protected:
  // Set the configuration of the model being served.
  tensorflow::Status SetModelConfig(
      const tensorflow::StringPiece& path, const ModelConfig& config);

  // Explicitly set the scheduler to use for inference requests to the
  // model. The scheduler can only be set once for a servable.
  tensorflow::Status SetScheduler(std::unique_ptr<Scheduler> scheduler);

  // Set the scheduler based on the model configuration. The scheduler
  // can only be set once for a servable.
  tensorflow::Status SetConfiguredScheduler(
      const uint32_t runner_cnt, Scheduler::StandardRunFunc OnRun);

 private:
  // Configuration of the model that this servable represents.
  ModelConfig config_;

  // Version of the model that this servable represents.
  int64_t version_;

  // Label provider for this model.
  LabelProvider label_provider_;

  // The scheduler to use for this servable.
  std::unique_ptr<Scheduler> scheduler_;

  // Map from input name to the model configuration for that input.
  std::unordered_map<std::string, ModelInput> input_map_;

  // Map from output name to the model configuration for that output.
  std::unordered_map<std::string, ModelOutput> output_map_;

  // Tags of the model that this servable represents.
  std::map<std::string, std::string> tags_;

  void GetMetricLabels(
      std::map<std::string, std::string>* labels, const int gpu_device) const;
  prometheus::Counter& GetCounterMetric(
      std::map<int, prometheus::Counter*>& metrics,
      prometheus::Family<prometheus::Counter>& family,
      const int gpu_device) const;

  mutable std::map<int, prometheus::Counter*> metric_inf_success_;
  mutable std::map<int, prometheus::Counter*> metric_inf_failure_;
  mutable std::map<int, prometheus::Counter*> metric_inf_count_;
  mutable std::map<int, prometheus::Counter*> metric_inf_exec_count_;
  mutable std::map<int, prometheus::Counter*> metric_inf_request_duration_us_;
  mutable std::map<int, prometheus::Counter*> metric_inf_compute_duration_us_;
  mutable std::map<int, prometheus::Counter*> metric_inf_queue_duration_us_;
  mutable std::map<int, prometheus::Histogram*> metric_inf_load_ratio_;
};

}}  // namespace nvidia::inferenceserver
