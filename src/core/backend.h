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
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferRequestProvider;
class InferResponseProvider;
class MetricModelReporter;

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

  // Get the metric reportoer for the model being served.
  const std::shared_ptr<MetricModelReporter>& MetricReporter() const
  {
    return metric_reporter_;
  }

  // Get the model configuration for a named input.
  Status GetInput(const std::string& name, const ModelInput** input) const;

  // Get the model configuration for a named output.
  Status GetOutput(const std::string& name, const ModelOutput** output) const;

  // Get a label provider for the model.
  const std::shared_ptr<LabelProvider>& GetLabelProvider() const
  {
    return label_provider_;
  }

  // Run inference using the provided request to produce outputs in the provide
  // response. The inference will run asynchronously and "OnCompleteHandleInfer"
  // callback will be called once the inference is completed
  void Run(
      std::shared_ptr<ModelInferStats> stats,
      std::shared_ptr<InferRequestProvider> request_provider,
      std::shared_ptr<InferResponseProvider> response_provider,
      std::function<void(const Status&)> OnCompleteHandleInfer);

 protected:
  // Set the configuration of the model being served.
  Status SetModelConfig(const std::string& path, const ModelConfig& config);

  // Explicitly set the scheduler to use for inference requests to the
  // model. The scheduler can only be set once for a backend.
  Status SetScheduler(std::unique_ptr<Scheduler> scheduler);

  // Set the scheduler based on the model configuration. The scheduler
  // can only be set once for a backend.
  Status SetConfiguredScheduler(
      const uint32_t runner_cnt, Scheduler::StandardInitFunc OnInit,
      Scheduler::StandardRunFunc OnRun);

  // Get the raw pointer to the scheduler of this backend.
  Scheduler* BackendScheduler() { return scheduler_.get(); }

 private:
  // Configuration of the model that this backend represents.
  ModelConfig config_;

  // Version of the model that this backend represents.
  int64_t version_;

  // The metric reporter for the model that this backend represents.
  std::shared_ptr<MetricModelReporter> metric_reporter_;

  // Label provider for this model.
  std::shared_ptr<LabelProvider> label_provider_;

  // The scheduler to use for this backend.
  std::unique_ptr<Scheduler> scheduler_;

  // Map from input name to the model configuration for that input.
  std::unordered_map<std::string, ModelInput> input_map_;

  // Map from output name to the model configuration for that output.
  std::unordered_map<std::string, ModelOutput> output_map_;
};

}}  // namespace nvidia::inferenceserver
