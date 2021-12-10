// Copyright 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "model_config.pb.h"
#include "src/core/infer_stats.h"
#include "src/core/label_provider.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferenceRequest;

//
// Interface for models that handle inference requests.
//
class Model {
 public:
  explicit Model(
      const double min_compute_capability, const std::string& model_dir,
      const int64_t version, const inference::ModelConfig& config)
      : config_(config), min_compute_capability_(min_compute_capability),
        version_(version), required_input_count_(0), model_dir_(model_dir)
  {
  }
  virtual ~Model() {}

  // Get the name of model being served.
  const std::string& Name() const { return config_.name(); }

  // Get the version of model being served.
  int64_t Version() const { return version_; }

  // Get the configuration of model being served.
  const inference::ModelConfig& Config() const { return config_; }

  // Get the number of required inputs
  size_t RequiredInputCount() const { return required_input_count_; }

  // Get the stats collector for the model being served.
  InferenceStatsAggregator* MutableStatsAggregator()
  {
    return &stats_aggregator_;
  }
  const InferenceStatsAggregator& StatsAggregator() const
  {
    return stats_aggregator_;
  }

  // Get the model configuration for a named input.
  Status GetInput(
      const std::string& name, const inference::ModelInput** input) const;

  // Get the model configuration for a named output.
  Status GetOutput(
      const std::string& name, const inference::ModelOutput** output) const;

  // Get a label provider for the model.
  const std::shared_ptr<LabelProvider>& GetLabelProvider() const
  {
    return label_provider_;
  }

  // Initialize the instance for Triton core usage
  Status Init();

  // Enqueue a request for execution. If Status::Success is returned
  // then the model has taken ownership of the request object and so
  // 'request' will be nullptr. If non-success is returned then the
  // caller still retains ownership of 'request'.
  Status Enqueue(std::unique_ptr<InferenceRequest>& request)
  {
    return scheduler_->Enqueue(request);
  }

  uint32_t DefaultPriorityLevel() const { return default_priority_level_; }

  uint32_t MaxPriorityLevel() const { return max_priority_level_; }

 protected:
  // Set the configuration of the model being served.
  Status SetModelConfig(const inference::ModelConfig& config);

  // Explicitly set the scheduler to use for inference requests to the
  // model. The scheduler can only be set once for a model.
  Status SetScheduler(std::unique_ptr<Scheduler> scheduler);

  // The scheduler to use for this model.
  std::unique_ptr<Scheduler> scheduler_;

  // Configuration of the model.
  inference::ModelConfig config_;

 private:
  // The minimum supported CUDA compute capability.
  const double min_compute_capability_;

  // Version of the model.
  int64_t version_;

  // The stats collector for the model.
  InferenceStatsAggregator stats_aggregator_;

  // Label provider for this model.
  std::shared_ptr<LabelProvider> label_provider_;

  size_t required_input_count_;

  // Map from input name to the model configuration for that input.
  std::unordered_map<std::string, inference::ModelInput> input_map_;

  // Map from output name to the model configuration for that output.
  std::unordered_map<std::string, inference::ModelOutput> output_map_;

  // Path to model
  std::string model_dir_;

  // The default priority level for the model.
  uint32_t default_priority_level_;

  // The largest priority value for the model.
  uint32_t max_priority_level_;
};

}}  // namespace nvidia::inferenceserver
