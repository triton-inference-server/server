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
#pragma once

#include <memory>
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"
#include "src/core/provider.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferenceServer;

struct EnsembleInfo {
  struct StepInfo {
    StepInfo(const std::string& model_name, const int64_t model_version)
        : model_name_(model_name), model_version_(model_version)
    {
    }

    std::string model_name_;
    int64_t model_version_;
    std::unordered_map<std::string, std::string> input_to_tensor_;
    std::unordered_map<std::string, std::string> output_to_tensor_;
  };

  std::string ensemble_name_;

  bool allow_batching_;

  // the ensemble output (re)shape expected by the ensemble
  std::unordered_map<std::string, DimsList> ensemble_output_shape_;

  std::vector<StepInfo> steps_;

  // Only include a step if the ensemble tensor is used as input in that step
  std::unordered_map<std::string, std::set<size_t>> tensor_to_step_;
};

// Scheduler that implements ensemble scheduling.
class EnsembleScheduler : public Scheduler {
 public:
  // Create a scheduler to process ensemble requests and
  // to dispatch requests to models in ensemble internally.
  static Status Create(
      InferenceServer* const server, const ModelConfig& config,
      std::unique_ptr<Scheduler>* scheduler);

  // \see Scheduler::Enqueue()
  void Enqueue(
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(const Status&)> OnComplete) override;

 private:
  EnsembleScheduler(InferenceServer* const server, const ModelConfig& config);

  InferenceServer* const is_;

  // Ensemble information that is built from model config
  std::unique_ptr<EnsembleInfo> info_;
};

}}  // namespace nvidia::inferenceserver
