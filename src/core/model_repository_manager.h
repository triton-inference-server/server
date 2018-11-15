// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
//
#pragma once

#include <mutex>
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/config/model_server_config.pb.h"

namespace tfs = tensorflow::serving;

namespace nvidia { namespace inferenceserver {

// A singleton to manage the model repository active in the server. A
// singleton is used because the servables have no connection to the
// server itself but they need to have access to the configuration.
class ModelRepositoryManager {
 public:
  // Map from model name to a model configuration.
  using ModelConfigMap = std::unordered_map<std::string, ModelConfig>;

  // Get the configuration for a named model. Return OK if found,
  // NOT_FOUND otherwise.
  static tensorflow::Status GetModelConfig(
    const std::string& name, ModelConfig* model_config);

  // Get the platform for a named model. Return OK if found, NO_FOUND
  // otherwise.
  static tensorflow::Status GetModelPlatform(
    const std::string& name, Platform* platform);

  // Set the model configurations, removing any existing model
  // configurations.
  static tensorflow::Status SetModelConfigs(
    const ModelConfigMap& model_configs);

  // Read the model configurations from all models in a model
  // repository.
  static tensorflow::Status ReadModelConfigs(
    const std::string& model_store_path, const bool autofill,
    ModelConfigMap* model_configs, tfs::ModelServerConfig* tfs_model_configs);

  static bool CompareModelConfigs(
    const ModelConfigMap& next, std::set<std::string>* added,
    std::set<std::string>* removed);

 private:
  ModelRepositoryManager() = default;
  ~ModelRepositoryManager() = default;
  static ModelRepositoryManager* GetSingleton();
  tensorflow::Status GetModelConfigInternal(
    const std::string& name, ModelConfig* model_config);

  std::mutex mu_;
  ModelConfigMap configs_;
  std::map<std::string, Platform> platforms_;
};

}}  // namespace nvidia::inferenceserver
