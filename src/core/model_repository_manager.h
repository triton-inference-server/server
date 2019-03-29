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
//
#pragma once

#include <mutex>
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/status.h"

namespace tensorflow { namespace serving {
class ModelConfig;
}}  // namespace tensorflow::serving
namespace tfs = tensorflow::serving;

namespace nvidia { namespace inferenceserver {

/// A singleton to manage the model repository active in the server. A
/// singleton is used because the servables have no connection to the
/// server itself but they need to have access to the configuration.
class ModelRepositoryManager {
 public:
  /// Create a manager for a repository.
  /// \param repositpory_path The file-system path of the repository.
  /// \param platform_config_map Map from platform name to the backend
  /// configuration for that platform.
  /// \param autofill If true attempt to autofill missing required
  /// information in each model configuration.
  /// \return The error status.
  static Status Create(
      const std::string& repository_path,
      const PlatformConfigMap& platform_config_map, const bool autofill);

  /// Poll the model repository to determine the new set of models and
  /// compare with the current set. Return the additions, deletions,
  /// and modifications that have occurred since the last Poll().
  /// \param added The names of the models added to the repository.
  /// \param deleted The names of the models removed from the repository.
  /// \param modified The names of the models remaining in the
  /// repository that have been changed.
  /// \param unmodified The names of the models remaining in the
  /// repository that have not changed.
  /// \return The error status.
  static Status Poll(
      std::set<std::string>* added, std::set<std::string>* deleted,
      std::set<std::string>* modified, std::set<std::string>* unmodified);

  /// Get the configuration for a named model.
  /// \param name The model name.
  /// \param model_config Returns the model configuration.
  /// \return OK if found, NOT_FOUND otherwise.
  static Status GetModelConfig(
      const std::string& name, ModelConfig* model_config);

  /// Get TFS-style configuration for a named model.
  /// \param name The model name.
  /// \param tfs_model_config Returns the TFS-style model configuration.
  /// \return OK if found, NOT_FOUND otherwise.
  static Status GetTFSModelConfig(
      const std::string& name, tfs::ModelConfig* tfs_model_config);

  /// Get the platform for a named model.
  /// \param name The model name.
  /// \param platform Returns the Platform.
  /// \return OK if found, NOT_FOUND otherwise.
  static Status GetModelPlatform(const std::string& name, Platform* platform);

 private:
  struct ModelInfo;

  // Map from model name to information about the model.
  using ModelInfoMap =
      std::unordered_map<std::string, std::unique_ptr<ModelInfo>>;

  ModelRepositoryManager(
      const std::string& repository_path,
      const PlatformConfigMap& platform_config_map, const bool autofill);
  ~ModelRepositoryManager() = default;

  static ModelRepositoryManager* singleton;

  const std::string repository_path_;
  const PlatformConfigMap platform_config_map_;
  const bool autofill_;

  std::mutex poll_mu_;
  std::mutex infos_mu_;
  ModelInfoMap infos_;
};

}}  // namespace nvidia::inferenceserver
