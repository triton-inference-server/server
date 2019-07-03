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
#include "src/core/server_status.pb.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferenceServer;
class InferenceBackend;
class ServerStatusManager;

/// An object to manage the model repository active in the server.
class ModelRepositoryManager {
 public:
  using VersionStateMap = std::map<int64_t, ModelReadyState>;
  using ModelStateMap = std::map<std::string, VersionStateMap>;

  enum ActionType { NO_ACTION, LOAD, UNLOAD };

  ~ModelRepositoryManager();

  /// Create a manager for a repository.
  /// \param server The pointer to the inference server.
  /// \param server_version The version of the inference server.
  /// \param status_manager The status manager that the model repository manager
  /// will update model configuration and state to.
  /// \param repositpory_path The file-system path of the repository.
  /// \param strict_model_config If false attempt to autofill missing required
  /// information in each model configuration.
  /// \param tf_gpu_memory_fraction The portion of GPU memory to be reserved
  /// for TensorFlow models.
  /// \param tf_allow_soft_placement If true instruct TensorFlow to use CPU
  /// implementation of an operation when a GPU implementation is not available
  /// \param polling_enabled If true, then PollAndUpdate() is allowed and
  /// LoadUnloadModel() is not allowed. If false, LoadUnloadModel() is allowed
  /// and PollAndUpdate() is not allowed.
  /// \return The error status.
  static Status Create(
      InferenceServer* server, const std::string& server_version,
      const std::shared_ptr<ServerStatusManager>& status_manager,
      const std::string& repository_path, const bool strict_model_config,
      const float tf_gpu_memory_fraction, const bool tf_allow_soft_placement,
      const bool polling_enabled,
      std::unique_ptr<ModelRepositoryManager>* model_repository_manager);

  /// Poll the model repository to determine the new set of models and
  /// compare with the current set. And serve the new set of models based
  /// on their version policy.
  Status PollAndUpdate();

  /// Load or unload a specified model.
  /// \parm model_name The name of the model to be loaded or unloaded
  /// \parm type The type action to be performed. If the action is LOAD and
  /// the model has been loaded, the model will be re-loaded.
  /// \param OnCompleteUpdate The callback function to be invoked once the
  /// action is completed.
  /// \return error status. Return "NOT_FOUND" if it tries to load
  /// a non-existing model or if it tries to unload a model that hasn't been
  /// loaded.
  Status LoadUnloadModel(
      const std::string& model_name, ActionType type,
      std::function<void(Status)> OnCompleteUpdate);

  /// Unload all models. This function should be called before shutting down
  /// the model repository manager.
  Status UnloadAllModels();

  /// \return the states of all versions of all live model backends.
  const ModelStateMap GetLiveBackendStates();

  /// ModelRepositoryManager is improved as it will manage the backends
  /// directly. \param model_name The model to get version states from. \return
  /// the states of all versions of the specified model backends.
  ///
  /// [TODO] Instead of providing this function for server status manager to
  /// poll version state, adding a mirror function in ServerStatusManager and
  /// publish the version state changes via that mirror function.
  const VersionStateMap GetVersionStates(const std::string& model_name);

  /// Obtain the specified backend.
  /// \param model_name The model name of the backend handle.
  /// \param model_version The model version of the backend handle.
  /// \param backend Return the inference backend object.
  /// \return error status.
  Status GetInferenceBackend(
      const std::string& model_name, const int64_t model_version,
      std::shared_ptr<InferenceBackend>* backend);

 private:
  struct ModelInfo;
  class BackendLifeCycle;

  // Map from model name to information about the model.
  using ModelInfoMap =
      std::unordered_map<std::string, std::unique_ptr<ModelInfo>>;

  ModelRepositoryManager(
      const std::shared_ptr<ServerStatusManager>& status_manager,
      const std::string& repository_path,
      const BackendConfigMap& backend_config_map, const bool autofill,
      const bool polling_enabled, std::unique_ptr<BackendLifeCycle> life_cycle);

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
  Status Poll(
      std::set<std::string>* added, std::set<std::string>* deleted,
      std::set<std::string>* modified, std::set<std::string>* unmodified);

  /// Update the configuration of newly added / modified model and serve
  /// the model based on its version policy.
  /// \param model_name The name of the model to be updated.
  /// \param is_added If the model is being added to the model repository.
  Status Update(const std::string& model_name, bool is_added);

  /// Get the configuration for a named model.
  /// \param name The model name.
  /// \param model_config Returns the model configuration.
  /// \return OK if found, NOT_FOUND otherwise.
  Status GetModelConfig(const std::string& name, ModelConfig* model_config);

  /// Get the list of versions to be loaded for a named model based on version
  /// policy.
  /// \param name The model name.
  /// \param model_config The model configuration.
  /// \param versions Returns the versions to be loaded
  /// \return The error status.
  Status VersionsToLoad(
      const std::string& name, const ModelConfig& model_config,
      std::vector<int64_t>& versions);

  const std::string repository_path_;
  const BackendConfigMap backend_config_map_;
  const bool autofill_;
  const bool polling_enabled_;

  std::mutex poll_mu_;
  std::mutex infos_mu_;
  ModelInfoMap infos_;

  std::shared_ptr<ServerStatusManager> status_manager_;

  std::unique_ptr<BackendLifeCycle> backend_life_cycle_;
};

}}  // namespace nvidia::inferenceserver
