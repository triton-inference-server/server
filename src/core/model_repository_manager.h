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

#include <functional>
#include <map>
#include <mutex>
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferenceServer;
class InferenceBackend;

/// Predefined reason strings
#define MODEL_READY_REASON_DUPLICATE "model appears in two or more repositories"

/// Readiness status for models.
enum class ModelReadyState {
  // The model is in an unknown state. The model is not available for
  // inferencing.
  UNKNOWN,

  // The model is ready and available for inferencing.
  READY,

  // The model is unavailable, indicating that the model failed to
  // load or has been implicitly or explicitly unloaded. The model is
  // not available for inferencing.
  UNAVAILABLE,

  // The model is being loaded by the inference server. The model is
  // not available for inferencing.
  LOADING,

  // The model is being unloaded by the inference server. The model is
  // not available for inferencing.
  UNLOADING
};

/// Get the string representation for a ModelReadyState
const std::string& ModelReadyStateString(ModelReadyState state);

/// An object to manage the model repository active in the server.
class ModelRepositoryManager {
 public:
  using VersionStateMap =
      std::map<int64_t, std::pair<ModelReadyState, std::string>>;
  using ModelStateMap = std::map<std::string, VersionStateMap>;

  // Index information for a model.
  struct ModelIndex {
    ModelIndex(const std::string& n)
        : name_only_(true), name_(n), version_(-1),
          state_(ModelReadyState::UNKNOWN)
    {
    }
    ModelIndex(
        const std::string& n, const int64_t v, const ModelReadyState s,
        const std::string& r)
        : name_only_(false), name_(n), version_(v), state_(s), reason_(r)
    {
    }
    const bool name_only_;
    const std::string name_;
    const int64_t version_;
    const ModelReadyState state_;
    const std::string reason_;
  };

  enum ActionType { NO_ACTION, LOAD, UNLOAD };

  /// A basic unit in dependency graph that records the models seen by the model
  /// repository manager.
  struct DependencyNode {
    DependencyNode(const std::string& model_name)
        : model_name_(model_name), status_(Status::Success), checked_(false)
    {
    }

    std::string model_name_;
    Status status_;
    bool checked_;
    inference::ModelConfig model_config_;
    std::set<int64_t> loaded_versions_;
    std::set<DependencyNode*> missing_upstreams_;
    std::unordered_map<DependencyNode*, std::set<int64_t>> upstreams_;
    std::set<DependencyNode*> downstreams_;
  };

  ~ModelRepositoryManager();

  /// Create a manager for a repository.
  /// \param server The pointer to the inference server.
  /// \param server_version The version of the inference server.
  /// \param repositpory_paths A set of file-system paths of the repositories.
  /// \param startup_models A set of models to be loaded at startup
  /// if model control is enabled.
  /// \param strict_model_config If false attempt to autofill missing required
  /// information in each model configuration.
  /// \param backend_cmdline_config_map The backend configuration setting
  /// specified on the command-line.
  /// \param tf_gpu_memory_fraction The portion of GPU memory to be reserved
  /// for TensorFlow models.
  /// \param tf_allow_soft_placement If true instruct TensorFlow to use CPU
  /// implementation of an operation when a GPU implementation is not available
  /// \param polling_enabled If true, then PollAndUpdate() is allowed.
  /// Otherwise, it is not allowed.
  /// \param model_control_enabled If true, then LoadUnloadModel() is allowed
  /// and the models in the model repository will not be loaded at startup.
  /// Otherwise, LoadUnloadModel() is not allowed and the models will be loaded.
  /// Cannot be set to true if polling_enabled is true.
  /// \param min_compute_capability The minimum support CUDA compute
  /// capability.
  /// \param model_repository_manager Return the model repository manager.
  /// \return The error status.
  static Status Create(
      InferenceServer* server, const std::string& server_version,
      const std::set<std::string>& repository_paths,
      const std::set<std::string>& startup_models,
      const bool strict_model_config,
      const BackendCmdlineConfigMap& backend_cmdline_config_map,
      const float tf_gpu_memory_fraction, const bool tf_allow_soft_placement,
      const bool polling_enabled, const bool model_control_enabled,
      const double min_compute_capability,
      std::unique_ptr<ModelRepositoryManager>* model_repository_manager);

  /// Poll the model repository to determine the new set of models and
  /// compare with the current set. And serve the new set of models based
  /// on their version policy.
  Status PollAndUpdate();

  /// Load or unload a specified model.
  /// \parm model_name The name of the model to be loaded or unloaded
  /// \parm type The type action to be performed. If the action is LOAD and
  /// the model has been loaded, the model will be re-loaded.
  /// \return error status. Return "NOT_FOUND" if it tries to load
  /// a non-existing model or if it tries to unload a model that hasn't been
  /// loaded.
  Status LoadUnloadModel(const std::string& model_name, ActionType type);

  /// Unload all models. This function should be called before shutting down
  /// the model repository manager.
  Status UnloadAllModels();

  /// \param strict_readiness If true, only models that have at least one
  /// ready version will be considered as live. Otherwise, the models that
  /// have loading / unloading versions will also be live.
  /// \return the state of all versions of all live models.
  const ModelStateMap LiveBackendStates(bool strict_readiness = false);

  /// \return the state of all versions of all models that have every
  /// been (attempted) loaded over the lifetime of the server.
  const ModelStateMap BackendStates();

  /// \return the states of all versions of a specific model.
  const VersionStateMap VersionStates(const std::string& model_name);

  /// \return the ready-state of a specific model version.
  Status ModelState(
      const std::string& model_name, const int64_t model_version,
      ModelReadyState* state);

  /// Get the index of all models in all repositories.
  /// \param ready_only If true return only index of models that are ready.
  /// \param index Returns the index.
  /// \return error status.
  Status RepositoryIndex(const bool ready_only, std::vector<ModelIndex>* index);

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

  // Set of DependencyNode
  using NodeSet = std::set<DependencyNode*>;

  ModelRepositoryManager(
      const std::set<std::string>& repository_paths,
      const BackendConfigMap& backend_config_map, const bool autofill,
      const bool polling_enabled, const bool model_control_enabled,
      const double min_compute_capability,
      std::unique_ptr<BackendLifeCycle> life_cycle);

  /// The internal function that are called in Create() and PollAndUpdate().
  Status PollAndUpdateInternal(bool* all_models_polled);

  /// The internal function that load or unload a set of models.
  Status LoadUnloadModels(
      const std::set<std::string>& models, ActionType type,
      bool* all_models_polled);

  /// Poll the requested models in the model repository and
  /// compare with the current set. Return the additions, deletions,
  /// and modifications that have occurred. This function will not updated
  /// the current model info, it is caller's responsibility to do so.
  /// \param models The set of models to be polled
  /// \param added The names of the models added to the repository.
  /// \param deleted The names of the models removed from the repository.
  /// \param modified The names of the models remaining in the
  /// repository that have been changed.
  /// \param unmodified The names of the models remaining in the
  /// repository that have not changed.
  /// \param updated_infos The model infos retrieved from the poll.
  /// \param all_models_polled Return true if all models are polled and
  /// their model configuration are validated successfully. Instead of aborting
  /// the polling, the models that fail will be ignored and their model infos
  /// will stay in the previous state.
  /// \return The error status.
  Status Poll(
      const std::set<std::string>& models, std::set<std::string>* added,
      std::set<std::string>* deleted, std::set<std::string>* modified,
      std::set<std::string>* unmodified, ModelInfoMap* updated_infos,
      bool* all_models_polled);

  /// Load models based on the dependency graph. The function will iteratively
  /// load models that all the models they depend on has been loaded, and unload
  /// models if their dependencies are no longer satisfied.
  /// \return The error status.
  Status LoadModelByDependency();

  /// Helper function to update the dependency graph based on the poll result
  /// \param added The names of the models added to the repository.
  /// \param deleted The names of the models removed from the repository.
  /// \param modified The names of the models remaining in the
  /// repository that have been changed.
  /// \return The error status.
  Status UpdateDependencyGraph(
      const std::set<std::string>& added, const std::set<std::string>& deleted,
      const std::set<std::string>& modified);

  /// Helper function to uncheck the nodes because the model that they depends
  /// on has changed. The unchecked nodes will be validated again.
  /// The function will be call recursively to uncheck all downstreams.
  /// \param downstreams The nodes to be unchecked.
  /// \param updated_nodes Return the nodes that have been unchecked
  void UncheckDownstream(NodeSet* downstreams, NodeSet* updated_nodes);

  /// Helper function to construct the edges between nodes in dependency graph.
  /// \param updated_node The node that is newly added or modified.
  /// \return True if the node represents an ensemble model. False otherwise.
  bool ConnectDependencyGraph(DependencyNode* updated_node);

  /// Get the configuration for a named model.
  /// \param name The model name.
  /// \param model_config Returns the model configuration.
  /// \return OK if found, NOT_FOUND otherwise.
  Status GetModelConfig(
      const std::string& name, inference::ModelConfig* model_config);

  /// Get the models to be loaded / unloaded based on the model loaded in
  /// previous iteration.
  /// \param loaded_models The models loaded / unloaded in previous iteration.
  /// Unloaded models will be represented as models with no loaded versions.
  /// \return A pair of node set containing models to be loaded and models to be
  /// unloaded for the next iteration.
  std::pair<NodeSet, NodeSet> ModelsToLoadUnload(const NodeSet& loaded_models);

  /// Check if the node is ready for the next iteration. A node is ready if the
  /// node is invalid (containing invalid model config or its depdencies failed
  /// to load) or all of its dependencies are satisfied.
  /// \param node The node to be checked.
  /// \return True if the node is ready. False otherwise.
  bool CheckNode(DependencyNode* node);

  /// Get the list of versions to be loaded for a named model based on version
  /// policy. Version directories that are not numerically named,
  /// or that have zero prefix will be ignored.
  /// \param model_repository_path The file-system path of the repository that
  /// the model is at.
  /// \param name The model name.
  /// \param model_config The model configuration.
  /// \param versions Returns the versions to be loaded
  /// \return The error status.
  Status VersionsToLoad(
      const std::string model_repository_path, const std::string& name,
      const inference::ModelConfig& model_config, std::set<int64_t>* versions);

  const std::set<std::string> repository_paths_;
  const BackendConfigMap backend_config_map_;
  const bool autofill_;
  const bool polling_enabled_;
  const bool model_control_enabled_;
  const double min_compute_capability_;

  std::mutex poll_mu_;
  ModelInfoMap infos_;

  std::unordered_map<std::string, std::unique_ptr<DependencyNode>>
      dependency_graph_;
  std::unordered_map<std::string, std::unique_ptr<DependencyNode>>
      missing_nodes_;

  std::unique_ptr<BackendLifeCycle> backend_life_cycle_;
};

}}  // namespace nvidia::inferenceserver
