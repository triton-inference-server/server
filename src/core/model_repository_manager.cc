// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/model_repository_manager.h"

#include <algorithm>
#include <deque>
#include <future>
#include <stdexcept>
#include <thread>
#include "src/core/backend.h"
#include "src/core/constants.h"
#include "src/core/ensemble_utils.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif

#ifdef TRITON_ENABLE_CAFFE2
#include "src/backends/caffe2/netdef_backend_factory.h"
#endif  // TRITON_ENABLE_CAFFE2
#ifdef TRITON_ENABLE_CUSTOM
#include "src/backends/backend/backend_factory.h"
#include "src/backends/custom/custom_backend_factory.h"
#endif  // TRITON_ENABLE_CUSTOM
#ifdef TRITON_ENABLE_ENSEMBLE
#include "src/backends/ensemble/ensemble_backend_factory.h"
#endif  // TRITON_ENABLE_ENSEMBLE
#ifdef TRITON_ENABLE_ONNXRUNTIME
#include "src/backends/onnx/onnx_backend_factory.h"
#endif  // TRITON_ENABLE_ONNXRUNTIME
#ifdef TRITON_ENABLE_PYTORCH
#include "src/backends/pytorch/libtorch_backend_factory.h"
#endif  // TRITON_ENABLE_PYTORCH
#ifdef TRITON_ENABLE_TENSORFLOW
#include "src/backends/tensorflow/graphdef_backend_factory.h"
#include "src/backends/tensorflow/savedmodel_backend_factory.h"
#endif  // TRITON_ENABLE_TENSORFLOW
#ifdef TRITON_ENABLE_TENSORRT
#include "src/backends/tensorrt/plan_backend_factory.h"
#endif  // TRITON_ENABLE_TENSORRT

namespace nvidia { namespace inferenceserver {

const std::string&
ModelReadyStateString(ModelReadyState state)
{
  switch (state) {
    case ModelReadyState::UNKNOWN: {
      static std::string m("UNKNOWN");
      return m;
    }
    case ModelReadyState::READY: {
      static std::string m("READY");
      return m;
    }
    case ModelReadyState::UNAVAILABLE: {
      static std::string m("UNAVAILABLE");
      return m;
    }
    case ModelReadyState::LOADING: {
      static std::string m("LOADING");
      return m;
    }
    case ModelReadyState::UNLOADING: {
      static std::string m("UNLOADING");
      return m;
    }
  }

  static std::string m("<unknown>");
  return m;
}

namespace {

void
BuildBackendConfigMap(
    const std::string& version, const bool strict_model_config,
    const float tf_gpu_memory_fraction, const bool tf_allow_soft_placement,
    const std::map<int, std::pair<int, uint64_t>> tf_vgpu_memory_limit_mb,
    BackendConfigMap* backend_configs)
{
#ifdef TRITON_ENABLE_TENSORFLOW
  //// Tensorflow GraphDef and SavedModel
  {
    auto graphdef_config = std::make_shared<GraphDefBackendFactory::Config>();
    graphdef_config->autofill = !strict_model_config;

    if (tf_gpu_memory_fraction == 0.0) {
      graphdef_config->allow_gpu_memory_growth = true;
    } else {
      graphdef_config->allow_gpu_memory_growth = false;
      graphdef_config->per_process_gpu_memory_fraction = tf_gpu_memory_fraction;
    }

#ifdef TRITON_ENABLE_GPU
    int device_cnt = 0;
    cudaError_t cuerr = cudaGetDeviceCount(&device_cnt);
    if ((cuerr == cudaErrorNoDevice) ||
        (cuerr == cudaErrorInsufficientDriver)) {
      device_cnt = 0;
    } else if (cuerr != cudaSuccess) {
      LOG_ERROR << "unable to get number of CUDA devices while building "
                   "BackendConfigMap: ("
                << cuerr << ") " << cudaGetErrorString(cuerr);
      device_cnt = 0;
    }

    if (!tf_vgpu_memory_limit_mb.empty()) {
      for (int device = 0; device < device_cnt; device++) {
        auto device_mapping = tf_vgpu_memory_limit_mb.find(device);
        if (device_mapping != tf_vgpu_memory_limit_mb.end()) {
          graphdef_config->memory_limit_mb[device] = std::vector<float>(
              device_mapping->second.first, device_mapping->second.second);
        } else {
          graphdef_config->memory_limit_mb[device] = {};
        }
      }
      graphdef_config->per_process_gpu_memory_fraction = 0.0;
    }
#endif  // TRITON_ENABLE_GPU

    graphdef_config->allow_soft_placement = tf_allow_soft_placement;

    (*backend_configs)[kTensorFlowGraphDefPlatform] = graphdef_config;
    (*backend_configs)[kTensorFlowSavedModelPlatform] = graphdef_config;
  }
#endif  // TRITON_ENABLE_TENSORFLOW

#ifdef TRITON_ENABLE_CAFFE2
  //// Caffe NetDef
  {
    auto netdef_config = std::make_shared<NetDefBackendFactory::Config>();
    netdef_config->autofill = !strict_model_config;
    (*backend_configs)[kCaffe2NetDefPlatform] = netdef_config;
  }
#endif  // TRITON_ENABLE_CAFFE2

#ifdef TRITON_ENABLE_TENSORRT
  //// TensorRT
  {
    auto plan_config = std::make_shared<PlanBackendFactory::Config>();
    plan_config->autofill = !strict_model_config;
    (*backend_configs)[kTensorRTPlanPlatform] = plan_config;
  }
#endif  // TRITON_ENABLE_TENSORRT

#ifdef TRITON_ENABLE_ONNXRUNTIME
  //// OnnxRuntime Onnx
  {
    auto onnx_config = std::make_shared<OnnxBackendFactory::Config>();
    onnx_config->autofill = !strict_model_config;
    (*backend_configs)[kOnnxRuntimeOnnxPlatform] = onnx_config;
  }
#endif  // TRITON_ENABLE_ONNXRUNTIME

#ifdef TRITON_ENABLE_PYTORCH
  //// PyTorch LibTorch
  {
    auto libtorch_config = std::make_shared<LibTorchBackendFactory::Config>();
    libtorch_config->autofill = !strict_model_config;
    (*backend_configs)[kPyTorchLibTorchPlatform] = libtorch_config;
  }
#endif  // TRITON_ENABLE_PYTORCH

#ifdef TRITON_ENABLE_CUSTOM
  //// Custom
  {
    auto custom_config = std::make_shared<CustomBackendFactory::Config>();
    custom_config->inference_server_version = version;
    (*backend_configs)[kCustomPlatform] = custom_config;
  }
#endif  // TRITON_ENABLE_CUSTOM

#ifdef TRITON_ENABLE_ENSEMBLE
  //// Ensemble
  {
    auto ensemble_config = std::make_shared<EnsembleBackendFactory::Config>();
    (*backend_configs)[kEnsemblePlatform] = ensemble_config;
  }
#endif  // TRITON_ENABLE_ENSEMBLE
  ;     // Need this semicolon to keep code formatter from freaking out
}

int64_t
GetModifiedTime(const std::string& path)
{
  // If there is an error in any step the fall-back default
  // modification time is 0. This means that in error cases 'path'
  // will show as not modified. This is the safe fall-back to avoid
  // assuming a model is constantly being modified.
  bool path_is_dir;
  Status status = IsDirectory(path, &path_is_dir);
  if (!status.IsOk()) {
    LOG_ERROR << "Failed to determine modification time for '" << path
              << "': " << status.AsString();
    return 0;
  }

  // If 'path' is a file return its mtime. Otherwise, using the modification
  // time of the directory as baseline in case of file deletion
  int64_t mtime;
  status = FileModificationTime(path, &mtime);
  if (!status.IsOk()) {
    LOG_ERROR << "Failed to determine modification time for '" << path
              << "': " << status.AsString();
    return 0;
  }
  if (!path_is_dir) {
    return mtime;
  }

  // 'path' is a directory. Return the most recent mtime of the
  // contents of the directory.
  std::set<std::string> contents;
  status = GetDirectoryContents(path, &contents);
  if (!status.IsOk()) {
    LOG_ERROR << "Failed to determine modification time for '" << path
              << "': " << status.AsString();
    return 0;
  }

  for (const auto& child : contents) {
    const auto full_path = JoinPath({path, child});
    mtime = std::max(mtime, GetModifiedTime(full_path));
  }

  return mtime;
}
// Return true if any file in the subdirectory root at 'path' has been
// modified more recently than 'last'. Return the most-recent modified
// time in 'last'.
bool
IsModified(const std::string& path, int64_t* last_ns)
{
  const int64_t repo_ns = GetModifiedTime(path);
  bool modified = repo_ns > *last_ns;
  *last_ns = repo_ns;
  return modified;
}

// Use smart pointer with custom deleter so that model state will be updated
// to UNAVAILABLE if all smart pointer copies are out of scope
struct BackendDeleter {
  BackendDeleter(std::function<void()> OnDestroyBackend)
      : OnDestroyBackend_(std::move(OnDestroyBackend))
  {
  }

  void operator()(InferenceBackend* backend)
  {
    // The actual model object must be destroyed in a different
    // thread. This thread could have a callstack that includes the
    // model/backend itself because this deleter could be triggered by
    // a request release or response send in the backend. Following
    // delete will lead to the model destructor which may wait on this
    // same thread... so deadlock if we don't use a different thread
    // here.
    std::function<void()> destroy_fn = OnDestroyBackend_;
    std::thread dthd([backend, destroy_fn]() {
      delete backend;
      destroy_fn();
    });

    dthd.detach();
  }

  // Use to inform the BackendLifeCycle that the backend handle is destroyed
  std::function<void()> OnDestroyBackend_;
};

}  // namespace

struct ModelRepositoryManager::ModelInfo {
  // [TODO] split modification time into versions' and model's
  // so that we have more information on whether the model reload
  // is necessary
  int64_t mtime_nsec_;
  ModelConfig model_config_;
  Platform platform_;
  std::string model_repository_path_;
};

class ModelRepositoryManager::BackendLifeCycle {
 public:
  static Status Create(
      InferenceServer* server, const double min_compute_capability,
      const BackendConfigMap& backend_map,
      std::unique_ptr<BackendLifeCycle>* life_cycle);

  ~BackendLifeCycle() = default;

  // Start loading model backends with specified versions asynchronously.
  // If 'force_unload', all versions that are being served will
  // be unloaded before loading the specified versions.
  Status AsyncLoad(
      const std::string& repository_path, const std::string& model_name,
      const std::set<int64_t>& versions, const ModelConfig& model_config,
      bool force_unload = true,
      std::function<void(int64_t, ModelReadyState, size_t)> OnComplete =
          nullptr);

  // Get specified model version's backend. Latest ready version will
  // be retrieved if 'version' is -1. Return error if the version specified is
  // not found or it is not ready.
  Status GetInferenceBackend(
      const std::string& model_name, const int64_t version,
      std::shared_ptr<InferenceBackend>* backend);

  // Get the ModelStateMap representation of the live backends. A backend is
  // live if at least one of the versions is not unknown nor unavailable.
  // If 'strict_readiness' is true, a backend is only live if
  // at least one of the versions is ready.
  const ModelStateMap LiveBackendStates(bool strict_readiness = false);

  // Get the ModelStateMap representation of the backends.
  const ModelStateMap BackendStates();

  // Get the VersionStateMap representation of the specified model.
  const VersionStateMap VersionStates(const std::string& model_name);

  // Get the state of a specific model version.
  Status ModelState(
      const std::string& model_name, const int64_t model_version,
      ModelReadyState* state);

 private:
  struct BackendInfo {
    BackendInfo(
        const std::string& repository_path, const ModelReadyState state,
        const ActionType next_action, const ModelConfig& model_config)
        : repository_path_(repository_path),
          platform_(GetPlatform(model_config.platform())), state_(state),
          next_action_(next_action), model_config_(model_config)
    {
    }

    std::string repository_path_;
    Platform platform_;

    std::recursive_mutex mtx_;
    ModelReadyState state_;
    std::string state_reason_;

    // next_action will be set in the case where a load / unload is requested
    // while the backend is already in loading / unloading state. Then the new
    // load / unload will be postponed as next action.
    ActionType next_action_;
    // callback function that will be triggered when there is no next action
    std::function<void()> OnComplete_;
    ModelConfig model_config_;

    std::shared_ptr<InferenceBackend> backend_;
  };

  BackendLifeCycle(const double min_compute_capability)
      : min_compute_capability_(min_compute_capability)
  {
  }

  // Function called after backend state / next action is updated.
  // Caller must obtain the mutex of 'backend_info' before calling this function
  Status TriggerNextAction(
      const std::string& model_name, const int64_t version,
      BackendInfo* backend_info);

  // Helper function called by TriggerNextAction()
  Status Load(
      const std::string& model_name, const int64_t version,
      BackendInfo* backend_info);

  // Helper function called by TriggerNextAction()
  Status Unload(
      const std::string& model_name, const int64_t version,
      BackendInfo* backend_info);

  Status CreateInferenceBackend(
      const std::string& model_name, const int64_t version,
      BackendInfo* backend_info);

  const double min_compute_capability_;

  using VersionMap = std::map<int64_t, std::unique_ptr<BackendInfo>>;
  using BackendMap = std::map<std::string, VersionMap>;
  BackendMap map_;
  std::mutex map_mtx_;

#ifdef TRITON_ENABLE_CAFFE2
  std::unique_ptr<NetDefBackendFactory> netdef_factory_;
#endif  // TRITON_ENABLE_CAFFE2
#ifdef TRITON_ENABLE_CUSTOM
  std::unique_ptr<TritonBackendFactory> triton_backend_factory_;
  std::unique_ptr<CustomBackendFactory> custom_factory_;
#endif  // TRITON_ENABLE_CUSTOM
#ifdef TRITON_ENABLE_TENSORFLOW
  std::unique_ptr<GraphDefBackendFactory> graphdef_factory_;
  std::unique_ptr<SavedModelBackendFactory> savedmodel_factory_;
#endif  // TRITON_ENABLE_TENSORFLOW
#ifdef TRITON_ENABLE_TENSORRT
  std::unique_ptr<PlanBackendFactory> plan_factory_;
#endif  // TRITON_ENABLE_TENSORRT
#ifdef TRITON_ENABLE_ONNXRUNTIME
  std::unique_ptr<OnnxBackendFactory> onnx_factory_;
#endif  // TRITON_ENABLE_ONNXRUNTIME
#ifdef TRITON_ENABLE_PYTORCH
  std::unique_ptr<LibTorchBackendFactory> libtorch_factory_;
#endif  // TRITON_ENABLE_PYTORCH
#ifdef TRITON_ENABLE_ENSEMBLE
  std::unique_ptr<EnsembleBackendFactory> ensemble_factory_;
#endif  // TRITON_ENABLE_ENSEMBLE
};

Status
ModelRepositoryManager::BackendLifeCycle::Create(
    InferenceServer* server, const double min_compute_capability,
    const BackendConfigMap& backend_map,
    std::unique_ptr<BackendLifeCycle>* life_cycle)
{
  std::unique_ptr<BackendLifeCycle> local_life_cycle(
      new BackendLifeCycle(min_compute_capability));

#ifdef TRITON_ENABLE_TENSORFLOW
  {
    const std::shared_ptr<BackendConfig>& config =
        backend_map.find(kTensorFlowGraphDefPlatform)->second;
    RETURN_IF_ERROR(GraphDefBackendFactory::Create(
        config, &(local_life_cycle->graphdef_factory_)));
  }
  {
    const std::shared_ptr<BackendConfig>& config =
        backend_map.find(kTensorFlowSavedModelPlatform)->second;
    RETURN_IF_ERROR(SavedModelBackendFactory::Create(
        config, &(local_life_cycle->savedmodel_factory_)));
  }
#endif  // TRITON_ENABLE_TENSORFLOW
#ifdef TRITON_ENABLE_CAFFE2
  {
    const std::shared_ptr<BackendConfig>& config =
        backend_map.find(kCaffe2NetDefPlatform)->second;
    RETURN_IF_ERROR(NetDefBackendFactory::Create(
        config, &(local_life_cycle->netdef_factory_)));
  }
#endif  // TRITON_ENABLE_CAFFE2
#ifdef TRITON_ENABLE_TENSORRT
  {
    const std::shared_ptr<BackendConfig>& config =
        backend_map.find(kTensorRTPlanPlatform)->second;
    RETURN_IF_ERROR(
        PlanBackendFactory::Create(config, &(local_life_cycle->plan_factory_)));
  }
#endif  // TRITON_ENABLE_TENSORRT
#ifdef TRITON_ENABLE_ONNXRUNTIME
  {
    const std::shared_ptr<BackendConfig>& config =
        backend_map.find(kOnnxRuntimeOnnxPlatform)->second;
    RETURN_IF_ERROR(
        OnnxBackendFactory::Create(config, &(local_life_cycle->onnx_factory_)));
  }
#endif  // TRITON_ENABLE_ONNXRUNTIME
#ifdef TRITON_ENABLE_PYTORCH
  {
    const std::shared_ptr<BackendConfig>& config =
        backend_map.find(kPyTorchLibTorchPlatform)->second;
    RETURN_IF_ERROR(LibTorchBackendFactory::Create(
        config, &(local_life_cycle->libtorch_factory_)));
  }
#endif  // TRITON_ENABLE_PYTORCH
#ifdef TRITON_ENABLE_CUSTOM
  {
    const std::shared_ptr<BackendConfig>& config =
        backend_map.find(kCustomPlatform)->second;
    RETURN_IF_ERROR(CustomBackendFactory::Create(
        config, &(local_life_cycle->custom_factory_)));
  }
  {
    const std::shared_ptr<BackendConfig> config;
    RETURN_IF_ERROR(TritonBackendFactory::Create(
        server, config, &(local_life_cycle->triton_backend_factory_)));
  }
#endif  // TRITON_ENABLE_CUSTOM
#ifdef TRITON_ENABLE_ENSEMBLE
  {
    const std::shared_ptr<BackendConfig>& config =
        backend_map.find(kEnsemblePlatform)->second;
    RETURN_IF_ERROR(EnsembleBackendFactory::Create(
        server, config, &(local_life_cycle->ensemble_factory_)));
  }
#endif  // TRITON_ENABLE_ENSEMBLE

  *life_cycle = std::move(local_life_cycle);
  return Status::Success;
}

const ModelRepositoryManager::ModelStateMap
ModelRepositoryManager::BackendLifeCycle::LiveBackendStates(
    bool strict_readiness)
{
  LOG_VERBOSE(1) << "LiveBackendStates()";
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  ModelStateMap live_backend_states;
  for (auto& model_version : map_) {
    bool live = false;
    VersionStateMap version_map;

    for (auto& version_backend : model_version.second) {
      std::lock_guard<std::recursive_mutex> lock(version_backend.second->mtx_);
      if (strict_readiness &&
          version_backend.second->state_ != ModelReadyState::READY) {
        continue;
      }

      // At lease one version is live (ready / loading / unloading)
      if ((version_backend.second->state_ != ModelReadyState::UNKNOWN) &&
          (version_backend.second->state_ != ModelReadyState::UNAVAILABLE)) {
        live = true;
        version_map[version_backend.first] = std::make_pair(
            version_backend.second->state_,
            version_backend.second->state_reason_);
      }
    }

    if (live) {
      live_backend_states[model_version.first] = std::move(version_map);
    }
  }
  return live_backend_states;
}

const ModelRepositoryManager::ModelStateMap
ModelRepositoryManager::BackendLifeCycle::BackendStates()
{
  LOG_VERBOSE(1) << "BackendStates()";
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  ModelStateMap backend_states;
  for (auto& model_version : map_) {
    VersionStateMap version_map;

    for (auto& version_backend : model_version.second) {
      std::lock_guard<std::recursive_mutex> lock(version_backend.second->mtx_);
      version_map[version_backend.first] = std::make_pair(
          version_backend.second->state_,
          version_backend.second->state_reason_);
    }

    backend_states[model_version.first] = std::move(version_map);
  }

  return backend_states;
}

const ModelRepositoryManager::VersionStateMap
ModelRepositoryManager::BackendLifeCycle::VersionStates(
    const std::string& model_name)
{
  LOG_VERBOSE(1) << "VersionStates() '" << model_name << "'";
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  VersionStateMap version_map;
  auto mit = map_.find(model_name);
  if (mit != map_.end()) {
    for (auto& version_backend : mit->second) {
      std::lock_guard<std::recursive_mutex> lock(version_backend.second->mtx_);
      version_map[version_backend.first] = std::make_pair(
          version_backend.second->state_,
          version_backend.second->state_reason_);
    }
  }

  return version_map;
}

Status
ModelRepositoryManager::BackendLifeCycle::ModelState(
    const std::string& model_name, const int64_t model_version,
    ModelReadyState* state)
{
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  auto mit = map_.find(model_name);
  if (mit != map_.end()) {
    auto vit = mit->second.find(model_version);
    if (vit != mit->second.end()) {
      const auto& backend_info = vit->second;
      std::lock_guard<std::recursive_mutex> lock(backend_info->mtx_);
      *state = backend_info->state_;
      return Status::Success;
    }
  }

  return Status(
      Status::Code::NOT_FOUND, "model '" + model_name + "', version " +
                                   std::to_string(model_version) +
                                   " is not found");
}

Status
ModelRepositoryManager::BackendLifeCycle::GetInferenceBackend(
    const std::string& model_name, const int64_t version,
    std::shared_ptr<InferenceBackend>* backend)
{
  LOG_VERBOSE(1) << "GetInferenceBackend() '" << model_name << "' version "
                 << version;
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  auto mit = map_.find(model_name);
  if (mit == map_.end()) {
    return Status(Status::Code::NOT_FOUND, "'" + model_name + "' is not found");
  }

  auto vit = mit->second.find(version);
  if (vit == mit->second.end()) {
    // In case the request is asking for latest version
    int64_t latest = -1;
    if (version == -1) {
      for (auto& version_backend : mit->second) {
        if (version_backend.first > latest) {
          std::lock_guard<std::recursive_mutex> lock(
              version_backend.second->mtx_);
          if (version_backend.second->state_ == ModelReadyState::READY) {
            latest = version_backend.first;
            // Tedious, but have to set handle for any "latest" version
            // at the moment to avoid edge case like the following:
            // "versions : 1 3 2", version 3 is latest but is requested
            // to be unloaded when the iterator is examining version 2.
            *backend = version_backend.second->backend_;
          }
        }
      }
      if (latest == -1) {
        return Status(
            Status::Code::NOT_FOUND,
            "'" + model_name + "' has no available versions");
      }
    } else {
      return Status(
          Status::Code::NOT_FOUND, "'" + model_name + "' version " +
                                       std::to_string(version) +
                                       " is not found");
    }
  } else {
    std::lock_guard<std::recursive_mutex> lock(vit->second->mtx_);
    if (vit->second->state_ == ModelReadyState::READY) {
      *backend = vit->second->backend_;
    } else {
      return Status(
          Status::Code::UNAVAILABLE, "'" + model_name + "' version " +
                                         std::to_string(version) +
                                         " is not at ready state");
    }
  }
  return Status::Success;
}

Status
ModelRepositoryManager::BackendLifeCycle::AsyncLoad(
    const std::string& repository_path, const std::string& model_name,
    const std::set<int64_t>& versions, const ModelConfig& model_config,
    bool force_unload,
    std::function<void(int64_t, ModelReadyState, size_t)> OnComplete)
{
  LOG_VERBOSE(1) << "AsyncLoad() '" << model_name << "'";
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  auto it = map_.find(model_name);
  if (it == map_.end()) {
    it = map_.emplace(std::make_pair(model_name, VersionMap())).first;
  }

  for (const auto& version : versions) {
    auto res = it->second.emplace(
        std::make_pair(version, std::unique_ptr<BackendInfo>()));
    if (res.second) {
      res.first->second.reset(new BackendInfo(
          repository_path, ModelReadyState::UNKNOWN, ActionType::NO_ACTION,
          model_config));
    }
  }

  Status status = Status::Success;
  size_t affected_version_cnt =
      force_unload ? it->second.size() : versions.size();
  for (auto& version_backend : it->second) {
    std::lock_guard<std::recursive_mutex> lock(version_backend.second->mtx_);
    if (versions.find(version_backend.first) != versions.end()) {
      version_backend.second->repository_path_ = repository_path;
      version_backend.second->model_config_ = model_config;
      version_backend.second->next_action_ = ActionType::LOAD;
    } else if (force_unload) {
      version_backend.second->next_action_ = ActionType::UNLOAD;
    }

    auto version = version_backend.first;
    auto backend_info = version_backend.second.get();
    // set version-wise callback before triggering next action
    if (OnComplete != nullptr) {
      version_backend.second->OnComplete_ =
          [version, backend_info, affected_version_cnt, OnComplete]() {
            OnComplete(version, backend_info->state_, affected_version_cnt);
          };
    }
    Status action_status = TriggerNextAction(model_name, version, backend_info);
    // Only care about status on unloading case
    if (!action_status.IsOk() && versions.empty()) {
      status = action_status;
    }
  }

  if (versions.empty()) {
    return Status(
        Status::Code::INVALID_ARG,
        "at least one version must be available under the version policy of "
        "model '" +
            model_name + "'");
  }

  return status;
}

Status
ModelRepositoryManager::BackendLifeCycle::TriggerNextAction(
    const std::string& model_name, const int64_t version,
    BackendInfo* backend_info)
{
  LOG_VERBOSE(1) << "TriggerNextAction() '" << model_name << "' version "
                 << version << ": "
                 << std::to_string(backend_info->next_action_);
  ActionType next_action = backend_info->next_action_;
  backend_info->next_action_ = ActionType::NO_ACTION;
  Status status = Status::Success;
  switch (next_action) {
    case ActionType::LOAD:
      status = Load(model_name, version, backend_info);
      break;
    case ActionType::UNLOAD:
      status = Unload(model_name, version, backend_info);
      break;
    default:
      if (backend_info->OnComplete_ != nullptr) {
        LOG_VERBOSE(1) << "no next action, trigger OnComplete()";
        backend_info->OnComplete_();
        backend_info->OnComplete_ = nullptr;
      }
      break;
  }

  // If status is not ok, "next action" path ends here and thus need to
  // invoke callback by this point
  if ((!status.IsOk()) && (backend_info->OnComplete_ != nullptr)) {
    LOG_VERBOSE(1) << "failed to execute next action, trigger OnComplete()";
    backend_info->OnComplete_();
    backend_info->OnComplete_ = nullptr;
  }

  return status;
}

Status
ModelRepositoryManager::BackendLifeCycle::Load(
    const std::string& model_name, const int64_t version,
    BackendInfo* backend_info)
{
  LOG_VERBOSE(1) << "Load() '" << model_name << "' version " << version;
  Status status = Status::Success;

  backend_info->next_action_ = ActionType::NO_ACTION;

  switch (backend_info->state_) {
    case ModelReadyState::READY:
      LOG_INFO << "re-loading: " << model_name << ":" << version;
      backend_info->state_ = ModelReadyState::UNLOADING;
      backend_info->state_reason_.clear();
      backend_info->next_action_ = ActionType::LOAD;
      // The load will be triggered once the unload is done (deleter is called)
      backend_info->backend_.reset();
      break;
    case ModelReadyState::LOADING:
    case ModelReadyState::UNLOADING:
      backend_info->next_action_ = ActionType::LOAD;
      break;
    default:
      LOG_INFO << "loading: " << model_name << ":" << version;
      backend_info->state_ = ModelReadyState::LOADING;
      backend_info->state_reason_.clear();
      {
        std::thread worker(
            &ModelRepositoryManager::BackendLifeCycle::CreateInferenceBackend,
            this, model_name, version, backend_info);
        worker.detach();
      }
      break;
  }

  return status;
}

Status
ModelRepositoryManager::BackendLifeCycle::Unload(
    const std::string& model_name, const int64_t version,
    BackendInfo* backend_info)
{
  LOG_VERBOSE(1) << "Unload() '" << model_name << "' version " << version;
  Status status = Status::Success;

  backend_info->next_action_ = ActionType::NO_ACTION;

  switch (backend_info->state_) {
    case ModelReadyState::READY:
      LOG_INFO << "unloading: " << model_name << ":" << version;
      backend_info->state_ = ModelReadyState::UNLOADING;
      backend_info->state_reason_.clear();
      backend_info->backend_.reset();
      break;
    case ModelReadyState::LOADING:
    case ModelReadyState::UNLOADING:
      backend_info->next_action_ = ActionType::UNLOAD;
      break;
    default:
      status = Status(
          Status::Code::NOT_FOUND,
          "tried to unload model '" + model_name + "' version " +
              std::to_string(version) + " which is at model state: " +
              ModelReadyStateString(backend_info->state_));
      break;
  }

  return status;
}

Status
ModelRepositoryManager::BackendLifeCycle::CreateInferenceBackend(
    const std::string& model_name, const int64_t version,
    BackendInfo* backend_info)
{
  LOG_VERBOSE(1) << "CreateInferenceBackend() '" << model_name << "' version "
                 << version;
  const auto version_path = JoinPath(
      {backend_info->repository_path_, model_name, std::to_string(version)});
  // make copy of the current model config in case model config in backend info
  // is updated (another poll) during the creation of backend handle
  ModelConfig model_config;
  {
    std::lock_guard<std::recursive_mutex> lock(backend_info->mtx_);
    model_config = backend_info->model_config_;
  }

  // Create backend
  Status status;
  std::unique_ptr<InferenceBackend> is;

  // If 'backend' is specified in the config then use the new triton
  // backend.
#ifdef TRITON_ENABLE_CUSTOM
  if (!model_config.backend().empty()) {
    status = triton_backend_factory_->CreateBackend(
        backend_info->repository_path_, model_name, version, model_config,
        min_compute_capability_, &is);
  } else
#endif  // TRITON_ENABLE_CUSTOM
  {
    switch (backend_info->platform_) {
#ifdef TRITON_ENABLE_TENSORFLOW
      case Platform::PLATFORM_TENSORFLOW_GRAPHDEF:
        status = graphdef_factory_->CreateBackend(
            version_path, model_config, min_compute_capability_, &is);
        break;
      case Platform::PLATFORM_TENSORFLOW_SAVEDMODEL:
        status = savedmodel_factory_->CreateBackend(
            version_path, model_config, min_compute_capability_, &is);
        break;
#endif  // TRITON_ENABLE_TENSORFLOW
#ifdef TRITON_ENABLE_TENSORRT
      case Platform::PLATFORM_TENSORRT_PLAN:
        status = plan_factory_->CreateBackend(
            version_path, model_config, min_compute_capability_, &is);
        break;
#endif  // TRITON_ENABLE_TENSORRT
#ifdef TRITON_ENABLE_CAFFE2
      case Platform::PLATFORM_CAFFE2_NETDEF:
        status = netdef_factory_->CreateBackend(
            version_path, model_config, min_compute_capability_, &is);
        break;
#endif  // TRITON_ENABLE_CAFFE2
#ifdef TRITON_ENABLE_ONNXRUNTIME
      case Platform::PLATFORM_ONNXRUNTIME_ONNX:
        status = onnx_factory_->CreateBackend(
            version_path, model_config, min_compute_capability_, &is);
        break;
#endif  // TRITON_ENABLE_ONNXRUNTIME
#ifdef TRITON_ENABLE_PYTORCH
      case Platform::PLATFORM_PYTORCH_LIBTORCH:
        status = libtorch_factory_->CreateBackend(
            version_path, model_config, min_compute_capability_, &is);
        break;
#endif  // TRITON_ENABLE_PYTORCH
#ifdef TRITON_ENABLE_CUSTOM
      case Platform::PLATFORM_CUSTOM:
        status = custom_factory_->CreateBackend(
            backend_info->repository_path_, model_name, version, model_config,
            min_compute_capability_, &is);
        break;
#endif  // TRITON_ENABLE_CUSTOM
#ifdef TRITON_ENABLE_ENSEMBLE
      case Platform::PLATFORM_ENSEMBLE: {
        status = ensemble_factory_->CreateBackend(
            version_path, model_config, min_compute_capability_, &is);
        // Complete label provider with label information from involved backends
        // Must be done here because involved backends may not be able to
        // obtained from server because this may happen during server
        // initialization.
        if (status.IsOk()) {
          std::set<std::string> no_label_outputs;
          const auto& label_provider = is->GetLabelProvider();
          for (const auto& output : model_config.output()) {
            if (label_provider->GetLabel(output.name(), 0).empty()) {
              no_label_outputs.emplace(output.name());
            }
          }
          for (const auto& element :
               model_config.ensemble_scheduling().step()) {
            for (const auto& pair : element.output_map()) {
              // Found model that produce one of the missing output
              if (no_label_outputs.find(pair.second) !=
                  no_label_outputs.end()) {
                std::shared_ptr<InferenceBackend> backend;
                // Safe to obtain backend because the ensemble can't be loaded
                // until the involved backends are ready
                GetInferenceBackend(
                    element.model_name(), element.model_version(), &backend);
                label_provider->AddLabels(
                    pair.second,
                    backend->GetLabelProvider()->GetLabels(pair.first));
              }
            }
          }
        }
        break;
      }
#endif  // TRITON_ENABLE_ENSEMBLE
      default:
        status = Status(
            Status::Code::INVALID_ARG,
            "unknown platform '" + model_config.platform() + "'");
        break;
    }
  }

  // Update backend state
  std::lock_guard<std::recursive_mutex> lock(backend_info->mtx_);
  // Sanity check
  if (backend_info->backend_ != nullptr) {
    LOG_ERROR << "trying to load model '" << model_name << "' version "
              << version << " while it is being served";
  } else {
    if (status.IsOk()) {
      // Unless the handle is nullptr, always reset handle out of the mutex,
      // otherwise the handle's destructor will try to acquire the mutex and
      // cause deadlock.
      backend_info->backend_.reset(
          is.release(),
          BackendDeleter([this, model_name, version, backend_info]() mutable {
            LOG_VERBOSE(1) << "OnDestroy callback() '" << model_name
                           << "' version " << version;
            LOG_INFO << "successfully unloaded '" << model_name << "' version "
                     << version;
            // Use recursive mutex as this deleter is likely to to be called
            // within BackendLifeCycle class where the same mutex is being hold.
            // However, mutex acquisition is needed here for the case where
            // the backend is requested to be unloaded while there are inflight
            // requests, then the deleter will be called from the request thread
            {
              std::lock_guard<std::recursive_mutex> lock(backend_info->mtx_);
              backend_info->state_ = ModelReadyState::UNAVAILABLE;
              backend_info->state_reason_ = "unloaded";
              // Check if next action is requested
              this->TriggerNextAction(model_name, version, backend_info);
            }
          }));
      backend_info->state_ = ModelReadyState::READY;
      backend_info->state_reason_.clear();
      LOG_INFO << "successfully loaded '" << model_name << "' version "
               << version;
    } else {
      LOG_ERROR << "failed to load '" << model_name << "' version " << version
                << ": " << status.AsString();
      backend_info->state_ = ModelReadyState::UNAVAILABLE;
      backend_info->state_reason_ = status.AsString();
    }
  }

  // Check if next action is requested
  return TriggerNextAction(model_name, version, backend_info);
}

ModelRepositoryManager::ModelRepositoryManager(
    const std::set<std::string>& repository_paths,
    const BackendConfigMap& backend_config_map, const bool autofill,
    const bool polling_enabled, const bool model_control_enabled,
    const double min_compute_capability,
    std::unique_ptr<BackendLifeCycle> life_cycle)
    : repository_paths_(repository_paths),
      backend_config_map_(backend_config_map), autofill_(autofill),
      polling_enabled_(polling_enabled),
      model_control_enabled_(model_control_enabled),
      min_compute_capability_(min_compute_capability),
      backend_life_cycle_(std::move(life_cycle))
{
}

ModelRepositoryManager::~ModelRepositoryManager() {}

Status
ModelRepositoryManager::Create(
    InferenceServer* server, const std::string& server_version,
    const std::set<std::string>& repository_paths,
    const std::set<std::string>& startup_models, const bool strict_model_config,
    const float tf_gpu_memory_fraction, const bool tf_allow_soft_placement,
    const std::map<int, std::pair<int, uint64_t>> tf_memory_limit_mb,
    const bool polling_enabled, const bool model_control_enabled,
    const double min_compute_capability,
    std::unique_ptr<ModelRepositoryManager>* model_repository_manager)
{
  // The rest only matters if repository path is valid directory
  for (const auto& path : repository_paths) {
    bool path_is_dir;
    RETURN_IF_ERROR(IsDirectory(path, &path_is_dir));
    if (!path_is_dir) {
      return Status(
          Status::Code::INVALID_ARG,
          "repository path is not a valid directory");
    }
  }

  if (polling_enabled && model_control_enabled) {
    return Status(
        Status::Code::INVALID_ARG,
        "cannot enable both polling and explicit model control");
  }

  BackendConfigMap backend_config_map;

  BuildBackendConfigMap(
      server_version, strict_model_config, tf_gpu_memory_fraction,
      tf_allow_soft_placement, tf_memory_limit_mb, &backend_config_map);

  std::unique_ptr<BackendLifeCycle> life_cycle;
  RETURN_IF_ERROR(BackendLifeCycle::Create(
      server, min_compute_capability, backend_config_map, &life_cycle));

  // Not setting the smart pointer directly to simplify clean up
  std::unique_ptr<ModelRepositoryManager> local_manager(
      new ModelRepositoryManager(
          repository_paths, backend_config_map, !strict_model_config,
          polling_enabled, model_control_enabled, min_compute_capability,
          std::move(life_cycle)));

  bool all_models_polled = true;
  if (!model_control_enabled) {
    // only error happens before model load / unload will be return
    // model loading / unloading error will be printed but ignored
    RETURN_IF_ERROR(local_manager->PollAndUpdateInternal(&all_models_polled));
  } else {
    RETURN_IF_ERROR(local_manager->LoadUnloadModels(
        startup_models, ActionType::LOAD, &all_models_polled));
  }

  *model_repository_manager = std::move(local_manager);

  if (!all_models_polled) {
    return Status(Status::Code::INTERNAL, "failed to load all models");
  }
  // Some models may failed to be loaded after model manager is created,
  // return proper error and let function caller decide whether to proceed.
  for (const auto& model : (*model_repository_manager)->infos_) {
    const auto version_states =
        (*model_repository_manager)
            ->backend_life_cycle_->VersionStates(model.first);
    // Return general error message, detail of each model's loading state
    // is logged separately.
    if (version_states.empty()) {
      return Status(Status::Code::INTERNAL, "failed to load all models");
    }
    for (const auto& state : version_states) {
      if (state.second.first != ModelReadyState::READY) {
        return Status(Status::Code::INTERNAL, "failed to load all models");
      }
    }
  }

  return Status::Success;
}

Status
ModelRepositoryManager::PollAndUpdate()
{
  if (!polling_enabled_) {
    return Status(Status::Code::UNAVAILABLE, "polling is disabled");
  }

  bool all_models_polled;
  return PollAndUpdateInternal(&all_models_polled);
}

Status
ModelRepositoryManager::PollAndUpdateInternal(bool* all_models_polled)
{
  // Serialize all operations that change model state
  std::lock_guard<std::mutex> lock(poll_mu_);

  std::set<std::string> added, deleted, modified, unmodified;

  // We don't modify 'infos_' in place to minimize how long we need to
  // hold the lock and also prevent any partial changes to do an error
  // during processing.
  ModelInfoMap new_infos;

  // Each subdirectory of repository path is a model directory from
  // which we read the model configuration.
  std::set<std::string> subdirs;
  RETURN_IF_ERROR(Poll(
      subdirs, &added, &deleted, &modified, &unmodified, &new_infos,
      all_models_polled));

  // Anything in 'infos_' that is not in "added", "modified", or
  // "unmodified" is deleted.
  for (const auto& pr : infos_) {
    if ((added.find(pr.first) == added.end()) &&
        (modified.find(pr.first) == modified.end()) &&
        (unmodified.find(pr.first) == unmodified.end())) {
      deleted.insert(pr.first);
    }
  }

  // Nothing to do if no model adds, deletes or modifies.
  if (added.empty() && deleted.empty() && modified.empty()) {
    return Status::Success;
  }

  infos_.swap(new_infos);

  UpdateDependencyGraph(added, deleted, modified);

  for (const auto& name : deleted) {
    ModelConfig model_config;
    std::set<int64_t> versions;
    std::string empty_path;
    // Utilize "force_unload" of AsyncLoad()
    backend_life_cycle_->AsyncLoad(empty_path, name, versions, model_config);
  }

  // model loading / unloading error will be printed but ignored
  LoadModelByDependency();

  return Status::Success;
}

Status
ModelRepositoryManager::LoadModelByDependency()
{
  struct ModelState {
    ModelState(DependencyNode* node) : node_(node) {}
    DependencyNode* node_;
    std::set<int64_t> loaded_versions_;
    std::set<int64_t> unloaded_versions_;
    std::mutex mtx_;
    std::promise<void> ready_;
  };
  NodeSet loaded_models;
  auto set_pair = ModelsToLoadUnload(loaded_models);
  // Loop until all model are loaded / unloaded
  while ((!set_pair.first.empty()) || (!set_pair.second.empty())) {
    loaded_models.clear();
    // Unload invalid models first
    for (auto& invalid_model : set_pair.second) {
      ModelConfig model_config;
      std::set<int64_t> versions;
      std::string empty_path;
      // Utilize "force_unload" of AsyncLoad()
      backend_life_cycle_->AsyncLoad(
          empty_path, invalid_model->model_name_, versions, model_config);
      LOG_ERROR << invalid_model->status_.AsString();
      invalid_model->loaded_versions_ = std::set<int64_t>();
      loaded_models.emplace(invalid_model);
    }
    // load valid models and wait for load results
    std::vector<std::unique_ptr<ModelState>> model_states;
    for (auto& valid_model : set_pair.first) {
      std::string repository_path;
      const auto itr = infos_.find(valid_model->model_name_);
      repository_path = itr->second->model_repository_path_;

      model_states.emplace_back(new ModelState(valid_model));
      auto model_state = model_states.back().get();
      std::set<int64_t> versions;
      Status status;
      status = VersionsToLoad(
          repository_path, valid_model->model_name_, valid_model->model_config_,
          &versions);
      if (status.IsOk()) {
        status = backend_life_cycle_->AsyncLoad(
            repository_path, valid_model->model_name_, versions,
            valid_model->model_config_, true,
            [model_state](
                int64_t version, ModelReadyState state,
                size_t total_version_cnt) {
              std::lock_guard<std::mutex> lk(model_state->mtx_);
              if (state == ModelReadyState::READY) {
                model_state->loaded_versions_.emplace(version);
              } else {
                model_state->unloaded_versions_.emplace(version);
              }
              if ((model_state->loaded_versions_.size() +
                   model_state->unloaded_versions_.size()) ==
                  total_version_cnt) {
                model_state->ready_.set_value();
              }
            });
      }
      if (!status.IsOk()) {
        model_states.pop_back();
        LOG_ERROR << "failed to load model '" << valid_model->model_name_
                  << "': " << status.Message();
        valid_model->status_ = status;
        valid_model->loaded_versions_ = std::set<int64_t>();
      }
      loaded_models.emplace(valid_model);
    }
    for (auto& model_state : model_states) {
      model_state->ready_.get_future().wait();
      model_state->node_->loaded_versions_ = model_state->loaded_versions_;
    }
    set_pair = ModelsToLoadUnload(loaded_models);
  }
  return Status::Success;
}

Status
ModelRepositoryManager::LoadUnloadModel(
    const std::string& model_name, ActionType type)
{
  if (!model_control_enabled_) {
    return Status(
        Status::Code::UNAVAILABLE,
        "explicit model load / unload is not allowed if polling is enabled");
  }

  // Serialize all operations that change model state
  std::lock_guard<std::mutex> lock(poll_mu_);

  bool polled = true;
  RETURN_IF_ERROR(LoadUnloadModels({model_name}, type, &polled));

  // Check if model is loaded / unloaded properly
  const auto version_states = backend_life_cycle_->VersionStates(model_name);
  if (type == ActionType::LOAD) {
    if (version_states.empty()) {
      return Status(
          Status::Code::INTERNAL,
          "failed to load '" + model_name + "', no version is available");
    }
    auto it = infos_.find(model_name);
    if (it == infos_.end()) {
      return Status(
          Status::Code::INTERNAL,
          "failed to load '" + model_name +
              "', failed to poll from model repository");
    }

    const auto& info = it->second;
    const auto& config = info->model_config_;
    const auto& repository = info->model_repository_path_;
    std::set<int64_t> expected_versions;
    RETURN_IF_ERROR(
        VersionsToLoad(repository, model_name, config, &expected_versions));

    if (expected_versions.empty()) {
      return Status(
          Status::Code::INVALID_ARG,
          "at least one version must be available under the version policy of "
          "model '" +
              model_name + "'");
    }

    std::string not_ready_version_str;
    for (const auto version : expected_versions) {
      const auto it = version_states.find(version);
      if ((it == version_states.end()) ||
          (it->second.first != ModelReadyState::READY)) {
        not_ready_version_str += std::to_string(version);
        not_ready_version_str += ",";
      }
    }
    if (!not_ready_version_str.empty()) {
      not_ready_version_str.pop_back();
      return Status(
          Status::Code::INTERNAL,
          "failed to load '" + model_name +
              "', versions that are not available: " + not_ready_version_str);
    }
  } else {
    std::string ready_version_str;
    for (const auto& version_state : version_states) {
      if (version_state.second.first == ModelReadyState::READY) {
        ready_version_str += std::to_string(version_state.first);
        ready_version_str += ",";
      }
    }
    if (!ready_version_str.empty()) {
      ready_version_str.pop_back();
      return Status(
          Status::Code::INTERNAL,
          "failed to unload '" + model_name +
              "', versions that are still available: " + ready_version_str);
    }
  }

  return Status::Success;
}

Status
ModelRepositoryManager::LoadUnloadModels(
    const std::set<std::string>& model_names, ActionType type,
    bool* all_models_polled)
{
  *all_models_polled = true;
  // Update ModelInfo related to file system accordingly
  std::set<std::string> added, deleted, modified, unmodified;
  {
    if (type == ActionType::UNLOAD) {
      for (const auto& model_name : model_names) {
        deleted.insert(model_name);
      }
    } else {
      std::set<std::string> checked_modes = model_names;
      std::set<std::string> models = model_names;

      ModelInfoMap new_infos;
      while (!models.empty()) {
        bool polled = true;
        RETURN_IF_ERROR(Poll(
            models, &added, &deleted, &modified, &unmodified, &new_infos,
            &polled));
        *all_models_polled &= polled;

        // More models should be polled if the polled models are ensembles
        std::set<std::string> next_models;
#ifdef TRITON_ENABLE_ENSEMBLE
        for (const auto& model : models) {
          auto it = new_infos.find(model);
          // Some models may be marked as deleted and not in 'new_infos'
          if (it != new_infos.end()) {
            const auto& config = it->second->model_config_;
            if (config.has_ensemble_scheduling()) {
              for (const auto& step : config.ensemble_scheduling().step()) {
                bool need_poll =
                    checked_modes.emplace(step.model_name()).second;
                if (need_poll) {
                  next_models.emplace(step.model_name());
                }
              }
            }
          }
        }
#endif  // TRITON_ENABLE_ENSEMBLE

        models.swap(next_models);
      }

      // Only update the infos when all validation is completed
      for (const auto& model_name : added) {
        auto nitr = new_infos.find(model_name);
        infos_.emplace(model_name, std::move(nitr->second));
      }
      for (const auto& model_name : modified) {
        auto nitr = new_infos.find(model_name);
        auto itr = infos_.find(model_name);
        itr->second = std::move(nitr->second);
      }
    }
  }
  // The models are in 'deleted' either when they are asked to be unloaded or
  // they are not found / are duplicated across all model repositories.
  // In all cases, should unload them and remove from 'infos_' explicitly.
  for (const auto& name : deleted) {
    infos_.erase(name);
    ModelConfig model_config;
    std::set<int64_t> versions;
    std::string empty_path;
    // Utilize "force_unload" of AsyncLoad()
    backend_life_cycle_->AsyncLoad(empty_path, name, versions, model_config);
  }

  // Update dependency graph and load
  UpdateDependencyGraph(added, deleted, modified);

  // model loading / unloading error will be printed but ignored
  LoadModelByDependency();

  return Status::Success;
}

Status
ModelRepositoryManager::UnloadAllModels()
{
  Status status;
  // Reload an empty version list to cause the model to unload.
  ModelConfig model_config;
  std::set<int64_t> versions;
  std::string empty_path;
  for (const auto& name_info : infos_) {
    Status unload_status = backend_life_cycle_->AsyncLoad(
        empty_path, name_info.first, versions, model_config);
    if (!unload_status.IsOk()) {
      status = Status(
          Status::Code::INTERNAL,
          "Failed to gracefully unload models: " + unload_status.Message());
    }
  }
  return Status::Success;
}

const ModelRepositoryManager::ModelStateMap
ModelRepositoryManager::LiveBackendStates(bool strict_readiness)
{
  return backend_life_cycle_->LiveBackendStates(strict_readiness);
}

const ModelRepositoryManager::ModelStateMap
ModelRepositoryManager::BackendStates()
{
  return backend_life_cycle_->BackendStates();
}

const ModelRepositoryManager::VersionStateMap
ModelRepositoryManager::VersionStates(const std::string& model_name)
{
  return backend_life_cycle_->VersionStates(model_name);
}

Status
ModelRepositoryManager::ModelState(
    const std::string& model_name, const int64_t model_version,
    ModelReadyState* state)
{
  return backend_life_cycle_->ModelState(model_name, model_version, state);
}

Status
ModelRepositoryManager::RepositoryIndex(
    const bool ready_only, std::vector<ModelIndex>* index)
{
  std::set<std::string> seen_models;
  std::set<std::string> duplicate_models;
  for (const auto& repository_path : repository_paths_) {
    std::set<std::string> subdirs;
    RETURN_IF_ERROR(GetDirectorySubdirs(repository_path, &subdirs));
    for (const auto& subdir : subdirs) {
      if (seen_models.find(subdir) != seen_models.end()) {
        duplicate_models.insert(subdir);
      }

      seen_models.insert(subdir);
    }
  }

  ModelStateMap states = BackendStates();

  for (const auto& model : seen_models) {
    // If the same model appears in multiple repostories then show it
    // as unavailable since duplicate models are not allowed to load.
    if (duplicate_models.find(model) != duplicate_models.end()) {
      index->emplace_back(
          model, -1 /* version */, ModelReadyState::UNAVAILABLE,
          MODEL_READY_REASON_DUPLICATE);
      continue;
    }

    // If there is any version/state/reason associated with the model
    // then include that in the index.
    auto sitr = states.find(model);
    if (sitr == states.end()) {
      if (!ready_only) {
        index->emplace_back(model);
      }
    } else {
      for (const auto& pr : sitr->second) {
        if (!ready_only || (pr.second.first == ModelReadyState::READY)) {
          index->emplace_back(
              model, pr.first, pr.second.first, pr.second.second);
        }
      }
    }
  }

  return Status::Success;
}

Status
ModelRepositoryManager::GetInferenceBackend(
    const std::string& model_name, const int64_t model_version,
    std::shared_ptr<InferenceBackend>* backend)
{
  Status status = backend_life_cycle_->GetInferenceBackend(
      model_name, model_version, backend);
  if (!status.IsOk()) {
    backend->reset();
    status = Status(
        Status::Code::UNAVAILABLE,
        "Request for unknown model: " + status.Message());
  }
  return status;
}

Status
ModelRepositoryManager::Poll(
    const std::set<std::string>& models, std::set<std::string>* added,
    std::set<std::string>* deleted, std::set<std::string>* modified,
    std::set<std::string>* unmodified, ModelInfoMap* updated_infos,
    bool* all_models_polled)
{
  *all_models_polled = true;
  std::map<std::string, std::string> model_to_repository;

  // If no model is specified, poll all models in all model repositories.
  // Otherwise, only poll the specified models
  if (models.empty()) {
    std::set<std::string> duplicated_models;
    for (const auto& repository_path : repository_paths_) {
      std::set<std::string> subdirs;
      Status status = GetDirectorySubdirs(repository_path, &subdirs);
      if (!status.IsOk()) {
        LOG_ERROR << "failed to poll model repository '" << repository_path
                  << "': " << status.Message();
        *all_models_polled = false;
      } else {
        for (const auto& subdir : subdirs) {
          if (!model_to_repository.emplace(subdir, repository_path).second) {
            duplicated_models.insert(subdir);
            *all_models_polled = false;
          }
        }
      }
    }
    // If the model is not unique, mark as deleted to unload it
    for (const auto& model : duplicated_models) {
      model_to_repository.erase(model);
      deleted->insert(model);
      LOG_ERROR << "failed to poll model '" << model
                << "': not unique across all model repositories";
    }
  } else {
    for (const auto& model : models) {
      bool exists = false;
      for (const auto repository_path : repository_paths_) {
        bool exists_in_this_repo = false;
        const auto full_path = JoinPath({repository_path, model});
        Status status = FileExists(full_path, &exists_in_this_repo);
        if (!status.IsOk()) {
          LOG_ERROR << "failed to poll model repository '" << repository_path
                    << "' for model '" << model << "': " << status.Message();
          *all_models_polled = false;
        } else if (exists_in_this_repo) {
          auto res = model_to_repository.emplace(model, repository_path);
          if (res.second) {
            exists = true;
          } else {
            exists = false;
            model_to_repository.erase(res.first);
            LOG_ERROR << "failed to poll model '" << model
                      << "': not unique across all model repositories";
            *all_models_polled = false;
            break;
          }
        }
      }
      if (!exists) {
        deleted->insert(model);
      }
    }
  }

  // State of the model in terms of polling. If error happens during polling
  // an individual model, its state will fallback to different states to be
  // ignored from the polling. i.e. STATE_ADDED -> STATE_INVALID,
  // STATE_MODIFIED -> STATE_UNMODIFIED.
  enum ModelPollState {
    STATE_ADDED,
    STATE_MODIFIED,
    STATE_UNMODIFIED,
    STATE_INVALID
  };

  for (const auto& pair : model_to_repository) {
    const auto& child = pair.first;
    const auto& repository = pair.second;

    auto model_poll_state = STATE_UNMODIFIED;
    const auto full_path = JoinPath({repository, child});


    std::unique_ptr<ModelInfo> model_info;
    const auto iitr = infos_.find(child);
    // If 'child' is a new model or an existing model that has been
    // modified since the last time it was polled, then need to
    // (re)load, normalize and validate the configuration.
    int64_t mtime_ns;
    if (iitr == infos_.end()) {
      mtime_ns = GetModifiedTime(std::string(full_path));
      model_poll_state = STATE_ADDED;
    } else {
      mtime_ns = iitr->second->mtime_nsec_;
      if (IsModified(std::string(full_path), &mtime_ns)) {
        model_poll_state = STATE_MODIFIED;
      }
    }

    Status status = Status::Success;
    if (model_poll_state != STATE_UNMODIFIED) {
      model_info.reset(new ModelInfo());
      ModelConfig& model_config = model_info->model_config_;
      model_info->mtime_nsec_ = mtime_ns;
      model_info->model_repository_path_ = repository;

      // If enabled, try to automatically generate missing parts of
      // the model configuration (autofill) from the model
      // definition. In all cases normalize and validate the config.
      status = GetNormalizedModelConfig(
          full_path, backend_config_map_, autofill_, min_compute_capability_,
          &model_config);
      if (status.IsOk()) {
        status = ValidateModelConfig(
            model_config, std::string(), min_compute_capability_);
      }
      if (status.IsOk()) {
        model_info->platform_ = GetPlatform(model_config.platform());

        // Make sure the name of the model matches the name of the
        // directory. This is a somewhat arbitrary requirement but seems
        // like good practice to require it of the user. It also acts as a
        // check to make sure we don't have two different models with the
        // same name.
        if (model_config.name() != child) {
          status = Status(
              Status::Code::INVALID_ARG,
              "unexpected directory name '" + child + "' for model '" +
                  model_config.name() +
                  "', directory name must equal model name");
        }
      }

      if (!status.IsOk()) {
        if (model_poll_state == STATE_MODIFIED) {
          model_poll_state = STATE_UNMODIFIED;
        } else {
          model_poll_state = STATE_INVALID;
        }
      }
    }

    if (model_poll_state != STATE_INVALID) {
      const auto& ret = updated_infos->emplace(child, nullptr);
      if (!ret.second) {
        return Status(
            Status::Code::ALREADY_EXISTS,
            "unexpected model info for model '" + child + "'");
      }

      if (model_poll_state == STATE_UNMODIFIED) {
        ret.first->second.reset(new ModelInfo(*iitr->second));
        unmodified->insert(child);
      } else {
        ret.first->second = std::move(model_info);
        if (model_poll_state == STATE_ADDED) {
          added->insert(child);
        } else {
          modified->insert(child);
        }
      }
    }

    if (!status.IsOk()) {
      LOG_ERROR << status.Message();
      *all_models_polled = false;
    }
  }

  return Status::Success;
}

Status
ModelRepositoryManager::UpdateDependencyGraph(
    const std::set<std::string>& added, const std::set<std::string>& deleted,
    const std::set<std::string>& modified)
{
  // update dependency graph, if the state of a node is changed, all its
  // downstreams will be affected

  // deleted, drop from dependency_graph, add to missing_nodes if downstreams is
  // not empty affected_nodes are all ensembles as only ensembles are depending
  // on other models
  std::set<DependencyNode*> affected_nodes;
  std::set<DependencyNode*> updated_nodes;
  for (const auto& model_name : deleted) {
    auto it = dependency_graph_.find(model_name);
    if (it != dependency_graph_.end()) {
      // remove this node from its upstreams
      for (auto& upstream : it->second->upstreams_) {
        upstream.first->downstreams_.erase(it->second.get());
      }
      it->second->upstreams_.clear();

      if (!it->second->downstreams_.empty()) {
        UncheckDownstream(&it->second->downstreams_, &affected_nodes);
        // mark this node as missing upstream in its downstreams
        for (auto& downstream : it->second->downstreams_) {
          downstream->missing_upstreams_.emplace(it->second.get());
        }
        missing_nodes_.emplace(
            std::make_pair(model_name, std::move(it->second)));
      }

      // Make sure deleted node will not be in affected nodes
      affected_nodes.erase(it->second.get());
      dependency_graph_.erase(it);
    }
  }

  // modified, invalidate (uncheck) all downstreams
  for (const auto& model_name : modified) {
    auto it = dependency_graph_.find(model_name);
    if (it != dependency_graph_.end()) {
      UncheckDownstream(&it->second->downstreams_, &affected_nodes);
      GetModelConfig(model_name, &it->second->model_config_);
      // remove this node from its upstream node
      for (auto& upstream : it->second->upstreams_) {
        upstream.first->downstreams_.erase(it->second.get());
      }
      it->second->upstreams_.clear();
      it->second->checked_ = false;
      it->second->status_ = Status::Success;
      updated_nodes.emplace(it->second.get());
    }
  }

  // added, add to dependency_graph, if in missing_node, invalidate (uncheck)
  // and associate all downstreams, remove from missing_node
  for (const auto& model_name : added) {
    std::unique_ptr<DependencyNode> added_node;
    auto it = missing_nodes_.find(model_name);
    if (it != missing_nodes_.end()) {
      UncheckDownstream(&it->second->downstreams_, &affected_nodes);
      // remove this node from missing upstream node in its downstream nodes
      for (auto& downstream : it->second->downstreams_) {
        downstream->missing_upstreams_.erase(it->second.get());
      }

      it->second->checked_ = false;
      added_node = std::move(it->second);
      missing_nodes_.erase(it);
    } else {
      // Right now, nothing is going to be filled until validation
      added_node.reset(new DependencyNode(model_name));
    }
    GetModelConfig(model_name, &added_node->model_config_);
    updated_nodes.emplace(added_node.get());
    dependency_graph_.emplace(
        std::make_pair(model_name, std::move(added_node)));
  }

  auto& affected_ensembles = affected_nodes;
  for (auto& updated_node : updated_nodes) {
    bool is_ensemble = ConnectDependencyGraph(updated_node);
    if (is_ensemble) {
      affected_ensembles.emplace(updated_node);
    }
  }

#ifdef TRITON_ENABLE_ENSEMBLE
  ValidateEnsembleConfig(&affected_ensembles);
#endif  // TRITON_ENABLE_ENSEMBLE

  return Status::Success;
}

void
ModelRepositoryManager::UncheckDownstream(
    NodeSet* downstreams, NodeSet* updated_nodes)
{
  // Mark downstream nodes as unchecked recursively
  for (auto& node : *downstreams) {
    if (node->checked_) {
      node->checked_ = false;
      node->status_ = Status::Success;
      UncheckDownstream(&node->downstreams_, updated_nodes);
      updated_nodes->emplace(node);
    }
  }
}

bool
ModelRepositoryManager::ConnectDependencyGraph(DependencyNode* updated_node)
{
  // Check the node's model config to determine if it depends on other models
  // and if those models are present
  updated_node->upstreams_.clear();
  updated_node->missing_upstreams_.clear();
  if (updated_node->model_config_.has_ensemble_scheduling()) {
    for (const auto& step :
         updated_node->model_config_.ensemble_scheduling().step()) {
      DependencyNode* upstream_node = nullptr;
      const auto& model_name = step.model_name();
      auto dit = dependency_graph_.find(model_name);
      if (dit == dependency_graph_.end()) {
        auto mit = missing_nodes_.find(model_name);
        if (mit == missing_nodes_.end()) {
          std::unique_ptr<DependencyNode> node(new DependencyNode(model_name));
          updated_node->missing_upstreams_.emplace(node.get());
          mit = missing_nodes_.emplace(model_name, std::move(node)).first;
        }
        // Add the node to missing node's downstream so that when the missing
        // node is added, the downstreams can be found easily.
        mit->second->downstreams_.emplace(updated_node);
        upstream_node = mit->second.get();
      } else {
        dit->second->downstreams_.emplace(updated_node);
        upstream_node = dit->second.get();
      }
      auto res = updated_node->upstreams_.emplace(
          upstream_node, std::set<int64_t>({step.model_version()}));
      // If map insertion doesn't happen, the same model is required in
      // different step, insert the version to existing required version set.
      if (!res.second) {
        res.first->second.insert(step.model_version());
      }
    }
    return true;
  }
  return false;
}

Status
ModelRepositoryManager::GetModelConfig(
    const std::string& name, ModelConfig* model_config)
{
  const auto itr = infos_.find(name);
  if (itr == infos_.end()) {
    return Status(
        Status::Code::NOT_FOUND, "no configuration for model '" + name + "'");
  }

  *model_config = itr->second->model_config_;
  return Status::Success;
}

std::pair<ModelRepositoryManager::NodeSet, ModelRepositoryManager::NodeSet>
ModelRepositoryManager::ModelsToLoadUnload(const NodeSet& loaded_models)
{
  // <valid model set, invalid model set>
  std::pair<NodeSet, NodeSet> res;
  // first call to this function
  if (loaded_models.empty()) {
    for (auto& pair : dependency_graph_) {
      auto node = pair.second.get();
      // only care about nodes that are affected by the update
      if (!node->checked_) {
        if (CheckNode(node)) {
          if (node->status_.IsOk()) {
            res.first.emplace(node);
          } else {
            res.second.emplace(node);
          }
        }
      }
    }
  } else {
    for (const auto& model : loaded_models) {
      for (auto node : model->downstreams_) {
        // only care about nodes that are affected by the update
        if (!node->checked_) {
          if (CheckNode(node)) {
            if (node->status_.IsOk()) {
              res.first.emplace(node);
            } else {
              res.second.emplace(node);
            }
          }
        }
      }
    }
  }
  for (auto& node : res.first) {
    node->checked_ = true;
  }
  for (auto& node : res.second) {
    node->checked_ = true;
  }
  return res;
}

bool
ModelRepositoryManager::CheckNode(DependencyNode* node)
{
  bool node_ready = true;
  // if the node failed on validation, mark as ready as we know
  // it should not be loaded
  if (node->status_.IsOk()) {
    for (auto& upstream : node->upstreams_) {
      if (!upstream.first->checked_) {
        node_ready = false;
        break;
      }
      if (!upstream.first->status_.IsOk()) {
        node->status_ = Status(
            Status::Code::INVALID_ARG,
            "ensemble '" + node->model_name_ + "' depends on '" +
                upstream.first->model_name_ + "' which is not valid");
      } else if (upstream.first->loaded_versions_.empty()) {
        node->status_ = Status(
            Status::Code::INVALID_ARG,
            "ensemble '" + node->model_name_ + "' depends on '" +
                upstream.first->model_name_ + "' which has no loaded version");
      } else {
        for (const auto& required_version : upstream.second) {
          if (required_version == -1) {
            continue;
          }

          auto it = upstream.first->loaded_versions_.find(required_version);
          if (it == upstream.first->loaded_versions_.end()) {
            node->status_ = Status(
                Status::Code::INVALID_ARG,
                "ensemble '" + node->model_name_ + "' depends on '" +
                    upstream.first->model_name_ + "' whose required version " +
                    std::to_string(required_version) + " is not loaded");
          }
        }
      }
      if (!node->status_.IsOk()) {
        break;
      }
    }
  }
  return node_ready;
}

Status
ModelRepositoryManager::VersionsToLoad(
    const std::string model_repository_path, const std::string& name,
    const ModelConfig& model_config, std::set<int64_t>* versions)
{
  versions->clear();

  // Get integral number of the version directory
  const auto model_path = JoinPath({model_repository_path, name});
  std::set<std::string> subdirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(model_path, &subdirs));
  std::set<int64_t, std::greater<int64_t>> existing_versions;
  for (const auto& subdir : subdirs) {
    if (subdir == kWarmupDataFolder) {
      continue;
    }
    if ((subdir.length() > 1) && (subdir.front() == '0')) {
      LOG_WARNING << "ignore version directory '" << subdir
                  << "' which contains leading zeros in its directory name";
      continue;
    }
    try {
      int64_t version = std::stoll(subdir);
      existing_versions.insert(version);
    }
    catch (const std::invalid_argument& ia) {
      LOG_ERROR << "failed to convert version directory '" << subdir
                << "' to integral number";
    }
  }

  if (model_config.version_policy().has_specific()) {
    for (const auto& v : model_config.version_policy().specific().versions()) {
      // Only load the specific versions that are presented in model directory
      bool version_not_exist = existing_versions.insert(v).second;
      if (!version_not_exist) {
        versions->emplace(v);
      } else {
        LOG_ERROR << "version " << v << " is specified for model '" << name
                  << "', but the version directory is not present";
      }
    }
  } else {
    if (model_config.version_policy().has_latest()) {
      // std::set is sorted with std::greater
      for (const auto& v : existing_versions) {
        if (versions->size() >=
            model_config.version_policy().latest().num_versions()) {
          break;
        }
        versions->emplace(v);
      }
    } else {
      // all
      versions->insert(existing_versions.begin(), existing_versions.end());
    }
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
