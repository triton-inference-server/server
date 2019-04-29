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

#include "src/core/model_repository_manager.h"

#include "src/core/backend.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"
#include "src/core/server_status.h"
#include "src/servables/caffe2/netdef_bundle.h"
#include "src/servables/caffe2/netdef_bundle.pb.h"
#include "src/servables/custom/custom_bundle.h"
#include "src/servables/custom/custom_bundle.pb.h"
#include "src/servables/ensemble/ensemble_bundle.h"
#include "src/servables/ensemble/ensemble_bundle.pb.h"
#include "src/servables/tensorflow/graphdef_bundle.h"
#include "src/servables/tensorflow/graphdef_bundle.pb.h"
#include "src/servables/tensorflow/savedmodel_bundle.h"
#include "src/servables/tensorflow/savedmodel_bundle.pb.h"
#include "src/servables/tensorrt/plan_bundle.h"
#include "src/servables/tensorrt/plan_bundle.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/config/platform_config.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace nvidia { namespace inferenceserver {

struct ModelRepositoryManager::ModelInfo {
  int64_t mtime_nsec_;
  ModelConfig model_config_;
  tfs::ModelConfig tfs_model_config_;
  Platform platform_;
};

namespace {

void
BuildPlatformConfigMap(
    const std::string& version, const std::string& model_store_path,
    const bool strict_model_config, const float tf_gpu_memory_fraction,
    const bool tf_allow_soft_placement, PlatformConfigMap* platform_configs,
    tfs::PlatformConfigMap* tfs_platform_configs)
{
  ::google::protobuf::Any graphdef_source_adapter_config;
  ::google::protobuf::Any saved_model_source_adapter_config;
  ::google::protobuf::Any plan_source_adapter_config;
  ::google::protobuf::Any netdef_source_adapter_config;
  ::google::protobuf::Any custom_source_adapter_config;
  ::google::protobuf::Any ensemble_source_adapter_config;

  //// Tensorflow GraphDef
  {
    GraphDefBundleSourceAdapterConfig graphdef_config;

    graphdef_config.set_autofill(!strict_model_config);

    // Tensorflow session config
    if (tf_gpu_memory_fraction == 0.0) {
      graphdef_config.mutable_session_config()
          ->mutable_gpu_options()
          ->set_allow_growth(true);
    } else {
      graphdef_config.mutable_session_config()
          ->mutable_gpu_options()
          ->set_per_process_gpu_memory_fraction(tf_gpu_memory_fraction);
    }

    graphdef_config.mutable_session_config()->set_allow_soft_placement(
        tf_allow_soft_placement);
    graphdef_source_adapter_config.PackFrom(graphdef_config);
  }

  //// Tensorflow SavedModel
  {
    SavedModelBundleSourceAdapterConfig saved_model_config;

    saved_model_config.set_autofill(!strict_model_config);

    if (tf_gpu_memory_fraction == 0.0) {
      saved_model_config.mutable_session_config()
          ->mutable_gpu_options()
          ->set_allow_growth(true);
    } else {
      saved_model_config.mutable_session_config()
          ->mutable_gpu_options()
          ->set_per_process_gpu_memory_fraction(tf_gpu_memory_fraction);
    }

    saved_model_config.mutable_session_config()->set_allow_soft_placement(
        tf_allow_soft_placement);
    saved_model_source_adapter_config.PackFrom(saved_model_config);
  }

  //// Caffe NetDef
  {
    NetDefBundleSourceAdapterConfig netdef_config;
    netdef_config.set_autofill(!strict_model_config);
    netdef_source_adapter_config.PackFrom(netdef_config);
  }

  //// TensorRT
  {
    PlanBundleSourceAdapterConfig plan_config;
    plan_config.set_autofill(!strict_model_config);
    plan_source_adapter_config.PackFrom(plan_config);
  }

  //// Custom
  {
    CustomBundleSourceAdapterConfig custom_config;
    custom_config.set_inference_server_version(version);
    custom_config.set_model_repository_path(model_store_path);
    custom_source_adapter_config.PackFrom(custom_config);
  }

  //// Ensemble
  {
    EnsembleBundleSourceAdapterConfig ensemble_config;
    ensemble_source_adapter_config.PackFrom(ensemble_config);
  }

  (*platform_configs)[kTensorFlowGraphDefPlatform] =
      graphdef_source_adapter_config;
  (*platform_configs)[kTensorFlowSavedModelPlatform] =
      saved_model_source_adapter_config;
  (*platform_configs)[kCaffe2NetDefPlatform] = netdef_source_adapter_config;
  (*platform_configs)[kTensorRTPlanPlatform] = plan_source_adapter_config;
  (*platform_configs)[kCustomPlatform] = custom_source_adapter_config;
  (*platform_configs)[kEnsemblePlatform] = ensemble_source_adapter_config;

  // Must also return the configs in format required by TFS for
  // ServerCore.
  (*(*tfs_platform_configs
          ->mutable_platform_configs())[kTensorFlowGraphDefPlatform]
        .mutable_source_adapter_config()) = graphdef_source_adapter_config;
  (*(*tfs_platform_configs
          ->mutable_platform_configs())[kTensorFlowSavedModelPlatform]
        .mutable_source_adapter_config()) = saved_model_source_adapter_config;
  (*(*tfs_platform_configs->mutable_platform_configs())[kCaffe2NetDefPlatform]
        .mutable_source_adapter_config()) = netdef_source_adapter_config;
  (*(*tfs_platform_configs->mutable_platform_configs())[kTensorRTPlanPlatform]
        .mutable_source_adapter_config()) = plan_source_adapter_config;
  (*(*tfs_platform_configs->mutable_platform_configs())[kCustomPlatform]
        .mutable_source_adapter_config()) = custom_source_adapter_config;
  (*(*tfs_platform_configs->mutable_platform_configs())[kEnsemblePlatform]
        .mutable_source_adapter_config()) = ensemble_source_adapter_config;
}

ModelReadyState
ManagerStateToModelReadyState(tfs::ServableState::ManagerState manager_state)
{
  switch (manager_state) {
    case tfs::ServableState::ManagerState::kLoading:
      return ModelReadyState::MODEL_LOADING;
      break;
    case tfs::ServableState::ManagerState::kUnloading:
      return ModelReadyState::MODEL_UNLOADING;
      break;
    case tfs::ServableState::ManagerState::kAvailable:
      return ModelReadyState::MODEL_READY;
      break;
    default:
      return ModelReadyState::MODEL_UNAVAILABLE;
      break;
  }
  return ModelReadyState::MODEL_UNKNOWN;
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

  // If 'path' is a file return its mtime.
  if (!path_is_dir) {
    int64_t mtime;
    status = FileModificationTime(path, &mtime);
    if (!status.IsOk()) {
      LOG_ERROR << "Failed to determine modification time for '" << path
                << "': " << status.AsString();
      return 0;
    }

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

  int64_t mtime = 0;
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

}  // namespace

ModelRepositoryManager* ModelRepositoryManager::singleton = nullptr;

class BackendHandleImpl : public ModelRepositoryManager::BackendHandle {
 public:
  ~BackendHandleImpl() = default;
  BackendHandleImpl(
      const Platform& platform, const tfs::ModelSpec& model_spec,
      tfs::ServerCore* core);
  InferenceBackend* GetInferenceBackend() override { return is_; }

 private:
  InferenceBackend* is_;
  tfs::ServableHandle<GraphDefBundle> graphdef_bundle_;
  tfs::ServableHandle<PlanBundle> plan_bundle_;
  tfs::ServableHandle<NetDefBundle> netdef_bundle_;
  tfs::ServableHandle<SavedModelBundle> saved_model_bundle_;
  tfs::ServableHandle<CustomBundle> custom_bundle_;
  tfs::ServableHandle<EnsembleBundle> ensemble_bundle_;
};

BackendHandleImpl::BackendHandleImpl(
    const Platform& platform, const tfs::ModelSpec& model_spec,
    tfs::ServerCore* core)
{
  tensorflow::Status tfstatus;
  is_ = nullptr;

  switch (platform) {
    case Platform::PLATFORM_TENSORFLOW_GRAPHDEF:
      tfstatus = core->GetServableHandle(model_spec, &(graphdef_bundle_));
      if (tfstatus.ok()) {
        is_ = static_cast<InferenceBackend*>(graphdef_bundle_.get());
      }
      break;
    case Platform::PLATFORM_TENSORFLOW_SAVEDMODEL:
      tfstatus = core->GetServableHandle(model_spec, &(saved_model_bundle_));
      if (tfstatus.ok()) {
        is_ = static_cast<InferenceBackend*>(saved_model_bundle_.get());
      }
      break;
    case Platform::PLATFORM_TENSORRT_PLAN:
      tfstatus = core->GetServableHandle(model_spec, &(plan_bundle_));
      if (tfstatus.ok()) {
        is_ = static_cast<InferenceBackend*>(plan_bundle_.get());
      }
      break;
    case Platform::PLATFORM_CAFFE2_NETDEF:
      tfstatus = core->GetServableHandle(model_spec, &(netdef_bundle_));
      if (tfstatus.ok()) {
        is_ = static_cast<InferenceBackend*>(netdef_bundle_.get());
      }
      break;
    case Platform::PLATFORM_CUSTOM:
      tfstatus = core->GetServableHandle(model_spec, &(custom_bundle_));
      if (tfstatus.ok()) {
        is_ = static_cast<InferenceBackend*>(custom_bundle_.get());
      }
      break;
    case Platform::PLATFORM_ENSEMBLE:
      tfstatus = core->GetServableHandle(model_spec, &(ensemble_bundle_));
      if (tfstatus.ok()) {
        is_ = static_cast<InferenceBackend*>(ensemble_bundle_.get());
      }
      break;
    default:
      break;
  }
}

ModelRepositoryManager::ModelRepositoryManager(
    const std::shared_ptr<ServerStatusManager>& status_manager,
    const std::string& repository_path,
    const PlatformConfigMap& platform_config_map, const bool autofill,
    const bool polling_enabled)
    : repository_path_(repository_path),
      platform_config_map_(platform_config_map), autofill_(autofill),
      polling_enabled_(polling_enabled), status_manager_(status_manager)
{
}

ModelRepositoryManager::~ModelRepositoryManager()
{
  singleton = nullptr;
}

Status
ModelRepositoryManager::Create(
    const std::string& server_version,
    const std::shared_ptr<ServerStatusManager>& status_manager,
    const std::string& repository_path, const bool strict_model_config,
    const float tf_gpu_memory_fraction, const bool tf_allow_soft_placement,
    const uint32_t repository_poll_secs, const bool polling_enabled,
    std::unique_ptr<ModelRepositoryManager>* model_repository_manager)
{
  if (singleton != nullptr) {
    return Status(
        RequestStatusCode::ALREADY_EXISTS,
        "ModelRepositoryManager singleton already created");
  }

  // For ServerCore Options, we leave servable_state_monitor_creator unspecified
  // so the default servable_state_monitor_creator will be used.
  tfs::ServerCore::Options options;

  // Set some default values in Options
  options.aspired_version_policy = std::unique_ptr<tfs::AspiredVersionPolicy>(
      new tfs::AvailabilityPreservingPolicy);

  // If not polling the model repository then set the poll secs to 0
  // in TFS so that repository is only checked a single time at
  // startup.
  // [TODO] Remove 'repository_poll_secs' parameter once ModelRepositoryManager
  // is improved as model version will also be monitored and polled
  // on PollAndUpdate()
  options.max_num_load_retries = 0;
  options.file_system_poll_wait_seconds = repository_poll_secs;

  PlatformConfigMap platform_config_map;

  BuildPlatformConfigMap(
      server_version, repository_path, strict_model_config,
      tf_gpu_memory_fraction, tf_allow_soft_placement, &platform_config_map,
      &options.platform_config_map);

  LOG_VERBOSE(1) << options.platform_config_map.DebugString();

  // Not setting the singleton / smart pointer directly because error on TFS
  // core creation may not be considered as initialization failure. So only
  // setting it before core creation to simplify clean up
  std::unique_ptr<ModelRepositoryManager> local_manager(
      new ModelRepositoryManager(
          status_manager, repository_path, platform_config_map,
          !strict_model_config, polling_enabled));

  // Similar to PollAndUpdate(), but simplier
  std::set<std::string> added, deleted, modified, unmodified;
  if (polling_enabled) {
    RETURN_IF_ERROR(
        local_manager->Poll(&added, &deleted, &modified, &unmodified));
  }
  if (!deleted.empty() || !modified.empty() || !unmodified.empty()) {
    return Status(
        RequestStatusCode::INTERNAL,
        "Unexpected initial state for model repository");
  }

  for (const auto& name : added) {
    tfs::ModelConfig* tfs_config =
        options.model_server_config.mutable_model_config_list()->add_config();
    Status status = local_manager->GetTFSModelConfig(name, tfs_config);
    if (!status.IsOk()) {
      return Status(
          RequestStatusCode::INTERNAL,
          "Internal: model repository manager inconsistency");
    }

    ModelConfig model_config;
    RETURN_IF_ERROR(
        local_manager->GetModelConfigFromInstance(name, &model_config));
    status = local_manager->status_manager_->InitForModel(name, model_config);
    if (!status.IsOk()) {
      return status;
    }
  }

  LOG_VERBOSE(1) << options.model_server_config.DebugString();

  // Create the server core. We assume that any failure is due to a
  // model not loading correctly so we just continue if not exiting on
  // error.
  *model_repository_manager = std::move(local_manager);
  singleton = model_repository_manager->get();
  RETURN_IF_TF_ERROR(
      tfs::ServerCore::Create(std::move(options), &singleton->core_));

  return Status::Success;
}

Status
ModelRepositoryManager::GetModelConfig(
    const std::string& name, ModelConfig* model_config)
{
  return singleton->GetModelConfigFromInstance(name, model_config);
}

Status
ModelRepositoryManager::PollAndUpdate()
{
  if (!polling_enabled_) {
    return Status(RequestStatusCode::INVALID, "polling is disabled");
  }
  std::set<std::string> added, deleted, modified, unmodified;
  RETURN_IF_ERROR(Poll(&added, &deleted, &modified, &unmodified));
  // Nothing to do if no model adds, deletes or modifies.
  if (added.empty() && deleted.empty() && modified.empty()) {
    return Status::Success;
  }

  // [TODO] Once the model repository manager is improved,
  // model load / unload should be done in separate thread

  // There was a change in the model repository so need to
  // create a new TFS model configuration and reload it into the
  // server to cause the appropriate models to be loaded and
  // unloaded.
  tfs::ModelServerConfig msc;
  msc.mutable_model_config_list();

  // Added models should be loaded and be initialized for status
  // reporting.
  for (const auto& name : added) {
    tfs::ModelConfig* tfs_config =
        msc.mutable_model_config_list()->add_config();
    RETURN_IF_ERROR(GetTFSModelConfig(name, tfs_config));
    ModelConfig model_config;
    RETURN_IF_ERROR(GetModelConfigFromInstance(name, &model_config));
    RETURN_IF_ERROR(status_manager_->InitForModel(name, model_config));
  }

  // Keep unmodified models...
  for (const auto& name : unmodified) {
    tfs::ModelConfig* tfs_config =
        msc.mutable_model_config_list()->add_config();
    RETURN_IF_ERROR(GetTFSModelConfig(name, tfs_config));
  }

  RETURN_IF_TF_ERROR(core_->ReloadConfig(msc));

  // If there are any modified model, (re)load them to pick up
  // the changes. We want to keep the current status information
  // so don't re-init it.
  if (!modified.empty()) {
    for (const auto& name : modified) {
      tfs::ModelConfig* tfs_config =
          msc.mutable_model_config_list()->add_config();
      RETURN_IF_ERROR(GetTFSModelConfig(name, tfs_config));
      ModelConfig model_config;
      RETURN_IF_ERROR(GetModelConfigFromInstance(name, &model_config));
      RETURN_IF_ERROR(
          status_manager_->UpdateConfigForModel(name, model_config));
    }

    RETURN_IF_TF_ERROR(core_->ReloadConfig(msc));
  }
  return Status::Success;
}

Status
ModelRepositoryManager::LoadUnloadModel(
    const std::string& model_name, ActionType type,
    std::function<void(Status)> OnCompleteUpdate)
{
  if (polling_enabled_) {
    return Status(
        RequestStatusCode::INVALID,
        "explicit model load / unload is not allowed if polling is enabled");
  }
  // [TODO] model load / unload should be done in separate thread
  Status status = Status(RequestStatusCode::UNSUPPORTED, "not implemented");
  OnCompleteUpdate(status);
  return status;
}

Status
ModelRepositoryManager::UnloadAllModels()
{
  // Reload an empty configuration to cause all models to unload.
  tfs::ModelServerConfig msc;
  msc.mutable_model_config_list();
  tensorflow::Status tfstatus = core_->ReloadConfig(msc);
  if (!tfstatus.ok()) {
    return Status(
        RequestStatusCode::INTERNAL,
        "Failed to gracefully unload models: " + tfstatus.error_message());
  }
  return Status::Success;
}

const ModelRepositoryManager::ModelMap
ModelRepositoryManager::GetLiveBackendStates()
{
  // [TODO] maintain its own ModelMap
  ModelMap res;
  const tfs::ServableStateMonitor& monitor = *(core_->servable_state_monitor());
  const auto& live_models = monitor.GetLiveServableStates();
  for (const auto& m : live_models) {
    VersionStateMap map;
    for (const auto& v : m.second) {
      map[v.first] =
          ManagerStateToModelReadyState(v.second.state.manager_state);
    }
    res[m.first] = map;
  }
  return res;
}

const ModelRepositoryManager::VersionStateMap
ModelRepositoryManager::GetVersionStates(const std::string& model_name)
{
  VersionStateMap res;
  const tfs::ServableStateMonitor& monitor = *(core_->servable_state_monitor());
  const tensorflow::serving::ServableStateMonitor::VersionMap
      versions_and_states = monitor.GetVersionStates(model_name);
  for (const auto& version_and_state : versions_and_states) {
    const int64_t version = version_and_state.first;
    const tensorflow::serving::ServableState& servable_state =
        version_and_state.second.state;

    ModelReadyState ready_state =
        ManagerStateToModelReadyState(servable_state.manager_state);

    if (ready_state != ModelReadyState::MODEL_UNKNOWN) {
      res[version] = ready_state;
    }
  }
  return res;
}

Status
ModelRepositoryManager::GetBackendHandle(
    const std::string& model_name, const int64_t model_version,
    std::unique_ptr<BackendHandle>* handle)
{
  // Create the model-spec. A negative version indicates that the
  // latest version of the model should be used.
  tfs::ModelSpec model_spec;
  model_spec.set_name(model_name);
  if (model_version >= 0) {
    model_spec.mutable_version()->set_value(model_version);
  }

  // Get the InferenceBackend appropriate for the request.
  Platform platform;
  Status status = GetModelPlatform(model_name, &platform);
  if (status.IsOk()) {
    handle->reset(new BackendHandleImpl(platform, model_spec, core_.get()));
    if ((*handle)->GetInferenceBackend() == nullptr) {
      handle->reset();
    }
  }
  if (*handle == nullptr) {
    status = Status(
        RequestStatusCode::UNAVAILABLE,
        "Inference request for unknown model '" + model_name + "'");
  }

  return status;
}

Status
ModelRepositoryManager::Poll(
    std::set<std::string>* added, std::set<std::string>* deleted,
    std::set<std::string>* modified, std::set<std::string>* unmodified)
{
  // Serialize all polling operation...
  std::lock_guard<std::mutex> lock(poll_mu_);

  added->clear();
  deleted->clear();
  modified->clear();
  unmodified->clear();

  // We don't modify 'infos_' in place to minimize how long we need to
  // hold the lock and also prevent any partial changes to do an error
  // during processing.
  ModelInfoMap new_infos;

  // Each subdirectory of repository path is a model directory from
  // which we read the model configuration.
  std::set<std::string> subdirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(repository_path_, &subdirs));

  for (const auto& child : subdirs) {
    const auto full_path = JoinPath({repository_path_, child});

    // If 'child' is a new model or an existing model that has been
    // modified since the last time it was polled, then need to
    // (re)load, normalize and validate the configuration.
    bool need_load = false;
    int64_t mtime_ns;
    const auto iitr = infos_.find(child);
    if (iitr == infos_.end()) {
      added->insert(child);
      mtime_ns = GetModifiedTime(std::string(full_path));
      need_load = true;
    } else {
      mtime_ns = iitr->second->mtime_nsec_;
      if (IsModified(std::string(full_path), &mtime_ns)) {
        modified->insert(child);
        need_load = true;
      } else {
        unmodified->insert(child);
        const auto& ret = new_infos.emplace(child, nullptr);
        if (!ret.second) {
          return Status(
              RequestStatusCode::ALREADY_EXISTS,
              "unexpected model info for model '" + child + "'");
        }

        std::unique_ptr<ModelInfo>& model_info = ret.first->second;
        model_info.reset(new ModelInfo(*iitr->second));
      }
    }

    if (need_load) {
      const auto& ret = new_infos.emplace(child, nullptr);
      if (!ret.second) {
        return Status(
            RequestStatusCode::ALREADY_EXISTS,
            "unexpected model info for model '" + child + "'");
      }

      std::unique_ptr<ModelInfo>& model_info = ret.first->second;
      model_info.reset(new ModelInfo());
      ModelConfig& model_config = model_info->model_config_;
      tfs::ModelConfig& tfs_config = model_info->tfs_model_config_;
      model_info->mtime_nsec_ = mtime_ns;

      // If enabled, try to automatically generate missing parts of
      // the model configuration (autofill) from the model
      // definition. In all cases normalize and validate the config.
      RETURN_IF_ERROR(GetNormalizedModelConfig(
          full_path, platform_config_map_, autofill_, &model_config));
      RETURN_IF_ERROR(ValidateModelConfig(model_config, std::string()));

      model_info->platform_ = GetPlatform(model_config.platform());

      // Make sure the name of the model matches the name of the
      // directory. This is a somewhat arbitrary requirement but seems
      // like good practice to require it of the user. It also acts as a
      // check to make sure we don't have two different models with the
      // same name.
      if (model_config.name() != child) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "unexpected directory name '" + child + "' for model '" +
                model_config.name() +
                "', directory name must equal model name");
      }

      tfs_config.set_name(model_config.name());
      tfs_config.set_base_path(full_path);
      tfs_config.set_model_platform(model_config.platform());

      // Create the appropriate TFS version policy from the model
      // configuration policy.
      if (model_config.version_policy().has_latest()) {
        tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::Latest
            latest;
        latest.set_num_versions(
            model_config.version_policy().latest().num_versions());
        tfs_config.mutable_model_version_policy()->mutable_latest()->CopyFrom(
            latest);
      } else if (model_config.version_policy().has_all()) {
        tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::All all;
        tfs_config.mutable_model_version_policy()->mutable_all()->CopyFrom(all);
      } else if (model_config.version_policy().has_specific()) {
        tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::Specific
            specific;
        specific.mutable_versions()->CopyFrom(
            model_config.version_policy().specific().versions());
        tfs_config.mutable_model_version_policy()->mutable_specific()->CopyFrom(
            specific);
      } else {
        return Status(
            RequestStatusCode::INTERNAL,
            "expected version policy for model '" + model_config.name());
      }
    }
  }

  // Anything in 'infos_' that is not in "added", "modified", or
  // "unmodified" is deleted.
  for (const auto& pr : infos_) {
    if ((added->find(pr.first) == added->end()) &&
        (modified->find(pr.first) == modified->end()) &&
        (unmodified->find(pr.first) == unmodified->end())) {
      deleted->insert(pr.first);
    }
  }

  // Swap the new infos in place under a short-lived lock and only if
  // there were no errors encountered during polling.
  {
    std::lock_guard<std::mutex> lock(infos_mu_);
    infos_.swap(new_infos);
  }

  return Status::Success;
}


Status
ModelRepositoryManager::GetModelConfigFromInstance(
    const std::string& name, ModelConfig* model_config)
{
  std::lock_guard<std::mutex> lock(infos_mu_);

  const auto itr = infos_.find(name);
  if (itr == infos_.end()) {
    return Status(
        RequestStatusCode::NOT_FOUND,
        "no configuration for model '" + name + "'");
  }

  *model_config = itr->second->model_config_;
  return Status::Success;
}

Status
ModelRepositoryManager::GetTFSModelConfig(
    const std::string& name, tfs::ModelConfig* tfs_model_config)
{
  std::lock_guard<std::mutex> lock(infos_mu_);

  const auto itr = infos_.find(name);
  if (itr == infos_.end()) {
    return Status(
        RequestStatusCode::NOT_FOUND,
        "no TFS configuration for model '" + name + "'");
  }

  *tfs_model_config = itr->second->tfs_model_config_;
  return Status::Success;
}

Status
ModelRepositoryManager::GetModelPlatform(
    const std::string& name, Platform* platform)
{
  std::lock_guard<std::mutex> lock(infos_mu_);

  const auto itr = infos_.find(name);
  if (itr == infos_.end()) {
    *platform = Platform::PLATFORM_UNKNOWN;
  } else {
    *platform = itr->second->platform_;
  }

  if (*platform == Platform::PLATFORM_UNKNOWN) {
    return Status(
        RequestStatusCode::NOT_FOUND,
        "unknown platform for model '" + name + "'");
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
