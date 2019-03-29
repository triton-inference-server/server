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

#include "src/core/server.h"

#include <cuda_profiler_api.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <csignal>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "src/core/api.pb.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"
#include "src/core/model_repository_manager.h"
#include "src/core/profile.h"
#include "src/core/provider.h"
#include "src/core/request_status.h"
#include "src/core/server.h"
#include "src/core/server_status.pb.h"
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
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/config/platform_config.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace tfs = tensorflow::serving;

namespace nvidia { namespace inferenceserver {

namespace {

// Scoped increment / decrement of atomic
class ScopedAtomicIncrement {
 public:
  explicit ScopedAtomicIncrement(std::atomic<uint64_t>& counter)
      : counter_(counter)
  {
    counter_++;
  }

  ~ScopedAtomicIncrement() { counter_--; }

 private:
  std::atomic<uint64_t>& counter_;
};

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

}  // namespace

//
// InferenceServer
//
InferenceServer::InferenceServer()
    : ready_state_(ServerReadyState::SERVER_INVALID), next_request_id_(1)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  start_time_ns_ = ts.tv_sec * NANOS_PER_SECOND + ts.tv_nsec;

  const char* vstr = getenv("TENSORRT_SERVER_VERSION");
  if (vstr != nullptr) {
    version_.assign(vstr);
  }

  id_ = "inference:0";
  strict_model_config_ = true;
  strict_readiness_ = true;
  profiling_enabled_ = false;
  exit_timeout_secs_ = 30;
  repository_poll_secs_ = 15;

  tf_soft_placement_enabled_ = true;
  tf_gpu_memory_fraction_ = 0.0;

  inflight_request_counter_ = 0;

  status_manager_.reset(new ServerStatusManager(version_));
}

bool
InferenceServer::Init()
{
  Status status;

  ready_state_ = ServerReadyState::SERVER_INITIALIZING;

  LOG_INFO << "Initializing TensorRT Inference Server";

  if (model_store_path_.empty()) {
    LOG_ERROR << "--model-store must be specified";
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return false;
  }

  // Disable profiling at server start. Server API can be used to
  // start/stop profiling.
  status = ProfileStopAll();
  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return false;
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
  options.max_num_load_retries = 0;
  options.file_system_poll_wait_seconds = repository_poll_secs_;

  PlatformConfigMap platform_configs;
  BuildPlatformConfigMap(
      version_, model_store_path_, strict_model_config_,
      tf_gpu_memory_fraction_, tf_soft_placement_enabled_, &platform_configs,
      &options.platform_config_map);
  LOG_VERBOSE(1) << options.platform_config_map.DebugString();

  // Create the global manager for the repository. Add all models'
  // into the server core 'options' so that they are eagerly loaded
  // below when ServerCore is created.
  status = ModelRepositoryManager::Create(
      model_store_path_, platform_configs, !strict_model_config_);
  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return false;
  }

  std::set<std::string> added, deleted, modified, unmodified;
  status =
      ModelRepositoryManager::Poll(&added, &deleted, &modified, &unmodified);
  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return false;
  }

  if (!deleted.empty() || !modified.empty() || !unmodified.empty()) {
    LOG_ERROR << "Unexpected initial state for model repository";
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return false;
  }

  for (const auto& name : added) {
    tfs::ModelConfig* tfs_config =
        options.model_server_config.mutable_model_config_list()->add_config();
    status = ModelRepositoryManager::GetTFSModelConfig(name, tfs_config);
    if (!status.IsOk()) {
      LOG_ERROR << "Internal: model repository manager inconsistency";
      ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
      return false;
    }

    status = status_manager_->InitForModel(name);
    if (!status.IsOk()) {
      LOG_ERROR << status.Message();
      ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
      return false;
    }
  }

  LOG_VERBOSE(1) << options.model_server_config.DebugString();

  // Create the server core. We assume that any failure is due to a
  // model not loading correctly so we just continue if not exiting on
  // error.
  tensorflow::Status tfstatus =
      tfs::ServerCore::Create(std::move(options), &core_);
  if (!tfstatus.ok()) {
    LOG_ERROR << tfstatus;
    ready_state_ = ServerReadyState::SERVER_READY;
    return false;
  }

  ready_state_ = ServerReadyState::SERVER_READY;
  return true;
}

bool
InferenceServer::Stop()
{
  ready_state_ = ServerReadyState::SERVER_EXITING;

  if (core_ == nullptr) {
    LOG_INFO << "No server context available. Exiting immediately.";
    return true;
  } else {
    LOG_INFO << "Waiting for in-flight inferences to complete.";
  }

  // Reload an empty configuration to cause all models to unload.
  tfs::ModelServerConfig msc;
  msc.mutable_model_config_list();
  tensorflow::Status tfstatus = core_->ReloadConfig(msc);
  if (!tfstatus.ok()) {
    LOG_ERROR << "Failed to gracefully unload models: " << tfstatus;
  }

  // Wait for all in-flight requests to complete and all loaded models
  // to unload, or for the exit timeout to expire.
  const tfs::ServableStateMonitor& monitor = *core_->servable_state_monitor();
  uint32_t exit_timeout_iters = exit_timeout_secs_;

  while (true) {
    const auto& live_models = monitor.GetLiveServableStates();

    LOG_INFO << "Timeout " << exit_timeout_iters << ": Found "
             << live_models.size() << " live models and "
             << inflight_request_counter_ << " in-flight requests";
    if (LOG_VERBOSE_IS_ON(1)) {
      for (const auto& m : live_models) {
        for (const auto& v : m.second) {
          LOG_VERBOSE(1) << m.first << "v" << v.first << ": "
                         << v.second.DebugString();
        }
      }
    }

    if ((live_models.size() == 0) && (inflight_request_counter_ == 0)) {
      return true;
    }
    if (exit_timeout_iters <= 0) {
      LOG_ERROR << "Exit timeout expired. Exiting immediately.";
      break;
    }

    exit_timeout_iters--;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  return false;
}

Status
InferenceServer::PollModelRepository()
{
  LOG_VERBOSE(1) << "Polling model repository";

  // Look for changes and update the loaded model configurations
  // appropriately.
  if (ready_state_ == ServerReadyState::SERVER_READY) {
    std::set<std::string> added, deleted, modified, unmodified;
    RETURN_IF_ERROR(
        ModelRepositoryManager::Poll(&added, &deleted, &modified, &unmodified));

    // Nothing to do if no model adds, deletes or modifies.
    if (added.empty() && deleted.empty() && modified.empty()) {
      return Status::Success;
    }

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
      RETURN_IF_ERROR(
          ModelRepositoryManager::GetTFSModelConfig(name, tfs_config));
      RETURN_IF_ERROR(status_manager_->InitForModel(name));
    }

    // Keep unmodified models...
    for (const auto& name : unmodified) {
      tfs::ModelConfig* tfs_config =
          msc.mutable_model_config_list()->add_config();
      RETURN_IF_ERROR(
          ModelRepositoryManager::GetTFSModelConfig(name, tfs_config));
    }

    RETURN_IF_TF_ERROR(core_->ReloadConfig(msc));

    // If there are any modified model, (re)load them to pick up
    // the changes. We want to keep the current status information
    // so don't re-init it.
    if (!modified.empty()) {
      for (const auto& name : modified) {
        tfs::ModelConfig* tfs_config =
            msc.mutable_model_config_list()->add_config();
        RETURN_IF_ERROR(
            ModelRepositoryManager::GetTFSModelConfig(name, tfs_config));
        RETURN_IF_ERROR(status_manager_->UpdateConfigForModel(name));
      }

      RETURN_IF_TF_ERROR(core_->ReloadConfig(msc));
    }
  }

  return Status::Success;
}

void
InferenceServer::HandleHealth(
    RequestStatus* request_status, bool* health, const std::string& mode)
{
  *health = false;

  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    RequestStatusFactory::Create(
        request_status, 0, id_, RequestStatusCode::UNAVAILABLE,
        "Server exiting");
    return;
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);
  const uint64_t request_id = NextRequestId();

  // Server is considered live if it can respond to this health
  // request and it was able to initialize.
  if (mode == "live") {
    *health =
        ((ready_state_ != ServerReadyState::SERVER_INVALID) &&
         (ready_state_ != ServerReadyState::SERVER_FAILED_TO_INITIALIZE));
    RequestStatusFactory::Create(
        request_status, request_id, id_, RequestStatusCode::SUCCESS);
  }
  // Server is considered ready if it is in the ready state.
  // Additionally can report ready only when all models are ready.
  else if (mode == "ready") {
    *health = (ready_state_ == ServerReadyState::SERVER_READY);
    if (*health && strict_readiness_) {
      // Strict readiness... get the model status and make sure all
      // models are ready.
      tfs::ServableStateMonitor* monitor = nullptr;
      if (core_ != nullptr) {
        monitor = core_->servable_state_monitor();
      }

      ServerStatus server_status;
      Status status = status_manager_->Get(
          &server_status, id_, ready_state_, UptimeNs(), monitor);

      *health = status.IsOk();
      if (*health) {
        for (const auto& ms : server_status.model_status()) {
          for (const auto& vs : ms.second.version_status()) {
            if (vs.second.ready_state() != ModelReadyState::MODEL_READY) {
              *health = false;
              goto strict_done;
            }
          }
        }
      strict_done:;
      }
    }

    RequestStatusFactory::Create(
        request_status, request_id, id_, RequestStatusCode::SUCCESS);
  } else {
    RequestStatusFactory::Create(
        request_status, request_id, id_, RequestStatusCode::UNKNOWN,
        "unknown health mode '" + mode + "'");
  }
}

void
InferenceServer::HandleProfile(
    RequestStatus* request_status, const std::string& cmd)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    RequestStatusFactory::Create(
        request_status, 0, id_, RequestStatusCode::UNAVAILABLE,
        "Server not ready");
    return;
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);
  const uint64_t request_id = NextRequestId();

  if (!profiling_enabled_) {
    RequestStatusFactory::Create(
        request_status, request_id, id_, RequestStatusCode::UNSUPPORTED,
        "Profile API not enabled");
  } else if (cmd == "start") {
    RequestStatusFactory::Create(
        request_status, request_id, id_, ProfileStartAll());
  } else if (cmd == "stop") {
    RequestStatusFactory::Create(
        request_status, request_id, id_, ProfileStopAll());
  } else {
    RequestStatusFactory::Create(
        request_status, request_id, id_, RequestStatusCode::INVALID_ARG,
        "Unknown profile command '" + std::string(cmd) + "'");
  }
}

void
InferenceServer::HandleInfer(
    RequestStatus* request_status,
    const std::shared_ptr<InferBackendHandle>& backend,
    std::shared_ptr<InferRequestProvider> request_provider,
    std::shared_ptr<InferResponseProvider> response_provider,
    std::shared_ptr<ModelInferStats> infer_stats,
    std::function<void()> OnCompleteInferRPC)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    RequestStatusFactory::Create(
        request_status, 0, id_, RequestStatusCode::UNAVAILABLE,
        "Server not ready");
    OnCompleteInferRPC();
    return;
  }

  std::shared_ptr<ScopedAtomicIncrement> inflight(
      new ScopedAtomicIncrement(inflight_request_counter_));
  const uint64_t request_id = NextRequestId();

  // Need to capture 'backend' to keep it alive... it goes away when
  // it goes out of scope which can cause the model to be unloaded,
  // and we don't want that to happen when a request is in flight.
  auto OnCompleteHandleInfer = [this, OnCompleteInferRPC, backend,
                                response_provider, request_status, request_id,
                                infer_stats, inflight](Status status) mutable {
    if (status.IsOk()) {
      status =
          response_provider->FinalizeResponse(*backend->GetInferenceBackend());
      if (status.IsOk()) {
        RequestStatusFactory::Create(request_status, request_id, id_, status);
        OnCompleteInferRPC();
        return;
      }
    }

    // Report only stats that are relevant for a failed inference run.
    infer_stats->SetFailed(true);
    LOG_VERBOSE(1) << "Infer failed: " << status.Message();
    RequestStatusFactory::Create(request_status, request_id, id_, status);
    OnCompleteInferRPC();
  };

  // Need to set 'this' in each backend even though it is redundant after
  // the first time. Once we remove TFS dependency we can construct each backend
  // in a way that makes it directly aware of the inference server
  backend->GetInferenceBackend()->SetInferenceServer(this);
  backend->GetInferenceBackend()->Run(
      infer_stats, request_provider, response_provider, OnCompleteHandleInfer);
}

void
InferenceServer::HandleStatus(
    RequestStatus* request_status, ServerStatus* server_status,
    const std::string& model_name)
{
  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    RequestStatusFactory::Create(
        request_status, 0, id_, RequestStatusCode::UNAVAILABLE,
        "Server exiting");
    return;
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);
  const uint64_t request_id = NextRequestId();

  tfs::ServableStateMonitor* monitor = nullptr;
  if (core_ != nullptr) {
    monitor = core_->servable_state_monitor();
  }

  // If no specific model request just return the entire status
  // object.
  if (model_name.empty()) {
    RequestStatusFactory::Create(
        request_status, request_id, id_,
        status_manager_->Get(
            server_status, id_, ready_state_, UptimeNs(), monitor));
  } else {
    RequestStatusFactory::Create(
        request_status, request_id, id_,
        status_manager_->Get(
            server_status, id_, ready_state_, UptimeNs(), model_name, monitor));
  }
}

uint64_t
InferenceServer::UptimeNs() const
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);

  uint64_t now_ns = now.tv_sec * NANOS_PER_SECOND + now.tv_nsec;
  return now_ns - start_time_ns_;
}

//
// InferBackendHandle
//
class InferBackendHandleImpl : public InferenceServer::InferBackendHandle {
 public:
  InferBackendHandleImpl() : is_(nullptr) {}
  Status Init(
      const std::string& model_name, const int64_t model_version,
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

Status
InferBackendHandleImpl::Init(
    const std::string& model_name, const int64_t model_version,
    tfs::ServerCore* core)
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
  Status status =
      ModelRepositoryManager::GetModelPlatform(model_name, &platform);
  if (status.IsOk()) {
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

  if (is_ == nullptr) {
    status = Status(
        RequestStatusCode::UNAVAILABLE,
        "Inference request for unknown model '" + model_name + "'");
  }

  return status;
}

Status
InferenceServer::InferBackendHandle::Create(
    const InferenceServer* server, const std::string& model_name,
    const int64_t model_version, std::shared_ptr<InferBackendHandle>* handle)
{
  InferBackendHandleImpl* bh = new InferBackendHandleImpl();
  Status status = bh->Init(model_name, model_version, server->core_.get());
  if (status.IsOk()) {
    handle->reset(bh);
  }

  return status;
}

}}  // namespace nvidia::inferenceserver
