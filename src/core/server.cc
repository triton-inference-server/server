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

#include "src/core/server.h"

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
#include "src/core/backend.h"
#include "src/core/constants.h"
#include "src/core/cuda_utils.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"
#include "src/core/model_repository_manager.h"
#include "src/core/pinned_memory_manager.h"
#include "src/core/provider.h"
#include "src/core/server.h"
#include "src/core/server_status.pb.h"

#ifdef TRTIS_ENABLE_GPU
#include "src/core/cuda_memory_manager.h"
#endif  // TRTIS_ENABLE_GPU

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

}  // namespace

//
// InferenceServer
//
InferenceServer::InferenceServer()
    : version_(TRTIS_VERSION), ready_state_(ServerReadyState::SERVER_INVALID)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  start_time_ns_ = TIMESPEC_TO_NANOS(ts);

  id_ = "inference:0";
  protocol_version_ = 1;
  extensions_.push_back("classification");
  extensions_.push_back("sequence");
  extensions_.push_back("model_repository");
  extensions_.push_back("model_configuration");
  extensions_.push_back("system_shared_memory");
  extensions_.push_back("cuda_shared_memory");
#ifdef TRTIS_ENABLE_STATS
  extensions_.push_back("statistics");
#endif

  strict_model_config_ = true;
  strict_readiness_ = true;
  exit_timeout_secs_ = 30;
  pinned_memory_pool_size_ = 1 << 28;
#ifdef TRTIS_ENABLE_GPU
  min_supported_compute_capability_ = TRTIS_MIN_COMPUTE_CAPABILITY;
#else
  min_supported_compute_capability_ = 0.0;
#endif  // TRTIS_ENABLE_GPU

  tf_soft_placement_enabled_ = true;
  tf_gpu_memory_fraction_ = 0.0;
  tf_vgpu_memory_limits_ = {};

  inflight_request_counter_ = 0;

  status_manager_.reset(new ServerStatusManager(version_));
}

Status
InferenceServer::Init()
{
  Status status;

  ready_state_ = ServerReadyState::SERVER_INITIALIZING;

  LOG_INFO << "Initializing Triton Inference Server";

  if (model_repository_paths_.empty()) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return Status(
        RequestStatusCode::INVALID_ARG, "--model-repository must be specified");
  }

  PinnedMemoryManager::Options options(pinned_memory_pool_size_);
  status = PinnedMemoryManager::Create(options);
  if (!status.IsOk()) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return status;
  }

#ifdef TRTIS_ENABLE_GPU
  // Defer the setting of default CUDA memory pool value here as
  // 'min_supported_compute_capability_' is finalized
  std::set<int> supported_gpus;
  if (GetSupportedGPUs(&supported_gpus, min_supported_compute_capability_)
          .IsOk()) {
    for (const auto gpu : supported_gpus) {
      if (cuda_memory_pool_size_.find(gpu) == cuda_memory_pool_size_.end()) {
        cuda_memory_pool_size_[gpu] = 1 << 26;
      }
    }
  }
  CudaMemoryManager::Options cuda_options(
      min_supported_compute_capability_, cuda_memory_pool_size_);
  status = CudaMemoryManager::Create(cuda_options);
  // If CUDA memory manager can't be created, just log error as the
  // server can still function properly
  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
  }
#endif  // TRTIS_ENABLE_GPU


  status = EnablePeerAccess(min_supported_compute_capability_);
  if (!status.IsOk()) {
    // failed to enable peer access is not critical, just inefficient.
    LOG_ERROR << status.Message();
  }

  // Create the model manager for the repository. Unless model control
  // is disabled, all models are eagerly loaded when the manager is created.
  bool polling_enabled = (model_control_mode_ == MODE_POLL);
  bool model_control_enabled = (model_control_mode_ == MODE_EXPLICIT);
  status = ModelRepositoryManager::Create(
      this, version_, status_manager_, model_repository_paths_, startup_models_,
      strict_model_config_, tf_gpu_memory_fraction_, tf_soft_placement_enabled_,
      tf_vgpu_memory_limits_, polling_enabled, model_control_enabled,
      min_supported_compute_capability_, &model_repository_manager_);
  if (!status.IsOk()) {
    if (model_repository_manager_ == nullptr) {
      ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    } else {
      // If error is returned while the manager is set, we assume the
      // failure is due to a model not loading correctly so we just
      // continue if not exiting on error.
      ready_state_ = ServerReadyState::SERVER_READY;
    }
    return status;
  }

  ready_state_ = ServerReadyState::SERVER_READY;
  return Status::Success;
}

Status
InferenceServer::Stop()
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status::Success;
  }

  ready_state_ = ServerReadyState::SERVER_EXITING;

  if (model_repository_manager_ == nullptr) {
    LOG_INFO << "No server context available. Exiting immediately.";
    return Status::Success;
  } else {
    LOG_INFO << "Waiting for in-flight inferences to complete.";
  }

  Status status = model_repository_manager_->UnloadAllModels();
  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
  }

  // Wait for all in-flight requests to complete and all loaded models
  // to unload, or for the exit timeout to expire.
  uint32_t exit_timeout_iters = exit_timeout_secs_;

  while (true) {
    const auto& live_models = model_repository_manager_->GetLiveBackendStates();

    LOG_INFO << "Timeout " << exit_timeout_iters << ": Found "
             << live_models.size() << " live models and "
             << inflight_request_counter_ << " in-flight requests";
    if (LOG_VERBOSE_IS_ON(1)) {
      for (const auto& m : live_models) {
        for (const auto& v : m.second) {
          LOG_VERBOSE(1) << m.first << " v" << v.first << ": " << v.second;
        }
      }
    }

    if ((live_models.size() == 0) && (inflight_request_counter_ == 0)) {
      return Status::Success;
    }
    if (exit_timeout_iters <= 0) {
      break;
    }

    exit_timeout_iters--;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  return Status(
      RequestStatusCode::INTERNAL,
      "Exit timeout expired. Exiting immediately.");
}

Status
InferenceServer::PollModelRepository()
{
  LOG_VERBOSE(1) << "Polling model repository";

  // Look for changes and update the loaded model configurations
  // appropriately.
  if (ready_state_ == ServerReadyState::SERVER_READY) {
    RETURN_IF_ERROR(model_repository_manager_->PollAndUpdate());
  }

  return Status::Success;
}

Status
InferenceServer::IsLive(bool* live)
{
  *live = false;

  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    return Status(RequestStatusCode::UNAVAILABLE, "Server exiting");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  // Server is considered live if it can respond to this health
  // request and it was able to initialize.
  *live =
      ((ready_state_ != ServerReadyState::SERVER_INVALID) &&
       (ready_state_ != ServerReadyState::SERVER_INITIALIZING) &&
       (ready_state_ != ServerReadyState::SERVER_FAILED_TO_INITIALIZE));
  return Status::Success;
}

Status
InferenceServer::IsReady(bool* ready)
{
  *ready = false;

  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    return Status(RequestStatusCode::UNAVAILABLE, "Server exiting");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  // Server is considered ready if it is in the ready state.
  // Additionally can report ready only when all models are ready.
  *ready = (ready_state_ == ServerReadyState::SERVER_READY);
  if (*ready && strict_readiness_) {
    // Strict readiness... get the model status and make sure all
    // models are ready.
    ServerStatus server_status;
    Status status =
        status_manager_->Get(&server_status, id_, ready_state_, UptimeNs());

    *ready = status.IsOk();
    if (*ready) {
      for (const auto& ms : server_status.model_status()) {
        // If a model status is present but no version status,
        // the model is not ready as there is no proper version to be served
        if (ms.second.version_status().size() == 0) {
          *ready = false;
          goto strict_done;
        }
        for (const auto& vs : ms.second.version_status()) {
          if (vs.second.ready_state() != ModelReadyState::MODEL_READY) {
            *ready = false;
            goto strict_done;
          }
        }
      }
    strict_done:;
    }
  }

  return Status::Success;
}

void
InferenceServer::InferAsync(
    const std::shared_ptr<InferenceBackend>& backend,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    const std::shared_ptr<ModelInferStats>& infer_stats,
    std::function<void(const Status&)> OnCompleteInfer)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    OnCompleteInfer(Status(RequestStatusCode::UNAVAILABLE, "Server not ready"));
    return;
  }

  std::shared_ptr<ScopedAtomicIncrement> inflight(
      new ScopedAtomicIncrement(inflight_request_counter_));

  // Need to capture 'backend' to keep it alive... it goes away when
  // it goes out of scope which can cause the model to be unloaded,
  // and we don't want that to happen when a request is in flight.
  auto OnCompleteHandleInfer = [this, OnCompleteInfer, backend,
                                response_provider,
                                inflight](const Status& status) mutable {
    if (status.IsOk()) {
      OnCompleteInfer(response_provider->FinalizeResponse(*backend));
    } else {
      OnCompleteInfer(status);
    }
  };

  backend->Run(
      infer_stats, request_provider, response_provider, OnCompleteHandleInfer);
}

Status
InferenceServer::GetStatus(
    ServerStatus* server_status, const std::string& model_name)
{
  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    return Status(RequestStatusCode::UNAVAILABLE, "Server exiting");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  // If no specific model request just return the entire status
  // object.
  if (model_name.empty()) {
    return status_manager_->Get(server_status, id_, ready_state_, UptimeNs());
  } else {
    return status_manager_->Get(
        server_status, id_, ready_state_, UptimeNs(), model_name);
  }

  return Status::Success;
}

Status
InferenceServer::GetModelRepositoryIndex(ModelRepositoryIndex* repository_index)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(RequestStatusCode::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  return model_repository_manager_->GetModelRepositoryIndex(repository_index);
}

Status
InferenceServer::LoadModel(const std::string& model_name)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(RequestStatusCode::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  auto action_type = ModelRepositoryManager::ActionType::LOAD;
  return model_repository_manager_->LoadUnloadModel(model_name, action_type);
}

Status
InferenceServer::UnloadModel(const std::string& model_name)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(RequestStatusCode::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  auto action_type = ModelRepositoryManager::ActionType::UNLOAD;
  return model_repository_manager_->LoadUnloadModel(model_name, action_type);
}

uint64_t
InferenceServer::UptimeNs() const
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);

  uint64_t now_ns = TIMESPEC_TO_NANOS(now);
  return now_ns - start_time_ns_;
}

}}  // namespace nvidia::inferenceserver
