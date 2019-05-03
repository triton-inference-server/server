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
#include "src/core/backend.h"
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
#include "tensorflow/core/platform/env.h"

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

  // Create the global manager for the repository. For now, all models are
  // eagerly loaded below when the manager is created.
  status = ModelRepositoryManager::Create(
      version_, status_manager_, model_store_path_, strict_model_config_,
      tf_gpu_memory_fraction_, tf_soft_placement_enabled_,
      repository_poll_secs_, true /* polling */, &model_repository_manager_);
  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
    if (model_repository_manager_ == nullptr) {
      ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    } else {
      // If error is returned while the manager is set, we assume the failure
      // is due to a model not loading correctly so we just continue
      // if not exiting on error.
      ready_state_ = ServerReadyState::SERVER_READY;
    }
    return false;
  }

  ready_state_ = ServerReadyState::SERVER_READY;
  return true;
}

bool
InferenceServer::Stop()
{
  ready_state_ = ServerReadyState::SERVER_EXITING;

  if (model_repository_manager_ == nullptr) {
    LOG_INFO << "No server context available. Exiting immediately.";
    return true;
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
          LOG_VERBOSE(1) << m.first << "v" << v.first << ": " << v.second;
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
    RETURN_IF_ERROR(model_repository_manager_->PollAndUpdate());
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
      ServerStatus server_status;
      Status status = status_manager_->Get(
          &server_status, id_, ready_state_, UptimeNs(),
          model_repository_manager_.get());

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

  // If no specific model request just return the entire status
  // object.
  if (model_name.empty()) {
    RequestStatusFactory::Create(
        request_status, request_id, id_,
        status_manager_->Get(
            server_status, id_, ready_state_, UptimeNs(),
            model_repository_manager_.get()));
  } else {
    RequestStatusFactory::Create(
        request_status, request_id, id_,
        status_manager_->Get(
            server_status, id_, ready_state_, UptimeNs(), model_name,
            model_repository_manager_.get()));
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
  InferBackendHandleImpl() = default;
  Status Init(
      const std::string& model_name, const int64_t model_version,
      ModelRepositoryManager* model_repository_manager);

  InferenceBackend* GetInferenceBackend() override
  {
    return backend_handle_->GetInferenceBackend();
  }

 private:
  std::shared_ptr<ModelRepositoryManager::BackendHandle> backend_handle_;
};

Status
InferBackendHandleImpl::Init(
    const std::string& model_name, const int64_t model_version,
    ModelRepositoryManager* model_repository_manager)
{
  return model_repository_manager->GetBackendHandle(
      model_name, model_version, &backend_handle_);
}

Status
InferenceServer::InferBackendHandle::Create(
    const InferenceServer* server, const std::string& model_name,
    const int64_t model_version, std::shared_ptr<InferBackendHandle>* handle)
{
  InferBackendHandleImpl* bh = new InferBackendHandleImpl();
  Status status = bh->Init(model_name, model_version, server->ModelManager());
  if (status.IsOk()) {
    handle->reset(bh);
  }

  return status;
}

}}  // namespace nvidia::inferenceserver
