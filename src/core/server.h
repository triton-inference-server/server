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
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <atomic>
#include <string>
#include <thread>
#include <unordered_map>

#include "src/core/api.pb.h"
#include "src/core/model_config.pb.h"
#include "src/core/provider.h"
#include "src/core/request_status.pb.h"
#include "src/core/server_status.h"
#include "src/core/server_status.pb.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferenceBackend;

// Inference server information.
class InferenceServer {
 public:
  // Construct an inference server.
  InferenceServer();

  // Initialize the server. Return true on success, false otherwise.
  bool Init();

  // Stop the server.  Return true if all models are unloaded, false
  // if exit timeout occurs.
  bool Stop();

  // Check the model repository for changes and update server state
  // based on those changes.
  Status PollModelRepository();

  // Run health check indicated by 'mode'
  void HandleHealth(
      RequestStatus* request_status, bool* health, const std::string& mode);

  // Run profile 'cmd' for profiling all the all GPU devices
  void HandleProfile(RequestStatus* request_status, const std::string& cmd);

  // Perform inference on the given input for specified model and
  // update RequestStatus object with the status of the inference.
  void HandleInfer(
      RequestStatus* request_status,
      const std::shared_ptr<InferenceBackend>& backend,
      std::shared_ptr<InferRequestProvider> request_provider,
      std::shared_ptr<InferResponseProvider> response_provider,
      std::shared_ptr<ModelInferStats> infer_stats,
      std::function<void()> OnCompleteInferRPC);

  // Update the RequestStatus object and ServerStatus object with the
  // status of the model. If 'model_name' is empty, update with the
  // status of all models.
  void HandleStatus(
      RequestStatus* request_status, ServerStatus* server_status,
      const std::string& model_name);

  // Return the ready state for the server.
  ServerReadyState ReadyState() const { return ready_state_; }

  // Return the server version.
  const std::string& Version() const { return version_; }

  // Get / set the ID of the server.
  const std::string& Id() const { return id_; }
  void SetId(const std::string& id) { id_ = id; }

  // Get / set the model repository path
  const std::string& ModelStorePath() const { return model_store_path_; }
  void SetModelStorePath(const std::string& p) { model_store_path_ = p; }

  // Get / set strict model configuration enable.
  bool StrictModelConfigEnabled() const { return strict_model_config_; }
  void SetStrictModelConfigEnabled(bool e) { strict_model_config_ = e; }

  // Get / set strict readiness enable.
  bool StrictReadinessEnabled() const { return strict_readiness_; }
  void SetStrictReadinessEnabled(bool e) { strict_readiness_ = e; }

  // Get / set profiling enable.
  bool ProfilingEnabled() const { return profiling_enabled_; }
  void SetProfilingEnabled(bool e) { profiling_enabled_ = e; }

  // Get / set the server exit timeout, in seconds.
  int32_t ExitTimeoutSeconds() const { return exit_timeout_secs_; }
  void SetExitTimeoutSeconds(int32_t s) { exit_timeout_secs_ = std::max(0, s); }

  // Get / set Tensorflow soft placement enable.
  bool TensorFlowSoftPlacementEnabled() const
  {
    return tf_soft_placement_enabled_;
  }
  void SetTensorFlowSoftPlacementEnabled(bool e)
  {
    tf_soft_placement_enabled_ = e;
  }

  // Get / set Tensorflow GPU memory fraction.
  float TensorFlowGPUMemoryFraction() const { return tf_gpu_memory_fraction_; }
  void SetTensorFlowGPUMemoryFraction(float f) { tf_gpu_memory_fraction_ = f; }

  // Return the status manager for this server.
  std::shared_ptr<ServerStatusManager> StatusManager() const
  {
    return status_manager_;
  }

  // Return the requested InferenceBackend object.
  Status GetInferenceBackend(
      const std::string& model_name, const int64_t model_version,
      std::shared_ptr<InferenceBackend>* backend)
  {
    return model_repository_manager_->GetInferenceBackend(
        model_name, model_version, backend);
  }

 private:
  // Return the uptime of the server in nanoseconds.
  uint64_t UptimeNs() const;

  // Return the next request ID for this server.
  uint64_t NextRequestId() { return next_request_id_++; }

  std::string version_;
  std::string id_;
  uint64_t start_time_ns_;

  std::string model_store_path_;
  bool strict_model_config_;
  bool strict_readiness_;
  bool profiling_enabled_;
  uint32_t exit_timeout_secs_;

  bool tf_soft_placement_enabled_;
  float tf_gpu_memory_fraction_;

  // Current state of the inference server.
  ServerReadyState ready_state_;

  // Each request is assigned a unique id.
  std::atomic<uint64_t> next_request_id_;

  // Number of in-flight requests. During shutdown we attempt to wait
  // for all in-flight requests to complete before exiting.
  std::atomic<uint64_t> inflight_request_counter_;

  std::shared_ptr<ServerStatusManager> status_manager_;
  std::unique_ptr<ModelRepositoryManager> model_repository_manager_;
};

}}  // namespace nvidia::inferenceserver
