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
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace tfs = tensorflow::serving;

namespace nvidia { namespace inferenceserver {

class CustomBundle;
class GraphDefBundle;
class NetDefBundle;
class PlanBundle;
class SavedModelBundle;
class GRPCServer;
class HTTPServer;

// Inference server information.
class InferenceServer {
 public:
  class InferBackendHandle;

  // Construct an inference server.
  InferenceServer();

  // Initialize the server.
  // Return true on success, false otherwise.
  bool Init(int argc, char** argv);

  // Close the server.
  // Return true if all models are unloaded, false if exit timeout occurs.
  bool Close();

  // Wait for server. Does not return until server is shutdown.
  void Wait();

  // Run health check indicated by 'mode'
  void HandleHealth(
      RequestStatus* request_status, bool* health, const std::string& mode);

  // Run profile 'cmd' for profiling all the all GPU devices
  void HandleProfile(RequestStatus* request_status, const std::string& cmd);

  // Perform inference on the given input for specified model and
  // update RequestStatus object with the status of the inference.
  void HandleInfer(
      RequestStatus* request_status,
      const std::shared_ptr<InferBackendHandle>& backend,
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

  // Return the server version.
  const std::string& Version() const { return version_; }

  // Return the ID of the server.
  const std::string& Id() const { return id_; }

  // Return the ready state for the server.
  ServerReadyState ReadyState() const { return ready_state_; }

  // Return the HTTP port of the server, or -1 if HTTP is not enabled.
  int HttpPort() const { return http_port_; }

  // Return the gRPC port of the server, or -1 if gRPC is not enabled.
  int GrpcPort() const { return grpc_port_; }

  // Return the metrics port of the server, or -1 if metrics are not
  // enabled.
  int MetricsPort() const { return metrics_port_; }

  // Return the status manager for this server.
  std::shared_ptr<ServerStatusManager> StatusManager() const
  {
    return status_manager_;
  }

  // A handle to a backend.
  class InferBackendHandle {
   public:
    InferBackendHandle() : is_(nullptr) {}
    tensorflow::Status Init(
        const std::string& model_name, const int64_t model_version,
        tfs::ServerCore* core);
    InferenceBackend* operator()() { return is_; }

   private:
    InferenceBackend* is_;
    tfs::ServableHandle<GraphDefBundle> graphdef_bundle_;
    tfs::ServableHandle<PlanBundle> plan_bundle_;
    tfs::ServableHandle<NetDefBundle> netdef_bundle_;
    tfs::ServableHandle<SavedModelBundle> saved_model_bundle_;
    tfs::ServableHandle<CustomBundle> custom_bundle_;
  };

  tensorflow::Status CreateBackendHandle(
      const std::string& model_name, const int64_t model_version,
      const std::shared_ptr<InferBackendHandle>& handle);

 private:
  // Start server running and listening on gRPC and/or HTTP endpoints.
  void Start();

  std::unique_ptr<GRPCServer> StartGrpcServer();
  std::unique_ptr<HTTPServer> StartHttpServer();
  tensorflow::Status ParseProtoTextFile(
      const std::string& file, google::protobuf::Message* message);
  tfs::PlatformConfigMap BuildPlatformConfigMap(
      float tf_gpu_memory_fraction, bool tf_allow_soft_placement);

  // Return the uptime of the server in nanoseconds.
  uint64_t UptimeNs() const;

  // Return the next request ID for this server.
  uint64_t NextRequestId() { return next_request_id_++; }

  // Helper function to perform repeated task during initialization.
  void LogInitError(const std::string& msg);

  std::string version_;
  std::string id_;

  // Use -1 for a port to indicate the corresponding service is
  // disabled
  int http_port_;
  int grpc_port_;
  int metrics_port_;

  std::string model_store_path_;
  int http_thread_cnt_;
  bool strict_model_config_;
  bool strict_readiness_;
  bool profiling_enabled_;
  bool poll_model_repository_enabled_;
  uint32_t repository_poll_secs_;
  uint32_t exit_timeout_secs_;
  uint64_t start_time_ns_;

  // Current state of the inference server.
  ServerReadyState ready_state_;

  // Each request is assigned a unique id.
  std::atomic<uint64_t> next_request_id_;

  // Number of in-flight requests. During shutdown we attempt to wait
  // for all in-flight requests to complete before exiting.
  std::atomic<uint64_t> inflight_request_counter_;

  std::unique_ptr<tfs::ServerCore> core_;
  std::shared_ptr<ServerStatusManager> status_manager_;

  std::unique_ptr<HTTPServer> http_server_;

  std::unique_ptr<GRPCServer> grpc_server_;
};

}}  // namespace nvidia::inferenceserver
