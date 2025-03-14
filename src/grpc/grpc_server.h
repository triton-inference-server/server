// Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <grpc++/grpc++.h>

#include <vector>

#include "../common.h"
#include "../restricted_features.h"
#include "../shared_memory_manager.h"
#include "../tracer.h"
#include "grpc_handler.h"
#include "grpc_service.grpc.pb.h"
#include "grpc_utils.h"
#include "health.grpc.pb.h"
#include "infer_handler.h"
#include "stream_infer_handler.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace server { namespace grpc {

// GRPC uses HTTP2 which requires header to be in lowercase, so the Triton
// specific header that may be set for GRPC is defined to be all lowercases
constexpr char kRestrictedProtocolHeaderTemplate[] = "triton-grpc-protocol-";

struct SocketOptions {
  std::string address_{"0.0.0.0"};
  int32_t port_{8001};
  bool reuse_port_{false};
};

struct SslOptions {
  // Whether SSL is used for communication
  bool use_ssl_{false};
  // File holding PEM-encoded server certificate
  std::string server_cert_{""};
  // File holding PEM-encoded server key
  std::string server_key_{""};
  // File holding PEM-encoded root certificate
  std::string root_cert_{""};
  // Whether to use Mutual Authentication
  bool use_mutual_auth_{false};
};

// GRPC KeepAlive: https://grpc.github.io/grpc/cpp/md_doc_keepalive.html
// https://grpc.io/docs/guides/keepalive/
struct KeepAliveOptions {
  int keepalive_time_ms_{7200000};
  int keepalive_timeout_ms_{20000};
  bool keepalive_permit_without_calls_{false};
  int http2_max_pings_without_data_{2};
  int http2_min_recv_ping_interval_without_data_ms_{300000};
  int http2_max_ping_strikes_{2};
  int max_connection_age_ms_{0};
  int max_connection_age_grace_ms_{0};
};

struct Options {
  SocketOptions socket_;
  SslOptions ssl_;
  KeepAliveOptions keep_alive_;
  grpc_compression_level infer_compression_level_{GRPC_COMPRESS_LEVEL_NONE};
  // The number of gRPC inference handler threads. Useful for
  // throughput tuning of models that are request handling bounded.
  int infer_thread_count_{2};
  // The maximum number of inference request/response objects that
  // remain allocated for reuse. As long as the number of in-flight
  // requests doesn't exceed this value there will be no
  // allocation/deallocation of request/response objects.
  int infer_allocation_pool_size_{8};
  int max_response_pool_size_{INT_MAX};
  RestrictedFeatures restricted_protocols_;
  std::string forward_header_pattern_;
};

class Server {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const Options& server_options, std::unique_ptr<Server>* server);

  static TRITONSERVER_Error* Create(
      std::shared_ptr<TRITONSERVER_Server>& server, UnorderedMapType& options,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const RestrictedFeatures& restricted_features,
      std::unique_ptr<Server>* service);

  ~Server();

  TRITONSERVER_Error* Start();
  TRITONSERVER_Error* Stop();

 private:
  Server(
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const Options& server_options);

  static TRITONSERVER_Error* GetSocketOptions(
      SocketOptions& options, UnorderedMapType& options_map);
  static TRITONSERVER_Error* GetSslOptions(
      SslOptions& options, UnorderedMapType& options_map);
  static TRITONSERVER_Error* GetKeepAliveOptions(
      KeepAliveOptions& options, UnorderedMapType& options_map);

  static TRITONSERVER_Error* GetOptions(
      Options& options, UnorderedMapType& options_map);

  std::shared_ptr<TRITONSERVER_Server> tritonserver_;
  TraceManager* trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  const std::string server_addr_;

  ::grpc::ServerBuilder builder_;

  inference::GRPCInferenceService::AsyncService service_;
  ::grpc::health::v1::Health::AsyncService health_service_;

  std::unique_ptr<::grpc::Server> server_;

  std::unique_ptr<::grpc::ServerCompletionQueue> common_cq_;
  std::unique_ptr<::grpc::ServerCompletionQueue> model_infer_cq_;
  std::unique_ptr<::grpc::ServerCompletionQueue> model_stream_infer_cq_;

  std::unique_ptr<HandlerBase> common_handler_;
  std::vector<std::unique_ptr<HandlerBase>> model_infer_handlers_;
  std::vector<std::unique_ptr<HandlerBase>> model_stream_infer_handlers_;

  int bound_port_{0};
  bool running_{false};
};

}}}  // namespace triton::server::grpc
