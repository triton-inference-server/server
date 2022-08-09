// Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "grpc_service.grpc.pb.h"
#include "shared_memory_manager.h"
#include "tracer.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace server {

struct SslOptions {
  explicit SslOptions() {}
  // File holding PEM-encoded server certificate
  std::string server_cert;
  // File holding PEM-encoded server key
  std::string server_key;
  // File holding PEM-encoded root certificate
  std::string root_cert;
  // Whether to use Mutual Authentication
  bool use_mutual_auth;
};

// GRPC KeepAlive: https://grpc.github.io/grpc/cpp/md_doc_keepalive.html
struct KeepAliveOptions {
  explicit KeepAliveOptions()
      : keepalive_time_ms(7200000), keepalive_timeout_ms(20000),
        keepalive_permit_without_calls(false), http2_max_pings_without_data(2),
        http2_min_recv_ping_interval_without_data_ms(300000),
        http2_max_ping_strikes(2)
  {
  }
  int keepalive_time_ms;
  int keepalive_timeout_ms;
  bool keepalive_permit_without_calls;
  int http2_max_pings_without_data;
  int http2_min_recv_ping_interval_without_data_ms;
  int http2_max_ping_strikes;
};

class GRPCServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager, int32_t port,
      const bool reuse_port, std::string address, bool use_ssl,
      const SslOptions& ssl_options, int infer_allocation_pool_size,
      grpc_compression_level compression_level,
      const KeepAliveOptions& keepalive_options,
      std::unique_ptr<GRPCServer>* grpc_server);

  ~GRPCServer();

  TRITONSERVER_Error* Start();
  TRITONSERVER_Error* Stop();

 public:
  class HandlerBase {
   public:
    virtual ~HandlerBase() = default;
  };

  class ICallData {
   public:
    virtual ~ICallData() = default;
    virtual bool Process(bool ok) = 0;
    virtual std::string Name() = 0;
    virtual uint64_t Id() = 0;
  };

 private:
  GRPCServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const std::string& server_addr, const bool reuse_port, bool use_ssl,
      const SslOptions& ssl_options, const int infer_allocation_pool_size,
      grpc_compression_level compression_level,
      const KeepAliveOptions& keepalive_options);

  std::shared_ptr<TRITONSERVER_Server> server_;
  TraceManager* trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  const std::string server_addr_;
  const bool reuse_port_;
  const bool use_ssl_;
  const SslOptions ssl_options_;

  const int infer_allocation_pool_size_;
  grpc_compression_level compression_level_;

  const KeepAliveOptions keepalive_options_;

  std::unique_ptr<grpc::ServerCompletionQueue> common_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> model_infer_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> model_stream_infer_cq_;

  grpc::ServerBuilder grpc_builder_;
  std::unique_ptr<grpc::Server> grpc_server_;

  std::unique_ptr<HandlerBase> common_handler_;
  std::vector<std::unique_ptr<HandlerBase>> model_infer_handlers_;
  std::vector<std::unique_ptr<HandlerBase>> model_stream_infer_handlers_;

  inference::GRPCInferenceService::AsyncService service_;
  bool running_;
};

}}  // namespace triton::server
