// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "src/core/grpc_service.grpc.pb.h"
#include "src/core/tritonserver.h"
#include "src/servers/shared_memory_manager.h"
#include "src/servers/tracer.h"

namespace nvidia { namespace inferenceserver {

struct SslOptions {
  explicit SslOptions() {}
  // File holding PEM-encoded server certificate
  std::string server_cert;
  // File holding PEM-encoded server key
  std::string server_key;
  // File holding PEM-encoded root certificate
  std::string root_cert;
};

class GRPCServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      nvidia::inferenceserver::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager, int32_t port,
      bool use_ssl, const SslOptions& ssl_options,
      int infer_allocation_pool_size, std::unique_ptr<GRPCServer>* grpc_server);

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
      nvidia::inferenceserver::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const std::string& server_addr, bool use_ssl,
      const SslOptions& ssl_options, const int infer_allocation_pool_size);

  std::shared_ptr<TRITONSERVER_Server> server_;
  TraceManager* trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  const std::string server_addr_;
  const bool use_ssl_;
  const SslOptions ssl_options_;

  const int infer_allocation_pool_size_;

  std::unique_ptr<grpc::ServerCompletionQueue> common_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> model_infer_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> model_stream_infer_cq_;

  grpc::ServerBuilder grpc_builder_;
  std::unique_ptr<grpc::Server> grpc_server_;

  std::unique_ptr<HandlerBase> common_handler_;
  std::unique_ptr<HandlerBase> model_infer_handler_;
  std::unique_ptr<HandlerBase> model_stream_infer_handler_;

  GRPCInferenceService::AsyncService service_;
  bool running_;
};

}}  // namespace nvidia::inferenceserver
