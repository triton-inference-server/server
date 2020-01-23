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
#include "src/core/grpc_service_v2.grpc.pb.h"
#include "src/core/trtserver.h"
#include "src/servers/shared_memory_block_manager.h"
#include "src/servers/tracer.h"

namespace nvidia { namespace inferenceserver {

class GRPCServerV2 {
 public:
  static TRTSERVER_Error* Create(
      const std::shared_ptr<TRTSERVER_Server>& server,
      const std::shared_ptr<nvidia::inferenceserver::TraceManager>&
          trace_manager,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      int32_t port, int infer_thread_cnt, int stream_infer_thread_cnt,
      int infer_allocation_pool_size,
      std::unique_ptr<GRPCServerV2>* grpc_server);

  ~GRPCServerV2();

  TRTSERVER_Error* Start();
  TRTSERVER_Error* Stop();

 public:
  class HandlerBase {
   public:
    virtual ~HandlerBase() = default;
  };

 private:
  GRPCServerV2(
      const std::shared_ptr<TRTSERVER_Server>& server,
      const std::shared_ptr<nvidia::inferenceserver::TraceManager>&
          trace_manager,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      const char* server_id, const std::string& server_addr,
      const int infer_thread_cnt, const int stream_infer_thread_cnt,
      const int infer_allocation_pool_size);

  std::shared_ptr<TRTSERVER_Server> server_;
  std::shared_ptr<TraceManager> trace_manager_;
  std::shared_ptr<SharedMemoryBlockManager> smb_manager_;
  const char* server_id_;
  const std::string server_addr_;

  const int infer_thread_cnt_;
  const int stream_infer_thread_cnt_;
  const int infer_allocation_pool_size_;

  std::unique_ptr<grpc::ServerCompletionQueue> server_live_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> server_ready_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> model_ready_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> server_metadata_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> model_metadata_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> repository_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> infer_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> stream_infer_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> modelcontrol_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> shmcontrol_cq_;

  grpc::ServerBuilder grpc_builder_;
  std::unique_ptr<grpc::Server> grpc_server_;

  std::unique_ptr<HandlerBase> server_live_handler_;
  std::unique_ptr<HandlerBase> server_ready_handler_;
  std::unique_ptr<HandlerBase> model_ready_handler_;
  std::unique_ptr<HandlerBase> server_metadata_handler_;
  std::unique_ptr<HandlerBase> model_metadata_handler_;
  std::unique_ptr<HandlerBase> repository_handler_;
  std::unique_ptr<HandlerBase> infer_handler_;
  std::unique_ptr<HandlerBase> stream_infer_handler_;
  std::unique_ptr<HandlerBase> modelcontrol_handler_;
  std::unique_ptr<HandlerBase> shmcontrol_handler_;

  GRPCInferenceService::AsyncService service_;
  bool running_;
};

}}  // namespace nvidia::inferenceserver
