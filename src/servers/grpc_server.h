// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/trtserver.h"
#include "src/nvrpc/Interfaces.h"
#include "src/nvrpc/Server.h"

namespace nvidia { namespace inferenceserver {

class InferenceServer;
class RequestStatus;
class ServerStatus;

class GRPCServer : private nvrpc::Server {
 public:
  static TRTSERVER_Error* Create(
      InferenceServer* server, int32_t port, int infer_thread_cnt,
      int stream_infer_thread_cnt, std::unique_ptr<GRPCServer>* grpc_servers);
  TRTSERVER_Error* Start();
  TRTSERVER_Error* Stop();

  ~GRPCServer();

 private:
  GRPCServer(
      const std::string& addr, const int infer_thread_cnt,
      const int stream_infer_thread_cnt);

  nvrpc::IRPC* rpcInfer_;
  nvrpc::IRPC* rpcStreamInfer_;
  nvrpc::IRPC* rpcStatus_;
  nvrpc::IRPC* rpcProfile_;
  nvrpc::IRPC* rpcHealth_;
  int infer_thread_cnt_;
  int stream_infer_thread_cnt_;
  bool running_;
};

}}  // namespace nvidia::inferenceserver
