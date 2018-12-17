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
///
#pragma once

#include <chrono>

#include "src/nvrpc/Service.h"

namespace nvrpc {

using std::chrono::milliseconds;

class Server {
 public:
  Server(std::string server_address);

  Server() : Server("0.0.0.0:50051") {}

  template <class ServiceType>
  AsyncService<typename ServiceType::AsyncService>* RegisterAsyncService();

  IExecutor* RegisterExecutor(IExecutor* executor)
  {
    m_Executors.emplace_back(executor);
    executor->Initialize(m_Builder);
    return executor;
  }

  void Run();
  void Run(milliseconds timeout, std::function<void()> control_fn);
  void AsyncRun();

  void Shutdown();

  ::grpc::ServerBuilder& GetBuilder();

 private:
  bool m_Running;
  std::string m_ServerAddress;
  ::grpc::ServerBuilder m_Builder;
  std::vector<std::unique_ptr<IService>> m_Services;
  std::vector<std::unique_ptr<IExecutor>> m_Executors;
  std::unique_ptr<::grpc::Server> m_Server;
};


template <class ServiceType>
AsyncService<typename ServiceType::AsyncService>*
Server::RegisterAsyncService()
{
  if (m_Running) {
    throw std::runtime_error(
        "Error: cannot register service on a running server");
  }
  auto service = new AsyncService<typename ServiceType::AsyncService>;
  auto base = static_cast<IService*>(service);
  m_Services.emplace_back(base);
  service->Initialize(m_Builder);
  return service;
}

}  // end namespace nvrpc
