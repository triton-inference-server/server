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
//
#include "src/nvrpc/Server.h"

#include <thread>

namespace nvrpc {

Server::Server(std::string server_address)
    : m_Running(false), m_ServerAddress(server_address)
{
  m_Builder.AddListeningPort(
      m_ServerAddress, ::grpc::InsecureServerCredentials());
}

::grpc::ServerBuilder&
Server::GetBuilder()
{
  if (m_Running) {
    throw std::runtime_error(
        "Unable to access Builder after the Server is running.");
  }
  return m_Builder;
}

void
Server::Run()
{
  Run(std::chrono::milliseconds(5000), [] {});
}

void
Server::Run(std::chrono::milliseconds timeout, std::function<void()> control_fn)
{
  AsyncRun();
  for (;;) {
    control_fn();
    std::this_thread::sleep_for(timeout);
  }
  // TODO: gracefully shutdown each service and join threads
}

void
Server::AsyncRun()
{
  m_Running = true;
  m_Server = m_Builder.BuildAndStart();
  for (size_t i = 0; i < m_Executors.size(); i++) {
    m_Executors[i]->Run();
  }
}

void
Server::Shutdown()
{
  m_Server->Shutdown();
  for (size_t i = 0; i < m_Executors.size(); i++) {
    m_Executors[i]->Shutdown();
  }

  // This should cause enforce a join on all async Executor threads
  m_Executors.clear();
}

}  // end namespace nvrpc
