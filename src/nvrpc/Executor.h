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
#pragma once

#include "src/nvrpc/Interfaces.h"
#include "src/nvrpc/Resources.h"
#include "src/nvrpc/ThreadPool.h"

#include <thread>

namespace nvrpc {

class Executor : public IExecutor {
 public:
  Executor();
  Executor(int numThreads);
  Executor(std::unique_ptr<ThreadPool> threadpool);
  ~Executor() override {}

  void Initialize(::grpc::ServerBuilder& builder) final override
  {
    for (int i = 0; i < m_ThreadPool->Size(); i++) {
      m_ServerCompletionQueues.emplace_back(builder.AddCompletionQueue());
    }
  }

  void RegisterContexts(
      IRPC* rpc, std::shared_ptr<Resources> resources,
      int numContextsPerThread) final override
  {
    // CHECK_EQ(m_ThreadPool->Size(), m_ServerCompletionQueues.size()) <<
    // "Incorrect number of CQs";
    for (int i = 0; i < m_ThreadPool->Size(); i++) {
      auto cq = m_ServerCompletionQueues[i].get();
      for (int j = 0; j < numContextsPerThread; j++) {
        m_Contexts.emplace_back(this->CreateContext(rpc, cq, resources));
      }
    }
  }

  void Run() final override
  {
    m_Running = true;
    // Launch the threads polling on their CQs
    for (int i = 0; i < m_ThreadPool->Size(); i++) {
      m_ThreadPool->enqueue([this, i] { ProgressEngine(i); });
    }
    // Queue the Execution Contexts in the recieve queue
    for (size_t i = 0; i < m_Contexts.size(); i++) {
      // Reseting the context decrements the gauge
      ResetContext(m_Contexts[i].get());
    }
  }

  void Shutdown() final override
  {
    m_Running = false;
    for (auto& cq : m_ServerCompletionQueues) {
      cq->Shutdown();
    }
  }

 private:
  void ProgressEngine(int thread_id);

  volatile bool m_Running;
  std::vector<std::unique_ptr<IContext>> m_Contexts;
  std::vector<std::unique_ptr<::grpc::ServerCompletionQueue>>
      m_ServerCompletionQueues;
  std::unique_ptr<ThreadPool> m_ThreadPool;
};

}  // end namespace nvrpc
