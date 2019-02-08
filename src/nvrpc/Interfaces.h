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
//
#pragma once

#include <grpc++/grpc++.h>

#include "src/nvrpc/Resources.h"

namespace nvrpc {

class IContext;
class IExecutor;
class IContextLifeCycle;
class IRPC;
class IService;


/**
 * The IContext object and it's subsequent derivations are the single more
 * important class in this library. Contexts are responsible for maintaining the
 * state of a message and performing the custom code for an RPC invocation.
 */
class IContext {
 public:
  virtual ~IContext() {}
  static IContext* Detag(void* tag) { return static_cast<IContext*>(tag); }

 protected:
  IContext() : m_MasterContext(this) {}
  IContext(IContext* master) : m_MasterContext(master) {}
  void* Tag() { return reinterpret_cast<void*>(this); }

  IContext* m_MasterContext;

 private:
  virtual bool RunNextState(bool) = 0;
  virtual void Reset() = 0;

  friend class IRPC;
  friend class IExecutor;
};

class IContextLifeCycle : public IContext {
 public:
  ~IContextLifeCycle() override {}

 protected:
  IContextLifeCycle() = default;

  virtual void OnLifeCycleStart() = 0;
  virtual void OnLifeCycleReset() = 0;

  // Function to get the context which the current RPC is related to
  virtual uintptr_t GetExecutionContext() = 0;

  // Callback function to be called at the end of RPC execution
  virtual void CompleteExecution(uintptr_t execution_context) = 0;

  virtual void FinishResponse() = 0;
  virtual void CancelResponse() = 0;
};

class IService {
 public:
  IService() = default;
  virtual ~IService() {}

  virtual void Initialize(::grpc::ServerBuilder&) = 0;
};

class IRPC {
 public:
  IRPC() = default;
  virtual ~IRPC() {}

 protected:
  virtual std::unique_ptr<IContext> CreateContext(
      ::grpc::ServerCompletionQueue*, std::shared_ptr<Resources>) = 0;

  friend class IExecutor;
};

class IExecutor {
 public:
  IExecutor() = default;
  virtual ~IExecutor() {}

  virtual void Initialize(::grpc::ServerBuilder&) = 0;
  virtual void Run() = 0;
  virtual void Shutdown() = 0;
  virtual void RegisterContexts(
      IRPC* rpc, std::shared_ptr<Resources> resources,
      int numContextsPerThread) = 0;
  // virtual void Shutdown() = 0;

 protected:
  inline bool RunContext(IContext* ctx, bool ok)
  {
    return ctx->RunNextState(ok);
  }
  inline void ResetContext(IContext* ctx) { ctx->Reset(); }
  inline std::unique_ptr<IContext> CreateContext(
      IRPC* rpc, ::grpc::ServerCompletionQueue* cq,
      std::shared_ptr<Resources> res)
  {
    return rpc->CreateContext(cq, res);
  }
};

}  // end namespace nvrpc
