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

namespace nvrpc {

template <class Request, class Response>
class LifeCycleUnary : public IContextLifeCycle {
 public:
  using RequestType = Request;
  using ResponseType = Response;
  using ServiceQueueFuncType = std::function<void(
      ::grpc::ServerContext*, RequestType*,
      ::grpc::ServerAsyncResponseWriter<ResponseType>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*)>;
  using ExecutorQueueFuncType = std::function<void(
      ::grpc::ServerContext*, RequestType*,
      ::grpc::ServerAsyncResponseWriter<ResponseType>*, void*)>;

  ~LifeCycleUnary() override {}

 protected:
  LifeCycleUnary() = default;
  void SetQueueFunc(ExecutorQueueFuncType);

  virtual void ExecuteRPC(RequestType& request, ResponseType& response) = 0;

  uintptr_t GetExecutionContext() final override { return 0; }
  void CompleteExecution(uintptr_t execution_context) final override
  {
    FinishResponse();
  }

  void FinishResponse() final override;
  void CancelResponse() final override;

 private:
  // IContext Methods
  bool RunNextState(bool ok) final override;
  void Reset() final override;

  // LifeCycleUnary Specific Methods
  bool StateRequestDone(bool ok);
  bool StateFinishedDone(bool ok);

  // Function pointers
  ExecutorQueueFuncType m_QueuingFunc;
  bool (LifeCycleUnary<RequestType, ResponseType>::*m_NextState)(bool);

  // Variables
  std::unique_ptr<RequestType> m_Request;
  std::unique_ptr<ResponseType> m_Response;
  std::unique_ptr<::grpc::ServerContext> m_Context;
  std::unique_ptr<::grpc::ServerAsyncResponseWriter<ResponseType>>
      m_ResponseWriter;

 public:
  template <class RequestFuncType, class ServiceType>
  static ServiceQueueFuncType BindServiceQueueFunc(
      /*
      std::function<void(
          ServiceType *, ::grpc::ServerContext *, RequestType *,
          ::grpc::ServerAsyncResponseWriter<ResponseType> *,
          ::grpc::CompletionQueue *, ::grpc::ServerCompletionQueue *, void *)>
      */
      RequestFuncType request_fn, ServiceType* service_type)
  {
    return std::bind(
        request_fn, service_type,
        std::placeholders::_1,  // ServerContext*
        std::placeholders::_2,  // InputType
        std::placeholders::_3,  // AsyncResponseWriter<OutputType>
        std::placeholders::_4,  // CQ
        std::placeholders::_5,  // ServerCQ
        std::placeholders::_6   // Tag
    );
  }

  static ExecutorQueueFuncType BindExecutorQueueFunc(
      ServiceQueueFuncType service_q_fn, ::grpc::ServerCompletionQueue* cq)
  {
    return std::bind(
        service_q_fn,
        std::placeholders::_1,  // ServerContext*
        std::placeholders::_2,  // Request *
        std::placeholders::_3,  // AsyncResponseWriter<Response> *
        cq, cq,
        std::placeholders::_4  // Tag
    );
  }
};

// Implementation

template <class Request, class Response>
bool
LifeCycleUnary<Request, Response>::RunNextState(bool ok)
{
  return (this->*m_NextState)(ok);
}

template <class Request, class Response>
void
LifeCycleUnary<Request, Response>::Reset()
{
  OnLifeCycleReset();
  m_Request.reset(new Request);
  m_Response.reset(new Response);
  m_Context.reset(new ::grpc::ServerContext);
  m_ResponseWriter.reset(
      new ::grpc::ServerAsyncResponseWriter<ResponseType>(m_Context.get()));
  m_NextState = &LifeCycleUnary<RequestType, ResponseType>::StateRequestDone;
  m_QueuingFunc(
      m_Context.get(), m_Request.get(), m_ResponseWriter.get(),
      IContext::Tag());
}

template <class Request, class Response>
bool
LifeCycleUnary<Request, Response>::StateRequestDone(bool ok)
{
  if (!ok)
    return false;
  OnLifeCycleStart();
  ExecuteRPC(*m_Request, *m_Response);
  return true;
}

template <class Request, class Response>
bool
LifeCycleUnary<Request, Response>::StateFinishedDone(bool ok)
{
  return false;
}

template <class Request, class Response>
void
LifeCycleUnary<Request, Response>::FinishResponse()
{
  m_NextState = &LifeCycleUnary<RequestType, ResponseType>::StateFinishedDone;
  m_ResponseWriter->Finish(*m_Response, ::grpc::Status::OK, IContext::Tag());
}

template <class Request, class Response>
void
LifeCycleUnary<Request, Response>::CancelResponse()
{
  m_NextState = &LifeCycleUnary<RequestType, ResponseType>::StateFinishedDone;
  m_ResponseWriter->Finish(
      *m_Response, ::grpc::Status::CANCELLED, IContext::Tag());
}

template <class Request, class Response>
void
LifeCycleUnary<Request, Response>::SetQueueFunc(ExecutorQueueFuncType queue_fn)
{
  m_QueuingFunc = queue_fn;
}

}  // namespace nvrpc
