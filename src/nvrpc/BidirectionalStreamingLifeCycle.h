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

#include <memory>
#include <queue>
#include <unordered_map>

#include "src/nvrpc/Interfaces.h"

namespace nvrpc {

// A bidirectional streaming version of LifeCycleUnary class
// Note that the bidirectional streaming feature in gRPC supports
// arbitrary call order of ServerReaderWriter::Read() and
// ServerReaderWriter::Write(), so we are able to handle
// reading request and writing response seperately.
template <class Request, class Response>
class BidirectionalStreamingLifeCycle : public IContextLifeCycle {
 public:
  using RequestType = Request;
  using ResponseType = Response;
  using ServiceQueueFuncType = std::function<void(
      ::grpc::ServerContext*,
      ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>*,
      ::grpc::CompletionQueue*, ::grpc::ServerCompletionQueue*, void*)>;
  using ExecutorQueueFuncType = std::function<void(
      ::grpc::ServerContext*,
      ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>*, void*)>;

  ~BidirectionalStreamingLifeCycle() override {}

 protected:
  // Execution context used to identify the completed response
  // so that the resource can be released
  template <class RequestType, class ResponseType>
  struct ExecutionContext {
    RequestType m_Request;
    ResponseType m_Response;
  };

  using ExecutionContextType = ExecutionContext<RequestType, ResponseType>;

  // Class to wrap over the State function pointers to allow the use of
  // different tags while referencing to the same Context.
  // Executor Detag() the tag, which points to the StateContext, which contains
  // a pointer to the actual context (master context).
  template <class RequestType, class ResponseType>
  class StateContext : public IContext {
   public:
    StateContext(IContext* master) : IContext(master) {}

   private:
    // IContext Methods
    bool RunNextState(bool ok) final override
    {
      return (
          static_cast<BidirectionalStreamingLifeCycle*>(m_MasterContext)
              ->*m_NextState)(ok);
    }
    void Reset() final override {}

    bool (BidirectionalStreamingLifeCycle<RequestType, ResponseType>::*
              m_NextState)(bool);

    friend class BidirectionalStreamingLifeCycle<RequestType, ResponseType>;
  };

  BidirectionalStreamingLifeCycle();
  void SetQueueFunc(ExecutorQueueFuncType);

  // Function to actually process the request
  virtual void ExecuteRPC(RequestType& request, ResponseType& response) = 0;

  uintptr_t GetExecutionContext() final override;
  void CompleteExecution(uintptr_t execution_context) final override;

  void FinishResponse() final override;
  void CancelResponse() final override;

 private:
  // IContext Methods
  bool RunNextState(bool ok) final override;
  void Reset() final override;

  // BidirectionalStreamingLifeCycle Specific Methods
  bool StateInitializedDone(bool ok);
  bool StateRequestDone(bool ok);
  bool StateResponseDone(bool ok);
  bool StateFinishedDone(bool ok);

  // Function pointers
  ExecutorQueueFuncType m_QueuingFunc;
  bool (BidirectionalStreamingLifeCycle<RequestType, ResponseType>::*
            m_NextState)(bool);

  // Variables
  // The mutex will be more useful once we can keep reading requests
  // without waiting for response to be sent
  std::mutex m_QueueMutex;
  std::unordered_map<uintptr_t, std::shared_ptr<ExecutionContextType>>
      live_contexts;
  std::queue<std::shared_ptr<ExecutionContextType>> m_WriteBackQueue;

  std::shared_ptr<ExecutionContextType> m_ExecutionContext;

  StateContext<RequestType, ResponseType> m_ReadStateContext;
  StateContext<RequestType, ResponseType> m_WriteStateContext;

  std::unique_ptr<::grpc::ServerContext> m_Context;
  std::unique_ptr<::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>>
      m_ReaderWriter;

  friend class StateContext<RequestType, ResponseType>;

 public:
  template <class RequestFuncType, class ServiceType>
  static ServiceQueueFuncType BindServiceQueueFunc(
      /*
      std::function<void(
          ServiceType *, ::grpc::ServerContext *,
          ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>*,
          ::grpc::CompletionQueue *, ::grpc::ServerCompletionQueue *, void *)>
      */
      RequestFuncType request_fn, ServiceType* service_type)
  {
    return std::bind(
        request_fn, service_type,
        std::placeholders::_1,  // ServerContext*
        std::placeholders::_2,  // AsyncReaderWriter<OutputType, InputType>
        std::placeholders::_3,  // CQ
        std::placeholders::_4,  // ServerCQ
        std::placeholders::_5   // Tag
    );
  }

  static ExecutorQueueFuncType BindExecutorQueueFunc(
      ServiceQueueFuncType service_q_fn, ::grpc::ServerCompletionQueue* cq)
  {
    return std::bind(
        service_q_fn,
        std::placeholders::_1,  // ServerContext*
        std::placeholders::_2,  // AsyncReaderWriter<Response, Request> *
        cq, cq,
        std::placeholders::_3  // Tag
    );
  }
};

// Implementation
template <class Request, class Response>
bool
BidirectionalStreamingLifeCycle<Request, Response>::RunNextState(bool ok)
{
  return (this->*m_NextState)(ok);
}


template <class Request, class Response>
BidirectionalStreamingLifeCycle<
    Request, Response>::BidirectionalStreamingLifeCycle()
    : m_ReadStateContext(static_cast<IContext*>(this)),
      m_WriteStateContext(static_cast<IContext*>(this))
{
  m_ReadStateContext.m_NextState = &BidirectionalStreamingLifeCycle<
      RequestType, ResponseType>::StateRequestDone;
  m_WriteStateContext.m_NextState = &BidirectionalStreamingLifeCycle<
      RequestType, ResponseType>::StateResponseDone;
}

template <class Request, class Response>
void
BidirectionalStreamingLifeCycle<Request, Response>::Reset()
{
  std::queue<std::shared_ptr<ExecutionContextType>> empty_queue;
  OnLifeCycleReset();
  {
    std::lock_guard<std::mutex> lock(m_QueueMutex);
    m_WriteBackQueue.swap(empty_queue);
    live_contexts.clear();
    m_ExecutionContext.reset();
    m_Context.reset(new ::grpc::ServerContext);
    m_ReaderWriter.reset(
        new ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>(
            m_Context.get()));

    m_NextState = &BidirectionalStreamingLifeCycle<
        RequestType, ResponseType>::StateInitializedDone;
  }
  m_QueuingFunc(m_Context.get(), m_ReaderWriter.get(), IContext::Tag());
}

// The following are a set of functions used as function pointers
// to keep track of the state of the context.
template <class Request, class Response>
bool
BidirectionalStreamingLifeCycle<Request, Response>::StateInitializedDone(
    bool ok)
{
  if (!ok) {
    return false;
  }

  OnLifeCycleStart();
  // Start reading once connection is created
  {
    std::lock_guard<std::mutex> lock(m_QueueMutex);
    m_ExecutionContext.reset(new ExecutionContext<Request, Response>);
    m_NextState = &BidirectionalStreamingLifeCycle<
        RequestType, ResponseType>::StateRequestDone;
  }
  m_ReaderWriter->Read(
      &m_ExecutionContext->m_Request, m_ReadStateContext.IContext::Tag());
  return true;
}

// If the Context.m_NextState is at this state or at StateResponseDone state,
// it will keep reading requests from the stream until no more requests will
// be read (Read() brings back status ok==false)
template <class Request, class Response>
bool
BidirectionalStreamingLifeCycle<Request, Response>::StateRequestDone(bool ok)
{
  // No more message to be read from this stream, however, if there are still
  // requests to be processed, then a ServerReaderWriter::Write() will be
  // called. In that case, change m_NextState as hint for no more requests.
  if (!ok) {
    {
      std::lock_guard<std::mutex> lock(m_QueueMutex);
      if (!live_contexts.empty() || !m_WriteBackQueue.empty()) {
        // if state is not StateRequestDone, Finish() is called
        // or is going to be called, don't "undo" the state change
        if (m_NextState == &BidirectionalStreamingLifeCycle<
                               RequestType, ResponseType>::StateRequestDone) {
          m_NextState = &BidirectionalStreamingLifeCycle<
              RequestType, ResponseType>::StateResponseDone;
        }
        return true;
      }
    }
    // No pending requests, finish the call
    FinishResponse();
    return true;
  }

  {
    std::lock_guard<std::mutex> lock(m_QueueMutex);

    // Put the execution context in live context set
    live_contexts.emplace(GetExecutionContext(), m_ExecutionContext);
  }
  // Always start RPC on receiving request successfully
  ExecuteRPC(m_ExecutionContext->m_Request, m_ExecutionContext->m_Response);

  // Start reading the next request
  m_ExecutionContext.reset(new ExecutionContext<Request, Response>);
  m_ReaderWriter->Read(
      &m_ExecutionContext->m_Request, m_ReadStateContext.IContext::Tag());

  return true;
}

// If the Context.m_NextState is at this state or at StateResponseDone state,
// it will keep writing completed response to the stream until it is closed
// (Write() brings back status ok==false)
template <class Request, class Response>
bool
BidirectionalStreamingLifeCycle<Request, Response>::StateResponseDone(bool ok)
{
  // If write didn't go through, then the call is dead unexpectedly.
  // Start reseting
  if (!ok) {
    CancelResponse();
    return true;
  }

  // Done writing back one response
  bool should_write = true;
  bool should_finish = false;
  {
    std::lock_guard<std::mutex> lock(m_QueueMutex);

    // Front of the queue is done, clean up that execution context
    m_WriteBackQueue.pop();
    should_write = !m_WriteBackQueue.empty();

    // check if all requests are processed
    if (!should_write && live_contexts.empty() &&
        m_NextState == &BidirectionalStreamingLifeCycle<
                           RequestType, ResponseType>::StateResponseDone) {
      should_finish = true;
    }
  }
  if (should_finish) {
    FinishResponse();
    return true;
  }
  // Only call Write() if the write back queue is not empty,
  // the CompleteExecution() will call Write() otherwise.
  if (should_write) {
    m_ReaderWriter->Write(
        m_WriteBackQueue.front()->m_Response,
        m_WriteStateContext.IContext::Tag());
  }
  return true;
}

template <class Request, class Response>
bool
BidirectionalStreamingLifeCycle<Request, Response>::StateFinishedDone(bool ok)
{
  return false;
}

template <class Request, class Response>
uintptr_t
BidirectionalStreamingLifeCycle<Request, Response>::GetExecutionContext()
{
  return reinterpret_cast<uintptr_t>(m_ExecutionContext.get());
}

template <class Request, class Response>
void
BidirectionalStreamingLifeCycle<Request, Response>::CompleteExecution(
    uintptr_t execution_context)
{
  bool should_write = false;
  bool should_cancel = false;
  {
    std::lock_guard<std::mutex> lock(m_QueueMutex);
    auto it = live_contexts.find(execution_context);
    if (it != live_contexts.end()) {
      should_write = m_WriteBackQueue.empty();
      m_WriteBackQueue.push(it->second);
      live_contexts.erase(execution_context);
    } else {
      // Unexpected behavior, cancel the stream
      should_cancel = true;
    }

    // Check if the stream should have been cancelled but need to wait for RPCs
    if (m_NextState == &BidirectionalStreamingLifeCycle<
                           RequestType, ResponseType>::StateFinishedDone) {
      should_cancel = true;
    }
  }
  if (should_cancel) {
    CancelResponse();
  }
  if (should_write) {
    m_ReaderWriter->Write(
        m_WriteBackQueue.front()->m_Response,
        m_WriteStateContext.IContext::Tag());
  }
}

template <class Request, class Response>
void
BidirectionalStreamingLifeCycle<Request, Response>::FinishResponse()
{
  {
    std::lock_guard<std::mutex> lock(m_QueueMutex);
    m_NextState = &BidirectionalStreamingLifeCycle<
        RequestType, ResponseType>::StateFinishedDone;
  }

  m_ReaderWriter->Finish(::grpc::Status::OK, IContext::Tag());
}

template <class Request, class Response>
void
BidirectionalStreamingLifeCycle<Request, Response>::CancelResponse()
{
  // Need to be carefully in this case
  // Only call Finish() when no RPC is being executed to avoid clearing
  // request and response while they are being referenced in the RPC
  bool reset_ready = false;
  {
    std::lock_guard<std::mutex> lock(m_QueueMutex);
    m_NextState = &BidirectionalStreamingLifeCycle<
        RequestType, ResponseType>::StateFinishedDone;
    reset_ready = live_contexts.empty();
  }
  // Only call Finish() when no RPC is being executed to avoid clearing
  // request and response while they are being referenced in the RPC
  if (reset_ready) {
    m_ReaderWriter->Finish(::grpc::Status::CANCELLED, IContext::Tag());
  }
}

template <class Request, class Response>
void
BidirectionalStreamingLifeCycle<Request, Response>::SetQueueFunc(
    ExecutorQueueFuncType queue_fn)
{
  m_QueuingFunc = queue_fn;
}

}  // namespace nvrpc
