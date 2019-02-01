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

#include <queue>

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
  std::queue<RequestType> m_RequestQueue;
  std::queue<ResponseType> m_ResponseQueue;
  std::queue<ResponseType> m_ResponseWriteBackQueue;

  StateContext<RequestType, ResponseType> m_ReadStateContext;
  StateContext<RequestType, ResponseType> m_WriteStateContext;

  bool m_Executing;

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
  std::queue<RequestType> empty_request_queue;
  std::queue<ResponseType> empty_response_queue;
  std::queue<ResponseType> empty_response_write_back_queue;
  OnLifeCycleReset();
  {
    std::unique_lock<std::mutex> lock(m_QueueMutex);
    m_Executing = false;
    m_RequestQueue.swap(empty_request_queue);
    m_ResponseQueue.swap(empty_response_queue);
    m_ResponseWriteBackQueue.swap(empty_response_write_back_queue);
    m_Context.reset(new ::grpc::ServerContext);
    m_ReaderWriter.reset(
        new ::grpc::ServerAsyncReaderWriter<ResponseType, RequestType>(
            m_Context.get()));
  }
  m_NextState = &BidirectionalStreamingLifeCycle<
      RequestType, ResponseType>::StateInitializedDone;
  m_QueuingFunc(m_Context.get(), m_ReaderWriter.get(), IContext::Tag());
}

// The following are a set of functions used as function pointers
// to keep track of the state of the context.
template <class Request, class Response>
bool
BidirectionalStreamingLifeCycle<Request, Response>::StateInitializedDone(
    bool ok)
{
  if (!ok)
    return false;

  OnLifeCycleStart();
  // Start reading once connection is created
  {
    std::unique_lock<std::mutex> lock(m_QueueMutex);
    m_RequestQueue.emplace();
  }
  m_NextState = &BidirectionalStreamingLifeCycle<
      RequestType, ResponseType>::StateRequestDone;
  m_ReaderWriter->Read(
      &m_RequestQueue.back(), m_ReadStateContext.IContext::Tag());
  return true;
}

// If the Context.m_NextState is at this state or at StateResponseDone state,
// it will keep reading requests from the stream until no more requests will
// be read (Read() brings back status ok==false)
template <class Request, class Response>
bool
BidirectionalStreamingLifeCycle<Request, Response>::StateRequestDone(bool ok)
{
  // No more message to be read from this stream, however, if it is executing
  // a request, then a ServerReaderWriter::Write() will be called. In that case,
  // let WriteStateContext handle the reset procedure.
  if (!ok) {
    {
      std::unique_lock<std::mutex> lock(m_QueueMutex);
      if (m_Executing)
        return true;
    }
    return false;
  }

  // Successfully receive request
  bool should_execute = false;
  {
    std::unique_lock<std::mutex> lock(m_QueueMutex);

    m_ResponseQueue.emplace();
    // [TODO] Need to talk about how we want to do inference with
    //   multiple requests in the same stream. For instance, in RNN models,
    //   the previous request will alter model's inner representation. We can't
    //   just initiate the requests once they arrive and let the scheduler do
    //   the work.

    // At this stage, only start execution when executing flag is false
    // (just received the first request in the queue).
    // Other wise, ExecuteRPC() will be called in FinishResponse() to be sure
    // that we only process next request after the previous request is completed
    // Note that we only add item in response queue when we receive request
    // successfully, FinishResponse() will use this
    // information to determine if it should call ExecuteRPC()
    if (!m_Executing) {
      should_execute = true;
      m_Executing = true;
    }

    // Start reading the next request
    m_RequestQueue.emplace();
  }
  if (m_NextState == &BidirectionalStreamingLifeCycle<
                         RequestType, ResponseType>::StateRequestDone)
    m_NextState = &BidirectionalStreamingLifeCycle<
        RequestType, ResponseType>::StateResponseDone;
  m_ReaderWriter->Read(
      &m_RequestQueue.back(), m_ReadStateContext.IContext::Tag());

  if (should_execute)
    ExecuteRPC(m_RequestQueue.front(), m_ResponseQueue.front());
  return true;
}

// If the Context.m_NextState is at this state or at StateResponseDone state,
// it will keep writing completed response to the stream until it is closed
// (Write() brings back status ok==false)
template <class Request, class Response>
bool
BidirectionalStreamingLifeCycle<Request, Response>::StateResponseDone(bool ok)
{
  // If write didn't go through, then the call is dead. Start reseting
  if (!ok) {
    CancelResponse();
    return true;
  }

  // Done writing back one response
  bool should_write = true;
  {
    std::unique_lock<std::mutex> lock(m_QueueMutex);

    m_ResponseWriteBackQueue.pop();
    should_write = !m_ResponseWriteBackQueue.empty();
  }
  // Only call Write() if the write back queue is not empty,
  // the FinishResponse() will call Write() otherwise.
  if (should_write)
    m_ReaderWriter->Write(
        m_ResponseWriteBackQueue.front(), m_WriteStateContext.IContext::Tag());
  return true;
}

template <class Request, class Response>
bool
BidirectionalStreamingLifeCycle<Request, Response>::StateFinishedDone(bool ok)
{
  return false;
}

template <class Request, class Response>
void
BidirectionalStreamingLifeCycle<Request, Response>::FinishResponse()
{
  bool should_write = false;
  bool should_execute = true;
  {
    std::unique_lock<std::mutex> lock(m_QueueMutex);

    should_write = m_ResponseWriteBackQueue.empty();
    // Push queue::front() because we are not calling ExecuteRPC() concurrently
    // on the same stream, the first item will always be the only completed item
    // in the queue.
    m_ResponseWriteBackQueue.push(std::move(m_ResponseQueue.front()));
    m_RequestQueue.pop();
    m_ResponseQueue.pop();
    if (m_ResponseQueue.empty()) {
      should_execute = false;
      m_Executing = false;
    }
  }
  if (should_write)
    m_ReaderWriter->Write(
        m_ResponseWriteBackQueue.front(), m_WriteStateContext.IContext::Tag());
  if (should_execute)
    ExecuteRPC(m_RequestQueue.front(), m_ResponseQueue.front());
}

template <class Request, class Response>
void
BidirectionalStreamingLifeCycle<Request, Response>::CancelResponse()
{
  m_NextState = &BidirectionalStreamingLifeCycle<
      RequestType, ResponseType>::StateFinishedDone;
  m_ReaderWriter->Finish(::grpc::Status::CANCELLED, IContext::Tag());
}

template <class Request, class Response>
void
BidirectionalStreamingLifeCycle<Request, Response>::SetQueueFunc(
    ExecutorQueueFuncType queue_fn)
{
  m_QueuingFunc = queue_fn;
}

}  // namespace nvrpc
