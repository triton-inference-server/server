// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "stream_infer_handler.h"
#include <regex>

namespace triton { namespace server { namespace grpc {

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error*
StreamInferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload<inference::ModelStreamInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelStreamInferResponse>*>(
          userp);

  auto response = payload->response_queue_->GetLastAllocatedResponse();

  if (response == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Unable to access the last allocated response");
  }

  return ResponseAllocatorHelper(
      allocator, tensor_name, byte_size, preferred_memory_type,
      preferred_memory_type_id, response->mutable_infer_response(),
      payload->shm_map_, buffer, buffer_userp, actual_memory_type,
      actual_memory_type_id);
}

TRITONSERVER_Error*
StreamInferResponseStart(TRITONSERVER_ResponseAllocator* allocator, void* userp)
{
  AllocPayload<inference::ModelStreamInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelStreamInferResponse>*>(
          userp);

  // Move to the next response object
  payload->response_queue_->AllocateResponse();

  return nullptr;  // success
}

// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error*
StreamOutputBufferQuery(
    TRITONSERVER_ResponseAllocator* allocator, void* userp,
    const char* tensor_name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  AllocPayload<inference::ModelStreamInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelStreamInferResponse>*>(
          userp);
  return OutputBufferQueryHelper(
      allocator, tensor_name, byte_size, payload->shm_map_, memory_type,
      memory_type_id);
}

// Make sure to keep InferResponseAlloc, OutputBufferQuery, and
// OutputBufferAttributes logic in sync
TRITONSERVER_Error*
StreamOutputBufferAttributes(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
    void* buffer_userp)
{
  AllocPayload<inference::ModelStreamInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelStreamInferResponse>*>(
          userp);

  return OutputBufferAttributesHelper(
      allocator, tensor_name, payload->shm_map_, buffer_attributes);
}

void
ModelStreamInferHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(cq_, NEXT_UNIQUE_ID);
  context->SetCompressionLevel(compression_level_);
  State* state = StateNew(tritonserver_.get(), context);

#ifdef TRITON_ENABLE_TRACING
  // Can't create trace as we don't know the model to be requested,
  // track timestamps in 'state'
  state->trace_timestamps_.emplace_back(
      std::make_pair("GRPC_WAITREAD_START", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

  service_->RequestModelStreamInfer(
      state->context_->ctx_.get(), state->context_->responder_.get(), cq_, cq_,
      state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ModelStreamInferHandler::Process(InferHandler::State* state, bool rpc_ok)
{
  LOG_VERBOSE(1) << "Process for " << Name() << ", rpc_ok=" << rpc_ok
                 << ", context " << state->context_->unique_id_ << ", "
                 << state->unique_id_ << " step " << state->step_;

  // We need an explicit finish indicator. Can't use 'state->step_'
  // because we launch an async thread that could update 'state's
  // step_ to be FINISH before this thread exits this function.
  bool finished = false;

  if (state->step_ == Steps::START) {
    // A new stream connection... If RPC failed on a new request then
    // the server is shutting down and so we should do nothing.
    if (!rpc_ok) {
      state->step_ = Steps::FINISH;
      return false;
    }

    // Start a new request to replace this one...
    StartNewRequest();

    if (ExecutePrecondition(state)) {
      // Since this is the start of a connection, 'state' hasn't been
      // used yet so use it to read a request off the connection.
      state->context_->step_ = Steps::READ;
      state->step_ = Steps::READ;
      state->context_->responder_->Read(&state->request_, state);
    } else {
      // Precondition is not satisfied, cancel the stream
      state->context_->step_ = Steps::COMPLETE;
      state->step_ = Steps::COMPLETE;
      ::grpc::Status status = ::grpc::Status(
          ::grpc::StatusCode::UNAVAILABLE,
          std::string("This protocol is restricted, expecting header '") +
              restricted_kv_.first + "'");
      state->context_->responder_->Finish(status, state);
      return !finished;
    }

  } else if (state->step_ == Steps::READ) {
    TRITONSERVER_Error* err = nullptr;
    const inference::ModelInferRequest& request = state->request_;
#ifdef TRITON_ENABLE_TRACING
    state->trace_timestamps_.emplace_back(
        std::make_pair("GRPC_WAITREAD_END", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

    // If done reading and no in-flight requests then can finish the
    // entire stream. Otherwise just finish this state.
    if (!rpc_ok) {
      state->context_->step_ = Steps::WRITEREADY;
      if (state->context_->IsRequestsCompleted()) {
        state->context_->step_ = Steps::COMPLETE;
        state->step_ = Steps::COMPLETE;
        state->context_->responder_->Finish(
            state->context_->finish_ok_ ? ::grpc::Status::OK
                                        : ::grpc::Status::CANCELLED,
            state);
      } else {
        state->step_ = Steps::FINISH;
        finished = true;
      }

      return !finished;
    }

    int64_t requested_model_version;
    err = GetModelVersionFromString(
        request.model_version(), &requested_model_version);

    // Record the transaction policy of the model into the current state
    // object.
    if (err == nullptr) {
      uint32_t txn_flags;
      err = TRITONSERVER_ServerModelTransactionProperties(
          tritonserver_.get(), request.model_name().c_str(),
          requested_model_version, &txn_flags, nullptr /* voidp */);
      if (err == nullptr) {
        state->is_decoupled_ = ((txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0);
      }
    }

    // Request has been successfully read, increment the context request
    // counter.
    state->context_->IncrementRequestCounter();

    // If the request is not for a model with decoupled transaction policy
    // then put it in the context queue so thats it's response is sent in
    // the same order as the request was received.
    if (!state->is_decoupled_) {
      state->context_->EnqueueForResponse(state);
    }

    // Need to get context here as it is needed below. 'state' can
    // complete inference, write response, and finish (which releases
    // context) before we make any forward progress.... so need to
    // hold onto context here while we know it is good.
    std::shared_ptr<StateContext> context = state->context_;

    // Issue the inference request into server...
    auto response_queue_ = state->response_queue_;

    // Create the inference request which contains all the
    // input information needed for an inference.
    TRITONSERVER_InferenceRequest* irequest = nullptr;
    if (err == nullptr) {
      err = TRITONSERVER_InferenceRequestNew(
          &irequest, tritonserver_.get(), request.model_name().c_str(),
          requested_model_version);
    }

    if (err == nullptr) {
      err = SetInferenceRequestMetadata(irequest, request);
    }

    if (err == nullptr) {
      err = ForwardHeadersAsParameters(irequest, state);
    }

    // Will be used to hold the serialized data in case explicit string
    // tensors are present in the request.
    std::list<std::string> serialized_data;

    if (err == nullptr) {
      err = InferGRPCToInput(
          tritonserver_, shm_manager_, request, &serialized_data, irequest);
    }
    if (err == nullptr) {
      err = InferAllocatorPayload<inference::ModelStreamInferResponse>(
          tritonserver_, shm_manager_, request, std::move(serialized_data),
          response_queue_, &state->alloc_payload_);
    }
    if (err == nullptr) {
      err = TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest, InferRequestComplete, nullptr /* request_release_userp */);
    }
    if (err == nullptr) {
      err = TRITONSERVER_InferenceRequestSetResponseCallback(
          irequest, allocator_,
          &state->alloc_payload_ /* response_allocator_userp */,
          StreamInferResponseComplete, reinterpret_cast<void*>(state));
    }
    // Get request ID for logging in case of error.
    const char* request_id = "";
    if (irequest != nullptr) {
      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_InferenceRequestId(irequest, &request_id),
          "unable to retrieve request ID string");
    }
    if (!strncmp(request_id, "", 1)) {
      request_id = "<id_unknown>";
    }
    if (err == nullptr) {
      TRITONSERVER_InferenceTrace* triton_trace = nullptr;
#ifdef TRITON_ENABLE_TRACING
      state->trace_ =
          std::move(trace_manager_->SampleTrace(request.model_name()));
      if (state->trace_ != nullptr) {
        triton_trace = state->trace_->trace_;
      }
#endif  // TRITON_ENABLE_TRACING

      state->step_ = ISSUED;
      err = TRITONSERVER_ServerInferAsync(
          tritonserver_.get(), irequest, triton_trace);
    }

    // If there was not an error in issuing the 'state' request then
    // state->step_ == ISSUED and inference request has
    // initiated... the completion callback will transition to
    // WRITEREADY or WRITTEN. If there was an error then enqueue the
    // error response and show it to be ready for writing.
    if (err != nullptr) {
      inference::ModelStreamInferResponse* response;
      if (state->is_decoupled_) {
        state->response_queue_->AllocateResponse();
        response = state->response_queue_->GetLastAllocatedResponse();
      } else {
        response = state->response_queue_->GetNonDecoupledResponse();
      }
      LOG_VERBOSE(1) << "[request id: " << request_id << "] "
                     << "Infer failed: " << TRITONSERVER_ErrorMessage(err);

      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_InferenceRequestDelete(irequest),
          "deleting GRPC inference request");

      ::grpc::Status status;
      GrpcStatusUtil::Create(&status, err);
      TRITONSERVER_ErrorDelete(err);
      response->set_error_message(status.error_message());

      response->mutable_infer_response()->Clear();
      // repopulate the id so that client knows which request failed.
      response->mutable_infer_response()->set_id(request.id());
      state->step_ = Steps::WRITEREADY;
      if (!state->is_decoupled_) {
        state->context_->WriteResponseIfReady(state);
      } else {
        state->response_queue_->MarkNextResponseComplete();
        state->complete_ = true;
        state->context_->PutTaskBackToQueue(state);
      }
    }

    // Now that the inference request is in flight, create a copy of
    // 'state' and use it to attempt another read from the connection
    // (i.e the next request in the stream).
    State* next_read_state =
        StateNew(tritonserver_.get(), context, Steps::READ);

#ifdef TRITON_ENABLE_TRACING
    // Capture a timestamp for the time when we start waiting for this
    // next request to read.
    // Can't create trace as we don't know the model to be requested,
    // track timestamps in 'state'
    next_read_state->trace_timestamps_.emplace_back(std::make_pair(
        "GRPC_WAITREAD_START", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

    next_read_state->context_->responder_->Read(
        &next_read_state->request_, next_read_state);

  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
    finished = true;
  } else if (!state->is_decoupled_) {
    // We handle the WRITTEN and WRITEREADY states little
    // differently depending whether the inference request
    // is for a decoupled model or not. This is because the
    // grpc contract requires us to call Write() only once
    // on a task. Hence, for decoupled writes, we call only
    // one write and then wait for another notification from
    // the completion queue to execute pending Write()'s, if
    // any.

    //
    // Non-Decoupled state transitions
    //
    if (state->step_ == Steps::WRITTEN) {
      state->context_->ongoing_write_ = false;
#ifdef TRITON_ENABLE_TRACING
      state->trace_timestamps_.emplace_back(
          std::make_pair("GRPC_SEND_END", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

      // If the write failed (for example, client closed the stream)
      // mark that the stream did not complete successfully but don't
      // cancel right away... need to wait for any pending reads,
      // inferences and writes to complete.
      if (!rpc_ok) {
        LOG_VERBOSE(1) << "Write for " << Name() << ", rpc_ok=" << rpc_ok
                       << ", context " << state->context_->unique_id_ << ", "
                       << state->unique_id_ << " step " << state->step_
                       << ", failed";
        state->context_->finish_ok_ = false;
      }

      // Log an error if 'state' is not the expected next response. Mark
      // that the stream did not complete successfully but don't cancel
      // right away... need to wait for any pending reads, inferences
      // and writes to complete.
      if (!state->context_->PopCompletedResponse(state)) {
        LOG_ERROR << "Unexpected response for " << Name()
                  << ", rpc_ok=" << rpc_ok << ", context "
                  << state->context_->unique_id_ << ", " << state->unique_id_
                  << " step " << state->step_;
        state->context_->finish_ok_ = false;
      }

      // Write the next response if it is ready...
      state->context_->WriteResponseIfReady(nullptr);

      // The response for the request has been written completely.
      // The counter can be safely decremented.
      state->context_->DecrementRequestCounter();
      finished = Finish(state);

    } else if (state->step_ == Steps::COMPLETE) {
      state->step_ = Steps::FINISH;
      finished = true;
    }
  } else {
    //
    //  Decoupled state transitions
    //
    if (state->step_ == Steps::WRITTEN) {
      state->context_->ongoing_write_ = false;
#ifdef TRITON_ENABLE_TRACING
      state->trace_timestamps_.emplace_back(
          std::make_pair("GRPC_SEND_END", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

      // If the write failed (for example, client closed the stream)
      // mark that the stream did not complete successfully but don't
      // cancel right away... need to wait for any pending reads,
      // inferences and writes to complete.
      if (!rpc_ok) {
        LOG_VERBOSE(1) << "Write for " << Name() << ", rpc_ok=" << rpc_ok
                       << ", context " << state->context_->unique_id_ << ", "
                       << state->unique_id_ << " step " << state->step_
                       << ", failed";
        state->context_->finish_ok_ = false;
      }

      // Finish the state if all the transactions associated with
      // the state have completed.
      if (state->IsComplete()) {
        state->context_->DecrementRequestCounter();
        finished = Finish(state);
      } else {
        std::lock_guard<std::mutex> lock(state->step_mtx_);

        // If there is an available response to be written
        // to the stream, then transition directly to WRITEREADY
        // state and enqueue itself to the completion queue to be
        // taken up later. Otherwise, go to ISSUED state and wait
        // for the callback to make a response available.
        if (state->response_queue_->HasReadyResponse()) {
          state->step_ = Steps::WRITEREADY;
          state->context_->PutTaskBackToQueue(state);
        } else {
          state->step_ = Steps::ISSUED;
        }
      }
    } else if (state->step_ == Steps::WRITEREADY) {
      if (state->delay_response_ms_ != 0) {
        // Will delay the write of the response by the specified time.
        // This can be used to test the flow where there are other
        // responses available to be written.
        LOG_INFO << "Delaying the write of the response by "
                 << state->delay_response_ms_ << " ms...";
        std::this_thread::sleep_for(
            std::chrono::milliseconds(state->delay_response_ms_));
      }

      // Finish the state if all the transactions associated with
      // the state have completed.
      if (state->IsComplete()) {
        state->context_->DecrementRequestCounter();
        finished = Finish(state);
      } else {
        // GRPC doesn't allow to issue another write till
        // the notification from previous write has been
        // delivered. If there is an ongoing write then
        // defer writing and place the task at the back
        // of the completion queue to be taken up later.
        if (!state->context_->ongoing_write_) {
          state->context_->ongoing_write_ = true;
          state->context_->DecoupledWriteResponse(state);
        } else {
          state->context_->PutTaskBackToQueue(state);
        }
      }
    }
  }

  return !finished;
}

bool
ModelStreamInferHandler::Finish(InferHandler::State* state)
{
  // If done reading and no in-flight requests then can finish the
  // entire stream. Otherwise just finish this state.
  if (state->context_->IsRequestsCompleted()) {
    state->context_->step_ = Steps::COMPLETE;
    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(
        state->context_->finish_ok_ ? ::grpc::Status::OK
                                    : ::grpc::Status::CANCELLED,
        state);
  } else {
    state->step_ = Steps::FINISH;
    return true;
  }

  return false;
}

void
ModelStreamInferHandler::StreamInferResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
    void* userp)
{
  State* state = reinterpret_cast<State*>(userp);

  // Increment the callback index
  uint32_t response_index = state->cb_count_++;

  LOG_VERBOSE(1) << "ModelStreamInferHandler::StreamInferComplete, context "
                 << state->context_->unique_id_ << ", " << state->unique_id_
                 << " step " << state->step_ << ", callback index "
                 << state->cb_count_ << ", flags " << flags;

#ifdef TRITON_ENABLE_TRACING
  if (state->cb_count_ == 1) {
    state->trace_timestamps_.emplace_back(std::make_pair(
        "INFER_RESPONSE_COMPLETE", TraceManager::CaptureTimestamp()));
  }
#endif  // TRITON_ENABLE_TRACING

  // Log appropriate errors
  if (!state->is_decoupled_) {
    if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
      LOG_ERROR << "[INTERNAL] ModelStreamInfer received a response without "
                   "FINAL flag for a model with one-to-one transaction";
    }
    if (iresponse == nullptr) {
      LOG_ERROR << "[INTERNAL] ModelStreamInfer received a null response for a "
                   "model with one-to-one transaction";
    }
  }

  auto& response_queue = state->response_queue_;

  if (iresponse != nullptr) {
    auto response = response_queue->GetResponseAt(response_index);
    if (response == nullptr) {
      LOG_ERROR << "expected the response allocator to have added the response";
    }

    TRITONSERVER_Error* err = nullptr;
    if (iresponse != nullptr) {
      inference::ModelInferResponse& infer_response =
          *(response->mutable_infer_response());
      err = InferResponseCompleteCommon<inference::ModelStreamInferResponse>(
          state->tritonserver_, iresponse, infer_response,
          state->alloc_payload_);
    }

    if (err != nullptr) {
      ::grpc::Status status;
      GrpcStatusUtil::Create(&status, err);
      response->mutable_infer_response()->Clear();
      response->set_error_message(status.error_message());

      // repopulate the id so that client knows which request failed.
      const char* id;
      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_InferenceResponseId(iresponse, &id),
          "couldn't retrieve id for failed request");
      LOG_VERBOSE(1) << "Failed for ID: " << id << std::endl;
      response->mutable_infer_response()->set_id(id);
    }

    TRITONSERVER_ErrorDelete(err);

    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseDelete(iresponse),
        "deleting GRPC inference response");
  }

  state->complete_ = ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0);
  if (!state->is_decoupled_) {
    state->step_ = Steps::WRITEREADY;
    state->context_->WriteResponseIfReady(state);
  } else {
    std::lock_guard<std::mutex> lock(state->step_mtx_);
    if (iresponse != nullptr) {
      state->response_queue_->MarkNextResponseComplete();
    }
    if (state->step_ == Steps::ISSUED) {
      state->step_ = Steps::WRITEREADY;
      state->context_->PutTaskBackToQueue(state);
    }
  }
}

}}}  // namespace triton::server::grpc
