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

//=============================================================================
//  The following section contains the handling mechanism for ModelStreamInfer
//  RPC. This implementation is tuned towards performance and reducing latency.
//=============================================================================

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
    // Transitions to READ state on success, or COMPLETE/FINISH on errors.
    finished = RequestStartStep(state, rpc_ok);
  } else if (state->step_ == Steps::READ) {
    // Transitions to ISSUED state on successfully sending inference request to
    // Triton, or COMPLETE/FINISH on errors. The ISSUED state is checked in the
    // request's response callback to handle transitioning to writing responses.
    finished = RequestReadStep(state, rpc_ok);
  }
  // We handle the WRITTEN and WRITEREADY states little
  // differently depending whether the inference request
  // is for a decoupled model or not. This is because the
  // grpc contract requires us to call Write() only once
  // on a task. Hence, for decoupled writes, we call only
  // one write and then wait for another notification from
  // the completion queue to execute pending Write()'s, if
  // any.
  else if (state->step_ == Steps::WRITEREADY) {
    // The non-decoupled transition to WRITEREADY state immediately attempts to
    // WriteResponseIfReady() and go to WRITTEN state, so only handle the
    // decoupled case here.
    if (state->is_decoupled_) {
      // Transitions to WRITTEN state if no other writes are ongoing, otherwise
      // remains in WRITEREADY state and is moved to the back of the task queue.
      // If there are no responses left to write, then this transitions to
      // COMPLETE/FINISH states.
      finished = RequestWriteReadyStepDecoupled(state);
    }
  } else if (state->step_ == Steps::WRITTEN) {
    if (state->is_decoupled_) {
      // Transitions to COMPLETE/FINISH state if all responses have been
      // written. Otherwise, transitions to WRITEREADY or ISSUED depending on
      // whether additional responses are ready to write or not.
      finished = RequestWrittenStepDecoupled(state, rpc_ok);
    } else {
      // Transitions to COMPLETE/FINISH state from WRITTEN state here, because
      // there is only one response per-request in the non-decoupled case.
      finished = RequestWrittenStepNonDecoupled(state, rpc_ok);
    }
  }
  // COMPLETE step simply marks that we're finished with the request.
  else if (state->step_ == Steps::COMPLETE) {
    finished = RequestCompleteStep(state);
  }
  // No special handling currently needed here for remaining states like
  // ISSUED and FINISH.

  return finished;
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
ModelStreamInferHandler::RequestStartStep(
    InferHandler::State* state, bool rpc_ok)
{
  // A new stream connection... If RPC failed on a new request then
  // the server is shutting down and so we should do nothing.
  if (!rpc_ok) {
    state->step_ = Steps::FINISH;
    return true;
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
  }

  // Not finished with a request on the start step unless an error occurs above.
  return false;
}

void
ModelStreamInferHandler::PrepareAndSendTritonRequest(InferHandler::State* state)
{
  TRITONSERVER_Error* err = nullptr;
  const inference::ModelInferRequest& request = state->request_;
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
  // then put it in the context queue so that its response is sent in
  // the same order as the request was received.
  if (!state->is_decoupled_) {
    state->context_->EnqueueForResponse(state);
  }

  // Create the inference request which contains all the
  // input information needed for an inference.
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestNew(
        &irequest, tritonserver_.get(), request.model_name().c_str(),
        requested_model_version);
  }

  if (err == nullptr) {
    err = SetInferenceRequestMetadata(irequest, request, state->parameters_);
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
        state->response_queue_, &state->alloc_payload_);
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
    EnqueueErrorResponse(state, irequest, err);
  }
}

void
ModelStreamInferHandler::EnqueueErrorResponse(
    InferHandler::State* state, TRITONSERVER_InferenceRequest* irequest,
    TRITONSERVER_Error* err)
{
  const inference::ModelInferRequest& request = state->request_;
  inference::ModelStreamInferResponse* response;
  if (state->is_decoupled_) {
    state->response_queue_->AllocateResponse();
    response = state->response_queue_->GetLastAllocatedResponse();
  } else {
    response = state->response_queue_->GetNonDecoupledResponse();
  }

  // Get request ID for logging in case of error.
  std::string log_request_id = request.id();
  if (log_request_id.empty()) {
    log_request_id = "<id_unknown>";
  }
  LOG_VERBOSE(1) << "[request id: " << log_request_id << "] "
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

bool
ModelStreamInferHandler::RequestReadStep(
    InferHandler::State* state, bool rpc_ok)
{
  bool finished = false;
#ifdef TRITON_ENABLE_TRACING
  state->trace_timestamps_.emplace_back(
      std::make_pair("GRPC_WAITREAD_END", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

  // If done reading and no in-flight requests then can finish the
  // entire stream. Otherwise just finish this state.
  if (!rpc_ok) {
    // Mark as WRITEREADY to indicate that there are no reads being processed
    // for checking IsRequestsCompleted().
    state->context_->step_ = Steps::WRITEREADY;
    finished = Finish(state);
    return finished;
  }

  // Need to get context here as it is needed below. 'state' can
  // complete inference, write response, and finish (which releases
  // context) before we make any forward progress.... so need to
  // hold onto context here while we know it is good.
  std::shared_ptr<StateContext> context = state->context_;

  // Read protobuf request, convert it into a Triton request, and execute it.
  // Update any related state metadata, such as if model is decoupled or not.
  // Send an error response if any error occurs.
  PrepareAndSendTritonRequest(state);

  // Now that the inference request is in flight, create a copy of
  // 'state' and use it to attempt another read from the connection
  // (i.e the next request in the stream).
  State* next_read_state = StateNew(tritonserver_.get(), context, Steps::READ);

#ifdef TRITON_ENABLE_TRACING
  // Capture a timestamp for the time when we start waiting for this
  // next request to read.
  // Can't create trace as we don't know the model to be requested,
  // track timestamps in 'state'
  next_read_state->trace_timestamps_.emplace_back(
      std::make_pair("GRPC_WAITREAD_START", TraceManager::CaptureTimestamp()));
#endif  // TRITON_ENABLE_TRACING

  next_read_state->context_->responder_->Read(
      &next_read_state->request_, next_read_state);

  // Not finished with a request on the read step unless an error occurs above.
  return false;
}

bool
ModelStreamInferHandler::RequestWrittenStepDecoupled(
    InferHandler::State* state, bool rpc_ok)
{
  bool finished = false;
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

  return finished;
}

bool
ModelStreamInferHandler::RequestWrittenStepNonDecoupled(
    InferHandler::State* state, bool rpc_ok)
{
  bool finished = false;
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
    LOG_ERROR << "Unexpected response for " << Name() << ", rpc_ok=" << rpc_ok
              << ", context " << state->context_->unique_id_ << ", "
              << state->unique_id_ << " step " << state->step_;
    state->context_->finish_ok_ = false;
  }

  // Write the next response if it is ready...
  state->context_->WriteResponseIfReady(nullptr);

  // The response for the request has been written completely.
  // The counter can be safely decremented.
  state->context_->DecrementRequestCounter();
  finished = Finish(state);
  return finished;
}

bool
ModelStreamInferHandler::RequestWriteReadyStepDecoupled(
    InferHandler::State* state)
{
  bool finished = false;
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

  return finished;
}

bool
ModelStreamInferHandler::RequestCompleteStep(InferHandler::State* state)
{
  state->step_ = Steps::FINISH;
  return true;
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
  state->complete_ = ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0);
  if (!state->is_decoupled_) {
    if (!state->complete_) {
      LOG_ERROR << "[INTERNAL] ModelStreamInfer received a response without "
                   "FINAL flag for a model with one-to-one transaction";
    }
    if (iresponse == nullptr) {
      LOG_ERROR << "[INTERNAL] ModelStreamInfer received a null response for a "
                   "model with one-to-one transaction";
    }
  }

  auto& response_queue = state->response_queue_;
  std::string log_request_id = state->request_.id();
  if (log_request_id.empty()) {
    log_request_id = "<id_unknown>";
  }

  inference::ModelStreamInferResponse* response = nullptr;
  bool failed = false;
  if (iresponse) {
    // Backend returned a non-null response
    TRITONSERVER_Error* err = nullptr;
    response = response_queue->GetResponseAt(response_index);
    if (response) {
      inference::ModelInferResponse& infer_response =
          *(response->mutable_infer_response());
      // Validate Triton iresponse and set grpc/protobuf response fields from
      // it
      err = InferResponseCompleteCommon<inference::ModelStreamInferResponse>(
          state->tritonserver_, iresponse, infer_response,
          state->alloc_payload_);
    } else {
      LOG_ERROR << "expected the response allocator to have added the response";
    }

    if (err != nullptr) {
      failed = true;
      ::grpc::Status status;
      GrpcStatusUtil::Create(&status, err);
      response->mutable_infer_response()->Clear();
      response->set_error_message(status.error_message());
      LOG_VERBOSE(1) << "Failed for ID: " << log_request_id << std::endl;
    }

    TRITONSERVER_ErrorDelete(err);
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseDelete(iresponse),
        "deleting GRPC inference response");
  }

  // Decoupled backends can return a null response via
  // TRITONBACKEND_ResponseFactorySendFlags. By default, these null
  // "empty" responses are not sent back to the client. Clients can
  // opt-in to receiving these empty responses via request parameters.
  // NOTE: The complete flag is the only flag used for this case at this time.
  const bool empty_final =
      (!iresponse && state->is_decoupled_ && state->complete_);
  const bool enable_empty_final =
      state->parameters_.enable_empty_final_response_;

  const bool create_empty_response = (empty_final && enable_empty_final);
  if (create_empty_response) {
    // Assume decoupled here based on prior checks.
    state->response_queue_->AllocateResponse();
    response = state->response_queue_->GetLastAllocatedResponse();
    if (response) {
      LOG_VERBOSE(1) << "[request id: " << log_request_id << "] "
                     << "Creating empty final response";
      response->mutable_infer_response()->Clear();
    } else {
      LOG_ERROR << "expected the response allocator to have added the response";
    }
  }

  if (response) {
    auto& infer_response = *(response->mutable_infer_response());
    // Set response metadata to associate it with request. These will be set
    // by InferResponseCompleteCommon for successful inference.
    if (create_empty_response || failed) {
      infer_response.set_id(state->request_.id());
      infer_response.set_model_name(state->request_.model_name());
      infer_response.set_model_version(state->request_.model_version());
    }
    auto& params = *(infer_response.mutable_parameters());
    params["triton_final_response"].set_bool_param(state->complete_);
  }

  // Update states to signal that response/error is ready to write to stream
  if (state->is_decoupled_) {
    std::lock_guard<std::mutex> lock(state->step_mtx_);
    if (response) {
      state->response_queue_->MarkNextResponseComplete();
    }
    if (state->step_ == Steps::ISSUED) {
      state->step_ = Steps::WRITEREADY;
      state->context_->PutTaskBackToQueue(state);
    }
  } else {
    state->step_ = Steps::WRITEREADY;
    state->context_->WriteResponseIfReady(state);
  }
}
}}}  // namespace triton::server::grpc
