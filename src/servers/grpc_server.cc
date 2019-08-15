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

#include "src/servers/grpc_server.h"

#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "grpc/grpc.h"
#include "src/core/constants.h"
#include "src/core/trtserver.h"
#include "src/servers/common.h"

namespace nvidia { namespace inferenceserver {

namespace {

//
// C++11 doesn't have a barrier so we implement our own.
//
class Barrier {
 public:
  explicit Barrier(size_t cnt) : threshold_(cnt), count_(cnt), generation_(0) {}

  void Wait()
  {
    std::unique_lock<std::mutex> lock(mu_);
    auto lgen = generation_;
    if (--count_ == 0) {
      generation_++;
      count_ = threshold_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, lgen] { return lgen != generation_; });
    }
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  const size_t threshold_;
  size_t count_;
  size_t generation_;
};

// The step of processing that the state is in. Every state must
// recognize START, COMPLETE and FINISH and the others are optional.
typedef enum {
  START,
  COMPLETE,
  FINISH,
  ISSUED,
  READ,
  WRITEREADY,
  WRITTEN
} Steps;

std::ostream&
operator<<(std::ostream& out, const Steps& step)
{
  switch (step) {
    case START:
      out << "START";
      break;
    case COMPLETE:
      out << "COMPLETE";
      break;
    case FINISH:
      out << "FINISH";
      break;
    case ISSUED:
      out << "ISSUED";
      break;
    case READ:
      out << "READ";
      break;
    case WRITEREADY:
      out << "WRITEREADY";
      break;
    case WRITTEN:
      out << "WRITTEN";
      break;
  }

  return out;
}

//
// AllocPayload
//
// Simple structure that carries the userp payload needed for
// allocation. These are just pointers into a HandlerState
// object. HandlerState lifetime is always longer than what is
// required for allocation callback so HandlerState manages the
// lifetime of the actual objects referenced by those pointers.
//
struct AllocPayload {
  struct ShmInfo {
    void* base_;
    size_t byte_size_;
  };

  using TensorShmMap = std::unordered_map<std::string, ShmInfo>;

  explicit AllocPayload() : response_(nullptr), shm_map_(nullptr) {}
  ~AllocPayload()
  {
    // Don't delete 'response_'.. it is owned by the HandlerState
    delete shm_map_;
  }

  InferResponse* response_;
  TensorShmMap* shm_map_;
};

//
// HandlerState
//
template <
    typename ServerResponderType, typename RequestType, typename ResponseType>
class HandlerState {
 public:
  using HandlerStateType =
      HandlerState<ServerResponderType, RequestType, ResponseType>;

  // State that is shared across all state objects that make up a GRPC
  // transaction (e.g. a stream).
  struct Context {
    explicit Context(const char* server_id, const uint64_t unique_id = 0)
        : server_id_(server_id), unique_id_(unique_id), step_(Steps::START),
          finish_ok_(true)
    {
      ctx_.reset(new grpc::ServerContext());
      responder_.reset(new ServerResponderType(ctx_.get()));
    }

    // Enqueue 'state' so that its response is delivered in the
    // correct order.
    void EnqueueForResponse(HandlerStateType* state)
    {
      std::unique_lock<std::mutex> lock(mu_);
      states_.push(state);
    }

    // If the state at the front of the queue is ready for writing
    // then return it, otherwise return nullptr.
    HandlerStateType* ReadyResponse()
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (states_.empty()) {
        return nullptr;
      }

      HandlerStateType* state = states_.front();
      if (state->step_ == Steps::WRITEREADY) {
        return state;
      }

      return nullptr;
    }

    // If 'state' is at the front of the queue and written, pop it and
    // return true. Other return false.
    bool PopCompletedResponse(HandlerStateType* state)
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (states_.empty()) {
        return false;
      }

      HandlerStateType* front = states_.front();
      if ((front == state) && (state->step_ == Steps::WRITTEN)) {
        states_.pop();
        return true;
      }

      return false;
    }

    // Return true if this context has completed all reads and writes.
    bool IsRequestsCompleted()
    {
      std::unique_lock<std::mutex> lock(mu_);
      return ((step_ == Steps::WRITEREADY) && states_.empty());
    }

    // ID for the server this context is on
    const char* const server_id_;

    // Unique ID for the context.
    const uint64_t unique_id_;

    // Context for the rpc, allowing to tweak aspects of it such as
    // the use of compression, authentication, as well as to send
    // metadata back to the client.
    std::unique_ptr<grpc::ServerContext> ctx_;
    std::unique_ptr<ServerResponderType> responder_;

    // The states associated with this context that are currently
    // active. Used by stream handlers to maintain request / response
    // orders. A state enters this queue when it has successfully read
    // a request and exits the queue when it is written.
    std::mutex mu_;
    std::queue<HandlerStateType*> states_;

    // The step of the entire context.
    Steps step_;

    // True if this context should finish with OK status, false if
    // should finish with CANCELLED status.
    bool finish_ok_;
  };

  explicit HandlerState(
      const std::shared_ptr<Context>& context, Steps start_step = Steps::START)
  {
    Reset(context, start_step);
  }

  void Reset(
      const std::shared_ptr<Context>& context, Steps start_step = Steps::START)
  {
    context_ = context;
    unique_id_ = RequestStatusUtil::NextUniqueRequestId();
    step_ = start_step;
    request_.Clear();
    response_.Clear();
  }

  void Release()
  {
    context_ = nullptr;
    request_.Clear();
    response_.Clear();
  }

  std::shared_ptr<Context> context_;

  uint64_t unique_id_;
  Steps step_;

  RequestType request_;
  ResponseType response_;

  // For inference requests the allocator payload, unused for other
  // requests.
  AllocPayload alloc_payload_;
};

//
// Handler
//
template <
    typename ServiceType, typename ServerResponderType, typename RequestType,
    typename ResponseType>
class Handler : public GRPCServer::HandlerBase {
 public:
  Handler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      ServiceType* service, grpc::ServerCompletionQueue* cq,
      size_t max_state_bucket_count);
  virtual ~Handler();

  // Descriptive name of of the handler.
  const std::string& Name() const { return name_; }

  // Start handling requests using 'thread_cnt' threads.
  void Start(int thread_cnt);

  // Stop handling requests.
  void Stop();

 protected:
  using State = HandlerState<ServerResponderType, RequestType, ResponseType>;
  using StateContext = typename State::Context;

  State* StateNew(
      const std::shared_ptr<StateContext>& context,
      Steps start_step = Steps::START)
  {
    State* state = nullptr;

    {
      std::unique_lock<std::mutex> lock(alloc_mu_);

      if (!state_bucket_.empty()) {
        state = state_bucket_.back();
        state->Reset(context, start_step);
        state_bucket_.pop_back();
      }
    }

    if (state == nullptr) {
      state = new State(context, start_step);
    }

    return state;
  }

  void StateRelease(State* state)
  {
    std::unique_lock<std::mutex> lock(alloc_mu_);

    state->Release();
    if (state_bucket_.size() < max_state_bucket_count_) {
      state_bucket_.push_back(state);
    } else {
      delete state;
    }
  }

  virtual void StartNewRequest() = 0;
  virtual bool Process(State* state, bool rpc_ok) = 0;

  const std::string name_;
  std::shared_ptr<TRTSERVER_Server> trtserver_;
  const char* const server_id_;

  ServiceType* service_;
  grpc::ServerCompletionQueue* cq_;
  std::vector<std::unique_ptr<std::thread>> threads_;

  // Mutex to serialize State allocation
  std::mutex alloc_mu_;

  // Keep some number of state objects for reuse to avoid the overhead
  // of creating a state for every new request.
  const size_t max_state_bucket_count_;
  std::vector<State*> state_bucket_;
};

template <
    typename ServiceType, typename ServerResponderType, typename RequestType,
    typename ResponseType>
Handler<ServiceType, ServerResponderType, RequestType, ResponseType>::Handler(
    const std::string& name, const std::shared_ptr<TRTSERVER_Server>& trtserver,
    const char* server_id, ServiceType* service,
    grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
    : name_(name), trtserver_(trtserver), server_id_(server_id),
      service_(service), cq_(cq),
      max_state_bucket_count_(max_state_bucket_count)
{
}

template <
    typename ServiceType, typename ServerResponderType, typename RequestType,
    typename ResponseType>
Handler<ServiceType, ServerResponderType, RequestType, ResponseType>::~Handler()
{
  for (State* state : state_bucket_) {
    delete state;
  }
  state_bucket_.clear();

  LOG_VERBOSE(1) << "Destructed " << Name();
}

template <
    typename ServiceType, typename ServerResponderType, typename RequestType,
    typename ResponseType>
void
Handler<ServiceType, ServerResponderType, RequestType, ResponseType>::Start(
    int thread_cnt)
{
  // Use a barrier to make sure we don't return until all threads have
  // started.
  auto barrier = std::make_shared<Barrier>(thread_cnt + 1);

  for (int t = 0; t < thread_cnt; ++t) {
    threads_.emplace_back(new std::thread([this, barrier] {
      StartNewRequest();
      barrier->Wait();

      void* tag;
      bool ok;

      while (cq_->Next(&tag, &ok)) {
        State* state = static_cast<State*>(tag);
        if (!Process(state, ok)) {
          LOG_VERBOSE(1) << "Done for " << Name() << ", " << state->unique_id_;
          StateRelease(state);
        }
      }
    }));
  }

  barrier->Wait();
  LOG_VERBOSE(1) << "Threads started for " << Name();
}

template <
    typename ServiceType, typename ServerResponderType, typename RequestType,
    typename ResponseType>
void
Handler<ServiceType, ServerResponderType, RequestType, ResponseType>::Stop()
{
  for (const auto& thread : threads_) {
    thread->join();
  }

  LOG_VERBOSE(1) << "Threads exited for " << Name();
}

//
// HealthHandler
//
class HealthHandler : public Handler<
                          GRPCService::AsyncService,
                          grpc::ServerAsyncResponseWriter<HealthResponse>,
                          HealthRequest, HealthResponse> {
 public:
  HealthHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCService::AsyncService* service, grpc::ServerCompletionQueue* cq,
      size_t max_state_bucket_count)
      : Handler(name, trtserver, server_id, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
HealthHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestHealth(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
HealthHandler::Process(Handler::State* state, bool rpc_ok)
{
  LOG_VERBOSE(1) << "Process for " << Name() << ", rpc_ok=" << rpc_ok << ", "
                 << state->unique_id_ << " step " << state->step_;

  // If RPC failed on a new request then the server is shutting down
  // and so we should do nothing (including not registering for a new
  // request). If RPC failed on a non-START step then there is nothing
  // we can do since we one execute one step.
  const bool shutdown = (!rpc_ok && (state->step_ == Steps::START));
  if (shutdown) {
    state->step_ = Steps::FINISH;
  }

  const HealthRequest& request = state->request_;
  HealthResponse& response = state->response_;

  if (state->step_ == Steps::START) {
    TRTSERVER_Error* err = nullptr;
    bool health = false;

    if (request.mode() == "live") {
      err = TRTSERVER_ServerIsLive(trtserver_.get(), &health);
    } else if (request.mode() == "ready") {
      err = TRTSERVER_ServerIsReady(trtserver_.get(), &health);
    } else {
      err = TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_UNKNOWN,
          std::string("unknown health mode '" + request.mode() + "'").c_str());
    }

    response.set_health((err == nullptr) && health);

    RequestStatusUtil::Create(
        response.mutable_request_status(), err, state->unique_id_, server_id_);

    TRTSERVER_ErrorDelete(err);

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(response, grpc::Status::OK, state);
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
  }

  // Only handle one status request at a time (to avoid having status
  // request cause too much load on server), so register for next
  // request only after this one finished.
  if (!shutdown && (state->step_ == Steps::FINISH)) {
    StartNewRequest();
  }

  return state->step_ != Steps::FINISH;
}

//
// StatusHandler
//
class StatusHandler : public Handler<
                          GRPCService::AsyncService,
                          grpc::ServerAsyncResponseWriter<StatusResponse>,
                          StatusRequest, StatusResponse> {
 public:
  StatusHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCService::AsyncService* service, grpc::ServerCompletionQueue* cq,
      size_t max_state_bucket_count)
      : Handler(name, trtserver, server_id, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
StatusHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestStatus(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
StatusHandler::Process(Handler::State* state, bool rpc_ok)
{
  LOG_VERBOSE(1) << "Process for " << Name() << ", rpc_ok=" << rpc_ok << ", "
                 << state->unique_id_ << " step " << state->step_;

  // If RPC failed on a new request then the server is shutting down
  // and so we should do nothing (including not registering for a new
  // request). If RPC failed on a non-START step then there is nothing
  // we can do since we one execute one step.
  const bool shutdown = (!rpc_ok && (state->step_ == Steps::START));
  if (shutdown) {
    state->step_ = Steps::FINISH;
  }

  const StatusRequest& request = state->request_;
  StatusResponse& response = state->response_;

  if (state->step_ == Steps::START) {
    TRTSERVER_Protobuf* server_status_protobuf = nullptr;
    TRTSERVER_Error* err =
        (request.model_name().empty())
            ? TRTSERVER_ServerStatus(trtserver_.get(), &server_status_protobuf)
            : TRTSERVER_ServerModelStatus(
                  trtserver_.get(), request.model_name().c_str(),
                  &server_status_protobuf);
    if (err == nullptr) {
      const char* status_buffer;
      size_t status_byte_size;
      err = TRTSERVER_ProtobufSerialize(
          server_status_protobuf, &status_buffer, &status_byte_size);
      if (err == nullptr) {
        if (!response.mutable_server_status()->ParseFromArray(
                status_buffer, status_byte_size)) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
        }
      }
    }

    TRTSERVER_ProtobufDelete(server_status_protobuf);

    RequestStatusUtil::Create(
        response.mutable_request_status(), err, state->unique_id_, server_id_);

    TRTSERVER_ErrorDelete(err);

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(response, grpc::Status::OK, state);
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
  }

  // Only handle one status request at a time (to avoid having status
  // request cause too much load on server), so register for next
  // request only after this one finished.
  if (!shutdown && (state->step_ == Steps::FINISH)) {
    StartNewRequest();
  }

  return state->step_ != Steps::FINISH;
}

//
// Infer utilities
//
TRTSERVER_Error*
InferResponseAlloc(
    TRTSERVER_ResponseAllocator* allocator, void** buffer, void** buffer_userp,
    const char* tensor_name, size_t byte_size,
    TRTSERVER_Memory_Type memory_type, int64_t memory_type_id, void* userp)
{
  AllocPayload* payload = reinterpret_cast<AllocPayload*>(userp);
  InferResponse* response = payload->response_;
  const AllocPayload::TensorShmMap* shm_map = payload->shm_map_;

  *buffer = nullptr;
  *buffer_userp = nullptr;

  // Can't allocate for any memory type other than CPU. If byte size is 0,
  // proceed regardless of memory type as no allocation is required.
  if ((memory_type != TRTSERVER_MEMORY_CPU) && (byte_size != 0)) {
    LOG_VERBOSE(1) << "GRPC allocation failed for type " << memory_type
                   << " for " << tensor_name;
    return nullptr;
  }

  // Called once for each result tensor in the inference request. Must
  // always add a raw output into the response's list of outputs so
  // that the number and order of raw output entries equals the output
  // meta-data.
  std::string* raw_output = response->add_raw_output();
  if (byte_size > 0) {
    bool use_shm = false;

    if (shm_map != nullptr) {
      const auto& pr = shm_map->find(tensor_name);
      if (pr != shm_map->end()) {
        // If the output is in shared memory then just need to check
        // that the requested size matches what is expected by the
        // request.
        if (byte_size != pr->second.byte_size_) {
          return TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL,
              std::string(
                  "for output " + std::string(tensor_name) +
                  " expected requested buffer size to be " +
                  std::to_string(pr->second.byte_size_) + " bytes but got " +
                  std::to_string(byte_size) + " in actual request")
                  .c_str());
        }

        *buffer = const_cast<void*>(pr->second.base_);
        use_shm = true;

        LOG_VERBOSE(1) << "GRPC shared-memory: " << tensor_name << ", size "
                       << byte_size << ", addr " << *buffer;
      }
    }

    if (!use_shm) {
      raw_output->resize(byte_size);
      *buffer = static_cast<void*>(&((*raw_output)[0]));

      LOG_VERBOSE(1) << "GRPC allocation: " << tensor_name << ", size "
                     << byte_size << ", addr " << *buffer;
    }
  }

  return nullptr;  // Success
}

TRTSERVER_Error*
InferResponseRelease(
    TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRTSERVER_Memory_Type memory_type, int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "GRPC release: "
                 << "size " << byte_size << ", addr " << buffer;

  // Don't do anything when releasing a buffer since ResponseAlloc
  // wrote directly into the response protobuf.
  return nullptr;  // Success
}

TRTSERVER_Error*
InferAllocatorPayload(
    const std::shared_ptr<TRTSERVER_Server>& trtserver,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
    const InferRequestHeader& request_header, InferResponse& response,
    AllocPayload* alloc_payload)
{
  alloc_payload->response_ = &response;

  // If any of the outputs use shared memory, then we must calculate
  // the memory address for that output and store it in the allocator
  // payload so that it is available when the allocation callback is
  // invoked.
  for (const auto& io : request_header.output()) {
    if (io.has_shared_memory()) {
      TRTSERVER_SharedMemoryBlock* smb = nullptr;
      RETURN_IF_ERR(smb_manager->Get(&smb, io.shared_memory().name()));

      void* base;
      RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
          trtserver.get(), smb, io.shared_memory().offset(),
          io.shared_memory().byte_size(), &base));

      if (alloc_payload->shm_map_ == nullptr) {
        alloc_payload->shm_map_ = new AllocPayload::TensorShmMap;
      }

      alloc_payload->shm_map_->emplace(
          io.name(),
          AllocPayload::ShmInfo{base, io.shared_memory().byte_size()});
    }
  }

  return nullptr;  // Success
}

TRTSERVER_Error*
InferGRPCToInput(
    const std::shared_ptr<TRTSERVER_Server>& trtserver,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
    const InferRequestHeader& request_header, const InferRequest& request,
    TRTSERVER_InferenceRequestProvider* request_provider)
{
  // Verify that the batch-byte-size of each input matches the size of
  // the provided tensor data (provided raw or from shared memory)
  size_t idx = 0;
  for (const auto& io : request_header.input()) {
    const void* base;
    size_t byte_size;
    if (io.has_shared_memory()) {
      TRTSERVER_SharedMemoryBlock* smb = nullptr;
      RETURN_IF_ERR(smb_manager->Get(&smb, io.shared_memory().name()));
      RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
          trtserver.get(), smb, io.shared_memory().offset(),
          io.shared_memory().byte_size(), const_cast<void**>(&base)));
      byte_size = io.shared_memory().byte_size();
    } else if ((int)idx >= request.raw_input_size()) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected tensor data for at least " + std::to_string(idx) +
              " inputs but got only " +
              std::to_string(request.raw_input_size()) +
              " sets of data for model '" + request.model_name() + "'")
              .c_str());
    } else {
      const std::string& raw = request.raw_input(idx++);
      base = raw.c_str();
      byte_size = raw.size();
    }

    uint64_t expected_byte_size = 0;
    RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderInputBatchByteSize(
        request_provider, io.name().c_str(), &expected_byte_size));

    if (byte_size != expected_byte_size) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INVALID_ARG,
          std::string(
              "unexpected size " + std::to_string(byte_size) + " for input '" +
              io.name() + "', expecting " + std::to_string(expected_byte_size) +
              " for model '" + request.model_name() + "'")
              .c_str());
    }

    RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
        request_provider, io.name().c_str(), base, byte_size,
        TRTSERVER_MEMORY_CPU));
  }

  return nullptr;  // success
}

//
// InferHandler
//
class InferHandler : public Handler<
                         GRPCService::AsyncService,
                         grpc::ServerAsyncResponseWriter<InferResponse>,
                         InferRequest, InferResponse> {
 public:
  InferHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      GRPCService::AsyncService* service, grpc::ServerCompletionQueue* cq,
      size_t max_state_bucket_count)
      : Handler(
            name, trtserver, server_id, service, cq, max_state_bucket_count),
        smb_manager_(smb_manager)
  {
    // Create the allocator that will be used to allocate buffers for
    // the result tensors.
    FAIL_IF_ERR(
        TRTSERVER_ResponseAllocatorNew(
            &allocator_, InferResponseAlloc, InferResponseRelease),
        "creating response allocator");
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;

 private:
  static void InferComplete(
      TRTSERVER_Server* server, TRTSERVER_InferenceResponse* response,
      void* userp);

  std::shared_ptr<SharedMemoryBlockManager> smb_manager_;
  TRTSERVER_ResponseAllocator* allocator_;
};

void
InferHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestInfer(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
InferHandler::Process(Handler::State* state, bool rpc_ok)
{
  LOG_VERBOSE(1) << "Process for " << Name() << ", rpc_ok=" << rpc_ok << ", "
                 << state->unique_id_ << " step " << state->step_;

  // We need an explicit finish indicator. Can use 'state->step_'
  // because we launch an async thread that could update 'state's
  // step_ to be FINISH before this thread exits this function.
  bool finished = false;

  // If RPC failed on a new request then the server is shutting down
  // and so we should do nothing (including not registering for a new
  // request). If RPC failed on a non-START step then there is nothing
  // we can do since we one execute one step.
  const bool shutdown = (!rpc_ok && (state->step_ == Steps::START));
  if (shutdown) {
    state->step_ = Steps::FINISH;
    finished = true;
  }

  const InferRequest& request = state->request_;
  InferResponse& response = state->response_;

  if (state->step_ == Steps::START) {
    // Start a new request to replace this one...
    if (!shutdown) {
      StartNewRequest();
    }

    TRTSERVER_Error* err = nullptr;

    std::string request_header_serialized;
    if (!request.meta_data().SerializeToString(&request_header_serialized)) {
      err = TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_UNKNOWN, "failed to serialize request header");
    } else {
      // Create the inference request provider which provides all the
      // input information needed for an inference.
      TRTSERVER_InferenceRequestProvider* request_provider = nullptr;
      err = TRTSERVER_InferenceRequestProviderNew(
          &request_provider, trtserver_.get(), request.model_name().c_str(),
          request.model_version(), request_header_serialized.c_str(),
          request_header_serialized.size());
      if (err == nullptr) {
        err = InferGRPCToInput(
            trtserver_, smb_manager_, request.meta_data(), request,
            request_provider);
        if (err == nullptr) {
          err = InferAllocatorPayload(
              trtserver_, smb_manager_, request.meta_data(), response,
              &state->alloc_payload_);
          if (err == nullptr) {
            state->step_ = ISSUED;
            err = TRTSERVER_ServerInferAsync(
                trtserver_.get(), request_provider, allocator_,
                &state->alloc_payload_ /* response_allocator_userp */,
                InferComplete, reinterpret_cast<void*>(state));

            // The request provider can be deleted immediately after the
            // ServerInferAsync call returns.
            TRTSERVER_InferenceRequestProviderDelete(request_provider);
          }
        }
      }
    }

    // If not error then state->step_ == ISSUED and inference request
    // has initiated... completion callback will transition to
    // COMPLETE. If error go immediately to COMPLETE.
    if (err != nullptr) {
      RequestStatusUtil::Create(
          response.mutable_request_status(), err, state->unique_id_,
          server_id_);

      LOG_VERBOSE(1) << "Infer failed: " << TRTSERVER_ErrorMessage(err);
      TRTSERVER_ErrorDelete(err);

      // Clear the meta-data and raw output as they may be partially
      // or un-initialized.
      response.mutable_meta_data()->Clear();
      response.mutable_raw_output()->Clear();

      response.mutable_meta_data()->set_id(request.meta_data().id());

      state->step_ = COMPLETE;
      state->context_->responder_->Finish(response, grpc::Status::OK, state);
    }
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
    finished = true;
  }

  return !finished;
}

void
InferHandler::InferComplete(
    TRTSERVER_Server* server, TRTSERVER_InferenceResponse* trtserver_response,
    void* userp)
{
  State* state = reinterpret_cast<State*>(userp);

  LOG_VERBOSE(1) << "InferHandler::InferComplete, " << state->unique_id_
                 << " step " << state->step_;

  const InferRequest& request = state->request_;
  InferResponse& response = state->response_;

  TRTSERVER_Error* response_status =
      TRTSERVER_InferenceResponseStatus(trtserver_response);
  if ((response_status == nullptr) && (response.ByteSizeLong() > INT_MAX)) {
    response_status = TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string(
            "Response has byte size " +
            std::to_string(response.ByteSizeLong()) +
            " which exceeds gRPC's byte size limit " + std::to_string(INT_MAX) +
            ".")
            .c_str());
  }

  if (response_status == nullptr) {
    TRTSERVER_Protobuf* response_protobuf = nullptr;
    response_status = TRTSERVER_InferenceResponseHeader(
        trtserver_response, &response_protobuf);
    if (response_status == nullptr) {
      const char* buffer;
      size_t byte_size;
      response_status =
          TRTSERVER_ProtobufSerialize(response_protobuf, &buffer, &byte_size);
      if (response_status == nullptr) {
        if (!response.mutable_meta_data()->ParseFromArray(buffer, byte_size)) {
          response_status = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL, "failed to parse response header");
        }
      }

      TRTSERVER_ProtobufDelete(response_protobuf);
    }
  }

  // If the response is an error then clear the meta-data and raw
  // output as they may be partially or un-initialized.
  if (response_status != nullptr) {
    response.mutable_meta_data()->Clear();
    response.mutable_raw_output()->Clear();
  }

  RequestStatusUtil::Create(
      response.mutable_request_status(), response_status, state->unique_id_,
      state->context_->server_id_);

  LOG_IF_ERR(
      TRTSERVER_InferenceResponseDelete(trtserver_response),
      "deleting GRPC response");
  TRTSERVER_ErrorDelete(response_status);

  response.mutable_meta_data()->set_id(request.meta_data().id());

  state->step_ = COMPLETE;
  state->context_->responder_->Finish(response, grpc::Status::OK, state);
}

//
// StreamInferHandler
//
class StreamInferHandler
    : public Handler<
          GRPCService::AsyncService,
          grpc::ServerAsyncReaderWriter<InferResponse, InferRequest>,
          InferRequest, InferResponse> {
 public:
  StreamInferHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      GRPCService::AsyncService* service, grpc::ServerCompletionQueue* cq,
      size_t max_state_bucket_count)
      : Handler(
            name, trtserver, server_id, service, cq, max_state_bucket_count),
        smb_manager_(smb_manager)
  {
    // Create the allocator that will be used to allocate buffers for
    // the result tensors.
    FAIL_IF_ERR(
        TRTSERVER_ResponseAllocatorNew(
            &allocator_, InferResponseAlloc, InferResponseRelease),
        "creating response allocator");
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;

 private:
  static void StreamInferComplete(
      TRTSERVER_Server* server, TRTSERVER_InferenceResponse* response,
      void* userp);
  static void CompleteResponse(Handler::State* state);

  std::shared_ptr<SharedMemoryBlockManager> smb_manager_;
  TRTSERVER_ResponseAllocator* allocator_;
};

void
StreamInferHandler::StartNewRequest()
{
  const uint64_t unique_id = RequestStatusUtil::NextUniqueRequestId();
  auto context = std::make_shared<State::Context>(server_id_, unique_id);
  State* state = StateNew(context);
  service_->RequestStreamInfer(
      state->context_->ctx_.get(), state->context_->responder_.get(), cq_, cq_,
      state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
StreamInferHandler::Process(Handler::State* state, bool rpc_ok)
{
  LOG_VERBOSE(1) << "Process for " << Name() << ", rpc_ok=" << rpc_ok
                 << ", context " << state->context_->unique_id_ << ", "
                 << state->unique_id_ << " step " << state->step_;

  // We need an explicit finish indicator. Can use 'state->step_'
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

    // Since this is the start of a connection, 'state' hasn't been
    // used yet so use it to read a request off the connection.
    state->context_->step_ = Steps::READ;
    state->step_ = Steps::READ;
    state->context_->responder_->Read(&state->request_, state);

  } else if (state->step_ == Steps::READ) {
    // If done reading and no in-flight requests then can finish the
    // entire stream. Otherwise just finish this state.
    if (!rpc_ok) {
      state->context_->step_ = Steps::WRITEREADY;
      if (state->context_->IsRequestsCompleted()) {
        state->context_->step_ = Steps::COMPLETE;
        state->step_ = Steps::COMPLETE;
        state->context_->responder_->Finish(
            state->context_->finish_ok_ ? grpc::Status::OK
                                        : grpc::Status::CANCELLED,
            state);
      } else {
        state->step_ = Steps::FINISH;
        finished = true;
      }

      return !finished;
    }

    // Request has been successfully read so put it in the context
    // queue so that it's response is sent in the same order as the
    // request was received.
    state->context_->EnqueueForResponse(state);

    // Need to get context here as it is needed below. 'state' can
    // complete inference, write response, and finish (which releases
    // context) before we make any forward progress.... so need to
    // hold onto context here while we know it is good.
    std::shared_ptr<StateContext> context = state->context_;

    // Issue the inference request into server...
    const InferRequest& request = state->request_;
    InferResponse& response = state->response_;

    TRTSERVER_Error* err = nullptr;
    std::string request_header_serialized;
    if (!request.meta_data().SerializeToString(&request_header_serialized)) {
      err = TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_UNKNOWN, "failed to serialize request header");
    } else {
      // Create the inference request provider which provides all the
      // input information needed for an inference.
      TRTSERVER_InferenceRequestProvider* request_provider = nullptr;
      err = TRTSERVER_InferenceRequestProviderNew(
          &request_provider, trtserver_.get(), request.model_name().c_str(),
          request.model_version(), request_header_serialized.c_str(),
          request_header_serialized.size());
      if (err == nullptr) {
        err = InferGRPCToInput(
            trtserver_, smb_manager_, request.meta_data(), request,
            request_provider);
        if (err == nullptr) {
          err = InferAllocatorPayload(
              trtserver_, smb_manager_, request.meta_data(), response,
              &state->alloc_payload_);
          if (err == nullptr) {
            state->step_ = ISSUED;
            err = TRTSERVER_ServerInferAsync(
                trtserver_.get(), request_provider, allocator_,
                &state->alloc_payload_ /* response_allocator_userp */,
                StreamInferComplete, reinterpret_cast<void*>(state));

            // The request provider can be deleted immediately after the
            // ServerInferAsync call returns.
            TRTSERVER_InferenceRequestProviderDelete(request_provider);
          }
        }
      }
    }

    // If there was not an error in issuing the 'state' request then
    // state->step_ == ISSUED and inference request has
    // initiated... the completion callback will transition to
    // WRITEREADY or WRITTEN. If there was an error then enqueue the
    // error response and show it to be ready for writing.
    if (err != nullptr) {
      RequestStatusUtil::Create(
          response.mutable_request_status(), err, state->unique_id_,
          server_id_);

      LOG_VERBOSE(1) << "Infer failed: " << TRTSERVER_ErrorMessage(err);
      TRTSERVER_ErrorDelete(err);

      // Clear the meta-data and raw output as they may be partially
      // or un-initialized.
      response.mutable_meta_data()->Clear();
      response.mutable_raw_output()->Clear();

      response.mutable_meta_data()->set_id(request.meta_data().id());

      state->step_ = Steps::WRITEREADY;
      CompleteResponse(state);
    }

    // Now that the inference request is in flight, create a copy of
    // 'state' and use it to attempt another read from the connection
    // (i.e the next request in the stream).
    State* next_read_state = StateNew(context, Steps::READ);
    next_read_state->context_->responder_->Read(
        &next_read_state->request_, next_read_state);

  } else if (state->step_ == Steps::WRITTEN) {
    // Log an error and cancel the stream if write failed or 'state'
    // is not the expected next response. We don't necessarily cancel
    // right away. Need to wait for any pending reads, inferences and
    // writes to complete.
    if (!rpc_ok || !state->context_->PopCompletedResponse(state)) {
      LOG_ERROR << "Process for " << Name() << ", rpc_ok=" << rpc_ok
                << ", context " << state->context_->unique_id_ << ", "
                << state->unique_id_ << " step " << state->step_
                << ", failed or unexpected to be response";
      state->context_->finish_ok_ = false;
    }

    // Write the next response if it is ready...
    auto next_state = state->context_->ReadyResponse();
    if (next_state != nullptr) {
      next_state->step_ = Steps::WRITTEN;
      next_state->context_->responder_->Write(
          next_state->response_, next_state);
    }

    // If done reading and no in-flight requests then can finish the
    // entire stream. Otherwise just finish this state.
    if (state->context_->IsRequestsCompleted()) {
      state->context_->step_ = Steps::COMPLETE;
      state->step_ = Steps::COMPLETE;
      state->context_->responder_->Finish(
          state->context_->finish_ok_ ? grpc::Status::OK
                                      : grpc::Status::CANCELLED,
          state);
    } else {
      state->step_ = Steps::FINISH;
      finished = true;
    }

  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
    finished = true;
  }

  return !finished;
}

void
StreamInferHandler::StreamInferComplete(
    TRTSERVER_Server* server, TRTSERVER_InferenceResponse* trtserver_response,
    void* userp)
{
  State* state = reinterpret_cast<State*>(userp);

  LOG_VERBOSE(1) << "StreamInferHandler::StreamInferComplete, context "
                 << state->context_->unique_id_ << ", " << state->unique_id_
                 << " step " << state->step_;

  const InferRequest& request = state->request_;
  InferResponse& response = state->response_;

  TRTSERVER_Error* response_status =
      TRTSERVER_InferenceResponseStatus(trtserver_response);
  if ((response_status == nullptr) && (response.ByteSizeLong() > INT_MAX)) {
    response_status = TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string(
            "Response has byte size " +
            std::to_string(response.ByteSizeLong()) +
            " which exceeds gRPC's byte size limit " + std::to_string(INT_MAX) +
            ".")
            .c_str());
  }

  if (response_status == nullptr) {
    TRTSERVER_Protobuf* response_protobuf = nullptr;
    response_status = TRTSERVER_InferenceResponseHeader(
        trtserver_response, &response_protobuf);
    if (response_status == nullptr) {
      const char* buffer;
      size_t byte_size;
      response_status =
          TRTSERVER_ProtobufSerialize(response_protobuf, &buffer, &byte_size);
      if (response_status == nullptr) {
        if (!response.mutable_meta_data()->ParseFromArray(buffer, byte_size)) {
          response_status = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL, "failed to parse response header");
        }
      }

      TRTSERVER_ProtobufDelete(response_protobuf);
    }
  }

  // If the response is an error then clear the meta-data and raw
  // output as they may be partially or un-initialized.
  if (response_status != nullptr) {
    response.mutable_meta_data()->Clear();
    response.mutable_raw_output()->Clear();
  }

  RequestStatusUtil::Create(
      response.mutable_request_status(), response_status, state->unique_id_,
      state->context_->server_id_);

  LOG_IF_ERR(
      TRTSERVER_InferenceResponseDelete(trtserver_response),
      "deleting GRPC response");
  TRTSERVER_ErrorDelete(response_status);

  response.mutable_meta_data()->set_id(request.meta_data().id());

  state->step_ = Steps::WRITEREADY;
  CompleteResponse(state);
}

void
StreamInferHandler::CompleteResponse(Handler::State* state)
{
  // If 'state' is at the front of the queued states then it's
  // response should be sent next... so go ahead and send it. If
  // 'state' is not the next to be sent then do nothing. When the
  // front state is eventually written it will trigger a write of the
  // next state in the queue.
  if (state == state->context_->ReadyResponse()) {
    state->step_ = Steps::WRITTEN;
    state->context_->responder_->Write(state->response_, state);
  }
}

//
// ProfileHandler
//
class ProfileHandler : public Handler<
                           GRPCService::AsyncService,
                           grpc::ServerAsyncResponseWriter<ProfileResponse>,
                           ProfileRequest, ProfileResponse> {
 public:
  ProfileHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCService::AsyncService* service, grpc::ServerCompletionQueue* cq,
      size_t max_state_bucket_count)
      : Handler(name, trtserver, server_id, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ProfileHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestProfile(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ProfileHandler::Process(Handler::State* state, bool rpc_ok)
{
  LOG_VERBOSE(1) << "Process for " << Name() << ", rpc_ok=" << rpc_ok << ", "
                 << state->unique_id_ << " step " << state->step_;

  // If RPC failed on a new request then the server is shutting down
  // and so we should do nothing (including not registering for a new
  // request). If RPC failed on a non-START step then there is nothing
  // we can do since we one execute one step.
  const bool shutdown = (!rpc_ok && (state->step_ == Steps::START));
  if (shutdown) {
    state->step_ = Steps::FINISH;
  }

  ProfileResponse& response = state->response_;

  if (state->step_ == START) {
    // For now profile is a nop...

    RequestStatusUtil::Create(
        response.mutable_request_status(), nullptr /* success */,
        state->unique_id_, server_id_);

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(response, grpc::Status::OK, state);
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
  }

  // Only handle one status request at a time (to avoid having status
  // request cause too much load on server), so register for next
  // request only after this one finished.
  if (!shutdown && (state->step_ == Steps::FINISH)) {
    StartNewRequest();
  }

  return state->step_ != Steps::FINISH;
}

//
// ModelControlHandler
//
class ModelControlHandler
    : public Handler<
          GRPCService::AsyncService,
          grpc::ServerAsyncResponseWriter<ModelControlResponse>,
          ModelControlRequest, ModelControlResponse> {
 public:
  ModelControlHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCService::AsyncService* service, grpc::ServerCompletionQueue* cq,
      size_t max_state_bucket_count)
      : Handler(name, trtserver, server_id, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ModelControlHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestModelControl(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ModelControlHandler::Process(Handler::State* state, bool rpc_ok)
{
  LOG_VERBOSE(1) << "Process for " << Name() << ", rpc_ok=" << rpc_ok << ", "
                 << state->unique_id_ << " step " << state->step_;

  // If RPC failed on a new request then the server is shutting down
  // and so we should do nothing (including not registering for a new
  // request). If RPC failed on a non-START step then there is nothing
  // we can do since we one execute one step.
  const bool shutdown = (!rpc_ok && (state->step_ == Steps::START));
  if (shutdown) {
    state->step_ = Steps::FINISH;
  }

  const ModelControlRequest& request = state->request_;
  ModelControlResponse& response = state->response_;

  if (state->step_ == START) {
    TRTSERVER_Error* err = nullptr;
    if (request.type() == ModelControlRequest::LOAD) {
      err = TRTSERVER_ServerLoadModel(
          trtserver_.get(), request.model_name().c_str());
    } else {
      err = TRTSERVER_ServerUnloadModel(
          trtserver_.get(), request.model_name().c_str());
    }

    RequestStatusUtil::Create(
        response.mutable_request_status(), err, state->unique_id_, server_id_);

    TRTSERVER_ErrorDelete(err);

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(response, grpc::Status::OK, state);
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
  }

  // Only handle one status request at a time (to avoid having status
  // request cause too much load on server), so register for next
  // request only after this one finished.
  if (!shutdown && (state->step_ == Steps::FINISH)) {
    StartNewRequest();
  }

  return state->step_ != Steps::FINISH;
}

//
// SharedMemoryControlHandler
//
class SharedMemoryControlHandler
    : public Handler<
          GRPCService::AsyncService,
          grpc::ServerAsyncResponseWriter<SharedMemoryControlResponse>,
          SharedMemoryControlRequest, SharedMemoryControlResponse> {
 public:
  SharedMemoryControlHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      GRPCService::AsyncService* service, grpc::ServerCompletionQueue* cq,
      size_t max_state_bucket_count)
      : Handler(
            name, trtserver, server_id, service, cq, max_state_bucket_count),
        smb_manager_(smb_manager)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;

 private:
  std::shared_ptr<SharedMemoryBlockManager> smb_manager_;
};

void
SharedMemoryControlHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestSharedMemoryControl(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
SharedMemoryControlHandler::Process(Handler::State* state, bool rpc_ok)
{
  LOG_VERBOSE(1) << "Process for " << Name() << ", rpc_ok=" << rpc_ok << ", "
                 << state->unique_id_ << " step " << state->step_;

  // If RPC failed on a new request then the server is shutting down
  // and so we should do nothing (including not registering for a new
  // request). If RPC failed on a non-START step then there is nothing
  // we can do since we one execute one step.
  const bool shutdown = (!rpc_ok && (state->step_ == Steps::START));
  if (shutdown) {
    state->step_ = Steps::FINISH;
  }

  const SharedMemoryControlRequest& request = state->request_;
  SharedMemoryControlResponse& response = state->response_;

  if (state->step_ == START) {
    TRTSERVER_SharedMemoryBlock* smb = nullptr;

    TRTSERVER_Error* err = nullptr;
    switch (request.type()) {
      case SharedMemoryControlRequest::REGISTER:
        err = smb_manager_->Create(
            &smb, request.shared_memory_region().name(),
            request.shared_memory_region().shm_key(),
            request.shared_memory_region().offset(),
            request.shared_memory_region().byte_size());
        if (err == nullptr) {
          err = TRTSERVER_ServerRegisterSharedMemory(trtserver_.get(), smb);
        }
        break;
      case SharedMemoryControlRequest::UNREGISTER:
        err = smb_manager_->Remove(&smb, request.shared_memory_region().name());
        if ((err == nullptr) && (smb != nullptr)) {
          err = TRTSERVER_ServerUnregisterSharedMemory(trtserver_.get(), smb);
          TRTSERVER_Error* del_err = TRTSERVER_SharedMemoryBlockDelete(smb);
          if (del_err != nullptr) {
            LOG_ERROR << "failed to delete shared memory block: "
                      << TRTSERVER_ErrorMessage(del_err);
          }
        }
        break;
      case SharedMemoryControlRequest::UNREGISTER_ALL:
        err = smb_manager_->Clear();
        if (err == nullptr) {
          err = TRTSERVER_ServerUnregisterAllSharedMemory(trtserver_.get());
        }
        break;
      default:
        err = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_UNKNOWN,
            "unknown sharedmemorycontrol request type");
        break;
    }

    RequestStatusUtil::Create(
        response.mutable_request_status(), err, state->unique_id_, server_id_);

    TRTSERVER_ErrorDelete(err);

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(response, grpc::Status::OK, state);
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
  }

  // Only handle one status request at a time (to avoid having status
  // request cause too much load on server), so register for next
  // request only after this one finished.
  if (!shutdown && (state->step_ == Steps::FINISH)) {
    StartNewRequest();
  }

  return state->step_ != Steps::FINISH;
}

}  // namespace

//
// GRPCServer
//
GRPCServer::GRPCServer(
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
    const char* server_id, const std::string& server_addr,
    const int infer_thread_cnt, const int stream_infer_thread_cnt,
    const int infer_allocation_pool_size)
    : server_(server), smb_manager_(smb_manager), server_id_(server_id),
      server_addr_(server_addr), infer_thread_cnt_(infer_thread_cnt),
      stream_infer_thread_cnt_(stream_infer_thread_cnt),
      infer_allocation_pool_size_(infer_allocation_pool_size), running_(false)
{
}

GRPCServer::~GRPCServer()
{
  Stop();
}

TRTSERVER_Error*
GRPCServer::Create(
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager, int32_t port,
    int infer_thread_cnt, int stream_infer_thread_cnt,
    int infer_allocation_pool_size, std::unique_ptr<GRPCServer>* grpc_server)
{
  const char* server_id = nullptr;
  TRTSERVER_Error* err = TRTSERVER_ServerId(server.get(), &server_id);
  if (err != nullptr) {
    server_id = "unknown:0";
    TRTSERVER_ErrorDelete(err);
  }

  const std::string addr = "0.0.0.0:" + std::to_string(port);
  grpc_server->reset(new GRPCServer(
      server, smb_manager, server_id, addr, infer_thread_cnt,
      stream_infer_thread_cnt, infer_allocation_pool_size));

  return nullptr;  // success
}

TRTSERVER_Error*
GRPCServer::Start()
{
  if (running_) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_ALREADY_EXISTS, "GRPC server is already running.");
  }

  grpc_builder_.AddListeningPort(
      server_addr_, grpc::InsecureServerCredentials());
  grpc_builder_.SetMaxMessageSize(MAX_GRPC_MESSAGE_SIZE);
  grpc_builder_.RegisterService(&service_);
  health_cq_ = grpc_builder_.AddCompletionQueue();
  status_cq_ = grpc_builder_.AddCompletionQueue();
  infer_cq_ = grpc_builder_.AddCompletionQueue();
  stream_infer_cq_ = grpc_builder_.AddCompletionQueue();
  profile_cq_ = grpc_builder_.AddCompletionQueue();
  modelcontrol_cq_ = grpc_builder_.AddCompletionQueue();
  shmcontrol_cq_ = grpc_builder_.AddCompletionQueue();
  grpc_server_ = grpc_builder_.BuildAndStart();

  // Handler for health requests. A single thread processes all of
  // these requests.
  HealthHandler* hhealth = new HealthHandler(
      "HealthHandler", server_, server_id_, &service_, health_cq_.get(),
      2 /* max_state_bucket_count */);
  hhealth->Start(1 /* thread_cnt */);
  health_handler_.reset(hhealth);

  // Handler for status requests. A single thread processes all of
  // these requests.
  StatusHandler* hstatus = new StatusHandler(
      "StatusHandler", server_, server_id_, &service_, status_cq_.get(),
      2 /* max_state_bucket_count */);
  hstatus->Start(1 /* thread_cnt */);
  status_handler_.reset(hstatus);

  // Handler for inference requests.
  InferHandler* hinfer = new InferHandler(
      "InferHandler", server_, server_id_, smb_manager_, &service_,
      infer_cq_.get(),
      infer_allocation_pool_size_ /* max_state_bucket_count */);
  hinfer->Start(infer_thread_cnt_);
  infer_handler_.reset(hinfer);

  // Handler for streaming inference requests.
  StreamInferHandler* hstreaminfer = new StreamInferHandler(
      "StreamInferHandler", server_, server_id_, smb_manager_, &service_,
      stream_infer_cq_.get(),
      infer_allocation_pool_size_ /* max_state_bucket_count */);
  hstreaminfer->Start(stream_infer_thread_cnt_);
  stream_infer_handler_.reset(hstreaminfer);

  // Handler for profile requests. A single thread processes all of
  // these requests.
  ProfileHandler* hprofile = new ProfileHandler(
      "ProfileHandler", server_, server_id_, &service_, profile_cq_.get(),
      2 /* max_state_bucket_count */);
  hprofile->Start(1 /* thread_cnt */);
  profile_handler_.reset(hprofile);

  // Handler for model-control requests. A single thread processes all
  // of these requests.
  ModelControlHandler* hmodelcontrol = new ModelControlHandler(
      "ModelControlHandler", server_, server_id_, &service_,
      modelcontrol_cq_.get(), 2 /* max_state_bucket_count */);
  hmodelcontrol->Start(1 /* thread_cnt */);
  modelcontrol_handler_.reset(hmodelcontrol);

  // Handler for shared-memory-control requests. A single thread
  // processes all of these requests.
  SharedMemoryControlHandler* hshmcontrol = new SharedMemoryControlHandler(
      "SharedMemoryControlHandler", server_, server_id_, smb_manager_,
      &service_, shmcontrol_cq_.get(), 2 /* max_state_bucket_count */);
  hshmcontrol->Start(1 /* thread_cnt */);
  shmcontrol_handler_.reset(hshmcontrol);

  running_ = true;
  LOG_INFO << "Started GRPCService at " << server_addr_;
  return nullptr;  // success
}

TRTSERVER_Error*
GRPCServer::Stop()
{
  if (!running_) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_UNAVAILABLE, "GRPC server is not running.");
  }

  // Always shutdown the completion queue after the server.
  grpc_server_->Shutdown();

  health_cq_->Shutdown();
  status_cq_->Shutdown();
  infer_cq_->Shutdown();
  stream_infer_cq_->Shutdown();
  profile_cq_->Shutdown();
  modelcontrol_cq_->Shutdown();
  shmcontrol_cq_->Shutdown();

  // Must stop all handlers explicitly to wait for all the handler
  // threads to join since they are referencing completion queue, etc.
  dynamic_cast<HealthHandler*>(health_handler_.get())->Stop();
  dynamic_cast<StatusHandler*>(status_handler_.get())->Stop();
  dynamic_cast<InferHandler*>(infer_handler_.get())->Stop();
  dynamic_cast<StreamInferHandler*>(stream_infer_handler_.get())->Stop();
  dynamic_cast<ProfileHandler*>(profile_handler_.get())->Stop();
  dynamic_cast<ModelControlHandler*>(modelcontrol_handler_.get())->Stop();
  dynamic_cast<SharedMemoryControlHandler*>(shmcontrol_handler_.get())->Stop();

  running_ = false;
  return nullptr;  // success
}

}}  // namespace nvidia::inferenceserver
