// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/servers/grpc_server_v2.h"

#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include "grpc++/grpc++.h"
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/server_status.pb.h"
#include "src/core/trtserver.h"
#include "src/servers/common.h"

#ifdef TRTIS_ENABLE_TRACING
#include "src/servers/tracer.h"
#endif  // TRTIS_ENABLE_TRACING

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

//
// GrpcStatusUtil
//
class GrpcStatusUtil {
 public:
  static void Create(grpc::Status* status, TRTSERVER_Error* err);
  static grpc::StatusCode CodeToStatus(TRTSERVER_Error_Code code);
};

void
GrpcStatusUtil::Create(grpc::Status* status, TRTSERVER_Error* err)
{
  if (err == nullptr) {
    *status = grpc::Status::OK;
  } else {
    *status = grpc::Status(
        GrpcStatusUtil::CodeToStatus(TRTSERVER_ErrorCode(err)),
        TRTSERVER_ErrorMessage(err));
  }
}

grpc::StatusCode
GrpcStatusUtil::CodeToStatus(TRTSERVER_Error_Code code)
{
  // GRPC status codes:
  // https://github.com/grpc/grpc/blob/master/include/grpc/impl/codegen/status.h
  switch (code) {
    case TRTSERVER_ERROR_UNKNOWN:
      return grpc::StatusCode::UNKNOWN;
    case TRTSERVER_ERROR_INTERNAL:
      return grpc::StatusCode::INTERNAL;
    case TRTSERVER_ERROR_NOT_FOUND:
      return grpc::StatusCode::NOT_FOUND;
    case TRTSERVER_ERROR_INVALID_ARG:
      return grpc::StatusCode::INVALID_ARGUMENT;
    case TRTSERVER_ERROR_UNAVAILABLE:
      return grpc::StatusCode::UNAVAILABLE;
    case TRTSERVER_ERROR_UNSUPPORTED:
      return grpc::StatusCode::UNIMPLEMENTED;
    case TRTSERVER_ERROR_ALREADY_EXISTS:
      return grpc::StatusCode::ALREADY_EXISTS;
  }

  return grpc::StatusCode::UNKNOWN;
}

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
    TRTSERVER_Memory_Type memory_type_;
    int64_t device_id_;
  };

  using TensorShmMap = std::unordered_map<std::string, ShmInfo>;
  using TensorSerializedDataMap =
      std::unordered_map<std::string, std::shared_ptr<std::string>>;

  explicit AllocPayload()
      : response_(nullptr), shm_map_(nullptr), serialized_data_map_(nullptr)
  {
  }
  ~AllocPayload()
  {
    // Don't delete 'response_'.. it is owned by the HandlerState
    delete shm_map_;
    delete serialized_data_map_;
  }

  ModelInferResponse* response_;
  TensorShmMap* shm_map_;
  // Used to extend the lifetime of the serialized data in case
  // repeated byte contents were provided in the request.
  TensorSerializedDataMap* serialized_data_map_;
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
      std::lock_guard<std::mutex> lock(mu_);
      states_.push(state);
    }

    // Check the state at the front of the queue and write it if
    // ready. The state at the front of the queue is ready if it is in
    // the WRITEREADY state and it equals 'required_state' (or
    // 'required_state' is nullptr). Return nullptr if front of queue
    // was not ready (and so not written), or return the state if it
    // was ready and written.
    HandlerStateType* WriteResponseIfReady(HandlerStateType* required_state)
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (states_.empty()) {
        return nullptr;
      }

      HandlerStateType* state = states_.front();
      if (state->step_ != Steps::WRITEREADY) {
        return nullptr;
      }

      if ((required_state != nullptr) && (state != required_state)) {
        return nullptr;
      }

#ifdef TRTIS_ENABLE_TRACING
      if (state->trace_meta_data_ != nullptr) {
        state->trace_meta_data_->tracer_->CaptureTimestamp(
            TRTSERVER_TRACE_LEVEL_MIN, "grpc send start");
      }
#endif  // TRTIS_ENABLE_TRACING

      state->step_ = Steps::WRITTEN;
      responder_->Write(state->response_, state);

      return state;
    }

    // If 'state' is at the front of the queue and written, pop it and
    // return true. Other return false.
    bool PopCompletedResponse(HandlerStateType* state)
    {
      std::lock_guard<std::mutex> lock(mu_);
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
      std::lock_guard<std::mutex> lock(mu_);
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

  void Release() { context_ = nullptr; }

  std::shared_ptr<Context> context_;

  uint64_t unique_id_;
  Steps step_;

#ifdef TRTIS_ENABLE_TRACING
  std::unique_ptr<TraceMetaData> trace_meta_data_;
#endif  // TRTIS_ENABLE_TRACING

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
class Handler : public GRPCServerV2::HandlerBase {
 public:
  Handler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      ServiceType* service, grpc::ServerCompletionQueue* cq,
      size_t max_state_bucket_count);
  virtual ~Handler();

  // Descriptive name of of the handler.
  const std::string& Name() const { return name_; }

  // Start handling requests.
  void Start();

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

    if (max_state_bucket_count_ > 0) {
      std::lock_guard<std::mutex> lock(alloc_mu_);

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
    if (max_state_bucket_count_ > 0) {
      std::lock_guard<std::mutex> lock(alloc_mu_);

      if (state_bucket_.size() < max_state_bucket_count_) {
        state->Release();
        state_bucket_.push_back(state);
        return;
      }
    }

    delete state;
  }

  virtual void StartNewRequest() = 0;
  virtual bool Process(State* state, bool rpc_ok) = 0;

  const std::string name_;
  std::shared_ptr<TRTSERVER_Server> trtserver_;
  const char* const server_id_;

  ServiceType* service_;
  grpc::ServerCompletionQueue* cq_;
  std::unique_ptr<std::thread> thread_;

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
Handler<ServiceType, ServerResponderType, RequestType, ResponseType>::Start()
{
  // Use a barrier to make sure we don't return until thread has
  // started.
  auto barrier = std::make_shared<Barrier>(2);

  thread_.reset(new std::thread([this, barrier] {
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

  barrier->Wait();
  LOG_VERBOSE(1) << "Thread started for " << Name();
}

template <
    typename ServiceType, typename ServerResponderType, typename RequestType,
    typename ResponseType>
void
Handler<ServiceType, ServerResponderType, RequestType, ResponseType>::Stop()
{
  if (thread_->joinable()) {
    thread_->join();
  }

  LOG_VERBOSE(1) << "Thread exited for " << Name();
}

//
// ServerLiveHandler
//
class ServerLiveHandler
    : public Handler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncResponseWriter<ServerLiveResponse>,
          ServerLiveRequest, ServerLiveResponse> {
 public:
  ServerLiveHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, trtserver, server_id, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ServerLiveHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestServerLive(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ServerLiveHandler::Process(Handler::State* state, bool rpc_ok)
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

  ServerLiveResponse& response = state->response_;

  if (state->step_ == Steps::START) {
    bool live = false;
    TRTSERVER_Error* err = TRTSERVER_ServerIsLive(trtserver_.get(), &live);

    response.set_live((err == nullptr) && live);

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRTSERVER_ErrorDelete(err);

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(response, status, state);
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
  }

  // Only handle one request at a time (to avoid having request cause
  // too much load on server), so register for next request only after
  // this one finished.
  if (!shutdown && (state->step_ == Steps::FINISH)) {
    StartNewRequest();
  }

  return state->step_ != Steps::FINISH;
}

//
// ServerReadyHandler
//
class ServerReadyHandler
    : public Handler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncResponseWriter<ServerReadyResponse>,
          ServerReadyRequest, ServerReadyResponse> {
 public:
  ServerReadyHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, trtserver, server_id, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ServerReadyHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestServerReady(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ServerReadyHandler::Process(Handler::State* state, bool rpc_ok)
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

  ServerReadyResponse& response = state->response_;

  if (state->step_ == Steps::START) {
    bool ready = false;
    TRTSERVER_Error* err = TRTSERVER_ServerIsReady(trtserver_.get(), &ready);

    response.set_ready((err == nullptr) && ready);

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRTSERVER_ErrorDelete(err);

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(response, status, state);
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
  }

  // Only handle one request at a time (to avoid having request cause
  // too much load on server), so register for next request only after
  // this one finished.
  if (!shutdown && (state->step_ == Steps::FINISH)) {
    StartNewRequest();
  }

  return state->step_ != Steps::FINISH;
}

//
// ModelReadyHandler
//
class ModelReadyHandler
    : public Handler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncResponseWriter<ModelReadyResponse>,
          ModelReadyRequest, ModelReadyResponse> {
 public:
  ModelReadyHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, trtserver, server_id, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ModelReadyHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestModelReady(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ModelReadyHandler::Process(Handler::State* state, bool rpc_ok)
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

  const ModelReadyRequest& request = state->request_;
  ModelReadyResponse& response = state->response_;

  if (state->step_ == Steps::START) {
    ServerStatus server_status;

    TRTSERVER_Protobuf* model_status_protobuf = nullptr;
    TRTSERVER_Error* err = TRTSERVER_ServerModelStatus(
        trtserver_.get(), request.name().c_str(), &model_status_protobuf);
    if (err == nullptr) {
      const char* status_buffer;
      size_t status_byte_size;
      err = TRTSERVER_ProtobufSerialize(
          model_status_protobuf, &status_buffer, &status_byte_size);
      if (err == nullptr) {
        if (!server_status.ParseFromArray(status_buffer, status_byte_size)) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
        }
      }
    }

    TRTSERVER_ProtobufDelete(model_status_protobuf);

    bool ready = false;
    if (err == nullptr) {
      const auto& nitr = server_status.model_status().find(request.name());
      if (nitr == server_status.model_status().end()) {
        err = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            std::string(
                "no status available for unknown model '" + request.name() +
                "'")
                .c_str());
      } else {
        const ModelStatus& model_status = nitr->second;

        // If requested version is -1 then find the highest valued
        // version.
        int64_t requested_version = request.version();
        if (requested_version == -1) {
          for (const auto& pr : model_status.version_status()) {
            requested_version = std::max(requested_version, pr.first);
          }
        }

        const auto& vitr =
            model_status.version_status().find(requested_version);
        if (vitr == model_status.version_status().end()) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INVALID_ARG,
              std::string(
                  "no status available for model '" + request.name() +
                  "', version " + std::to_string(request.version()))
                  .c_str());
        } else {
          const ModelVersionStatus& version_status = vitr->second;
          ready = version_status.ready_state() == ModelReadyState::MODEL_READY;
        }
      }
    }

    response.set_ready(ready);

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRTSERVER_ErrorDelete(err);

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(response, status, state);
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
  }

  // Only handle one request at a time (to avoid having request cause
  // too much load on server), so register for next request only after
  // this one finished.
  if (!shutdown && (state->step_ == Steps::FINISH)) {
    StartNewRequest();
  }

  return state->step_ != Steps::FINISH;
}

//
// ServerMetadataHandler
//
class ServerMetadataHandler
    : public Handler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncResponseWriter<ServerMetadataResponse>,
          ServerMetadataRequest, ServerMetadataResponse> {
 public:
  ServerMetadataHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, trtserver, server_id, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ServerMetadataHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestServerMetadata(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ServerMetadataHandler::Process(Handler::State* state, bool rpc_ok)
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

  ServerMetadataResponse& response = state->response_;

  if (state->step_ == Steps::START) {
    const char* name = nullptr;
    TRTSERVER_Error* err = TRTSERVER_ServerId(trtserver_.get(), &name);
    if (err == nullptr) {
      response.set_name(name);

      const char* version = nullptr;
      err = TRTSERVER_ServerVersion(trtserver_.get(), &version);
      if (err == nullptr) {
        response.set_version(version);

        uint64_t extensions_count;
        const char* const* extensions;
        err = TRTSERVER_ServerExtensions(
            trtserver_.get(), &extensions, &extensions_count);
        if (err == nullptr) {
          for (uint64_t i = 0; i < extensions_count; ++i) {
            response.add_extensions(extensions[i]);
          }
        }
      }
    }

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRTSERVER_ErrorDelete(err);

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(response, status, state);
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
  }

  // Only handle one request at a time (to avoid having request cause
  // too much load on server), so register for next request only after
  // this one finished.
  if (!shutdown && (state->step_ == Steps::FINISH)) {
    StartNewRequest();
  }

  return state->step_ != Steps::FINISH;
}

//
// ModelMetadataHandler
//
class ModelMetadataHandler
    : public Handler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncResponseWriter<ModelMetadataResponse>,
          ModelMetadataRequest, ModelMetadataResponse> {
 public:
  ModelMetadataHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, trtserver, server_id, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ModelMetadataHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestModelMetadata(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ModelMetadataHandler::Process(Handler::State* state, bool rpc_ok)
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

  const ModelMetadataRequest& request = state->request_;
  ModelMetadataResponse& response = state->response_;

  if (state->step_ == Steps::START) {
    ServerStatus server_status;

    TRTSERVER_Protobuf* model_status_protobuf = nullptr;
    TRTSERVER_Error* err = TRTSERVER_ServerModelStatus(
        trtserver_.get(), request.name().c_str(), &model_status_protobuf);
    if (err == nullptr) {
      const char* status_buffer;
      size_t status_byte_size;
      err = TRTSERVER_ProtobufSerialize(
          model_status_protobuf, &status_buffer, &status_byte_size);
      if (err == nullptr) {
        if (!server_status.ParseFromArray(status_buffer, status_byte_size)) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
        }
      }
    }

    TRTSERVER_ProtobufDelete(model_status_protobuf);

    if (err == nullptr) {
      const auto& nitr = server_status.model_status().find(request.name());
      if (nitr == server_status.model_status().end()) {
        err = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            std::string(
                "no metadata available for unknown model '" + request.name() +
                "'")
                .c_str());
      } else {
        // All models share the same metadata across versions so we
        // ignore request.version().
        const ModelStatus& model_status = nitr->second;
        const ModelConfig& model_config = model_status.config();
        response.set_name(model_config.name());
        response.set_platform(model_config.platform());
        for (const auto& pr : model_status.version_status()) {
          response.add_versions(pr.first);
        }

        for (const auto& io : model_config.input()) {
          ModelMetadataResponse::TensorMetadata* input = response.add_inputs();
          input->set_name(io.name());
          input->set_datatype(GetDataTypeProtocolString(io.data_type()));
          for (const auto d : io.dims()) {
            input->add_shape(d);
          }
        }

        for (const auto& io : model_config.output()) {
          ModelMetadataResponse::TensorMetadata* output =
              response.add_outputs();
          output->set_name(io.name());
          output->set_datatype(GetDataTypeProtocolString(io.data_type()));
          for (const auto d : io.dims()) {
            output->add_shape(d);
          }
        }
      }
    }

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRTSERVER_ErrorDelete(err);

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(response, status, state);
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
  }

  // Only handle one request at a time (to avoid having request cause
  // too much load on server), so register for next request only after
  // this one finished.
  if (!shutdown && (state->step_ == Steps::FINISH)) {
    StartNewRequest();
  }

  return state->step_ != Steps::FINISH;
}

//
// ModelConfigHandler
//
class ModelConfigHandler
    : public Handler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncResponseWriter<ModelConfigResponse>,
          ModelConfigRequest, ModelConfigResponse> {
 public:
  ModelConfigHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, trtserver, server_id, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ModelConfigHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestModelConfig(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ModelConfigHandler::Process(Handler::State* state, bool rpc_ok)
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

  const ModelConfigRequest& request = state->request_;
  ModelConfigResponse& response = state->response_;

  if (state->step_ == Steps::START) {
    ServerStatus server_status;

    TRTSERVER_Protobuf* model_status_protobuf = nullptr;
    TRTSERVER_Error* err = TRTSERVER_ServerModelStatus(
        trtserver_.get(), request.name().c_str(), &model_status_protobuf);
    if (err == nullptr) {
      const char* status_buffer;
      size_t status_byte_size;
      err = TRTSERVER_ProtobufSerialize(
          model_status_protobuf, &status_buffer, &status_byte_size);
      if (err == nullptr) {
        if (!server_status.ParseFromArray(status_buffer, status_byte_size)) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
        }
      }
    }

    TRTSERVER_ProtobufDelete(model_status_protobuf);

    if (err == nullptr) {
      const auto& nitr = server_status.model_status().find(request.name());
      if (nitr == server_status.model_status().end()) {
        err = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            std::string(
                "no config available for unknown model '" + request.name() +
                "'")
                .c_str());
      } else {
        // All models share the same config across versions so we
        // ignore request.version().
        const ModelStatus& model_status = nitr->second;
        response.mutable_config()->CopyFrom(model_status.config());
      }
    }

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRTSERVER_ErrorDelete(err);

    state->step_ = Steps::COMPLETE;
    state->context_->responder_->Finish(response, status, state);
  } else if (state->step_ == Steps::COMPLETE) {
    state->step_ = Steps::FINISH;
  }

  // Only handle one request at a time (to avoid having request cause
  // too much load on server), so register for next request only after
  // this one finished.
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
    TRTSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRTSERVER_Memory_Type preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRTSERVER_Memory_Type* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload* payload = reinterpret_cast<AllocPayload*>(userp);
  ModelInferResponse* response = payload->response_;
  const AllocPayload::TensorShmMap* shm_map = payload->shm_map_;

  *buffer = nullptr;
  *buffer_userp = nullptr;
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // Called once for each result tensor in the inference request.
  ModelInferResponse::InferOutputTensor* output_tensor =
      response->add_outputs();
  output_tensor->set_name(tensor_name);
  std::string* raw_output =
      output_tensor->mutable_contents()->mutable_raw_contents();

  if (byte_size > 0) {
    bool use_shm = false;

    if (shm_map != nullptr) {
      const auto& pr = shm_map->find(tensor_name);
      if (pr != shm_map->end()) {
        // If the output is in shared memory then check that the expected
        // requested buffer size is at least the byte size of the output.
        if (byte_size > pr->second.byte_size_) {
          return TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL,
              std::string(
                  "for output " + std::string(tensor_name) +
                  " expected requested buffer size to be at least" +
                  std::to_string(pr->second.byte_size_) + " bytes but got " +
                  std::to_string(byte_size) + " in actual request")
                  .c_str());
        }

        *buffer = const_cast<void*>(pr->second.base_);
        *actual_memory_type = pr->second.memory_type_;
        *actual_memory_type_id = pr->second.device_id_;
        use_shm = true;

        LOG_VERBOSE(1) << "GRPC: using shared-memory for '" << tensor_name
                       << "', size: " << byte_size << ", addr: " << *buffer;
      }
    }

    if (!use_shm) {
      // Can't allocate for any memory type other than CPU. If asked to
      // allocate on GPU memory then force allocation on CPU instead.
      if (*actual_memory_type != TRTSERVER_MEMORY_CPU) {
        LOG_VERBOSE(1) << "GRPC: unable to provide '" << tensor_name << "' in "
                       << MemoryTypeString(*actual_memory_type) << ", will use "
                       << MemoryTypeString(TRTSERVER_MEMORY_CPU);
        *actual_memory_type = TRTSERVER_MEMORY_CPU;
        *actual_memory_type_id = 0;
      }

      raw_output->resize(byte_size);
      *buffer = static_cast<void*>(&((*raw_output)[0]));

      LOG_VERBOSE(1) << "GRPC: using buffer for '" << tensor_name
                     << "', size: " << byte_size << ", addr: " << *buffer;
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

  // Don't do anything when releasing a buffer since InferResponseAlloc
  // wrote directly into the response protobuf.
  return nullptr;  // Success
}

TRTSERVER_Error*
InferAllocatorPayload(
    const std::shared_ptr<TRTSERVER_Server>& trtserver,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
    const ModelInferRequest& request,
    AllocPayload::TensorSerializedDataMap* serialized_data_map,
    ModelInferResponse& response, AllocPayload* alloc_payload)
{
  alloc_payload->response_ = &response;
  if (alloc_payload->shm_map_ != nullptr) {
    alloc_payload->shm_map_->clear();
  }
  if (alloc_payload->serialized_data_map_ != nullptr) {
    alloc_payload->serialized_data_map_->clear();
  }
  if (!serialized_data_map->empty()) {
    alloc_payload->serialized_data_map_ = serialized_data_map;
  }

  // If any of the outputs use shared memory, then we must calculate
  // the memory address for that output and store it in the allocator
  // payload so that it is available when the allocation callback is
  // invoked.

#if 0
  for (const auto& io : request.outputs()) {
    if (io.has_shared_memory()) {
      void* base;
      TRTSERVER_SharedMemoryBlock* smb = nullptr;
      RETURN_IF_ERR(smb_manager->Get(&smb, io.shared_memory().name()));
      RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
          trtserver.get(), smb, io.shared_memory().offset(),
          io.shared_memory().byte_size(), &base));

      TRTSERVER_Memory_Type memory_type;
      int64_t memory_type_id;
      TRTSERVER_SharedMemoryBlockMemoryType(smb, &memory_type);
      TRTSERVER_SharedMemoryBlockMemoryTypeId(smb, &memory_type_id);

      // if shm_map_ does not exist, then create an empty shm_map
      if (alloc_payload->shm_map_ == nullptr) {
        alloc_payload->shm_map_ = new AllocPayload::TensorShmMap;
      }

      alloc_payload->shm_map_->emplace(
          io.name(), AllocPayload::ShmInfo{base, io.shared_memory().byte_size(),
                                           memory_type, memory_type_id});
    }
  }
#endif

  return nullptr;  // Success
}

TRTSERVER_Error*
InferGRPCToInputHelper(
    const std::string& input_name, const std::string& model_name,
    const std::string& tensor_dt, const std::string input_dt,
    const size_t byte_size)
{
  if (input_dt.compare(tensor_dt) != 0) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected explicit tensor data for input tensor '" + input_name +
            "' for model '" + model_name + "' of type '" + tensor_dt +
            "', expected datatype '" + input_dt + "'")
            .c_str());
  }
  if (byte_size != 0) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected explicit tensor data for input tensor '" + input_name +
            "' for model '" + model_name +
            "', binary data was already supplied.")
            .c_str());
  }
  return nullptr;  // success
}

TRTSERVER_Error*
InferGRPCToInput(
    const std::shared_ptr<TRTSERVER_Server>& trtserver,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
    const ModelInferRequest& request,
    AllocPayload::TensorSerializedDataMap* serialized_data_map,
    TRTSERVER_InferenceRequestProvider* request_provider)
{
  // Verify that the batch-byte-size of each input matches the size of
  // the provided tensor data (provided raw or from shared memory)
  for (const auto& io : request.inputs()) {
    const void* base;
    size_t byte_size;
    TRTSERVER_Memory_Type memory_type = TRTSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;

#if 0
    if (io.has_shared_memory()) {
      TRTSERVER_SharedMemoryBlock* smb = nullptr;
      RETURN_IF_ERR(smb_manager->Get(&smb, io.shared_memory().name()));
      RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
          trtserver.get(), smb, io.shared_memory().offset(),
          io.shared_memory().byte_size(), const_cast<void**>(&base)));
      byte_size = io.shared_memory().byte_size();
      TRTSERVER_SharedMemoryBlockMemoryType(smb, &memory_type);
      TRTSERVER_SharedMemoryBlockMemoryTypeId(smb, &memory_type_id);
    } else
#endif
    {
      if (!io.has_contents()) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            std::string(
                "expected tensor data for input tensor '" + io.name() +
                "' for model '" + request.model_name() + "'")
                .c_str());
      }

      // Try to read the raw contents if available
      const std::string& raw = io.contents().raw_contents();
      base = raw.c_str();
      byte_size = raw.size();

      // Check the presence of explicit tensors
      if (io.contents().bool_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), "BOOL", io.datatype(), byte_size));
        base = (const void*)io.contents().bool_contents().data();
        byte_size = io.contents().bool_contents_size();
      }

      if (io.contents().int_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), "INT32", io.datatype(),
            byte_size));
        base = (const void*)io.contents().int_contents().data();
        byte_size = io.contents().int_contents_size() * 4;
      }

      if (io.contents().int64_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), "INT64", io.datatype(),
            byte_size));
        base = (const void*)io.contents().int64_contents().data();
        byte_size = io.contents().int64_contents_size() * 8;
      }

      if (io.contents().uint_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), "UINT32", io.datatype(),
            byte_size));
        base = (const void*)io.contents().uint_contents().data();
        byte_size = io.contents().uint_contents_size() * 4;
      }

      if (io.contents().uint64_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), "UINT64", io.datatype(),
            byte_size));
        base = (const void*)io.contents().uint64_contents().data();
        byte_size = io.contents().uint64_contents_size() * 8;
      }

      if (io.contents().fp32_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), "FP32", io.datatype(), byte_size));
        base = (const void*)io.contents().fp32_contents().data();
        byte_size = io.contents().fp32_contents_size() * 4;
      }

      if (io.contents().fp64_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), "FP64", io.datatype(), byte_size));
        base = (const void*)io.contents().fp64_contents().data();
        byte_size = io.contents().fp64_contents_size() * 8;
      }

      if (io.contents().byte_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), "STRING", io.datatype(),
            byte_size));
        std::shared_ptr<std::string> serialized(new std::string(""));
        serialized_data_map->emplace(io.name(), serialized);

        // Serialize the output tensor strings. Each string is
        // serialized as a 4-byte length followed by the string itself
        // with no null-terminator.
        for (int e = 0; e < io.contents().byte_contents_size(); ++e) {
          const std::string& element = io.contents().byte_contents().Get(e);
          uint32_t len{(uint32_t)element.size()};
          serialized->append(
              reinterpret_cast<const char*>(&len), sizeof(uint32_t));
          if (element.size() > 0) {
            serialized->append(element.c_str(), len);
          }
        }
        base = serialized->c_str();
        byte_size = serialized->size();
      }
    }

    uint64_t expected_byte_size = 0;
    RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderInputBatchByteSize(
        request_provider, io.name().c_str(), &expected_byte_size));

    if ((expected_byte_size != 0) && (byte_size != expected_byte_size)) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INVALID_ARG,
          std::string(
              "unexpected size " + std::to_string(byte_size) + " for input '" +
              io.name() + "', expecting " + std::to_string(expected_byte_size) +
              " for model '" + request.model_name() + "'")
              .c_str());
    }

    RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
        request_provider, io.name().c_str(), base, byte_size, memory_type,
        memory_type_id));
  }

  return nullptr;  // success
}

//
// ModelInferHandler
//
class ModelInferHandler
    : public Handler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncResponseWriter<ModelInferResponse>,
          ModelInferRequest, ModelInferResponse> {
 public:
  ModelInferHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      const std::shared_ptr<TraceManager>& trace_manager,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(
            name, trtserver, server_id, service, cq, max_state_bucket_count),
        trace_manager_(trace_manager), smb_manager_(smb_manager)
  {
    // Create the allocator that will be used to allocate buffers for
    // the result tensors.
    FAIL_IF_ERR(
        TRTSERVER_ResponseAllocatorNew(
            &allocator_, InferResponseAlloc, InferResponseRelease),
        "creating inference response allocator");
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;

 private:
  static void InferComplete(
      TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
      TRTSERVER_InferenceResponse* response, void* userp);

  std::shared_ptr<TraceManager> trace_manager_;
  std::shared_ptr<SharedMemoryBlockManager> smb_manager_;
  TRTSERVER_ResponseAllocator* allocator_;
};

void
ModelInferHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);

#ifdef TRTIS_ENABLE_TRACING
  if (trace_manager_ != nullptr) {
    state->trace_meta_data_.reset(trace_manager_->SampleTrace());
    if (state->trace_meta_data_ != nullptr) {
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "grpc wait/read start");
    }
  }
#endif  // TRTIS_ENABLE_TRACING

  service_->RequestModelInfer(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ModelInferHandler::Process(Handler::State* state, bool rpc_ok)
{
  LOG_VERBOSE(1) << "Process for " << Name() << ", rpc_ok=" << rpc_ok << ", "
                 << state->unique_id_ << " step " << state->step_;

  // We need an explicit finish indicator. Can't use 'state->step_'
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

  const ModelInferRequest& request = state->request_;
  ModelInferResponse& response = state->response_;

  if (state->step_ == Steps::START) {
#ifdef TRTIS_ENABLE_TRACING
    if (state->trace_meta_data_ != nullptr) {
      state->trace_meta_data_->tracer_->SetModel(
          request.model_name(), request.model_version());
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "grpc wait/read end");
    }
#endif  // TRTIS_ENABLE_TRACING

    // Start a new request to replace this one...
    if (!shutdown) {
      StartNewRequest();
    }

    // Create the inference request provider which provides all the
    // input information needed for an inference.
    TRTSERVER_InferenceRequestOptions* request_options = nullptr;
    TRTSERVER_Error* err = TRTSERVER_InferenceRequestOptionsNew(
        &request_options, request.model_name().c_str(),
        request.model_version());
    if (err == nullptr) {
      err = SetInferenceRequestOptions(request_options, request);
    }

    TRTSERVER_InferenceRequestProvider* request_provider = nullptr;
    if (err == nullptr) {
      err = TRTSERVER_InferenceRequestProviderNewV2(
          &request_provider, trtserver_.get(), request_options);
    }

    // Will be used to hold the serialized data in case explicit string tensors
    // are present in the request.
    AllocPayload::TensorSerializedDataMap* serialized_data_map =
        new AllocPayload::TensorSerializedDataMap();

    if (err == nullptr) {
      err = InferGRPCToInput(
          trtserver_, smb_manager_, request, serialized_data_map,
          request_provider);
    }
    if (err == nullptr) {
      err = InferAllocatorPayload(
          trtserver_, smb_manager_, request, serialized_data_map, response,
          &state->alloc_payload_);
    }
    if (err == nullptr) {
      // Provide the trace manager object to use for this request, if
      // nullptr then no tracing will be performed.
      TRTSERVER_TraceManager* trace_manager = nullptr;
#ifdef TRTIS_ENABLE_TRACING
      if (state->trace_meta_data_ != nullptr) {
        TRTSERVER_TraceManagerNew(
            &trace_manager, TraceManager::CreateTrace,
            TraceManager::ReleaseTrace, state->trace_meta_data_.get());
      }
#endif  // TRTIS_ENABLE_TRACING

      state->step_ = ISSUED;
      err = TRTSERVER_ServerInferAsync(
          trtserver_.get(), trace_manager, request_provider, allocator_,
          &state->alloc_payload_ /* response_allocator_userp */, InferComplete,
          reinterpret_cast<void*>(state));
    }

    // The request provider can be deleted before ServerInferAsync
    // callback completes.
    TRTSERVER_InferenceRequestProviderDelete(request_provider);
    TRTSERVER_InferenceRequestOptionsDelete(request_options);

    // If not error then state->step_ == ISSUED and inference request
    // has initiated... completion callback will transition to
    // COMPLETE. If error go immediately to COMPLETE.
    if (err != nullptr) {
      LOG_VERBOSE(1) << "Infer failed: " << TRTSERVER_ErrorMessage(err);

      grpc::Status status;
      GrpcStatusUtil::Create(&status, err);
      TRTSERVER_ErrorDelete(err);

      response.Clear();

#ifdef TRTIS_ENABLE_TRACING
      if (state->trace_meta_data_ != nullptr) {
        state->trace_meta_data_->tracer_->CaptureTimestamp(
            TRTSERVER_TRACE_LEVEL_MIN, "grpc send start");
      }
#endif  // TRTIS_ENABLE_TRACING

      state->step_ = COMPLETE;
      state->context_->responder_->Finish(response, status, state);
    }
  } else if (state->step_ == Steps::COMPLETE) {
#ifdef TRTIS_ENABLE_TRACING
    if (state->trace_meta_data_ != nullptr) {
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "grpc send end");
    }
#endif  // TRTIS_ENABLE_TRACING

    state->step_ = Steps::FINISH;
    finished = true;
  }

  return !finished;
}

void
ModelInferHandler::InferComplete(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER_InferenceResponse* trtserver_response, void* userp)
{
  State* state = reinterpret_cast<State*>(userp);

  LOG_VERBOSE(1) << "ModelInferHandler::InferComplete, " << state->unique_id_
                 << " step " << state->step_;

  ModelInferResponse& response = state->response_;
  InferResponseHeader response_header;

  TRTSERVER_Error* err = TRTSERVER_InferenceResponseStatus(trtserver_response);
  if (err == nullptr) {
    TRTSERVER_Protobuf* response_protobuf = nullptr;
    err = TRTSERVER_InferenceResponseHeader(
        trtserver_response, &response_protobuf);
    if (err == nullptr) {
      const char* buffer;
      size_t byte_size;
      err = TRTSERVER_ProtobufSerialize(response_protobuf, &buffer, &byte_size);
      if (err == nullptr) {
        if (!response_header.ParseFromArray(buffer, byte_size)) {
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_INTERNAL, "failed to parse response header");
        }
      }

      TRTSERVER_ProtobufDelete(response_protobuf);
    }
  }

  const char* id;
  if (err == nullptr) {
    err = TRTSERVER_InferenceResponseIdStr(trtserver_response, &id);
  }

  // Convert the InferResponseHeader to the V2 response
  if (err == nullptr) {
    response.set_model_name(response_header.model_name());
    response.set_model_version(response_header.model_version());
    response.set_id(id);
    for (const auto& io : response_header.output()) {
      // Find the tensor in the response and set its shape.
      for (auto& output : *(response.mutable_outputs())) {
        if (output.name() == io.name()) {
          if (io.batch_classes().size() == 0) {
            for (const auto d : io.raw().dims()) {
              output.add_shape(d);
            }
            output.set_datatype(GetDataTypeProtocolString(io.data_type()));
          } else {
            int cls_count = 0;
            for (const auto& classes : io.batch_classes()) {
              cls_count = classes.cls().size();
              for (const auto& cls : classes.cls()) {
                if (!cls.label().empty()) {
                  output.mutable_contents()->add_byte_contents(std::string(
                      std::to_string(cls.idx()) + ":" +
                      std::to_string(cls.value()) + ":" + cls.label()));
                } else {
                  output.mutable_contents()->add_byte_contents(std::string(
                      std::to_string(cls.idx()) + ":" +
                      std::to_string(cls.value())));
                }
              }
            }
            output.add_shape(io.batch_classes().size());
            output.add_shape(cls_count);

            output.set_datatype("STRING");
          }
          break;
        }
      }
    }
  }

  // Make sure response doesn't exceed GRPC limits.
  if ((err == nullptr) && (response.ByteSizeLong() > INT_MAX)) {
    err = TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        std::string(
            "Response has byte size " +
            std::to_string(response.ByteSizeLong()) +
            " which exceeds gRPC's byte size limit " + std::to_string(INT_MAX) +
            ".")
            .c_str());
  }

  if (err != nullptr) {
    response.Clear();
  }

  grpc::Status status;
  GrpcStatusUtil::Create(&status, err);

  // Don't need to explicitly delete 'trace_manager'. It will be deleted by
  // the TraceMetaData object in 'state'.
  LOG_TRTSERVER_ERROR(
      TRTSERVER_InferenceResponseDelete(trtserver_response),
      "deleting GRPC response");
  TRTSERVER_ErrorDelete(err);

#ifdef TRTIS_ENABLE_TRACING
  if (state->trace_meta_data_ != nullptr) {
    state->trace_meta_data_->tracer_->CaptureTimestamp(
        TRTSERVER_TRACE_LEVEL_MIN, "grpc send start");
  }
#endif  // TRTIS_ENABLE_TRACING

  state->step_ = COMPLETE;
  state->context_->responder_->Finish(response, status, state);
}

#if 0
//
// StreamInferHandler
//
class StreamInferHandler
    : public Handler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncReaderWriter<InferResponse, InferRequest>,
          InferRequest, InferResponse> {
 public:
  StreamInferHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      const std::shared_ptr<TraceManager>& trace_manager,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(
            name, trtserver, server_id, service, cq, max_state_bucket_count),
        trace_manager_(trace_manager), smb_manager_(smb_manager)
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
      TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
      TRTSERVER_InferenceResponse* response, void* userp);

  std::shared_ptr<TraceManager> trace_manager_;
  std::shared_ptr<SharedMemoryBlockManager> smb_manager_;
  TRTSERVER_ResponseAllocator* allocator_;
};

void
StreamInferHandler::StartNewRequest()
{
  const uint64_t unique_id = RequestStatusUtil::NextUniqueRequestId();
  auto context = std::make_shared<State::Context>(server_id_, unique_id);
  State* state = StateNew(context);

#ifdef TRTIS_ENABLE_TRACING
  if (trace_manager_ != nullptr) {
    state->trace_meta_data_.reset(trace_manager_->SampleTrace());
    if (state->trace_meta_data_ != nullptr) {
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "grpc wait/read start");
    }
  }
#endif  // TRTIS_ENABLE_TRACING

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

    // Since this is the start of a connection, 'state' hasn't been
    // used yet so use it to read a request off the connection.
    state->context_->step_ = Steps::READ;
    state->step_ = Steps::READ;
    state->context_->responder_->Read(&state->request_, state);

  } else if (state->step_ == Steps::READ) {
#ifdef TRTIS_ENABLE_TRACING
    if (state->trace_meta_data_ != nullptr) {
      state->trace_meta_data_->tracer_->SetModel(
          state->request_.model_name(), state->request_.model_version());
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "grpc wait/read end");
    }
#endif  // TRTIS_ENABLE_TRACING

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

    // Create the inference request provider which provides all the
    // input information needed for an inference.
    TRTSERVER_InferenceRequestOptions* request_options = nullptr;
    TRTSERVER_Error* err = TRTSERVER_InferenceRequestOptionsNew(
        &request_options, request.model_name().c_str(),
        request.model_version());
    if (err == nullptr) {
      err = SetTRTSERVER_InferenceRequestOptions(
          request_options, request.meta_data());
    }

    TRTSERVER_InferenceRequestProvider* request_provider = nullptr;
    if (err == nullptr) {
      err = TRTSERVER_InferenceRequestProviderNewV2(
          &request_provider, trtserver_.get(), request_options);
    }

    // Will be used when GRPC request contains explicit bytes tensors
    AllocPayload::TensorSerializedDataMap* serialized_data_map =
      new AllocPayload::TensorSerializedDataMap();

    if (err == nullptr) {
      err = InferGRPCToInput(
          trtserver_, smb_manager_, request.meta_data(), request,
          serialized_data_map, request_provider);
    }
    if (err == nullptr) {
      err = InferAllocatorPayload(
          trtserver_, smb_manager_, request.meta_data(),
          serialized_data_map, response, &state->alloc_payload_);
    }
    if (err == nullptr) {
      // Provide the trace manager object to use for this request, if
      // nullptr then no tracing will be performed.
      TRTSERVER_TraceManager* trace_manager = nullptr;
#ifdef TRTIS_ENABLE_TRACING
      if (state->trace_meta_data_ != nullptr) {
        TRTSERVER_TraceManagerNew(
            &trace_manager, TraceManager::CreateTrace,
            TraceManager::ReleaseTrace, state->trace_meta_data_.get());
      }
#endif  // TRTIS_ENABLE_TRACING

      state->step_ = ISSUED;
      err = TRTSERVER_ServerInferAsync(
          trtserver_.get(), trace_manager, request_provider, allocator_,
          &state->alloc_payload_ /* response_allocator_userp */,
          StreamInferComplete, reinterpret_cast<void*>(state));
    }

    // The request provider can be deleted before ServerInferAsync
    // callback completes.
    TRTSERVER_InferenceRequestProviderDelete(request_provider);
    TRTSERVER_InferenceRequestOptionsDelete(request_options);

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
      // initialized or uninitialized.
      response.mutable_meta_data()->Clear();
      response.mutable_raw_output()->Clear();

      response.mutable_meta_data()->set_id(request.meta_data().id());

      state->step_ = Steps::WRITEREADY;
      state->context_->WriteResponseIfReady(state);
    }

    // Now that the inference request is in flight, create a copy of
    // 'state' and use it to attempt another read from the connection
    // (i.e the next request in the stream).
    State* next_read_state = StateNew(context, Steps::READ);

#ifdef TRTIS_ENABLE_TRACING
    // Capture a timestamp for the time when we start waiting for this
    // next request to read.
    if (trace_manager_ != nullptr) {
      next_read_state->trace_meta_data_.reset(trace_manager_->SampleTrace());
      if (next_read_state->trace_meta_data_ != nullptr) {
        next_read_state->trace_meta_data_->tracer_->CaptureTimestamp(
            TRTSERVER_TRACE_LEVEL_MIN, "grpc wait/read start");
      }
    }
#endif  // TRTIS_ENABLE_TRACING

    next_read_state->context_->responder_->Read(
        &next_read_state->request_, next_read_state);

  } else if (state->step_ == Steps::WRITTEN) {
#ifdef TRTIS_ENABLE_TRACING
    if (state->trace_meta_data_ != nullptr) {
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRTSERVER_TRACE_LEVEL_MIN, "grpc send end");
    }
#endif  // TRTIS_ENABLE_TRACING

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
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER_InferenceResponse* trtserver_response, void* userp)
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
  // output as they may be partially initialized or uninitialized.
  if (response_status != nullptr) {
    response.mutable_meta_data()->Clear();
    response.mutable_raw_output()->Clear();
  }

  RequestStatusUtil::Create(
      response.mutable_request_status(), response_status, state->unique_id_,
      state->context_->server_id_);

  // Don't need to explicitly delete 'trace_manager'. It will be deleted by
  // the TraceMetaData object in 'state'.
  LOG_TRTSERVER_ERROR(
      TRTSERVER_InferenceResponseDelete(trtserver_response),
      "deleting GRPC response");
  TRTSERVER_ErrorDelete(response_status);

  response.mutable_meta_data()->set_id(request.meta_data().id());

  state->step_ = Steps::WRITEREADY;
  state->context_->WriteResponseIfReady(state);
}

//
// RepositoryHandler
//
class RepositoryHandler
    : public Handler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncResponseWriter<RepositoryResponse>,
          RepositoryRequest, RepositoryResponse> {
 public:
  RepositoryHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, trtserver, server_id, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
RepositoryHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(server_id_);
  State* state = StateNew(context);
  service_->RequestRepository(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
RepositoryHandler::Process(Handler::State* state, bool rpc_ok)
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

  const RepositoryRequest& request = state->request_;
  RepositoryResponse& response = state->response_;

  if (state->step_ == Steps::START) {
    TRTSERVER_Error* err = nullptr;
    switch (request.request_type_case()) {
      case RepositoryRequest::RequestTypeCase::kIndex: {
        TRTSERVER_Protobuf* repository_index_protobuf = nullptr;
        err = TRTSERVER_ServerModelRepositoryIndex(
            trtserver_.get(), &repository_index_protobuf);
        if (err == nullptr) {
          const char* serialized_buffer;
          size_t serialized_byte_size;
          err = TRTSERVER_ProtobufSerialize(
              repository_index_protobuf, &serialized_buffer,
              &serialized_byte_size);
          if (err == nullptr) {
            if (!response.mutable_index()->ParseFromArray(
                    serialized_buffer, serialized_byte_size)) {
              err = TRTSERVER_ErrorNew(
                  TRTSERVER_ERROR_UNKNOWN, "failed to parse repository index");
            }
          }
        }

        TRTSERVER_ProtobufDelete(repository_index_protobuf);
        break;
      }
      default:
        err = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_UNKNOWN, "request type is not specified");
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

  // Only handle one model repository request at a time (to avoid having
  // model repository request cause too much load on server),
  // so register for next request only after this one finished.
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
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncResponseWriter<ModelControlResponse>,
          ModelControlRequest, ModelControlResponse> {
 public:
  ModelControlHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
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
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncResponseWriter<SharedMemoryControlResponse>,
          SharedMemoryControlRequest, SharedMemoryControlResponse> {
 public:
  SharedMemoryControlHandler(
      const std::string& name,
      const std::shared_ptr<TRTSERVER_Server>& trtserver, const char* server_id,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
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
    if (request.has_register_()) {
      if (request.register_().has_system_shared_memory()) {
        // system shared memory
        err = smb_manager_->CpuCreate(
            &smb, request.register_().name(),
            request.register_().system_shared_memory().shared_memory_key(),
            request.register_().system_shared_memory().offset(),
            request.register_().byte_size());
        if (err == nullptr) {
          err = TRTSERVER_ServerRegisterSharedMemory(trtserver_.get(), smb);
        }
      } else if (request.register_().has_cuda_shared_memory()) {
        // cuda shared memory
#ifdef TRTIS_ENABLE_GPU
        const std::string& raw_handle =
            request.register_().cuda_shared_memory().raw_handle();
        char* handle_base = const_cast<char*>(raw_handle.c_str());
        cudaIpcMemHandle_t* cuda_shm_handle =
            reinterpret_cast<cudaIpcMemHandle_t*>(handle_base);
        err = smb_manager_->GpuCreate(
            &smb, request.register_().name(), cuda_shm_handle,
            request.register_().byte_size(),
            request.register_().cuda_shared_memory().device_id());
        if (err == nullptr) {
          err = TRTSERVER_ServerRegisterSharedMemory(trtserver_.get(), smb);
        }
#else
        err = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            std::string(
                "failed to register CUDA shared memory region: '" +
                request.register_().name() + "', GPUs not supported")
                .c_str());
#endif  // TRTIS_ENABLE_GPU
      } else {
        err = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            std::string(
                "failed to register shared memory region: '" +
                request.register_().name() + "', improperly formed request.")
                .c_str());
      }
    } else if (request.has_unregister()) {
      err = smb_manager_->Remove(&smb, request.unregister().name());
      if ((err == nullptr) && (smb != nullptr)) {
        err = TRTSERVER_ServerUnregisterSharedMemory(trtserver_.get(), smb);
        TRTSERVER_Error* del_err = TRTSERVER_SharedMemoryBlockDelete(smb);
        if (del_err != nullptr) {
          LOG_ERROR << "failed to delete shared memory block: "
                    << TRTSERVER_ErrorMessage(del_err);
          TRTSERVER_ErrorDelete(del_err);
        }
      }
    } else if (request.has_unregister_all()) {
      err = smb_manager_->Clear();
      if (err == nullptr) {
        err = TRTSERVER_ServerUnregisterAllSharedMemory(trtserver_.get());
      }
    } else if (request.has_status()) {
      TRTSERVER_Protobuf* shm_status_protobuf = nullptr;
      err = TRTSERVER_ServerSharedMemoryStatus(
          trtserver_.get(), &shm_status_protobuf);
      if (err == nullptr) {
        const char* status_buffer;
        size_t status_byte_size;
        err = TRTSERVER_ProtobufSerialize(
            shm_status_protobuf, &status_buffer, &status_byte_size);
        auto shm_status_response = response.mutable_shared_memory_status();
        if (err == nullptr) {
          if (!shm_status_response->ParseFromArray(
                  status_buffer, status_byte_size)) {
            err = TRTSERVER_ErrorNew(
                TRTSERVER_ERROR_INTERNAL,
                "failed to parse shared memory status");
          }
        }
      }
      TRTSERVER_ProtobufDelete(shm_status_protobuf);
    } else {
      err = TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_UNKNOWN, "unknown sharedmemorycontrol request type");
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
#endif

}  // namespace

//
// GRPCServerV2
//
GRPCServerV2::GRPCServerV2(
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
    const char* server_id, const std::string& server_addr,
    const int infer_allocation_pool_size)
    : server_(server), trace_manager_(trace_manager), smb_manager_(smb_manager),
      server_id_(server_id), server_addr_(server_addr),
      infer_allocation_pool_size_(infer_allocation_pool_size), running_(false)
{
}

GRPCServerV2::~GRPCServerV2()
{
  Stop();
}

TRTSERVER_Error*
GRPCServerV2::Create(
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager, int32_t port,
    int infer_allocation_pool_size, std::unique_ptr<GRPCServerV2>* grpc_server)
{
  const char* server_id = nullptr;
  TRTSERVER_Error* err = TRTSERVER_ServerId(server.get(), &server_id);
  if (err != nullptr) {
    server_id = "unknown:0";
    TRTSERVER_ErrorDelete(err);
  }

  const std::string addr = "0.0.0.0:" + std::to_string(port);
  grpc_server->reset(new GRPCServerV2(
      server, trace_manager, smb_manager, server_id, addr,
      infer_allocation_pool_size));

  return nullptr;  // success
}

TRTSERVER_Error*
GRPCServerV2::Start()
{
  if (running_) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_ALREADY_EXISTS, "GRPC server is already running.");
  }

  grpc_builder_.AddListeningPort(
      server_addr_, grpc::InsecureServerCredentials());
  grpc_builder_.SetMaxMessageSize(MAX_GRPC_MESSAGE_SIZE);
  grpc_builder_.RegisterService(&service_);
  server_live_cq_ = grpc_builder_.AddCompletionQueue();
  server_ready_cq_ = grpc_builder_.AddCompletionQueue();
  model_ready_cq_ = grpc_builder_.AddCompletionQueue();
  server_metadata_cq_ = grpc_builder_.AddCompletionQueue();
  model_metadata_cq_ = grpc_builder_.AddCompletionQueue();
  model_config_cq_ = grpc_builder_.AddCompletionQueue();
  model_infer_cq_ = grpc_builder_.AddCompletionQueue();
#if 0
  stream_infer_cq_ = grpc_builder_.AddCompletionQueue();
  repository_cq_ = grpc_builder_.AddCompletionQueue();
  modelcontrol_cq_ = grpc_builder_.AddCompletionQueue();
  shmcontrol_cq_ = grpc_builder_.AddCompletionQueue();
#endif
  grpc_server_ = grpc_builder_.BuildAndStart();

  // Handler for server-live requests.
  ServerLiveHandler* hserverlive = new ServerLiveHandler(
      "ServerLiveHandler", server_, server_id_, &service_,
      server_live_cq_.get(), 2 /* max_state_bucket_count */);
  hserverlive->Start();
  server_live_handler_.reset(hserverlive);

  // Handler for server-ready requests.
  ServerReadyHandler* hserverready = new ServerReadyHandler(
      "ServerReadyHandler", server_, server_id_, &service_,
      server_ready_cq_.get(), 2 /* max_state_bucket_count */);
  hserverready->Start();
  server_ready_handler_.reset(hserverready);

  // Handler for model-ready requests.
  ModelReadyHandler* hmodelready = new ModelReadyHandler(
      "ModelReadyHandler", server_, server_id_, &service_,
      model_ready_cq_.get(), 2 /* max_state_bucket_count */);
  hmodelready->Start();
  model_ready_handler_.reset(hmodelready);

  // Handler for server-metadata requests.
  ServerMetadataHandler* hservermetadata = new ServerMetadataHandler(
      "ServerMetadataHandler", server_, server_id_, &service_,
      server_metadata_cq_.get(), 2 /* max_state_bucket_count */);
  hservermetadata->Start();
  server_metadata_handler_.reset(hservermetadata);

  // Handler for model-metadata requests.
  ModelMetadataHandler* hmodelmetadata = new ModelMetadataHandler(
      "ModelMetadataHandler", server_, server_id_, &service_,
      model_metadata_cq_.get(), 2 /* max_state_bucket_count */);
  hmodelmetadata->Start();
  model_metadata_handler_.reset(hmodelmetadata);

  // Handler for model-config requests.
  ModelConfigHandler* hmodelconfig = new ModelConfigHandler(
      "ModelConfigHandler", server_, server_id_, &service_,
      model_config_cq_.get(), 2 /* max_state_bucket_count */);
  hmodelconfig->Start();
  model_config_handler_.reset(hmodelconfig);

  // Handler for model inference requests.
  ModelInferHandler* hmodelinfer = new ModelInferHandler(
      "ModelInferHandler", server_, server_id_, trace_manager_, smb_manager_,
      &service_, model_infer_cq_.get(),
      infer_allocation_pool_size_ /* max_state_bucket_count */);
  hmodelinfer->Start();
  model_infer_handler_.reset(hmodelinfer);

#if 0
  // Handler for streaming inference requests.
  StreamInferHandler* hstreaminfer = new StreamInferHandler(
      "StreamInferHandler", server_, server_id_, trace_manager_, smb_manager_,
      &service_, stream_infer_cq_.get(),
      infer_allocation_pool_size_ /* max_state_bucket_count */);
  hstreaminfer->Start();
  stream_infer_handler_.reset(hstreaminfer);

  // Handler for status requests.
  RepositoryHandler* hrepository = new RepositoryHandler(
      "RepositoryHandler", server_, server_id_, &service_, repository_cq_.get(),
      2 /* max_state_bucket_count */);
  hrepository->Start();
  repository_handler_.reset(hrepository);

  // Handler for model-control requests.
  ModelControlHandler* hmodelcontrol = new ModelControlHandler(
      "ModelControlHandler", server_, server_id_, &service_,
      modelcontrol_cq_.get(), 2 /* max_state_bucket_count */);
  hmodelcontrol->Start();
  modelcontrol_handler_.reset(hmodelcontrol);

  // Handler for shared-memory-control requests.
  SharedMemoryControlHandler* hshmcontrol = new SharedMemoryControlHandler(
      "SharedMemoryControlHandler", server_, server_id_, smb_manager_,
      &service_, shmcontrol_cq_.get(), 2 /* max_state_bucket_count */);
  hshmcontrol->Start();
  shmcontrol_handler_.reset(hshmcontrol);
#endif

  running_ = true;
  LOG_INFO << "Started GRPCInferenceService at " << server_addr_;
  return nullptr;  // success
}

TRTSERVER_Error*
GRPCServerV2::Stop()
{
  if (!running_) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_UNAVAILABLE, "GRPC server is not running.");
  }

  // Always shutdown the completion queue after the server.
  grpc_server_->Shutdown();

  server_live_cq_->Shutdown();
  server_ready_cq_->Shutdown();
  model_ready_cq_->Shutdown();
  server_metadata_cq_->Shutdown();
  model_metadata_cq_->Shutdown();
  model_config_cq_->Shutdown();
  model_infer_cq_->Shutdown();
#if 0
  repository_cq_->Shutdown();
  stream_infer_cq_->Shutdown();
  modelcontrol_cq_->Shutdown();
  shmcontrol_cq_->Shutdown();
#endif

  // Must stop all handlers explicitly to wait for all the handler
  // threads to join since they are referencing completion queue, etc.
  dynamic_cast<ServerLiveHandler*>(server_live_handler_.get())->Stop();
  dynamic_cast<ServerReadyHandler*>(server_ready_handler_.get())->Stop();
  dynamic_cast<ModelReadyHandler*>(model_ready_handler_.get())->Stop();
  dynamic_cast<ServerMetadataHandler*>(server_metadata_handler_.get())->Stop();
  dynamic_cast<ModelMetadataHandler*>(model_metadata_handler_.get())->Stop();
  dynamic_cast<ModelConfigHandler*>(model_config_handler_.get())->Stop();
  dynamic_cast<ModelInferHandler*>(model_infer_handler_.get())->Stop();
#if 0
  dynamic_cast<StreamInferHandler*>(stream_infer_handler_.get())->Stop();
  dynamic_cast<RepositoryHandler*>(repository_handler_.get())->Stop();
  dynamic_cast<ModelControlHandler*>(modelcontrol_handler_.get())->Stop();
  dynamic_cast<SharedMemoryControlHandler*>(shmcontrol_handler_.get())->Stop();
#endif

  running_ = false;
  return nullptr;  // success
}

}}  // namespace nvidia::inferenceserver
