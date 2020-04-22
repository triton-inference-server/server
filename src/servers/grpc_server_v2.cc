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
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/server_status.pb.h"
#include "src/core/tritonserver.h"
#include "src/servers/common.h"

#ifdef TRTIS_ENABLE_TRACING
#include "src/servers/tracer.h"
#endif  // TRTIS_ENABLE_TRACING

namespace nvidia { namespace inferenceserver {

namespace {

// Unique IDs are only needed when debugging. They only appear in
// verbose logging.
#ifndef NDEBUG
uint64_t
NextUniqueId()
{
  static std::atomic<uint64_t> id(0);
  return ++id;
}
#define NEXT_UNIQUE_ID NextUniqueId()
#else
#define NEXT_UNIQUE_ID (0)
#endif

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
  static void Create(grpc::Status* status, TRITONSERVER_Error* err);
  static grpc::StatusCode CodeToStatus(TRITONSERVER_Error_Code code);
};

void
GrpcStatusUtil::Create(grpc::Status* status, TRITONSERVER_Error* err)
{
  if (err == nullptr) {
    *status = grpc::Status::OK;
  } else {
    *status = grpc::Status(
        GrpcStatusUtil::CodeToStatus(TRITONSERVER_ErrorCode(err)),
        TRITONSERVER_ErrorMessage(err));
  }
}

grpc::StatusCode
GrpcStatusUtil::CodeToStatus(TRITONSERVER_Error_Code code)
{
  // GRPC status codes:
  // https://github.com/grpc/grpc/blob/master/include/grpc/impl/codegen/status.h
  switch (code) {
    case TRITONSERVER_ERROR_UNKNOWN:
      return grpc::StatusCode::UNKNOWN;
    case TRITONSERVER_ERROR_INTERNAL:
      return grpc::StatusCode::INTERNAL;
    case TRITONSERVER_ERROR_NOT_FOUND:
      return grpc::StatusCode::NOT_FOUND;
    case TRITONSERVER_ERROR_INVALID_ARG:
      return grpc::StatusCode::INVALID_ARGUMENT;
    case TRITONSERVER_ERROR_UNAVAILABLE:
      return grpc::StatusCode::UNAVAILABLE;
    case TRITONSERVER_ERROR_UNSUPPORTED:
      return grpc::StatusCode::UNIMPLEMENTED;
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
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
// allocation.
//
struct AllocPayload {
  struct ShmInfo {
    void* base_;
    size_t byte_size_;
    TRITONSERVER_MemoryType memory_type_;
    int64_t memory_type_id_;
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
  // repeated byte contents were provided in the request. It actual
  // lifetime is that of the request whereas AllocPayload's lifetime
  // is that of a response... but it is convenient to keep it here.
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
    explicit Context(const uint64_t unique_id = 0)
        : unique_id_(unique_id), step_(Steps::START), finish_ok_(true)
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
            TRITONSERVER_TRACE_LEVEL_MIN, "grpc send start");
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

    // Unique ID for the context. Used only for debugging so will
    // always be 0 in non-debug builds.
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
    unique_id_ = NEXT_UNIQUE_ID;
    context_ = context;
    step_ = start_step;
    request_.Clear();
    response_.Clear();
  }

  void Release() { context_ = nullptr; }

  // Unique ID for the state. Used only for debugging so will
  // always be 0 in non-debug builds.
  uint64_t unique_id_;

  std::shared_ptr<Context> context_;
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
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
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
  std::shared_ptr<TRITONSERVER_Server> tritonserver_;

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
    const std::string& name,
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    ServiceType* service, grpc::ServerCompletionQueue* cq,
    size_t max_state_bucket_count)
    : name_(name), tritonserver_(tritonserver), service_(service), cq_(cq),
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

template <typename ResponderType, typename RequestType, typename ResponseType>
class CommonCallData : public GRPCServerV2::ICallData {
 public:
  using StandardRegisterFunc = std::function<void(
      grpc::ServerContext*, RequestType*, ResponderType*, void*)>;
  using StandardCallbackFunc =
      std::function<void(RequestType&, ResponseType*, grpc::Status*)>;

  CommonCallData(
      const std::string& name, const uint64_t id,
      const StandardRegisterFunc OnRegister,
      const StandardCallbackFunc OnCallback)
      : name_(name), id_(id), OnRegister_(OnRegister), OnCallback_(OnCallback),
        responder_(&ctx_), step_(Steps::START)
  {
    OnRegister_(&ctx_, &request_, &responder_, this);
    LOG_VERBOSE(1) << "Ready for RPC '" << name_ << "', " << id_;
  }

  bool Process(bool ok) override;

  std::string Name() override { return name_; }

  uint64_t Id() override { return id_; }

 private:
  const std::string name_;
  const uint64_t id_;
  const StandardRegisterFunc OnRegister_;
  const StandardCallbackFunc OnCallback_;

  grpc::ServerContext ctx_;

  ResponderType responder_;
  RequestType request_;

  Steps step_;
};

template <typename ResponderType, typename RequestType, typename ResponseType>
bool
CommonCallData<ResponderType, RequestType, ResponseType>::Process(bool rpc_ok)
{
  LOG_VERBOSE(1) << "Process for " << name_ << ", rpc_ok=" << rpc_ok << ", "
                 << id_ << " step " << step_;

  // If RPC failed on a new request then the server is shutting down
  // and so we should do nothing (including not registering for a new
  // request). If RPC failed on a non-START step then there is nothing
  // we can do since we one execute one step.
  const bool shutdown = (!rpc_ok && (step_ == Steps::START));
  if (shutdown) {
    step_ = Steps::FINISH;
  }

  if (step_ == Steps::START) {
    ResponseType response;
    grpc::Status status;

    OnCallback_(request_, &response, &status);

    step_ = Steps::COMPLETE;

    responder_.Finish(response, status, this);
  } else if (step_ == Steps::COMPLETE) {
    step_ = Steps::FINISH;
  }

  if (!shutdown && (step_ == Steps::FINISH)) {
    new CommonCallData<ResponderType, RequestType, ResponseType>(
        name_, id_ + 1, OnRegister_, OnCallback_);
  }

  return step_ != Steps::FINISH;
}

//
// CommonHandler
//
class CommonHandler : public GRPCServerV2::HandlerBase {
 public:
  CommonHandler(
      const std::string& name,
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq);

  // Descriptive name of of the handler.
  const std::string& Name() const { return name_; }

  // Start handling requests.
  void Start();

  // Stop handling requests.
  void Stop();

 private:
  void SetUpAllRequests();

  const std::string name_;
  std::shared_ptr<TRITONSERVER_Server> tritonserver_;

  std::shared_ptr<SharedMemoryManager> shm_manager_;

  GRPCInferenceService::AsyncService* service_;
  grpc::ServerCompletionQueue* cq_;
  std::unique_ptr<std::thread> thread_;
};

CommonHandler::CommonHandler(
    const std::string& name,
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    GRPCInferenceService::AsyncService* service,
    grpc::ServerCompletionQueue* cq)
    : name_(name), tritonserver_(tritonserver), shm_manager_(shm_manager),
      service_(service), cq_(cq)
{
}

void
CommonHandler::Start()
{
  // Use a barrier to make sure we don't return until thread has
  // started.
  auto barrier = std::make_shared<Barrier>(2);

  thread_.reset(new std::thread([this, barrier] {
    SetUpAllRequests();
    barrier->Wait();

    void* tag;
    bool ok;

    while (cq_->Next(&tag, &ok)) {
      GRPCServerV2::ICallData* call_data =
          static_cast<GRPCServerV2::ICallData*>(tag);
      if (!call_data->Process(ok)) {
        LOG_VERBOSE(1) << "Done for " << call_data->Name() << ", "
                       << call_data->Id();
        delete call_data;
      }
    }
  }));

  barrier->Wait();
  LOG_VERBOSE(1) << "Thread started for " << Name();
}

void
CommonHandler::Stop()
{
  if (thread_->joinable()) {
    thread_->join();
  }

  LOG_VERBOSE(1) << "Thread exited for " << Name();
}

void
CommonHandler::SetUpAllRequests()
{
  // Define all the RPCs to be handled by this handler below
  //
  // The format of each RPC specification is :
  // 1. A OnRegister function: This will be called when the
  //    server is ready to receive the requests for this RPC.
  // 2. A OnExecute function: This will be called when the
  //    to process the request.
  // 3. Create a CommonCallData object with the above callback
  //    functions

  //
  //  ModelStatistics
  //
  auto OnRegisterModelStatistics =
      [this](
          grpc::ServerContext* ctx, ModelStatisticsRequest* request,
          grpc::ServerAsyncResponseWriter<ModelStatisticsResponse>* responder,
          void* tag) {
        this->service_->RequestModelStatistics(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteModelStatistics = [this](
                                      ModelStatisticsRequest& request,
                                      ModelStatisticsResponse* response,
                                      grpc::Status* status) {
#ifdef TRTIS_ENABLE_STATS
    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request.version(), &requested_model_version);
    if (err == nullptr) {
      TRITONSERVER_Message* model_stats_message = nullptr;
      err = TRITONSERVER_ServerModelStatistics(
          tritonserver_.get(), request.name().c_str(), requested_model_version,
          &model_stats_message);
      rapidjson::Document model_stats_json;
      if (err == nullptr) {
        const char* buffer;
        size_t byte_size;
        err = TRITONSERVER_MessageSerializeToJson(
            model_stats_message, &buffer, &byte_size);
        if (err == nullptr) {
          model_stats_json.Parse(buffer, byte_size);
          if (model_stats_json.HasParseError()) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                std::string(
                    "failed to parse the model statistics JSON buffer: " +
                    std::string(
                        GetParseError_En(model_stats_json.GetParseError())) +
                    " at " + std::to_string(model_stats_json.GetErrorOffset()))
                    .c_str());
          }
        }
        TRITONSERVER_MessageDelete(model_stats_message);
      }
    }

    if (err == nullptr) {
      for (const auto& version_stats :
           model_stats_json["version_stats"].GetArray()) {
        const auto& infer_stats_json = version_stats["stats"]["inference"];
        InferStatistics infer_stats;
        infer_stats.mutable_success()->set_count(
            infer_stats_json["success"]["count"].GetInt());
        infer_stats.mutable_success()->set_ns(
            infer_stats_json["success"]["ns"].GetInt());
        infer_stats.mutable_fail()->set_count(
            infer_stats_json["fail"]["count"].GetInt());
        infer_stats.mutable_fail()->set_ns(
            infer_stats_json["fail"]["ns"].GetInt());
        infer_stats.mutable_queue()->set_count(
            infer_stats_json["queue"]["count"].GetInt());
        infer_stats.mutable_queue()->set_ns(
            infer_stats_json["queue"]["ns"].GetInt());
        infer_stats.mutable_compute_input()->set_count(
            infer_stats_json["compute_input"]["count"].GetInt());
        infer_stats.mutable_compute_input()->set_ns(
            infer_stats_json["compute_input"]["ns"].GetInt());
        infer_stats.mutable_compute_infer()->set_count(
            infer_stats_json["compute_infer"]["count"].GetInt());
        infer_stats.mutable_compute_infer()->set_ns(
            infer_stats_json["compute_infer"]["ns"].GetInt());
        infer_stats.mutable_compute_output()->set_count(
            infer_stats_json["compute_output"]["count"].GetInt());
        infer_stats.mutable_compute_output()->set_ns(
            infer_stats_json["compute_output"]["ns"].GetInt());

        // Add the statistics to the response
        (*response->mutable_inference())[version_stats["version"].GetString()] =
            infer_stats;
      }
    }
#else
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "the server does not suppport model statistics");
#endif

    GrpcStatusUtil::Create(status, err);
    TRITONSERVER_ErrorDelete(err);
  };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<ModelStatisticsResponse>,
      ModelStatisticsRequest, ModelStatisticsResponse>(
      "ModelStatistics", 0, OnRegisterModelStatistics,
      OnExecuteModelStatistics);


  //
  // SystemSharedMemoryStatus
  //
  auto OnRegisterSystemSharedMemoryStatus =
      [this](
          grpc::ServerContext* ctx, SystemSharedMemoryStatusRequest* request,
          grpc::ServerAsyncResponseWriter<SystemSharedMemoryStatusResponse>*
              responder,
          void* tag) {
        this->service_->RequestSystemSharedMemoryStatus(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteSystemSharedMemoryStatus =
      [this](
          SystemSharedMemoryStatusRequest& request,
          SystemSharedMemoryStatusResponse* response, grpc::Status* status) {
        TRITONSERVER_Error* err =
            shm_manager_->GetStatus(request.name(), response);

        GrpcStatusUtil::Create(status, err);
        TRITONSERVER_ErrorDelete(err);
      };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<SystemSharedMemoryStatusResponse>,
      SystemSharedMemoryStatusRequest, SystemSharedMemoryStatusResponse>(
      "SystemSharedMemoryStatus", 0, OnRegisterSystemSharedMemoryStatus,
      OnExecuteSystemSharedMemoryStatus);


  //
  // SystemSharedMemoryRegister
  //
  auto OnRegisterSystemSharedMemoryRegister =
      [this](
          grpc::ServerContext* ctx, SystemSharedMemoryRegisterRequest* request,
          grpc::ServerAsyncResponseWriter<SystemSharedMemoryRegisterResponse>*
              responder,
          void* tag) {
        this->service_->RequestSystemSharedMemoryRegister(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteSystemSharedMemoryRegister =
      [this](
          SystemSharedMemoryRegisterRequest& request,
          SystemSharedMemoryRegisterResponse* response, grpc::Status* status) {
        TRITONSERVER_Error* err = shm_manager_->RegisterSystemSharedMemory(
            request.name(), request.key(), request.offset(),
            request.byte_size());

        GrpcStatusUtil::Create(status, err);
        TRITONSERVER_ErrorDelete(err);
      };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<SystemSharedMemoryRegisterResponse>,
      SystemSharedMemoryRegisterRequest, SystemSharedMemoryRegisterResponse>(
      "SystemSharedMemoryRegister", 0, OnRegisterSystemSharedMemoryRegister,
      OnExecuteSystemSharedMemoryRegister);


  //
  // SystemSharedMemoryUnregister
  //
  auto OnRegisterSystemSharedMemoryUnregister =
      [this](
          grpc::ServerContext* ctx,
          SystemSharedMemoryUnregisterRequest* request,
          grpc::ServerAsyncResponseWriter<SystemSharedMemoryUnregisterResponse>*
              responder,
          void* tag) {
        this->service_->RequestSystemSharedMemoryUnregister(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteSystemSharedMemoryUnregister =
      [this](
          SystemSharedMemoryUnregisterRequest& request,
          SystemSharedMemoryUnregisterResponse* response,
          grpc::Status* status) {
        TRITONSERVER_Error* err = nullptr;
        if (request.name().empty()) {
          err = shm_manager_->UnregisterAll(TRITONSERVER_MEMORY_CPU);
        } else {
          err =
              shm_manager_->Unregister(request.name(), TRITONSERVER_MEMORY_CPU);
        }

        GrpcStatusUtil::Create(status, err);
        TRITONSERVER_ErrorDelete(err);
      };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<SystemSharedMemoryUnregisterResponse>,
      SystemSharedMemoryUnregisterRequest,
      SystemSharedMemoryUnregisterResponse>(
      "SystemSharedMemoryUnregister", 0, OnRegisterSystemSharedMemoryUnregister,
      OnExecuteSystemSharedMemoryUnregister);


  //
  // CudaSharedMemoryStatus
  //
  auto OnRegisterCudaSharedMemoryStatus =
      [this](
          grpc::ServerContext* ctx, CudaSharedMemoryStatusRequest* request,
          grpc::ServerAsyncResponseWriter<CudaSharedMemoryStatusResponse>*
              responder,
          void* tag) {
        this->service_->RequestCudaSharedMemoryStatus(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };
  auto OnExecuteCudaSharedMemoryStatus =
      [this](
          CudaSharedMemoryStatusRequest& request,
          CudaSharedMemoryStatusResponse* response, grpc::Status* status) {
        TRITONSERVER_Error* err =
            shm_manager_->GetStatus(request.name(), response);

        GrpcStatusUtil::Create(status, err);
        TRITONSERVER_ErrorDelete(err);
      };
  new CommonCallData<
      grpc::ServerAsyncResponseWriter<CudaSharedMemoryStatusResponse>,
      CudaSharedMemoryStatusRequest, CudaSharedMemoryStatusResponse>(
      "CudaSharedMemoryStatus", 0, OnRegisterCudaSharedMemoryStatus,
      OnExecuteCudaSharedMemoryStatus);


  //
  // CudaSharedMemoryRegister
  //
  auto OnRegisterCudaSharedMemoryRegister =
      [this](
          grpc::ServerContext* ctx, CudaSharedMemoryRegisterRequest* request,
          grpc::ServerAsyncResponseWriter<CudaSharedMemoryRegisterResponse>*
              responder,
          void* tag) {
        this->service_->RequestCudaSharedMemoryRegister(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteCudaSharedMemoryRegister =
      [this](
          CudaSharedMemoryRegisterRequest& request,
          CudaSharedMemoryRegisterResponse* response, grpc::Status* status) {
        TRITONSERVER_Error* err = nullptr;
#ifdef TRTIS_ENABLE_GPU
        err = shm_manager_->RegisterCUDASharedMemory(
            request.name(),
            reinterpret_cast<const cudaIpcMemHandle_t*>(
                request.raw_handle().c_str()),
            request.byte_size(), request.device_id());
#else
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "failed to register CUDA shared memory region: '" +
                request.name() + "', GPUs not supported")
                .c_str());
#endif  // TRTIS_ENABLE_GPU

        GrpcStatusUtil::Create(status, err);
        TRITONSERVER_ErrorDelete(err);
      };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<CudaSharedMemoryRegisterResponse>,
      CudaSharedMemoryRegisterRequest, CudaSharedMemoryRegisterResponse>(
      "CudaSharedMemoryRegister", 0, OnRegisterCudaSharedMemoryRegister,
      OnExecuteCudaSharedMemoryRegister);

  //
  // CudaSharedMemoryUnregister
  //
  auto OnRegisterCudaSharedMemoryUnregister =
      [this](
          grpc::ServerContext* ctx, CudaSharedMemoryUnregisterRequest* request,
          grpc::ServerAsyncResponseWriter<CudaSharedMemoryUnregisterResponse>*
              responder,
          void* tag) {
        this->service_->RequestCudaSharedMemoryUnregister(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteCudaSharedMemoryUnregister =
      [this](
          CudaSharedMemoryUnregisterRequest& request,
          CudaSharedMemoryUnregisterResponse* response, grpc::Status* status) {
        TRITONSERVER_Error* err = nullptr;
        if (request.name().empty()) {
          err = shm_manager_->UnregisterAll(TRITONSERVER_MEMORY_GPU);
        } else {
          err =
              shm_manager_->Unregister(request.name(), TRITONSERVER_MEMORY_GPU);
        }

        GrpcStatusUtil::Create(status, err);
        TRITONSERVER_ErrorDelete(err);
      };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<CudaSharedMemoryUnregisterResponse>,
      CudaSharedMemoryUnregisterRequest, CudaSharedMemoryUnregisterResponse>(
      "CudaSharedMemoryUnregister", 0, OnRegisterCudaSharedMemoryUnregister,
      OnExecuteCudaSharedMemoryUnregister);

  //
  // RepositoryIndex
  //
  auto OnRegisterRepositoryIndex =
      [this](
          grpc::ServerContext* ctx, RepositoryIndexRequest* request,
          grpc::ServerAsyncResponseWriter<RepositoryIndexResponse>* responder,
          void* tag) {
        this->service_->RequestRepositoryIndex(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteRepositoryIndex = [this](
                                      RepositoryIndexRequest& request,
                                      RepositoryIndexResponse* response,
                                      grpc::Status* status) {
    TRITONSERVER_Error* err = nullptr;
    if (request.repository_name().empty()) {
      TRITONSERVER_Message* model_index_message = nullptr;
      err = TRITONSERVER_ServerModelIndex(
          tritonserver_.get(), &model_index_message);
      if (err == nullptr) {
        const char* buffer;
        size_t byte_size;
        err = TRITONSERVER_MessageSerializeToJson(
            model_index_message, &buffer, &byte_size);
        if (err == nullptr) {
          rapidjson::Document model_index_json;
          model_index_json.Parse(buffer, byte_size);
          if (model_index_json.HasParseError()) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                std::string(
                    "failed to parse the repository index JSON buffer: " +
                    std::string(
                        GetParseError_En(model_index_json.GetParseError())) +
                    " at " + std::to_string(model_index_json.GetErrorOffset()))
                    .c_str());
          } else {
            for (const auto& model : model_index_json.GetArray()) {
              auto model_index = response->add_models();
              model_index->set_name(model["name"].GetString());
            }
          }
        }
        TRITONSERVER_MessageDelete(model_index_message);
      };
    } else {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "'repository_name' specification is not supported");
    }

    GrpcStatusUtil::Create(status, err);
    TRITONSERVER_ErrorDelete(err);
  };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<RepositoryIndexResponse>,
      RepositoryIndexRequest, RepositoryIndexResponse>(
      "RepositoryIndex", 0, OnRegisterRepositoryIndex,
      OnExecuteRepositoryIndex);

  //
  // RepositoryModelLoad
  //
  auto OnRegisterRepositoryModelLoad =
      [this](
          grpc::ServerContext* ctx, RepositoryModelLoadRequest* request,
          grpc::ServerAsyncResponseWriter<RepositoryModelLoadResponse>*
              responder,
          void* tag) {
        this->service_->RequestRepositoryModelLoad(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteRepositoryModelLoad = [this](
                                          RepositoryModelLoadRequest& request,
                                          RepositoryModelLoadResponse* response,
                                          grpc::Status* status) {
    TRITONSERVER_Error* err = nullptr;
    if (request.repository_name().empty()) {
      err = TRITONSERVER_ServerLoadModel(
          tritonserver_.get(), request.model_name().c_str());
    } else {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "'repository_name' specification is not supported");
    }

    GrpcStatusUtil::Create(status, err);
    TRITONSERVER_ErrorDelete(err);
  };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<RepositoryModelLoadResponse>,
      RepositoryModelLoadRequest, RepositoryModelLoadResponse>(
      "RepositoryModelLoad", 0, OnRegisterRepositoryModelLoad,
      OnExecuteRepositoryModelLoad);

  //
  // RepositoryModelUnload
  //
  auto OnRegisterRepositoryModelUnload =
      [this](
          grpc::ServerContext* ctx, RepositoryModelUnloadRequest* request,
          grpc::ServerAsyncResponseWriter<RepositoryModelUnloadResponse>*
              responder,
          void* tag) {
        this->service_->RequestRepositoryModelUnload(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteRepositoryModelUnload =
      [this](
          RepositoryModelUnloadRequest& request,
          RepositoryModelUnloadResponse* response, grpc::Status* status) {
        TRITONSERVER_Error* err = nullptr;
        if (request.repository_name().empty()) {
          err = TRITONSERVER_ServerUnloadModel(
              tritonserver_.get(), request.model_name().c_str());
        } else {
          err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "'repository_name' specification is not supported");
        }

        GrpcStatusUtil::Create(status, err);
        TRITONSERVER_ErrorDelete(err);
      };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<RepositoryModelUnloadResponse>,
      RepositoryModelUnloadRequest, RepositoryModelUnloadResponse>(
      "RepositoryModelUnload", 0, OnRegisterRepositoryModelUnload,
      OnExecuteRepositoryModelUnload);
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
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, tritonserver, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ServerLiveHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>();
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
    TRITONSERVER_Error* err =
        TRITONSERVER_ServerIsLive(tritonserver_.get(), &live);

    response.set_live((err == nullptr) && live);

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);

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
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, tritonserver, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ServerReadyHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>();
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
    TRITONSERVER_Error* err =
        TRITONSERVER_ServerIsReady(tritonserver_.get(), &ready);

    response.set_ready((err == nullptr) && ready);

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);

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
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, tritonserver, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ModelReadyHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>();
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
    bool is_ready = false;
    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request.version(), &requested_model_version);
    if (err == nullptr) {
      err = TRITONSERVER_ServerModelIsReady(
          tritonserver_.get(), request.name().c_str(), requested_model_version,
          &is_ready);
    }

    response.set_ready(is_ready);

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);

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
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, tritonserver, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ServerMetadataHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>();
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
    TRITONSERVER_Message* server_metadata_message = nullptr;
    TRITONSERVER_Error* err = TRITONSERVER_ServerMetadata(
        tritonserver_.get(), &server_metadata_message);
    if (err == nullptr) {
      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(
          server_metadata_message, &buffer, &byte_size);
      if (err == nullptr) {
        rapidjson::Document server_metadata_json;
        server_metadata_json.Parse(buffer, byte_size);
        if (server_metadata_json.HasParseError()) {
          err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "failed to parse the server metadata JSON buffer: " +
                  std::string(
                      GetParseError_En(server_metadata_json.GetParseError())) +
                  " at " +
                  std::to_string(server_metadata_json.GetErrorOffset()))
                  .c_str());
        } else {
          response.set_name(server_metadata_json["name"].GetString());
          response.set_version(server_metadata_json["version"].GetString());
          for (const auto& extension :
               server_metadata_json["extensions"].GetArray()) {
            response.add_extensions(extension.GetString());
          }
        }
      }
      TRITONSERVER_MessageDelete(server_metadata_message);
    }

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);

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
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, tritonserver, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ModelMetadataHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>();
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
    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request.version(), &requested_model_version);
    if (err == nullptr) {
      TRITONSERVER_Message* model_metadata_message = nullptr;
      err = TRITONSERVER_ServerModelMetadata(
          tritonserver_.get(), request.name().c_str(), requested_model_version,
          &model_metadata_message);
      if (err == nullptr) {
        const char* buffer;
        size_t byte_size;
        err = TRITONSERVER_MessageSerializeToJson(
            model_metadata_message, &buffer, &byte_size);
        if (err == nullptr) {
          rapidjson::Document model_metadata_json;
          model_metadata_json.Parse(buffer, byte_size);
          if (model_metadata_json.HasParseError()) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                std::string(
                    "failed to parse the model metadata JSON buffer: " +
                    std::string(
                        GetParseError_En(model_metadata_json.GetParseError())) +
                    " at " +
                    std::to_string(model_metadata_json.GetErrorOffset()))
                    .c_str());
          } else {
            response.set_name(model_metadata_json["name"].GetString());
            for (const auto& version :
                 model_metadata_json["versions"].GetArray()) {
              response.add_versions(version.GetString());
            }
            response.set_platform(model_metadata_json["platform"].GetString());

            for (const auto& io_json :
                 model_metadata_json["inputs"].GetArray()) {
              ModelMetadataResponse::TensorMetadata* io = response.add_inputs();
              io->set_name(io_json["name"].GetString());
              io->set_datatype(io_json["datatype"].GetString());
              for (const auto& d : io_json["shape"].GetArray()) {
                io->add_shape(d.GetInt());
              }
            }

            for (const auto& io_json :
                 model_metadata_json["outputs"].GetArray()) {
              ModelMetadataResponse::TensorMetadata* io =
                  response.add_outputs();
              io->set_name(io_json["name"].GetString());
              io->set_datatype(io_json["datatype"].GetString());
              for (const auto& d : io_json["shape"].GetArray()) {
                io->add_shape(d.GetInt());
              }
            }
          }
        }
        TRITONSERVER_MessageDelete(model_metadata_message);
      }
    }

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);

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
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, tritonserver, service, cq, max_state_bucket_count)
  {
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;
};

void
ModelConfigHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>();
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
    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request.version(), &requested_model_version);
    if (err == nullptr) {
      TRITONSERVER_Message* model_config_message = nullptr;
      err = TRITONSERVER_ServerModelConfig(
          tritonserver_.get(), request.name().c_str(), requested_model_version,
          &model_config_message);
      if (err == nullptr) {
        const char* buffer;
        size_t byte_size;
        err = TRITONSERVER_MessageSerializeToJson(
            model_config_message, &buffer, &byte_size);
        if (err == nullptr) {
          ::google::protobuf::util::JsonStringToMessage(
              {buffer, (int)byte_size}, response.mutable_config());
        }
        TRITONSERVER_MessageDelete(model_config_message);
      }
    }

    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);

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
TRITONSERVER_Error*
InferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload* payload = reinterpret_cast<AllocPayload*>(userp);
  ModelInferResponse* response = payload->response_;
  const AllocPayload::TensorShmMap* shm_map = payload->shm_map_;

  *buffer = nullptr;
  *buffer_userp = nullptr;
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // We add an output contents even if the 'byte_size' == 0 because we
  // expect to have a contents for every output.
  ModelInferResponse::InferOutputTensor* output_tensor =
      response->add_outputs();
  output_tensor->set_name(tensor_name);
  std::string* raw_output =
      output_tensor->mutable_contents()->mutable_raw_contents();

  if (byte_size > 0) {
    if (shm_map != nullptr) {
      const auto& pr = shm_map->find(tensor_name);
      if (pr != shm_map->end()) {
        // The output is in shared memory so check that shared memory
        // size is at least large enough for the output.
        if (byte_size > pr->second.byte_size_) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "shared memory size specified with the request for output '" +
                  std::string(tensor_name) + "' (" +
                  std::to_string(pr->second.byte_size_) +
                  " bytes) should be at least " + std::to_string(byte_size) +
                  " bytes to hold the results")
                  .c_str());
        }

        *buffer = const_cast<void*>(pr->second.base_);
        *actual_memory_type = pr->second.memory_type_;
        *actual_memory_type_id = pr->second.memory_type_id_;

        LOG_VERBOSE(1) << "GRPC: using shared-memory for '" << tensor_name
                       << "', size: " << byte_size << ", addr: " << *buffer;
        return nullptr;  // Success
      }
    }

    // Not using shared memory so allocate a buffer. The buffer we
    // create is directly in the response protobuf so we can't
    // allocate any type other than CPU.
    //
    // FIXME we could use pinned CPU memory here.
    if (*actual_memory_type != TRITONSERVER_MEMORY_CPU) {
      LOG_VERBOSE(1) << "GRPC: unable to provide '" << tensor_name << "' in "
                     << TRITONSERVER_MemoryTypeString(*actual_memory_type)
                     << ", will use "
                     << TRITONSERVER_MemoryTypeString(TRITONSERVER_MEMORY_CPU);
      *actual_memory_type = TRITONSERVER_MEMORY_CPU;
      *actual_memory_type_id = 0;
    }

    raw_output->resize(byte_size);
    *buffer = static_cast<void*>(&((*raw_output)[0]));

    LOG_VERBOSE(1) << "GRPC: using buffer for '" << tensor_name
                   << "', size: " << byte_size << ", addr: " << *buffer;
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
InferResponseFree(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "GRPC free: "
                 << "size " << byte_size << ", addr " << buffer;

  // Don't do anything when releasing a buffer since InferResponseAlloc
  // wrote directly into the response protobuf.
  return nullptr;  // Success
}

template <typename TensorType>
TRITONSERVER_Error*
ParseSharedMemoryParams(
    const TensorType& tensor, bool* has_shared_memory, std::string* region_name,
    int64_t* offset, size_t* byte_size)
{
  *has_shared_memory = false;
  *offset = 0 /* default value */;
  const auto& region_it = tensor.parameters().find("shared_memory_region");
  if (region_it != tensor.parameters().end()) {
    *has_shared_memory = true;
    const auto& infer_param = region_it->second;
    if (infer_param.parameter_choice_case() !=
        InferParameter::ParameterChoiceCase::kStringParam) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "invalid value type for 'shared_memory_region' parameter for "
              "tensor '" +
              tensor.name() + "', expected string_param.")
              .c_str());
    }
    *region_name = infer_param.string_param();
  }

  const auto& offset_it = tensor.parameters().find("shared_memory_offset");
  if (offset_it != tensor.parameters().end()) {
    if (!*has_shared_memory) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "'shared_memory_offset' can not be specified without "
              "'shared_memory_region' parameter for tensor '" +
              tensor.name() + "'")
              .c_str());
    }
    const auto& infer_param = offset_it->second;
    if (infer_param.parameter_choice_case() !=
        InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "invalid value type for 'shared_memory_offset' parameter for "
              "tensor '" +
              tensor.name() + "', expected int64_param.")
              .c_str());
    }
    *offset = infer_param.int64_param();
  }

  const auto& bs_it = tensor.parameters().find("shared_memory_byte_size");
  if (bs_it != tensor.parameters().end()) {
    if (!*has_shared_memory) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "'shared_memory_byte_size' can not be specified without "
              "'shared_memory_region' parameter for tensor '" +
              tensor.name() + "'")
              .c_str());
    }
    const auto& infer_param = bs_it->second;
    if (infer_param.parameter_choice_case() !=
        InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "invalid value type for 'shared_memory_byte_size' parameter for "
              "tensor '" +
              tensor.name() + "', expected int64_param.")
              .c_str());
    }
    *byte_size = infer_param.int64_param();
  } else {
    if (*has_shared_memory) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "'shared_memory_byte_size' must be specified along with "
              "'shared_memory_region' parameter for tensor '" +
              tensor.name() + "'")
              .c_str());
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
InferAllocatorPayload(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const ModelInferRequest& request,
    AllocPayload::TensorSerializedDataMap* serialized_data_map,
    ModelInferResponse* response, AllocPayload* alloc_payload)
{
  alloc_payload->response_ = response;
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
  for (const auto& io : request.outputs()) {
    std::string region_name;
    int64_t offset;
    size_t byte_size;
    bool has_shared_memory;

    RETURN_IF_ERR(
        ParseSharedMemoryParams<ModelInferRequest::InferRequestedOutputTensor>(
            io, &has_shared_memory, &region_name, &offset, &byte_size));

    if (has_shared_memory) {
      void* base;
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      RETURN_IF_ERR(shm_manager->GetMemoryInfo(
          region_name, offset, &base, &memory_type, &memory_type_id));

      // if shm_map_ does not exist, then create an empty shm_map
      if (alloc_payload->shm_map_ == nullptr) {
        alloc_payload->shm_map_ = new AllocPayload::TensorShmMap;
      }

      alloc_payload->shm_map_->emplace(
          io.name(),
          AllocPayload::ShmInfo{base, byte_size, memory_type, memory_type_id});
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
InferGRPCToInputHelper(
    const std::string& input_name, const std::string& model_name,
    const TRITONSERVER_DataType tensor_dt, const TRITONSERVER_DataType input_dt,
    const size_t binary_data_byte_size)
{
  if (binary_data_byte_size != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected explicit tensor data for input tensor '" + input_name +
            "' for model '" + model_name +
            "', binary data was already supplied.")
            .c_str());
  }

  if (tensor_dt != input_dt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected explicit tensor data for input tensor '" + input_name +
            "' for model '" + model_name + "' of type '" +
            TRITONSERVER_DataTypeString(tensor_dt) + "', expected datatype '" +
            TRITONSERVER_DataTypeString(input_dt) + "'")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
InferGRPCToInput(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const ModelInferRequest& request,
    AllocPayload::TensorSerializedDataMap* serialized_data_map,
    TRITONSERVER_InferenceRequest* inference_request)
{
  // Verify that the batch-byte-size of each input matches the size of
  // the provided tensor data (provided raw or from shared memory)
  for (const auto& io : request.inputs()) {
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;

    std::string region_name;
    int64_t offset;
    bool has_shared_memory;
    RETURN_IF_ERR(ParseSharedMemoryParams<ModelInferRequest::InferInputTensor>(
        io, &has_shared_memory, &region_name, &offset, &byte_size));

    if (has_shared_memory) {
      void* tmp;
      RETURN_IF_ERR(shm_manager->GetMemoryInfo(
          region_name, offset, &tmp, &memory_type, &memory_type_id));
      base = tmp;
    } else {
      if (!io.has_contents()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
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
      TRITONSERVER_DataType dtype =
          TRITONSERVER_StringToDataType(io.datatype().c_str());
      const size_t elem_byte_size = TRITONSERVER_DataTypeByteSize(dtype);
      if (io.contents().bool_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), TRITONSERVER_TYPE_BOOL, dtype,
            byte_size));
        base = (const void*)io.contents().bool_contents().data();
        byte_size = io.contents().bool_contents_size() * elem_byte_size;
      }

      if (io.contents().int_contents_size() != 0) {
        if (dtype == TRITONSERVER_TYPE_INT8) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_INT8, dtype,
              byte_size));
          std::shared_ptr<std::string> serialized(new std::string());
          serialized->reserve(
              io.contents().int_contents_size() * elem_byte_size);
          serialized_data_map->emplace(io.name(), serialized);
          for (const auto& element : io.contents().int_contents()) {
            // Assuming the system is little-endian, picking the
            // least significant byte of 32-bit integer as a
            // int8 element
            serialized->append(
                reinterpret_cast<const char*>(&element), elem_byte_size);
          }
          base = serialized->c_str();
          byte_size = serialized->size();
        } else if (dtype == TRITONSERVER_TYPE_INT16) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_INT16, dtype,
              byte_size));
          std::shared_ptr<std::string> serialized(new std::string());
          serialized->reserve(
              io.contents().int_contents_size() * elem_byte_size);
          serialized_data_map->emplace(io.name(), serialized);
          for (const auto& element : io.contents().int_contents()) {
            // Assuming the system is little-endian, picking the
            // least 2 significant bytes of 32-bit integer as a
            // int16 element
            serialized->append(
                reinterpret_cast<const char*>(&element), elem_byte_size);
          }
          base = serialized->c_str();
          byte_size = serialized->size();
        } else {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_INT32, dtype,
              byte_size));
          base = (const void*)io.contents().int_contents().data();
          byte_size = io.contents().int_contents_size() * elem_byte_size;
        }
      }

      if (io.contents().int64_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), TRITONSERVER_TYPE_INT64, dtype,
            byte_size));
        base = (const void*)io.contents().int64_contents().data();
        byte_size = io.contents().int64_contents_size() * elem_byte_size;
      }

      if (io.contents().uint_contents_size() != 0) {
        if (dtype == TRITONSERVER_TYPE_UINT8) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_UINT8, dtype,
              byte_size));
          std::shared_ptr<std::string> serialized(new std::string());
          serialized_data_map->emplace(io.name(), serialized);
          serialized->reserve(
              io.contents().uint_contents_size() * elem_byte_size);
          for (const auto& element : io.contents().uint_contents()) {
            // Assuming the system is little-endian, picking the
            // least significant byte of 32-bit unsigned integer as a
            // uint8 element
            serialized->append(
                reinterpret_cast<const char*>(&element), elem_byte_size);
          }
          base = serialized->c_str();
          byte_size = serialized->size();
        } else if (dtype == TRITONSERVER_TYPE_UINT16) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_UINT16, dtype,
              byte_size));
          std::shared_ptr<std::string> serialized(new std::string());
          serialized_data_map->emplace(io.name(), serialized);
          serialized->reserve(
              io.contents().uint_contents_size() * elem_byte_size);
          for (const auto& element : io.contents().uint_contents()) {
            // Assuming the system is little-endian, picking the
            // least 2 significant bytes of 32-bit integer as a
            // uint16 element
            serialized->append(
                reinterpret_cast<const char*>(&element), elem_byte_size);
          }
          base = serialized->c_str();
          byte_size = serialized->size();
        } else {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_UINT32, dtype,
              byte_size));
          base = (const void*)io.contents().int_contents().data();
          byte_size = io.contents().int_contents_size() * elem_byte_size;
        }
      }

      if (io.contents().uint64_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), TRITONSERVER_TYPE_UINT64, dtype,
            byte_size));
        base = (const void*)io.contents().uint64_contents().data();
        byte_size = io.contents().uint64_contents_size() * elem_byte_size;
      }

      if (io.contents().fp32_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), TRITONSERVER_TYPE_FP32, dtype,
            byte_size));
        base = (const void*)io.contents().fp32_contents().data();
        byte_size = io.contents().fp32_contents_size() * elem_byte_size;
      }

      if (io.contents().fp64_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), TRITONSERVER_TYPE_FP64, dtype,
            byte_size));
        base = (const void*)io.contents().fp64_contents().data();
        byte_size = io.contents().fp64_contents_size() * elem_byte_size;
      }

      if (io.contents().byte_contents_size() != 0) {
        RETURN_IF_ERR(InferGRPCToInputHelper(
            io.name(), request.model_name(), TRITONSERVER_TYPE_BYTES, dtype,
            byte_size));
        std::shared_ptr<std::string> serialized(new std::string());
        serialized_data_map->emplace(io.name(), serialized);

        // Serialize the output tensor strings. Each string is
        // serialized as a 4-byte length followed by the string itself
        // with no null-terminator.
        for (const auto& element : io.contents().byte_contents()) {
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

    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
        inference_request, io.name().c_str(), base, byte_size, memory_type,
        memory_type_id));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
SetInferenceRequestMetadata(
    TRITONSERVER_InferenceRequest* inference_request,
    const ModelInferRequest& request)
{
  RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetId(
      inference_request, request.id().c_str()));

  // FIXME, instead of find perhaps we should just iterate through the
  // parameters...
  const auto& sequence_id_it = request.parameters().find("sequence_id");
  if (sequence_id_it != request.parameters().end()) {
    const auto& infer_param = sequence_id_it->second;
    if (infer_param.parameter_choice_case() !=
        InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "invalid value type for 'sequence_id' parameter, expected "
          "int64_param.");
    }
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetCorrelationId(
        inference_request, infer_param.int64_param()));
    uint32_t flags = TRITONSERVER_REQUEST_FLAG_NONE;
    const auto& sequence_start_it = request.parameters().find("sequence_start");
    if (sequence_start_it != request.parameters().end()) {
      const auto& infer_param = sequence_start_it->second;
      if (infer_param.parameter_choice_case() !=
          InferParameter::ParameterChoiceCase::kBoolParam) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'sequence_start' parameter, expected "
            "bool_param.");
      }
      flags |=
          infer_param.bool_param() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
    }
    const auto& sequence_end_it = request.parameters().find("sequence_end");
    if (sequence_end_it != request.parameters().end()) {
      const auto& infer_param = sequence_end_it->second;
      if (infer_param.parameter_choice_case() !=
          InferParameter::ParameterChoiceCase::kBoolParam) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'sequence_end' parameter, expected "
            "bool_param.");
      }
      flags |=
          infer_param.bool_param() & TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
    }
    RETURN_IF_ERR(
        TRITONSERVER_InferenceRequestSetFlags(inference_request, flags));
  }

  const auto& priority_it = request.parameters().find("priority");
  if (priority_it != request.parameters().end()) {
    const auto& infer_param = priority_it->second;
    if (infer_param.parameter_choice_case() !=
        InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "invalid value type for 'sequence_id' parameter, expected "
          "int64_param.");
    }
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetPriority(
        inference_request, infer_param.int64_param()));
  }

  const auto& timeout_it = request.parameters().find("timeout");
  if (timeout_it != request.parameters().end()) {
    const auto& infer_param = timeout_it->second;
    if (infer_param.parameter_choice_case() !=
        InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "invalid value type for 'sequence_id' parameter, expected "
          "int64_param.");
    }
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
        inference_request, infer_param.int64_param()));
  }

  for (const auto& input : request.inputs()) {
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
        inference_request, input.name().c_str(),
        TRITONSERVER_StringToDataType(input.datatype().c_str()),
        input.shape().data(), input.shape_size()));
  }

  for (const auto& output : request.outputs()) {
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(
        inference_request, output.name().c_str()));

    const auto& class_it = output.parameters().find("classification");
    if (class_it != output.parameters().end()) {
      const auto& infer_param = class_it->second;
      if (infer_param.parameter_choice_case() !=
          InferParameter::ParameterChoiceCase::kInt64Param) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "invalid value type for 'classification' parameter, expected "
            "int64_param.");
      }
      RETURN_IF_ERR(
          TRITONSERVER_InferenceRequestSetRequestedOutputClassificationCount(
              inference_request, output.name().c_str(),
              infer_param.int64_param()));
    }
  }

  return nullptr;  // Success
}

void
TraceManagerComplete(TRITONSERVER_TraceManager* trace_manager, void* userp)
{
  LOG_VERBOSE(1) << "ModelInferHandler::TraceManagerComplete";

  // FIXME need to sort out trace manager handling
}

void
InferRequestComplete(TRITONSERVER_InferenceRequest* request, void* userp)
{
  LOG_VERBOSE(1) << "ModelInferHandler::InferRequestComplete";

  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceRequestDelete(request),
      "deleting GRPC inference request");
}

TRITONSERVER_Error*
InferResponseCompleteCommon(
    TRITONSERVER_InferenceResponse* iresponse, ModelInferResponse& response,
    const AllocPayload& alloc_payload)
{
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseError(iresponse));

  const char *model_name, *id;
  int64_t model_version;
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseModel(
      iresponse, &model_name, &model_version));
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseId(iresponse, &id));

  response.set_id(id);
  response.set_model_name(model_name);
  response.set_model_version(std::to_string(model_version));

  // Go through each response output and transfer information to the
  // corresponding GRPC response output.
  uint32_t output_count;
  RETURN_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(iresponse, &output_count));
  if (output_count != (uint32_t)response.outputs_size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "response output count mismatch");
  }

  for (uint32_t idx = 0; idx < output_count; ++idx) {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;

    RETURN_IF_ERR(TRITONSERVER_InferenceResponseOutput(
        iresponse, idx, &cname, &datatype, &shape, &dim_count, &base,
        &byte_size, &memory_type, &memory_type_id));

    const std::string name(cname);

    // There are usually very few outputs so fastest just to look for
    // the one we want... could create a map for cases where there are
    // a large number of outputs. Or rely on order to be same...
    ModelInferResponse::InferOutputTensor* output = nullptr;
    for (auto& io : *(response.mutable_outputs())) {
      if (io.name() == name) {
        output = &io;
        break;
      }
    }

    if (output == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "unable to find expected response output");
    }

    output->set_datatype(TRITONSERVER_DataTypeString(datatype));
    for (size_t idx = 0; idx < dim_count; idx++) {
      output->add_shape(shape[idx]);
    }

#if 0
        // FIXMEV2, different handling if the output requested
        // classification results
        //
        // Check if the output is classification results
        // (no raw_contents and not using shared memory)
        if (output.contents().raw_contents().size() == 0) {
          if ((alloc_payload.shm_map_ == nullptr) ||
              (alloc_payload.shm_map_->find(output.name()) ==
               alloc_payload.shm_map_->end())) {
            size_t element_cnt = shape[0] * shape[1];
            const char* base;
            size_t byte_size;
            TRITONSERVER_MemoryType mem_type;
            int64_t mem_id;
            err = TRITONSERVER_InferenceResponseOutputData(
                iresponse, output.name().c_str(), (const void**)&base,
                &byte_size, &mem_type, &mem_id);
            if (err != nullptr) {
              break;
            }
            size_t offset = 0;
            for (size_t idx = 0; idx < element_cnt; idx++) {
              size_t length =
                  *(reinterpret_cast<const uint32_t*>(base + offset));
              offset += sizeof(uint32_t);
              output.mutable_contents()->add_byte_contents(
                  base + offset, length);
              offset += length;
            }
          }
        }
#endif
  }

  // Make sure response doesn't exceed GRPC limits.
  if (response.ByteSizeLong() > MAX_GRPC_MESSAGE_SIZE) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "Response has byte size " +
            std::to_string(response.ByteSizeLong()) +
            " which exceeds gRPC's byte size limit " + std::to_string(INT_MAX) +
            ".")
            .c_str());
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
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      const std::shared_ptr<TraceManager>& trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, tritonserver, service, cq, max_state_bucket_count),
        trace_manager_(trace_manager), shm_manager_(shm_manager)
  {
    // Create the allocator that will be used to allocate buffers for
    // the result tensors.
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator_, InferResponseAlloc, InferResponseFree),
        "creating inference response allocator");
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;

 private:
  static void InferResponseComplete(
      TRITONSERVER_InferenceResponse* response, void* userp);

  std::shared_ptr<TraceManager> trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  TRITONSERVER_ResponseAllocator* allocator_;
};

void
ModelInferHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>();
  State* state = StateNew(context);

#ifdef TRTIS_ENABLE_TRACING
  if (trace_manager_ != nullptr) {
    state->trace_meta_data_.reset(trace_manager_->SampleTrace());
    if (state->trace_meta_data_ != nullptr) {
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRITONSERVER_TRACE_LEVEL_MIN, "grpc wait/read start");
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
    TRITONSERVER_Error* err = nullptr;
#ifdef TRTIS_ENABLE_TRACING
    if (state->trace_meta_data_ != nullptr) {
      int64_t requested_model_version;
      err = GetModelVersionFromString(
          request.model_version(), &requested_model_version);
      if (err == nullptr) {
        state->trace_meta_data_->tracer_->SetModel(
            request.model_name(), requested_model_version);
      } else {
        // If failed to retrieve the requested_model_version
        // then use the default model version just to record
        // the timestamps in the tracer
        state->trace_meta_data_->tracer_->SetModel(request.model_name(), -1);
      }
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRITONSERVER_TRACE_LEVEL_MIN, "grpc wait/read end");
    }
#endif  // TRTIS_ENABLE_TRACING

    // Start a new request to replace this one...
    if (!shutdown) {
      StartNewRequest();
    }
    // Create the inference request which contains all the
    // input information needed for an inference.
    TRITONSERVER_InferenceRequest* irequest = nullptr;
    if (err == nullptr) {
      int64_t requested_model_version;
      err = GetModelVersionFromString(
          request.model_version(), &requested_model_version);
      if (err == nullptr) {
        err = TRITONSERVER_InferenceRequestNew(
            &irequest, tritonserver_.get(), request.model_name().c_str(),
            requested_model_version);
      }
    }

    if (err == nullptr) {
      err = SetInferenceRequestMetadata(irequest, request);
    }

    // Will be used to hold the serialized data in case explicit string
    // tensors are present in the request.
    AllocPayload::TensorSerializedDataMap* serialized_data_map =
        new AllocPayload::TensorSerializedDataMap();

    if (err == nullptr) {
      err = InferGRPCToInput(
          tritonserver_, shm_manager_, request, serialized_data_map, irequest);
    }
    if (err == nullptr) {
      err = InferAllocatorPayload(
          tritonserver_, shm_manager_, request, serialized_data_map, &response,
          &state->alloc_payload_);
    }
    if (err == nullptr) {
      err = TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest, InferRequestComplete, nullptr /* request_release_userp */);
    }
    if (err == nullptr) {
      err = TRITONSERVER_InferenceRequestSetResponseCallback(
          irequest, allocator_,
          &state->alloc_payload_ /* response_allocator_userp */,
          InferResponseComplete, reinterpret_cast<void*>(state));
    }
    if (err == nullptr) {
      // Provide the trace manager object to use for this request, if
      // nullptr then no tracing will be performed.
      TRITONSERVER_TraceManager* trace_manager = nullptr;
#ifdef TRTIS_ENABLE_TRACING
      if (state->trace_meta_data_ != nullptr) {
        TRITONSERVER_TraceManagerNew(
            &trace_manager, TraceManager::CreateTrace,
            TraceManager::ReleaseTrace, state->trace_meta_data_.get());
      }
#endif  // TRTIS_ENABLE_TRACING

      state->step_ = ISSUED;

      err = TRITONSERVER_ServerInferAsync(
          tritonserver_.get(), irequest, trace_manager, TraceManagerComplete,
          nullptr /* trace_release_userp */);
    }

    // If not error then state->step_ == ISSUED and inference request
    // has initiated... completion callback will transition to
    // COMPLETE. If error go immediately to COMPLETE.
    if (err != nullptr) {
      LOG_VERBOSE(1) << "Infer failed: " << TRITONSERVER_ErrorMessage(err);

      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_InferenceRequestDelete(irequest),
          "deleting GRPC inference request");

      grpc::Status status;
      GrpcStatusUtil::Create(&status, err);
      TRITONSERVER_ErrorDelete(err);

      response.Clear();

#ifdef TRTIS_ENABLE_TRACING
      if (state->trace_meta_data_ != nullptr) {
        state->trace_meta_data_->tracer_->CaptureTimestamp(
            TRITONSERVER_TRACE_LEVEL_MIN, "grpc send start");
      }
#endif  // TRTIS_ENABLE_TRACING

      state->step_ = COMPLETE;
      state->context_->responder_->Finish(response, status, state);
    }
  } else if (state->step_ == Steps::COMPLETE) {
#ifdef TRTIS_ENABLE_TRACING
    if (state->trace_meta_data_ != nullptr) {
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRITONSERVER_TRACE_LEVEL_MIN, "grpc send end");
    }
#endif  // TRTIS_ENABLE_TRACING

    state->step_ = Steps::FINISH;
    finished = true;
  }

  return !finished;
}

void
ModelInferHandler::InferResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, void* userp)
{
  State* state = reinterpret_cast<State*>(userp);

  LOG_VERBOSE(1) << "ModelInferHandler::InferResponseComplete, "
                 << state->unique_id_ << " step " << state->step_;


  ModelInferResponse& response = state->response_;
  TRITONSERVER_Error* err =
      InferResponseCompleteCommon(iresponse, response, state->alloc_payload_);

  if (err != nullptr) {
    response.Clear();
  }

  grpc::Status status;
  GrpcStatusUtil::Create(&status, err);
  TRITONSERVER_ErrorDelete(err);

  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceResponseDelete(iresponse),
      "deleting GRPC inference response");

  // FIXME
  // Don't need to explicitly delete 'trace_manager'. It will be deleted by
  // the TraceMetaData object in 'state'.
#ifdef TRTIS_ENABLE_TRACING
  if (state->trace_meta_data_ != nullptr) {
    state->trace_meta_data_->tracer_->CaptureTimestamp(
        TRITONSERVER_TRACE_LEVEL_MIN, "grpc send start");
  }
#endif  // TRTIS_ENABLE_TRACING

  state->step_ = COMPLETE;
  state->context_->responder_->Finish(response, status, state);
}

//
// ModelStreamInferHandler
//
class ModelStreamInferHandler
    : public Handler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncReaderWriter<
              ModelStreamInferResponse, ModelInferRequest>,
          ModelInferRequest, ModelStreamInferResponse> {
 public:
  ModelStreamInferHandler(
      const std::string& name,
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      const std::shared_ptr<TraceManager>& trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : Handler(name, tritonserver, service, cq, max_state_bucket_count),
        trace_manager_(trace_manager), shm_manager_(shm_manager)
  {
    // Create the allocator that will be used to allocate buffers for
    // the result tensors.
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator_, InferResponseAlloc, InferResponseFree),
        "creating response allocator");
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;

 private:
  static void StreamInferResponseComplete(
      TRITONSERVER_InferenceResponse* response, void* userp);

  std::shared_ptr<TraceManager> trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  TRITONSERVER_ResponseAllocator* allocator_;
};

void
ModelStreamInferHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(NEXT_UNIQUE_ID);
  State* state = StateNew(context);

#ifdef TRTIS_ENABLE_TRACING
  if (trace_manager_ != nullptr) {
    state->trace_meta_data_.reset(trace_manager_->SampleTrace());
    if (state->trace_meta_data_ != nullptr) {
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRITONSERVER_TRACE_LEVEL_MIN, "grpc wait/read start");
    }
  }
#endif  // TRTIS_ENABLE_TRACING

  service_->RequestModelStreamInfer(
      state->context_->ctx_.get(), state->context_->responder_.get(), cq_, cq_,
      state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ModelStreamInferHandler::Process(Handler::State* state, bool rpc_ok)
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
    TRITONSERVER_Error* err = nullptr;
    const ModelInferRequest& request = state->request_;
#ifdef TRTIS_ENABLE_TRACING
    if (state->trace_meta_data_ != nullptr) {
      int64_t requested_model_version;
      err = GetModelVersionFromString(
          request.model_version(), &requested_model_version);
      if (err == nullptr) {
        state->trace_meta_data_->tracer_->SetModel(
            state->request_.model_name(), requested_model_version);
      } else {
        // If failed to retrieve the requested_model_version
        // then use the default model version just to record
        // the timestamps in the tracer
        state->trace_meta_data_->tracer_->SetModel(
            state->request_.model_name(), -1);
      }
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRITONSERVER_TRACE_LEVEL_MIN, "grpc wait/read end");
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
    ModelStreamInferResponse& response = state->response_;

    // Create the inference request which contains all the
    // input information needed for an inference.
    TRITONSERVER_InferenceRequest* irequest = nullptr;
    if (err == nullptr) {
      int64_t requested_model_version;
      err = GetModelVersionFromString(
          request.model_version(), &requested_model_version);
      if (err == nullptr) {
        err = TRITONSERVER_InferenceRequestNew(
            &irequest, tritonserver_.get(), request.model_name().c_str(),
            requested_model_version);
      }
    }

    if (err == nullptr) {
      err = SetInferenceRequestMetadata(irequest, request);
    }

    // Will be used to hold the serialized data in case explicit string
    // tensors are present in the request.
    AllocPayload::TensorSerializedDataMap* serialized_data_map =
        new AllocPayload::TensorSerializedDataMap();

    if (err == nullptr) {
      err = InferGRPCToInput(
          tritonserver_, shm_manager_, request, serialized_data_map, irequest);
    }
    if (err == nullptr) {
      err = InferAllocatorPayload(
          tritonserver_, shm_manager_, request, serialized_data_map,
          response.mutable_infer_response(), &state->alloc_payload_);
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
      // Provide the trace manager object to use for this request, if
      // nullptr then no tracing will be performed.
      TRITONSERVER_TraceManager* trace_manager = nullptr;
#ifdef TRTIS_ENABLE_TRACING
      if (state->trace_meta_data_ != nullptr) {
        TRITONSERVER_TraceManagerNew(
            &trace_manager, TraceManager::CreateTrace,
            TraceManager::ReleaseTrace, state->trace_meta_data_.get());
      }
#endif  // TRTIS_ENABLE_TRACING

      state->step_ = ISSUED;

      err = TRITONSERVER_ServerInferAsync(
          tritonserver_.get(), irequest, trace_manager, TraceManagerComplete,
          nullptr /* trace_release_userp */);
    }

    // If there was not an error in issuing the 'state' request then
    // state->step_ == ISSUED and inference request has
    // initiated... the completion callback will transition to
    // WRITEREADY or WRITTEN. If there was an error then enqueue the
    // error response and show it to be ready for writing.
    if (err != nullptr) {
      LOG_VERBOSE(1) << "Infer failed: " << TRITONSERVER_ErrorMessage(err);

      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_InferenceRequestDelete(irequest),
          "deleting GRPC inference request");

      grpc::Status status;
      GrpcStatusUtil::Create(&status, err);
      TRITONSERVER_ErrorDelete(err);
      response.set_error_message(status.error_message());

      response.mutable_infer_response()->Clear();

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
            TRITONSERVER_TRACE_LEVEL_MIN, "grpc wait/read start");
      }
    }
#endif  // TRTIS_ENABLE_TRACING

    next_read_state->context_->responder_->Read(
        &next_read_state->request_, next_read_state);

  } else if (state->step_ == Steps::WRITTEN) {
#ifdef TRTIS_ENABLE_TRACING
    if (state->trace_meta_data_ != nullptr) {
      state->trace_meta_data_->tracer_->CaptureTimestamp(
          TRITONSERVER_TRACE_LEVEL_MIN, "grpc send end");
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
ModelStreamInferHandler::StreamInferResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, void* userp)
{
  State* state = reinterpret_cast<State*>(userp);

  LOG_VERBOSE(1) << "ModelStreamInferHandler::StreamInferComplete, context "
                 << state->context_->unique_id_ << ", " << state->unique_id_
                 << " step " << state->step_;

  ModelInferResponse& response = *(state->response_.mutable_infer_response());
  TRITONSERVER_Error* err =
      InferResponseCompleteCommon(iresponse, response, state->alloc_payload_);

  if (err != nullptr) {
    grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    state->response_.Clear();
    state->response_.set_error_message(status.error_message());
  }

  TRITONSERVER_ErrorDelete(err);

  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceResponseDelete(iresponse),
      "deleting GRPC inference response");

  // FIXME
  // Don't need to explicitly delete 'trace_manager'. It will be deleted by
  // the TraceMetaData object in 'state'.

  state->step_ = Steps::WRITEREADY;
  state->context_->WriteResponseIfReady(state);
}

}  // namespace

//
// GRPCServerV2
//
GRPCServerV2::GRPCServerV2(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const std::string& server_addr, const int infer_allocation_pool_size)
    : server_(server), trace_manager_(trace_manager), shm_manager_(shm_manager),
      server_addr_(server_addr),
      infer_allocation_pool_size_(infer_allocation_pool_size), running_(false)
{
}

GRPCServerV2::~GRPCServerV2()
{
  Stop();
}

TRITONSERVER_Error*
GRPCServerV2::Create(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager, int32_t port,
    int infer_allocation_pool_size, std::unique_ptr<GRPCServerV2>* grpc_server)
{
  const std::string addr = "0.0.0.0:" + std::to_string(port);
  grpc_server->reset(new GRPCServerV2(
      server, trace_manager, shm_manager, addr, infer_allocation_pool_size));

  return nullptr;  // success
}

TRITONSERVER_Error*
GRPCServerV2::Start()
{
  if (running_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS, "GRPC server is already running.");
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
  common_cq_ = grpc_builder_.AddCompletionQueue();
  model_stream_infer_cq_ = grpc_builder_.AddCompletionQueue();
  grpc_server_ = grpc_builder_.BuildAndStart();

  // Handler for server-live requests.
  ServerLiveHandler* hserverlive = new ServerLiveHandler(
      "ServerLiveHandler", server_, &service_, server_live_cq_.get(),
      2 /* max_state_bucket_count */);
  hserverlive->Start();
  server_live_handler_.reset(hserverlive);

  // Handler for server-ready requests.
  ServerReadyHandler* hserverready = new ServerReadyHandler(
      "ServerReadyHandler", server_, &service_, server_ready_cq_.get(),
      2 /* max_state_bucket_count */);
  hserverready->Start();
  server_ready_handler_.reset(hserverready);

  // Handler for model-ready requests.
  ModelReadyHandler* hmodelready = new ModelReadyHandler(
      "ModelReadyHandler", server_, &service_, model_ready_cq_.get(),
      2 /* max_state_bucket_count */);
  hmodelready->Start();
  model_ready_handler_.reset(hmodelready);

  // Handler for server-metadata requests.
  ServerMetadataHandler* hservermetadata = new ServerMetadataHandler(
      "ServerMetadataHandler", server_, &service_, server_metadata_cq_.get(),
      2 /* max_state_bucket_count */);
  hservermetadata->Start();
  server_metadata_handler_.reset(hservermetadata);

  // Handler for model-metadata requests.
  ModelMetadataHandler* hmodelmetadata = new ModelMetadataHandler(
      "ModelMetadataHandler", server_, &service_, model_metadata_cq_.get(),
      2 /* max_state_bucket_count */);
  hmodelmetadata->Start();
  model_metadata_handler_.reset(hmodelmetadata);

  // Handler for model-config requests.
  ModelConfigHandler* hmodelconfig = new ModelConfigHandler(
      "ModelConfigHandler", server_, &service_, model_config_cq_.get(),
      2 /* max_state_bucket_count */);
  hmodelconfig->Start();
  model_config_handler_.reset(hmodelconfig);

  // Handler for model inference requests.
  ModelInferHandler* hmodelinfer = new ModelInferHandler(
      "ModelInferHandler", server_, trace_manager_, shm_manager_, &service_,
      model_infer_cq_.get(),
      infer_allocation_pool_size_ /* max_state_bucket_count */);
  hmodelinfer->Start();
  model_infer_handler_.reset(hmodelinfer);

  // Handler for streaming inference requests.
  ModelStreamInferHandler* hmodelstreaminfer = new ModelStreamInferHandler(
      "ModelStreamInferHandler", server_, trace_manager_, shm_manager_,
      &service_, model_stream_infer_cq_.get(),
      infer_allocation_pool_size_ /* max_state_bucket_count */);
  hmodelstreaminfer->Start();
  model_stream_infer_handler_.reset(hmodelstreaminfer);

  // A common Handler for other non-critical requests
  CommonHandler* hcommon = new CommonHandler(
      "CommonHandler", server_, shm_manager_, &service_, common_cq_.get());
  hcommon->Start();
  common_handler_.reset(hcommon);

  running_ = true;
  LOG_INFO << "Started GRPCInferenceService at " << server_addr_;
  return nullptr;  // success
}

TRITONSERVER_Error*
GRPCServerV2::Stop()
{
  if (!running_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE, "GRPC server is not running.");
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
  common_cq_->Shutdown();
  model_stream_infer_cq_->Shutdown();

  // Must stop all handlers explicitly to wait for all the handler
  // threads to join since they are referencing completion queue, etc.
  dynamic_cast<ServerLiveHandler*>(server_live_handler_.get())->Stop();
  dynamic_cast<ServerReadyHandler*>(server_ready_handler_.get())->Stop();
  dynamic_cast<ModelReadyHandler*>(model_ready_handler_.get())->Stop();
  dynamic_cast<ServerMetadataHandler*>(server_metadata_handler_.get())->Stop();
  dynamic_cast<ModelMetadataHandler*>(model_metadata_handler_.get())->Stop();
  dynamic_cast<ModelConfigHandler*>(model_config_handler_.get())->Stop();
  dynamic_cast<ModelInferHandler*>(model_infer_handler_.get())->Stop();
  dynamic_cast<CommonHandler*>(common_handler_.get())->Stop();
  dynamic_cast<ModelStreamInferHandler*>(model_stream_infer_handler_.get())
      ->Stop();

  running_ = false;
  return nullptr;  // success
}

}}  // namespace nvidia::inferenceserver
