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

#include "src/servers/grpc_server.h"

#include <grpc++/alarm.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <list>
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
#include "src/core/tritonserver.h"
#include "src/servers/classification.h"
#include "src/servers/common.h"

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "src/core/json.h"

#ifdef TRITON_ENABLE_TRACING
#include "src/servers/tracer.h"
#endif  // TRITON_ENABLE_TRACING

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
#endif  // NDEBUG

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
// The server has separate handling mechanisms for inference RPCs
// and non-inference RPCs.
//

//=========================================================================
//  The following section contains the handling mechanism for non-inference
//  RPCs. A single thread is created to handle all these requests as they
//  are deemed to be not performance critical.
//=========================================================================

template <typename ResponderType, typename RequestType, typename ResponseType>
class CommonCallData : public GRPCServer::ICallData {
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
// A common handler for all non-inference requests.
//
class CommonHandler : public GRPCServer::HandlerBase {
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
      GRPCServer::ICallData* call_data =
          static_cast<GRPCServer::ICallData*>(tag);
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
  //  ServerLive
  //
  auto OnRegisterServerLive =
      [this](
          grpc::ServerContext* ctx, ServerLiveRequest* request,
          grpc::ServerAsyncResponseWriter<ServerLiveResponse>* responder,
          void* tag) {
        this->service_->RequestServerLive(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteServerLive = [this](
                                 ServerLiveRequest& request,
                                 ServerLiveResponse* response,
                                 grpc::Status* status) {
    bool live = false;
    TRITONSERVER_Error* err =
        TRITONSERVER_ServerIsLive(tritonserver_.get(), &live);

    response->set_live((err == nullptr) && live);

    GrpcStatusUtil::Create(status, err);
    TRITONSERVER_ErrorDelete(err);
  };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<ServerLiveResponse>, ServerLiveRequest,
      ServerLiveResponse>(
      "ServerLive", 0, OnRegisterServerLive, OnExecuteServerLive);

  //
  //  ServerReady
  //
  auto OnRegisterServerReady =
      [this](
          grpc::ServerContext* ctx, ServerReadyRequest* request,
          grpc::ServerAsyncResponseWriter<ServerReadyResponse>* responder,
          void* tag) {
        this->service_->RequestServerReady(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteServerReady = [this](
                                  ServerReadyRequest& request,
                                  ServerReadyResponse* response,
                                  grpc::Status* status) {
    bool ready = false;
    TRITONSERVER_Error* err =
        TRITONSERVER_ServerIsReady(tritonserver_.get(), &ready);

    response->set_ready((err == nullptr) && ready);

    GrpcStatusUtil::Create(status, err);
    TRITONSERVER_ErrorDelete(err);
  };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<ServerReadyResponse>, ServerReadyRequest,
      ServerReadyResponse>(
      "ServerReady", 0, OnRegisterServerReady, OnExecuteServerReady);

  //
  //  ModelReady
  //
  auto OnRegisterModelReady =
      [this](
          grpc::ServerContext* ctx, ModelReadyRequest* request,
          grpc::ServerAsyncResponseWriter<ModelReadyResponse>* responder,
          void* tag) {
        this->service_->RequestModelReady(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteModelReady = [this](
                                 ModelReadyRequest& request,
                                 ModelReadyResponse* response,
                                 grpc::Status* status) {
    bool is_ready = false;
    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request.version(), &requested_model_version);
    if (err == nullptr) {
      err = TRITONSERVER_ServerModelIsReady(
          tritonserver_.get(), request.name().c_str(), requested_model_version,
          &is_ready);
    }

    response->set_ready(is_ready);

    GrpcStatusUtil::Create(status, err);
    TRITONSERVER_ErrorDelete(err);
  };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<ModelReadyResponse>, ModelReadyRequest,
      ModelReadyResponse>(
      "ModelReady", 0, OnRegisterModelReady, OnExecuteModelReady);

  //
  //  ServerMetadata
  //
  auto OnRegisterServerMetadata =
      [this](
          grpc::ServerContext* ctx, ServerMetadataRequest* request,
          grpc::ServerAsyncResponseWriter<ServerMetadataResponse>* responder,
          void* tag) {
        this->service_->RequestServerMetadata(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteServerMetadata = [this](
                                     ServerMetadataRequest& request,
                                     ServerMetadataResponse* response,
                                     grpc::Status* status) {
    TRITONSERVER_Message* server_metadata_message = nullptr;
    TRITONSERVER_Error* err = TRITONSERVER_ServerMetadata(
        tritonserver_.get(), &server_metadata_message);
    GOTO_IF_ERR(err, earlyexit);

    const char* buffer;
    size_t byte_size;
    err = TRITONSERVER_MessageSerializeToJson(
        server_metadata_message, &buffer, &byte_size);
    GOTO_IF_ERR(err, earlyexit);

    {
      TritonJson::Value server_metadata_json;
      err = server_metadata_json.Parse(buffer, byte_size);
      GOTO_IF_ERR(err, earlyexit);

      const char* name;
      size_t namelen;
      err = server_metadata_json.MemberAsString("name", &name, &namelen);
      GOTO_IF_ERR(err, earlyexit);

      const char* version;
      size_t versionlen;
      err =
          server_metadata_json.MemberAsString("version", &version, &versionlen);
      GOTO_IF_ERR(err, earlyexit);

      response->set_name(std::string(name, namelen));
      response->set_version(std::string(version, versionlen));

      if (server_metadata_json.Find("extensions")) {
        TritonJson::Value extensions_json;
        err =
            server_metadata_json.MemberAsArray("extensions", &extensions_json);
        GOTO_IF_ERR(err, earlyexit);

        for (size_t idx = 0; idx < extensions_json.ArraySize(); ++idx) {
          const char* ext;
          size_t extlen;
          err = extensions_json.IndexAsString(idx, &ext, &extlen);
          GOTO_IF_ERR(err, earlyexit);
          response->add_extensions(std::string(ext, extlen));
        }
      }
      TRITONSERVER_MessageDelete(server_metadata_message);
    }

  earlyexit:
    GrpcStatusUtil::Create(status, err);
    TRITONSERVER_ErrorDelete(err);
  };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<ServerMetadataResponse>,
      ServerMetadataRequest, ServerMetadataResponse>(
      "ServerMetadata", 0, OnRegisterServerMetadata, OnExecuteServerMetadata);

  //
  //  ModelMetadata
  //
  auto OnRegisterModelMetadata =
      [this](
          grpc::ServerContext* ctx, ModelMetadataRequest* request,
          grpc::ServerAsyncResponseWriter<ModelMetadataResponse>* responder,
          void* tag) {
        this->service_->RequestModelMetadata(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteModelMetadata = [this](
                                    ModelMetadataRequest& request,
                                    ModelMetadataResponse* response,
                                    grpc::Status* status) {
    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request.version(), &requested_model_version);
    GOTO_IF_ERR(err, earlyexit);

    {
      TRITONSERVER_Message* model_metadata_message = nullptr;
      err = TRITONSERVER_ServerModelMetadata(
          tritonserver_.get(), request.name().c_str(), requested_model_version,
          &model_metadata_message);
      GOTO_IF_ERR(err, earlyexit);

      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(
          model_metadata_message, &buffer, &byte_size);
      GOTO_IF_ERR(err, earlyexit);

      TritonJson::Value model_metadata_json;
      err = model_metadata_json.Parse(buffer, byte_size);
      GOTO_IF_ERR(err, earlyexit);

      const char* name;
      size_t namelen;
      err = model_metadata_json.MemberAsString("name", &name, &namelen);
      GOTO_IF_ERR(err, earlyexit);

      response->set_name(std::string(name, namelen));

      if (model_metadata_json.Find("versions")) {
        TritonJson::Value versions_json;
        err = model_metadata_json.MemberAsArray("versions", &versions_json);
        GOTO_IF_ERR(err, earlyexit);

        for (size_t idx = 0; idx < versions_json.ArraySize(); ++idx) {
          const char* version;
          size_t versionlen;
          err = versions_json.IndexAsString(idx, &version, &versionlen);
          GOTO_IF_ERR(err, earlyexit);
          response->add_versions(std::string(version, versionlen));
        }
      }

      const char* platform;
      size_t platformlen;
      err = model_metadata_json.MemberAsString(
          "platform", &platform, &platformlen);
      GOTO_IF_ERR(err, earlyexit);
      response->set_platform(std::string(platform, platformlen));

      if (model_metadata_json.Find("inputs")) {
        TritonJson::Value inputs_json;
        err = model_metadata_json.MemberAsArray("inputs", &inputs_json);
        GOTO_IF_ERR(err, earlyexit);

        for (size_t idx = 0; idx < inputs_json.ArraySize(); ++idx) {
          TritonJson::Value io_json;
          err = inputs_json.IndexAsObject(idx, &io_json);
          GOTO_IF_ERR(err, earlyexit);

          ModelMetadataResponse::TensorMetadata* io = response->add_inputs();

          const char* name;
          size_t namelen;
          err = io_json.MemberAsString("name", &name, &namelen);
          GOTO_IF_ERR(err, earlyexit);

          const char* datatype;
          size_t datatypelen;
          err = io_json.MemberAsString("datatype", &datatype, &datatypelen);
          GOTO_IF_ERR(err, earlyexit);

          io->set_name(std::string(name, namelen));
          io->set_datatype(std::string(datatype, datatypelen));

          if (io_json.Find("shape")) {
            TritonJson::Value shape_json;
            err = io_json.MemberAsArray("shape", &shape_json);
            GOTO_IF_ERR(err, earlyexit);

            for (size_t sidx = 0; sidx < shape_json.ArraySize(); ++sidx) {
              int64_t d;
              err = shape_json.IndexAsInt(sidx, &d);
              GOTO_IF_ERR(err, earlyexit);

              io->add_shape(d);
            }
          }
        }
      }

      if (model_metadata_json.Find("outputs")) {
        TritonJson::Value outputs_json;
        err = model_metadata_json.MemberAsArray("outputs", &outputs_json);
        GOTO_IF_ERR(err, earlyexit);

        for (size_t idx = 0; idx < outputs_json.ArraySize(); ++idx) {
          TritonJson::Value io_json;
          err = outputs_json.IndexAsObject(idx, &io_json);
          GOTO_IF_ERR(err, earlyexit);

          ModelMetadataResponse::TensorMetadata* io = response->add_outputs();

          const char* name;
          size_t namelen;
          err = io_json.MemberAsString("name", &name, &namelen);
          GOTO_IF_ERR(err, earlyexit);

          const char* datatype;
          size_t datatypelen;
          err = io_json.MemberAsString("datatype", &datatype, &datatypelen);
          GOTO_IF_ERR(err, earlyexit);

          io->set_name(std::string(name, namelen));
          io->set_datatype(std::string(datatype, datatypelen));

          if (io_json.Find("shape")) {
            TritonJson::Value shape_json;
            err = io_json.MemberAsArray("shape", &shape_json);
            GOTO_IF_ERR(err, earlyexit);

            for (size_t sidx = 0; sidx < shape_json.ArraySize(); ++sidx) {
              int64_t d;
              err = shape_json.IndexAsInt(sidx, &d);
              GOTO_IF_ERR(err, earlyexit);

              io->add_shape(d);
            }
          }
        }
      }

      TRITONSERVER_MessageDelete(model_metadata_message);
    }

  earlyexit:
    GrpcStatusUtil::Create(status, err);
    TRITONSERVER_ErrorDelete(err);
  };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<ModelMetadataResponse>,
      ModelMetadataRequest, ModelMetadataResponse>(
      "ModelMetadata", 0, OnRegisterModelMetadata, OnExecuteModelMetadata);

  //
  //  ModelConfig
  //
  auto OnRegisterModelConfig =
      [this](
          grpc::ServerContext* ctx, ModelConfigRequest* request,
          grpc::ServerAsyncResponseWriter<ModelConfigResponse>* responder,
          void* tag) {
        this->service_->RequestModelConfig(
            ctx, request, responder, this->cq_, this->cq_, tag);
      };

  auto OnExecuteModelConfig = [this](
                                  ModelConfigRequest& request,
                                  ModelConfigResponse* response,
                                  grpc::Status* status) {
    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request.version(), &requested_model_version);
    if (err == nullptr) {
      TRITONSERVER_Message* model_config_message = nullptr;
      err = TRITONSERVER_ServerModelConfig(
          tritonserver_.get(), request.name().c_str(), requested_model_version,
          1 /* config_version */, &model_config_message);
      if (err == nullptr) {
        const char* buffer;
        size_t byte_size;
        err = TRITONSERVER_MessageSerializeToJson(
            model_config_message, &buffer, &byte_size);
        if (err == nullptr) {
          ::google::protobuf::util::JsonStringToMessage(
              {buffer, (int)byte_size}, response->mutable_config());
        }
        TRITONSERVER_MessageDelete(model_config_message);
      }
    }

    GrpcStatusUtil::Create(status, err);
    TRITONSERVER_ErrorDelete(err);
  };

  new CommonCallData<
      grpc::ServerAsyncResponseWriter<ModelConfigResponse>, ModelConfigRequest,
      ModelConfigResponse>(
      "ModelConfig", 0, OnRegisterModelConfig, OnExecuteModelConfig);

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
#ifdef TRITON_ENABLE_STATS
    TritonJson::Value model_stats_json;

    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request.version(), &requested_model_version);
    GOTO_IF_ERR(err, earlyexit);

    {
      TRITONSERVER_Message* model_stats_message = nullptr;
      err = TRITONSERVER_ServerModelStatistics(
          tritonserver_.get(), request.name().c_str(), requested_model_version,
          &model_stats_message);
      GOTO_IF_ERR(err, earlyexit);

      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(
          model_stats_message, &buffer, &byte_size);
      GOTO_IF_ERR(err, earlyexit);

      err = model_stats_json.Parse(buffer, byte_size);
      GOTO_IF_ERR(err, earlyexit);

      TRITONSERVER_MessageDelete(model_stats_message);
    }

    if (model_stats_json.Find("model_stats")) {
      TritonJson::Value stats_json;
      err = model_stats_json.MemberAsArray("model_stats", &stats_json);
      GOTO_IF_ERR(err, earlyexit);

      for (size_t idx = 0; idx < stats_json.ArraySize(); ++idx) {
        TritonJson::Value model_stat;
        err = stats_json.IndexAsObject(idx, &model_stat);
        GOTO_IF_ERR(err, earlyexit);

        auto statistics = response->add_model_stats();

        const char* name;
        size_t namelen;
        err = model_stat.MemberAsString("name", &name, &namelen);
        GOTO_IF_ERR(err, earlyexit);

        const char* version;
        size_t versionlen;
        err = model_stat.MemberAsString("version", &version, &versionlen);
        GOTO_IF_ERR(err, earlyexit);

        statistics->set_name(std::string(name, namelen));
        statistics->set_version(std::string(version, versionlen));

        uint64_t ucnt;
        err = model_stat.MemberAsUInt("last_inference", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->set_last_inference(ucnt);

        err = model_stat.MemberAsUInt("inference_count", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->set_inference_count(ucnt);

        err = model_stat.MemberAsUInt("execution_count", &ucnt);
        GOTO_IF_ERR(err, earlyexit);
        statistics->set_execution_count(ucnt);

        TritonJson::Value infer_stats_json;
        err = model_stat.MemberAsObject("inference_stats", &infer_stats_json);
        GOTO_IF_ERR(err, earlyexit);

        {
          TritonJson::Value success_json;
          err = infer_stats_json.MemberAsObject("success", &success_json);
          GOTO_IF_ERR(err, earlyexit);

          err = success_json.MemberAsUInt("count", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()->mutable_success()->set_count(
              ucnt);
          err = success_json.MemberAsUInt("ns", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()->mutable_success()->set_ns(
              ucnt);
        }

        {
          TritonJson::Value fail_json;
          err = infer_stats_json.MemberAsObject("fail", &fail_json);
          GOTO_IF_ERR(err, earlyexit);

          err = fail_json.MemberAsUInt("count", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()->mutable_fail()->set_count(
              ucnt);
          err = fail_json.MemberAsUInt("ns", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()->mutable_fail()->set_ns(ucnt);
        }

        {
          TritonJson::Value queue_json;
          err = infer_stats_json.MemberAsObject("queue", &queue_json);
          GOTO_IF_ERR(err, earlyexit);

          err = queue_json.MemberAsUInt("count", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()->mutable_queue()->set_count(
              ucnt);
          err = queue_json.MemberAsUInt("ns", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()->mutable_queue()->set_ns(ucnt);
        }

        {
          TritonJson::Value compute_input_json;
          err = infer_stats_json.MemberAsObject(
              "compute_input", &compute_input_json);
          GOTO_IF_ERR(err, earlyexit);

          err = compute_input_json.MemberAsUInt("count", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()
              ->mutable_compute_input()
              ->set_count(ucnt);
          err = compute_input_json.MemberAsUInt("ns", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()
              ->mutable_compute_input()
              ->set_ns(ucnt);
        }

        {
          TritonJson::Value compute_infer_json;
          err = infer_stats_json.MemberAsObject(
              "compute_infer", &compute_infer_json);
          GOTO_IF_ERR(err, earlyexit);

          err = compute_infer_json.MemberAsUInt("count", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()
              ->mutable_compute_infer()
              ->set_count(ucnt);
          err = compute_infer_json.MemberAsUInt("ns", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()
              ->mutable_compute_infer()
              ->set_ns(ucnt);
        }

        {
          TritonJson::Value compute_output_json;
          err = infer_stats_json.MemberAsObject(
              "compute_output", &compute_output_json);
          GOTO_IF_ERR(err, earlyexit);

          err = compute_output_json.MemberAsUInt("count", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()
              ->mutable_compute_output()
              ->set_count(ucnt);
          err = compute_output_json.MemberAsUInt("ns", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          statistics->mutable_inference_stats()
              ->mutable_compute_output()
              ->set_ns(ucnt);
        }


        TritonJson::Value batches_json;
        err = model_stat.MemberAsArray("batch_stats", &batches_json);
        GOTO_IF_ERR(err, earlyexit);

        for (size_t idx = 0; idx < batches_json.ArraySize(); ++idx) {
          TritonJson::Value batch_stat;
          err = batches_json.IndexAsObject(idx, &batch_stat);
          GOTO_IF_ERR(err, earlyexit);

          auto batch_statistics = statistics->add_batch_stats();

          uint64_t ucnt;
          err = batch_stat.MemberAsUInt("batch_size", &ucnt);
          GOTO_IF_ERR(err, earlyexit);
          batch_statistics->set_batch_size(ucnt);

          {
            TritonJson::Value compute_input_json;
            err =
                batch_stat.MemberAsObject("compute_input", &compute_input_json);
            GOTO_IF_ERR(err, earlyexit);

            err = compute_input_json.MemberAsUInt("count", &ucnt);
            GOTO_IF_ERR(err, earlyexit);
            batch_statistics->mutable_compute_input()->set_count(ucnt);
            err = compute_input_json.MemberAsUInt("ns", &ucnt);
            GOTO_IF_ERR(err, earlyexit);
            batch_statistics->mutable_compute_input()->set_ns(ucnt);
          }

          {
            TritonJson::Value compute_infer_json;
            err =
                batch_stat.MemberAsObject("compute_infer", &compute_infer_json);
            GOTO_IF_ERR(err, earlyexit);

            err = compute_infer_json.MemberAsUInt("count", &ucnt);
            GOTO_IF_ERR(err, earlyexit);
            batch_statistics->mutable_compute_infer()->set_count(ucnt);
            err = compute_infer_json.MemberAsUInt("ns", &ucnt);
            GOTO_IF_ERR(err, earlyexit);
            batch_statistics->mutable_compute_infer()->set_ns(ucnt);
          }

          {
            TritonJson::Value compute_output_json;
            err = batch_stat.MemberAsObject(
                "compute_output", &compute_output_json);
            GOTO_IF_ERR(err, earlyexit);

            err = compute_output_json.MemberAsUInt("count", &ucnt);
            GOTO_IF_ERR(err, earlyexit);
            batch_statistics->mutable_compute_output()->set_count(ucnt);
            err = compute_output_json.MemberAsUInt("ns", &ucnt);
            GOTO_IF_ERR(err, earlyexit);
            batch_statistics->mutable_compute_output()->set_ns(ucnt);
          }
        }
      }
    }

  earlyexit:
    GrpcStatusUtil::Create(status, err);
    TRITONSERVER_ErrorDelete(err);
#else
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "the server does not suppport model statistics");
    GrpcStatusUtil::Create(status, err);
    TRITONSERVER_ErrorDelete(err);
#endif
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
        TritonJson::Value shm_status_json(TritonJson::ValueType::ARRAY);
        TRITONSERVER_Error* err = shm_manager_->GetStatus(
            request.name(), TRITONSERVER_MEMORY_CPU, &shm_status_json);
        GOTO_IF_ERR(err, earlyexit);

        for (size_t idx = 0; idx < shm_status_json.ArraySize(); ++idx) {
          TritonJson::Value shm_region_json;
          err = shm_status_json.IndexAsObject(idx, &shm_region_json);
          GOTO_IF_ERR(err, earlyexit);

          const char* name;
          size_t namelen;
          err = shm_region_json.MemberAsString("name", &name, &namelen);
          GOTO_IF_ERR(err, earlyexit);

          const char* key;
          size_t keylen;
          err = shm_region_json.MemberAsString("key", &key, &keylen);
          GOTO_IF_ERR(err, earlyexit);

          uint64_t offset;
          err = shm_region_json.MemberAsUInt("offset", &offset);
          GOTO_IF_ERR(err, earlyexit);

          uint64_t byte_size;
          err = shm_region_json.MemberAsUInt("byte_size", &byte_size);
          GOTO_IF_ERR(err, earlyexit);

          SystemSharedMemoryStatusResponse::RegionStatus region_status;
          region_status.set_name(std::string(name, namelen));
          region_status.set_key(std::string(key, keylen));
          region_status.set_offset(offset);
          region_status.set_byte_size(byte_size);

          (*response->mutable_regions())[name] = region_status;
        }

      earlyexit:
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
        TritonJson::Value shm_status_json(TritonJson::ValueType::ARRAY);
        TRITONSERVER_Error* err = shm_manager_->GetStatus(
            request.name(), TRITONSERVER_MEMORY_GPU, &shm_status_json);
        GOTO_IF_ERR(err, earlyexit);

        for (size_t idx = 0; idx < shm_status_json.ArraySize(); ++idx) {
          TritonJson::Value shm_region_json;
          err = shm_status_json.IndexAsObject(idx, &shm_region_json);
          GOTO_IF_ERR(err, earlyexit);

          const char* name;
          size_t namelen;
          err = shm_region_json.MemberAsString("name", &name, &namelen);
          GOTO_IF_ERR(err, earlyexit);

          uint64_t device_id;
          err = shm_region_json.MemberAsUInt("device_id", &device_id);
          GOTO_IF_ERR(err, earlyexit);

          uint64_t byte_size;
          err = shm_region_json.MemberAsUInt("byte_size", &byte_size);
          GOTO_IF_ERR(err, earlyexit);


          CudaSharedMemoryStatusResponse::RegionStatus region_status;
          region_status.set_name(std::string(name, namelen));
          region_status.set_device_id(device_id);
          region_status.set_byte_size(byte_size);

          (*response->mutable_regions())[name] = region_status;
        }
      earlyexit:
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
#ifdef TRITON_ENABLE_GPU
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
#endif  // TRITON_ENABLE_GPU

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
      uint32_t flags = TRITONSERVER_INDEX_FLAG_NONE;
      if (request.ready()) {
        flags |= TRITONSERVER_INDEX_FLAG_READY;
      }

      TRITONSERVER_Message* model_index_message = nullptr;
      err = TRITONSERVER_ServerModelIndex(
          tritonserver_.get(), flags, &model_index_message);
      GOTO_IF_ERR(err, earlyexit);

      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(
          model_index_message, &buffer, &byte_size);
      GOTO_IF_ERR(err, earlyexit);

      TritonJson::Value model_index_json;
      err = model_index_json.Parse(buffer, byte_size);
      GOTO_IF_ERR(err, earlyexit);

      err = model_index_json.AssertType(TritonJson::ValueType::ARRAY);
      GOTO_IF_ERR(err, earlyexit);

      for (size_t idx = 0; idx < model_index_json.ArraySize(); ++idx) {
        TritonJson::Value index_json;
        err = model_index_json.IndexAsObject(idx, &index_json);
        GOTO_IF_ERR(err, earlyexit);

        auto model_index = response->add_models();

        const char* name;
        size_t namelen;
        err = index_json.MemberAsString("name", &name, &namelen);
        GOTO_IF_ERR(err, earlyexit);
        model_index->set_name(std::string(name, namelen));

        if (index_json.Find("version")) {
          const char* version;
          size_t versionlen;
          err = index_json.MemberAsString("version", &version, &versionlen);
          GOTO_IF_ERR(err, earlyexit);
          model_index->set_version(std::string(version, versionlen));
        }
        if (index_json.Find("state")) {
          const char* state;
          size_t statelen;
          err = index_json.MemberAsString("state", &state, &statelen);
          GOTO_IF_ERR(err, earlyexit);
          model_index->set_state(std::string(state, statelen));
        }
        if (index_json.Find("reason")) {
          const char* reason;
          size_t reasonlen;
          err = index_json.MemberAsString("reason", &reason, &reasonlen);
          GOTO_IF_ERR(err, earlyexit);
          model_index->set_reason(std::string(reason, reasonlen));
        }
      }

      TRITONSERVER_MessageDelete(model_index_message);
    } else {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "'repository_name' specification is not supported");
    }

  earlyexit:
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

//=========================================================================
//  The following section contains the handling mechanism for inference
//  RPCs such as ModelInfer and ModelStreamInfer. This implementation
//  is tuned more towards performance and reducing the latency.
//=========================================================================

//
// AllocPayload
//
// Simple structure that carries the userp payload needed for
// allocation.
//
template <typename ResponseType>
struct AllocPayload {
  struct ShmInfo {
    ShmInfo(
        void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
        int64_t memory_type_id)
        : base_(base), byte_size_(byte_size), memory_type_(memory_type),
          memory_type_id_(memory_type_id)
    {
    }
    void* base_;
    size_t byte_size_;
    TRITONSERVER_MemoryType memory_type_;
    int64_t memory_type_id_;
  };

  using TensorShmMap = std::unordered_map<std::string, ShmInfo>;
  using ClassificationMap = std::unordered_map<std::string, uint32_t>;

  explicit AllocPayload() : response_list_(nullptr) {}
  ~AllocPayload()
  {
    // Don't delete 'response_'.. it is owned by the InferHandlerState
  }

  std::shared_ptr<std::vector<ResponseType*>> response_list_;
  uint32_t response_alloc_count_;
  TensorShmMap shm_map_;
  ClassificationMap classification_map_;

  // Used to extend the lifetime of the serialized data in case
  // non-raw contents were provided in the request. Serialized data's
  // actual lifetime is that of the request whereas AllocPayload's
  // lifetime is that of a response... but it is convenient to keep it
  // here.
  std::list<std::string> serialized_data_;
};

//
// InferHandlerState
//
template <
    typename ServerResponderType, typename RequestType, typename ResponseType>
class InferHandlerState {
 public:
  using InferHandlerStateType =
      InferHandlerState<ServerResponderType, RequestType, ResponseType>;

  // State that is shared across all state objects that make up a GRPC
  // transaction (e.g. a stream).
  struct Context {
    explicit Context(
        grpc::ServerCompletionQueue* cq, const uint64_t unique_id = 0)
        : cq_(cq), unique_id_(unique_id), ongoing_requests_(0),
          step_(Steps::START), finish_ok_(true), ongoing_write_(false)
    {
      ctx_.reset(new grpc::ServerContext());
      responder_.reset(new ServerResponderType(ctx_.get()));
    }

    // Increments the ongoing request counter
    void IncrementRequestCounter() { ongoing_requests_++; }

    // Decrements the ongoing request counter
    void DecrementRequestCounter() { ongoing_requests_--; }

    // Enqueue 'state' so that its response is delivered in the
    // correct order.
    void EnqueueForResponse(InferHandlerStateType* state)
    {
      std::lock_guard<std::mutex> lock(mu_);
      states_.push(state);
    }

    // Write the response to the stream directly.
    void DecoupledWriteResponse(InferHandlerStateType* state)
    {
#ifdef TRITON_ENABLE_TRACING
      if ((state->trace_manager_ != nullptr) && (state->trace_id_ != 0)) {
        state->trace_manager_->CaptureTimestamp(
            state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "GRPC_SEND_START");
      }
#endif  // TRITON_ENABLE_TRACING
      state->step_ = Steps::WRITTEN;
      responder_->Write(*state->GetCurrentResponse(), state);

      // Clear the outputs in the response
      state->GetCurrentResponse()
          ->mutable_infer_response()
          ->mutable_outputs()
          ->Clear();

      // Proceed to next response object
      state->Next();
    }

    // Adds the state object to the completion queue so
    // that it can be processed later
    void PutTaskBackToQueue(InferHandlerStateType* state)
    {
      // FIXME: Is there a better way to put task on the
      // completion queue rather than using alarm object?
      // The alarm object will add a new task to the back of the
      // completion queue when it expires or when it’s cancelled.
      grpc::Alarm alarm;
      alarm.Set(cq_, gpr_now(gpr_clock_type::GPR_CLOCK_REALTIME), state);
    }

    // Check the state at the front of the queue and write it if
    // ready. The state at the front of the queue is ready if it is in
    // the WRITEREADY state and it equals 'required_state' (or
    // 'required_state' is nullptr). Return nullptr if front of queue
    // was not ready (and so not written), or return the state if it
    // was ready and written.
    InferHandlerStateType* WriteResponseIfReady(
        InferHandlerStateType* required_state)
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (states_.empty()) {
        return nullptr;
      }

      InferHandlerStateType* state = states_.front();
      if (state->step_ != Steps::WRITEREADY) {
        return nullptr;
      }

      if ((required_state != nullptr) && (state != required_state)) {
        return nullptr;
      }

#ifdef TRITON_ENABLE_TRACING
      if ((state->trace_manager_ != nullptr) && (state->trace_id_ != 0)) {
        state->trace_manager_->CaptureTimestamp(
            state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "GRPC_SEND_START");
      }
#endif  // TRITON_ENABLE_TRACING

      state->step_ = Steps::WRITTEN;
      state->context_->ongoing_write_ = true;
      responder_->Write(*state->GetCurrentResponse(), state);

      return state;
    }

    // If 'state' is at the front of the queue and written, pop it and
    // return true. Other return false.
    bool PopCompletedResponse(InferHandlerStateType* state)
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (states_.empty()) {
        return false;
      }

      InferHandlerStateType* front = states_.front();
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
      return (
          (step_ == Steps::WRITEREADY) && states_.empty() &&
          (ongoing_requests_ == 0));
    }

    // The grpc completion queue associated with the RPC.
    grpc::ServerCompletionQueue* cq_;

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
    std::queue<InferHandlerStateType*> states_;
    std::atomic<uint32_t> ongoing_requests_;

    // The step of the entire context.
    Steps step_;

    // True if this context should finish with OK status, false if
    // should finish with CANCELLED status.
    bool finish_ok_;

    // True if there is an ongoing write to the grpc stream
    std::atomic<bool> ongoing_write_;
  };

  explicit InferHandlerState(
      TRITONSERVER_Server* tritonserver,
      const std::shared_ptr<Context>& context, Steps start_step = Steps::START)
      : tritonserver_(tritonserver)
  {
    // For debugging and testing,
    const char* dstr = getenv("TRITONSERVER_DELAY_GRPC_RESPONSE");
    delay_response_ms_ = 0;
    if (dstr != nullptr) {
      delay_response_ms_ = atoi(dstr);
    }
    response_list_.reset(new std::vector<ResponseType*>());
    Reset(context, start_step);
  }

  ~InferHandlerState()
  {
    for (auto response : *response_list_) {
      delete response;
    }
  }

  void Reset(
      const std::shared_ptr<Context>& context, Steps start_step = Steps::START)
  {
    unique_id_ = NEXT_UNIQUE_ID;
    context_ = context;
    step_ = start_step;
    current_index_ = 0;
    available_count_ = 0;
    is_decoupled_ = false;
    complete_ = false;
    request_.Clear();
    for (auto response : *response_list_) {
      response->Clear();
    }
  }

  void Release() { context_ = nullptr; }

  // Get the response object next in line to be written
  // to the stream.
  ResponseType* GetCurrentResponse()
  {
    if (response_list_->size() <= current_index_) {
      return nullptr;
    }
    return (*response_list_)[current_index_];
  }

  // Get the response object at the specified index.
  ResponseType* GetResponseAt(uint32_t index)
  {
    if (response_list_->size() <= index) {
      return nullptr;
    }
    return (*response_list_)[index];
  }

  // Advance the response index to point to the next
  // response.
  bool Next()
  {
    current_index_++;
    return true;
  }

  // Returns whether all the responses from the state
  // are delivered and successfully written on the
  // stream.
  bool IsComplete()
  {
    return (complete_ && (current_index_ == available_count_));
  }

  // Returns whether the state has a pending response
  // to be written on the stream.
  bool WriteReady() { return (current_index_ < available_count_); }

  // Needed in the response handle for classification outputs.
  TRITONSERVER_Server* tritonserver_;

  // Unique ID for the state. Used only for debugging so will
  // always be 0 in non-debug builds.
  uint64_t unique_id_;

  std::shared_ptr<Context> context_;
  // Mutex to protect the step_
  std::mutex mu_;
  Steps step_;

#ifdef TRITON_ENABLE_TRACING
  TraceManager* trace_manager_;
  TRITONSERVER_InferenceTrace* trace_;
  uint64_t trace_id_;
#endif  // TRITON_ENABLE_TRACING

  bool is_decoupled_;
  uint32_t current_index_;
  uint32_t available_count_;
  bool complete_;

  RequestType request_;
  std::shared_ptr<std::vector<ResponseType*>> response_list_;

  // For testing and debugging
  int delay_response_ms_;

  // For inference requests the allocator payload, unused for other
  // requests.
  AllocPayload<ResponseType> alloc_payload_;
};

//
// InferHandler
//
template <
    typename ServiceType, typename ServerResponderType, typename RequestType,
    typename ResponseType>
class InferHandler : public GRPCServer::HandlerBase {
 public:
  InferHandler(
      const std::string& name,
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      ServiceType* service, grpc::ServerCompletionQueue* cq,
      size_t max_state_bucket_count);
  virtual ~InferHandler();

  // Descriptive name of of the handler.
  const std::string& Name() const { return name_; }

  // Start handling requests.
  void Start();

  // Stop handling requests.
  void Stop();

 protected:
  using State =
      InferHandlerState<ServerResponderType, RequestType, ResponseType>;
  using StateContext = typename State::Context;

  State* StateNew(
      TRITONSERVER_Server* tritonserver,
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
      state = new State(tritonserver, context, start_step);
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
InferHandler<ServiceType, ServerResponderType, RequestType, ResponseType>::
    InferHandler(
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
InferHandler<ServiceType, ServerResponderType, RequestType, ResponseType>::
    ~InferHandler()
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
InferHandler<
    ServiceType, ServerResponderType, RequestType, ResponseType>::Start()
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
InferHandler<
    ServiceType, ServerResponderType, RequestType, ResponseType>::Stop()
{
  if (thread_->joinable()) {
    thread_->join();
  }

  LOG_VERBOSE(1) << "Thread exited for " << Name();
}

//
// Infer utilities
//
template <typename ShmMapType>
TRITONSERVER_Error*
ResponseAllocatorHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, ModelInferResponse* response,
    const ShmMapType& shm_map, void** buffer, void** buffer_userp,
    TRITONSERVER_MemoryType* actual_memory_type, int64_t* actual_memory_type_id)
{
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
    const auto& pr = shm_map.find(tensor_name);
    if (pr != shm_map.end()) {
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
InferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload<ModelInferResponse>* payload =
      reinterpret_cast<AllocPayload<ModelInferResponse>*>(userp);

  // ModelInfer RPC expects exactly one response per request. Hence,
  // will be creating and using just one response object.
  if (payload->response_list_->size() < 1) {
    payload->response_list_->push_back(new ModelInferResponse());
  }

  return ResponseAllocatorHelper<
      AllocPayload<ModelInferResponse>::TensorShmMap>(
      allocator, tensor_name, byte_size, preferred_memory_type,
      preferred_memory_type_id, (*payload->response_list_)[0],
      payload->shm_map_, buffer, buffer_userp, actual_memory_type,
      actual_memory_type_id);
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
              "invalid value type for 'shared_memory_byte_size' parameter "
              "for "
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
ParseClassificationParams(
    const ModelInferRequest::InferRequestedOutputTensor& output,
    bool* has_classification, uint32_t* classification_count)
{
  *has_classification = false;

  const auto& class_it = output.parameters().find("classification");
  if (class_it != output.parameters().end()) {
    *has_classification = true;

    const auto& param = class_it->second;
    if (param.parameter_choice_case() !=
        InferParameter::ParameterChoiceCase::kInt64Param) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "invalid value type for 'classification' parameter, expected "
          "int64_param");
    }

    const int64_t cnt = param.int64_param();
    if (cnt <= 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "invalid value for 'classification' parameter, expected >= 0");
    }

    *classification_count = cnt;
  }

  return nullptr;  // success
}

template <typename ResponseType>
TRITONSERVER_Error*
InferAllocatorPayload(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const ModelInferRequest& request, std::list<std::string>&& serialized_data,
    std::shared_ptr<std::vector<ResponseType*>> response_list_,
    AllocPayload<ResponseType>* alloc_payload)
{
  alloc_payload->response_list_ = response_list_;
  alloc_payload->response_alloc_count_ = 0;
  alloc_payload->shm_map_.clear();
  alloc_payload->classification_map_.clear();
  alloc_payload->serialized_data_ = std::move(serialized_data);

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

    bool has_classification;
    uint32_t classification_count;
    RETURN_IF_ERR(ParseClassificationParams(
        io, &has_classification, &classification_count));

    if (has_shared_memory && has_classification) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "output can't set both 'shared_memory_region' and "
          "'classification'");
    }

    if (has_shared_memory) {
      void* base;
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      RETURN_IF_ERR(shm_manager->GetMemoryInfo(
          region_name, offset, &base, &memory_type, &memory_type_id));

      alloc_payload->shm_map_.emplace(
          io.name(), typename AllocPayload<ResponseType>::ShmInfo(
                         base, byte_size, memory_type, memory_type_id));
    } else if (has_classification) {
      alloc_payload->classification_map_.emplace(
          io.name(), classification_count);
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
    const ModelInferRequest& request, std::list<std::string>* serialized_data,
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
      if (io.has_contents()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "unexpected 'content' provided when using shared memory "
                "for "
                "input tensor '" +
                io.name() + "' for model '" + request.model_name() + "'")
                .c_str());
      }
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
          serialized_data->emplace_back();
          auto& serialized = serialized_data->back();
          serialized.reserve(
              io.contents().int_contents_size() * elem_byte_size);
          for (const auto& element : io.contents().int_contents()) {
            // Assuming the system is little-endian, picking the
            // least significant byte of 32-bit integer as a
            // int8 element
            serialized.append(
                reinterpret_cast<const char*>(&element), elem_byte_size);
          }
          base = serialized.c_str();
          byte_size = serialized.size();
        } else if (dtype == TRITONSERVER_TYPE_INT16) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_INT16, dtype,
              byte_size));
          serialized_data->emplace_back();
          auto& serialized = serialized_data->back();
          serialized.reserve(
              io.contents().int_contents_size() * elem_byte_size);
          for (const auto& element : io.contents().int_contents()) {
            // Assuming the system is little-endian, picking the
            // least 2 significant bytes of 32-bit integer as a
            // int16 element
            serialized.append(
                reinterpret_cast<const char*>(&element), elem_byte_size);
          }
          base = serialized.c_str();
          byte_size = serialized.size();
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
          serialized_data->emplace_back();
          auto& serialized = serialized_data->back();
          serialized.reserve(
              io.contents().uint_contents_size() * elem_byte_size);
          for (const auto& element : io.contents().uint_contents()) {
            // Assuming the system is little-endian, picking the
            // least significant byte of 32-bit unsigned integer as a
            // uint8 element
            serialized.append(
                reinterpret_cast<const char*>(&element), elem_byte_size);
          }
          base = serialized.c_str();
          byte_size = serialized.size();
        } else if (dtype == TRITONSERVER_TYPE_UINT16) {
          RETURN_IF_ERR(InferGRPCToInputHelper(
              io.name(), request.model_name(), TRITONSERVER_TYPE_UINT16, dtype,
              byte_size));
          serialized_data->emplace_back();
          auto& serialized = serialized_data->back();
          serialized.reserve(
              io.contents().uint_contents_size() * elem_byte_size);
          for (const auto& element : io.contents().uint_contents()) {
            // Assuming the system is little-endian, picking the
            // least 2 significant bytes of 32-bit integer as a
            // uint16 element
            serialized.append(
                reinterpret_cast<const char*>(&element), elem_byte_size);
          }
          base = serialized.c_str();
          byte_size = serialized.size();
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

        serialized_data->emplace_back();
        auto& serialized = serialized_data->back();

        // Serialize the output tensor strings. Each string is
        // serialized as a 4-byte length followed by the string itself
        // with no null-terminator.
        for (const auto& element : io.contents().byte_contents()) {
          uint32_t len{(uint32_t)element.size()};
          serialized.append(
              reinterpret_cast<const char*>(&len), sizeof(uint32_t));
          if (element.size() > 0) {
            serialized.append(element.c_str(), len);
          }
        }
        base = serialized.c_str();
        byte_size = serialized.size();
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
      if (infer_param.bool_param()) {
        flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
      }
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
      if (infer_param.bool_param()) {
        flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
      }
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
  }

  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  LOG_VERBOSE(1) << "ModelInferHandler::InferRequestComplete";

  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "deleting GRPC inference request");
  }
}

template <typename ResponseType>
TRITONSERVER_Error*
InferResponseCompleteCommon(
    TRITONSERVER_Server* server, TRITONSERVER_InferenceResponse* iresponse,
    ModelInferResponse& response,
    const AllocPayload<ResponseType>& alloc_payload)
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

  for (uint32_t output_idx = 0; output_idx < output_count; ++output_idx) {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;

    RETURN_IF_ERR(TRITONSERVER_InferenceResponseOutput(
        iresponse, output_idx, &cname, &datatype, &shape, &dim_count, &base,
        &byte_size, &memory_type, &memory_type_id, &userp));

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

    // If this output was requested as classification then remove the
    // raw output from the response and instead return classification
    // results as a string tensor
    const auto itr = alloc_payload.classification_map_.find(name);
    if (itr == alloc_payload.classification_map_.end()) {
      // Not classification...
      output->set_datatype(TRITONSERVER_DataTypeString(datatype));
      for (size_t idx = 0; idx < dim_count; idx++) {
        output->add_shape(shape[idx]);
      }
    } else {
      // Classification
      const uint32_t classification_count = itr->second;

      // For classification need to determine the batch size, if any,
      // because need to use that to break up the response for each
      // batch entry.
      uint32_t batch_size = 0;

      uint32_t batch_flags;
      RETURN_IF_ERR(TRITONSERVER_ServerModelBatchProperties(
          server, model_name, model_version, &batch_flags,
          nullptr /* voidp */));
      if ((dim_count > 0) &&
          ((batch_flags & TRITONSERVER_BATCH_FIRST_DIM) != 0)) {
        batch_size = shape[0];
      }

      // Determine the batch1 byte size of the tensor... needed when
      // the response tensor batch-size > 1 so that we know how to
      // stride though the tensor data.
      size_t batch1_element_count = 1;
      for (size_t idx = ((batch_size == 0) ? 0 : 1); idx < dim_count; idx++) {
        batch1_element_count *= shape[idx];
      }

      const size_t batch1_byte_size =
          batch1_element_count * TRITONSERVER_DataTypeByteSize(datatype);

      // Create the classification contents
      std::string serialized;

      size_t class_offset = 0;
      for (uint32_t bs = 0; bs < std::max((uint32_t)1, batch_size); ++bs) {
        std::vector<std::string> class_strs;
        RETURN_IF_ERR(TopkClassifications(
            iresponse, output_idx,
            reinterpret_cast<const char*>(base) + class_offset,
            ((class_offset + batch1_byte_size) > byte_size) ? 0
                                                            : batch1_byte_size,
            datatype, classification_count, &class_strs));

        // Serialize for binary representation...
        for (const auto& str : class_strs) {
          uint32_t len = str.size();
          serialized.append(reinterpret_cast<const char*>(&len), sizeof(len));
          if (len > 0) {
            serialized.append(str);
          }
        }

        class_offset += batch1_byte_size;
      }

      // Update the output with new datatype, shape and contents.
      output->set_datatype(
          TRITONSERVER_DataTypeString(TRITONSERVER_TYPE_BYTES));

      if (batch_size > 0) {
        output->add_shape(batch_size);
      }
      output->add_shape(classification_count);

      output->mutable_contents()->Clear();
      *(output->mutable_contents()->mutable_raw_contents()) =
          std::move(serialized);
    }
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
    : public InferHandler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncResponseWriter<ModelInferResponse>,
          ModelInferRequest, ModelInferResponse> {
 public:
  ModelInferHandler(
      const std::string& name,
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : InferHandler(name, tritonserver, service, cq, max_state_bucket_count),
        trace_manager_(trace_manager), shm_manager_(shm_manager)
  {
    // Create the allocator that will be used to allocate buffers for
    // the result tensors.
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator_, InferResponseAlloc, InferResponseFree,
            nullptr /* start_fn */),
        "creating inference response allocator");
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;

 private:
  static void InferResponseComplete(
      TRITONSERVER_InferenceResponse* response, const uint32_t flags,
      void* userp);

  TraceManager* trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  TRITONSERVER_ResponseAllocator* allocator_;
};

void
ModelInferHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(cq_);
  State* state = StateNew(tritonserver_.get(), context);

#ifdef TRITON_ENABLE_TRACING
  state->trace_manager_ = nullptr;
  state->trace_ = nullptr;
  state->trace_id_ = 0;
  if (trace_manager_ != nullptr) {
    state->trace_ = trace_manager_->SampleTrace();
    if (state->trace_ != nullptr) {
      state->trace_manager_ = trace_manager_;
      TRITONSERVER_InferenceTraceId(state->trace_, &state->trace_id_);
      trace_manager_->CaptureTimestamp(
          state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN,
          "GRPC_WAITREAD_START");
    }
  }
#endif  // TRITON_ENABLE_TRACING

  service_->RequestModelInfer(
      state->context_->ctx_.get(), &state->request_,
      state->context_->responder_.get(), cq_, cq_, state);

  LOG_VERBOSE(1) << "New request handler for " << Name() << ", "
                 << state->unique_id_;
}

bool
ModelInferHandler::Process(InferHandler::State* state, bool rpc_ok)
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
  auto response_list = state->response_list_;

  if (state->step_ == Steps::START) {
    TRITONSERVER_Error* err = nullptr;
#ifdef TRITON_ENABLE_TRACING
    if ((state->trace_manager_ != nullptr) && (state->trace_id_ != 0)) {
      state->trace_manager_->CaptureTimestamp(
          state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "GRPC_WAITREAD_END");
    }
#endif  // TRITON_ENABLE_TRACING

    // Start a new request to replace this one...
    if (!shutdown) {
      StartNewRequest();
    }

    int64_t requested_model_version;
    if (err == nullptr) {
      err = GetModelVersionFromString(
          request.model_version(), &requested_model_version);
    }

    if (err == nullptr) {
      uint32_t txn_flags;
      err = TRITONSERVER_ServerModelTransactionProperties(
          tritonserver_.get(), request.model_name().c_str(),
          requested_model_version, &txn_flags, nullptr /* voidp */);
      if ((txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0) {
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "ModelInfer RPC doesn't support models with decoupled "
            "transaction policy");
      }
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
      err = SetInferenceRequestMetadata(irequest, request);
    }

    // Will be used to hold the serialized data in case explicit string
    // tensors are present in the request.
    std::list<std::string> serialized_data;

    if (err == nullptr) {
      err = InferGRPCToInput(
          tritonserver_, shm_manager_, request, &serialized_data, irequest);
    }
    if (err == nullptr) {
      err = InferAllocatorPayload<ModelInferResponse>(
          tritonserver_, shm_manager_, request, std::move(serialized_data),
          response_list, &state->alloc_payload_);
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
      TRITONSERVER_InferenceTrace* trace = nullptr;
#ifdef TRITON_ENABLE_TRACING
      trace = state->trace_;
#endif  // TRITON_ENABLE_TRACING

      state->step_ = ISSUED;
      err = TRITONSERVER_ServerInferAsync(tritonserver_.get(), irequest, trace);
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

      ModelInferResponse error_response;

#ifdef TRITON_ENABLE_TRACING
      if ((state->trace_manager_ != nullptr) && (state->trace_id_ != 0)) {
        state->trace_manager_->CaptureTimestamp(
            state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "GRPC_SEND_START");
      }
#endif  // TRITON_ENABLE_TRACING

      state->step_ = COMPLETE;
      state->context_->responder_->Finish(error_response, status, state);
    }
  } else if (state->step_ == Steps::COMPLETE) {
#ifdef TRITON_ENABLE_TRACING
    if ((state->trace_manager_ != nullptr) && (state->trace_id_ != 0)) {
      state->trace_manager_->CaptureTimestamp(
          state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "GRPC_SEND_END");
    }
#endif  // TRITON_ENABLE_TRACING

    state->step_ = Steps::FINISH;
    finished = true;
  }

  return !finished;
}

void
ModelInferHandler::InferResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
    void* userp)
{
  State* state = reinterpret_cast<State*>(userp);

  // Increment the callback index
  state->available_count_++;

  LOG_VERBOSE(1) << "ModelInferHandler::InferResponseComplete, "
                 << state->unique_id_ << " step " << state->step_;

  // Defer to the callback with the final response
  if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
    LOG_ERROR << "[INTERNAL] ModelInfer received a response without FINAL flag";
    return;
  }

  auto response_list = state->response_list_;

  TRITONSERVER_Error* err = nullptr;
  // This callback is expected to be called exactly once for each request.
  // Will use the single response object in the response list to hold the
  // information.
  if (state->response_list_->size() < 1) {
    LOG_ERROR << "expected allocator to have created a response object";
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "No response object found in the callback");
  }

  ModelInferResponse* response;
  bool response_created = false;
  if (err == nullptr) {
    response = (*state->response_list_)[0];
  } else {
    response_created = true;
    response = new ModelInferResponse();
  }

  if (state->available_count_ != 1) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "expected a single response, got " +
            std::to_string(state->available_count_))
            .c_str());
  } else if (iresponse == nullptr) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "received an unexpected null response");
  } else {
    err = InferResponseCompleteCommon<ModelInferResponse>(
        state->tritonserver_, iresponse, *response, state->alloc_payload_);
  }

  if (err != nullptr) {
    response->Clear();
  }

  grpc::Status status;
  GrpcStatusUtil::Create(&status, err);
  TRITONSERVER_ErrorDelete(err);

  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceResponseDelete(iresponse),
      "deleting GRPC inference response");

#ifdef TRITON_ENABLE_TRACING
  if ((state->trace_manager_ != nullptr) && (state->trace_id_ != 0)) {
    state->trace_manager_->CaptureTimestamp(
        state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "GRPC_SEND_START");
  }
#endif  // TRITON_ENABLE_TRACING

  state->step_ = COMPLETE;
  state->context_->responder_->Finish(*response, status, state);
  if (response_created) {
    delete response;
  }
}

//
// Additional Stream Infer utilities
//
TRITONSERVER_Error*
StreamInferResponseStart(TRITONSERVER_ResponseAllocator* allocator, void* userp)
{
  AllocPayload<ModelStreamInferResponse>* payload =
      reinterpret_cast<AllocPayload<ModelStreamInferResponse>*>(userp);

  // Move to the next response object
  payload->response_alloc_count_++;

  // Create a response object only when needed
  if (payload->response_list_->size() < payload->response_alloc_count_) {
    payload->response_list_->push_back(new ModelStreamInferResponse());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
StreamInferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload<ModelStreamInferResponse>* payload =
      reinterpret_cast<AllocPayload<ModelStreamInferResponse>*>(userp);

  uint32_t index = payload->response_alloc_count_ - 1;

  return ResponseAllocatorHelper<
      AllocPayload<ModelStreamInferResponse>::TensorShmMap>(
      allocator, tensor_name, byte_size, preferred_memory_type,
      preferred_memory_type_id,
      (*payload->response_list_)[index]->mutable_infer_response(),
      payload->shm_map_, buffer, buffer_userp, actual_memory_type,
      actual_memory_type_id);
}

//
// ModelStreamInferHandler
//
class ModelStreamInferHandler
    : public InferHandler<
          GRPCInferenceService::AsyncService,
          grpc::ServerAsyncReaderWriter<
              ModelStreamInferResponse, ModelInferRequest>,
          ModelInferRequest, ModelStreamInferResponse> {
 public:
  ModelStreamInferHandler(
      const std::string& name,
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      GRPCInferenceService::AsyncService* service,
      grpc::ServerCompletionQueue* cq, size_t max_state_bucket_count)
      : InferHandler(name, tritonserver, service, cq, max_state_bucket_count),
        trace_manager_(trace_manager), shm_manager_(shm_manager)
  {
    // Create the allocator that will be used to allocate buffers for
    // the result tensors.
    FAIL_IF_ERR(
        TRITONSERVER_ResponseAllocatorNew(
            &allocator_, StreamInferResponseAlloc, InferResponseFree,
            StreamInferResponseStart),
        "creating response allocator");
  }

 protected:
  void StartNewRequest() override;
  bool Process(State* state, bool rpc_ok) override;

 private:
  static void StreamInferResponseComplete(
      TRITONSERVER_InferenceResponse* response, const uint32_t flags,
      void* userp);
  bool Finish(State* state);

  TraceManager* trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  TRITONSERVER_ResponseAllocator* allocator_;
};

void
ModelStreamInferHandler::StartNewRequest()
{
  auto context = std::make_shared<State::Context>(cq_, NEXT_UNIQUE_ID);
  State* state = StateNew(tritonserver_.get(), context);

#ifdef TRITON_ENABLE_TRACING
  state->trace_manager_ = nullptr;
  state->trace_ = nullptr;
  state->trace_id_ = 0;
  if (trace_manager_ != nullptr) {
    state->trace_ = trace_manager_->SampleTrace();
    if (state->trace_ != nullptr) {
      state->trace_manager_ = trace_manager_;
      TRITONSERVER_InferenceTraceId(state->trace_, &state->trace_id_);
      state->trace_manager_->CaptureTimestamp(
          state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN,
          "GRPC_WAITREAD_START");
    }
  }
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

    // Since this is the start of a connection, 'state' hasn't been
    // used yet so use it to read a request off the connection.
    state->context_->step_ = Steps::READ;
    state->step_ = Steps::READ;
    state->context_->responder_->Read(&state->request_, state);

  } else if (state->step_ == Steps::READ) {
    TRITONSERVER_Error* err = nullptr;
    const ModelInferRequest& request = state->request_;
#ifdef TRITON_ENABLE_TRACING
    if ((state->trace_manager_ != nullptr) && (state->trace_id_ != 0)) {
      state->trace_manager_->CaptureTimestamp(
          state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "GRPC_WAITREAD_END");
    }
#endif  // TRITON_ENABLE_TRACING

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
      state->is_decoupled_ = ((txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0);
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
    auto response_list_ = state->response_list_;

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

    // Will be used to hold the serialized data in case explicit string
    // tensors are present in the request.
    std::list<std::string> serialized_data;

    if (err == nullptr) {
      err = InferGRPCToInput(
          tritonserver_, shm_manager_, request, &serialized_data, irequest);
    }
    if (err == nullptr) {
      err = InferAllocatorPayload<ModelStreamInferResponse>(
          tritonserver_, shm_manager_, request, std::move(serialized_data),
          response_list_, &state->alloc_payload_);
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
      TRITONSERVER_InferenceTrace* trace = nullptr;
#ifdef TRITON_ENABLE_TRACING
      trace = state->trace_;
#endif  // TRITON_ENABLE_TRACING

      state->step_ = ISSUED;
      err = TRITONSERVER_ServerInferAsync(tritonserver_.get(), irequest, trace);
    }

    // If there was not an error in issuing the 'state' request then
    // state->step_ == ISSUED and inference request has
    // initiated... the completion callback will transition to
    // WRITEREADY or WRITTEN. If there was an error then enqueue the
    // error response and show it to be ready for writing.
    if (err != nullptr) {
      // Create a response object only when needed
      if (state->response_list_->size() < 1) {
        state->response_list_->push_back(new ModelStreamInferResponse());
      }
      ModelStreamInferResponse* response = state->GetCurrentResponse();
      LOG_VERBOSE(1) << "Infer failed: " << TRITONSERVER_ErrorMessage(err);

      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_InferenceRequestDelete(irequest),
          "deleting GRPC inference request");

      grpc::Status status;
      GrpcStatusUtil::Create(&status, err);
      TRITONSERVER_ErrorDelete(err);
      response->set_error_message(status.error_message());

      response->mutable_infer_response()->Clear();
      state->step_ = Steps::WRITEREADY;
      if (!state->is_decoupled_) {
        state->context_->WriteResponseIfReady(state);
      } else {
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
    next_read_state->trace_manager_ = nullptr;
    next_read_state->trace_ = nullptr;
    next_read_state->trace_id_ = 0;
    if (trace_manager_ != nullptr) {
      next_read_state->trace_ = trace_manager_->SampleTrace();
      if (next_read_state->trace_ != nullptr) {
        next_read_state->trace_manager_ = trace_manager_;
        TRITONSERVER_InferenceTraceId(
            next_read_state->trace_, &next_read_state->trace_id_);
        next_read_state->trace_manager_->CaptureTimestamp(
            next_read_state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN,
            "GRPC_WAITREAD_START");
      }
    }
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
      if ((state->trace_manager_ != nullptr) && (state->trace_id_ != 0)) {
        state->trace_manager_->CaptureTimestamp(
            state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "GRPC_SEND_END");
      }
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
      if ((state->trace_manager_ != nullptr) && (state->trace_id_ != 0)) {
        state->trace_manager_->CaptureTimestamp(
            state->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "GRPC_SEND_END");
      }
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
        std::lock_guard<std::mutex> lock(state->mu_);

        // If there is an available response to be written
        // to the stream, then transition directly to WRITEREADY
        // state and enqueue itself to the completion queue to be
        // taken up later. Otherwise, go to ISSUED state and wait
        // for the callback to make a response available.
        if (state->WriteReady()) {
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
        state->context_->finish_ok_ ? grpc::Status::OK
                                    : grpc::Status::CANCELLED,
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
  state->available_count_++;

  LOG_VERBOSE(1) << "ModelStreamInferHandler::StreamInferComplete, context "
                 << state->context_->unique_id_ << ", " << state->unique_id_
                 << " step " << state->step_ << ", callback index "
                 << state->available_count_ << ", flags " << flags;

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

  auto& response_list = state->response_list_;

  // Add a response object to the list if needed
  if (iresponse != nullptr) {
    if (response_list->size() < state->available_count_) {
      LOG_ERROR << "expected the response allocator to have added the response";
    }

    ModelStreamInferResponse* response =
        state->GetResponseAt(state->available_count_ - 1);

    TRITONSERVER_Error* err = nullptr;
    if (iresponse != nullptr) {
      ModelInferResponse& infer_response =
          *(response->mutable_infer_response());
      err = InferResponseCompleteCommon<ModelStreamInferResponse>(
          state->tritonserver_, iresponse, infer_response,
          state->alloc_payload_);
    }

    if (err != nullptr) {
      grpc::Status status;
      GrpcStatusUtil::Create(&status, err);
      response->mutable_infer_response()->Clear();
      response->set_error_message(status.error_message());
    }

    TRITONSERVER_ErrorDelete(err);

    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceResponseDelete(iresponse),
        "deleting GRPC inference response");
  } else {
    // Should not count if (iresponse==nullptr)
    state->available_count_--;
  }

  state->complete_ = ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0);
  if (!state->is_decoupled_) {
    state->step_ = Steps::WRITEREADY;
    state->context_->WriteResponseIfReady(state);
  } else {
    std::lock_guard<std::mutex> lock(state->mu_);

    if (state->step_ == Steps::ISSUED) {
      state->step_ = Steps::WRITEREADY;
      state->context_->PutTaskBackToQueue(state);
    }
  }
}

}  // namespace

//
// GRPCServer
//
GRPCServer::GRPCServer(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    nvidia::inferenceserver::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const std::string& server_addr, const int infer_allocation_pool_size)
    : server_(server), trace_manager_(trace_manager), shm_manager_(shm_manager),
      server_addr_(server_addr),
      infer_allocation_pool_size_(infer_allocation_pool_size), running_(false)
{
}

GRPCServer::~GRPCServer()
{
  Stop();
}

TRITONSERVER_Error*
GRPCServer::Create(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    nvidia::inferenceserver::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager, int32_t port,
    int infer_allocation_pool_size, std::unique_ptr<GRPCServer>* grpc_server)
{
  const std::string addr = "0.0.0.0:" + std::to_string(port);
  grpc_server->reset(new GRPCServer(
      server, trace_manager, shm_manager, addr, infer_allocation_pool_size));

  return nullptr;  // success
}

TRITONSERVER_Error*
GRPCServer::Start()
{
  if (running_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS, "GRPC server is already running.");
  }

  grpc_builder_.AddListeningPort(
      server_addr_, grpc::InsecureServerCredentials());
  grpc_builder_.SetMaxMessageSize(MAX_GRPC_MESSAGE_SIZE);
  grpc_builder_.RegisterService(&service_);
  common_cq_ = grpc_builder_.AddCompletionQueue();
  model_infer_cq_ = grpc_builder_.AddCompletionQueue();
  model_stream_infer_cq_ = grpc_builder_.AddCompletionQueue();
  grpc_server_ = grpc_builder_.BuildAndStart();

  // A common Handler for other non-inference requests
  CommonHandler* hcommon = new CommonHandler(
      "CommonHandler", server_, shm_manager_, &service_, common_cq_.get());
  hcommon->Start();
  common_handler_.reset(hcommon);

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

  running_ = true;
  LOG_INFO << "Started GRPCInferenceService at " << server_addr_;
  return nullptr;  // success
}

TRITONSERVER_Error*
GRPCServer::Stop()
{
  if (!running_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE, "GRPC server is not running.");
  }

  // Always shutdown the completion queue after the server.
  grpc_server_->Shutdown();

  common_cq_->Shutdown();
  model_infer_cq_->Shutdown();
  model_stream_infer_cq_->Shutdown();

  // Must stop all handlers explicitly to wait for all the handler
  // threads to join since they are referencing completion queue, etc.
  dynamic_cast<CommonHandler*>(common_handler_.get())->Stop();
  dynamic_cast<ModelInferHandler*>(model_infer_handler_.get())->Stop();
  dynamic_cast<ModelStreamInferHandler*>(model_stream_infer_handler_.get())
      ->Stop();

  running_ = false;
  return nullptr;  // success
}

}}  // namespace nvidia::inferenceserver
