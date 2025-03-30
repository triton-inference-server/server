// Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "grpc_server.h"

#include <google/protobuf/arena.h>
#include <grpc++/alarm.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>

#include "../classification.h"
#include "../common.h"
#include "grpc++/grpc++.h"
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "triton/common/logging.h"
#include "triton/common/table_printer.h"
#include "triton/core/tritonserver.h"

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "triton/common/triton_json.h"

#ifdef TRITON_ENABLE_TRACING
#include "../tracer.h"
#endif  // TRITON_ENABLE_TRACING

#define REGISTER_GRPC_INFER_THREAD_COUNT 2

namespace triton { namespace server { namespace grpc {

namespace {

//
// The server has separate handling mechanisms for inference RPCs
// and non-inference RPCs.
//

//=========================================================================
//  The following section contains the handling mechanism for non-inference
//  RPCs. A single thread is created to handle all these requests as they
//  are deemed to be not performance critical.
//=========================================================================

// Define a dedicated health service that implements the health check RPC
class HealthCallbackService
    : public ::grpc::health::v1::Health::CallbackService {
 public:
  HealthCallbackService(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      RestrictedFeatures& restricted_keys_)
      : tritonserver_(server), restricted_keys_(restricted_keys_)
  {
  }

  ::grpc::ServerUnaryReactor* Check(
      ::grpc::CallbackServerContext* context,
      const ::grpc::health::v1::HealthCheckRequest* request,
      ::grpc::health::v1::HealthCheckResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // Check restricted access if configured
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::HEALTH);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Check if server is ready
    bool ready = false;
    TRITONSERVER_Error* err =
        TRITONSERVER_ServerIsReady(tritonserver_.get(), &ready);

    // Set health status based on server readiness
    if (err == nullptr && ready) {
      response->set_status(::grpc::health::v1::HealthCheckResponse::SERVING);
    } else {
      response->set_status(
          ::grpc::health::v1::HealthCheckResponse::NOT_SERVING);
    }

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

 private:
  std::shared_ptr<TRITONSERVER_Server> tritonserver_;
  RestrictedFeatures restricted_keys_;
};

class UnifiedCallbackService
    : public inference::GRPCInferenceServiceCallback::CallbackService {
 public:
  UnifiedCallbackService(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      grpc_compression_level compression_level,
      RestrictedFeatures& restricted_keys_,
      const std::string& forward_header_pattern)
      : tritonserver_(server), shm_manager_(shm_manager),
        trace_manager_(trace_manager), restricted_keys_(restricted_keys_),
        model_infer_handler_(
            "ModelInferCallbackHandler", tritonserver_, trace_manager_,
            shm_manager_, compression_level, restricted_keys_,
            forward_header_pattern)
  {
  }

  template <typename PBTYPE>
  TRITONSERVER_Error* SetStatisticsDuration(
      triton::common::TritonJson::Value& statistics_json,
      const std::string& statistics_name,
      PBTYPE* mutable_statistics_duration_protobuf)
  {
    triton::common::TritonJson::Value statistics_duration_json;
    RETURN_IF_ERR(statistics_json.MemberAsObject(
        statistics_name.c_str(), &statistics_duration_json));

    uint64_t value;
    RETURN_IF_ERR(statistics_duration_json.MemberAsUInt("count", &value));
    mutable_statistics_duration_protobuf->set_count(value);
    RETURN_IF_ERR(statistics_duration_json.MemberAsUInt("ns", &value));
    mutable_statistics_duration_protobuf->set_ns(value);
    return nullptr;
  }

  ::grpc::ServerUnaryReactor* ModelInfer(
      ::grpc::CallbackServerContext* context,
      const inference::ModelInferRequest* request,
      inference::ModelInferResponse* response) override
  {
    // 1. Create reactor for this RPC - This is incorrect for callback API.
    //    The reactor is obtained from the context, but we don't need it here
    //    directly. The handler function will obtain and manage it.

    // 2. Process request and start inference by calling the *member handler*.
    //    The handler function itself returns the reactor.
    return model_infer_handler_.HandleModelInfer(
        context, request, response);  // CORRECTED CALL

    // 3. Return reactor to gRPC - Handled by returning the result of the line
    // above.
  }

  // Example RPC method: ServerLive
  ::grpc::ServerUnaryReactor* ServerLive(
      ::grpc::CallbackServerContext* context,
      const inference::ServerLiveRequest* request,
      inference::ServerLiveResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::HEALTH);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Business logic for ServerLive.
    bool live = false;
    TRITONSERVER_Error* err =
        TRITONSERVER_ServerIsLive(tritonserver_.get(), &live);
    response->set_live((err == nullptr) && live);

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* ServerReady(
      ::grpc::CallbackServerContext* context,
      const inference::ServerReadyRequest* request,
      inference::ServerReadyResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::HEALTH);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Business logic for ServerReady.
    bool ready = false;
    TRITONSERVER_Error* err =
        TRITONSERVER_ServerIsReady(tritonserver_.get(), &ready);
    response->set_ready((err == nullptr) && ready);

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* ModelReady(
      ::grpc::CallbackServerContext* context,
      const inference::ModelReadyRequest* request,
      inference::ModelReadyResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::HEALTH);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Business logic for ModelReady.
    bool is_ready = false;
    int64_t requested_model_version;
    TRITONSERVER_Error* err =
        GetModelVersionFromString(request->version(), &requested_model_version);
    if (err == nullptr) {
      err = TRITONSERVER_ServerModelIsReady(
          tritonserver_.get(), request->name().c_str(), requested_model_version,
          &is_ready);
    }

    response->set_ready(is_ready);

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* ServerMetadata(
      ::grpc::CallbackServerContext* context,
      const inference::ServerMetadataRequest* request,
      inference::ServerMetadataResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::METADATA);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Business logic for ServerMetadata.
    TRITONSERVER_Message* server_metadata_message = nullptr;
    TRITONSERVER_Error* err = TRITONSERVER_ServerMetadata(
        tritonserver_.get(), &server_metadata_message);
    if (err == nullptr) {
      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(
          server_metadata_message, &buffer, &byte_size);
      if (err == nullptr) {
        triton::common::TritonJson::Value server_metadata_json;
        err = server_metadata_json.Parse(buffer, byte_size);
        if (err == nullptr) {
          const char* name;
          size_t namelen;
          err = server_metadata_json.MemberAsString("name", &name, &namelen);
          if (err == nullptr) {
            const char* version;
            size_t versionlen;
            err = server_metadata_json.MemberAsString(
                "version", &version, &versionlen);
            if (err == nullptr) {
              response->set_name(std::string(name, namelen));
              response->set_version(std::string(version, versionlen));

              if (server_metadata_json.Find("extensions")) {
                triton::common::TritonJson::Value extensions_json;
                err = server_metadata_json.MemberAsArray(
                    "extensions", &extensions_json);
                if (err == nullptr) {
                  for (size_t idx = 0; idx < extensions_json.ArraySize();
                       ++idx) {
                    const char* ext;
                    size_t extlen;
                    err = extensions_json.IndexAsString(idx, &ext, &extlen);
                    if (err == nullptr) {
                      response->add_extensions(std::string(ext, extlen));
                    }
                  }
                }
              }
            }
          }
        }
      }
      TRITONSERVER_MessageDelete(server_metadata_message);
    }

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* ModelMetadata(
      ::grpc::CallbackServerContext* context,
      const inference::ModelMetadataRequest* request,
      inference::ModelMetadataResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::METADATA);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic - kept same as original
    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request->version(), &requested_model_version);
    GOTO_IF_ERR(err, earlyexit);
    {
      TRITONSERVER_Message* model_metadata_message = nullptr;
      err = TRITONSERVER_ServerModelMetadata(
          tritonserver_.get(), request->name().c_str(), requested_model_version,
          &model_metadata_message);
      GOTO_IF_ERR(err, earlyexit);

      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(
          model_metadata_message, &buffer, &byte_size);
      GOTO_IF_ERR(err, earlyexit);

      triton::common::TritonJson::Value model_metadata_json;
      err = model_metadata_json.Parse(buffer, byte_size);
      GOTO_IF_ERR(err, earlyexit);

      const char* name;
      size_t namelen;
      err = model_metadata_json.MemberAsString("name", &name, &namelen);
      GOTO_IF_ERR(err, earlyexit);

      response->set_name(std::string(name, namelen));

      if (model_metadata_json.Find("versions")) {
        triton::common::TritonJson::Value versions_json;
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
        triton::common::TritonJson::Value inputs_json;
        err = model_metadata_json.MemberAsArray("inputs", &inputs_json);
        GOTO_IF_ERR(err, earlyexit);

        for (size_t idx = 0; idx < inputs_json.ArraySize(); ++idx) {
          triton::common::TritonJson::Value io_json;
          err = inputs_json.IndexAsObject(idx, &io_json);
          GOTO_IF_ERR(err, earlyexit);

          inference::ModelMetadataResponse::TensorMetadata* io =
              response->add_inputs();

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
            triton::common::TritonJson::Value shape_json;
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
        triton::common::TritonJson::Value outputs_json;
        err = model_metadata_json.MemberAsArray("outputs", &outputs_json);
        GOTO_IF_ERR(err, earlyexit);

        for (size_t idx = 0; idx < outputs_json.ArraySize(); ++idx) {
          triton::common::TritonJson::Value io_json;
          err = outputs_json.IndexAsObject(idx, &io_json);
          GOTO_IF_ERR(err, earlyexit);

          inference::ModelMetadataResponse::TensorMetadata* io =
              response->add_outputs();

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
            triton::common::TritonJson::Value shape_json;
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

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* ModelConfig(
      ::grpc::CallbackServerContext* context,
      const inference::ModelConfigRequest* request,
      inference::ModelConfigResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::MODEL_CONFIG);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic
    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request->version(), &requested_model_version);
    if (err == nullptr) {
      TRITONSERVER_Message* model_config_message = nullptr;
      err = TRITONSERVER_ServerModelConfig(
          tritonserver_.get(), request->name().c_str(), requested_model_version,
          1 /* config_version */, &model_config_message);
      if (err == nullptr) {
        const char* buffer;
        size_t byte_size;
        err = TRITONSERVER_MessageSerializeToJson(
            model_config_message, &buffer, &byte_size);
        if (err == nullptr) {
          ::google::protobuf::util::JsonStringToMessage(
              ::google::protobuf::stringpiece_internal::StringPiece(
                  buffer, (int)byte_size),
              response->mutable_config());
        }
        TRITONSERVER_MessageDelete(model_config_message);
      }
    }

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  // Other RPC methods (e.g., ServerReady, HealthCheck) would be implemented
  // similarly.
  ::grpc::ServerUnaryReactor* ModelStatistics(
      ::grpc::CallbackServerContext* context,
      const inference::ModelStatisticsRequest* request,
      inference::ModelStatisticsResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::STATISTICS);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic - kept same as original
#ifdef TRITON_ENABLE_STATS
    triton::common::TritonJson::Value model_stats_json;

    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(request->version(), &requested_model_version);
    GOTO_IF_ERR(err, earlyexit);
    {
      TRITONSERVER_Message* model_stats_message = nullptr;
      err = TRITONSERVER_ServerModelStatistics(
          tritonserver_.get(), request->name().c_str(), requested_model_version,
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
      triton::common::TritonJson::Value stats_json;
      err = model_stats_json.MemberAsArray("model_stats", &stats_json);
      GOTO_IF_ERR(err, earlyexit);

      for (size_t idx = 0; idx < stats_json.ArraySize(); ++idx) {
        triton::common::TritonJson::Value model_stat;
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

        {
          triton::common::TritonJson::Value infer_stats_json;
          err = model_stat.MemberAsObject("inference_stats", &infer_stats_json);
          GOTO_IF_ERR(err, earlyexit);

          err = SetStatisticsDuration(
              infer_stats_json, "success",
              statistics->mutable_inference_stats()->mutable_success());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "fail",
              statistics->mutable_inference_stats()->mutable_fail());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "queue",
              statistics->mutable_inference_stats()->mutable_queue());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "compute_input",
              statistics->mutable_inference_stats()->mutable_compute_input());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "compute_infer",
              statistics->mutable_inference_stats()->mutable_compute_infer());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "compute_output",
              statistics->mutable_inference_stats()->mutable_compute_output());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "cache_hit",
              statistics->mutable_inference_stats()->mutable_cache_hit());
          GOTO_IF_ERR(err, earlyexit);
          err = SetStatisticsDuration(
              infer_stats_json, "cache_miss",
              statistics->mutable_inference_stats()->mutable_cache_miss());
          GOTO_IF_ERR(err, earlyexit);
        }

        {
          triton::common::TritonJson::Value responses_json;
          err = model_stat.MemberAsObject("response_stats", &responses_json);
          GOTO_IF_ERR(err, earlyexit);

          std::vector<std::string> keys;
          err = responses_json.Members(&keys);
          GOTO_IF_ERR(err, earlyexit);

          for (const auto& key : keys) {
            triton::common::TritonJson::Value res_json;
            err = responses_json.MemberAsObject(key.c_str(), &res_json);
            GOTO_IF_ERR(err, earlyexit);

            inference::InferResponseStatistics res;

            err = SetStatisticsDuration(
                res_json, "compute_infer", res.mutable_compute_infer());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(
                res_json, "compute_output", res.mutable_compute_output());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(
                res_json, "success", res.mutable_success());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(res_json, "fail", res.mutable_fail());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(
                res_json, "empty_response", res.mutable_empty_response());
            GOTO_IF_ERR(err, earlyexit);
            err =
                SetStatisticsDuration(res_json, "cancel", res.mutable_cancel());
            GOTO_IF_ERR(err, earlyexit);

            (*statistics->mutable_response_stats())[key] = std::move(res);
          }
        }

        {
          triton::common::TritonJson::Value batches_json;
          err = model_stat.MemberAsArray("batch_stats", &batches_json);
          GOTO_IF_ERR(err, earlyexit);

          for (size_t idx = 0; idx < batches_json.ArraySize(); ++idx) {
            triton::common::TritonJson::Value batch_stat;
            err = batches_json.IndexAsObject(idx, &batch_stat);
            GOTO_IF_ERR(err, earlyexit);

            auto batch_statistics = statistics->add_batch_stats();

            uint64_t ucnt;
            err = batch_stat.MemberAsUInt("batch_size", &ucnt);
            GOTO_IF_ERR(err, earlyexit);
            batch_statistics->set_batch_size(ucnt);

            err = SetStatisticsDuration(
                batch_stat, "compute_input",
                batch_statistics->mutable_compute_input());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(
                batch_stat, "compute_infer",
                batch_statistics->mutable_compute_infer());
            GOTO_IF_ERR(err, earlyexit);
            err = SetStatisticsDuration(
                batch_stat, "compute_output",
                batch_statistics->mutable_compute_output());
            GOTO_IF_ERR(err, earlyexit);
          }
        }

        {
          triton::common::TritonJson::Value memory_usage_json;
          err = model_stat.MemberAsArray("memory_usage", &memory_usage_json);
          GOTO_IF_ERR(err, earlyexit);

          for (size_t idx = 0; idx < memory_usage_json.ArraySize(); ++idx) {
            triton::common::TritonJson::Value usage;
            err = memory_usage_json.IndexAsObject(idx, &usage);
            GOTO_IF_ERR(err, earlyexit);

            auto memory_usage = statistics->add_memory_usage();
            {
              const char* type;
              size_t type_len;
              err = usage.MemberAsString("type", &type, &type_len);
              GOTO_IF_ERR(err, earlyexit);
              memory_usage->set_type(std::string(type, type_len));
            }
            {
              int64_t id;
              err = usage.MemberAsInt("id", &id);
              GOTO_IF_ERR(err, earlyexit);
              memory_usage->set_id(id);
            }
            {
              uint64_t byte_size;
              err = usage.MemberAsUInt("byte_size", &byte_size);
              GOTO_IF_ERR(err, earlyexit);
              memory_usage->set_byte_size(byte_size);
            }
          }
        }
      }
    }

  earlyexit:
    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
#else
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "the server does not support model statistics");
    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
#endif

    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* TraceSetting(
      ::grpc::CallbackServerContext* context,
      const inference::TraceSettingRequest* request,
      inference::TraceSettingResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::TRACE);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic - kept same as original
#ifdef TRITON_ENABLE_TRACING
    TRITONSERVER_Error* err = nullptr;
    TRITONSERVER_InferenceTraceLevel level = TRITONSERVER_TRACE_LEVEL_DISABLED;
    uint32_t rate;
    int32_t count;
    uint32_t log_frequency;
    std::string filepath;
    InferenceTraceMode trace_mode;
    TraceConfigMap config_map;

    if (!request->model_name().empty()) {
      bool ready = false;
      err = TRITONSERVER_ServerModelIsReady(
          tritonserver_.get(), request->model_name().c_str(),
          -1 /* model version */, &ready);
      GOTO_IF_ERR(err, earlyexit);
      if (!ready) {
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Request for unknown model : ") +
             request->model_name())
                .c_str());
        GOTO_IF_ERR(err, earlyexit);
      }
    }

    // Update trace setting
    if (!request->settings().empty()) {
      TraceManager::NewSetting new_setting;
      {
        static std::string setting_name = "trace_file";
        auto it = request->settings().find(setting_name);
        if (it != request->settings().end()) {
          err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "trace file location can not be updated through network "
              "protocol");
          GOTO_IF_ERR(err, earlyexit);
        }
      }
      {
        static std::string setting_name = "trace_level";
        auto it = request->settings().find(setting_name);
        if (it != request->settings().end()) {
          if (it->second.value().size() == 0) {
            new_setting.clear_level_ = true;
          } else {
            for (const auto& level_str : it->second.value()) {
              if (level_str == "OFF") {
                if (it->second.value().size() == 1) {
                  level = TRITONSERVER_TRACE_LEVEL_DISABLED;
                  new_setting.level_ = &level;
                } else {
                  err = TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INVALID_ARG,
                      "Expect only one trace level 'OFF' is specified");
                  GOTO_IF_ERR(err, earlyexit);
                }
              } else if (level_str == "TIMESTAMPS") {
                level = static_cast<TRITONSERVER_InferenceTraceLevel>(
                    level | TRITONSERVER_TRACE_LEVEL_TIMESTAMPS);
                new_setting.level_ = &level;
              } else if (level_str == "TENSORS") {
                level = static_cast<TRITONSERVER_InferenceTraceLevel>(
                    level | TRITONSERVER_TRACE_LEVEL_TENSORS);
                new_setting.level_ = &level;
              }
            }
          }
        }
      }
      {
        static std::string setting_name = "trace_rate";
        auto it = request->settings().find(setting_name);
        if (it != request->settings().end()) {
          if (it->second.value().size() == 0) {
            new_setting.clear_rate_ = true;
          } else if (it->second.value().size() == 1) {
            try {
              rate = std::stoi(it->second.value()[0]);
              new_setting.rate_ = &rate;
            }
            catch (const std::invalid_argument& ia) {
              err = TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("Unable to parse '") + setting_name +
                   "', got: " + it->second.value()[0])
                      .c_str());
              GOTO_IF_ERR(err, earlyexit);
            }
            catch (const std::out_of_range& oor) {
              err = TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("Unable to parse '") + setting_name +
                   "', value is out of range [ " +
                   std::to_string(std::numeric_limits<std::uint32_t>::min()) +
                   ", " +
                   std::to_string(std::numeric_limits<std::uint32_t>::max()) +
                   " ], got: " + it->second.value()[0])
                      .c_str());
              GOTO_IF_ERR(err, earlyexit);
            }
          } else {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("expect only 1 value for '") + setting_name + "'")
                    .c_str());
            GOTO_IF_ERR(err, earlyexit);
          }
        }
      }
      {
        static std::string setting_name = "trace_count";
        auto it = request->settings().find(setting_name);
        if (it != request->settings().end()) {
          if (it->second.value().size() == 0) {
            new_setting.clear_count_ = true;
          } else if (it->second.value().size() == 1) {
            try {
              count = std::stoi(it->second.value()[0]);
              if (count < TraceManager::MIN_TRACE_COUNT_VALUE) {
                err = TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    (std::string("Unable to parse '") + setting_name +
                     "'. Expecting value >= " +
                     std::to_string(TraceManager::MIN_TRACE_COUNT_VALUE) +
                     ", got: " + it->second.value()[0])
                        .c_str());
                GOTO_IF_ERR(err, earlyexit);
              }
              new_setting.count_ = &count;
            }
            catch (const std::invalid_argument& ia) {
              err = TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("Unable to parse '") + setting_name +
                   "', got: " + it->second.value()[0])
                      .c_str());
              GOTO_IF_ERR(err, earlyexit);
            }
            catch (const std::out_of_range& oor) {
              err = TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("Unable to parse '") + setting_name +
                   "', value is out of range [ " +
                   std::to_string(TraceManager::MIN_TRACE_COUNT_VALUE) + ", " +
                   std::to_string(std::numeric_limits<std::int32_t>::max()) +
                   " ], got: " + it->second.value()[0])
                      .c_str());
              GOTO_IF_ERR(err, earlyexit);
            }
          } else {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("expect only 1 value for '") + setting_name + "'")
                    .c_str());
            GOTO_IF_ERR(err, earlyexit);
          }
        }
      }
      {
        static std::string setting_name = "log_frequency";
        auto it = request->settings().find(setting_name);
        if (it != request->settings().end()) {
          if (it->second.value().size() == 0) {
            new_setting.clear_log_frequency_ = true;
          } else if (it->second.value().size() == 1) {
            try {
              log_frequency = std::stoi(it->second.value()[0]);
              new_setting.log_frequency_ = &log_frequency;
            }
            catch (const std::invalid_argument& ia) {
              err = TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("Unable to parse '") + setting_name +
                   "', got: " + it->second.value()[0])
                      .c_str());
              GOTO_IF_ERR(err, earlyexit);
            }
            catch (const std::out_of_range& oor) {
              err = TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("Unable to parse '") + setting_name +
                   "', value is out of range [ " +
                   std::to_string(std::numeric_limits<std::uint32_t>::min()) +
                   ", " +
                   std::to_string(std::numeric_limits<std::uint32_t>::max()) +
                   " ], got: " + it->second.value()[0])
                      .c_str());
              GOTO_IF_ERR(err, earlyexit);
            }
          } else {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("expect only 1 value for '") + setting_name + "'")
                    .c_str());
            GOTO_IF_ERR(err, earlyexit);
          }
        }
      }

      err = trace_manager_->UpdateTraceSetting(
          request->model_name(), new_setting);
      GOTO_IF_ERR(err, earlyexit);
    }

    // Get current trace setting
    trace_manager_->GetTraceSetting(
        request->model_name(), &level, &rate, &count, &log_frequency, &filepath,
        &trace_mode, &config_map);

    // level
    {
      inference::TraceSettingResponse::SettingValue level_setting;
      if (level == TRITONSERVER_TRACE_LEVEL_DISABLED) {
        level_setting.add_value("OFF");
      } else {
        if (level & TRITONSERVER_TRACE_LEVEL_TIMESTAMPS) {
          level_setting.add_value("TIMESTAMPS");
        }
        if (level & TRITONSERVER_TRACE_LEVEL_TENSORS) {
          level_setting.add_value("TENSORS");
        }
      }
      (*response->mutable_settings())["trace_level"] = level_setting;
    }

    (*response->mutable_settings())["trace_rate"].add_value(
        std::to_string(rate));
    (*response->mutable_settings())["trace_count"].add_value(
        std::to_string(count));
    if (trace_mode == TRACE_MODE_TRITON) {
      (*response->mutable_settings())["log_frequency"].add_value(
          std::to_string(log_frequency));
      (*response->mutable_settings())["trace_file"].add_value(filepath);
    }
    (*response->mutable_settings())["trace_mode"].add_value(
        trace_manager_->InferenceTraceModeString(trace_mode));
    {
      auto mode_key = std::to_string(trace_mode);
      auto trace_options_it = config_map.find(mode_key);
      if (trace_options_it != config_map.end()) {
        for (const auto& [key, value] : trace_options_it->second) {
          if ((key == "file") || (key == "log-frequency")) {
            continue;
          }
          std::string valueAsString;
          if (std::holds_alternative<std::string>(value)) {
            valueAsString = std::get<std::string>(value);
          } else if (std::holds_alternative<int>(value)) {
            valueAsString = std::to_string(std::get<int>(value));
          } else if (std::holds_alternative<uint32_t>(value)) {
            valueAsString = std::to_string(std::get<uint32_t>(value));
          }
          (*response->mutable_settings())[key].add_value(valueAsString);
        }
      }
    }

  earlyexit:
    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
#else
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE, "the server does not support trace");
    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
#endif

    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* LogSettings(
      ::grpc::CallbackServerContext* context,
      const inference::LogSettingsRequest* request,
      inference::LogSettingsResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::LOGGING);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic - kept same as original
#ifdef TRITON_ENABLE_LOGGING
    TRITONSERVER_Error* err = nullptr;
    // Update log settings
    // Server and Core repos do not have the same Logger object
    // Each update must be applied to both server and core repo versions
    if (!request->settings().empty()) {
      {
        static std::string setting_name = "log_file";
        auto it = request->settings().find(setting_name);
        if (it != request->settings().end()) {
          err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "log file location can not be updated through network protocol");
          GOTO_IF_ERR(err, earlyexit);
        }
      }
      {
        static std::string setting_name = "log_info";
        auto it = request->settings().find(setting_name);
        if (it != request->settings().end()) {
          const auto& log_param = it->second;
          if (log_param.parameter_choice_case() !=
              inference::LogSettingsRequest_SettingValue::ParameterChoiceCase::
                  kBoolParam) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("expect boolean for '") + setting_name + "'")
                    .c_str());
            GOTO_IF_ERR(err, earlyexit);
          } else {
            bool log_info_status = it->second.bool_param();
            LOG_ENABLE_INFO(log_info_status);
            TRITONSERVER_ServerOptionsSetLogInfo(nullptr, log_info_status);
          }
        }
      }
      {
        static std::string setting_name = "log_warning";
        auto it = request->settings().find(setting_name);
        if (it != request->settings().end()) {
          const auto& log_param = it->second;
          if (log_param.parameter_choice_case() !=
              inference::LogSettingsRequest_SettingValue::ParameterChoiceCase::
                  kBoolParam) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("expect boolean for '") + setting_name + "'")
                    .c_str());
            GOTO_IF_ERR(err, earlyexit);
          } else {
            bool log_warn_status = it->second.bool_param();
            LOG_ENABLE_WARNING(log_warn_status);
            TRITONSERVER_ServerOptionsSetLogWarn(nullptr, log_warn_status);
          }
        }
      }
      {
        static std::string setting_name = "log_error";
        auto it = request->settings().find(setting_name);
        if (it != request->settings().end()) {
          const auto& log_param = it->second;
          if (log_param.parameter_choice_case() !=
              inference::LogSettingsRequest_SettingValue::ParameterChoiceCase::
                  kBoolParam) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("expect boolean for '") + setting_name + "'")
                    .c_str());
            GOTO_IF_ERR(err, earlyexit);
          } else {
            bool log_error_status = it->second.bool_param();
            LOG_ENABLE_ERROR(log_error_status);
            TRITONSERVER_ServerOptionsSetLogError(nullptr, log_error_status);
          }
        }
      }
      {
        static std::string setting_name = "log_verbose_level";
        auto it = request->settings().find(setting_name);
        if (it != request->settings().end()) {
          const auto& log_param = it->second;
          if (log_param.parameter_choice_case() !=
              inference::LogSettingsRequest_SettingValue::ParameterChoiceCase::
                  kUint32Param) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("expect int32 for '") + setting_name + "'")
                    .c_str());
            GOTO_IF_ERR(err, earlyexit);
          } else {
            uint32_t verbose_level = it->second.uint32_param();
            LOG_SET_VERBOSE(static_cast<int32_t>(verbose_level));
            TRITONSERVER_ServerOptionsSetLogVerbose(nullptr, verbose_level);
          }
        }
      }
      {
        static std::string setting_name = "log_format";
        auto it = request->settings().find(setting_name);
        if (it != request->settings().end()) {
          const auto& log_param = it->second;
          if (log_param.parameter_choice_case() !=
              inference::LogSettingsRequest_SettingValue::ParameterChoiceCase::
                  kStringParam) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("expect string for '") + setting_name + "'")
                    .c_str());
            GOTO_IF_ERR(err, earlyexit);
          } else {
            const std::string& log_format_parse = it->second.string_param();
            triton::common::Logger::Format log_format_final =
                triton::common::Logger::Format::kDEFAULT;
            if (log_format_parse == "ISO8601") {
              log_format_final = triton::common::Logger::Format::kISO8601;
            } else if (log_format_parse != "default") {
              err = TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  ("invalid argument for log_format, got: " + log_format_parse)
                      .c_str());
              GOTO_IF_ERR(err, earlyexit);
            }
            LOG_SET_FORMAT(log_format_final);
            switch (log_format_final) {
              case triton::common::Logger::Format::kDEFAULT:
                TRITONSERVER_ServerOptionsSetLogFormat(
                    nullptr, TRITONSERVER_LOG_DEFAULT);
                break;
              case triton::common::Logger::Format::kISO8601:
                TRITONSERVER_ServerOptionsSetLogFormat(
                    nullptr, TRITONSERVER_LOG_ISO8601);
                break;
            }
          }
        }
      }
      GOTO_IF_ERR(err, earlyexit);
    }

    (*response->mutable_settings())["log_file"].set_string_param(LOG_FILE);
    (*response->mutable_settings())["log_info"].set_bool_param(LOG_INFO_IS_ON);
    (*response->mutable_settings())["log_warning"].set_bool_param(
        LOG_WARNING_IS_ON);
    (*response->mutable_settings())["log_error"].set_bool_param(
        LOG_ERROR_IS_ON);
    (*response->mutable_settings())["log_verbose_level"].set_uint32_param(
        LOG_VERBOSE_LEVEL);
    (*response->mutable_settings())["log_format"].set_string_param(
        LOG_FORMAT_STRING);

  earlyexit:
    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
#else
    auto err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "the server does not support dynamic logging");
    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
#endif

    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* SystemSharedMemoryRegister(
      ::grpc::CallbackServerContext* context,
      const inference::SystemSharedMemoryRegisterRequest* request,
      inference::SystemSharedMemoryRegisterResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::SHARED_MEMORY);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic - kept same as original
    TRITONSERVER_Error* err = shm_manager_->RegisterSystemSharedMemory(
        request->name(), request->key(), request->offset(),
        request->byte_size());

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* SystemSharedMemoryStatus(
      ::grpc::CallbackServerContext* context,
      const inference::SystemSharedMemoryStatusRequest* request,
      inference::SystemSharedMemoryStatusResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::SHARED_MEMORY);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic - kept same as original
    triton::common::TritonJson::Value shm_status_json(
        triton::common::TritonJson::ValueType::ARRAY);
    TRITONSERVER_Error* err = shm_manager_->GetStatus(
        request->name(), TRITONSERVER_MEMORY_CPU, &shm_status_json);
    GOTO_IF_ERR(err, earlyexit);

    for (size_t idx = 0; idx < shm_status_json.ArraySize(); ++idx) {
      triton::common::TritonJson::Value shm_region_json;
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

      inference::SystemSharedMemoryStatusResponse::RegionStatus region_status;
      region_status.set_name(std::string(name, namelen));
      region_status.set_key(std::string(key, keylen));
      region_status.set_offset(offset);
      region_status.set_byte_size(byte_size);

      (*response->mutable_regions())[name] = region_status;
    }

  earlyexit:
    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* CudaSharedMemoryRegister(
      ::grpc::CallbackServerContext* context,
      const inference::CudaSharedMemoryRegisterRequest* request,
      inference::CudaSharedMemoryRegisterResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::SHARED_MEMORY);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic
    TRITONSERVER_Error* err = nullptr;
#ifdef TRITON_ENABLE_GPU
    err = shm_manager_->RegisterCUDASharedMemory(
        request->name(),
        reinterpret_cast<const cudaIpcMemHandle_t*>(
            request->raw_handle().c_str()),
        request->byte_size(), request->device_id());
#else
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "failed to register CUDA shared memory region: '" +
            request->name() + "', GPUs not supported")
            .c_str());
#endif  // TRITON_ENABLE_GPU

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* CudaSharedMemoryStatus(
      ::grpc::CallbackServerContext* context,
      const inference::CudaSharedMemoryStatusRequest* request,
      inference::CudaSharedMemoryStatusResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::SHARED_MEMORY);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic - kept same as original
    triton::common::TritonJson::Value shm_status_json(
        triton::common::TritonJson::ValueType::ARRAY);
    TRITONSERVER_Error* err = shm_manager_->GetStatus(
        request->name(), TRITONSERVER_MEMORY_GPU, &shm_status_json);
    GOTO_IF_ERR(err, earlyexit);

    for (size_t idx = 0; idx < shm_status_json.ArraySize(); ++idx) {
      triton::common::TritonJson::Value shm_region_json;
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

      inference::CudaSharedMemoryStatusResponse::RegionStatus region_status;
      region_status.set_name(std::string(name, namelen));
      region_status.set_device_id(device_id);
      region_status.set_byte_size(byte_size);

      (*response->mutable_regions())[name] = region_status;
    }

  earlyexit:
    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* SystemSharedMemoryUnregister(
      ::grpc::CallbackServerContext* context,
      const inference::SystemSharedMemoryUnregisterRequest* request,
      inference::SystemSharedMemoryUnregisterResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::SHARED_MEMORY);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic - kept same as original
    TRITONSERVER_Error* err = nullptr;
    if (request->name().empty()) {
      err = shm_manager_->UnregisterAll(TRITONSERVER_MEMORY_CPU);
    } else {
      err = shm_manager_->Unregister(request->name(), TRITONSERVER_MEMORY_CPU);
    }

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  // Add here
  ::grpc::ServerUnaryReactor* CudaSharedMemoryUnregister(
      ::grpc::CallbackServerContext* context,
      const inference::CudaSharedMemoryUnregisterRequest* request,
      inference::CudaSharedMemoryUnregisterResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::SHARED_MEMORY);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic - kept same as original
    TRITONSERVER_Error* err = nullptr;
    if (request->name().empty()) {
      err = shm_manager_->UnregisterAll(TRITONSERVER_MEMORY_GPU);
    } else {
      err = shm_manager_->Unregister(request->name(), TRITONSERVER_MEMORY_GPU);
    }

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* RepositoryIndex(
      ::grpc::CallbackServerContext* context,
      const inference::RepositoryIndexRequest* request,
      inference::RepositoryIndexResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::MODEL_REPOSITORY);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic
    TRITONSERVER_Error* err = nullptr;
    if (request->repository_name().empty()) {
      uint32_t flags = 0;
      if (request->ready()) {
        flags |= TRITONSERVER_INDEX_FLAG_READY;
      }

      TRITONSERVER_Message* model_index_message = nullptr;
      err = TRITONSERVER_ServerModelIndex(
          tritonserver_.get(), flags, &model_index_message);
      if (err == nullptr) {
        const char* buffer;
        size_t byte_size;
        err = TRITONSERVER_MessageSerializeToJson(
            model_index_message, &buffer, &byte_size);
        if (err == nullptr) {
          triton::common::TritonJson::Value model_index_json;
          err = model_index_json.Parse(buffer, byte_size);
          if (err == nullptr) {
            err = model_index_json.AssertType(
                triton::common::TritonJson::ValueType::ARRAY);
            if (err == nullptr) {
              for (size_t idx = 0; idx < model_index_json.ArraySize(); ++idx) {
                triton::common::TritonJson::Value index_json;
                err = model_index_json.IndexAsObject(idx, &index_json);
                if (err != nullptr) {
                  break;
                }

                auto model_index = response->add_models();

                const char* name;
                size_t namelen;
                err = index_json.MemberAsString("name", &name, &namelen);
                if (err != nullptr) {
                  break;
                }
                model_index->set_name(std::string(name, namelen));

                if (index_json.Find("version")) {
                  const char* version;
                  size_t versionlen;
                  err = index_json.MemberAsString(
                      "version", &version, &versionlen);
                  if (err != nullptr) {
                    break;
                  }
                  model_index->set_version(std::string(version, versionlen));
                }
                if (index_json.Find("state")) {
                  const char* state;
                  size_t statelen;
                  err = index_json.MemberAsString("state", &state, &statelen);
                  if (err != nullptr) {
                    break;
                  }
                  model_index->set_state(std::string(state, statelen));
                }
                if (index_json.Find("reason")) {
                  const char* reason;
                  size_t reasonlen;
                  err =
                      index_json.MemberAsString("reason", &reason, &reasonlen);
                  if (err != nullptr) {
                    break;
                  }
                  model_index->set_reason(std::string(reason, reasonlen));
                }
              }
            }
          }
        }
        TRITONSERVER_MessageDelete(model_index_message);
      }
    } else {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "'repository_name' specification is not supported");
    }

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* RepositoryModelLoad(
      ::grpc::CallbackServerContext* context,
      const inference::RepositoryModelLoadRequest* request,
      inference::RepositoryModelLoadResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::MODEL_REPOSITORY);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic
    TRITONSERVER_Error* err = nullptr;
    if (request->repository_name().empty()) {
      std::vector<TRITONSERVER_Parameter*> params;
      // WAR for the const-ness check
      std::vector<const TRITONSERVER_Parameter*> const_params;

      for (const auto& param_proto : request->parameters()) {
        if (param_proto.first == "config") {
          if (param_proto.second.parameter_choice_case() !=
              inference::ModelRepositoryParameter::ParameterChoiceCase::
                  kStringParam) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("invalid value type for load parameter '") +
                 param_proto.first + "', expected string_param.")
                    .c_str());
            break;
          } else {
            auto param = TRITONSERVER_ParameterNew(
                param_proto.first.c_str(), TRITONSERVER_PARAMETER_STRING,
                param_proto.second.string_param().c_str());
            if (param != nullptr) {
              params.emplace_back(param);
              const_params.emplace_back(param);
            } else {
              err = TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  "unexpected error on creating Triton parameter");
              break;
            }
          }
        } else if (param_proto.first.rfind("file:", 0) == 0) {
          if (param_proto.second.parameter_choice_case() !=
              inference::ModelRepositoryParameter::ParameterChoiceCase::
                  kBytesParam) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("invalid value type for load parameter '") +
                 param_proto.first + "', expected bytes_param.")
                    .c_str());
            break;
          } else {
            auto param = TRITONSERVER_ParameterBytesNew(
                param_proto.first.c_str(),
                param_proto.second.bytes_param().data(),
                param_proto.second.bytes_param().length());
            if (param != nullptr) {
              params.emplace_back(param);
              const_params.emplace_back(param);
            } else {
              err = TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  "unexpected error on creating Triton parameter");
              break;
            }
          }
        } else {
          err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("unrecognized load parameter '") +
               param_proto.first + "'.")
                  .c_str());
          break;
        }
      }

      if (err == nullptr) {
        err = TRITONSERVER_ServerLoadModelWithParameters(
            tritonserver_.get(), request->model_name().c_str(),
            const_params.data(), const_params.size());
      }

      // Assumes no further 'params' access after load API returns
      for (auto& param : params) {
        TRITONSERVER_ParameterDelete(param);
      }
    } else {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "'repository_name' specification is not supported");
    }

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

  ::grpc::ServerUnaryReactor* RepositoryModelUnload(
      ::grpc::CallbackServerContext* context,
      const inference::RepositoryModelUnloadRequest* request,
      inference::RepositoryModelUnloadResponse* response) override
  {
    auto* reactor = context->DefaultReactor();

    // (Optionally) Check client metadata for restricted access.
    const std::pair<std::string, std::string>& restricted_kv =
        restricted_keys_.Get(RestrictedCategory::MODEL_REPOSITORY);
    if (!restricted_kv.first.empty()) {
      const auto& metadata = context->client_metadata();
      auto it = metadata.find(restricted_kv.first);
      if (it == metadata.end() || it->second != restricted_kv.second) {
        reactor->Finish(::grpc::Status(
            ::grpc::StatusCode::UNAVAILABLE,
            "Missing or mismatched restricted header"));
        return reactor;
      }
    }

    // Core business logic
    TRITONSERVER_Error* err = nullptr;
    if (request->repository_name().empty()) {
      // Check if the dependent models should be removed
      bool unload_dependents = false;
      for (const auto& param : request->parameters()) {
        if (param.first.compare("unload_dependents") == 0) {
          const auto& unload_param = param.second;
          if (unload_param.parameter_choice_case() !=
              inference::ModelRepositoryParameter::ParameterChoiceCase::
                  kBoolParam) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                "invalid value type for 'unload_dependents' parameter, "
                "expected bool_param.");
          }
          unload_dependents = unload_param.bool_param();
          break;
        }
      }

      if (err == nullptr) {
        if (unload_dependents) {
          err = TRITONSERVER_ServerUnloadModelAndDependents(
              tritonserver_.get(), request->model_name().c_str());
        } else {
          err = TRITONSERVER_ServerUnloadModel(
              tritonserver_.get(), request->model_name().c_str());
        }
      }
    } else {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "'repository_name' specification is not supported");
    }

    ::grpc::Status status;
    GrpcStatusUtil::Create(&status, err);
    TRITONSERVER_ErrorDelete(err);
    reactor->Finish(status);
    return reactor;
  }

 private:
  std::shared_ptr<TRITONSERVER_Server> tritonserver_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  TraceManager* trace_manager_;
  RestrictedFeatures restricted_keys_;
  ModelInferCallbackHandler model_infer_handler_;
};

//
// CommonHandler
//
// A common handler for all non-inference requests.
//
class CommonHandler : public HandlerBase {
 public:
  CommonHandler(
      const std::string& name,
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      TraceManager* trace_manager,
      inference::GRPCInferenceService::AsyncService* service,
      ::grpc::health::v1::Health::AsyncService* health_service,
      inference::GRPCInferenceServiceCallback::CallbackService*
          non_inference_callback_service,
      const RestrictedFeatures& restricted_keys, const uint64_t response_delay);

  // Implement pure virtual functions
  void Start() override {}  // No-op for callback implementation
  void Stop() override {}   // No-op for callback implementation

  // Descriptive name of of the handler.
  const std::string& Name() const { return name_; }

  void CreateCallbackServices();

  // Add methods to return the callback services
  inference::GRPCInferenceServiceCallback::CallbackService*
  GetUnifiedCallbackService()
  {
    return non_inference_callback_service_;
  }

  ::grpc::health::v1::Health::CallbackService* GetHealthCallbackService()
  {
    return health_callback_service_;
  }

 private:
  const std::string name_;
  std::shared_ptr<TRITONSERVER_Server> tritonserver_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;
  TraceManager* trace_manager_;
  inference::GRPCInferenceService::AsyncService* service_;
  ::grpc::health::v1::Health::AsyncService* health_service_;
  inference::GRPCInferenceServiceCallback::CallbackService*
      non_inference_callback_service_;
  ::grpc::health::v1::Health::CallbackService* health_callback_service_;
  std::unique_ptr<std::thread> thread_;
  RestrictedFeatures restricted_keys_;
  const uint64_t response_delay_;
};

CommonHandler::CommonHandler(
    const std::string& name,
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    TraceManager* trace_manager,
    inference::GRPCInferenceService::AsyncService* service,
    ::grpc::health::v1::Health::AsyncService* health_service,
    inference::GRPCInferenceServiceCallback::CallbackService*
        non_inference_callback_service,
    const RestrictedFeatures& restricted_keys, const uint64_t response_delay)
    : name_(name), tritonserver_(tritonserver), shm_manager_(shm_manager),
      trace_manager_(trace_manager), service_(service),
      health_service_(health_service),
      non_inference_callback_service_(non_inference_callback_service),
      health_callback_service_(nullptr), restricted_keys_(restricted_keys),
      response_delay_(response_delay)
{
  CreateCallbackServices();
}

void
CommonHandler::CreateCallbackServices()
{
  // Create the unified callback service for non-inference operations
  // Pass all required arguments to the UnifiedCallbackService constructor
  non_inference_callback_service_ = new UnifiedCallbackService(
      "CommonHandler", tritonserver_,
      trace_manager_,  // Pass the trace manager from CommonHandler
      shm_manager_,
      grpc_compression_level::GRPC_COMPRESS_LEVEL_NONE,  // Provide a default
                                                         // compression level
      restricted_keys_,
      ""  // Provide an empty default for forward_header_pattern
  );
  // Create the health callback service
  health_callback_service_ =
      new HealthCallbackService(tritonserver_, restricted_keys_);
}

}  // namespace

//
// Server
//
Server::Server(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const Options& options)
    : tritonserver_(tritonserver), trace_manager_(trace_manager),
      shm_manager_(shm_manager), server_addr_(
                                     options.socket_.address_ + ":" +
                                     std::to_string(options.socket_.port_))
{
  std::shared_ptr<::grpc::ServerCredentials> credentials;
  const auto& ssl_options = options.ssl_;
  if (ssl_options.use_ssl_) {
    std::string key;
    std::string cert;
    std::string root;
    ReadFile(ssl_options.server_cert_, cert);
    ReadFile(ssl_options.server_key_, key);
    ReadFile(ssl_options.root_cert_, root);
    ::grpc::SslServerCredentialsOptions::PemKeyCertPair keycert = {key, cert};
    ::grpc::SslServerCredentialsOptions sslOpts;
    sslOpts.pem_root_certs = root;
    sslOpts.pem_key_cert_pairs.push_back(keycert);
    if (ssl_options.use_mutual_auth_) {
      sslOpts.client_certificate_request =
          GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
    }
    credentials = ::grpc::SslServerCredentials(sslOpts);
  } else {
    credentials = ::grpc::InsecureServerCredentials();
  }

  builder_.AddListeningPort(server_addr_, credentials, &bound_port_);
  builder_.SetMaxMessageSize(MAX_GRPC_MESSAGE_SIZE);
  builder_.RegisterService(&service_);
  builder_.AddChannelArgument(
      GRPC_ARG_ALLOW_REUSEPORT, options.socket_.reuse_port_);

  {
    // GRPC KeepAlive Docs:
    // https://grpc.github.io/grpc/cpp/md_doc_keepalive.html NOTE: In order to
    // work properly, the client-side settings should be in agreement with
    // server-side settings.
    const auto& keepalive_options = options.keep_alive_;
    builder_.AddChannelArgument(
        GRPC_ARG_KEEPALIVE_TIME_MS, keepalive_options.keepalive_time_ms_);
    builder_.AddChannelArgument(
        GRPC_ARG_KEEPALIVE_TIMEOUT_MS, keepalive_options.keepalive_timeout_ms_);
    builder_.AddChannelArgument(
        GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS,
        keepalive_options.keepalive_permit_without_calls_);
    builder_.AddChannelArgument(
        GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA,
        keepalive_options.http2_max_pings_without_data_);
    builder_.AddChannelArgument(
        GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS,
        keepalive_options.http2_min_recv_ping_interval_without_data_ms_);
    builder_.AddChannelArgument(
        GRPC_ARG_HTTP2_MAX_PING_STRIKES,
        keepalive_options.http2_max_ping_strikes_);
    if (keepalive_options.max_connection_age_ms_ != 0) {
      builder_.AddChannelArgument(
          GRPC_ARG_MAX_CONNECTION_AGE_MS,
          keepalive_options.max_connection_age_ms_);
    }
    if (keepalive_options.max_connection_age_grace_ms_ != 0) {
      builder_.AddChannelArgument(
          GRPC_ARG_MAX_CONNECTION_AGE_GRACE_MS,
          keepalive_options.max_connection_age_grace_ms_);
    }

    std::vector<std::string> headers{"GRPC KeepAlive Option", "Value"};
    triton::common::TablePrinter table_printer(headers);
    std::vector<std::string> row{
        "keepalive_time_ms",
        std::to_string(keepalive_options.keepalive_time_ms_)};
    table_printer.InsertRow(row);

    row = {
        "keepalive_timeout_ms",
        std::to_string(keepalive_options.keepalive_timeout_ms_)};
    table_printer.InsertRow(row);

    row = {
        "keepalive_permit_without_calls",
        std::to_string(keepalive_options.keepalive_permit_without_calls_)};
    table_printer.InsertRow(row);

    row = {
        "http2_max_pings_without_data",
        std::to_string(keepalive_options.http2_max_pings_without_data_)};
    table_printer.InsertRow(row);

    row = {
        "http2_min_recv_ping_interval_without_data_ms",
        std::to_string(
            keepalive_options.http2_min_recv_ping_interval_without_data_ms_)};
    table_printer.InsertRow(row);

    row = {
        "http2_max_ping_strikes",
        std::to_string(keepalive_options.http2_max_ping_strikes_)};
    table_printer.InsertRow(row);

    if (keepalive_options.max_connection_age_ms_ != 0) {
      row = {
          "max_connection_age_ms",
          std::to_string(keepalive_options.max_connection_age_ms_)};
      table_printer.InsertRow(row);
    }

    if (keepalive_options.max_connection_age_grace_ms_ != 0) {
      row = {
          "max_connection_age_grace_ms",
          std::to_string(keepalive_options.max_connection_age_grace_ms_)};
      table_printer.InsertRow(row);
    }
    LOG_TABLE_VERBOSE(1, table_printer);
  }

  model_infer_cq_ = builder_.AddCompletionQueue();
  model_stream_infer_cq_ = builder_.AddCompletionQueue();

  // For testing purposes only, add artificial delay in grpc responses.
  const char* dstr = getenv("TRITONSERVER_SERVER_DELAY_GRPC_RESPONSE_SEC");
  uint64_t response_delay = 0;
  if (dstr != nullptr) {
    response_delay = atoi(dstr);
  }
  // A common Handler for other non-inference requests
  common_handler_.reset(new CommonHandler(
      "CommonHandler", tritonserver_, shm_manager_, trace_manager_, &service_,
      &health_service_, nullptr /* non_inference_callback_service */,
      options.restricted_protocols_, response_delay));
  // Use common_handler_ and register services
  auto* handler = dynamic_cast<CommonHandler*>(common_handler_.get());
  if (handler != nullptr) {
    // Register both the unified service and health service
    builder_.RegisterService(handler->GetUnifiedCallbackService());
    builder_.RegisterService(handler->GetHealthCallbackService());
  }

  // [FIXME] "register" logic is different for infer
  // Handler for model inference requests.
  std::pair<std::string, std::string> restricted_kv =
      options.restricted_protocols_.Get(RestrictedCategory::INFERENCE);
  for (int i = 0; i < REGISTER_GRPC_INFER_THREAD_COUNT; ++i) {
    model_infer_handlers_.emplace_back(new ModelInferHandler(
        "ModelInferHandler", tritonserver_, trace_manager_, shm_manager_,
        &service_, model_infer_cq_.get(),
        options.infer_allocation_pool_size_ /* max_state_bucket_count */,
        options.infer_compression_level_, restricted_kv,
        options.forward_header_pattern_));
  }

  // Handler for streaming inference requests. Keeps one handler for streaming
  // to avoid possible concurrent writes which is not allowed
  model_stream_infer_handlers_.emplace_back(new ModelStreamInferHandler(
      "ModelStreamInferHandler", tritonserver_, trace_manager_, shm_manager_,
      &service_, model_stream_infer_cq_.get(),
      options.infer_allocation_pool_size_ /* max_state_bucket_count */,
      options.infer_compression_level_, restricted_kv,
      options.forward_header_pattern_));
}

Server::~Server()
{
  IGNORE_ERR(Stop());
}

TRITONSERVER_Error*
Server::Create(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const Options& server_options, std::unique_ptr<Server>* server)
{
  const std::string addr = server_options.socket_.address_ + ":" +
                           std::to_string(server_options.socket_.port_);
  try {
    server->reset(
        new Server(tritonserver, trace_manager, shm_manager, server_options));
  }
  catch (const std::invalid_argument& pe) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, pe.what());
    ;
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
Server::Create(
    std::shared_ptr<TRITONSERVER_Server>& server, UnorderedMapType& options,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager,
    const RestrictedFeatures& restricted_features,
    std::unique_ptr<Server>* service)
{
  Options grpc_options;

  RETURN_IF_ERR(GetOptions(grpc_options, options));

  return Create(server, trace_manager, shm_manager, grpc_options, service);
}

TRITONSERVER_Error*
Server::GetOptions(Options& options, UnorderedMapType& options_map)
{
  SocketOptions socket_selection;
  SslOptions ssl_selection;
  KeepAliveOptions keep_alive_selection;

  RETURN_IF_ERR(GetSocketOptions(options.socket_, options_map));
  RETURN_IF_ERR(GetSslOptions(options.ssl_, options_map));
  RETURN_IF_ERR(GetKeepAliveOptions(options.keep_alive_, options_map));

  int infer_compression_level_key;

  RETURN_IF_ERR(GetValue(
      options_map, "infer_compression_level", &infer_compression_level_key));

  options.infer_compression_level_ =
      static_cast<grpc_compression_level>(infer_compression_level_key);

  RETURN_IF_ERR(GetValue(
      options_map, "infer_allocation_pool_size",
      &options.infer_allocation_pool_size_));
  RETURN_IF_ERR(GetValue(
      options_map, "forward_header_pattern", &options.forward_header_pattern_));

  return nullptr;
}

TRITONSERVER_Error*
Server::GetSocketOptions(SocketOptions& options, UnorderedMapType& options_map)
{
  RETURN_IF_ERR(GetValue(options_map, "address", &options.address_));
  RETURN_IF_ERR(GetValue(options_map, "port", &options.port_));
  RETURN_IF_ERR(GetValue(options_map, "reuse_port", &options.reuse_port_));

  return nullptr;
}

TRITONSERVER_Error*
Server::GetSslOptions(SslOptions& options, UnorderedMapType& options_map)
{
  RETURN_IF_ERR(GetValue(options_map, "use_ssl", &options.use_ssl_));
  RETURN_IF_ERR(GetValue(options_map, "server_cert", &options.server_cert_));
  RETURN_IF_ERR(GetValue(options_map, "server_key", &options.server_key_));
  RETURN_IF_ERR(GetValue(options_map, "root_cert", &options.root_cert_));
  RETURN_IF_ERR(
      GetValue(options_map, "use_mutual_auth", &options.use_mutual_auth_));

  return nullptr;
}

TRITONSERVER_Error*
Server::GetKeepAliveOptions(
    KeepAliveOptions& options, UnorderedMapType& options_map)
{
  RETURN_IF_ERR(
      GetValue(options_map, "keepalive_time_ms", &options.keepalive_time_ms_));
  RETURN_IF_ERR(GetValue(
      options_map, "keepalive_timeout_ms", &options.keepalive_timeout_ms_));
  RETURN_IF_ERR(GetValue(
      options_map, "keepalive_permit_without_calls",
      &options.keepalive_permit_without_calls_));
  RETURN_IF_ERR(GetValue(
      options_map, "http2_max_pings_without_data",
      &options.http2_max_pings_without_data_));
  RETURN_IF_ERR(GetValue(
      options_map, "http2_min_recv_ping_interval_without_data_ms",
      &options.http2_min_recv_ping_interval_without_data_ms_));
  RETURN_IF_ERR(GetValue(
      options_map, "http2_max_ping_strikes", &options.http2_max_ping_strikes_));
  RETURN_IF_ERR(GetValue(
      options_map, "max_connection_age_ms", &options.max_connection_age_ms_));
  RETURN_IF_ERR(GetValue(
      options_map, "max_connection_age_grace_ms",
      &options.max_connection_age_grace_ms_));

  return nullptr;
}


TRITONSERVER_Error*
Server::Start()
{
  if (running_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS, "GRPC server is already running.");
  }

  server_ = builder_.BuildAndStart();
  // Check if binding port failed
  if (bound_port_ == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        (std::string("Socket '") + server_addr_ + "' already in use ").c_str());
  }

  // Remove this
  for (auto& model_infer_handler : model_infer_handlers_) {
    model_infer_handler->Start();
  }
  for (auto& model_stream_infer_handler : model_stream_infer_handlers_) {
    model_stream_infer_handler->Start();
  }

  running_ = true;
  LOG_INFO << "Started GRPCInferenceService at " << server_addr_;
  return nullptr;  // success
}

TRITONSERVER_Error*
Server::Stop()
{
  if (!running_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE, "GRPC server is not running.");
  }

  // Always shutdown the completion queue after the server.
  server_->Shutdown();

  // common_cq_->Shutdown();
  model_infer_cq_->Shutdown();
  model_stream_infer_cq_->Shutdown();

  // Must stop all handlers explicitly to wait for all the handler
  // threads to join since they are referencing completion queue, etc.
  // common_handler_->Stop();
  for (auto& model_infer_handler : model_infer_handlers_) {
    model_infer_handler->Stop();
  }
  for (auto& model_stream_infer_handler : model_stream_infer_handlers_) {
    model_stream_infer_handler->Stop();
  }

  running_ = false;
  return nullptr;  // success
}

}}}  // namespace triton::server::grpc
