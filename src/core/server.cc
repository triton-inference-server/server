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

#include "src/core/server.h"

#include <cuda_profiler_api.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <csignal>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/wrappers.pb.h"
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "grpc/grpc.h"
#include "re2/re2.h"

#include "src/core/api.pb.h"
#include "src/core/constants.h"
#include "src/core/grpc_service.grpc.pb.h"
#include "src/core/infer.h"
#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_repository_manager.h"
#include "src/core/profile.h"
#include "src/core/request_status.h"
#include "src/core/server.h"
#include "src/core/server_status.pb.h"
#include "src/core/utils.h"
#include "src/servables/caffe2/netdef_bundle.h"
#include "src/servables/caffe2/netdef_bundle.pb.h"
#include "src/servables/custom/custom_bundle.h"
#include "src/servables/custom/custom_bundle.pb.h"
#include "src/servables/tensorflow/graphdef_bundle.h"
#include "src/servables/tensorflow/graphdef_bundle.pb.h"
#include "src/servables/tensorflow/savedmodel_bundle.h"
#include "src/servables/tensorflow/savedmodel_bundle.pb.h"
#include "src/servables/tensorrt/plan_bundle.h"
#include "src/servables/tensorrt/plan_bundle.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/config/platform_config.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver.h"
#include "tensorflow_serving/util/net_http/server/public/response_code_enum.h"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
#include "tensorflow_serving/util/threadpool_executor.h"

#include "src/nvrpc/Context.h"
#include "src/nvrpc/Executor.h"
#include "src/nvrpc/Resources.h"
#include "src/nvrpc/Server.h"
#include "src/nvrpc/Service.h"
#include "src/nvrpc/ThreadPool.h"

using nvrpc::Context;
using nvrpc::ThreadPool;

namespace nvidia { namespace inferenceserver {


namespace {

class AsyncResources : public nvrpc::Resources {
 public:
  explicit AsyncResources(
    InferenceServer* server, int infer_threads, int mgmt_threads)
      : m_Server(server), m_MgmtThreadPool(mgmt_threads),
        m_InferThreadPool(infer_threads)
  {
  }

  InferenceServer* GetServer() { return m_Server; }
  ThreadPool& GetMgmtThreadPool() { return m_MgmtThreadPool; }
  ThreadPool& GetInferThreadPool() { return m_InferThreadPool; }

 private:
  InferenceServer* m_Server;

  // We can and should get specific on thread affinity.  It might not be as
  // important on the frontend, but the backend threadpool should be aligned
  // with the respective devices.
  ThreadPool m_MgmtThreadPool;
  ThreadPool m_InferThreadPool;
};

static std::shared_ptr<AsyncResources> g_Resources;

class StatusContext final
    : public Context<StatusRequest, StatusResponse, AsyncResources> {
  void ExecuteRPC(
    StatusRequest& request, StatusResponse& response) final override
  {
    GetResources()->GetMgmtThreadPool().enqueue([this, &request, &response] {
      ServerStatTimerScoped timer(
        GetResources()->GetServer()->StatusManager(),
        ServerStatTimerScoped::Kind::STATUS);

      RequestStatus* request_status = response.mutable_request_status();
      ServerStatus* server_status = response.mutable_server_status();

      GetResources()->GetServer()->HandleStatus(
        request_status, server_status, request.model_name());
      this->FinishResponse();
    });
  }
};

class InferContext final
    : public Context<InferRequest, InferResponse, AsyncResources> {
  void ExecuteRPC(InferRequest& request, InferResponse& response) final override
  {
    auto server = GetResources()->GetServer();
    auto infer_stats = std::make_shared<ModelInferStats>(
      server->StatusManager(), request.model_name());
    auto timer = std::make_shared<ModelInferStats::ScopedTimer>();
    infer_stats->StartRequestTimer(timer.get());

    RequestStatus* request_status = response.mutable_request_status();

    std::shared_ptr<GRPCInferRequestProvider> request_provider;
    tensorflow::Status status =
      GRPCInferRequestProvider::Create(request, &request_provider);
    if (status.ok()) {
      std::shared_ptr<GRPCInferResponseProvider> response_provider;
      status = GRPCInferResponseProvider::Create(
        request.meta_data(), &response, &response_provider);
      if (status.ok()) {
        server->HandleInfer(
          request_status, request_provider, response_provider, infer_stats,
          [this, request_status, &response, infer_stats, timer]() mutable {
            // If the response is an error then clear the meta-data
            // and raw output as they may be partially or
            // un-initialized.
            if (request_status->code() != RequestStatusCode::SUCCESS) {
              response.mutable_meta_data()->Clear();
              response.mutable_raw_output()->Clear();
            }

            timer.reset();
            this->FinishResponse();
          },
          true  // async_frontend
        );
      }
    }

    if (!status.ok()) {
      LOG_VERBOSE(1) << "Infer failed: " << status.error_message();
      infer_stats->SetFailed(true);
      RequestStatusFactory::Create(
        request_status, 0 /* request_id */, server->Id(), status);

      // If the response is an error then clear the meta-data and raw
      // output as they may be partially or un-initialized.
      response.mutable_meta_data()->Clear();
      response.mutable_raw_output()->Clear();

      this->FinishResponse();
    }
  }
};

class ProfileContext final
    : public Context<ProfileRequest, ProfileResponse, AsyncResources> {
  void ExecuteRPC(
    ProfileRequest& request, ProfileResponse& response) final override
  {
    GetResources()->GetMgmtThreadPool().enqueue([this, &request, &response] {
      auto server = GetResources()->GetServer();
      ServerStatTimerScoped timer(
        server->StatusManager(), ServerStatTimerScoped::Kind::PROFILE);

      RequestStatus* request_status = response.mutable_request_status();
      server->HandleProfile(request_status, request.cmd());
      this->FinishResponse();
    });
  }
};

class HealthContext final
    : public Context<HealthRequest, HealthResponse, AsyncResources> {
  void ExecuteRPC(
    HealthRequest& request, HealthResponse& response) final override
  {
    GetResources()->GetMgmtThreadPool().enqueue([this, &request, &response] {
      auto server = GetResources()->GetServer();
      ServerStatTimerScoped timer(
        server->StatusManager(), ServerStatTimerScoped::Kind::HEALTH);

      RequestStatus* request_status = response.mutable_request_status();
      bool health;

      server->HandleHealth(request_status, &health, request.mode());
      response.set_health(health);
      this->FinishResponse();
    });
  }
};

//
// Handle HTTP requests
//
class HTTPServiceImpl {
 public:
  explicit HTTPServiceImpl(InferenceServer* server)
      : server_(server),
        api_regex_(R"(/api/(health|profile|infer|status)(.*))"),
        health_regex_(R"(/(live|ready))"),
        infer_regex_(R"(/([^/]+)(?:/(\d+))?)"), status_regex_(R"(/(.*))")
  {
  }

  tfs::net_http::RequestHandler Dispatch(
    tfs::net_http::ServerRequestInterface* req);

 private:
  void Handle(tfs::net_http::ServerRequestInterface* req);
  tfs::net_http::HTTPStatusCode Health(
    tfs::net_http::ServerRequestInterface* req, const std::string& health_uri);
  tfs::net_http::HTTPStatusCode Profile(
    tfs::net_http::ServerRequestInterface* req, const std::string& profile_uri);
  tfs::net_http::HTTPStatusCode Infer(
    tfs::net_http::ServerRequestInterface* req, const std::string& infer_uri);
  tfs::net_http::HTTPStatusCode Status(
    tfs::net_http::ServerRequestInterface* req, const std::string& status_uri);

  InferenceServer* server_;
  re2::RE2 api_regex_;
  re2::RE2 health_regex_;
  re2::RE2 infer_regex_;
  re2::RE2 status_regex_;
};

tfs::net_http::RequestHandler
HTTPServiceImpl::Dispatch(tfs::net_http::ServerRequestInterface* req)
{
  return
    [this](tfs::net_http::ServerRequestInterface* req) { this->Handle(req); };
}

void
HTTPServiceImpl::Handle(tfs::net_http::ServerRequestInterface* req)
{
  LOG_VERBOSE(1) << "HTTP request: " << req->http_method() << " "
                 << req->uri_path();

  tfs::net_http::HTTPStatusCode status =
    tfs::net_http::HTTPStatusCode::BAD_REQUEST;

  std::string endpoint, rest;
  if (RE2::FullMatch(
        std::string(req->uri_path()), api_regex_, &endpoint, &rest)) {
    // health
    if (endpoint == "health") {
      status = Health(req, rest);
    }
    // profile
    else if (endpoint == "profile") {
      status = Profile(req, rest);
    }
    // infer
    else if (endpoint == "infer") {
      status = Infer(req, rest);
    }
    // status
    else if (endpoint == "status") {
      status = Status(req, rest);
    }
  }

  if (status != tfs::net_http::HTTPStatusCode::OK) {
    LOG_VERBOSE(1) << "HTTP error: " << req->http_method() << " "
                   << req->uri_path() << " - " << static_cast<int>(status);
  }

  req->ReplyWithStatus(status);
}

tfs::net_http::HTTPStatusCode
HTTPServiceImpl::Health(
  tfs::net_http::ServerRequestInterface* req, const std::string& health_uri)
{
  ServerStatTimerScoped timer(
    server_->StatusManager(), ServerStatTimerScoped::Kind::HEALTH);

  if (req->http_method() != "GET") {
    return tfs::net_http::HTTPStatusCode::METHOD_NA;
  }

  std::string mode;
  if (!health_uri.empty()) {
    if (!RE2::FullMatch(health_uri, health_regex_, &mode)) {
      return tfs::net_http::HTTPStatusCode::BAD_REQUEST;
    }
  }

  RequestStatus request_status;
  bool health;
  server_->HandleHealth(&request_status, &health, mode);

  req->OverwriteResponseHeader(
    kStatusHTTPHeader, request_status.ShortDebugString());

  return (
    (health) ? tfs::net_http::HTTPStatusCode::OK
             : tfs::net_http::HTTPStatusCode::BAD_REQUEST);
}

tfs::net_http::HTTPStatusCode
HTTPServiceImpl::Profile(
  tfs::net_http::ServerRequestInterface* req, const std::string& profile_uri)
{
  ServerStatTimerScoped timer(
    server_->StatusManager(), ServerStatTimerScoped::Kind::PROFILE);

  if (req->http_method() != "GET") {
    return tfs::net_http::HTTPStatusCode::METHOD_NA;
  }

  if (!profile_uri.empty() && (profile_uri != "/")) {
    return tfs::net_http::HTTPStatusCode::BAD_REQUEST;
  }

  std::string cmd;
  if (!req->QueryParam("cmd", &cmd)) {
    cmd.clear();
  }

  RequestStatus request_status;
  server_->HandleProfile(&request_status, cmd);

  req->OverwriteResponseHeader(
    kStatusHTTPHeader, request_status.ShortDebugString());

  return (
    (request_status.code() == RequestStatusCode::SUCCESS)
      ? tfs::net_http::HTTPStatusCode::OK
      : tfs::net_http::HTTPStatusCode::BAD_REQUEST);
}

tfs::net_http::HTTPStatusCode
HTTPServiceImpl::Infer(
  tfs::net_http::ServerRequestInterface* req, const std::string& infer_uri)
{
  if (req->http_method() != "POST") {
    return tfs::net_http::HTTPStatusCode::METHOD_NA;
  }

  std::string model_name, model_version_str;
  if (!infer_uri.empty()) {
    if (!RE2::FullMatch(
          infer_uri, infer_regex_, &model_name, &model_version_str)) {
      return tfs::net_http::HTTPStatusCode::BAD_REQUEST;
    }
  }

  auto infer_stats =
    std::make_shared<ModelInferStats>(server_->StatusManager(), model_name);
  auto timer = std::make_shared<ModelInferStats::ScopedTimer>();
  infer_stats->StartRequestTimer(timer.get());

  absl::string_view infer_request_header =
    req->GetRequestHeader(kInferRequestHTTPHeader);
  std::string infer_request_header_str(
    infer_request_header.data(), infer_request_header.size());

  RequestStatus request_status;

  std::shared_ptr<HTTPInferRequestProvider> request_provider;
  tensorflow::Status status = HTTPInferRequestProvider::Create(
    req->InputBuffer(), model_name, model_version_str, infer_request_header_str,
    &request_provider);
  if (status.ok()) {
    std::shared_ptr<HTTPInferResponseProvider> response_provider;
    status = HTTPInferResponseProvider::Create(
      req->OutputBuffer(), request_provider->RequestHeader(),
      &response_provider);
    if (status.ok()) {
      server_->HandleInfer(
        &request_status, request_provider, response_provider, infer_stats,
        [&request_status, request_provider, response_provider, infer_stats,
         req]() mutable {
          if (request_status.code() == RequestStatusCode::SUCCESS) {
            std::string format;
            if (!req->QueryParam("format", &format)) {
              format = "text";
            }

            const InferResponseHeader& response_header =
              response_provider->ResponseHeader();

            std::string rstr;
            if (format == "binary") {
              response_header.SerializeToString(&rstr);
            } else {
              rstr = response_header.DebugString();
            }
            req->WriteResponseBytes(rstr.c_str(), rstr.size());
          }
        },
        false  // async frontend
      );
    }
  }

  if (!status.ok()) {
    LOG_VERBOSE(1) << "Infer failed: " << status.error_message();
    infer_stats->SetFailed(true);
    RequestStatusFactory::Create(
      &request_status, 0 /* request_id */, server_->Id(), status);
  }

  // this part still needs to be implemented in teh completer
  req->OverwriteResponseHeader(
    kStatusHTTPHeader, request_status.ShortDebugString());
  req->OverwriteResponseHeader("Content-Type", "application/octet-stream");

  return (
    (request_status.code() == RequestStatusCode::SUCCESS)
      ? tfs::net_http::HTTPStatusCode::OK
      : tfs::net_http::HTTPStatusCode::BAD_REQUEST);
}

tfs::net_http::HTTPStatusCode
HTTPServiceImpl::Status(
  tfs::net_http::ServerRequestInterface* req, const std::string& status_uri)
{
  ServerStatTimerScoped timer(
    server_->StatusManager(), ServerStatTimerScoped::Kind::STATUS);

  if (req->http_method() != "GET") {
    return tfs::net_http::HTTPStatusCode::METHOD_NA;
  }

  std::string model_name;
  if (!status_uri.empty()) {
    if (!RE2::FullMatch(status_uri, status_regex_, &model_name)) {
      return tfs::net_http::HTTPStatusCode::BAD_REQUEST;
    }
  }

  RequestStatus request_status;
  ServerStatus server_status;
  server_->HandleStatus(&request_status, &server_status, model_name);

  // If got status successfully then send it...
  if (request_status.code() == RequestStatusCode::SUCCESS) {
    std::string format;
    if (!req->QueryParam("format", &format)) {
      format = "text";
    }

    if (format == "binary") {
      std::string server_status_str;
      server_status.SerializeToString(&server_status_str);
      req->WriteResponseString(server_status_str);
      req->OverwriteResponseHeader("Content-Type", "application/octet-stream");
    } else {
      req->WriteResponseString(server_status.DebugString());
    }
  }

  req->OverwriteResponseHeader(
    kStatusHTTPHeader, request_status.ShortDebugString());

  return (
    (request_status.code() == RequestStatusCode::SUCCESS)
      ? tfs::net_http::HTTPStatusCode::OK
      : tfs::net_http::HTTPStatusCode::BAD_REQUEST);
}

// Scoped increment / decrement of atomic
class ScopedAtomicIncrement {
 public:
  explicit ScopedAtomicIncrement(std::atomic<uint64_t>& counter)
      : counter_(counter)
  {
    counter_++;
  }

  ~ScopedAtomicIncrement() { counter_--; }

 private:
  std::atomic<uint64_t>& counter_;
};

}  // namespace

//
// InferenceServer
//
InferenceServer::InferenceServer()
    : ready_state_(ServerReadyState::SERVER_INVALID), next_request_id_(1)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  start_time_ns_ = ts.tv_sec * NANOS_PER_SECOND + ts.tv_nsec;

  const char* vstr = getenv("TENSORRT_SERVER_VERSION");
  if (vstr != nullptr) {
    version_.assign(vstr);
  }

  id_ = "inference:0";
  http_port_ = 8000;
  grpc_port_ = 8001;
  metrics_port_ = 8002;
  http_thread_cnt_ = 8;
  strict_model_config_ = true;
  strict_readiness_ = true;
  profiling_enabled_ = false;
  poll_model_repository_enabled_ = true;
  repository_poll_secs_ = 15;
  exit_timeout_secs_ = 30;

  inflight_request_counter_ = 0;

  status_manager_.reset(new ServerStatusManager(version_));
}

void
InferenceServer::LogInitError(const std::string& msg)
{
  LOG_ERROR << msg;
  ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
}

bool
InferenceServer::Init(int argc, char** argv)
{
  tensorflow::Status status;

  ready_state_ = ServerReadyState::SERVER_INITIALIZING;

  std::string server_id("inference:0");
  std::string model_store_path;
  std::string platform_config_file;

  // On error, the init process will stop.
  // The difference is if the server will be terminated.
  bool exit_on_error = true;
  bool strict_model_config = strict_model_config_;
  bool strict_readiness = strict_readiness_;
  bool allow_profiling = profiling_enabled_;
  bool tf_allow_soft_placement = true;
  float tf_gpu_memory_fraction = 0.0;
  bool allow_poll_model_repository = poll_model_repository_enabled_;
  int32_t repository_poll_secs = repository_poll_secs_;
  int32_t exit_timeout_secs = exit_timeout_secs_;

  bool allow_http = true;
  bool allow_grpc = true;
  bool allow_metrics = true;
  int32_t http_port = http_port_;
  int32_t grpc_port = grpc_port_;
  int32_t metrics_port = metrics_port_;
  int32_t http_thread_cnt = http_thread_cnt_;

  bool log_info = true;
  bool log_warn = true;
  bool log_error = true;
  int32_t log_verbose = 0;

  std::vector<tensorflow::Flag> flag_list = {
    tensorflow::Flag("log-info", &log_info, "Enable/Disable info logging"),
    tensorflow::Flag(
      "log-warning", &log_warn, "Enable/Disable warning logging"),
    tensorflow::Flag("log-error", &log_error, "Enable/Disable error logging"),
    tensorflow::Flag("log-verbose", &log_verbose, "Verbose logging level"),
    tensorflow::Flag("id", &server_id, "Identifier for this server"),
    tensorflow::Flag(
      "model-store", &model_store_path, "Path to model store directory."),
    tensorflow::Flag(
      "platform-config-file", &platform_config_file,
      "If non-empty, read an ASCII PlatformConfigMap protobuf "
      "from the supplied file name, and use that platform "
      "config instead of the default platform."),
    tensorflow::Flag(
      "exit-on-error", &exit_on_error,
      "Exit the inference server if an error occurs during "
      "initialization."),
    tensorflow::Flag(
      "strict-model-config", &strict_model_config,
      "If true model configuration files must be provided and all required "
      "configuration settings must be specified. If false the model "
      "configuration may be absent or only partially specified and the "
      "server will attempt to derive the missing required configuration."),
    tensorflow::Flag(
      "strict-readiness", &strict_readiness,
      "If true /api/health/ready endpoint indicates ready if the server "
      "is responsive and all models are available. If false "
      "/api/health/ready endpoint indicates ready if server is responsive even "
      "if some/all models are unavailable."),
    tensorflow::Flag(
      "allow-profiling", &allow_profiling, "Allow server profiling."),
    tensorflow::Flag(
      "allow-http", &allow_http,
      "Allow the server to listen on for HTTP requests."),
    tensorflow::Flag(
      "allow-grpc", &allow_grpc,
      "Allow the server to listen on for gRPC requests."),
    tensorflow::Flag(
      "allow-metrics", &allow_metrics,
      "Allow the server to provide prometheus metrics."),
    tensorflow::Flag(
      "http-port", &http_port,
      "The port for the server to listen on for HTTP requests."),
    tensorflow::Flag(
      "grpc-port", &grpc_port,
      "The port for the server to listen on for gRPC requests."),
    tensorflow::Flag(
      "metrics-port", &metrics_port, "The port exposing prometheus metrics."),
    tensorflow::Flag(
      "http-thread-count", &http_thread_cnt,
      "Number of threads handling HTTP requests."),
    tensorflow::Flag(
      "allow-poll-model-repository", &allow_poll_model_repository,
      "Poll the model repository to detect changes. The poll rate is "
      "controlled by 'repository-poll-secs'."),
    tensorflow::Flag(
      "repository-poll-secs", &repository_poll_secs,
      "Interval in seconds between each poll of the model repository to check "
      "for changes. A value of zero indicates that the repository is checked "
      "only a single time at startup. Valid only when "
      "--allow-poll-model-repository=true is specified."),
    tensorflow::Flag(
      "exit-timeout-secs", &exit_timeout_secs,
      "Timeout (in seconds) when exiting to wait for in-flight inferences to "
      "finish. After the timeout expires the server exits even if inferences "
      "are still in flight."),
    tensorflow::Flag(
      "tf-allow-soft-placement", &tf_allow_soft_placement,
      "Instruct TensorFlow to use CPU implementation of an operation when a "
      "GPU implementation is not available."),
    tensorflow::Flag(
      "tf-gpu-memory-fraction", &tf_gpu_memory_fraction,
      "Reserve a portion of GPU memory for TensorFlow models. Default value "
      "0.0 indicates that TensorFlow should dynamically allocate memory as "
      "needed. Value of 1.0 indicates that TensorFlow should allocate all of "
      "GPU memory."),
  };

  std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LogInitError(usage);
    return false;
  }


  LOG_ENABLE_INFO(log_info);
  LOG_ENABLE_WARNING(log_warn);
  LOG_ENABLE_ERROR(log_error);
  LOG_SET_VERBOSE(log_verbose);

  LOG_INFO << "Initializing TensorRT Inference Server";

  // So the server was init with default value first, then the settings
  // will be modified after parsing arguments
  id_ = server_id;
  http_port_ = allow_http ? http_port : -1;
  grpc_port_ = allow_grpc ? grpc_port : -1;
  metrics_port_ = allow_metrics ? metrics_port : -1;
  model_store_path_ = model_store_path;
  http_thread_cnt_ = http_thread_cnt;
  strict_model_config_ = strict_model_config;
  strict_readiness_ = strict_readiness;
  profiling_enabled_ = allow_profiling;
  poll_model_repository_enabled_ = allow_poll_model_repository;
  repository_poll_secs_ = std::max(0, repository_poll_secs);
  exit_timeout_secs_ = std::max(0, exit_timeout_secs);

  if (argc != 1) {
    LOG_ERROR << "Unrecognized option: " << argv[1];
    LogInitError(usage);
    return false;
  }

  if (model_store_path_.empty()) {
    LOG_ERROR << "--model-store must be specified";
    LogInitError(usage);
    return false;
  }

  if (!allow_http && !allow_grpc) {
    LOG_ERROR << "At least one of the following options must be true: "
              << "--allow-http, --allow-grpc";
    LogInitError(usage);
    return false;
  } else if (allow_http && allow_grpc && http_port == grpc_port) {
    LOG_ERROR << "The server cannot listen to HTTP requests "
              << "and gRPC requests at the same port";
    LogInitError(usage);
    return false;
  } else if (
    allow_metrics && ((allow_grpc && (metrics_port == grpc_port)) ||
                      (allow_http && (metrics_port == http_port)))) {
    LOG_ERROR << "The server cannot provide metrics on same port used for "
              << "HTTP or gRPC requests";
    LogInitError(usage);
    return false;
  }

  // Initialize metrics reporting port.
  if (metrics_port_ >= 0) {
    LOG_INFO << "Reporting prometheus metrics on port " << metrics_port_;
    Metrics::Initialize(metrics_port_);
  }

  // Start the HTTP and/or gRPC server accepting connections.
  Start();

  // Disable profiling at server start. Server API can be used to
  // start/stop profiling (unless disabled as indicated by
  // 'allow_profiling').
  status = ProfileStopAll();
  if (!status.ok()) {
    LogInitError(status.error_message());
    return !exit_on_error;
  }

  // For ServerCore Options, we leave servable_state_monitor_creator unspecified
  // so the default servable_state_monitor_creator will be used.
  tfs::ServerCore::Options options;

  // Set some default values in Options
  options.aspired_version_policy = std::unique_ptr<tfs::AspiredVersionPolicy>(
    new tfs::AvailabilityPreservingPolicy);

  // If not polling the model repository then set the poll secs to 0
  // in TFS so that repository is only checked a single time at
  // startup.
  options.max_num_load_retries = 0;
  options.file_system_poll_wait_seconds =
    (allow_poll_model_repository) ? repository_poll_secs_ : 0;

  // Platform configuration
  if (platform_config_file.empty()) {
    options.platform_config_map =
      BuildPlatformConfigMap(tf_gpu_memory_fraction, tf_allow_soft_placement);
  } else {
    status =
      ParseProtoTextFile(platform_config_file, &options.platform_config_map);
    if (!status.ok()) {
      LogInitError(status.error_message());
      return !exit_on_error;
    }
  }
  LOG_VERBOSE(1) << options.platform_config_map.DebugString();

  // Create the global manager for the repository. Add all models'
  // into the server core 'options' so that they are eagerly loaded
  // below when ServerCore is created.
  status =
    ModelRepositoryManager::Create(model_store_path_, !strict_model_config_);
  if (!status.ok()) {
    LogInitError(status.error_message());
    return !exit_on_error;
  }

  std::set<std::string> added, deleted, modified, unmodified;
  status =
    ModelRepositoryManager::Poll(&added, &deleted, &modified, &unmodified);
  if (!status.ok()) {
    LogInitError(status.error_message());
    return !exit_on_error;
  }

  if (!deleted.empty() || !modified.empty() || !unmodified.empty()) {
    LogInitError("Unexpected initial state for model repository");
    return !exit_on_error;
  }

  for (const auto& name : added) {
    tfs::ModelConfig* tfs_config =
      options.model_server_config.mutable_model_config_list()->add_config();
    status = ModelRepositoryManager::GetTFSModelConfig(name, tfs_config);
    if (!status.ok()) {
      LogInitError("Internal: model repository manager inconsistency");
      return !exit_on_error;
    }

    status = status_manager_->InitForModel(name);
    if (!status.ok()) {
      LogInitError(status.error_message());
      return !exit_on_error;
    }
  }

  LOG_VERBOSE(1) << options.model_server_config.DebugString();

  // Create the server core. We assume that any failure is due to a
  // model not loading correctly so we just continue if not exiting on
  // error.
  status = tfs::ServerCore::Create(std::move(options), &core_);
  if (!status.ok()) {
    LOG_ERROR << status;
    if (exit_on_error) {
      return false;
    }
  }

  ready_state_ = ServerReadyState::SERVER_READY;
  return true;
}

bool
InferenceServer::Close()
{
  ready_state_ = ServerReadyState::SERVER_EXITING;

  if (core_ == nullptr) {
    LOG_INFO << "No server context available. Exiting immediately.";
    return true;
  } else {
    LOG_INFO << "Waiting for in-flight inferences to complete.";
  }

  // Reload an empty configuration to cause all models to unload.
  tfs::ModelServerConfig msc;
  msc.mutable_model_config_list();
  tensorflow::Status status = core_->ReloadConfig(msc);
  if (!status.ok()) {
    LOG_ERROR << "Failed to gracefully unload models: " << status;
  }

  // Wait for all in-flight requests to complete and all loaded models
  // to unload, or for the exit timeout to expire.
  const tfs::ServableStateMonitor& monitor = *core_->servable_state_monitor();
  uint32_t exit_timeout_iters = exit_timeout_secs_;

  while (true) {
    const auto& live_models = monitor.GetLiveServableStates();

    LOG_INFO << "Timeout " << exit_timeout_iters << ": Found "
             << live_models.size() << " live models and "
             << inflight_request_counter_ << " in-flight requests";
    if (LOG_VERBOSE_IS_ON(1)) {
      for (const auto& m : live_models) {
        for (const auto& v : m.second) {
          LOG_VERBOSE(1) << m.first << "v" << v.first << ": "
                         << v.second.DebugString();
        }
      }
    }

    if ((live_models.size() == 0) && (inflight_request_counter_ == 0)) {
      return true;
    }
    if (exit_timeout_iters <= 0) {
      LOG_ERROR << "Exit timeout expired. Exiting immediately.";
      break;
    }

    exit_timeout_iters--;
    tensorflow::Env::Default()->SleepForMicroseconds(1000 * 1000);
  }

  return false;
}

void
InferenceServer::Wait()
{
  tensorflow::Status status;

  // If model load/unload is enabled for the model store, then
  // periodically look for changes and update the loaded model
  // configurations appropriately.
  if (poll_model_repository_enabled_) {
    while (ready_state_ != ServerReadyState::SERVER_EXITING) {
      if (ready_state_ == ServerReadyState::SERVER_READY) {
        std::set<std::string> added, deleted, modified, unmodified;
        status = ModelRepositoryManager::Poll(
          &added, &deleted, &modified, &unmodified);
        if (!status.ok()) {
          LOG_ERROR << "Failed to poll model repository: "
                    << status.error_message();
          goto next;
        }

        // Nothing to do if no model adds, deletes or modifies.
        if (added.empty() && deleted.empty() && modified.empty()) {
          goto next;
        }

        // There was a change in the model repository so need to
        // create a new TFS model configuration and reload it into the
        // server to cause the appropriate models to be loaded and
        // unloaded.
        tfs::ModelServerConfig msc;
        msc.mutable_model_config_list();

        // Added models should be loaded and be initialized for status
        // reporting.
        for (const auto& name : added) {
          tfs::ModelConfig* tfs_config =
            msc.mutable_model_config_list()->add_config();
          status = ModelRepositoryManager::GetTFSModelConfig(name, tfs_config);
          if (!status.ok()) {
            LOG_ERROR << "Failed to create server config for '" << name
                      << "': " << status.error_message();
            goto next;
          }

          status = status_manager_->InitForModel(name);
          if (!status.ok()) {
            LOG_ERROR << "Failed to initialize status for '" << name
                      << "': " << status.error_message();
            goto next;
          }
        }

        // Keep unmodified models...
        for (const auto& name : unmodified) {
          tfs::ModelConfig* tfs_config =
            msc.mutable_model_config_list()->add_config();
          status = ModelRepositoryManager::GetTFSModelConfig(name, tfs_config);
          if (!status.ok()) {
            LOG_ERROR << "Failed to create server config for '" << name
                      << "': " << status.error_message();
            goto next;
          }
        }

        status = core_->ReloadConfig(msc);
        if (!status.ok()) {
          LOG_ERROR << "Failed to reload model configurations: "
                    << status.error_message();
          goto next;
        }

        // If there are any modified model, (re)load them to pick up
        // the changes. We want to keep the current status information
        // so don't re-init it.
        if (!modified.empty()) {
          for (const auto& name : modified) {
            tfs::ModelConfig* tfs_config =
              msc.mutable_model_config_list()->add_config();
            status =
              ModelRepositoryManager::GetTFSModelConfig(name, tfs_config);
            if (!status.ok()) {
              LOG_ERROR << "Failed to create server config for '" << name
                        << "': " << status.error_message();
              goto next;
            }
          }

          status = core_->ReloadConfig(msc);
          if (!status.ok()) {
            LOG_ERROR << "Failed to reload modified model configurations: "
                      << status.error_message();
            goto next;
          }
        }
      }

    next:
      tensorflow::Env::Default()->SleepForMicroseconds(
        repository_poll_secs_ * 1000 * 1000);
    }
  }

  if (grpc_server_) {
    grpc_server_->Shutdown();
  }

  if (http_server_ != nullptr) {
    http_server_->WaitForTermination();
  }
}

std::unique_ptr<nvrpc::Server>
InferenceServer::StartGrpcServer()
{
  // DLIS-162 - provide global defaults and cli overridable options
  g_Resources = std::make_shared<AsyncResources>(
    this,  // InferenceServer*,
    1,     // infer threads
    1      // mgmt threads
  );
  // PlanBundle::SetInferenceManager(g_Resources);

  LOG_INFO << "Building nvrpc server";
  const std::string addr = "0.0.0.0:" + std::to_string(grpc_port_);
  auto server = nvrpc::make_unique<nvrpc::Server>(addr);

  server->GetBuilder().SetMaxMessageSize(MAX_GRPC_MESSAGE_SIZE);

  LOG_INFO << "Register TensorRT GRPCService";
  auto inferenceService = server->RegisterAsyncService<GRPCService>();

  LOG_INFO << "Register Infer RPC";
  auto rpcInfer = inferenceService->RegisterRPC<InferContext>(
    &GRPCService::AsyncService::RequestInfer);

  LOG_INFO << "Register Status RPC";
  auto rpcStatus = inferenceService->RegisterRPC<StatusContext>(
    &GRPCService::AsyncService::RequestStatus);

  LOG_INFO << "Register Profile RPC";
  auto rpcProfile = inferenceService->RegisterRPC<ProfileContext>(
    &GRPCService::AsyncService::RequestProfile);

  LOG_INFO << "Register Health RPC";
  auto rpcHealth = inferenceService->RegisterRPC<HealthContext>(
    &GRPCService::AsyncService::RequestHealth);

  LOG_INFO << "Register Executor";
  auto executor = server->RegisterExecutor(new ::nvrpc::Executor(1));

  // You can register RPC execution contexts from any registered RPC on any
  // executor.
  executor->RegisterContexts(
    rpcInfer, g_Resources, 1000);  // Configurable DLIS-161
  executor->RegisterContexts(rpcStatus, g_Resources, 1);
  executor->RegisterContexts(rpcHealth, g_Resources, 1);
  executor->RegisterContexts(rpcProfile, g_Resources, 1);

  server->AsyncRun();

  return std::move(server);
}

namespace {

class HTTPRequestExecutor final : public tfs::net_http::EventExecutor {
 public:
  // Create executor for HTTP server. Seems to require at least 2
  // threads or else it hangs.
  explicit HTTPRequestExecutor(int num_threads)
      : executor_(
          tensorflow::Env::Default(), "httpserver", std::max(2, num_threads))
  {
  }

  void Schedule(std::function<void()> fn) override { executor_.Schedule(fn); }

 private:
  tfs::ThreadPoolExecutor executor_;
};

}  // namespace

std::unique_ptr<tfs::net_http::HTTPServerInterface>
InferenceServer::StartHttpServer()
{
  auto options = absl::make_unique<tfs::net_http::ServerOptions>();
  options->AddPort(static_cast<uint32_t>(http_port_));
  options->SetExecutor(
    absl::make_unique<HTTPRequestExecutor>(http_thread_cnt_));

  auto server = tfs::net_http::CreateEvHTTPServer(std::move(options));
  if (server != nullptr) {
    std::shared_ptr<HTTPServiceImpl> service =
      std::make_shared<HTTPServiceImpl>(this);

    tfs::net_http::RequestHandlerOptions handler_options;
    server->RegisterRequestDispatcher(
      [service](tfs::net_http::ServerRequestInterface* req) {
        return service->Dispatch(req);
      },
      handler_options);

    if (!server->StartAcceptingRequests()) {
      server.reset();
    }
  }

  return std::move(server);
}

void
InferenceServer::Start()
{
  LOG_INFO << "Starting server '" << id_ << "' listening on";

  // Enable gRPC endpoint if requested...
  if (grpc_port_ != -1) {
    LOG_INFO << " localhost:" << std::to_string(grpc_port_)
             << " for gRPC requests";
    grpc_server_ = StartGrpcServer();
    if (grpc_server_ == nullptr) {
      LOG_ERROR << "Failed to start gRPC server";
    }
  }

  // Enable HTTP endpoint if requested...
  if (http_port_ != -1) {
    LOG_INFO << " localhost:" << std::to_string(http_port_)
             << " for HTTP requests";

    http_server_ = StartHttpServer();
    if (http_server_ == nullptr) {
      LOG_ERROR << "Failed to start HTTP server";
    }
  }
}

void
InferenceServer::HandleHealth(
  RequestStatus* request_status, bool* health, const std::string& mode)
{
  *health = false;

  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    RequestStatusFactory::Create(
      request_status, 0, id_, RequestStatusCode::UNAVAILABLE, "Server exiting");
    return;
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);
  const uint64_t request_id = NextRequestId();

  // Server is considered live if it can respond to this health
  // request and it was able to initialize.
  if (mode == "live") {
    *health =
      ((ready_state_ != ServerReadyState::SERVER_INVALID) &&
       (ready_state_ != ServerReadyState::SERVER_FAILED_TO_INITIALIZE));
    RequestStatusFactory::Create(
      request_status, request_id, id_, RequestStatusCode::SUCCESS);
  }
  // Server is considered ready if it is in the ready state.
  // Additionally can report ready only when all models are ready.
  else if (mode == "ready") {
    *health = (ready_state_ == ServerReadyState::SERVER_READY);
    if (*health && strict_readiness_) {
      // Strict readiness... get the model status and make sure all
      // models are ready.
      tfs::ServableStateMonitor* monitor = nullptr;
      if (core_ != nullptr) {
        monitor = core_->servable_state_monitor();
      }

      ServerStatus server_status;
      tensorflow::Status status = status_manager_->Get(
        &server_status, id_, ready_state_, UptimeNs(), monitor);

      *health = status.ok();
      if (*health) {
        for (const auto& ms : server_status.model_status()) {
          for (const auto& vs : ms.second.version_status()) {
            if (vs.second.ready_state() != ModelReadyState::MODEL_READY) {
              *health = false;
              goto strict_done;
            }
          }
        }
      strict_done:;
      }
    }

    RequestStatusFactory::Create(
      request_status, request_id, id_, RequestStatusCode::SUCCESS);
  } else {
    RequestStatusFactory::Create(
      request_status, request_id, id_, RequestStatusCode::UNKNOWN,
      "unknown health mode '" + mode + "'");
  }
}

void
InferenceServer::HandleProfile(
  RequestStatus* request_status, const std::string& cmd)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    RequestStatusFactory::Create(
      request_status, 0, id_, RequestStatusCode::UNAVAILABLE,
      "Server not ready");
    return;
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);
  const uint64_t request_id = NextRequestId();

  if (!profiling_enabled_) {
    RequestStatusFactory::Create(
      request_status, request_id, id_, RequestStatusCode::UNSUPPORTED,
      "Profile API not enabled");
  } else if (cmd == "start") {
    RequestStatusFactory::Create(
      request_status, request_id, id_, ProfileStartAll());
  } else if (cmd == "stop") {
    RequestStatusFactory::Create(
      request_status, request_id, id_, ProfileStopAll());
  } else {
    RequestStatusFactory::Create(
      request_status, request_id, id_, RequestStatusCode::INVALID_ARG,
      "Unknown profile command '" + std::string(cmd) + "'");
  }
}

namespace {

// Use the servable appropriate for the requested model's platform.
struct AsyncState {
  InferenceServable* is = nullptr;
  tfs::ServableHandle<GraphDefBundle> graphdef_bundle;
  tfs::ServableHandle<PlanBundle> plan_bundle;
  tfs::ServableHandle<NetDefBundle> netdef_bundle;
  tfs::ServableHandle<SavedModelBundle> saved_model_bundle;
  tfs::ServableHandle<CustomBundle> custom_bundle;
};
}  // namespace

void
InferenceServer::HandleInfer(
  RequestStatus* request_status,
  std::shared_ptr<InferRequestProvider> request_provider,
  std::shared_ptr<InferResponseProvider> response_provider,
  std::shared_ptr<ModelInferStats> infer_stats,
  std::function<void()> OnCompleteInferRPC, bool async_frontend)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    RequestStatusFactory::Create(
      request_status, 0, id_, RequestStatusCode::UNAVAILABLE,
      "Server not ready");
    OnCompleteInferRPC();
    return;
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);
  const uint64_t request_id = NextRequestId();

  tensorflow::Status status = tensorflow::Status::OK();

  // Create the model-spec. A negative version indicates that the
  // latest version of the model should be used.
  tfs::ModelSpec model_spec;
  model_spec.set_name(request_provider->ModelName());
  if (request_provider->ModelVersion() >= 0) {
    model_spec.mutable_version()->set_value(request_provider->ModelVersion());
  }

  auto state = std::make_shared<AsyncState>();

  std::function<void()> handle;

  Platform platform;
  status = ModelRepositoryManager::GetModelPlatform(
    request_provider->ModelName(), &platform);
  if (status.ok()) {
    switch (platform) {
      case Platform::PLATFORM_TENSORFLOW_GRAPHDEF:
        status =
          core_->GetServableHandle(model_spec, &(state->graphdef_bundle));
        if (status.ok()) {
          state->is =
            static_cast<InferenceServable*>(state->graphdef_bundle.get());
        }
        break;
      case Platform::PLATFORM_TENSORFLOW_SAVEDMODEL:
        status =
          core_->GetServableHandle(model_spec, &(state->saved_model_bundle));
        if (status.ok()) {
          state->is =
            static_cast<InferenceServable*>(state->saved_model_bundle.get());
        }
        break;
      case Platform::PLATFORM_TENSORRT_PLAN:
        status = core_->GetServableHandle(model_spec, &(state->plan_bundle));
        if (status.ok()) {
          state->is = static_cast<InferenceServable*>(state->plan_bundle.get());
        }
        break;
      case Platform::PLATFORM_CAFFE2_NETDEF:
        status = core_->GetServableHandle(model_spec, &(state->netdef_bundle));
        if (status.ok()) {
          state->is =
            static_cast<InferenceServable*>(state->netdef_bundle.get());
        }
        break;
      case Platform::PLATFORM_CUSTOM:
        status = core_->GetServableHandle(model_spec, &(state->custom_bundle));
        if (status.ok()) {
          state->is =
            static_cast<InferenceServable*>(state->custom_bundle.get());
        }
        break;
      default:
        break;
    }
  }

  infer_stats->SetRequestedVersion(request_provider->ModelVersion());
  infer_stats->SetModelServable(state->is);
  infer_stats->SetBatchSize(request_provider->RequestHeader().batch_size());

  if (!status.ok() || (state->is == nullptr)) {
    status = tensorflow::errors::Unavailable(
      "Inference request for unknown model '", request_provider->ModelName(),
      "'");
  }

  auto OnCompleteHandleInfer =
    [this, OnCompleteInferRPC, state, response_provider, request_status,
     request_id, infer_stats](tensorflow::Status status) mutable {
      if (status.ok()) {
        auto status = response_provider->FinalizeResponse(*(state->is));
        if (status.ok()) {
          RequestStatusFactory::Create(request_status, request_id, id_, status);
          OnCompleteInferRPC();
          return;
        }
      }
      // Report only stats that are relevant for a failed inference run.
      infer_stats->SetFailed(true);
      LOG_VERBOSE(1) << "Infer failed: "
                     << status.error_message();  // should logged as an error
      RequestStatusFactory::Create(request_status, request_id, id_, status);
      OnCompleteInferRPC();
    };

  if (status.ok()) {
    // we need to capture the servable handle to keep it alive
    // it goes away when it goes out of scope
    if (async_frontend) {
      state->is->AsyncRun(
        infer_stats, request_provider, response_provider,
        OnCompleteHandleInfer);
    } else {
      state->is->Run(
        infer_stats, request_provider, response_provider,
        OnCompleteHandleInfer);
    }
  } else {
    OnCompleteHandleInfer(status);
  }
}

void
InferenceServer::HandleStatus(
  RequestStatus* request_status, ServerStatus* server_status,
  const std::string& model_name)
{
  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    RequestStatusFactory::Create(
      request_status, 0, id_, RequestStatusCode::UNAVAILABLE, "Server exiting");
    return;
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);
  const uint64_t request_id = NextRequestId();

  tfs::ServableStateMonitor* monitor = nullptr;
  if (core_ != nullptr) {
    monitor = core_->servable_state_monitor();
  }

  // If no specific model request just return the entire status
  // object.
  if (model_name.empty()) {
    RequestStatusFactory::Create(
      request_status, request_id, id_,
      status_manager_->Get(
        server_status, id_, ready_state_, UptimeNs(), monitor));
  } else {
    RequestStatusFactory::Create(
      request_status, request_id, id_,
      status_manager_->Get(
        server_status, id_, ready_state_, UptimeNs(), model_name, monitor));
  }
}

uint64_t
InferenceServer::UptimeNs() const
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);

  uint64_t now_ns = now.tv_sec * NANOS_PER_SECOND + now.tv_nsec;
  return now_ns - start_time_ns_;
}

tensorflow::Status
InferenceServer::ParseProtoTextFile(
  const std::string& file, google::protobuf::Message* message)
{
  std::unique_ptr<tensorflow::ReadOnlyMemoryRegion> file_data;
  TF_RETURN_IF_ERROR(
    tensorflow::Env::Default()->NewReadOnlyMemoryRegionFromFile(
      file, &file_data));
  std::string file_data_str(
    reinterpret_cast<const char*>(file_data->data()), file_data->length());
  if (tensorflow::protobuf::TextFormat::ParseFromString(
        file_data_str, message)) {
    return tensorflow::Status::OK();
  } else {
    return tensorflow::errors::InvalidArgument(
      "Invalid protobuf file: '", file, "'");
  }
}

tfs::PlatformConfigMap
InferenceServer::BuildPlatformConfigMap(
  float tf_gpu_memory_fraction, bool tf_allow_soft_placement)
{
  ::google::protobuf::Any graphdef_source_adapter_config;
  ::google::protobuf::Any saved_model_source_adapter_config;
  ::google::protobuf::Any plan_source_adapter_config;
  ::google::protobuf::Any netdef_source_adapter_config;
  ::google::protobuf::Any custom_source_adapter_config;

  //// Tensorflow GraphDef
  {
    GraphDefBundleSourceAdapterConfig graphdef_config;

    graphdef_config.set_autofill(!strict_model_config_);

    // Tensorflow session config
    if (tf_gpu_memory_fraction == 0.0) {
      graphdef_config.mutable_session_config()
        ->mutable_gpu_options()
        ->set_allow_growth(true);
    } else {
      graphdef_config.mutable_session_config()
        ->mutable_gpu_options()
        ->set_per_process_gpu_memory_fraction(tf_gpu_memory_fraction);
    }

    graphdef_config.mutable_session_config()->set_allow_soft_placement(
      tf_allow_soft_placement);
    graphdef_source_adapter_config.PackFrom(graphdef_config);
  }

  //// Tensorflow SavedModel
  {
    SavedModelBundleSourceAdapterConfig saved_model_config;

    saved_model_config.set_autofill(!strict_model_config_);

    if (tf_gpu_memory_fraction == 0.0) {
      saved_model_config.mutable_session_config()
        ->mutable_gpu_options()
        ->set_allow_growth(true);
    } else {
      saved_model_config.mutable_session_config()
        ->mutable_gpu_options()
        ->set_per_process_gpu_memory_fraction(tf_gpu_memory_fraction);
    }

    saved_model_config.mutable_session_config()->set_allow_soft_placement(
      tf_allow_soft_placement);
    saved_model_source_adapter_config.PackFrom(saved_model_config);
  }

  //// Caffe NetDef
  {
    NetDefBundleSourceAdapterConfig netdef_config;
    netdef_config.set_autofill(!strict_model_config_);
    netdef_source_adapter_config.PackFrom(netdef_config);
  }

  //// TensorRT
  {
    PlanBundleSourceAdapterConfig plan_config;
    plan_config.set_autofill(!strict_model_config_);
    plan_source_adapter_config.PackFrom(plan_config);
  }

  //// Custom
  {
    CustomBundleSourceAdapterConfig custom_config;
    custom_source_adapter_config.PackFrom(custom_config);
  }

  tfs::PlatformConfigMap platform_config_map;

  (*(*platform_config_map
        .mutable_platform_configs())[kTensorFlowGraphDefPlatform]
      .mutable_source_adapter_config()) = graphdef_source_adapter_config;
  (*(*platform_config_map
        .mutable_platform_configs())[kTensorFlowSavedModelPlatform]
      .mutable_source_adapter_config()) = saved_model_source_adapter_config;
  (*(*platform_config_map.mutable_platform_configs())[kCaffe2NetDefPlatform]
      .mutable_source_adapter_config()) = netdef_source_adapter_config;
  (*(*platform_config_map.mutable_platform_configs())[kTensorRTPlanPlatform]
      .mutable_source_adapter_config()) = plan_source_adapter_config;
  (*(*platform_config_map.mutable_platform_configs())[kCustomPlatform]
      .mutable_source_adapter_config()) = custom_source_adapter_config;

  return platform_config_map;
}

}}  // namespace nvidia::inferenceserver
