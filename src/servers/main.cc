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

#include <getopt.h>
#include <stdint.h>
#include <unistd.h>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <mutex>

#include "src/core/logging.h"
#include "src/core/metrics.h"
#include "src/core/server.h"
#include "src/core/status.h"
#include "src/servers/grpc_server.h"
#include "src/servers/http_server.h"

namespace {

// The inference server object. Once this server is successfully
// created it does *not* transition back to a nullptr value and it is
// *not* explicitly destructed. Thus we assume that 'server_' can
// always be dereferenced.
nvidia::inferenceserver::InferenceServer* server_ = nullptr;

// Exit mutex and cv used to signal the main thread that it should
// close the server and exit.
volatile bool exiting_ = false;
std::mutex exit_mu_;
std::condition_variable exit_cv_;

// If true then exit if the inference-server fails to initialize
// completely but is still in an operational state (i.e. one or more
// models fail to load but the server is otherwise ok). If false then
// exit if inference-server doesn't completely initialize (e.g. will
// exit if even one model fails to load).
bool exit_on_failed_init_ = true;

// The HTTP and GRPC services
std::unique_ptr<nvidia::inferenceserver::HTTPServer> http_service_;
std::unique_ptr<nvidia::inferenceserver::GRPCServer> grpc_service_;

// The HTTP and GRPC ports. Initialized to default values and
// modifyied based on command-line args. Set to -1 to indicate the
// protocol is disabled.
int http_port_ = 8000;
int grpc_port_ = 8001;

// The metric port. Initialized to default values and modifyied based
// on command-line args. Set to -1 to indicate the protocol is
// disabled.
int metrics_port_ = 8002;

// The number of threads to initialize for the HTTP front-end.
int http_thread_cnt_ = 8;

// Command-line options
enum OptionId {
  OPTION_HELP = 1000,
  OPTION_LOG_VERBOSE,
  OPTION_LOG_INFO,
  OPTION_LOG_WARNING,
  OPTION_LOG_ERROR,
  OPTION_ID,
  OPTION_MODEL_STORE,
  OPTION_EXIT_ON_ERROR,
  OPTION_STRICT_MODEL_CONFIG,
  OPTION_STRICT_READINESS,
  OPTION_ALLOW_PROFILING,
  OPTION_ALLOW_GRPC,
  OPTION_ALLOW_HTTP,
  OPTION_ALLOW_METRICS,
  OPTION_GRPC_PORT,
  OPTION_HTTP_PORT,
  OPTION_METRICS_PORT,
  OPTION_HTTP_THREAD_COUNT,
  OPTION_ALLOW_POLL_REPO,
  OPTION_POLL_REPO_SECS,
  OPTION_EXIT_TIMEOUT_SECS,
  OPTION_TF_ALLOW_SOFT_PLACEMENT,
  OPTION_TF_GPU_MEMORY_FRACTION,
};

struct Option {
  Option(OptionId id, std::string flag, std::string desc, bool has_arg = true)
      : id_(id), flag_(flag), desc_(desc), has_arg_(has_arg)
  {
  }

  struct option GetLongOption() const
  {
    struct option lo {
      flag_.c_str(), (has_arg_) ? required_argument : no_argument, nullptr, id_
    };
    return lo;
  }

  const OptionId id_;
  const std::string flag_;
  const std::string desc_;
  const bool has_arg_;
};

std::vector<Option> options_{
    {OPTION_HELP, "help", "Print usage", false},
    {OPTION_LOG_VERBOSE, "log-verbose", "Enable/disable verbose-level logging"},
    {OPTION_LOG_INFO, "log-info", "Enable/disable info-level logging"},
    {OPTION_LOG_WARNING, "log-warning", "Enable/disable warning-level logging"},
    {OPTION_LOG_ERROR, "log-error", "Enable/disable error-level logging"},
    {OPTION_ID, "id", "Identifier for this server"},
    {OPTION_MODEL_STORE, "model-store", "Path to model repository directory"},
    {OPTION_EXIT_ON_ERROR, "exit-on-error",
     "Exit the inference server if an error occurs during initialization."},
    {OPTION_STRICT_MODEL_CONFIG, "strict-model-config",
     "If true model configuration files must be provided and all required "
     "configuration settings must be specified. If false the model "
     "configuration may be absent or only partially specified and the "
     "server will attempt to derive the missing required configuration."},
    {OPTION_STRICT_READINESS, "strict-readiness",
     "If true /api/health/ready endpoint indicates ready if the server "
     "is responsive and all models are available. If false "
     "/api/health/ready endpoint indicates ready if server is responsive "
     "even if some/all models are unavailable."},
    {OPTION_ALLOW_PROFILING, "allow-profiling", "Allow server profiling."},
    {OPTION_ALLOW_GRPC, "allow-grpc",
     "Allow the server to listen for GRPC requests."},
    {OPTION_ALLOW_HTTP, "allow-http",
     "Allow the server to listen for HTTP requests."},
    {OPTION_ALLOW_METRICS, "allow-metrics",
     "Allow the server to provide prometheus metrics."},
    {OPTION_GRPC_PORT, "grpc-port",
     "The port for the server to listen on for GRPC requests."},
    {OPTION_HTTP_PORT, "http-port",
     "The port for the server to listen on for HTTP requests."},
    {OPTION_METRICS_PORT, "metrics-port",
     "The port reporting prometheus metrics."},
    {OPTION_HTTP_THREAD_COUNT, "http-thread-count",
     "Number of threads handling HTTP requests."},
    {OPTION_ALLOW_POLL_REPO, "allow-poll-model-repository",
     "Poll the model repository to detect changes. The poll rate is "
     "controlled by 'repository-poll-secs'."},
    {OPTION_POLL_REPO_SECS, "repository-poll-secs",
     "Interval in seconds between each poll of the model repository to check "
     "for changes. A value of zero indicates that the repository is checked "
     "only a single time at startup. Valid only when "
     "--allow-poll-model-repository=true is specified."},
    {OPTION_EXIT_TIMEOUT_SECS, "exit-timeout-secs",
     "Timeout (in seconds) when exiting to wait for in-flight inferences to "
     "finish. After the timeout expires the server exits even if inferences "
     "are still in flight."},
    {OPTION_TF_ALLOW_SOFT_PLACEMENT, "tf-allow-soft-placement",
     "Instruct TensorFlow to use CPU implementation of an operation when "
     "a GPU implementation is not available."},
    {OPTION_TF_GPU_MEMORY_FRACTION, "tf-gpu-memory-fraction",
     "Reserve a portion of GPU memory for TensorFlow models. Default "
     "value 0.0 indicates that TensorFlow should dynamically allocate "
     "memory as needed. Value of 1.0 indicates that TensorFlow should "
     "allocate all of GPU memory."}};


void
SignalHandler(int signum)
{
  // Don't need a mutex here since signals should be disabled while in
  // the handler.
  LOG_INFO << "Interrupt signal (" << signum << ") received.";

  // Do nothing if already exiting...
  if (exiting_)
    return;

  {
    std::unique_lock<std::mutex> lock(exit_mu_);
    exiting_ = true;
  }

  exit_cv_.notify_all();
}

std::unique_ptr<nvidia::inferenceserver::GRPCServer>
StartGrpcService(nvidia::inferenceserver::InferenceServer* server)
{
  std::unique_ptr<nvidia::inferenceserver::GRPCServer> service;
  nvidia::inferenceserver::Status status =
      nvidia::inferenceserver::GRPCServer::Create(server, grpc_port_, &service);
  if (status.IsOk()) {
    status = service->Start();
  }

  if (!status.IsOk()) {
    service.reset();
  }

  return std::move(service);
}

std::unique_ptr<nvidia::inferenceserver::HTTPServer>
StartHttpService(nvidia::inferenceserver::InferenceServer* server)
{
  std::unique_ptr<nvidia::inferenceserver::HTTPServer> service;
  nvidia::inferenceserver::Status status =
      nvidia::inferenceserver::HTTPServer::Create(
          server, http_port_, http_thread_cnt_, &service);
  if (status.IsOk()) {
    status = service->Start();
  }

  if (!status.IsOk()) {
    service.reset();
  }

  return std::move(service);
}

bool
StartEndpoints(nvidia::inferenceserver::InferenceServer* server)
{
  LOG_INFO << "Starting endpoints, '" << server->Id() << "' listening on";

  // Enable gRPC endpoint if requested...
  if (grpc_port_ != -1) {
    LOG_INFO << " localhost:" << std::to_string(grpc_port_)
             << " for gRPC requests";
    grpc_service_ = StartGrpcService(server);
    if (grpc_service_ == nullptr) {
      LOG_ERROR << "Failed to start gRPC service";
      return false;
    }
  }

  // Enable HTTP endpoint if requested...
  if (http_port_ != -1) {
    LOG_INFO << " localhost:" << std::to_string(http_port_)
             << " for HTTP requests";

    http_service_ = StartHttpService(server);
    if (http_service_ == nullptr) {
      LOG_ERROR << "Failed to start HTTP service";
      return false;
    }
  }

  // Enable metrics endpoint if requested...
  if (metrics_port_ != -1) {
    LOG_INFO << " localhost:" << std::to_string(metrics_port_)
             << " for metric reporting";
    nvidia::inferenceserver::Metrics::Initialize(metrics_port_);
  }

  return true;
}

std::string
Usage()
{
  std::string usage("Usage:\n");
  for (const auto& o : options_) {
    usage += "--" + o.flag_ + "\t" + o.desc_ + "\n";
  }

  return usage;
}

bool
ParseBoolOption(const std::string arg)
{
  if ((arg == "true") || (arg == "True") || (arg == "1")) {
    return true;
  }
  if ((arg == "false") || (arg == "False") || (arg == "0")) {
    return false;
  }

  LOG_ERROR << "invalid value for bool option: " << arg;
  LOG_ERROR << Usage();
  exit(1);
}

int
ParseIntOption(const std::string arg)
{
  return std::stoi(arg);
}

float
ParseFloatOption(const std::string arg)
{
  return std::stof(arg);
}

bool
Parse(nvidia::inferenceserver::InferenceServer* server, int argc, char** argv)
{
  std::string server_id(server->Id());
  std::string model_store_path(server->ModelStorePath());
  bool strict_model_config = server->StrictModelConfigEnabled();
  bool strict_readiness = server->StrictReadinessEnabled();
  bool allow_profiling = server->ProfilingEnabled();
  bool tf_allow_soft_placement = server->TensorFlowSoftPlacementEnabled();
  float tf_gpu_memory_fraction = server->TensorFlowGPUMemoryFraction();
  int32_t exit_timeout_secs = server->ExitTimeoutSeconds();
  int32_t repository_poll_secs = server->RepositoryPollSeconds();

  bool exit_on_error = exit_on_failed_init_;

  bool allow_http = true;
  bool allow_grpc = true;
  bool allow_metrics = true;
  int32_t http_port = http_port_;
  int32_t grpc_port = grpc_port_;
  int32_t metrics_port = metrics_port_;
  int32_t http_thread_cnt = http_thread_cnt_;

  bool allow_poll_model_repository = repository_poll_secs > 0;

  bool log_info = true;
  bool log_warn = true;
  bool log_error = true;
  int32_t log_verbose = 0;

  std::vector<struct option> long_options;
  for (const auto& o : options_) {
    long_options.push_back(o.GetLongOption());
  }
  long_options.push_back({nullptr, 0, nullptr, 0});

  int flag;
  while ((flag = getopt_long(argc, argv, "", &long_options[0], NULL)) != -1) {
    switch (flag) {
      case OPTION_HELP:
      case '?':
        LOG_ERROR << Usage();
        return false;

      case OPTION_LOG_VERBOSE:
        log_verbose = ParseIntOption(optarg);
        break;
      case OPTION_LOG_INFO:
        log_info = ParseBoolOption(optarg);
        break;
      case OPTION_LOG_WARNING:
        log_warn = ParseBoolOption(optarg);
        break;
      case OPTION_LOG_ERROR:
        log_error = ParseBoolOption(optarg);
        break;

      case OPTION_ID:
        server_id = optarg;
        break;
      case OPTION_MODEL_STORE:
        model_store_path = optarg;
        break;

      case OPTION_EXIT_ON_ERROR:
        exit_on_error = ParseBoolOption(optarg);
        break;
      case OPTION_STRICT_MODEL_CONFIG:
        strict_model_config = ParseBoolOption(optarg);
        break;
      case OPTION_STRICT_READINESS:
        strict_readiness = ParseBoolOption(optarg);
        break;

      case OPTION_ALLOW_PROFILING:
        allow_profiling = ParseBoolOption(optarg);
        break;
      case OPTION_ALLOW_GRPC:
        allow_grpc = ParseBoolOption(optarg);
        break;
      case OPTION_ALLOW_HTTP:
        allow_http = ParseBoolOption(optarg);
        break;
      case OPTION_ALLOW_METRICS:
        allow_metrics = ParseBoolOption(optarg);
        break;
      case OPTION_GRPC_PORT:
        grpc_port = ParseIntOption(optarg);
        break;
      case OPTION_HTTP_PORT:
        http_port = ParseIntOption(optarg);
        break;
      case OPTION_METRICS_PORT:
        metrics_port = ParseIntOption(optarg);
        break;

      case OPTION_HTTP_THREAD_COUNT:
        http_thread_cnt = ParseIntOption(optarg);
        break;
      case OPTION_ALLOW_POLL_REPO:
        allow_poll_model_repository = ParseBoolOption(optarg);
        break;
      case OPTION_POLL_REPO_SECS:
        repository_poll_secs = ParseIntOption(optarg);
        break;
      case OPTION_EXIT_TIMEOUT_SECS:
        exit_timeout_secs = ParseIntOption(optarg);
        break;

      case OPTION_TF_ALLOW_SOFT_PLACEMENT:
        tf_allow_soft_placement = ParseBoolOption(optarg);
        break;
      case OPTION_TF_GPU_MEMORY_FRACTION:
        tf_gpu_memory_fraction = ParseFloatOption(optarg);
        break;
    }
  }

  if (optind < argc) {
    LOG_ERROR << "Unexpected argument: " << argv[optind];
    LOG_ERROR << Usage();
    return false;
  }

  LOG_ENABLE_INFO(log_info);
  LOG_ENABLE_WARNING(log_warn);
  LOG_ENABLE_ERROR(log_error);
  LOG_SET_VERBOSE(log_verbose);


  if (!allow_http && !allow_grpc) {
    LOG_ERROR << "At least one of the following options must be true: "
              << "--allow-http, --allow-grpc";
    return false;
  } else if (allow_http && allow_grpc && (http_port == grpc_port)) {
    LOG_ERROR << "The server cannot listen to HTTP requests "
              << "and gRPC requests at the same port";
    return false;
  } else if (
      allow_metrics && ((allow_grpc && (metrics_port == grpc_port)) ||
                        (allow_http && (metrics_port == http_port)))) {
    LOG_ERROR << "The server cannot provide metrics on same port used for "
              << "HTTP or gRPC requests";
    return false;
  }

  exit_on_failed_init_ = exit_on_error;
  http_port_ = allow_http ? http_port : -1;
  grpc_port_ = allow_grpc ? grpc_port : -1;
  metrics_port_ = allow_metrics ? metrics_port : -1;
  http_thread_cnt_ = http_thread_cnt;

  server->SetId(server_id);
  server->SetModelStorePath(model_store_path);
  server->SetStrictModelConfigEnabled(strict_model_config);
  server->SetStrictReadinessEnabled(strict_readiness);
  server->SetProfilingEnabled(allow_profiling);
  server->SetExitTimeoutSeconds(exit_timeout_secs);

  server->SetRepositoryPollSeconds(
      (allow_poll_model_repository) ? std::max(0, repository_poll_secs) : 0);

  server->SetTensorFlowSoftPlacementEnabled(tf_allow_soft_placement);
  server->SetTensorFlowGPUMemoryFraction(tf_gpu_memory_fraction);

  return true;
}

}  // namespace

int
main(int argc, char** argv)
{
  // Create the inference server
  server_ = new nvidia::inferenceserver::InferenceServer();

  // Parse command-line using defaults provided by the inference
  // server. Update inference server options appropriately.
  if (!Parse(server_, argc, argv)) {
    exit(1);
  }

  // Start the HTTP, GRPC, and metrics endpoints.
  if (!StartEndpoints(server_)) {
    exit(1);
  }

  // Initialize the inference server
  if (!server_->Init() && exit_on_failed_init_) {
    exit(1);
  }

  // Trap SIGINT and SIGTERM to allow server to exit gracefully
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  // Wait until a signal terminates the server...
  while (!exiting_) {
    uint32_t poll_secs = server_->RepositoryPollSeconds();

    // If enabled, poll the model repository to see if there have been
    // any changes.
    if (poll_secs > 0) {
      nvidia::inferenceserver::Status status = server_->PollModelRepository();
      if (!status.IsOk()) {
        LOG_ERROR << "Failed to poll model repository: " << status.Message();
      }
    }

    // Wait for the polling interval (or a long time if polling is not
    // enabled). Will be woken if the server is exiting.
    std::unique_lock<std::mutex> lock(exit_mu_);
    std::chrono::seconds wait_timeout((poll_secs == 0) ? 3600 : poll_secs);
    exit_cv_.wait_for(lock, wait_timeout);
  }

  bool stop_status = server_->Stop();

  if (grpc_service_) {
    grpc_service_->Stop();
  }

  if (http_service_ != nullptr) {
    http_service_->Stop();
  }

  return (stop_status) ? 0 : 1;
}
