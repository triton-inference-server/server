// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <list>
#include <mutex>
#include <set>
#include <sstream>

#ifdef TRITON_ENABLE_ASAN
#include <sanitizer/lsan_interface.h>
#endif  // TRITON_ENABLE_ASAN

#include "src/core/logging.h"
#include "src/core/tritonserver.h"
#include "src/servers/common.h"
#include "src/servers/shared_memory_manager.h"
#include "src/servers/tracer.h"

#if defined(TRITON_ENABLE_HTTP) || defined(TRITON_ENABLE_METRICS)
#include "src/servers/http_server.h"
#endif  // TRITON_ENABLE_HTTP|| TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_GRPC
#include "src/servers/grpc_server.h"
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_GPU
static_assert(
    TRITON_MIN_COMPUTE_CAPABILITY >= 1.0,
    "Invalid TRITON_MIN_COMPUTE_CAPABILITY specified");
#endif  // TRITON_ENABLE_GPU

namespace {

// Exit mutex and cv used to signal the main thread that it should
// close the server and exit.
volatile bool exiting_ = false;
std::mutex exit_mu_;
std::condition_variable exit_cv_;

// Interval, in seconds, when the model repository is polled for
// changes.
int32_t repository_poll_secs_ = 15;

// The HTTP, GRPC and metrics service/s and ports. Initialized to
// default values and modifyied based on command-line args. Set to -1
// to indicate the protocol is disabled.
#ifdef TRITON_ENABLE_HTTP
std::unique_ptr<nvidia::inferenceserver::HTTPServer> http_service_;
bool allow_http_ = true;
int32_t http_port_ = 8000;
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_GRPC
std::unique_ptr<nvidia::inferenceserver::GRPCServer> grpc_service_;
bool allow_grpc_ = true;
int32_t grpc_port_ = 8001;
bool grpc_use_ssl_ = false;
nvidia::inferenceserver::SslOptions grpc_ssl_options_;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
std::unique_ptr<nvidia::inferenceserver::HTTPServer> metrics_service_;
bool allow_metrics_ = true;
int32_t metrics_port_ = 8002;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_TRACING
std::string trace_filepath_;
TRITONSERVER_InferenceTraceLevel trace_level_ =
    TRITONSERVER_TRACE_LEVEL_DISABLED;
int32_t trace_rate_ = 1000;
#endif  // TRITON_ENABLE_TRACING

#if defined(TRITON_ENABLE_GRPC)
// The maximum number of inference request/response objects that
// remain allocated for reuse. As long as the number of in-flight
// requests doesn't exceed this value there will be no
// allocation/deallocation of request/response objects.
int grpc_infer_allocation_pool_size_ = 8;
#endif  // TRITON_ENABLE_GRPC

#if defined(TRITON_ENABLE_HTTP)
// The number of threads to initialize for the HTTP front-end.
int http_thread_cnt_ = 8;
#endif  // TRITON_ENABLE_HTTP

// Command-line options
enum OptionId {
  OPTION_HELP = 1000,
#ifdef TRITON_ENABLE_LOGGING
  OPTION_LOG_VERBOSE,
  OPTION_LOG_INFO,
  OPTION_LOG_WARNING,
  OPTION_LOG_ERROR,
#endif  // TRITON_ENABLE_LOGGING
  OPTION_ID,
  OPTION_MODEL_REPOSITORY,
  OPTION_EXIT_ON_ERROR,
  OPTION_STRICT_MODEL_CONFIG,
  OPTION_STRICT_READINESS,
#if defined(TRITON_ENABLE_HTTP)
  OPTION_ALLOW_HTTP,
  OPTION_HTTP_PORT,
  OPTION_HTTP_THREAD_COUNT,
#endif  // TRITON_ENABLE_HTTP
#if defined(TRITON_ENABLE_GRPC)
  OPTION_ALLOW_GRPC,
  OPTION_GRPC_PORT,
  OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE,
  OPTION_GRPC_USE_SSL,
  OPTION_GRPC_SERVER_CERT,
  OPTION_GRPC_SERVER_KEY,
  OPTION_GRPC_ROOT_CERT,
#endif  // TRITON_ENABLE_GRPC
#ifdef TRITON_ENABLE_METRICS
  OPTION_ALLOW_METRICS,
  OPTION_ALLOW_GPU_METRICS,
  OPTION_METRICS_PORT,
#endif  // TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_TRACING
  OPTION_TRACE_FILEPATH,
  OPTION_TRACE_LEVEL,
  OPTION_TRACE_RATE,
#endif  // TRITON_ENABLE_TRACING
  OPTION_MODEL_CONTROL_MODE,
  OPTION_POLL_REPO_SECS,
  OPTION_STARTUP_MODEL,
  OPTION_PINNED_MEMORY_POOL_BYTE_SIZE,
  OPTION_CUDA_MEMORY_POOL_BYTE_SIZE,
  OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY,
  OPTION_EXIT_TIMEOUT_SECS,
  OPTION_BACKEND_DIR,
  OPTION_BUFFER_MANAGER_THREAD_COUNT,
  OPTION_BACKEND_CONFIG
};

struct Option {
  static constexpr const char* ArgNone = "";
  static constexpr const char* ArgBool = "boolean";
  static constexpr const char* ArgFloat = "float";
  static constexpr const char* ArgInt = "integer";
  static constexpr const char* ArgStr = "string";

  Option(OptionId id, std::string flag, std::string arg_desc, std::string desc)
      : id_(id), flag_(flag), arg_desc_(arg_desc), desc_(desc)
  {
  }

  struct option GetLongOption() const
  {
    struct option lo {
      flag_.c_str(), (!arg_desc_.empty()) ? required_argument : no_argument,
          nullptr, id_
    };
    return lo;
  }

  const OptionId id_;
  const std::string flag_;
  const std::string arg_desc_;
  const std::string desc_;
};

std::vector<Option> options_
{
  {OPTION_HELP, "help", Option::ArgNone, "Print usage"},
#ifdef TRITON_ENABLE_LOGGING
      {OPTION_LOG_VERBOSE, "log-verbose", Option::ArgInt,
       "Set verbose logging level. Zero (0) disables verbose logging and "
       "values >= 1 enable verbose logging."},
      {OPTION_LOG_INFO, "log-info", Option::ArgBool,
       "Enable/disable info-level logging."},
      {OPTION_LOG_WARNING, "log-warning", Option::ArgBool,
       "Enable/disable warning-level logging."},
      {OPTION_LOG_ERROR, "log-error", Option::ArgBool,
       "Enable/disable error-level logging."},
#endif  // TRITON_ENABLE_LOGGING
      {OPTION_ID, "id", Option::ArgStr, "Identifier for this server."},
      {OPTION_MODEL_REPOSITORY, "model-store", Option::ArgStr,
       "Equivalent to --model-repository."},
      {OPTION_MODEL_REPOSITORY, "model-repository", Option::ArgStr,
       "Path to model repository directory. It may be specified multiple times "
       "to add multiple model repositories. Note that if a model is not unique "
       "across all model repositories at any time, the model will not be "
       "available."},
      {OPTION_EXIT_ON_ERROR, "exit-on-error", Option::ArgBool,
       "Exit the inference server if an error occurs during initialization."},
      {OPTION_STRICT_MODEL_CONFIG, "strict-model-config", Option::ArgBool,
       "If true model configuration files must be provided and all required "
       "configuration settings must be specified. If false the model "
       "configuration may be absent or only partially specified and the "
       "server will attempt to derive the missing required configuration."},
      {OPTION_STRICT_READINESS, "strict-readiness", Option::ArgBool,
       "If true /v2/health/ready endpoint indicates ready if the server "
       "is responsive and all models are available. If false "
       "/v2/health/ready endpoint indicates ready if server is responsive "
       "even if some/all models are unavailable."},
#if defined(TRITON_ENABLE_HTTP)
      {OPTION_ALLOW_HTTP, "allow-http", Option::ArgBool,
       "Allow the server to listen for HTTP requests."},
      {OPTION_HTTP_PORT, "http-port", Option::ArgInt,
       "The port for the server to listen on for HTTP requests."},
      {OPTION_HTTP_THREAD_COUNT, "http-thread-count", Option::ArgInt,
       "Number of threads handling HTTP requests."},
#endif  // TRITON_ENABLE_HTTP
#if defined(TRITON_ENABLE_GRPC)
      {OPTION_ALLOW_GRPC, "allow-grpc", Option::ArgBool,
       "Allow the server to listen for GRPC requests."},
      {OPTION_GRPC_PORT, "grpc-port", Option::ArgInt,
       "The port for the server to listen on for GRPC requests."},
      {OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE,
       "grpc-infer-allocation-pool-size", Option::ArgInt,
       "The maximum number of inference request/response objects that remain "
       "allocated for reuse. As long as the number of in-flight requests "
       "doesn't exceed this value there will be no allocation/deallocation of "
       "request/response objects."},
      {OPTION_GRPC_USE_SSL, "grpc-use-ssl", Option::ArgBool,
       "Use SSL authentication for GRPC requests. Default is false."},
      {OPTION_GRPC_SERVER_CERT, "grpc-server-cert", Option::ArgStr,
       "File holding PEM-encoded server certificate. Ignored unless "
       "--grpc-use-ssl is true."},
      {OPTION_GRPC_SERVER_KEY, "grpc-server-key", Option::ArgStr,
       "File holding PEM-encoded server key. Ignored unless "
       "--grpc-use-ssl is true."},
      {OPTION_GRPC_ROOT_CERT, "grpc-root-cert", Option::ArgStr,
       "File holding PEM-encoded root certificate. Ignore unless "
       "--grpc-use-ssl is false."},
#endif  // TRITON_ENABLE_GRPC
#ifdef TRITON_ENABLE_METRICS
      {OPTION_ALLOW_METRICS, "allow-metrics", Option::ArgBool,
       "Allow the server to provide prometheus metrics."},
      {OPTION_ALLOW_GPU_METRICS, "allow-gpu-metrics", Option::ArgBool,
       "Allow the server to provide GPU metrics. Ignored unless "
       "--allow-metrics is true."},
      {OPTION_METRICS_PORT, "metrics-port", Option::ArgInt,
       "The port reporting prometheus metrics."},
#endif  // TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_TRACING
      {OPTION_TRACE_FILEPATH, "trace-file", Option::ArgStr,
       "Set the file where trace output will be saved."},
      {OPTION_TRACE_LEVEL, "trace-level", Option::ArgStr,
       "Set the trace level. OFF to disable tracing, MIN for minimal tracing, "
       "MAX for maximal tracing. Default is OFF."},
      {OPTION_TRACE_RATE, "trace-rate", Option::ArgInt,
       "Set the trace sampling rate. Default is 1000."},
#endif  // TRITON_ENABLE_TRACING
      {OPTION_MODEL_CONTROL_MODE, "model-control-mode", Option::ArgStr,
       "Specify the mode for model management. Options are \"none\", \"poll\" "
       "and \"explicit\". The default is \"none\". "
       "For \"none\", the server will load all models in the model "
       "repository(s) at startup and will not make any changes to the load "
       "models after that. For \"poll\", the server will poll the model "
       "repository(s) to detect changes and will load/unload models based on "
       "those changes. The poll rate is controlled by 'repository-poll-secs'. "
       "For \"explicit\", model load and unload is initiated by using the "
       "model control APIs, and only models specified with --load-model will "
       "be loaded at startup."},
      {OPTION_POLL_REPO_SECS, "repository-poll-secs", Option::ArgInt,
       "Interval in seconds between each poll of the model repository to check "
       "for changes. Valid only when --model-control-mode=poll is "
       "specified."},
      {OPTION_STARTUP_MODEL, "load-model", Option::ArgStr,
       "Name of the model to be loaded on server startup. It may be specified "
       "multiple times to add multiple models. Note that this option will only "
       "take affect if --model-control-mode=explicit is true."},
      {OPTION_PINNED_MEMORY_POOL_BYTE_SIZE, "pinned-memory-pool-byte-size",
       Option::ArgInt,
       "The total byte size that can be allocated as pinned system memory. "
       "If GPU support is enabled, the server will allocate pinned system "
       "memory to accelerate data transfer between host and devices until it "
       "exceeds the specified byte size. This option will not affect the "
       "allocation conducted by the backend frameworks. Default is 256 MB."},
      {OPTION_CUDA_MEMORY_POOL_BYTE_SIZE, "cuda-memory-pool-byte-size",
       "<integer>:<integer>",
       "The total byte size that can be allocated as CUDA memory for the GPU "
       "device. If GPU support is enabled, the server will allocate CUDA "
       "memory to minimize data transfer between host and devices until it "
       "exceeds the specified byte size. This option will not affect the "
       "allocation conducted by the backend frameworks. The argument should be "
       "2 integers separated by colons in the format "
       "<GPU device ID>:<pool byte size>. This option can be used multiple "
       "times, but only once per GPU device. Subsequent uses will overwrite "
       "previous uses for the same GPU device. Default is 64 MB."},
      {OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY,
       "min-supported-compute-capability", Option::ArgFloat,
       "The minimum supported CUDA compute capability. GPUs that don't support "
       "this compute capability will not be used by the server."},
      {OPTION_EXIT_TIMEOUT_SECS, "exit-timeout-secs", Option::ArgInt,
       "Timeout (in seconds) when exiting to wait for in-flight inferences to "
       "finish. After the timeout expires the server exits even if inferences "
       "are still in flight."},
      {OPTION_BACKEND_DIR, "backend-directory", Option::ArgStr,
       "The global directory searched for backend shared libraries. Default is "
       "'/opt/tritonserver/backends'."},
      {OPTION_BUFFER_MANAGER_THREAD_COUNT, "buffer-manager-thread-count",
       Option::ArgInt,
       "The number of threads used to accelerate copies and other operations "
       "required to manage input and output tensor contents. Default is 0."},
  {
    OPTION_BACKEND_CONFIG, "backend-config", "<string>,<string>=<string>",
        "Specify a backend-specific configuration setting. The format of this "
        "flag is --backend-config=<backend_name>,<setting>=<value>. Where "
        "<backend_name> is the name of the backend, such as 'tensorrt'."
  }
};

void
SignalHandler(int signum)
{
  // Don't need a mutex here since signals should be disabled while in
  // the handler.
  std::cout << "Interrupt signal (" << signum << ") received." << std::endl;

  // Do nothing if already exiting...
  if (exiting_)
    return;

  {
    std::unique_lock<std::mutex> lock(exit_mu_);
    exiting_ = true;
  }

  exit_cv_.notify_all();
}

bool
CheckPortCollision()
{
#if defined(TRITON_ENABLE_HTTP) && defined(TRITON_ENABLE_GRPC)
  // Check if HTTP and GRPC have shared ports
  if ((grpc_port_ == http_port_) && (grpc_port_ != -1) && allow_http_ &&
      allow_grpc_) {
    std::cerr << "The server cannot listen to HTTP requests "
              << "and GRPC requests at the same port" << std::endl;
    return true;
  }
#endif  // TRITON_ENABLE_HTTP && TRITON_ENABLE_GRPC

#if defined(TRITON_ENABLE_GRPC) && defined(TRITON_ENABLE_METRICS)
  // Check if Metric and GRPC have shared ports
  if ((grpc_port_ == metrics_port_) && (metrics_port_ != -1) && allow_grpc_ &&
      allow_metrics_) {
    std::cerr << "The server cannot provide metrics on same port used for "
              << "GRPC requests" << std::endl;
    return true;
  }
#endif  // TRITON_ENABLE_GRPC && TRITON_ENABLE_METRICS

#if defined(TRITON_ENABLE_HTTP) && defined(TRITON_ENABLE_METRICS)
  // Check if Metric and HTTP have shared ports
  if ((http_port_ == metrics_port_) && (metrics_port_ != -1) && allow_http_ &&
      allow_metrics_) {
    std::cerr << "The server cannot provide metrics on same port used for "
              << "HTTP requests" << std::endl;
    return true;
  }
#endif  // TRITON_ENABLE_HTTP && TRITON_ENABLE_METRICS

  return false;
}

#ifdef TRITON_ENABLE_GRPC
TRITONSERVER_Error*
StartGrpcService(
    std::unique_ptr<nvidia::inferenceserver::GRPCServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    nvidia::inferenceserver::TraceManager* trace_manager,
    const std::shared_ptr<nvidia::inferenceserver::SharedMemoryManager>&
        shm_manager)
{
  TRITONSERVER_Error* err = nvidia::inferenceserver::GRPCServer::Create(
      server, trace_manager, shm_manager, grpc_port_, grpc_use_ssl_,
      grpc_ssl_options_, grpc_infer_allocation_pool_size_, service);
  if (err == nullptr) {
    err = (*service)->Start();
  }

  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_HTTP
TRITONSERVER_Error*
StartHttpService(
    std::unique_ptr<nvidia::inferenceserver::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    nvidia::inferenceserver::TraceManager* trace_manager,
    const std::shared_ptr<nvidia::inferenceserver::SharedMemoryManager>&
        shm_manager)
{
  TRITONSERVER_Error* err =
      nvidia::inferenceserver::HTTPServer::CreateAPIServer(
          server, trace_manager, shm_manager, http_port_, http_thread_cnt_,
          service);
  if (err == nullptr) {
    err = (*service)->Start();
  }

  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_METRICS
TRITONSERVER_Error*
StartMetricsService(
    std::unique_ptr<nvidia::inferenceserver::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server)
{
  TRITONSERVER_Error* err =
      nvidia::inferenceserver::HTTPServer::CreateMetricsServer(
          server, metrics_port_, 1 /* HTTP thread count */, service);
  if (err == nullptr) {
    err = (*service)->Start();
  }
  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRITON_ENABLE_METRICS

bool
StartEndpoints(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    nvidia::inferenceserver::TraceManager* trace_manager,
    const std::shared_ptr<nvidia::inferenceserver::SharedMemoryManager>&
        shm_manager)
{
#ifdef TRITON_ENABLE_GRPC
  // Enable GRPC endpoints if requested...
  if (allow_grpc_ && (grpc_port_ != -1)) {
    TRITONSERVER_Error* err =
        StartGrpcService(&grpc_service_, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start GRPC service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_HTTP
  // Enable HTTP endpoints if requested...
  if (allow_http_ && (http_port_ != -1)) {
    TRITONSERVER_Error* err =
        StartHttpService(&http_service_, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start HTTP service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_METRICS
  // Enable metrics endpoint if requested...
  if (metrics_port_ != -1) {
    TRITONSERVER_Error* err = StartMetricsService(&metrics_service_, server);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start Metrics service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_METRICS

  return true;
}

bool
StopEndpoints()
{
  bool ret = true;

#ifdef TRITON_ENABLE_HTTP
  if (http_service_) {
    TRITONSERVER_Error* err = http_service_->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop HTTP service");
      ret = false;
    }

    http_service_.reset();
  }
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_GRPC
  if (grpc_service_) {
    TRITONSERVER_Error* err = grpc_service_->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop GRPC service");
      ret = false;
    }

    grpc_service_.reset();
  }
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
  if (metrics_service_) {
    TRITONSERVER_Error* err = metrics_service_->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop Metrics service");
      ret = false;
    }

    metrics_service_.reset();
  }
#endif  // TRITON_ENABLE_METRICS

  return ret;
}

bool
StartTracing(nvidia::inferenceserver::TraceManager** trace_manager)
{
  *trace_manager = nullptr;

#ifdef TRITON_ENABLE_TRACING
  TRITONSERVER_Error* err = nullptr;

  // Configure tracing if host is specified.
  if (trace_level_ != TRITONSERVER_TRACE_LEVEL_DISABLED) {
    err = nvidia::inferenceserver::TraceManager::Create(
        trace_manager, trace_level_, trace_rate_, trace_filepath_);
  }

  if (err != nullptr) {
    LOG_TRITONSERVER_ERROR(err, "failed to configure tracing");
    *trace_manager = nullptr;
    return false;
  }
#endif  // TRITON_ENABLE_TRACING

  return true;
}

bool
StopTracing(nvidia::inferenceserver::TraceManager** trace_manager)
{
#ifdef TRITON_ENABLE_TRACING
  // We assume that at this point Triton has been stopped gracefully,
  // so can delete the trace manager to finalize the output.
  delete (*trace_manager);
  *trace_manager = nullptr;
#endif  // TRITON_ENABLE_TRACING

  return true;
}

std::string
FormatUsageMessage(std::string str, int offset)
{
  int width = 60;
  int current_pos = offset;
  while (current_pos + width < int(str.length())) {
    int n = str.rfind(' ', current_pos + width);
    if (n != int(std::string::npos)) {
      str.replace(n, 1, "\n\t");
      current_pos += (width + 9);
    }
  }

  return str;
}

std::string
Usage()
{
  std::stringstream ss;

  ss << "Usage: tritonserver [options]" << std::endl;
  for (const auto& o : options_) {
    if (!o.arg_desc_.empty()) {
      ss << "  --" << o.flag_ << " <" << o.arg_desc_ << ">" << std::endl
         << "\t" << FormatUsageMessage(o.desc_, 0) << std::endl;
    } else {
      ss << "  --" << o.flag_ << std::endl
         << "\t" << FormatUsageMessage(o.desc_, 0) << std::endl;
    }
  }

  // FIXME, once backends are more separated from core Triton the
  // documentation for these flags should be moved to the backend
  // documentation.
  ss << std::endl;
  ss << "For --backend-config for the 'tensorflow' backend the following flags "
        "are accepted."
     << std::endl;
  ss << "  --backend-config=tensorflow,allow-soft-placement=<boolean>"
     << std::endl;
  ss << "\t"
     << FormatUsageMessage(
            "Instruct TensorFlow to use CPU implementation of an operation "
            "when a GPU implementation is not available.",
            0)
     << std::endl;
  ss << "  --backend-config=tensorflow,gpu-memory-fraction=<float>"
     << std::endl;
  ss << "\t"
     << FormatUsageMessage(
            "Reserve a portion of GPU memory for TensorFlow models. Default "
            "value 0.0 indicates that TensorFlow should dynamically allocate "
            "memory as needed. Value of 1.0 indicates that TensorFlow should "
            "allocate all of GPU memory.",
            0)
     << std::endl;
  ss << "  --backend-config=tensorflow,version=<int>" << std::endl;
  ss << "\t"
     << FormatUsageMessage(
            "Select the version of the TensorFlow library to be used, "
            "available version is 1 and 2. Default TensorFlow version is 1.",
            0)
     << std::endl;

  return ss.str();
}

bool
ParseBoolOption(std::string arg)
{
  std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  if ((arg == "true") || (arg == "on") || (arg == "1")) {
    return true;
  }
  if ((arg == "false") || (arg == "off") || (arg == "0")) {
    return false;
  }

  std::cerr << "invalid value for bool option: " << arg << std::endl;
  std::cerr << Usage() << std::endl;
  exit(1);
}

int
ParseIntOption(const std::string arg)
{
  return std::stoi(arg);
}

int64_t
ParseLongLongOption(const std::string arg)
{
  return std::stoll(arg);
}

#if 0
float
ParseFloatOption(const std::string arg)
{
  return std::stof(arg);
}
#endif

double
ParseDoubleOption(const std::string arg)
{
  return std::stod(arg);
}

// Condition here merely to avoid compilation error, this function will
// be defined but not used otherwise.
#ifdef TRITON_ENABLE_LOGGING
int
ParseIntBoolOption(std::string arg)
{
  std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  if (arg == "true") {
    return 1;
  }
  if (arg == "false") {
    return 0;
  }

  return ParseIntOption(arg);
}
#endif  // TRITON_ENABLE_LOGGING

#ifdef TRITON_ENABLE_TRACING
TRITONSERVER_InferenceTraceLevel
ParseTraceLevelOption(std::string arg)
{
  std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  if ((arg == "false") || (arg == "off")) {
    return TRITONSERVER_TRACE_LEVEL_DISABLED;
  }
  if ((arg == "true") || (arg == "on") || (arg == "min")) {
    return TRITONSERVER_TRACE_LEVEL_MIN;
  }
  if (arg == "max") {
    return TRITONSERVER_TRACE_LEVEL_MAX;
  }

  std::cerr << "invalid value for trace level option: " << arg << std::endl;
  std::cerr << Usage() << std::endl;
  exit(1);
}
#endif  // TRITON_ENABLE_TRACING

std::tuple<std::string, std::string, std::string>
ParseBackendConfigOption(const std::string arg)
{
  // Format is "<backend_name>,<setting>=<value>"
  int delim_name = arg.find(",");
  int delim_setting = arg.find("=", delim_name + 1);

  // Check for 2 semicolons
  if ((delim_name < 0) || (delim_setting < 0)) {
    std::cerr << "--backend-config option format is '<backend "
                 "name>,<setting>=<value>'. Got "
              << arg << std::endl;
    exit(1);
  }

  std::string name_string = arg.substr(0, delim_name);
  std::string setting_string =
      arg.substr(delim_name + 1, delim_setting - delim_name - 1);
  std::string value_string = arg.substr(delim_setting + 1);

  if (name_string.empty() || setting_string.empty() || value_string.empty()) {
    std::cerr << "--backend-config option format is '<backend "
                 "name>,<setting>=<value>'. Got "
              << arg << std::endl;
    exit(1);
  }

  return {name_string, setting_string, value_string};
}

std::pair<int, uint64_t>
ParsePairOption(const std::string arg)
{
  int delim = arg.find(":");

  if ((delim < 0)) {
    std::cerr << "Cannot parse pair option due to incorrect number of inputs."
                 "--<pair option> argument requires format <key>:<value>. "
              << "Found: " << arg << std::endl;
    std::cerr << Usage() << std::endl;
    exit(1);
  }

  std::string key_string = arg.substr(0, delim);
  std::string value_string = arg.substr(delim + 1);

  // Specific conversion from key-value string to actual key-value type,
  // should be extracted out of this function if we need to parse
  // more pair option of different types.
  int key = ParseIntOption(key_string);
  uint64_t value = ParseLongLongOption(value_string);

  return {key, value};
}

bool
Parse(TRITONSERVER_ServerOptions** server_options, int argc, char** argv)
{
  std::string server_id("triton");
  std::set<std::string> model_repository_paths;
  bool exit_on_error = true;
  bool strict_model_config = true;
  bool strict_readiness = true;
  std::list<std::pair<int, uint64_t>> cuda_pools;
  int32_t exit_timeout_secs = 30;
  int32_t repository_poll_secs = repository_poll_secs_;
  int64_t pinned_memory_pool_byte_size = 1 << 28;
  int32_t buffer_manager_thread_count = 0;

  std::string backend_dir = "/opt/tritonserver/backends";
  std::vector<std::tuple<std::string, std::string, std::string>>
      backend_config_settings;

#ifdef TRITON_ENABLE_GPU
  double min_supported_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;
#else
  double min_supported_compute_capability = 0;
#endif  // TRITON_ENABLE_GPU

#if defined(TRITON_ENABLE_HTTP)
  int32_t http_port = http_port_;
  int32_t http_thread_cnt = http_thread_cnt_;
#endif  // TRITON_ENABLE_HTTP

#if defined(TRITON_ENABLE_GRPC)
  int32_t grpc_port = grpc_port_;
  int32_t grpc_use_ssl = grpc_use_ssl_;
  int32_t grpc_infer_allocation_pool_size = grpc_infer_allocation_pool_size_;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
  int32_t metrics_port = metrics_port_;
  bool allow_gpu_metrics = true;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_TRACING
  std::string trace_filepath = trace_filepath_;
  TRITONSERVER_InferenceTraceLevel trace_level = trace_level_;
  int32_t trace_rate = trace_rate_;
#endif  // TRITON_ENABLE_TRACING

  TRITONSERVER_ModelControlMode control_mode = TRITONSERVER_MODEL_CONTROL_NONE;
  std::set<std::string> startup_models_;

#ifdef TRITON_ENABLE_LOGGING
  bool log_info = true;
  bool log_warn = true;
  bool log_error = true;
  int32_t log_verbose = 0;
#endif  // TRITON_ENABLE_LOGGING

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
        std::cerr << Usage() << std::endl;
        return false;
#ifdef TRITON_ENABLE_LOGGING
      case OPTION_LOG_VERBOSE:
        log_verbose = ParseIntBoolOption(optarg);
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
#endif  // TRITON_ENABLE_LOGGING

      case OPTION_ID:
        server_id = optarg;
        break;
      case OPTION_MODEL_REPOSITORY:
        model_repository_paths.insert(optarg);
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

#if defined(TRITON_ENABLE_HTTP)
      case OPTION_ALLOW_HTTP:
        allow_http_ = ParseBoolOption(optarg);
        break;
      case OPTION_HTTP_PORT:
        http_port = ParseIntOption(optarg);
        break;
      case OPTION_HTTP_THREAD_COUNT:
        http_thread_cnt = ParseIntOption(optarg);
        break;
#endif  // TRITON_ENABLE_HTTP

#if defined(TRITON_ENABLE_GRPC)
      case OPTION_ALLOW_GRPC:
        allow_grpc_ = ParseBoolOption(optarg);
        break;
      case OPTION_GRPC_PORT:
        grpc_port = ParseIntOption(optarg);
        break;
      case OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE:
        grpc_infer_allocation_pool_size = ParseIntOption(optarg);
        break;
      case OPTION_GRPC_USE_SSL:
        grpc_use_ssl = ParseBoolOption(optarg);
        break;
      case OPTION_GRPC_SERVER_CERT:
        grpc_ssl_options_.server_cert = optarg;
        break;
      case OPTION_GRPC_SERVER_KEY:
        grpc_ssl_options_.server_key = optarg;
        break;
      case OPTION_GRPC_ROOT_CERT:
        grpc_ssl_options_.root_cert = optarg;
        break;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
      case OPTION_ALLOW_METRICS:
        allow_metrics_ = ParseBoolOption(optarg);
        break;
      case OPTION_ALLOW_GPU_METRICS:
        allow_gpu_metrics = ParseBoolOption(optarg);
        break;
      case OPTION_METRICS_PORT:
        metrics_port = ParseIntOption(optarg);
        break;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_TRACING
      case OPTION_TRACE_FILEPATH:
        trace_filepath = optarg;
        break;
      case OPTION_TRACE_LEVEL:
        trace_level = ParseTraceLevelOption(optarg);
        break;
      case OPTION_TRACE_RATE:
        trace_rate = ParseIntOption(optarg);
        break;
#endif  // TRITON_ENABLE_TRACING

      case OPTION_POLL_REPO_SECS:
        repository_poll_secs = ParseIntOption(optarg);
        break;
      case OPTION_STARTUP_MODEL:
        startup_models_.insert(optarg);
        break;
      case OPTION_MODEL_CONTROL_MODE: {
        std::string mode_str(optarg);
        std::transform(
            mode_str.begin(), mode_str.end(), mode_str.begin(), ::tolower);
        if (mode_str == "none") {
          control_mode = TRITONSERVER_MODEL_CONTROL_NONE;
        } else if (mode_str == "poll") {
          control_mode = TRITONSERVER_MODEL_CONTROL_POLL;
        } else if (mode_str == "explicit") {
          control_mode = TRITONSERVER_MODEL_CONTROL_EXPLICIT;
        } else {
          std::cerr << "invalid argument for --model-control-mode" << std::endl;
          std::cerr << Usage() << std::endl;
          return false;
        }
        break;
      }
      case OPTION_PINNED_MEMORY_POOL_BYTE_SIZE:
        pinned_memory_pool_byte_size = ParseLongLongOption(optarg);
        break;
      case OPTION_CUDA_MEMORY_POOL_BYTE_SIZE:
        cuda_pools.push_back(ParsePairOption(optarg));
        break;
      case OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY:
        min_supported_compute_capability = ParseDoubleOption(optarg);
        break;
      case OPTION_EXIT_TIMEOUT_SECS:
        exit_timeout_secs = ParseIntOption(optarg);
        break;
      case OPTION_BACKEND_DIR:
        backend_dir = optarg;
        break;
      case OPTION_BUFFER_MANAGER_THREAD_COUNT:
        buffer_manager_thread_count = ParseIntOption(optarg);
        break;
      case OPTION_BACKEND_CONFIG:
        backend_config_settings.push_back(ParseBackendConfigOption(optarg));
        break;
    }
  }

  if (optind < argc) {
    std::cerr << "Unexpected argument: " << argv[optind] << std::endl;
    std::cerr << Usage() << std::endl;
    return false;
  }

#ifdef TRITON_ENABLE_LOGGING
  // Initialize our own logging instance since it is used by GRPC and
  // HTTP endpoints. This logging instance is separate from the one in
  // libtritonserver so we must initialize explicitly.
  LOG_ENABLE_INFO(log_info);
  LOG_ENABLE_WARNING(log_warn);
  LOG_ENABLE_ERROR(log_error);
  LOG_SET_VERBOSE(log_verbose);
#endif  // TRITON_ENABLE_LOGGING

  repository_poll_secs_ = 0;
  if (control_mode == TRITONSERVER_MODEL_CONTROL_POLL) {
    repository_poll_secs_ = std::max(0, repository_poll_secs);
  }

#if defined(TRITON_ENABLE_HTTP)
  http_port_ = http_port;
  http_thread_cnt_ = http_thread_cnt;
#endif  // TRITON_ENABLE_HTTP

#if defined(TRITON_ENABLE_GRPC)
  grpc_port_ = grpc_port;
  grpc_infer_allocation_pool_size_ = grpc_infer_allocation_pool_size;
  grpc_use_ssl_ = grpc_use_ssl;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
  metrics_port_ = allow_metrics_ ? metrics_port : -1;
  allow_gpu_metrics = allow_metrics_ ? allow_gpu_metrics : false;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_TRACING
  trace_filepath_ = trace_filepath;
  trace_level_ = trace_level;
  trace_rate_ = trace_rate;
#endif  // TRITON_ENABLE_TRACING

  // Check if HTTP, GRPC and metrics port clash
  if (CheckPortCollision()) {
    return false;
  }

  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsNew(server_options), "creating server options");
  auto loptions = *server_options;
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetServerId(loptions, server_id.c_str()),
      "setting server ID");
  for (const auto& model_repository_path : model_repository_paths) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetModelRepositoryPath(
            loptions, model_repository_path.c_str()),
        "setting model repository path");
  }
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelControlMode(loptions, control_mode),
      "setting model control mode");
  for (const auto& model : startup_models_) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetStartupModel(loptions, model.c_str()),
        "setting startup model");
  }
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
          loptions, pinned_memory_pool_byte_size),
      "setting total pinned memory byte size");
  for (const auto& cuda_pool : cuda_pools) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
            loptions, cuda_pool.first, cuda_pool.second),
        "setting total CUDA memory byte size");
  }
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
          loptions, min_supported_compute_capability),
      "setting minimum supported CUDA compute capability");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetExitOnError(loptions, exit_on_error),
      "setting exit on error");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictModelConfig(
          loptions, strict_model_config),
      "setting strict model configuration");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictReadiness(loptions, strict_readiness),
      "setting strict readiness");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetExitTimeout(
          loptions, std::max(0, exit_timeout_secs)),
      "setting exit timeout");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(
          loptions, std::max(0, buffer_manager_thread_count)),
      "setting exit timeout");

#ifdef TRITON_ENABLE_LOGGING
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogInfo(loptions, log_info),
      "setting log info enable");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogWarn(loptions, log_warn),
      "setting log warn enable");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogError(loptions, log_error),
      "setting log error enable");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogVerbose(loptions, log_verbose),
      "setting log verbose level");
#endif  // TRITON_ENABLE_LOGGING

#ifdef TRITON_ENABLE_METRICS
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMetrics(loptions, allow_metrics_),
      "setting metrics enable");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetGpuMetrics(loptions, allow_gpu_metrics),
      "setting GPU metrics enable");
#endif  // TRITON_ENABLE_METRICS

  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetBackendDirectory(
          loptions, backend_dir.c_str()),
      "setting backend directory");
  for (const auto& bcs : backend_config_settings) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetBackendConfig(
            loptions, std::get<0>(bcs).c_str(), std::get<1>(bcs).c_str(),
            std::get<2>(bcs).c_str()),
        "setting backend configurtion");
  }

  return true;
}

}  // namespace

int
main(int argc, char** argv)
{
  // Parse command-line to create the options for the inference
  // server.
  TRITONSERVER_ServerOptions* server_options = nullptr;
  if (!Parse(&server_options, argc, argv)) {
    exit(1);
  }

  // Trace manager.
  nvidia::inferenceserver::TraceManager* trace_manager;

  // Manager for shared memory blocks.
  auto shm_manager =
      std::make_shared<nvidia::inferenceserver::SharedMemoryManager>();

  // Create the server...
  TRITONSERVER_Server* server_ptr = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ServerNew(&server_ptr, server_options), "creating server");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsDelete(server_options),
      "deleting server options");

  std::shared_ptr<TRITONSERVER_Server> server(
      server_ptr, TRITONSERVER_ServerDelete);

  // Configure and start tracing if specified on the command line.
  if (!StartTracing(&trace_manager)) {
    exit(1);
  }

  // Start the HTTP, GRPC, and metrics endpoints.
  if (!StartEndpoints(server, trace_manager, shm_manager)) {
    exit(1);
  }

  // Trap SIGINT and SIGTERM to allow server to exit gracefully
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  // Wait until a signal terminates the server...
  while (!exiting_) {
    // If enabled, poll the model repository to see if there have been
    // any changes.
    if (repository_poll_secs_ > 0) {
      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_ServerPollModelRepository(server_ptr),
          "failed to poll model repository");
    }

    // Wait for the polling interval (or a long time if polling is not
    // enabled). Will be woken if the server is exiting.
    std::unique_lock<std::mutex> lock(exit_mu_);
    std::chrono::seconds wait_timeout(
        (repository_poll_secs_ == 0) ? 3600 : repository_poll_secs_);
    exit_cv_.wait_for(lock, wait_timeout);
  }

  TRITONSERVER_Error* stop_err = TRITONSERVER_ServerStop(server_ptr);

  // If unable to gracefully stop the server then Triton threads and
  // state are potentially in an invalid state, so just exit
  // immediately.
  if (stop_err != nullptr) {
    LOG_TRITONSERVER_ERROR(stop_err, "failed to stop server");
    exit(1);
  }

  // Stop tracing and the HTTP, GRPC, and metrics endpoints.
  StopEndpoints();
  StopTracing(&trace_manager);

#ifdef TRITON_ENABLE_ASAN
  // Can invoke ASAN before exit though this is typically not very
  // useful since there are many objects that are not yet destructed.
  //  __lsan_do_leak_check();
#endif  // TRITON_ENABLE_ASAN

  return 0;
}
