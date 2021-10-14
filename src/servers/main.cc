// Copyright 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#ifndef _WIN32
#include <getopt.h>
#include <unistd.h>
#endif

#include <stdint.h>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <list>
#include <set>
#include <sstream>
#include "src/servers/signal.h"

#ifdef TRITON_ENABLE_ASAN
#include <sanitizer/lsan_interface.h>
#endif  // TRITON_ENABLE_ASAN

#include "src/core/logging.h"
#include "src/servers/common.h"
#include "src/servers/shared_memory_manager.h"
#include "src/servers/tracer.h"
#include "triton/core/tritonserver.h"

#if defined(TRITON_ENABLE_HTTP) || defined(TRITON_ENABLE_METRICS)
#include "src/servers/http_server.h"
#endif  // TRITON_ENABLE_HTTP|| TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_SAGEMAKER
#include "src/servers/sagemaker_server.h"
#endif  // TRITON_ENABLE_SAGEMAKER
#ifdef TRITON_ENABLE_VERTEX_AI
#include "src/servers/vertex_ai_server.h"
#endif  // TRITON_ENABLE_VERTEX_AI
#ifdef TRITON_ENABLE_GRPC
#include "src/servers/grpc_server.h"
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_GPU
static_assert(
    TRITON_MIN_COMPUTE_CAPABILITY >= 1.0,
    "Invalid TRITON_MIN_COMPUTE_CAPABILITY specified");
#endif  // TRITON_ENABLE_GPU

namespace {

// Interval, in seconds, when the model repository is polled for
// changes.
int32_t repository_poll_secs_ = 15;

// The HTTP, GRPC and metrics service/s and ports. Initialized to
// default values and modifyied based on command-line args.
#ifdef TRITON_ENABLE_HTTP
std::unique_ptr<nvidia::inferenceserver::HTTPServer> http_service_;
bool allow_http_ = true;
int32_t http_port_ = 8000;
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_SAGEMAKER
std::unique_ptr<nvidia::inferenceserver::HTTPServer> sagemaker_service_;
bool allow_sagemaker_ = false;
int32_t sagemaker_port_ = 8080;
bool sagemaker_safe_range_set_ = false;
std::pair<int32_t, int32_t> sagemaker_safe_range_ = {-1, -1};
// The number of threads to initialize for the SageMaker HTTP front-end.
int sagemaker_thread_cnt_ = 8;
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
std::unique_ptr<nvidia::inferenceserver::HTTPServer> vertex_ai_service_;
bool allow_vertex_ai_ = false;
int32_t vertex_ai_port_ = 8080;
// The number of threads to initialize for the Vertex AI HTTP front-end.
int vertex_ai_thread_cnt_ = 8;
std::string vertex_ai_default_model_;
#endif  // TRITON_ENABLE_VERTEX_AI

#ifdef TRITON_ENABLE_GRPC
std::unique_ptr<nvidia::inferenceserver::GRPCServer> grpc_service_;
bool allow_grpc_ = true;
int32_t grpc_port_ = 8001;
bool grpc_use_ssl_ = false;
nvidia::inferenceserver::SslOptions grpc_ssl_options_;
grpc_compression_level grpc_response_compression_level_ =
    GRPC_COMPRESS_LEVEL_NONE;
// KeepAlive defaults: https://grpc.github.io/grpc/cpp/md_doc_keepalive.html
nvidia::inferenceserver::KeepAliveOptions grpc_keepalive_options_;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
std::unique_ptr<nvidia::inferenceserver::HTTPServer> metrics_service_;
bool allow_metrics_ = true;
int32_t metrics_port_ = 8002;
float metrics_interval_ms_ = 2000;
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

#ifdef _WIN32
// Minimum implementation of <getopt.h> for Windows
#define required_argument 1
#define no_argument 2

int optind = 1;
const char* optarg = nullptr;

struct option {
  option(const char* name, int has_arg, int* flag, int val)
      : name_(name), has_arg_(has_arg), flag_(flag), val_(val)
  {
  }
  const char* name_;
  int has_arg_;
  int* flag_;
  int val_;
};

bool
end_of_long_opts(const struct option* longopts)
{
  return (
      (longopts->name_ == nullptr) && (longopts->has_arg_ == 0) &&
      (longopts->flag_ == nullptr) && (longopts->val_ == 0));
}

int
getopt_long(
    int argc, char* const argv[], const char* optstring,
    const struct option* longopts, int* longindex)
{
  if ((longindex != NULL) || (optind >= argc)) {
    return -1;
  }
  const struct option* curr_longopt = longopts;
  std::string argv_str = argv[optind];
  size_t found = argv_str.find_first_of("=");
  std::string key = argv_str.substr(
      2, (found == std::string::npos) ? std::string::npos : (found - 2));
  while (!end_of_long_opts(curr_longopt)) {
    if (key == curr_longopt->name_) {
      if (curr_longopt->has_arg_ == required_argument) {
        if (found == std::string::npos) {
          optind++;
          if (optind >= argc) {
            std::cerr << argv[0] << ": option '" << argv_str
                      << "' requires an argument" << std::endl;
            return '?';
          }
          optarg = argv[optind];
        } else {
          optarg = (argv[optind] + found + 1);
        }
      }
      optind++;
      return curr_longopt->val_;
    }
    curr_longopt++;
  }
  return -1;
}
#endif

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
  OPTION_GRPC_USE_SSL_MUTUAL,
  OPTION_GRPC_SERVER_CERT,
  OPTION_GRPC_SERVER_KEY,
  OPTION_GRPC_ROOT_CERT,
  OPTION_GRPC_RESPONSE_COMPRESSION_LEVEL,
  OPTION_GRPC_ARG_KEEPALIVE_TIME_MS,
  OPTION_GRPC_ARG_KEEPALIVE_TIMEOUT_MS,
  OPTION_GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS,
  OPTION_GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA,
  OPTION_GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS,
  OPTION_GRPC_ARG_HTTP2_MAX_PING_STRIKES,
#endif  // TRITON_ENABLE_GRPC
#if defined(TRITON_ENABLE_SAGEMAKER)
  OPTION_ALLOW_SAGEMAKER,
  OPTION_SAGEMAKER_PORT,
  OPTION_SAGEMAKER_SAFE_PORT_RANGE,
  OPTION_SAGEMAKER_THREAD_COUNT,
#endif  // TRITON_ENABLE_SAGEMAKER
#if defined(TRITON_ENABLE_VERTEX_AI)
  OPTION_ALLOW_VERTEX_AI,
  OPTION_VERTEX_AI_PORT,
  OPTION_VERTEX_AI_THREAD_COUNT,
  OPTION_VERTEX_AI_DEFAULT_MODEL,
#endif  // TRITON_ENABLE_VERTEX_AI
#ifdef TRITON_ENABLE_METRICS
  OPTION_ALLOW_METRICS,
  OPTION_ALLOW_GPU_METRICS,
  OPTION_METRICS_PORT,
  OPTION_METRICS_INTERVAL_MS,
#endif  // TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_TRACING
  OPTION_TRACE_FILEPATH,
  OPTION_TRACE_LEVEL,
  OPTION_TRACE_RATE,
#endif  // TRITON_ENABLE_TRACING
  OPTION_MODEL_CONTROL_MODE,
  OPTION_POLL_REPO_SECS,
  OPTION_STARTUP_MODEL,
  OPTION_RATE_LIMIT,
  OPTION_RATE_LIMIT_RESOURCE,
  OPTION_PINNED_MEMORY_POOL_BYTE_SIZE,
  OPTION_CUDA_MEMORY_POOL_BYTE_SIZE,
  OPTION_RESPONSE_CACHE_BYTE_SIZE,
  OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY,
  OPTION_EXIT_TIMEOUT_SECS,
  OPTION_BACKEND_DIR,
  OPTION_REPOAGENT_DIR,
  OPTION_BUFFER_MANAGER_THREAD_COUNT,
  OPTION_BACKEND_CONFIG,
  OPTION_HOST_POLICY
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
      {OPTION_GRPC_USE_SSL_MUTUAL, "grpc-use-ssl-mutual", Option::ArgBool,
       "Use mututal SSL authentication for GRPC requests. Default is false."},
      {OPTION_GRPC_SERVER_CERT, "grpc-server-cert", Option::ArgStr,
       "File holding PEM-encoded server certificate. Ignored unless "
       "--grpc-use-ssl is true."},
      {OPTION_GRPC_SERVER_KEY, "grpc-server-key", Option::ArgStr,
       "File holding PEM-encoded server key. Ignored unless "
       "--grpc-use-ssl is true."},
      {OPTION_GRPC_ROOT_CERT, "grpc-root-cert", Option::ArgStr,
       "File holding PEM-encoded root certificate. Ignore unless "
       "--grpc-use-ssl is false."},
      {OPTION_GRPC_RESPONSE_COMPRESSION_LEVEL,
       "grpc-infer-response-compression-level", Option::ArgStr,
       "The compression level to be used while returning the infer response to "
       "the peer. Allowed values are none, low, medium and high. By default, "
       "compression level is selected as none."},
      {OPTION_GRPC_ARG_KEEPALIVE_TIME_MS, "grpc-keepalive-time", Option::ArgInt,
       "The period (in milliseconds) after which a keepalive ping is sent on "
       "the transport. Default is 7200000 (2 hours)."},
      {OPTION_GRPC_ARG_KEEPALIVE_TIMEOUT_MS, "grpc-keepalive-timeout",
       Option::ArgInt,
       "The period (in milliseconds) the sender of the keepalive ping waits "
       "for an acknowledgement. If it does not receive an acknowledgment "
       "within this time, it will close the connection. "
       "Default is 20000 (20 seconds)."},
      {OPTION_GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS,
       "grpc-keepalive-permit-without-calls", Option::ArgBool,
       "Allows keepalive pings to be sent even if there are no calls in flight "
       "(0 : false; 1 : true). Default is 0 (false)."},
      {OPTION_GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA,
       "grpc-http2-max-pings-without-data", Option::ArgInt,
       "The maximum number of pings that can be sent when there is no "
       "data/header frame to be sent. gRPC Core will not continue sending "
       "pings if we run over the limit. Setting it to 0 allows sending pings "
       "without such a restriction. Default is 2."},
      {OPTION_GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS,
       "grpc-http2-min-recv-ping-interval-without-data", Option::ArgInt,
       "If there are no data/header frames being sent on the transport, this "
       "channel argument on the server side controls the minimum time "
       "(in milliseconds) that gRPC Core would expect between receiving "
       "successive pings. If the time between successive pings is less than "
       "this time, then the ping will be considered a bad ping from the peer. "
       "Such a ping counts as a ‘ping strike’. Default is 300000 (5 minutes)."},
      {OPTION_GRPC_ARG_HTTP2_MAX_PING_STRIKES, "grpc-http2-max-ping-strikes",
       Option::ArgInt,
       "Maximum number of bad pings that the server will tolerate before "
       "sending an HTTP2 GOAWAY frame and closing the transport. Setting it to "
       "0 allows the server to accept any number of bad pings. Default is 2."},
#endif  // TRITON_ENABLE_GRPC
#if defined(TRITON_ENABLE_SAGEMAKER)
      {OPTION_ALLOW_SAGEMAKER, "allow-sagemaker", Option::ArgBool,
       "Allow the server to listen for Sagemaker requests. Default is false."},
      {OPTION_SAGEMAKER_PORT, "sagemaker-port", Option::ArgInt,
       "The port for the server to listen on for Sagemaker requests. Default "
       "is 8080."},
      {OPTION_SAGEMAKER_SAFE_PORT_RANGE, "sagemaker-safe-port-range",
       "<integer>-<integer>",
       "Set the allowed port range for endpoints other than the SageMaker "
       "endpoints."},
      {OPTION_SAGEMAKER_THREAD_COUNT, "sagemaker-thread-count", Option::ArgInt,
       "Number of threads handling Sagemaker requests. Default is 8."},
#endif  // TRITON_ENABLE_SAGEMAKER
#if defined(TRITON_ENABLE_VERTEX_AI)
      {OPTION_ALLOW_VERTEX_AI, "allow-vertex-ai", Option::ArgBool,
       "Allow the server to listen for Vertex AI requests. Default is true if "
       "AIP_MODE=PREDICTION, false otherwise."},
      {OPTION_VERTEX_AI_PORT, "vertex-ai-port", Option::ArgInt,
       "The port for the server to listen on for Vertex AI requests. Default "
       "is AIP_HTTP_PORT if set, 8080 otherwise."},
      {OPTION_VERTEX_AI_THREAD_COUNT, "vertex-ai-thread-count", Option::ArgInt,
       "Number of threads handling Vertex AI requests. Default is 8."},
      {OPTION_VERTEX_AI_DEFAULT_MODEL, "vertex-ai-default-model",
       Option::ArgStr,
       "The name of the model to use for single-model inference requests."},
#endif  // TRITON_ENABLE_VERTEX_AI
#ifdef TRITON_ENABLE_METRICS
      {OPTION_ALLOW_METRICS, "allow-metrics", Option::ArgBool,
       "Allow the server to provide prometheus metrics."},
      {OPTION_ALLOW_GPU_METRICS, "allow-gpu-metrics", Option::ArgBool,
       "Allow the server to provide GPU metrics. Ignored unless "
       "--allow-metrics is true."},
      {OPTION_METRICS_PORT, "metrics-port", Option::ArgInt,
       "The port reporting prometheus metrics."},
      {OPTION_METRICS_INTERVAL_MS, "metrics-interval-ms", Option::ArgFloat,
       "Metrics will be collected once every <metrics-interval-ms> "
       "milliseconds. Default is 2000 milliseconds."},
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
      // FIXME:  fix the default to execution_count once RL logic is complete.
      {OPTION_RATE_LIMIT, "rate-limit", Option::ArgStr,
       "Specify the mode for rate limiting. Options are \"execution_count\" "
       "and \"off\". The default is \"off\". For "
       "\"execution_count\", the server will determine the instance using "
       "configured priority and the number of time the instance has been "
       "used to run inference. The inference will finally be executed once "
       "the required resources are available. For \"off\", the server will "
       "ignore any rate limiter config and run inference as soon as an "
       "instance is ready."},
      {OPTION_RATE_LIMIT_RESOURCE, "rate-limit-resource",
       "<string>:<integer>:<integer>",
       "The number of resources available to the server. The format of this "
       "flag is --rate-limit-resource=<resource_name>:<count>:<device>. The "
       "<device> is optional and if not listed will be applied to every "
       "device. If the resource is specified as \"GLOBAL\" in the model "
       "configuration the resource is considered shared among all the devices "
       "in the system. The <device> property is ignored for such resources. "
       "This flag can be specified multiple times to specify each resources "
       "and their availability. By default, the max across all instances that "
       "list the resource is selected as its availability. The values for this "
       "flag is case-insensitive."},
      {OPTION_PINNED_MEMORY_POOL_BYTE_SIZE, "pinned-memory-pool-byte-size",
       Option::ArgInt,
       "The total byte size that can be allocated as pinned system memory. "
       "If GPU support is enabled, the server will allocate pinned system "
       "memory to accelerate data transfer between host and devices until it "
       "exceeds the specified byte size. If 'numa-node' is configured via "
       "--host-policy, the pinned system memory of the pool size will be "
       "allocated on each numa node. This option will not affect the "
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
      {OPTION_RESPONSE_CACHE_BYTE_SIZE, "response-cache-byte-size",
       Option::ArgInt,
       "The size in bytes to allocate for a request/response cache. When "
       "non-zero, Triton allocates the requested size in CPU memory and "
       "shares the cache across all inference requests and across all models. "
       "For a given model to use request caching, the model must enable "
       "request caching in the model configuration. By default, no model uses "
       "request caching even if the request cache is enabled with the "
       "--response-cache-byte-size flag. Default is 0."},
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
      {OPTION_REPOAGENT_DIR, "repoagent-directory", Option::ArgStr,
       "The global directory searched for repository agent shared libraries. "
       "Default is '/opt/tritonserver/repoagents'."},
      {OPTION_BUFFER_MANAGER_THREAD_COUNT, "buffer-manager-thread-count",
       Option::ArgInt,
       "The number of threads used to accelerate copies and other operations "
       "required to manage input and output tensor contents. Default is 0."},
      {OPTION_BACKEND_CONFIG, "backend-config", "<string>,<string>=<string>",
       "Specify a backend-specific configuration setting. The format of this "
       "flag is --backend-config=<backend_name>,<setting>=<value>. Where "
       "<backend_name> is the name of the backend, such as 'tensorrt'."},
  {
    OPTION_HOST_POLICY, "host-policy", "<string>,<string>=<string>",
        "Specify a host policy setting associated with a policy name. The "
        "format of this flag is --host-policy=<policy_name>,<setting>=<value>."
        "Currently supported settings are 'numa-node', 'cpu-cores'. Note that "
        "'numa-node' setting will affect pinned memory pool behavior, see "
        "--pinned-memory-pool for more detail."
  }
};

bool
CheckPortCollision()
{
  // List of enabled services and their constraints
  std::vector<std::tuple<std::string, int32_t, bool, int32_t, int32_t>> ports;
#ifdef TRITON_ENABLE_HTTP
  if (allow_http_) {
    ports.emplace_back("HTTP", http_port_, false, -1, -1);
  }
#endif  // TRITON_ENABLE_HTTP
#ifdef TRITON_ENABLE_GRPC
  if (allow_grpc_) {
    ports.emplace_back("GRPC", grpc_port_, false, -1, -1);
  }
#endif  // TRITON_ENABLE_GRPC
#ifdef TRITON_ENABLE_METRICS
  if (allow_metrics_) {
    ports.emplace_back("metrics", metrics_port_, false, -1, -1);
  }
#endif  // TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_SAGEMAKER
  if (allow_sagemaker_) {
    ports.emplace_back(
        "SageMaker", sagemaker_port_, sagemaker_safe_range_set_,
        sagemaker_safe_range_.first, sagemaker_safe_range_.second);
  }
#endif  // TRITON_ENABLE_SAGEMAKER
#ifdef TRITON_ENABLE_VERTEX_AI
  if (allow_vertex_ai_) {
    ports.emplace_back("Vertex AI", vertex_ai_port_, false, -1, -1);
  }
#endif  // TRITON_ENABLE_VERTEX_AI

  for (auto curr_it = ports.begin(); curr_it != ports.end(); ++curr_it) {
    // If the current service doesn't specify the allow port range for other
    // services, then we don't need to revisit the checked services
    auto comparing_it = (std::get<2>(*curr_it)) ? ports.begin() : (curr_it + 1);
    for (; comparing_it != ports.end(); ++comparing_it) {
      if (comparing_it == curr_it) {
        continue;
      }
      // Set range and comparing service port is out of range
      if (std::get<2>(*curr_it) &&
          ((std::get<1>(*comparing_it) < std::get<3>(*curr_it)) ||
           (std::get<1>(*comparing_it) > std::get<4>(*curr_it)))) {
        std::cerr << "The server cannot listen to "
                  << std::get<0>(*comparing_it) << " requests at port "
                  << std::get<1>(*comparing_it) << ", allowed port range is ["
                  << std::get<3>(*curr_it) << ", " << std::get<4>(*curr_it)
                  << "]" << std::endl;
        return true;
      }
      if (std::get<1>(*curr_it) == std::get<1>(*comparing_it)) {
        std::cerr << "The server cannot listen to " << std::get<0>(*curr_it)
                  << " requests "
                  << "and " << std::get<0>(*comparing_it)
                  << " requests at the same port " << std::get<1>(*curr_it)
                  << std::endl;
        return true;
      }
    }
  }

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
      grpc_ssl_options_, grpc_infer_allocation_pool_size_,
      grpc_response_compression_level_, grpc_keepalive_options_, service);
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
  TRITONSERVER_Error* err = nvidia::inferenceserver::HTTPAPIServer::Create(
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
  TRITONSERVER_Error* err = nvidia::inferenceserver::HTTPMetricsServer::Create(
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

#ifdef TRITON_ENABLE_SAGEMAKER
TRITONSERVER_Error*
StartSagemakerService(
    std::unique_ptr<nvidia::inferenceserver::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    nvidia::inferenceserver::TraceManager* trace_manager,
    const std::shared_ptr<nvidia::inferenceserver::SharedMemoryManager>&
        shm_manager)
{
  TRITONSERVER_Error* err = nvidia::inferenceserver::SagemakerAPIServer::Create(
      server, trace_manager, shm_manager, sagemaker_port_,
      sagemaker_thread_cnt_, service);
  if (err == nullptr) {
    err = (*service)->Start();
  }

  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
TRITONSERVER_Error*
StartVertexAiService(
    std::unique_ptr<nvidia::inferenceserver::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    nvidia::inferenceserver::TraceManager* trace_manager,
    const std::shared_ptr<nvidia::inferenceserver::SharedMemoryManager>&
        shm_manager)
{
  TRITONSERVER_Error* err = nvidia::inferenceserver::VertexAiAPIServer::Create(
      server, trace_manager, shm_manager, vertex_ai_port_,
      vertex_ai_thread_cnt_, vertex_ai_default_model_, service);
  if (err == nullptr) {
    err = (*service)->Start();
  }

  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRITON_ENABLE_VERTEX_AI

bool
StartEndpoints(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    nvidia::inferenceserver::TraceManager* trace_manager,
    const std::shared_ptr<nvidia::inferenceserver::SharedMemoryManager>&
        shm_manager)
{
#ifdef _WIN32
  WSADATA wsaData;
  int wsa_ret = WSAStartup(MAKEWORD(2, 2), &wsaData);

  if (wsa_ret != 0) {
    LOG_ERROR << "Error in WSAStartup " << wsa_ret;
    return false;
  }
#endif

#ifdef TRITON_ENABLE_GRPC
  // Enable GRPC endpoints if requested...
  if (allow_grpc_) {
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
  if (allow_http_) {
    TRITONSERVER_Error* err =
        StartHttpService(&http_service_, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start HTTP service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_HTTP


#ifdef TRITON_ENABLE_SAGEMAKER
  // Enable Sagemaker endpoints if requested...
  if (allow_sagemaker_) {
    TRITONSERVER_Error* err = StartSagemakerService(
        &sagemaker_service_, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start Sagemaker service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
  // Enable Vertex AI endpoints if requested...
  if (allow_vertex_ai_) {
    TRITONSERVER_Error* err = StartVertexAiService(
        &vertex_ai_service_, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start Vertex AI service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_VERTEX_AI

#ifdef TRITON_ENABLE_METRICS
  // Enable metrics endpoint if requested...
  if (allow_metrics_) {
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

#ifdef TRITON_ENABLE_SAGEMAKER
  if (sagemaker_service_) {
    TRITONSERVER_Error* err = sagemaker_service_->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop Sagemaker service");
      ret = false;
    }

    sagemaker_service_.reset();
  }
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
  if (vertex_ai_service_) {
    TRITONSERVER_Error* err = vertex_ai_service_->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop Vertex AI service");
      ret = false;
    }

    vertex_ai_service_.reset();
  }
#endif  // TRITON_ENABLE_VERTEX_AI

#ifdef _WIN32
  int wsa_ret = WSACleanup();

  if (wsa_ret != 0) {
    LOG_ERROR << "Error in WSACleanup " << wsa_ret;
    ret = false;
  }
#endif

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

// Template specialization for ParsePairOption
// [FIXME] replace ParseXXXOPtion with these
template <typename T>
T ParseOption(const std::string& arg);

template <>
int
ParseOption(const std::string& arg)
{
  return std::stoi(arg);
}

template <>
uint64_t
ParseOption(const std::string& arg)
{
  return std::stoll(arg);
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

std::tuple<std::string, int, int>
ParseRateLimiterResourceOption(const std::string arg)
{
  std::string error_string(
      "--rate-limit-resource option format is "
      "'<resource_name>:<count>:<device>' or '<resource_name>:<count>'. Got " +
      arg);

  std::string name_string("");
  int count = -1;
  int device_id = -1;

  size_t delim_first = arg.find(":");
  size_t delim_second = arg.find(":", delim_first + 1);

  if (delim_second != std::string::npos) {
    // Handle format `<resource_name>:<count>:<device>'
    size_t delim_third = arg.find(":", delim_second + 1);
    if (delim_third != std::string::npos) {
      std::cerr << error_string << std::endl;
      exit(1);
    }
    name_string = arg.substr(0, delim_first);
    count = ParseIntOption(
        arg.substr(delim_first + 1, delim_second - delim_first - 1));
    device_id = ParseIntOption(arg.substr(delim_second + 1));
  } else if (delim_first != std::string::npos) {
    // Handle format `<resource_name>:<count>'
    name_string = arg.substr(0, delim_first);
    count = ParseIntOption(arg.substr(delim_first + 1));
  } else {
    // If no colons found
    std::cerr << error_string << std::endl;
    exit(1);
  }

  return {name_string, count, device_id};
}

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

std::tuple<std::string, std::string, std::string>
ParseHostPolicyOption(const std::string arg)
{
  // Format is "<backend_name>,<setting>=<value>"
  int delim_name = arg.find(",");
  int delim_setting = arg.find("=", delim_name + 1);

  // Check for 2 semicolons
  if ((delim_name < 0) || (delim_setting < 0)) {
    std::cerr << "--host-policy option format is '<policy "
                 "name>,<setting>=<value>'. Got "
              << arg << std::endl;
    exit(1);
  }

  std::string name_string = arg.substr(0, delim_name);
  std::string setting_string =
      arg.substr(delim_name + 1, delim_setting - delim_name - 1);
  std::string value_string = arg.substr(delim_setting + 1);

  if (name_string.empty() || setting_string.empty() || value_string.empty()) {
    std::cerr << "--host-policy option format is '<policy "
                 "name>,<setting>=<value>'. Got "
              << arg << std::endl;
    exit(1);
  }

  return {name_string, setting_string, value_string};
}

template <typename T1, typename T2>
std::pair<T1, T2>
ParsePairOption(const std::string& arg, const std::string& delim_str)
{
  int delim = arg.find(delim_str);

  if ((delim < 0)) {
    std::cerr << "Cannot parse pair option due to incorrect number of inputs."
                 "--<pair option> argument requires format <first>"
              << delim_str << "<second>. "
              << "Found: " << arg << std::endl;
    std::cerr << Usage() << std::endl;
    exit(1);
  }

  std::string first_string = arg.substr(0, delim);
  std::string second_string = arg.substr(delim + delim_str.length());

  // Specific conversion from key-value string to actual key-value type,
  // should be extracted out of this function if we need to parse
  // more pair option of different types.
  return {ParseOption<T1>(first_string), ParseOption<T2>(second_string)};
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
  uint64_t response_cache_byte_size = 0;

  std::string backend_dir = "/opt/tritonserver/backends";
  std::string repoagent_dir = "/opt/tritonserver/repoagents";
  std::vector<std::tuple<std::string, std::string, std::string>>
      backend_config_settings;
  std::vector<std::tuple<std::string, std::string, std::string>> host_policies;

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
  grpc_compression_level grpc_response_compression_level =
      grpc_response_compression_level_;
#endif  // TRITON_ENABLE_GRPC

#if defined(TRITON_ENABLE_SAGEMAKER)
  int32_t sagemaker_port = sagemaker_port_;
  int32_t sagemaker_thread_cnt = sagemaker_thread_cnt_;
  bool sagemaker_safe_range_set = sagemaker_safe_range_set_;
  std::pair<int32_t, int32_t> sagemaker_safe_range = sagemaker_safe_range_;
#endif  // TRITON_ENABLE_SAGEMAKER

#if defined(TRITON_ENABLE_VERTEX_AI)
  // Set different default value if specific flag is set
  {
    auto aip_mode = nvidia::inferenceserver::GetEnvironmentVariableOrDefault(
        "AIP_MODE", "");
    allow_vertex_ai_ = (aip_mode == "PREDICTION");
    auto port = nvidia::inferenceserver::GetEnvironmentVariableOrDefault(
        "AIP_HTTP_PORT", "8080");
    vertex_ai_port_ = ParseIntOption(port);
  }
  int32_t vertex_ai_port = vertex_ai_port_;
  int32_t vertex_ai_thread_cnt = vertex_ai_thread_cnt_;
  std::string vertex_ai_default_model = vertex_ai_default_model_;
#endif  // TRITON_ENABLE_VERTEX_AI

#ifdef TRITON_ENABLE_METRICS
  int32_t metrics_port = metrics_port_;
  bool allow_gpu_metrics = true;
  float metrics_interval_ms = metrics_interval_ms_;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_TRACING
  std::string trace_filepath = trace_filepath_;
  TRITONSERVER_InferenceTraceLevel trace_level = trace_level_;
  int32_t trace_rate = trace_rate_;
#endif  // TRITON_ENABLE_TRACING

  TRITONSERVER_ModelControlMode control_mode = TRITONSERVER_MODEL_CONTROL_NONE;
  std::set<std::string> startup_models_;

  // FIXME: Once the rate limiter implementation is complete make
  // EXEC_COUNT the default.
  // TRITONSERVER_RateLimitMode rate_limit_mode =
  //    TRITONSERVER_RATE_LIMIT_EXEC_COUNT;
  TRITONSERVER_RateLimitMode rate_limit_mode = TRITONSERVER_RATE_LIMIT_OFF;
  std::vector<std::tuple<std::string, int, int>> rate_limit_resources;

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

#if defined(TRITON_ENABLE_SAGEMAKER)
      case OPTION_ALLOW_SAGEMAKER:
        allow_sagemaker_ = ParseBoolOption(optarg);
        break;
      case OPTION_SAGEMAKER_PORT:
        sagemaker_port = ParseIntOption(optarg);
        break;
      case OPTION_SAGEMAKER_SAFE_PORT_RANGE:
        sagemaker_safe_range_set = true;
        sagemaker_safe_range = ParsePairOption<int, int>(optarg, "-");
        break;
      case OPTION_SAGEMAKER_THREAD_COUNT:
        sagemaker_thread_cnt = ParseIntOption(optarg);
        break;
#endif  // TRITON_ENABLE_SAGEMAKER

#if defined(TRITON_ENABLE_VERTEX_AI)
      case OPTION_ALLOW_VERTEX_AI:
        allow_vertex_ai_ = ParseBoolOption(optarg);
        break;
      case OPTION_VERTEX_AI_PORT:
        vertex_ai_port = ParseIntOption(optarg);
        break;
      case OPTION_VERTEX_AI_THREAD_COUNT:
        vertex_ai_thread_cnt = ParseIntOption(optarg);
        break;
      case OPTION_VERTEX_AI_DEFAULT_MODEL:
        vertex_ai_default_model = optarg;
        break;
#endif  // TRITON_ENABLE_VERTEX_AI

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
      case OPTION_GRPC_USE_SSL_MUTUAL:
        grpc_ssl_options_.use_mutual_auth = ParseBoolOption(optarg);
        grpc_use_ssl = true;
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
      case OPTION_GRPC_RESPONSE_COMPRESSION_LEVEL: {
        std::string mode_str(optarg);
        std::transform(
            mode_str.begin(), mode_str.end(), mode_str.begin(), ::tolower);
        if (mode_str == "none") {
          grpc_response_compression_level = GRPC_COMPRESS_LEVEL_NONE;
        } else if (mode_str == "low") {
          grpc_response_compression_level = GRPC_COMPRESS_LEVEL_LOW;
        } else if (mode_str == "medium") {
          grpc_response_compression_level = GRPC_COMPRESS_LEVEL_MED;
        } else if (mode_str == "high") {
          grpc_response_compression_level = GRPC_COMPRESS_LEVEL_HIGH;
        } else {
          std::cerr
              << "invalid argument for --grpc_infer_response_compression_level"
              << std::endl;
          std::cerr << Usage() << std::endl;
          return false;
        }
        break;
      }
      case OPTION_GRPC_ARG_KEEPALIVE_TIME_MS:
        grpc_keepalive_options_.keepalive_time_ms = ParseIntOption(optarg);
        break;
      case OPTION_GRPC_ARG_KEEPALIVE_TIMEOUT_MS:
        grpc_keepalive_options_.keepalive_timeout_ms = ParseIntOption(optarg);
        break;
      case OPTION_GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS:
        grpc_keepalive_options_.keepalive_permit_without_calls =
            ParseBoolOption(optarg);
        break;
      case OPTION_GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA:
        grpc_keepalive_options_.http2_max_pings_without_data =
            ParseIntOption(optarg);
        break;
      case OPTION_GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS:
        grpc_keepalive_options_.http2_min_recv_ping_interval_without_data_ms =
            ParseIntOption(optarg);
        break;
      case OPTION_GRPC_ARG_HTTP2_MAX_PING_STRIKES:
        grpc_keepalive_options_.http2_max_ping_strikes = ParseIntOption(optarg);
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
      case OPTION_METRICS_INTERVAL_MS:
        metrics_interval_ms = ParseIntOption(optarg);
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
      case OPTION_RATE_LIMIT: {
        std::string rate_limit_str(optarg);
        std::transform(
            rate_limit_str.begin(), rate_limit_str.end(),
            rate_limit_str.begin(), ::tolower);
        if (rate_limit_str == "execution_count") {
          rate_limit_mode = TRITONSERVER_RATE_LIMIT_EXEC_COUNT;
        } else if (rate_limit_str == "off") {
          rate_limit_mode = TRITONSERVER_RATE_LIMIT_OFF;
        } else {
          std::cerr << "invalid argument for --rate-limit" << std::endl;
          std::cerr << Usage() << std::endl;
          return false;
        }
        break;
      }
      case OPTION_RATE_LIMIT_RESOURCE: {
        std::string rate_limit_resource_str(optarg);
        std::transform(
            rate_limit_resource_str.begin(), rate_limit_resource_str.end(),
            rate_limit_resource_str.begin(), ::tolower);
        try {
          rate_limit_resources.push_back(
              ParseRateLimiterResourceOption(optarg));
        }
        catch (const std::invalid_argument& ia) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("failed to parse '") + optarg +
               "' as <str>:<int>:<int>")
                  .c_str());
        }
        break;
      }
      case OPTION_PINNED_MEMORY_POOL_BYTE_SIZE:
        pinned_memory_pool_byte_size = ParseLongLongOption(optarg);
        break;
      case OPTION_CUDA_MEMORY_POOL_BYTE_SIZE:
        cuda_pools.push_back(ParsePairOption<int, uint64_t>(optarg, ":"));
        break;
      case OPTION_RESPONSE_CACHE_BYTE_SIZE:
        response_cache_byte_size = (uint64_t)ParseLongLongOption(optarg);
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
      case OPTION_REPOAGENT_DIR:
        repoagent_dir = optarg;
        break;
      case OPTION_BUFFER_MANAGER_THREAD_COUNT:
        buffer_manager_thread_count = ParseIntOption(optarg);
        break;
      case OPTION_BACKEND_CONFIG:
        backend_config_settings.push_back(ParseBackendConfigOption(optarg));
        break;
      case OPTION_HOST_POLICY:
        host_policies.push_back(ParseHostPolicyOption(optarg));
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

#if defined(TRITON_ENABLE_SAGEMAKER)
  sagemaker_port_ = sagemaker_port;
  sagemaker_thread_cnt_ = sagemaker_thread_cnt;
  sagemaker_safe_range_set_ = sagemaker_safe_range_set;
  sagemaker_safe_range_ = sagemaker_safe_range;
#endif  // TRITON_ENABLE_SAGEMAKER

#if defined(TRITON_ENABLE_VERTEX_AI)
  // Set default model repository if specific flag is set, postpone the
  // check to after parsing so we only monitor the default repository if
  // Vertex service is allowed
  {
    auto aip_storage_uri =
        nvidia::inferenceserver::GetEnvironmentVariableOrDefault(
            "AIP_STORAGE_URI", "");
    if (!aip_storage_uri.empty() && model_repository_paths.empty()) {
      model_repository_paths.insert(aip_storage_uri);
    }
  }
  vertex_ai_port_ = vertex_ai_port;
  vertex_ai_thread_cnt_ = vertex_ai_thread_cnt;
  vertex_ai_default_model_ = vertex_ai_default_model;
#endif  // TRITON_ENABLE_VERTEX_AI


#if defined(TRITON_ENABLE_GRPC)
  grpc_port_ = grpc_port;
  grpc_infer_allocation_pool_size_ = grpc_infer_allocation_pool_size;
  grpc_use_ssl_ = grpc_use_ssl;
  grpc_response_compression_level_ = grpc_response_compression_level;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
  metrics_port_ = metrics_port;
  allow_gpu_metrics = allow_metrics_ ? allow_gpu_metrics : false;
  metrics_interval_ms_ = metrics_interval_ms;
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
      TRITONSERVER_ServerOptionsSetRateLimiterMode(loptions, rate_limit_mode),
      "setting rate limiter configuration");
  for (const auto& resource : rate_limit_resources) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsAddRateLimiterResource(
            loptions, std::get<0>(resource).c_str(), std::get<1>(resource),
            std::get<2>(resource)),
        "setting rate limiter resource");
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
      TRITONSERVER_ServerOptionsSetResponseCacheByteSize(
          loptions, response_cache_byte_size),
      "setting total response cache byte size");
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
      "setting buffer manager thread count");

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
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMetricsInterval(
          loptions, metrics_interval_ms_),
      "setting metrics interval");
#endif  // TRITON_ENABLE_METRICS

  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetBackendDirectory(
          loptions, backend_dir.c_str()),
      "setting backend directory");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
          loptions, repoagent_dir.c_str()),
      "setting repository agent directory");
  for (const auto& bcs : backend_config_settings) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetBackendConfig(
            loptions, std::get<0>(bcs).c_str(), std::get<1>(bcs).c_str(),
            std::get<2>(bcs).c_str()),
        "setting backend configurtion");
  }
  for (const auto& hp : host_policies) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetHostPolicy(
            loptions, std::get<0>(hp).c_str(), std::get<1>(hp).c_str(),
            std::get<2>(hp).c_str()),
        "setting host policy");
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

  // Trap SIGINT and SIGTERM to allow server to exit gracefully
  TRITONSERVER_Error* signal_err =
      nvidia::inferenceserver::RegisterSignalHandler();
  if (signal_err != nullptr) {
    LOG_TRITONSERVER_ERROR(signal_err, "failed to register signal handler");
    exit(1);
  }

  // Start the HTTP, GRPC, and metrics endpoints.
  if (!StartEndpoints(server, trace_manager, shm_manager)) {
    exit(1);
  }

  // Wait until a signal terminates the server...
  while (!nvidia::inferenceserver::signal_exiting_) {
    // If enabled, poll the model repository to see if there have been
    // any changes.
    if (repository_poll_secs_ > 0) {
      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_ServerPollModelRepository(server_ptr),
          "failed to poll model repository");
    }

    // Wait for the polling interval (or a long time if polling is not
    // enabled). Will be woken if the server is exiting.
    std::unique_lock<std::mutex> lock(nvidia::inferenceserver::signal_exit_mu_);
    std::chrono::seconds wait_timeout(
        (repository_poll_secs_ == 0) ? 3600 : repository_poll_secs_);
    nvidia::inferenceserver::signal_exit_cv_.wait_for(lock, wait_timeout);
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
