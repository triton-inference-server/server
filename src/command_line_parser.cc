// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//

#include "command_line_parser.h"

#include <iomanip>
#include <iostream>
#include <string>

namespace triton { namespace server {

enum TritonOptionId {
  OPTION_HELP = 1000,
#ifdef TRITON_ENABLE_LOGGING
  OPTION_LOG_VERBOSE,
  OPTION_LOG_INFO,
  OPTION_LOG_WARNING,
  OPTION_LOG_ERROR,
  OPTION_LOG_FORMAT,
  OPTION_LOG_FILE,
#endif  // TRITON_ENABLE_LOGGING
  OPTION_ID,
  OPTION_MODEL_REPOSITORY,
  OPTION_EXIT_ON_ERROR,
  OPTION_DISABLE_AUTO_COMPLETE_CONFIG,
  OPTION_STRICT_MODEL_CONFIG,
  OPTION_STRICT_READINESS,
#if defined(TRITON_ENABLE_HTTP)
  OPTION_ALLOW_HTTP,
  OPTION_HTTP_PORT,
  OPTION_REUSE_HTTP_PORT,
  OPTION_HTTP_ADDRESS,
  OPTION_HTTP_THREAD_COUNT,
#endif  // TRITON_ENABLE_HTTP
#if defined(TRITON_ENABLE_GRPC)
  OPTION_ALLOW_GRPC,
  OPTION_GRPC_PORT,
  OPTION_REUSE_GRPC_PORT,
  OPTION_GRPC_ADDRESS,
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
  OPTION_ALLOW_CPU_METRICS,
  OPTION_METRICS_PORT,
  OPTION_METRICS_INTERVAL_MS,
#endif  // TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_TRACING
  OPTION_TRACE_FILEPATH,
  OPTION_TRACE_LEVEL,
  OPTION_TRACE_RATE,
  OPTION_TRACE_COUNT,
  OPTION_TRACE_LOG_FREQUENCY,
#endif  // TRITON_ENABLE_TRACING
  OPTION_MODEL_CONTROL_MODE,
  OPTION_POLL_REPO_SECS,
  OPTION_STARTUP_MODEL,
  OPTION_RATE_LIMIT,
  OPTION_RATE_LIMIT_RESOURCE,
  OPTION_PINNED_MEMORY_POOL_BYTE_SIZE,
  OPTION_CUDA_MEMORY_POOL_BYTE_SIZE,
  OPTION_RESPONSE_CACHE_BYTE_SIZE,
  OPTION_CACHE_CONFIG,
  OPTION_CACHE_DIR,
  OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY,
  OPTION_EXIT_TIMEOUT_SECS,
  OPTION_BACKEND_DIR,
  OPTION_REPOAGENT_DIR,
  OPTION_BUFFER_MANAGER_THREAD_COUNT,
  OPTION_MODEL_LOAD_THREAD_COUNT,
  OPTION_BACKEND_CONFIG,
  OPTION_HOST_POLICY,
  OPTION_MODEL_LOAD_GPU_LIMIT,
  OPTION_MODEL_NAMESPACING
};

std::vector<Option> FallThroughParser::recognized_options_
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
      {OPTION_LOG_FORMAT, "log-format", Option::ArgStr,
       "Set the logging format. Options are \"default\" and \"ISO8601\". "
       "The default is \"default\". For \"default\", the log severity (L) and "
       "timestamp will be logged as \"LMMDD hh:mm:ss.ssssss\". "
       "For \"ISO8601\", the log format will be \"YYYY-MM-DDThh:mm:ssZ L\"."},
      {OPTION_LOG_FILE, "log-file", Option::ArgStr,
       "Set the name of the log output file. If specified, log outputs will be "
       "saved to this file. If not specified, log outputs will stream to the "
       "console."},
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
      {OPTION_DISABLE_AUTO_COMPLETE_CONFIG, "disable-auto-complete-config",
       Option::ArgNone,
       "If set, disables the triton and backends from auto completing model "
       "configuration files. Model configuration files must be provided and "
       "all required "
       "configuration settings must be specified."},
      {OPTION_STRICT_MODEL_CONFIG, "strict-model-config", Option::ArgBool,
       "DEPRECATED: If true model configuration files must be provided and all "
       "required "
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
      {OPTION_REUSE_HTTP_PORT, "reuse-http-port", Option::ArgBool,
       "Allow multiple servers to listen on the same HTTP port when every "
       "server has this option set. If you plan to use this option as a way to "
       "load balance between different Triton servers, the same model "
       "repository or set of models must be used for every server."},
      {OPTION_HTTP_ADDRESS, "http-address", Option::ArgStr,
       "The address for the http server to binds to."},
      {OPTION_HTTP_THREAD_COUNT, "http-thread-count", Option::ArgInt,
       "Number of threads handling HTTP requests."},
#endif  // TRITON_ENABLE_HTTP
#if defined(TRITON_ENABLE_GRPC)
      {OPTION_ALLOW_GRPC, "allow-grpc", Option::ArgBool,
       "Allow the server to listen for GRPC requests."},
      {OPTION_GRPC_PORT, "grpc-port", Option::ArgInt,
       "The port for the server to listen on for GRPC requests."},
      {OPTION_REUSE_GRPC_PORT, "reuse-grpc-port", Option::ArgBool,
       "Allow multiple servers to listen on the same GRPC port when every "
       "server has this option set. If you plan to use this option as a way to "
       "load balance between different Triton servers, the same model "
       "repository or set of models must be used for every server."},
      {OPTION_GRPC_ADDRESS, "grpc-address", Option::ArgStr,
       "The address for the grpc server to binds to."},
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
      {OPTION_ALLOW_CPU_METRICS, "allow-cpu-metrics", Option::ArgBool,
       "Allow the server to provide CPU metrics. Ignored unless "
       "--allow-metrics is true."},
      {OPTION_METRICS_PORT, "metrics-port", Option::ArgInt,
       "The port reporting prometheus metrics."},
      {OPTION_METRICS_INTERVAL_MS, "metrics-interval-ms", Option::ArgFloat,
       "Metrics will be collected once every <metrics-interval-ms> "
       "milliseconds. Default is 2000 milliseconds."},
#endif  // TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_TRACING
      {OPTION_TRACE_FILEPATH, "trace-file", Option::ArgStr,
       "Set the file where trace output will be saved. If --trace-log-frequency"
       " is also specified, this argument value will be the prefix of the files"
       " to save the trace output. See --trace-log-frequency for detail."},
      {OPTION_TRACE_LEVEL, "trace-level", Option::ArgStr,
       "Specify a trace level. OFF to disable tracing, TIMESTAMPS to "
       "trace timestamps, TENSORS to trace tensors. It may be specified "
       "multiple times to trace multiple informations. Default is OFF."},
      {OPTION_TRACE_RATE, "trace-rate", Option::ArgInt,
       "Set the trace sampling rate. Default is 1000."},
      {OPTION_TRACE_COUNT, "trace-count", Option::ArgInt,
       "Set the number of traces to be sampled. If the value is -1, the number "
       "of traces to be sampled will not be limited. Default is -1."},
      {OPTION_TRACE_LOG_FREQUENCY, "trace-log-frequency", Option::ArgInt,
       "Set the trace log frequency. If the value is 0, Triton will only log "
       "the trace output to <trace-file> when shutting down. Otherwise, Triton "
       "will log the trace output to <trace-file>.<idx> when it collects the "
       "specified number of traces. For example, if the log frequency is 100, "
       "when Triton collects the 100-th trace, it logs the traces to file "
       "<trace-file>.0, and when it collects the 200-th trace, it logs the "
       "101-th to the 200-th traces to file <trace-file>.1. Default is 0."},
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
       "multiple times to add multiple models. To load ALL models at startup, "
       "specify '*' as the model name with --load-model=* as the ONLY "
       "--load-model argument, this does not imply any pattern matching. "
       "Specifying --load-model=* in conjunction with another --load-model "
       "argument will result in error. Note that this option will only take "
       "effect if --model-control-mode=explicit is true."},
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
       Option::ArgInt, "DEPRECATED: Please use --cache-config instead."},
      {OPTION_CACHE_CONFIG, "cache-config", "<string>,<string>=<string>",
       "Specify a cache-specific configuration setting. The format of this "
       "flag is --cache-config=<cache_name>,<setting>=<value>. Where "
       "<cache_name> is the name of the cache, such as 'local' or 'redis'. "
       "Example: --cache-config=local,size=1048576 will configure a 'local' "
       "cache implementation with a fixed buffer pool of size 1048576 bytes."},
      {OPTION_CACHE_DIR, "cache-directory", Option::ArgStr,
       "The global directory searched for cache shared libraries. Default is "
       "'/opt/tritonserver/caches'. This directory is expected to contain a "
       "cache implementation as a shared library with the name "
       "'libtritoncache.so'."},
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
      {OPTION_MODEL_LOAD_THREAD_COUNT, "model-load-thread-count",
       Option::ArgInt,
       "The number of threads used to concurrently load models in "
       "model repositories. Default is 2*<num_cpu_cores>."},
      {OPTION_BACKEND_CONFIG, "backend-config", "<string>,<string>=<string>",
       "Specify a backend-specific configuration setting. The format of this "
       "flag is --backend-config=<backend_name>,<setting>=<value>. Where "
       "<backend_name> is the name of the backend, such as 'tensorrt'."},
      {OPTION_HOST_POLICY, "host-policy", "<string>,<string>=<string>",
       "Specify a host policy setting associated with a policy name. The "
       "format of this flag is --host-policy=<policy_name>,<setting>=<value>. "
       "Currently supported settings are 'numa-node', 'cpu-cores'. Note that "
       "'numa-node' setting will affect pinned memory pool behavior, see "
       "--pinned-memory-pool for more detail."},
      {OPTION_MODEL_LOAD_GPU_LIMIT, "model-load-gpu-limit",
       "<device_id>:<fraction>",
       "Specify the limit on GPU memory usage as a fraction. If model loading "
       "on the device is requested and the current memory usage exceeds the "
       "limit, the load will be rejected. If not specified, the limit will "
       "not be set."},
  {
    OPTION_MODEL_NAMESPACING, "model-namespacing", Option::ArgBool,
        "Whether model namespacing is enable or not. If true, models with the "
        "same name can be served if they are in different namespace."
  }
};

bool
TritonServerParameters::CheckPortCollision()
{
  // List of enabled services and their constraints
  std::vector<
      std::tuple<std::string, std::string, int32_t, bool, int32_t, int32_t>>
      ports;
#ifdef TRITON_ENABLE_HTTP
  if (allow_http_) {
    ports.emplace_back("HTTP", http_address_, http_port_, false, -1, -1);
  }
#endif  // TRITON_ENABLE_HTTP
#ifdef TRITON_ENABLE_GRPC
  if (allow_grpc_) {
    ports.emplace_back(
        "GRPC", grpc_options_.socket_.address_, grpc_options_.socket_.port_,
        false, -1, -1);
  }
#endif  // TRITON_ENABLE_GRPC
#ifdef TRITON_ENABLE_METRICS
  if (allow_metrics_) {
    ports.emplace_back("metrics", http_address_, metrics_port_, false, -1, -1);
  }
#endif  // TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_SAGEMAKER
  if (allow_sagemaker_) {
    ports.emplace_back(
        "SageMaker", sagemaker_address_, sagemaker_port_,
        sagemaker_safe_range_set_, sagemaker_safe_range_.first,
        sagemaker_safe_range_.second);
  }
#endif  // TRITON_ENABLE_SAGEMAKER
#ifdef TRITON_ENABLE_VERTEX_AI
  if (allow_vertex_ai_) {
    ports.emplace_back(
        "Vertex AI", vertex_ai_address_, vertex_ai_port_, false, -1, -1);
  }
#endif  // TRITON_ENABLE_VERTEX_AI

  for (auto curr_it = ports.begin(); curr_it != ports.end(); ++curr_it) {
    // If the current service doesn't specify the allow port range for other
    // services, then we don't need to revisit the checked services
    auto comparing_it = (std::get<3>(*curr_it)) ? ports.begin() : (curr_it + 1);
    for (; comparing_it != ports.end(); ++comparing_it) {
      if (comparing_it == curr_it) {
        continue;
      }
      if (std::get<1>(*curr_it) != std::get<1>(*comparing_it)) {
        continue;
      }
      // Set range and comparing service port is out of range
      if (std::get<3>(*curr_it) &&
          ((std::get<2>(*comparing_it) < std::get<4>(*curr_it)) ||
           (std::get<2>(*comparing_it) > std::get<5>(*curr_it)))) {
        std::cerr << "The server cannot listen to "
                  << std::get<0>(*comparing_it) << " requests at port "
                  << std::get<2>(*comparing_it) << ", allowed port range is ["
                  << std::get<4>(*curr_it) << ", " << std::get<5>(*curr_it)
                  << "]" << std::endl;
        return true;
      }
      if (std::get<2>(*curr_it) == std::get<2>(*comparing_it)) {
        std::cerr << "The server cannot listen to " << std::get<0>(*curr_it)
                  << " requests "
                  << "and " << std::get<0>(*comparing_it)
                  << " requests at the same address and port "
                  << std::get<1>(*curr_it) << ":" << std::get<2>(*curr_it)
                  << std::endl;
        return true;
      }
    }
  }

  return false;
}

TritonServerParameters::ManagedTritonServerOptionPtr
TritonServerParameters::BuildTritonServerOptions()
{
  TRITONSERVER_ServerOptions* loptions = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsNew(&loptions), "creating server options");
  ManagedTritonServerOptionPtr managed_ptr(
      loptions, TRITONSERVER_ServerOptionsDelete);
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetServerId(loptions, server_id_.c_str()),
      "setting server ID");
  for (const auto& model_repository_path : model_repository_paths_) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetModelRepositoryPath(
            loptions, model_repository_path.c_str()),
        "setting model repository path");
  }
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelControlMode(loptions, control_mode_),
      "setting model control mode");
  for (const auto& model : startup_models_) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetStartupModel(loptions, model.c_str()),
        "setting startup model");
  }
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetRateLimiterMode(loptions, rate_limit_mode_),
      "setting rate limiter configuration");
  for (const auto& resource : rate_limit_resources_) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsAddRateLimiterResource(
            loptions, std::get<0>(resource).c_str(), std::get<1>(resource),
            std::get<2>(resource)),
        "setting rate limiter resource");
  }
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
          loptions, pinned_memory_pool_byte_size_),
      "setting total pinned memory byte size");
  for (const auto& cuda_pool : cuda_pools_) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
            loptions, cuda_pool.first, cuda_pool.second),
        "setting total CUDA memory byte size");
  }
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
          loptions, min_supported_compute_capability_),
      "setting minimum supported CUDA compute capability");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetExitOnError(loptions, exit_on_error_),
      "setting exit on error");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictModelConfig(
          loptions, strict_model_config_),
      "setting strict model configuration");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictReadiness(loptions, strict_readiness_),
      "setting strict readiness");
  // [FIXME] std::max seems to be part of Parse()
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetExitTimeout(
          loptions, std::max(0, exit_timeout_secs_)),
      "setting exit timeout");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(
          loptions, std::max(0, buffer_manager_thread_count_)),
      "setting buffer manager thread count");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelLoadThreadCount(
          loptions, std::max(1u, model_load_thread_count_)),
      "setting model load thread count");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelNamespacing(
          loptions, enable_model_namespacing_),
      "setting model namespacing");

#ifdef TRITON_ENABLE_LOGGING
  TRITONSERVER_ServerOptionsSetLogFile(loptions, log_file_.c_str());
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogInfo(loptions, log_info_),
      "setting log info enable");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogWarn(loptions, log_warn_),
      "setting log warn enable");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogError(loptions, log_error_),
      "setting log error enable");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogVerbose(loptions, log_verbose_),
      "setting log verbose level");
  switch (log_format_) {
    case triton::common::Logger::Format::kDEFAULT:
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetLogFormat(
              loptions, TRITONSERVER_LOG_DEFAULT),
          "setting log format");
      break;
    case triton::common::Logger::Format::kISO8601:
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetLogFormat(
              loptions, TRITONSERVER_LOG_ISO8601),
          "setting log format");
      break;
  }
#endif  // TRITON_ENABLE_LOGGING

#ifdef TRITON_ENABLE_METRICS
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMetrics(loptions, allow_metrics_),
      "setting metrics enable");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetGpuMetrics(loptions, allow_gpu_metrics_),
      "setting GPU metrics enable");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetCpuMetrics(loptions, allow_cpu_metrics_),
      "setting CPU metrics enable");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMetricsInterval(
          loptions, metrics_interval_ms_),
      "setting metrics interval");
#endif  // TRITON_ENABLE_METRICS

  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetBackendDirectory(
          loptions, backend_dir_.c_str()),
      "setting backend directory");

  // Enable cache and configure it if a cache CLI arg is passed,
  // this will allow for an empty configuration.
  if (enable_cache_) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetCacheDirectory(
            loptions, cache_dir_.c_str()),
        "setting cache directory");

    for (const auto& cache_pair : cache_config_settings_) {
      const auto& cache_name = cache_pair.first;
      const auto& settings = cache_pair.second;
      const auto& json_config_str = PairsToJsonStr(settings);
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetCacheConfig(
              loptions, cache_name.c_str(), json_config_str.c_str()),
          "setting cache configurtion");
    }
  }

  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
          loptions, repoagent_dir_.c_str()),
      "setting repository agent directory");
  for (const auto& bcs : backend_config_settings_) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetBackendConfig(
            loptions, std::get<0>(bcs).c_str(), std::get<1>(bcs).c_str(),
            std::get<2>(bcs).c_str()),
        "setting backend configurtion");
  }
  for (const auto& limit : load_gpu_limit_) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit(
            loptions, TRITONSERVER_INSTANCEGROUPKIND_GPU, limit.first,
            limit.second),
        "setting model load GPU limit");
  }
  for (const auto& hp : host_policies_) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerOptionsSetHostPolicy(
            loptions, std::get<0>(hp).c_str(), std::get<1>(hp).c_str(),
            std::get<2>(hp).c_str()),
        "setting host policy");
  }
  return std::move(managed_ptr);
}

std::pair<TritonServerParameters, std::vector<char*>>
FallThroughParser::Parse(int argc, char** argv)
{
  TritonServerParameters lparams;
  bool strict_model_config_present{false};
  bool disable_auto_complete_config{false};
  bool cache_size_present{false};
  bool cache_config_present{false};

  // [WIP] Parse

  // [WIP]after parse checking...
  // Check if there is a conflict between --response-cache-byte-size
  // and --cache-config
  if (cache_size_present && cache_config_present) {
    std::cerr
        << "Error: Incompatible flags --response-cache-byte-size and "
           "--cache-config both provided. Please provide one or the other."
        << std::endl;
    exit(1);
  }
}

// Used to format the usage message
std::string
CLParser::FormatMessage(std::string str, int offset) const
{
  int width = 60;
  int current_pos = offset;
  while (current_pos + width < int(str.length())) {
    int n = str.rfind(' ', current_pos + width);
    if (n != int(std::string::npos)) {
      str.replace(n, 1, "\n\t ");
      current_pos += (width + 10);
    }
  }
  return str;
}

void
CLParser::Usage(const std::string& msg)
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv_[0] << " [options]" << std::endl;
  std::cerr << "==== SYNOPSIS ====\n \n";
  std::cerr << "\t--service-kind "
               "<\"triton\"|\"tfserving\"|\"torchserve\"|\"triton_c_api\">"
            << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-x <model version>" << std::endl;
  std::cerr << "\t--model-signature-name <model signature name>" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << std::endl;
  std::cerr << "I. MEASUREMENT PARAMETERS: " << std::endl;
  std::cerr << "\t--async (-a)" << std::endl;
  std::cerr << "\t--sync" << std::endl;
  std::cerr << "\t--measurement-interval (-p) <measurement window (in msec)>"
            << std::endl;
  std::cerr << "\t--concurrency-range <start:end:step>" << std::endl;
  std::cerr << "\t--request-rate-range <start:end:step>" << std::endl;
  std::cerr << "\t--request-distribution <\"poisson\"|\"constant\">"
            << std::endl;
  std::cerr << "\t--request-intervals <path to file containing time intervals "
               "in microseconds>"
            << std::endl;
  std::cerr << "\t--binary-search" << std::endl;
  std::cerr << "\t--num-of-sequences <number of concurrent sequences>"
            << std::endl;
  std::cerr << "\t--latency-threshold (-l) <latency threshold (in msec)>"
            << std::endl;
  std::cerr << "\t--max-threads <thread counts>" << std::endl;
  std::cerr << "\t--stability-percentage (-s) <deviation threshold for stable "
               "measurement (in percentage)>"
            << std::endl;
  std::cerr << "\t--max-trials (-r)  <maximum number of measurements for each "
               "profiling>"
            << std::endl;
  std::cerr << "\t--percentile <percentile>" << std::endl;
  std::cerr << "\tDEPRECATED OPTIONS" << std::endl;
  std::cerr << "\t-t <number of concurrent requests>" << std::endl;
  std::cerr << "\t-c <maximum concurrency>" << std::endl;
  std::cerr << "\t-d" << std::endl;
  std::cerr << std::endl;
  std::cerr << "II. INPUT DATA OPTIONS: " << std::endl;
  std::cerr << "\t-b <batch size>" << std::endl;
  std::cerr << "\t--input-data <\"zero\"|\"random\"|<path>>" << std::endl;
  std::cerr << "\t--shared-memory <\"system\"|\"cuda\"|\"none\">" << std::endl;
  std::cerr << "\t--output-shared-memory-size <size in bytes>" << std::endl;
  std::cerr << "\t--shape <name:shape>" << std::endl;
  std::cerr << "\t--sequence-length <length>" << std::endl;
  std::cerr << "\t--sequence-id-range <start:end>" << std::endl;
  std::cerr << "\t--string-length <length>" << std::endl;
  std::cerr << "\t--string-data <string>" << std::endl;
  std::cerr << "\tDEPRECATED OPTIONS" << std::endl;
  std::cerr << "\t-z" << std::endl;
  std::cerr << "\t--data-directory <path>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "III. SERVER DETAILS: " << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
            << std::endl;
  std::cerr << "\t--ssl-grpc-use-ssl <bool>" << std::endl;
  std::cerr << "\t--ssl-grpc-root-certifications-file <path>" << std::endl;
  std::cerr << "\t--ssl-grpc-private-key-file <path>" << std::endl;
  std::cerr << "\t--ssl-grpc-certificate-chain-file <path>" << std::endl;
  std::cerr << "\t--ssl-https-verify-peer <number>" << std::endl;
  std::cerr << "\t--ssl-https-verify-host <number>" << std::endl;
  std::cerr << "\t--ssl-https-ca-certificates-file <path>" << std::endl;
  std::cerr << "\t--ssl-https-client-certificate-file <path>" << std::endl;
  std::cerr << "\t--ssl-https-client-certificate-type <string>" << std::endl;
  std::cerr << "\t--ssl-https-private-key-file <path>" << std::endl;
  std::cerr << "\t--ssl-https-private-key-type <string>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "IV. OTHER OPTIONS: " << std::endl;
  std::cerr << "\t-f <filename for storing report in csv format>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << "\t--streaming" << std::endl;
  std::cerr << "\t--grpc-compression-algorithm <compression_algorithm>"
            << std::endl;
  std::cerr << "\t--trace-file" << std::endl;
  std::cerr << "\t--trace-level" << std::endl;
  std::cerr << "\t--trace-rate" << std::endl;
  std::cerr << "\t--trace-count" << std::endl;
  std::cerr << "\t--log-frequency" << std::endl;
  std::cerr << "\t--collect-metrics" << std::endl;
  std::cerr << "\t--metrics-url" << std::endl;
  std::cerr << "\t--metrics-interval" << std::endl;
  std::cerr << std::endl;
  std::cerr << "==== OPTIONS ==== \n \n";

  std::cerr
      << FormatMessage(
             " --service-kind: Describes the kind of service perf_analyzer to "
             "generate load for. The options are \"triton\", \"triton_c_api\", "
             "\"tfserving\" and \"torchserve\". Default value is \"triton\". "
             "Note in order to use \"torchserve\" backend --input-data option "
             "must point to a json file holding data in the following format "
             "{\"data\" : [{\"TORCHSERVE_INPUT\" : [\"<complete path to the "
             "content file>\"]}, {...}...]}. The type of file here will depend "
             "on the model. In order to use \"triton_c_api\" you must specify "
             "the Triton server install path and the model repository "
             "path via the --library-name and --model-repo flags",
             18)
      << std::endl;

  std::cerr
      << std::setw(9) << std::left << " -m: "
      << FormatMessage(
             "This is a required argument and is used to specify the model"
             " against which to run perf_analyzer.",
             9)
      << std::endl;
  std::cerr << std::setw(9) << std::left << " -x: "
            << FormatMessage(
                   "The version of the above model to be used. If not specified"
                   " the most recent version (that is, the highest numbered"
                   " version) of the model will be used.",
                   9)
            << std::endl;
  std::cerr << FormatMessage(
                   " --model-signature-name: The signature name of the saved "
                   "model to use. Default value is \"serving_default\". This "
                   "option will be ignored if --service-kind is not "
                   "\"tfserving\".",
                   18)
            << std::endl;
  std::cerr << std::setw(9) << std::left
            << " -v: " << FormatMessage("Enables verbose mode.", 9)
            << std::endl;
  std::cerr << std::setw(9) << std::left
            << " -v -v: " << FormatMessage("Enables extra verbose mode.", 9)
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "I. MEASUREMENT PARAMETERS: " << std::endl;
  std::cerr
      << FormatMessage(
             " --async (-a): Enables asynchronous mode in perf_analyzer. "
             "By default, perf_analyzer will use synchronous API to "
             "request inference. However, if the model is sequential "
             "then default mode is asynchronous. Specify --sync to "
             "operate sequential models in synchronous mode. In synchronous "
             "mode, perf_analyzer will start threads equal to the concurrency "
             "level. Use asynchronous mode to limit the number of threads, yet "
             "maintain the concurrency.",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --sync: Force enables synchronous mode in perf_analyzer. "
                   "Can be used to operate perf_analyzer with sequential model "
                   "in synchronous mode.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --measurement-interval (-p): Indicates the time interval used "
             "for each measurement in milliseconds. The perf analyzer will "
             "sample a time interval specified by -p and take measurement over "
             "the requests completed within that time interval. The default "
             "value is 5000 msec.",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --measurement-mode <\"time_windows\"|\"count_windows\">: "
                   "Indicates the mode used for stabilizing measurements."
                   " \"time_windows\" will create windows such that the length "
                   "of each window is equal to --measurement-interval. "
                   "\"count_windows\" will create "
                   "windows such that there are at least "
                   "--measurement-request-count requests in each window.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --measurement-request-count: "
             "Indicates the minimum number of requests to be collected in each "
             "measurement window when \"count_windows\" mode is used. This "
             "mode can "
             "be enabled using the --measurement-mode flag.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --concurrency-range <start:end:step>: Determines the range of "
             "concurrency levels covered by the perf_analyzer. The "
             "perf_analyzer "
             "will start from the concurrency level of 'start' and go till "
             "'end' with a stride of 'step'. The default value of 'end' and "
             "'step' are 1. If 'end' is not specified then perf_analyzer will "
             "run for a single concurrency level determined by 'start'. If "
             "'end' is set as 0, then the concurrency limit will be "
             "incremented by 'step' till latency threshold is met. 'end' and "
             "--latency-threshold can not be both 0 simultaneously. 'end' can "
             "not be 0 for sequence models while using asynchronous mode.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-rate-range <start:end:step>: Determines the range of "
             "request rates for load generated by analyzer. This option can "
             "take floating-point values. The search along the request rate "
             "range is enabled only when using this option. If not specified, "
             "then analyzer will search along the concurrency-range. The "
             "perf_analyzer will start from the request rate of 'start' and go "
             "till 'end' with a stride of 'step'. The default values of "
             "'start', 'end' and 'step' are all 1.0. If 'end' is not specified "
             "then perf_analyzer will run for a single request rate as "
             "determined by 'start'. If 'end' is set as 0.0, then the request "
             "rate will be incremented by 'step' till latency threshold is "
             "met. 'end' and --latency-threshold can not be both 0 "
             "simultaneously.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-distribution <\"poisson\"|\"constant\">: Specifies "
             "the time interval distribution between dispatching inference "
             "requests to the server. Poisson distribution closely mimics the "
             "real-world work load on a server. This option is ignored if not "
             "using --request-rate-range. By default, this option is set to be "
             "constant.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --request-intervals: Specifies a path to a file containing time "
             "intervals in microseconds. Each time interval should be in a new "
             "line. The analyzer will try to maintain time intervals between "
             "successive generated requests to be as close as possible in this "
             "file. This option can be used to apply custom load to server "
             "with a certain pattern of interest. The analyzer will loop "
             "around the file if the duration of execution exceeds to that "
             "accounted for by the intervals. This option can not be used with "
             "--request-rate-range or --concurrency-range.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             "--binary-search: Enables the binary search on the specified "
             "search range. This option requires 'start' and 'end' to be "
             "expilicitly specified in the --concurrency-range or "
             "--request-rate-range. When using this option, 'step' is more "
             "like the precision. Lower the 'step', more the number of "
             "iterations along the search path to find suitable convergence. "
             "By default, linear search is used.",
             18)
      << std::endl;

  std::cerr << FormatMessage(
                   "--num-of-sequences: Sets the number of concurrent "
                   "sequences for sequence models. This option is ignored when "
                   "--request-rate-range is not specified. By default, its "
                   "value is 4.",
                   18)
            << std::endl;

  std::cerr
      << FormatMessage(
             " --latency-threshold (-l): Sets the limit on the observed "
             "latency. Analyzer will terminate the concurrency search once "
             "the measured latency exceeds this threshold. By default, "
             "latency threshold is set 0 and the perf_analyzer will run "
             "for entire --concurrency-range.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --max-threads: Sets the maximum number of threads that will be "
             "created for providing desired concurrency or request rate. "
             "However, when running"
             "in synchronous mode with concurrency-range having explicit 'end' "
             "specification,"
             "this value will be ignored. Default is 4 if --request-rate-range "
             "is specified otherwise default is 16.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --stability-percentage (-s): Indicates the allowed variation in "
             "latency measurements when determining if a result is stable. The "
             "measurement is considered as stable if the ratio of max / min "
             "from the recent 3 measurements is within (stability percentage)% "
             "in terms of both infer per second and latency. Default is "
             "10(%).",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --max-trials (-r): Indicates the maximum number of "
                   "measurements for each concurrency level visited during "
                   "search. The perf analyzer will take multiple measurements "
                   "and report the measurement until it is stable. The perf "
                   "analyzer will abort if the measurement is still unstable "
                   "after the maximum number of measurements. The default "
                   "value is 10.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --percentile: Indicates the confidence value as a percentile "
             "that will be used to determine if a measurement is stable. For "
             "example, a value of 85 indicates that the 85th percentile "
             "latency will be used to determine stability. The percentile will "
             "also be reported in the results. The default is -1 indicating "
             "that the average latency is used to determine stability",
             18)
      << std::endl;
  std::cerr << std::endl;
  std::cerr << "II. INPUT DATA OPTIONS: " << std::endl;
  std::cerr << std::setw(9) << std::left
            << " -b: " << FormatMessage("Batch size for each request sent.", 9)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --input-data: Select the type of data that will be used "
             "for input in inference requests. The available options are "
             "\"zero\", \"random\", path to a directory or a json file. If the "
             "option is path to a directory then the directory must "
             "contain a binary/text file for each non-string/string input "
             "respectively, named the same as the input. Each "
             "file must contain the data required for that input for a batch-1 "
             "request. Each binary file should contain the raw binary "
             "representation of the input in row-major order for non-string "
             "inputs. The text file should contain all strings needed by "
             "batch-1, each in a new line, listed in row-major order. When "
             "pointing to a json file, user must adhere to the format "
             "described in the Performance Analyzer documentation. By "
             "specifying json data users can control data used with every "
             "request. Multiple data streams can be specified for a sequence "
             "model and the analyzer will select a data stream in a "
             "round-robin fashion for every new sequence. Muliple json files "
             "can also be provided (--input-data json_file1 --input-data "
             "json-file2 and so on) and the analyzer will append data streams "
             "from each file. When using --service-kind=torchserve make sure "
             "this option points to a json file. Default is \"random\".",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --shared-memory <\"system\"|\"cuda\"|\"none\">: Specifies "
                   "the type of the shared memory to use for input and output "
                   "data. Default is none.",
                   18)
            << std::endl;

  std::cerr
      << FormatMessage(
             " --output-shared-memory-size: The size in bytes of the shared "
             "memory region to allocate per output tensor. Only needed when "
             "one or more of the outputs are of string type and/or variable "
             "shape. The value should be larger than the size of the largest "
             "output tensor the model is expected to return. The analyzer will "
             "use the following formula to calculate the total shared memory "
             "to allocate: output_shared_memory_size * number_of_outputs * "
             "batch_size. Defaults to 100KB.",
             18)
      << std::endl;

  std::cerr << FormatMessage(
                   " --shape: The shape used for the specified input. The "
                   "argument must be specified as 'name:shape' where the shape "
                   "is a comma-separated list for dimension sizes, for example "
                   "'--shape input_name:1,2,3' indicate tensor shape [ 1, 2, 3 "
                   "]. --shape may be specified multiple times to specify "
                   "shapes for different inputs.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --sequence-length: Indicates the base length of a "
                   "sequence used for sequence models. A sequence with length "
                   "x will be composed of x requests to be sent as the "
                   "elements in the sequence. The length of the actual "
                   "sequence will be within +/- 20% of the base length.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --sequence-id-range <start:end>: Determines the range of "
             "sequence id used by the perf_analyzer. The perf_analyzer "
             "will start from the sequence id of 'start' and go till "
             "'end' (excluded). If 'end' is not specified then perf_analyzer "
             "will use new sequence id without bounds. If 'end' is specified "
             "and the concurrency setting may result in maintaining a number "
             "of sequences more than the range of available sequence id, "
             "perf analyzer will exit with error due to possible sequence id "
             "collision. The default setting is start from sequence id 1 and "
             "without bounds",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --string-length: Specifies the length of the random "
                   "strings to be generated by the analyzer for string input. "
                   "This option is ignored if --input-data points to a "
                   "directory. Default is 128.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --string-data: If provided, analyzer will use this string "
                   "to initialize string input buffers. The perf analyzer will "
                   "replicate the given string to build tensors of required "
                   "shape. --string-length will not have any effect. This "
                   "option is ignored if --input-data points to a directory.",
                   18)
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "III. SERVER DETAILS: " << std::endl;
  std::cerr << std::setw(38) << std::left << " -u: "
            << FormatMessage(
                   "Specify URL to the server. When using triton default is "
                   "\"localhost:8000\" if using HTTP and \"localhost:8001\" "
                   "if using gRPC. When using tfserving default is "
                   "\"localhost:8500\". ",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " -i: "
            << FormatMessage(
                   "The communication protocol to use. The available protocols "
                   "are gRPC and HTTP. Default is HTTP.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " --ssl-grpc-use-ssl: "
            << FormatMessage(
                   "Bool (true|false) for whether "
                   "to use encrypted channel to the server. Default false.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left
            << " --ssl-grpc-root-certifications-file: "
            << FormatMessage(
                   "Path to file containing the "
                   "PEM encoding of the server root certificates.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " --ssl-grpc-private-key-file: "
            << FormatMessage(
                   "Path to file containing the "
                   "PEM encoding of the client's private key.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left
            << " --ssl-grpc-certificate-chain-file: "
            << FormatMessage(
                   "Path to file containing the "
                   "PEM encoding of the client's certificate chain.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " --ssl-https-verify-peer: "
            << FormatMessage(
                   "Number (0|1) to verify the "
                   "peer's SSL certificate. See "
                   "https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYPEER.html for "
                   "the meaning of each value. Default is 1.",
                   38)
            << std::endl;
  std::cerr
      << std::setw(38) << std::left << " --ssl-https-verify-host: "
      << FormatMessage(
             "Number (0|1|2) to verify the "
             "certificate's name against host. "
             "See https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYHOST.html for "
             "the meaning of each value. Default is 2.",
             38)
      << std::endl;
  std::cerr << std::setw(38) << std::left
            << " --ssl-https-ca-certificates-file: "
            << FormatMessage(
                   "Path to Certificate Authority "
                   "(CA) bundle.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left
            << " --ssl-https-client-certificate-file: "
            << FormatMessage("Path to the SSL client certificate.", 38)
            << std::endl;
  std::cerr << std::setw(38) << std::left
            << " --ssl-https-client-certificate-type: "
            << FormatMessage(
                   "Type (PEM|DER) of the client "
                   "SSL certificate. Default is PEM.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " --ssl-https-private-key-file: "
            << FormatMessage(
                   "Path to the private keyfile "
                   "for TLS and SSL client cert.",
                   38)
            << std::endl;
  std::cerr << std::setw(38) << std::left << " --ssl-https-private-key-type: "
            << FormatMessage(
                   "Type (PEM|DER) of the private "
                   "key file. Default is PEM.",
                   38)
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "IV. OTHER OPTIONS: " << std::endl;
  std::cerr
      << std::setw(9) << std::left << " -f: "
      << FormatMessage(
             "The latency report will be stored in the file named by "
             "this option. By default, the result is not recorded in a file.",
             9)
      << std::endl;
  std::cerr
      << std::setw(9) << std::left << " -H: "
      << FormatMessage(
             "The header will be added to HTTP requests (ignored for GRPC "
             "requests). The header must be specified as 'Header:Value'. -H "
             "may be specified multiple times to add multiple headers.",
             9)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --streaming: Enables the use of streaming API. This flag is "
             "only valid with gRPC protocol. By default, it is set false.",
             18)
      << std::endl;

  std::cerr << FormatMessage(
                   " --grpc-compression-algorithm: The compression algorithm "
                   "to be used by gRPC when sending request. Only supported "
                   "when grpc protocol is being used. The supported values are "
                   "none, gzip, and deflate. Default value is none.",
                   18)
            << std::endl;

  std::cerr
      << FormatMessage(
             " --trace-file: Set the file where trace output will be saved."
             " If --trace-log-frequency is also specified, this argument "
             "value will be the prefix of the files to save the trace "
             "output. See --trace-log-frequency for details. Only used for "
             "service-kind of triton. Default value is none.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             "--trace-level: Specify a trace level. OFF to disable tracing, "
             "TIMESTAMPS to trace timestamps, TENSORS to trace tensors. It "
             "may be specified multiple times to trace multiple "
             "informations. Default is OFF.",
             18)
      << std::endl;
  std::cerr
      << FormatMessage(
             " --trace-rate: Set the trace sampling rate. Default is 1000.", 18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --trace-count: Set the number of traces to be sampled. "
                   "If the value is -1, the number of traces to be sampled "
                   "will not be limited. Default is -1.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --log-frequency:  Set the trace log frequency. If the "
             "value is 0, Triton will only log the trace output to "
             "<trace-file> when shutting down. Otherwise, Triton will log "
             "the trace output to <trace-file>.<idx> when it collects the "
             "specified number of traces. For example, if the log frequency "
             "is 100, when Triton collects the 100-th trace, it logs the "
             "traces to file <trace-file>.0, and when it collects the 200-th "
             "trace, it logs the 101-th to the 200-th traces to file "
             "<trace-file>.1. Default is 0.",
             18)
      << std::endl;

  std::cerr << FormatMessage(
                   " --triton-server-directory: The Triton server install "
                   "path. Required by and only used when C API "
                   "is used (--service-kind=triton_c_api). "
                   "eg:--triton-server-directory=/opt/tritonserver.",
                   18)
            << std::endl;
  std::cerr
      << FormatMessage(
             " --model-repository: The model repository of which the model is "
             "loaded. Required by and only used when C API is used "
             "(--service-kind=triton_c_api). "
             "eg:--model-repository=/tmp/host/docker-data/model_unit_test.",
             18)
      << std::endl;
  std::cerr << FormatMessage(
                   " --verbose-csv: The csv files generated by perf analyzer "
                   "will include additional information.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --collect-metrics: Enables collection of server-side "
                   "inference server metrics. Outputs metrics in the csv file "
                   "generated with the -f option. Must enable `--verbose-csv` "
                   "option to use the `--collect-metrics`.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --metrics-url: The URL to query for server-side inference "
                   "server metrics. Default is 'localhost:8002/metrics'.",
                   18)
            << std::endl;
  std::cerr << FormatMessage(
                   " --metrics-interval: How often in milliseconds, within "
                   "each measurement window, to query for server-side "
                   "inference server metrics. Default is 1000.",
                   18)
            << std::endl;
  exit(GENERIC_ERROR);
}

void
CLParser::ParseCommandLine(int argc, char** argv)
{
  argc_ = argc;
  argv_ = argv;

  // {name, has_arg, *flag, val}
  static struct option long_options[] = {
      {"streaming", no_argument, 0, 0},
      {"max-threads", required_argument, 0, 1},
      {"sequence-length", required_argument, 0, 2},
      {"percentile", required_argument, 0, 3},
      {"data-directory", required_argument, 0, 4},
      {"shape", required_argument, 0, 5},
      {"measurement-interval", required_argument, 0, 6},
      {"concurrency-range", required_argument, 0, 7},
      {"latency-threshold", required_argument, 0, 8},
      {"stability-percentage", required_argument, 0, 9},
      {"max-trials", required_argument, 0, 10},
      {"input-data", required_argument, 0, 11},
      {"string-length", required_argument, 0, 12},
      {"string-data", required_argument, 0, 13},
      {"async", no_argument, 0, 14},
      {"sync", no_argument, 0, 15},
      {"request-rate-range", required_argument, 0, 16},
      {"num-of-sequences", required_argument, 0, 17},
      {"binary-search", no_argument, 0, 18},
      {"request-distribution", required_argument, 0, 19},
      {"request-intervals", required_argument, 0, 20},
      {"shared-memory", required_argument, 0, 21},
      {"output-shared-memory-size", required_argument, 0, 22},
      {"service-kind", required_argument, 0, 23},
      {"model-signature-name", required_argument, 0, 24},
      {"grpc-compression-algorithm", required_argument, 0, 25},
      {"measurement-mode", required_argument, 0, 26},
      {"measurement-request-count", required_argument, 0, 27},
      {"triton-server-directory", required_argument, 0, 28},
      {"model-repository", required_argument, 0, 29},
      {"sequence-id-range", required_argument, 0, 30},
      {"ssl-grpc-use-ssl", no_argument, 0, 31},
      {"ssl-grpc-root-certifications-file", required_argument, 0, 32},
      {"ssl-grpc-private-key-file", required_argument, 0, 33},
      {"ssl-grpc-certificate-chain-file", required_argument, 0, 34},
      {"ssl-https-verify-peer", required_argument, 0, 35},
      {"ssl-https-verify-host", required_argument, 0, 36},
      {"ssl-https-ca-certificates-file", required_argument, 0, 37},
      {"ssl-https-client-certificate-file", required_argument, 0, 38},
      {"ssl-https-client-certificate-type", required_argument, 0, 39},
      {"ssl-https-private-key-file", required_argument, 0, 40},
      {"ssl-https-private-key-type", required_argument, 0, 41},
      {"verbose-csv", no_argument, 0, 42},
      {"enable-mpi", no_argument, 0, 43},
      {"trace-file", required_argument, 0, 44},
      {"trace-level", required_argument, 0, 45},
      {"trace-rate", required_argument, 0, 46},
      {"trace-count", required_argument, 0, 47},
      {"log-frequency", required_argument, 0, 48},
      {"collect-metrics", no_argument, 0, 49},
      {"metrics-url", required_argument, 0, 50},
      {"metrics-interval", required_argument, 0, 51},
      {0, 0, 0, 0}};

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(
              argc, argv, "vdazc:u:m:x:b:t:p:i:H:l:r:s:f:", long_options,
              NULL)) != -1) {
    switch (opt) {
      case 0:
        params_->streaming = true;
        break;
      case 1:
        params_->max_threads = std::atoi(optarg);
        params_->max_threads_specified = true;
        break;
      case 2:
        params_->sequence_length = std::atoi(optarg);
        break;
      case 3:
        params_->percentile = std::atoi(optarg);
        break;
      case 4:
        params_->user_data.push_back(optarg);
        break;
      case 5: {
        std::string arg = optarg;
        auto colon_pos = arg.rfind(":");
        if (colon_pos == std::string::npos) {
          Usage(
              "failed to parse input shape. There must be a colon after input "
              "name.");
        }
        std::string name = arg.substr(0, colon_pos);
        std::string shape_str = arg.substr(name.size() + 1);
        size_t pos = 0;
        std::vector<int64_t> shape;
        try {
          while (pos != std::string::npos) {
            size_t comma_pos = shape_str.find(",", pos);
            int64_t dim;
            if (comma_pos == std::string::npos) {
              dim = std::stoll(shape_str.substr(pos, comma_pos));
              pos = comma_pos;
            } else {
              dim = std::stoll(shape_str.substr(pos, comma_pos - pos));
              pos = comma_pos + 1;
            }
            if (dim <= 0) {
              Usage("input shape must be > 0");
            }
            shape.emplace_back(dim);
          }
        }
        catch (const std::invalid_argument& ia) {
          Usage("failed to parse input shape: " + std::string(optarg));
        }
        params_->input_shapes[name] = shape;
        break;
      }
      case 6: {
        params_->measurement_window_ms = std::atoi(optarg);
        break;
      }
      case 7: {
        params_->using_concurrency_range = true;
        std::string arg = optarg;
        size_t pos = 0;
        int index = 0;
        try {
          while (pos != std::string::npos) {
            size_t colon_pos = arg.find(":", pos);
            if (index > 2) {
              Usage(
                  "option concurrency-range can have maximum of three "
                  "elements");
            }
            int64_t val;
            if (colon_pos == std::string::npos) {
              val = std::stoll(arg.substr(pos, colon_pos));
              pos = colon_pos;
            } else {
              val = std::stoll(arg.substr(pos, colon_pos - pos));
              pos = colon_pos + 1;
            }
            switch (index) {
              case 0:
                params_->concurrency_range.start = val;
                break;
              case 1:
                params_->concurrency_range.end = val;
                break;
              case 2:
                params_->concurrency_range.step = val;
                break;
            }
            index++;
          }
        }
        catch (const std::invalid_argument& ia) {
          Usage("failed to parse concurrency range: " + std::string(optarg));
        }
        break;
      }
      case 8: {
        params_->latency_threshold_ms = std::atoi(optarg);
        break;
      }
      case 9: {
        params_->stability_threshold = atof(optarg) / 100;
        break;
      }
      case 10: {
        params_->max_trials = std::atoi(optarg);
        break;
      }
      case 11: {
        std::string arg = optarg;
        // Check whether the argument is a directory
        if (IsDirectory(arg) || IsFile(arg)) {
          params_->user_data.push_back(optarg);
        } else if (arg.compare("zero") == 0) {
          params_->zero_input = true;
        } else if (arg.compare("random") == 0) {
          break;
        } else {
          Usage("unsupported input data provided " + std::string(optarg));
        }
        break;
      }
      case 12: {
        params_->string_length = std::atoi(optarg);
        break;
      }
      case 13: {
        params_->string_data = optarg;
        break;
      }
      case 14: {
        params_->async = true;
        break;
      }
      case 15: {
        params_->forced_sync = true;
        break;
      }
      case 16: {
        params_->using_request_rate_range = true;
        std::string arg = optarg;
        size_t pos = 0;
        int index = 0;
        try {
          while (pos != std::string::npos) {
            size_t colon_pos = arg.find(":", pos);
            if (index > 2) {
              Usage(
                  "option request_rate_range can have maximum of three "
                  "elements");
            }
            if (colon_pos == std::string::npos) {
              params_->request_rate_range[index] =
                  std::stod(arg.substr(pos, colon_pos));
              pos = colon_pos;
            } else {
              params_->request_rate_range[index] =
                  std::stod(arg.substr(pos, colon_pos - pos));
              pos = colon_pos + 1;
              index++;
            }
          }
        }
        catch (const std::invalid_argument& ia) {
          Usage("failed to parse request rate range: " + std::string(optarg));
        }
        break;
      }
      case 17: {
        params_->num_of_sequences = std::atoi(optarg);
        break;
      }
      case 18: {
        params_->search_mode = SearchMode::BINARY;
        break;
      }
      case 19: {
        std::string arg = optarg;
        if (arg.compare("poisson") == 0) {
          params_->request_distribution = Distribution::POISSON;
        } else if (arg.compare("constant") == 0) {
          params_->request_distribution = Distribution::CONSTANT;
        } else {
          Usage(
              "unsupported request distribution provided " +
              std::string(optarg));
        }
        break;
      }
      case 20:
        params_->using_custom_intervals = true;
        params_->request_intervals_file = optarg;
        break;
      case 21: {
        std::string arg = optarg;
        if (arg.compare("system") == 0) {
          params_->shared_memory_type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
        } else if (arg.compare("cuda") == 0) {
#ifdef TRITON_ENABLE_GPU
          params_->shared_memory_type = SharedMemoryType::CUDA_SHARED_MEMORY;
#else
          Usage("cuda shared memory is not supported when TRITON_ENABLE_GPU=0");
#endif  // TRITON_ENABLE_GPU
        }
        break;
      }
      case 22: {
        params_->output_shm_size = std::atoi(optarg);
        break;
      }
      case 23: {
        std::string arg = optarg;
        if (arg.compare("triton") == 0) {
          params_->kind = cb::TRITON;
        } else if (arg.compare("tfserving") == 0) {
          params_->kind = cb::TENSORFLOW_SERVING;
        } else if (arg.compare("torchserve") == 0) {
          params_->kind = cb::TORCHSERVE;
        } else if (arg.compare("triton_c_api") == 0) {
          params_->kind = cb::TRITON_C_API;
        } else {
          Usage("unsupported --service-kind specified");
        }
        break;
      }
      case 24:
        params_->model_signature_name = optarg;
        break;
      case 25: {
        params_->using_grpc_compression = true;
        std::string arg = optarg;
        if (arg.compare("none") == 0) {
          params_->compression_algorithm = cb::COMPRESS_NONE;
        } else if (arg.compare("deflate") == 0) {
          params_->compression_algorithm = cb::COMPRESS_DEFLATE;
        } else if (arg.compare("gzip") == 0) {
          params_->compression_algorithm = cb::COMPRESS_GZIP;
        } else {
          Usage("unsupported --grpc-compression-algorithm specified");
        }
        break;
      }
      case 26: {
        std::string arg = optarg;
        if (arg.compare("time_windows") == 0) {
          params_->measurement_mode = MeasurementMode::TIME_WINDOWS;
        } else if (arg.compare("count_windows") == 0) {
          params_->measurement_mode = MeasurementMode::COUNT_WINDOWS;
        } else {
          Usage("unsupported --measurement-mode specified");
        }
        break;
      }
      case 27: {
        params_->measurement_request_count = std::atoi(optarg);
        break;
      }
      case 28: {
        params_->triton_server_path = optarg;
        break;
      }
      case 29: {
        params_->model_repository_path = optarg;
        break;
      }
      case 30: {
        std::string arg = optarg;
        size_t pos = 0;
        int index = 0;
        try {
          while (pos != std::string::npos) {
            size_t colon_pos = arg.find(":", pos);
            if (index > 1) {
              Usage(
                  "option sequence-id-range can have maximum of two "
                  "elements");
            }
            if (colon_pos == std::string::npos) {
              if (index == 0) {
                params_->start_sequence_id =
                    std::stoll(arg.substr(pos, colon_pos));
              } else {
                params_->sequence_id_range =
                    std::stoll(arg.substr(pos, colon_pos)) -
                    params_->start_sequence_id;
              }
              pos = colon_pos;
            } else {
              params_->start_sequence_id =
                  std::stoll(arg.substr(pos, colon_pos - pos));
              pos = colon_pos + 1;
              index++;
            }
          }
        }
        catch (const std::invalid_argument& ia) {
          Usage("failed to parse sequence-id-range: " + std::string(optarg));
        }
        break;
      }
      case 31: {
        params_->ssl_options.ssl_grpc_use_ssl = true;
        break;
      }
      case 32: {
        if (IsFile(optarg)) {
          params_->ssl_options.ssl_grpc_root_certifications_file = optarg;
        } else {
          Usage(
              "--ssl-grpc-root-certifications-file must be a valid file path");
        }
        break;
      }
      case 33: {
        if (IsFile(optarg)) {
          params_->ssl_options.ssl_grpc_private_key_file = optarg;
        } else {
          Usage("--ssl-grpc-private-key-file must be a valid file path");
        }
        break;
      }
      case 34: {
        if (IsFile(optarg)) {
          params_->ssl_options.ssl_grpc_certificate_chain_file = optarg;
        } else {
          Usage("--ssl-grpc-certificate-chain-file must be a valid file path");
        }
        break;
      }
      case 35: {
        if (std::atol(optarg) == 0 || std::atol(optarg) == 1) {
          params_->ssl_options.ssl_https_verify_peer = std::atol(optarg);
        } else {
          Usage("--ssl-https-verify-peer must be 0 or 1");
        }
        break;
      }
      case 36: {
        if (std::atol(optarg) == 0 || std::atol(optarg) == 1 ||
            std::atol(optarg) == 2) {
          params_->ssl_options.ssl_https_verify_host = std::atol(optarg);
        } else {
          Usage("--ssl-https-verify-host must be 0, 1, or 2");
        }
        break;
      }
      case 37: {
        if (IsFile(optarg)) {
          params_->ssl_options.ssl_https_ca_certificates_file = optarg;
        } else {
          Usage("--ssl-https-ca-certificates-file must be a valid file path");
        }
        break;
      }
      case 38: {
        if (IsFile(optarg)) {
          params_->ssl_options.ssl_https_client_certificate_file = optarg;
        } else {
          Usage(
              "--ssl-https-client-certificate-file must be a valid file path");
        }
        break;
      }
      case 39: {
        if (std::string(optarg) == "PEM" || std::string(optarg) == "DER") {
          params_->ssl_options.ssl_https_client_certificate_type = optarg;
        } else {
          Usage("--ssl-https-client-certificate-type must be 'PEM' or 'DER'");
        }
        break;
      }
      case 40: {
        if (IsFile(optarg)) {
          params_->ssl_options.ssl_https_private_key_file = optarg;
        } else {
          Usage("--ssl-https-private-key-file must be a valid file path");
        }
        break;
      }
      case 41: {
        if (std::string(optarg) == "PEM" || std::string(optarg) == "DER") {
          params_->ssl_options.ssl_https_private_key_type = optarg;
        } else {
          Usage("--ssl-https-private-key-type must be 'PEM' or 'DER'");
        }
        break;
      }
      case 42: {
        params_->verbose_csv = true;
        break;
      }
      case 43: {
        params_->enable_mpi = true;
        break;
      }
      case 44: {
        params_->trace_options["trace_file"] = {optarg};
        break;
      }
      case 45: {
        params_->trace_options["trace_level"] = {optarg};
        break;
      }
      case 46: {
        params_->trace_options["trace_rate"] = {optarg};
        break;
      }
      case 47: {
        params_->trace_options["trace_count"] = {optarg};
        break;
      }
      case 48: {
        params_->trace_options["log_frequency"] = {optarg};
        break;
      }
      case 49: {
        params_->should_collect_metrics = true;
        break;
      }
      case 50: {
        params_->metrics_url = optarg;
        params_->metrics_url_specified = true;
        break;
      }
      case 51: {
        params_->metrics_interval_ms = std::stoull(optarg);
        params_->metrics_interval_ms_specified = true;
        break;
      }
      case 'v':
        params_->extra_verbose = params_->verbose;
        params_->verbose = true;
        break;
      case 'z':
        params_->zero_input = true;
        break;
      case 'd':
        params_->using_old_options = true;
        params_->dynamic_concurrency_mode = true;
        break;
      case 'u':
        params_->url_specified = true;
        params_->url = optarg;
        break;
      case 'm':
        params_->model_name = optarg;
        break;
      case 'x':
        params_->model_version = optarg;
        break;
      case 'b':
        params_->batch_size = std::atoi(optarg);
        params_->using_batch_size = true;
        break;
      case 't':
        params_->using_old_options = true;
        params_->concurrent_request_count = std::atoi(optarg);
        break;
      case 'p':
        params_->measurement_window_ms = std::atoi(optarg);
        break;
      case 'i':
        params_->protocol = ParseProtocol(optarg);
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        (*params_->http_headers)[header] = arg.substr(header.size() + 1);
        break;
      }
      case 'l':
        params_->latency_threshold_ms = std::atoi(optarg);
        break;
      case 'c':
        params_->using_old_options = true;
        params_->max_concurrency = std::atoi(optarg);
        break;
      case 'r':
        params_->max_trials = std::atoi(optarg);
        break;
      case 's':
        params_->stability_threshold = atof(optarg) / 100;
        break;
      case 'f':
        params_->filename = optarg;
        break;
      case 'a':
        params_->async = true;
        break;
      case '?':
        Usage();
        break;
    }
  }

  params_->mpi_driver = std::shared_ptr<triton::perfanalyzer::MPIDriver>{
      std::make_shared<triton::perfanalyzer::MPIDriver>(params_->enable_mpi)};
  params_->mpi_driver->MPIInit(&argc, &argv);

  if (!params_->url_specified &&
      (params_->protocol == cb::ProtocolType::GRPC)) {
    if (params_->kind == cb::BackendKind::TRITON) {
      params_->url = "localhost:8001";
    } else if (params_->kind == cb::BackendKind::TENSORFLOW_SERVING) {
      params_->url = "localhost:8500";
    }
  }

  // Overriding the max_threads default for request_rate search
  if (!params_->max_threads_specified && params_->targeting_concurrency()) {
    params_->max_threads = 16;
  }

  if (params_->using_custom_intervals) {
    // Will be using user-provided time intervals, hence no control variable.
    params_->search_mode = SearchMode::NONE;
  }
}

void
CLParser::VerifyOptions()
{
  if (params_->model_name.empty()) {
    Usage("-m flag must be specified");
  }
  if (params_->batch_size <= 0) {
    Usage("batch size must be > 0");
  }
  if (params_->measurement_window_ms <= 0) {
    Usage("measurement window must be > 0 in msec");
  }
  if (params_->measurement_request_count <= 0) {
    Usage("measurement request count must be > 0");
  }
  if (params_->concurrency_range.start <= 0 ||
      params_->concurrent_request_count < 0) {
    Usage("The start of the search range must be > 0");
  }
  if (params_->request_rate_range[SEARCH_RANGE::kSTART] <= 0) {
    Usage("The start of the search range must be > 0");
  }
  if (params_->protocol == cb::ProtocolType::UNKNOWN) {
    Usage("protocol should be either HTTP or gRPC");
  }
  if (params_->streaming && (params_->protocol != cb::ProtocolType::GRPC)) {
    Usage("streaming is only allowed with gRPC protocol");
  }
  if (params_->using_grpc_compression &&
      (params_->protocol != cb::ProtocolType::GRPC)) {
    Usage("compression is only allowed with gRPC protocol");
  }
  if (params_->max_threads == 0) {
    Usage("maximum number of threads must be > 0");
  }
  if (params_->sequence_length == 0) {
    params_->sequence_length = 20;
    std::cerr << "WARNING: using an invalid sequence length. Perf Analyzer will"
              << " use default value if it is measuring on sequence model."
              << std::endl;
  }
  if (params_->start_sequence_id == 0) {
    params_->start_sequence_id = 1;
    std::cerr << "WARNING: using an invalid start sequence id. Perf Analyzer"
              << " will use default value if it is measuring on sequence model."
              << std::endl;
  }
  if (params_->percentile != -1 &&
      (params_->percentile > 99 || params_->percentile < 1)) {
    Usage("percentile must be -1 for not reporting or in range (0, 100)");
  }
  if (params_->zero_input && !params_->user_data.empty()) {
    Usage("zero input can't be set when data directory is provided");
  }
  if (params_->async && params_->forced_sync) {
    Usage("Both --async and --sync can not be specified simultaneously.");
  }

  if (params_->using_concurrency_range && params_->using_old_options) {
    Usage("can not use deprecated options with --concurrency-range");
  } else if (params_->using_old_options) {
    if (params_->dynamic_concurrency_mode) {
      params_->concurrency_range.end = params_->max_concurrency;
    }
    params_->concurrency_range.start = params_->concurrent_request_count;
  }

  if (params_->using_request_rate_range && params_->using_old_options) {
    Usage("can not use concurrency options with --request-rate-range");
  }

  if (params_->using_request_rate_range && params_->using_concurrency_range) {
    Usage(
        "can not specify concurrency_range and request_rate_range "
        "simultaneously");
  }

  if (params_->using_request_rate_range && params_->mpi_driver->IsMPIRun() &&
      (params_->request_rate_range[SEARCH_RANGE::kEND] != 1.0 ||
       params_->request_rate_range[SEARCH_RANGE::kSTEP] != 1.0)) {
    Usage("cannot use request rate range with multi-model mode");
  }

  if (params_->using_custom_intervals && params_->using_old_options) {
    Usage("can not use deprecated options with --request-intervals");
  }

  if ((params_->using_custom_intervals) &&
      (params_->using_request_rate_range || params_->using_concurrency_range)) {
    Usage(
        "can not use --concurrency-range or --request-rate-range "
        "along with --request-intervals");
  }

  if (params_->using_concurrency_range && params_->mpi_driver->IsMPIRun() &&
      (params_->concurrency_range.end != 1 ||
       params_->concurrency_range.step != 1)) {
    Usage("cannot use concurrency range with multi-model mode");
  }

  if (((params_->concurrency_range.end == NO_LIMIT) ||
       (params_->request_rate_range[SEARCH_RANGE::kEND] ==
        static_cast<double>(NO_LIMIT))) &&
      (params_->latency_threshold_ms == NO_LIMIT)) {
    Usage(
        "The end of the search range and the latency limit can not be both 0 "
        "(or 0.0) simultaneously");
  }

  if (((params_->concurrency_range.end == NO_LIMIT) ||
       (params_->request_rate_range[SEARCH_RANGE::kEND] ==
        static_cast<double>(NO_LIMIT))) &&
      (params_->search_mode == SearchMode::BINARY)) {
    Usage("The end of the range can not be 0 (or 0.0) for binary search mode.");
  }

  if ((params_->search_mode == SearchMode::BINARY) &&
      (params_->latency_threshold_ms == NO_LIMIT)) {
    Usage("The latency threshold can not be 0 for binary search mode.");
  }

  if (((params_->concurrency_range.end < params_->concurrency_range.start) ||
       (params_->request_rate_range[SEARCH_RANGE::kEND] <
        params_->request_rate_range[SEARCH_RANGE::kSTART])) &&
      (params_->search_mode == SearchMode::BINARY)) {
    Usage(
        "The end of the range can not be less than start of the range for "
        "binary search mode.");
  }

  if (params_->kind == cb::TENSORFLOW_SERVING) {
    if (params_->protocol != cb::ProtocolType::GRPC) {
      Usage(
          "perf_analyzer supports only grpc protocol for TensorFlow Serving.");
    } else if (params_->streaming) {
      Usage("perf_analyzer does not support streaming for TensorFlow Serving.");
    } else if (params_->async) {
      Usage("perf_analyzer does not support async API for TensorFlow Serving.");
    } else if (!params_->using_batch_size) {
      params_->batch_size = 0;
    }
  } else if (params_->kind == cb::TORCHSERVE) {
    if (params_->user_data.empty()) {
      Usage(
          "--input-data should be provided with a json file with "
          "input data for torchserve");
    }
  }

  if (params_->kind == cb::BackendKind::TRITON_C_API) {
    if (params_->triton_server_path.empty()) {
      Usage(
          "--triton-server-path should not be empty when using "
          "service-kind=triton_c_api.");
    }

    if (params_->model_repository_path.empty()) {
      Usage(
          "--model-repository should not be empty when using "
          "service-kind=triton_c_api.");
    }

    if (params_->async) {
      Usage(
          "Async mode is not supported by triton_c_api service "
          "kind.");
    }

    params_->protocol = cb::ProtocolType::UNKNOWN;
  }

  if (params_->should_collect_metrics &&
      params_->kind != cb::BackendKind::TRITON) {
    Usage(
        "Server-side metric collection is only supported with Triton client "
        "backend.");
  }

  if (params_->metrics_url_specified &&
      params_->should_collect_metrics == false) {
    Usage(
        "Must specify --collect-metrics when using the --metrics-url option.");
  }

  if (params_->metrics_interval_ms_specified &&
      params_->should_collect_metrics == false) {
    Usage(
        "Must specify --collect-metrics when using the --metrics-interval "
        "option.");
  }

  if (params_->metrics_interval_ms == 0) {
    Usage("Metrics interval must be larger than 0 milliseconds.");
  }
}

}}  // namespace triton::server
