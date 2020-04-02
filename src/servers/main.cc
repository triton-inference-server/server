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
#include <iostream>
#include <list>
#include <mutex>

#ifdef TRTIS_ENABLE_ASAN
#include <sanitizer/lsan_interface.h>
#endif  // TRTIS_ENABLE_ASAN

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "src/core/logging.h"
#include "src/core/tritonserver.h"
#include "src/core/trtserver.h"
#include "src/servers/common.h"
#include "src/servers/shared_memory_manager.h"
#include "src/servers/tracer.h"

#ifdef TRTIS_ENABLE_GPU
static_assert(
    TRTIS_MIN_COMPUTE_CAPABILITY >= 1.0,
    "Invalid TRTIS_MIN_COMPUTE_CAPABILITY specified");
#endif  // TRTIS_ENABLE_GPU

#if defined(TRTIS_ENABLE_HTTP) || defined(TRTIS_ENABLE_METRICS)
#include "src/servers/http_server.h"
#endif  // TRTIS_ENABLE_HTTP || TRTIS_ENABLE_METRICS
#if defined(TRTIS_ENABLE_HTTP_V2) || defined(TRTIS_ENABLE_METRICS)
#include "src/servers/http_server_v2.h"
#endif  // TRTIS_ENABLE_HTTP_V2|| TRTIS_ENABLE_METRICS

#ifdef TRTIS_ENABLE_GRPC
#include "src/servers/grpc_server.h"
#endif  // TRTIS_ENABLE_GRPC
#ifdef TRTIS_ENABLE_GRPC_V2
#include "src/servers/grpc_server_v2.h"
#endif  // TRTIS_ENABLE_GRPC_V2

namespace {

// Exit mutex and cv used to signal the main thread that it should
// close the server and exit.
volatile bool exiting_ = false;
std::mutex exit_mu_;
std::condition_variable exit_cv_;

// Interval, in seconds, when the model repository is polled for
// changes.
int32_t repository_poll_secs_ = 15;

// Whether explicit model control is allowed
bool allow_model_control_ = false;

// Default to using the V1 protocol.
int32_t api_version_ = 1;

// The HTTP, GRPC and metrics service/s and ports. Initialized to
// default values and modifyied based on command-line args. Set to -1
// to indicate the protocol is disabled.
#ifdef TRTIS_ENABLE_HTTP
std::vector<std::unique_ptr<nvidia::inferenceserver::HTTPServer>>
    http_services_;
std::vector<std::string> endpoint_names = {
    "status",    "health", "infer", "modelcontrol", "sharedmemorycontrol",
    "repository"};
#endif  // TRTIS_ENABLE_HTTP
#ifdef TRTIS_ENABLE_HTTP_V2
std::vector<std::unique_ptr<nvidia::inferenceserver::HTTPServerV2>>
    http_services_v2_;
std::vector<std::string> endpoint_names_v2 = {"health", "infer"};
#endif  // TRTIS_ENABLE_HTTP_V2
#if defined(TRTIS_ENABLE_HTTP) || defined(TRTIS_ENABLE_HTTP_V2)
bool allow_http_ = true;
int32_t http_port_ = 8000;
int32_t http_health_port_ = -1;
std::vector<int32_t> http_ports_;
#endif  // TRTIS_ENABLE_HTTP || TRTIS_ENABLE_HTTP_V2

#ifdef TRTIS_ENABLE_GRPC
std::unique_ptr<nvidia::inferenceserver::GRPCServer> grpc_service_;
#endif  // TRTIS_ENABLE_GRPC
#ifdef TRTIS_ENABLE_GRPC_V2
std::unique_ptr<nvidia::inferenceserver::GRPCServerV2> grpc_service_v2_;
#endif  // TRTIS_ENABLE_GRPC_V2
#if defined(TRTIS_ENABLE_GRPC) || defined(TRTIS_ENABLE_GRPC_V2)
bool allow_grpc_ = true;
int32_t grpc_port_ = 8001;
#endif  // TRTIS_ENABLE_GRPC || TRTIS_ENABLE_GRPC_V2

#ifdef TRTIS_ENABLE_METRICS
std::unique_ptr<nvidia::inferenceserver::HTTPServer> metrics_service_;
std::unique_ptr<nvidia::inferenceserver::HTTPServerV2> metrics_service_v2_;
bool allow_metrics_ = true;
int32_t metrics_port_ = 8002;
#endif  // TRTIS_ENABLE_METRICS

#ifdef TRTIS_ENABLE_TRACING
std::string trace_filepath_;
auto trace_level_ = TRITONSERVER_TRACE_LEVEL_DISABLED;
int32_t trace_rate_ = 1000;
#endif  // TRTIS_ENABLE_TRACING

#if defined(TRTIS_ENABLE_GRPC) || defined(TRTIS_ENABLE_GRPC_V2)
// The number of threads to initialize for handling GRPC infer
// requests.
int grpc_infer_thread_cnt_ = 1;

// The number of threads to initialize for handling GRPC stream infer
// requests.
int grpc_stream_infer_thread_cnt_ = 1;

// The maximum number of inference request/response objects that
// remain allocated for reuse. As long as the number of in-flight
// requests doesn't exceed this value there will be no
// allocation/deallocation of request/response objects.
int grpc_infer_allocation_pool_size_ = 8;
#endif  // TRTIS_ENABLE_GRPC || TRTIS_ENABLE_GRPC_V2

#if defined(TRTIS_ENABLE_HTTP) || defined(TRTIS_ENABLE_HTTP_V2)
// The number of threads to initialize for the HTTP front-end.
int http_thread_cnt_ = 8;
#endif  // TRTIS_ENABLE_HTTP || TRTIS_ENABLE_HTTP_V2

// Command-line options
enum OptionId {
  OPTION_HELP = 1000,
#ifdef TRTIS_ENABLE_LOGGING
  OPTION_LOG_VERBOSE,
  OPTION_LOG_INFO,
  OPTION_LOG_WARNING,
  OPTION_LOG_ERROR,
#endif  // TRTIS_ENABLE_LOGGING
  OPTION_ID,
  OPTION_MODEL_REPOSITORY,
  OPTION_EXIT_ON_ERROR,
  OPTION_STRICT_MODEL_CONFIG,
  OPTION_STRICT_READINESS,
  OPTION_API_VERSION,
#if defined(TRTIS_ENABLE_HTTP) || defined(TRTIS_ENABLE_HTTP_V2)
  OPTION_ALLOW_HTTP,
  OPTION_HTTP_PORT,
  OPTION_HTTP_HEALTH_PORT,
  OPTION_HTTP_THREAD_COUNT,
#endif  // TRTIS_ENABLE_HTTP || TRTIS_ENABLE_HTTP_V2
#if defined(TRTIS_ENABLE_GRPC) || defined(TRTIS_ENABLE_GRPC_V2)
  OPTION_ALLOW_GRPC,
  OPTION_GRPC_PORT,
  OPTION_GRPC_INFER_THREAD_COUNT,
  OPTION_GRPC_STREAM_INFER_THREAD_COUNT,
  OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE,
#endif  // TRTIS_ENABLE_GRPC || TRTIS_ENABLE_GRPC_V2
#ifdef TRTIS_ENABLE_METRICS
  OPTION_ALLOW_METRICS,
  OPTION_ALLOW_GPU_METRICS,
  OPTION_METRICS_PORT,
#endif  // TRTIS_ENABLE_METRICS
#ifdef TRTIS_ENABLE_TRACING
  OPTION_TRACE_FILEPATH,
  OPTION_TRACE_LEVEL,
  OPTION_TRACE_RATE,
#endif  // TRTIS_ENABLE_TRACING
  OPTION_MODEL_CONTROL_MODE,
  OPTION_ALLOW_POLL_REPO,
  OPTION_POLL_REPO_SECS,
  OPTION_ALLOW_MODEL_CONTROL,
  OPTION_STARTUP_MODEL,
  OPTION_PINNED_MEMORY_POOL_BYTE_SIZE,
  OPTION_CUDA_MEMORY_POOL_BYTE_SIZE,
  OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY,
  OPTION_EXIT_TIMEOUT_SECS,
  OPTION_TF_ALLOW_SOFT_PLACEMENT,
  OPTION_TF_GPU_MEMORY_FRACTION,
  OPTION_TF_ADD_VGPU,
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

std::vector<Option> options_
{
  {OPTION_HELP, "help", "Print usage", false},
#ifdef TRTIS_ENABLE_LOGGING
      {OPTION_LOG_VERBOSE, "log-verbose",
       "Set verbose logging level. Zero (0) disables verbose logging and "
       "values >= 1 enable verbose logging"},
      {OPTION_LOG_INFO, "log-info", "Enable/disable info-level logging"},
      {OPTION_LOG_WARNING, "log-warning",
       "Enable/disable warning-level logging"},
      {OPTION_LOG_ERROR, "log-error", "Enable/disable error-level logging"},
#endif  // TRTIS_ENABLE_LOGGING
      {OPTION_ID, "id", "Identifier for this server"},
      {OPTION_MODEL_REPOSITORY, "model-store",
       "Path to model repository directory. It may be specified multiple times "
       "to add multiple model repositories. Note that if a model is not unique "
       "across all model repositories at any time, the model will not be "
       "available. This option is deprecated, the preferred usage is "
       "--model-repository"},
      {OPTION_MODEL_REPOSITORY, "model-repository",
       "Path to model repository directory. It may be specified multiple times "
       "to add multiple model repositories. Note that if a model is not unique "
       "across all model repositories at any time, the model will not be "
       "available."},
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
      {OPTION_API_VERSION, "api-version",
       "Version of the GRPC/HTTP API to use. Default is version 1. Allowed "
       "versions are 1 and 2."},
#if defined(TRTIS_ENABLE_HTTP) || defined(TRTIS_ENABLE_HTTP_V2)
      {OPTION_ALLOW_HTTP, "allow-http",
       "Allow the server to listen for HTTP requests."},
      {OPTION_HTTP_PORT, "http-port",
       "The port for the server to listen on for HTTP requests."},
      {OPTION_HTTP_HEALTH_PORT, "http-health-port",
       "The port for the server to listen on for HTTP Health requests."},
      {OPTION_HTTP_THREAD_COUNT, "http-thread-count",
       "Number of threads handling HTTP requests."},
#endif  // TRTIS_ENABLE_HTTP || TRTIS_ENABLE_HTTP_V2
#if defined(TRTIS_ENABLE_GRPC) || defined(TRTIS_ENABLE_GRPC_V2)
      {OPTION_ALLOW_GRPC, "allow-grpc",
       "Allow the server to listen for GRPC requests."},
      {OPTION_GRPC_PORT, "grpc-port",
       "The port for the server to listen on for GRPC requests."},
      {OPTION_GRPC_INFER_THREAD_COUNT, "grpc-infer-thread-count",
       "Number of threads handling GRPC inference requests."},
      {OPTION_GRPC_STREAM_INFER_THREAD_COUNT, "grpc-stream-infer-thread-count",
       "Number of threads handling GRPC stream inference requests."},
      {OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE,
       "grpc-infer-allocation-pool-size",
       "The maximum number of inference request/response objects that remain "
       "allocated for reuse. As long as the number of in-flight requests "
       "doesn't exceed this value there will be no allocation/deallocation of "
       "request/response objects."},
#endif  // TRTIS_ENABLE_GRPC || TRTIS_ENABLE_GRPC_V2
#ifdef TRTIS_ENABLE_METRICS
      {OPTION_ALLOW_METRICS, "allow-metrics",
       "Allow the server to provide prometheus metrics."},
      {OPTION_ALLOW_GPU_METRICS, "allow-gpu-metrics",
       "Allow the server to provide GPU metrics. Ignored unless "
       "--allow-metrics is true."},
      {OPTION_METRICS_PORT, "metrics-port",
       "The port reporting prometheus metrics."},
#endif  // TRTIS_ENABLE_METRICS
#ifdef TRTIS_ENABLE_TRACING
      {OPTION_TRACE_FILEPATH, "trace-file",
       "Set the file where trace output will be saved."},
      {OPTION_TRACE_LEVEL, "trace-level",
       "Set the trace level. OFF to disable tracing, MIN for minimal tracing, "
       "MAX for maximal tracing. Default is OFF."},
      {OPTION_TRACE_RATE, "trace-rate",
       "Set the trace sampling rate. Default is 1000."},
#endif  // TRTIS_ENABLE_TRACING
      {OPTION_MODEL_CONTROL_MODE, "model-control-mode",
       "Specify the mode for model management. Options are \"none\", \"poll\" "
       "and \"explicit\". The default is \"poll\". "
       "For \"none\", the server will load all models in the model "
       "repositories only once at startup. For \"poll\", the server will poll "
       "the model repository to detect changes. The poll rate is controlled by "
       "'repository-poll-secs'. For \"explicit\", the model load and unload is "
       "initiated by using the model control APIs, and the models in the model "
       "repository will not be loaded at startup, unless the model is "
       "specified by --load-model."},
      {OPTION_ALLOW_POLL_REPO, "allow-poll-model-repository",
       "[DEPRECATED] Poll the model repository to detect changes. The poll "
       "rate is controlled by 'repository-poll-secs'. This option is "
       "deprecated by --model-control-mode, this option can not be specified "
       "if --model-control-mode is specified."},
      {OPTION_POLL_REPO_SECS, "repository-poll-secs",
       "Interval in seconds between each poll of the model repository to check "
       "for changes. Valid only when --allow-poll-model-repository=true is "
       "specified."},
      {OPTION_ALLOW_MODEL_CONTROL, "allow-model-control",
       "[DEPRECATED] Allow to load or to unload models explicitly using model "
       "control API. If true the models in the model repository will not be "
       "loaded at startup, unless the model is specified by --load-model. "
       "Cannot be specified if --allow-poll-model-repository is true. This "
       "option is deprecated by --model-control-mode, this option can not be "
       "specified if --model-control-mode is specified."},
      {OPTION_STARTUP_MODEL, "load-model",
       "Name of the model to be loaded on server startup. It may be specified "
       "multiple times to add multiple models. Note that this option will only "
       "take affect if --allow-model-control is true."},
      {OPTION_PINNED_MEMORY_POOL_BYTE_SIZE, "pinned-memory-pool-byte-size",
       "The total byte size that can be allocated as pinned system memory. "
       "If GPU support is enabled, the server will allocate pinned system "
       "memory to accelerate data transfer between host and devices until it "
       "exceeds the specified byte size. This option will not affect the "
       "allocation conducted by the backend frameworks. Default is 256 MB."},
      {OPTION_CUDA_MEMORY_POOL_BYTE_SIZE, "cuda-memory-pool-byte-size",
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
       "min-supported-compute-capability",
       "The minimum supported CUDA compute capability. GPUs that don't support "
       "this compute capability will not be used by the server."},
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
       "allocate all of GPU memory."},
  {
    OPTION_TF_ADD_VGPU, "tf-add-vgpu",
        "Add a tensorflow virtual GPU instances on a physical GPU. Input "
        "should be 2 integers and 1 float separated by semicolons in the "
        "format <physical GPU>;<number of virtual GPUs>;<memory limit per VGPU "
        "in megabytes>. This option can be used multiple times, but only once "
        "per physical GPU device. Subsequent uses will overwrite previous uses "
        "with the same physical device. By default, no VGPUs are enabled."
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
#if (defined(TRTIS_ENABLE_HTTP) || defined(TRTIS_ENABLE_HTTP_V2)) && \
    (defined(TRTIS_ENABLE_GRPC) || defined(TRTIS_ENABLE_GRPC_V2))
  // Check if HTTP and GRPC have shared ports
  if ((std::find(http_ports_.begin(), http_ports_.end(), grpc_port_) !=
       http_ports_.end()) &&
      (grpc_port_ != -1) && allow_http_ && allow_grpc_) {
    std::cerr << "The server cannot listen to HTTP requests "
              << "and GRPC requests at the same port" << std::endl;
    return true;
  }
#endif  // (TRTIS_ENABLE_HTTP || TRTIS_ENABLE_HTTP_V2) && (TRTIS_ENABLE_GRPC ||
        // TRTIS_ENABLE_GRPC_V2)

#if (defined(TRTIS_ENABLE_GRPC) || defined(TRTIS_ENABLE_GRPC_V2)) && \
    defined(TRTIS_ENABLE_METRICS)
  // Check if Metric and GRPC have shared ports
  if ((grpc_port_ == metrics_port_) && (metrics_port_ != -1) && allow_grpc_ &&
      allow_metrics_) {
    std::cerr << "The server cannot provide metrics on same port used for "
              << "GRPC requests" << std::endl;
    return true;
  }
#endif  // (TRTIS_ENABLE_GRPC || TRTIS_ENABLE_GRPC_V2) && TRTIS_ENABLE_METRICS

#if defined(TRTIS_ENABLE_HTTP) && defined(TRTIS_ENABLE_METRICS)
  // Check if Metric and HTTP have shared ports
  if ((std::find(http_ports_.begin(), http_ports_.end(), metrics_port_) !=
       http_ports_.end()) &&
      (metrics_port_ != -1) && allow_http_ && allow_metrics_) {
    std::cerr << "The server cannot provide metrics on same port used for "
              << "HTTP requests" << std::endl;
    return true;
  }
#endif  // TRTIS_ENABLE_HTTP && TRTIS_ENABLE_METRICS

  return false;
}

#ifdef TRTIS_ENABLE_GRPC
TRTSERVER_Error*
StartGrpcService(
    std::unique_ptr<nvidia::inferenceserver::GRPCServer>* service,
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<nvidia::inferenceserver::SharedMemoryManager>&
        shm_manager)
{
  TRTSERVER_Error* err = nvidia::inferenceserver::GRPCServer::Create(
      server, trace_manager, shm_manager, grpc_port_, grpc_infer_thread_cnt_,
      grpc_stream_infer_thread_cnt_, grpc_infer_allocation_pool_size_, service);
  if (err == nullptr) {
    err = (*service)->Start();
  }

  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRTIS_ENABLE_GRPC

#ifdef TRTIS_ENABLE_GRPC_V2
TRITONSERVER_Error*
StartGrpcServiceV2(
    std::unique_ptr<nvidia::inferenceserver::GRPCServerV2>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<nvidia::inferenceserver::SharedMemoryManager>&
        shm_manager)
{
  TRITONSERVER_Error* err = nvidia::inferenceserver::GRPCServerV2::Create(
      server, trace_manager, shm_manager, grpc_port_,
      grpc_infer_allocation_pool_size_, service);
  if (err == nullptr) {
    err = (*service)->Start();
  }

  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRTIS_ENABLE_GRPC_V2

#ifdef TRTIS_ENABLE_HTTP
TRTSERVER_Error*
StartHttpService(
    std::vector<std::unique_ptr<nvidia::inferenceserver::HTTPServer>>* services,
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<nvidia::inferenceserver::SharedMemoryManager>&
        shm_manager,
    std::map<int32_t, std::vector<std::string>>& port_map)
{
  TRTSERVER_Error* err = nvidia::inferenceserver::HTTPServer::CreateAPIServer(
      server, trace_manager, shm_manager, port_map, http_thread_cnt_, services);
  if (err == nullptr) {
    for (auto& http_eps : *services) {
      if (http_eps != nullptr) {
        err = http_eps->Start();
      }
    }
  }

  if (err != nullptr) {
    for (auto& http_eps : *services) {
      if (http_eps != nullptr) {
        http_eps.reset();
      }
    }
  }

  return err;
}
#endif  // TRTIS_ENABLE_HTTP

#ifdef TRTIS_ENABLE_HTTP_V2
TRITONSERVER_Error*
StartHttpV2Service(
    std::vector<std::unique_ptr<nvidia::inferenceserver::HTTPServerV2>>*
        services,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<nvidia::inferenceserver::SharedMemoryManager>&
        shm_manager,
    std::map<int32_t, std::vector<std::string>>& port_map)
{
  TRITONSERVER_Error* err =
      nvidia::inferenceserver::HTTPServerV2::CreateAPIServer(
          server, trace_manager, shm_manager, port_map, http_thread_cnt_,
          services);
  if (err == nullptr) {
    for (auto& http_eps : *services) {
      if (http_eps != nullptr) {
        err = http_eps->Start();
      }
    }
  }

  if (err != nullptr) {
    for (auto& http_eps : *services) {
      if (http_eps != nullptr) {
        http_eps.reset();
      }
    }
  }

  return err;
}
#endif  // TRTIS_ENABLE_HTTP_V2

#ifdef TRTIS_ENABLE_METRICS
TRTSERVER_Error*
StartMetricsService(
    std::unique_ptr<nvidia::inferenceserver::HTTPServer>* service,
    const std::shared_ptr<TRTSERVER_Server>& server)
{
  TRTSERVER_Error* err =
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

TRITONSERVER_Error*
StartMetricsV2Service(
    std::unique_ptr<nvidia::inferenceserver::HTTPServerV2>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server)
{
  TRITONSERVER_Error* err =
      nvidia::inferenceserver::HTTPServerV2::CreateMetricsServer(
          server, metrics_port_, 1 /* HTTP thread count */, service);
  if (err == nullptr) {
    err = (*service)->Start();
  }
  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRTIS_ENABLE_METRICS

bool
StartEndpoints(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    const std::shared_ptr<TRTSERVER_Server>& trt_server,
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager,
    const std::shared_ptr<nvidia::inferenceserver::SharedMemoryManager>&
        shm_manager)
{
  if (api_version_ == 1) {
    const char* id;
    FAIL_IF_ERR(TRTSERVER_ServerId(trt_server.get(), &id), "getting server ID");
    std::cout << "Starting endpoints, '" << id << "' listening on" << std::endl;
  } else {
    TRITONSERVER_Message* message = nullptr;
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerMetadata(server.get(), &message),
        "failed to get server metadata message");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_MessageSerializeToJson(message, &buffer, &byte_size),
        "failed to get server metadata json string");
    rapidjson::Document server_metadata_json;
    server_metadata_json.Parse(buffer, byte_size);
    if (server_metadata_json.HasParseError()) {
      FAIL_IF_TRITON_ERR(
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "failed to parse the server metadata JSON buffer: " +
                  std::string(
                      GetParseError_En(server_metadata_json.GetParseError())) +
                  " at " +
                  std::to_string(server_metadata_json.GetErrorOffset()))
                  .c_str()),
          "");
    }
    std::cout << "Starting endpoints, '"
              << server_metadata_json["name"].GetString() << "' listening on"
              << std::endl;
  }
#ifdef TRTIS_ENABLE_GRPC
  // Enable GRPC endpoints if requested...
  if (allow_grpc_ && (api_version_ == 1) && (grpc_port_ != -1)) {
    TRTSERVER_Error* err = StartGrpcService(
        &grpc_service_, trt_server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRTSERVER_ERROR(err, "failed to start GRPC service");
      return false;
    }
  }
#endif  // TRTIS_ENABLE_GRPC

#ifdef TRTIS_ENABLE_GRPC_V2
  // Enable GRPC V2 endpoints if requested...
  if (allow_grpc_ && (api_version_ == 2) && (grpc_port_ != -1)) {
    TRITONSERVER_Error* err = StartGrpcServiceV2(
        &grpc_service_v2_, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start GRPC V2 service");
      return false;
    }
  }
#endif  // TRTIS_ENABLE_GRPC_V2

#ifdef TRTIS_ENABLE_HTTP
  // Enable HTTP endpoints if requested...
  if (allow_http_ && (api_version_ == 1)) {
    std::map<int32_t, std::vector<std::string>> port_map;

    // Group by port numbers
    for (size_t i = 0; i < http_ports_.size(); i++) {
      if (http_ports_[i] != -1) {
        port_map[http_ports_[i]].push_back(endpoint_names[i]);
      }
    }

    TRTSERVER_Error* err = StartHttpService(
        &http_services_, trt_server, trace_manager, shm_manager, port_map);
    if (err != nullptr) {
      LOG_TRTSERVER_ERROR(err, "failed to start HTTP service");
      return false;
    }
  }
#endif  // TRTIS_ENABLE_HTTP

#ifdef TRTIS_ENABLE_HTTP_V2
  // Enable HTTP endpoints if requested...
  if (allow_http_ && (api_version_ == 2)) {
    std::map<int32_t, std::vector<std::string>> port_map;

    // Group by port numbers
    for (size_t i = 0; i < http_ports_.size(); i++) {
      if (http_ports_[i] != -1) {
        port_map[http_ports_[i]].push_back(endpoint_names_v2[i]);
      }
    }

    TRITONSERVER_Error* err = StartHttpV2Service(
        &http_services_v2_, server, trace_manager, shm_manager, port_map);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start HTTP V2 service");
      return false;
    }
  }
#endif  // TRTIS_ENABLE_HTTP_V2

#ifdef TRTIS_ENABLE_METRICS
  // Enable metrics endpoint if requested...
  if (metrics_port_ != -1) {
    if (api_version_ == 1) {
      TRTSERVER_Error* err = StartMetricsService(&metrics_service_, trt_server);
      if (err != nullptr) {
        LOG_TRTSERVER_ERROR(err, "failed to start Metrics service");
        return false;
      }
    } else if (api_version_ == 2) {
      TRITONSERVER_Error* err =
          StartMetricsV2Service(&metrics_service_v2_, server);
      if (err != nullptr) {
        LOG_TRITONSERVER_ERROR(err, "failed to start Metrics V2 service");
        return false;
      }
    }
  }
#endif  // TRTIS_ENABLE_METRICS

  return true;
}

bool
StopEndpoints()
{
  bool ret = true;

#ifdef TRTIS_ENABLE_HTTP
  for (auto& http_eps : http_services_) {
    if (http_eps != nullptr) {
      TRTSERVER_Error* err = http_eps->Stop();
      if (err != nullptr) {
        LOG_TRTSERVER_ERROR(err, "failed to stop HTTP service");
        ret = false;
      }
    }
  }

  http_services_.clear();
#endif  // TRTIS_ENABLE_HTTP

#ifdef TRTIS_ENABLE_HTTP_V2
  for (auto& http_eps : http_services_v2_) {
    if (http_eps != nullptr) {
      TRITONSERVER_Error* err = http_eps->Stop();
      if (err != nullptr) {
        LOG_TRITONSERVER_ERROR(err, "failed to stop HTTP V2 service");
        ret = false;
      }
    }
  }

  http_services_v2_.clear();
#endif  // TRTIS_ENABLE_HTTP_V2

#ifdef TRTIS_ENABLE_GRPC
  if (grpc_service_) {
    TRTSERVER_Error* err = grpc_service_->Stop();
    if (err != nullptr) {
      LOG_TRTSERVER_ERROR(err, "failed to stop GRPC service");
      ret = false;
    }

    grpc_service_.reset();
  }
#endif  // TRTIS_ENABLE_GRPC

#ifdef TRTIS_ENABLE_GRPC_V2
  if (grpc_service_v2_) {
    TRITONSERVER_Error* err = grpc_service_v2_->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop GRPC V2 service");
      ret = false;
    }

    grpc_service_v2_.reset();
  }
#endif  // TRTIS_ENABLE_GRPC_V2

#ifdef TRTIS_ENABLE_METRICS
  if (metrics_service_) {
    TRTSERVER_Error* err = metrics_service_->Stop();
    if (err != nullptr) {
      LOG_TRTSERVER_ERROR(err, "failed to stop Metrics service");
      ret = false;
    }

    metrics_service_.reset();
  }

  if (metrics_service_v2_) {
    TRITONSERVER_Error* err = metrics_service_v2_->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop Metrics V2 service");
      ret = false;
    }

    metrics_service_v2_.reset();
  }
#endif  // TRTIS_ENABLE_METRICS

  return ret;
}

bool
StartTracing(
    std::shared_ptr<nvidia::inferenceserver::TraceManager>* trace_manager)
{
  trace_manager->reset();

#ifdef TRTIS_ENABLE_TRACING
  TRITONSERVER_Error* err = nullptr;

  // Configure tracing if host is specified.
  if (trace_level_ != TRITONSERVER_TRACE_LEVEL_DISABLED) {
    err = nvidia::inferenceserver::TraceManager::Create(
        trace_manager, trace_filepath_);
    if (err == nullptr) {
      err = (*trace_manager)->SetRate(trace_rate_);
      if (err == nullptr) {
        err = (*trace_manager)->SetLevel(trace_level_);
      }
    }
  }

  if (err != nullptr) {
    LOG_TRITONSERVER_ERROR(err, "failed to configure tracing");
    trace_manager->reset();
    return false;
  }
#endif  // TRTIS_ENABLE_TRACING

  return true;
}

bool
StopTracing(
    const std::shared_ptr<nvidia::inferenceserver::TraceManager>& trace_manager)
{
#ifdef TRTIS_ENABLE_TRACING
#endif  // TRTIS_ENABLE_TRACING

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
ParseBoolOption(std::string arg)
{
  std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  if ((arg == "true") || (arg == "1")) {
    return true;
  }
  if ((arg == "false") || (arg == "0")) {
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

float
ParseFloatOption(const std::string arg)
{
  return std::stof(arg);
}

double
ParseDoubleOption(const std::string arg)
{
  return std::stod(arg);
}

// Condition here merely to avoid compilation error, this function will
// be defined but not used otherwise.
#ifdef TRTIS_ENABLE_LOGGING
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
#endif  // TRTIS_ENABLE_LOGGING

#ifdef TRTIS_ENABLE_TRACING
TRITONSERVER_Trace_Level
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
#endif  // TRTIS_ENABLE_TRACING

struct VgpuOption {
  int gpu_device_;
  int num_vgpus_;
  uint64_t mem_limit_mbytes_;
};

VgpuOption
ParseVGPUOption(const std::string arg)
{
  int delim_gpu = arg.find(";");
  int delim_num_vgpus = arg.find(";", delim_gpu + 1);

  // Check for 2 semicolons
  if ((delim_gpu < 0) || (delim_num_vgpus < 0)) {
    std::cerr << "Cannot add virtual devices due to incorrect number of inputs."
                 "--tf-add-vgpu argument requires format <physical "
                 "GPU>;<number of virtual GPUs>;<memory limit per VGPU in "
                 "megabytes>. "
              << "Found: " << arg << std::endl;
    std::cerr << Usage() << std::endl;
    exit(1);
  }

  std::string gpu_string = arg.substr(0, delim_gpu);
  std::string vgpu_string =
      arg.substr(delim_gpu + 1, delim_num_vgpus - delim_gpu - 1);
  std::string mem_limit_string = arg.substr(delim_num_vgpus + 1);

  // Ensure that options are non-empty otherwise calling stoi/stof will throw an
  // exception
  if (gpu_string.empty() || vgpu_string.empty() || mem_limit_string.empty()) {
    std::cerr << "Cannot add virtual devices due to empty inputs."
                 "--tf-add-vgpu argument requires format <physical "
                 "GPU>;<number of virtual GPUs>;<memory limit per VGPU in "
                 "megabytes>. "
              << "Found: " << arg << std::endl;
    std::cerr << Usage() << std::endl;
    exit(1);
  }

  int gpu_device = std::stoi(gpu_string);
  int num_vgpus_on_device = std::stoi(vgpu_string);
  uint64_t mem_limit = std::stoi(mem_limit_string);

  if (gpu_device < 0) {
    std::cerr << "Cannot add virtual devices. Physical GPU device index must "
                 "be >= 0. "
              << "Found: " << gpu_string << std::endl;
    std::cerr << Usage() << std::endl;
    exit(1);
  }

  if (num_vgpus_on_device <= 0) {
    std::cerr
        << "Cannot add virtual devices. Number of virtual GPUs must be > 0. "
        << "Found: " << vgpu_string << std::endl;
    std::cerr << Usage() << std::endl;
    exit(1);
  }

  return {gpu_device, num_vgpus_on_device, mem_limit};
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
  std::string server_id("inference:0");
  std::set<std::string> model_repository_paths;
  bool exit_on_error = true;
  bool strict_model_config = true;
  bool strict_readiness = true;
  bool tf_allow_soft_placement = true;
  float tf_gpu_memory_fraction = 0.0;
  std::list<VgpuOption> tf_vgpus;
  std::list<std::pair<int, uint64_t>> cuda_pools;
  int32_t exit_timeout_secs = 30;
  int32_t repository_poll_secs = repository_poll_secs_;
  int64_t pinned_memory_pool_byte_size = 1 << 28;

#ifdef TRTIS_ENABLE_GPU
  double min_supported_compute_capability = TRTIS_MIN_COMPUTE_CAPABILITY;
#else
  double min_supported_compute_capability = 0;
#endif  // TRTIS_ENABLE_GPU

#if defined(TRTIS_ENABLE_HTTP) || defined(TRTIS_ENABLE_HTTP_V2)
  int32_t http_port = http_port_;
  int32_t http_thread_cnt = http_thread_cnt_;
  int32_t http_health_port = http_port_;
#endif  // TRTIS_ENABLE_HTTP || TRTIS_ENABLE_HTTP_V2

#if defined(TRTIS_ENABLE_GRPC) || defined(TRTIS_ENABLE_GRPC_V2)
  int32_t grpc_port = grpc_port_;
  int32_t grpc_infer_thread_cnt = grpc_infer_thread_cnt_;
  int32_t grpc_stream_infer_thread_cnt = grpc_stream_infer_thread_cnt_;
  int32_t grpc_infer_allocation_pool_size = grpc_infer_allocation_pool_size_;
#endif  // TRTIS_ENABLE_GRPC || TRTIS_ENABLE_GRPC_V2

#if defined(TRTIS_ENABLE_HTTP) || defined(TRTIS_ENABLE_GRPC)
  int32_t api_version = 1;
#else
  int32_t api_version = 2;
#endif  // TRTIS_ENABLE_HTTP || TRTIS_ENABLE_GRPC

#ifdef TRTIS_ENABLE_METRICS
  int32_t metrics_port = metrics_port_;
  bool allow_gpu_metrics = true;
#endif  // TRTIS_ENABLE_METRICS

#ifdef TRTIS_ENABLE_TRACING
  std::string trace_filepath = trace_filepath_;
  auto trace_level = trace_level_;
  int32_t trace_rate = trace_rate_;
#endif  // TRTIS_ENABLE_TRACING

  bool deprecated_control_mode_set = false;
  bool allow_poll_model_repository = repository_poll_secs > 0;
  bool allow_model_control = allow_model_control_;
  std::set<std::string> startup_models_;

  bool control_mode_set = false;
  TRITONSERVER_Model_Control_Mode control_mode =
      TRITONSERVER_MODEL_CONTROL_POLL;

#ifdef TRTIS_ENABLE_LOGGING
  bool log_info = true;
  bool log_warn = true;
  bool log_error = true;
  int32_t log_verbose = 0;
#endif  // TRTIS_ENABLE_LOGGING

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
#ifdef TRTIS_ENABLE_LOGGING
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
#endif  // TRTIS_ENABLE_LOGGING

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
      case OPTION_API_VERSION:
        api_version = ParseIntOption(optarg);
        break;

#if defined(TRTIS_ENABLE_HTTP) || defined(TRTIS_ENABLE_HTTP_V2)
      case OPTION_ALLOW_HTTP:
        allow_http_ = ParseBoolOption(optarg);
        break;
      case OPTION_HTTP_PORT:
        http_port = ParseIntOption(optarg);
        http_health_port = http_port;
        break;
      case OPTION_HTTP_HEALTH_PORT:
        http_health_port = ParseIntOption(optarg);
        break;
      case OPTION_HTTP_THREAD_COUNT:
        http_thread_cnt = ParseIntOption(optarg);
        break;
#endif  // TRTIS_ENABLE_HTTP

#if defined(TRTIS_ENABLE_GRPC) || defined(TRTIS_ENABLE_GRPC_V2)
      case OPTION_ALLOW_GRPC:
        allow_grpc_ = ParseBoolOption(optarg);
        break;
      case OPTION_GRPC_PORT:
        grpc_port = ParseIntOption(optarg);
        break;
      case OPTION_GRPC_INFER_THREAD_COUNT:
        grpc_infer_thread_cnt = ParseIntOption(optarg);
        break;
      case OPTION_GRPC_STREAM_INFER_THREAD_COUNT:
        grpc_stream_infer_thread_cnt = ParseIntOption(optarg);
        break;
      case OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE:
        grpc_infer_allocation_pool_size = ParseIntOption(optarg);
        break;
#endif  // TRTIS_ENABLE_GRPC || TRTIS_ENABLE_GRPC_V2

#ifdef TRTIS_ENABLE_METRICS
      case OPTION_ALLOW_METRICS:
        allow_metrics_ = ParseBoolOption(optarg);
        break;
      case OPTION_ALLOW_GPU_METRICS:
        allow_gpu_metrics = ParseBoolOption(optarg);
        break;
      case OPTION_METRICS_PORT:
        metrics_port = ParseIntOption(optarg);
        break;
#endif  // TRTIS_ENABLE_METRICS

#ifdef TRTIS_ENABLE_TRACING
      case OPTION_TRACE_FILEPATH:
        trace_filepath = optarg;
        break;
      case OPTION_TRACE_LEVEL:
        trace_level = ParseTraceLevelOption(optarg);
        break;
      case OPTION_TRACE_RATE:
        trace_rate = ParseIntOption(optarg);
        break;
#endif  // TRTIS_ENABLE_TRACING

      case OPTION_ALLOW_POLL_REPO:
        allow_poll_model_repository = ParseBoolOption(optarg);
        deprecated_control_mode_set = true;
        break;
      case OPTION_POLL_REPO_SECS:
        repository_poll_secs = ParseIntOption(optarg);
        break;
      case OPTION_ALLOW_MODEL_CONTROL:
        allow_model_control = ParseBoolOption(optarg);
        deprecated_control_mode_set = true;
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
        control_mode_set = true;
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

      case OPTION_TF_ALLOW_SOFT_PLACEMENT:
        tf_allow_soft_placement = ParseBoolOption(optarg);
        break;
      case OPTION_TF_GPU_MEMORY_FRACTION:
        tf_gpu_memory_fraction = ParseFloatOption(optarg);
        break;
      case OPTION_TF_ADD_VGPU:
        tf_vgpus.push_back(ParseVGPUOption(optarg));
        break;
    }
  }

  if (optind < argc) {
    std::cerr << "Unexpected argument: " << argv[optind] << std::endl;
    std::cerr << Usage() << std::endl;
    return false;
  }

#ifdef TRTIS_ENABLE_LOGGING
  // Initialize our own logging instance since it is used by GRPC and
  // HTTP endpoints. This logging instance is separate from the one in
  // libtritonserver so we must initialize explicitly.
  LOG_ENABLE_INFO(log_info);
  LOG_ENABLE_WARNING(log_warn);
  LOG_ENABLE_ERROR(log_error);
  LOG_SET_VERBOSE(log_verbose);
#endif  // TRTIS_ENABLE_LOGGING

  repository_poll_secs_ =
      (allow_poll_model_repository) ? std::max(0, repository_poll_secs) : 0;

  if (control_mode_set && deprecated_control_mode_set) {
    std::cerr << "--allow-model-control or --allow-poll-model-repository "
              << "can not be specified if --model-control-mode is specified"
              << std::endl;
    std::cerr << Usage() << std::endl;
  }

  if (!control_mode_set) {
    if (allow_model_control && allow_poll_model_repository) {
      std::cerr << "--allow-model-control and --allow-poll-model-repository "
                << "can not be both set to true" << std::endl;
      std::cerr << Usage() << std::endl;
      return false;
    }

    if (allow_model_control) {
      control_mode = TRITONSERVER_MODEL_CONTROL_EXPLICIT;
    } else if (repository_poll_secs_ > 0) {
      control_mode = TRITONSERVER_MODEL_CONTROL_POLL;
    } else {
      control_mode = TRITONSERVER_MODEL_CONTROL_NONE;
    }
  } else {
    // May sure main() will not try to invoke polling periodically
    if (control_mode != TRITONSERVER_MODEL_CONTROL_POLL) {
      repository_poll_secs_ = 0;
    }
  }

  api_version_ = api_version;

#if defined(TRTIS_ENABLE_HTTP) || defined(TRTIS_ENABLE_HTTP_V2)
  http_port_ = http_port;
  http_health_port_ = http_health_port;
  http_thread_cnt_ = http_thread_cnt;
  if (api_version_ == 2) {
    http_ports_ = {http_health_port_, http_port_};
  } else {
    http_ports_ = {http_port_, http_health_port_, http_port_,
                   http_port_, http_port_,        http_port_};
  }
#endif  // TRTIS_ENABLE_HTTP || TRTIS_ENABLE_HTTP_V2

#if defined(TRTIS_ENABLE_GRPC) || defined(TRTIS_ENABLE_GRPC_V2)
  grpc_port_ = grpc_port;
  grpc_infer_thread_cnt_ = grpc_infer_thread_cnt;
  grpc_stream_infer_thread_cnt_ = grpc_stream_infer_thread_cnt;
  grpc_infer_allocation_pool_size_ = grpc_infer_allocation_pool_size;
#endif  // TRTIS_ENABLE_GRPC || TRTIS_ENABLE_GRPC_V2

#ifdef TRTIS_ENABLE_METRICS
  metrics_port_ = allow_metrics_ ? metrics_port : -1;
  allow_gpu_metrics = allow_metrics_ ? allow_gpu_metrics : false;
#endif  // TRTIS_ENABLE_METRICS

#ifdef TRTIS_ENABLE_TRACING
  trace_filepath_ = trace_filepath;
  trace_level_ = trace_level;
  trace_rate_ = trace_rate;
#endif  // TRTIS_ENABLE_TRACING

  // Check if HTTP, GRPC and metrics port clash
  if (CheckPortCollision())
    return false;

  // Highly duplicated code conditioned by api_version,
  // just remove the api_version 1 case when dropping TRTSERVER API
  if (api_version == 1) {
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsNew(
            reinterpret_cast<TRTSERVER_ServerOptions**>(server_options)),
        "creating server options");
    auto loptions = reinterpret_cast<TRTSERVER_ServerOptions*>(*server_options);

    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetServerId(loptions, server_id.c_str()),
        "setting server ID");
#if defined(TRTIS_ENABLE_GRPC) || defined(TRTIS_ENABLE_GRPC_V2)
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetServerProtocolVersion(loptions, api_version),
        "setting server protocol version");
#endif  // TRTIS_ENABLE_GRPC || TRTIS_ENABLE_GRPC_V2
    for (const auto& model_repository_path : model_repository_paths) {
      FAIL_IF_ERR(
          TRTSERVER_ServerOptionsSetModelRepositoryPath(
              loptions, model_repository_path.c_str()),
          "setting model repository path");
    }
    TRTSERVER_Model_Control_Mode trt_control_mode;
    switch (control_mode) {
      case TRITONSERVER_MODEL_CONTROL_EXPLICIT:
        trt_control_mode = TRTSERVER_MODEL_CONTROL_EXPLICIT;
        break;
      case TRITONSERVER_MODEL_CONTROL_POLL:
        trt_control_mode = TRTSERVER_MODEL_CONTROL_POLL;
        break;
      case TRITONSERVER_MODEL_CONTROL_NONE:
      default:
        trt_control_mode = TRTSERVER_MODEL_CONTROL_NONE;
        break;
    }
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetModelControlMode(loptions, trt_control_mode),
        "setting model control mode");
    for (const auto& model : startup_models_) {
      FAIL_IF_ERR(
          TRTSERVER_ServerOptionsSetStartupModel(loptions, model.c_str()),
          "setting startup model");
    }
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
            loptions, pinned_memory_pool_byte_size),
        "setting total pinned memory byte size");
    for (const auto& cuda_pool : cuda_pools) {
      FAIL_IF_ERR(
          TRTSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
              loptions, cuda_pool.first, cuda_pool.second),
          "setting total CUDA memory byte size");
    }
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetMinSupportedComputeCapability(
            loptions, min_supported_compute_capability),
        "setting minimum supported CUDA compute capability");
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetExitOnError(loptions, exit_on_error),
        "setting exit on error");
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetStrictModelConfig(
            loptions, strict_model_config),
        "setting strict model configuration");
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetStrictReadiness(loptions, strict_readiness),
        "setting strict readiness");
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetExitTimeout(
            loptions, std::max(0, exit_timeout_secs)),
        "setting exit timeout");

#ifdef TRTIS_ENABLE_LOGGING
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetLogInfo(loptions, log_info),
        "setting log info enable");
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetLogWarn(loptions, log_warn),
        "setting log warn enable");
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetLogError(loptions, log_error),
        "setting log error enable");
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetLogVerbose(loptions, log_verbose),
        "setting log verbose level");
#endif  // TRTIS_ENABLE_LOGGING

#ifdef TRTIS_ENABLE_METRICS
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetMetrics(loptions, allow_metrics_),
        "setting metrics enable");
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetGpuMetrics(loptions, allow_gpu_metrics),
        "setting GPU metrics enable");
#endif  // TRTIS_ENABLE_METRICS

    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetTensorFlowSoftPlacement(
            loptions, tf_allow_soft_placement),
        "setting tensorflow soft placement");
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsSetTensorFlowGpuMemoryFraction(
            loptions, tf_gpu_memory_fraction),
        "setting tensorflow GPU memory fraction");
    for (const auto& tf_vgpu : tf_vgpus) {
      FAIL_IF_ERR(
          TRTSERVER_ServerOptionsAddTensorFlowVgpuMemoryLimits(
              loptions, tf_vgpu.gpu_device_, tf_vgpu.num_vgpus_,
              tf_vgpu.mem_limit_mbytes_),
          "adding tensorflow VGPU instances");
    }
  } else {
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsNew(server_options),
        "creating server options");
    auto loptions = *server_options;
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetServerId(loptions, server_id.c_str()),
        "setting server ID");
    for (const auto& model_repository_path : model_repository_paths) {
      FAIL_IF_TRITON_ERR(
          TRITONSERVER_ServerOptionsSetModelRepositoryPath(
              loptions, model_repository_path.c_str()),
          "setting model repository path");
    }
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetModelControlMode(loptions, control_mode),
        "setting model control mode");
    for (const auto& model : startup_models_) {
      FAIL_IF_TRITON_ERR(
          TRITONSERVER_ServerOptionsSetStartupModel(loptions, model.c_str()),
          "setting startup model");
    }
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
            loptions, pinned_memory_pool_byte_size),
        "setting total pinned memory byte size");
    for (const auto& cuda_pool : cuda_pools) {
      FAIL_IF_TRITON_ERR(
          TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
              loptions, cuda_pool.first, cuda_pool.second),
          "setting total CUDA memory byte size");
    }
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
            loptions, min_supported_compute_capability),
        "setting minimum supported CUDA compute capability");
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetExitOnError(loptions, exit_on_error),
        "setting exit on error");
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetStrictModelConfig(
            loptions, strict_model_config),
        "setting strict model configuration");
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetStrictReadiness(
            loptions, strict_readiness),
        "setting strict readiness");
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetExitTimeout(
            loptions, std::max(0, exit_timeout_secs)),
        "setting exit timeout");

#ifdef TRTIS_ENABLE_LOGGING
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetLogInfo(loptions, log_info),
        "setting log info enable");
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetLogWarn(loptions, log_warn),
        "setting log warn enable");
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetLogError(loptions, log_error),
        "setting log error enable");
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetLogVerbose(loptions, log_verbose),
        "setting log verbose level");
#endif  // TRTIS_ENABLE_LOGGING

#ifdef TRTIS_ENABLE_METRICS
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetMetrics(loptions, allow_metrics_),
        "setting metrics enable");
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetGpuMetrics(loptions, allow_gpu_metrics),
        "setting GPU metrics enable");
#endif  // TRTIS_ENABLE_METRICS

    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetTensorFlowSoftPlacement(
            loptions, tf_allow_soft_placement),
        "setting tensorflow soft placement");
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsSetTensorFlowGpuMemoryFraction(
            loptions, tf_gpu_memory_fraction),
        "setting tensorflow GPU memory fraction");
    for (const auto& tf_vgpu : tf_vgpus) {
      FAIL_IF_TRITON_ERR(
          TRITONSERVER_ServerOptionsAddTensorFlowVgpuMemoryLimits(
              loptions, tf_vgpu.gpu_device_, tf_vgpu.num_vgpus_,
              tf_vgpu.mem_limit_mbytes_),
          "adding tensorflow VGPU instances");
    }
  }
  return true;
}
}  // namespace

int
main(int argc, char** argv)
{
  // FIXMEV2 currently casting server_ptr / server_options for corresponding
  // api version: TRTSERVER_XXX for v1, TRITONSERVER_XXX for v2
  // Parse command-line to create the options for the inference
  // server.
  TRITONSERVER_ServerOptions* server_options = nullptr;
  if (!Parse(&server_options, argc, argv)) {
    exit(1);
  }

  // Trace manager.
  std::shared_ptr<nvidia::inferenceserver::TraceManager> trace_manager;

  // Manager for shared memory blocks.
  auto shm_manager =
      std::make_shared<nvidia::inferenceserver::SharedMemoryManager>();

  // Create the server...
  TRITONSERVER_Server* server_ptr = nullptr;
  if (api_version_ == 1) {
    FAIL_IF_ERR(
        TRTSERVER_ServerNew(
            reinterpret_cast<TRTSERVER_Server**>(&server_ptr),
            reinterpret_cast<TRTSERVER_ServerOptions*>(server_options)),
        "creating server");
    FAIL_IF_ERR(
        TRTSERVER_ServerOptionsDelete(
            reinterpret_cast<TRTSERVER_ServerOptions*>(server_options)),
        "deleting server options");
  } else {
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerNew(&server_ptr, server_options), "creating server");
    FAIL_IF_TRITON_ERR(
        TRITONSERVER_ServerOptionsDelete(server_options),
        "deleting server options");
  }

  // FIXMEV2 for the Server object APIs, just use TRITONSERVER_ServerXXX as
  // we know the underlying object is the same.
  std::shared_ptr<TRITONSERVER_Server> server(
      (api_version_ == 1 ? nullptr : server_ptr), TRITONSERVER_ServerDelete);
  // FIXMEV2 Create a TRTSERVER_Server shared pointer for V1 server frontends,
  // ideally the frontends only needs the raw pointer of the server as we can
  // ensure the server object lives longer than the frontend objects.
  std::shared_ptr<TRTSERVER_Server> trt_server(
      (api_version_ == 1 ? reinterpret_cast<TRTSERVER_Server*>(server_ptr)
                         : nullptr),
      TRTSERVER_ServerDelete);

  // Configure and start tracing if specified on the command line.
  if (!StartTracing(&trace_manager)) {
    exit(1);
  }

  // Start the HTTP, GRPC, and metrics endpoints.
  if (!StartEndpoints(server, trt_server, trace_manager, shm_manager)) {
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
  const int ret = (stop_err != nullptr) ? 1 : 0;
  LOG_TRITONSERVER_ERROR(stop_err, "failed to stop server");

  // Stop tracing and the HTTP, GRPC, and metrics endpoints.
  StopTracing(trace_manager);
  StopEndpoints();

#ifdef TRTIS_ENABLE_ASAN
  // Can invoke ASAN before exit though this is typically not very
  // useful since there are many objects that are not yet destructed.
  //  __lsan_do_leak_check();
#endif  // TRTIS_ENABLE_ASAN

  // FIXME. TF backend aborts if we attempt cleanup...
  if (api_version_ == 1) {
    std::shared_ptr<TRTSERVER_Server>* keep_alive =
        new std::shared_ptr<TRTSERVER_Server>(trt_server);
    if (keep_alive == nullptr) {
      return 1;
    }
  } else {
    std::shared_ptr<TRITONSERVER_Server>* keep_alive =
        new std::shared_ptr<TRITONSERVER_Server>(server);
    if (keep_alive == nullptr) {
      return 1;
    }
  }

  return ret;
}
