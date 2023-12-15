// Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <sstream>
#include <thread>

#include "triton_signal.h"

#ifdef TRITON_ENABLE_ASAN
#include <sanitizer/lsan_interface.h>
#endif  // TRITON_ENABLE_ASAN

#include "command_line_parser.h"
#include "common.h"
#include "shared_memory_manager.h"
#include "tracer.h"
#include "triton/common/logging.h"
#include "triton/core/tritonserver.h"

#if defined(TRITON_ENABLE_HTTP) || defined(TRITON_ENABLE_METRICS)
#include "http_server.h"
#endif  // TRITON_ENABLE_HTTP|| TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_SAGEMAKER
#include "sagemaker_server.h"
#endif  // TRITON_ENABLE_SAGEMAKER
#ifdef TRITON_ENABLE_VERTEX_AI
#include "vertex_ai_server.h"
#endif  // TRITON_ENABLE_VERTEX_AI
#ifdef TRITON_ENABLE_GRPC
#include "grpc/grpc_server.h"
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_GPU
static_assert(
    TRITON_MIN_COMPUTE_CAPABILITY >= 1.0,
    "Invalid TRITON_MIN_COMPUTE_CAPABILITY specified");
#endif  // TRITON_ENABLE_GPU

namespace {

#ifdef TRITON_ENABLE_HTTP
std::unique_ptr<triton::server::HTTPServer> g_http_service;
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_GRPC
std::unique_ptr<triton::server::grpc::Server> g_grpc_service;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
std::unique_ptr<triton::server::HTTPServer> g_metrics_service;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_SAGEMAKER
std::unique_ptr<triton::server::HTTPServer> g_sagemaker_service;
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
std::unique_ptr<triton::server::HTTPServer> g_vertex_ai_service;
#endif  // TRITON_ENABLE_VERTEX_AI

triton::server::TritonServerParameters g_triton_params;

#ifdef TRITON_ENABLE_GRPC
TRITONSERVER_Error*
StartGrpcService(
    std::unique_ptr<triton::server::grpc::Server>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager)
{
  TRITONSERVER_Error* err = triton::server::grpc::Server::Create(
      server, trace_manager, shm_manager, g_triton_params.grpc_options_,
      service);
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
    std::unique_ptr<triton::server::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager)
{
  TRITONSERVER_Error* err = triton::server::HTTPAPIServer::Create(
      server, trace_manager, shm_manager, g_triton_params.http_port_,
      g_triton_params.reuse_http_port_, g_triton_params.http_address_,
      g_triton_params.http_forward_header_pattern_,
      g_triton_params.http_thread_cnt_, g_triton_params.http_restricted_apis_,
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
    std::unique_ptr<triton::server::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server)
{
  TRITONSERVER_Error* err = triton::server::HTTPMetricsServer::Create(
      server, g_triton_params.metrics_port_, g_triton_params.metrics_address_,
      1 /* HTTP thread count */, service);
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
    std::unique_ptr<triton::server::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager)
{
  TRITONSERVER_Error* err = triton::server::SagemakerAPIServer::Create(
      server, trace_manager, shm_manager, g_triton_params.sagemaker_port_,
      g_triton_params.sagemaker_address_, g_triton_params.sagemaker_thread_cnt_,
      service);
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
    std::unique_ptr<triton::server::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager)
{
  TRITONSERVER_Error* err = triton::server::VertexAiAPIServer::Create(
      server, trace_manager, shm_manager, g_triton_params.vertex_ai_port_,
      g_triton_params.vertex_ai_address_, g_triton_params.vertex_ai_thread_cnt_,
      g_triton_params.vertex_ai_default_model_, service);
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
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager)
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
  if (g_triton_params.allow_grpc_) {
    TRITONSERVER_Error* err =
        StartGrpcService(&g_grpc_service, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start GRPC service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_HTTP
  // Enable HTTP endpoints if requested...
  if (g_triton_params.allow_http_) {
    TRITONSERVER_Error* err =
        StartHttpService(&g_http_service, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start HTTP service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_HTTP


#ifdef TRITON_ENABLE_SAGEMAKER
  // Enable Sagemaker endpoints if requested...
  if (g_triton_params.allow_sagemaker_) {
    TRITONSERVER_Error* err = StartSagemakerService(
        &g_sagemaker_service, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start Sagemaker service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
  // Enable Vertex AI endpoints if requested...
  if (g_triton_params.allow_vertex_ai_) {
    TRITONSERVER_Error* err = StartVertexAiService(
        &g_vertex_ai_service, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start Vertex AI service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_VERTEX_AI

#ifdef TRITON_ENABLE_METRICS
  // Enable metrics endpoint if requested...
  if (g_triton_params.allow_metrics_) {
    TRITONSERVER_Error* err = StartMetricsService(&g_metrics_service, server);
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
  if (g_http_service) {
    TRITONSERVER_Error* err = g_http_service->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop HTTP service");
      ret = false;
    }

    g_http_service.reset();
  }
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_GRPC
  if (g_grpc_service) {
    TRITONSERVER_Error* err = g_grpc_service->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop GRPC service");
      ret = false;
    }

    g_grpc_service.reset();
  }
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
  if (g_metrics_service) {
    TRITONSERVER_Error* err = g_metrics_service->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop Metrics service");
      ret = false;
    }

    g_metrics_service.reset();
  }
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_SAGEMAKER
  if (g_sagemaker_service) {
    TRITONSERVER_Error* err = g_sagemaker_service->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop Sagemaker service");
      ret = false;
    }

    g_sagemaker_service.reset();
  }
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
  if (g_vertex_ai_service) {
    TRITONSERVER_Error* err = g_vertex_ai_service->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop Vertex AI service");
      ret = false;
    }

    g_vertex_ai_service.reset();
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
StartTracing(triton::server::TraceManager** trace_manager)
{
  *trace_manager = nullptr;

#ifdef TRITON_ENABLE_TRACING
  TRITONSERVER_Error* err = triton::server::TraceManager::Create(
      trace_manager, g_triton_params.trace_level_, g_triton_params.trace_rate_,
      g_triton_params.trace_count_, g_triton_params.trace_log_frequency_,
      g_triton_params.trace_filepath_, g_triton_params.trace_mode_,
      g_triton_params.trace_config_map_);

  if (err != nullptr) {
    LOG_TRITONSERVER_ERROR(err, "failed to configure tracing");
    if (*trace_manager != nullptr) {
      delete (*trace_manager);
    }
    *trace_manager = nullptr;
    return false;
  }
#endif  // TRITON_ENABLE_TRACING

  return true;
}

bool
StopTracing(triton::server::TraceManager** trace_manager)
{
#ifdef TRITON_ENABLE_TRACING
  // We assume that at this point Triton has been stopped gracefully,
  // so can delete the trace manager to finalize the output.
  delete (*trace_manager);
  *trace_manager = nullptr;
#endif  // TRITON_ENABLE_TRACING

  return true;
}

}  // namespace

int
main(int argc, char** argv)
{
  // Parse command-line to create the options for the inference
  // server.
  triton::server::TritonParser tp;
  try {
    auto res = tp.Parse(argc, argv);
    g_triton_params = res.first;
    g_triton_params.CheckPortCollision();
  }
  catch (const triton::server::ParseException& pe) {
    std::cerr << pe.what() << std::endl;
    std::cerr << "Usage: tritonserver [options]" << std::endl;
    std::cerr << tp.Usage() << std::endl;
    exit(1);
  }

  triton::server::TritonServerParameters::ManagedTritonServerOptionPtr
      triton_options(nullptr, TRITONSERVER_ServerOptionsDelete);
  try {
    triton_options = g_triton_params.BuildTritonServerOptions();
  }
  catch (const triton::server::ParseException& pe) {
    std::cerr << "Failed to build Triton option:" << std::endl;
    std::cerr << pe.what() << std::endl;
    exit(1);
  }

#ifdef TRITON_ENABLE_LOGGING
  // Initialize our own logging instance since it is used by GRPC and
  // HTTP endpoints. This logging instance is separate from the one in
  // libtritonserver so we must initialize explicitly.
  LOG_ENABLE_INFO(g_triton_params.log_info_);
  LOG_ENABLE_WARNING(g_triton_params.log_warn_);
  LOG_ENABLE_ERROR(g_triton_params.log_error_);
  LOG_SET_VERBOSE(g_triton_params.log_verbose_);
  LOG_SET_FORMAT(g_triton_params.log_format_);
  LOG_SET_OUT_FILE(g_triton_params.log_file_);
#endif  // TRITON_ENABLE_LOGGING

  // Trace manager.
  triton::server::TraceManager* trace_manager;

  // Manager for shared memory blocks.
  auto shm_manager = std::make_shared<triton::server::SharedMemoryManager>();

  // Create the server...
  TRITONSERVER_Server* server_ptr = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ServerNew(&server_ptr, triton_options.get()),
      "creating server");

  std::shared_ptr<TRITONSERVER_Server> server(
      server_ptr, TRITONSERVER_ServerDelete);

  // Configure and start tracing if specified on the command line.
  if (!StartTracing(&trace_manager)) {
    exit(1);
  }

  // Trap SIGINT and SIGTERM to allow server to exit gracefully
  TRITONSERVER_Error* signal_err = triton::server::RegisterSignalHandler();
  if (signal_err != nullptr) {
    LOG_TRITONSERVER_ERROR(signal_err, "failed to register signal handler");
    exit(1);
  }

  // Start the HTTP, GRPC, and metrics endpoints.
  if (!StartEndpoints(server, trace_manager, shm_manager)) {
    exit(1);
  }

  // Wait until a signal terminates the server...
  while (!triton::server::signal_exiting_) {
    // If enabled, poll the model repository to see if there have been
    // any changes.
    if (g_triton_params.repository_poll_secs_ > 0) {
      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_ServerPollModelRepository(server_ptr),
          "failed to poll model repository");
    }

    // Wait for the polling interval (or a long time if polling is not
    // enabled). Will be woken if the server is exiting.
    std::unique_lock<std::mutex> lock(triton::server::signal_exit_mu_);
    std::chrono::seconds wait_timeout(
        (g_triton_params.repository_poll_secs_ == 0)
            ? 3600
            : g_triton_params.repository_poll_secs_);
    triton::server::signal_exit_cv_.wait_for(lock, wait_timeout);
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
