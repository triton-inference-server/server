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
#include <list>
#include <set>
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
#include "grpc_server.h"
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_GPU
static_assert(
    TRITON_MIN_COMPUTE_CAPABILITY >= 1.0,
    "Invalid TRITON_MIN_COMPUTE_CAPABILITY specified");
#endif  // TRITON_ENABLE_GPU

namespace {

#ifdef TRITON_ENABLE_HTTP
std::unique_ptr<triton::server::HTTPServer> http_service_;
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_GRPC
// [FIXME] global variable should use different naming convention "g_xxx"
std::unique_ptr<triton::server::grpc::Server> grpc_service_;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
std::unique_ptr<triton::server::HTTPServer> metrics_service_;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_SAGEMAKER
std::unique_ptr<triton::server::HTTPServer> sagemaker_service_;
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
std::unique_ptr<triton::server::HTTPServer> vertex_ai_service_;
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
      server, trace_manager, shm_manager, g_triton_server_parameters, service);
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
      server, trace_manager, shm_manager, http_port_, reuse_http_port_,
      http_address_, http_thread_cnt_, service);
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
      server, metrics_port_, http_address_, 1 /* HTTP thread count */, service);
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
      server, trace_manager, shm_manager, sagemaker_port_, sagemaker_address_,
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
    std::unique_ptr<triton::server::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager)
{
  TRITONSERVER_Error* err = triton::server::VertexAiAPIServer::Create(
      server, trace_manager, shm_manager, vertex_ai_port_, vertex_ai_address_,
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
StartTracing(triton::server::TraceManager** trace_manager)
{
  *trace_manager = nullptr;

#ifdef TRITON_ENABLE_TRACING
  TRITONSERVER_Error* err = triton::server::TraceManager::Create(
      trace_manager, g_triton_params.trace_level_, g_triton_params.trace_rate_,
      g_triton_params.trace_count_, g_triton_params.trace_log_frequency_,
      g_triton_params.trace_filepath_);

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

template <>
double
ParseOption(const std::string& arg)
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
  if ((arg == "true") || (arg == "on") || (arg == "min") || (arg == "max") ||
      (arg == "timestamps")) {
    return TRITONSERVER_TRACE_LEVEL_TIMESTAMPS;
  }
  if (arg == "tensors") {
    return TRITONSERVER_TRACE_LEVEL_TENSORS;
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
  // Format is "<backend_name>,<setting>=<value>" for specific
  // config/settings and "<setting>=<value>" for backend agnostic
  // configs/settings
  int delim_name = arg.find(",");
  int delim_setting = arg.find("=", delim_name + 1);

  std::string name_string = std::string();
  if (delim_name > 0) {
    name_string = arg.substr(0, delim_name);
  } else if (delim_name == 0) {
    std::cerr << "No backend specified. --backend-config option format is "
              << "<backend name>,<setting>=<value> or "
              << "<setting>=<value>. Got " << arg << std::endl;
    exit(1);
  }  // else global backend config

  if (delim_setting < 0) {
    std::cerr << "--backend-config option format is '<backend "
                 "name>,<setting>=<value>'. Got "
              << arg << std::endl;
    exit(1);
  }
  std::string setting_string =
      arg.substr(delim_name + 1, delim_setting - delim_name - 1);
  std::string value_string = arg.substr(delim_setting + 1);

  if (setting_string.empty() || value_string.empty()) {
    std::cerr << "--backend-config option format is '<backend "
                 "name>,<setting>=<value>'. Got "
              << arg << std::endl;
    exit(1);
  }

  return {name_string, setting_string, value_string};
}

std::string
PairsToJsonStr(std::vector<std::pair<std::string, std::string>> settings)
{
  triton::common::TritonJson::Value json(
      triton::common::TritonJson::ValueType::OBJECT);
  for (const auto& setting : settings) {
    const auto& key = setting.first;
    const auto& value = setting.second;
    json.SetStringObject(key.c_str(), value);
  }
  triton::common::TritonJson::WriteBuffer buffer;
  auto err = json.Write(&buffer);
  if (err != nullptr) {
    LOG_TRITONSERVER_ERROR(err, "failed to convert config to JSON");
  }
  return buffer.Contents();
}

std::tuple<std::string, std::string, std::string>
ParseCacheConfigOption(const std::string arg)
{
  // Format is "<cache_name>,<setting>=<value>" for specific
  // config/settings and "<setting>=<value>" for cache agnostic
  // configs/settings
  int delim_name = arg.find(",");
  int delim_setting = arg.find("=", delim_name + 1);

  std::string name_string = std::string();
  if (delim_name > 0) {
    name_string = arg.substr(0, delim_name);
  }
  // No cache-agnostic global settings are currently supported
  else {
    std::cerr << "No cache specified. --cache-config option format is "
              << "<cache name>,<setting>=<value>. Got " << arg << std::endl;
    exit(1);
  }

  if (delim_setting < 0) {
    std::cerr << "--cache-config option format is '<cache "
                 "name>,<setting>=<value>'. Got "
              << arg << std::endl;
    exit(1);
  }
  std::string setting_string =
      arg.substr(delim_name + 1, delim_setting - delim_name - 1);
  std::string value_string = arg.substr(delim_setting + 1);

  if (setting_string.empty() || value_string.empty()) {
    std::cerr << "--cache-config option format is '<cache "
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
#if defined(TRITON_ENABLE_HTTP)
  int32_t http_port = http_port_;
  bool reuse_http_port = reuse_http_port_;
  std::string http_address = http_address_;
  int32_t http_thread_cnt = http_thread_cnt_;
#endif  // TRITON_ENABLE_HTTP

#if defined(TRITON_ENABLE_GRPC)
  triton::server::grpc::Options lgrpc_options;
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
    auto aip_mode =
        triton::server::GetEnvironmentVariableOrDefault("AIP_MODE", "");
    // Enable Vertex AI service and disable HTTP / GRPC service by default
    // if detecting Vertex AI environment
    if (aip_mode == "PREDICTION") {
      allow_vertex_ai_ = true;
#ifdef TRITON_ENABLE_HTTP
      allow_http_ = false;
#endif  // TRITON_ENABLE_HTTP
#ifdef TRITON_ENABLE_GRPC
      allow_grpc_ = false;
#endif  // TRITON_ENABLE_GRPC
    }
    auto port = triton::server::GetEnvironmentVariableOrDefault(
        "AIP_HTTP_PORT", "8080");
    vertex_ai_port_ = ParseIntOption(port);
  }
  int32_t vertex_ai_port = vertex_ai_port_;
  int32_t vertex_ai_thread_cnt = vertex_ai_thread_cnt_;
  std::string vertex_ai_default_model = vertex_ai_default_model_;
#endif  // TRITON_ENABLE_VERTEX_AI

#ifdef TRITON_ENABLE_METRICS
  int32_t metrics_port = metrics_port_;
  float metrics_interval_ms = metrics_interval_ms_;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_TRACING
  std::string trace_filepath = trace_filepath_;
  std::vector<TRITONSERVER_InferenceTraceLevel> trace_level_settings = {
      trace_level_};
  int32_t trace_rate = trace_rate_;
  int32_t trace_count = trace_count_;
  int32_t trace_log_frequency = trace_log_frequency_;
#endif  // TRITON_ENABLE_TRACING

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
      case OPTION_LOG_FORMAT: {
        std::string format_str(optarg);
        if (format_str == "default") {
          log_format = triton::common::Logger::Format::kDEFAULT;
        } else if (format_str == "ISO8601") {
          log_format = triton::common::Logger::Format::kISO8601;
        } else {
          std::cerr << "invalid argument for --log-format" << std::endl;
          std::cerr << Usage() << std::endl;
          return false;
        }
        break;
      }
      case OPTION_LOG_FILE:
        log_file = optarg;
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
      case OPTION_DISABLE_AUTO_COMPLETE_CONFIG:
        disable_auto_complete_config = true;
        break;
      case OPTION_STRICT_MODEL_CONFIG:
        std::cerr << "Warning: '--strict-model-config' has been deprecated! "
                     "Please use '--disable-auto-complete-config' instead."
                  << std::endl;
        strict_model_config_present = true;
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
      case OPTION_REUSE_HTTP_PORT:
        reuse_http_port = ParseIntOption(optarg);
        break;
      case OPTION_HTTP_ADDRESS:
        http_address = optarg;
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
        lgrpc_options.socket_.port_ = ParseIntOption(optarg);
        break;
      case OPTION_REUSE_GRPC_PORT:
        lgrpc_options.socket_.reuse_port_ = ParseIntOption(optarg);
        break;
      case OPTION_GRPC_ADDRESS:
        lgrpc_options.socket_.address_ = optarg;
        break;
      case OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE:
        lgrpc_options.infer_allocation_pool_size_ = ParseIntOption(optarg);
        break;
      case OPTION_GRPC_USE_SSL:
        lgrpc_options.ssl_.use_ssl_ = ParseBoolOption(optarg);
        break;
      case OPTION_GRPC_USE_SSL_MUTUAL:
        lgrpc_options.ssl_.use_mutual_auth_ = ParseBoolOption(optarg);
        // [FIXME] this implies use SSL, take priority over OPTION_GRPC_USE_SSL?
        lgrpc_options.ssl_.use_ssl_ = true;
        break;
      case OPTION_GRPC_SERVER_CERT:
        lgrpc_options.ssl_.server_cert_ = optarg;
        break;
      case OPTION_GRPC_SERVER_KEY:
        lgrpc_options.ssl_.server_key_ = optarg;
        break;
      case OPTION_GRPC_ROOT_CERT:
        lgrpc_options.ssl_.root_cert_ = optarg;
        break;
      case OPTION_GRPC_RESPONSE_COMPRESSION_LEVEL: {
        std::string mode_str(optarg);
        std::transform(
            mode_str.begin(), mode_str.end(), mode_str.begin(), ::tolower);
        if (mode_str == "none") {
          lgrpc_options.infer_compression_level_ = GRPC_COMPRESS_LEVEL_NONE;
        } else if (mode_str == "low") {
          lgrpc_options.infer_compression_level_ = GRPC_COMPRESS_LEVEL_LOW;
        } else if (mode_str == "medium") {
          lgrpc_options.infer_compression_level_ = GRPC_COMPRESS_LEVEL_MED;
        } else if (mode_str == "high") {
          lgrpc_options.infer_compression_level_ = GRPC_COMPRESS_LEVEL_HIGH;
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
        lgrpc_options.keep_alive_.keepalive_time_ms_ = ParseIntOption(optarg);
        break;
      case OPTION_GRPC_ARG_KEEPALIVE_TIMEOUT_MS:
        lgrpc_options.keep_alive_.keepalive_timeout_ms_ =
            ParseIntOption(optarg);
        break;
      case OPTION_GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS:
        lgrpc_options.keep_alive_.keepalive_permit_without_calls_ =
            ParseBoolOption(optarg);
        break;
      case OPTION_GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA:
        lgrpc_options.keep_alive_.http2_max_pings_without_data_ =
            ParseIntOption(optarg);
        break;
      case OPTION_GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS:
        lgrpc_options.keep_alive_
            .http2_min_recv_ping_interval_without_data_ms_ =
            ParseIntOption(optarg);
        break;
      case OPTION_GRPC_ARG_HTTP2_MAX_PING_STRIKES:
        lgrpc_options.keep_alive_.http2_max_ping_strikes_ =
            ParseIntOption(optarg);
        break;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
      case OPTION_ALLOW_METRICS:
        allow_metrics_ = ParseBoolOption(optarg);
        break;
      case OPTION_ALLOW_GPU_METRICS:
        allow_gpu_metrics = ParseBoolOption(optarg);
        break;
      case OPTION_ALLOW_CPU_METRICS:
        allow_cpu_metrics = ParseBoolOption(optarg);
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
        trace_level_settings.push_back(ParseTraceLevelOption(optarg));
        break;
      case OPTION_TRACE_RATE:
        trace_rate = ParseIntOption(optarg);
        break;
      case OPTION_TRACE_COUNT:
        trace_count = ParseIntOption(optarg);
        break;
      case OPTION_TRACE_LOG_FREQUENCY:
        trace_log_frequency = ParseIntOption(optarg);
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
      case OPTION_RESPONSE_CACHE_BYTE_SIZE: {
        cache_size_present = true;
        const auto byte_size = std::to_string(ParseLongLongOption(optarg));
        cache_config_settings["local"] = {{"size", byte_size}};
        std::cerr
            << "Warning: '--response-cache-byte-size' has been deprecated! "
               "This will default to the 'local' cache implementation with "
               "the provided byte size for its config. Please use "
               "'--cache-config' instead. The equivalent "
               "--cache-config CLI args would be: "
               "'--cache-config=local,size=" +
                   byte_size + "'"
            << std::endl;
        break;
      }
      case OPTION_CACHE_CONFIG: {
        cache_config_present = true;
        const auto cache_setting = ParseCacheConfigOption(optarg);
        const auto& cache_name = std::get<0>(cache_setting);
        const auto& key = std::get<1>(cache_setting);
        const auto& value = std::get<2>(cache_setting);
        cache_config_settings[cache_name].push_back({key, value});
        break;
      }
      case OPTION_CACHE_DIR:
        cache_dir = optarg;
        break;
      case OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY:
        min_supported_compute_capability = ParseOption<double>(optarg);
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
      case OPTION_MODEL_LOAD_THREAD_COUNT:
        model_load_thread_count = ParseIntOption(optarg);
        break;
      case OPTION_BACKEND_CONFIG:
        backend_config_settings.push_back(ParseBackendConfigOption(optarg));
        break;
      case OPTION_HOST_POLICY:
        host_policies.push_back(ParseHostPolicyOption(optarg));
        break;
      case OPTION_MODEL_LOAD_GPU_LIMIT:
        load_gpu_limit.emplace(ParsePairOption<int, double>(optarg, ":"));
        break;
      case OPTION_MODEL_NAMESPACING:
        enable_model_namespacing = ParseBoolOption(optarg);
        break;
    }
  }

  if (optind < argc) {
    std::cerr << "Unexpected argument: " << argv[optind] << std::endl;
    std::cerr << Usage() << std::endl;
    return false;
  }

  // [FIXME] should move outside parse, first thing to do after successful
  // parsing
#ifdef TRITON_ENABLE_LOGGING
  // Initialize our own logging instance since it is used by GRPC and
  // HTTP endpoints. This logging instance is separate from the one in
  // libtritonserver so we must initialize explicitly.
  LOG_ENABLE_INFO(log_info);
  LOG_ENABLE_WARNING(log_warn);
  LOG_ENABLE_ERROR(log_error);
  LOG_SET_VERBOSE(log_verbose);
  LOG_SET_FORMAT(log_format);
  LOG_SET_OUT_FILE(log_file);
#endif  // TRITON_ENABLE_LOGGING

  repository_poll_secs_ = 0;
  if (control_mode == TRITONSERVER_MODEL_CONTROL_POLL) {
    repository_poll_secs_ = std::max(0, repository_poll_secs);
  }

#if defined(TRITON_ENABLE_HTTP)
  http_port_ = http_port;
  reuse_http_port_ = reuse_http_port;
  http_address_ = http_address;
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
        triton::server::GetEnvironmentVariableOrDefault("AIP_STORAGE_URI", "");
    if (!aip_storage_uri.empty() && model_repository_paths.empty()) {
      model_repository_paths.insert(aip_storage_uri);
    }
  }
  vertex_ai_port_ = vertex_ai_port;
  vertex_ai_thread_cnt_ = vertex_ai_thread_cnt;
  vertex_ai_default_model_ = vertex_ai_default_model;
#endif  // TRITON_ENABLE_VERTEX_AI


#if defined(TRITON_ENABLE_GRPC)
  g_triton_server_parameters.grpc_options_ = lgrpc_options;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
  metrics_port_ = metrics_port;
  allow_gpu_metrics = allow_metrics_ ? allow_gpu_metrics : false;
  allow_cpu_metrics = allow_metrics_ ? allow_cpu_metrics : false;
  metrics_interval_ms_ = metrics_interval_ms;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_TRACING
  trace_filepath_ = trace_filepath;
  for (auto& trace_level : trace_level_settings) {
    trace_level_ = static_cast<TRITONSERVER_InferenceTraceLevel>(
        trace_level_ | trace_level);
  }
  trace_rate_ = trace_rate;
  trace_count_ = trace_count;
  trace_log_frequency_ = trace_log_frequency;
#endif  // TRITON_ENABLE_TRACING

  // Check if HTTP, GRPC and metrics port clash
  if (CheckPortCollision()) {
    return false;
  }

  // Check if there is a conflict between --disable-auto-complete-config
  // and --strict-model-config
  if (disable_auto_complete_config) {
    if (strict_model_config_present && !strict_model_config) {
      std::cerr
          << "Warning: Overriding deprecated '--strict-model-config' from "
             "False to True in favor of '--disable-auto-complete-config'!"
          << std::endl;
    }
    strict_model_config = true;
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
  triton::server::TraceManager* trace_manager;

  // Manager for shared memory blocks.
  auto shm_manager = std::make_shared<triton::server::SharedMemoryManager>();

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
    if (repository_poll_secs_ > 0) {
      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_ServerPollModelRepository(server_ptr),
          "failed to poll model repository");
    }

    // Wait for the polling interval (or a long time if polling is not
    // enabled). Will be woken if the server is exiting.
    std::unique_lock<std::mutex> lock(triton::server::signal_exit_mu_);
    std::chrono::seconds wait_timeout(
        (repository_poll_secs_ == 0) ? 3600 : repository_poll_secs_);
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
