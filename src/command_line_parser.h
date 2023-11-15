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
#pragma once

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "restricted_features.h"
#include "triton/common/logging.h"
#include "triton/core/tritonserver.h"
#ifdef TRITON_ENABLE_GRPC
// To avoid ambiguous reference during build
// grpc headers should be imported first
// https://github.com/open-telemetry/opentelemetry-cpp/blob/main/examples/otlp/README.md#additional-notes-regarding-abseil-library
#include "grpc/grpc_server.h"
#endif  // TRITON_ENABLE_GRPC
#if defined(TRITON_ENABLE_HTTP) || defined(TRITON_ENABLE_METRICS)
#include "http_server.h"
#endif  // TRITON_ENABLE_HTTP || TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_SAGEMAKER
#include "sagemaker_server.h"
#endif  // TRITON_ENABLE_SAGEMAKER
#ifdef TRITON_ENABLE_VERTEX_AI
#include "vertex_ai_server.h"
#endif  // TRITON_ENABLE_VERTEX_AI

#ifndef _WIN32
#include <getopt.h>
#include <unistd.h>
#else
// Minimum implementation of <getopt.h> for Windows
#define required_argument 1
#define no_argument 2
struct option {
  option(const char* name, int has_arg, int* flag, int val)
      : name(name), has_arg(has_arg), flag(flag), val(val)
  {
  }
  const char* name;
  int has_arg;
  int* flag;
  int val;
};
#endif
#ifdef TRITON_ENABLE_TRACING
#include "tracer.h"
#endif


namespace triton { namespace server {

// Command-line options
struct Option {
  static constexpr const char* ArgNone = "";
  static constexpr const char* ArgBool = "boolean";
  static constexpr const char* ArgFloat = "float";
  static constexpr const char* ArgInt = "integer";
  static constexpr const char* ArgStr = "string";

  Option(int id, std::string flag, std::string arg_desc, std::string desc)
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

  const int id_;
  const std::string flag_;
  const std::string arg_desc_;
  const std::string desc_;
};

struct TritonServerParameters {
  std::string server_id_{"triton"};
  bool exit_on_error_{true};
  bool strict_model_config_{false};
  bool strict_readiness_{true};
  int32_t exit_timeout_secs_{30};
#ifdef TRITON_ENABLE_GPU
  double min_supported_compute_capability_{TRITON_MIN_COMPUTE_CAPABILITY};
#else
  double min_supported_compute_capability_{0.0};
#endif  // TRITON_ENABLE_GPU
  std::string repoagent_dir_{"/opt/tritonserver/repoagents"};
  std::string backend_dir_{"/opt/tritonserver/backends"};
  std::vector<std::tuple<std::string, std::string, std::string>>
      backend_config_settings_;

  // Model repository manager configuration
  bool enable_model_namespacing_{false};
  std::set<std::string> model_repository_paths_{};
  TRITONSERVER_ModelControlMode control_mode_{TRITONSERVER_MODEL_CONTROL_NONE};
  std::set<std::string> startup_models_{};
  // Interval, in seconds, when the model repository is polled for changes.
  int32_t repository_poll_secs_{15};
  // Number of threads to use for concurrently loading models
  uint32_t model_load_thread_count_{4};
  std::map<int, double> load_gpu_limit_;

  // Rate limiter configuration
  // FIXME: Once the rate limiter implementation is complete make
  // EXEC_COUNT the default.
  // TRITONSERVER_RateLimitMode
  // rate_limit_mode_{TRITONSERVER_RATE_LIMIT_EXEC_COUNT};
  TRITONSERVER_RateLimitMode rate_limit_mode_{TRITONSERVER_RATE_LIMIT_OFF};
  std::vector<std::tuple<std::string, int, int>> rate_limit_resources_;

  // memory pool configuration
  int64_t pinned_memory_pool_byte_size_{1 << 28};
  std::list<std::pair<int, uint64_t>> cuda_pools_;
  std::list<std::pair<int, size_t>> cuda_virtual_address_size_;

  // [FIXME] this option is broken after backend separation: this should have
  // controlled backend copy behavior but not properly propagate to backend
  // after separation, need to go through backend config.
  int32_t buffer_manager_thread_count_{0};

  std::vector<std::tuple<std::string, std::string, std::string>> host_policies_;

  // Cache configuration
  bool enable_cache_{false};
  std::string cache_dir_{"/opt/tritonserver/caches"};
  std::unordered_map<
      std::string, std::vector<std::pair<std::string, std::string>>>
      cache_config_settings_;

#ifdef TRITON_ENABLE_LOGGING
  bool log_info_{true};
  bool log_warn_{true};
  bool log_error_{true};
  int32_t log_verbose_{0};
  triton::common::Logger::Format log_format_{
      triton::common::Logger::Format::kDEFAULT};
  std::string log_file_{};
#endif  // TRITON_ENABLE_LOGGING

#ifdef TRITON_ENABLE_TRACING
  std::string trace_filepath_{};
  TRITONSERVER_InferenceTraceLevel trace_level_{
      TRITONSERVER_TRACE_LEVEL_DISABLED};
  int32_t trace_rate_{1000};
  int32_t trace_count_{-1};
  int32_t trace_log_frequency_{0};
  InferenceTraceMode trace_mode_{TRACE_MODE_TRITON};
  TraceConfigMap trace_config_map_;
#endif  // TRITON_ENABLE_TRACING

// The configurations for various endpoints (i.e. HTTP, GRPC and metrics)
#ifdef TRITON_ENABLE_HTTP
  bool allow_http_{true};
  std::string http_address_{"0.0.0.0"};
  int32_t http_port_{8000};
  bool reuse_http_port_{false};
  std::string http_forward_header_pattern_;
  // The number of threads to initialize for the HTTP front-end.
  int http_thread_cnt_{8};
  RestrictedFeatures http_restricted_apis_{};
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_GRPC
  bool allow_grpc_{true};
  triton::server::grpc::Options grpc_options_;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
  bool allow_metrics_{true};
  // Defaults to http_address_ if TRITON_ENABLE_HTTP is enabled for backwards,
  // otherwise defaults to "0.0.0.0" for TRITON_ENABLE_HTTP is disabled.
  std::string metrics_address_{""};
  int32_t metrics_port_{8002};
  // Metric settings for Triton core
  float metrics_interval_ms_{2000};
  bool allow_gpu_metrics_{true};
  bool allow_cpu_metrics_{true};
  std::vector<std::tuple<std::string, std::string, std::string>>
      metrics_config_settings_;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_SAGEMAKER
  bool allow_sagemaker_{false};
  std::string sagemaker_address_{"0.0.0.0"};
  int32_t sagemaker_port_{8080};
  bool sagemaker_safe_range_set_{false};
  std::pair<int32_t, int32_t> sagemaker_safe_range_{-1, -1};
  // The number of threads to initialize for the SageMaker HTTP front-end.
  int sagemaker_thread_cnt_{8};
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
  bool allow_vertex_ai_{false};
  std::string vertex_ai_address_{"0.0.0.0"};
  int32_t vertex_ai_port_{8080};
  // The number of threads to initialize for the Vertex AI HTTP front-end.
  int vertex_ai_thread_cnt_{8};
  std::string vertex_ai_default_model_{};
#endif  // TRITON_ENABLE_VERTEX_AI

  // [FIXME] who should call this function?
  void CheckPortCollision();
  using ManagedTritonServerOptionPtr = std::unique_ptr<
      TRITONSERVER_ServerOptions, decltype(&TRITONSERVER_ServerOptionsDelete)>;
  ManagedTritonServerOptionPtr BuildTritonServerOptions();
};

// Exception type to be thrown if the error is parsing related
class ParseException : public std::exception {
 public:
  ParseException() = default;
  ParseException(const std::string& message) : message_(message) {}

  virtual const char* what() const throw() { return message_.c_str(); }

 private:
  const std::string message_{""};
};

// [WIP] Fall-through parser, Parse() will convert the recognized options into
// parameter object and return the unrecognized options to be another argument
// list for other parser to consume.
// This allows the composition of parser chain.
// [FIXME] abstract interface, concrete class below should only parse Triton
// core and endpoint control options (endpoint specific options in their own
// parser)
class TritonParser {
 public:
  TritonParser();
  // Parse command line arguments into a parameters struct and transform
  // the argument list to contain only unrecognized options. The content of
  // unrecognized argument list shares the same lifecycle as 'argv'.
  // Raise ParseException if fail to parse recognized options.
  std::pair<TritonServerParameters, std::vector<char*>> Parse(
      int argc, char** argv);

  // Return usage of all recognized options
  std::string Usage();

 private:
  std::string FormatUsageMessage(std::string str, int offset);
  // Helper functions for parsing options that require multi-value parsing.
  std::tuple<std::string, std::string, std::string> ParseCacheConfigOption(
      const std::string& arg);
  std::tuple<std::string, int, int> ParseRateLimiterResourceOption(
      const std::string& arg);
  std::tuple<std::string, std::string, std::string> ParseBackendConfigOption(
      const std::string& arg);
  std::tuple<std::string, std::string, std::string> ParseHostPolicyOption(
      const std::string& arg);
  std::tuple<std::string, std::string, std::string> ParseMetricsConfigOption(
      const std::string& arg);
  void ParseRestrictedFeatureOption(
      const std::string& arg, const std::string& option_name,
      const std::string& header_prefix, const std::string& feature_type,
      RestrictedFeatures& restricted_features);
#ifdef TRITON_ENABLE_TRACING
  TRITONSERVER_InferenceTraceLevel ParseTraceLevelOption(std::string arg);
  InferenceTraceMode ParseTraceModeOption(std::string arg);
  std::tuple<std::string, std::string, std::string> ParseTraceConfigOption(
      const std::string& arg);
  // Helper functions for post processing for collected trace arguments.
  void SetGlobalTraceArgs(
      TritonServerParameters& lparams, bool trace_level_present,
      bool trace_rate_present, bool trace_count_present,
      bool explicit_disable_trace);
  void SetTritonTraceArgs(
      TritonServerParameters& lparams, bool trace_filepath_present,
      bool trace_log_frequency_present);
  void VerifyOpentelemetryTraceArgs(
      bool trace_filepath_present, bool trace_log_frequency_present);
  void PostProcessTraceArgs(
      TritonServerParameters& lparams, bool trace_level_present,
      bool trace_rate_present, bool trace_count_present,
      bool trace_filepath_present, bool trace_log_frequency_present,
      bool explicit_disable_trace);
#endif  // TRITON_ENABLE_TRACING
  // Helper function to parse option in
  // "<string>[1st_delim]<string>[2nd_delim]<string>" format
  std::tuple<std::string, std::string, std::string> ParseGenericConfigOption(
      const std::string& arg, const std::string& first_delim,
      const std::string& second_delim, const std::string& option_name,
      const std::string& config_name);

  // Initialize individual option groups
  void SetupOptions();
  // Initialize option group mappings
  void SetupOptionGroups();

  // Sum of option groups: vector to maintain insertion order for Usage()
  std::vector<std::pair<std::string, std::vector<Option>&>> option_groups_;
  // Individual option groups
  std::vector<Option> global_options_;
  std::vector<Option> server_options_;
  std::vector<Option> model_repo_options_;
  std::vector<Option> logging_options_;
  std::vector<Option> http_options_;
  std::vector<Option> grpc_options_;
  std::vector<Option> sagemaker_options_;
  std::vector<Option> vertex_options_;
  std::vector<Option> metric_options_;
  std::vector<Option> tracing_options_;
  std::vector<Option> backend_options_;
  std::vector<Option> repo_agent_options_;
  std::vector<Option> cache_options_;
  std::vector<Option> rate_limiter_options_;
  std::vector<Option> memory_device_options_;
  // Group deprecated options to keep preferred options more succinct
  std::vector<Option> deprecated_options_;
};
}}  // namespace triton::server
