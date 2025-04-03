// Copyright 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
constexpr const char* GLOBAL_OPTION_GROUP = "";

#ifdef _WIN32
int optind = 1;
const char* optarg = nullptr;

/// Implementation of `getopt_long` for Windows.
/// Linux uses available implementation:
/// https://github.com/gcc-mirror/gcc/blob/fab08d12b40ad637c5a4ce8e026fb43cd3f0fad1/include/getopt.h
/// and
/// https://github.com/gcc-mirror/gcc/blob/fab08d12b40ad637c5a4ce8e026fb43cd3f0fad1/libiberty/getopt.c#L521
/// Parameters' description is available here:
/// https://github.com/gcc-mirror/gcc/blob/fab08d12b40ad637c5a4ce8e026fb43cd3f0fad1/libiberty/getopt.c#L464-L518
/// `optind' is an index to iterate over `argv`, (whose length is `argc`),
/// and starts from 1, since argv[0] is the program name.
/// Text in the current `argv`-element is returned in `optarg'.
/// Note: if option was provided in the form of --<key>=<value>, then
/// optarg is (argv[optind] + found + 1), i.e. everything after `=`.
/// Alternatively, option can be provided as --<key> <value>.
/// In this case, <value> is storred as a separate parameter in `argv`.
/// `longind` returns the index in `longopts` of the long-named option found.

int
getopt_long(
    int argc, char* const argv[], const char* optstring,
    const struct option* longopts, int* longind)
{
  if (optind >= argc) {
    return -1;
  }
  const struct option* curr_longopt = longopts;
  std::string argv_str = argv[optind];
  size_t found = argv_str.find_first_of("=");
  std::string key = argv_str.substr(
      2, (found == std::string::npos) ? std::string::npos : (found - 2));
  int option_index = 0;
  for (curr_longopt, option_index; curr_longopt->name;
       curr_longopt++, option_index++) {
    if (key == curr_longopt->name) {
      if (longind != NULL)
        (*longind) = option_index;
      if (curr_longopt->has_arg == required_argument) {
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
      return curr_longopt->val;
    }
  }
  return -1;
}
#endif

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

#include "common.h"

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "triton/common/triton_json.h"


namespace triton { namespace server {

// [FIXME] expose following parse helpers for other type of parser
namespace {

// A wrapper around std::stoi, std::stoull, std::stoll, std::stod
// to catch `invalid argument` and `out of range` exceptions
template <typename T>
T StringTo(const std::string& arg);

template <>
int
StringTo(const std::string& arg)
{
  return std::stoi(arg);
}

#ifdef TRITON_ENABLE_TRACING
template <>
uint32_t
StringTo(const std::string& arg)
{
  return std::stoul(arg);
}
#endif  // TRITON_ENABLE_TRACING

template <>
uint64_t
StringTo(const std::string& arg)
{
  return std::stoull(arg);
}

template <>
int64_t
StringTo(const std::string& arg)
{
  return std::stoll(arg);
}

template <>
double
StringTo(const std::string& arg)
{
  return std::stod(arg);
}

// There must be specialization for the types to be parsed into so that
// the argument is properly validated and parsed. Attempted to use input
// operator (>>) but it will consume improper argument without error
// (i.e. parse "1.4" to 'int' will return 1 but we want to report error).
template <typename T>
T
ParseOption(const std::string& arg)
{
  try {
    return StringTo<T>(arg);
  }
  catch (const std::invalid_argument& ia) {
    std::stringstream ss;
    ss << "Invalid option value. Got " << arg << std::endl;
    throw ParseException(ss.str());
  }
  catch (const std::out_of_range& oor) {
    std::stringstream ss;
    ss << "Provided option value is out of bound. Got " << arg << std::endl;
    throw ParseException(ss.str());
  }
}

template <>
bool
ParseOption(const std::string& arg)
{
  // 'arg' need to comply with template declaration
  std::string larg = arg;
  std::transform(larg.begin(), larg.end(), larg.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  if ((larg == "true") || (larg == "on") || (larg == "1")) {
    return true;
  }
  if ((larg == "false") || (larg == "off") || (larg == "0")) {
    return false;
  }

  throw ParseException("invalid value for bool option: " + arg);
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

  return ParseOption<int>(arg);
}
#endif  // TRITON_ENABLE_LOGGING

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

template <typename T1, typename T2>
std::pair<T1, T2>
ParsePairOption(const std::string& arg, const std::string& delim_str)
{
  int delim = arg.find(delim_str);

  if ((delim < 0)) {
    std::stringstream ss;
    ss << "Cannot parse pair option due to incorrect number of inputs."
          "--<pair option> argument requires format <first>"
       << delim_str << "<second>. "
       << "Found: " << arg << std::endl;
    throw ParseException(ss.str());
  }

  std::string first_string = arg.substr(0, delim);
  std::string second_string = arg.substr(delim + delim_str.length());

  // Specific conversion from key-value string to actual key-value type,
  // should be extracted out of this function if we need to parse
  // more pair option of different types.
  return {ParseOption<T1>(first_string), ParseOption<T2>(second_string)};
}

// Split 'options' by 'delim_str' and place split strings into a vector
std::vector<std::string>
SplitOptions(std::string options, const std::string& delim_str)
{
  std::vector<std::string> res;

  int delim = options.find(delim_str);
  while ((delim >= 0)) {
    res.emplace_back(options.substr(0, delim));
    options = options.substr(delim + delim_str.length());
    delim = options.find(delim_str);
  }
  // include last element
  res.emplace_back(options);
  return res;
}

}  // namespace

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
  OPTION_HTTP_HEADER_FORWARD_PATTERN,
  OPTION_HTTP_PORT,
  OPTION_REUSE_HTTP_PORT,
  OPTION_HTTP_ADDRESS,
  OPTION_HTTP_THREAD_COUNT,
  OPTION_HTTP_RESTRICTED_API,
#endif  // TRITON_ENABLE_HTTP
#if defined(TRITON_ENABLE_GRPC)
  OPTION_ALLOW_GRPC,
  OPTION_GRPC_PORT,
  OPTION_REUSE_GRPC_PORT,
  OPTION_GRPC_ADDRESS,
  OPTION_GRPC_HEADER_FORWARD_PATTERN,
  OPTION_GRPC_INFER_THREAD_COUNT,
  OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE,
  OPTION_GRPC_MAX_RESPONSE_POOL_SIZE,
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
  OPTION_GRPC_RESTRICTED_PROTOCOL,
  OPTION_GRPC_ARG_MAX_CONNECTION_AGE_MS,
  OPTION_GRPC_ARG_MAX_CONNECTION_AGE_GRACE_MS,
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
  OPTION_METRICS_ADDRESS,
  OPTION_METRICS_PORT,
  OPTION_METRICS_INTERVAL_MS,
  OPTION_METRICS_CONFIG,
#endif  // TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_TRACING
  OPTION_TRACE_FILEPATH,
  OPTION_TRACE_LEVEL,
  OPTION_TRACE_RATE,
  OPTION_TRACE_COUNT,
  OPTION_TRACE_LOG_FREQUENCY,
  OPTION_TRACE_CONFIG,
#endif  // TRITON_ENABLE_TRACING
  OPTION_MODEL_CONTROL_MODE,
  OPTION_POLL_REPO_SECS,
  OPTION_STARTUP_MODEL,
  OPTION_CUSTOM_MODEL_CONFIG_NAME,
  OPTION_RATE_LIMIT,
  OPTION_RATE_LIMIT_RESOURCE,
  OPTION_PINNED_MEMORY_POOL_BYTE_SIZE,
  OPTION_CUDA_MEMORY_POOL_BYTE_SIZE,
  OPTION_CUDA_VIRTUAL_ADDRESS_SIZE,
  OPTION_RESPONSE_CACHE_BYTE_SIZE,
  OPTION_CACHE_CONFIG,
  OPTION_CACHE_DIR,
  OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY,
  OPTION_EXIT_TIMEOUT_SECS,
  OPTION_BACKEND_DIR,
  OPTION_REPOAGENT_DIR,
  OPTION_BUFFER_MANAGER_THREAD_COUNT,
  OPTION_MODEL_LOAD_THREAD_COUNT,
  OPTION_MODEL_LOAD_RETRY_COUNT,
  OPTION_BACKEND_CONFIG,
  OPTION_HOST_POLICY,
  OPTION_MODEL_LOAD_GPU_LIMIT,
  OPTION_MODEL_NAMESPACING,
  OPTION_ENABLE_PEER_ACCESS
};

void
TritonParser::SetupOptions()
{
  global_options_.push_back(
      {OPTION_HELP, "help", Option::ArgNone, "Print usage"});

  server_options_.push_back(
      {OPTION_ID, "id", Option::ArgStr, "Identifier for this server."});
  server_options_.push_back(
      {OPTION_EXIT_TIMEOUT_SECS, "exit-timeout-secs", Option::ArgInt,
       "Timeout (in seconds) when exiting to wait for in-flight inferences to "
       "finish. After the timeout expires the server exits even if inferences "
       "are still in flight."});

  model_repo_options_.push_back(
      {OPTION_MODEL_REPOSITORY, "model-store", Option::ArgStr,
       "Equivalent to --model-repository."});
  model_repo_options_.push_back(
      {OPTION_MODEL_REPOSITORY, "model-repository", Option::ArgStr,
       "Path to model repository directory. It may be specified multiple times "
       "to add multiple model repositories. Note that if a model is not unique "
       "across all model repositories at any time, the model will not be "
       "available."});
  model_repo_options_.push_back(
      {OPTION_EXIT_ON_ERROR, "exit-on-error", Option::ArgBool,
       "Exit the inference server if an error occurs during initialization."});
  model_repo_options_.push_back(
      {OPTION_DISABLE_AUTO_COMPLETE_CONFIG, "disable-auto-complete-config",
       Option::ArgNone,
       "If set, disables the triton and backends from auto completing model "
       "configuration files. Model configuration files must be provided and "
       "all required "
       "configuration settings must be specified."});
  model_repo_options_.push_back(
      {OPTION_STRICT_READINESS, "strict-readiness", Option::ArgBool,
       "If true /v2/health/ready endpoint indicates ready if the server "
       "is responsive and all models are available. If false "
       "/v2/health/ready endpoint indicates ready if server is responsive "
       "even if some/all models are unavailable."});
  model_repo_options_.push_back(
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
       "be loaded at startup."});
  model_repo_options_.push_back(
      {OPTION_POLL_REPO_SECS, "repository-poll-secs", Option::ArgInt,
       "Interval in seconds between each poll of the model repository to check "
       "for changes. Valid only when --model-control-mode=poll is "
       "specified."});
  model_repo_options_.push_back(
      {OPTION_STARTUP_MODEL, "load-model", Option::ArgStr,
       "Name of the model to be loaded on server startup. It may be specified "
       "multiple times to add multiple models. To load ALL models at startup, "
       "specify '*' as the model name with --load-model=* as the ONLY "
       "--load-model argument, this does not imply any pattern matching. "
       "Specifying --load-model=* in conjunction with another --load-model "
       "argument will result in error. Note that this option will only take "
       "effect if --model-control-mode=explicit is true."});
  model_repo_options_.push_back(
      {OPTION_CUSTOM_MODEL_CONFIG_NAME, "model-config-name", Option::ArgStr,
       "The custom configuration name for models to load."
       "The name should not contain any space character."
       "For example: --model-config-name=h100. "
       "If --model-config-name is not set, Triton will use the default "
       "config.pbtxt."});
  model_repo_options_.push_back(
      {OPTION_MODEL_LOAD_THREAD_COUNT, "model-load-thread-count",
       Option::ArgInt,
       "The number of threads used to concurrently load models in "
       "model repositories. Default is 4."});
  model_repo_options_.push_back(
      {OPTION_MODEL_LOAD_RETRY_COUNT, "model-load-retry-count", Option::ArgInt,
       "The number of retry to load a model in "
       "model repositories. Default is 0."});
  model_repo_options_.push_back(
      {OPTION_MODEL_NAMESPACING, "model-namespacing", Option::ArgBool,
       "Whether model namespacing is enable or not. If true, models with the "
       "same name can be served if they are in different namespace."});
  model_repo_options_.push_back(
      {OPTION_ENABLE_PEER_ACCESS, "enable-peer-access", Option::ArgBool,
       "Whether the server tries to enable peer access or not. Even when this "
       "options is set to true,  "
       "peer access could still be not enabled because the underlying system "
       "doesn't support it."
       " The server will log a warning in this case. Default is true."});

#if defined(TRITON_ENABLE_HTTP)
  http_options_.push_back(
      {OPTION_ALLOW_HTTP, "allow-http", Option::ArgBool,
       "Allow the server to listen for HTTP requests."});
  http_options_.push_back(
      {OPTION_HTTP_ADDRESS, "http-address", Option::ArgStr,
       "The address for the http server to bind to. Default is 0.0.0.0"});
  http_options_.push_back(
      {OPTION_HTTP_PORT, "http-port", Option::ArgInt,
       "The port for the server to listen on for HTTP "
       "requests. Default is 8000."});
  http_options_.push_back(
      {OPTION_REUSE_HTTP_PORT, "reuse-http-port", Option::ArgBool,
       "Allow multiple servers to listen on the same HTTP port when every "
       "server has this option set. If you plan to use this option as a way to "
       "load balance between different Triton servers, the same model "
       "repository or set of models must be used for every server."});
  http_options_.push_back(
      {OPTION_HTTP_HEADER_FORWARD_PATTERN, "http-header-forward-pattern",
       Option::ArgStr,
       "The regular expression pattern that will be used for forwarding HTTP "
       "headers as inference request parameters."});
  http_options_.push_back(
      {OPTION_HTTP_THREAD_COUNT, "http-thread-count", Option::ArgInt,
       "Number of threads handling HTTP requests."});
  http_options_.push_back(
      {OPTION_HTTP_RESTRICTED_API, "http-restricted-api",
       "<string>:<string>=<string>",
       "Specify restricted HTTP api setting. The format of this "
       "flag is --http-restricted-api=<apis>,<key>=<value>. Where "
       "<api> is a comma-separated list of apis to be restricted. "
       "<key> will be additional header key to be checked when a HTTP request "
       "is received, and <value> is the value expected to be matched."
       " Allowed APIs: " +
           Join(RESTRICTED_CATEGORY_NAMES, ", ")});
#endif  // TRITON_ENABLE_HTTP

#if defined(TRITON_ENABLE_GRPC)
  grpc_options_.push_back(
      {OPTION_ALLOW_GRPC, "allow-grpc", Option::ArgBool,
       "Allow the server to listen for GRPC requests."});
  grpc_options_.push_back(
      {OPTION_GRPC_ADDRESS, "grpc-address", Option::ArgStr,
       "The address for the grpc server to binds to. Default is 0.0.0.0"});
  grpc_options_.push_back(
      {OPTION_GRPC_PORT, "grpc-port", Option::ArgInt,
       "The port for the server to listen on for GRPC "
       "requests. Default is 8001."});
  grpc_options_.push_back(
      {OPTION_REUSE_GRPC_PORT, "reuse-grpc-port", Option::ArgBool,
       "Allow multiple servers to listen on the same GRPC port when every "
       "server has this option set. If you plan to use this option as a way to "
       "load balance between different Triton servers, the same model "
       "repository or set of models must be used for every server."});
  grpc_options_.push_back(
      {OPTION_GRPC_HEADER_FORWARD_PATTERN, "grpc-header-forward-pattern",
       Option::ArgStr,
       "The regular expression pattern that will be used for forwarding GRPC "
       "headers as inference request parameters."});
  grpc_options_.push_back(
      {OPTION_GRPC_INFER_THREAD_COUNT, "grpc-infer-thread-count",
       Option::ArgInt,
       "The number of gRPC inference handler threads. Default is 2."});
  grpc_options_.push_back(
      {OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE,
       "grpc-infer-allocation-pool-size", Option::ArgInt,
       "The maximum number of states (inference request/response queues) that "
       "remain allocated for reuse. As long as the number of in-flight "
       "requests doesn't exceed this value there will be no "
       "allocation/deallocation of request/response objects."});
  grpc_options_.push_back(
      {OPTION_GRPC_MAX_RESPONSE_POOL_SIZE, "grpc-max-response-pool-size",
       Option::ArgInt,
       "The maximum number of inference response objects that can remain "
       "allocated in the response queue at any given time."});
  grpc_options_.push_back(
      {OPTION_GRPC_USE_SSL, "grpc-use-ssl", Option::ArgBool,
       "Use SSL authentication for GRPC requests. Default is false."});
  grpc_options_.push_back(
      {OPTION_GRPC_USE_SSL_MUTUAL, "grpc-use-ssl-mutual", Option::ArgBool,
       "Use mututal SSL authentication for GRPC requests. This option will "
       "preempt '--grpc-use-ssl' if it is also specified. Default is false."});
  grpc_options_.push_back(
      {OPTION_GRPC_SERVER_CERT, "grpc-server-cert", Option::ArgStr,
       "File holding PEM-encoded server certificate. Ignored unless "
       "--grpc-use-ssl is true."});
  grpc_options_.push_back(
      {OPTION_GRPC_SERVER_KEY, "grpc-server-key", Option::ArgStr,
       "File holding PEM-encoded server key. Ignored unless "
       "--grpc-use-ssl is true."});
  grpc_options_.push_back(
      {OPTION_GRPC_ROOT_CERT, "grpc-root-cert", Option::ArgStr,
       "File holding PEM-encoded root certificate. Ignore unless "
       "--grpc-use-ssl is false."});
  grpc_options_.push_back(
      {OPTION_GRPC_RESPONSE_COMPRESSION_LEVEL,
       "grpc-infer-response-compression-level", Option::ArgStr,
       "The compression level to be used while returning the infer response to "
       "the peer. Allowed values are none, low, medium and high. By default, "
       "compression level is selected as none."});
  grpc_options_.push_back(
      {OPTION_GRPC_ARG_KEEPALIVE_TIME_MS, "grpc-keepalive-time", Option::ArgInt,
       "The period (in milliseconds) after which a keepalive ping is sent on "
       "the transport. Default is 7200000 (2 hours)."});
  grpc_options_.push_back(
      {OPTION_GRPC_ARG_KEEPALIVE_TIMEOUT_MS, "grpc-keepalive-timeout",
       Option::ArgInt,
       "The period (in milliseconds) the sender of the keepalive ping waits "
       "for an acknowledgement. If it does not receive an acknowledgment "
       "within this time, it will close the connection. "
       "Default is 20000 (20 seconds)."});
  grpc_options_.push_back(
      {OPTION_GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS,
       "grpc-keepalive-permit-without-calls", Option::ArgBool,
       "Allows keepalive pings to be sent even if there are no calls in flight "
       "(0 : false; 1 : true). Default is 0 (false)."});
  grpc_options_.push_back(
      {OPTION_GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA,
       "grpc-http2-max-pings-without-data", Option::ArgInt,
       "The maximum number of pings that can be sent when there is no "
       "data/header frame to be sent. gRPC Core will not continue sending "
       "pings if we run over the limit. Setting it to 0 allows sending pings "
       "without such a restriction. Default is 2."});
  grpc_options_.push_back(
      {OPTION_GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS,
       "grpc-http2-min-recv-ping-interval-without-data", Option::ArgInt,
       "If there are no data/header frames being sent on the transport, this "
       "channel argument on the server side controls the minimum time "
       "(in milliseconds) that gRPC Core would expect between receiving "
       "successive pings. If the time between successive pings is less than "
       "this time, then the ping will be considered a bad ping from the peer. "
       "Such a ping counts as a ‘ping strike’. Default is 300000 (5 "
       "minutes)."});
  grpc_options_.push_back(
      {OPTION_GRPC_ARG_HTTP2_MAX_PING_STRIKES, "grpc-http2-max-ping-strikes",
       Option::ArgInt,
       "Maximum number of bad pings that the server will tolerate before "
       "sending an HTTP2 GOAWAY frame and closing the transport. Setting it to "
       "0 allows the server to accept any number of bad pings. Default is 2."});
  grpc_options_.push_back(
      {OPTION_GRPC_ARG_MAX_CONNECTION_AGE_MS, "grpc-max-connection-age",
       Option::ArgInt,
       "Maximum time that a channel may exist in milliseconds."
       "Default is undefined."});
  grpc_options_.push_back(
      {OPTION_GRPC_ARG_MAX_CONNECTION_AGE_GRACE_MS,
       "grpc-max-connection-age-grace", Option::ArgInt,
       "Grace period after the channel reaches its max age. "
       "Default is undefined."});
  grpc_options_.push_back(
      {OPTION_GRPC_RESTRICTED_PROTOCOL, "grpc-restricted-protocol",
       "<string>:<string>=<string>",
       "Specify restricted GRPC protocol setting. The format of this "
       "flag is --grpc-restricted-protocol=<protocols>,<key>=<value>. Where "
       "<protocol> is a comma-separated list of protocols to be restricted. "
       "<key> will be additional header key to be checked when a GRPC request "
       "is received, and <value> is the value expected to be matched."
       " Allowed protocols: " +
           Join(RESTRICTED_CATEGORY_NAMES, ", ")});
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_LOGGING
  logging_options_.push_back(
      {OPTION_LOG_VERBOSE, "log-verbose", Option::ArgInt,
       "Set verbose logging level. Zero (0) disables verbose logging and "
       "values >= 1 enable verbose logging."});
  logging_options_.push_back(
      {OPTION_LOG_INFO, "log-info", Option::ArgBool,
       "Enable/disable info-level logging."});
  logging_options_.push_back(
      {OPTION_LOG_WARNING, "log-warning", Option::ArgBool,
       "Enable/disable warning-level logging."});
  logging_options_.push_back(
      {OPTION_LOG_ERROR, "log-error", Option::ArgBool,
       "Enable/disable error-level logging."});
  logging_options_.push_back(
      {OPTION_LOG_FORMAT, "log-format", Option::ArgStr,
       "Set the logging format. Options are \"default\" and \"ISO8601\". "
       "The default is \"default\". For \"default\", the log severity (L) and "
       "timestamp will be logged as \"LMMDD hh:mm:ss.ssssss\". "
       "For \"ISO8601\", the log format will be \"YYYY-MM-DDThh:mm:ssZ L\"."});
  logging_options_.push_back(
      {OPTION_LOG_FILE, "log-file", Option::ArgStr,
       "Set the name of the log output file. If specified, log outputs will be "
       "saved to this file. If not specified, log outputs will stream to the "
       "console."});
#endif  // TRITON_ENABLE_LOGGING

#if defined(TRITON_ENABLE_SAGEMAKER)
  sagemaker_options_.push_back(
      {OPTION_ALLOW_SAGEMAKER, "allow-sagemaker", Option::ArgBool,
       "Allow the server to listen for Sagemaker requests. Default is false."});
  sagemaker_options_.push_back(
      {OPTION_SAGEMAKER_PORT, "sagemaker-port", Option::ArgInt,
       "The port for the server to listen on for Sagemaker requests. Default "
       "is 8080."});
  sagemaker_options_.push_back(
      {OPTION_SAGEMAKER_SAFE_PORT_RANGE, "sagemaker-safe-port-range",
       "<integer>-<integer>",
       "Set the allowed port range for endpoints other than the SageMaker "
       "endpoints."});
  sagemaker_options_.push_back(
      {OPTION_SAGEMAKER_THREAD_COUNT, "sagemaker-thread-count", Option::ArgInt,
       "Number of threads handling Sagemaker requests. Default is 8."});
#endif  // TRITON_ENABLE_SAGEMAKER

#if defined(TRITON_ENABLE_VERTEX_AI)
  vertex_options_.push_back(
      {OPTION_ALLOW_VERTEX_AI, "allow-vertex-ai", Option::ArgBool,
       "Allow the server to listen for Vertex AI requests. Default is true if "
       "AIP_MODE=PREDICTION, false otherwise."});
  vertex_options_.push_back(
      {OPTION_VERTEX_AI_PORT, "vertex-ai-port", Option::ArgInt,
       "The port for the server to listen on for Vertex AI requests. Default "
       "is AIP_HTTP_PORT if set, 8080 otherwise."});
  vertex_options_.push_back(
      {OPTION_VERTEX_AI_THREAD_COUNT, "vertex-ai-thread-count", Option::ArgInt,
       "Number of threads handling Vertex AI requests. Default is 8."});
  vertex_options_.push_back(
      {OPTION_VERTEX_AI_DEFAULT_MODEL, "vertex-ai-default-model",
       Option::ArgStr,
       "The name of the model to use for single-model inference requests."});
#endif  // TRITON_ENABLE_VERTEX_AI

#if defined(TRITON_ENABLE_METRICS)
  metric_options_.push_back(
      {OPTION_ALLOW_METRICS, "allow-metrics", Option::ArgBool,
       "Allow the server to provide prometheus metrics."});
  metric_options_.push_back(
      {OPTION_ALLOW_GPU_METRICS, "allow-gpu-metrics", Option::ArgBool,
       "Allow the server to provide GPU metrics. Ignored unless "
       "--allow-metrics is true."});
  metric_options_.push_back(
      {OPTION_ALLOW_CPU_METRICS, "allow-cpu-metrics", Option::ArgBool,
       "Allow the server to provide CPU metrics. Ignored unless "
       "--allow-metrics is true."});
  metric_options_.push_back(
      {OPTION_METRICS_ADDRESS, "metrics-address", Option::ArgStr,
       "The address for the metrics server to bind to. Default is the same as "
       "--http-address if built with HTTP support. Otherwise, default is "
       "0.0.0.0"});
  metric_options_.push_back(
      {OPTION_METRICS_PORT, "metrics-port", Option::ArgInt,
       "The port reporting prometheus metrics. Default is 8002."});
  metric_options_.push_back(
      {OPTION_METRICS_INTERVAL_MS, "metrics-interval-ms", Option::ArgFloat,
       "Metrics will be collected once every <metrics-interval-ms> "
       "milliseconds. Default is 2000 milliseconds."});
  metric_options_.push_back(
      {OPTION_METRICS_CONFIG, "metrics-config", "<string>=<string>",
       "Specify a metrics-specific configuration setting. The format of this "
       "flag is --metrics-config=<setting>=<value>. It can be specified "
       "multiple times."});
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_TRACING
  tracing_options_.push_back(
      {OPTION_TRACE_CONFIG, "trace-config", "<string>,<string>=<string>",
       "Specify global or trace mode specific configuration setting. "
       "The format of this flag is --trace-config "
       "<mode>,<setting>=<value>. "
       "Where <mode> is either \"triton\" or \"opentelemetry\". "
       "The default is \"triton\". To specify global trace settings "
       "(level, rate, count, or mode), the format would be "
       "--trace-config <setting>=<value>. For \"triton\" mode, the server will "
       "use "
       "Triton's Trace APIs. For \"opentelemetry\" mode, the server will use "
       "OpenTelemetry's APIs to generate, collect and export traces for "
       "individual inference requests."});
#endif  // TRITON_ENABLE_TRACING

  cache_options_.push_back(
      {OPTION_CACHE_CONFIG, "cache-config", "<string>,<string>=<string>",
       "Specify a cache-specific configuration setting. The format of this "
       "flag is --cache-config=<cache_name>,<setting>=<value>. Where "
       "<cache_name> is the name of the cache, such as 'local' or 'redis'. "
       "Example: --cache-config=local,size=1048576 will configure a 'local' "
       "cache implementation with a fixed buffer pool of size 1048576 bytes."});
  cache_options_.push_back(
      {OPTION_CACHE_DIR, "cache-directory", Option::ArgStr,
       "The global directory searched for cache shared libraries. Default is "
       "'/opt/tritonserver/caches'. This directory is expected to contain a "
       "cache implementation as a shared library with the name "
       "'libtritoncache.so'."});


  rate_limiter_options_.push_back(
      // FIXME:  fix the default to execution_count once RL logic is complete.
      {OPTION_RATE_LIMIT, "rate-limit", Option::ArgStr,
       "Specify the mode for rate limiting. Options are \"execution_count\" "
       "and \"off\". The default is \"off\". For "
       "\"execution_count\", the server will determine the instance using "
       "configured priority and the number of time the instance has been "
       "used to run inference. The inference will finally be executed once "
       "the required resources are available. For \"off\", the server will "
       "ignore any rate limiter config and run inference as soon as an "
       "instance is ready."});
  rate_limiter_options_.push_back(
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
       "flag is case-insensitive."});

  memory_device_options_.push_back(
      {OPTION_PINNED_MEMORY_POOL_BYTE_SIZE, "pinned-memory-pool-byte-size",
       Option::ArgInt,
       "The total byte size that can be allocated as pinned system memory. "
       "If GPU support is enabled, the server will allocate pinned system "
       "memory to accelerate data transfer between host and devices until it "
       "exceeds the specified byte size. If 'numa-node' is configured via "
       "--host-policy, the pinned system memory of the pool size will be "
       "allocated on each numa node. This option will not affect the "
       "allocation conducted by the backend frameworks. Default is 256 MB."});
  memory_device_options_.push_back(
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
       "previous uses for the same GPU device. Default is 64 MB."});
  memory_device_options_.push_back(
      {OPTION_CUDA_VIRTUAL_ADDRESS_SIZE, "cuda-virtual-address-size",
       "<integer>:<integer>",
       "The total CUDA virtual address size that will be used for each "
       "implicit state when growable memory is used. This value determines "
       "the maximum size of each implicit state. The state size cannot go "
       "beyond this value. The argument should be "
       "2 integers separated by colons in the format "
       "<GPU device ID>:<CUDA virtual address size>. This option can be used "
       "multiple "
       "times, but only once per GPU device. Subsequent uses will overwrite "
       "previous uses for the same GPU device. Default is 1 GB."});
  memory_device_options_.push_back(
      {OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY,
       "min-supported-compute-capability", Option::ArgFloat,
       "The minimum supported CUDA compute capability. GPUs that don't support "
       "this compute capability will not be used by the server."});
  memory_device_options_.push_back(
      {OPTION_BUFFER_MANAGER_THREAD_COUNT, "buffer-manager-thread-count",
       Option::ArgInt,
       "The number of threads used to accelerate copies and other operations "
       "required to manage input and output tensor contents. Default is 0."});
  memory_device_options_.push_back(
      {OPTION_HOST_POLICY, "host-policy", "<string>,<string>=<string>",
       "Specify a host policy setting associated with a policy name. The "
       "format of this flag is --host-policy=<policy_name>,<setting>=<value>. "
       "Currently supported settings are 'numa-node', 'cpu-cores'. Note that "
       "'numa-node' setting will affect pinned memory pool behavior, see "
       "--pinned-memory-pool for more detail."});
  memory_device_options_.push_back(
      {OPTION_MODEL_LOAD_GPU_LIMIT, "model-load-gpu-limit",
       "<device_id>:<fraction>",
       "Specify the limit on GPU memory usage as a fraction. If model loading "
       "on the device is requested and the current memory usage exceeds the "
       "limit, the load will be rejected. If not specified, the limit will "
       "not be set."});

  backend_options_.push_back(
      {OPTION_BACKEND_DIR, "backend-directory", Option::ArgStr,
       "The global directory searched for backend shared libraries. Default is "
       "'/opt/tritonserver/backends'."});
  backend_options_.push_back(
      {OPTION_BACKEND_CONFIG, "backend-config", "<string>,<string>=<string>",
       "Specify a backend-specific configuration setting. The format of this "
       "flag is --backend-config=<backend_name>,<setting>=<value>. Where "
       "<backend_name> is the name of the backend, such as 'tensorrt'."});

  repo_agent_options_.push_back(
      {OPTION_REPOAGENT_DIR, "repoagent-directory", Option::ArgStr,
       "The global directory searched for repository agent shared libraries. "
       "Default is '/opt/tritonserver/repoagents'."});

  // Deprecations
  deprecated_options_.push_back(
      {OPTION_STRICT_MODEL_CONFIG, "strict-model-config", Option::ArgBool,
       "DEPRECATED: If true model configuration files must be provided and all "
       "required "
       "configuration settings must be specified. If false the model "
       "configuration may be absent or only partially specified and the "
       "server will attempt to derive the missing required configuration."});
  deprecated_options_.push_back(
      {OPTION_RESPONSE_CACHE_BYTE_SIZE, "response-cache-byte-size",
       Option::ArgInt, "DEPRECATED: Please use --cache-config instead."});
#ifdef TRITON_ENABLE_TRACING
  deprecated_options_.push_back(
      {OPTION_TRACE_FILEPATH, "trace-file", Option::ArgStr,
       "DEPRECATED: Please use --trace-config triton,file=<path/to/your/file>"
       " Set the file where trace output will be saved. If "
       "--trace-log-frequency"
       " is also specified, this argument value will be the prefix of the files"
       " to save the trace output. See --trace-log-frequency for detail."});
  deprecated_options_.push_back(
      {OPTION_TRACE_LEVEL, "trace-level", Option::ArgStr,
       "DEPRECATED: Please use --trace-config level=<OFF|TIMESTAMPS|TENSORS>"
       "Specify a trace level. OFF to disable tracing, TIMESTAMPS to "
       "trace timestamps, TENSORS to trace tensors. It may be specified "
       "multiple times to trace multiple information. Default is OFF."});
  deprecated_options_.push_back(
      {OPTION_TRACE_RATE, "trace-rate", Option::ArgInt,
       "DEPRECATED: Please use --trace-config rate=<rate value>"
       "Set the trace sampling rate. Default is 1000."});
  deprecated_options_.push_back(
      {OPTION_TRACE_COUNT, "trace-count", Option::ArgInt,
       "DEPRECATED: Please use --trace-config count=<count value>"
       "Set the number of traces to be sampled. If the value is -1, the number "
       "of traces to be sampled will not be limited. Default is -1."});
  deprecated_options_.push_back(
      {OPTION_TRACE_LOG_FREQUENCY, "trace-log-frequency", Option::ArgInt,
       "DEPRECATED: Please use --trace-config triton,log-frequency=<value>"
       "Set the trace log frequency. If the value is 0, Triton will only log "
       "the trace output to <trace-file> when shutting down. Otherwise, Triton "
       "will log the trace output to <trace-file>.<idx> when it collects the "
       "specified number of traces. For example, if the log frequency is 100, "
       "when Triton collects the 100-th trace, it logs the traces to file "
       "<trace-file>.0, and when it collects the 200-th trace, it logs the "
       "101-th to the 200-th traces to file <trace-file>.1. Default is 0."});
#endif  // TRITON_ENABLE_TRACING
}

void
TritonParser::SetupOptionGroups()
{
  SetupOptions();
  option_groups_.emplace_back(GLOBAL_OPTION_GROUP, global_options_);
  option_groups_.emplace_back("Server", server_options_);
  option_groups_.emplace_back("Logging", logging_options_);
  option_groups_.emplace_back("Model Repository", model_repo_options_);
  option_groups_.emplace_back("HTTP", http_options_);
  option_groups_.emplace_back("GRPC", grpc_options_);
  option_groups_.emplace_back("Sagemaker", sagemaker_options_);
  option_groups_.emplace_back("Vertex", vertex_options_);
  option_groups_.emplace_back("Metrics", metric_options_);
  option_groups_.emplace_back("Tracing", tracing_options_);
  option_groups_.emplace_back("Backend", backend_options_);
  option_groups_.emplace_back("Repository Agent", repo_agent_options_);
  option_groups_.emplace_back("Response Cache", cache_options_);
  option_groups_.emplace_back("Rate Limiter", rate_limiter_options_);
  option_groups_.emplace_back(
      "Memory/Device Management", memory_device_options_);
  option_groups_.emplace_back("DEPRECATED", deprecated_options_);
}

TritonParser::TritonParser()
{
  SetupOptionGroups();
}

void
TritonServerParameters::CheckPortCollision()
{
  // [FIXME] try to make this function endpoint type agnostic
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
    ports.emplace_back(
        "metrics", metrics_address_, metrics_port_, false, -1, -1);
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
        std::stringstream ss;
        ss << "The server cannot listen to " << std::get<0>(*comparing_it)
           << " requests at port " << std::get<2>(*comparing_it)
           << ", allowed port range is [" << std::get<4>(*curr_it) << ", "
           << std::get<5>(*curr_it) << "]" << std::endl;
        throw ParseException(ss.str());
      }
      if (std::get<2>(*curr_it) == std::get<2>(*comparing_it)) {
        std::stringstream ss;
        ss << "The server cannot listen to " << std::get<0>(*curr_it)
           << " requests "
           << "and " << std::get<0>(*comparing_it)
           << " requests at the same address and port " << std::get<1>(*curr_it)
           << ":" << std::get<2>(*curr_it) << std::endl;
        throw ParseException(ss.str());
      }
    }
  }
}

TritonServerParameters::ManagedTritonServerOptionPtr
TritonServerParameters::BuildTritonServerOptions()
{
  TRITONSERVER_ServerOptions* loptions = nullptr;
  THROW_IF_ERR(
      ParseException, TRITONSERVER_ServerOptionsNew(&loptions),
      "creating server options");
  ManagedTritonServerOptionPtr managed_ptr(
      loptions, TRITONSERVER_ServerOptionsDelete);
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetServerId(loptions, server_id_.c_str()),
      "setting server ID");
  for (const auto& model_repository_path : model_repository_paths_) {
    THROW_IF_ERR(
        ParseException,
        TRITONSERVER_ServerOptionsSetModelRepositoryPath(
            loptions, model_repository_path.c_str()),
        "setting model repository path");
  }
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetModelControlMode(loptions, control_mode_),
      "setting model control mode");
  for (const auto& model : startup_models_) {
    THROW_IF_ERR(
        ParseException,
        TRITONSERVER_ServerOptionsSetStartupModel(loptions, model.c_str()),
        "setting startup model");
  }
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetModelConfigName(
          loptions, model_config_name_.c_str()),
      "setting custom model configuration name for models");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetRateLimiterMode(loptions, rate_limit_mode_),
      "setting rate limiter configuration");
  for (const auto& resource : rate_limit_resources_) {
    THROW_IF_ERR(
        ParseException,
        TRITONSERVER_ServerOptionsAddRateLimiterResource(
            loptions, std::get<0>(resource).c_str(), std::get<1>(resource),
            std::get<2>(resource)),
        "setting rate limiter resource");
  }
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
          loptions, pinned_memory_pool_byte_size_),
      "setting total pinned memory byte size");
  for (const auto& cuda_pool : cuda_pools_) {
    THROW_IF_ERR(
        ParseException,
        TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
            loptions, cuda_pool.first, cuda_pool.second),
        "setting total CUDA memory byte size");
  }
  for (const auto& cuda_virtual_address_size : cuda_virtual_address_size_) {
    THROW_IF_ERR(
        ParseException,
        TRITONSERVER_ServerOptionsSetCudaVirtualAddressSize(
            loptions, cuda_virtual_address_size.first,
            cuda_virtual_address_size.second),
        "setting total CUDA virtual address size");
  }
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
          loptions, min_supported_compute_capability_),
      "setting minimum supported CUDA compute capability");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetExitOnError(loptions, exit_on_error_),
      "setting exit on error");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetStrictModelConfig(
          loptions, strict_model_config_),
      "setting strict model configuration");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetStrictReadiness(loptions, strict_readiness_),
      "setting strict readiness");
  // [FIXME] std::max seems to be part of Parse()
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetExitTimeout(
          loptions, std::max(0, exit_timeout_secs_)),
      "setting exit timeout");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(
          loptions, std::max(0, buffer_manager_thread_count_)),
      "setting buffer manager thread count");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetModelLoadThreadCount(
          loptions, std::max(1u, model_load_thread_count_)),
      "setting model load thread count");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetModelLoadRetryCount(
          loptions, std::max(0u, model_load_retry_count_)),
      "setting model load retry count");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetModelNamespacing(
          loptions, enable_model_namespacing_),
      "setting model namespacing");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetEnablePeerAccess(
          loptions, enable_peer_access_),
      "setting peer access");

#ifdef TRITON_ENABLE_LOGGING
  TRITONSERVER_ServerOptionsSetLogFile(loptions, log_file_.c_str());
  THROW_IF_ERR(
      ParseException, TRITONSERVER_ServerOptionsSetLogInfo(loptions, log_info_),
      "setting log info enable");
  THROW_IF_ERR(
      ParseException, TRITONSERVER_ServerOptionsSetLogWarn(loptions, log_warn_),
      "setting log warn enable");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetLogError(loptions, log_error_),
      "setting log error enable");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetLogVerbose(loptions, log_verbose_),
      "setting log verbose level");
  switch (log_format_) {
    case triton::common::Logger::Format::kDEFAULT:
      THROW_IF_ERR(
          ParseException,
          TRITONSERVER_ServerOptionsSetLogFormat(
              loptions, TRITONSERVER_LOG_DEFAULT),
          "setting log format");
      break;
    case triton::common::Logger::Format::kISO8601:
      THROW_IF_ERR(
          ParseException,
          TRITONSERVER_ServerOptionsSetLogFormat(
              loptions, TRITONSERVER_LOG_ISO8601),
          "setting log format");
      break;
  }
#endif  // TRITON_ENABLE_LOGGING

#ifdef TRITON_ENABLE_METRICS
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetMetrics(loptions, allow_metrics_),
      "setting metrics enable");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetGpuMetrics(loptions, allow_gpu_metrics_),
      "setting GPU metrics enable");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetCpuMetrics(loptions, allow_cpu_metrics_),
      "setting CPU metrics enable");
  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetMetricsInterval(
          loptions, metrics_interval_ms_),
      "setting metrics interval");
  for (const auto& mcs : metrics_config_settings_) {
    THROW_IF_ERR(
        ParseException,
        TRITONSERVER_ServerOptionsSetMetricsConfig(
            loptions, std::get<0>(mcs).c_str(), std::get<1>(mcs).c_str(),
            std::get<2>(mcs).c_str()),
        "setting metrics configuration");
  }

#endif  // TRITON_ENABLE_METRICS

  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetBackendDirectory(
          loptions, backend_dir_.c_str()),
      "setting backend directory");

  // Enable cache and configure it if a cache CLI arg is passed,
  // this will allow for an empty configuration.
  if (enable_cache_) {
    THROW_IF_ERR(
        ParseException,
        TRITONSERVER_ServerOptionsSetCacheDirectory(
            loptions, cache_dir_.c_str()),
        "setting cache directory");

    for (const auto& cache_pair : cache_config_settings_) {
      const auto& cache_name = cache_pair.first;
      const auto& settings = cache_pair.second;
      const auto& json_config_str = PairsToJsonStr(settings);
      THROW_IF_ERR(
          ParseException,
          TRITONSERVER_ServerOptionsSetCacheConfig(
              loptions, cache_name.c_str(), json_config_str.c_str()),
          "setting cache configuration");
    }
  }

  THROW_IF_ERR(
      ParseException,
      TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
          loptions, repoagent_dir_.c_str()),
      "setting repository agent directory");
  for (const auto& bcs : backend_config_settings_) {
    THROW_IF_ERR(
        ParseException,
        TRITONSERVER_ServerOptionsSetBackendConfig(
            loptions, std::get<0>(bcs).c_str(), std::get<1>(bcs).c_str(),
            std::get<2>(bcs).c_str()),
        "setting backend configuration");
  }
  for (const auto& limit : load_gpu_limit_) {
    THROW_IF_ERR(
        ParseException,
        TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit(
            loptions, TRITONSERVER_INSTANCEGROUPKIND_GPU, limit.first,
            limit.second),
        "setting model load GPU limit");
  }
  for (const auto& hp : host_policies_) {
    THROW_IF_ERR(
        ParseException,
        TRITONSERVER_ServerOptionsSetHostPolicy(
            loptions, std::get<0>(hp).c_str(), std::get<1>(hp).c_str(),
            std::get<2>(hp).c_str()),
        "setting host policy");
  }
  return managed_ptr;
}

std::pair<TritonServerParameters, std::vector<char*>>
TritonParser::Parse(int argc, char** argv)
{
  //
  // Step 1. Before parsing setup
  //
  TritonServerParameters lparams;
  bool strict_model_config_present{false};
  bool disable_auto_complete_config{false};
  bool cache_size_present{false};
  bool cache_config_present{false};
#ifdef TRITON_ENABLE_TRACING
  bool explicit_disable_trace{false};
  bool trace_filepath_present{false};
  bool trace_level_present{false};
  bool trace_rate_present{false};
  bool trace_count_present{false};
  bool trace_log_frequency_present{false};
#endif  // TRITON_ENABLE_TRACING
  int option_index = 0;

#ifdef TRITON_ENABLE_GRPC
  triton::server::grpc::Options& lgrpc_options = lparams.grpc_options_;
#endif  // TRITON_ENABLE_GRPC

#if defined TRITON_ENABLE_HTTP || defined TRITON_ENABLE_GRPC
  // According to HTTP specification header names are case-insensitive.
  const std::string case_insensitive_prefix{"(?i)"};
#endif  // TRITON_ENABLE_HTTP || TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_VERTEX_AI
  // Set different default value if specific flag is set
  {
    auto aip_mode =
        triton::server::GetEnvironmentVariableOrDefault("AIP_MODE", "");
    // Enable Vertex AI service and disable HTTP / GRPC service by default
    // if detecting Vertex AI environment
    if (aip_mode == "PREDICTION") {
      lparams.allow_vertex_ai_ = true;
#ifdef TRITON_ENABLE_HTTP
      lparams.allow_http_ = false;
#endif  // TRITON_ENABLE_HTTP
#ifdef TRITON_ENABLE_GRPC
      lparams.allow_grpc_ = false;
#endif  // TRITON_ENABLE_GRPC
    }
    auto port = triton::server::GetEnvironmentVariableOrDefault(
        "AIP_HTTP_PORT", "8080");
    lparams.vertex_ai_port_ = ParseOption<int>(port);
  }
#endif  // TRITON_ENABLE_VERTEX_AI

  //
  // Step 2. parse options
  //
  std::vector<struct option> long_options;
  for (const auto& group : option_groups_) {
    for (const auto& o : group.second) {
      long_options.push_back(o.GetLongOption());
    }
  }
  long_options.push_back({nullptr, 0, nullptr, 0});

  int flag;
  while ((flag = getopt_long(
              argc, argv, "", &long_options[0], &option_index)) != -1) {
    try {
      switch (flag) {
        case OPTION_HELP:
          // [FIXME] how help is printed?
        case '?':
          // [FIXME] fall through when seeing this, currently consumes all
          // options [FIXME] disable stderr output of `getopt_long`
          throw ParseException();
#ifdef TRITON_ENABLE_LOGGING
        case OPTION_LOG_VERBOSE:
          lparams.log_verbose_ = ParseIntBoolOption(optarg);
          break;
        case OPTION_LOG_INFO:
          lparams.log_info_ = ParseOption<bool>(optarg);
          break;
        case OPTION_LOG_WARNING:
          lparams.log_warn_ = ParseOption<bool>(optarg);
          break;
        case OPTION_LOG_ERROR:
          lparams.log_error_ = ParseOption<bool>(optarg);
          break;
        case OPTION_LOG_FORMAT: {
          std::string format_str(optarg);
          if (format_str == "default") {
            lparams.log_format_ = triton::common::Logger::Format::kDEFAULT;
          } else if (format_str == "ISO8601") {
            lparams.log_format_ = triton::common::Logger::Format::kISO8601;
          } else {
            throw ParseException("invalid argument for --log-format");
          }
          break;
        }
        case OPTION_LOG_FILE:
          lparams.log_file_ = optarg;
          break;
#endif  // TRITON_ENABLE_LOGGING

        case OPTION_ID:
          lparams.server_id_ = optarg;
          break;
        case OPTION_MODEL_REPOSITORY:
          lparams.model_repository_paths_.insert(optarg);
          break;
        case OPTION_EXIT_ON_ERROR:
          lparams.exit_on_error_ = ParseOption<bool>(optarg);
          break;
        case OPTION_DISABLE_AUTO_COMPLETE_CONFIG:
          disable_auto_complete_config = true;
          break;
        case OPTION_STRICT_MODEL_CONFIG:
          std::cerr << "Warning: '--strict-model-config' has been deprecated! "
                       "Please use '--disable-auto-complete-config' instead."
                    << std::endl;
          strict_model_config_present = true;
          lparams.strict_model_config_ = ParseOption<bool>(optarg);
          break;
        case OPTION_STRICT_READINESS:
          lparams.strict_readiness_ = ParseOption<bool>(optarg);
          break;

#ifdef TRITON_ENABLE_HTTP
        case OPTION_ALLOW_HTTP:
          lparams.allow_http_ = ParseOption<bool>(optarg);
          break;
        case OPTION_HTTP_PORT:
          lparams.http_port_ = ParseOption<int>(optarg);
          break;
        case OPTION_REUSE_HTTP_PORT:
          lparams.reuse_http_port_ = ParseOption<bool>(optarg);
          break;
        case OPTION_HTTP_ADDRESS:
          lparams.http_address_ = optarg;
          break;
        case OPTION_HTTP_HEADER_FORWARD_PATTERN:
          lparams.http_forward_header_pattern_ =
              std::move(case_insensitive_prefix + optarg);
          break;
        case OPTION_HTTP_THREAD_COUNT:
          lparams.http_thread_cnt_ = ParseOption<int>(optarg);
          break;
        case OPTION_HTTP_RESTRICTED_API:
          ParseRestrictedFeatureOption(
              optarg, long_options[option_index].name, "", "api",
              lparams.http_restricted_apis_);
          break;

#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_SAGEMAKER
        case OPTION_ALLOW_SAGEMAKER:
          lparams.allow_sagemaker_ = ParseOption<bool>(optarg);
          break;
        case OPTION_SAGEMAKER_PORT:
          lparams.sagemaker_port_ = ParseOption<int>(optarg);
          break;
        case OPTION_SAGEMAKER_SAFE_PORT_RANGE:
          lparams.sagemaker_safe_range_set_ = true;
          lparams.sagemaker_safe_range_ =
              ParsePairOption<int, int>(optarg, "-");
          break;
        case OPTION_SAGEMAKER_THREAD_COUNT:
          lparams.sagemaker_thread_cnt_ = ParseOption<int>(optarg);
          break;
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
        case OPTION_ALLOW_VERTEX_AI:
          lparams.allow_vertex_ai_ = ParseOption<bool>(optarg);
          break;
        case OPTION_VERTEX_AI_PORT:
          lparams.vertex_ai_port_ = ParseOption<int>(optarg);
          break;
        case OPTION_VERTEX_AI_THREAD_COUNT:
          lparams.vertex_ai_thread_cnt_ = ParseOption<int>(optarg);
          break;
        case OPTION_VERTEX_AI_DEFAULT_MODEL:
          lparams.vertex_ai_default_model_ = optarg;
          break;
#endif  // TRITON_ENABLE_VERTEX_AI

#ifdef TRITON_ENABLE_GRPC
        case OPTION_ALLOW_GRPC:
          lparams.allow_grpc_ = ParseOption<bool>(optarg);
          break;
        case OPTION_GRPC_PORT:
          lgrpc_options.socket_.port_ = ParseOption<int>(optarg);
          break;
        case OPTION_REUSE_GRPC_PORT:
          lgrpc_options.socket_.reuse_port_ = ParseOption<bool>(optarg);
          break;
        case OPTION_GRPC_ADDRESS:
          lgrpc_options.socket_.address_ = optarg;
          break;
        case OPTION_GRPC_INFER_THREAD_COUNT:
          lgrpc_options.infer_thread_count_ = ParseOption<int>(optarg);
          if (lgrpc_options.infer_thread_count_ < 2 ||
              lgrpc_options.infer_thread_count_ > 128) {
            throw ParseException(
                "invalid argument for --grpc_infer_thread_count. Must be in "
                "the range 2 to 128.");
          }
          break;
        case OPTION_GRPC_INFER_ALLOCATION_POOL_SIZE:
          lgrpc_options.infer_allocation_pool_size_ = ParseOption<int>(optarg);
          break;
        case OPTION_GRPC_MAX_RESPONSE_POOL_SIZE:
          lgrpc_options.max_response_pool_size_ = ParseOption<int>(optarg);
          if (lgrpc_options.max_response_pool_size_ <= 0) {
            throw ParseException(
                "Error: --grpc-max-response-pool-size must be greater "
                "than 0.");
          }
          break;
        case OPTION_GRPC_USE_SSL:
          lgrpc_options.ssl_.use_ssl_ = ParseOption<bool>(optarg);
          break;
        case OPTION_GRPC_USE_SSL_MUTUAL:
          lgrpc_options.ssl_.use_mutual_auth_ = ParseOption<bool>(optarg);
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
            throw ParseException(
                "invalid argument for "
                "--grpc_infer_response_compression_level");
          }
          break;
        }
        case OPTION_GRPC_ARG_KEEPALIVE_TIME_MS:
          lgrpc_options.keep_alive_.keepalive_time_ms_ =
              ParseOption<int>(optarg);
          break;
        case OPTION_GRPC_ARG_KEEPALIVE_TIMEOUT_MS:
          lgrpc_options.keep_alive_.keepalive_timeout_ms_ =
              ParseOption<int>(optarg);
          break;
        case OPTION_GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS:
          lgrpc_options.keep_alive_.keepalive_permit_without_calls_ =
              ParseOption<bool>(optarg);
          break;
        case OPTION_GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA:
          lgrpc_options.keep_alive_.http2_max_pings_without_data_ =
              ParseOption<int>(optarg);
          break;
        case OPTION_GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS:
          lgrpc_options.keep_alive_
              .http2_min_recv_ping_interval_without_data_ms_ =
              ParseOption<int>(optarg);
          break;
        case OPTION_GRPC_ARG_HTTP2_MAX_PING_STRIKES:
          lgrpc_options.keep_alive_.http2_max_ping_strikes_ =
              ParseOption<int>(optarg);
          break;
        case OPTION_GRPC_ARG_MAX_CONNECTION_AGE_MS:
          lgrpc_options.keep_alive_.max_connection_age_ms_ =
              ParseOption<int>(optarg);
          break;
        case OPTION_GRPC_ARG_MAX_CONNECTION_AGE_GRACE_MS:
          lgrpc_options.keep_alive_.max_connection_age_grace_ms_ =
              ParseOption<int>(optarg);
          break;
        case OPTION_GRPC_RESTRICTED_PROTOCOL: {
          ParseRestrictedFeatureOption(
              optarg, long_options[option_index].name,
              std::string(
                  triton::server::grpc::kRestrictedProtocolHeaderTemplate),
              "protocol", lgrpc_options.restricted_protocols_);
          break;
        }
        case OPTION_GRPC_HEADER_FORWARD_PATTERN:
          lgrpc_options.forward_header_pattern_ =
              std::move(case_insensitive_prefix + optarg);
          break;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
        case OPTION_ALLOW_METRICS:
          lparams.allow_metrics_ = ParseOption<bool>(optarg);
          break;
        case OPTION_ALLOW_GPU_METRICS:
          lparams.allow_gpu_metrics_ = ParseOption<bool>(optarg);
          break;
        case OPTION_ALLOW_CPU_METRICS:
          lparams.allow_cpu_metrics_ = ParseOption<bool>(optarg);
          break;
        case OPTION_METRICS_ADDRESS:
          lparams.metrics_address_ = optarg;
          break;
        case OPTION_METRICS_PORT:
          lparams.metrics_port_ = ParseOption<int>(optarg);
          break;
        case OPTION_METRICS_INTERVAL_MS:
          lparams.metrics_interval_ms_ = ParseOption<int>(optarg);
          break;
        case OPTION_METRICS_CONFIG:
          lparams.metrics_config_settings_.push_back(
              ParseMetricsConfigOption(optarg));
          break;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_TRACING
        case OPTION_TRACE_FILEPATH: {
          std::cerr << "Warning: '--trace-file' has been deprecated and will be"
                       " removed in future releases. Please use "
                       "'--trace-config triton,file=<filepath> instead."
                    << std::endl;
          trace_filepath_present = true;
          lparams.trace_filepath_ = optarg;
          break;
        }
        case OPTION_TRACE_LEVEL: {
          std::cerr
              << "Warning: '--trace-level' has been deprecated and will be"
                 " removed in future releases. Please use "
                 "'--trace-config level=<OFF|TIMESTAMPS|TENSORS> instead."
              << std::endl;
          trace_level_present = true;
          auto parsed_level = ParseTraceLevelOption(optarg);
          explicit_disable_trace |=
              (parsed_level == TRITONSERVER_TRACE_LEVEL_DISABLED);
          lparams.trace_level_ = static_cast<TRITONSERVER_InferenceTraceLevel>(
              lparams.trace_level_ | parsed_level);
          break;
        }
        case OPTION_TRACE_RATE:
          std::cerr << "Warning: '--trace-rate' has been deprecated and will be"
                       " removed in future releases. Please use "
                       "'--trace-config rate=<rate value> instead."
                    << std::endl;
          trace_rate_present = true;
          lparams.trace_rate_ = ParseOption<int>(optarg);
          break;

        case OPTION_TRACE_COUNT:
          std::cerr
              << "Warning: '--trace-count' has been deprecated and will be"
                 " removed in future releases. Please use "
                 "'--trace-config count=<count value> instead."
              << std::endl;
          trace_count_present = true;
          lparams.trace_count_ = ParseOption<int>(optarg);
          break;
        case OPTION_TRACE_LOG_FREQUENCY:
          std::cerr
              << "Warning: '--trace-log-frequency' has been deprecated and "
                 "will be"
                 " removed in future releases. Please use "
                 "'--trace-config triton,log-frequency=<log frequency "
                 "value> instead."
              << std::endl;
          trace_log_frequency_present = true;
          lparams.trace_log_frequency_ = ParseOption<int>(optarg);
          break;
        case OPTION_TRACE_CONFIG: {
          auto trace_config_setting = ParseTraceConfigOption(optarg);
          triton::server::TraceConfig& tc =
              lparams
                  .trace_config_map_[std::get<0>(trace_config_setting).c_str()];
          tc.push_back(std::make_pair(
              std::get<1>(trace_config_setting).c_str(),
              std::get<2>(trace_config_setting).c_str()));
          break;
        }
#endif  // TRITON_ENABLE_TRACING

        case OPTION_POLL_REPO_SECS:
          lparams.repository_poll_secs_ = ParseOption<int>(optarg);
          break;
        case OPTION_STARTUP_MODEL:
          lparams.startup_models_.insert(optarg);
          break;
        case OPTION_CUSTOM_MODEL_CONFIG_NAME:
          if (std::strlen(optarg) == 0) {
            throw ParseException(
                "Error: empty argument for --model-config-name");
          }
          lparams.model_config_name_ = optarg;
          break;
        case OPTION_MODEL_CONTROL_MODE: {
          std::string mode_str(optarg);
          std::transform(
              mode_str.begin(), mode_str.end(), mode_str.begin(), ::tolower);
          if (mode_str == "none") {
            lparams.control_mode_ = TRITONSERVER_MODEL_CONTROL_NONE;
          } else if (mode_str == "poll") {
            lparams.control_mode_ = TRITONSERVER_MODEL_CONTROL_POLL;
          } else if (mode_str == "explicit") {
            lparams.control_mode_ = TRITONSERVER_MODEL_CONTROL_EXPLICIT;
          } else {
            throw ParseException("invalid argument for --model-control-mode");
          }
          break;
        }
        case OPTION_RATE_LIMIT: {
          std::string rate_limit_str(optarg);
          std::transform(
              rate_limit_str.begin(), rate_limit_str.end(),
              rate_limit_str.begin(), ::tolower);
          if (rate_limit_str == "execution_count") {
            lparams.rate_limit_mode_ = TRITONSERVER_RATE_LIMIT_EXEC_COUNT;
          } else if (rate_limit_str == "off") {
            lparams.rate_limit_mode_ = TRITONSERVER_RATE_LIMIT_OFF;
          } else {
            throw ParseException("invalid argument for --rate-limit");
          }
          break;
        }
        case OPTION_RATE_LIMIT_RESOURCE: {
          std::string rate_limit_resource_str(optarg);
          std::transform(
              rate_limit_resource_str.begin(), rate_limit_resource_str.end(),
              rate_limit_resource_str.begin(), ::tolower);
          lparams.rate_limit_resources_.push_back(
              ParseRateLimiterResourceOption(optarg));
          break;
        }
        case OPTION_PINNED_MEMORY_POOL_BYTE_SIZE:
          lparams.pinned_memory_pool_byte_size_ = ParseOption<int64_t>(optarg);
          break;
        case OPTION_CUDA_MEMORY_POOL_BYTE_SIZE:
          lparams.cuda_pools_.push_back(
              ParsePairOption<int, uint64_t>(optarg, ":"));
          break;
        case OPTION_CUDA_VIRTUAL_ADDRESS_SIZE:
          lparams.cuda_virtual_address_size_.push_back(
              ParsePairOption<int, size_t>(optarg, ":"));
          break;
        case OPTION_RESPONSE_CACHE_BYTE_SIZE: {
          cache_size_present = true;
          const auto byte_size = std::to_string(ParseOption<int64_t>(optarg));
          lparams.cache_config_settings_["local"] = {{"size", byte_size}};
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
          lparams.cache_config_settings_[cache_name].push_back({key, value});
          break;
        }
        case OPTION_CACHE_DIR:
          lparams.cache_dir_ = optarg;
          break;
        case OPTION_MIN_SUPPORTED_COMPUTE_CAPABILITY:
          lparams.min_supported_compute_capability_ =
              ParseOption<double>(optarg);
          break;
        case OPTION_EXIT_TIMEOUT_SECS:
          lparams.exit_timeout_secs_ = ParseOption<int>(optarg);
          break;
        case OPTION_BACKEND_DIR:
          lparams.backend_dir_ = optarg;
          break;
        case OPTION_REPOAGENT_DIR:
          lparams.repoagent_dir_ = optarg;
          break;
        case OPTION_BUFFER_MANAGER_THREAD_COUNT:
          lparams.buffer_manager_thread_count_ = ParseOption<int>(optarg);
          break;
        case OPTION_MODEL_LOAD_THREAD_COUNT:
          lparams.model_load_thread_count_ = ParseOption<int>(optarg);
          break;
        case OPTION_MODEL_LOAD_RETRY_COUNT:
          lparams.model_load_retry_count_ = ParseOption<int>(optarg);
          break;
        case OPTION_BACKEND_CONFIG:
          lparams.backend_config_settings_.push_back(
              ParseBackendConfigOption(optarg));
          break;
        case OPTION_HOST_POLICY:
          lparams.host_policies_.push_back(ParseHostPolicyOption(optarg));
          break;
        case OPTION_MODEL_LOAD_GPU_LIMIT:
          lparams.load_gpu_limit_.emplace(
              ParsePairOption<int, double>(optarg, ":"));
          break;
        case OPTION_MODEL_NAMESPACING:
          lparams.enable_model_namespacing_ = ParseOption<bool>(optarg);
          break;
        case OPTION_ENABLE_PEER_ACCESS:
          lparams.enable_peer_access_ = ParseOption<bool>(optarg);
          break;
      }
    }
    catch (const ParseException& pe) {
      if ((pe.what() != NULL) && (strlen(pe.what()) != 0)) {
        std::stringstream ss;
        ss << "Bad option: \"--" << long_options[option_index].name << "\".\n"
           << pe.what() << std::endl;
        throw ParseException(ss.str());
      } else {
        // In case of `Unrecognized option` or `Help` option, just throw a
        // ParseException
        throw ParseException();
      }
    }
  }

  if (optind < argc) {
    throw ParseException(std::string("Unexpected argument: ") + argv[optind]);
  }

  //
  // Step 3. Post parsing validation, usually for options that depend on the
  // others which are not determined until after parsing.
  //

  if (lparams.control_mode_ != TRITONSERVER_MODEL_CONTROL_POLL) {
    lparams.repository_poll_secs_ = 0;
  }

  if (lparams.startup_models_.size() > 0 &&
      lparams.control_mode_ != TRITONSERVER_MODEL_CONTROL_EXPLICIT) {
    throw ParseException(
        "Error: Use of '--load-model' requires setting "
        "'--model-control-mode=explicit' as well.");
  }


#ifdef TRITON_ENABLE_VERTEX_AI
  // Set default model repository if specific flag is set, postpone the
  // check to after parsing so we only monitor the default repository if
  // Vertex service is allowed
  if (lparams.model_repository_paths_.empty()) {
    auto aip_storage_uri =
        triton::server::GetEnvironmentVariableOrDefault("AIP_STORAGE_URI", "");
    if (!aip_storage_uri.empty()) {
      lparams.model_repository_paths_.insert(aip_storage_uri);
    }
  }
#endif  // TRITON_ENABLE_VERTEX_AI

#ifdef TRITON_ENABLE_METRICS
  lparams.allow_gpu_metrics_ &= lparams.allow_metrics_;
  lparams.allow_cpu_metrics_ &= lparams.allow_metrics_;
  // Set metrics_address to default if never specified
  if (lparams.metrics_address_.empty()) {
#ifdef TRITON_ENABLE_HTTP
    // If built with HTTP support, default to HTTP address
    lparams.metrics_address_ = lparams.http_address_;
#else
    // Otherwise have default for builds without HTTP support
    lparams.metrics_address_ = "0.0.0.0";
#endif  // TRITON_ENABLE_HTTP
  }
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_TRACING
  PostProcessTraceArgs(
      lparams, trace_level_present, trace_rate_present, trace_count_present,
      trace_filepath_present, trace_log_frequency_present,
      explicit_disable_trace);
#endif  // TRITON_ENABLE_TRACING

  // Check if there is a conflict between --disable-auto-complete-config
  // and --strict-model-config
  if (disable_auto_complete_config) {
    if (strict_model_config_present && !lparams.strict_model_config_) {
      std::cerr
          << "Warning: Overriding deprecated '--strict-model-config' from "
             "False to True in favor of '--disable-auto-complete-config'!"
          << std::endl;
    }
    lparams.strict_model_config_ = true;
  }

  // Check if there is a conflict between --response-cache-byte-size
  // and --cache-config
  if (cache_size_present && cache_config_present) {
    throw ParseException(
        "Error: Incompatible flags --response-cache-byte-size and "
        "--cache-config both provided. Please provide one or the other.");
  }
  lparams.enable_cache_ = (cache_size_present || cache_config_present);
  return {lparams, {}};
}

std::string
TritonParser::FormatUsageMessage(std::string str, int offset)
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
TritonParser::Usage()
{
  std::stringstream ss;
  for (const auto& group : option_groups_) {
    if (!group.first.empty() && !group.second.empty()) {
      ss << std::endl << group.first << ":" << std::endl;
    }

    for (const auto& o : group.second) {
      if (!o.arg_desc_.empty()) {
        ss << "  --" << o.flag_ << " <" << o.arg_desc_ << ">" << std::endl
           << "\t" << FormatUsageMessage(o.desc_, 0) << std::endl;
      } else {
        ss << "  --" << o.flag_ << std::endl
           << "\t" << FormatUsageMessage(o.desc_, 0) << std::endl;
      }
    }
  }
  return ss.str();
}

std::tuple<std::string, std::string, std::string>
TritonParser::ParseMetricsConfigOption(const std::string& arg)
{
  // Format is "<setting>=<value>" for generic configs/settings
  int delim_setting = arg.find("=");
  if (delim_setting < 0) {
    std::stringstream ss;
    ss << "--metrics-config option format is "
       << "<setting>=<value>. Got " << arg << std::endl;
    throw ParseException(ss.str());
  }

  // Break section before "=" into substr to avoid matching commas
  // in setting values.
  auto name_substr = arg.substr(0, delim_setting);
  int delim_name = name_substr.find(",");

  // No name-specific configs currently supported, though it may be in
  // the future. Map global configs to empty string like other configs for
  // now.
  std::string name_string = std::string();
  if (delim_name >= 0) {
    std::stringstream ss;
    ss << "--metrics-config option format is "
       << "<setting>=<value>. Got " << arg << std::endl;
    throw ParseException(ss.str());
  }  // else global metrics config

  std::string setting_string =
      arg.substr(delim_name + 1, delim_setting - delim_name - 1);
  std::string value_string = arg.substr(delim_setting + 1);

  if (setting_string.empty() || value_string.empty()) {
    std::stringstream ss;
    ss << "--metrics-config option format is "
       << "<setting>=<value>. Got " << arg << std::endl;
    throw ParseException(ss.str());
  }

  return {name_string, setting_string, value_string};
}

std::tuple<std::string, std::string, std::string>
TritonParser::ParseCacheConfigOption(const std::string& arg)
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
    std::stringstream ss;
    ss << "No cache specified. --cache-config option format is "
       << "<cache name>,<setting>=<value>. Got " << arg << std::endl;
    throw ParseException(ss.str());
  }

  if (delim_setting < 0) {
    std::stringstream ss;
    ss << "--cache-config option format is '<cache "
          "name>,<setting>=<value>'. Got "
       << arg << std::endl;
    throw ParseException(ss.str());
  }
  std::string setting_string =
      arg.substr(delim_name + 1, delim_setting - delim_name - 1);
  std::string value_string = arg.substr(delim_setting + 1);

  if (setting_string.empty() || value_string.empty()) {
    std::stringstream ss;
    ss << "--cache-config option format is '<cache "
          "name>,<setting>=<value>'. Got "
       << arg << std::endl;
    throw ParseException(ss.str());
  }

  return {name_string, setting_string, value_string};
}

std::tuple<std::string, int, int>
TritonParser::ParseRateLimiterResourceOption(const std::string& arg)
{
  std::string error_string(
      "--rate-limit-resource option format is "
      "'<resource_name>:<count>:<device>' or '<resource_name>:<count>'. "
      "Got " +
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
      throw ParseException(error_string);
    }
    name_string = arg.substr(0, delim_first);
    count = ParseOption<int>(
        arg.substr(delim_first + 1, delim_second - delim_first - 1));
    device_id = ParseOption<int>(arg.substr(delim_second + 1));
  } else if (delim_first != std::string::npos) {
    // Handle format `<resource_name>:<count>'
    name_string = arg.substr(0, delim_first);
    count = ParseOption<int>(arg.substr(delim_first + 1));
  } else {
    // If no colons found
    throw ParseException(error_string);
  }

  return {name_string, count, device_id};
}

std::tuple<std::string, std::string, std::string>
TritonParser::ParseBackendConfigOption(const std::string& arg)
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
    std::stringstream ss;
    ss << "No backend specified. --backend-config option format is "
       << "<backend name>,<setting>=<value> or "
       << "<setting>=<value>. Got " << arg << std::endl;
    throw ParseException(ss.str());
  }  // else global backend config

  if (delim_setting < 0) {
    std::stringstream ss;
    ss << "--backend-config option format is '<backend "
          "name>,<setting>=<value>'. Got "
       << arg << std::endl;
    throw ParseException(ss.str());
  }
  std::string setting_string =
      arg.substr(delim_name + 1, delim_setting - delim_name - 1);
  std::string value_string = arg.substr(delim_setting + 1);

  if (setting_string.empty() || value_string.empty()) {
    std::stringstream ss;
    ss << "--backend-config option format is '<backend "
          "name>,<setting>=<value>'. Got "
       << arg << std::endl;
    throw ParseException(ss.str());
  }

  return {name_string, setting_string, value_string};
}

void
TritonParser::ParseRestrictedFeatureOption(
    const std::string& arg, const std::string& option_name,
    const std::string& key_prefix, const std::string& feature_type,
    RestrictedFeatures& restricted_features)
{
  const auto& parsed_tuple =
      ParseGenericConfigOption(arg, ":", "=", option_name, "config name");

  const auto& features = SplitOptions(std::get<0>(parsed_tuple), ",");
  const auto& key = std::get<1>(parsed_tuple);
  const auto& value = std::get<2>(parsed_tuple);

  for (const auto& feature : features) {
    const auto& category = RestrictedFeatures::ToCategory(feature);

    if (category == RestrictedCategory::INVALID) {
      std::stringstream ss;
      ss << "unknown restricted " << feature_type << " '" << feature << "' "
         << std::endl;
      throw ParseException(ss.str());
    }

    if (restricted_features.IsRestricted(category)) {
      // restricted feature can only be in one group
      std::stringstream ss;
      ss << "restricted " << feature_type << " '" << feature
         << "' can not be specified in multiple config groups" << std::endl;
      throw ParseException(ss.str());
    }
    restricted_features.Insert(
        category, std::make_pair(key_prefix + key, value));
  }
}

std::tuple<std::string, std::string, std::string>
TritonParser::ParseHostPolicyOption(const std::string& arg)
{
  return ParseGenericConfigOption(arg, ",", "=", "host-policy", "policy name");
}

std::tuple<std::string, std::string, std::string>
TritonParser::ParseGenericConfigOption(
    const std::string& arg, const std::string& first_delim,
    const std::string& second_delim, const std::string& option_name,
    const std::string& config_name)
{
  // Format is "<string>,<string>=<string>"
  int delim_name = arg.find(first_delim);
  int delim_setting = arg.find(second_delim, delim_name + 1);

  std::string error_string = "--" + option_name + " option format is '<" +
                             config_name + ">" + first_delim + "<setting>" +
                             second_delim + "<value>'. Got " + arg + "\n";

  // Check for 2 semicolons
  if ((delim_name < 0) || (delim_setting < 0)) {
    throw ParseException(error_string);
  }

  std::string name_string = arg.substr(0, delim_name);
  std::string setting_string =
      arg.substr(delim_name + 1, delim_setting - delim_name - 1);
  std::string value_string = arg.substr(delim_setting + 1);

  if (name_string.empty() || setting_string.empty() || value_string.empty()) {
    throw ParseException(error_string);
  }

  return {name_string, setting_string, value_string};
}

#ifdef TRITON_ENABLE_TRACING
TRITONSERVER_InferenceTraceLevel
TritonParser::ParseTraceLevelOption(std::string arg)
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

  throw ParseException("invalid value for trace level option: " + arg);
}

InferenceTraceMode
TritonParser::ParseTraceModeOption(std::string arg)
{
  std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  if (arg == "triton") {
    return TRACE_MODE_TRITON;
  }
  if (arg == "opentelemetry") {
    return TRACE_MODE_OPENTELEMETRY;
  }

  throw ParseException(
      "invalid value for trace mode option: " + arg +
      ". Available options are \"triton\" and \"opentelemetry\"");
}

std::tuple<std::string, std::string, std::string>
TritonParser::ParseTraceConfigOption(const std::string& arg)
{
  int delim_name = arg.find(",");
  int delim_setting = arg.find("=", delim_name + 1);

  std::string name_string = std::string();
  if (delim_name > 0) {
    name_string =
        std::to_string(ParseTraceModeOption(arg.substr(0, delim_name)));
  } else if (delim_name == 0) {
    std::stringstream ss;
    ss << "No trace mode specified. --trace-config option format is "
       << "<trace mode>,<setting>=<value> or "
       << "<setting>=<value>. Got " << arg << std::endl;
    throw ParseException(ss.str());
  }  // else global trace config

  if (delim_setting < 0) {
    std::stringstream ss;
    ss << "--trace-config option format is '<trace mode>,<setting>=<value>'. "
          "Got "
       << arg << std::endl;
    throw ParseException(ss.str());
  }
  std::string setting_string =
      arg.substr(delim_name + 1, delim_setting - delim_name - 1);
  std::string value_string = arg.substr(delim_setting + 1);

  if (setting_string.empty() || value_string.empty()) {
    std::stringstream ss;
    ss << "--trace-config option format is '<trace mode>,<setting>=<value>'. "
          "Got "
       << arg << std::endl;
    throw ParseException(ss.str());
  }

  return {name_string, setting_string, value_string};
}

void
TritonParser::SetGlobalTraceArgs(
    TritonServerParameters& lparams, bool trace_level_present,
    bool trace_rate_present, bool trace_count_present,
    bool explicit_disable_trace)
{
  for (const auto& [setting, value_variant] : lparams.trace_config_map_[""]) {
    auto value = std::get<std::string>(value_variant);
    try {
      if (setting == "rate") {
        if (trace_rate_present) {
          std::cerr << "Warning: Overriding deprecated '--trace-rate' "
                       "in favor of provided rate value in --trace-config!"
                    << std::endl;
        }
        lparams.trace_rate_ = ParseOption<int>(value);
      }
      if (setting == "level") {
        if (trace_level_present) {
          std::cerr << "Warning: Overriding deprecated '--trace-level' "
                       "in favor of provided level in --trace-config!"
                    << std::endl;
        }
        auto parsed_level_config = ParseTraceLevelOption(value);
        explicit_disable_trace |=
            (parsed_level_config == TRITONSERVER_TRACE_LEVEL_DISABLED);
        lparams.trace_level_ = static_cast<TRITONSERVER_InferenceTraceLevel>(
            lparams.trace_level_ | parsed_level_config);
      }
      if (setting == "mode") {
        lparams.trace_mode_ = ParseTraceModeOption(value);
      }
      if (setting == "count") {
        if (trace_count_present) {
          std::cerr << "Warning: Overriding deprecated '--trace-count' "
                       "in favor of provided count in --trace-config!"
                    << std::endl;
        }
        lparams.trace_count_ = ParseOption<int>(value);
      }
    }
    catch (const ParseException& pe) {
      std::stringstream ss;
      ss << "Bad option: \"--trace-config " << setting << "\".\n"
         << pe.what() << std::endl;
      throw ParseException(ss.str());
    }
  }
}

void
TritonParser::SetTritonTraceArgs(
    TritonServerParameters& lparams, bool trace_filepath_present,
    bool trace_log_frequency_present)
{
  for (const auto& [setting, value_variant] :
       lparams.trace_config_map_[std::to_string(TRACE_MODE_TRITON)]) {
    auto value = std::get<std::string>(value_variant);
    try {
      if (setting == "file") {
        if (trace_filepath_present) {
          std::cerr << "Warning: Overriding deprecated '--trace-file' "
                       "in favor of provided file in --trace-config!"
                    << std::endl;
        }
        lparams.trace_filepath_ = value;
      } else if (setting == "log-frequency") {
        if (trace_log_frequency_present) {
          std::cerr << "Warning: Overriding deprecated '--trace-log-frequency' "
                       "in favor of provided log-frequency in --trace-config!"
                    << std::endl;
        }
        lparams.trace_log_frequency_ = ParseOption<int>(value);
      }
    }
    catch (const ParseException& pe) {
      std::stringstream ss;
      ss << "Bad option: \"--trace-config triton," << setting << "\".\n"
         << pe.what() << std::endl;
      throw ParseException(ss.str());
    }
  }
}

void
TritonParser::SetOpenTelemetryTraceArgs(
    TritonServerParameters& lparams, bool trace_filepath_present,
    bool trace_log_frequency_present)
{
  if (trace_filepath_present) {
    std::cerr << "Warning: '--trace-file' is deprecated and will "
                 "be ignored with opentelemetry tracing mode. "
              << std::endl;
  }
  if (trace_log_frequency_present) {
    std::cerr << "Warning: '--trace-log-frequency' is deprecated "
                 "and will be ignored with opentelemetry tracing mode."
              << std::endl;
  }
  triton::server::TraceConfig& otel_trace_settings =
      lparams.trace_config_map_[std::to_string(TRACE_MODE_OPENTELEMETRY)];
  ProcessOpenTelemetryBatchSpanProcessorArgs(otel_trace_settings);
}

void
TritonParser::ProcessOpenTelemetryBatchSpanProcessorArgs(
    TraceConfig& otel_trace_settings)
{
  std::unordered_map<std::string, std::string> otel_bsp_default_settings = {};
  // Set up default BatchSpanProcessor parameters, or use
  // parameters, specified by environment variables
  auto env_bsp_max_queue_size = triton::server::GetEnvironmentVariableOrDefault(
      "OTEL_BSP_MAX_QUEUE_SIZE", "2048");
  otel_bsp_default_settings.insert(std::make_pair(
      std::string("bsp_max_queue_size"), env_bsp_max_queue_size));
  auto env_bsp_schedule_delay = triton::server::GetEnvironmentVariableOrDefault(
      "OTEL_BSP_SCHEDULE_DELAY", "5000");
  otel_bsp_default_settings.insert(std::make_pair(
      std::string("bsp_schedule_delay"), env_bsp_schedule_delay));
  auto env_bsp_max_export_batch_size =
      triton::server::GetEnvironmentVariableOrDefault(
          "OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "512");
  otel_bsp_default_settings.insert(std::make_pair(
      std::string("bsp_max_export_batch_size"), env_bsp_max_export_batch_size));

  // Process cmd args and convert string arguments to integers.
  // Throw a ParseException for invalid arguments
  for (auto& [setting, value_variant] : otel_trace_settings) {
    try {
      auto value = std::get<std::string>(value_variant);
      if (setting == "bsp_max_queue_size") {
        value_variant = ParseOption<uint32_t>(value);
        otel_bsp_default_settings.erase("bsp_max_queue_size");
      } else if (setting == "bsp_schedule_delay") {
        value_variant = ParseOption<uint32_t>(value);
        otel_bsp_default_settings.erase("bsp_schedule_delay");
      } else if (setting == "bsp_max_export_batch_size") {
        value_variant = ParseOption<uint32_t>(value);
        otel_bsp_default_settings.erase("bsp_max_export_batch_size");
      }
    }
    catch (const ParseException& pe) {
      std::stringstream ss;
      ss << "Bad option: \"--trace-config opentelemetry," << setting << "\".\n"
         << pe.what() << std::endl;
      throw ParseException(ss.str());
    }
  }
  // If not all BSP settings were provided through cmd,
  // populate OpenTelemetry's trace settings with the default value.
  if (!otel_bsp_default_settings.empty()) {
    for (const auto& [setting, value] : otel_bsp_default_settings) {
      try {
        otel_trace_settings.push_back(
            std::make_pair(setting, ParseOption<uint32_t>(value)));
      }
      catch (const ParseException& pe) {
        std::stringstream ss;
        ss << "Bad option: \"OTEL_";
        for (auto& ch : setting) {
          ss << static_cast<char>(std::toupper(ch));
        }
        ss << "\".\n" << pe.what() << std::endl;
        throw ParseException(ss.str());
      }
    }
  }
}

void
TritonParser::PostProcessTraceArgs(
    TritonServerParameters& lparams, bool trace_level_present,
    bool trace_rate_present, bool trace_count_present,
    bool trace_filepath_present, bool trace_log_frequency_present,
    bool explicit_disable_trace)
{
  SetGlobalTraceArgs(
      lparams, trace_level_present, trace_rate_present, trace_count_present,
      explicit_disable_trace);

  if (lparams.trace_mode_ == TRACE_MODE_OPENTELEMETRY) {
    SetOpenTelemetryTraceArgs(
        lparams, trace_filepath_present, trace_log_frequency_present);
  } else if (lparams.trace_mode_ == TRACE_MODE_TRITON) {
    SetTritonTraceArgs(
        lparams, trace_filepath_present, trace_log_frequency_present);
  }

  if (explicit_disable_trace) {
    lparams.trace_level_ = TRITONSERVER_TRACE_LEVEL_DISABLED;
  }
}

#endif  // TRITON_ENABLE_TRACING
}}      // namespace triton::server
