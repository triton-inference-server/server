// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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


#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <src/custom/sdk/error_codes.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Fix include path for protobuf
#include "python_host.grpc.pb.h"
#include "src/backends/backend/tritonbackend.h"
#include "src/core/json.h"
#include "src/core/model_config.h"

namespace nic = nvidia::inferenceserver::custom;
namespace ni = nvidia::inferenceserver;


namespace {

#define SHOW_INPUT_JSON_PYTHON_BACKEND 0

namespace nvidia { namespace inferenceserver { namespace backend {

#define TRITON_MSG_LOG(MSG)                        \
  do {                                             \
    TRITONSERVER_LogMessage(                       \
        TRITONSERVER_LOG_INFO, __FILE__, __LINE__, \
        std::string(MSG + '\n').c_str());          \
  } while (false)

#define LOG_IF_ERROR(X, MSG)                                                   \
  do {                                                                         \
    TRITONSERVER_Error* lie_err__ = (X);                                       \
    if (lie_err__ != nullptr) {                                                \
      TRITONSERVER_LogMessage(                                                 \
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,                           \
          (std::string(MSG) + ": " + TRITONSERVER_ErrorCodeString(lie_err__) + \
           " - " + TRITONSERVER_ErrorMessage(lie_err__))                       \
              .c_str());                                                       \
      TRITONSERVER_ErrorDelete(lie_err__);                                     \
    }                                                                          \
  } while (false)

#define RETURN_ERROR_IF_FALSE(P, C, MSG)              \
  do {                                                \
    if (!(P)) {                                       \
      return TRITONSERVER_ErrorNew(C, (MSG).c_str()); \
    }                                                 \
  } while (false)

#define RETURN_IF_ERROR(X)               \
  do {                                   \
    TRITONSERVER_Error* rie_err__ = (X); \
    if (rie_err__ != nullptr) {          \
      return rie_err__;                  \
    }                                    \
  } while (false)

#define LOG_MESSAGE(LEVEL, MSG)                                  \
  do {                                                           \
    LOG_IF_ERROR(                                                \
        TRITONSERVER_LogMessage(LEVEL, __FILE__, __LINE__, MSG), \
        ("failed to log message: "));                            \
  } while (false)

#ifdef TRITON_ENABLE_STATS
#define TIMESPEC_TO_NANOS(TS) ((TS).tv_sec * 1000000000 + (TS).tv_nsec)
#define SET_TIMESTAMP(TS_NS)             \
  {                                      \
    struct timespec ts;                  \
    clock_gettime(CLOCK_MONOTONIC, &ts); \
    TS_NS = TIMESPEC_TO_NANOS(ts);       \
  }
#define DECL_TIMESTAMP(TS_NS) \
  uint64_t TS_NS;             \
  SET_TIMESTAMP(TS_NS);
#else
#define DECL_TIMESTAMP(TS_NS)
#define SET_TIMESTAMP(TS_NS)
#endif  // TRITON_ENABLE_STATS

/// Convenience deleter for TRITONBACKEND_ResponseFactory.
struct ResponseFactoryDeleter {
  void operator()(TRITONBACKEND_ResponseFactory* f)
  {
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseFactoryDelete(f),
        "failed deleting response factory");
  }
};

TRITONSERVER_Error* ParseShape(
    ni::TritonJson::Value& io, const std::string& name,
    std::vector<int64_t>* shape);

TRITONSERVER_Error*
ParseShape(
    ni::TritonJson::Value& io, const std::string& name,
    std::vector<int64_t>* shape)
{
  ni::TritonJson::Value shape_array;
  RETURN_IF_ERROR(io.MemberAsArray(name.c_str(), &shape_array));
  for (size_t i = 0; i < shape_array.ArraySize(); ++i) {
    int64_t d = 0;
    RETURN_IF_ERROR(shape_array.IndexAsInt(i, &d));
    shape->push_back(d);
  }

  return nullptr;
}

}}}  // namespace nvidia::inferenceserver::backend

namespace nib = nvidia::inferenceserver::backend;

#define LOG_ERROR std::cerr
#define LOG_INFO std::cout

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)


#define RESPOND_AND_RETURN_IF_ERROR(REQUEST, X)                         \
  do {                                                                  \
    TRITONSERVER_Error* rarie_err__ = (X);                              \
    if (rarie_err__ != nullptr) {                                       \
      TRITONBACKEND_Response* rarie_response__ = nullptr;               \
      LOG_IF_ERROR(                                                     \
          TRITONBACKEND_ResponseNew(&rarie_response__, REQUEST),        \
          "failed to create response");                                 \
      if (rarie_response__ != nullptr) {                                \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                rarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                rarie_err__),                                           \
            "failed to send error response");                           \
      }                                                                 \
      TRITONSERVER_ErrorDelete(rarie_err__);                            \
      return nic::ErrorCodes::Unknown;                                  \
    }                                                                   \
  } while (false)

class ModelState;

class ModelInstanceState {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance,
      ModelInstanceState** model_instance_state);

  /// Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  /// Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  ~ModelInstanceState();

  TRITONSERVER_Error* ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*> responses);

  /// Create a python child process running startup.py
  void CreatePythonInterpreter();

  int GetInputTensor(
      const uint32_t iidx, TRITONBACKEND_Request* request,
      ni::InferenceData* input_tensor,
      std::vector<TRITONBACKEND_Response*>& responses, size_t r,
      uint32_t& batch_size);

  // TODO: Create getter and setters
  std::unique_ptr<ni::PythonInterpreter::Stub> stub;

 private:
  ModelInstanceState(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance,
      const char* name, const TRITONSERVER_InstanceGroupKind kind,
      const int32_t device_id, ni::TritonJson::Value&& model_config);

  int ConnectPythonInterpreter(const std::string& module_path);


  std::string pymodule_path_;
  std::string pyinterpreter_;
  ModelState* model_state_;
  std::string domain_socket_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;

 public:
  ni::TritonJson::Value model_config;

 private:
  pid_t interpreter_pid_;
};

class ModelState {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Get the name and version of the model.
  const std::string& ModelName() const { return model_name_; }
  uint64_t ModelVersion() const { return model_version_; }

  // Does this model support batching in the first dimension.
  TRITONSERVER_Error* SupportsFirstDimBatching(bool* supports);

 private:
  ModelState(
      TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
      const char* model_name, const uint64_t model_version,
      ni::TritonJson::Value&& model_config);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string model_name_;
  const uint64_t model_version_;
  ni::TritonJson::Value model_config_;
};

void
ModelInstanceState::CreatePythonInterpreter()
{
  std::string module_path;

  const char* subinterpreter_commandline[] = {
      nullptr, nullptr,           "--socket", nullptr, "--model_path",
      nullptr, "--instance_name", nullptr,    nullptr};

  constexpr int max_tmpfile_name = 255;
  char tmp_socket_name[max_tmpfile_name], full_socket_name[max_tmpfile_name];
  if (!tmpnam(tmp_socket_name)) {
    TRITON_MSG_LOG("Failed to create a temporary socket name");
  } else {
    snprintf(full_socket_name, max_tmpfile_name, "unix://%s", tmp_socket_name);
    subinterpreter_commandline[3] = full_socket_name;
    domain_socket_ = std::string(full_socket_name);
  }

  ni::TritonJson::Value param_json, module_path_json;
  LOG_IF_ERROR(
      model_config.MemberAsObject("parameters", &param_json),
      "can't get param json value");
  LOG_IF_ERROR(
      param_json.MemberAsObject("module_path", &module_path_json),
      "can't get module path json value");

  LOG_IF_ERROR(
      module_path_json.MemberAsString("string_value", &module_path),
      "can't get module path");

  pymodule_path_ = module_path;
  interpreter_pid_ = fork();

  if (interpreter_pid_ == 0) {
    // Use the python available in $PATH
    // TODO: Make this overridable by config
    std::string python_interpreter_path =
        "/workspace/src/custom/python/install/env/bin/python";
    std::string python_interpreter_startup =
        "/workspace/src/custom/python/install/startup.py";

    subinterpreter_commandline[0] = python_interpreter_path.c_str();
    subinterpreter_commandline[1] = python_interpreter_startup.c_str();
    subinterpreter_commandline[5] = pymodule_path_.c_str();
    subinterpreter_commandline[7] = name_.c_str();
    if (execve(
            subinterpreter_commandline[0], (char**)subinterpreter_commandline,
            nullptr) == -1) {
      LOG_ERROR << "Cannot run interpreter host. Errno = " << errno << '\n'
                << "python_interpreter_path: " << python_interpreter_path
                << '\n'
                << "python_interpreter_startup: " << python_interpreter_startup
                << '\n'
                << "pymodule_path_: " << pymodule_path_ << '\n'
                << "instance_name: " << name_ << '\n';
    }
  } else {
    ConnectPythonInterpreter(module_path);
  }
}

int
ModelInstanceState::ConnectPythonInterpreter(const std::string& module_path)
{
  auto grpc_channel =
      grpc::CreateChannel(domain_socket_, grpc::InsecureChannelCredentials());
  stub.reset(new ni::PythonInterpreter::Stub(grpc_channel));


  std::shared_ptr<ni::InitializationCommand> initialization_params(
      new ni::InitializationCommand());

  std::vector<std::string> keys;
  LOG_IF_ERROR(model_config.Members(&keys), "can't get key names");
  std::string val;

  const auto insert_model_param =
      [&initialization_params](const std::string& val, const std::string& key) {
        auto* value_pair = initialization_params->add_model_command();
        value_pair->set_key(key);
        value_pair->set_value(val);
      };

  insert_model_param("module_path", module_path);

  // Attempting to connect to the python runtime
  constexpr uint8_t conn_attempts = 5;
  for (int i = 0; i < conn_attempts; ++i) {
    grpc::ClientContext context;
    ni::StatusCode result;
    const auto err =
        stub->InterpreterInit(&context, *initialization_params, &result);
    if (err.ok() && result.code() == 0) {
      return nic::ErrorCodes::Success;
    } else {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  std::this_thread::sleep_for(std::chrono::seconds(5));
  grpc::ClientContext context;
  ni::StatusCode result;
  const auto err =
      stub->InterpreterInit(&context, *initialization_params, &result);
  if (err.ok() && result.code() == 0) {
    return nic::ErrorCodes::Success;
  }
  return nic::ErrorCodes::CreationFailure;
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id, ni::TritonJson::Value&& model_config)
    : model_state_(model_state), triton_model_instance_(triton_model_instance),
      name_(name), kind_(kind), device_id_(device_id),
      model_config(std::move(model_config))
{
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  const char* instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  TRITONBACKEND_Model* triton_model;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceModel(triton_model_instance, &triton_model));

  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // Parse JSON config
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  ni::TritonJson::Value model_config;

  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  *state = new ModelInstanceState(
      model_state, triton_model_instance, instance_name, instance_kind,
      instance_id, std::move(model_config));

  return nullptr;
}

ModelInstanceState::~ModelInstanceState()
{
  // Close python interpreter.
  grpc::ClientContext context;
  ni::StatusCode result;
  ni::Empty null_msg;

  const auto err = stub->InterpreterShutdown(&context, null_msg, &result);
  if (!err.ok() || result.code() != 0) {
    LOG_ERROR << "Cannot shutdown interpreter gracefully: "
              << err.error_message() << std::endl;
  }

  int status;
  waitpid(interpreter_pid_, &status, 0);
  unlink(domain_socket_.substr(7).c_str());
}

int
ModelInstanceState::GetInputTensor(
    const uint32_t iidx, TRITONBACKEND_Request* request,
    ni::InferenceData* input_tensor,
    std::vector<TRITONBACKEND_Response*>& responses, size_t r,
    uint32_t& batch_size)
{
  const char* input_name;
  // Load iidx'th input name
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_RequestInputName(request, iidx, &input_name));

  // Load iidx'th input
  TRITONBACKEND_Input* in;
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_RequestInput(request, input_name, &in));

  // Load input properties
  TRITONSERVER_DataType input_dtype;
  const int64_t* input_shape;
  uint32_t input_dims_count;
  uint64_t input_byte_size;
  uint32_t input_buffer_count;
  GUARDED_RESPOND_IF_ERROR(
      responses, r,
      TRITONBACKEND_InputProperties(
          in, &input_name, &input_dtype, &input_shape, &input_dims_count,
          &input_byte_size, &input_buffer_count));

  if (iidx == 0) {
    batch_size = input_shape[0];
  }

  // Update input_tensor.
  input_tensor->set_name(input_name);
  input_tensor->set_dtype(static_cast<inference::DataType>(input_dtype));

  for (size_t j = 0; j < input_dims_count; ++j) {
    input_tensor->add_dims(input_shape[j]);
  }

  // Load raw data into input_tensor raw data.
  std::string* data_buffer = input_tensor->mutable_raw_data();
  const void* input_buffer = nullptr;
  uint64_t buffer_byte_size = 0;
  TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t input_memory_type_id = 0;
  for (size_t j = 0; j < input_buffer_count; ++j) {
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputBuffer(
            in, j, &input_buffer, &buffer_byte_size, &input_memory_type,
            &input_memory_type_id));
    data_buffer->append((const char*)input_buffer, buffer_byte_size);
  }

  return nic::ErrorCodes::Success;
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  ni::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

  uint64_t model_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &model_version));

  TRITONSERVER_Server* triton_server;
  RETURN_IF_ERROR(TRITONBACKEND_ModelServer(triton_model, &triton_server));

  *state = new ModelState(
      triton_server, triton_model, model_name, model_version,
      std::move(model_config));
  return nullptr;  // success
}

ModelState::ModelState(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* model_name, const uint64_t model_version,
    ni::TritonJson::Value&& model_config)
    : triton_server_(triton_server), triton_model_(triton_model),
      model_name_(model_name), model_version_(model_version),
      model_config_(std::move(model_config))
{
}

TRITONSERVER_Error*
ModelState::SupportsFirstDimBatching(bool* supports)
{
  uint32_t flags = 0;
  RETURN_IF_ERROR(TRITONSERVER_ServerModelBatchProperties(
      triton_server_, model_name_.c_str(), model_version_, &flags,
      nullptr /* voidp */));
  *supports = ((flags & TRITONSERVER_BATCH_FIRST_DIM) != 0);
  return nullptr;  // success
}

}  // namespace

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  // Check backend version to ensure compatibility
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_VERBOSE, __FILE__, __LINE__,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor);
  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "Triton backend API version does not support this backend");
  }

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  delete model_state;

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  ModelState* model_state = instance_state->StateForModel();

  instance_state->CreatePythonInterpreter();

  bool supports_batching_initialized = false;
  bool supports_batching = false;

  if (!supports_batching_initialized) {
    LOG_IF_ERROR(
        model_state->SupportsFirstDimBatching(&supports_batching),
        "failed to determine batching support");
    supports_batching_initialized = true;
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Response* response = responses[r];

    ni::InferenceBatch input_batch;

    uint32_t requested_input_count = 0;
    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputCount(request, &requested_input_count));
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    // Batching all the inputs in a single request
    uint32_t batch_size = 0;
    for (size_t iidx = 0; iidx < requested_input_count; ++iidx) {
      ni::InferenceData* input_tensor = input_batch.add_tensors();
      int err = instance_state->GetInputTensor(
          iidx, request, input_tensor, responses, r, batch_size);
      if (err != nic::ErrorCodes::Success) {
        LOG_ERROR << "Can't get input tensor\n";
      }
    }

    input_batch.set_batch_size(batch_size);

    // Inference request
    grpc::ClientContext context;
    ni::InferenceBatch result_batch;

    const auto err = instance_state->stub->InferenceRequest(
        &context, input_batch, &result_batch);

    if (!err.ok()) {
      LOG_ERROR << "something went wrong with interfere request\n"
                << err.error_message();
    }

    // Time to output.
    ni::TritonJson::Value outputs;
    LOG_IF_ERROR(
        instance_state->model_config.MemberAsArray("output", &outputs),
        "can't get model outputs");

    for (size_t j = 0; j < requested_output_count; ++j) {
      // Prepare output buffers.
      TRITONBACKEND_Output* output;
      ni::TritonJson::Value out;
      LOG_IF_ERROR(outputs.IndexAsObject(j, &out), "can't get out");
      std::string out_name, out_dtype;
      LOG_IF_ERROR(out.MemberAsString("name", &out_name), "can't get out name");
      std::vector<int64_t> out_shape;
      LOG_IF_ERROR(
          nib::ParseShape(out, "dims", &out_shape), "can't get out shape");
      LOG_IF_ERROR(
          out.MemberAsString("data_type", &out_dtype),
          "can't get out data type");

      // Remove TYPE_ from the datatype
      out_dtype = out_dtype.substr(5);
      out_shape.insert(out_shape.begin(), batch_size);

      inference::DataType out_dt = ni::ProtocolStringToDataType(out_dtype);

      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &output, out_name.data(), ni::DataTypeToTriton(out_dt),
              out_shape.data(), out_shape.size()));

      uint64_t acc = std::accumulate(
          out_shape.begin(), out_shape.end(), 1,
          [](uint64_t acc, const int64_t& val) { return acc * val; });

      size_t type_size = ni::GetDataTypeByteSize(out_dt);

      void* output_buffer;

      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, type_size * acc, &output_memory_type,
              &output_memory_type_id));


      if ((responses[r] == nullptr) ||
          (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "can't create response in GPU memory."));
        TRITONSERVER_LogMessage(
            TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
            (std::string("request ") + std::to_string(r) +
             ": failed to create output buffer in CPU memory.")
                .c_str());
        continue;
      }
      auto output_itr = std::find_if(
          result_batch.tensors().begin(), result_batch.tensors().end(),
          [&out_name](const ni::InferenceData& itr) {
            return itr.name() == out_name;
          });

      if (output_itr == result_batch.tensors().end()) {
        LOG_ERROR << "can't find output tensor with name " << out_name << '\n';
      }

      // Copy Python output to Triton output buffers
      std::copy(
          output_itr->raw_data().begin(), output_itr->raw_data().end(),
          (char*)output_buffer);
    }

    if (responses[r] == nullptr) {
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
          (std::string("Request ") + std::to_string(r) +
           ": failed to create output response")
              .c_str());
      continue;
    }

    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            nullptr /* success */),
        "failed sending response");
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    // if (responses[r] == nullptr) {
    //   LOG_IF_ERROR(
    //       TRITONBACKEND_ModelReportStatistics(
    //           model_state->TritonModel(), request, false /* success */,
    //           TRITONBACKEND_NO_DEVICE, 0, 0, 0, 0),
    //       "failed reporting request statistics");
    // }

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
       " (device " + std::to_string(device_id) + ")")
          .c_str());

  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  RETURN_ERROR_IF_FALSE(
      instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("'python_interpreter' backend only supports CPU instances"));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;
}

}  // extern "C"
