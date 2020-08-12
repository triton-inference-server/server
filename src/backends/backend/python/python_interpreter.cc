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
#include "src/backends/backend/examples/backend_utils.h"
#include "src/backends/backend/tritonbackend.h"
#include "src/core/json.h"
#include "src/core/tritonserver.h"

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
      return rarie_err__;                                               \
    }                                                                   \
  } while (false)

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

  /// Create a python child process running startup.py
  TRITONSERVER_Error* CreatePythonInterpreter();

  /// Load Triton inputs to the appropriate Protobufs
  TRITONSERVER_Error* GetInputTensor(
      const uint32_t iidx, TRITONBACKEND_Request* request,
      ni::Tensor* input_tensor, std::vector<TRITONBACKEND_Response*>& responses,
      size_t r, uint32_t& batch_size);

  // TODO: Create getter and setters
  std::unique_ptr<ni::PythonInterpreter::Stub> stub;

 private:
  ModelInstanceState(
      ModelState* model_state, TRITONBACKEND_ModelInstance* model_instance,
      const char* name, const TRITONSERVER_InstanceGroupKind kind,
      const int32_t device_id, ni::TritonJson::Value&& model_config);

  TRITONSERVER_Error* ConnectPythonInterpreter(const std::string& module_path);

  std::string pymodule_path_;
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

TRITONSERVER_Error*
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
    std::string python_interpreter_path = "/usr/bin/python3";
    std::string python_interpreter_startup =
        "/workspace/builddir/server/install/lib/python/runtime/startup.py";

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

      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("Failed to initialize model instance ") + name_)
              .c_str());
    }
  } else {
    ConnectPythonInterpreter(module_path);
  }

  return nullptr;
}

TRITONSERVER_Error*
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
    ni::Empty null_msg;
    const auto err = stub->Init(&context, *initialization_params, &null_msg);
    if (err.ok()) {
      return nullptr;
    } else {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL, "failed to initialize grpc stub");
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
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1, &config_message));

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
  ni::Empty null_msg;

  const auto err = stub->Fini(&context, null_msg, &null_msg);
  if (!err.ok()) {
    LOG_ERROR << "Cannot shutdown interpreter gracefully: "
              << err.error_message() << std::endl;
  }

  int status;
  waitpid(interpreter_pid_, &status, 0);
  unlink(domain_socket_.substr(7).c_str());
}

TRITONSERVER_Error*
ModelInstanceState::GetInputTensor(
    const uint32_t iidx, TRITONBACKEND_Request* request,
    ni::Tensor* input_tensor, std::vector<TRITONBACKEND_Response*>& responses,
    size_t r, uint32_t& batch_size)
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

  // Update input_tensor
  input_tensor->set_name(input_name);
  input_tensor->set_dtype(static_cast<int>(input_dtype));

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
    if ((responses[iidx] == nullptr) ||
        (input_memory_type == TRITONSERVER_MEMORY_GPU)) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "failed to get input buffer in CPU memory"));
    }
    data_buffer->append((const char*)input_buffer, buffer_byte_size);
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1, &config_message));

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
  return nullptr;
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
      TRITONSERVER_LOG_VERBOSE, __FILE__, __LINE__,
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
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
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

  RETURN_IF_ERROR(instance_state->CreatePythonInterpreter());

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
    TRITONBACKEND_Request* req = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, req));
    responses.push_back(response);
  }

  // Create ExecuteRequest
  ni::ExecuteRequest execute_request;
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    ni::InferenceRequest* inference_request = execute_request.add_requests();

    uint32_t requested_input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputCount(request, &requested_input_count));

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    uint32_t batch_size = 0;
    for (size_t iidx = 0; iidx < requested_input_count; ++iidx) {
      ni::Tensor* input_tensor = inference_request->add_inputs();
      TRITONSERVER_Error* err = instance_state->GetInputTensor(
          iidx, request, input_tensor, responses, r, batch_size);
      if (err != nullptr) {
        return err;
      }
    }

    for (size_t iidx = 0; iidx < requested_output_count; ++iidx) {
      const char* requested_output_name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(
              request, iidx, &requested_output_name));

      inference_request->add_requested_output_names(requested_output_name);
    }

    const char *id;
    TRITONBACKEND_RequestId(request, &id);
    inference_request->set_id(id);

    uint64_t correlation_id;
    TRITONBACKEND_RequestCorrelationId(request, &correlation_id);
    inference_request->set_correlation_id(correlation_id);
  }

  // ExecuteResponse
  grpc::ClientContext context;
  ni::ExecuteResponse execute_response;

  // Perform inference on the Python side
  const auto err = instance_state->stub->Execute(
      &context, execute_request, &execute_response);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Response* response = responses[r];
    TRITONBACKEND_Request* request = requests[r];
    uint32_t requested_output_count = 0;

    // Get response r
    ni::InferenceResponse inference_response = execute_response.responses(r);

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));
    for (size_t j = 0; j < requested_output_count; ++j) {
      // Prepare output buffers.
      const ni::Tensor output_tensor = inference_response.inputs(j);
      TRITONBACKEND_Output* output;
      TRITONSERVER_DataType triton_dt =
          static_cast<TRITONSERVER_DataType>(output_tensor.dtype());

      auto output_tensor_dims = output_tensor.dims();
      const std::string output_tensor_name = output_tensor.name();
      int64_t output_shape[output_tensor_dims.size()];

      for (int i = 0; i < output_tensor_dims.size(); i++) {
        output_shape[i] = output_tensor_dims.data()[i];
      }

      uint32_t dims_count = output_tensor_dims.size();

      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &output, output_tensor.name().c_str(), triton_dt,
              output_shape, dims_count));

      uint64_t total_output_size = std::accumulate(
          output_tensor_dims.begin(), output_tensor_dims.end(), 1,
          [](uint64_t acc, const int64_t& val) { return acc * val; });

      size_t type_size = TRITONSERVER_DataTypeByteSize(triton_dt);

      void* output_buffer;

      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, type_size * total_output_size,
              &output_memory_type, &output_memory_type_id));

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

      // Try to find the matching output name we don't use indexing here because
      // the output inference batch may be missing from the response
      auto output_response_tensor = std::find_if(
          inference_response.inputs().begin(),
          inference_response.inputs().end(),
          [&output_tensor_name](const ni::Tensor& itr) {
            return itr.name() == output_tensor_name;
          });

      // Continue to the next inference batch if the corresponding output
      // response can't be found
      if (output_response_tensor == inference_response.inputs().end()) {
        LOG_ERROR << "can't find output tensor with name " << output_tensor_name
                  << '\n';
        continue;
      }

      // Copy Python output to Triton output buffers
      std::copy(
          output_response_tensor->raw_data().begin(),
          output_response_tensor->raw_data().end(), (char*)output_buffer);
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
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
        "failed sending response");
    for (uint32_t r = 0; r < request_count; ++r) {
      TRITONBACKEND_Request* request = requests[r];

      // TODO: Add resposne/request statistics

      LOG_IF_ERROR(
          TRITONBACKEND_RequestRelease(
              request, TRITONSERVER_REQUEST_RELEASE_ALL),
          "failed releasing request");
    }
  }


  return nullptr;
}


TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  delete instance_state;

  return nullptr;
}

}  // extern "C"
