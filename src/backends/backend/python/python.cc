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

#include <grpc/grpc.h>
#include <grpc/support/time.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
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

#include "python_host.grpc.pb.h"
#include "src/backends/backend/examples/backend_utils.h"
#include "src/backends/backend/tritonbackend.h"
#include "src/core/json.h"
#include "src/core/tritonserver.h"

namespace ni = nvidia::inferenceserver;
namespace nib = nvidia::inferenceserver::backend;

namespace nvidia { namespace inferenceserver { namespace backend {

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

namespace {

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

struct BackendState {
  std::string python_lib;
  std::string python_runtime;
};

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

  /// Creates a python child process running startup.py
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
      const int32_t device_id, ni::TritonJson::Value&& model_config,
      TRITONBACKEND_Model* trition_model);

  TRITONSERVER_Error* ConnectPythonInterpreter();

  std::string pymodule_path_;
  ModelState* model_state_;
  std::string domain_socket_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;
  bool connected_ = false;

 public:
  ni::TritonJson::Value model_config;

 private:
  TRITONBACKEND_Model* triton_model_;
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

  // Get backend state
  ::BackendState* BackendState() { return backend_state_; }

  // Get Model Path
  const char* ModelPath() { return model_path_; }

 private:
  ModelState(
      TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
      const char* model_name, const uint64_t model_version,
      ni::TritonJson::Value&& model_config, ::BackendState* backend_state, const char* model_path);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string model_name_;
  const uint64_t model_version_;
  ni::TritonJson::Value model_config_;
  ::BackendState* backend_state_;
  const char* model_path_;
};

TRITONSERVER_Error*
ModelInstanceState::CreatePythonInterpreter()
{
  const char* subinterpreter_commandline[] = {
      nullptr, nullptr,           "--socket", nullptr, "--model-path",
      nullptr, "--instance-name", nullptr,    nullptr};

  constexpr int max_tmpfile_name = 255;
  char tmp_dir_name[max_tmpfile_name] = "/tmp/XXXXXX";
  char full_socket_name[max_tmpfile_name];

  // Create a temporary directory and use <tmp_dir>/unix.socket for GRPC socket
  // This is the only way that we can make sure that the unix socket path used
  // for GRPC is unique
  char* tmp_dir_response = mkdtemp(tmp_dir_name);

  if (!tmp_dir_response) {
    TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to create a temporary socket name");
    return err;
  } else {
    std::stringstream ss;
    ss << tmp_dir_name << "/unix.socket";
    snprintf(full_socket_name, max_tmpfile_name, "unix://%s", ss.str().c_str());
    subinterpreter_commandline[3] = full_socket_name;
    domain_socket_ = std::string(full_socket_name);
  }

  uint64_t model_version = model_state_->ModelVersion();
  const char* model_path = model_state_->ModelPath();

  std::stringstream ss;
  // Use <path>/version/model.py as the model location
  ss << model_path << "/" << model_version << "/model.py";
  pymodule_path_ = ss.str();
  interpreter_pid_ = fork();

  if (interpreter_pid_ == 0) {
    // Use the python available in $PATH
    std::string python_interpreter_path =
        model_state_->BackendState()->python_runtime;

    std::stringstream ss;
    ss << model_state_->BackendState()->python_lib << "/startup.py";
    std::string python_interpreter_startup = ss.str();

    subinterpreter_commandline[0] = python_interpreter_path.c_str();
    subinterpreter_commandline[1] = python_interpreter_startup.c_str();
    subinterpreter_commandline[5] = pymodule_path_.c_str();
    subinterpreter_commandline[7] = name_.c_str();
    if (execvpe(
            subinterpreter_commandline[0], (char**)subinterpreter_commandline,
            nullptr) == -1) {
      std::stringstream ss;
      ss << "Cannot run interpreter host. Errno = " << errno << '\n'
         << "python_interpreter_path: " << python_interpreter_path << '\n'
         << "python_interpreter_startup: " << python_interpreter_startup << '\n'
         << "pymodule_path_: " << pymodule_path_ << '\n'
         << "instance_name: " << name_ << '\n';
      std::string log_message = ss.str();
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, log_message.c_str());

      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("Failed to initialize model instance ") + name_)
              .c_str());
    }
  } else {
    RETURN_IF_ERROR(ConnectPythonInterpreter());
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ConnectPythonInterpreter()
{
  auto grpc_channel =
      grpc::CreateChannel(domain_socket_, grpc::InsecureChannelCredentials());

  stub = ni::PythonInterpreter::NewStub(grpc_channel);

  std::shared_ptr<ni::InitializationCommand> initialization_params(
      new ni::InitializationCommand());

  std::vector<std::string> keys;
  LOG_IF_ERROR(model_config.Members(&keys), "can't get key names");
  std::string val;

  const auto insert_model_param =
      [&initialization_params](const std::string& key, const std::string& val) {
        auto* value_pair = initialization_params->add_args();
        value_pair->set_key(key);
        value_pair->set_value(val);
      };

  ni::TritonJson::WriteBuffer buffer;
  model_config.Write(&buffer);

  insert_model_param("model_config", std::move(buffer.MutableContents()));
  insert_model_param("model_instance_kind", TRITONSERVER_InstanceGroupKindString(kind_));
  insert_model_param("model_instance_name", name_);
  insert_model_param("model_instance_device_id", std::to_string(device_id_));
  insert_model_param("model_repository", model_state_->ModelPath());
  insert_model_param("model_version", std::to_string(model_state_->ModelVersion()));
  insert_model_param("model_name", model_state_->ModelName());

  // Attempting to connect to the python runtime
  constexpr uint8_t conn_attempts = 5;
  for (int i = 0; i < conn_attempts; ++i) {
    grpc::ClientContext context;
    ni::Empty null_msg;
    const auto status = stub->Init(&context, *initialization_params, &null_msg);
    if (status.ok()) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("GRPC connection was successful ") +
          name_ + " (device " + std::to_string(device_id_) + ")")
              .c_str());
      connected_ = true;
      return nullptr;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL, "failed to initialize grpc stub");
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id, ni::TritonJson::Value&& model_config,
    TRITONBACKEND_Model* triton_model)
    : model_state_(model_state), triton_model_instance_(triton_model_instance),
      name_(name), kind_(kind), device_id_(device_id),
      model_config(std::move(model_config)), triton_model_(triton_model)
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
      instance_id, std::move(model_config), triton_model);

  return nullptr;
}

ModelInstanceState::~ModelInstanceState()
{
  // Intentional empty scope, without this empty scope
  // GRPC will NOT shutdown gracefully
  {
    // Close python interpreter.
    grpc::ClientContext context;
    ni::Empty null_msg;

    if (connected_) {
      auto err = stub->Fini(&context, null_msg, &null_msg);
      if (!err.ok()) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            ("Cannot shutdown interpreter gracefully: " + err.error_message())
                .c_str());
      }
    }

  }

  stub.reset();

  int status;
  kill(interpreter_pid_, SIGTERM);
  waitpid(interpreter_pid_, &status, 0);
  unlink(domain_socket_.substr(7).c_str());

  // FIXME currently GRPC client uses an async thread for cleaning up and
  // shutting down the connection, however, as reported in
  // https://github.com/grpc/grpc/issues/22479 the clean up thread may continue
  // to live after resources have been deallocated an cause a segfault. This is
  // a workaround to do a blocking shutdown of the GRPC client
  grpc_shutdown_blocking();

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "GRPC shutdown complete");
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

  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(triton_model, &backend));

  void* bstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &bstate));
  ::BackendState* backend_state = reinterpret_cast<::BackendState*>(bstate);

  const char* path = nullptr;
  TRITONBACKEND_ModelArtifactType artifact_type;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(triton_model, &artifact_type, &path));

  if (artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("unsupported artifact type for model '") + model_name + "'")
            .c_str());
  }

  *state = new ModelState(
      triton_server, triton_model, model_name, model_version,
      std::move(model_config), backend_state, path);

  return nullptr;
}

ModelState::ModelState(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* model_name, const uint64_t model_version,
    ni::TritonJson::Value&& model_config, ::BackendState* backend_state, const char* model_path)
    : triton_server_(triton_server), triton_model_(triton_model),
      model_name_(model_name), model_version_(model_version),
      model_config_(std::move(model_config)), backend_state_(backend_state),
      model_path_(model_path)
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
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
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

  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("backend configuration:\n") + buffer).c_str());

  ni::TritonJson::Value backend_config;
  if (byte_size != 0) {
    RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
  }

  std::unique_ptr<BackendState> backend_state(new BackendState());
  ni::TritonJson::Value cmdline;
  bool found_py_lib_config = false;
  bool found_py_runtime_config = false;

  if (backend_config.Find("cmdline", &cmdline)) {
    ni::TritonJson::Value python_lib;
    if (cmdline.Find("python-lib", &python_lib)) {
      RETURN_IF_ERROR(python_lib.AsString(&backend_state->python_lib));
      found_py_lib_config = true;
    }

    ni::TritonJson::Value python_runtime;
    if (cmdline.Find("python-runtime", &python_runtime)) {
      RETURN_IF_ERROR(python_runtime.AsString(&backend_state->python_runtime));
      found_py_runtime_config = true;
    }
  }

  // Set the default path for python runtime and library
  if (!found_py_lib_config) {
    backend_state->python_lib = "/opt/tritonserver/lib/python/runtime";
  }

  if (!found_py_runtime_config) {
    backend_state->python_runtime = "python3";
  }


  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(backend_state.get())));

  backend_state.release();
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "TRITONBACKEND_Finalize: Start");
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  auto backend_state = reinterpret_cast<BackendState*>(vstate);
  delete backend_state;
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "TRITONBACKEND_Finalize: End");
  return nullptr;  // success
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

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_ModelFinalize: delete model state");

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
      std::string("python backend only supports CPU instances"));

  RETURN_IF_ERROR(instance_state->CreatePythonInterpreter());

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: instance "
                   "initialization successful ") +
       name + " (device " + std::to_string(device_id) + ")")
          .c_str());

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

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

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
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          instance_state->GetInputTensor(
              iidx, request, input_tensor, responses, r, batch_size));
    }

    // Append the list of requested outputs to the inference_request
    for (size_t iidx = 0; iidx < requested_output_count; ++iidx) {
      const char* requested_output_name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(
              request, iidx, &requested_output_name));

      inference_request->add_requested_output_names(requested_output_name);
    }

    const char* id;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestId(request, &id));
    inference_request->set_id(id);

    uint64_t correlation_id;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));
    inference_request->set_correlation_id(correlation_id);
  }

  // ExecuteResponse
  grpc::ClientContext context;
  ni::ExecuteResponse execute_response;

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // Perform inference on the Python side
  const auto status = instance_state->stub->Execute(
      &context, execute_request, &execute_response);

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  // If inference fails, release all the requests and send an error response If
  // inference fails at this stage, it usually indicates a bug in the model code
  if (!status.ok()) {
    for (uint32_t r = 0; r < request_count; ++r) {
      if (responses[r] == nullptr) {
        continue;
      }
      TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, ("GRPC Execute Failed, message: " +
                                        std::string(status.error_message()))
                                           .c_str());
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
          "failed sending response");
      responses[r] = nullptr;
      TRITONSERVER_ErrorDelete(err);
    }

    for (uint32_t r = 0; r < request_count; ++r) {
      TRITONBACKEND_Request* request = requests[r];
      LOG_IF_ERROR(
          TRITONBACKEND_ModelInstanceReportStatistics(
              instance, request, false /* success */, exec_start_ns,
              compute_start_ns, compute_end_ns, compute_end_ns),
          "failed reporting request statistics");

      LOG_IF_ERROR(
          TRITONBACKEND_RequestRelease(
              request, TRITONSERVER_REQUEST_RELEASE_ALL),
          "failed releasing request");
    }

    return nullptr;
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Response* response = responses[r];
    TRITONBACKEND_Request* request = requests[r];
    uint32_t requested_output_count = 0;

    // Get response r
    ni::InferenceResponse inference_response = execute_response.responses(r);

    if (inference_response.failed()) {
      TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, (inference_response.error().message()).c_str());
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
          "failed sending response");
      responses[r] = nullptr;
      TRITONSERVER_ErrorDelete(err);

      // If has_error is true, we do not look at the response even if the response
      // is set.
      continue;
    }

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));
    for (size_t j = 0; j < requested_output_count; ++j) {
      // Prepare output buffers.
      const ni::Tensor python_output_result = inference_response.outputs(j);
      TRITONBACKEND_Output* triton_output;
      TRITONSERVER_DataType triton_dt =
          static_cast<TRITONSERVER_DataType>(python_output_result.dtype());

      auto python_output_dims = python_output_result.dims();
      const std::string output_tensor_name = python_output_result.name();

      uint32_t dims_count = python_output_dims.size();

      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &triton_output, python_output_result.name().c_str(),
              triton_dt, python_output_dims.data(), dims_count));

      std::vector<int64_t> output_dims(
          python_output_dims.begin(), python_output_dims.end());
      int64_t output_byte_size = nib::GetByteSize(triton_dt, output_dims);
      void* output_buffer;

      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              triton_output, &output_buffer, output_byte_size,
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
          inference_response.outputs().begin(),
          inference_response.outputs().end(),
          [&output_tensor_name](const ni::Tensor& itr) {
            return itr.name() == output_tensor_name;
          });

      // Continue to the next inference batch if the corresponding output
      // response can't be found
      if (output_response_tensor == inference_response.outputs().end()) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            ("can't find output tensor with name " + output_tensor_name)
                .c_str());
        continue;
      }

      // Copy Python output to Triton output buffers
      std::copy(
          output_response_tensor->raw_data().begin(),
          output_response_tensor->raw_data().end(), (char*)output_buffer);
    }

    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR, (std::string("Request ") + std::to_string(r) +
                                   ": failed to create output response")
                                      .c_str());
      continue;
    }

    // If error happens at this stage, we can only log it
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
        "failed sending response");
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    // Report statistics for the request. Note that there could
    // still be responses that have not yet been sent but those
    // cannot be captured in the statistics as they reflect only the
    // request object. We use the execution start/end time for
    // compute also so that the entire execution time is associated
    // with the inference computation.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance, request, (responses[r] != nullptr) /* success */,
            exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics. This backend does not support
  // batching so the total batch size is always 1.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance, 1, exec_start_ns, compute_start_ns, compute_end_ns,
          exec_end_ns),
      "failed reporting batch request statistics");


  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelInstanceExecute: model instance name ") +
       instance_state->Name() + " released " + std::to_string(request_count) +
       " requests")
          .c_str());

  return nullptr;
}


TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;
}

}  // extern "C"
