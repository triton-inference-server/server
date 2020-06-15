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

#include <iostream>
#include <string>
#include <vector>
#include "src/backends/backend/tritonbackend.h"

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "src/core/json.h"

namespace ni = nvidia::inferenceserver;

//
// Simple backend that demonstrates the TRITONBACKEND API for a
// sequential backend. A sequential backend produces exactly 1
// response for every request and sends the response before exiting
// the TRITONBACKEND_ModelExecute function.
//
// This backend supports any model that has exactly 1 input and
// exactly 1 output. The input and output can have any name, datatype
// and shape but the shape and datatype of the input and output must
// match. The backend simply responds with the output tensor equal to
// the input tensor.
//

namespace {

#define RETURN_ERROR_IF_FALSE(P, C, MSG)              \
  do {                                                \
    if (!(P)) {                                       \
      return TRITONSERVER_ErrorNew(C, (MSG).c_str()); \
    }                                                 \
  } while (false)

#define RETURN_IF_ERROR(X)           \
  do {                               \
    TRITONSERVER_Error* err__ = (X); \
    if (err__ != nullptr) {          \
      return err__;                  \
    }                                \
  } while (false)

#define LOG_IF_ERROR(X, MSG)                                               \
  do {                                                                     \
    TRITONSERVER_Error* err__ = (X);                                       \
    if (err__ != nullptr) {                                                \
      TRITONSERVER_LogMessage(                                             \
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,                       \
          (std::string(MSG) + ": " + TRITONSERVER_ErrorCodeString(err__) + \
           " - " + TRITONSERVER_ErrorMessage(err__))                       \
              .c_str());                                                   \
      TRITONSERVER_ErrorDelete(err__);                                     \
    }                                                                      \
  } while (false)

TRITONSERVER_Error*
ParseShape(
    ni::TritonJson::Value& io, const std::string& name,
    std::vector<int64_t>* shape)
{
  ni::TritonJson::Value shape_array;
  RETURN_IF_ERROR(io.MemberAsArray(name.c_str(), &shape_array));
  for (size_t i = 0; i < shape_array.ArraySize(); ++i) {
    int64_t d;
    RETURN_IF_ERROR(shape_array.IndexAsInt(i, &d));
    shape->push_back(d);
  }

  return nullptr;  // success
}

std::string
ShapeToString(const std::vector<int64_t>& shape)
{
  bool first = true;

  std::string str("[");
  for (const auto& dim : shape) {
    if (!first) {
      str += ",";
    }
    str += std::to_string(dim);
    first = false;
  }

  str += "]";
  return str;
}

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ModelState(
      TRITONBACKEND_Model* triton_model, ni::TritonJson::Value&& model_config);

  TRITONBACKEND_Model* triton_model_;
  ni::TritonJson::Value model_config_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, &config_message));

  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  ni::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  *state = new ModelState(triton_model, std::move(model_config));
  return nullptr;  // success
}

ModelState::ModelState(
    TRITONBACKEND_Model* triton_model, ni::TritonJson::Value&& model_config)
    : triton_model_(triton_model), model_config_(std::move(model_config))
{
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  ni::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  ni::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // There must be 1 input and 1 output.
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 1 input, got ") +
          std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 1 output, got ") +
          std::to_string(outputs.ArraySize()));

  ni::TritonJson::Value input, output;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));

  // Input and output must have same datatype
  std::string input_dtype, output_dtype;
  RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
  RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));

  RETURN_ERROR_IF_FALSE(
      input_dtype == output_dtype, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected input and output datatype to match, got ") +
          input_dtype + " and " + output_dtype);

  // Input and output must have same shape
  std::vector<int64_t> input_shape, output_shape;
  RETURN_IF_ERROR(ParseShape(input, "dims", &input_shape));
  RETURN_IF_ERROR(ParseShape(output, "dims", &output_shape));

  RETURN_ERROR_IF_FALSE(
      input_shape == output_shape, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected input and output shape to match, got ") +
          ShapeToString(input_shape) + " and " + ShapeToString(output_shape));

  return nullptr;  // success
}

}  // namespace

/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version;
  RETURN_IF_ERROR(TRITONBACKEND_BackendApiVersion(backend, &api_version));

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version))
          .c_str());
  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION))
          .c_str());

  if (api_version < TRITONBACKEND_API_VERSION) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string(" triton backend API version '") +
         std::to_string(api_version) +
         "' is less than backend's API version '" +
         std::to_string(TRITONBACKEND_API_VERSION) + "'")
            .c_str());
  }

  // If we have any global backend state we create and set it here. We
  // don't need anything for this backend but for demonstration
  // purposes we just create something...
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
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

  // Can get the full path to the filesystem directory containing this
  // model... in case we wanted to load something from the repo.
  const char* cdir;
  RETURN_IF_ERROR(TRITONBACKEND_ModelRepositoryPath(model, &cdir));
  std::string dir(cdir);

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("Repository path: ") + dir).c_str());

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  std::string* backend_state = reinterpret_cast<std::string*>(vbackendstate);

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("backend state is '") + *backend_state + "'").c_str());

  // With each model we create a Model object and associate it with
  // the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  // FIXME here we should look at the instance-group and decide how we
  // are going to handle multiple instances (or raise and error if
  // disallowing them).

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelExecute(
    TRITONBACKEND_Model* model, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelExecute: ") +
       std::to_string(request_count) + " requests")
          .c_str());

  return nullptr;  // success
}

}  // extern "C"
