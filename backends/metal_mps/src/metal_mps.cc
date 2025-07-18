// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <sstream>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

#include "mps_engine.h"
#include "mps_model.h"

namespace triton { namespace backend { namespace metal_mps {

// Metal Performance Shaders backend implementation for Triton
//
// This backend loads and executes models using Apple's Metal Performance Shaders.
// The model must be in a supported format (ONNX-based) and placed in the model
// directory as 'model.mps' or 'model.onnx'.

#define RESPOND_IF_ERROR(RESPONSE, X)                                   \
  do {                                                                  \
    if (RESPONSE != nullptr) {                                          \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                RESPONSE, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err__), \
            "failed to send error response");                           \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

#define RESPOND_AND_SET_NULL_IF_ERROR(RESPONSE_PTR, X)                 \
  do {                                                                  \
    TRITONSERVER_Error* err__ = (X);                                    \
    if (err__ != nullptr) {                                             \
      LOG_IF_ERROR(                                                     \
          TRITONBACKEND_ResponseSend(                                   \
              RESPONSE_PTR, TRITONSERVER_RESPONSE_COMPLETE_FINAL,       \
              err__),                                                   \
          "failed to send error response");                             \
      TRITONSERVER_ErrorDelete(err__);                                  \
      RESPONSE_PTR = nullptr;                                           \
    }                                                                   \
  } while (false)

// MPS Model State
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Load the MPS model
  TRITONSERVER_Error* LoadModel();

  // Get the loaded model
  MPSModel* GetModel() { return model_.get(); }

  // Validate model configuration
  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ModelState(TRITONBACKEND_Model* triton_model)
      : BackendModel(triton_model) {}

  std::unique_ptr<MPSModel> model_;
  std::string model_path_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // Validate that model configuration is supported
  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &ios));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &ios));

  // Check max batch size
  int64_t max_batch_size = 0;
  RETURN_IF_ERROR(model_config_.MemberAsInt("max_batch_size", &max_batch_size));
  
  if (max_batch_size < 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "max_batch_size must be >= 0");
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::LoadModel()
{
  // Validate configuration first
  RETURN_IF_ERROR(ValidateModelConfig());

  // Get the model repository path
  TRITONBACKEND_ArtifactType artifact_type;
  const char* path;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(triton_model_, &artifact_type, &path));

  if (artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "Metal MPS backend only supports filesystem model repository");
  }

  // Try different model file names
  std::vector<std::string> possible_files = {"model.mps", "model.onnx"};
  bool model_found = false;
  
  for (const auto& filename : possible_files) {
    model_path_ = std::string(path) + "/" + filename;
    std::ifstream file(model_path_);
    if (file.good()) {
      model_found = true;
      file.close();
      break;
    }
  }

  if (!model_found) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        "Could not find model file (model.mps or model.onnx) in repository");
  }

  try {
    // Create and load the MPS model
    model_ = std::make_unique<MPSModel>();
    RETURN_IF_ERROR(model_->LoadFromFile(model_path_));
    
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Loaded MPS model from: ") + model_path_).c_str());
  }
  catch (const std::exception& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("Failed to load MPS model: ") + e.what()).c_str());
  }

  return nullptr;  // success
}

// MPS Model Instance State
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Execute inference request
  void ProcessRequest(
      TRITONBACKEND_Request* request,
      std::vector<TRITONBACKEND_Response*>& responses);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state)
  {
    // Create MPS execution engine
    engine_ = std::make_unique<MPSEngine>();
  }

  ModelState* model_state_;
  std::unique_ptr<MPSEngine> engine_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequest(
    TRITONBACKEND_Request* request,
    std::vector<TRITONBACKEND_Response*>& responses)
{
  // Get the response
  TRITONBACKEND_Response* response;
  TRITONBACKEND_ResponseNew(&response, request);
  responses.push_back(response);

  // Get input count
  uint32_t input_count;
  RESPOND_AND_SET_NULL_IF_ERROR(
      response,
      TRITONBACKEND_RequestInputCount(request, &input_count));
  if (response == nullptr) return;

  // Get output count
  uint32_t requested_output_count;
  RESPOND_AND_SET_NULL_IF_ERROR(
      response,
      TRITONBACKEND_RequestOutputCount(request, &requested_output_count));
  if (response == nullptr) return;

  // Prepare inputs
  std::vector<MPSTensor> inputs;
  for (uint32_t i = 0; i < input_count; i++) {
    TRITONBACKEND_Input* input;
    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONBACKEND_RequestInputByIndex(request, i, &input));
    if (response == nullptr) return;

    const char* name;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint32_t dims_count;
    uint64_t byte_size;
    uint32_t buffer_count;

    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONBACKEND_InputProperties(
            input, &name, &datatype, &shape, &dims_count, &byte_size,
            &buffer_count));
    if (response == nullptr) return;

    // Create input tensor
    MPSTensor tensor;
    tensor.name = name;
    tensor.datatype = datatype;
    tensor.shape.assign(shape, shape + dims_count);
    tensor.byte_size = byte_size;

    // Get input data
    const void* buffer;
    uint64_t buffer_byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;

    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONBACKEND_InputBuffer(
            input, 0, &buffer, &buffer_byte_size, &memory_type,
            &memory_type_id));
    if (response == nullptr) return;

    // For now, we only support CPU memory
    if (memory_type != TRITONSERVER_MEMORY_CPU) {
      RESPOND_IF_ERROR(
          response,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "MPS backend only supports CPU memory input"));
      return;
    }

    tensor.data = const_cast<void*>(buffer);
    inputs.push_back(tensor);
  }

  // Execute the model
  std::vector<MPSTensor> outputs;
  RESPOND_AND_SET_NULL_IF_ERROR(
      response,
      engine_->Execute(model_state_->GetModel(), inputs, outputs));
  if (response == nullptr) return;

  // Create response outputs
  for (const auto& output : outputs) {
    TRITONBACKEND_Output* response_output;
    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONBACKEND_ResponseOutput(
            response, &response_output, output.name.c_str(), output.datatype,
            output.shape.data(), output.shape.size()));
    if (response == nullptr) return;

    // Allocate output buffer
    void* output_buffer;
    TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t output_memory_type_id = 0;
    
    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONBACKEND_OutputBuffer(
            response_output, &output_buffer, output.byte_size,
            &output_memory_type, &output_memory_type_id));
    if (response == nullptr) return;

    // Copy output data
    memcpy(output_buffer, output.data, output.byte_size);
  }

  // Send the response
  LOG_IF_ERROR(
      TRITONBACKEND_ResponseSend(
          response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
      "failed to send response");
}

/////////////

extern "C" {

// Backend initialization
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check backend API version
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));
  
  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "Triton backend API version does not support this backend");
  }

  // Backend configuration
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  return nullptr;  // success
}

// Backend finalization
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: ") + name).c_str());

  return nullptr;  // success
}

// Model initialization
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")").c_str());

  // Create model state
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // Load the model
  RETURN_IF_ERROR(model_state->LoadModel());

  return nullptr;  // success
}

// Model finalization
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Model instance initialization
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  
  // For MPS, device_id represents the GPU device (usually 0 for the main GPU)
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")").c_str());

  // Get model state
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create instance state
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Model instance finalization
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

  return nullptr;  // success
}

// Model instance execution
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Get instance state
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model instance ") + instance_state->Name() + 
       ", executing " + std::to_string(request_count) + " requests").c_str());

  // Process each request
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    
    // Create response list for this request
    std::vector<TRITONBACKEND_Response*> responses;
    instance_state->ProcessRequest(request, responses);

    // Release request
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::metal_mps