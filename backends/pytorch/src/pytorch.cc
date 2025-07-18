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

#include <torch/script.h>
#include <torch/torch.h>

#ifdef __APPLE__
#include <torch/mps.h>
#endif

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace pytorch {

// PyTorch backend implementation for Triton
//
// This backend loads and executes TorchScript models. The model must be
// serialized using torch.jit.save() and placed in the model directory
// as 'model.pt'.
//
// Supported devices:
// - CPU
// - CUDA (GPU)
// - MPS (Metal Performance Shaders on macOS)

#ifdef __APPLE__
// MPS-specific utilities for memory management
namespace mps_utils {

// Check if MPS is available and log device information
void LogMPSInfo() {
  if (torch::mps::is_available()) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        "MPS (Metal Performance Shaders) is available on this system");
    
    // Log if MPS is built into PyTorch
    if (torch::mps::is_built()) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          "PyTorch was built with MPS support");
    }
  } else {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        "MPS (Metal Performance Shaders) is not available on this system");
  }
}

// Synchronize MPS operations if needed
void SynchronizeMPS() {
  if (torch::mps::is_available()) {
    // MPS operations are automatically synchronized when copying to CPU
    // but we can add explicit synchronization if needed in the future
  }
}

} // namespace mps_utils
#endif

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

// Convert Triton data type to PyTorch data type
static c10::ScalarType
ConvertDataType(TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_BOOL:
      return c10::ScalarType::Bool;
    case TRITONSERVER_TYPE_UINT8:
      return c10::ScalarType::Byte;
    case TRITONSERVER_TYPE_INT8:
      return c10::ScalarType::Char;
    case TRITONSERVER_TYPE_INT16:
      return c10::ScalarType::Short;
    case TRITONSERVER_TYPE_INT32:
      return c10::ScalarType::Int;
    case TRITONSERVER_TYPE_INT64:
      return c10::ScalarType::Long;
    case TRITONSERVER_TYPE_FP16:
      return c10::ScalarType::Half;
    case TRITONSERVER_TYPE_FP32:
      return c10::ScalarType::Float;
    case TRITONSERVER_TYPE_FP64:
      return c10::ScalarType::Double;
    default:
      return c10::ScalarType::Undefined;
  }
}

// Convert PyTorch data type to Triton data type
static TRITONSERVER_DataType
ConvertDataType(c10::ScalarType dtype)
{
  switch (dtype) {
    case c10::ScalarType::Bool:
      return TRITONSERVER_TYPE_BOOL;
    case c10::ScalarType::Byte:
      return TRITONSERVER_TYPE_UINT8;
    case c10::ScalarType::Char:
      return TRITONSERVER_TYPE_INT8;
    case c10::ScalarType::Short:
      return TRITONSERVER_TYPE_INT16;
    case c10::ScalarType::Int:
      return TRITONSERVER_TYPE_INT32;
    case c10::ScalarType::Long:
      return TRITONSERVER_TYPE_INT64;
    case c10::ScalarType::Half:
      return TRITONSERVER_TYPE_FP16;
    case c10::ScalarType::Float:
      return TRITONSERVER_TYPE_FP32;
    case c10::ScalarType::Double:
      return TRITONSERVER_TYPE_FP64;
    default:
      return TRITONSERVER_TYPE_INVALID;
  }
}

// PyTorch Model State
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Load the TorchScript model
  TRITONSERVER_Error* LoadModel();

  // Get the loaded model
  torch::jit::Module& Model() { return model_; }
  
  // Get the device for this model
  torch::Device GetDevice() const { return device_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model)
      : BackendModel(triton_model), device_(torch::kCPU) {}

  torch::jit::Module model_;
  std::string model_path_;
  torch::Device device_;
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
ModelState::LoadModel()
{
  // Get the model repository path
  TRITONBACKEND_ArtifactType artifact_type;
  const char* path;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(triton_model_, &artifact_type, &path));

  if (artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "PyTorch backend only supports filesystem model repository");
  }

  // Construct the model path
  model_path_ = std::string(path) + "/model.pt";

  // Determine the device from model configuration
  triton::common::TritonJson::Value optimization;
  bool has_optimization = false;
  if (model_config_.Find("optimization", &optimization)) {
    has_optimization = true;
  }

  // Check for explicit device selection in optimization settings
  std::string device_str = "cpu";
  if (has_optimization) {
    triton::common::TritonJson::Value execution_accelerators;
    if (optimization.Find("execution_accelerators", &execution_accelerators)) {
      triton::common::TritonJson::Value gpu_execution_accelerator;
      if (execution_accelerators.Find("gpu_execution_accelerator", &gpu_execution_accelerator)) {
        for (size_t i = 0; i < gpu_execution_accelerator.ArraySize(); ++i) {
          triton::common::TritonJson::Value accelerator;
          gpu_execution_accelerator.IndexAsObject(i, &accelerator);
          const char* name;
          size_t name_len;
          if (accelerator.MemberAsString("name", &name, &name_len) == nullptr) {
            std::string accel_name(name, name_len);
            if (accel_name == "mps") {
              device_str = "mps";
              break;
            }
          }
        }
      }
    }
  }

  // Check instance groups for device type
  triton::common::TritonJson::Value instance_groups;
  if (model_config_.Find("instance_group", &instance_groups)) {
    for (size_t i = 0; i < instance_groups.ArraySize(); ++i) {
      triton::common::TritonJson::Value instance_group;
      instance_groups.IndexAsObject(i, &instance_group);
      const char* kind_str;
      size_t kind_len;
      if (instance_group.MemberAsString("kind", &kind_str, &kind_len) == nullptr) {
        std::string kind(kind_str, kind_len);
        if (kind == "KIND_GPU") {
          // Check if CUDA is available first, then MPS
          if (torch::cuda::is_available()) {
            device_str = "cuda";
          }
#ifdef __APPLE__
          else if (torch::mps::is_available()) {
            device_str = "mps";
          }
#endif
        } else if (kind == "KIND_CPU") {
          device_str = "cpu";
        }
      }
    }
  }

  // Set the device
  if (device_str == "cuda" && torch::cuda::is_available()) {
    device_ = torch::Device(torch::kCUDA);
  }
#ifdef __APPLE__
  else if (device_str == "mps" && torch::mps::is_available()) {
    device_ = torch::Device(torch::kMPS);
  }
#endif
  else {
    device_ = torch::Device(torch::kCPU);
  }

  try {
    // Disable gradient computation for inference
    torch::NoGradGuard no_grad;
    
    // Load the model
    model_ = torch::jit::load(model_path_);
    model_.eval();
    
    // Move model to the target device
    model_.to(device_);
    
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Loaded PyTorch model from: ") + model_path_ + 
         " on device: " + device_.str()).c_str());
  }
  catch (const c10::Error& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("Failed to load PyTorch model: ") + e.what()).c_str());
  }

  return nullptr;  // success
}

// PyTorch Model Instance State
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }
  
  // Get the device for this instance
  torch::Device GetDevice() const { return device_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state),
        device_(model_state->GetDevice())
  {
  }

  ModelState* model_state_;
  torch::Device device_;
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

// Execute a single request on the PyTorch model
static TRITONSERVER_Error*
ExecuteRequest(
    ModelState* model_state, ModelInstanceState* instance_state,
    TRITONBACKEND_Request* request, TRITONBACKEND_Response* response)
{
  // Get input count
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &input_count));

  // Prepare input tensors
  std::vector<torch::jit::IValue> inputs;
  inputs.reserve(input_count);

  // Process each input
  for (uint32_t i = 0; i < input_count; ++i) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(request, i, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    uint32_t input_buffer_count;

    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        &input_byte_size, &input_buffer_count));

    // Get input buffer
    const void* input_buffer = nullptr;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t input_memory_type_id = 0;

    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        input, 0, &input_buffer, &buffer_byte_size, &input_memory_type,
        &input_memory_type_id));

    // Convert to PyTorch tensor
    c10::ScalarType torch_dtype = ConvertDataType(input_datatype);
    if (torch_dtype == c10::ScalarType::Undefined) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          (std::string("Unsupported data type for input '") + input_name + "'")
              .c_str());
    }

    // Create shape vector
    std::vector<int64_t> torch_shape(
        input_shape, input_shape + input_dims_count);

    // Create PyTorch tensor from input buffer
    auto options = torch::TensorOptions().dtype(torch_dtype).device(torch::kCPU);
    torch::Tensor tensor = torch::from_blob(
        const_cast<void*>(input_buffer), torch_shape, options);

    // Move tensor to the target device if needed
    torch::Device target_device = instance_state->GetDevice();
    if (target_device != torch::kCPU) {
      // For MPS and CUDA, we need to copy the tensor to the device
      tensor = tensor.to(target_device);
    }

    // Add to inputs
    inputs.push_back(tensor);
  }

  // Run inference
  torch::jit::IValue output;
  try {
    torch::NoGradGuard no_grad;
    output = model_state->Model().forward(inputs);
  }
  catch (const c10::Error& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("PyTorch inference failed: ") + e.what()).c_str());
  }

  // Process outputs
  if (output.isTensor()) {
    // Single tensor output
    torch::Tensor output_tensor = output.toTensor();
    
    // Move output back to CPU if needed
    if (output_tensor.device() != torch::kCPU) {
      output_tensor = output_tensor.to(torch::kCPU);
    }
    
    // Get output info from model config
    triton::common::TritonJson::Value outputs;
    RETURN_IF_ERROR(model_state->ModelConfig().MemberAsArray("output", &outputs));
    
    triton::common::TritonJson::Value output_config;
    RETURN_IF_ERROR(outputs.IndexAsObject(0, &output_config));
    
    const char* output_name;
    size_t output_name_len;
    RETURN_IF_ERROR(output_config.MemberAsString("name", &output_name, &output_name_len));
    
    // Create output
    TRITONBACKEND_Output* output;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
        response, &output, output_name, ConvertDataType(output_tensor.scalar_type()),
        output_tensor.sizes().data(), output_tensor.dim()));
    
    // Allocate output buffer
    void* output_buffer;
    TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t output_memory_type_id = 0;
    RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
        output, &output_buffer, output_tensor.nbytes(), &output_memory_type,
        &output_memory_type_id));
    
    // Copy output data
    memcpy(output_buffer, output_tensor.data_ptr(), output_tensor.nbytes());
  }
  else if (output.isTuple()) {
    // Multiple tensor outputs
    auto output_tuple = output.toTuple()->elements();
    
    triton::common::TritonJson::Value outputs;
    RETURN_IF_ERROR(model_state->ModelConfig().MemberAsArray("output", &outputs));
    
    for (size_t i = 0; i < output_tuple.size(); ++i) {
      if (!output_tuple[i].isTensor()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "PyTorch backend only supports tensor outputs");
      }
      
      torch::Tensor output_tensor = output_tuple[i].toTensor();
      
      // Move output back to CPU if needed
      if (output_tensor.device() != torch::kCPU) {
        output_tensor = output_tensor.to(torch::kCPU);
      }
      
      triton::common::TritonJson::Value output_config;
      RETURN_IF_ERROR(outputs.IndexAsObject(i, &output_config));
      
      const char* output_name;
      size_t output_name_len;
      RETURN_IF_ERROR(output_config.MemberAsString("name", &output_name, &output_name_len));
      
      // Create output
      TRITONBACKEND_Output* output;
      RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
          response, &output, output_name, ConvertDataType(output_tensor.scalar_type()),
          output_tensor.sizes().data(), output_tensor.dim()));
      
      // Allocate output buffer
      void* output_buffer;
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
          output, &output_buffer, output_tensor.nbytes(), &output_memory_type,
          &output_memory_type_id));
      
      // Copy output data
      memcpy(output_buffer, output_tensor.data_ptr(), output_tensor.nbytes());
    }
  }
  else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "PyTorch backend only supports tensor or tuple of tensor outputs");
  }

  return nullptr;  // success
}

}}}  // namespace triton::backend::pytorch

/////////////

extern "C" {

// Triton calls TRITONBACKEND_Initialize when the backend is loaded
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

#ifdef __APPLE__
  // Log MPS availability information
  mps_utils::LogMPSInfo();
#endif

  // We should check the backend API version that Triton supports
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton backend API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support this backend")
            .c_str());
  }

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
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
       std::to_string(version) + ")")
          .c_str());

  // Create model state
  triton::backend::pytorch::ModelState* model_state;
  RETURN_IF_ERROR(triton::backend::pytorch::ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // Load the model
  RETURN_IF_ERROR(model_state->LoadModel());

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is unloaded
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  triton::backend::pytorch::ModelState* model_state =
      reinterpret_cast<triton::backend::pytorch::ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model instance is created
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
       " (device " + std::to_string(device_id) + ")")
          .c_str());

  // Get model state
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  triton::backend::pytorch::ModelState* model_state =
      reinterpret_cast<triton::backend::pytorch::ModelState*>(vmodelstate);

  // Create instance state
  triton::backend::pytorch::ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      triton::backend::pytorch::ModelInstanceState::Create(
          model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  // Log device information
  torch::Device device = instance_state->GetDevice();
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("PyTorch instance initialized on device: ") + device.str()).c_str());

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model instance is destroyed
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  triton::backend::pytorch::ModelInstanceState* instance_state =
      reinterpret_cast<triton::backend::pytorch::ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceExecute to perform inference
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Get instance state
  triton::backend::pytorch::ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  triton::backend::pytorch::ModelState* model_state =
      instance_state->StateForModel();

  // Process each request
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    // Create response
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));

    // Execute request
    TRITONSERVER_Error* err = triton::backend::pytorch::ExecuteRequest(
        model_state, instance_state, request, response);

    if (err != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
          "failed to send error response");
      TRITONSERVER_ErrorDelete(err);
    }
    else {
      // Send the response
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send response");
    }

    // Release request
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed to release request");
  }

  return nullptr;  // success
}

}  // extern "C"