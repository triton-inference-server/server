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
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <queue>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace coreml {

// CoreML backend implementation for Triton
//
// This backend loads and executes Core ML models (.mlmodel or .mlpackage).
// The model must be placed in the model directory as 'model.mlmodel' or
// 'model.mlpackage'.

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

// Convert between Triton and CoreML data types
static MLMultiArrayDataType
ConvertToCoreMLDataType(TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_FP32:
      return MLMultiArrayDataTypeFloat32;
    case TRITONSERVER_TYPE_FP64:
      return MLMultiArrayDataTypeDouble;
    case TRITONSERVER_TYPE_INT32:
      return MLMultiArrayDataTypeInt32;
    default:
      // CoreML has limited data type support
      return MLMultiArrayDataTypeFloat32;
  }
}

static TRITONSERVER_DataType
ConvertFromCoreMLDataType(MLMultiArrayDataType dtype)
{
  switch (dtype) {
    case MLMultiArrayDataTypeFloat32:
      return TRITONSERVER_TYPE_FP32;
    case MLMultiArrayDataTypeDouble:
      return TRITONSERVER_TYPE_FP64;
    case MLMultiArrayDataTypeInt32:
      return TRITONSERVER_TYPE_INT32;
    default:
      return TRITONSERVER_TYPE_FP32;
  }
}

// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Validate that model configuration is supported
  TRITONSERVER_Error* ValidateModelConfig();

  // Get the CoreML model URL
  const NSURL* GetModelURL() const { return model_url_; }

  // Get compute units configuration
  MLComputeUnits GetComputeUnits() const { return compute_units_; }

  // Get whether to enable Neural Engine
  bool UseNeuralEngine() const { return use_neural_engine_; }

  // Get power efficiency preference
  bool PreferPowerEfficiency() const { return prefer_power_efficiency_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  // Model URL
  NSURL* model_url_;

  // Compute configuration
  MLComputeUnits compute_units_;
  bool use_neural_engine_;
  bool prefer_power_efficiency_;
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

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), model_url_(nullptr),
      compute_units_(MLComputeUnitsAll), use_neural_engine_(true),
      prefer_power_efficiency_(false)
{
  // Get the model configuration
  model_config_.reset(new triton::common::TritonJson::Value());
  THROW_IF_BACKEND_MODEL_ERROR(
      backend::CommonGetModelConfig(triton_model, model_config_.get()));

  // Validate and initialize based on configuration
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());

  // Construct the path to the model file
  auto model_dir = JoinPath({RepositoryPath(), std::to_string(Version())});
  
  // Check for .mlpackage first (preferred format)
  std::string mlpackage_path = JoinPath({model_dir, "model.mlpackage"});
  std::string mlmodel_path = JoinPath({model_dir, "model.mlmodel"});
  
  NSFileManager* fileManager = [NSFileManager defaultManager];
  NSString* nsMLPackagePath = [NSString stringWithUTF8String:mlpackage_path.c_str()];
  NSString* nsMLModelPath = [NSString stringWithUTF8String:mlmodel_path.c_str()];
  
  if ([fileManager fileExistsAtPath:nsMLPackagePath]) {
    model_url_ = [NSURL fileURLWithPath:nsMLPackagePath];
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Loading CoreML model from: ") + mlpackage_path).c_str());
  } else if ([fileManager fileExistsAtPath:nsMLModelPath]) {
    model_url_ = [NSURL fileURLWithPath:nsMLModelPath];
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Loading CoreML model from: ") + mlmodel_path).c_str());
  } else {
    THROW_IF_BACKEND_MODEL_ERROR(
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_NOT_FOUND,
            (std::string("Could not find model file at ") + mlpackage_path +
             " or " + mlmodel_path).c_str()));
  }

  // Parse backend-specific parameters
  triton::common::TritonJson::Value params;
  if (model_config_->Find("parameters", &params)) {
    // Check for compute units preference
    triton::common::TritonJson::Value compute_units_param;
    if (params.Find("compute_units", &compute_units_param)) {
      std::string compute_units_str;
      THROW_IF_BACKEND_MODEL_ERROR(
          compute_units_param.MemberAsString("string_value", &compute_units_str));
      
      if (compute_units_str == "CPU_ONLY") {
        compute_units_ = MLComputeUnitsCPUOnly;
      } else if (compute_units_str == "CPU_AND_GPU") {
        compute_units_ = MLComputeUnitsCPUAndGPU;
      } else if (compute_units_str == "ALL") {
        compute_units_ = MLComputeUnitsAll;
      } else if (compute_units_str == "CPU_AND_NE") {
        compute_units_ = MLComputeUnitsCPUAndNeuralEngine;
      }
    }

    // Check for Neural Engine preference
    triton::common::TritonJson::Value ne_param;
    if (params.Find("use_neural_engine", &ne_param)) {
      std::string ne_str;
      THROW_IF_BACKEND_MODEL_ERROR(
          ne_param.MemberAsString("string_value", &ne_str));
      use_neural_engine_ = (ne_str == "true" || ne_str == "1");
    }

    // Check for power efficiency preference
    triton::common::TritonJson::Value power_param;
    if (params.Find("prefer_power_efficiency", &power_param)) {
      std::string power_str;
      THROW_IF_BACKEND_MODEL_ERROR(
          power_param.MemberAsString("string_value", &power_str));
      prefer_power_efficiency_ = (power_str == "true" || power_str == "1");
    }
  }
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_->PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  // Check batching support
  triton::common::TritonJson::Value max_batch_size_value;
  RETURN_IF_ERROR(model_config_->Find("max_batch_size", &max_batch_size_value));
  
  int64_t max_batch_size = 0;
  RETURN_IF_ERROR(max_batch_size_value.AsInt(&max_batch_size));
  
  // CoreML doesn't natively support batching in the same way as other frameworks
  // We'll handle batch size 1 or implement manual batching
  if (max_batch_size > 1) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        "CoreML backend will process batches sequentially. Native batching is not supported.");
  }

  return nullptr;  // success
}

// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute inference requests
  void ProcessRequests(
      const uint32_t request_count, TRITONBACKEND_Request** requests);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  TRITONSERVER_Error* LoadModel();

  ModelState* model_state_;
  
  // CoreML model and configuration
  MLModel* model_;
  MLModelConfiguration* model_config_;
  
  // Input/output feature names
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  
  // Cached input/output descriptions
  std::unordered_map<std::string, MLFeatureDescription*> input_descriptions_;
  std::unordered_map<std::string, MLFeatureDescription*> output_descriptions_;
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

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), model_(nullptr), model_config_(nullptr)
{
  THROW_IF_BACKEND_MODEL_ERROR(LoadModel());
}

ModelInstanceState::~ModelInstanceState()
{
  // Objective-C objects will be automatically released by ARC
}

TRITONSERVER_Error*
ModelInstanceState::LoadModel()
{
  // Create model configuration
  model_config_ = [[MLModelConfiguration alloc] init];
  
  // Set compute units based on model configuration
  model_config_.computeUnits = model_state_->GetComputeUnits();
  
  // Set additional configuration options
  if (@available(macOS 11.0, iOS 14.0, *)) {
    // Enable or disable specific compute units
    if (!model_state_->UseNeuralEngine() && 
        model_config_.computeUnits == MLComputeUnitsAll) {
      model_config_.computeUnits = MLComputeUnitsCPUAndGPU;
    }
  }

  // Load the model
  NSError* error = nil;
  model_ = [MLModel modelWithContentsOfURL:model_state_->GetModelURL()
                             configuration:model_config_
                                     error:&error];
  
  if (error != nil) {
    NSString* errorString = [error localizedDescription];
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("Failed to load CoreML model: ") + 
         [errorString UTF8String]).c_str());
  }

  // Get model description
  MLModelDescription* description = model_.modelDescription;
  
  // Cache input descriptions
  NSDictionary<NSString*, MLFeatureDescription*>* inputs = 
      description.inputDescriptionsByName;
  for (NSString* key in inputs) {
    std::string input_name = [key UTF8String];
    input_names_.push_back(input_name);
    input_descriptions_[input_name] = inputs[key];
  }

  // Cache output descriptions
  NSDictionary<NSString*, MLFeatureDescription*>* outputs = 
      description.outputDescriptionsByName;
  for (NSString* key in outputs) {
    std::string output_name = [key UTF8String];
    output_names_.push_back(output_name);
    output_descriptions_[output_name] = outputs[key];
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Successfully loaded CoreML model with ") +
       std::to_string(input_names_.size()) + " inputs and " +
       std::to_string(output_names_.size()) + " outputs").c_str());

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    const uint32_t request_count, TRITONBACKEND_Request** requests)
{
  // Process each request
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    
    // Create response
    TRITONBACKEND_Response* response;
    RESPOND_IF_ERROR(
        nullptr, TRITONBACKEND_ResponseNew(&response, request));

    // Get input count
    uint32_t input_count;
    RESPOND_AND_SET_NULL_IF_ERROR(
        response, TRITONBACKEND_RequestInputCount(request, &input_count));

    // Create feature provider dictionary
    NSMutableDictionary* featureDict = [[NSMutableDictionary alloc] init];

    // Process each input
    for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
      TRITONBACKEND_Input* input;
      RESPOND_AND_SET_NULL_IF_ERROR(
          response, TRITONBACKEND_RequestInputByIndex(
              request, input_idx, &input));

      const char* input_name;
      TRITONSERVER_DataType input_datatype;
      const int64_t* input_shape;
      uint32_t input_dims_count;
      RESPOND_AND_SET_NULL_IF_ERROR(
          response,
          TRITONBACKEND_InputProperties(
              input, &input_name, &input_datatype, &input_shape,
              &input_dims_count, nullptr, nullptr));

      // Get the expected feature description
      auto it = input_descriptions_.find(input_name);
      if (it == input_descriptions_.end()) {
        RESPOND_AND_SET_NULL_IF_ERROR(
            response,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("Unknown input '") + input_name + "'").c_str()));
        continue;
      }

      MLFeatureDescription* featureDesc = it->second;
      
      // Only support multiArray type for now
      if (featureDesc.type != MLFeatureTypeMultiArray) {
        RESPOND_AND_SET_NULL_IF_ERROR(
            response,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                (std::string("Input '") + input_name + 
                 "' is not a MultiArray type").c_str()));
        continue;
      }

      // Create shape array for MLMultiArray
      NSMutableArray<NSNumber*>* shape = [[NSMutableArray alloc] init];
      size_t total_size = 1;
      for (uint32_t i = 0; i < input_dims_count; i++) {
        [shape addObject:@(input_shape[i])];
        total_size *= input_shape[i];
      }

      // Get input buffer
      const void* input_buffer;
      size_t input_byte_size;
      TRITONSERVER_MemoryType input_memory_type;
      int64_t input_memory_type_id;
      RESPOND_AND_SET_NULL_IF_ERROR(
          response,
          TRITONBACKEND_InputBuffer(
              input, 0, &input_buffer, &input_byte_size, &input_memory_type,
              &input_memory_type_id));

      // Create MLMultiArray
      NSError* error = nil;
      MLMultiArray* multiArray = [[MLMultiArray alloc]
          initWithShape:shape
               dataType:ConvertToCoreMLDataType(input_datatype)
                  error:&error];
      
      if (error != nil) {
        NSString* errorString = [error localizedDescription];
        RESPOND_AND_SET_NULL_IF_ERROR(
            response,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                (std::string("Failed to create MLMultiArray: ") + 
                 [errorString UTF8String]).c_str()));
        continue;
      }

      // Copy data to MLMultiArray
      size_t element_size = input_byte_size / total_size;
      memcpy(multiArray.dataPointer, input_buffer, input_byte_size);

      // Add to feature provider
      NSString* nsInputName = [NSString stringWithUTF8String:input_name];
      MLFeatureValue* featureValue = [MLFeatureValue featureValueWithMultiArray:multiArray];
      featureDict[nsInputName] = featureValue;
    }

    // Create feature provider
    NSError* error = nil;
    MLDictionaryFeatureProvider* featureProvider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:featureDict
                                                           error:&error];
    
    if (error != nil) {
      NSString* errorString = [error localizedDescription];
      RESPOND_AND_SET_NULL_IF_ERROR(
          response,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("Failed to create feature provider: ") + 
               [errorString UTF8String]).c_str()));
      continue;
    }

    // Create prediction options
    MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
    
    // Perform prediction
    id<MLFeatureProvider> output = [model_ predictionFromFeatures:featureProvider
                                                          options:options
                                                            error:&error];
    
    if (error != nil) {
      NSString* errorString = [error localizedDescription];
      RESPOND_AND_SET_NULL_IF_ERROR(
          response,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("Prediction failed: ") + 
               [errorString UTF8String]).c_str()));
      continue;
    }

    // Process outputs
    for (const auto& output_name : output_names_) {
      NSString* nsOutputName = [NSString stringWithUTF8String:output_name.c_str()];
      MLFeatureValue* outputValue = [output featureValueForName:nsOutputName];
      
      if (outputValue == nil) {
        RESPOND_AND_SET_NULL_IF_ERROR(
            response,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                (std::string("Missing output '") + output_name + "'").c_str()));
        continue;
      }

      // Only support multiArray output for now
      if (outputValue.type != MLFeatureTypeMultiArray) {
        RESPOND_AND_SET_NULL_IF_ERROR(
            response,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                (std::string("Output '") + output_name + 
                 "' is not a MultiArray type").c_str()));
        continue;
      }

      MLMultiArray* outputArray = outputValue.multiArrayValue;
      
      // Get output shape
      std::vector<int64_t> output_shape;
      for (NSNumber* dim in outputArray.shape) {
        output_shape.push_back([dim longLongValue]);
      }

      // Create output
      TRITONBACKEND_Output* output;
      RESPOND_AND_SET_NULL_IF_ERROR(
          response,
          TRITONBACKEND_ResponseOutput(
              response, &output, output_name.c_str(),
              ConvertFromCoreMLDataType(outputArray.dataType),
              output_shape.data(), output_shape.size()));

      // Get output buffer
      void* output_buffer;
      size_t output_byte_size = outputArray.count * 
          TRITONSERVER_DataTypeByteSize(ConvertFromCoreMLDataType(outputArray.dataType));
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;

      RESPOND_AND_SET_NULL_IF_ERROR(
          response,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, output_byte_size, &output_memory_type,
              &output_memory_type_id));

      // Copy output data
      memcpy(output_buffer, outputArray.dataPointer, output_byte_size);
    }

    // Send response
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
        "failed to send response");
  }
}

/////////////

extern "C" {

// Triton backend API implementation

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

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

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

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
       " (GPU device " + std::to_string(device_id) + ", kind " +
       TRITONSERVER_InstanceGroupKindString(kind) + ")")
          .c_str());

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

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

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(request_count, requests);

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::coreml