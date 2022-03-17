// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <atomic>
#include <memory>
#include <thread>
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace distributed_addsub {


// Addsub backend that distributes partial computation to different model
// instances and gather the results to form the response internally.
// This backend is designed in the way that CPU instance will perform add task
// and GPU instance will perform sub task, and only the CPU instances will
// accept inference request from Triton core. Note that GPU instance has
// different meaning in this backend.
//
// The backend supports models that take two input tensors, two variable-size
// INT32 [ -1 ] value inputs INPUT0 and INPUT1; and produces two output tensors:
// OUTPUT0 as the element-wise sum of INPUT0 and INPUT1, OUTPUT1 as
// the element-wise difference of INPUT0 and INPUT1
//

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

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelInstanceState;
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Keep track of the model instance that will only accept works distributed
  // from within the model (instance)
  void AddSubInstance(ModelInstanceState* instance) { instance_ = instance; }

  ModelInstanceState* SubInstance() { return instance_; }

  std::atomic<size_t> instance_counter_;

 private:
  ModelState(TRITONBACKEND_Model* triton_model)
      : BackendModel(triton_model), instance_counter_(0), instance_(nullptr)
  {
  }

  ModelInstanceState* instance_;
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
  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // There must be one INT32 input called INPUT defined in the model
  // configuration and it must be a 1D vector (of any length).
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model must have two inputs"));

  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model must have two outputs"));

  int64_t dim_value = 0;
  for (size_t idx = 0; idx < 2; ++idx) {
    common::TritonJson::Value input;
    RETURN_IF_ERROR(inputs.IndexAsObject(idx, &input));

    std::vector<int64_t> input_shape;
    RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));

    RETURN_ERROR_IF_FALSE(
        input_shape.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model must have input of one-dimensional shape"));
    if (idx == 0) {
      dim_value = input_shape[0];
    } else {
      RETURN_ERROR_IF_FALSE(
          dim_value == input_shape[0], TRITONSERVER_ERROR_INVALID_ARG,
          std::string("model must have consistent shape for all tensors"));
    }

    std::string input_dtype;
    RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));

    RETURN_ERROR_IF_FALSE(
        input_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model input must have TYPE_INT32 data-type"));

    const char* input_name;
    size_t input_name_len;
    RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));

    auto expected_name = (std::string("INPUT") + std::to_string(idx));
    RETURN_ERROR_IF_FALSE(
        expected_name == input_name, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model input must be named '") + expected_name +
            "' at index " + std::to_string(idx));

    common::TritonJson::Value output;
    RETURN_IF_ERROR(outputs.IndexAsObject(idx, &output));

    std::vector<int64_t> output_shape;
    RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

    RETURN_ERROR_IF_FALSE(
        (output_shape.size() == 1) && (output_shape[0] == input_shape[0]),
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model must have consistent shape for all tensors"));

    std::string output_dtype;
    RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));

    RETURN_ERROR_IF_FALSE(
        output_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model output must have TYPE_INT32 data-type"));

    const char* output_name;
    size_t output_name_len;
    RETURN_IF_ERROR(
        output.MemberAsString("name", &output_name, &output_name_len));

    expected_name = (std::string("OUTPUT") + std::to_string(idx));
    RETURN_ERROR_IF_FALSE(
        expected_name == output_name, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model output must be named '") + expected_name +
            "' at index " + std::to_string(idx));
  }

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }
  bool IsPassive() const { return passive_; }

  TRITONSERVER_Error* Add(
      const size_t element_count, const int32_t* input_0,
      const int32_t* input_1, int32_t* output);
  TRITONSERVER_Error* Sub(
      const size_t element_count, const int32_t* input_0,
      const int32_t* input_1, int32_t* output);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  ModelState* model_state_;
  bool passive_;
};

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
  // Check if the setup is correct
  THROW_IF_BACKEND_INSTANCE_ERROR(
      TRITONBACKEND_ModelInstanceIsPassive(triton_model_instance, &passive_));
  switch (kind_) {
    case TRITONSERVER_INSTANCEGROUPKIND_CPU: {
      if (passive_) {
        throw BackendModelInstanceException(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string("CPU instance should not be passive").c_str()));
      }
      break;
    }
    case TRITONSERVER_INSTANCEGROUPKIND_GPU:
      if (!passive_) {
        throw BackendModelInstanceException(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string("GPU instance should be passive").c_str()));
      }
      break;
    default:
      throw BackendModelInstanceException(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("instance kind must be CPU or GPU").c_str()));
      break;
  }
}

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

TRITONSERVER_Error*
ModelInstanceState::Add(
    const size_t element_count, const int32_t* input_0, const int32_t* input_1,
    int32_t* output)
{
  if (kind_ != TRITONSERVER_INSTANCEGROUPKIND_CPU) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string("Add operation must be done by CPU instance").c_str());
  }

  for (size_t i = 0; i < element_count; ++i) {
    output[i] = input_0[i] + input_1[i];
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::Sub(
    const size_t element_count, const int32_t* input_0, const int32_t* input_1,
    int32_t* output)
{
  if (kind_ != TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string("Sub operation must be done by GPU instance").c_str());
  }

  for (size_t i = 0; i < element_count; ++i) {
    output[i] = input_0[i] - input_1[i];
  }

  return nullptr;  // success
}

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
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments. This backend doesn't use
  // any such configuration but we print whatever is available.
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

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

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
  TRITONSERVER_Error* err = nullptr;
  if (model_state->instance_counter_ != 0) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Unexpected unfinalized model instance(s)");
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return err;
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
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
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));
  if (instance_state->IsPassive()) {
    model_state->AddSubInstance(instance_state);
  }

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
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

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
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
  // we should not return from this function until execution is complete. Triton
  // will automatically release 'instance' on return from this function so that
  // it is again available to be used for another call to
  // TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + " (" +
       TRITONSERVER_InstanceGroupKindString(instance_state->Kind()) +
       " device " + std::to_string(instance_state->DeviceId()) + ")" +
       ", executing " + std::to_string(request_count) + " requests")
          .c_str());


  if (instance_state->Kind() != TRITONSERVER_INSTANCEGROUPKIND_CPU) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Unexpected inference request sent to non-CPU instance");
  }

  auto sub_instance_state = model_state->SubInstance();

  // 'responses' is initialized with the response objects below and
  // if/when an error response is sent the corresponding entry in
  // 'responses' is set to nullptr to indicate that that response has
  // already been sent.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  // Create a single response object for each request. If something
  // goes wrong when attempting to create the response objects just
  // fail all of the requests by returning an error.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // After this point we take ownership of 'requests', which means that a
  // response must be sent for every request. If something does go wrong in
  // processing a particular request then we send an error response just for the
  // specific request.

  // The way we collect these batch timestamps is not entirely accurate.
  // Normally, in a performant backend you would execute all the requests at the
  // same time, and so there would be a single compute-start / compute-end
  // time-range. But here we execute each request separately so there is no
  // single range. As a result we just show the entire execute time as being the
  // compute time as well.
  uint64_t batch_exec_start_ns = 0;
  SET_TIMESTAMP(batch_exec_start_ns);
  uint64_t batch_exec_end_ns = 0;
  uint64_t total_batch_size = 0;

  // For simplicity we just process each request separately... in
  // general a backend should try to operate on the entire batch of
  // requests at the same time for improved performance.
  std::vector<uint8_t> start_buffer, ready_buffer, input_buffer;
  for (uint32_t r = 0; r < request_count; ++r) {
    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);

    TRITONBACKEND_Request* request = requests[r];

    uint32_t input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInputCount(request, &input_count));

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    // If an error response was sent for the above then display an error
    // message and move on to next request.
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read request input/output counts, error response "
           "sent")
              .c_str());
      continue;
    }

    TRITONBACKEND_Input* input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, "INPUT0", &input));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input, error response sent")
              .c_str());
      continue;
    }

    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    uint32_t input_buffer_count;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            input, nullptr /* input_name */, &input_datatype, &input_shape,
            &input_dims_count, &input_byte_size, &input_buffer_count));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input properties, error response sent")
              .c_str());
      continue;
    }
    if (input_dims_count > 1) {
      total_batch_size += input_shape[0];
    } else {
      ++total_batch_size;
    }

    std::vector<char> input_0(input_byte_size);
    std::vector<char> input_1(input_byte_size);
    uint64_t input_0_byte_size = input_byte_size;
    uint64_t input_1_byte_size = input_byte_size;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        ReadInputTensor(request, "INPUT0", input_0.data(), &input_0_byte_size));
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        ReadInputTensor(request, "INPUT1", input_1.data(), &input_1_byte_size));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to get input buffer in CPU memory, error "
           "response sent")
              .c_str());
      continue;
    }

    // Compute... Get GPU instance from model state and let it compute
    // the subtraction, while the CPU instance computes the addition.
    // In real world some parallelization should be used, but here just
    // seralize the "distributed" work.
    TRITONBACKEND_Response* response = responses[r];

    uint64_t compute_start_ns = 0;
    SET_TIMESTAMP(compute_start_ns);
    for (size_t out_idx = 0; out_idx < requested_output_count; ++out_idx) {
      const char* output_name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(request, out_idx, &output_name));

      TRITONBACKEND_Output* output;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &output, output_name, input_datatype, input_shape,
              input_dims_count));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to create response output, error response sent")
                .c_str());
        break;
      }

      // Get the output buffer. We request a buffer in CPU memory but we have
      // to handle any returned type. If we get back a buffer in GPU memory we
      // just fail the request.
      void* output_buffer;
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, input_byte_size, &output_memory_type,
              &output_memory_type_id));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to create output buffer in CPU memory, error "
             "response sent")
                .c_str());
        break;
      }
      if (output_memory_type == TRITONSERVER_MEMORY_GPU) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to create output buffer in CPU memory"));
        break;
      }

      static std::string output_0_name("OUTPUT0");
      if (output_0_name == output_name) {
        instance_state->Add(
            (input_byte_size / sizeof(int32_t)),
            reinterpret_cast<int32_t*>(input_0.data()),
            reinterpret_cast<int32_t*>(input_1.data()),
            reinterpret_cast<int32_t*>(output_buffer));
      } else {
        sub_instance_state->Sub(
            (input_byte_size / sizeof(int32_t)),
            reinterpret_cast<int32_t*>(input_0.data()),
            reinterpret_cast<int32_t*>(input_1.data()),
            reinterpret_cast<int32_t*>(output_buffer));
      }
    }
    uint64_t compute_end_ns = 0;
    SET_TIMESTAMP(compute_end_ns);

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);
    batch_exec_end_ns = exec_end_ns;

    // Send all the responses that haven't already been sent because of an
    // earlier error.
    if (responses[r] != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
              nullptr /* success */),
          "failed sending response");
    }

    // Report statistics for each request.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_state->TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), total_batch_size,
          batch_exec_start_ns, batch_exec_start_ns, batch_exec_end_ns,
          batch_exec_end_ns),
      "failed reporting batch request statistics");

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::distributed_addsub
