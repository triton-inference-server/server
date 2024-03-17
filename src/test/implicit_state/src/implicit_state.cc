// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <vector>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace implicit {

// Implicit state backend that is solely used with testing implicit state
// management functionality in the backend API.
//
// The backend supports models that take 4 input tensors, three INT32 [ 1 ]
// control values, one UINT64 [ 1 ] correlation ID control, one INT32 [ 1 ]
// value input, and one INT32 [ 1 ] input indicating the test case. The input
// tensors must be named "START", "END", "READY", "CORRID", "UPDATE", "INPUT",
// and "TEST_CASE". The output tensor must be named "OUTPUT".
//
// The list of accepted values for the "TEST_CASE" field are:
//
//   * STATE_NEW_NON_EXISTENT = 0: This tests calling the TRITONBACKEND_StateNew
//   for a non existent state or a model that doesn't have states section in
//   sequence batching.
//
//   * STATE_UPDATE_FALSE = 1: Tests not calling the state update and expecting
//   the implicit state to not be updated.
//
//   * USE_SINGLE_STATE_BUFFER = 2: For this scenario we will be using the same
//   buffer for both input and output state. In total there will be 3 requests
//   sent in a sequence.
//
//   * USE_GROWABLE_STATE_BUFFER = 3: In this test case we use growable state
//   buffer. Currently, growable state buffer only supports CUDA memory.

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, REQUEST, X)            \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_RequestRelease(                               \
                REQUEST, TRITONSERVER_REQUEST_RELEASE_ALL),             \
            "failed to release the request.");                          \
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
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
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
    : BackendModel(triton_model)
{
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

  // The model configuration must specify the sequence batcher and
  // must use the START, END, READ and CORRID input to indicate
  // control values.
  triton::common::TritonJson::Value sequence_batching;
  RETURN_IF_ERROR(
      model_config_.MemberAsObject("sequence_batching", &sequence_batching));
  common::TritonJson::Value control_inputs;
  RETURN_IF_ERROR(
      sequence_batching.MemberAsArray("control_input", &control_inputs));
  RETURN_ERROR_IF_FALSE(
      control_inputs.ArraySize() == 3, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("'START', 'END, and 'READY' must be configured as "
                  "the control inputs"));

  std::vector<std::string> control_input_names;
  for (size_t io_index = 0; io_index < control_inputs.ArraySize(); io_index++) {
    common::TritonJson::Value control_input;
    RETURN_IF_ERROR(control_inputs.IndexAsObject(io_index, &control_input));
    const char* input_name = nullptr;
    size_t input_name_len;
    RETURN_IF_ERROR(
        control_input.MemberAsString("name", &input_name, &input_name_len));
    control_input_names.push_back(input_name);
  }

  RETURN_ERROR_IF_FALSE(
      ((std::find(
            control_input_names.begin(), control_input_names.end(), "START") !=
        control_input_names.end()) ||
       (std::find(
            control_input_names.begin(), control_input_names.end(), "END") !=
        control_input_names.end()) ||
       (std::find(
            control_input_names.begin(), control_input_names.end(), "READY") !=
        control_input_names.end())),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("'START', 'END, and 'READY' must be configured as "
                  "the control inputs"));

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

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }
  void* state_ = nullptr;

  // Index of the request in the sequence
  uint32_t request_index_ = 0;

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  ModelState* model_state_;
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
      model_state_(model_state)
{
}

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

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
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
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  bool supports_batching = false;
  RETURN_IF_ERROR(model_state->SupportsFirstDimBatching(&supports_batching));

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

  // The way we collect these batch timestamps is not entirely
  // accurate. Normally, in a performant backend you would execute all
  // the requests at the same time, and so there would be a single
  // compute-start / compute-end time-range. But here we execute each
  // request separately so there is no single range. As a result we
  // just show the entire execute time as being the compute time as
  // well.
  uint64_t min_exec_start_ns = std::numeric_limits<uint64_t>::max();
  uint64_t max_exec_end_ns = 0;
  uint64_t total_batch_size = 0;

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  // For simplicity we just process each request separately... in
  // general a backend should try to operate on the entire batch of
  // requests at the same time for improved performance.
  std::vector<uint8_t> start_buffer, end_buffer, ready_buffer, corrid_buffer,
      input_buffer;
  for (uint32_t r = 0; r < request_count; ++r) {
    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);
    min_exec_start_ns = std::min(min_exec_start_ns, exec_start_ns);

    TRITONBACKEND_Request* request = requests[r];

    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request, TRITONBACKEND_RequestId(request, &request_id));

    uint32_t input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
        TRITONBACKEND_RequestInputCount(request, &input_count));

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
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

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("request ") + std::to_string(r) + ": id = \"" +
         request_id + "\", input_count = " + std::to_string(input_count) +
         ", requested_output_count = " + std::to_string(requested_output_count))
            .c_str());

    // For statistics we need to collect the total batch size of all the
    // requests. If the model doesn't support batching then each request is
    // necessarily batch-size 1. If the model does support batching then the
    // first dimension of the shape is the batch size. We only the first input
    // for this.
    if (supports_batching) {
      TRITONBACKEND_Input* input = nullptr;
      GUARDED_RESPOND_IF_ERROR(
          responses, r, request,
          TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to read input, error response sent")
                .c_str());
        continue;
      }

      const int64_t* input_shape;
      uint32_t input_dims_count;
      GUARDED_RESPOND_IF_ERROR(
          responses, r, request,
          TRITONBACKEND_InputProperties(
              input, nullptr, nullptr, &input_shape, &input_dims_count, nullptr,
              nullptr));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to read input properties, error response sent")
                .c_str());
        continue;
      }

      if (input_dims_count > 0) {
        if (input_shape[0] != 1) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r, request,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "unable to execute more than one timestep at a time"));
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": unable to execute more than one timestep at a time, error "
               "response sent")
                  .c_str());
          continue;
        }
        total_batch_size += input_shape[0];
      }
    } else {
      total_batch_size++;
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("total_batch_size: ") + std::to_string(total_batch_size))
            .c_str());

    std::set<uint64_t> seen_corrids;

    // Get the input tensors.
    TRITONBACKEND_Input* start_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
        TRITONBACKEND_RequestInput(request, "START", &start_input));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input 'START', error response sent")
              .c_str());
      continue;
    }

    TRITONBACKEND_Input* end_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
        TRITONBACKEND_RequestInput(request, "END", &end_input));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input 'END', error response sent")
              .c_str());
      continue;
    }

    TRITONBACKEND_Input* ready_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
        TRITONBACKEND_RequestInput(request, "READY", &ready_input));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input 'READY', error response sent")
              .c_str());
      continue;
    }

    const void* start_buffer = nullptr;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t input_memory_type_id = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
        TRITONBACKEND_InputBuffer(
            start_input, 0 /* input_buffer_count */, &start_buffer,
            &buffer_byte_size, &input_memory_type, &input_memory_type_id));
    if (responses[r] == nullptr) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r, request,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED, "failed to get input buffer"));
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR, (std::string("request ") + std::to_string(r) +
                                   ": failed to get input buffer, error "
                                   "response sent")
                                      .c_str());
      continue;
    }

    const float* lstart_buffer = reinterpret_cast<const float*>(start_buffer);
    if (*lstart_buffer == 1) {
      instance_state->request_index_ = 0;
      instance_state->state_ = nullptr;
    }

    const void* end_buffer = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
        TRITONBACKEND_InputBuffer(
            end_input, 0 /* input_buffer_count */, &end_buffer,
            &buffer_byte_size, &input_memory_type, &input_memory_type_id));
    if (responses[r] == nullptr) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r, request,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED, "failed to get input buffer"));
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR, (std::string("request ") + std::to_string(r) +
                                   ": failed to get input buffer, error "
                                   "response sent")
                                      .c_str());
      continue;
    }

    const void* ready_buffer = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
        TRITONBACKEND_InputBuffer(
            ready_input, 0 /* input_buffer_count */, &ready_buffer,
            &buffer_byte_size, &input_memory_type, &input_memory_type_id));
    if (responses[r] == nullptr) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r, request,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED, "failed to get input buffer"));
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR, (std::string("request ") + std::to_string(r) +
                                   ": failed to get input buffer, error "
                                   "response sent")
                                      .c_str());
      continue;
    }

    TRITONBACKEND_Input* input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
        TRITONBACKEND_RequestInput(request, "INPUT", &input));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input 'INPUT', error response sent")
              .c_str());
      continue;
    }

    const void* input_buffer = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
        TRITONBACKEND_InputBuffer(
            input, 0 /* input_buffer_count */, &input_buffer, &buffer_byte_size,
            &input_memory_type, &input_memory_type_id));
    if ((responses[r] == nullptr) ||
        (input_memory_type == TRITONSERVER_MEMORY_GPU)) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r, request,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "failed to get input buffer in CPU memory"));
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to get input buffer in CPU memory, error "
           "response sent")
              .c_str());
      continue;
    }

    TRITONBACKEND_Input* test_case = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
        TRITONBACKEND_RequestInput(request, "TEST_CASE", &test_case));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input 'TEST_CASE', error response sent")
              .c_str());
      continue;
    }

    const void* test_case_buffer = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, request,
        TRITONBACKEND_InputBuffer(
            test_case, 0 /* test_case_buffer_count */, &test_case_buffer,
            &buffer_byte_size, &input_memory_type, &input_memory_type_id));
    if ((responses[r] == nullptr) ||
        (input_memory_type == TRITONSERVER_MEMORY_GPU)) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r, request,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "failed to get input buffer in CPU memory"));
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to get input buffer in CPU memory, error "
           "response sent")
              .c_str());
      continue;
    }
    const int32_t test_case_buffer_int =
        *reinterpret_cast<const int32_t*>(test_case_buffer);
    const int32_t ipbuffer_int =
        *reinterpret_cast<const int32_t*>(input_buffer);
    int32_t ipbuffer_state_int = 0;

    if (test_case_buffer_int != 0) {
      TRITONBACKEND_Input* input_state = nullptr;
      GUARDED_RESPOND_IF_ERROR(
          responses, r, request,
          TRITONBACKEND_RequestInput(request, "INPUT_STATE", &input_state));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to read input 'INPUT_STATE', error response sent")
                .c_str());
        continue;
      }

      const void* input_state_buffer = nullptr;
      GUARDED_RESPOND_IF_ERROR(
          responses, r, request,
          TRITONBACKEND_InputBuffer(
              input_state, 0 /* input_buffer_count */, &input_state_buffer,
              &buffer_byte_size, &input_memory_type, &input_memory_type_id));
      if ((responses[r] == nullptr) ||
          (test_case_buffer_int == 3 &&
           input_memory_type != TRITONSERVER_MEMORY_GPU)) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "growable memory should always provide memory in GPU"));
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to get input buffer in GPU memory, error "
             "response sent")
                .c_str());
        continue;
      } else if (
          (responses[r] == nullptr) ||
          (input_memory_type == TRITONSERVER_MEMORY_GPU &&
           test_case_buffer_int != 3)) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to get input buffer in CPU memory"));
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to get input buffer in CPU memory, error "
             "response sent")
                .c_str());
        continue;
      }

      // When using single state buffer, input/output tensors should point to
      // the buffer.
      if ((test_case_buffer_int == 2 || test_case_buffer_int == 3) &&
          instance_state->state_ != nullptr) {
        if (input_state_buffer != instance_state->state_) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r, request,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "Input and output state are using different buffers."));
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": input and output state are using different buffers, error "
               "response sent")
                  .c_str());
          continue;
        }
      }

      if (test_case_buffer_int == 2 || test_case_buffer_int == 1 ||
          test_case_buffer_int == 0) {
        const int32_t ipbuffer_state =
            *reinterpret_cast<const int32_t*>(input_state_buffer);
        ipbuffer_state_int = ipbuffer_state;
      }
    }

    switch (test_case_buffer_int) {
      // STATE_NEW_NON_EXISTENT. The behavior for both of the test cases is
      // the same.
      case 0: {
        TRITONBACKEND_State* response_state;
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_StateNew(
                &response_state, request, "undefined_state",
                TRITONSERVER_TYPE_INT32, nullptr /* shape */,
                0 /* dim_count */));
        if (responses[r] == nullptr) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to create the output state 'OUTPUT_STATE', error "
               "response sent")
                  .c_str());
          continue;
        }
      } break;
      // STATE_UPDATE_FALSE
      case 1: {
        TRITONBACKEND_State* response_state;
        TRITONBACKEND_Output* response_output;
        std::vector<int64_t> shape{1};
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_StateNew(
                &response_state, request, "OUTPUT_STATE",
                TRITONSERVER_TYPE_INT32, shape.data() /* data */,
                shape.size() /* dim_count */));

        if (responses[r] == nullptr) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to create the output state 'OUTPUT_STATE', error "
               "response sent")
                  .c_str());
          continue;
        }
        TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_GPU;
        int64_t actual_memory_type_id = 0;
        char* buffer;

        // Request an output buffer in GPU. This is only for testing purposes
        // to make sure that GPU output buffers can be requested.
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_StateBuffer(
                response_state, reinterpret_cast<void**>(&buffer),
                sizeof(int32_t), &actual_memory_type, &actual_memory_type_id));


        if ((responses[r] == nullptr) ||
            (actual_memory_type == TRITONSERVER_MEMORY_CPU)) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r, request,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "failed to create the state buffer in GPU memory"));
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to create the state buffer in GPU memory, error "
               "response sent")
                  .c_str());
          continue;
        }

        actual_memory_type = TRITONSERVER_MEMORY_CPU;
        actual_memory_type_id = 0;
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_StateBuffer(
                response_state, reinterpret_cast<void**>(&buffer),
                sizeof(int32_t), &actual_memory_type, &actual_memory_type_id));

        if ((responses[r] == nullptr) ||
            (actual_memory_type == TRITONSERVER_MEMORY_GPU)) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r, request,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "failed to create the state buffer in CPU memory"));
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to create the state buffer in CPU memory, error "
               "response sent")
                  .c_str());
          continue;
        }

        TRITONSERVER_BufferAttributes* buffer_attributes;
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_StateBufferAttributes(
                response_state, &buffer_attributes));

        // Testing for the StateBuffer attributes
        TRITONSERVER_MemoryType ba_memory_type;
        int64_t ba_memory_type_id;
        size_t ba_byte_size;

        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONSERVER_BufferAttributesMemoryType(
                buffer_attributes, &ba_memory_type));

        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONSERVER_BufferAttributesMemoryTypeId(
                buffer_attributes, &ba_memory_type_id));

        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONSERVER_BufferAttributesByteSize(
                buffer_attributes, &ba_byte_size));

        if (!((actual_memory_type == ba_memory_type) &&
              (sizeof(int32_t) == ba_byte_size) &&
              (ba_memory_type_id == actual_memory_type_id)) ||
            responses[r] == nullptr) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r, request,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "State buffer attributes are not set correctly."));
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": State buffer attributes are not set correctly., error "
               "response sent")
                  .c_str());
          continue;
        }

        // Put the new state in the output buffer but intentionally do not
        // call the TRITONBACKEND_StateUpdate function.
        int32_t* lbuffer = reinterpret_cast<int32_t*>(buffer);
        *lbuffer = ipbuffer_int + ipbuffer_state_int;

        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_ResponseOutput(
                responses[r], &response_output, "OUTPUT",
                TRITONSERVER_TYPE_INT32, shape.data() /* data */,
                shape.size() /* dim_count */));

        if (responses[r] == nullptr) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to create the output state 'OUTPUT_STATE', error "
               "response sent")
                  .c_str());
          continue;
        }

        actual_memory_type = TRITONSERVER_MEMORY_CPU;
        actual_memory_type_id = 0;
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_OutputBuffer(
                response_output, reinterpret_cast<void**>(&buffer),
                sizeof(int32_t), &actual_memory_type, &actual_memory_type_id));

        if ((responses[r] == nullptr) ||
            (actual_memory_type == TRITONSERVER_MEMORY_GPU)) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r, request,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "failed to create the state buffer in CPU memory"));
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to create the state buffer in CPU memory, error "
               "response sent")
                  .c_str());
          continue;
        }
        lbuffer = reinterpret_cast<int32_t*>(buffer);
        *lbuffer = ipbuffer_int + ipbuffer_state_int;
      } break;
      // USE_SINGLE_BUFFER
      case 2: {
        TRITONBACKEND_State* response_state;
        std::vector<int64_t> shape{1};
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_StateNew(
                &response_state, request, "OUTPUT_STATE",
                TRITONSERVER_TYPE_INT32, shape.data() /* data */,
                shape.size() /* dim_count */));

        if (responses[r] == nullptr) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to create the output state 'OUTPUT_STATE', error "
               "response sent")
                  .c_str());
          continue;
        }
        TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t actual_memory_type_id = 0;
        char* buffer;

        // Request an output buffer in GPU. This is only for testing purposes
        // to make sure that GPU output buffers can be requested.
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_StateBuffer(
                response_state, reinterpret_cast<void**>(&buffer),
                sizeof(int32_t), &actual_memory_type, &actual_memory_type_id));

        instance_state->state_ = buffer;
      } break;
      case 3: {
        TRITONBACKEND_State* response_state;
        size_t block_size = sizeof(int8_t) * 1024 * 1024;
        int64_t current_elements =
            (instance_state->request_index_ + 1) * 1024 * 1024;
        std::cout << "current elements are "
                  << (instance_state->request_index_ + 1) << std::endl;
        std::vector<int64_t> shape{current_elements};
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_StateNew(
                &response_state, request, "OUTPUT_STATE",
                TRITONSERVER_TYPE_INT8, shape.data() /* data */,
                shape.size() /* dim_count */));

        if (responses[r] == nullptr) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to create the output state 'OUTPUT_STATE', error "
               "response sent")
                  .c_str());
          continue;
        }
        TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_GPU;
        int64_t actual_memory_type_id = 0;
        char* buffer;

        // Request an output buffer in GPU. This is only for testing purposes
        // to make sure that GPU output buffers can be requested.
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_StateBuffer(
                response_state, reinterpret_cast<void**>(&buffer),
                block_size * (instance_state->request_index_ + 1),
                &actual_memory_type, &actual_memory_type_id));

        // Only write the new data to the portion of the state buffer that
        // has been grown.
        cudaMemset(
            buffer + block_size * (instance_state->request_index_),
            instance_state->request_index_, block_size);

        TRITONBACKEND_Output* response_output;
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_ResponseOutput(
                responses[r], &response_output, "OUTPUT_STATE",
                TRITONSERVER_TYPE_INT8, shape.data() /* data */,
                shape.size() /* dim_count */));

        actual_memory_type = TRITONSERVER_MEMORY_CPU;
        actual_memory_type_id = 0;
        char* output_buffer;
        GUARDED_RESPOND_IF_ERROR(
            responses, r, request,
            TRITONBACKEND_OutputBuffer(
                response_output, reinterpret_cast<void**>(&output_buffer),
                block_size * (instance_state->request_index_ + 1),
                &actual_memory_type, &actual_memory_type_id));
        if ((responses[r] == nullptr) ||
            (actual_memory_type != TRITONSERVER_MEMORY_CPU)) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r, request,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "the backend can only handle CPU tensors"));
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               "the backend can only handle CPU tensors"
               "response sent")
                  .c_str());
          continue;
        }
        cudaMemcpy(
            output_buffer, buffer,
            block_size * (instance_state->request_index_ + 1),
            cudaMemcpyDeviceToHost);

        instance_state->state_ = buffer;
      } break;
    }
    const float* lend_buffer = reinterpret_cast<const float*>(end_buffer);

    if (*lend_buffer == 1) {
      instance_state->request_index_ = 0;
    } else {
      instance_state->request_index_ += 1;
    }

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);
    max_exec_end_ns = std::max(max_exec_end_ns, exec_end_ns);

    // Send all the responses that haven't already been sent because of
    // an earlier error.
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
            exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), total_batch_size,
          min_exec_start_ns, min_exec_start_ns, max_exec_end_ns,
          max_exec_end_ns),
      "failed reporting batch request statistics");

  return nullptr;  // success
}
}  // extern "C"
}}}  // namespace triton::backend::implicit
