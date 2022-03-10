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

#include <memory>
#include <thread>
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace sequence {


// Simple sequence backend that demonstrates the TRITONBACKEND API for a
// blocking backend. A blocking backend completes execution of the inference
// before returning from TRITONBACKEND_ModelInstanceExecute.
//
// The backend supports models that take three input tensors, two INT32 [ 1 ]
// control values and one variable-size INT32 [ -1 ] value input; and produces
// an output tensor with the same shape as the input tensor. The input tensors
// must be named "START", "READY" and "INPUT". The output tensor must be named
// "OUTPUT".
//
// The model maintains an INT32 accumulator which is updated based on the
// control values in "START" and "READY":
//
//   READY=0, START=x: Ignore value input, do not change accumulator value.
//
//   READY=1, START=1: Start accumulating. Set accumulator equal to sum of input
//   tensor.
//
//   READY=1, START=0: Add input tensor values to accumulator.
//
// When READY=1, the accumulator is returned in every element of the output.
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
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Get accumulator size and execution delay
  size_t AccumulatorSize() const { return accumulator_size_; }
  int ExecDelay() const { return execute_delay_ms_; }

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  // Delay to introduce into execution, in milliseconds.
  int execute_delay_ms_;

  // Accumulator size
  size_t accumulator_size_;
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
    : BackendModel(triton_model), execute_delay_ms_(0)
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

  triton::common::TritonJson::Value params;
  if (model_config_.Find("parameters", &params)) {
    common::TritonJson::Value exec_delay;
    if (params.Find("execute_delay_ms", &exec_delay)) {
      std::string exec_delay_str;
      RETURN_IF_ERROR(
          exec_delay.MemberAsString("string_value", &exec_delay_str));
      execute_delay_ms_ = std::stoi(exec_delay_str);
    }
  }

  int64_t max_batch_size = 0;
  RETURN_IF_ERROR(model_config_.MemberAsInt("max_batch_size", &max_batch_size));
  accumulator_size_ = (size_t)(std::max((int64_t)1, max_batch_size));

  // The model configuration must specify the sequence batcher and must use the
  // START and READY input to indicate control values.
  triton::common::TritonJson::Value sequence_batching;
  RETURN_IF_ERROR(
      model_config_.MemberAsObject("sequence_batching", &sequence_batching));
  common::TritonJson::Value control_inputs;
  RETURN_IF_ERROR(
      sequence_batching.MemberAsArray("control_input", &control_inputs));
  RETURN_ERROR_IF_FALSE(
      control_inputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "'START' and 'READY' must be configured as the control inputs"));

  std::vector<std::string> control_input_names;
  for (size_t io_index = 0; io_index < control_inputs.ArraySize(); io_index++) {
    common::TritonJson::Value control_input;
    RETURN_IF_ERROR(control_inputs.IndexAsObject(io_index, &control_input));
    const char* input_name;
    size_t input_name_len;
    RETURN_IF_ERROR(
        control_input.MemberAsString("name", &input_name, &input_name_len));
    control_input_names.push_back(input_name);
  }

  RETURN_ERROR_IF_FALSE(
      ((control_input_names[0] == "START") &&
       (control_input_names[1] == "READY")) ||
          ((control_input_names[0] == "READY") &&
           (control_input_names[1] == "START")),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "'START' and 'READY' must be configured as the control inputs"));

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // There must be one INT32 input called INPUT defined in the model
  // configuration and it must be a 1D vector (of any length).
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "model must have input 'INPUT' with vector shape, any length"));

  common::TritonJson::Value input;
  RETURN_IF_ERROR(inputs.IndexAsObject(0 /* index */, &input));

  std::vector<int64_t> input_shape;
  RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));

  RETURN_ERROR_IF_FALSE(
      input_shape.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "model must have one input 'INPUT' with vector shape, any length"));

  std::string input_dtype;
  RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));

  RETURN_ERROR_IF_FALSE(
      input_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model input must have TYPE_INT32 data-type"));

  const char* input_name;
  size_t input_name_len;
  RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));

  RETURN_ERROR_IF_FALSE(
      strcmp(input_name, "INPUT") == 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model input must be named 'INPUT'"));

  // There must be one INT32 output with shape that matches the input. The
  // output must be named OUTPUT.
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "model must have one output 'OUTPUT' with vector shape, any length"));

  common::TritonJson::Value output;
  RETURN_IF_ERROR(outputs.IndexAsObject(0 /* index */, &output));

  std::vector<int64_t> output_shape;
  RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

  RETURN_ERROR_IF_FALSE(
      (output_shape.size() == 1) && (output_shape[0] == input_shape[0]),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "model must have output 'OUTPUT' with shape matching 'INPUT'"));

  std::string output_dtype;
  RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));

  RETURN_ERROR_IF_FALSE(
      output_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model output must have TYPE_INT32 data-type"));

  const char* output_name;
  size_t output_name_len;
  RETURN_IF_ERROR(
      output.MemberAsString("name", &output_name, &output_name_len));

  RETURN_ERROR_IF_FALSE(
      strcmp(output_name, "OUTPUT") == 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model output must be named 'OUTPUT'"));

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
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Get accumulator for this instance
  int32_t GetAccumulatorAt(size_t idx);
  void SetAccumulatorAt(size_t idx, int32_t value);
  void AddAccumulatorAt(size_t idx, int32_t value);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  ModelState* model_state_;

  // Accumulators maintained by this instance, one for each batch slot.
  std::vector<int32_t> accumulator_;
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
  accumulator_.resize(model_state->AccumulatorSize());
}

int32_t
ModelInstanceState::GetAccumulatorAt(size_t idx)
{
  return accumulator_[idx];
}

void
ModelInstanceState::SetAccumulatorAt(size_t idx, int32_t value)
{
  accumulator_[idx] = value;
}

void
ModelInstanceState::AddAccumulatorAt(size_t idx, int32_t value)
{
  accumulator_[idx] += value;
}

ModelInstanceState::~ModelInstanceState()
{
  accumulator_.clear();
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

  // Because this backend just copies IN -> OUT and requires that
  // input and output be in CPU memory, we fail if a GPU instances is
  // requested.
  RETURN_ERROR_IF_FALSE(
      instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("'sequence' backend only supports CPU instances"));

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
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  bool supports_batching = false;
  RETURN_IF_ERROR(model_state->SupportsFirstDimBatching(&supports_batching));

  // Each request represents a different sequence, which corresponds
  // to the accumulator at the same index. Each request must have
  // batch-size 1 inputs which is the next timestep for that sequence. The total
  // number of requests will not exceed the max-batch-size specified in the
  // model configuration.
  if (request_count > model_state->AccumulatorSize()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "unable to execute batch larger than max-batch-size");
  }

  // Delay if requested...
  if (model_state->ExecDelay() > 0) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(model_state->ExecDelay()));
  }

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
  uint64_t min_exec_start_ns = std::numeric_limits<uint64_t>::max();
  uint64_t max_exec_end_ns = 0;
  uint64_t total_batch_size = 0;

  // For simplicity we just process each request separately... in
  // general a backend should try to operate on the entire batch of
  // requests at the same time for improved performance.
  std::vector<uint8_t> start_buffer, ready_buffer, input_buffer;
  for (uint32_t r = 0; r < request_count; ++r) {
    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);
    min_exec_start_ns = std::min(min_exec_start_ns, exec_start_ns);

    TRITONBACKEND_Request* request = requests[r];

    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestId(request, &request_id));

    uint64_t correlation_id = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

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

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("request ") + std::to_string(r) + ": id = \"" +
         request_id + "\", correlation_id = " + std::to_string(correlation_id) +
         ", input_count = " + std::to_string(input_count) +
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
          responses, r,
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
          responses, r,
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
              responses, r,
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
        TRITONSERVER_LOG_ERROR,
        (std::string("total_batch_size: ") + std::to_string(total_batch_size))
            .c_str());

    // Get the input tensors.
    TRITONBACKEND_Input* start_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInput(request, "START", &start_input));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input 'START', error response sent")
              .c_str());
      continue;
    }

    TRITONBACKEND_Input* ready_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
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
        responses, r,
        TRITONBACKEND_InputBuffer(
            start_input, 0 /* input_buffer_count */, &start_buffer,
            &buffer_byte_size, &input_memory_type, &input_memory_type_id));
    if (responses[r] == nullptr) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
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
        responses, r,
        TRITONBACKEND_InputBuffer(
            ready_input, 0 /* input_buffer_count */, &ready_buffer,
            &buffer_byte_size, &input_memory_type, &input_memory_type_id));
    if (responses[r] == nullptr) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
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
        responses, r, TRITONBACKEND_RequestInput(request, "INPUT", &input));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input, error response sent")
              .c_str());
      continue;
    }

    const void* input_buffer = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputBuffer(
            input, 0 /* input_buffer_count */, &input_buffer, &buffer_byte_size,
            &input_memory_type, &input_memory_type_id));
    if (responses[r] == nullptr) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
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

    int64_t input_element_cnt = input_byte_size / sizeof(int32_t);
    const int32_t* start = reinterpret_cast<const int32_t*>(start_buffer);
    const int32_t* ready = reinterpret_cast<const int32_t*>(ready_buffer);
    const int32_t* ipbuffer_int = nullptr;
    std::vector<int32_t> ipbuffer_vec;

    if (input_memory_type == TRITONSERVER_MEMORY_GPU) {
      ipbuffer_vec.resize(input_element_cnt);
      ipbuffer_int = ipbuffer_vec.data();
      cudaMemcpy(
          const_cast<int32_t*>(ipbuffer_int), input_buffer, input_byte_size,
          cudaMemcpyDeviceToHost);
    } else {
      ipbuffer_int = reinterpret_cast<const int32_t*>(input_buffer);
    }

    // Update the accumulator value based on START/READY and calculate the
    // output value.
    if (ready[0] != 0) {
      if (start[0] == 0) {
        // Update accumulator.
        for (int64_t e = 0; e < input_element_cnt; ++e) {
          instance_state->AddAccumulatorAt(r, ipbuffer_int[e]);
        }
      } else {
        // Set accumulator.
        instance_state->SetAccumulatorAt(r, ipbuffer_int[0]);
        for (int64_t e = 1; e < input_element_cnt; ++e) {
          instance_state->AddAccumulatorAt(r, ipbuffer_int[e]);
        }
      }

      TRITONBACKEND_Response* response = responses[r];

      // If the output is requested, copy the calculated output value
      // into the output buffer.
      if (requested_output_count > 0) {
        // The output shape is [1, input_element_cnt] if the model configuration
        // supports batching, or just [input_element_cnt] if the model
        // configuration does not support batching.
        std::vector<int64_t> shape;
        if (supports_batching) {
          shape.push_back(1);
        }
        shape.push_back(input_element_cnt);

        TRITONBACKEND_Output* output;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_ResponseOutput(
                response, &output, "OUTPUT", input_datatype, input_shape,
                input_dims_count));
        if (responses[r] == nullptr) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to create response output, error response sent")
                  .c_str());
          continue;
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
                output, &output_buffer, buffer_byte_size, &output_memory_type,
                &output_memory_type_id));
        if (responses[r] == nullptr) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "failed to create output buffer in CPU memory"));
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("request ") + std::to_string(r) +
               ": failed to create output buffer in CPU memory, error "
               "response sent")
                  .c_str());
          continue;
        }

        int32_t* obuffer_int = nullptr;
        std::vector<int32_t> obuffer_vec;
        if (output_memory_type == TRITONSERVER_MEMORY_GPU) {
          obuffer_vec.resize(buffer_byte_size / sizeof(int32_t));
          obuffer_int = obuffer_vec.data();
        } else {
          obuffer_int = reinterpret_cast<int32_t*>(output_buffer);
        }

        for (int64_t i = 0; i < input_element_cnt; ++i) {
          obuffer_int[i] = instance_state->GetAccumulatorAt(r);
        }

        if (output_memory_type == TRITONSERVER_MEMORY_GPU) {
          cudaMemcpy(
              output_buffer, const_cast<int32_t*>(obuffer_int),
              buffer_byte_size, cudaMemcpyHostToDevice);
        }
      }
    }

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);
    max_exec_end_ns = std::max(max_exec_end_ns, exec_end_ns);

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

}}}  // namespace triton::backend::sequence
