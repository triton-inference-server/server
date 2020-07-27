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

#include "src/backends/backend/examples/backend_utils.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

namespace ni = nvidia::inferenceserver;
namespace nib = nvidia::inferenceserver::backend;

//
// Backend that demonstrates the TRITONBACKEND API for a decoupled
// backend where each request can generate 0 to many responses.
//
// This backend supports a model that has one input and one
// output. The model can support batching, with constraint that each
// request must be batch-1 request, but the shapes described
// here refer to the non-batch portion of the shape.
//
//   - Input 'IN' must have shape [1] and datatype INT32.
//
//   - Output 'OUT' must have shape [1] and datatype INT32.
//
// For a request, the backend will sent 'n' responses where 'n' is the
// element in IN. For each response, OUT will equal the element of IN.

namespace {

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
      return;                                                           \
    }                                                                   \
  } while (false)

#define RESPOND_FACTORY_AND_RETURN_IF_ERROR(FACTORY, X)                      \
  do {                                                                       \
    TRITONSERVER_Error* rfarie_err__ = (X);                                  \
    if (rfarie_err__ != nullptr) {                                           \
      TRITONBACKEND_Response* rfarie_response__ = nullptr;                   \
      LOG_IF_ERROR(                                                          \
          TRITONBACKEND_ResponseNewFromFactory(&rfarie_response__, FACTORY), \
          "failed to create response");                                      \
      if (rfarie_response__ != nullptr) {                                    \
        LOG_IF_ERROR(                                                        \
            TRITONBACKEND_ResponseSend(                                      \
                rfarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL,     \
                rfarie_err__),                                               \
            "failed to send error response");                                \
      }                                                                      \
      TRITONSERVER_ErrorDelete(rfarie_err__);                                \
      return;                                                                \
    }                                                                        \
  } while (false)


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

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Validate that model configuration is supported by this backend.
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
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

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
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  ni::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // There must be 3 inputs and 1 output.
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 1 input, got ") +
          std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 1 output, got ") +
          std::to_string(outputs.ArraySize()));

  ni::TritonJson::Value in, out;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &in));
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &out));

  // Check tensor names
  std::string in_name, out_name;
  RETURN_IF_ERROR(in.MemberAsString("name", &in_name));
  RETURN_IF_ERROR(out.MemberAsString("name", &out_name));

  RETURN_ERROR_IF_FALSE(
      in_name == "IN", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected first input tensor name to be IN, got ") + in_name);
  RETURN_ERROR_IF_FALSE(
      out_name == "OUT", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected first output tensor name to be OUT, got ") +
          out_name);

  // Check shapes
  std::vector<int64_t> in_shape, out_shape;
  RETURN_IF_ERROR(nib::ParseShape(in, "dims", &in_shape));
  RETURN_IF_ERROR(nib::ParseShape(out, "dims", &out_shape));

  RETURN_ERROR_IF_FALSE(
      (in_shape.size() == 1) && (in_shape[0] == 1),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IN shape to be [1], got ") +
          nib::ShapeToString(in_shape));
  RETURN_ERROR_IF_FALSE(
      (out_shape.size() == 1) && (out_shape[0] == 1),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected OUT shape to be [1], got ") +
          nib::ShapeToString(out_shape));

  // Check datatypes
  std::string in_dtype, out_dtype;
  RETURN_IF_ERROR(in.MemberAsString("data_type", &in_dtype));
  RETURN_IF_ERROR(out.MemberAsString("data_type", &out_dtype));

  RETURN_ERROR_IF_FALSE(
      in_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IN datatype to be INT32, got ") + in_dtype);
  RETURN_ERROR_IF_FALSE(
      out_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected OUT datatype to be INT32, got ") + out_dtype);

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  ~ModelInstanceState();

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance* TritonModelInstance()
  {
    return triton_model_instance_;
  }

  // Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Spawn a thread to produce outputs for a request. Return the
  // request wait time before it should release.
  void ProcessRequest(TRITONBACKEND_Request* request);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance, const char* name,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id);
  void RequestThread(
      TRITONBACKEND_ResponseFactory* factory_ptr, const size_t element_count,
      uint32_t dims_count);

  ModelState* model_state_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;

  std::atomic<size_t> inflight_thread_count_;
};

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

  *state = new ModelInstanceState(
      model_state, triton_model_instance, instance_name, instance_kind,
      instance_id);
  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : model_state_(model_state), triton_model_instance_(triton_model_instance),
      name_(name), kind_(kind), device_id_(device_id), inflight_thread_count_(0)
{
}

ModelInstanceState::~ModelInstanceState()
{
  // Wait for all threads that have been launched by this instance to
  // exit...
  while (inflight_thread_count_ > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void
ModelInstanceState::ProcessRequest(TRITONBACKEND_Request* request)
{
  // Make sure the request is OK and if not just send a single error
  // response.
  TRITONBACKEND_Input* in;
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_RequestInput(request, "IN", &in));

  const int64_t* in_shape_arr;
  uint32_t in_dims_count;
  uint64_t in_byte_size;
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_InputProperties(
                   in, nullptr, nullptr, &in_shape_arr, &in_dims_count,
                   &in_byte_size, nullptr));
  std::vector<int64_t> in_shape(in_shape_arr, in_shape_arr + in_dims_count);

  // We need a copy of the inputs because we can release the request
  // before we are done with the inputs, and once the request is
  // released we are no longer allowed to access the input tensor
  // buffers directly.
  std::vector<int32_t> in_buffer(in_byte_size / sizeof(int32_t));
  RESPOND_AND_RETURN_IF_ERROR(
      request, nib::ReadInputTensor(
                   request, "IN", reinterpret_cast<char*>(in_buffer.data()),
                   &in_byte_size));

  if ((in_dims_count == 2) && (in_shape[0] != 1)) {
    RESPOND_AND_RETURN_IF_ERROR(
        request,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG, "unexpected shape for IN input"));
  }

  const size_t element_count = in_buffer[0];

  // 'request' may be released before all the responses are sent, so
  // create a response factory that will live until the RequestThread
  // exits.
  TRITONBACKEND_ResponseFactory* factory_ptr;
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_ResponseFactoryNew(&factory_ptr, request));

  // Start a detached thread to generate the responses. If a model is
  // being destroyed (because it is unloaded and there are no
  // in-flight requests) then that destruction must wait for all
  // threads to complete. We do this by maintaining an atomic counter
  // that tracks how many threads are running.
  inflight_thread_count_++;
  std::thread response_thread(
      [this, factory_ptr, element_count, in_dims_count]() {
        RequestThread(factory_ptr, element_count, in_dims_count);
      });

  response_thread.detach();
}

void
ModelInstanceState::RequestThread(
    TRITONBACKEND_ResponseFactory* factory_ptr, const size_t element_count,
    uint32_t dims_count)
{
  std::unique_ptr<TRITONBACKEND_ResponseFactory, nib::ResponseFactoryDeleter>
      factory(factory_ptr);

  // Copy IN->OUT, and send a response.
  const std::vector<int64_t> output_shape(dims_count, 1);
  for (size_t e = 0; e < element_count; ++e) {
    // Create the response with a single OUT output.
    TRITONBACKEND_Response* response;
    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        factory.get(),
        TRITONBACKEND_ResponseNewFromFactory(&response, factory.get()));

    TRITONBACKEND_Output* output;
    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        factory.get(), TRITONBACKEND_ResponseOutput(
                           response, &output, "OUT", TRITONSERVER_TYPE_INT32,
                           output_shape.data(), dims_count));

    // Get the output buffer. We request a buffer in CPU memory but we
    // have to handle any returned type. If we get back a buffer in
    // GPU memory we just fail the request.
    void* output_buffer;
    TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t output_memory_type_id = 0;
    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        factory.get(), TRITONBACKEND_OutputBuffer(
                           output, &output_buffer, sizeof(int32_t),
                           &output_memory_type, &output_memory_type_id));
    if (output_memory_type == TRITONSERVER_MEMORY_GPU) {
      RESPOND_FACTORY_AND_RETURN_IF_ERROR(
          factory.get(), TRITONSERVER_ErrorNew(
                             TRITONSERVER_ERROR_INTERNAL,
                             "failed to create output buffer in CPU memory"));
    }

    // Copy IN -> OUT
    *(reinterpret_cast<int32_t*>(output_buffer)) = element_count;

    // Send the response.
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            response, 0 /* flags */, nullptr /* success */),
        "failed sending response");

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("sent response ") + std::to_string(e + 1) + " of " +
         std::to_string(element_count))
            .c_str());
  }

  // Add some logging for the case where IN was size 0 and so no
  // responses were sent.
  if (element_count == 0) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "IN size is zero, no responses send ");
  }

  // All responses have been sent so we must signal that we are done
  // sending responses for the request. We could have been smarter
  // above and included the FINAL flag on the ResponseSend in the last
  // iteration of the loop... but instead we demonstrate how to use
  // the factory to send just response flags without a corresponding
  // response.
  LOG_IF_ERROR(
      TRITONBACKEND_ResponseFactorySendFlags(
          factory.get(), TRITONSERVER_RESPONSE_COMPLETE_FINAL),
      "failed sending final response");

  inflight_thread_count_--;
}

}  // namespace

/////////////

extern "C" {

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

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
       " (device " + std::to_string(device_id) + ")")
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

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until this instance has
  // completed processing 'requests'. On return from this function,
  // Triton will automatically show 'instance' as available to execute
  // a new batch of requests.

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model instance ") + instance_state->Name() +
       ", executing " + std::to_string(request_count) + " requests")
          .c_str());


  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.

  // For simplicity we process each request in a separate thread. If a
  // model has multiple instances and uses the dynamic batcher to
  // create large batches, then this could result in 1000's of
  // threads. A more performant implementation would likely need to
  // use a different technique than one-thread-per-request.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    instance_state->ProcessRequest(request);
  }

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  return nullptr;  // success
}

}  // extern "C"
