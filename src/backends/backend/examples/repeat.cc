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

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include "triton/backend/backend_common.h"

//
// Backend that demonstrates the TRITONBACKEND API for a decoupled
// backend where each request can generate 0 to many responses.
//
// This backend supports a model that has three inputs and two
// outputs. The backend does not support batching.
//
//   - Input 'IN' can have any vector shape (e.g. [4] or [-1]) and
//     datatype must be INT32.
//
//   - Input 'DELAY' must have same shape as IN and datatype must be
//     UINT32.
//
//   - Input 'WAIT' must have shape [1] and datatype UINT32.
//
//   - For each response, output 'OUT' must have shape [1] and
//     datatype INT32.
//
//   - For each response, output 'IDX' must have shape [1] and
//     datatype UINT32.
//
// For a request, the backend will sent 'n' responses where 'n' is the
// number of elements in IN. For the i'th response, OUT will equal the
// i'th element of IN and IDX will equal the zero-based count of this
// response for the request. For example, the first response for a
// request will have IDX 0, the second will have IDX 1, etc.  The
// backend will wait the i'th DELAY, in milliseconds, before sending
// the i'th response. If IN shape is [0] then no responses will be
// sent.
//
// After WAIT milliseconds the backend will release the request and
// return from the TRITONBACKEND_ModelInstanceExecute function so that
// Triton can provide another request to the backend. WAIT can be less
// than the sum of DELAY so that the request is released before all
// responses are sent. Thus, even if there is only one instance of the
// model, the backend can be processing multiple requests at the same
// time, and the responses for multiple requests can be intermixed,
// depending on the values of DELAY and WAIT.

namespace triton { namespace backend { namespace repeat {

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
      TRITONBACKEND_Model* triton_model,
      common::TritonJson::Value&& model_config);

  TRITONBACKEND_Model* triton_model_;
  common::TritonJson::Value model_config_;
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

  common::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  *state = new ModelState(triton_model, std::move(model_config));
  return nullptr;  // success
}

ModelState::ModelState(
    TRITONBACKEND_Model* triton_model, common::TritonJson::Value&& model_config)
    : triton_model_(triton_model), model_config_(std::move(model_config))
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

  // max_batch_size must be 0 because this backend does not support
  // batching
  int64_t max_batch_size;
  RETURN_IF_ERROR(model_config_.MemberAsInt("max_batch_size", &max_batch_size));
  RETURN_ERROR_IF_FALSE(
      max_batch_size == 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "repeat backend only supports models with max_batch_size == 0"));

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // There must be 3 inputs and 2 outputs.
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 3, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 3 inputs, got ") +
          std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 2 outputs, got ") +
          std::to_string(outputs.ArraySize()));

  // Here we rely on the model configuation listing the inputs and
  // outputs in a specific order, which we shouldn't really require...
  common::TritonJson::Value in, delay, wait, out, idx;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &in));
  RETURN_IF_ERROR(inputs.IndexAsObject(1, &delay));
  RETURN_IF_ERROR(inputs.IndexAsObject(2, &wait));
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &out));
  RETURN_IF_ERROR(outputs.IndexAsObject(1, &idx));

  // Check tensor names
  std::string in_name, delay_name, wait_name, out_name, idx_name;
  RETURN_IF_ERROR(in.MemberAsString("name", &in_name));
  RETURN_IF_ERROR(delay.MemberAsString("name", &delay_name));
  RETURN_IF_ERROR(wait.MemberAsString("name", &wait_name));
  RETURN_IF_ERROR(out.MemberAsString("name", &out_name));
  RETURN_IF_ERROR(idx.MemberAsString("name", &idx_name));

  RETURN_ERROR_IF_FALSE(
      in_name == "IN", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected first input tensor name to be IN, got ") + in_name);
  RETURN_ERROR_IF_FALSE(
      delay_name == "DELAY", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected second input tensor name to be DELAY, got ") +
          delay_name);
  RETURN_ERROR_IF_FALSE(
      wait_name == "WAIT", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected third input tensor name to be WAIT, got ") +
          wait_name);
  RETURN_ERROR_IF_FALSE(
      out_name == "OUT", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected first output tensor name to be OUT, got ") +
          out_name);
  RETURN_ERROR_IF_FALSE(
      idx_name == "IDX", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected second output tensor name to be IDX, got ") +
          idx_name);

  // Check shapes
  std::vector<int64_t> in_shape, delay_shape, wait_shape, out_shape, idx_shape;
  RETURN_IF_ERROR(backend::ParseShape(in, "dims", &in_shape));
  RETURN_IF_ERROR(backend::ParseShape(delay, "dims", &delay_shape));
  RETURN_IF_ERROR(backend::ParseShape(wait, "dims", &wait_shape));
  RETURN_IF_ERROR(backend::ParseShape(out, "dims", &out_shape));
  RETURN_IF_ERROR(backend::ParseShape(idx, "dims", &idx_shape));

  RETURN_ERROR_IF_FALSE(
      in_shape.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IN shape to have one dimension, got ") +
          backend::ShapeToString(in_shape));
  RETURN_ERROR_IF_FALSE(
      in_shape == delay_shape, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IN and DELAY shape to match, got ") +
          backend::ShapeToString(in_shape) + " and " +
          backend::ShapeToString(delay_shape));
  RETURN_ERROR_IF_FALSE(
      (wait_shape.size() == 1) && (wait_shape[0] == 1),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected WAIT shape to be [1], got ") +
          backend::ShapeToString(wait_shape));
  RETURN_ERROR_IF_FALSE(
      (out_shape.size() == 1) && (out_shape[0] == 1),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected OUT shape to be [1], got ") +
          backend::ShapeToString(out_shape));
  RETURN_ERROR_IF_FALSE(
      (idx_shape.size() == 1) && (idx_shape[0] == 1),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IDX shape to be [1], got ") +
          backend::ShapeToString(idx_shape));

  // Check datatypes
  std::string in_dtype, delay_dtype, wait_dtype, out_dtype, idx_dtype;
  RETURN_IF_ERROR(in.MemberAsString("data_type", &in_dtype));
  RETURN_IF_ERROR(delay.MemberAsString("data_type", &delay_dtype));
  RETURN_IF_ERROR(wait.MemberAsString("data_type", &wait_dtype));
  RETURN_IF_ERROR(out.MemberAsString("data_type", &out_dtype));
  RETURN_IF_ERROR(idx.MemberAsString("data_type", &idx_dtype));

  RETURN_ERROR_IF_FALSE(
      in_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IN datatype to be INT32, got ") + in_dtype);
  RETURN_ERROR_IF_FALSE(
      delay_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected DELAY datatype to be UINT32, got ") + delay_dtype);
  RETURN_ERROR_IF_FALSE(
      wait_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected WAIT datatype to be UINT32, got ") + wait_dtype);
  RETURN_ERROR_IF_FALSE(
      out_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected OUT datatype to be INT32, got ") + out_dtype);
  RETURN_ERROR_IF_FALSE(
      idx_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IDX datatype to be UINT32, got ") + idx_dtype);

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
  void ProcessRequest(TRITONBACKEND_Request* request, uint32_t* wait_ms);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance, const char* name,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id);
  void ResponseThread(
      TRITONBACKEND_ResponseFactory* factory_ptr, const int32_t* in_buffer_ptr,
      const int32_t* delay_buffer_ptr, const uint32_t element_count);

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
ModelInstanceState::ProcessRequest(
    TRITONBACKEND_Request* request, uint32_t* wait_ms)
{
  // Get the wait time for the request release.
  *wait_ms = 0;

  size_t wait_byte_size = sizeof(uint32_t);
  RESPOND_AND_RETURN_IF_ERROR(
      request,
      backend::ReadInputTensor(
          request, "WAIT", reinterpret_cast<char*>(wait_ms), &wait_byte_size));

  if (wait_byte_size != sizeof(uint32_t)) {
    RESPOND_AND_RETURN_IF_ERROR(
        request,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG, "unexpected size for WAIT input"));
  }

  // Make sure the request is OK and if not just send a single error
  // response. Make sure shape of IN and DELAY are the same. We
  // checked the model in ValidateModelConfig but the shape for IN and
  // DELAY could be [-1] in the model configuration and so we need to
  // check that for each request they are equal.
  TRITONBACKEND_Input* in;
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_RequestInput(request, "IN", &in));
  TRITONBACKEND_Input* delay;
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_RequestInput(request, "DELAY", &delay));

  const int64_t* in_shape_arr;
  uint32_t in_dims_count;
  uint64_t in_byte_size;
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_InputProperties(
                   in, nullptr, nullptr, &in_shape_arr, &in_dims_count,
                   &in_byte_size, nullptr));
  std::vector<int64_t> in_shape(in_shape_arr, in_shape_arr + in_dims_count);

  const int64_t* delay_shape_arr;
  uint32_t delay_dims_count;
  uint64_t delay_byte_size;
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_InputProperties(
                   delay, nullptr, nullptr, &delay_shape_arr, &delay_dims_count,
                   &delay_byte_size, nullptr));
  std::vector<int64_t> delay_shape(
      delay_shape_arr, delay_shape_arr + delay_dims_count);

  if (in_shape != delay_shape) {
    RESPOND_AND_RETURN_IF_ERROR(
        request,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("expected IN and DELAY shape to match, got ") +
             backend::ShapeToString(in_shape) + " and " +
             backend::ShapeToString(delay_shape))
                .c_str()));
  }

  const uint32_t element_count = in_byte_size / sizeof(int32_t);

  // We need a copy of the inputs because we can release the request
  // before we are done with the inputs, and once the request is
  // released we are no longer allowed to access the input tensor
  // buffers directly.
  std::unique_ptr<int32_t> in_buffer(new int32_t[element_count]);
  RESPOND_AND_RETURN_IF_ERROR(
      request, backend::ReadInputTensor(
                   request, "IN", reinterpret_cast<char*>(in_buffer.get()),
                   &in_byte_size));

  std::unique_ptr<int32_t> delay_buffer(new int32_t[element_count]);
  RESPOND_AND_RETURN_IF_ERROR(
      request,
      backend::ReadInputTensor(
          request, "DELAY", reinterpret_cast<char*>(delay_buffer.get()),
          &delay_byte_size));

  // 'request' may be released before all the responses are sent, so
  // create a response factory that will live until the ResponseThread
  // exits.
  TRITONBACKEND_ResponseFactory* factory_ptr;
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_ResponseFactoryNew(&factory_ptr, request));

  int32_t* in_buffer_ptr = in_buffer.release();
  int32_t* delay_buffer_ptr = delay_buffer.release();

  // Start a detached thread to generate the responses. If a model is
  // being destroyed (because it is unloaded and there are no
  // in-flight requests) then that destruction must wait for all
  // threads to complete. We do this by maintaining an atomic counter
  // that tracks how many threads are running.
  inflight_thread_count_++;
  std::thread response_thread([this, factory_ptr, in_buffer_ptr,
                               delay_buffer_ptr, element_count]() {
    ResponseThread(factory_ptr, in_buffer_ptr, delay_buffer_ptr, element_count);
  });

  response_thread.detach();
}

void
ModelInstanceState::ResponseThread(
    TRITONBACKEND_ResponseFactory* factory_ptr, const int32_t* in_buffer_ptr,
    const int32_t* delay_buffer_ptr, const uint32_t element_count)
{
  std::unique_ptr<
      TRITONBACKEND_ResponseFactory, backend::ResponseFactoryDeleter>
      factory(factory_ptr);
  std::unique_ptr<const int32_t> in_buffer(in_buffer_ptr);
  std::unique_ptr<const int32_t> delay_buffer(delay_buffer_ptr);

  // IN and DELAY are INT32 vectors... wait, copy IN->OUT, and send a
  // response.
  for (uint32_t e = 0; e < element_count; ++e) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("waiting ") + std::to_string(delay_buffer.get()[e]) +
         " ms, then sending response " + std::to_string(e + 1) + " of " +
         std::to_string(element_count))
            .c_str());

    std::this_thread::sleep_for(
        std::chrono::milliseconds(delay_buffer.get()[e]));

    // Create the response with OUT and IDX outputs.
    TRITONBACKEND_Response* response;
    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        factory.get(),
        TRITONBACKEND_ResponseNewFromFactory(&response, factory.get()));

    const int64_t out_shape = 1;
    TRITONBACKEND_Output* out;
    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        factory.get(), TRITONBACKEND_ResponseOutput(
                           response, &out, "OUT", TRITONSERVER_TYPE_INT32,
                           &out_shape, 1 /* dims_count */));

    const int64_t idx_shape = 1;
    TRITONBACKEND_Output* idx;
    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        factory.get(), TRITONBACKEND_ResponseOutput(
                           response, &idx, "IDX", TRITONSERVER_TYPE_UINT32,
                           &idx_shape, 1 /* dims_count */));

    // Get the OUT output buffer. We request a buffer in CPU memory
    // but we have to handle any returned type. If we get back a
    // buffer in GPU memory we just fail the request.
    void* out_buffer;
    TRITONSERVER_MemoryType out_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t out_memory_type_id = 0;
    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        factory.get(), TRITONBACKEND_OutputBuffer(
                           out, &out_buffer, sizeof(int32_t), &out_memory_type,
                           &out_memory_type_id));
    if (out_memory_type == TRITONSERVER_MEMORY_GPU) {
      RESPOND_FACTORY_AND_RETURN_IF_ERROR(
          factory.get(),
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "failed to create OUT output buffer in CPU memory"));
    }

    // Get the IDX output buffer. We request a buffer in CPU memory
    // but we have to handle any returned type. If we get back a
    // buffer in GPU memory we just fail the request.
    void* idx_buffer;
    TRITONSERVER_MemoryType idx_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t idx_memory_type_id = 0;
    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        factory.get(), TRITONBACKEND_OutputBuffer(
                           idx, &idx_buffer, sizeof(uint32_t), &idx_memory_type,
                           &idx_memory_type_id));
    if (idx_memory_type == TRITONSERVER_MEMORY_GPU) {
      RESPOND_FACTORY_AND_RETURN_IF_ERROR(
          factory.get(),
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "failed to create IDX output buffer in CPU memory"));
    }

    // Copy IN -> OUT
    *(reinterpret_cast<int32_t*>(out_buffer)) = in_buffer.get()[e];

    // Set response IDX
    *(reinterpret_cast<uint32_t*>(idx_buffer)) = e;

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
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "IN size is zero, no responses sent");
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
  // completed prcoessing 'requests'. On return from this function,
  // Triton will automatically show 'instance' as available to execute
  // a new batch of requests.

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model instance ") + instance_state->Name() +
       ", executing " + std::to_string(request_count) + " requests")
          .c_str());

  // This backend does not support models that support batching, so
  // 'request_count' should always be 1.
  RETURN_ERROR_IF_FALSE(
      request_count <= 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("repeat backend does not support batched request execution"));

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);
  uint32_t wait_milliseconds = 0;

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.

  // For simplicity we process each request in a separate thread.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    instance_state->ProcessRequest(request, &wait_milliseconds);
  }

  // Wait, release, return...
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("waiting ") + std::to_string(wait_milliseconds) +
       " ms before releasing requests")
          .c_str());

  std::this_thread::sleep_for(std::chrono::milliseconds(wait_milliseconds));

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
            instance_state->TritonModelInstance(), request, true /* success */,
            exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics. This backend does not support
  // batching so the total batch size is always 1.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), 1 /*total_batch_size*/,
          exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::repeat
