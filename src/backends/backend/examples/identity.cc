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

#include <memory>
#include <thread>
#include "src/backends/backend/examples/backend_utils.h"

namespace ni = nvidia::inferenceserver;
namespace nib = nvidia::inferenceserver::backend;

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

//
// ModelInstance
//
// A model instance.
//
class ModelInstance {
 public:
  explicit ModelInstance(ModelState* ms, const nib::InstanceProperties p);
  ~ModelInstance();

  // Execute a batch of requests on this instance.
  TRITONSERVER_Error* Execute(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  struct ExecRequests {
    TRITONBACKEND_Request** requests_;
    uint32_t request_count_;
    std::vector<TRITONBACKEND_Response*> responses_;
  };

  void ExecuteThread();

  ModelState* model_state_;
  const nib::InstanceProperties props_;

  nib::BlockingQueue<std::unique_ptr<ExecRequests>> exec_queue_;
  std::unique_ptr<std::thread> thread_;
};

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

  // Get the name and version of the model.
  const std::string& ModelName() const { return model_name_; }
  uint64_t ModelVersion() const { return model_version_; }

  // Does this model support batching in the first dimension.
  TRITONSERVER_Error* SupportsFirstDimBatching(bool* supports);

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Determine the instances required for this model.
  TRITONSERVER_Error* CreateModelInstances();

  // Execute on a ready model instance.
  TRITONSERVER_Error* Execute(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  // Show that instance with index 'idx' is now ready to execute.
  void SetInstanceReady(const size_t idx);

  // Wait for a ready model instance.
  void WaitForReadyInstance();

 private:
  ModelState(
      TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
      const char* model_name, const uint64_t model_version,
      ni::TritonJson::Value&& model_config);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string model_name_;
  const uint64_t model_version_;
  ni::TritonJson::Value model_config_;

  std::vector<std::unique_ptr<ModelInstance>> instances_;
  nib::BlockingQueue<size_t> ready_instance_idx_;
};

//
// ModelInstance
//
ModelInstance::ModelInstance(ModelState* ms, const nib::InstanceProperties p)
    : model_state_(ms), props_(p)
{
  // Launch a thread for the instance that will perform the actual
  // execution.
  thread_.reset(new std::thread([this]() { ExecuteThread(); }));
}

ModelInstance::~ModelInstance()
{
  // Send a "null" batch of requests to the execute thread to cause it
  // to finish what it is doing and exit.
  if (thread_ != nullptr) {
    std::unique_ptr<ExecRequests> null_requests;
    exec_queue_.Push(std::move(null_requests));
    thread_->join();
  }
}

TRITONSERVER_Error*
ModelInstance::Execute(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
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

  // Send the requests to the execute thread...
  std::unique_ptr<ExecRequests> exec_requests(new ExecRequests);
  exec_requests->requests_ = requests;
  exec_requests->request_count_ = request_count;
  exec_requests->responses_ = std::move(responses);
  exec_queue_.Push(std::move(exec_requests));

  return nullptr;  // success
}

void
ModelInstance::ExecuteThread()
{
  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("model ") + model_state_->ModelName() +
       ": starting execute thread for instance " + props_.AsString())
          .c_str());

  bool supports_batching = false;
  LOG_IF_ERROR(
      model_state_->SupportsFirstDimBatching(&supports_batching),
      "failed to determine batching support");

  // Process requests until receive a nullptr... then exit.
  do {
    // The way we collect these batch timestamps is not entirely
    // accurate. Normally, in a performant backend you would execute
    // all the requests at the same time, and so there would be a
    // single compute-start / compute-end time-range. But here we
    // execute each request separately so there is no single range.
    uint64_t min_exec_start_ns = std::numeric_limits<uint64_t>::max();
    uint64_t max_exec_end_ns = 0;
    uint64_t min_compute_start_ns = std::numeric_limits<uint64_t>::max();
    uint64_t max_compute_end_ns = 0;
    uint64_t total_batch_size = 0;

    // Show that this instance is ready to execute...
    model_state_->SetInstanceReady(props_.id_);

    // Wait for a batch of requests...
    std::unique_ptr<ExecRequests> exec_requests = exec_queue_.Pop();
    if (exec_requests == nullptr) {
      break;
    }

    TRITONBACKEND_Request** requests = exec_requests->requests_;
    uint32_t request_count = exec_requests->request_count_;
    std::vector<TRITONBACKEND_Response*>& responses = exec_requests->responses_;

    TRITONSERVER_LogMessage(
        TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
        (std::string("model ") + model_state_->ModelName() + ": executing " +
         std::to_string(request_count) + " requests on instance " +
         props_.AsString())
            .c_str());

    // This thread has ownership of 'requests', which means that a
    // response must be sent for every request. If something does go
    // wrong in processing a particular request then we send an error
    // response just for the specific request.

    // For simplicity we just process each request separately... in
    // general a backend should try to operate on the entire batch of
    // requests at the same time for improved performance.
    for (uint32_t r = 0; r < request_count; ++r) {
      DECL_TIMESTAMP(exec_start_ns);
      min_exec_start_ns = std::min(min_exec_start_ns, exec_start_ns);

      TRITONBACKEND_Request* request = requests[r];

      const char* request_id = "";
      GUARDED_RESPOND_IF_ERROR(
          responses, r, TRITONBACKEND_RequestId(request, &request_id));

      uint64_t correlation_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

      // Triton ensures that there is only a single input since that is
      // what is specified in the model configuration, so normally there
      // would be no reason to check it but we do here to demonstate the
      // API.
      uint32_t input_count = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r, TRITONBACKEND_RequestInputCount(request, &input_count));

      uint32_t requested_output_count = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

      // If an error response was sent for the above then display an
      // error message and move on to next request.
      if (responses[r] == nullptr) {
        TRITONSERVER_LogMessage(
            TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
            (std::string("request ") + std::to_string(r) +
             ": failed to read request properties, error response sent")
                .c_str());
        continue;
      }

      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
          (std::string("request ") + std::to_string(r) + ": id = \"" +
           request_id +
           "\", correlation_id = " + std::to_string(correlation_id) +
           ", input_count = " + std::to_string(input_count) +
           ", requested_output_count = " +
           std::to_string(requested_output_count))
              .c_str());

      // We already validated that the model configuration specifies
      // only a single input and Triton enforces that.
      TRITONBACKEND_Input* input = nullptr;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestInput(request, 0 /* index */, &input));

      // We also validated that the model configuration specifies only a
      // single output, but the request is not required to request any
      // output at all so we only produce an output if requested.
      const char* requested_output_name = nullptr;
      if (requested_output_count > 0) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_RequestOutputName(
                request, 0 /* index */, &requested_output_name));
      }

      // If an error response was sent while getting the input or
      // requested output name then display an error message and move on
      // to next request.
      if (responses[r] == nullptr) {
        TRITONSERVER_LogMessage(
            TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
            (std::string("request ") + std::to_string(r) +
             ": failed to read input or requested output name, error response "
             "sent")
                .c_str());
        continue;
      }

      const char* input_name;
      TRITONSERVER_DataType input_datatype;
      const int64_t* input_shape;
      uint32_t input_dims_count;
      uint64_t input_byte_size;
      uint32_t input_buffer_count;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_InputProperties(
              input, &input_name, &input_datatype, &input_shape,
              &input_dims_count, &input_byte_size, &input_buffer_count));
      if (responses[r] == nullptr) {
        TRITONSERVER_LogMessage(
            TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
            (std::string("request ") + std::to_string(r) +
             ": failed to read input properties, error response sent")
                .c_str());
        continue;
      }

      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
          (std::string("\tinput ") + input_name +
           ": datatype = " + TRITONSERVER_DataTypeString(input_datatype) +
           ", shape = " + nib::ShapeToString(input_shape, input_dims_count) +
           ", byte_size = " + std::to_string(input_byte_size) +
           ", buffer_count = " + std::to_string(input_buffer_count))
              .c_str());
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
          (std::string("\trequested_output ") + requested_output_name).c_str());

      // For statistics we need to collect the total batch size of all
      // the requests. If the model doesn't support batching then each
      // request is necessarily batch-size 1. If the model does
      // support batching then the first dimension of the shape is the
      // batch size.
      if (supports_batching && (input_dims_count > 0)) {
        total_batch_size += input_shape[0];
      } else {
        total_batch_size++;
      }

      uint64_t compute_start_ns = 0;
      uint64_t compute_end_ns = 0;

      // We only need to produce an output if it was requested.
      if (requested_output_count > 0) {
        // This backend simply copies the input tensor to the output
        // tensor. The input tensor contents are available in one or
        // more contiguous buffers. To do the copy we:
        //
        //   1. Create an output tensor in the response.
        //
        //   2. Allocate appropriately sized buffer in the output
        //      tensor.
        //
        //   3. Iterate over the input tensor buffers and copy the
        //      contents into the output buffer.
        TRITONBACKEND_Response* response = responses[r];

        // Step 1. Input and output have same datatype and shape...
        TRITONBACKEND_Output* output;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_ResponseOutput(
                response, &output, requested_output_name, input_datatype,
                input_shape, input_dims_count));
        if (responses[r] == nullptr) {
          TRITONSERVER_LogMessage(
              TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
              (std::string("request ") + std::to_string(r) +
               ": failed to create response output, error response sent")
                  .c_str());
          continue;
        }

        // Step 2. Get the output buffer. We request a buffer in CPU
        // memory but we have to handle any returned type. If we get
        // back a buffer in GPU memory we just fail the request.
        void* output_buffer;
        TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t output_memory_type_id = 0;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_OutputBuffer(
                output, &output_buffer, input_byte_size, &output_memory_type,
                &output_memory_type_id));
        if ((responses[r] == nullptr) ||
            (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "failed to create output buffer in CPU memory"));
          TRITONSERVER_LogMessage(
              TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
              (std::string("request ") + std::to_string(r) +
               ": failed to create output buffer in CPU memory, error response "
               "sent")
                  .c_str());
          continue;
        }

        SET_TIMESTAMP(compute_start_ns);
        min_compute_start_ns = std::min(min_compute_start_ns, compute_start_ns);

        // Step 3. Copy input -> output. We can only handle if the input
        // buffers are on CPU so fail otherwise.
        size_t output_buffer_offset = 0;
        for (uint32_t b = 0; b < input_buffer_count; ++b) {
          const void* input_buffer = nullptr;
          uint64_t buffer_byte_size = 0;
          TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
          int64_t input_memory_type_id = 0;
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONBACKEND_InputBuffer(
                  input, b, &input_buffer, &buffer_byte_size,
                  &input_memory_type, &input_memory_type_id));
          if ((responses[r] == nullptr) ||
              (input_memory_type == TRITONSERVER_MEMORY_GPU)) {
            GUARDED_RESPOND_IF_ERROR(
                responses, r,
                TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_UNSUPPORTED,
                    "failed to get input buffer in CPU memory"));
          }

          memcpy(
              reinterpret_cast<char*>(output_buffer) + output_buffer_offset,
              input_buffer, buffer_byte_size);
          output_buffer_offset += buffer_byte_size;
        }

        SET_TIMESTAMP(compute_end_ns);
        max_compute_end_ns = std::max(max_compute_end_ns, compute_end_ns);

        if (responses[r] == nullptr) {
          TRITONSERVER_LogMessage(
              TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
              (std::string("request ") + std::to_string(r) +
               ": failed to get input buffer in CPU memory, error response "
               "sent")
                  .c_str());
          continue;
        }
      }

      // If we get to this point then there hasn't been any error and
      // the response is complete and we can send it. This is the last
      // (and only) response that we are sending for the request so we
      // must mark it FINAL. If there is an error when sending all we
      // can do is log it.
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
              nullptr /* success */),
          "failed sending response");

      DECL_TIMESTAMP(exec_end_ns);
      max_exec_end_ns = std::max(max_exec_end_ns, exec_end_ns);

      // If there are no compute timestamps the the request didn't ask
      // for any outputs. Just set the compute start/end to equal
      // exec_end.
      if (compute_start_ns == 0) {
        compute_start_ns = compute_end_ns = exec_end_ns;
      }

      // Report statistics for the successful request. For instance
      // using the CPU we don't associate any device with the
      // statistics, otherwise we associate the instance's device.
      LOG_IF_ERROR(
          TRITONBACKEND_ModelReportStatistics(
              model_state_->TritonModel(), request, true /* success */,
              (props_.kind_ == nib::InstanceProperties::Kind::CPU)
                  ? TRITONBACKEND_NO_DEVICE
                  : props_.device_id_,
              exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
          "failed reporting request statistics");
    }

    // Done with requests...

    // There are two types of statistics that we can report... the
    // statistics for the entire batch of requests that we just
    // executed and statistics for each individual request. Statistics
    // for each individual request were reported above inside the loop
    // as each request was processed (or for failed requests we report
    // that failure below). Here we report statistics for the entire
    // batch of requests.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelReportBatchStatistics(
            model_state_->TritonModel(), total_batch_size, min_exec_start_ns,
            min_compute_start_ns, max_compute_end_ns, max_exec_end_ns),
        "failed reporting batch request statistics");

    // We could have released each request as soon as we sent the
    // corresponding response. But for clarity we just release them
    // all here. Note that is something goes wrong when releasing a
    // request all we can do is log it... there is no response left to
    // use to report an error.
    for (uint32_t r = 0; r < request_count; ++r) {
      TRITONBACKEND_Request* request = requests[r];

      // Before releasing, record failed requests as those where
      // responses[r] is nullptr. The timestamps are ignored in this
      // case.
      if (responses[r] == nullptr) {
        LOG_IF_ERROR(
            TRITONBACKEND_ModelReportStatistics(
                model_state_->TritonModel(), request, false /* success */,
                TRITONBACKEND_NO_DEVICE, 0, 0, 0, 0),
            "failed reporting request statistics");
      }

      LOG_IF_ERROR(
          TRITONBACKEND_RequestRelease(
              request, TRITONSERVER_REQUEST_RELEASE_ALL),
          "failed releasing request");
    }
  } while (true);

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("model ") + model_state_->ModelName() +
       ": ending execute thread for instance " + props_.AsString())
          .c_str());
}

//
// ModelState
//
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

  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

  uint64_t model_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &model_version));

  TRITONSERVER_Server* triton_server;
  RETURN_IF_ERROR(TRITONBACKEND_ModelServer(triton_model, &triton_server));

  *state = new ModelState(
      triton_server, triton_model, model_name, model_version,
      std::move(model_config));
  return nullptr;  // success
}

ModelState::ModelState(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* model_name, const uint64_t model_version,
    ni::TritonJson::Value&& model_config)
    : triton_server_(triton_server), triton_model_(triton_model),
      model_name_(model_name), model_version_(model_version),
      model_config_(std::move(model_config))
{
}

TRITONSERVER_Error*
ModelState::SupportsFirstDimBatching(bool* supports)
{
  // We can't determine this during model initialization because
  // TRITONSERVER_ServerModelBatchProperties can't be called until the
  // model is loaded...
  uint32_t flags = 0;
  RETURN_IF_ERROR(TRITONSERVER_ServerModelBatchProperties(
      triton_server_, model_name_.c_str(), model_version_, &flags,
      nullptr /* voidp */));
  *supports = ((flags & TRITONSERVER_BATCH_FIRST_DIM) != 0);
  return nullptr;  // success
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
  RETURN_IF_ERROR(nib::ParseShape(input, "dims", &input_shape));
  RETURN_IF_ERROR(nib::ParseShape(output, "dims", &output_shape));

  RETURN_ERROR_IF_FALSE(
      input_shape == output_shape, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected input and output shape to match, got ") +
          nib::ShapeToString(input_shape) + " and " +
          nib::ShapeToString(output_shape));

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::CreateModelInstances()
{
  std::vector<nib::InstanceProperties> instances;
  RETURN_IF_ERROR(nib::ParseInstanceGroups(model_config_, &instances));

  // Create the instances...
  size_t idx = 0;
  for (auto& instance : instances) {
    // Because this backend just copies IN -> OUT and requires that
    // input and output be in CPU memory, we fail if a GPU instances
    // is requested.
    RETURN_ERROR_IF_FALSE(
        instance.kind_ == nib::InstanceProperties::Kind::CPU,
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("backend only supports CPU instances"));

    // We rely on the instance ID to match the index in 'instances_',
    // so set it appropriately.
    instance.id_ = idx++;

    TRITONSERVER_LogMessage(
        TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
        (std::string("created instance: ") + instance.AsString()).c_str());

    instances_.emplace_back(new ModelInstance(this, instance));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::Execute(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  RETURN_ERROR_IF_FALSE(
      !ready_instance_idx_.Empty(), TRITONSERVER_ERROR_INTERNAL,
      std::string("expected available instance"));

  size_t idx = ready_instance_idx_.Pop();
  instances_[idx]->Execute(requests, request_count);

  return nullptr;  // success
}

void
ModelState::SetInstanceReady(const size_t idx)
{
  ready_instance_idx_.Push(idx);
}


void
ModelState::WaitForReadyInstance()
{
  ready_instance_idx_.WaitNotEmpty();
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

  // Parse instance_group to determine how many model instances are
  // being requested. This backend maintains a "ready list" of
  // instances and for each call to ModelExecute uses the next
  // instance in the list.
  RETURN_IF_ERROR(model_state->CreateModelInstances());

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
  // Triton only calls model execute from a single thread at a time
  // *for a given model*. But since this backend could be used by
  // multiple models the implementation needs to handle multiple
  // models executing at the same time. Good practice for this is to
  // use only function-local and model-specific state (obtained from
  // 'model'), which is what we do here.
  ModelState* state;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelState(model, reinterpret_cast<void**>(&state)));

  RETURN_IF_ERROR(state->Execute(requests, request_count));

  // Now wait for there to be an instance ready before returning. We
  // want ModelExecute to not be called until there is an instance
  // ready to execute, so we wait here because once we return Triton
  // is free to call ModelExecute again immediately.
  state->WaitForReadyInstance();

  return nullptr;  // success
}

}  // extern "C"
