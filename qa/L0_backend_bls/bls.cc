// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "bls.h"

namespace triton { namespace backend { namespace bls {

BLSExecutor::BLSExecutor(TRITONSERVER_Server* server)
    : server_(server), model_executor_(server)
{
}

TRITONSERVER_Error*
BLSExecutor::PrepareInferenceRequest(
    TRITONBACKEND_Request* bls_request,
    TRITONSERVER_InferenceRequest** irequest, const std::string model_name)
{
  // Get request_id, correlation_id, and flags from the current request
  // for preparing a new inference request that we will send to 'addsub_python'
  // or 'addsub_tf' model later.
  const char* request_id;
  uint64_t correlation_id;
  uint32_t flags;
  RETURN_IF_ERROR(TRITONBACKEND_RequestId(bls_request, &request_id));
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestCorrelationId(bls_request, &correlation_id));
  RETURN_IF_ERROR(TRITONBACKEND_RequestFlags(bls_request, &flags));

  // Create an inference request object. The inference request object
  // is where we set the name of the model we want to use for
  // inference and the input tensors.
  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestNew(
      irequest, server_, model_name.c_str(), -1 /* model_version */));
  // Set request_id, correlation_id, and flags for the new request.
  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetId(*irequest, request_id));
  RETURN_IF_ERROR(
      TRITONSERVER_InferenceRequestSetCorrelationId(*irequest, correlation_id));
  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetFlags(*irequest, flags));
  RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetReleaseCallback(
      *irequest, InferRequestComplete, nullptr /* request_release_userp */));

  return nullptr;  // success
}

TRITONSERVER_Error*
BLSExecutor::PrepareInferenceInput(
    TRITONBACKEND_Request* bls_request, TRITONSERVER_InferenceRequest* irequest)
{
  // Get the properties of the two inputs from the current request.
  // Then, add the two input tensors and append the input data to the new
  // request.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(bls_request, &input_count));

  TRITONBACKEND_Input* input;
  const char* name;
  TRITONSERVER_DataType datatype;
  const int64_t* shape;
  uint32_t dims_count;
  size_t data_byte_size;
  TRITONSERVER_MemoryType data_memory_type;
  int64_t data_memory_id;
  const char* data_buffer;

  for (size_t count = 0; count < input_count; count++) {
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(
        bls_request, count /* index */, &input));
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &name, &datatype, &shape, &dims_count, nullptr, nullptr));
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        input, 0 /* idx */, reinterpret_cast<const void**>(&data_buffer),
        &data_byte_size, &data_memory_type, &data_memory_id));
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAddInput(
        irequest, name, datatype, shape, dims_count));
    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestAppendInputData(
        irequest, name, &data_buffer[0], data_byte_size, data_memory_type,
        data_memory_id));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
BLSExecutor::PrepareInferenceOutput(
    TRITONBACKEND_Request* bls_request, TRITONSERVER_InferenceRequest* irequest)
{
  // Indicate the output tensors to be calculated and returned
  // for the inference request.
  uint32_t output_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestOutputCount(bls_request, &output_count));
  const char* output_name;
  for (size_t count = 0; count < output_count; count++) {
    RETURN_IF_ERROR(TRITONBACKEND_RequestOutputName(
        bls_request, count /* index */, &output_name));
    RETURN_IF_ERROR(
        TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output_name));
  }

  return nullptr;  // success
}

void
BLSExecutor::Execute(
    TRITONBACKEND_Request* bls_request, TRITONBACKEND_Response** response)
{
  // The names of the models that we will send internal requests on.
  std::vector<std::string> model_names = {"addsub_python", "addsub_tf"};

  // Check if both models are valid before executing request.
  try {
    for (size_t i = 0; i < 2; i++) {
      // Check if the model is ready.
      bool is_ready = false;
      THROW_IF_TRITON_ERROR(TRITONSERVER_ServerModelIsReady(
          server_, model_names[i].c_str(), -1 /* model_version */, &is_ready));
      if (!is_ready) {
        throw BLSBackendException(
            (std::string("Failed to execute the inference request. Model '") +
             model_names[i].c_str() + "' is not ready.")
                .c_str());
      }
      // For simplicity, decoupled API is not supported in this BLS backend. You
      // can implement your own backend that supports decoupled models.
      uint32_t txn_flags;
      THROW_IF_TRITON_ERROR(TRITONSERVER_ServerModelTransactionProperties(
          server_, model_names[i].c_str(), -1 /* model_version */, &txn_flags,
          nullptr /* voidp */));
      if ((txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0) {
        throw BLSBackendException(
            std::string("Model '") + model_names[i].c_str() +
            "' is using the decoupled. This BLS Backend doesn't support models "
            "using the decoupled transaction policy.");
      }
    }
  }
  catch (const BLSBackendException& bls_exception) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, bls_exception.what());
    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Failed to send inference requests"));
    return;
  }

  // Prepare std::future for each model. Since this BLS backend
  // can handle requests in parallel, we will send all the inference
  // requests first and then retrieve them later.
  std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures(2);

  // The inference request object for sending internal requests.
  TRITONSERVER_InferenceRequest* irequest = nullptr;

  // For each inference request, the backend sends two requests on the
  // 'addsub_python' and 'addsub_tf' models.
  try {
    for (size_t icount = 0; icount < 2; icount++) {
      // Initialize the inference request with required information.
      THROW_IF_TRITON_ERROR(
          PrepareInferenceRequest(bls_request, &irequest, model_names[icount]));
      THROW_IF_TRITON_ERROR(PrepareInferenceInput(bls_request, irequest));
      THROW_IF_TRITON_ERROR(PrepareInferenceOutput(bls_request, irequest));

      // Execute inference request.
      THROW_IF_TRITON_ERROR(
          model_executor_.AsyncExecute(irequest, &futures[icount]));
    }
  }
  catch (const BLSBackendException& bls_exception) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, bls_exception.what());
    LOG_IF_ERROR(
        TRITONSERVER_InferenceRequestDelete(irequest),
        "Failed to delete inference request.");
    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Failed to send inference requests"));
    return;
  }

  // If both internal requests are sent successfully, retrieve the output from
  // each request and construct the final response.
  ConstructFinalResponse(response, std::move(futures));
}

void
BLSExecutor::ConstructFinalResponse(
    TRITONBACKEND_Response** response,
    std::vector<std::future<TRITONSERVER_InferenceResponse*>> futures)
{
  // Prepare two TRITONSERVER_InferenceResponse* objects for 'addsub_python' and
  // 'addsub_tf' respectively.
  std::vector<TRITONSERVER_InferenceResponse*> completed_responses = {
      nullptr, nullptr};

  const char* output_name_server;
  TRITONSERVER_DataType output_datatype_server;
  const int64_t* output_shape_server;
  uint64_t dims_count_server;
  size_t output_byte_size_server;
  TRITONSERVER_MemoryType output_memory_type_server;
  int64_t output_memory_id_server;
  const void* output_base_server;
  void* userp_server;

  const char* output_name_backend;
  TRITONSERVER_DataType output_datatype_backend;
  const int64_t* output_shape_backend;
  uint64_t dims_count_backend;
  size_t output_byte_size_backend;
  TRITONSERVER_MemoryType output_memory_type_backend;
  int64_t output_memory_id_backend;
  const void* output_base_backend;
  void* userp_backend;
  for (size_t icount = 0; icount < 2; icount++) {
    // Retrieve the corresponding TRITONSERVER_InferenceResponse object from
    // 'futures'. The InferResponseComplete function sets the std::promise
    // so that this thread will block until the response is returned.
    completed_responses[icount] = futures[icount].get();
    try {
      THROW_IF_TRITON_ERROR(
          TRITONSERVER_InferenceResponseError(completed_responses[icount]));
    }
    catch (const BLSBackendException& bls_exception) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, bls_exception.what());

      if (completed_responses[icount] != nullptr) {
        LOG_IF_ERROR(
            TRITONSERVER_InferenceResponseDelete(completed_responses[icount]),
            "Failed to delete inference response.");
      }
      return;
    }
    // Retrieve outputs from 'completed_responses'.
    // Extract OUTPUT0 from the 'addsub_python' and OUTPUT1 from the
    // 'addsub_tf' model to form the final inference response object.
    // Get all the information about the output tensor.
    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONSERVER_InferenceResponseOutput(
            completed_responses[icount], icount, &output_name_server, &output_datatype_server,
            &output_shape_server, &dims_count_server, &output_base_server, &output_byte_size_server,
            &output_memory_type_server, &output_memory_id_server, &userp_server));

    // Create an output tensor in the final response with
    // the information retrieved above.
    TRITONBACKEND_Output* output;
    RESPOND_AND_SET_NULL_IF_ERROR(
        response, TRITONBACKEND_ResponseOutput(
                      *response, &output, output_name_server, output_datatype_server,
                      output_shape_server, dims_count_server));

    // Get a buffer that holds the tensor data for the output.
    // We request a buffer in CPU memory but we have to handle any returned
    // type. If we get back a buffer in GPU memory we just fail the request.
    void* output_buffer;
    output_memory_type_server = TRITONSERVER_MEMORY_CPU;
    RESPOND_AND_SET_NULL_IF_ERROR(
        response, TRITONBACKEND_OutputBuffer(
                      output, &output_buffer, output_byte_size_server,
                      &output_memory_type_server, &output_memory_id_server));
    if (output_memory_type_server == TRITONSERVER_MEMORY_GPU) {
      RESPOND_AND_SET_NULL_IF_ERROR(
          response, TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        "failed to create output buffer in CPU memory"));
    }
    // Validate new backen APIs, ensuring that the retrieved values match those retrieved by the server
    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONBACKEND_InferenceResponseOutput(
            *response, icount, &output_name_backend, &output_datatype_backend,
            &output_shape_backend, &dims_count_backend, &output_base_backend, &output_byte_size_backend,
            &output_memory_type_backend, &output_memory_id_backend, &userp_backend));

  
    if((*output_name_server != *output_name_backend) || (output_datatype_server != output_datatype_backend) || (*output_shape_server != *output_shape_backend) || (dims_count_server != dims_count_backend) ||
    (output_byte_size_server != output_byte_size_backend) || (output_memory_type_server != output_memory_type_backend) || (output_memory_id_server != output_memory_id_backend)) {
      RESPOND_AND_SET_NULL_IF_ERROR(response, TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Returned values do not match when using TRITONBACKEND_InferenceResponseOutput API"));
    }

    RESPOND_AND_SET_NULL_IF_ERROR(
        response,
        TRITONBACKEND_InferenceResponseOutputByName(
            *response, output_name_backend, &output_datatype_backend,
            &output_shape_backend, &dims_count_backend, &output_base_backend, &output_byte_size_backend,
            &output_memory_type_backend, &output_memory_id_backend, &userp_backend));

    if((*output_name_server != *output_name_backend) || (output_datatype_server != output_datatype_backend) || (*output_shape_server != *output_shape_backend) || (dims_count_server != dims_count_backend) ||
    (output_byte_size_server != output_byte_size_backend) || (output_memory_type_server != output_memory_type_backend) || (output_memory_id_server != output_memory_id_backend)) {
      RESPOND_AND_SET_NULL_IF_ERROR(response, TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Returned values do not match when using TRITONBACKEND_InferenceResponseOutputByName API"));
    }

    // Fill the BLS output buffer with output data returned by internal
    // requests.
    memcpy(output_buffer, output_base_server, output_byte_size_server);

    LOG_IF_ERROR(
        TRITONSERVER_InferenceResponseDelete(completed_responses[icount]),
        "Failed to delete inference response.");
  }
}

}}}  // namespace triton::backend::bls
