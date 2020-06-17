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
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "src/core/tritonserver.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#define TRITONBACKEND_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONBACKEND_EXPORT __attribute__((__visibility__("default")))
#else
#define TRITONBACKEND_EXPORT
#endif

struct TRITONBACKEND_Input;
struct TRITONBACKEND_RequestedOutput;
struct TRITONBACKEND_Output;
struct TRITONBACKEND_Request;
struct TRITONBACKEND_ResponseFactory;
struct TRITONBACKEND_Response;
struct TRITONBACKEND_Backend;
struct TRITONBACKEND_Model;

// Version of this TRITONBACKEND API.
#define TRITONBACKEND_API_VERSION 1

/// GPU device number that indicates that no GPU is available for a
/// backend. Typically a backend will then execute using only CPU.
#define TRITONBACKEND_NO_GPU_DEVICE -1

///
/// TRITONBACKEND_Input
///
/// Object representing an input tensor.
///

/// Get the name and properties of an input tensor. The returned
/// strings and other properties are owned by the input, not the
/// caller, and so should not be modified or freed.
///
/// \param input The input tensor.
/// \param name If non-nullptr, returns the tensor name.
/// \param datatype If non-nullptr, returns the tensor datatype.
/// \param shape If non-nullptr, returns the tensor shape.
/// \param dim_count If non-nullptr, returns the number of dimensions
/// in the tensor shape.
/// \param buffer_count If non-nullptr, returns the number of buffers
/// holding the contents of the tensor.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_InputProperties(
    TRITONBACKEND_Input* input, const char** name,
    TRITONSERVER_DataType* datatype, int64_t** shape, uint32_t* dims_count,
    uint32_t* buffer_count);

/// Get a buffer holding (part of) the tensor data for an input. The
/// returned buffer is owned by the input and so should not be
/// modified or freed by the caller. The lifetime of the buffer
/// matches that of the input and so the buffer should not be accessed
/// after the input tensor object is released.
///
/// \param input The input tensor.
/// \param index The index of the buffer. Must be 0 <= index <
/// buffer_count, where buffer_count is the value returned by
/// TRITONBACKEND_InputProperties.
/// \param buffer Returns a pointer to a contiguous block of data for
/// the named input.
/// \param buffer_byte_size Returns the size, in bytes, of 'buffer'.
/// \param memory_type Acts as both input and output. On input gives
/// the buffer memory type preferred by the function caller.  Returns
/// the actual memory type of 'buffer'.
/// \param memory_type_id Acts as both input and output. On input
/// gives the buffer memory type id preferred by the function caller.
/// Returns the actual memory type id of 'buffer'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_InputBuffer(
    TRITONBACKEND_Input* input, const uint32_t index, const void** buffer,
    uint64_t* buffer_byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id);

///
/// TRITONBACKEND_RequestedOutput
///
/// Object representing a requested output tensor.
///

/// Get the name of a requested output tensor. The returned string is
/// owned by the output, not the caller, and so should not be modified
/// or freed.
///
/// \param request The inference request.
/// \param output The output tensor.
/// \param name Returns the tensor name.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_RequestedOutputName(
    TRITONBACKEND_RequestedOutput* output, const char** name);

///
/// TRITONBACKEND_Output
///
/// Object representing a response output tensor.
///

/// Get a buffer to use to hold the tensor data for the output. The
/// returned buffer is owned by the output and so should not be freed
/// by the caller. The caller can and should fill the buffer with the
/// output data for the tensor. The lifetime of the buffer matches
/// that of the output and so the buffer should not be accessed after
/// the output tensor object is released.
///
/// \param buffer Returns a pointer to a buffer where the contents of
/// the output tensor should be placed.
/// \param buffer_byte_size The size, in bytes, of the buffer required
/// by the caller.
/// \param memory_type Acts as both input and output. On input gives
/// the buffer memory type preferred by the caller.  Returns the
/// actual memory type of 'buffer'.
/// \param memory_type_id Acts as both input and output. On input
/// gives the buffer memory type id preferred by the caller. Returns
/// the actual memory type id of 'buffer'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_OutputBuffer(
    TRITONBACKEND_Output* output, void** buffer,
    const uint64_t buffer_byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id);

///
/// TRITONBACKEND_Request
///
/// Object representing an inference request.
///

/// Get the ID of the request. Can be nullptr if request doesn't have
/// an ID. The returned string is owned by the request, not the
/// caller, and so should not be modified or freed.
///
/// \param request The inference request.
/// \param id Returns the ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_RequestId(
    TRITONBACKEND_Request* request, const char** id);

/// Get the correlation ID of the request. Zero indicates that the
/// request does not have a correlation ID.
///
/// \param request The inference request.
/// \param id Returns the correlation ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_RequestCorrelationId(
    TRITONBACKEND_Request* request, uint64_t* id);

/// Get the batch size of the request.
///
/// \param request The inference request.
/// \param batch_size Returns the batch size.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_RequestBatchSize(
    TRITONBACKEND_Request* request, uint32_t* batch_size);

/// Get the number of input tensors specified in the request.
///
/// \param request The inference request.
/// \param count Returns the number of input tensors.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_RequestInputCount(
    TRITONBACKEND_Request* request, uint32_t* count);

/// Get a request input tensor. The lifetime of the returned input
/// tensor object matches that of the request and so the input tensor
/// object should not be accessed after the request object is
/// released.
///
/// \param request The inference request.
/// \param index The index of the input tensor. Must be 0 <= index <
/// count, where count is the value returned by
/// TRITONBACKEND_RequestInputCount.
/// \param input Returns the input tensor corresponding to the index.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_RequestInput(
    TRITONBACKEND_Request* request, const uint32_t index,
    TRITONBACKEND_Input** input);

/// Get the number of output tensors specified in the request.
///
/// \param request The inference request.
/// \param count Returns the number of output tensors.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_RequestOutputCount(
    TRITONBACKEND_Request* request, uint32_t* count);

/// Get a requested output tensor. The lifetime of the returned output
/// tensor object matches that of the request and so the output tensor
/// object should not be accessed after the request object is
/// released.
///
/// \param request The inference request.
/// \param index The index of the output tensor. Must be 0 <= index <
/// count, where count is the value returned by
/// TRITONBACKEND_RequestOutputCount.
/// \param output Returns the output tensor corresponding to the index.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_RequestOutput(
    TRITONBACKEND_Request* request, const uint32_t index,
    TRITONBACKEND_RequestedOutput** output);

/// Release the request. The request should be released when it is no
/// longer needed by the backend. After this call returns the
/// 'request' object is no longer valid and must not be used. Any
/// tensor names, data types, shapes, etc. returned by
/// TRITONBACKEND_Request* functions for this request is no longer
/// valid. If a persistent copy of that data is required it must be
/// created before calling this function.
///
/// \param request The inference request.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_RequestRelease(
    TRITONBACKEND_Request* request);

///
/// TRITONBACKEND_ResponseFactory
///
/// Object representing an inference response factory.
///

/// Create the response factory associated with a request. This
/// function can be called only once for a given request. Subsequent
/// calls with result in a TRITONSERVER_ERROR_ALREADY_EXISTS error.
///
/// \param factory Returns the new response factory.
/// \param request The inference request.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ResponseFactoryNew(
    TRITONBACKEND_ResponseFactory** factory, TRITONBACKEND_Request* request);

/// Destroy a response factory.
///
/// \param factory The response factory.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ResponseFactoryDelete(
    TRITONBACKEND_ResponseFactory* factory);

///
/// TRITONBACKEND_Response
///
/// Object representing an inference response.
///

/// Create a response using a factory.
///
/// \param response Returns the new response.
/// \param factory The response factory.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ResponseNew(
    TRITONBACKEND_Response** response, TRITONBACKEND_ResponseFactory* factory);

/// Destroy a response. It is not necessary to delete a response if
/// TRITONBACKEND_ResponseSend is called as that function transfers
/// ownership of the response object to Triton.
///
/// \param response The response.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ResponseDelete(
    TRITONBACKEND_Response* response);

/// Get an output tensor in the response, creating if necessary. The
/// lifetime of the returned output tensor object matches that of the
/// response and so the output tensor object should not be accessed
/// after the response object is deleted.
///
/// \param response The response.
/// \param output Returns the new response output.
/// \param name The name of the output tensor.
/// \param datatype The datatype of the output tensor.
/// \param shape The shape of the output tensor.
/// \param dims_count The number of dimensions in the output tensor
/// shape.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ResponseOutput(
    TRITONBACKEND_Response* response, TRITONBACKEND_Output** output,
    const char* name, const TRITONSERVER_DataType datatype,
    const int64_t* shape, const uint32_t dims_count);

/// Send a successful response. Calling this function transfers
/// ownership of the response object to Triton. The caller must not
/// access or delete the response object after calling this function.
///
/// \param response The response.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ResponseSend(
    TRITONBACKEND_Response* response);

/// Send an error response. Calling this function transfers ownership
/// of the response object to Triton. The caller must not access or
/// delete the response object after calling this function.
///
/// \param response The response.
/// \param error The TRITONSERVER_Error to send.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ResponseSendError(
    TRITONBACKEND_Response* response, TRITONSERVER_Error* error);

///
/// TRITONBACKEND_Backend
///
/// Object representing a backend.
///

/// Get the name of the backend. The caller does not own the returned
/// string and must not modify or delete it. The lifetime of the
/// returned string extends only as long as 'backend'.
///
/// \param backend The backend.
/// \param name Returns the name of the backend.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONBACKEND_BackendName(
    TRITONBACKEND_Backend* backend, const char** name);

/// Get the TRITONBACKEND API version supported by Triton. This value
/// can be compared against the TRITONBACKEND_API_VERSION used to
/// compile the backend to ensure that Triton is compatible with the
/// backend.
///
/// \param backend The backend.
/// \param api_version Returns the TRITONBACKEND API version supported
/// by Triton.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_BackendApiVersion(
    TRITONBACKEND_Backend* backend, uint32_t* api_version);

/// Get the user-specified state associated with the backend. The
/// state is completely owned and managed by the backend.
///
/// \param backend The backend.
/// \param state Returns the user state, or nullptr if no user state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_BackendState(
    TRITONBACKEND_Backend* backend, void** state);

/// Set the user-specified state associated with the backend. The
/// state is completely owned and managed by the backend.
///
/// \param backend The backend.
/// \param state The user state, or nullptr if no user state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_BackendSetState(
    TRITONBACKEND_Backend* backend, void* state);

///
/// TRITONBACKEND_Model
///
/// Object representing a model implemented using the backend.
///

/// Get the name of the model. The returned string is owned by the
/// model object, not the caller, and so should not be modified or
/// freed.
///
/// \param model The model.
/// \param name Returns the model name.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ModelName(
    TRITONBACKEND_Model* model, const char** name);

/// Get the version of the model.
///
/// \param model The model.
/// \param version Returns the model version.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ModelVersion(
    TRITONBACKEND_Model* model, uint64_t* version);

/// Get the full path to the directory in the model repository that
/// contains this model. The returned string is owned by the model
/// object, not the caller, and so should not be modified or freed.
///
/// \param model The model.
/// \param path Returns the full path.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ModelRepositoryPath(
    TRITONBACKEND_Model* model, const char** path);

/// Get the model configuration. The caller takes ownership of the
/// message object and must call TRITONSERVER_MessageDelete to release
/// the object. The configuration is available via this call even
/// before the model is loaded and so can be used in
/// TRITONBACKEND_ModelInitialize. TRITONSERVER_ServerModelConfig
/// returns equivalent information but is not useable until after the
/// model loads.
///
/// \param model The model.
/// \param model_config Returns the model configuration as a message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ModelConfig(
    TRITONBACKEND_Model* model, TRITONSERVER_Message** model_config);

/// Get the backend used by the model.
///
/// \param model The model.
/// \param model Returns the model object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ModelBackend(
    TRITONBACKEND_Model* model, TRITONBACKEND_Backend** backend);

/// Get the user-specified state associated with the model. The
/// state is completely owned and managed by the backend.
///
/// \param model The model.
/// \param state Returns the user state, or nullptr if no user state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ModelState(
    TRITONBACKEND_Model* model, void** state);

/// Set the user-specified state associated with the model. The
/// state is completely owned and managed by the backend.
///
/// \param model The model.
/// \param state The user state, or nullptr if no user state.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ModelSetState(
    TRITONBACKEND_Model* model, void* state);

///
/// The following functions can be implemented by a backend. Functions
/// indicated as required must be implemented or the backend will fail
/// to load.
///

/// Initialize a backend. This function is optional, a backend is not
/// required to implement it. This function is called once when a
/// backend is loaded to allow the backend to initialize any state
/// associated with the backend. A backend has a single state that is
/// shared across all models that use the backend.
///
/// \param backend The backend.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_Initialize(
    TRITONBACKEND_Backend* backend);

/// Finalize for a backend. This function is optional, a backend is
/// not required to implement it. This function is called once, just
/// before the backend is unloaded. All state associated with the
/// backend should be freed and any threads created for the backend
/// should be exited/joined before returning from this function.
///
/// \param backend The backend.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_Finalize(
    TRITONBACKEND_Backend* backend);

/// Initialize for a model. This function is optional, a backend is
/// not required to implement it. This function is called once when a
/// model that uses the backend is loaded to allow the backend to
/// initialize any state associated with the model. The backend should
/// also examine the model configuration to determine if the
/// configuration is suitable for the backend. Any errors reported by
/// this function will prevent the model from loading.
///
/// \param model The model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(
    TRITONBACKEND_Model* model);

/// Finalize for a model. This function is optional, a backend is not
/// required to implement it. This function is called once, just
/// before the model is unloaded from Triton. All state associated
/// with the model should be freed and any threads created for the
/// model should be exited/joined before returning from this function.
///
/// \param model The model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(
    TRITONBACKEND_Model* model);

/// Execute a batch of one or more requests. This function is
/// required.
///
/// If an error is returned the ownership of the request objects
/// remains with Triton and the backend must not retain references to
/// the request objects or access them in any way.
///
/// If success is returned ownership of the request objects is
/// transferred to the backend and it is then responsible for creating
/// responses and releasing the request objects. If the backend needs
/// to access 'requests' after this function returns the backend must
/// make a copy of 'requests'.
///
/// \param model The model.
/// \param requests The requests.
/// \param request_count The number of requests in the batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_EXPORT TRITONSERVER_Error* TRITONBACKEND_ModelExecute(
    TRITONBACKEND_Model* model, TRITONBACKEND_Request** requests,
    const uint32_t request_count);


#ifdef __cplusplus
}
#endif
