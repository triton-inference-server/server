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

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#define TRITONCUSTOM_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONCUSTOM_EXPORT __attribute__((__visibility__("default")))
#else
#define TRITONCUSTOM_EXPORT
#endif

struct TRITONCUSTOM_Error;
struct TRITONCUSTOM_Message;
struct TRITONCUSTOM_Input;
struct TRITONCUSTOM_RequestedOutput;
struct TRITONCUSTOM_Output;
struct TRITONCUSTOM_Request;
struct TRITONCUSTOM_ResponseFactory;
struct TRITONCUSTOM_Response;
struct TRITONCUSTOM_Instance;

/// GPU device number that indicates that no GPU is available for a
/// custom backend instance. Typically a custom backend will then
/// execute using only CPU.
#define TRITONCUSTOM_NO_GPU_DEVICE -1

/// Type for the TRITONCUSTOM_InstanceNew function.
typedef TRITONCUSTOM_Error* TRITONCUSTOM_InstanceNewFn_t(
    TRITONCUSTOM_Instance**, const char*, const char*, const int64_t);

/// Type for the TRITONCUSTOM_InstanceDelete function.
typedef TRITONCUSTOM_Error* TRITONCUSTOM_InstanceDeleteFn_t(
    TRITONCUSTOM_Instance*);

/// Type for the TRITONCUSTOM_InstanceExecute function.
typedef TRITONCUSTOM_Error* TRITONCUSTOM_InstanceExecuteFn_t(
    TRITONCUSTOM_Instance*, const uint32_t, TRITONCUSTOM_Request**);

/// Tensor data types recognized by Triton and custom backends.
typedef enum custom_datatype_enum {
  TRITONCUSTOM_TYPE_BOOL,
  TRITONCUSTOM_TYPE_UINT8,
  TRITONCUSTOM_TYPE_UINT16,
  TRITONCUSTOM_TYPE_UINT32,
  TRITONCUSTOM_TYPE_UINT64,
  TRITONCUSTOM_TYPE_INT8,
  TRITONCUSTOM_TYPE_INT16,
  TRITONCUSTOM_TYPE_INT32,
  TRITONCUSTOM_TYPE_INT64,
  TRITONCUSTOM_TYPE_FP16,
  TRITONCUSTOM_TYPE_FP32,
  TRITONCUSTOM_TYPE_FP64,
  TRITONCUSTOM_TYPE_BYTES
} TRITONCUSTOM_DataType;

/// Types of memory recognized by Triton and custom backends.
typedef enum custom_memorytype_enum {
  TRITONCUSTOM_MEMORY_CPU,
  TRITONCUSTOM_MEMORY_CPU_PINNED,
  TRITONCUSTOM_MEMORY_GPU
} TRITONCUSTOM_MemoryType;

///
/// TRITONCUSTOM_Error
///
/// Errors are reported by a TRITONCUSTOM_Error object. A NULL
/// TRITONCUSTOM_Error indicates no error, a non-NULL
/// TRITONCUSTOM_Error indicates error and the code and message for
/// the error can be retrieved from the object.
///
/// For TRITONCUSTOM_Error objects returned by functions, the caller
/// takes ownership of a TRITONCUSTOM_Error object returned by the API
/// and must call TRITONCUSTOM_ErrorDelete to release the object.
///

/// The TRITONCUSTOM_Error error codes
typedef enum TRITONCUSTOM_errorcode_enum {
  TRITONCUSTOM_ERROR_UNKNOWN,
  TRITONCUSTOM_ERROR_INTERNAL,
  TRITONCUSTOM_ERROR_NOT_FOUND,
  TRITONCUSTOM_ERROR_INVALID_ARG,
  TRITONCUSTOM_ERROR_UNAVAILABLE,
  TRITONCUSTOM_ERROR_UNSUPPORTED,
  TRITONCUSTOM_ERROR_ALREADY_EXISTS
} TRITONCUSTOM_Error_Code;

/// Create a new error object. The caller takes ownership of the
/// TRITONCUSTOM_Error object and must call TRITONCUSTOM_ErrorDelete
/// to release the object.
///
/// \param code The error code.
/// \param msg The error message.
/// \return A new TRITONCUSTOM_Error object.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_ErrorNew(
    const TRITONCUSTOM_Error_Code code, const char* msg);

/// Delete an error object.
///
/// \param error The error object.
TRITONCUSTOM_EXPORT void TRITONCUSTOM_ErrorDelete(TRITONCUSTOM_Error* error);

/// Get the error code.
///
/// \param error The error object.
/// \return The error code.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error_Code
TRITONCUSTOM_ErrorCode(TRITONCUSTOM_Error* error);

/// Get the string representation of an error code. The returned
/// string is not owned by the caller and so should not be modified or
/// freed. The lifetime of the returned string extends only as long as
/// 'error' and must not be accessed once 'error' is deleted.
///
/// \param error The error object.
/// \return The string representation of the error code.
TRITONCUSTOM_EXPORT const char* TRITONCUSTOM_ErrorCodeString(
    TRITONCUSTOM_Error* error);

/// Get the error message. The returned string is not owned by the
/// caller and so should not be modified or freed. The lifetime of the
/// returned string extends only as long as 'error' and must not be
/// accessed once 'error' is deleted.
///
/// \param error The error object.
/// \return The error message.
TRITONCUSTOM_EXPORT const char* TRITONCUSTOM_ErrorMessage(
    TRITONCUSTOM_Error* error);

///
/// TRITONCUSTOM_Message
///
/// Object representing a Triton custom backend message.
///

/// Delete a message object.
/// \param message The message object.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_MessageDelete(
    TRITONCUSTOM_Message* message);

/// Get the base and size of the buffer containing the serialized
/// message in JSON format. The buffer is owned by the
/// TRITONCUSTOM_Message object and should not be modified or freed by
/// the caller. The lifetime of the buffer extends only as long as
/// 'message' and must not be accessed once 'message' is deleted.
/// \param message The message object.
/// \param base Returns the base of the JSON.
/// \param byte_size Returns the size, in bytes, of the JSON.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_MessageSerializeToJson(
    TRITONCUSTOM_Message* message, const char** base, size_t* byte_size);

///
/// TRITONCUSTOM_Input
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
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_InputProperties(
    TRITONCUSTOM_Input* input, const char** name,
    TRITONCUSTOM_DataType* datatype, int64_t** shape, uint32_t* dims_count,
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
/// TRITONCUSTOM_InputProperties.
/// \param buffer Returns a pointer to a contiguous block of data for
/// the named input.
/// \param buffer_byte_size Returns the size, in bytes, of 'buffer'.
/// \param memory_type Acts as both input and output. On input gives
/// the buffer memory type preferred by the function caller.  Returns
/// the actual memory type of 'buffer'.
/// \param memory_type_id Acts as both input and output. On input
/// gives the buffer memory type id preferred by the function caller.
/// Returns the actual memory type id of 'buffer'.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_InputBuffer(
    TRITONCUSTOM_Input* input, const uint32_t index, const void** buffer,
    uint64_t* buffer_byte_size, TRITONCUSTOM_MemoryType* memory_type,
    int64_t* memory_type_id);

///
/// TRITONCUSTOM_RequestedOutput
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
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_RequestedOutputName(
    TRITONCUSTOM_RequestedOutput* output, const char** name);

///
/// TRITONCUSTOM_Output
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
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_OutputBuffer(
    TRITONCUSTOM_Output* output, void** buffer,
    const uint64_t buffer_byte_size, TRITONCUSTOM_MemoryType* memory_type,
    int64_t* memory_type_id);

///
/// TRITONCUSTOM_Request
///
/// Object representing an inference request.
///

/// Get the ID of the request. Can be nullptr if request doesn't have
/// an ID. The returned string is owned by the request, not the
/// caller, and so should not be modified or freed.
///
/// \param request The inference request.
/// \param id Returns the ID.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_RequestId(
    TRITONCUSTOM_Request* request, const char** id);

/// Get the correlation ID of the request. Zero indicates that the
/// request does not have a correlation ID.
///
/// \param request The inference request.
/// \param id Returns the correlation ID.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_RequestCorrelationId(
    TRITONCUSTOM_Request* request, uint64_t* id);

/// Get the batch size of the request.
///
/// \param request The inference request.
/// \param batch_size Returns the batch size.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_RequestBatchSize(
    TRITONCUSTOM_Request* request, uint32_t* batch_size);

/// Get the number of input tensors specified in the request.
///
/// \param request The inference request.
/// \param count Returns the number of input tensors.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_RequestInputCount(
    TRITONCUSTOM_Request* request, uint32_t* count);

/// Get a request input tensor. The lifetime of the returned input
/// tensor object matches that of the request and so the input tensor
/// object should not be accessed after the request object is
/// released.
///
/// \param request The inference request.
/// \param index The index of the input tensor. Must be 0 <= index <
/// count, where count is the value returned by
/// TRITONCUSTOM_RequestInputCount.
/// \param input Returns the input tensor corresponding to the index.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_RequestInput(
    TRITONCUSTOM_Request* request, const uint32_t index,
    TRITONCUSTOM_Input** input);

/// Get the number of output tensors specified in the request.
///
/// \param request The inference request.
/// \param count Returns the number of output tensors.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_RequestOutputCount(
    TRITONCUSTOM_Request* request, uint32_t* count);

/// Get a requested output tensor. The lifetime of the returned output
/// tensor object matches that of the request and so the output tensor
/// object should not be accessed after the request object is
/// released.
///
/// \param request The inference request.
/// \param index The index of the output tensor. Must be 0 <= index <
/// count, where count is the value returned by
/// TRITONCUSTOM_RequestOutputCount.
/// \param output Returns the output tensor corresponding to the index.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_RequestOutput(
    TRITONCUSTOM_Request* request, const uint32_t index,
    TRITONCUSTOM_RequestedOutput** output);

/// Get the error associated with the request. No error will be set
/// when the request is passsed to the backend by
/// TRITONCUSTOM_InstanceExecute. The backend should set an error if
/// something goes wrong during the processing of the request.
///
/// \param request The inference request.
/// \return a TRITONCUSTOM_Error indicating the error associated with
/// the request, or nullptr if no error is associated with the
/// request.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_RequestError(
    TRITONCUSTOM_Request* request);

/// Set the error for the request. No error will be set when the
/// request is passsed to the backend by TRITONCUSTOM_InstanceExecute.
/// The backend should set an error if something goes wrong during the
/// processing of the request.
///
/// \param request The inference request.
/// \param error The TRITONCUSTOM_Error to set for the request.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_RequestSetError(
    TRITONCUSTOM_Request* request, TRITONCUSTOM_Error* error);

/// Release the request. The request should be released when it is no
/// longer needed by the backend. After this call returns the
/// 'request' object is no longer valid and must not be used. Any
/// tensor names, data types, shapes, etc. returned by
/// TRITONCUSTOM_Request* functions for this request is no longer
/// valid. If a persistent copy of that data is required it must be
/// created before calling this function.
///
/// \param request The inference request.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_RequestRelease(
    TRITONCUSTOM_Request* request);

///
/// TRITONCUSTOM_ResponseFactory
///
/// Object representing an inference response factory.
///

/// Create the response factory associated with a request. This
/// function can be called only once for a given request. Subsequent
/// calls with result in a TRITONCUSTOM_ERROR_ALREADY_EXISTS error.
///
/// \param factory Returns the new response factory.
/// \param request The inference request.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_ResponseFactoryNew(
    TRITONCUSTOM_ResponseFactory** factory, TRITONCUSTOM_Request* request);

/// Destroy a response factory.
///
/// \param factory The response factory.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_ResponseFactoryDelete(
    TRITONCUSTOM_ResponseFactory* factory);

///
/// TRITONCUSTOM_Response
///
/// Object representing an inference response.
///

/// Create a response using a factory.
///
/// \param response Returns the new response.
/// \param factory The response factory.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_ResponseNew(
    TRITONCUSTOM_Response** response, TRITONCUSTOM_ResponseFactory* factory);

/// Destroy a response. It is not necessary to delete a response if
/// TRITONCUSTOM_ResponseSend is called as that function transfers
/// ownership of the response object to Triton.
///
/// \param response The response.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_ResponseDelete(
    TRITONCUSTOM_Response* response);

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
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_ResponseOutput(
    TRITONCUSTOM_Response* response, TRITONCUSTOM_Output** output,
    const char* name, const TRITONCUSTOM_DataType datatype,
    const int64_t* shape, const uint32_t dims_count);

/// Send a response. Calling this function transfers ownership of the
/// response object to Triton. The caller must not access or delete
/// the response object after calling this function.
///
/// \param response The response.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_ResponseSend(
    TRITONCUSTOM_Response* response);

///
/// TRITONCUSTOM_Instance
///
/// Object representing a custom backend instance.
///

/// Get the user-specified state associated with the instance. The
/// state is completely owned and managed by the custom backend
/// instance.
///
/// \param instance The custom backend instance.
/// \param state Returns the user state, or nullptr if no user state.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_InstanceState(
    TRITONCUSTOM_Instance* instance, void** state);

/// Set the user-specified state associated with the instance. The
/// state is completely owned and managed by the custom backend
/// instance.
///
/// \param instance The custom backend instance.
/// \param state The user state, or nullptr if no user state.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_InstanceSetState(
    TRITONCUSTOM_Instance* instance, void* state);

/// Get the GPU device ID to initialize for, or
/// TRITONCUSTOM_NO_GPU_DEVICE if should initialize for CPU.
///
/// \param instance The custom backend instance.
/// \param message Returns the instance's device ID.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_InstanceDeviceId(
    TRITONCUSTOM_Instance* instance, int* device_id);

/// Get the model configuration associated with the custom backend
/// instance.  The caller takes ownership of the message object and
/// must call TRITONCUSTOM_MessageDelete to release the object.
///
/// \param instance The custom backend instance.
/// \param message Returns the model configuration as a message.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_InstanceModelConfig(
    TRITONCUSTOM_Instance* instance, TRITONCUSTOM_Message** message);

/// Get the full path to the root of the model repository that
/// contains this model. The returned string is owned by the instance,
/// not the caller, and so should not be modified or freed.
///
/// \param instance The custom backend instance.
/// \param path Returns the full path.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_InstanceRepositoryPath(
    TRITONCUSTOM_Instance* instance, const char** path);


/// Create a custom backend instance for a given model and version
/// and get the object associated with the instance.
///
/// \param instance Returns the new custom backend instance.
/// \param instance_name The unique name assigned to this instance of
/// the custom backend.
/// \param model_name The name of the model.
/// \param model_version The version of the model.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_InstanceNew(
    TRITONCUSTOM_Instance** instance, const char* instance_name,
    const char* model_name, const int64_t model_version);

/// Delete a custom backend instance. All state associated with the
/// custom backend instance should be freed and any threads created by
/// the instance should be exited/joined before returning from this
/// function.
///
/// \param instance The custom backend instance to finalize.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_InstanceDelete(
    TRITONCUSTOM_Instance* instance);

/// Execute a batch of one or more requests on the custom backend.
///
/// \param instance The custom backend instance.
/// \param request_cnt The number of requests in the batch.
/// \param requests The requests.
/// \return a TRITONCUSTOM_Error indicating success or failure.
TRITONCUSTOM_EXPORT TRITONCUSTOM_Error* TRITONCUSTOM_InstanceExecute(
    TRITONCUSTOM_Instance* instance, const uint32_t request_cnt,
    TRITONCUSTOM_Request** requests);


#ifdef __cplusplus
}
#endif
