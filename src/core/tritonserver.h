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

/// \file

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#else
typedef void cudaIpcMemHandle_t;
#endif  // TRTIS_ENABLE_GPU

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#define TRITONSERVER_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONSERVER_EXPORT __attribute__((__visibility__("default")))
#else
#define TRITONSERVER_EXPORT
#endif

struct TRITONSERVER_Error;
struct TRITONSERVER_ResponseAllocator;
struct TRITONSERVER_Server;
struct TRITONSERVER_ServerOptions;
struct TRITONSERVER_Trace;
struct TRITONSERVER_TraceManager;

struct TRITONSERVER_Metrics;
struct TRITONSERVER_Message;
struct TRITONSERVER_InferenceRequest;

/// Types of memory recognized by TRITONSERVER.
typedef enum TRITONSERVER_memorytype_enum {
  TRITONSERVER_MEMORY_CPU,
  TRITONSERVER_MEMORY_GPU,
  TRITONSERVER_MEMORY_CPU_PINNED
} TRITONSERVER_Memory_Type;

/// TRITONSERVER_Error
///
/// Errors are reported by a TRITONSERVER_Error object. A NULL
/// TRITONSERVER_Error indicates no error, a non-NULL TRITONSERVER_Error
/// indicates error and the code and message for the error can be
/// retrieved from the object.
///
/// The caller takes ownership of a TRITONSERVER_Error object returned by
/// the API and must call TRITONSERVER_ErrorDelete to release the object.
///

/// The TRITONSERVER_Error error codes
typedef enum TRITONSERVER_errorcode_enum {
  TRITONSERVER_ERROR_UNKNOWN,
  TRITONSERVER_ERROR_INTERNAL,
  TRITONSERVER_ERROR_NOT_FOUND,
  TRITONSERVER_ERROR_INVALID_ARG,
  TRITONSERVER_ERROR_UNAVAILABLE,
  TRITONSERVER_ERROR_UNSUPPORTED,
  TRITONSERVER_ERROR_ALREADY_EXISTS
} TRITONSERVER_Error_Code;

/// Create a new error object. The caller takes ownership of the
/// TRITONSERVER_Error object and must call TRITONSERVER_ErrorDelete to
/// release the object.
///
/// \param code The error code.
/// \param msg The error message.
/// \return A new TRITONSERVER_Error object.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ErrorNew(
    TRITONSERVER_Error_Code code, const char* msg);

/// Delete an error object.
///
/// \param error The error object.
TRITONSERVER_EXPORT void TRITONSERVER_ErrorDelete(TRITONSERVER_Error* error);

/// Get the error code.
///
/// \param error The error object.
/// \return The error code.
TRITONSERVER_EXPORT TRITONSERVER_Error_Code
TRITONSERVER_ErrorCode(TRITONSERVER_Error* error);

/// Get the string representation of an error code. The returned
/// string is not owned by the caller and so should not be modified or
/// freed. The lifetime of the returned string extends only as long as
/// 'error' and must not be accessed once 'error' is deleted.
///
/// \param error The error object.
/// \return The string representation of the error code.
TRITONSERVER_EXPORT const char* TRITONSERVER_ErrorCodeString(
    TRITONSERVER_Error* error);

/// Get the error message. The returned string is not owned by the
/// caller and so should not be modified or freed. The lifetime of the
/// returned string extends only as long as 'error' and must not be
/// accessed once 'error' is deleted.
///
/// \param error The error object.
/// \return The error message.
TRITONSERVER_EXPORT const char* TRITONSERVER_ErrorMessage(
    TRITONSERVER_Error* error);

/// TRITONSERVER_ResponseAllocator
///
/// Object representing a memory allocator for inference response
/// tensors.
///

/// Type for allocation function that allocates a buffer to hold a
/// result tensor.
///
/// Return in 'buffer' a pointer to the contiguous memory block of
/// size 'byte_size' for result tensor called 'tensor_name'. The
/// buffer must be allocated in the memory type identified by
/// 'memory_type' and 'memory_type_id'. The 'userp' data is the same
/// as what is supplied in the call to TRITONSERVER_ServerInferAsync.
///
/// Return in 'buffer_userp' a user-specified value to associate with
/// the buffer. This value will be provided in the call to
/// TRITONSERVER_ResponseAllocatorReleaseFn_t.
///
/// The function will be called once for each result tensor, even if
/// the 'byte_size' required for that tensor is zero. When 'byte_size'
/// is zero the function does not need to allocate any memory but may
/// perform other tasks associated with the result tensor. In this
/// case the function should return success and set 'buffer' ==
/// nullptr.
///
/// If the function is called with 'byte_size' non-zero the function should
/// allocate a contiguous buffer of the requested size. If possible the function
/// should allocate the buffer in the requested 'memory_type' and
/// 'memory_type_id', but the function is free to allocate the buffer in any
/// memory. The function must return in 'actual_memory_type' and
/// 'actual_memory_type_id' the memory where the buffer is allocated.
///
/// The function should return a TRITONSERVER_Error object if a failure
/// occurs while attempting an allocation. If an error object is
/// returned for a result tensor, the inference server will assume allocation
/// is not possible for the result buffer and will abort the inference
/// request.
typedef TRITONSERVER_Error* (*TRITONSERVER_ResponseAllocatorAllocFn_t)(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_Memory_Type memory_type,
    int64_t memory_type_id, void* userp, void** buffer, void** buffer_userp,
    TRITONSERVER_Memory_Type* actual_memory_type,
    int64_t* actual_memory_type_id);

/// Type for function that is called when the server no longer holds
/// any reference to a buffer allocated by
/// TRITONSERVER_ResponseAllocatorAllocFn_t. In practice this function is
/// called when the response object associated with the buffer is
/// deleted by TRITONSERVER_InferenceResponseDelete.
///
/// The 'buffer' and 'buffer_userp' arguments equal those returned by
/// TRITONSERVER_ResponseAllocatorAllocFn_t and 'byte_size',
/// 'memory_type' and 'memory_type_id' equal the values passed to
/// TRITONSERVER_ResponseAllocatorAllocFn_t.
///
/// Return a TRITONSERVER_Error object on failure, return nullptr on
/// success.
typedef TRITONSERVER_Error* (*TRITONSERVER_ResponseAllocatorReleaseFn_t)(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_Memory_Type memory_type,
    int64_t memory_type_id);

/// Create a new response allocator object.
///
/// \param allocator Returns the new response allocator object.
/// \param alloc_fn The function to call to allocate buffers for result
/// tensors.
/// \param release_fn The function to call when the server no longer
/// holds a reference to an allocated buffer.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ResponseAllocatorNew(
    TRITONSERVER_ResponseAllocator** allocator,
    TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
    TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn);

/// Delete a response allocator.
///
/// \param allocator The response allocator object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ResponseAllocatorDelete(
    TRITONSERVER_ResponseAllocator* allocator);

/// TRITONSERVER_Message
///
/// Object representing a Triton Server message.
///

/// Delete a message object.
///
/// \param message The message object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_MessageDelete(
    TRITONSERVER_Message* message);

/// Get the base and size of the buffer containing the serialized
/// message in JSON format. The buffer is owned by the
/// TRITONSERVER_Message object and should not be modified or freed by
/// the caller. The lifetime of the buffer extends only as long as
/// 'message' and must not be accessed once 'message' is deleted.
///
/// \param message The message object.
/// \param base Returns the base of the serialized message.
/// \param byte_size Returns the size, in bytes, of the serialized
/// message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_MessageSerializeToJson(
    TRITONSERVER_Message* message, const char** base, size_t* byte_size);

/// TRITONSERVER_Metrics
///
/// Object representing metrics.
///

/// Metric format types
typedef enum tritonserver_metricformat_enum {
  TRITONSERVER_METRIC_PROMETHEUS
} TRITONSERVER_Metric_Format;

/// Delete a metrics object.
///
/// \param metrics The metrics object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_MetricsDelete(
    TRITONSERVER_Metrics* metrics);

/// Get a buffer containing the metrics in the specified format. For
/// each format the buffer contains the following:
///
///   TRITONSERVER_METRIC_PROMETHEUS: 'base' points to a single multiline
///   string (char*) that gives a text representation of the metrics in
///   prometheus format. 'byte_size' returns the length of the string
///   in bytes.
///
/// The buffer is owned by the 'metrics' object and should not be
/// modified or freed by the caller. The lifetime of the buffer
/// extends only as long as 'metrics' and must not be accessed once
/// 'metrics' is deleted.
///
/// \param metrics The metrics object.
/// \param format The format to use for the returned metrics.
/// \param base Returns a pointer to the base of the formatted
/// metrics, as described above.
/// \param byte_size Returns the size, in bytes, of the formatted
/// metrics.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_MetricsFormatted(
    TRITONSERVER_Metrics* metrics, TRITONSERVER_Metric_Format format,
    const char** base, size_t* byte_size);

/// TRITONSERVER_Trace
///
/// Object that represents tracing for a request.
///

/// Trace levels
typedef enum tritonserver_tracelevel_enum {
  TRITONSERVER_TRACE_LEVEL_DISABLED,
  TRITONSERVER_TRACE_LEVEL_MIN,
  TRITONSERVER_TRACE_LEVEL_MAX
} TRITONSERVER_Trace_Level;

// Trace activities
typedef enum tritonserver_traceactivity_enum {
  TRITONSERVER_TRACE_REQUEST_START,
  TRITONSERVER_TRACE_QUEUE_START,
  TRITONSERVER_TRACE_COMPUTE_START,
  TRITONSERVER_TRACE_COMPUTE_INPUT_END,
  TRITONSERVER_TRACE_COMPUTE_OUTPUT_START,
  TRITONSERVER_TRACE_COMPUTE_END,
  TRITONSERVER_TRACE_REQUEST_END
} TRITONSERVER_Trace_Activity;

/// Type for trace activity callback function. This callback function
/// is used to report activity occurring during a traced request. The
/// 'userp' data is the same as what is supplied in the call to
/// TRITONSERVER_TraceNew.
typedef void (*TRITONSERVER_TraceActivityFn_t)(
    TRITONSERVER_Trace* trace, TRITONSERVER_Trace_Activity activity,
    uint64_t timestamp_ns, void* userp);

/// Create a new trace object. The caller takes ownership of the
/// TRITONSERVER_Trace object and must call TRITONSERVER_TraceDelete to
/// release the object.
///
/// \param trace Returns the new trace object.
/// \param level The tracing level.
/// \param activity_fn The callback function where activity for the
/// trace is reported.
/// \param activity_userp User-provided pointer that is delivered to
/// the activity function.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_TraceNew(
    TRITONSERVER_Trace** trace, TRITONSERVER_Trace_Level level,
    TRITONSERVER_TraceActivityFn_t activity_fn, void* activity_userp);

/// Delete a trace object.
///
/// \param trace The trace object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_TraceDelete(
    TRITONSERVER_Trace* trace);

/// Get the name of the model being traced. The caller
/// does not own the returned string and must not modify or delete
/// it. The lifetime of the returned string extends only as long as
/// 'trace' and must not be accessed once 'trace' is deleted.
/// This method is only guaranteed to correctly return the 'model_name' in the
/// invocation of TRITONSERVER_TraceManagerReleaseTraceFn_t
///
/// \param trace The trace object.
/// \param model_name Returns the name of the model being traced.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_TraceModelName(
    TRITONSERVER_Trace* trace, const char** model_name);

/// Get the version of the model being traced.
/// This method is only guaranteed to correctly return the 'model_version' in
/// the invocation of TRITONSERVER_TraceManagerReleaseTraceFn_t
///
/// \param trace The trace object.
/// \param model_version Returns the version of the model being traced.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_TraceModelVersion(
    TRITONSERVER_Trace* trace, int64_t* model_version);

/// Get the id of the trace object. Each trace object created during execution
/// of the inference server has a unique id.
///
/// \param trace The trace object.
/// \param id Returns the id of the trace object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_TraceId(
    TRITONSERVER_Trace* trace, int64_t* id);

/// Get the parent id of the trace object. The parent id will be set if the
/// trace object is created from within another traced request, and the parent
/// id will be set to the id of the trace object associated with that
/// traced request.
/// This method is only guaranteed to correctly return the 'parent_id' in
/// the invocation of TRITONSERVER_TraceManagerReleaseTraceFn_t
///
/// \param trace The trace object.
/// \param parent_id Returns the parent id of the trace object. -1 indicates
/// that the trace object has no parent.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_TraceParentId(
    TRITONSERVER_Trace* trace, int64_t* parent_id);

/// TRITONSERVER_TraceManager
///
/// Object representing a manager for initiating traces.
///

/// Type for trace creation callback function. This callback function
/// is used when a model execution is initiated within the request, if the
/// request is to be traced. The user should call TRITONSERVER_TraceNew and
/// return the new trace object if the user decides to trace the model
/// execution. Otherwise, the user should set 'trace' == nullptr. The 'userp'
/// data is the same as 'userp' supplied in the call to
/// TRITONSERVER_TraceManagerNew.
typedef void (*TRITONSERVER_TraceManagerCreateTraceFn_t)(
    TRITONSERVER_Trace** trace, const char* model_name, int64_t version,
    void* userp);

/// Type for trace release callback function. This callback function
/// is invoked when the model execution being traced is completed. By this
/// point, it is the user's responsiblity to delete 'trace' object created from
/// TRITONSERVER_TraceManagerCreateTraceFn_t by calling
/// TRITONSERVER_TraceDelete. The 'activity_userp' data is the same as
/// 'activity_userp' supplied in the call to TRITONSERVER_TraceNew. The 'userp'
/// data is the same as 'userp' supplied in the call to
/// TRITONSERVER_TraceManagerNew.
typedef void (*TRITONSERVER_TraceManagerReleaseTraceFn_t)(
    TRITONSERVER_Trace* trace, void* activity_userp, void* userp);

/// Create a new trace manager object.
///
/// \param trace_manager Returns the new trace manager object.
/// \param create_fn The function to call to create trace object for a request.
/// \param release_fn The function to call when the request associated with a
/// trace object is complete.
/// \param userp User-provided pointer that is delivered to the trace
/// creation and release function.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_TraceManagerNew(
    TRITONSERVER_TraceManager** trace_manager,
    TRITONSERVER_TraceManagerCreateTraceFn_t create_fn,
    TRITONSERVER_TraceManagerReleaseTraceFn_t release_fn, void* userp);

/// Delete a trace manager.
///
/// \param trace_manager The trace manager object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_TraceManagerDelete(
    TRITONSERVER_TraceManager* trace_manager);

/// TRITONSERVER_InferenceRequest
///
/// Object representing an inference request. The inference request
/// provides the meta-data and input tensor values needed for an
/// inference and returns the inference result meta-data and output
/// tensors. An inference request object can be modified and reused
/// multiple times.
///

/// Inference request flags. The enum values must be power-of-2 values.
typedef enum tritonserver_requestflag_enum {
  TRITONSERVER_REQUEST_FLAG_NONE = 0,
  TRITONSERVER_REQUEST_FLAG_SEQUENCE_START = 1,
  TRITONSERVER_REQUEST_FLAG_SEQUENCE_END = 2
} TRITONSERVER_Request_Flag;

/// Create a new inference request object.
///
/// \param inference_request Returns the new request object.
/// \param server the inference server object.
/// \param model_name The name of the model to use for the request.
/// \param model_version The version of the model to use for the
/// request. If nullptr or empty then the server will choose a version
/// based on the model's policy.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestNew(
    TRITONSERVER_InferenceRequest** inference_request,
    TRITONSERVER_Server* server, const char* model_name,
    const char* model_version);

/// Delete an inference request object.
///
/// \param inference_request The request object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestDelete(
    TRITONSERVER_InferenceRequest* inference_request);

/// Get the ID for a request. The returned ID is owned by
/// 'inference_request' and must not be modified or freed by the
/// caller.
///
/// \param inference_request The request object.
/// \param id Returns the ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestId(
    TRITONSERVER_InferenceRequest* inference_request, const char** id);

/// Set the ID for a request.
///
/// \param inference_request The request object.
/// \param id The ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestSetId(
    TRITONSERVER_InferenceRequest* inference_request, const char* id);

/// Get the flag(s) associated with a request. On return 'flags' holds
/// a bitwise-or of all flag values, see
/// TRITONSERVER_Request_Options_Flag for available flags.
///
/// \param inference_request The request object.
/// \param flags Returns the flags.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestFlags(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t* flags);

/// Set the flag(s) associated with a request. 'flags'
/// should holds a bitwise-or of all flag values, see
/// TRITONSERVER_Request_Flag for available flags.
///
/// \param inference_request The request object.
/// \param flags The flags.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestSetFlags(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t flags);

/// Get the correlation ID of the inference request. Default is 0,
/// which indictes that the request has no correlation ID. The
/// correlation ID is used to indicate two or more inference request
/// are related to each other. How this relationship is handled by the
/// inference server is determined by the model's scheduling
/// policy.
///
/// \param inference_request The request object.
/// \param correlation_id Returns the correlation ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestCorrelationId(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t* correlation_id);

/// Set the correlation ID of the inference request. Default is 0, which
/// indictes that the request has no correlation ID. The correlation ID
/// is used to indicate two or more inference request are related to
/// each other. How this relationship is handled by the inference
/// server is determined by the model's scheduling policy.
///
/// \param inference_request The request object.
/// \param correlation_id The correlation ID.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetCorrelationId(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t correlation_id);

/// Get the priority for a request. The default is 0 indicating that
/// the request does not specify a priority and so will use the
/// model's default priority.
///
/// \param inference_request The request object.
/// \param priority Returns the priority level.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestPriority(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t* priority);

/// Set the priority for a request. The default is 0 indicating that
/// the request does not specify a priority and so will use the
/// model's default priority.
///
/// \param inference_request The request object.
/// \param priority The priority level.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetPriority(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t priority);

/// Get the timeout for a request, in microseconds. The default is 0
/// which indicates that the request has no timeout.
///
/// \param inference_request The request object.
/// \param timeout_us Returns the timeout, in microseconds.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestTimeoutMicroseconds(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t* timeout_us);

/// Set the timeout for a request, in microseconds. The default is 0
/// which indicates that the request has no timeout.
///
/// \param inference_request The request object.
/// \param timeout_us The timeout, in microseconds.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t timeout_us);

/// Add an input to a request.
///
/// \param inference_request The request object.
/// \param name The name of the input.
/// \param datatype The type of the input. Valid type names are BOOL,
/// UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FP16,
/// FP32, FP64, and BYTES.
/// \param shape The shape of the input.
/// \param dim_count The number of dimensions of 'shape'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestAddInput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const char* datatype, const int64_t* shape, uint64_t dim_count);

/// Remove an input from a request.
///
/// \param inference_request The request object.
/// \param name The name of the input.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveInput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name);

/// Remove all inputs from a request.
///
/// \param inference_request The request object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllInputs(
    TRITONSERVER_InferenceRequest* inference_request);

/// Assign a buffer of data to an input. The buffer will be appended
/// to any existing buffers for that input. The 'inference_request'
/// object takes ownership of the buffer and so the caller should not
/// modify or free the buffer until that ownership is released by
/// 'inference_request' being deleted or by the input being removed
/// from 'inference_request'.
///
/// \param inference_request The request object.
/// \param name The name of the input.
/// \param base The base address of the input data.
/// \param byte_size The size, in bytes, of the input data.
/// \param memory_type The memory type of the input data.
/// \param memory_type_id The memory type id of the input data.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAppendInputData(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void* base, size_t byte_size, TRITONSERVER_Memory_Type memory_type,
    int64_t memory_type_id);

/// Clear all input data from an input, releasing ownership of the
/// buffer(s) that were appended to the input with
/// TRITONSERVER_InferenceRequestAppendInputData.
///
/// \param inference_request The request object.
/// \param name The name of the input.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllInputData(
    TRITONSERVER_InferenceRequest* inference_request, const char* name);

/// Add an output request to an inference request.
///
/// \param inference_request The request object.
/// \param name The name of the output.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestAddRequestedOutput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name);

/// Remove an output request from an inference request.
///
/// \param inference_request The request object.
/// \param name The name of the output.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveRequestedOutput(
    TRITONSERVER_InferenceRequest* inference_request, const char* name);

/// Remove all output requests from an inference request.
///
/// \param inference_request The request object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllRequestedOutputs(
    TRITONSERVER_InferenceRequest* inference_request);

/// Set that a requested output should be returned as a tensor of
/// classification strings instead of as the tensor defined by the model.
///
/// \param inference_request The request object.
/// \param name The name of the output.
/// \param count Indicates how many classification values should be
/// returned for the output. The 'count' highest priority values are
/// returned. The default is 0, indicating that the output tensor
/// should not be returned as a classification.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetRequestedOutputClassificationCount(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    uint32_t count);

/// Return the error status of an inference request corresponding to
/// the most recent call to TRITONSERVER_ServerInferAsync. Return a
/// TRITONSERVER_Error object on failure, return nullptr on success.  The
/// returned error object is owned by 'inference_request' and so
/// should not be deleted by the caller.
///
/// \param inference_request The request object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestError(
    TRITONSERVER_InferenceRequest* inference_request);

/// Get the datatype of an output tensor.
///
/// \param inference_request The request object.
/// \param name The name of the output.
/// \param datatype Returns the type of the output. The returned
/// datatype is owned by 'inference_request' and must not be modified
/// or freed by the caller.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestOutputDataType(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const char** datatype);

/// Get the shape of an output tensor.
///
/// \param inference_request The request object.
/// \param name The name of the output.
/// \param shape Return the shape of the output. The returned value is owned by
/// 'inference_request' and must not be modified or freed by the caller.
/// \param dim_count Returns the number of dimensions of the returned shape.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestOutputShape(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const int64_t** shape, uint64_t* dim_count);

/// Get the number of output tensors in the result.
///
/// \param inference_request The request object.
/// \param output_count Returns the number of outputs from a inference request.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestOutputCount(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t* output_count);

/// Get the name of the output tensor at a specific index.
///
/// \param inference_request The request object.
/// \param index The index of the output tensor.
/// \param name The name of the output at 'index'.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestOutputName(
    TRITONSERVER_InferenceRequest* inference_request, uint64_t index,
    const char** name);

/// Get the results data for a named output. The result data is
/// returned as the base pointer to the data and the size, in bytes,
/// of the data. The caller does not own the returned data and must
/// not modify or delete it. The lifetime of the returned data extends
/// until 'inference_request' is deleted or until 'inference_request' is
/// reused in a call to TRITONSERVER_ServerInferAsync.
///
/// \param inference_request The request object.
/// \param name The name of the output.
/// \param base Returns the result data for the named output.
/// \param byte_size Returns the size, in bytes, of the output data.
/// \param memory_type Returns the memory type of the output data.
/// \param memory_type_id Returns the memory type id of the output data.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestOutputData(
    TRITONSERVER_InferenceRequest* inference_request, const char* name,
    const void** base, size_t* byte_size, TRITONSERVER_Memory_Type* memory_type,
    int64_t* memory_type_id);

/// Remove all the output tensors. The meta data of the output tensors will
/// become unaccesible and the result data will be released.
///
/// \param inference_request The request object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestRemoveAllOutputs(
    TRITONSERVER_InferenceRequest* inference_request);

/// TRITONSERVER_ServerOptions
///
/// Options to use when creating an inference server.
///

/// Model control modes
typedef enum tritonserver_modelcontrolmode_enum {
  TRITONSERVER_MODEL_CONTROL_NONE,
  TRITONSERVER_MODEL_CONTROL_POLL,
  TRITONSERVER_MODEL_CONTROL_EXPLICIT
} TRITONSERVER_Model_Control_Mode;

/// Create a new server options object. The caller takes ownership of
/// the TRITONSERVER_ServerOptions object and must call
/// TRITONSERVER_ServerOptionsDelete to release the object.
///
/// \param options Returns the new server options object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerOptionsNew(
    TRITONSERVER_ServerOptions** options);

/// Delete a server options object.
///
/// \param options The server options object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerOptionsDelete(
    TRITONSERVER_ServerOptions* options);

/// Set the textual ID for the server in a server options. The ID is a
/// name that identifies the server.
///
/// \param options The server options object.
/// \param server_id The server identifier.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerOptionsSetServerId(
    TRITONSERVER_ServerOptions* options, const char* server_id);

/// Set the model repository path in a server options. The path must be
/// the full absolute path to the model repository. This function can be called
/// multiple times with different paths to set multiple model repositories.
/// Note that if a model is not unique across all model repositories
/// at any time, the model will not be available.
///
/// \param options The server options object.
/// \param model_repository_path The full path to the model repository.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelRepositoryPath(
    TRITONSERVER_ServerOptions* options, const char* model_repository_path);

/// Set the model control mode in a server options. For each mode the models
/// will be managed as the following:
///
///   TRITONSERVER_MODEL_CONTROL_NONE: the models in model repository will be
///   loaded on startup. After startup any changes to the model repository will
///   be ignored. Calling TRITONSERVER_ServerPollModelRepository will result in
///   an error.
///
///   TRITONSERVER_MODEL_CONTROL_POLL: the models in model repository will be
///   loaded on startup. The model repository can be polled periodically using
///   TRITONSERVER_ServerPollModelRepository and the server will load, unload,
///   and updated models according to changes in the model repository.
///
///   TRITONSERVER_MODEL_CONTROL_EXPLICIT: the models in model repository will
///   not be loaded on startup. The corresponding model control APIs must be
///   called to load / unload a model in the model repository.
///
/// \param options The server options object.
/// \param mode The mode to use for the model control.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetModelControlMode(
    TRITONSERVER_ServerOptions* options, TRITONSERVER_Model_Control_Mode mode);

/// Set the model to be loaded at startup in a server options. The model must be
/// present in one, and only one, of the specified model repositories.
/// This function can be called multiple times with different model name
/// to set multiple startup models.
/// Note that it only takes affect on TRITONSERVER_MODEL_CONTROL_EXPLICIT mode.
///
/// \param options The server options object.
/// \param mode_name The name of the model to load on startup.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStartupModel(
    TRITONSERVER_ServerOptions* options, const char* model_name);

/// Enable or disable strict model configuration handling in a server
/// options.
///
/// \param options The server options object.
/// \param strict True to enable strict model configuration handling,
/// false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStrictModelConfig(
    TRITONSERVER_ServerOptions* options, bool strict);

/// Set the total pinned memory byte size that the server can allocate
/// in a server options. This option will not affect the allocation conducted
/// by the backend frameworks.
///
/// \param options The server options object.
/// \param size The pinned memory pool byte size.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
    TRITONSERVER_ServerOptions* options, uint64_t size);

/// Set the total CUDA memory byte size that the server can allocate on given
/// GPU device in a server options. This option will not affect the allocation
/// conducted by the backend frameworks.
///
/// \param options The server options object.
/// \param gpu_device The GPU device to allocate the memory pool.
/// \param size The CUDA memory pool byte size.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize(
    TRITONSERVER_ServerOptions* options, int gpu_device, uint64_t size);

/// Set the minimum support CUDA compute capability in a server
/// options.
///
/// \param options The server options object.
/// \param cc The minimum CUDA compute capability.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
    TRITONSERVER_ServerOptions* options, double cc);

/// Enable or disable exit-on-error in a server options.
///
/// \param options The server options object.
/// \param exit True to enable exiting on intialization error, false
/// to continue.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetExitOnError(
    TRITONSERVER_ServerOptions* options, bool exit);

/// Enable or disable strict readiness handling in a server options.
///
/// \param options The server options object.
/// \param strict True to enable strict readiness handling, false to
/// disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetStrictReadiness(
    TRITONSERVER_ServerOptions* options, bool strict);

/// Set the exit timeout, in seconds, for the server in a server
/// options.
///
/// \param options The server options object.
/// \param timeout The exit timeout, in seconds.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetExitTimeout(
    TRITONSERVER_ServerOptions* options, unsigned int timeout);

/// Enable or disable info level logging.
///
/// \param options The server options object.
/// \param log True to enable info logging, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerOptionsSetLogInfo(
    TRITONSERVER_ServerOptions* options, bool log);

/// Enable or disable warning level logging.
///
/// \param options The server options object.
/// \param log True to enable warning logging, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerOptionsSetLogWarn(
    TRITONSERVER_ServerOptions* options, bool log);

/// Enable or disable error level logging.
///
/// \param options The server options object.
/// \param log True to enable error logging, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerOptionsSetLogError(
    TRITONSERVER_ServerOptions* options, bool log);

/// Set verbose logging level. Level zero disables verbose logging.
///
/// \param options The server options object.
/// \param level The verbose logging level.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerOptionsSetLogVerbose(
    TRITONSERVER_ServerOptions* options, int level);

/// Enable or disable metrics collection in a server options.
///
/// \param options The server options object.
/// \param metrics True to enable metrics, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerOptionsSetMetrics(
    TRITONSERVER_ServerOptions* options, bool metrics);

/// Enable or disable GPU metrics collection in a server options. GPU
/// metrics are collected if both this option and
/// TRITONSERVER_ServerOptionsSetMetrics are true.
///
/// \param options The server options object.
/// \param gpu_metrics True to enable GPU metrics, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerOptionsSetGpuMetrics(
    TRITONSERVER_ServerOptions* options, bool gpu_metrics);

/// Enable or disable TensorFlow soft-placement of operators.
///
/// \param options The server options object.
/// \param soft_placement True to enable, false to disable.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetTensorFlowSoftPlacement(
    TRITONSERVER_ServerOptions* options, bool soft_placement);

/// Set the fraction of GPU memory dedicated to TensorFlow models on
/// each GPU visible to the inference server. Zero (0) indicates that
/// no memory will be dedicated to TensorFlow and that it will instead
/// allocate memory as needed.
///
/// \param options The server options object.
/// \param fraction The fraction of the GPU memory dedicated to
/// TensorFlow.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsSetTensorFlowGpuMemoryFraction(
    TRITONSERVER_ServerOptions* options, float fraction);

/// Add Tensorflow virtual GPU instances to a physical GPU.
///
/// \param options The server options object.
/// \param gpu_device The physical GPU device id.
/// \param num_vgpus The number of virtual GPUs to create on the
/// physical GPU.
/// \param per_vgpu_memory_mbytes The amount of GPU memory, in
/// megabytes, to dedicate to each virtual GPU instance.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerOptionsAddTensorFlowVgpuMemoryLimits(
    TRITONSERVER_ServerOptions* options, int gpu_device, int num_vgpus,
    uint64_t per_vgpu_memory_mbytes);

/// TRITONSERVER_Server
///
/// An inference server.
///

/// Create a new server object. The caller takes ownership of the
/// TRITONSERVER_Server object and must call TRITONSERVER_ServerDelete
/// to release the object.
///
/// \param server Returns the new inference server object.
/// \param options The inference server options object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerNew(
    TRITONSERVER_Server** server, TRITONSERVER_ServerOptions* options);

/// Delete a server object. If server is not already stopped it is
/// stopped before being deleted.
///
/// \param server The inference server object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerDelete(
    TRITONSERVER_Server* server);

/// Stop a server object. A server can't be restarted once it is
/// stopped.
///
/// \param server The inference server object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerStop(
    TRITONSERVER_Server* server);

/// Check the model repository for changes and update server state
/// based on those changes.
///
/// \param server The inference server object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerPollModelRepository(
    TRITONSERVER_Server* server);

/// Is the server live?
///
/// \param server The inference server object.
/// \param live Returns true if server is live, false otherwise.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerIsLive(
    TRITONSERVER_Server* server, bool* live);

/// Is the server ready?
///
/// \param server The inference server object.
/// \param ready Returns true if server is ready, false otherwise.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerIsReady(
    TRITONSERVER_Server* server, bool* ready);

/// Is the model ready?
///
/// \param server The inference server object.
/// \param model_name The name of the model to get readiness for.
/// \param model_version The version of the model to get readiness for.
/// If nullptr or empty then the server will choose a version based on
/// the model's policy.
/// \param ready Returns true if server is ready, false otherwise.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerModelIsReady(
    TRITONSERVER_Server* server, const char* model_name,
    const char* model_version, bool* ready);

/// Get the metadata of the server as a TRITONSERVER_Message object.
/// The caller takes ownership of the message object and must call
/// TRITONSERVER_MessageDelete to release the object.
///
/// \param server The inference server object.
/// \param server_metadata Returns the server metadata message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerMetadata(
    TRITONSERVER_Server* server, TRITONSERVER_Message** server_metadata);

/// Get the metadata of the model being served as a TRITONSERVER_Message object.
/// The caller takes ownership of the message object and must call
/// TRITONSERVER_MessageDelete to release the object.
///
/// \param server The inference server object.
/// \param model_name The name of the model to get metadata for.
/// \param model_version The version of the model to get metadata for.
/// If nullptr or empty then the server will choose a version based on
/// the model's policy.
/// \param model_metadata Returns the model metadata message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerModelMetadata(
    TRITONSERVER_Server* server, const char* model_name,
    const char* model_version, TRITONSERVER_Message** model_metadata);

/// Get the statistics of the model being served as a TRITONSERVER_Message
/// object. The caller takes ownership of the object and must call
/// TRITONSERVER_MessageDelete to release the object.
///
/// \param server The inference server object.
/// \param model_name The name of the model to get statistics for.
/// \param model_version The version of the model to get statistics for.
/// If nullptr or empty then the server will choose a version based on
/// the model's policy.
/// \param model_stats Returns the model statistics message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerModelStatistics(
    TRITONSERVER_Server* server, const char* model_name,
    const char* model_version, TRITONSERVER_Message** model_stats);

/// Get the configuration of the model being served as a
/// TRITONSERVER_Message object.  The caller takes ownership of the
/// message object and must call TRITONSERVER_MessageDelete to release
/// the object.
///
/// \param server The inference server object.
/// \param model_name The name of the model to get configuration for.
/// \param model_version The version of the model to get configuration for.
/// If nullptr or empty then the server will choose a version based on
/// the model's policy.
/// \param model_config Returns the model config message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerModelConfig(
    TRITONSERVER_Server* server, const char* model_name,
    const char* model_version, TRITONSERVER_Message** model_config);

/// Get the index of all unique models in the model repositories as a
/// TRITONSERVER_Message object. The caller takes ownership of the
/// message object and must call TRITONSERVER_MessageDelete to release
/// the object.
///
/// \param server The inference server object.
/// \param model_index Return the model index message that holds the
/// index of all models contained in the server's model repository(s).
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerModelIndex(
    TRITONSERVER_Server* server, TRITONSERVER_Message** model_index);

/// Load the requested model or reload the model if it is already
/// loaded. The function does not return until the model is loaded or
/// fails to load. Returned error indicates if model loaded
/// successfully or not.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerLoadModel(
    TRITONSERVER_Server* server, const char* model_name);

/// Unload the requested model. Unloading a model that is not loaded
/// on server has no affect and success code will be returned.
/// The function does not return until the model is unloaded or fails to unload.
/// Returned error indicates if model unloaded successfully or not.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerUnloadModel(
    TRITONSERVER_Server* server, const char* model_name);

/// Get the current metrics for the server. The caller takes ownership
/// of the metrics object and must call TRITONSERVER_MetricsDelete to
/// release the object.
///
/// \param server The inference server object.
/// \param metrics Returns the metrics.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerMetrics(
    TRITONSERVER_Server* server, TRITONSERVER_Metrics** metrics);

/// Type for inference completion callback function. If non-nullptr,
/// the 'trace_manager' object is the trace manager associated with
/// the request that is completing. The callback function takes
/// ownership of the TRITONSERVER_TraceManager object and must call
/// TRITONSERVER_TraceManagerDelete to release the object. The callback
/// function takes ownership of the TRITONSERVER_InferenceRequest object
/// and must call TRITONSERVER_InferenceRequestDelete to release the
/// object. The 'userp' data is the same as what is supplied in the
/// call to TRITONSERVER_ServerInferAsync.
typedef void (*TRITONSERVER_InferenceCompleteFn_t)(
    TRITONSERVER_Server* server, TRITONSERVER_TraceManager* trace_manager,
    TRITONSERVER_InferenceRequest* request, void* userp);

/// Perform inference using the meta-data and inputs supplied by the
/// 'inference_request'. The caller releases ownership of
/// 'inference_request' and 'trace_manager' and must not access them
/// in any way after this call, until ownership is returned via the
/// completion function.
///
/// \param server The inference server object.
/// \param trace_manager The trace manager object for this request, or
/// nullptr if no tracing.
/// \param inference_request The request object.
/// \param response_allocator The TRITONSERVER_ResponseAllocator to use
/// to allocate buffers to hold inference results.
/// \param response_allocator_userp User-provided pointer that is
/// delivered to the response allocator's allocation function.
/// \param complete_fn The function called when the inference
/// completes.
/// \param complete_userp User-provided pointer that is delivered to
/// the completion function.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerInferAsync(
    TRITONSERVER_Server* server, TRITONSERVER_TraceManager* trace_manager,
    TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp,
    TRITONSERVER_InferenceCompleteFn_t complete_fn, void* complete_userp);

#ifdef __cplusplus
}
#endif
