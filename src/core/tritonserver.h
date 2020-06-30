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
struct TRITONSERVER_InferenceRequest;
struct TRITONSERVER_InferenceResponse;
struct TRITONSERVER_InferenceTrace;
struct TRITONSERVER_Message;
struct TRITONSERVER_Metrics;
struct TRITONSERVER_ResponseAllocator;
struct TRITONSERVER_Server;
struct TRITONSERVER_ServerOptions;

/// TRITONSERVER_DataType
///
/// Tensor data types recognized by TRITONSERVER.
///
typedef enum TRITONSERVER_datatype_enum {
  TRITONSERVER_TYPE_INVALID,
  TRITONSERVER_TYPE_BOOL,
  TRITONSERVER_TYPE_UINT8,
  TRITONSERVER_TYPE_UINT16,
  TRITONSERVER_TYPE_UINT32,
  TRITONSERVER_TYPE_UINT64,
  TRITONSERVER_TYPE_INT8,
  TRITONSERVER_TYPE_INT16,
  TRITONSERVER_TYPE_INT32,
  TRITONSERVER_TYPE_INT64,
  TRITONSERVER_TYPE_FP16,
  TRITONSERVER_TYPE_FP32,
  TRITONSERVER_TYPE_FP64,
  TRITONSERVER_TYPE_BYTES
} TRITONSERVER_DataType;

/// Get the string representation of a data type. The returned string
/// is not owned by the caller and so should not be modified or freed.
///
/// \param datatype The data type.
/// \return The string representation of the data type.
TRITONSERVER_EXPORT const char* TRITONSERVER_DataTypeString(
    TRITONSERVER_DataType datatype);

/// Get the Triton datatype corresponding to a string representation
/// of a datatype.
///
/// \param dtype The datatype string representation.
/// \return The Triton data type or TRITONSERVER_TYPE_INVALID if the
/// string does not represent a data type.
TRITONSERVER_EXPORT TRITONSERVER_DataType
TRITONSERVER_StringToDataType(const char* dtype);

/// Get the size of a Triton datatype in bytes. Zero is returned for
/// TRITONSERVER_TYPE_BYTES because it have variable size. Zero is
/// returned for TRITONSERVER_TYPE_INVALID.
///
/// \param dtype The datatype.
/// \return The size of the datatype.
TRITONSERVER_EXPORT uint32_t
TRITONSERVER_DataTypeByteSize(TRITONSERVER_DataType datatype);

/// TRITONSERVER_MemoryType
///
/// Types of memory recognized by TRITONSERVER.
///
typedef enum TRITONSERVER_memorytype_enum {
  TRITONSERVER_MEMORY_CPU,
  TRITONSERVER_MEMORY_CPU_PINNED,
  TRITONSERVER_MEMORY_GPU
} TRITONSERVER_MemoryType;

/// Get the string representation of a memory type. The returned
/// string is not owned by the caller and so should not be modified or
/// freed.
///
/// \param memtype The memory type.
/// \return The string representation of the memory type.
TRITONSERVER_EXPORT const char* TRITONSERVER_MemoryTypeString(
    TRITONSERVER_MemoryType memtype);

/// TRITONSERVER_Logging
///
/// Types/levels of logging.
///
typedef enum TRITONSERVER_loglevel_enum {
  TRITONSERVER_LOG_INFO,
  TRITONSERVER_LOG_WARN,
  TRITONSERVER_LOG_ERROR,
  TRITONSERVER_LOG_VERBOSE
} TRITONSERVER_LogLevel;

/// Is a log level enabled?
///
/// \param level The log level.
/// \return True if the log level is enabled, false if not enabled.
TRITONSERVER_EXPORT bool TRITONSERVER_LogIsEnabled(TRITONSERVER_LogLevel level);

/// Log a message at a given log level if that level is enabled.
///
/// \param level The log level.
/// \param filename The file name of the location of the log message.
/// \param line The line number of the log message.
/// \param msg The log message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_LogMessage(
    TRITONSERVER_LogLevel level, const char* filename, const int line,
    const char* msg);

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
/// Object representing a memory allocator for output tensors in an
/// inference response.
///

/// Type for allocation function that allocates a buffer to hold an
/// output tensor.
///
/// \param allocator The allocator that is provided in the call to
/// TRITONSERVER_ServerInferAsync.
/// \param tensor_name The name of the output tensor to allocate for.
/// \param byte_size The size of the buffer to allocate.
/// \param memory_type The type of memory that the caller prefers for
/// the buffer allocation.
/// \param memory_type_id The ID of the memory that the caller prefers
/// for the buffer allocation.
/// \param userp The user data pointer that is provided in the call to
/// TRITONSERVER_ServerInferAsync.
/// \param buffer Returns a pointer to the allocated memory.
/// \param buffer_userp Returns a user-specified value to associate
/// with the buffer, or nullptr if no user-specified value should be
/// associated with the buffer. This value will be provided in the
/// call to TRITONSERVER_ResponseAllocatorReleaseFn_t when the buffer
/// is released and will also be returned by
/// TRITONSERVER_InferenceResponseOutput.
/// \param actual_memory_type Returns the type of memory where the
/// allocation resides. May be different than the type of memory
/// requested by 'memory_type'.
/// \param actual_memory_type_id Returns the ID of the memory where
/// the allocation resides. May be different than the ID of the memory
/// requested by 'memory_type_id'.
/// \return a TRITONSERVER_Error object if a failure occurs while
/// attempting an allocation. If an error is returned all other return
/// values will be ignored.
typedef TRITONSERVER_Error* (*TRITONSERVER_ResponseAllocatorAllocFn_t)(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, void* userp, void** buffer, void** buffer_userp,
    TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id);

/// Type for function that is called when the server no longer holds
/// any reference to a buffer allocated by
/// TRITONSERVER_ResponseAllocatorAllocFn_t. In practice this function
/// is typically called when the response object associated with the
/// buffer is deleted by TRITONSERVER_InferenceResponseDelete.
///
/// \param allocator The allocator that is provided in the call to
/// TRITONSERVER_ServerInferAsync.
/// \param buffer Pointer to the buffer to be freed.
/// \param buffer_userp The user-specified value associated
/// with the buffer in TRITONSERVER_ResponseAllocatorAllocFn_t.
/// \param byte_size The size of the buffer.
/// \param memory_type The type of memory holding the buffer.
/// \param memory_type_id The ID of the memory holding the buffer.
/// \return a TRITONSERVER_Error object if a failure occurs while
/// attempting the release. If an error is returned Triton will not
/// attempt to release the buffer again.
typedef TRITONSERVER_Error* (*TRITONSERVER_ResponseAllocatorReleaseFn_t)(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id);

/// Type for function that is called to indicate that subsequent
/// allocation requests will refer to a new response.
///
/// \param allocator The allocator that is provided in the call to
/// TRITONSERVER_ServerInferAsync.
/// \param userp The user data pointer that is provided in the call to
/// TRITONSERVER_ServerInferAsync.
/// \return a TRITONSERVER_Error object if a failure occurs.
typedef TRITONSERVER_Error* (*TRITONSERVER_ResponseAllocatorStartFn_t)(
    TRITONSERVER_ResponseAllocator* allocator, void* userp);

/// Create a new response allocator object.
///
/// The response allocator object is used by Triton to allocate
/// buffers to hold the output tensors in inference responses. Most
/// models generate a single response for each inference request
/// (TRITONSERVER_TXN_ONE_TO_ONE). For these models the order of
/// callbacks will be:
///
///   TRITONSERVER_ServerInferAsync called
///    - start_fn : optional (and typically not required)
///    - alloc_fn : called once for each output tensor in response
///   TRITONSERVER_InferenceResponseDelete called
///    - release_fn: called once for each output tensor in response
///
/// For models that generate multiple responses for each inference
/// request (TRITONSERVER_TXN_DECOUPLED), the start_fn callback can be
/// used to determine sets of alloc_fn callbacks that belong to the
/// same response:
///
///   TRITONSERVER_ServerInferAsync called
///    - start_fn
///    - alloc_fn : called once for each output tensor in response
///    - start_fn
///    - alloc_fn : called once for each output tensor in response
///      ...
///   For each response, TRITONSERVER_InferenceResponseDelete called
///    - release_fn: called once for each output tensor in the response
///
/// \param allocator Returns the new response allocator object.
/// \param alloc_fn The function to call to allocate buffers for result
/// tensors.
/// \param release_fn The function to call when the server no longer
/// holds a reference to an allocated buffer.
/// \param start_fn The function to call to indicate that the
/// subsequent 'alloc_fn' calls are for a new response. This callback
/// is optional (use nullptr to indicate that it should not be
/// invoked).

/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ResponseAllocatorNew(
    TRITONSERVER_ResponseAllocator** allocator,
    TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
    TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn,
    TRITONSERVER_ResponseAllocatorStartFn_t start_fn);

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
} TRITONSERVER_MetricFormat;

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
    TRITONSERVER_Metrics* metrics, TRITONSERVER_MetricFormat format,
    const char** base, size_t* byte_size);

/// TRITONSERVER_InferenceTrace
///
/// Object that represents tracing for an inference request.
///

/// Trace levels
typedef enum tritonserver_tracelevel_enum {
  TRITONSERVER_TRACE_LEVEL_DISABLED,
  TRITONSERVER_TRACE_LEVEL_MIN,
  TRITONSERVER_TRACE_LEVEL_MAX
} TRITONSERVER_InferenceTraceLevel;

/// Get the string representation of a trace level. The returned
/// string is not owned by the caller and so should not be modified or
/// freed.
///
/// \param level The trace level.
/// \return The string representation of the trace level.
TRITONSERVER_EXPORT const char* TRITONSERVER_InferenceTraceLevelString(
    TRITONSERVER_InferenceTraceLevel level);

// Trace activities
typedef enum tritonserver_traceactivity_enum {
  TRITONSERVER_TRACE_REQUEST_START = 0,
  TRITONSERVER_TRACE_QUEUE_START = 1,
  TRITONSERVER_TRACE_COMPUTE_START = 2,
  TRITONSERVER_TRACE_COMPUTE_INPUT_END = 3,
  TRITONSERVER_TRACE_COMPUTE_OUTPUT_START = 4,
  TRITONSERVER_TRACE_COMPUTE_END = 5,
  TRITONSERVER_TRACE_REQUEST_END = 6
} TRITONSERVER_InferenceTraceActivity;

/// Get the string representation of a trace activity. The returned
/// string is not owned by the caller and so should not be modified or
/// freed.
///
/// \param activity The trace activity.
/// \return The string representation of the trace activity.
TRITONSERVER_EXPORT const char* TRITONSERVER_InferenceTraceActivityString(
    TRITONSERVER_InferenceTraceActivity activity);

/// Type for trace activity callback function. This callback function
/// is used to report activity occurring for a trace. This function
/// does not take ownership of 'trace' and so any information needed
/// from that object must be copied before returning. The 'userp' data
/// is the same as what is supplied in the call to
/// TRITONSERVER_InferenceTraceNew.
typedef void (*TRITONSERVER_InferenceTraceActivityFn_t)(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns,
    void* userp);

/// Type for trace release callback function. This callback function
/// is called when all activity for the trace has completed. The
/// callback function takes ownership of the
/// TRITONSERVER_InferenceTrace object. The 'userp' data is the same
/// as what is supplied in the call to TRITONSERVER_InferenceTraceNew.
typedef void (*TRITONSERVER_InferenceTraceReleaseFn_t)(
    TRITONSERVER_InferenceTrace* trace, void* userp);

/// Create a new inference trace object. The caller takes ownership of
/// the TRITONSERVER_InferenceTrace object and must call
/// TRITONSERVER_TraceDelete to release the object.
///
/// The activity callback function will be called to report activity
/// for 'trace' as well as for any child traces that are spawned by
/// 'trace', and so the activity callback must check the trace object
/// to determine specifically what activity is being reported.
///
/// The release callback is called for both 'trace' and for any child
/// traces spawned by 'trace'.
///
/// \param trace Returns the new infernece trace object.
/// \param level The tracing level.
/// \param parent_id The parent trace id for this trace. A value of 0
/// indicates that there is not parent trace.
/// \param activity_fn The callback function where activity for the
/// trace is reported.
/// \param release_fn The callback function called when all activity
/// is complete for the trace.
/// \param trace_userp User-provided pointer that is delivered to
/// the activity and release callback functions.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceTraceNew(
    TRITONSERVER_InferenceTrace** trace, TRITONSERVER_InferenceTraceLevel level,
    uint64_t parent_id, TRITONSERVER_InferenceTraceActivityFn_t activity_fn,
    TRITONSERVER_InferenceTraceReleaseFn_t release_fn, void* trace_userp);

/// Delete a trace object.
///
/// \param trace The trace object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceTraceDelete(
    TRITONSERVER_InferenceTrace* trace);

/// Get the id associated with a trace. Every trace is assigned an id
/// that is unique across all traces created for a Triton server.
///
/// \param trace The trace.
/// \param id Returns the id associated with the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceTraceId(
    TRITONSERVER_InferenceTrace* trace, uint64_t* id);

/// Get the parent id associated with a trace. The parent id indicates
/// a parent-child relationship between two traces. A parent id value
/// of 0 indicates that there is no parent trace.
///
/// \param trace The trace.
/// \param id Returns the parent id associated with the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceTraceParentId(
    TRITONSERVER_InferenceTrace* trace, uint64_t* parent_id);

/// Get the name of the model associated with a trace. The caller does
/// not own the returned string and must not modify or delete it. The
/// lifetime of the returned string extends only as long as 'trace'.
///
/// \param trace The trace.
/// \param model_name Returns the name of the model associated with
/// the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceTraceModelName(
    TRITONSERVER_InferenceTrace* trace, const char** model_name);

/// Get the version of the model associated with a trace.
///
/// \param trace The trace.
/// \param model_version Returns the version of the model associated
/// with the trace.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceTraceModelVersion(
    TRITONSERVER_InferenceTrace* trace, int64_t* model_version);

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
  TRITONSERVER_REQUEST_FLAG_SEQUENCE_START = 1,
  TRITONSERVER_REQUEST_FLAG_SEQUENCE_END = 2
} TRITONSERVER_RequestFlag;

/// Inference request release flags. The enum values must be
/// power-of-2 values.
typedef enum tritonserver_requestreleaseflag_enum {
  TRITONSERVER_REQUEST_RELEASE_ALL = 1
} TRITONSERVER_RequestReleaseFlag;

/// Inference response complete flags. The enum values must be
/// power-of-2 values.
typedef enum tritonserver_responsecompleteflag_enum {
  TRITONSERVER_RESPONSE_COMPLETE_FINAL = 1
} TRITONSERVER_ResponseCompleteFlag;

/// Type for inference request release callback function. The callback
/// indicates what type of release is being performed on the request
/// and for some of these the callback function takes ownership of the
/// TRITONSERVER_InferenceRequest object. The 'userp' data is the same
/// as what is supplied in the call to TRITONSERVER_ServerInferAsync.
///
/// One or more flags will be specified when the callback is invoked,
/// and the callback must take the following actions:
///
///   - TRITONSERVER_REQUEST_RELEASE_ALL: The entire inference request
///     is being released and ownership is passed to the callback
///     function. Triton will not longer access the 'request' object
///     itself nor any input tensor data associated with the
///     request. The callback should free or otherwise manage the
///     'request' object and all associated tensor data.
///
/// Note that currently TRITONSERVER_REQUEST_RELEASE_ALL should always
/// be set when the callback is invoked but in the future that may
/// change, so the callback should explicitly check for the flag
/// before taking ownership of the request object.
///
typedef void (*TRITONSERVER_InferenceRequestReleaseFn_t)(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp);

/// Type for callback function indicating that an inference response
/// has completed. The callback function takes ownership of the
/// TRITONSERVER_InferenceResponse object. The 'userp' data is the
/// same as what is supplied in the call to
/// TRITONSERVER_ServerInferAsync.
///
/// One or more flags may be specified when the callback is invoked:
///
///   - TRITONSERVER_RESPONSE_COMPLETE_FINAL: Indicates that no more
///     responses will be generated for a given request (more
///     specifically, that no more responses will be generated for the
///     inference request that set this callback and 'userp'). When
///     this flag is set 'response' may be a response object or may be
///     nullptr. If 'response' is not nullptr, then 'response' is the
///     last response that Triton will produce for the request. If
///     'response' is nullptr then Triton is indicating that no more
///     responses will be produced for the request.
typedef void (*TRITONSERVER_InferenceResponseCompleteFn_t)(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags,
    void* userp);

/// Create a new inference request object.
///
/// \param inference_request Returns the new request object.
/// \param server the inference server object.
/// \param model_name The name of the model to use for the request.
/// \param model_version The version of the model to use for the
/// request. If -1 then the server will choose a version based on the
/// model's policy.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestNew(
    TRITONSERVER_InferenceRequest** inference_request,
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version);

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
/// a bitwise-or of all flag values, see TRITONSERVER_RequestFlag for
/// available flags.
///
/// \param inference_request The request object.
/// \param flags Returns the flags.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceRequestFlags(
    TRITONSERVER_InferenceRequest* inference_request, uint32_t* flags);

/// Set the flag(s) associated with a request. 'flags' should holds a
/// bitwise-or of all flag values, see TRITONSERVER_RequestFlag for
/// available flags.
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
    const TRITONSERVER_DataType datatype, const int64_t* shape,
    uint64_t dim_count);

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
    const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
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

/// Set the release callback for an inference request. The release
/// callback is called by Triton to return ownership of the request
/// object.
///
/// \param inference_request The request object.
/// \param request_release_fn The function called to return ownership
/// of the 'inference_request' object.
/// \param request_release_userp User-provided pointer that is
/// delivered to the 'request_release_fn' callback.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetReleaseCallback(
    TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_InferenceRequestReleaseFn_t request_release_fn,
    void* request_release_userp);

/// Set the allocator and response callback for an inference
/// request. The allocator is used to allocate buffers for any output
/// tensors included in responses that are produced for this
/// request. The response callback is called to return response
/// objects representing responses produced for this request.
///
/// \param inference_request The request object.
/// \param response_allocator The TRITONSERVER_ResponseAllocator to use
/// to allocate buffers to hold inference results.
/// \param response_allocator_userp User-provided pointer that is
/// delivered to the response allocator's allocation function.
/// \param response_fn The function called to deliver an inference
/// response for this request.
/// \param response_userp User-provided pointer that is delivered to
/// the 'response_fn' callback.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceRequestSetResponseCallback(
    TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp,
    TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
    void* response_userp);

/// TRITONSERVER_InferenceResponse
///
/// Object representing an inference response. The inference response
/// provides the meta-data and output tensor values calculated by the
/// inference.
///

/// Delete an inference response object.
///
/// \param inference_response The response object.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceResponseDelete(
    TRITONSERVER_InferenceResponse* inference_response);

/// Return the error status of an inference response. Return a
/// TRITONSERVER_Error object on failure, return nullptr on success.
/// The returned error object is owned by 'inference_response' and so
/// should not be deleted by the caller.
///
/// \param inference_response The response object.
/// \return a TRITONSERVER_Error indicating the success or failure
/// status of the response.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceResponseError(
    TRITONSERVER_InferenceResponse* inference_response);

/// Get model used to produce a response. The caller does not own the
/// returned model name value and must not modify or delete it. The
/// lifetime of all returned values extends until 'inference_response'
/// is deleted.
///
/// \param inference_response The response object.
/// \param model_name Returns the name of the model.
/// \param model_version Returns the version of the model.
/// this response.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceResponseModel(
    TRITONSERVER_InferenceResponse* inference_response, const char** model_name,
    int64_t* model_version);

/// Get the ID of the request corresponding to a response. The caller
/// does not own the returned ID and must not modify or delete it. The
/// lifetime of all returned values extends until 'inference_response'
/// is deleted.
///
/// \param inference_response The response object.
/// \param request_id Returns the ID of the request corresponding to
/// this response.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceResponseId(
    TRITONSERVER_InferenceResponse* inference_response,
    const char** request_id);

/// Get the number of outputs available in the response.
///
/// \param inference_response The response object.
/// \param count Returns the number of output tensors.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceResponseOutputCount(
    TRITONSERVER_InferenceResponse* inference_response, uint32_t* count);

/// Get all information about an output tensor.  The tensor data is
/// returned as the base pointer to the data and the size, in bytes,
/// of the data. The caller does not own any of the returned value and
/// must not modify or delete them. The lifetime of all returned
/// values extends until 'inference_response' is deleted.
///
/// \param inference_response The response object.
/// \param index The index of the output tensor, must be 0 <= index <
/// count, where 'count' is the value returned by
/// TRITONSERVER_InferenceResponseOutputCount.
/// \param name Returns the name of the output.
/// \param datatype Returns the type of the output.
/// \param shape Returns the shape of the output.
/// \param dim_count Returns the number of dimensions of the returned
/// shape.
/// \param base Returns the tensor data for the output.
/// \param byte_size Returns the size, in bytes, of the data.
/// \param memory_type Returns the memory type of the data.
/// \param memory_type_id Returns the memory type id of the data.
/// \param userp The user-specified value associated with the buffer
/// in TRITONSERVER_ResponseAllocatorAllocFn_t.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_InferenceResponseOutput(
    TRITONSERVER_InferenceResponse* inference_response, const uint32_t index,
    const char** name, TRITONSERVER_DataType* datatype, const int64_t** shape,
    uint64_t* dim_count, const void** base, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id,
    void** userp);

/// Get a classification label associated with an output for a given
/// index.  The caller does not own the returned label and must not
/// modify or delete ot. The lifetime of all returned label extends
/// until 'inference_response' is deleted.
///
/// \param inference_response The response object.
/// \param index The index of the output tensor, must be 0 <= index <
/// count, where 'count' is the value returned by
/// TRITONSERVER_InferenceResponseOutputCount.
/// \param class_index The index of the class.
/// \param name Returns the label corresponding to 'class_index' or
/// nullptr if no label.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_InferenceResponseOutputClassificationLabel(
    TRITONSERVER_InferenceResponse* inference_response, const uint32_t index,
    const size_t class_index, const char** label);


/// TRITONSERVER_ServerOptions
///
/// Options to use when creating an inference server.
///

/// Model control modes
typedef enum tritonserver_modelcontrolmode_enum {
  TRITONSERVER_MODEL_CONTROL_NONE,
  TRITONSERVER_MODEL_CONTROL_POLL,
  TRITONSERVER_MODEL_CONTROL_EXPLICIT
} TRITONSERVER_ModelControlMode;

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
    TRITONSERVER_ServerOptions* options, TRITONSERVER_ModelControlMode mode);

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

/// Model batch flags. The enum values must be power-of-2 values.
typedef enum tritonserver_batchflag_enum {
  TRITONSERVER_BATCH_UNKNOWN = 1,
  TRITONSERVER_BATCH_FIRST_DIM = 2
} TRITONSERVER_ModelBatchFlag;

/// Model index flags. The enum values must be power-of-2 values.
typedef enum tritonserver_modelindexflag_enum {
  TRITONSERVER_INDEX_FLAG_NONE = 0,
  TRITONSERVER_INDEX_FLAG_READY = 1
} TRITONSERVER_ModelIndexFlag;

/// Model transaction policy flags. The enum values must be
/// power-of-2 values.
typedef enum tritonserver_txn_property_flag_enum {
  TRITONSERVER_TXN_ONE_TO_ONE = 0,
  TRITONSERVER_TXN_DECOUPLED = 1
} TRITONSERVER_ModelTxnPropertyFlag;

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
/// \param model_version The version of the model to get readiness
/// for.  If -1 then the server will choose a version based on the
/// model's policy.
/// \param ready Returns true if server is ready, false otherwise.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerModelIsReady(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, bool* ready);

/// Get the batch properties of the model. The properties are
/// communicated by a flags value and an (optional) object returned by
/// 'voidp'.
///
///   - TRITONSERVER_BATCH_UNKNOWN: Triton cannot determine the
///     batching properties of the model. This means that the model
///     does not support batching in any way that is useable by
///     Triton. The returned 'voidp' value is nullptr.
///
///   - TRITONSERVER_BATCH_FIRST_DIM: The model supports batching
///     along the first dimension of every input and output
///     tensor. Triton schedulers that perform batching can
///     automatically batch inference requests along this dimension.
///     The returned 'voidp' value is nullptr.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \param model_version The version of the model.  If -1 then the
/// server will choose a version based on the model's policy.
/// \param flags Returns flags indicating the batch properties of the
/// model.
/// \param voidp If non-nullptr, returns a point specific to the 'flags' value.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerModelBatchProperties(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, uint32_t* flags, void** voidp);

/// Get the transaction policy of the model. The policy is communicated
/// by a flags value.
///
///   - TRITONSERVER_TXN_ONE_TO_ONE: The model generates exactly
///     one response per request.
///
///   - TRITONSERVER_TXN_DECOUPLED: The model may generate zero
///     to many responses per request.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \param model_version The version of the model.  If -1 then the
/// server will choose a version based on the model's policy.
/// \param txn_flags Returns flags indicating the transaction policy of the
/// model.
/// \param voidp If non-nullptr, returns a point specific to the 'flags' value.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error*
TRITONSERVER_ServerModelTransactionProperties(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, uint32_t* txn_flags, void** voidp);

/// Get the metadata of the server as a TRITONSERVER_Message object.
/// The caller takes ownership of the message object and must call
/// TRITONSERVER_MessageDelete to release the object.
///
/// \param server The inference server object.
/// \param server_metadata Returns the server metadata message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerMetadata(
    TRITONSERVER_Server* server, TRITONSERVER_Message** server_metadata);

/// Get the metadata of a model as a TRITONSERVER_Message
/// object.  The caller takes ownership of the message object and must
/// call TRITONSERVER_MessageDelete to release the object.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \param model_version The version of the model.
/// If -1 then the server will choose a version based on the model's
/// policy.
/// \param model_metadata Returns the model metadata message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerModelMetadata(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, TRITONSERVER_Message** model_metadata);

/// Get the statistics of a model as a TRITONSERVER_Message
/// object. The caller takes ownership of the object and must call
/// TRITONSERVER_MessageDelete to release the object.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// If empty, then statistics for all available models will be returned,
/// and the server will choose a version based on those models' policies.
/// \param model_version The version of the model.  If -1 then the
/// server will choose a version based on the model's policy.
/// \param model_stats Returns the model statistics message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerModelStatistics(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, TRITONSERVER_Message** model_stats);

/// Get the configuration of a model as a TRITONSERVER_Message object.
/// The caller takes ownership of the message object and must call
/// TRITONSERVER_MessageDelete to release the object.
///
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \param model_version The version of the model.  If -1 then the
/// server will choose a version based on the model's policy.
/// \param config_version The model configuration will be returned in
/// a format matching this version. If the configuration cannot be
/// represented in the requested version's format then an error will
/// be returned. Currently only version 1 is supported.
/// \param model_config Returns the model config message.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerModelConfig(
    TRITONSERVER_Server* server, const char* model_name,
    const int64_t model_version, const uint32_t config_version,
    TRITONSERVER_Message** model_config);

/// Get the index of all unique models in the model repositories as a
/// TRITONSERVER_Message object. The caller takes ownership of the
/// message object and must call TRITONSERVER_MessageDelete to release
/// the object.
///
/// \param server The inference server object.
/// \param flags TRITONSERVER_ModelIndexFlag flags that control how to
/// collect the index.
/// \param model_index Return the model index message that holds the
/// index of all models contained in the server's model repository(s).
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerModelIndex(
    TRITONSERVER_Server* server, uint32_t flags,
    TRITONSERVER_Message** model_index);

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

/// Perform inference using the meta-data and inputs supplied by the
/// 'inference_request'. If the function returns success, then the
/// caller releases ownership of 'inference_request' and must not
/// access it in any way after this call, until ownership is returned
/// via the 'request_release_fn' callback registered in the request
/// object with TRITONSERVER_InferenceRequestSetReleaseCallback.
///
/// The function unconditionally takes ownership of 'trace' and so the
/// caller must not access it in any way after this call (except in
/// the trace id callback) until ownership is returned via the trace's
/// release_fn callback.
///
/// Responses produced for this request are returned using the
/// allocator and callback register with the request by
/// TRITONSERVER_InferenceRequestSetResponseCallback.
///
/// \param server The inference server object.
/// \param inference_request The request object.
/// \param trace The trace object for this request, or nullptr if no
/// tracing.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_EXPORT TRITONSERVER_Error* TRITONSERVER_ServerInferAsync(
    TRITONSERVER_Server* server,
    TRITONSERVER_InferenceRequest* inference_request,
    TRITONSERVER_InferenceTrace* trace);


#ifdef __cplusplus
}
#endif
