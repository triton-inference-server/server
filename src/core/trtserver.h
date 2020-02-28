// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#define TRTSERVER_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define TRTSERVER_EXPORT __attribute__((__visibility__("default")))
#else
#define TRTSERVER_EXPORT
#endif

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#else
typedef void cudaIpcMemHandle_t;
#endif  // TRTIS_ENABLE_GPU

struct TRTSERVER_Error;
struct TRTSERVER_InferenceRequestOptions;
struct TRTSERVER_InferenceRequestProvider;
struct TRTSERVER_InferenceResponse;
struct TRTSERVER_Metrics;
struct TRTSERVER_Protobuf;
struct TRTSERVER_ResponseAllocator;
struct TRTSERVER_Server;
struct TRTSERVER_ServerOptions;
struct TRTSERVER_SharedMemoryBlock;
struct TRTSERVER_Trace;
struct TRTSERVER_TraceManager;

/// Types of memory recognized by TRTSERVER.
typedef enum trtserver_memorytype_enum {
  TRTSERVER_MEMORY_CPU,
  TRTSERVER_MEMORY_GPU,
  TRTSERVER_MEMORY_CPU_PINNED
} TRTSERVER_Memory_Type;

/// TRTSERVER_Error
///
/// Errors are reported by a TRTSERVER_Error object. A NULL
/// TRTSERVER_Error indicates no error, a non-NULL TRTSERVER_Error
/// indicates error and the code and message for the error can be
/// retrieved from the object.
///
/// The caller takes ownership of a TRTSERVER_Error object returned by
/// the API and must call TRTSERVER_ErrorDelete to release the object.
///

/// The TRTSERVER_Error error codes
typedef enum trtserver_errorcode_enum {
  TRTSERVER_ERROR_UNKNOWN,
  TRTSERVER_ERROR_INTERNAL,
  TRTSERVER_ERROR_NOT_FOUND,
  TRTSERVER_ERROR_INVALID_ARG,
  TRTSERVER_ERROR_UNAVAILABLE,
  TRTSERVER_ERROR_UNSUPPORTED,
  TRTSERVER_ERROR_ALREADY_EXISTS
} TRTSERVER_Error_Code;

/// Create a new error object. The caller takes ownership of the
/// TRTSERVER_Error object and must call TRTSERVER_ErrorDelete to
/// release the object.
/// \param code The error code.
/// \param msg The error message.
/// \return A new TRTSERVER_Error object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ErrorNew(
    TRTSERVER_Error_Code code, const char* msg);

/// Delete an error object.
/// \param error The error object.
TRTSERVER_EXPORT void TRTSERVER_ErrorDelete(TRTSERVER_Error* error);

/// Get the error code.
/// \param error The error object.
/// \return The error code.
TRTSERVER_EXPORT TRTSERVER_Error_Code
TRTSERVER_ErrorCode(TRTSERVER_Error* error);

/// Get the string representation of an error code. The returned
/// string is not owned by the caller and so should not be modified or
/// freed. The lifetime of the returned string extends only as long as
/// 'error' and must not be accessed once 'error' is deleted.
/// \param error The error object.
/// \return The string representation of the error code.
TRTSERVER_EXPORT const char* TRTSERVER_ErrorCodeString(TRTSERVER_Error* error);

/// Get the error message. The returned string is not owned by the
/// caller and so should not be modified or freed. The lifetime of the
/// returned string extends only as long as 'error' and must not be
/// accessed once 'error' is deleted.
/// \param error The error object.
/// \return The error message.
TRTSERVER_EXPORT const char* TRTSERVER_ErrorMessage(TRTSERVER_Error* error);

/// TRTSERVER_SharedMemoryBlock
///
/// Object representing a reference to a contiguous block of shared
/// memory. The TRTSERVER_SharedMemoryBlock object does not create or
/// manage the lifetime of the shared-memory block, it simply
/// maintains a reference into the block.
///

/// Create a new shared memory block object referencing a system shared
/// memory block residing in TRTSERVER_MEMORY_CPU type memory.
/// \param shared_memory_block Returns the new shared memory block object.
/// \param name A unique name for the shared memory block. This name
/// is used in inference requests to refer to this shared memory
/// block.
/// \param shm_key The name of the posix shared memory object containing
/// the block of memory.
/// \param offset The offset within the system shared memory object to the
/// start of the block.
/// \param byte_size The size, in bytes of the block.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_SharedMemoryBlockCpuNew(
    TRTSERVER_SharedMemoryBlock** shared_memory_block, const char* name,
    const char* shm_key, const size_t offset, const size_t byte_size);

/// Create a new shared memory block object referencing a CUDA shared
/// memory block residing in TRTSERVER_MEMORY_GPU type memory.
/// \param shared_memory_block Returns the new shared memory block object.
/// \param name A unique name for the shared memory block. This name
/// is used in inference requests to refer to this shared memory
/// block.
/// \param cuda_shm_handle The CUDA IPC handle.
/// \param byte_size The size, in bytes of the block.
/// \param device_id The GPU number the CUDA shared memory region is in.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_SharedMemoryBlockGpuNew(
    TRTSERVER_SharedMemoryBlock** shared_memory_block, const char* name,
    const cudaIpcMemHandle_t* cuda_shm_handle, const size_t byte_size,
    const int device_id);

/// Delete a shared memory block object.
/// \param shared_memory_block The object to delete.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_SharedMemoryBlockDelete(
    TRTSERVER_SharedMemoryBlock* shared_memory_block);

/// Get the memory type of a shared memory block object.
/// \param shared_memory_block The object whose memory type is required.
/// \param memory_type Returns the memory type of the shared memory block.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_Error* TRTSERVER_SharedMemoryBlockMemoryType(
    TRTSERVER_SharedMemoryBlock* shared_memory_block,
    TRTSERVER_Memory_Type* memory_type);

/// Get the memory type id of a shared memory block object.
/// \param shared_memory_block The object whose memory type is required.
/// \param memory_type_id The device ID if the region is in GPU shared memory.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_Error* TRTSERVER_SharedMemoryBlockMemoryTypeId(
    TRTSERVER_SharedMemoryBlock* shared_memory_block, int64_t* memory_type_id);

/// TRTSERVER_ResponseAllocator
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
/// as what is supplied in the call to TRTSERVER_ServerInferAsync.
///
/// Return in 'buffer_userp' a user-specified value to associate with
/// the buffer. This value will be provided in the call to
/// TRTSERVER_ResponseAllocatorReleaseFn_t.
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
/// The function should return a TRTSERVER_Error object if a failure
/// occurs while attempting an allocation. If an error object is
/// returned for a result tensor, the inference server will assume allocation
/// is not possible for the result buffer and will abort the inference
/// request.
typedef TRTSERVER_Error* (*TRTSERVER_ResponseAllocatorAllocFn_t)(
    TRTSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRTSERVER_Memory_Type memory_type, int64_t memory_type_id,
    void* userp, void** buffer, void** buffer_userp,
    TRTSERVER_Memory_Type* actual_memory_type, int64_t* actual_memory_type_id);

/// Type for function that is called when the server no longer holds
/// any reference to a buffer allocated by
/// TRTSERVER_ResponseAllocatorAllocFn_t. In practice this function is
/// called when the response object associated with the buffer is
/// deleted by TRTSERVER_InferenceResponseDelete.
///
/// The 'buffer' and 'buffer_userp' arguments equal those returned by
/// TRTSERVER_ResponseAllocatorAllocFn_t and 'byte_size',
/// 'memory_type' and 'memory_type_id' equal the values passed to
/// TRTSERVER_ResponseAllocatorAllocFn_t.
///
/// Return a TRTSERVER_Error object on failure, return nullptr on
/// success.
typedef TRTSERVER_Error* (*TRTSERVER_ResponseAllocatorReleaseFn_t)(
    TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRTSERVER_Memory_Type memory_type,
    int64_t memory_type_id);

/// Create a new response allocator object.
/// \param allocator Returns the new response allocator object.
/// \param alloc_fn The function to call to allocate buffers for result
/// tensors.
/// \param release_fn The function to call when the server no longer
/// holds a reference to an allocated buffer.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ResponseAllocatorNew(
    TRTSERVER_ResponseAllocator** allocator,
    TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn,
    TRTSERVER_ResponseAllocatorReleaseFn_t release_fn);

/// Delete a response allocator.
/// \param allocator The response allocator object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ResponseAllocatorDelete(
    TRTSERVER_ResponseAllocator* allocator);

/// TRTSERVER_Protobuf
///
/// Object representing a protobuf.
///

/// Delete a protobuf object.
/// \param protobuf The protobuf object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ProtobufDelete(
    TRTSERVER_Protobuf* protobuf);

/// Get the base and size of the buffer containing the serialized
/// version of the protobuf. The buffer is owned by the
/// TRTSERVER_Protobuf object and should not be modified or freed by
/// the caller. The lifetime of the buffer extends only as long as
/// 'protobuf' and must not be accessed once 'protobuf' is deleted.
/// \param protobuf The protobuf object.
/// \param base Returns the base of the serialized protobuf.
/// \param byte_size Returns the size, in bytes, of the serialized
/// protobuf.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ProtobufSerialize(
    TRTSERVER_Protobuf* protobuf, const char** base, size_t* byte_size);

/// TRTSERVER_Metrics
///
/// Object representing metrics.
///

/// Metric format types
typedef enum trtserver_metricformat_enum {
  TRTSERVER_METRIC_PROMETHEUS
} TRTSERVER_Metric_Format;

/// Delete a metrics object.
/// \param metrics The metrics object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_MetricsDelete(
    TRTSERVER_Metrics* metrics);

/// Get a buffer containing the metrics in the specified format. For
/// each format the buffer contains the following:
///
///   TRTSERVER_METRIC_PROMETHEUS: 'base' points to a single multiline
///   string (char*) that gives a text representation of the metrics in
///   prometheus format. 'byte_size' returns the length of the string
///   in bytes.
///
/// The buffer is owned by the 'metrics' object and should not be
/// modified or freed by the caller. The lifetime of the buffer
/// extends only as long as 'metrics' and must not be accessed once
/// 'metrics' is deleted.
/// \param metrics The metrics object.
/// \param format The format to use for the returned metrics.
/// \param base Returns a pointer to the base of the formatted
/// metrics, as described above.
/// \param byte_size Returns the size, in bytes, of the formatted
/// metrics.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_MetricsFormatted(
    TRTSERVER_Metrics* metrics, TRTSERVER_Metric_Format format,
    const char** base, size_t* byte_size);

/// TRTSERVER_Trace
///
/// Object that represents tracing for a request.
///

/// Trace levels
typedef enum trtserver_tracelevel_enum {
  TRTSERVER_TRACE_LEVEL_DISABLED,
  TRTSERVER_TRACE_LEVEL_MIN,
  TRTSERVER_TRACE_LEVEL_MAX
} TRTSERVER_Trace_Level;

// Trace activities
typedef enum trtserver_traceactivity_enum {
  TRTSERVER_TRACE_REQUEST_START,
  TRTSERVER_TRACE_QUEUE_START,
  TRTSERVER_TRACE_COMPUTE_START,
  TRTSERVER_TRACE_COMPUTE_INPUT_END,
  TRTSERVER_TRACE_COMPUTE_OUTPUT_START,
  TRTSERVER_TRACE_COMPUTE_END,
  TRTSERVER_TRACE_REQUEST_END
} TRTSERVER_Trace_Activity;

/// Type for trace activity callback function. This callback function
/// is used to report activity occurring during a traced request. The
/// 'userp' data is the same as what is supplied in the call to
/// TRTSERVER_TraceNew.
typedef void (*TRTSERVER_TraceActivityFn_t)(
    TRTSERVER_Trace* trace, TRTSERVER_Trace_Activity activity,
    uint64_t timestamp_ns, void* userp);

/// Create a new trace object. The caller takes ownership of the
/// TRTSERVER_Trace object and must call TRTSERVER_TraceDelete to
/// release the object.
/// \param trace Returns the new trace object.
/// \param level The tracing level.
/// \param activity_fn The callback function where activity for the
/// trace is reported.
/// \param activity_userp User-provided pointer that is delivered to
/// the activity function.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_TraceNew(
    TRTSERVER_Trace** trace, TRTSERVER_Trace_Level level,
    TRTSERVER_TraceActivityFn_t activity_fn, void* activity_userp);

/// Delete a trace object.
/// \param trace The trace object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_TraceDelete(TRTSERVER_Trace* trace);

/// Get the name of the model being traced. The caller
/// does not own the returned string and must not modify or delete
/// it. The lifetime of the returned string extends only as long as
/// 'trace' and must not be accessed once 'trace' is deleted.
/// This method is only guaranteed to correctly return the 'model_name' in the
/// invocation of TRTSERVER_TraceManagerReleaseTraceFn_t
/// \param trace The trace object.
/// \param model_name Returns the name of the model being traced.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_TraceModelName(
    TRTSERVER_Trace* trace, const char** model_name);

/// Get the version of the model being traced.
/// This method is only guaranteed to correctly return the 'model_version' in
/// the invocation of TRTSERVER_TraceManagerReleaseTraceFn_t
/// \param trace The trace object.
/// \param model_version Returns the version of the model being traced.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_TraceModelVersion(
    TRTSERVER_Trace* trace, int64_t* model_version);

/// Get the id of the trace object. Each trace object created during execution
/// of the inference server has a unique id.
/// \param trace The trace object.
/// \param id Returns the id of the trace object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_TraceId(
    TRTSERVER_Trace* trace, int64_t* id);

/// Get the parent id of the trace object. The parent id will be set if the
/// trace object is created from within another traced request, and the parent
/// id will be set to the id of the trace object associated with that
/// traced request.
/// This method is only guaranteed to correctly return the 'parent_id' in
/// the invocation of TRTSERVER_TraceManagerReleaseTraceFn_t
/// \param trace The trace object.
/// \param parent_id Returns the parent id of the trace object. -1 indicates
/// that the trace object has no parent.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_TraceParentId(
    TRTSERVER_Trace* trace, int64_t* parent_id);

/// TRTSERVER_TraceManager
///
/// Object representing a manager for initiating traces.
///

/// Type for trace creation callback function. This callback function
/// is used when a model execution is initiated within the request, if the
/// request is to be traced. The user should call TRTSERVER_TraceNew and return
/// the new trace object if the user decides to trace the model execution.
/// Otherwise, the user should set 'trace' == nullptr.
/// The 'userp' data is the same as 'userp' supplied in the call to
/// TRTSERVER_TraceManagerNew.
typedef void (*TRTSERVER_TraceManagerCreateTraceFn_t)(
    TRTSERVER_Trace** trace, const char* model_name, int64_t version,
    void* userp);

/// Type for trace release callback function. This callback function
/// is invoked when the model execution being traced is completed. By this
/// point, it is the user's responsiblity to delete 'trace' object created from
/// TRTSERVER_TraceManagerCreateTraceFn_t by calling TRTSERVER_TraceDelete.
/// The 'activity_userp' data is the same as 'activity_userp' supplied in the
/// call to TRTSERVER_TraceNew.
/// The 'userp' data is the same as 'userp' supplied in the call to
/// TRTSERVER_TraceManagerNew.
typedef void (*TRTSERVER_TraceManagerReleaseTraceFn_t)(
    TRTSERVER_Trace* trace, void* activity_userp, void* userp);

/// Create a new trace manager object.
/// \param trace_manager Returns the new trace manager object.
/// \param create_fn The function to call to create trace object for a request.
/// \param release_fn The function to call when the request associated with a
/// trace object is complete.
/// \param userp User-provided pointer that is delivered to the trace
/// creation and release function.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_TraceManagerNew(
    TRTSERVER_TraceManager** trace_manager,
    TRTSERVER_TraceManagerCreateTraceFn_t create_fn,
    TRTSERVER_TraceManagerReleaseTraceFn_t release_fn, void* userp);

/// Delete a trace manager.
/// \param trace_manager The trace manager object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_TraceManagerDelete(
    TRTSERVER_TraceManager* trace_manager);

/// TRTSERVER_InferenceRequestOptions
///
/// Object representing the request meta-data needed for an inference.
///

/// Inference request option flags. The enum values must be power-of-2 values.
typedef enum trtserver_requestoptionsflag_enum {
  TRTSERVER_REQUEST_FLAG_NONE = 0,
  TRTSERVER_REQUEST_FLAG_SEQUENCE_START = 1,
  TRTSERVER_REQUEST_FLAG_SEQUENCE_END = 2
} TRTSERVER_Request_Options_Flag;

/// Create a new inference request options object.
/// \param request_options Returns the new request options object.
/// \param model_name The name of the model that the inference request is for.
/// \param model_version The version of the model that the inference
/// request is for, or -1 to select the latest (highest numbered) version.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestOptionsNew(
    TRTSERVER_InferenceRequestOptions** request_options, const char* model_name,
    int64_t model_version);

/// Set the ID for the request in a request options. The response of the request
/// will contain the same ID. The request sender can use the ID to correlate
/// the response to corresponding request if needed. The default value is 0.
/// \param request_options The request options object.
/// \param id The ID.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestOptionsSetId(
    TRTSERVER_InferenceRequestOptions* request_options, uint64_t id);

#ifdef TRTIS_ENABLE_GRPC_V2
/// Set the ID for the request in a request options. The response of the request
/// will contain the same ID. The request sender can use the ID to correlate
/// the response to corresponding request if needed. The default value is 0.
/// \param request_options The request options object.
/// \param id The ID.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestOptionsSetIdStr(
    TRTSERVER_InferenceRequestOptions* request_options, const char* id);
#endif  // TRTIS_ENABLE_GRPC_V2

/// Set the flag associated with the request in a request options. 'flags'
/// should holds a bitwise-or of all flag values, see
/// TRTSERVER_Request_Options_Flag for available flags.
/// \param request_options The request options object.
/// \param flags The flags.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestOptionsSetFlags(
    TRTSERVER_InferenceRequestOptions* request_options, uint32_t flags);

/// The correlation ID of the inference request. Default is 0, which
/// indictes that the request has no correlation ID. The correlation ID
/// is used to indicate two or more inference request are related to
/// each other. How this relationship is handled by the inference
/// server is determined by the model's scheduling policy.
/// \param request_options The request options object.
/// \param correlation_id The correlation ID.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsSetCorrelationId(
    TRTSERVER_InferenceRequestOptions* request_options,
    uint64_t correlation_id);

/// Set the batch size for the request in a request options.
/// \param request_options The request options object.
/// \param batch_size The batch size.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestOptionsSetBatchSize(
    TRTSERVER_InferenceRequestOptions* request_options, uint32_t batch_size);

/// Set the priority for the request in a request options.
/// \param request_options The request options object.
/// \param priority The priority level.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestOptionsSetPriority(
    TRTSERVER_InferenceRequestOptions* request_options, uint32_t priority);

/// Set the timeout for the request in a request options, in microseconds.
/// \param request_options The request options object.
/// \param timeout_ms The timeout, in microseconds.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestOptionsSetTimeout(
    TRTSERVER_InferenceRequestOptions* request_options, uint64_t timeout_ms);

/// Add a input meta-data associated with the request in a request options.
/// \param request_options The request options object.
/// \param input_name The name of the input.
/// \param dims The shape of the input. nullptr can be used if the input shape
/// is fixed size.
/// \param dim_count The length 'dims'.
/// \param batch_byte_size The size of the full batch of the input tensor, in
/// bytes. This field is only requried for non-fixed-size data type (like
/// STRING). In the case of fixed-size data type, 0 can always be set for this
/// field.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestOptionsAddInput(
    TRTSERVER_InferenceRequestOptions* request_options, const char* input_name,
    const int64_t* dims, uint64_t dim_count, uint64_t batch_byte_size);

/// Request the raw data of the specified output to be returned for requests
/// associated with the request options.
/// \param request_options The request options object.
/// \param output_name The name of the output.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestOptionsAddOutput(
    TRTSERVER_InferenceRequestOptions* request_options,
    const char* output_name);

/// Request the classification values of the specified output to be returned
/// for requests associated with the request options.
/// \param request_options The request options object.
/// \param output_name The name of the output.
/// \param count Indicates how many classification values should be returned
/// for the output. The 'count' highest priority values are returned.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER_InferenceRequestOptionsAddClassificationOutput(
    TRTSERVER_InferenceRequestOptions* request_options, const char* output_name,
    uint32_t count);

/// Delete an inference request options object.
/// \param request_options The request options object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestOptionsDelete(
    TRTSERVER_InferenceRequestOptions* request_options);

/// TRTSERVER_InferenceRequestProvider
///
/// Object representing the request provider for an inference
/// request. The request provider provides the meta-data and input
/// tensor values needed for an inference.
///

/// [DEPRECATED] Please use TRTSERVER_InferenceRequestProviderNewV2() API
/// Create a new inference request provider object. The request header
/// protobuf must be serialized and provided as a base address and a
/// size, in bytes.
/// \param request_provider Returns the new request provider object.
/// \param server the inference server object.
/// \param model_name The name of the model that the inference request
/// is for.
/// \param model_version The version of the model that the inference
/// request is for, or -1 to select the latest (highest numbered)
/// version.
/// \param request_header_base Pointer to the serialized request
/// header protobuf.
/// \param request_header_byte_size The size of the serialized request
/// header in bytes.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestProviderNew(
    TRTSERVER_InferenceRequestProvider** request_provider,
    TRTSERVER_Server* server, const char* model_name, int64_t model_version,
    const char* request_header_base, size_t request_header_byte_size);

/// Create a new inference request provider object. The caller retains ownership
/// of 'request_options' and the caller must extend the options object's
/// lifetime to be longer than the lifetime of any providers created with the
/// options object.
/// \param request_provider Returns the new request provider object.
/// \param server the inference server object.
/// \param request_options The request options object for this inference
/// request. \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestProviderNewV2(
    TRTSERVER_InferenceRequestProvider** request_provider,
    TRTSERVER_Server* server,
    TRTSERVER_InferenceRequestOptions* request_options);

/// Delete an inference request provider object.
/// \param request_provider The request provider object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestProviderDelete(
    TRTSERVER_InferenceRequestProvider* request_provider);

/// Get the size, in bytes, expected by the inference server for the
/// named input tensor. The returned size is the total size for the
/// entire batch of the input.
/// \param request_provider The request provider object.
/// \param name The name of the input.
/// \param byte_size Returns the size, in bytes, of the full batch of
/// tensors for the named input.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderInputBatchByteSize(
    TRTSERVER_InferenceRequestProvider* request_provider, const char* name,
    uint64_t* byte_size);

/// Assign a buffer of data to an input. The buffer will be appended
/// to any existing buffers for that input. The 'request_provider'
/// takes ownership of the buffer and so the caller should not modify
/// or free the buffer until that ownership is released when
/// 'request_provider' is deleted. The total size of data that is
/// provided for an input must equal the value returned by
/// TRTSERVER_InferenceRequestProviderInputBatchByteSize().
/// \param request_provider The request provider object.
/// \param name The name of the input.
/// \param base The base address of the input data.
/// \param byte_size The size, in bytes, of the input data.
/// \param memory_type The memory type of the input data.
/// \param memory_type_id The memory type id of the input data.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderSetInputData(
    TRTSERVER_InferenceRequestProvider* request_provider, const char* name,
    const void* base, size_t byte_size, TRTSERVER_Memory_Type memory_type,
    int64_t memory_type_id);

/// TRTSERVER_InferenceResponse
///
/// Object representing the response for an inference request. The
/// response handler collects output tensor data and result meta-data.
///

/// Delete an inference response handler object.
/// \param response The response object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceResponseDelete(
    TRTSERVER_InferenceResponse* response);

/// Return the success or failure status of the inference
/// request. Return a TRTSERVER_Error object on failure, return nullptr
/// on success.
/// \param response The response object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceResponseStatus(
    TRTSERVER_InferenceResponse* response);

#ifdef TRTIS_ENABLE_GRPC_V2
/// Return the V2 request ID for the response.
/// \param response The response object.
/// \param Returns the request ID.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceResponseIdStr(
    TRTSERVER_InferenceResponse* response, const char** id);
#endif  // TRTIS_ENABLE_GRPC_V2

/// Get the response header as a TRTSERVER_Protobuf object. The caller
/// takes ownership of the object and must call
/// TRTSERVER_ProtobufDelete to release the object.
/// \param response The response object.
/// \param header Returns the response header as a TRTSERVER_Protobuf
/// object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceResponseHeader(
    TRTSERVER_InferenceResponse* response, TRTSERVER_Protobuf** header);

/// Get the results data for a named output. The result data is
/// returned as the base pointer to the data and the size, in bytes, of
/// the data. The caller does not own the returned data and must not
/// modify or delete it. The lifetime of the returned data extends only
/// as long as 'response' and must not be accessed once 'response' is
/// deleted.
/// \param response The response object.
/// \param name The name of the output.
/// \param base Returns the result data for the named output.
/// \param byte_size Returns the size, in bytes, of the output data.
/// \param memory_type Returns the memory type of the output data.
/// \param memory_type_id Returns the memory type id of the output data.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceResponseOutputData(
    TRTSERVER_InferenceResponse* response, const char* name, const void** base,
    size_t* byte_size, TRTSERVER_Memory_Type* memory_type,
    int64_t* memory_type_id);

/// TRTSERVER_ServerOptions
///
/// Options to use when creating an inference server.
///

/// Model control modes
typedef enum trtserver_modelcontrolmode_enum {
  TRTSERVER_MODEL_CONTROL_NONE,
  TRTSERVER_MODEL_CONTROL_POLL,
  TRTSERVER_MODEL_CONTROL_EXPLICIT
} TRTSERVER_Model_Control_Mode;

/// Create a new server options object. The caller takes ownership of
/// the TRTSERVER_ServerOptions object and must call
/// TRTSERVER_ServerOptionsDelete to release the object.
/// \param options Returns the new server options object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsNew(
    TRTSERVER_ServerOptions** options);

/// Delete a server options object.
/// \param options The server options object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsDelete(
    TRTSERVER_ServerOptions* options);

/// Set the textual ID for the server in a server options. The ID is a
/// name that identifies the server.
/// \param options The server options object.
/// \param server_id The server identifier.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetServerId(
    TRTSERVER_ServerOptions* options, const char* server_id);

/// Set the version of the data-plane protocol supported by the server
/// in a server options.
/// \param options The server options object.
/// \param server_protocol_version The protocol version.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER_ServerOptionsSetServerProtocolVersion(
    TRTSERVER_ServerOptions* options, const uint32_t server_protocol_version);

/// Set the model repository path in a server options. The path must be
/// the full absolute path to the model repository. This function can be called
/// multiple times with different paths to set multiple model repositories.
/// Note that if a model is not unique across all model repositories
/// at any time, the model will not be available.
/// \param options The server options object.
/// \param model_repository_path The full path to the model repository.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetModelRepositoryPath(
    TRTSERVER_ServerOptions* options, const char* model_repository_path);

/// Set the model control mode in a server options. For each mode the models
/// will be managed as the following:
///
///   TRTSERVER_MODEL_CONTROL_NONE: the models in model repository will be
///   loaded on startup. After startup any changes to the model repository will
///   be ignored. Calling TRTSERVER_ServerPollModelRepository will result in an
///   error.
///
///   TRTSERVER_MODEL_CONTROL_POLL: the models in model repository will be
///   loaded on startup. The model repository can be polled periodically using
///   TRTSERVER_ServerPollModelRepository and the server will load, unload, and
///   updated models according to changes in the model repository.
///
///   TRTSERVER_MODEL_CONTROL_EXPLICIT: the models in model repository will not
///   be loaded on startup. The corresponding model control APIs must be called
///   to load / unload a model in the model repository.
///
/// \param options The server options object.
/// \param mode The mode to use for the model control.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetModelControlMode(
    TRTSERVER_ServerOptions* options, TRTSERVER_Model_Control_Mode mode);

/// Set the model to be loaded at startup in a server options. The model must be
/// present in one, and only one, of the specified model repositories.
/// This function can be called multiple times with different model name
/// to set multiple startup models.
/// Note that it only takes affect on TRTSERVER_MODEL_CONTROL_EXPLICIT mode.
/// \param options The server options object.
/// \param mode_name The name of the model to load on startup.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetStartupModel(
    TRTSERVER_ServerOptions* options, const char* model_name);

/// Enable or disable strict model configuration handling in a server
/// options.
/// \param options The server options object.
/// \param strict True to enable strict model configuration handling,
/// false to disable.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetStrictModelConfig(
    TRTSERVER_ServerOptions* options, bool strict);

/// Set the total pinned memory byte size that the server can allocate
/// in a server options. This option will not affect the allocation conducted
/// by the backend frameworks.
/// \param options The server options object.
/// \param size The pinned memory pool byte size.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(
    TRTSERVER_ServerOptions* options, uint64_t size);

/// Set the minimum support CUDA compute capability in a server
/// options.
/// \param options The server options object.
/// \param cc The minimum CUDA compute capability.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER_ServerOptionsSetMinSupportedComputeCapability(
    TRTSERVER_ServerOptions* options, double cc);

/// Enable or disable exit-on-error in a server options.
/// \param options The server options object.
/// \param exit True to enable exiting on intialization error, false
/// to continue.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetExitOnError(
    TRTSERVER_ServerOptions* options, bool exit);

/// Enable or disable strict readiness handling in a server options.
/// \param options The server options object.
/// \param strict True to enable strict readiness handling, false to
/// disable.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetStrictReadiness(
    TRTSERVER_ServerOptions* options, bool strict);

/// Set the exit timeout, in seconds, for the server in a server
/// options.
/// \param options The server options object.
/// \param timeout The exit timeout, in seconds.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetExitTimeout(
    TRTSERVER_ServerOptions* options, unsigned int timeout);

/// Enable or disable info level logging.
/// \param options The server options object.
/// \param log True to enable info logging, false to disable.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetLogInfo(
    TRTSERVER_ServerOptions* options, bool log);

/// Enable or disable warning level logging.
/// \param options The server options object.
/// \param log True to enable warning logging, false to disable.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetLogWarn(
    TRTSERVER_ServerOptions* options, bool log);

/// Enable or disable error level logging.
/// \param options The server options object.
/// \param log True to enable error logging, false to disable.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetLogError(
    TRTSERVER_ServerOptions* options, bool log);

/// Set verbose logging level. Level zero disables verbose logging.
/// \param options The server options object.
/// \param level The verbose logging level.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetLogVerbose(
    TRTSERVER_ServerOptions* options, int level);

/// Enable or disable metrics collection in a server options.
/// \param options The server options object.
/// \param metrics True to enable metrics, false to disable.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetMetrics(
    TRTSERVER_ServerOptions* options, bool metrics);

/// Enable or disable GPU metrics collection in a server options. GPU
/// metrics are collected if both this option and
/// TRTSERVER_ServerOptionsSetMetrics are true.
/// \param options The server options object.
/// \param gpu_metrics True to enable GPU metrics, false to disable.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetGpuMetrics(
    TRTSERVER_ServerOptions* options, bool gpu_metrics);

/// Enable or disable TensorFlow soft-placement of operators.
/// \param options The server options object.
/// \param soft_placement True to enable, false to disable.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER_ServerOptionsSetTensorFlowSoftPlacement(
    TRTSERVER_ServerOptions* options, bool soft_placement);

/// Set the fraction of GPU memory dedicated to TensorFlow models on
/// each GPU visible to the inference server. Zero (0) indicates that
/// no memory will be dedicated to TensorFlow and that it will instead
/// allocate memory as needed.
/// \param options The server options object.
/// \param fraction The fraction of the GPU memory dedicated to
/// TensorFlow.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER_ServerOptionsSetTensorFlowGpuMemoryFraction(
    TRTSERVER_ServerOptions* options, float fraction);

/// Add Tensorflow virtual GPU instances to a physical GPU.
/// \param options The server options object.
/// \param gpu_device The physical GPU device id.
/// \param num_vgpus The number of virtual GPUs to create on the
/// physical GPU.
/// \param per_vgpu_memory_mbytes The amount of GPU memory, in
/// megabytes, to dedicate to each virtual GPU instance.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER_ServerOptionsAddTensorFlowVgpuMemoryLimits(
    TRTSERVER_ServerOptions* options, int gpu_device, int num_vgpus,
    uint64_t per_vgpu_memory_mbytes);

/// TRTSERVER_Server
///
/// An inference server.
///

/// Create a new server object. The caller takes ownership of the
/// TRTSERVER_Server object and must call TRTSERVER_ServerDelete
/// to release the object.
/// \param server Returns the new inference server object.
/// \param options The inference server options object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerNew(
    TRTSERVER_Server** server, TRTSERVER_ServerOptions* options);

/// Delete a server object. If server is not already stopped it is
/// stopped before being deleted.
/// \param server The inference server object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerDelete(
    TRTSERVER_Server* server);

/// Stop a server object. A server can't be restarted once it is
/// stopped.
/// \param server The inference server object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerStop(
    TRTSERVER_Server* server);

/// Get the string identifier (i.e. name) of the server. The caller
/// does not own the returned string and must not modify or delete
/// it. The lifetime of the returned string extends only as long as
/// 'server' and must not be accessed once 'server' is deleted.
/// \param server The inference server object.
/// \param id Returns the server identifier.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerId(
    TRTSERVER_Server* server, const char** id);

/// Get the version of the server. The caller does not own the
/// returned string and must not modify or delete it. The lifetime of
/// the returned string extends only as long as 'server' and must not
/// be accessed once 'server' is deleted.
/// \param server The inference server object.
/// \param version Returns the server version as a string.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerVersion(
    TRTSERVER_Server* server, const char** version);

/// Get the extensions supported by the server. The caller does not
/// own the returned strings and must not modify or delete them. The
/// lifetime of the returned strings extends only as long as 'server'
/// and must not be accessed once 'server' is deleted.
/// \param server The inference server object.
/// \param extensions Returns the server extensions as an array of
/// pointers to the name of each extension.
/// \param extensions_count Returns the number of server extensions.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerExtensions(
    TRTSERVER_Server* server, const char* const** extensions,
    uint64_t* extensions_count);

/// Check the model repository for changes and update server state
/// based on those changes.
/// \param server The inference server object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerPollModelRepository(
    TRTSERVER_Server* server);

/// Is the server live?
/// \param server The inference server object.
/// \param live Returns true if server is live, false otherwise.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerIsLive(
    TRTSERVER_Server* server, bool* live);

/// Is the server ready?
/// \param server The inference server object.
/// \param ready Returns true if server is ready, false otherwise.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerIsReady(
    TRTSERVER_Server* server, bool* ready);

/// Get the current server status for all models as a
/// TRTSERVER_Protobuf object. The caller takes ownership of the object
/// and must call TRTSERVER_ProtobufDelete to release the object.
/// \param server The inference server object.
/// \param status Returns the server status protobuf.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerStatus(
    TRTSERVER_Server* server, TRTSERVER_Protobuf** status);

/// Get the current server status for a single model as a
/// TRTSERVER_Protobuf object. The caller takes ownership of the object
/// and must call TRTSERVER_ProtobufDelete to release the object.
/// \param server The inference server object.
/// \param model_name The name of the model to get status for.
/// \param status Returns the server status protobuf.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerModelStatus(
    TRTSERVER_Server* server, const char* model_name,
    TRTSERVER_Protobuf** status);

/// Get the index for all unique models in the model repository  as a
/// TRTSERVER_Protobuf object. The caller takes ownership of the object
/// and must call TRTSERVER_ProtobufDelete to release the object.
/// \param server The inference server object.
/// \param repository_index Returns the model repository index protobuf.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerModelRepositoryIndex(
    TRTSERVER_Server* server, TRTSERVER_Protobuf** repository_index);

/// Load the requested model or reload the model if it is already
/// loaded. The function does not return until the model is loaded or
/// fails to load. Returned error indicates if model loaded
/// successfully or not.
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerLoadModel(
    TRTSERVER_Server* server, const char* model_name);

/// Unload the requested model. Unloading a model that is not loaded
/// on server has no affect and success code will be returned.
/// The function does not return until the model is unloaded or fails to unload.
/// Returned error indicates if model unloaded successfully or not.
/// \param server The inference server object.
/// \param model_name The name of the model.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerUnloadModel(
    TRTSERVER_Server* server, const char* model_name);

/// Register a shared memory block on the inference server. After a
/// block is registered, addresses within the block can be used for
/// input and output tensors in inference requests. If a shared memory
/// block with the same name is already registered
/// TRTSERVER_ERROR_ALREADY_EXISTS is returned.
/// \param server The inference server object.
/// \param shared_memory_block The shared memory block to register.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerRegisterSharedMemory(
    TRTSERVER_Server* server, TRTSERVER_SharedMemoryBlock* shared_memory_block);

/// Unregister a shared memory block on the inference server. No
/// operation is performed if the shared memory block is not
/// registered.
/// \param server The inference server object.
/// \param shared_memory_block The shared memory block to unregister.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerUnregisterSharedMemory(
    TRTSERVER_Server* server, TRTSERVER_SharedMemoryBlock* shared_memory_block);

/// Unregister all shared memory blocks that are currently registered
/// \param server The inference server object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerUnregisterAllSharedMemory(
    TRTSERVER_Server* server);

/// Get an address in a shared memory block that has been registered
/// with the inference server. Verify that a 'byte_size' block of
/// memory starting at that address is completely contained within the
/// shared memory block.

/// \param server The inference server object.
/// \param shared_memory_block The shared memory block.
/// \param offset The offset within the shared memory block to get the
/// address for.
/// \param byte_size The size of block to within the shared memory
/// block. Returns error if a block of this size (starting at
/// 'offset') isn't completely contained in the shared memory block.
/// \param base Returns the base address.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerSharedMemoryAddress(
    TRTSERVER_Server* server, TRTSERVER_SharedMemoryBlock* shared_memory_block,
    size_t offset, size_t byte_size, void** base);

/// Get the list of all active shared memory region on the inference server.
/// If there are none then the list is empty. Returned error indicates if it
/// was able to successfully get all active shared memory regions or not.
/// \param server The inference server object.
/// \param status Get the current shared memory region status of the inference
/// server. The caller takes ownership of 'status' and must call
/// TRTSERVER_ProtobufDelete to release the object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerSharedMemoryStatus(
    TRTSERVER_Server* server, TRTSERVER_Protobuf** status);

/// Get the current metrics for the server. The caller takes ownership
/// of the metrics object and must call TRTSERVER_MetricsDelete to
/// release the object.
/// \param server The inference server object.
/// \param metrics Returns the metrics.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerMetrics(
    TRTSERVER_Server* server, TRTSERVER_Metrics** metrics);

/// Type for inference completion callback function. If non-nullptr,
/// the 'trace_manager' object is the trace manager associated with the request
/// that is completing. The callback function takes ownership of the
/// TRTSERVER_TraceManager object and must call TRTSERVER_TraceManagerDelete to
/// release the object. The callback function takes ownership of the
/// TRTSERVER_InferenceResponse object and must call
/// TRTSERVER_InferenceResponseDelete to release the object. The
/// 'userp' data is the same as what is supplied in the call to
/// TRTSERVER_ServerInferAsync.
typedef void (*TRTSERVER_InferenceCompleteFn_t)(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER_InferenceResponse* response, void* userp);

/// Perform inference using the meta-data and inputs supplied by the
/// request provider. The caller retains ownership of
/// 'request_provider' but may release it by calling
/// TRTSERVER_InferenceRequestProviderDelete once this function
/// returns.
/// \param server The inference server object.
/// \param trace_manager The trace manager object for this request, or nullptr
/// if no tracing.
/// \param request_provider The request provider for the request.
/// \param response_allocator The TRTSERVER_ResponseAllocator to use
/// to allocate buffers to hold inference results.
/// \param response_allocator_userp User-provided pointer that is
/// delivered to the response allocator's allocation function.
/// \param complete_fn The function called when the inference
/// completes.
/// \param complete_userp User-provided pointer that is delivered to
/// the completion function.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerInferAsync(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER_InferenceRequestProvider* request_provider,
    TRTSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp, TRTSERVER_InferenceCompleteFn_t complete_fn,
    void* complete_userp);

#ifdef __cplusplus
}
#endif
