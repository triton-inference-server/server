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
#include "src/core/trtserver.h"

#ifdef __cplusplus
extern "C" {
#endif

struct TRTSERVER2_InferenceRequest;
struct TRTSERVER2_ModelIndex;

/// TRTSERVER2_InferenceRequest
///
/// Object representing an inference request. The inference request
/// provides the meta-data and input tensor values needed for an
/// inference and returns the inference result meta-data and output
/// tensors. An inference request object can be modified and reused
/// multiple times.
///

/// Create a new inference request object.
/// \param inference_request Returns the new request object.
/// \param server the inference server object.
/// \param model_name The name of the model to use for the request.
/// \param model_version The version of the model to use for the
/// request. If nullptr or empty then the server will choose a version
/// based on the model's policy.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestNew(
    TRTSERVER2_InferenceRequest** inference_request, TRTSERVER_Server* server,
    const char* model_name, const char* model_version);

/// Delete an inference request object.
/// \param inference_request The request object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestDelete(
    TRTSERVER2_InferenceRequest* inference_request);

/// Get the ID for a request. The returned ID is owned by
/// 'inference_request' and must not be modified or freed by the
/// caller.
/// \param inference_request The request object.
/// \param id Returns the ID.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestId(
    TRTSERVER2_InferenceRequest* inference_request, const char** id);

/// Set the ID for a request.
/// \param inference_request The request object.
/// \param id The ID.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestSetId(
    TRTSERVER2_InferenceRequest* inference_request, const char* id);

/// Get the flag(s) associated with a request. On return 'flags' holds
/// a bitwise-or of all flag values, see
/// TRTSERVER_Request_Options_Flag for available flags.
/// \param inference_request The request object.
/// \param flags Returns the flags.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestFlags(
    TRTSERVER2_InferenceRequest* inference_request, uint32_t* flags);

/// Set the flag(s) associated with a request. 'flags'
/// should holds a bitwise-or of all flag values, see
/// TRTSERVER_Request_Options_Flag for available flags.
/// \param inference_request The request object.
/// \param flags The flags.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestSetFlags(
    TRTSERVER2_InferenceRequest* inference_request, uint32_t flags);

/// Get the correlation ID of the inference request. Default is 0,
/// which indictes that the request has no correlation ID. The
/// correlation ID is used to indicate two or more inference request
/// are related to each other. How this relationship is handled by the
/// inference server is determined by the model's scheduling
/// policy.
/// \param inference_request The request object.
/// \param correlation_id Returns the correlation ID.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestCorrelationId(
    TRTSERVER2_InferenceRequest* inference_request, uint64_t* correlation_id);

/// Set the correlation ID of the inference request. Default is 0, which
/// indictes that the request has no correlation ID. The correlation ID
/// is used to indicate two or more inference request are related to
/// each other. How this relationship is handled by the inference
/// server is determined by the model's scheduling policy.
/// \param inference_request The request object.
/// \param correlation_id The correlation ID.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestSetCorrelationId(
    TRTSERVER2_InferenceRequest* inference_request, uint64_t correlation_id);

/// Get the priority for a request. The default is 0 indicating that
/// the request does not specify a priority and so will use the
/// model's default priority.
/// \param inference_request The request object.
/// \param priority Returns the priority level.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestPriority(
    TRTSERVER2_InferenceRequest* inference_request, uint32_t* priority);

/// Set the priority for a request. The default is 0 indicating that
/// the request does not specify a priority and so will use the
/// model's default priority.
/// \param inference_request The request object.
/// \param priority The priority level.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestSetPriority(
    TRTSERVER2_InferenceRequest* inference_request, uint32_t priority);

/// Get the timeout for a request, in microseconds. The default is 0
/// which indicates that the request has no timeout.
/// \param inference_request The request object.
/// \param timeout_us Returns the timeout, in microseconds.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER2_InferenceRequestTimeoutMicroseconds(
    TRTSERVER2_InferenceRequest* inference_request, uint64_t* timeout_us);

/// Set the timeout for a request, in microseconds. The default is 0
/// which indicates that the request has no timeout.
/// \param inference_request The request object.
/// \param timeout_us The timeout, in microseconds.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER2_InferenceRequestSetTimeoutMicroseconds(
    TRTSERVER2_InferenceRequest* inference_request, uint64_t timeout_us);

/// Add an input to a request.
/// \param inference_request The request object.
/// \param name The name of the input.
/// \param datatype The type of the input. Valid type names are BOOL,
/// UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FP16,
/// FP32, FP64, and BYTES.
/// \param shape The shape of the input.
/// \param dim_count The number of dimensions of 'shape'.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestAddInput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    const char* datatype, const int64_t* shape, uint64_t dim_count);

/// Remove an input from a request.
/// \param inference_request The request object.
/// \param name The name of the input.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestRemoveInput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name);

/// Remove all inputs from a request.
/// \param inference_request The request object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestRemoveAllInputs(
    TRTSERVER2_InferenceRequest* inference_request);

/// Assign a buffer of data to an input. The buffer will be appended
/// to any existing buffers for that input. The 'inference_request'
/// object takes ownership of the buffer and so the caller should not
/// modify or free the buffer until that ownership is released by
/// 'inference_request' being deleted or by the input being removed
/// from 'inference_request'.
/// \param inference_request The request object.
/// \param name The name of the input.
/// \param base The base address of the input data.
/// \param byte_size The size, in bytes, of the input data.
/// \param memory_type The memory type of the input data.
/// \param memory_type_id The memory type id of the input data.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestAppendInputData(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    const void* base, size_t byte_size, TRTSERVER_Memory_Type memory_type,
    int64_t memory_type_id);

/// Clear all input data from an input, releasing ownership of the
/// buffer(s) that were appended to the input with
/// TRTSERVER2_InferenceRequestAppendInputData.
/// \param inference_request The request object.
/// \param name The name of the input.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestRemoveAllInputData(
    TRTSERVER2_InferenceRequest* inference_request, const char* name);

/// Add an output request to an inference request.
/// \param inference_request The request object.
/// \param name The name of the output.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestAddRequestedOutput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name);

/// Remove an output request from an inference request.
/// \param inference_request The request object.
/// \param name The name of the output.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER2_InferenceRequestRemoveRequestedOutput(
    TRTSERVER2_InferenceRequest* inference_request, const char* name);

/// Remove all output requests from an inference request.
/// \param inference_request The request object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER2_InferenceRequestRemoveAllRequestedOutputs(
    TRTSERVER2_InferenceRequest* inference_request);

/// Set that a requested output should be returned as a tensor of
/// classification strings instead of as the tensor defined by the model.
/// \param inference_request The request object.
/// \param name The name of the output.
/// \param count Indicates how many classification values should be
/// returned for the output. The 'count' highest priority values are
/// returned. The default is 0, indicating that the output tensor
/// should not be returned as a classification.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER2_InferenceRequestSetRequestedOutputClassificationCount(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    uint32_t count);

/// Return the error status of an inference request corresponding to
/// the most recent call to TRTSERVER2_ServerInferAsync. Return a
/// TRTSERVER_Error object on failure, return nullptr on success.  The
/// returned error object is owned by 'inference_request' and so
/// should not be deleted by the caller.
/// \param inference_request The request object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestError(
    TRTSERVER2_InferenceRequest* inference_request);

/// Get the datatype of an output tensor.
/// \param inference_request The request object.
/// \param name The name of the output.
/// \param datatype Returns the type of the output. The returned
/// datatype is owned by 'inference_request' and must not be modified
/// or freed by the caller.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestOutputDataType(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    const char** datatype);

/// Get the shape of an output tensor.
/// \param inference_request The request object.
/// \param name The name of the output.
/// \param shape Buffer where the shape of the output is returned.
/// \param dim_count Acts as input and output. On input gives the
/// maximum number of dimensions that can be recorded in
/// 'shape'. Returns the number of dimensions of the returned shape.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestOutputShape(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    int64_t* shape, uint64_t* dim_count);

/// Get the results data for a named output. The result data is
/// returned as the base pointer to the data and the size, in bytes,
/// of the data. The caller does not own the returned data and must
/// not modify or delete it. The lifetime of the returned data extends
/// until 'inference_request' is deleted or until 'inference_request' is
/// reused in a call to TRTSERVER2_ServerInferAsync.
/// \param inference_request The request object.
/// \param name The name of the output.
/// \param base Returns the result data for the named output.
/// \param byte_size Returns the size, in bytes, of the output data.
/// \param memory_type Returns the memory type of the output data.
/// \param memory_type_id Returns the memory type id of the output data.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestOutputData(
    TRTSERVER2_InferenceRequest* inference_request, const char* name,
    const void** base, size_t* byte_size, TRTSERVER_Memory_Type* memory_type,
    int64_t* memory_type_id);

/// Remove all the output tensors. The meta data of the output tensors will
/// become unaccesible and the result data will be released.
/// \param inference_request The request object.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_InferenceRequestRemoveAllOutputs(
    TRTSERVER2_InferenceRequest* inference_request);

///
/// TRTSERVER2_ModelIndex
///
/// Object representing model index.
///

/// Get the object representing the index of all unique models in the
/// model repositories.
/// \param server The inference server object.
/// \param model_index Return the TRTSERVER2_ModelIndex object that holds the
/// index of all models contained in the server's model repository(s).
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_ServerModelIndex(
    TRTSERVER_Server* server, TRTSERVER2_ModelIndex** model_index);

/// Get the names of all unique models in the model repository.
/// The caller does not own the returned strings and must not modify or
/// delete them. The lifetime of the returned strings extends only as
/// long as 'model_index' and must not be accessed once 'model_index'
/// is deleted.
/// \param model_index The TRTSERVER2_ModelIndex object that
/// is used to manage the lifecycle of the returned strings.
/// \param models Returns the names of all unique models as an array of
/// pointers to the name of each model.
/// \param models_count Returns the number of unique models.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_ModelIndexNames(
    TRTSERVER2_ModelIndex* model_index, const char* const** models,
    uint64_t* models_count);

/// Delete a model indices object.
/// \param model_index The index of models.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_ModelIndexDelete(
    TRTSERVER2_ModelIndex* model_index);

/// Type for inference completion callback function. If non-nullptr,
/// the 'trace_manager' object is the trace manager associated with
/// the request that is completing. The callback function takes
/// ownership of the TRTSERVER_TraceManager object and must call
/// TRTSERVER_TraceManagerDelete to release the object. The callback
/// function takes ownership of the TRTSERVER2_InferenceRequest object
/// and must call TRTSERVER2_InferenceRequestDelete to release the
/// object. The 'userp' data is the same as what is supplied in the
/// call to TRTSERVER2_ServerInferAsync.
typedef void (*TRTSERVER2_InferenceCompleteFn_t)(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER2_InferenceRequest* request, void* userp);

/// Perform inference using the meta-data and inputs supplied by the
/// 'inference_request'. The caller releases ownership of
/// 'inference_request' and 'trace_manager' and must not access them
/// in any way after this call, until ownership is returned via the
/// completion function.
/// \param server The inference server object.
/// \param trace_manager The trace manager object for this request, or
/// nullptr if no tracing.
/// \param inference_request The request object.
/// \param response_allocator The TRTSERVER_ResponseAllocator to use
/// to allocate buffers to hold inference results.
/// \param response_allocator_userp User-provided pointer that is
/// delivered to the response allocator's allocation function.
/// \param complete_fn The function called when the inference
/// completes.
/// \param complete_userp User-provided pointer that is delivered to
/// the completion function.
/// \return a TRTSERVER_Error indicating success or failure.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER2_ServerInferAsync(
    TRTSERVER_Server* server, TRTSERVER_TraceManager* trace_manager,
    TRTSERVER2_InferenceRequest* inference_request,
    TRTSERVER_ResponseAllocator* response_allocator,
    void* response_allocator_userp,
    TRTSERVER2_InferenceCompleteFn_t complete_fn, void* complete_userp);

#ifdef __cplusplus
}
#endif
