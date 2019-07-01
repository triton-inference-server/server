// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#define TRTSERVER_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define TRTSERVER_EXPORT __attribute__((__visibility__("default")))
#else
#define TRTSERVER_EXPORT
#endif

//
// TRTSERVER_Error
//
// Errors are reported by a TRTSERVER_Error object. A NULL
// TRTSERVER_Error indicates no error, a non-NULL TRTSERVER_Error
// indicates error and the code and message for the error can be
// retrieved from the object.
//
// The caller takes ownership of a TRTSERVER_Error object returned by
// the API and must call TRTSERVER_ErrorDelete to release the object.
//
struct TRTSERVER_Error;

// The error codes
typedef enum trtserver_errorcode_enum {
  TRTSERVER_ERROR_UNKNOWN,
  TRTSERVER_ERROR_INTERNAL,
  TRTSERVER_ERROR_NOT_FOUND,
  TRTSERVER_ERROR_INVALID_ARG,
  TRTSERVER_ERROR_UNAVAILABLE,
  TRTSERVER_ERROR_UNSUPPORTED,
  TRTSERVER_ERROR_ALREADY_EXISTS
} TRTSERVER_Error_Code;

// Create a new error object. The caller takes ownership of the
// TRTSERVER_Error object and must call TRTSERVER_ErrorDelete to
// release the object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ErrorNew(
    TRTSERVER_Error_Code code, const char* msg);

// Delete an error object.
TRTSERVER_EXPORT void TRTSERVER_ErrorDelete(TRTSERVER_Error* error);

// Get the error code.
TRTSERVER_EXPORT TRTSERVER_Error_Code
TRTSERVER_ErrorCode(TRTSERVER_Error* error);

// Get the string representation of an error code. The returned string
// is not owned by the caller and so should not be modified or freed.
TRTSERVER_EXPORT const char* TRTSERVER_ErrorCodeString(TRTSERVER_Error* error);

// Get the error message.
TRTSERVER_EXPORT const char* TRTSERVER_ErrorMessage(TRTSERVER_Error* error);

//
// TRTSERVER_Protobuf
//
// Object representing a protobuf.
//
struct TRTSERVER_Protobuf;

// Delete a protobuf object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ProtobufDelete(
    TRTSERVER_Protobuf* protobuf);

// Get the base and size of the buffer containing the serialized
// version of the protobuf. The buffer is owned by the
// TRTSERVER_Protobuf object and should not be modified or freed by
// the caller.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ProtobufSerialize(
    TRTSERVER_Protobuf* protobuf, const char** base, size_t* byte_size);

//
// TRTSERVER_MemoryAllocator
//
// Object representing a memory allocator.
//
struct TRTSERVER_MemoryAllocator;

// Memory allocation regions.
typedef enum trtserver_memoryallocatorregions_enum {
  TRTSERVER_MEMORY_CPU,
  TRTSERVER_MEMORY_GPU
} TRTSERVER_MemoryAllocator_Region;

// Type for allocation function. Return in 'buffer' the pointer to the
// contiguous memory block of size 'byte_size'. Return a
// TRTSERVER_Error object on failure, return nullptr on success.
typedef TRTSERVER_Error* (*TRTSERVER_MemoryAllocFn_t)(
    void** buffer, size_t byte_size, TRTSERVER_MemoryAllocator_Region region,
    int64_t region_id);

// Type for delete function. Return a TRTSERVER_Error object on
// failure, return nullptr on success.
typedef TRTSERVER_Error* (*TRTSERVER_MemoryDeleteFn_t)(
    void* buffer, size_t byte_size, TRTSERVER_MemoryAllocator_Region region,
    int64_t region_id);

// Create a new memory allocator object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_MemoryAllocatorNew(
    TRTSERVER_MemoryAllocator** allocator, TRTSERVER_MemoryAllocFn_t alloc_fn,
    TRTSERVER_MemoryDeleteFn_t delete_fn);

// Delete a memory allocator.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_MemoryAllocatorDelete(
    TRTSERVER_MemoryAllocator* allocator);

//
// TRTSERVER_InferenceRequestProvider
//
// Object representing the request provider for an inference
// request. The request provider provides the meta-data and input
// tensor values needed for an inference.
//
struct TRTSERVER_InferenceRequestProvider;

// Create a new inference request provider object. The request header
// protobuf must be serialized and provided as a base address and a
// size, in bytes.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestProviderNew(
    TRTSERVER_InferenceRequestProvider** request_provider,
    const char* model_name, int64_t model_version,
    const char* request_header_base, size_t request_header_byte_size);

// Delete an inference request provider object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceRequestProviderDelete(
    TRTSERVER_InferenceRequestProvider* request_provider);

// Assign a buffer of data to an input. The buffer will be appended to
// any existing buffers for that input.
TRTSERVER_EXPORT TRTSERVER_Error*
TRTSERVER_InferenceRequestProviderSetInputData(
    TRTSERVER_InferenceRequestProvider* request_provider,
    const char* input_name, const void* base, size_t byte_size);

//
// TRTSERVER_InferenceResponse
//
// Object representing the response for an inference request. The
// response handler collects output tensor data and result meta-data.
//
struct TRTSERVER_InferenceResponse;

// Delete an inference response handler object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceResponseDelete(
    TRTSERVER_InferenceResponse* response);

// Return the success or failure status of the inference
// request. Return a TRTSERVER_Error object on failure, return nullptr
// on success.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceResponseStatus(
    TRTSERVER_InferenceResponse* response);

// Get the response header as a TRTSERVER_protobuf object. The caller
// takes ownership of the object and must call
// TRTSERVER_ProtobufDelete to release the object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceResponseHeader(
    TRTSERVER_InferenceResponse* response, TRTSERVER_Protobuf** header);

// Get the results data for a named output. The result data is
// returned as the base pointer to the data and the size, in bytes, of
// the data. The caller does not own the returned data and must not
// modify or delete it.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_InferenceResponseOutputData(
    TRTSERVER_InferenceResponse* response, const char* name, const void** base,
    size_t* byte_size);

//
// TRTSERVER_ServerOptions
//
// Options to use when creating an inference server.
//
struct TRTSERVER_ServerOptions;

// Create a new server options object. The caller takes ownership of
// the TRTSERVER_ServerOptions object and must call
// TRTSERVER_ServerOptionsDelete to release the object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsNew(
    TRTSERVER_ServerOptions** options);

// Delete a server options object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsDelete(
    TRTSERVER_ServerOptions* options);

// Set the model repository path in a server options. The path must be
// the full absolute path to the model repository.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetModelRepositoryPath(
    TRTSERVER_ServerOptions* options, const char* model_repository_path);

//
// TRTSERVER_Server
//
// An inference server.
//
struct TRTSERVER_Server;

// Create a new server object. The caller takes ownership of the
// TRTSERVER_Server object and must call TRTSERVER_ServerDelete
// to release the object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerNew(
    TRTSERVER_Server** server, TRTSERVER_ServerOptions* options);

// Delete a server object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerDelete(
    TRTSERVER_Server* server);

// Is the server live?
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerIsLive(
    TRTSERVER_Server* server, bool* live);

// Is the server ready?
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerIsReady(
    TRTSERVER_Server* server, bool* ready);

// Get the current server status for all models as a
// TRTSERVER_protobuf object. The caller takes ownership of the object
// and must call TRTSERVER_ProtobufDelete to release the object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerStatus(
    TRTSERVER_Server* server, TRTSERVER_Protobuf** status);

// Get the current server status for a single model as a
// TRTSERVER_protobuf object. The caller takes ownership of the object
// and must call TRTSERVER_ProtobufDelete to release the object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerModelStatus(
    TRTSERVER_Server* server, TRTSERVER_Protobuf** status,
    const char* model_name);

// Type for inference completion callback function. The callback
// function takes ownership of the TRTSERVER_InferenceResponse object
// and must call TRTSERVER_InferenceResponseDelete to release the
// object. The 'userp' data is the same as what is supplied in the
// call to TRTSERVER_ServerInferAsync.
typedef void (*TRTSERVER_InferenceCompleteFn_t)(
    TRTSERVER_Server* server, TRTSERVER_InferenceResponse* response,
    void* userp);

// Perform inference using the meta-data and inputs supplied by the
// request provider. The caller retains ownership of
// 'request_provider' but may release it by calling
// TRTSERVER_InferenceRequestProviderDelete once this function
// returns.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerInferAsync(
    TRTSERVER_Server* server,
    TRTSERVER_InferenceRequestProvider* request_provider,
    TRTSERVER_InferenceCompleteFn_t complete_fn, void* userp);

#ifdef __cplusplus
}
#endif
