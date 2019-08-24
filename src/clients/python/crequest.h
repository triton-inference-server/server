// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "src/clients/c++/request.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

#ifdef __cplusplus
extern "C" {
#endif


//==============================================================================
// Error
nic::Error* ErrorNew(const char* msg);
void ErrorDelete(nic::Error* ctx);
bool ErrorIsOk(nic::Error* ctx);
bool ErrorIsUnavailable(nic::Error* ctx);
const char* ErrorMessage(nic::Error* ctx);
const char* ErrorServerId(nic::Error* ctx);
uint64_t ErrorRequestId(nic::Error* ctx);

//==============================================================================
// ServerHealthContext
typedef struct ServerHealthContextCtx ServerHealthContextCtx;
nic::Error* ServerHealthContextNew(
    ServerHealthContextCtx** ctx, const char* url, int protocol_int,
    const char** headers, int num_headers, bool verbose);
void ServerHealthContextDelete(ServerHealthContextCtx* ctx);
nic::Error* ServerHealthContextGetReady(
    ServerHealthContextCtx* ctx, bool* ready);
nic::Error* ServerHealthContextGetLive(ServerHealthContextCtx* ctx, bool* live);

//==============================================================================
// ServerStatusContext
typedef struct ServerStatusContextCtx ServerStatusContextCtx;
nic::Error* ServerStatusContextNew(
    ServerStatusContextCtx** ctx, const char* url, int protocol_int,
    const char** headers, int num_headers, const char* model_name,
    bool verbose);
void ServerStatusContextDelete(ServerStatusContextCtx* ctx);
nic::Error* ServerStatusContextGetServerStatus(
    ServerStatusContextCtx* ctx, char** status, uint32_t* status_len);

//==============================================================================
// ModelControlContext
typedef struct ModelControlContextCtx ModelControlContextCtx;
nic::Error* ModelControlContextNew(
    ModelControlContextCtx** ctx, const char* url, int protocol_int,
    const char** headers, int num_headers, bool verbose);
void ModelControlContextDelete(ModelControlContextCtx* ctx);
nic::Error* ModelControlContextLoad(
    ModelControlContextCtx* ctx, const char* model_name);
nic::Error* ModelControlContextUnload(
    ModelControlContextCtx* ctx, const char* model_name);

//==============================================================================
// SharedMemoryControlContext
typedef struct SharedMemoryControlContextCtx SharedMemoryControlContextCtx;
nic::Error* SharedMemoryControlContextNew(
    SharedMemoryControlContextCtx** ctx, const char* url, int protocol_int,
    const char** headers, int num_headers, bool verbose);
void SharedMemoryControlContextDelete(SharedMemoryControlContextCtx* ctx);
nic::Error* SharedMemoryControlContextRegister(
    SharedMemoryControlContextCtx* ctx, const char* name, const char* shm_key,
    const int offset, const int byte_size);
nic::Error* SharedMemoryControlContextUnregister(
    SharedMemoryControlContextCtx* ctx, const char* model_name);
nic::Error* GetSharedMemoryHandleInfo(
    void* shm_handle, void** shm_addr, const char** shm_key, int* shm_fd);
//==============================================================================
// InferContext
typedef struct InferContextCtx InferContextCtx;
nic::Error* InferContextNew(
    InferContextCtx** ctx, const char* url, int protocol_int,
    const char** headers, int num_headers, const char* model_name,
    int64_t model_version, ni::CorrelationID correlation_id, bool streaming,
    bool verbose);
void InferContextDelete(InferContextCtx* ctx);
nic::Error* InferContextSetOptions(
    InferContextCtx* ctx, nic::InferContext::Options* options);
nic::Error* InferContextRun(InferContextCtx* ctx);
nic::Error* InferContextAsyncRun(InferContextCtx* ctx, uint64_t* request_id);
nic::Error* InferContextAsyncRunWithCallback(
    InferContextCtx* ctx, void (*callback)(InferContextCtx*, uint64_t));
nic::Error* InferContextGetAsyncRunResults(
    InferContextCtx* ctx, bool* is_ready, uint64_t request_id, bool wait);
nic::Error* InferContextGetReadyAsyncRequest(
    InferContextCtx* ctx, bool* is_ready, uint64_t* request_id, bool wait);

//==============================================================================
// InferContext::Options
nic::Error* InferContextOptionsNew(
    nic::InferContext::Options** ctx, uint32_t flags, uint64_t batch_size);
void InferContextOptionsDelete(nic::InferContext::Options* ctx);
nic::Error* InferContextOptionsAddRaw(
    InferContextCtx* infer_ctx, nic::InferContext::Options* ctx,
    const char* output_name);
nic::Error* InferContextOptionsAddClass(
    InferContextCtx* infer_ctx, nic::InferContext::Options* ctx,
    const char* output_name, uint64_t count);
nic::Error* InferContextOptionsAddSharedMemory(
    InferContextCtx* infer_ctx, nic::InferContext::Options* ctx,
    const char* output_name, const char* shm_name, uint64_t offset,
    uint64_t byte_size);

//==============================================================================
// InferContext::Input
typedef struct InferContextInputCtx InferContextInputCtx;
nic::Error* InferContextInputNew(
    InferContextInputCtx** ctx, InferContextCtx* infer_ctx,
    const char* input_name);
void InferContextInputDelete(InferContextInputCtx* ctx);
nic::Error* InferContextInputSetShape(
    InferContextInputCtx* ctx, const int64_t* dims, uint64_t size);
nic::Error* InferContextInputSetRaw(
    InferContextInputCtx* ctx, const void* data, uint64_t byte_size);
nic::Error* InferContextInputSetSharedMemory(
    InferContextInputCtx* ctx, const char* shm_key, size_t offset,
    size_t byte_size);

//==============================================================================
// InferContext::Result
typedef struct InferContextResultCtx InferContextResultCtx;
nic::Error* InferContextResultNew(
    InferContextResultCtx** ctx, InferContextCtx* infer_ctx,
    const char* result_name);
nic::Error* InferContextAsyncResultNew(
    InferContextResultCtx** ctx, InferContextCtx* infer_ctx,
    const uint64_t request_id, const char* result_name);
void InferContextResultDelete(InferContextResultCtx* ctx);
nic::Error* InferContextResultModelName(
    InferContextResultCtx* ctx, const char** model_name);
nic::Error* InferContextResultModelVersion(
    InferContextResultCtx* ctx, int64_t* model_version);
nic::Error* InferContextResultDataType(
    InferContextResultCtx* ctx, uint32_t* dtype);
nic::Error* InferContextResultShape(
    InferContextResultCtx* ctx, uint64_t max_dims, int64_t* shape,
    uint64_t* shape_len);
nic::Error* InferContextResultNextRaw(
    InferContextResultCtx* ctx, size_t batch_idx, const char** val,
    uint64_t* val_len);
nic::Error* InferContextResultClassCount(
    InferContextResultCtx* ctx, size_t batch_idx, uint64_t* count);
nic::Error* InferContextResultNextClass(
    InferContextResultCtx* ctx, size_t batch_idx, uint64_t* idx, float* prob,
    const char** label);

#ifdef __cplusplus
}
#endif
