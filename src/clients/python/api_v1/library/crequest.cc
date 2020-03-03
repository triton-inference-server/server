// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/clients/python/api_v1/library/crequest.h"

#include <iostream>
#include "src/clients/c++/library/request_grpc.h"
#include "src/clients/c++/library/request_http.h"
#include "src/clients/python/api_v1/library/shared_memory/shared_memory_handle.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

//==============================================================================
nic::Error*
ErrorNew(const char* msg)
{
  return new nic::Error(ni::RequestStatusCode::INTERNAL, std::string(msg));
}

void
ErrorDelete(nic::Error* ctx)
{
  delete ctx;
}

bool
ErrorIsOk(nic::Error* ctx)
{
  return ctx->IsOk();
}

const char*
ErrorMessage(nic::Error* ctx)
{
  return ctx->Message().c_str();
}

const char*
ErrorServerId(nic::Error* ctx)
{
  return ctx->ServerId().c_str();
}

uint64_t
ErrorRequestId(nic::Error* ctx)
{
  return ctx->RequestId();
}

//==============================================================================
namespace {

enum ProtocolType { HTTP = 0, GRPC = 1 };

nic::Error
ParseProtocol(ProtocolType* protocol, const int protocol_int)
{
  *protocol = ProtocolType::HTTP;
  if (protocol_int == 0) {
    return nic::Error::Success;
  }

  if (protocol_int == 1) {
    *protocol = ProtocolType::GRPC;
    return nic::Error::Success;
  }

  return nic::Error(
      ni::RequestStatusCode::INVALID_ARG,
      "unexpected protocol integer, expecting 0 for HTTP or 1 for gRPC");
}

nic::Error
ParseHttpHeaders(
    std::map<std::string, std::string>* http_headers, const char** headers,
    int num_headers)
{
  for (int i = 0; i < num_headers; ++i) {
    std::string full(headers[i]);
    std::string header = full.substr(0, full.find(":"));
    (*http_headers)[header] = full.substr(header.size() + 1);
  }

  return nic::Error::Success;
}

}  // namespace

//==============================================================================
struct ServerHealthContextCtx {
  std::unique_ptr<nic::ServerHealthContext> ctx;
};

nic::Error*
ServerHealthContextNew(
    ServerHealthContextCtx** ctx, const char* url, int protocol_int,
    const char** headers, int num_headers, bool verbose)
{
  nic::Error err;
  ProtocolType protocol;
  err = ParseProtocol(&protocol, protocol_int);
  if (err.IsOk()) {
    ServerHealthContextCtx* lctx = new ServerHealthContextCtx;
    if (protocol == ProtocolType::HTTP) {
      std::map<std::string, std::string> http_headers;
      err = ParseHttpHeaders(&http_headers, headers, num_headers);
      if (err.IsOk()) {
        err = nic::ServerHealthHttpContext::Create(
            &(lctx->ctx), std::string(url), http_headers, verbose);
      }
    } else {
      err = nic::ServerHealthGrpcContext::Create(
          &(lctx->ctx), std::string(url), verbose);
    }

    if (err.IsOk()) {
      *ctx = lctx;
      return nullptr;
    }

    delete lctx;
  }

  *ctx = nullptr;
  return new nic::Error(err);
}

void
ServerHealthContextDelete(ServerHealthContextCtx* ctx)
{
  delete ctx;
}

nic::Error*
ServerHealthContextGetReady(ServerHealthContextCtx* ctx, bool* ready)
{
  nic::Error err = ctx->ctx->GetReady(ready);
  if (err.IsOk()) {
    return nullptr;
  }

  return new nic::Error(err);
}

nic::Error*
ServerHealthContextGetLive(ServerHealthContextCtx* ctx, bool* live)
{
  nic::Error err = ctx->ctx->GetLive(live);
  if (err.IsOk()) {
    return nullptr;
  }

  return new nic::Error(err);
}

//==============================================================================
struct ServerStatusContextCtx {
  std::unique_ptr<nic::ServerStatusContext> ctx;
  std::string status_buf;
};

nic::Error*
ServerStatusContextNew(
    ServerStatusContextCtx** ctx, const char* url, int protocol_int,
    const char** headers, int num_headers, const char* model_name, bool verbose)
{
  nic::Error err;
  ProtocolType protocol;
  err = ParseProtocol(&protocol, protocol_int);
  if (err.IsOk()) {
    ServerStatusContextCtx* lctx = new ServerStatusContextCtx;
    if (model_name == nullptr) {
      if (protocol == ProtocolType::HTTP) {
        std::map<std::string, std::string> http_headers;
        err = ParseHttpHeaders(&http_headers, headers, num_headers);
        if (err.IsOk()) {
          err = nic::ServerStatusHttpContext::Create(
              &(lctx->ctx), std::string(url), http_headers, verbose);
        }
      } else {
        err = nic::ServerStatusGrpcContext::Create(
            &(lctx->ctx), std::string(url), verbose);
      }
    } else {
      if (protocol == ProtocolType::HTTP) {
        std::map<std::string, std::string> http_headers;
        err = ParseHttpHeaders(&http_headers, headers, num_headers);
        if (err.IsOk()) {
          err = nic::ServerStatusHttpContext::Create(
              &(lctx->ctx), std::string(url), http_headers,
              std::string(model_name), verbose);
        }
      } else {
        err = nic::ServerStatusGrpcContext::Create(
            &(lctx->ctx), std::string(url), std::string(model_name), verbose);
      }
    }

    if (err.IsOk()) {
      *ctx = lctx;
      return nullptr;
    }

    delete lctx;
  }

  *ctx = nullptr;
  return new nic::Error(err);
}

void
ServerStatusContextDelete(ServerStatusContextCtx* ctx)
{
  delete ctx;
}

nic::Error*
ServerStatusContextGetServerStatus(
    ServerStatusContextCtx* ctx, char** status, uint32_t* status_len)
{
  ctx->status_buf.clear();

  ni::ServerStatus server_status;
  nic::Error err = ctx->ctx->GetServerStatus(&server_status);
  if (err.IsOk()) {
    ctx->status_buf = server_status.ShortDebugString();
    *status = &ctx->status_buf[0];
    *status_len = ctx->status_buf.size();
  }

  return new nic::Error(err);
}

//==============================================================================
struct ModelRepositoryContextCtx {
  std::unique_ptr<nic::ModelRepositoryContext> ctx;
  std::string index_buf;
};

nic::Error*
ModelRepositoryContextNew(
    ModelRepositoryContextCtx** ctx, const char* url, int protocol_int,
    const char** headers, int num_headers, bool verbose)
{
  nic::Error err;
  ProtocolType protocol;
  err = ParseProtocol(&protocol, protocol_int);
  if (err.IsOk()) {
    ModelRepositoryContextCtx* lctx = new ModelRepositoryContextCtx;
    if (protocol == ProtocolType::HTTP) {
      std::map<std::string, std::string> http_headers;
      err = ParseHttpHeaders(&http_headers, headers, num_headers);
      if (err.IsOk()) {
        err = nic::ModelRepositoryHttpContext::Create(
            &(lctx->ctx), std::string(url), http_headers, verbose);
      }
    } else {
      err = nic::ModelRepositoryGrpcContext::Create(
          &(lctx->ctx), std::string(url), verbose);
    }
    if (err.IsOk()) {
      *ctx = lctx;
      return nullptr;
    }

    delete lctx;
  }

  *ctx = nullptr;
  return new nic::Error(err);
}

void
ModelRepositoryContextDelete(ModelRepositoryContextCtx* ctx)
{
  delete ctx;
}

nic::Error*
ModelRepositoryContextGetModelRepositoryIndex(
    ModelRepositoryContextCtx* ctx, char** index, uint32_t* index_len)
{
  ctx->index_buf.clear();

  ni::ModelRepositoryIndex repository_index;
  nic::Error err = ctx->ctx->GetModelRepositoryIndex(&repository_index);
  if (err.IsOk()) {
    if (repository_index.SerializeToString(&ctx->index_buf)) {
      *index = &ctx->index_buf[0];
      *index_len = ctx->index_buf.size();
    } else {
      err = nic::Error(
          ni::RequestStatusCode::INTERNAL,
          "failed to parse model repository index");
    }
  }

  return new nic::Error(err);
}

//==============================================================================
struct ModelControlContextCtx {
  std::unique_ptr<nic::ModelControlContext> ctx;
};

nic::Error*
ModelControlContextNew(
    ModelControlContextCtx** ctx, const char* url, int protocol_int,
    const char** headers, int num_headers, bool verbose)
{
  nic::Error err;
  ProtocolType protocol;
  err = ParseProtocol(&protocol, protocol_int);
  if (err.IsOk()) {
    ModelControlContextCtx* lctx = new ModelControlContextCtx;
    if (protocol == ProtocolType::HTTP) {
      std::map<std::string, std::string> http_headers;
      err = ParseHttpHeaders(&http_headers, headers, num_headers);
      if (err.IsOk()) {
        err = nic::ModelControlHttpContext::Create(
            &(lctx->ctx), std::string(url), http_headers, verbose);
      }
    } else {
      err = nic::ModelControlGrpcContext::Create(
          &(lctx->ctx), std::string(url), verbose);
    }

    if (err.IsOk()) {
      *ctx = lctx;
      return nullptr;
    }

    delete lctx;
  }

  *ctx = nullptr;
  return new nic::Error(err);
}

void
ModelControlContextDelete(ModelControlContextCtx* ctx)
{
  delete ctx;
}

nic::Error*
ModelControlContextLoad(ModelControlContextCtx* ctx, const char* model_name)
{
  nic::Error err = ctx->ctx->Load(std::string(model_name));
  if (err.IsOk()) {
    return nullptr;
  }

  return new nic::Error(err);
}

nic::Error*
ModelControlContextUnload(ModelControlContextCtx* ctx, const char* model_name)
{
  nic::Error err = ctx->ctx->Unload(std::string(model_name));
  if (err.IsOk()) {
    return nullptr;
  }

  return new nic::Error(err);
}

//==============================================================================
struct SharedMemoryControlContextCtx {
  std::unique_ptr<nic::SharedMemoryControlContext> ctx;
  std::string status_buf;
};

nic::Error*
SharedMemoryControlContextNew(
    SharedMemoryControlContextCtx** ctx, const char* url, int protocol_int,
    const char** headers, int num_headers, bool verbose)
{
  nic::Error err;
  ProtocolType protocol;
  err = ParseProtocol(&protocol, protocol_int);
  if (err.IsOk()) {
    SharedMemoryControlContextCtx* lctx = new SharedMemoryControlContextCtx;
    if (protocol == ProtocolType::HTTP) {
      std::map<std::string, std::string> http_headers;
      err = ParseHttpHeaders(&http_headers, headers, num_headers);
      if (err.IsOk()) {
        err = nic::SharedMemoryControlHttpContext::Create(
            &(lctx->ctx), std::string(url), http_headers, verbose);
      }
    } else {
      err = nic::SharedMemoryControlGrpcContext::Create(
          &(lctx->ctx), std::string(url), verbose);
    }

    if (err.IsOk()) {
      *ctx = lctx;
      return nullptr;
    }

    delete lctx;
  }

  *ctx = nullptr;
  return new nic::Error(err);
}

void
SharedMemoryControlContextDelete(SharedMemoryControlContextCtx* ctx)
{
  delete ctx;
}

nic::Error*
SharedMemoryControlContextRegister(
    SharedMemoryControlContextCtx* ctx, void* shm_handle)
{
  SharedMemoryHandle* handle =
      reinterpret_cast<SharedMemoryHandle*>(shm_handle);
  nic::Error err = ctx->ctx->RegisterSharedMemory(
      handle->trtis_shm_name_, handle->shm_key_, handle->offset_,
      handle->byte_size_);
  if (err.IsOk()) {
    return nullptr;
  }

  return new nic::Error(err);
}

#ifdef TRTIS_ENABLE_GPU
nic::Error*
SharedMemoryControlContextCudaRegister(
    SharedMemoryControlContextCtx* ctx, void* cuda_shm_handle)
{
  SharedMemoryHandle* handle =
      reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle);
  nic::Error err = ctx->ctx->RegisterCudaSharedMemory(
      handle->trtis_shm_name_, handle->cuda_shm_handle_, handle->byte_size_,
      handle->device_id_);
  if (err.IsOk()) {
    return nullptr;
  }

  return new nic::Error(err);
}
#endif  // TRTIS_ENABLE_GPU

nic::Error*
SharedMemoryControlContextUnregister(
    SharedMemoryControlContextCtx* ctx, void* shm_handle)
{
  SharedMemoryHandle* handle =
      reinterpret_cast<SharedMemoryHandle*>(shm_handle);
  nic::Error err = ctx->ctx->UnregisterSharedMemory(handle->trtis_shm_name_);
  if (err.IsOk()) {
    return nullptr;
  }

  return new nic::Error(err);
}

nic::Error*
SharedMemoryControlContextUnregisterAll(SharedMemoryControlContextCtx* ctx)
{
  nic::Error err = ctx->ctx->UnregisterAllSharedMemory();
  if (err.IsOk()) {
    return nullptr;
  }

  return new nic::Error(err);
}

nic::Error*
SharedMemoryControlContextGetStatus(
    SharedMemoryControlContextCtx* ctx, char** status, uint32_t* status_len)
{
  ctx->status_buf.clear();

  ni::SharedMemoryStatus shm_status;
  nic::Error err = ctx->ctx->GetSharedMemoryStatus(&shm_status);
  if (err.IsOk()) {
    ctx->status_buf = shm_status.ShortDebugString();
    *status = &ctx->status_buf[0];
    *status_len = ctx->status_buf.size();
  }

  return new nic::Error(err);
}

nic::Error*
SharedMemoryControlContextGetSharedMemoryHandleInfo(
    void* shm_handle, char** shm_addr, const char** shm_key, int* shm_fd,
    size_t* offset, size_t* byte_size)
{
  SharedMemoryHandle* handle =
      reinterpret_cast<SharedMemoryHandle*>(shm_handle);
  if (handle->shm_key_ == "") {
#ifdef TRTIS_ENABLE_GPU
    // Must call SharedMemoryControlContextReleaseBuffer to destroy 'new' object
    // after writing into results. Numpy cannot read buffer from GPU and hence
    // this is needed to maintain a copy of the data on GPU shared memory.
    *shm_addr = new char[handle->byte_size_];
    cudaError_t err = cudaMemcpy(
        *shm_addr, handle->base_addr_, handle->byte_size_,
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      return new nic::Error(
          ni::RequestStatusCode::INTERNAL,
          "failed to read GPU shared memory results: " +
              std::string(cudaGetErrorString(err)));
    }
#endif  // TRTIS_ENABLE_GPU
  } else {
    *shm_addr = reinterpret_cast<char*>(handle->base_addr_);
  }
  *shm_key = handle->shm_key_.c_str();
  *shm_fd = handle->shm_fd_;
  *offset = handle->offset_;
  *byte_size = handle->byte_size_;
  return nullptr;
}

nic::Error*
SharedMemoryControlContextReleaseBuffer(void* shm_handle, char* ptr)
{
  SharedMemoryHandle* handle =
      reinterpret_cast<SharedMemoryHandle*>(shm_handle);
  // Only destroy in case of GPU shared memory since it is a temporary buffer
  if (handle->shm_key_ == "") {
    if (ptr) {
      delete ptr;
    }
  }
  return nullptr;
}

//==============================================================================
struct InferContextCtx {
  std::unique_ptr<nic::InferContext> ctx;
  nic::InferContext::ResultMap results;
  std::unordered_map<size_t, nic::InferContext::ResultMap> async_results;
  std::unordered_map<size_t, std::shared_ptr<nic::InferContext::Request>>
      requests;
  std::mutex mu;
};

nic::Error*
InferContextNew(
    InferContextCtx** ctx, const char* url, int protocol_int,
    const char** headers, int num_headers, const char* model_name,
    int64_t model_version, ni::CorrelationID correlation_id, bool streaming,
    bool verbose)
{
  nic::Error err;
  ProtocolType protocol;
  err = ParseProtocol(&protocol, protocol_int);
  if (err.IsOk()) {
    if (streaming && protocol != ProtocolType::GRPC) {
      return new nic::Error(
          ni::RequestStatusCode::INVALID_ARG,
          "Streaming is only allowed with gRPC protocol");
    }
    InferContextCtx* lctx = new InferContextCtx;
    if (streaming) {
      err = nic::InferGrpcStreamContext::Create(
          &(lctx->ctx), correlation_id, std::string(url),
          std::string(model_name), model_version, verbose);
    } else if (protocol == ProtocolType::HTTP) {
      std::map<std::string, std::string> http_headers;
      err = ParseHttpHeaders(&http_headers, headers, num_headers);
      if (err.IsOk()) {
        err = nic::InferHttpContext::Create(
            &(lctx->ctx), correlation_id, std::string(url), http_headers,
            std::string(model_name), model_version, verbose);
      }
    } else {
      err = nic::InferGrpcContext::Create(
          &(lctx->ctx), correlation_id, std::string(url),
          std::string(model_name), model_version, verbose);
    }

    if (err.IsOk()) {
      *ctx = lctx;
      return nullptr;
    }
    delete lctx;
  }

  *ctx = nullptr;
  return new nic::Error(err);
}

void
InferContextDelete(InferContextCtx* ctx)
{
  delete ctx;
}

nic::Error*
InferContextSetOptions(
    InferContextCtx* ctx, nic::InferContext::Options* options)
{
  nic::Error err = ctx->ctx->SetRunOptions(*options);
  return new nic::Error(err);
}

nic::Error*
InferContextRun(InferContextCtx* ctx)
{
  ctx->results.clear();
  nic::Error err = ctx->ctx->Run(&ctx->results);
  return new nic::Error(err);
}

nic::Error*
InferContextAsyncRun(
    InferContextCtx* ctx, void (*callback)(InferContextCtx*, uint64_t))
{
  nic::Error err = ctx->ctx->AsyncRun(
      [ctx, callback](
          nic::InferContext*,
          std::shared_ptr<nic::InferContext::Request> request) {
        {
          std::lock_guard<std::mutex> lock(ctx->mu);
          ctx->requests.emplace(request->Id(), request);
        }

        (*callback)(ctx, request->Id());
      });

  return new nic::Error(err);
}

nic::Error*
InferContextGetAsyncRunResults(InferContextCtx* ctx, uint64_t request_id)
{
  std::lock_guard<std::mutex> lock(ctx->mu);

  auto itr = ctx->requests.find(request_id);
  if (itr != ctx->requests.end()) {
    nic::InferContext::ResultMap results;
    nic::Error err = ctx->ctx->GetAsyncRunResults(itr->second, &results);
    ctx->requests.erase(itr);
    ctx->async_results.emplace(request_id, std::move(results));

    return new nic::Error(err);
  }

  return new nic::Error(
      ni::RequestStatusCode::INVALID_ARG,
      "The request ID doesn't match any existing asynchronous requests");
}

//==============================================================================
nic::Error*
InferContextOptionsNew(
    nic::InferContext::Options** ctx, uint32_t flags, uint64_t batch_size,
    ni::CorrelationID corr_id = 0, uint32_t priority = 0,
    uint64_t timeout_ms = 0)
{
  std::unique_ptr<nic::InferContext::Options> uctx;
  nic::Error err = nic::InferContext::Options::Create(&uctx);
  if (err.IsOk()) {
    *ctx = uctx.release();
    (*ctx)->SetFlags(flags);
    (*ctx)->SetBatchSize(batch_size);
    (*ctx)->SetCorrelationId(corr_id);
    (*ctx)->SetPriority(priority);
    (*ctx)->SetTimeout(timeout_ms);
    return nullptr;
  }

  *ctx = nullptr;
  return new nic::Error(err);
}

void
InferContextOptionsDelete(nic::InferContext::Options* ctx)
{
  delete ctx;
}

nic::Error*
InferContextOptionsAddRaw(
    InferContextCtx* infer_ctx, nic::InferContext::Options* ctx,
    const char* output_name)
{
  std::shared_ptr<nic::InferContext::Output> output;
  nic::Error err = infer_ctx->ctx->GetOutput(std::string(output_name), &output);
  if (err.IsOk()) {
    err = ctx->AddRawResult(output);
  }

  return new nic::Error(err);
}

nic::Error*
InferContextOptionsAddClass(
    InferContextCtx* infer_ctx, nic::InferContext::Options* ctx,
    const char* output_name, uint64_t count)
{
  std::shared_ptr<nic::InferContext::Output> output;
  nic::Error err = infer_ctx->ctx->GetOutput(std::string(output_name), &output);
  if (err.IsOk()) {
    err = ctx->AddClassResult(output, count);
  }

  return new nic::Error(err);
}

nic::Error*
InferContextOptionsAddSharedMemory(
    InferContextCtx* infer_ctx, nic::InferContext::Options* ctx,
    const char* output_name, void* shm_handle)
{
  std::shared_ptr<nic::InferContext::Output> output;
  nic::Error err = infer_ctx->ctx->GetOutput(std::string(output_name), &output);
  SharedMemoryHandle* handle =
      reinterpret_cast<SharedMemoryHandle*>(shm_handle);
  if (err.IsOk()) {
    err = ctx->AddSharedMemoryResult(
        output, handle->trtis_shm_name_, handle->offset_, handle->byte_size_);
  }

  return new nic::Error(err);
}

ni::CorrelationID
CorrelationId(InferContextCtx* infer_ctx)
{
  return infer_ctx->ctx->CorrelationId();
}

//==============================================================================
struct InferContextInputCtx {
  std::shared_ptr<nic::InferContext::Input> input;
};

nic::Error*
InferContextInputNew(
    InferContextInputCtx** ctx, InferContextCtx* infer_ctx,
    const char* input_name)
{
  InferContextInputCtx* lctx = new InferContextInputCtx;
  nic::Error err =
      infer_ctx->ctx->GetInput(std::string(input_name), &lctx->input);
  if (err.IsOk()) {
    lctx->input->Reset();
  }
  *ctx = lctx;
  return new nic::Error(err);
}

void
InferContextInputDelete(InferContextInputCtx* ctx)
{
  delete ctx;
}

nic::Error*
InferContextInputSetShape(
    InferContextInputCtx* ctx, const int64_t* dims, uint64_t size)
{
  std::vector<int64_t> shape;
  shape.reserve(size);
  for (uint64_t i = 0; i < size; ++i) {
    shape.push_back(dims[i]);
  }

  nic::Error err = ctx->input->SetShape(shape);
  return new nic::Error(err);
}

nic::Error*
InferContextInputSetRaw(
    InferContextInputCtx* ctx, const void* data, uint64_t byte_size)
{
  nic::Error err =
      ctx->input->SetRaw(reinterpret_cast<const uint8_t*>(data), byte_size);
  return new nic::Error(err);
}

nic::Error*
InferContextInputSetSharedMemory(InferContextInputCtx* ctx, void* shm_handle)
{
  SharedMemoryHandle* handle =
      reinterpret_cast<SharedMemoryHandle*>(shm_handle);
  nic::Error err = ctx->input->SetSharedMemory(
      handle->trtis_shm_name_, handle->offset_, handle->byte_size_);
  return new nic::Error(err);
}

//==============================================================================
struct InferContextResultCtx {
  std::unique_ptr<nic::InferContext::Result> result;
  nic::InferContext::Result::ClassResult cr;
};

nic::Error*
InferContextResultNew(
    InferContextResultCtx** ctx, InferContextCtx* infer_ctx,
    const char* result_name)
{
  InferContextResultCtx* lctx = new InferContextResultCtx;

  auto itr = infer_ctx->results.find(result_name);
  if ((itr == infer_ctx->results.end()) || (itr->second == nullptr)) {
    return new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "unable to find result for output '" + std::string(result_name) + "'");
  }

  lctx->result.swap(itr->second);
  *ctx = lctx;
  return nullptr;
}

nic::Error*
InferContextAsyncResultNew(
    InferContextResultCtx** ctx, InferContextCtx* infer_ctx,
    const uint64_t request_id, const char* result_name)
{
  std::lock_guard<std::mutex> lock(infer_ctx->mu);

  auto res_itr = infer_ctx->async_results.find(request_id);
  if (res_itr == infer_ctx->async_results.end()) {
    return new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "unable to find results for request '" + std::to_string(request_id) +
            "'");
  }

  auto itr = res_itr->second.find(result_name);
  if ((itr == res_itr->second.end()) || (itr->second == nullptr)) {
    return new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "unable to find result for output '" + std::string(result_name) + "'");
  }

  InferContextResultCtx* lctx = new InferContextResultCtx;
  lctx->result.swap(itr->second);
  *ctx = lctx;

  // clean up async_requests if all outputs are retrieved
  res_itr->second.erase(itr);
  if (res_itr->second.empty()) {
    infer_ctx->async_results.erase(res_itr);
  }

  return nullptr;
}

void
InferContextResultDelete(InferContextResultCtx* ctx)
{
  delete ctx;
}

nic::Error*
InferContextResultModelName(InferContextResultCtx* ctx, const char** model_name)
{
  if (ctx->result == nullptr) {
    return new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "model name not available for empty result");
  }

  *model_name = ctx->result->ModelName().c_str();
  return nullptr;
}

nic::Error*
InferContextResultModelVersion(
    InferContextResultCtx* ctx, int64_t* model_version)
{
  if (ctx->result == nullptr) {
    return new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "model version not available for empty result");
  }

  *model_version = ctx->result->ModelVersion();
  return nullptr;
}

nic::Error*
InferContextResultDataType(InferContextResultCtx* ctx, uint32_t* dtype)
{
  if (ctx->result == nullptr) {
    return new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "datatype not available for empty result");
  }

  ni::DataType data_type = ctx->result->GetOutput()->DType();
  *dtype = static_cast<uint32_t>(data_type);

  return nullptr;
}

nic::Error*
InferContextResultShape(
    InferContextResultCtx* ctx, uint64_t max_dims, int64_t* shape,
    uint64_t* shape_len)
{
  if (ctx->result == nullptr) {
    return new nic::Error(
        ni::RequestStatusCode::INTERNAL, "dims not available for empty result");
  }

  std::vector<int64_t> rshape;
  nic::Error err = ctx->result->GetRawShape(&rshape);
  if (!err.IsOk()) {
    return new nic::Error(err);
  }

  if (rshape.size() > max_dims) {
    return new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "number of dimensions in result shape exceeds maximum of " +
            std::to_string(max_dims));
  }

  size_t cnt = 0;
  for (auto dim : rshape) {
    shape[cnt++] = dim;
  }

  *shape_len = rshape.size();

  return nullptr;
}

nic::Error*
InferContextResultNextRaw(
    InferContextResultCtx* ctx, size_t batch_idx, const char** val,
    uint64_t* val_len)
{
  if (ctx->result == nullptr) {
    return new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "no raw result available for empty result");
  }

  const uint8_t* content;
  size_t content_byte_size;
  nic::Error err = ctx->result->GetRaw(batch_idx, &content, &content_byte_size);
  if (err.IsOk()) {
    *val = reinterpret_cast<const char*>(content);
    *val_len = content_byte_size;
  }

  return new nic::Error(err);
}

nic::Error*
InferContextResultClassCount(
    InferContextResultCtx* ctx, size_t batch_idx, uint64_t* count)
{
  if (ctx->result == nullptr) {
    return new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "no classes available for empty result");
  }

  nic::Error err = ctx->result->GetClassCount(batch_idx, count);
  return new nic::Error(err);
}

nic::Error*
InferContextResultNextClass(
    InferContextResultCtx* ctx, size_t batch_idx, uint64_t* idx, float* prob,
    const char** label)
{
  if (ctx->result == nullptr) {
    return new nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "no classes available for empty result");
  }

  nic::Error err = ctx->result->GetClassAtCursor(batch_idx, &ctx->cr);
  if (err.IsOk()) {
    auto& cr = ctx->cr;
    *idx = cr.idx;
    *prob = cr.value;
    *label = cr.label.c_str();
  }

  return new nic::Error(err);
}

//==============================================================================
nic::Error*
InferContextGetStat(
    InferContextCtx* ctx, uint64_t* completed_request_count,
    uint64_t* cumulative_total_request_time_ns,
    uint64_t* cumulative_send_time_ns, uint64_t* cumulative_receive_time_ns)
{
  nic::InferContext::Stat stat;
  nic::Error err = ctx->ctx->GetStat(&stat);
  if (err.IsOk()) {
    *completed_request_count = stat.completed_request_count;
    *cumulative_total_request_time_ns = stat.cumulative_total_request_time_ns;
    *cumulative_send_time_ns = stat.cumulative_send_time_ns;
    *cumulative_receive_time_ns = stat.cumulative_receive_time_ns;
  }
  return new nic::Error(err);
}
