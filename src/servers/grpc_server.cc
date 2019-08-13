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

#include "src/servers/grpc_server.h"

#include <cstdint>
#include <map>
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "grpc/grpc.h"
#include "src/core/constants.h"
#include "src/core/grpc_service.grpc.pb.h"
#include "src/core/trtserver.h"
#include "src/nvrpc/Context.h"
#include "src/nvrpc/Executor.h"
#include "src/nvrpc/Resources.h"
#include "src/nvrpc/Service.h"
#include "src/nvrpc/ThreadPool.h"
#include "src/servers/common.h"

using nvrpc::BaseContext;
using nvrpc::BidirectionalStreamingLifeCycle;
using nvrpc::Context;
using nvrpc::LifeCycleUnary;
using nvrpc::ThreadPool;

namespace nvidia { namespace inferenceserver {
namespace {
class AsyncResources : public nvrpc::Resources {
 public:
  AsyncResources(
      const std::shared_ptr<TRTSERVER_Server>& server,
      const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
      int infer_threads, int mgmt_threads);
  ~AsyncResources();

  TRTSERVER_Server* Server() const { return server_.get(); }
  const char* ServerId() const { return server_id_; }

  const std::shared_ptr<SharedMemoryBlockManager>& SharedMemoryManager() const
  {
    return smb_manager_;
  }

  TRTSERVER_ResponseAllocator* Allocator() const { return allocator_; }

  ThreadPool& GetMgmtThreadPool() { return mgmt_thread_pool_; }
  ThreadPool& GetInferThreadPool() { return infer_thread_pool_; }

 private:
  static TRTSERVER_Error* ResponseAlloc(
      TRTSERVER_ResponseAllocator* allocator, void** buffer,
      void** buffer_userp, const char* tensor_name, size_t byte_size,
      TRTSERVER_Memory_Type memory_type, int64_t memory_type_id, void* userp);
  static TRTSERVER_Error* ResponseRelease(
      TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
      size_t byte_size, TRTSERVER_Memory_Type memory_type,
      int64_t memory_type_id);

  std::shared_ptr<TRTSERVER_Server> server_;
  const char* server_id_;

  std::shared_ptr<SharedMemoryBlockManager> smb_manager_;

  // The allocator that will be used to allocate buffers for the
  // inference result tensors.
  TRTSERVER_ResponseAllocator* allocator_;

  // We can and should get specific on thread affinity.  It might not
  // be as important on the frontend, but the backend threadpool
  // should be aligned with the respective devices.
  ThreadPool mgmt_thread_pool_;
  ThreadPool infer_thread_pool_;
};

AsyncResources::AsyncResources(
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager,
    int infer_threads, int mgmt_threads)
    : server_(server), smb_manager_(smb_manager), allocator_(nullptr),
      mgmt_thread_pool_(mgmt_threads), infer_thread_pool_(infer_threads)
{
  TRTSERVER_Error* err = TRTSERVER_ServerId(server_.get(), &server_id_);
  if (err != nullptr) {
    server_id_ = "unknown:0";
    TRTSERVER_ErrorDelete(err);
  }

  // Create the allocator that will be used to allocate buffers for
  // the result tensors.
  FAIL_IF_ERR(
      TRTSERVER_ResponseAllocatorNew(
          &allocator_, ResponseAlloc, ResponseRelease),
      "creating response allocator");
}

AsyncResources::~AsyncResources()
{
  LOG_IF_ERR(
      TRTSERVER_ResponseAllocatorDelete(allocator_),
      "deleting response allocator");
}

TRTSERVER_Error*
AsyncResources::ResponseAlloc(
    TRTSERVER_ResponseAllocator* allocator, void** buffer, void** buffer_userp,
    const char* tensor_name, size_t byte_size,
    TRTSERVER_Memory_Type memory_type, int64_t memory_type_id, void* userp)
{
  InferResponse* response = reinterpret_cast<InferResponse*>(userp);

  *buffer = nullptr;
  *buffer_userp = nullptr;

  // Can't allocate for any memory type other than CPU.
  if (memory_type != TRTSERVER_MEMORY_CPU) {
    LOG_VERBOSE(1) << "GRPC allocation failed for type " << memory_type
                   << " for " << tensor_name;
    return nullptr;  // Success
  }

  // Called once for each result tensor in the inference request. Must
  // always add a raw output into the response's list of outputs so
  // that the number and order of raw output entries equals the output
  // meta-data.
  std::string* raw_output = response->add_raw_output();
  if (byte_size > 0) {
    raw_output->resize(byte_size);
    *buffer = static_cast<void*>(&((*raw_output)[0]));
  }

  LOG_VERBOSE(1) << "GRPC allocation: " << tensor_name << ", size " << byte_size
                 << ", addr " << *buffer;

  return nullptr;  // Success
}

TRTSERVER_Error*
AsyncResources::ResponseRelease(
    TRTSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRTSERVER_Memory_Type memory_type, int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "GRPC release: "
                 << "size " << byte_size << ", addr " << buffer;

  // Don't do anything when releasing a buffer since ResponseAlloc
  // wrote directly into the response protobuf.
  return nullptr;  // Success
}

static std::shared_ptr<AsyncResources> g_Resources;

class StatusContext final
    : public Context<StatusRequest, StatusResponse, AsyncResources> {
  void ExecuteRPC(
      StatusRequest& request, StatusResponse& response) final override
  {
    uintptr_t execution_context = this->GetExecutionContext();
    GetResources()->GetMgmtThreadPool().enqueue([this, execution_context,
                                                 &request, &response] {
      TRTSERVER_Server* server = GetResources()->Server();

      TRTSERVER_Protobuf* server_status_protobuf = nullptr;
      TRTSERVER_Error* err =
          (request.model_name().empty())
              ? TRTSERVER_ServerStatus(server, &server_status_protobuf)
              : TRTSERVER_ServerModelStatus(
                    server, request.model_name().c_str(),
                    &server_status_protobuf);
      if (err == nullptr) {
        const char* status_buffer;
        size_t status_byte_size;
        err = TRTSERVER_ProtobufSerialize(
            server_status_protobuf, &status_buffer, &status_byte_size);
        if (err == nullptr) {
          if (!response.mutable_server_status()->ParseFromArray(
                  status_buffer, status_byte_size)) {
            err = TRTSERVER_ErrorNew(
                TRTSERVER_ERROR_UNKNOWN, "failed to parse server status");
          }
        }
      }

      TRTSERVER_ProtobufDelete(server_status_protobuf);

      RequestStatusUtil::Create(
          response.mutable_request_status(), err,
          RequestStatusUtil::NextUniqueRequestId(), GetResources()->ServerId());

      TRTSERVER_ErrorDelete(err);
      this->CompleteExecution(execution_context);
    });
  }
};

class ModelControlContext final
    : public Context<
          ModelControlRequest, ModelControlResponse, AsyncResources> {
  void ExecuteRPC(ModelControlRequest& request, ModelControlResponse& response)
      final override
  {
    uintptr_t execution_context = this->GetExecutionContext();
    GetResources()->GetMgmtThreadPool().enqueue([this, execution_context,
                                                 &request, &response] {
      TRTSERVER_Server* server = GetResources()->Server();

      TRTSERVER_Error* err = nullptr;
      if (request.type() == ModelControlRequest::LOAD) {
        err = TRTSERVER_ServerLoadModel(server, request.model_name().c_str());
      } else {
        err = TRTSERVER_ServerUnloadModel(server, request.model_name().c_str());
      }

      RequestStatusUtil::Create(
          response.mutable_request_status(), err,
          RequestStatusUtil::NextUniqueRequestId(), GetResources()->ServerId());

      TRTSERVER_ErrorDelete(err);
      this->CompleteExecution(execution_context);
    });
  }
};

class SharedMemoryControlContext final
    : public Context<
          SharedMemoryControlRequest, SharedMemoryControlResponse,
          AsyncResources> {
  void ExecuteRPC(
      SharedMemoryControlRequest& request,
      SharedMemoryControlResponse& response) final override
  {
    uintptr_t execution_context = this->GetExecutionContext();
    GetResources()->GetMgmtThreadPool().enqueue([this, execution_context,
                                                 &request, &response] {
      TRTSERVER_Server* server = GetResources()->Server();
      TRTSERVER_SharedMemoryBlock* smb = nullptr;

      TRTSERVER_Error* err = nullptr;
      switch (request.type()) {
        case SharedMemoryControlRequest::REGISTER:
          err = GetResources()->SharedMemoryManager()->Create(
              &smb, request.shared_memory_region().name(),
              request.shared_memory_region().shm_key(),
              request.shared_memory_region().offset(),
              request.shared_memory_region().byte_size());
          if (err == nullptr) {
            err = TRTSERVER_ServerRegisterSharedMemory(server, smb);
          }
          break;
        case SharedMemoryControlRequest::UNREGISTER:
          err = GetResources()->SharedMemoryManager()->Remove(
              &smb, request.shared_memory_region().name());
          if ((err == nullptr) && (smb != nullptr)) {
            err = TRTSERVER_ServerUnregisterSharedMemory(server, smb);
            TRTSERVER_Error* del_err = TRTSERVER_SharedMemoryBlockDelete(smb);
            if (del_err != nullptr) {
              LOG_ERROR << "failed to delete shared memory block: "
                        << TRTSERVER_ErrorMessage(del_err);
            }
          }
          break;
        case SharedMemoryControlRequest::UNREGISTER_ALL:
          err = GetResources()->SharedMemoryManager()->Clear();
          if (err == nullptr) {
            err = TRTSERVER_ServerUnregisterAllSharedMemory(server);
          }
          break;
        default:
          err = TRTSERVER_ErrorNew(
              TRTSERVER_ERROR_UNKNOWN,
              "unknown sharedmemorycontrol request type");
          break;
      }

      RequestStatusUtil::Create(
          response.mutable_request_status(), err,
          RequestStatusUtil::NextUniqueRequestId(), GetResources()->ServerId());

      TRTSERVER_ErrorDelete(err);
      this->CompleteExecution(execution_context);
    });
  }
};

template <class LifeCycle>
class InferBaseContext : public BaseContext<LifeCycle, AsyncResources> {
  class GRPCInferRequest {
   public:
    GRPCInferRequest(
        InferBaseContext<LifeCycle>* ctx, uintptr_t exec_ctx,
        InferResponse& response, uint64_t request_id, const char* server_id,
        uint64_t unique_id)
        : ctx_(ctx), exec_ctx_(exec_ctx), response_(response),
          request_id_(request_id), server_id_(server_id), unique_id_(unique_id)
    {
    }

    static void InferComplete(
        TRTSERVER_Server* server, TRTSERVER_InferenceResponse* response,
        void* userp)
    {
      std::unique_ptr<GRPCInferRequest> grpc_infer_request(
          reinterpret_cast<GRPCInferRequest*>(userp));

      TRTSERVER_Error* response_status =
          TRTSERVER_InferenceResponseStatus(response);
      if ((response_status == nullptr) &&
          (grpc_infer_request->response_.ByteSizeLong() > INT_MAX)) {
        response_status = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            std::string(
                "Response has byte size " +
                std::to_string(grpc_infer_request->response_.ByteSizeLong()) +
                " which exceed gRPC's byte size limit " +
                std::to_string(INT_MAX) + ".")
                .c_str());
      }

      if (response_status == nullptr) {
        TRTSERVER_Protobuf* response_protobuf = nullptr;
        response_status =
            TRTSERVER_InferenceResponseHeader(response, &response_protobuf);
        if (response_status == nullptr) {
          const char* buffer;
          size_t byte_size;
          response_status = TRTSERVER_ProtobufSerialize(
              response_protobuf, &buffer, &byte_size);
          if (response_status == nullptr) {
            if (!grpc_infer_request->response_.mutable_meta_data()
                     ->ParseFromArray(buffer, byte_size)) {
              response_status = TRTSERVER_ErrorNew(
                  TRTSERVER_ERROR_INTERNAL, "failed to parse response header");
            }
          }

          TRTSERVER_ProtobufDelete(response_protobuf);
        }
      }

      // If the response is an error then clear the meta-data
      // and raw output as they may be partially or
      // un-initialized.
      if (response_status != nullptr) {
        grpc_infer_request->response_.mutable_meta_data()->Clear();
        grpc_infer_request->response_.mutable_raw_output()->Clear();
      }

      RequestStatusUtil::Create(
          grpc_infer_request->response_.mutable_request_status(),
          response_status, grpc_infer_request->unique_id_,
          grpc_infer_request->server_id_);

      LOG_IF_ERR(
          TRTSERVER_InferenceResponseDelete(response),
          "deleting GRPC response");
      TRTSERVER_ErrorDelete(response_status);

      grpc_infer_request->response_.mutable_meta_data()->set_id(
          grpc_infer_request->request_id_);
      grpc_infer_request->ctx_->CompleteExecution(
          grpc_infer_request->exec_ctx_);
    }

   private:
    InferBaseContext<LifeCycle>* ctx_;
    uintptr_t exec_ctx_;
    InferResponse& response_;
    const uint64_t request_id_;
    const char* const server_id_;
    const uint64_t unique_id_;
  };

  TRTSERVER_Error* GRPCToInput(
      TRTSERVER_Server* server, const InferRequestHeader& request_header,
      const InferRequest& request,
      TRTSERVER_InferenceRequestProvider* request_provider)
  {
    // Make sure that the request is providing the same number of raw
    // + shared memory input tensor data.
    int shared_memory_input_count = 0;
    for (const auto& io : request_header.input()) {
      if (io.has_shared_memory()) {
        shared_memory_input_count++;
      }
    }
    if (request_header.input_size() !=
        (request.raw_input_size() + shared_memory_input_count)) {
      return TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_INVALID_ARG,
          std::string(
              "expected tensor data for " +
              std::to_string(request_header.input_size()) + " inputs but got " +
              std::to_string(request.raw_input_size()) +
              " sets of data for model '" + request.model_name() + "'")
              .c_str());
    }

    // Verify that the batch-byte-size of each input matches the size of
    // the provided tensor data (provided raw or from shared memory)
    size_t idx = 0;
    for (const auto& io : request_header.input()) {
      const void* base;
      size_t byte_size;
      if (io.has_shared_memory()) {
        LOG_VERBOSE(1) << io.name() << " has shared memory";
        TRTSERVER_SharedMemoryBlock* smb = nullptr;
        RETURN_IF_ERR(this->GetResources()->SharedMemoryManager()->Get(
            &smb, io.shared_memory().name()));
        RETURN_IF_ERR(TRTSERVER_ServerSharedMemoryAddress(
            server, smb, io.shared_memory().offset(),
            io.shared_memory().byte_size(), const_cast<void**>(&base)));
        byte_size = io.shared_memory().byte_size();
      } else {
        const std::string& raw = request.raw_input(idx++);
        base = raw.c_str();
        byte_size = raw.size();
      }

      uint64_t expected_byte_size = 0;
      RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderInputBatchByteSize(
          request_provider, io.name().c_str(), &expected_byte_size));

      if (byte_size != expected_byte_size) {
        return TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_INVALID_ARG,
            std::string(
                "unexpected size " + std::to_string(byte_size) +
                " for input '" + io.name() + "', expecting " +
                std::to_string(expected_byte_size) + " for model '" +
                request.model_name() + "'")
                .c_str());
      }

      RETURN_IF_ERR(TRTSERVER_InferenceRequestProviderSetInputData(
          request_provider, io.name().c_str(), base, byte_size));
    }

    return nullptr;  // success
  }

  void ExecuteRPC(InferRequest& request, InferResponse& response) final override
  {
    auto server = this->GetResources()->Server();
    auto server_id = this->GetResources()->ServerId();
    uintptr_t execution_context = this->GetExecutionContext();
    uint64_t unique_id = RequestStatusUtil::NextUniqueRequestId();

    TRTSERVER_Error* err = nullptr;

    std::string request_header_serialized;
    if (!request.meta_data().SerializeToString(&request_header_serialized)) {
      err = TRTSERVER_ErrorNew(
          TRTSERVER_ERROR_UNKNOWN, "failed to serialize request header");
    } else {
      // Create the inference request provider which provides all the
      // input information needed for an inference.
      TRTSERVER_InferenceRequestProvider* request_provider = nullptr;
      err = TRTSERVER_InferenceRequestProviderNew(
          &request_provider, server, request.model_name().c_str(),
          request.model_version(), request_header_serialized.c_str(),
          request_header_serialized.size());
      if (err == nullptr) {
        err =
            GRPCToInput(server, request.meta_data(), request, request_provider);
        if (err == nullptr) {
          GRPCInferRequest* grpc_infer_request = new GRPCInferRequest(
              this, execution_context, response, request.meta_data().id(),
              server_id, unique_id);

          err = TRTSERVER_ServerInferAsync(
              server, request_provider, g_Resources->Allocator(),
              &response /* response_allocator_userp */,
              GRPCInferRequest::InferComplete,
              reinterpret_cast<void*>(grpc_infer_request));
          if (err != nullptr) {
            delete grpc_infer_request;
            grpc_infer_request = nullptr;
          }

          // The request provider can be deleted immediately after the
          // ServerInferAsync call returns.
          TRTSERVER_InferenceRequestProviderDelete(request_provider);
        }
      }
    }

    if (err != nullptr) {
      RequestStatusUtil::Create(
          response.mutable_request_status(), err, unique_id, server_id);

      LOG_VERBOSE(1) << "Infer failed: " << TRTSERVER_ErrorMessage(err);
      TRTSERVER_ErrorDelete(err);

      // Clear the meta-data and raw output as they may be partially
      // or un-initialized.
      response.mutable_meta_data()->Clear();
      response.mutable_raw_output()->Clear();

      response.mutable_meta_data()->set_id(request.meta_data().id());
      this->CompleteExecution(execution_context);
    }
  }
};

class InferContext final
    : public InferBaseContext<LifeCycleUnary<InferRequest, InferResponse>> {
};

class StreamInferContext final
    : public InferBaseContext<
          BidirectionalStreamingLifeCycle<InferRequest, InferResponse>> {
};

class ProfileContext final
    : public Context<ProfileRequest, ProfileResponse, AsyncResources> {
  void ExecuteRPC(
      ProfileRequest& request, ProfileResponse& response) final override
  {
    uintptr_t execution_context = this->GetExecutionContext();
    GetResources()->GetMgmtThreadPool().enqueue([this, execution_context,
                                                 &request, &response] {
      // For now profile is a nop...

      RequestStatusUtil::Create(
          response.mutable_request_status(), nullptr /* err */,
          RequestStatusUtil::NextUniqueRequestId(), GetResources()->ServerId());

      this->CompleteExecution(execution_context);
    });
  }
};

class HealthContext final
    : public Context<HealthRequest, HealthResponse, AsyncResources> {
  void ExecuteRPC(
      HealthRequest& request, HealthResponse& response) final override
  {
    uintptr_t execution_context = this->GetExecutionContext();
    GetResources()->GetMgmtThreadPool().enqueue([this, execution_context,
                                                 &request, &response] {
      TRTSERVER_Server* server = GetResources()->Server();

      TRTSERVER_Error* err = nullptr;
      bool health = false;

      if (request.mode() == "live") {
        err = TRTSERVER_ServerIsLive(server, &health);
      } else if (request.mode() == "ready") {
        err = TRTSERVER_ServerIsReady(server, &health);
      } else {
        err = TRTSERVER_ErrorNew(
            TRTSERVER_ERROR_UNKNOWN,
            std::string("unknown health mode '" + request.mode() + "'")
                .c_str());
      }

      response.set_health((err == nullptr) && health);

      RequestStatusUtil::Create(
          response.mutable_request_status(), err,
          RequestStatusUtil::NextUniqueRequestId(), GetResources()->ServerId());

      TRTSERVER_ErrorDelete(err);
      this->CompleteExecution(execution_context);
    });
  }
};

}  // namespace

GRPCServer::GRPCServer(
    const std::string& addr, const int infer_thread_cnt,
    const int stream_infer_thread_cnt)
    : nvrpc::Server(addr), infer_thread_cnt_(infer_thread_cnt),
      stream_infer_thread_cnt_(stream_infer_thread_cnt), running_(false)
{
}

GRPCServer::~GRPCServer()
{
  Stop();
}

TRTSERVER_Error*
GRPCServer::Create(
    const std::shared_ptr<TRTSERVER_Server>& server,
    const std::shared_ptr<SharedMemoryBlockManager>& smb_manager, int32_t port,
    int infer_thread_cnt, int stream_infer_thread_cnt,
    std::unique_ptr<GRPCServer>* grpc_server)
{
  g_Resources = std::make_shared<AsyncResources>(
      server, smb_manager, 1 /* infer threads */, 1 /* mgmt threads */);

  std::string addr = "0.0.0.0:" + std::to_string(port);
  LOG_INFO << "Starting a GRPCService at " << addr;
  grpc_server->reset(
      new GRPCServer(addr, infer_thread_cnt, stream_infer_thread_cnt));

  (*grpc_server)->GetBuilder().SetMaxMessageSize(MAX_GRPC_MESSAGE_SIZE);

  LOG_VERBOSE(1) << "Register TensorRT GRPCService";
  auto inferenceService = (*grpc_server)->RegisterAsyncService<GRPCService>();

  LOG_VERBOSE(1) << "Register Infer RPC";
  (*grpc_server)->rpcInfer_ = inferenceService->RegisterRPC<InferContext>(
      &GRPCService::AsyncService::RequestInfer);

  LOG_VERBOSE(1) << "Register StreamInfer RPC";
  (*grpc_server)->rpcStreamInfer_ =
      inferenceService->RegisterRPC<StreamInferContext>(
          &GRPCService::AsyncService::RequestStreamInfer);

  LOG_VERBOSE(1) << "Register Status RPC";
  (*grpc_server)->rpcStatus_ = inferenceService->RegisterRPC<StatusContext>(
      &GRPCService::AsyncService::RequestStatus);

  LOG_VERBOSE(1) << "Register ModelControl RPC";
  (*grpc_server)->rpcModelControl_ =
      inferenceService->RegisterRPC<ModelControlContext>(
          &GRPCService::AsyncService::RequestModelControl);

  LOG_VERBOSE(1) << "Register SharedMemoryControl RPC";
  (*grpc_server)->rpcSharedMemoryControl_ =
      inferenceService->RegisterRPC<SharedMemoryControlContext>(
          &GRPCService::AsyncService::RequestSharedMemoryControl);

  LOG_VERBOSE(1) << "Register Profile RPC";
  (*grpc_server)->rpcProfile_ = inferenceService->RegisterRPC<ProfileContext>(
      &GRPCService::AsyncService::RequestProfile);

  LOG_VERBOSE(1) << "Register Health RPC";
  (*grpc_server)->rpcHealth_ = inferenceService->RegisterRPC<HealthContext>(
      &GRPCService::AsyncService::RequestHealth);

  return nullptr;
}

TRTSERVER_Error*
GRPCServer::Start()
{
  if (!running_) {
    running_ = true;
    LOG_VERBOSE(1) << "Register Executor";
    auto executor = RegisterExecutor(new ::nvrpc::Executor(1));

    // You can register RPC execution contexts from any registered RPC on any
    // executor.
    executor->RegisterContexts(rpcInfer_, g_Resources, infer_thread_cnt_);
    executor->RegisterContexts(
        rpcStreamInfer_, g_Resources, stream_infer_thread_cnt_);
    executor->RegisterContexts(rpcStatus_, g_Resources, 1);
    executor->RegisterContexts(rpcModelControl_, g_Resources, 1);
    executor->RegisterContexts(rpcSharedMemoryControl_, g_Resources, 1);
    executor->RegisterContexts(rpcHealth_, g_Resources, 1);
    executor->RegisterContexts(rpcProfile_, g_Resources, 1);

    AsyncRun();
    return nullptr;
  }

  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_ALREADY_EXISTS, "GRPC server is already running.");
}

TRTSERVER_Error*
GRPCServer::Stop()
{
  if (running_) {
    running_ = false;
    Shutdown();
    return nullptr;
  }

  return TRTSERVER_ErrorNew(
      TRTSERVER_ERROR_UNAVAILABLE, "GRPC server is not running.");
}

}}  // namespace nvidia::inferenceserver
