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

#include "src/core/grpc_server.h"

#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "grpc/grpc.h"
#include "src/core/constants.h"
#include "src/core/grpc_service.grpc.pb.h"
#include "src/core/logging.h"
#include "src/core/request_status.h"
#include "src/core/server.h"
#include "src/nvrpc/Context.h"
#include "src/nvrpc/Executor.h"
#include "src/nvrpc/Resources.h"
#include "src/nvrpc/Service.h"
#include "src/nvrpc/ThreadPool.h"


using nvrpc::BaseContext;
using nvrpc::BidirectionalStreamingLifeCycle;
using nvrpc::Context;
using nvrpc::LifeCycleUnary;
using nvrpc::ThreadPool;

namespace nvidia { namespace inferenceserver {
namespace {
class AsyncResources : public nvrpc::Resources {
 public:
  explicit AsyncResources(
      InferenceServer* server, int infer_threads, int mgmt_threads)
      : m_Server(server), m_MgmtThreadPool(mgmt_threads),
        m_InferThreadPool(infer_threads)
  {
  }

  InferenceServer* GetServer() { return m_Server; }
  ThreadPool& GetMgmtThreadPool() { return m_MgmtThreadPool; }
  ThreadPool& GetInferThreadPool() { return m_InferThreadPool; }

 private:
  InferenceServer* m_Server;

  // We can and should get specific on thread affinity.  It might not be as
  // important on the frontend, but the backend threadpool should be aligned
  // with the respective devices.
  ThreadPool m_MgmtThreadPool;
  ThreadPool m_InferThreadPool;
};

static std::shared_ptr<AsyncResources> g_Resources;

class StatusContext final
    : public Context<StatusRequest, StatusResponse, AsyncResources> {
  void ExecuteRPC(
      StatusRequest& request, StatusResponse& response) final override
  {
    uintptr_t execution_context = this->GetExecutionContext();
    GetResources()->GetMgmtThreadPool().enqueue(
        [this, execution_context, &request, &response] {
          ServerStatTimerScoped timer(
              GetResources()->GetServer()->StatusManager(),
              ServerStatTimerScoped::Kind::STATUS);

          RequestStatus* request_status = response.mutable_request_status();
          ServerStatus* server_status = response.mutable_server_status();

          GetResources()->GetServer()->HandleStatus(
              request_status, server_status, request.model_name());
          this->CompleteExecution(execution_context);
        });
  }
};

template <class LifeCycle>
class InferBaseContext : public BaseContext<LifeCycle, AsyncResources> {
  void ExecuteRPC(InferRequest& request, InferResponse& response) final override
  {
    uintptr_t execution_context = this->GetExecutionContext();
    auto server = this->GetResources()->GetServer();
    auto infer_stats = std::make_shared<ModelInferStats>(
        server->StatusManager(), request.model_name());
    auto timer = std::make_shared<ModelInferStats::ScopedTimer>();
    infer_stats->StartRequestTimer(timer.get());
    infer_stats->SetRequestedVersion(request.model_version());

    RequestStatus* request_status = response.mutable_request_status();

    // Get the request ID at top level so we can set the corresponding field
    // in ResponseHeader even on inference failure
    uint64_t id = request.meta_data().id();

    auto backend = std::make_shared<InferenceServer::InferBackendHandle>();
    tensorflow::Status status = server->CreateBackendHandle(
        request.model_name(), request.model_version(), backend);
    if (status.ok()) {
      infer_stats->SetModelBackend((*backend)());

      std::shared_ptr<GRPCInferRequestProvider> request_provider;
      status = GRPCInferRequestProvider::Create(
          *((*backend)()), request, &request_provider);
      if (status.ok()) {
        infer_stats->SetBatchSize(
            request_provider->RequestHeader().batch_size());

        std::shared_ptr<GRPCInferResponseProvider> response_provider;
        status = GRPCInferResponseProvider::Create(
            request.meta_data(), &response, &response_provider);
        if (status.ok()) {
          server->HandleInfer(
              request_status, backend, request_provider, response_provider,
              infer_stats,
              [this, execution_context, id, request_status, &response,
               infer_stats, timer]() mutable {
                // If the response is an error then clear the meta-data
                // and raw output as they may be partially or
                // un-initialized.
                if (request_status->code() != RequestStatusCode::SUCCESS) {
                  response.mutable_meta_data()->Clear();
                  response.mutable_raw_output()->Clear();
                }

                response.mutable_meta_data()->set_id(id);
                this->CompleteExecution(execution_context);
                timer.reset();
              });
        }
      }
    }

    if (!status.ok()) {
      LOG_VERBOSE(1) << "Infer failed: " << status.error_message();
      infer_stats->SetFailed(true);
      RequestStatusFactory::Create(
          request_status, 0 /* request_id */, server->Id(), status);

      // If the response is an error then clear the meta-data and raw
      // output as they may be partially or un-initialized.
      response.mutable_meta_data()->Clear();
      response.mutable_raw_output()->Clear();

      response.mutable_meta_data()->set_id(id);
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
    GetResources()->GetMgmtThreadPool().enqueue(
        [this, execution_context, &request, &response] {
          auto server = GetResources()->GetServer();
          ServerStatTimerScoped timer(
              server->StatusManager(), ServerStatTimerScoped::Kind::PROFILE);

          RequestStatus* request_status = response.mutable_request_status();
          server->HandleProfile(request_status, request.cmd());
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
    GetResources()->GetMgmtThreadPool().enqueue(
        [this, execution_context, &request, &response] {
          auto server = GetResources()->GetServer();
          ServerStatTimerScoped timer(
              server->StatusManager(), ServerStatTimerScoped::Kind::HEALTH);

          RequestStatus* request_status = response.mutable_request_status();
          bool health;

          server->HandleHealth(request_status, &health, request.mode());
          response.set_health(health);
          this->CompleteExecution(execution_context);
        });
  }
};
}  // namespace

GRPCServer::GRPCServer(const std::string& addr) : nvrpc::Server(addr) {}

tensorflow::Status
GRPCServer::Create(
    InferenceServer* server, uint16_t port,
    std::unique_ptr<GRPCServer>* grpc_server)
{
  // DLIS-162 - provide global defaults and cli overridable options
  g_Resources = std::make_shared<AsyncResources>(
      server,  // InferenceServer*,
      1,       // infer threads
      1        // mgmt threads
  );

  LOG_INFO << "Building nvrpc server";
  const std::string addr = "0.0.0.0:" + std::to_string(port);
  grpc_server->reset(new GRPCServer(addr));

  (*grpc_server)->GetBuilder().SetMaxMessageSize(MAX_GRPC_MESSAGE_SIZE);

  LOG_INFO << "Register TensorRT GRPCService";
  auto inferenceService = (*grpc_server)->RegisterAsyncService<GRPCService>();

  LOG_INFO << "Register Infer RPC";
  auto rpcInfer = inferenceService->RegisterRPC<InferContext>(
      &GRPCService::AsyncService::RequestInfer);

  LOG_INFO << "Register StreamInfer RPC";
  auto rpcStreamInfer = inferenceService->RegisterRPC<StreamInferContext>(
      &GRPCService::AsyncService::RequestStreamInfer);

  LOG_INFO << "Register Status RPC";
  auto rpcStatus = inferenceService->RegisterRPC<StatusContext>(
      &GRPCService::AsyncService::RequestStatus);

  LOG_INFO << "Register Profile RPC";
  auto rpcProfile = inferenceService->RegisterRPC<ProfileContext>(
      &GRPCService::AsyncService::RequestProfile);

  LOG_INFO << "Register Health RPC";
  auto rpcHealth = inferenceService->RegisterRPC<HealthContext>(
      &GRPCService::AsyncService::RequestHealth);

  LOG_INFO << "Register Executor";
  auto executor = (*grpc_server)->RegisterExecutor(new ::nvrpc::Executor(1));

  // You can register RPC execution contexts from any registered RPC on any
  // executor.
  executor->RegisterContexts(
      rpcInfer, g_Resources, 1000);  // Configurable DLIS-161
  executor->RegisterContexts(
      rpcStreamInfer, g_Resources, 100);  // Configurable DLIS-161
  executor->RegisterContexts(rpcStatus, g_Resources, 1);
  executor->RegisterContexts(rpcHealth, g_Resources, 1);
  executor->RegisterContexts(rpcProfile, g_Resources, 1);

  return tensorflow::Status::OK();
}

tensorflow::Status
GRPCServer::Start()
{
  AsyncRun();
  return tensorflow::Status::OK();
}

tensorflow::Status
GRPCServer::Stop()
{
  Shutdown();
  return tensorflow::Status::OK();
}

}}  // namespace nvidia::inferenceserver
