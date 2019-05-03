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

#include "src/core/request_inprocess.h"

#include "src/core/backend.h"
#include "src/core/logging.h"
#include "src/core/provider.h"
#include "src/core/provider_utils.h"
#include "src/core/request_common.h"
#include "src/core/server.h"

// If status is non-OK, return the corresponding Error.
#define RETURN_IF_STATUS_ERROR(S)                        \
  do {                                                   \
    const Status& status__ = (S);                        \
    if (status__.Code() != RequestStatusCode::SUCCESS) { \
      return Error(status__.Code(), status__.Message()); \
    }                                                    \
  } while (false)

namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================

namespace {

class ServerOptionsImpl : public InferenceServerContext::Options {
 public:
  const std::string& ModelRepositoryPath() const;
  void SetModelRepositoryPath(const std::string& path) override;

 private:
  std::string model_repository_path_;
};

const std::string&
ServerOptionsImpl::ModelRepositoryPath() const
{
  return model_repository_path_;
}

void
ServerOptionsImpl::SetModelRepositoryPath(const std::string& path)
{
  model_repository_path_ = path;
}

}  // namespace

InferenceServerContext::Options::~Options() {}

Error
InferenceServerContext::Options::Create(std::unique_ptr<Options>* options)
{
  options->reset(new ServerOptionsImpl());
  return Error::Success;
}

Error
InferenceServerContext::Create(
    std::unique_ptr<InferenceServerContext>* ctx,
    const std::unique_ptr<Options>& options)
{
  static bool initialized = false;
  if (initialized) {
    return Error(
        RequestStatusCode::UNSUPPORTED,
        "Attempt to create multiple InferenceServerContext objects");
  }

  ServerOptionsImpl* options_impl =
      dynamic_cast<ServerOptionsImpl*>(options.get());

  InferenceServer* server = new InferenceServer();

  server->SetModelStorePath(options_impl->ModelRepositoryPath());

  if (!server->Init()) {
    delete server;
    return Error(
        RequestStatusCode::INVALID_ARG,
        "Failed to initialize inference server");
  }

  ctx->reset(reinterpret_cast<InferenceServerContext*>(server));
  initialized = true;

  return Error::Success;
}

//==============================================================================

class ServerHealthInProcessContextImpl : public ServerHealthContext {
 public:
  ServerHealthInProcessContextImpl(InferenceServer* server, bool verbose);

  Error GetReady(bool* ready) override;
  Error GetLive(bool* live) override;

 private:
  Error GetHealth(const std::string& mode, bool* health);

  InferenceServer* const server_;
  const bool verbose_;
};

ServerHealthInProcessContextImpl::ServerHealthInProcessContextImpl(
    InferenceServer* server, bool verbose)
    : server_(server), verbose_(verbose)
{
}

Error
ServerHealthInProcessContextImpl::GetHealth(
    const std::string& mode, bool* health)
{
  RequestStatus request_status;
  server_->HandleHealth(&request_status, health, mode);

  if (verbose_) {
    if (request_status.code() != RequestStatusCode::SUCCESS) {
      LOG_ERROR << "server health failed, " << mode << ": "
                << request_status.ShortDebugString();
    } else {
      LOG_INFO << "server health, " << mode << ": " << *health;
    }
  }

  return Error(request_status);
}

Error
ServerHealthInProcessContextImpl::GetReady(bool* ready)
{
  return GetHealth("ready", ready);
}

Error
ServerHealthInProcessContextImpl::GetLive(bool* live)
{
  return GetHealth("live", live);
}

Error
ServerHealthInProcessContext::Create(
    std::unique_ptr<ServerHealthContext>* ctx,
    const std::unique_ptr<InferenceServerContext>& server_ctx, bool verbose)
{
  InferenceServer* server =
      reinterpret_cast<InferenceServer*>(server_ctx.get());
  ctx->reset(static_cast<ServerHealthContext*>(
      new ServerHealthInProcessContextImpl(server, verbose)));
  return Error::Success;
}

//==============================================================================

class ServerStatusInProcessContextImpl : public ServerStatusContext {
 public:
  ServerStatusInProcessContextImpl(
      InferenceServer* server, const std::string& model_name, bool verbose);
  Error GetServerStatus(ServerStatus* status) override;

 private:
  InferenceServer* const server_;
  const std::string model_name_;
  const bool verbose_;
};

ServerStatusInProcessContextImpl::ServerStatusInProcessContextImpl(
    InferenceServer* server, const std::string& model_name, bool verbose)
    : server_(server), model_name_(model_name), verbose_(verbose)
{
}

Error
ServerStatusInProcessContextImpl::GetServerStatus(ServerStatus* server_status)
{
  server_status->Clear();

  RequestStatus request_status;
  server_->HandleStatus(&request_status, server_status, model_name_);

  if (verbose_) {
    if (request_status.code() != RequestStatusCode::SUCCESS) {
      LOG_ERROR << "server status failed: "
                << request_status.ShortDebugString();
    } else {
      LOG_INFO << "server status: " << server_status->DebugString();
    }
  }

  return Error(request_status);
}

Error
ServerStatusInProcessContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx,
    const std::unique_ptr<InferenceServerContext>& server_ctx, bool verbose)
{
  InferenceServer* server =
      reinterpret_cast<InferenceServer*>(server_ctx.get());
  ctx->reset(static_cast<ServerStatusContext*>(
      new ServerStatusInProcessContextImpl(server, "", verbose)));
  return Error::Success;
}

Error
ServerStatusInProcessContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx,
    const std::unique_ptr<InferenceServerContext>& server_ctx,
    const std::string& model_name, bool verbose)
{
  InferenceServer* server =
      reinterpret_cast<InferenceServer*>(server_ctx.get());
  ctx->reset(static_cast<ServerStatusContext*>(
      new ServerStatusInProcessContextImpl(server, model_name, verbose)));
  return Error::Success;
}

//==============================================================================
class InProcessResultImpl : public ResultImpl {
 public:
  InProcessResultImpl(
      const std::shared_ptr<DelegatingInferResponseProvider>& response_provider,
      const std::shared_ptr<InferContext::Output>& output);

 private:
  // Result tensor data is used in-place from the response provider
  // object so we must hold a reference to it.
  std::shared_ptr<DelegatingInferResponseProvider> response_provider_;
};

InProcessResultImpl::InProcessResultImpl(
    const std::shared_ptr<DelegatingInferResponseProvider>& response_provider,
    const std::shared_ptr<InferContext::Output>& output)
    : ResultImpl(output, response_provider->ResponseHeader().batch_size()),
      response_provider_(response_provider)
{
}

//==============================================================================

class InferInProcessRequestImpl : public RequestImpl {
 public:
  InferInProcessRequestImpl(const uint64_t id) : RequestImpl(id) {}

  const RequestStatus& GetRequestStatus() const { return request_status_; }
  RequestStatus* MutableRequestStatus() { return &request_status_; }

  Error CreateResponseProvider(
      const InferRequestHeader& request_header,
      const std::shared_ptr<LabelProvider>& label_provider,
      std::shared_ptr<DelegatingInferResponseProvider>* response_provider);

  const std::shared_ptr<DelegatingInferResponseProvider>& GetResponseProvider()
      const
  {
    return response_provider_;
  }

  const InferResponseHeader& GetResponseHeader() const
  {
    return response_provider_->ResponseHeader();
  }

  Error GetOutputBufferContents(
      const std::string& name, void** content, size_t* content_byte_size) const;

  Error Release();
  Error Wait();

 private:
  RequestStatus request_status_;
  std::shared_ptr<DelegatingInferResponseProvider> response_provider_;

  std::mutex mu_;
  std::condition_variable cv_;
};

Error
InferInProcessRequestImpl::CreateResponseProvider(
    const InferRequestHeader& request_header,
    const std::shared_ptr<LabelProvider>& label_provider,
    std::shared_ptr<DelegatingInferResponseProvider>* response_provider)
{
  if (response_provider_ != nullptr) {
    return Error(
        RequestStatusCode::INTERNAL,
        "CreateResponseProvider called multiple times");
  }

  RETURN_IF_STATUS_ERROR(DelegatingInferResponseProvider::Create(
      request_header, label_provider, &response_provider_));

  *response_provider = response_provider_;
  return Error::Success;
}

Error
InferInProcessRequestImpl::GetOutputBufferContents(
    const std::string& name, void** content, size_t* content_byte_size) const
{
  RETURN_IF_STATUS_ERROR(response_provider_->OutputBufferContents(
      name, content, content_byte_size));
  return Error::Success;
}

Error
InferInProcessRequestImpl::Release()
{
  SetIsReady(true);
  cv_.notify_all();
  return Error::Success;
}

Error
InferInProcessRequestImpl::Wait()
{
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [this] { return IsReady(); });

  return Error::Success;
}

//==============================================================================

class InferInProcessContextImpl : public InferContextImpl {
 public:
  InferInProcessContextImpl(
      InferenceServer* server, CorrelationID correlation_id,
      const std::string& model_name, int64_t model_version, bool verbose);

  Error InitInProcess(
      const std::unique_ptr<InferenceServerContext>& server_ctx);

  virtual Error Run(ResultMap* results) override;
  virtual Error AsyncRun(std::shared_ptr<Request>* async_request) override;
  Error GetReadyAsyncRequest(
      std::shared_ptr<Request>* async_request, bool* is_ready,
      bool wait) override;
  Error GetAsyncRunResults(
      ResultMap* results, bool* is_ready,
      const std::shared_ptr<Request>& async_request, bool wait) override;

 private:
  Error AsyncInfer(
      std::shared_ptr<InferInProcessRequestImpl> request,
      std::function<void()> OnCompleteInfer);
  Status InferRequestToInputMap(
      std::unordered_map<std::string, std::shared_ptr<SystemMemory>>* input_map)
      const;
  Error GetResults(
      const InferInProcessRequestImpl& request,
      InferContext::ResultMap* results) const;
  Error InitResult(
      const InferInProcessRequestImpl& request,
      const std::shared_ptr<InferContext::Output>& infer_output,
      const InferResponseHeader::Output& output,
      InProcessResultImpl* result) const;

  InferenceServer* const server_;
  uint64_t next_id_;
};

InferInProcessContextImpl::InferInProcessContextImpl(
    InferenceServer* server, CorrelationID correlation_id,
    const std::string& model_name, int64_t model_version, bool verbose)
    : InferContextImpl(model_name, model_version, correlation_id, verbose),
      server_(server), next_id_(1)
{
}

Error
InferInProcessContextImpl::InitInProcess(
    const std::unique_ptr<InferenceServerContext>& server_ctx)
{
  std::unique_ptr<ServerStatusContext> sctx;
  Error err = ServerStatusInProcessContext::Create(
      &sctx, server_ctx, model_name_, verbose_);
  if (err.IsOk()) {
    err = Init(std::move(sctx));
  }

  return err;
}

Status
InferInProcessContextImpl::InferRequestToInputMap(
    std::unordered_map<std::string, std::shared_ptr<SystemMemory>>* input_map)
    const
{
  for (const auto& input : inputs_) {
    auto memory_ref = std::make_shared<SystemMemoryReference>();
    input_map->emplace(std::make_pair(
        input->Name(), std::static_pointer_cast<SystemMemory>(memory_ref)));

    InputImpl* input_impl = reinterpret_cast<InputImpl*>(input.get());

    for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
      const uint8_t* data_ptr;
      size_t data_byte_size;
      input_impl->GetRaw(batch_idx, &data_ptr, &data_byte_size);
      memory_ref->AddBuffer(
          reinterpret_cast<const char*>(data_ptr), data_byte_size);
    }
  }

  return Status::Success;
}

Error
InferInProcessContextImpl::AsyncInfer(
    std::shared_ptr<InferInProcessRequestImpl> request,
    std::function<void()> OnCompleteInfer)
{
  auto infer_stats =
      std::make_shared<ModelInferStats>(server_->StatusManager(), model_name_);
  auto timer = std::make_shared<ModelInferStats::ScopedTimer>();
  infer_stats->StartRequestTimer(timer.get());
  infer_stats->SetRequestedVersion(model_version_);
  infer_stats->SetFailed(true);

  std::shared_ptr<InferenceServer::InferBackendHandle> backend = nullptr;
  RETURN_IF_STATUS_ERROR(InferenceServer::InferBackendHandle::Create(
      server_, model_name_, model_version_, &backend));
  infer_stats->SetMetricReporter(
      backend->GetInferenceBackend()->MetricReporter());
  infer_stats->SetBatchSize(infer_request_.batch_size());

  // Create the input metadata for the request now that all input
  // sizes are known. For non-fixed-sized datatypes the
  // per-batch-instance byte-size can be different for different input
  // instances in the batch... so set the batch-byte-size to the total
  // size of the batch (see api.proto).
  infer_request_.mutable_input()->Clear();
  infer_request_.set_id(request->Id());
  for (auto& io : inputs_) {
    reinterpret_cast<InputImpl*>(io.get())->PrepareForRequest();

    auto rinput = infer_request_.add_input();
    rinput->set_name(io->Name());

    for (const auto s : io->Shape()) {
      rinput->add_dims(s);
    }
    if (!IsFixedSizeDataType(io->DType())) {
      rinput->set_batch_byte_size(io->TotalByteSize());
    }
  }

  RETURN_IF_STATUS_ERROR(
      NormalizeRequestHeader(*backend->GetInferenceBackend(), infer_request_));

  std::unordered_map<std::string, std::shared_ptr<SystemMemory>> input_map;
  RETURN_IF_STATUS_ERROR(InferRequestToInputMap(&input_map));

  std::shared_ptr<InferRequestProvider> request_provider;
  RETURN_IF_STATUS_ERROR(InferRequestProvider::Create(
      model_name_, model_version_, infer_request_, input_map,
      &request_provider));

  std::shared_ptr<DelegatingInferResponseProvider> response_provider;
  Error err = request->CreateResponseProvider(
      infer_request_, backend->GetInferenceBackend()->GetLabelProvider(),
      &response_provider);
  if (!err.IsOk()) {
    return err;
  }

  server_->HandleInfer(
      request->MutableRequestStatus(), backend, request_provider,
      response_provider, infer_stats,
      [this, infer_stats, timer, OnCompleteInfer]() mutable {
        infer_stats->SetFailed(false);
        timer.reset();
        OnCompleteInfer();
      });

  return Error::Success;
}

Error
InferInProcessContextImpl::InitResult(
    const InferInProcessRequestImpl& request,
    const std::shared_ptr<InferContext::Output>& infer_output,
    const InferResponseHeader::Output& output,
    InProcessResultImpl* result) const
{
  result->SetBatch1Shape(output.raw().dims());
  if (IsFixedSizeDataType(infer_output->DType())) {
    result->SetBatchnByteSize(output.raw().batch_byte_size());
  }

  if (result->ResultFormat() == InferContext::Result::ResultFormat::RAW) {
    void* content;
    size_t content_byte_size;
    Error err = request.GetOutputBufferContents(
        output.name(), &content, &content_byte_size);
    if (!err.IsOk()) {
      return err;
    }

    size_t result_bytes = 0;
    err = result->SetNextRawResult(
        reinterpret_cast<const uint8_t*>(content), content_byte_size,
        true /* inplace */, &result_bytes);
    if (!err.IsOk()) {
      return err;
    }

    if (result_bytes != output.raw().batch_byte_size()) {
      return Error(
          RequestStatusCode::INVALID,
          "Result size doesn't match expected size for output '" +
              output.name() + "'");
    }
  }

  return Error::Success;
}

Error
InferInProcessContextImpl::GetResults(
    const InferInProcessRequestImpl& request,
    InferContext::ResultMap* results) const
{
  results->clear();

  // Create a Result for each output. Each result holds a reference to
  // the response provider_ (shared_ptr) so it can use its specific
  // result in-place instead of copying it out.
  const InferResponseHeader& response_header = request.GetResponseHeader();
  for (const auto& output : response_header.output()) {
    std::shared_ptr<InferContext::Output> infer_output;
    Error err = GetOutput(output.name(), &infer_output);
    if (!err.IsOk()) {
      results->clear();
      return err;
    }

    std::unique_ptr<InProcessResultImpl> result(
        new InProcessResultImpl(request.GetResponseProvider(), infer_output));
    err = InitResult(request, infer_output, output, result.get());
    if (!err.IsOk()) {
      results->clear();
      return err;
    }

    results->insert(std::make_pair(output.name(), std::move(result)));
  }

  Error err = request.PostRunProcessing(response_header, results);
  if (!err.IsOk()) {
    results->clear();
  }

  return err;
}

Error
InferInProcessContextImpl::AsyncRun(std::shared_ptr<Request>* async_request)
{
  std::shared_ptr<InferInProcessRequestImpl> inprocess_request(
      std::make_shared<InferInProcessRequestImpl>(next_id_++));
  *async_request = inprocess_request;
  return AsyncInfer(inprocess_request, [inprocess_request]() {
    inprocess_request->Release();
  });
}

Error
InferInProcessContextImpl::GetAsyncRunResults(
    ResultMap* results, bool* is_ready,
    const std::shared_ptr<Request>& async_request, bool wait)
{
  std::shared_ptr<InferInProcessRequestImpl> inprocess_request =
      std::static_pointer_cast<InferInProcessRequestImpl>(async_request);

  *is_ready = inprocess_request->IsReady();

  if (!(*is_ready) && wait) {
    inprocess_request->Wait();
    *is_ready = inprocess_request->IsReady();
  }

  if (*is_ready) {
    return GetResults(*inprocess_request, results);
  }

  return Error::Success;
}

Error
InferInProcessContextImpl::GetReadyAsyncRequest(
    std::shared_ptr<Request>* async_request, bool* is_ready, bool wait)
{
  return Error(
      RequestStatusCode::UNSUPPORTED,
      "GetReadyAsyncRequest not supported for in-process API");
}

Error
InferInProcessContextImpl::Run(ResultMap* results)
{
  std::shared_ptr<Request> request;
  Error err = AsyncRun(&request);
  if (!err.IsOk()) {
    return err;
  }

  bool is_ready;
  err = GetAsyncRunResults(results, &is_ready, request, true /* wait */);
  if (!err.IsOk()) {
    return err;
  }

  std::shared_ptr<InferInProcessRequestImpl> inprocess_request =
      std::static_pointer_cast<InferInProcessRequestImpl>(request);

  err = Error(inprocess_request->GetRequestStatus());
  if (err.IsOk()) {
    err = GetResults(*inprocess_request, results);
  }

  return err;
}

Error
InferInProcessContext::Create(
    std::unique_ptr<InferContext>* ctx,
    const std::unique_ptr<InferenceServerContext>& server_ctx,
    const std::string& model_name, int64_t model_version, bool verbose)
{
  InferenceServer* server =
      reinterpret_cast<InferenceServer*>(server_ctx.get());
  InferInProcessContextImpl* ctx_ptr = new InferInProcessContextImpl(
      server, 0 /* correlation_id */, model_name, model_version, verbose);
  ctx->reset(static_cast<InferContext*>(ctx_ptr));

  Error err = ctx_ptr->InitInProcess(server_ctx);
  if (!err.IsOk()) {
    ctx->reset();
  }

  return err;
}

Error
InferInProcessContext::Create(
    std::unique_ptr<InferContext>* ctx,
    const std::unique_ptr<InferenceServerContext>& server_ctx,
    CorrelationID correlation_id, const std::string& model_name,
    int64_t model_version, bool verbose)
{
  InferenceServer* server =
      reinterpret_cast<InferenceServer*>(server_ctx.get());
  InferInProcessContextImpl* ctx_ptr = new InferInProcessContextImpl(
      server, correlation_id, model_name, model_version, verbose);
  ctx->reset(static_cast<InferContext*>(ctx_ptr));

  Error err = ctx_ptr->InitInProcess(server_ctx);
  if (!err.IsOk()) {
    ctx->reset();
  }

  return err;
}

}}}  // namespace nvidia::inferenceserver::client
