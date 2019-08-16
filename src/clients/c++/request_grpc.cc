// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#define DLL_EXPORTING

#include "src/clients/c++/request_grpc.h"

#include <grpcpp/grpcpp.h>
#include <cstdint>
#include <iostream>
#include "src/clients/c++/request_common.h"
#include "src/core/grpc_service.grpc.pb.h"
#include "src/core/grpc_service.pb.h"
#include "src/core/model_config.pb.h"

namespace nvidia { namespace inferenceserver { namespace client {

class InferGrpcContextImpl;

namespace {

//==============================================================================

// Use map to keep track of GRPC channels. <key, value> : <url, Channel*>
// If context is created on url that has established Channel, then reuse it.
std::map<std::string, std::shared_ptr<grpc::Channel>> grpc_channel_map_;
std::shared_ptr<grpc::Channel>
GetChannel(const std::string& url)
{
  const auto& channel_itr = grpc_channel_map_.find(url);
  if (channel_itr != grpc_channel_map_.end()) {
    return channel_itr->second;
  } else {
    grpc::ChannelArguments arguments;
    arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
    arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
        url, grpc::InsecureChannelCredentials(), arguments);
    grpc_channel_map_.insert(std::make_pair(url, channel));
    return channel;
  }
}

}  // namespace

//==============================================================================

class ServerHealthGrpcContextImpl : public ServerHealthContext {
 public:
  ServerHealthGrpcContextImpl(const std::string& url, bool verbose);

  Error GetReady(bool* ready) override;
  Error GetLive(bool* live) override;

 private:
  Error GetHealth(const std::string& mode, bool* health);

  // GRPC end point.
  std::unique_ptr<GRPCService::Stub> stub_;

  // Enable verbose output
  const bool verbose_;
};

ServerHealthGrpcContextImpl::ServerHealthGrpcContextImpl(
    const std::string& url, bool verbose)
    : stub_(GRPCService::NewStub(GetChannel(url))), verbose_(verbose)
{
}

Error
ServerHealthGrpcContextImpl::GetHealth(const std::string& mode, bool* health)
{
  Error err;

  HealthRequest request;
  HealthResponse response;
  grpc::ClientContext context;

  request.set_mode(mode);
  grpc::Status grpc_status = stub_->Health(&context, request, &response);
  if (grpc_status.ok()) {
    *health = response.health();
    err = Error(response.request_status());
  } else {
    // Something wrong with the GRPC connection
    err = Error(
        RequestStatusCode::INTERNAL,
        "GRPC client failed: " + std::to_string(grpc_status.error_code()) +
            ": " + grpc_status.error_message());
  }

  if (verbose_ && err.IsOk()) {
    std::cout << mode << ": " << *health << std::endl;
  }

  return err;
}

Error
ServerHealthGrpcContextImpl::GetReady(bool* ready)
{
  return GetHealth("ready", ready);
}

Error
ServerHealthGrpcContextImpl::GetLive(bool* live)
{
  return GetHealth("live", live);
}

Error
ServerHealthGrpcContext::Create(
    std::unique_ptr<ServerHealthContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ServerHealthContext*>(
      new ServerHealthGrpcContextImpl(server_url, verbose)));
  return Error::Success;
}

//==============================================================================

class ServerStatusGrpcContextImpl : public ServerStatusContext {
 public:
  ServerStatusGrpcContextImpl(const std::string& url, bool verbose);
  ServerStatusGrpcContextImpl(
      const std::string& url, const std::string& model_name, bool verbose);
  Error GetServerStatus(ServerStatus* status) override;

 private:
  // Model name
  const std::string model_name_;

  // GRPC end point.
  std::unique_ptr<GRPCService::Stub> stub_;

  // Enable verbose output
  const bool verbose_;
};

ServerStatusGrpcContextImpl::ServerStatusGrpcContextImpl(
    const std::string& url, bool verbose)
    : model_name_(""), stub_(GRPCService::NewStub(GetChannel(url))),
      verbose_(verbose)
{
}

ServerStatusGrpcContextImpl::ServerStatusGrpcContextImpl(
    const std::string& url, const std::string& model_name, bool verbose)
    : model_name_(model_name), stub_(GRPCService::NewStub(GetChannel(url))),
      verbose_(verbose)
{
}

Error
ServerStatusGrpcContextImpl::GetServerStatus(ServerStatus* server_status)
{
  server_status->Clear();

  Error grpc_status;

  StatusRequest request;
  StatusResponse response;
  grpc::ClientContext context;

  request.set_model_name(model_name_);
  grpc::Status status = stub_->Status(&context, request, &response);
  if (status.ok()) {
    server_status->Swap(response.mutable_server_status());
    grpc_status = Error(response.request_status());
  } else {
    // Something wrong with the GRPC conncection
    grpc_status = Error(
        RequestStatusCode::INTERNAL,
        "GRPC client failed: " + std::to_string(status.error_code()) + ": " +
            status.error_message());
  }

  // Log server status if request is SUCCESS and verbose is true.
  if (grpc_status.IsOk() && verbose_) {
    std::cout << server_status->DebugString() << std::endl;
  }
  return grpc_status;
}

Error
ServerStatusGrpcContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ServerStatusContext*>(
      new ServerStatusGrpcContextImpl(server_url, verbose)));
  return Error::Success;
}

Error
ServerStatusGrpcContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx, const std::string& server_url,
    const std::string& model_name, bool verbose)
{
  ctx->reset(static_cast<ServerStatusContext*>(
      new ServerStatusGrpcContextImpl(server_url, model_name, verbose)));
  return Error::Success;
}

//==============================================================================

class ProfileGrpcContextImpl : public ProfileContext {
 public:
  ProfileGrpcContextImpl(const std::string& url, bool verbose);
  Error StartProfile() override;
  Error StopProfile() override;

 private:
  Error SendCommand(const std::string& cmd_str);

  // GRPC end point.
  std::unique_ptr<GRPCService::Stub> stub_;

  // Enable verbose output
  const bool verbose_;
};

ProfileGrpcContextImpl::ProfileGrpcContextImpl(
    const std::string& url, bool verbose)
    : stub_(GRPCService::NewStub(GetChannel(url))), verbose_(verbose)
{
}

Error
ProfileGrpcContextImpl::StartProfile()
{
  return SendCommand("start");
}

Error
ProfileGrpcContextImpl::StopProfile()
{
  return SendCommand("stop");
}

Error
ProfileGrpcContextImpl::SendCommand(const std::string& cmd_str)
{
  ProfileRequest request;
  ProfileResponse response;
  grpc::ClientContext context;

  request.set_cmd(cmd_str);
  grpc::Status status = stub_->Profile(&context, request, &response);
  if (status.ok()) {
    return Error(response.request_status());
  } else {
    // Something wrong with the GRPC conncection
    return Error(
        RequestStatusCode::INTERNAL,
        "GRPC client failed: " + std::to_string(status.error_code()) + ": " +
            status.error_message());
  }
}

Error
ProfileGrpcContext::Create(
    std::unique_ptr<ProfileContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ProfileContext*>(
      new ProfileGrpcContextImpl(server_url, verbose)));
  return Error::Success;
}

//==============================================================================

class ModelControlGrpcContextImpl : public ModelControlContext {
 public:
  ModelControlGrpcContextImpl(const std::string& url, bool verbose);
  Error Load(const std::string& model_name) override;
  Error Unload(const std::string& model_name) override;

 private:
  Error SendRequest(const std::string& model_name, const bool is_load);

  // GRPC end point.
  std::unique_ptr<GRPCService::Stub> stub_;

  // Enable verbose output
  const bool verbose_;
};

ModelControlGrpcContextImpl::ModelControlGrpcContextImpl(
    const std::string& url, bool verbose)
    : stub_(GRPCService::NewStub(GetChannel(url))), verbose_(verbose)
{
}

Error
ModelControlGrpcContextImpl::Load(const std::string& model_name)
{
  return SendRequest(model_name, true);
}

Error
ModelControlGrpcContextImpl::Unload(const std::string& model_name)
{
  return SendRequest(model_name, false);
}

Error
ModelControlGrpcContextImpl::SendRequest(
    const std::string& model_name, const bool is_load)
{
  ModelControlRequest request;
  ModelControlResponse response;
  grpc::ClientContext context;

  request.set_model_name(model_name);
  if (is_load) {
    request.set_type(ModelControlRequest::LOAD);
  } else {
    request.set_type(ModelControlRequest::UNLOAD);
  }
  grpc::Status status = stub_->ModelControl(&context, request, &response);
  if (status.ok()) {
    return Error(response.request_status());
  } else {
    // Something wrong with the GRPC conncection
    return Error(
        RequestStatusCode::INTERNAL,
        "GRPC client failed: " + std::to_string(status.error_code()) + ": " +
            status.error_message());
  }
}

Error
ModelControlGrpcContext::Create(
    std::unique_ptr<ModelControlContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ModelControlContext*>(
      new ModelControlGrpcContextImpl(server_url, verbose)));
  return Error::Success;
}

//==============================================================================

class SharedMemoryControlGrpcContextImpl : public SharedMemoryControlContext {
 public:
  SharedMemoryControlGrpcContextImpl(const std::string& url, bool verbose);
  Error RegisterSharedMemory(
      const std::string& name, const std::string& shm_key, const size_t offset,
      const size_t byte_size) override;
  Error UnregisterSharedMemory(const std::string& name) override;
  Error UnregisterAllSharedMemory() override;

 private:
  Error SendRequest(
      const std::string& name, const SharedMemoryControlRequest::Type action,
      const std::string& shm_key, const size_t offset, const size_t byte_size);

  // GRPC end point.
  std::unique_ptr<GRPCService::Stub> stub_;

  // Enable verbose output
  const bool verbose_;
};

SharedMemoryControlGrpcContextImpl::SharedMemoryControlGrpcContextImpl(
    const std::string& url, bool verbose)
    : stub_(GRPCService::NewStub(GetChannel(url))), verbose_(verbose)
{
}

Error
SharedMemoryControlGrpcContextImpl::RegisterSharedMemory(
    const std::string& name, const std::string& shm_key, const size_t offset,
    const size_t byte_size)
{
  return SendRequest(
      name, SharedMemoryControlRequest::REGISTER, shm_key, offset, byte_size);
}

Error
SharedMemoryControlGrpcContextImpl::UnregisterSharedMemory(
    const std::string& name)
{
  return SendRequest(name, SharedMemoryControlRequest::UNREGISTER, "", 0, 0);
}

Error
SharedMemoryControlGrpcContextImpl::UnregisterAllSharedMemory()
{
  return SendRequest("", SharedMemoryControlRequest::UNREGISTER_ALL, "", 0, 0);
}

Error
SharedMemoryControlGrpcContextImpl::SendRequest(
    const std::string& name, const SharedMemoryControlRequest::Type action,
    const std::string& shm_key, const size_t offset, const size_t byte_size)
{
  SharedMemoryControlRequest request;
  SharedMemoryControlResponse response;
  grpc::ClientContext context;

  if (action == SharedMemoryControlRequest::REGISTER) {
    auto rshm_region = request.mutable_shared_memory_region();
    rshm_region->set_name(name);
    rshm_region->set_shm_key(shm_key);
    rshm_region->set_offset(offset);
    rshm_region->set_byte_size(byte_size);
    request.set_type(action);
  } else if (action == SharedMemoryControlRequest::UNREGISTER) {
    auto rshm_region = request.mutable_shared_memory_region();
    rshm_region->set_name(name);
    request.set_type(action);
  } else if (action == SharedMemoryControlRequest::UNREGISTER_ALL) {
    request.set_type(action);
  }

  grpc::Status status =
      stub_->SharedMemoryControl(&context, request, &response);
  if (status.ok()) {
    return Error(response.request_status());
  } else {
    // Something wrong with the GRPC conncection
    return Error(
        RequestStatusCode::INTERNAL,
        "GRPC client failed: " + std::to_string(status.error_code()) + ": " +
            status.error_message());
  }
}

Error
SharedMemoryControlGrpcContext::Create(
    std::unique_ptr<SharedMemoryControlContext>* ctx,
    const std::string& server_url, bool verbose)
{
  ctx->reset(static_cast<SharedMemoryControlContext*>(
      new SharedMemoryControlGrpcContextImpl(server_url, verbose)));
  return Error::Success;
}

//==============================================================================

class GrpcResultImpl : public ResultImpl {
 public:
  GrpcResultImpl(
      const std::shared_ptr<InferResponse>& response,
      const std::shared_ptr<InferContext::Output>& output);
  ~GrpcResultImpl() = default;

 private:
  // Result tensor data is used in-place from the GRPC response
  // object so we must hold a reference to it.
  std::shared_ptr<InferResponse> response_;
};

GrpcResultImpl::GrpcResultImpl(
    const std::shared_ptr<InferResponse>& response,
    const std::shared_ptr<InferContext::Output>& output)
    : ResultImpl(output, response->meta_data().batch_size()),
      response_(response)
{
}

//==============================================================================

class GrpcRequestImpl : public RequestImpl {
 public:
  GrpcRequestImpl(
      const uint64_t id, InferContext::OnCompleteFn callback = nullptr);

  Error GetResults(
      const InferGrpcContextImpl& ctx, InferContext::ResultMap* results) const;

 private:
  Error InitResult(
      const std::shared_ptr<InferContext::Output>& infer_output,
      const InferResponseHeader::Output& output, const size_t idx,
      GrpcResultImpl* result) const;

  friend class InferGrpcContextImpl;
  friend class InferGrpcStreamContextImpl;

  // Variables for GRPC call
  grpc::ClientContext grpc_context_;
  grpc::Status grpc_status_;
  std::shared_ptr<InferResponse> grpc_response_;
};

class InferGrpcContextImpl : public InferContextImpl {
 public:
  InferGrpcContextImpl(
      const std::string&, const std::string&, int64_t, CorrelationID, bool);
  virtual ~InferGrpcContextImpl();

  Error InitGrpc(const std::string& server_url);

  virtual Error Run(ResultMap* results) override;
  Error AsyncRun(std::shared_ptr<Request>* async_request) override;
  Error AsyncRun(OnCompleteFn callback) override;
  Error GetAsyncRunResults(
      ResultMap* results, bool* is_ready,
      const std::shared_ptr<Request>& async_request, bool wait) override;

 protected:
  virtual Error AsyncRun(
      std::shared_ptr<Request>* async_request, OnCompleteFn callback);
  virtual void AsyncTransfer();
  Error PreRunProcessing(std::shared_ptr<Request>& request);

  // The producer-consumer queue used to communicate asynchronously with
  // the GRPC runtime.
  grpc::CompletionQueue async_request_completion_queue_;

  // GRPC end point.
  std::unique_ptr<GRPCService::Stub> stub_;

  // request for GRPC call, one request object can be used for multiple calls
  // since it can be overwritten as soon as the GRPC send finishes.
  InferRequest request_;
};

//==============================================================================

GrpcRequestImpl::GrpcRequestImpl(
    const uint64_t id, InferContext::OnCompleteFn callback)
    : RequestImpl(id, std::move(callback)), grpc_status_(),
      grpc_response_(std::make_shared<InferResponse>())
{
  SetRunIndex(id);
}

Error
GrpcRequestImpl::InitResult(
    const std::shared_ptr<InferContext::Output>& infer_output,
    const InferResponseHeader::Output& output, const size_t idx,
    GrpcResultImpl* result) const
{
  result->SetBatch1Shape(output.raw().dims());
  if (IsFixedSizeDataType(infer_output->DType())) {
    result->SetBatchnByteSize(output.raw().batch_byte_size());
  }

  if ((result->ResultFormat() == InferContext::Result::ResultFormat::RAW) && (!result->UsesSharedMemory())) {
    if (grpc_response_->raw_output_size() <= (int)idx) {
      return Error(
          RequestStatusCode::INVALID,
          "Expected RAW output for result '" + output.name() + "'");
    }

    const std::string& raw_output = grpc_response_->raw_output(idx);
    const uint8_t* buf = reinterpret_cast<const uint8_t*>(&raw_output[0]);
    size_t size = raw_output.size();
    size_t result_bytes = 0;

    Error err =
        result->SetNextRawResult(buf, size, true /* inplace */, &result_bytes);
    if (!err.IsOk()) {
      return err;
    }

    if (result_bytes != size) {
      return Error(
          RequestStatusCode::INVALID,
          "Written bytes doesn't match received bytes for result '" +
              output.name() + "'");
    }
  }

  return Error::Success;
}

Error
GrpcRequestImpl::GetResults(
    const InferGrpcContextImpl& ctx, InferContext::ResultMap* results) const
{
  results->clear();

  // Something wrong with the GRPC connection
  if (!grpc_status_.ok()) {
    return Error(
        RequestStatusCode::INTERNAL,
        "GRPC client failed: " + std::to_string(grpc_status_.error_code()) +
            ": " + grpc_status_.error_message());
  }

  // Request failed...
  if (grpc_response_->request_status().code() != RequestStatusCode::SUCCESS) {
    return Error(grpc_response_->request_status());
  }

  const InferResponseHeader& response_header = grpc_response_->meta_data();

  // Create a Result for each output. Each result holds
  // grpc_response_ (shared_ptr) so it can use its specific result
  // in-place instead of copying it out.
  size_t idx = 0;
  for (const auto& output : response_header.output()) {
    std::shared_ptr<InferContext::Output> infer_output;
    Error err = ctx.GetOutput(output.name(), &infer_output);
    if (!err.IsOk()) {
      results->clear();
      return err;
    }

    std::unique_ptr<GrpcResultImpl> result(
        new GrpcResultImpl(grpc_response_, infer_output));
    err = InitResult(infer_output, output, idx, result.get());

    if (!err.IsOk()) {
      results->clear();
      return err;
    }
    results->insert(std::make_pair(output.name(), std::move(result)));
    ++idx;
  }

  Error err = PostRunProcessing(response_header, results);
  if (!err.IsOk()) {
    results->clear();
    return err;
  }

  return Error(grpc_response_->request_status());
}

//==============================================================================

Error
InferGrpcContextImpl::InitGrpc(const std::string& server_url)
{
  std::unique_ptr<ServerStatusContext> sctx;
  Error err =
      ServerStatusGrpcContext::Create(&sctx, server_url, model_name_, verbose_);
  if (err.IsOk()) {
    err = Init(std::move(sctx));
    if (err.IsOk()) {
      // Create request context for synchronous request.
      sync_request_.reset(
          static_cast<InferContext::Request*>(new GrpcRequestImpl(0)));
    }
  }

  return err;
}

InferGrpcContextImpl::InferGrpcContextImpl(
    const std::string& server_url, const std::string& model_name,
    int64_t model_version, CorrelationID correlation_id, bool verbose)
    : InferContextImpl(model_name, model_version, correlation_id, verbose),
      stub_(GRPCService::NewStub(GetChannel(server_url)))
{
}

InferGrpcContextImpl::~InferGrpcContextImpl()
{
  exiting_ = true;
  // thread not joinable if AsyncRun() is not called
  // (it is default constructed thread before the first AsyncRun() call)
  if (worker_.joinable()) {
    cv_.notify_all();
    worker_.join();
  }

  // Close complete queue and drain its content
  async_request_completion_queue_.Shutdown();
  bool has_next = true;
  void* tag;
  bool ok;
  do {
    has_next = async_request_completion_queue_.Next(&tag, &ok);
  } while (has_next);
}

Error
InferGrpcContextImpl::Run(ResultMap* results)
{
  grpc::ClientContext context;

  std::shared_ptr<GrpcRequestImpl> sync_request =
      std::static_pointer_cast<GrpcRequestImpl>(sync_request_);

  sync_request->Timer().Reset();
  // Use send timer to measure time for marshalling infer request
  sync_request->Timer().Record(RequestTimers::Kind::SEND_START);
  Error err = PreRunProcessing(sync_request_);
  if (!err.IsOk()) {
    return err;
  }
  sync_request->Timer().Record(RequestTimers::Kind::SEND_END);

  sync_request->Timer().Record(RequestTimers::Kind::REQUEST_START);
  sync_request->grpc_response_->Clear();
  sync_request->grpc_status_ =
      stub_->Infer(&context, request_, sync_request->grpc_response_.get());
  sync_request->Timer().Record(RequestTimers::Kind::REQUEST_END);

  sync_request->Timer().Record(RequestTimers::Kind::RECEIVE_START);
  Error request_status = sync_request->GetResults(*this, results);
  sync_request->Timer().Record(RequestTimers::Kind::RECEIVE_END);

  err = UpdateStat(sync_request->Timer());
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }

  return request_status;
}

Error
InferGrpcContextImpl::AsyncRun(std::shared_ptr<Request>* async_request)
{
  return AsyncRun(async_request, nullptr);
}

Error
InferGrpcContextImpl::AsyncRun(OnCompleteFn callback)
{
  std::shared_ptr<Request> request;
  return AsyncRun(&request, std::move(callback));
}

Error
InferGrpcContextImpl::AsyncRun(
    std::shared_ptr<Request>* async_request, OnCompleteFn callback)
{
  if (!worker_.joinable()) {
    worker_ = std::thread(&InferGrpcContextImpl::AsyncTransfer, this);
  }

  GrpcRequestImpl* current_context;
  uintptr_t run_index;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Also need to protect the id
    current_context =
        new GrpcRequestImpl(async_request_id_++, std::move(callback));
    async_request->reset(static_cast<Request*>(current_context));

    run_index = current_context->Id();
    auto insert_result = ongoing_async_requests_.emplace(
        std::make_pair(run_index, *async_request));
    if (!insert_result.second) {
      return Error(
          RequestStatusCode::INTERNAL,
          "Failed to insert new asynchronous request context.");
    }
  }

  current_context->Timer().Reset();
  current_context->Timer().Record(RequestTimers::Kind::SEND_START);
  Error err = PreRunProcessing(*async_request);
  if (!err.IsOk()) {
    ongoing_async_requests_.erase(current_context->Id());
    return err;
  }
  current_context->Timer().Record(RequestTimers::Kind::SEND_END);

  current_context->Timer().Record(RequestTimers::Kind::REQUEST_START);
  std::unique_ptr<grpc::ClientAsyncResponseReader<InferResponse>> rpc(
      stub_->PrepareAsyncInfer(
          &current_context->grpc_context_, request_,
          &async_request_completion_queue_));

  rpc->StartCall();

  rpc->Finish(
      current_context->grpc_response_.get(), &current_context->grpc_status_,
      (void*)run_index);

  cv_.notify_all();
  return Error(RequestStatusCode::SUCCESS);
}

Error
InferGrpcContextImpl::GetAsyncRunResults(
    ResultMap* results, bool* is_ready,
    const std::shared_ptr<Request>& async_request, bool wait)
{
  Error err = IsRequestReady(async_request, is_ready, wait);
  if (!err.IsOk() || !(*is_ready)) {
    return err;
  }

  std::shared_ptr<GrpcRequestImpl> grpc_request =
      std::static_pointer_cast<GrpcRequestImpl>(async_request);

  grpc_request->Timer().Record(RequestTimers::Kind::RECEIVE_START);
  Error request_status = grpc_request->GetResults(*this, results);
  grpc_request->Timer().Record(RequestTimers::Kind::RECEIVE_END);
  err = UpdateStat(grpc_request->Timer());
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ongoing_async_requests_.erase(grpc_request->RunIndex());
  }
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }
  return request_status;
}

Error
InferGrpcContextImpl::PreRunProcessing(std::shared_ptr<Request>& request)
{
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

    // set shared memory
    if (reinterpret_cast<InputImpl*>(io.get())->IsSharedMemory()) {
      auto rshared_memory = rinput->mutable_shared_memory();
      rshared_memory->set_name(
          reinterpret_cast<InputImpl*>(io.get())->GetSharedMemoryName());
      rshared_memory->set_offset(
          reinterpret_cast<InputImpl*>(io.get())->GetSharedMemoryOffset());
      rshared_memory->set_byte_size(io->TotalByteSize());
    }
  }

  request_.Clear();
  request_.set_model_name(model_name_);
  request_.set_model_version(model_version_);
  request_.mutable_meta_data()->MergeFrom(infer_request_);

  size_t input_pos_idx = 0;
  while (input_pos_idx < inputs_.size()) {
    InputImpl* io = reinterpret_cast<InputImpl*>(inputs_[input_pos_idx].get());

    // Append all batches of one input together (skip if using shared memory)
    if (!io->IsSharedMemory()) {
      std::string* new_input = request_.add_raw_input();
      for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
        const uint8_t* data_ptr;
        size_t data_byte_size;
        io->GetRaw(batch_idx, &data_ptr, &data_byte_size);
        new_input->append(
            reinterpret_cast<const char*>(data_ptr), data_byte_size);
      }
    }
    input_pos_idx++;
  }

  if (request_.ByteSizeLong() > INT_MAX) {
    size_t request_size = request_.ByteSizeLong();
    request_.Clear();
    return Error(
        RequestStatusCode::INVALID_ARG,
        "Request has byte size " + std::to_string(request_size) +
            " which exceed gRPC's byte size limit " + std::to_string(INT_MAX) +
            ".");
  }

  return Error::Success;
}

void
InferGrpcContextImpl::AsyncTransfer()
{
  do {
    // sleep if no work is available
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
      if (this->exiting_) {
        return true;
      }
      // wake up if at least one request is not ready
      for (auto& ongoing_async_request : this->ongoing_async_requests_) {
        if (std::static_pointer_cast<GrpcRequestImpl>(
                ongoing_async_request.second)
                ->IsReady() == false) {
          return true;
        }
      }
      return false;
    });
    lock.unlock();
    // GRPC async APIs are thread-safe https://github.com/grpc/grpc/issues/4486
    if (!exiting_) {
      std::shared_ptr<Request> request_with_callback;
      size_t got;
      bool ok = true;
      bool status = async_request_completion_queue_.Next((void**)(&got), &ok);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ok) {
          fprintf(stderr, "Unexpected not ok on client side.");
        }
        if (!status) {
          fprintf(stderr, "Completion queue is closed.");
        }
        auto itr = ongoing_async_requests_.find(got);
        if (itr == ongoing_async_requests_.end()) {
          fprintf(
              stderr,
              "Unexpected error: received completed request that"
              " is not in the list of asynchronous requests.\n");
          continue;
        }

        std::shared_ptr<GrpcRequestImpl> grpc_request =
            std::static_pointer_cast<GrpcRequestImpl>(itr->second);
        grpc_request->Timer().Record(RequestTimers::Kind::REQUEST_END);
        grpc_request->SetIsReady(true);
        if (grpc_request->HasCallback()) {
          request_with_callback = itr->second;
        }
      }
      // send signal in case the main thread is waiting
      cv_.notify_all();
      if (request_with_callback != nullptr) {
        GrpcRequestImpl* request =
            static_cast<GrpcRequestImpl*>(request_with_callback.get());
        request->callback_(this, std::move(request_with_callback));
      }
    }
  } while (!exiting_);
}

Error
InferGrpcContext::Create(
    std::unique_ptr<InferContext>* ctx, const std::string& server_url,
    const std::string& model_name, int64_t model_version, bool verbose)
{
  return Create(
      ctx, 0 /* correlation_id */, server_url, model_name, model_version,
      verbose);
}

Error
InferGrpcContext::Create(
    std::unique_ptr<InferContext>* ctx, CorrelationID correlation_id,
    const std::string& server_url, const std::string& model_name,
    int64_t model_version, bool verbose)
{
  InferGrpcContextImpl* ctx_ptr = new InferGrpcContextImpl(
      server_url, model_name, model_version, correlation_id, verbose);
  ctx->reset(static_cast<InferContext*>(ctx_ptr));

  Error err = ctx_ptr->InitGrpc(server_url);
  if (!err.IsOk()) {
    ctx->reset();
  }

  return err;
}

//==============================================================================

class InferGrpcStreamContextImpl : public InferGrpcContextImpl {
 public:
  using InferGrpcContextImpl::AsyncRun;

  InferGrpcStreamContextImpl(
      const std::string&, const std::string&, int64_t, CorrelationID, bool);
  virtual ~InferGrpcStreamContextImpl();

  Error Run(ResultMap* results) override;

 private:
  Error AsyncRun(
      std::shared_ptr<Request>* async_request, OnCompleteFn callback) override;
  void AsyncTransfer() override;

  // gRPC objects for using the streaming API
  grpc::ClientContext context_;
  std::shared_ptr<grpc::ClientReaderWriter<InferRequest, InferResponse>>
      stream_;
};

InferGrpcStreamContextImpl::InferGrpcStreamContextImpl(
    const std::string& server_url, const std::string& model_name,
    int64_t model_version, CorrelationID correlation_id, bool verbose)
    : InferGrpcContextImpl(
          server_url, model_name, model_version, correlation_id, verbose)
{
  stream_ = stub_->StreamInfer(&context_);
  // Initiate worker thread to read constantly
  worker_ = std::thread(&InferGrpcStreamContextImpl::AsyncTransfer, this);
}

InferGrpcStreamContextImpl::~InferGrpcStreamContextImpl()
{
  exiting_ = true;
  stream_->WritesDone();
  // The reader thread will drain the stream properly
  worker_.join();
}

Error
InferGrpcStreamContextImpl::Run(ResultMap* results)
{
  // Actually calling AsyncRun() and GetAsyncRunResults()
  std::shared_ptr<Request> req;
  Error err = AsyncRun(&req);
  if (!err.IsOk()) {
    return err;
  }
  bool is_ready;
  return GetAsyncRunResults(results, &is_ready, req, true);
}

Error
InferGrpcStreamContextImpl::AsyncRun(
    std::shared_ptr<Request>* async_request, OnCompleteFn callback)
{
  GrpcRequestImpl* current_context;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_context =
        new GrpcRequestImpl(async_request_id_++, std::move(callback));
    async_request->reset(static_cast<Request*>(current_context));

    uintptr_t run_index = current_context->Id();
    auto insert_result = ongoing_async_requests_.emplace(
        std::make_pair(run_index, *async_request));
    if (!insert_result.second) {
      return Error(
          RequestStatusCode::INTERNAL,
          "Failed to insert new asynchronous request context.");
    }
  }

  current_context->Timer().Reset();
  current_context->Timer().Record(RequestTimers::Kind::SEND_START);
  Error err = PreRunProcessing(*async_request);
  if (!err.IsOk()) {
    ongoing_async_requests_.erase(current_context->Id());
    return err;
  }
  current_context->Timer().Record(RequestTimers::Kind::SEND_END);

  current_context->Timer().Record(RequestTimers::Kind::REQUEST_START);
  bool ok = stream_->Write(request_);

  if (ok) {
    return Error::Success;
  } else {
    return Error(RequestStatusCode::INTERNAL, "Stream has been closed.");
  }
}

void
InferGrpcStreamContextImpl::AsyncTransfer()
{
  InferResponse response;
  // End loop if Read() returns false
  // (stream ended and all responses are drained)
  while (stream_->Read(&response)) {
    if (exiting_) {
      continue;
    }

    std::shared_ptr<Request> request_with_callback;
    uintptr_t run_index = response.meta_data().id();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto itr = ongoing_async_requests_.find(run_index);
      if (itr == ongoing_async_requests_.end()) {
        fprintf(
            stderr,
            "Unexpected error: received completed request that"
            " is not in the list of asynchronous requests.\n");
        continue;
      }

      std::shared_ptr<GrpcRequestImpl> grpc_request =
          std::static_pointer_cast<GrpcRequestImpl>(itr->second);
      grpc_request->grpc_response_->Swap(&response);
      grpc_request->Timer().Record(RequestTimers::Kind::REQUEST_END);
      grpc_request->SetIsReady(true);
      if (grpc_request->HasCallback()) {
        request_with_callback = itr->second;
      }
    }
    // send signal in case the main thread is waiting for response
    cv_.notify_all();
    if (request_with_callback != nullptr) {
      GrpcRequestImpl* request =
          static_cast<GrpcRequestImpl*>(request_with_callback.get());
      request->callback_(this, std::move(request_with_callback));
    }
  }
  stream_->Finish();
}

Error
InferGrpcStreamContext::Create(
    std::unique_ptr<InferContext>* ctx, const std::string& server_url,
    const std::string& model_name, int64_t model_version, bool verbose)
{
  return Create(
      ctx, 0 /* correlation_id */, server_url, model_name, model_version,
      verbose);
}

Error
InferGrpcStreamContext::Create(
    std::unique_ptr<InferContext>* ctx, CorrelationID correlation_id,
    const std::string& server_url, const std::string& model_name,
    int64_t model_version, bool verbose)
{
  InferGrpcStreamContextImpl* ctx_ptr = new InferGrpcStreamContextImpl(
      server_url, model_name, model_version, correlation_id, verbose);
  ctx->reset(static_cast<InferContext*>(ctx_ptr));

  Error err = ctx_ptr->InitGrpc(server_url);
  if (!err.IsOk()) {
    ctx->reset();
  }

  return err;
}

}}}  // namespace nvidia::inferenceserver::client
