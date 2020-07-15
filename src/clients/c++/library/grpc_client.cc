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

// Include this first to make sure we are a friend of common classes.
#define TRITON_INFERENCE_SERVER_CLIENT_CLASS InferenceServerGrpcClient
#include "src/clients/c++/library/common.h"

#include "src/clients/c++/library/grpc_client.h"

#include <grpcpp/grpcpp.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>

namespace nvidia { namespace inferenceserver { namespace client {
namespace {

//==============================================================================

// Use map to keep track of GRPC channels. <key, value> : <url, Channel*>
// If context is created on url that has established Channel, then reuse it.
std::map<std::string, std::shared_ptr<grpc::Channel>> grpc_channel_map_;
std::mutex grpc_channel_map_mtx_;

void
ReadFile(const std::string& filename, std::string& data)
{
  std::ifstream file(filename.c_str(), std::ios::in);
  if (file.is_open()) {
    std::stringstream ss;
    ss << file.rdbuf();
    file.close();
    data = ss.str();
  }
  return;
}

std::shared_ptr<grpc::Channel>
GetChannel(const std::string& url, bool use_ssl, const SslOptions& ssl_options)
{
  std::lock_guard<std::mutex> lock(grpc_channel_map_mtx_);

  const auto& channel_itr = grpc_channel_map_.find(url);
  if (channel_itr != grpc_channel_map_.end()) {
    return channel_itr->second;
  } else {
    grpc::ChannelArguments arguments;
    arguments.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_SIZE);
    arguments.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_SIZE);
    std::shared_ptr<grpc::ChannelCredentials> credentials;
    if (use_ssl) {
      std::string root;
      std::string key;
      std::string cert;
      ReadFile(ssl_options.root_certificates, root);
      ReadFile(ssl_options.private_key, key);
      ReadFile(ssl_options.certificate_chain, cert);
      grpc::SslCredentialsOptions opts = {root, key, cert};
      credentials = grpc::SslCredentials(opts);
    } else {
      credentials = grpc::InsecureChannelCredentials();
    }
    std::shared_ptr<grpc::Channel> channel =
        grpc::CreateCustomChannel(url, credentials, arguments);
    grpc_channel_map_.insert(std::make_pair(url, channel));
    return channel;
  }
}
}  // namespace

//==============================================================================
// An GrpcInferRequest represents an inflght inference request on gRPC.
//
class GrpcInferRequest : public InferRequest {
 public:
  GrpcInferRequest(InferenceServerClient::OnCompleteFn callback = nullptr)
      : InferRequest(callback), grpc_status_(),
        grpc_response_(std::make_shared<ModelInferResponse>())
  {
  }

  friend InferenceServerGrpcClient;

 private:
  // Variables for GRPC call
  grpc::ClientContext grpc_context_;
  grpc::Status grpc_status_;
  std::shared_ptr<ModelInferResponse> grpc_response_;
};

//==============================================================================

class InferResultGrpc : public InferResult {
 public:
  static Error Create(
      InferResult** infer_result, std::shared_ptr<ModelInferResponse> response,
      Error& request_status);
  static Error Create(
      InferResult** infer_result,
      std::shared_ptr<ModelStreamInferResponse> response);

  Error RequestStatus() const override;
  Error ModelName(std::string* name) const override;
  Error ModelVersion(std::string* version) const override;
  Error Id(std::string* id) const override;
  Error Shape(const std::string& output_name, std::vector<int64_t>* shape)
      const override;
  Error Datatype(
      const std::string& output_name, std::string* datatype) const override;
  Error RawData(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const override;
  Error StringData(
      const std::string& output_name,
      std::vector<std::string>* string_result) const override;
  std::string DebugString() const override { return response_->DebugString(); }

 private:
  InferResultGrpc(
      std::shared_ptr<ModelInferResponse> response, Error& request_status);
  InferResultGrpc(std::shared_ptr<ModelStreamInferResponse> response);

  std::map<std::string, const ModelInferResponse::InferOutputTensor*>
      output_name_to_result_map_;

  std::shared_ptr<ModelInferResponse> response_;
  std::shared_ptr<ModelStreamInferResponse> stream_response_;
  Error request_status_;
};

Error
InferResultGrpc::Create(
    InferResult** infer_result, std::shared_ptr<ModelInferResponse> response,
    Error& request_status)
{
  *infer_result = reinterpret_cast<InferResult*>(
      new InferResultGrpc(response, request_status));
  return Error::Success;
}

Error
InferResultGrpc::Create(
    InferResult** infer_result,
    std::shared_ptr<ModelStreamInferResponse> response)
{
  *infer_result = reinterpret_cast<InferResult*>(new InferResultGrpc(response));
  return Error::Success;
}

Error
InferResultGrpc::RequestStatus() const
{
  return request_status_;
}

Error
InferResultGrpc::ModelName(std::string* name) const
{
  *name = response_->model_name();
  return Error::Success;
}

Error
InferResultGrpc::ModelVersion(std::string* version) const
{
  *version = response_->model_version();
  return Error::Success;
}

Error
InferResultGrpc::Id(std::string* id) const
{
  *id = response_->id();
  return Error::Success;
}

Error
InferResultGrpc::Shape(
    const std::string& output_name, std::vector<int64_t>* shape) const
{
  shape->clear();
  auto it = output_name_to_result_map_.find(output_name);
  if (it != output_name_to_result_map_.end()) {
    for (const auto dim : it->second->shape()) {
      shape->push_back(dim);
    }
  } else {
    return Error(
        "The response does not contain shape for output name '" + output_name +
        "'");
  }
  return Error::Success;
}

Error
InferResultGrpc::Datatype(
    const std::string& output_name, std::string* datatype) const
{
  auto it = output_name_to_result_map_.find(output_name);
  if (it != output_name_to_result_map_.end()) {
    *datatype = it->second->datatype();
  } else {
    return Error(
        "The response does not contain datatype for output name '" +
        output_name + "'");
  }
  return Error::Success;
}


Error
InferResultGrpc::RawData(
    const std::string& output_name, const uint8_t** buf,
    size_t* byte_size) const
{
  auto it = output_name_to_result_map_.find(output_name);
  if (it != output_name_to_result_map_.end()) {
    *buf = (uint8_t*)&(it->second->contents().raw_contents()[0]);
    *byte_size = it->second->contents().raw_contents().size();
  } else {
    return Error(
        "The response does not contain results for output name '" +
        output_name + "'");
  }

  return Error::Success;
}

Error
InferResultGrpc::StringData(
    const std::string& output_name,
    std::vector<std::string>* string_result) const
{
  std::string datatype;
  Error err = Datatype(output_name, &datatype);
  if (!err.IsOk()) {
    return err;
  }
  if (datatype.compare("BYTES") != 0) {
    return Error(
        "This function supports tensors with datatype 'BYTES', requested "
        "output tensor '" +
        output_name + "' with datatype '" + datatype + "'");
  }

  const uint8_t* buf;
  size_t byte_size;
  err = RawData(output_name, &buf, &byte_size);
  string_result->clear();
  if (byte_size != 0) {
    size_t buf_offset = 0;
    while (byte_size > buf_offset) {
      const uint32_t element_size =
          *(reinterpret_cast<const uint32_t*>(buf + buf_offset));
      string_result->emplace_back(
          reinterpret_cast<const char*>(
              buf + buf_offset + sizeof(element_size)),
          element_size);
      buf_offset += (sizeof(element_size) + element_size);
    }
  } else {
    auto it = output_name_to_result_map_.find(output_name);
    for (const auto& element : it->second->contents().byte_contents()) {
      string_result->push_back(element);
    }
  }

  return Error::Success;
}

InferResultGrpc::InferResultGrpc(
    std::shared_ptr<ModelInferResponse> response, Error& request_status)
    : response_(response), request_status_(request_status)
{
  for (const auto& output : response_->outputs()) {
    output_name_to_result_map_[output.name()] = &output;
  }
}

InferResultGrpc::InferResultGrpc(
    std::shared_ptr<ModelStreamInferResponse> stream_response)
    : stream_response_(stream_response)
{
  request_status_ = Error(stream_response_->error_message());
  response_.reset(
      stream_response->mutable_infer_response(), [](ModelInferResponse*) {});
  for (const auto& output : response_->outputs()) {
    output_name_to_result_map_[output.name()] = &output;
  }
}

//==============================================================================

Error
InferenceServerGrpcClient::Create(
    std::unique_ptr<InferenceServerGrpcClient>* client,
    const std::string& server_url, bool verbose, bool use_ssl,
    const SslOptions& ssl_options)
{
  client->reset(
      new InferenceServerGrpcClient(server_url, verbose, use_ssl, ssl_options));
  return Error::Success;
}

Error
InferenceServerGrpcClient::IsServerLive(bool* live, const Headers& headers)
{
  Error err;

  ServerLiveRequest request;
  ServerLiveResponse response;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  grpc::Status grpc_status = stub_->ServerLive(&context, request, &response);
  if (grpc_status.ok()) {
    *live = response.live();
    if (verbose_) {
      std::cout << "Server Live : " << *live << std::endl;
    }
  } else {
    err = Error(grpc_status.error_message());
  }

  return err;
}

Error
InferenceServerGrpcClient::IsServerReady(bool* ready, const Headers& headers)
{
  Error err;

  ServerReadyRequest request;
  ServerReadyResponse response;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  grpc::Status grpc_status = stub_->ServerReady(&context, request, &response);
  if (grpc_status.ok()) {
    *ready = response.ready();
    if (verbose_) {
      std::cout << "Server Ready : " << *ready << std::endl;
    }
  } else {
    err = Error(grpc_status.error_message());
  }

  return err;
}

Error
InferenceServerGrpcClient::IsModelReady(
    bool* ready, const std::string& model_name,
    const std::string& model_version, const Headers& headers)
{
  Error err;

  ModelReadyRequest request;
  ModelReadyResponse response;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_name(model_name);
  request.set_version(model_version);
  grpc::Status grpc_status = stub_->ModelReady(&context, request, &response);
  if (grpc_status.ok()) {
    *ready = response.ready();
    if (verbose_) {
      std::cout << "Model Ready : name: " << model_name;
      if (!model_version.empty()) {
        std::cout << "(version: " << model_version << ") ";
      }
      std::cout << ": " << *ready << std::endl;
    }
  } else {
    err = Error(grpc_status.error_message());
  }

  return err;
}


Error
InferenceServerGrpcClient::ServerMetadata(
    ServerMetadataResponse* server_metadata, const Headers& headers)
{
  server_metadata->Clear();
  Error err;

  ServerMetadataRequest request;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  grpc::Status grpc_status =
      stub_->ServerMetadata(&context, request, server_metadata);
  if (grpc_status.ok()) {
    if (verbose_) {
      std::cout << server_metadata->DebugString() << std::endl;
    }
  } else {
    err = Error(grpc_status.error_message());
  }

  return err;
}


Error
InferenceServerGrpcClient::ModelMetadata(
    ModelMetadataResponse* model_metadata, const std::string& model_name,
    const std::string& model_version, const Headers& headers)
{
  model_metadata->Clear();
  Error err;

  ModelMetadataRequest request;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_name(model_name);
  request.set_version(model_version);
  grpc::Status grpc_status =
      stub_->ModelMetadata(&context, request, model_metadata);
  if (grpc_status.ok()) {
    if (verbose_) {
      std::cout << model_metadata->DebugString() << std::endl;
    }
  } else {
    err = Error(grpc_status.error_message());
  }

  return err;
}


Error
InferenceServerGrpcClient::ModelConfig(
    ModelConfigResponse* model_config, const std::string& model_name,
    const std::string& model_version, const Headers& headers)
{
  model_config->Clear();
  Error err;

  ModelConfigRequest request;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_name(model_name);
  request.set_version(model_version);
  grpc::Status grpc_status =
      stub_->ModelConfig(&context, request, model_config);
  if (grpc_status.ok()) {
    if (verbose_) {
      std::cout << model_config->DebugString() << std::endl;
    }
  } else {
    err = Error(grpc_status.error_message());
  }

  return err;
}

Error
InferenceServerGrpcClient::ModelRepositoryIndex(
    RepositoryIndexResponse* repository_index, const Headers& headers)
{
  repository_index->Clear();
  Error err;

  RepositoryIndexRequest request;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  grpc::Status grpc_status =
      stub_->RepositoryIndex(&context, request, repository_index);
  if (grpc_status.ok()) {
    if (verbose_) {
      std::cout << repository_index->DebugString() << std::endl;
    }
  } else {
    err = Error(grpc_status.error_message());
  }

  return err;
}

Error
InferenceServerGrpcClient::LoadModel(
    const std::string& model_name, const Headers& headers)
{
  Error err;

  RepositoryModelLoadRequest request;
  RepositoryModelLoadResponse response;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_model_name(model_name);
  grpc::Status grpc_status =
      stub_->RepositoryModelLoad(&context, request, &response);
  if (!grpc_status.ok()) {
    err = Error(grpc_status.error_message());
  } else {
    if (verbose_) {
      std::cout << "Loaded model '" << model_name << "'" << std::endl;
    }
  }

  return err;
}

Error
InferenceServerGrpcClient::UnloadModel(
    const std::string& model_name, const Headers& headers)
{
  Error err;

  RepositoryModelUnloadRequest request;
  RepositoryModelUnloadResponse response;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_model_name(model_name);
  grpc::Status grpc_status =
      stub_->RepositoryModelUnload(&context, request, &response);
  if (!grpc_status.ok()) {
    err = Error(grpc_status.error_message());
  } else {
    if (verbose_) {
      std::cout << "Unloaded model '" << model_name << "'" << std::endl;
    }
  }

  return err;
}

Error
InferenceServerGrpcClient::ModelInferenceStatistics(
    ModelStatisticsResponse* infer_stat, const std::string& model_name,
    const std::string& model_version, const Headers& headers)
{
  infer_stat->Clear();
  Error err;

  ModelStatisticsRequest request;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_name(model_name);
  request.set_version(model_version);
  grpc::Status grpc_status =
      stub_->ModelStatistics(&context, request, infer_stat);
  if (grpc_status.ok()) {
    if (verbose_) {
      std::cout << infer_stat->DebugString() << std::endl;
    }
  } else {
    err = Error(grpc_status.error_message());
  }

  return err;
}

Error
InferenceServerGrpcClient::SystemSharedMemoryStatus(
    SystemSharedMemoryStatusResponse* status, const std::string& region_name,
    const Headers& headers)
{
  status->Clear();
  Error err;

  SystemSharedMemoryStatusRequest request;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_name(region_name);
  grpc::Status grpc_status =
      stub_->SystemSharedMemoryStatus(&context, request, status);
  if (grpc_status.ok()) {
    if (verbose_) {
      std::cout << status->DebugString() << std::endl;
    }
  } else {
    err = Error(grpc_status.error_message());
  }

  return err;
}

Error
InferenceServerGrpcClient::RegisterSystemSharedMemory(
    const std::string& name, const std::string& key, const size_t byte_size,
    const size_t offset, const Headers& headers)
{
  Error err;

  SystemSharedMemoryRegisterRequest request;
  SystemSharedMemoryRegisterResponse response;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_name(name);
  request.set_key(key);
  request.set_offset(offset);
  request.set_byte_size(byte_size);
  grpc::Status grpc_status =
      stub_->SystemSharedMemoryRegister(&context, request, &response);
  if (!grpc_status.ok()) {
    err = Error(grpc_status.error_message());
  } else {
    if (verbose_) {
      std::cout << "Registered system shared memory with name  '" << name << "'"
                << std::endl;
    }
  }

  return err;
}

Error
InferenceServerGrpcClient::UnregisterSystemSharedMemory(
    const std::string& name, const Headers& headers)
{
  Error err;

  SystemSharedMemoryUnregisterRequest request;
  SystemSharedMemoryUnregisterResponse response;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_name(name);
  grpc::Status grpc_status =
      stub_->SystemSharedMemoryUnregister(&context, request, &response);
  if (!grpc_status.ok()) {
    err = Error(grpc_status.error_message());
  } else {
    if (verbose_) {
      if (name.size() != 0) {
        std::cout << "Unregistered system shared memory with name '" << name
                  << "'" << std::endl;
      } else {
        std::cout << "Unregistered all system shared memory regions"
                  << std::endl;
      }
    }
  }

  return err;
}

Error
InferenceServerGrpcClient::CudaSharedMemoryStatus(
    CudaSharedMemoryStatusResponse* status, const std::string& region_name,
    const Headers& headers)
{
  status->Clear();
  Error err;

  CudaSharedMemoryStatusRequest request;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_name(region_name);
  grpc::Status grpc_status =
      stub_->CudaSharedMemoryStatus(&context, request, status);
  if (grpc_status.ok()) {
    if (verbose_) {
      std::cout << status->DebugString() << std::endl;
    }
  } else {
    err = Error(grpc_status.error_message());
  }

  return err;
}

Error
InferenceServerGrpcClient::RegisterCudaSharedMemory(
    const std::string& name, const cudaIpcMemHandle_t& cuda_shm_handle,
    const size_t device_id, const size_t byte_size, const Headers& headers)
{
  Error err;

  CudaSharedMemoryRegisterRequest request;
  CudaSharedMemoryRegisterResponse response;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_name(name);
  request.set_raw_handle((char*)&cuda_shm_handle, sizeof(cudaIpcMemHandle_t));
  request.set_device_id(device_id);
  request.set_byte_size(byte_size);
  grpc::Status grpc_status =
      stub_->CudaSharedMemoryRegister(&context, request, &response);
  if (!grpc_status.ok()) {
    err = Error(grpc_status.error_message());
  } else {
    if (verbose_) {
      std::cout << "Registered cuda shared memory with name  '" << name << "'"
                << std::endl;
    }
  }

  return err;
}

Error
InferenceServerGrpcClient::UnregisterCudaSharedMemory(
    const std::string& name, const Headers& headers)
{
  Error err;

  CudaSharedMemoryUnregisterRequest request;
  CudaSharedMemoryUnregisterResponse response;
  grpc::ClientContext context;

  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  request.set_name(name);
  grpc::Status grpc_status =
      stub_->CudaSharedMemoryUnregister(&context, request, &response);
  if (!grpc_status.ok()) {
    err = Error(grpc_status.error_message());
  } else {
    if (verbose_) {
      if (name.size() != 0) {
        std::cout << "Unregistered system shared memory with name '" << name
                  << "'" << std::endl;
      } else {
        std::cout << "Unregistered all system shared memory regions"
                  << std::endl;
      }
    }
  }

  return err;
}

Error
InferenceServerGrpcClient::Infer(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers)
{
  Error err;

  grpc::ClientContext context;

  std::shared_ptr<GrpcInferRequest> sync_request(new GrpcInferRequest());

  sync_request->Timer().Reset();
  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_START);
  // Use send timer to measure time for marshalling infer request
  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_START);
  for (const auto& it : headers) {
    context.AddMetadata(it.first, it.second);
  }

  if (options.client_timeout_ != 0) {
    auto deadline = std::chrono::system_clock::now() +
                    std::chrono::microseconds(options.client_timeout_);
    context.set_deadline(deadline);
  }

  err = PreRunProcessing(options, inputs, outputs);
  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_END);
  if (!err.IsOk()) {
    return err;
  }
  sync_request->grpc_response_->Clear();
  sync_request->grpc_status_ = stub_->ModelInfer(
      &context, infer_request_, sync_request->grpc_response_.get());

  if (!sync_request->grpc_status_.ok()) {
    err = Error(sync_request->grpc_status_.error_message());
  }

  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::RECV_START);
  InferResultGrpc::Create(result, sync_request->grpc_response_, err);
  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::RECV_END);

  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_END);

  err = UpdateInferStat(sync_request->Timer());
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }

  if (sync_request->grpc_status_.ok()) {
    if (verbose_) {
      std::cout << sync_request->grpc_response_->DebugString() << std::endl;
    }
  }

  return (*result)->RequestStatus();
}

Error
InferenceServerGrpcClient::AsyncInfer(
    OnCompleteFn callback, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers)
{
  if (callback == nullptr) {
    return Error(
        "Callback function must be provided along with AsyncInfer() call.");
  }
  if (!worker_.joinable()) {
    worker_ = std::thread(&InferenceServerGrpcClient::AsyncTransfer, this);
  }

  GrpcInferRequest* async_request;
  async_request = new GrpcInferRequest(std::move(callback));

  async_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_START);
  async_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_START);
  for (const auto& it : headers) {
    async_request->grpc_context_.AddMetadata(it.first, it.second);
  }

  if (options.client_timeout_ != 0) {
    auto deadline = std::chrono::system_clock::now() +
                    std::chrono::microseconds(options.client_timeout_);
    async_request->grpc_context_.set_deadline(deadline);
  }

  Error err = PreRunProcessing(options, inputs, outputs);
  if (!err.IsOk()) {
    delete async_request;
    return err;
  }

  async_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_END);

  std::unique_ptr<grpc::ClientAsyncResponseReader<ModelInferResponse>> rpc(
      stub_->PrepareAsyncModelInfer(
          &async_request->grpc_context_, infer_request_,
          &async_request_completion_queue_));

  rpc->StartCall();

  rpc->Finish(
      async_request->grpc_response_.get(), &async_request->grpc_status_,
      (void*)async_request);

  if (verbose_) {
    std::cout << "Sent request";
    if (options.request_id_.size() != 0) {
      std::cout << " '" << options.request_id_ << "'";
    }
    std::cout << std::endl;
  }

  return Error::Success;
}

Error
InferenceServerGrpcClient::StartStream(
    OnCompleteFn callback, bool enable_stats, uint32_t stream_timeout,
    const Headers& headers)
{
  if (stream_worker_.joinable()) {
    return Error(
        "cannot start another stream with one already running. "
        "'InferenceServerClient' supports only a single active "
        "stream at a given time.");
  }

  if (callback == nullptr) {
    return Error(
        "Callback function must be provided along with StartStream() call.");
  }

  stream_callback_ = callback;
  enable_stream_stats_ = enable_stats;

  for (const auto& it : headers) {
    grpc_context_.AddMetadata(it.first, it.second);
  }

  if (stream_timeout != 0) {
    auto deadline = std::chrono::system_clock::now() +
                    std::chrono::microseconds(stream_timeout);
    grpc_context_.set_deadline(deadline);
  }

  grpc_stream_ = stub_->ModelStreamInfer(&grpc_context_);
  stream_worker_ =
      std::thread(&InferenceServerGrpcClient::AsyncStreamTransfer, this);

  if (verbose_) {
    std::cout << "Started stream..." << std::endl;
  }

  return Error::Success;
}

Error
InferenceServerGrpcClient::StopStream()
{
  if (stream_worker_.joinable()) {
    grpc_stream_->WritesDone();
    // The reader thread will drain the stream properly
    stream_worker_.join();
    if (verbose_) {
      std::cout << "Stopped stream..." << std::endl;
    }
  }

  return Error::Success;
}

Error
InferenceServerGrpcClient::AsyncStreamInfer(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  std::unique_ptr<RequestTimers> timer;
  if (enable_stream_stats_) {
    timer.reset(new RequestTimers());
    timer->CaptureTimestamp(RequestTimers::Kind::REQUEST_START);
    timer->CaptureTimestamp(RequestTimers::Kind::SEND_START);
  }

  Error err = PreRunProcessing(options, inputs, outputs);
  if (!err.IsOk()) {
    return err;
  }

  if (enable_stream_stats_) {
    timer->CaptureTimestamp(RequestTimers::Kind::SEND_END);
  }

  if (enable_stream_stats_) {
    std::lock_guard<std::mutex> lock(stream_mutex_);
    ongoing_stream_request_timers_.push(std::move(timer));
  }
  bool ok = grpc_stream_->Write(infer_request_);

  if (ok) {
    if (verbose_) {
      std::cout << "Sent request";
      if (options.request_id_.size() != 0) {
        std::cout << " '" << options.request_id_ << "'";
      }
      std::cout << " to the stream" << std::endl;
    }
    return Error::Success;
  } else {
    return Error("Stream has been closed.");
  }
}

Error
InferenceServerGrpcClient::PreRunProcessing(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  // Populate the request protobuf
  infer_request_.set_model_name(options.model_name_);
  infer_request_.set_model_version(options.model_version_);
  infer_request_.set_id(options.request_id_);

  infer_request_.mutable_parameters()->clear();
  if (options.sequence_id_ != 0) {
    (*infer_request_.mutable_parameters())["sequence_id"].set_int64_param(
        options.sequence_id_);
    (*infer_request_.mutable_parameters())["sequence_start"].set_bool_param(
        options.sequence_start_);
    (*infer_request_.mutable_parameters())["sequence_end"].set_bool_param(
        options.sequence_end_);
  }

  if (options.priority_ != 0) {
    (*infer_request_.mutable_parameters())["priority"].set_int64_param(
        options.priority_);
  }

  if (options.server_timeout_ != 0) {
    (*infer_request_.mutable_parameters())["timeout"].set_int64_param(
        options.server_timeout_);
  }

  int index = 0;
  for (const auto input : inputs) {
    // Add new InferInputTensor submessages only if required, otherwise
    // reuse the submessages already available.
    auto grpc_input = (infer_request_.inputs().size() <= index)
                          ? infer_request_.add_inputs()
                          : infer_request_.mutable_inputs()->Mutable(index);

    if (input->IsSharedMemory()) {
      // The input contents must be cleared when using shared memory.
      grpc_input->Clear();
    }

    grpc_input->set_name(input->Name());
    grpc_input->mutable_shape()->Clear();
    for (const auto dim : input->Shape()) {
      grpc_input->mutable_shape()->Add(dim);
    }
    grpc_input->set_datatype(input->Datatype());

    input->PrepareForRequest();
    grpc_input->mutable_parameters()->clear();
    if (input->IsSharedMemory()) {
      std::string region_name;
      size_t offset;
      size_t byte_size;
      input->SharedMemoryInfo(&region_name, &byte_size, &offset);

      (*grpc_input->mutable_parameters())["shared_memory_region"]
          .set_string_param(region_name);
      (*grpc_input->mutable_parameters())["shared_memory_byte_size"]
          .set_int64_param(byte_size);
      if (offset != 0) {
        (*grpc_input->mutable_parameters())["shared_memory_offset"]
            .set_int64_param(offset);
      }
    } else {
      bool end_of_input = false;
      std::string* contents =
          grpc_input->mutable_contents()->mutable_raw_contents();
      size_t content_size;
      input->ByteSize(&content_size);
      contents->reserve(content_size);
      contents->clear();
      while (!end_of_input) {
        const uint8_t* buf;
        size_t buf_size;
        input->GetNext(&buf, &buf_size, &end_of_input);
        if (buf != nullptr) {
          contents->append(reinterpret_cast<const char*>(buf), buf_size);
        }
      }
    }
    index++;
  }

  // Remove extra InferInputTensor submessages, that are not required for
  // this request.
  while (index < infer_request_.inputs().size()) {
    infer_request_.mutable_inputs()->RemoveLast();
  }

  index = 0;
  for (const auto routput : outputs) {
    // Add new InferRequestedOutputTensor submessage only if required, otherwise
    // reuse the submessages already available.
    auto grpc_output = (infer_request_.outputs().size() <= index)
                           ? infer_request_.add_outputs()
                           : infer_request_.mutable_outputs()->Mutable(index);
    grpc_output->Clear();
    grpc_output->set_name(routput->Name());
    size_t class_count = routput->ClassificationCount();
    if (class_count != 0) {
      (*grpc_output->mutable_parameters())["classification"].set_int64_param(
          class_count);
    }
    if (routput->IsSharedMemory()) {
      std::string region_name;
      size_t offset;
      size_t byte_size;
      routput->SharedMemoryInfo(&region_name, &byte_size, &offset);
      (*grpc_output->mutable_parameters())["shared_memory_region"]
          .set_string_param(region_name);
      (*grpc_output->mutable_parameters())["shared_memory_byte_size"]
          .set_int64_param(byte_size);
      if (offset != 0) {
        (*grpc_output->mutable_parameters())["shared_memory_offset"]
            .set_int64_param(offset);
      }
    }
    index++;
  }

  // Remove extra InferRequestedOutputTensor submessages, that are not required
  // for this request.
  while (index < infer_request_.outputs().size()) {
    infer_request_.mutable_outputs()->RemoveLast();
  }

  if (infer_request_.ByteSizeLong() > INT_MAX) {
    size_t request_size = infer_request_.ByteSizeLong();
    infer_request_.Clear();
    return Error(
        "Request has byte size " + std::to_string(request_size) +
        " which exceed gRPC's byte size limit " + std::to_string(INT_MAX) +
        ".");
  }

  return Error::Success;
}

void
InferenceServerGrpcClient::AsyncTransfer()
{
  while (!exiting_) {
    // GRPC async APIs are thread-safe https://github.com/grpc/grpc/issues/4486
    GrpcInferRequest* raw_async_request;
    bool ok = true;
    bool status =
        async_request_completion_queue_.Next((void**)(&raw_async_request), &ok);
    std::shared_ptr<GrpcInferRequest> async_request;
    if (!ok) {
      fprintf(stderr, "Unexpected not ok on client side.\n");
    }
    if (!status) {
      if (!exiting_) {
        fprintf(stderr, "Completion queue is closed.\n");
      }
    } else if (raw_async_request == nullptr) {
      fprintf(stderr, "Unexpected null tag received at client.\n");
    } else {
      async_request.reset(raw_async_request);
      InferResult* async_result;
      Error err;
      if (!async_request->grpc_status_.ok()) {
        err = Error(async_request->grpc_status_.error_message());
      }
      async_request->Timer().CaptureTimestamp(RequestTimers::Kind::RECV_START);
      InferResultGrpc::Create(
          &async_result, async_request->grpc_response_, err);
      async_request->Timer().CaptureTimestamp(RequestTimers::Kind::RECV_END);
      async_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_END);
      err = UpdateInferStat(async_request->Timer());
      if (!err.IsOk()) {
        std::cerr << "Failed to update context stat: " << err << std::endl;
      }
      if (async_request->grpc_status_.ok()) {
        if (verbose_) {
          std::cout << async_request->grpc_response_->DebugString()
                    << std::endl;
        }
      }
      async_request->callback_(async_result);
    }
  }
}

void
InferenceServerGrpcClient::AsyncStreamTransfer()
{
  std::shared_ptr<ModelStreamInferResponse> response =
      std::make_shared<ModelStreamInferResponse>();
  // End loop if Read() returns false
  // (stream ended and all responses are drained)
  while (grpc_stream_->Read(response.get())) {
    if (exiting_) {
      continue;
    }

    std::unique_ptr<RequestTimers> timer;
    if (enable_stream_stats_) {
      std::lock_guard<std::mutex> lock(stream_mutex_);
      if (!ongoing_stream_request_timers_.empty()) {
        timer = std::move(ongoing_stream_request_timers_.front());
        ongoing_stream_request_timers_.pop();
      }
    }

    InferResult* stream_result;
    // FIXME, DLIS-1263 there is no 1:1 mapping between
    // requests and response for decoupled streaming case
    // hence, this method will record incorrect statistics
    // for decoupled case.
    if (timer.get() != nullptr) {
      timer->CaptureTimestamp(RequestTimers::Kind::RECV_START);
    }
    InferResultGrpc::Create(&stream_result, response);
    if (timer.get() != nullptr) {
      timer->CaptureTimestamp(RequestTimers::Kind::RECV_END);
      timer->CaptureTimestamp(RequestTimers::Kind::REQUEST_END);
      Error err = UpdateInferStat(*timer);
      if (!err.IsOk()) {
        std::cerr << "Failed to update context stat: " << err << std::endl;
      }
    }
    if (verbose_) {
      std::cout << response->DebugString() << std::endl;
    }
    stream_callback_(stream_result);
    response = std::make_shared<ModelStreamInferResponse>();
  }
  grpc_stream_->Finish();
}


InferenceServerGrpcClient::InferenceServerGrpcClient(
    const std::string& url, bool verbose, bool use_ssl,
    const SslOptions& ssl_options)
    : InferenceServerClient(verbose),
      stub_(
          GRPCInferenceService::NewStub(GetChannel(url, use_ssl, ssl_options)))
{
}

InferenceServerGrpcClient::~InferenceServerGrpcClient()
{
  exiting_ = true;
  // Close complete queue and wait for the worker thread to return
  async_request_completion_queue_.Shutdown();

  // thread not joinable if AsyncInfer() is not called
  // (it is default constructed thread before the first AsyncInfer() call)
  if (worker_.joinable()) {
    worker_.join();
  }

  bool has_next = true;
  GrpcInferRequest* async_request;
  bool ok;
  do {
    has_next =
        async_request_completion_queue_.Next((void**)&async_request, &ok);
    if (has_next && async_request != nullptr) {
      delete async_request;
    }
  } while (has_next);

  StopStream();
}

//==============================================================================

}}}  // namespace nvidia::inferenceserver::client
