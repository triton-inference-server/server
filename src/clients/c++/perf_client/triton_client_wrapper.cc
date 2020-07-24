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

#include "src/clients/c++/perf_client/triton_client_wrapper.h"

#include "src/clients/c++/examples/json_utils.h"

//==============================================================================

nic::Error
TritonClientFactory::Create(
    const std::string& url, const ProtocolType protocol,
    std::shared_ptr<nic::Headers> http_headers, const bool verbose,
    std::shared_ptr<TritonClientFactory>* factory)
{
  factory->reset(new TritonClientFactory(url, protocol, http_headers, verbose));
  return nic::Error::Success;
}

nic::Error
TritonClientFactory::CreateTritonClient(
    std::unique_ptr<TritonClientWrapper>* client)
{
  RETURN_IF_ERROR(TritonClientWrapper::Create(
      url_, protocol_, http_headers_, verbose_, client));
  return nic::Error::Success;
}

//==============================================================================

nic::Error
TritonClientWrapper::Create(
    const std::string& url, const ProtocolType protocol,
    std::shared_ptr<nic::Headers> http_headers, const bool verbose,
    std::unique_ptr<TritonClientWrapper>* triton_client_wrapper)
{
  triton_client_wrapper->reset(new TritonClientWrapper(protocol, http_headers));
  if (protocol == ProtocolType::HTTP) {
    RETURN_IF_ERROR(nic::InferenceServerHttpClient::Create(
        &((*triton_client_wrapper)->client_.http_client_), url, verbose));
  } else {
    RETURN_IF_ERROR(nic::InferenceServerGrpcClient::Create(
        &((*triton_client_wrapper)->client_.grpc_client_), url, verbose));
  }
  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::ModelMetadata(
    rapidjson::Document* model_metadata, const std::string& model_name,
    const std::string& model_version)
{
  if (protocol_ == ProtocolType::HTTP) {
    std::string metadata;
    RETURN_IF_ERROR(client_.http_client_->ModelMetadata(
        &metadata, model_name, model_version, *http_headers_));
    RETURN_IF_ERROR(nic::ParseJson(model_metadata, metadata));
  } else {
    return nic::Error("gRPC can not return model metadata as json");
  }

  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::ModelMetadata(
    inference::ModelMetadataResponse* model_metadata, const std::string& model_name,
    const std::string& model_version)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(client_.grpc_client_->ModelMetadata(
        model_metadata, model_name, model_version, *http_headers_));
  } else {
    return nic::Error("HTTP can not return model metadata as protobuf message");
  }

  return nic::Error::Success;
}


nic::Error
TritonClientWrapper::ModelConfig(
    rapidjson::Document* model_config, const std::string& model_name,
    const std::string& model_version)
{
  if (protocol_ == ProtocolType::HTTP) {
    std::string config;
    RETURN_IF_ERROR(client_.http_client_->ModelConfig(
        &config, model_name, model_version, *http_headers_));
    RETURN_IF_ERROR(nic::ParseJson(model_config, config));
  } else {
    return nic::Error("gRPC can not return model config as json");
  }
  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::ModelConfig(
    inference::ModelConfigResponse* model_config, const std::string& model_name,
    const std::string& model_version)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(client_.grpc_client_->ModelConfig(
        model_config, model_name, model_version, *http_headers_));
  } else {
    return nic::Error("HTTP can not return model config as protobuf message");
  }
  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::Infer(
    nic::InferResult** result, const nic::InferOptions& options,
    const std::vector<nic::InferInput*>& inputs,
    const std::vector<const nic::InferRequestedOutput*>& outputs)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(client_.grpc_client_->Infer(
        result, options, inputs, outputs, *http_headers_));
  } else {
    RETURN_IF_ERROR(client_.http_client_->Infer(
        result, options, inputs, outputs, *http_headers_));
  }

  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::AsyncInfer(
    nic::InferenceServerClient::OnCompleteFn callback,
    const nic::InferOptions& options,
    const std::vector<nic::InferInput*>& inputs,
    const std::vector<const nic::InferRequestedOutput*>& outputs)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(client_.grpc_client_->AsyncInfer(
        callback, options, inputs, outputs, *http_headers_));
  } else {
    RETURN_IF_ERROR(client_.http_client_->AsyncInfer(
        callback, options, inputs, outputs, *http_headers_));
  }

  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::StartStream(
    nic::InferenceServerClient::OnCompleteFn callback, bool enable_stats)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(client_.grpc_client_->StartStream(
        callback, enable_stats, 0 /* stream_timeout */, *http_headers_));
  } else {
    return nic::Error("HTTP does not support starting streams");
  }

  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::AsyncStreamInfer(
    const nic::InferOptions& options,
    const std::vector<nic::InferInput*>& inputs,
    const std::vector<const nic::InferRequestedOutput*>& outputs)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(
        client_.grpc_client_->AsyncStreamInfer(options, inputs, outputs));
  } else {
    return nic::Error("HTTP does not support streaming inferences");
  }

  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::ClientInferStat(nic::InferStat* infer_stat)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(client_.grpc_client_->ClientInferStat(infer_stat));
  } else {
    RETURN_IF_ERROR(client_.http_client_->ClientInferStat(infer_stat));
  }
  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::ModelInferenceStatistics(
    std::map<ModelIdentifier, ModelStatistics>* model_stats,
    const std::string& model_name, const std::string& model_version)
{
  if (protocol_ == ProtocolType::GRPC) {
    inference::ModelStatisticsResponse infer_stat;
    RETURN_IF_ERROR(client_.grpc_client_->ModelInferenceStatistics(
        &infer_stat, model_name, model_version, *http_headers_));
    ParseStatistics(infer_stat, model_stats);
  } else {
    std::string infer_stat;
    RETURN_IF_ERROR(client_.http_client_->ModelInferenceStatistics(
        &infer_stat, model_name, model_version, *http_headers_));
    rapidjson::Document infer_stat_json;
    RETURN_IF_ERROR(nic::ParseJson(&infer_stat_json, infer_stat));
    ParseStatistics(infer_stat_json, model_stats);
  }

  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::UnregisterAllSharedMemory()
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(
        client_.grpc_client_->UnregisterSystemSharedMemory("", *http_headers_));
    RETURN_IF_ERROR(
        client_.grpc_client_->UnregisterCudaSharedMemory("", *http_headers_));
  } else {
    RETURN_IF_ERROR(
        client_.http_client_->UnregisterSystemSharedMemory("", *http_headers_));
    RETURN_IF_ERROR(
        client_.http_client_->UnregisterCudaSharedMemory("", *http_headers_));
  }

  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::RegisterSystemSharedMemory(
    const std::string& name, const std::string& key, const size_t byte_size)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(client_.grpc_client_->RegisterSystemSharedMemory(
        name, key, byte_size, 0 /* offset */, *http_headers_));

  } else {
    RETURN_IF_ERROR(client_.http_client_->RegisterSystemSharedMemory(
        name, key, byte_size, 0 /* offset */, *http_headers_));
  }

  return nic::Error::Success;
}

nic::Error
TritonClientWrapper::RegisterCudaSharedMemory(
    const std::string& name, const cudaIpcMemHandle_t& handle,
    const size_t byte_size)
{
  if (protocol_ == ProtocolType::GRPC) {
    RETURN_IF_ERROR(client_.grpc_client_->RegisterCudaSharedMemory(
        name, handle, 0 /*device id*/, byte_size, *http_headers_));

  } else {
    RETURN_IF_ERROR(client_.http_client_->RegisterCudaSharedMemory(
        name, handle, 0 /*device id*/, byte_size, *http_headers_));
  }

  return nic::Error::Success;
}

void
TritonClientWrapper::ParseStatistics(
    inference::ModelStatisticsResponse& infer_stat,
    std::map<ModelIdentifier, ModelStatistics>* model_stats)
{
  model_stats->clear();
  for (const auto& this_stat : infer_stat.model_stats()) {
    auto it = model_stats
                  ->emplace(
                      std::make_pair(this_stat.name(), this_stat.version()),
                      ModelStatistics())
                  .first;
    it->second.inference_count_ = this_stat.inference_count();
    it->second.execution_count_ = this_stat.execution_count();
    it->second.success_count_ = this_stat.inference_stats().success().count();
    it->second.cumm_time_ns_ = this_stat.inference_stats().success().ns();
    it->second.queue_time_ns_ = this_stat.inference_stats().queue().ns();
    it->second.compute_input_time_ns_ =
        this_stat.inference_stats().compute_input().ns();
    it->second.compute_infer_time_ns_ =
        this_stat.inference_stats().compute_infer().ns();
    it->second.compute_output_time_ns_ =
        this_stat.inference_stats().compute_output().ns();
  }
}

void
TritonClientWrapper::ParseStatistics(
    rapidjson::Document& infer_stat,
    std::map<ModelIdentifier, ModelStatistics>* model_stats)
{
  model_stats->clear();
  for (const auto& this_stat : infer_stat["model_stats"].GetArray()) {
    auto it = model_stats
                  ->emplace(
                      std::make_pair(
                          this_stat["name"].GetString(),
                          this_stat["version"].GetString()),
                      ModelStatistics())
                  .first;
    it->second.inference_count_ = this_stat["inference_count"].GetUint64();
    it->second.execution_count_ = this_stat["execution_count"].GetUint64();
    it->second.success_count_ =
        this_stat["inference_stats"]["success"]["count"].GetUint64();
    it->second.cumm_time_ns_ =
        this_stat["inference_stats"]["success"]["ns"].GetUint64();
    it->second.queue_time_ns_ =
        this_stat["inference_stats"]["queue"]["ns"].GetUint64();
    it->second.compute_input_time_ns_ =
        this_stat["inference_stats"]["compute_input"]["ns"].GetUint64();
    it->second.compute_infer_time_ns_ =
        this_stat["inference_stats"]["compute_infer"]["ns"].GetUint64();
    it->second.compute_output_time_ns_ =
        this_stat["inference_stats"]["compute_output"]["ns"].GetUint64();
  }
}

//==============================================================================
