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

#include <string>

#include "src/clients/c++/perf_client/perf_utils.h"

using ModelIdentifier = std::pair<std::string, std::string>;

struct ModelStatistics {
  uint64_t success_count_;
  uint64_t inference_count_;
  uint64_t execution_count_;
  uint64_t cumm_time_ns_;
  uint64_t queue_time_ns_;
  uint64_t compute_input_time_ns_;
  uint64_t compute_infer_time_ns_;
  uint64_t compute_output_time_ns_;
};

//==============================================================================
/// TritonClientWrapper is a light-weight wrapper around the triton client
/// library and is responsible for invoking right clients for a set protocol.
///
class TritonClientWrapper {
 public:
  /// Create a triton client which can be used to interact with the server.
  /// \param url The inference server name and port.
  /// \param protocol The protocol type used.
  /// \param http_headers Map of HTTP headers. The map key/value
  /// indicates the header name/value.
  /// \param verbose Enables the verbose mode.
  /// \param triton_client Returns a new TritonClientWrapper object.
  /// \return Error object indicating success or failure.
  static nic::Error Create(
      const std::string& url, const ProtocolType protocol,
      std::shared_ptr<nic::Headers> http_headers, const bool verbose,
      std::unique_ptr<TritonClientWrapper>* triton_client);

  /// Get the model metadata from the server for specified name and
  /// version as rapidjson DOM object.
  nic::Error ModelMetadata(
      rapidjson::Document* model_metadata, const std::string& model_name,
      const std::string& model_version);

  /// Get the model metadata from the server for specified name and
  /// version as message.
  nic::Error ModelMetadata(
      inference::ModelMetadataResponse* model_metadata, const std::string& model_name,
      const std::string& model_version);

  /// Get the model config from the server for specified name and
  /// version as rapidjson DOM object.
  nic::Error ModelConfig(
      rapidjson::Document* model_config, const std::string& model_name,
      const std::string& model_version);

  /// Get the model config from the server for specified name and
  /// version as message.
  nic::Error ModelConfig(
      inference::ModelConfigResponse* model_config, const std::string& model_name,
      const std::string& model_version);

  /// Issues a synchronous inference request to the server.
  nic::Error Infer(
      nic::InferResult** result, const nic::InferOptions& options,
      const std::vector<nic::InferInput*>& inputs,
      const std::vector<const nic::InferRequestedOutput*>& outputs);

  /// Issues an asynchronous inference request to the server.
  nic::Error AsyncInfer(
      nic::InferenceServerClient::OnCompleteFn callback,
      const nic::InferOptions& options,
      const std::vector<nic::InferInput*>& inputs,
      const std::vector<const nic::InferRequestedOutput*>& outputs);

  /// Established a stream to the server.
  nic::Error StartStream(
      nic::InferenceServerClient::OnCompleteFn callback, bool enable_stats);

  /// Issues an asynchronous inference request to the underlying stream.
  nic::Error AsyncStreamInfer(
      const nic::InferOptions& options,
      const std::vector<nic::InferInput*>& inputs,
      const std::vector<const nic::InferRequestedOutput*>& outputs);

  /// Gets the client side inference statistics from the client library.
  nic::Error ClientInferStat(nic::InferStat* infer_stat);

  /// Gets the server-side model inference statistics from the server.
  nic::Error ModelInferenceStatistics(
      std::map<ModelIdentifier, ModelStatistics>* model_stats,
      const std::string& model_name = "",
      const std::string& model_version = "");

  /// Unregisters all the shared memory from the server
  nic::Error UnregisterAllSharedMemory();

  /// Registers a system shared memory from the server
  nic::Error RegisterSystemSharedMemory(
      const std::string& name, const std::string& key, const size_t byte_size);

  /// Registers cuda shared memory to the server.
  nic::Error RegisterCudaSharedMemory(
      const std::string& name, const cudaIpcMemHandle_t& handle,
      const size_t byte_size);

 private:
  TritonClientWrapper(
      const ProtocolType protocol, std::shared_ptr<nic::Headers> http_headers)
      : protocol_(protocol), http_headers_(http_headers)
  {
  }

  void ParseStatistics(
      inference::ModelStatisticsResponse& infer_stat,
      std::map<ModelIdentifier, ModelStatistics>* model_stats);

  void ParseStatistics(
      rapidjson::Document& infer_stat,
      std::map<ModelIdentifier, ModelStatistics>* model_stats);

  /// Union to represent the underlying triton client belonging to one of
  /// the protocols
  union TritonClient {
    TritonClient()
    {
      new (&http_client_) std::unique_ptr<nic::InferenceServerHttpClient>{};
    }
    ~TritonClient() {}

    std::unique_ptr<nic::InferenceServerHttpClient> http_client_;
    std::unique_ptr<nic::InferenceServerGrpcClient> grpc_client_;
  } client_;

  const ProtocolType protocol_;
  std::shared_ptr<nic::Headers> http_headers_;
};

class TritonClientFactory {
 public:
  /// Create a factory that can be used to construct TritonClients.
  /// \param url The inference server name and port.
  /// \param protocol The protocol type used.
  /// \param http_headers Map of HTTP headers. The map key/value
  /// indicates the header name/value. The headers will be included
  /// with all the requests made to server using this client.
  /// \param verbose Enables the verbose mode.
  /// \param factory Returns a new TritonClientWrapper object.
  /// \return Error object indicating success or failure.
  static nic::Error Create(
      const std::string& url, const ProtocolType protocol,
      std::shared_ptr<nic::Headers> http_headers, const bool verbose,
      std::shared_ptr<TritonClientFactory>* factory);

  /// Create a TritonClientWrapper.
  /// \param client Returns a new TritonClientWrapper object.
  nic::Error CreateTritonClient(std::unique_ptr<TritonClientWrapper>* client);

 private:
  TritonClientFactory(
      const std::string& url, const ProtocolType protocol,
      const std::shared_ptr<nic::Headers> http_headers, const bool verbose)
      : url_(url), protocol_(protocol), http_headers_(http_headers),
        verbose_(verbose)
  {
  }

  const std::string url_;
  const ProtocolType protocol_;
  std::shared_ptr<nic::Headers> http_headers_;
  const bool verbose_;
};
