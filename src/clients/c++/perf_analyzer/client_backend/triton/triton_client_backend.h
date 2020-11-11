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

#include "src/clients/c++/perf_analyzer/client_backend/client_backend.h"
#include "src/clients/c++/perf_analyzer/perf_utils.h"

#include "src/clients/c++/examples/shm_utils.h"
#include "src/clients/c++/library/grpc_client.h"
#include "src/clients/c++/library/http_client.h"

#define RETURN_IF_TRITON_ERROR(S)       \
  do {                                  \
    const nic::Error& status__ = (S);   \
    if (!status__.IsOk()) {             \
      return Error(status__.Message()); \
    }                                   \
  } while (false)

#define FAIL_IF_TRITON_ERR(X, MSG)                                 \
  {                                                                \
    const nic::Error err = (X);                                    \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace nic = nvidia::inferenceserver::client;

namespace perfanalyzer { namespace clientbackend {

//==============================================================================
/// TritonClientBackend uses triton client C++ library to communicate with
/// triton inference service.
///
class TritonClientBackend : public ClientBackend {
 public:
  /// Create a triton client backend which can be used to interact with the
  /// server.
  /// \param url The inference server url and port.
  /// \param protocol The protocol type used.
  /// \param http_headers Map of HTTP headers. The map key/value indicates
  /// the header name/value.
  /// \param verbose Enables the verbose mode.
  /// \param client_backend Returns a new TritonClientBackend
  /// object.
  /// \return Error object indicating success or failure.
  static Error Create(
      const std::string& url, const ProtocolType protocol,
      std::shared_ptr<nic::Headers> http_headers, const bool verbose,
      std::unique_ptr<ClientBackend>* client_backend);

  /// See ClientBackend::ServerExtensions()
  Error ServerExtensions(std::set<std::string>* server_extensions) override;

  /// See ClientBackend::ModelMetadata()
  Error ModelMetadata(
      rapidjson::Document* model_metadata, const std::string& model_name,
      const std::string& model_version) override;

  /// See ClientBackend::ModelConfig()
  Error ModelConfig(
      rapidjson::Document* model_config, const std::string& model_name,
      const std::string& model_version) override;

  /// See ClientBackend::Infer()
  Error Infer(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override;

  /// See ClientBackend::AsyncInfer()
  Error AsyncInfer(
      OnCompleteFn callback, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override;

  /// See ClientBackend::StartStream()
  Error StartStream(OnCompleteFn callback, bool enable_stats) override;

  /// See ClientBackend::AsyncStreamInfer()
  Error AsyncStreamInfer(
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override;

  /// See ClientBackend::ClientInferStat()
  Error ClientInferStat(InferStat* infer_stat) override;

  /// See ClientBackend::ModelInferenceStatistics()
  Error ModelInferenceStatistics(
      std::map<ModelIdentifier, ModelStatistics>* model_stats,
      const std::string& model_name = "",
      const std::string& model_version = "") override;

  /// See ClientBackend::UnregisterAllSharedMemory()
  Error UnregisterAllSharedMemory() override;

  /// See ClientBackend::RegisterSystemSharedMemory()
  Error RegisterSystemSharedMemory(
      const std::string& name, const std::string& key,
      const size_t byte_size) override;

  /// See ClientBackend::RegisterCudaSharedMemory()
  Error RegisterCudaSharedMemory(
      const std::string& name, const cudaIpcMemHandle_t& handle,
      const size_t byte_size) override;

  /// See ClientBackend::CreateSharedMemoryRegion()
  Error CreateSharedMemoryRegion(
      std::string shm_key, size_t byte_size, int* shm_fd) override;

  /// See ClientBackend::MapSharedMemory()
  Error MapSharedMemory(
      int shm_fd, size_t offset, size_t byte_size, void** shm_addr) override;

  /// See ClientBackend::CloseSharedMemory()
  Error CloseSharedMemory(int shm_fd) override;

  /// See ClientBackend::UnlinkSharedMemoryRegion()
  Error UnlinkSharedMemoryRegion(std::string shm_key) override;

  /// See ClientBackend::UnmapSharedMemory()
  Error UnmapSharedMemory(void* shm_addr, size_t byte_size) override;

 private:
  TritonClientBackend(
      const ProtocolType protocol, std::shared_ptr<nic::Headers> http_headers)
      : ClientBackend(BackendKind::TRITON), protocol_(protocol),
        http_headers_(http_headers)
  {
  }

  void ParseInferInputToTriton(
      const std::vector<InferInput*>& inputs,
      std::vector<nic::InferInput*>* triton_inputs);
  void ParseInferRequestedOutputToTriton(
      const std::vector<const InferRequestedOutput*>& outputs,
      std::vector<const nic::InferRequestedOutput*>* triton_outputs);
  void ParseInferOptionsToTriton(
      const InferOptions& options, nic::InferOptions* triton_options);
  void ParseStatistics(
      const inference::ModelStatisticsResponse& infer_stat,
      std::map<ModelIdentifier, ModelStatistics>* model_stats);
  void ParseStatistics(
      const rapidjson::Document& infer_stat,
      std::map<ModelIdentifier, ModelStatistics>* model_stats);
  void ParseInferStat(
      const nic::InferStat& triton_infer_stat, InferStat* infer_stat);

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

//==============================================================
/// TritonInferInput is a wrapper around InferInput object of
/// triton client library.
///
class TritonInferInput : public InferInput {
 public:
  static Error Create(
      InferInput** infer_input, const std::string& name,
      const std::vector<int64_t>& dims, const std::string& datatype);
  /// Returns the raw InferInput object required by triton client library.
  nic::InferInput* Get() const { return input_.get(); }
  /// See InferInput::Shape()
  const std::vector<int64_t>& Shape() const override;
  /// See InferInput::SetShape()
  Error SetShape(const std::vector<int64_t>& shape) override;
  /// See InferInput::Reset()
  Error Reset() override;
  /// See InferInput::AppendRaw()
  Error AppendRaw(const uint8_t* input, size_t input_byte_size) override;
  /// See InferInput::SetSharedMemory()
  Error SetSharedMemory(
      const std::string& name, size_t byte_size, size_t offset = 0) override;

 private:
  explicit TritonInferInput(
      const std::string& name, const std::string& datatype);

  std::unique_ptr<nic::InferInput> input_;
};

//==============================================================
/// TritonInferRequestedOutput is a wrapper around
/// InferRequestedOutput object of triton client library.
///
class TritonInferRequestedOutput : public InferRequestedOutput {
 public:
  static Error Create(
      InferRequestedOutput** infer_output, const std::string name,
      const size_t class_count = 0);
  /// Returns the raw InferRequestedOutput object required by triton client
  /// library.
  nic::InferRequestedOutput* Get() const { return output_.get(); }
  // See InferRequestedOutput::SetSharedMemory()
  Error SetSharedMemory(
      const std::string& region_name, const size_t byte_size,
      const size_t offset = 0) override;

 private:
  explicit TritonInferRequestedOutput();

  std::unique_ptr<nic::InferRequestedOutput> output_;
};

//==============================================================
/// TritonInferResult is a wrapper around InferResult object of
/// triton client library.
///
class TritonInferResult : public InferResult {
 public:
  explicit TritonInferResult(nic::InferResult* result);
  /// See InferResult::Id()
  Error Id(std::string* id) const override;
  /// See InferResult::RequestStatus()
  Error RequestStatus() const override;

 private:
  std::unique_ptr<nic::InferResult> result_;
};

}}  // namespace perfanalyzer::clientbackend
