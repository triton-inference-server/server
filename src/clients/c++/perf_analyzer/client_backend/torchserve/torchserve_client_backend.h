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

#include "src/clients/c++/perf_analyzer/client_backend/torchserve/http_client.h"

#define RETURN_IF_TRITON_ERROR(S)       \
  do {                                  \
    const nic::Error& status__ = (S);   \
    if (!status__.IsOk()) {             \
      return Error(status__.Message()); \
    }                                   \
  } while (false)

namespace nic = nvidia::inferenceserver::client;
namespace ts = perfanalyzer::clientbackend::torchserve;

namespace perfanalyzer { namespace clientbackend {

//==============================================================================
/// TorchServeClientBackend is used to generate load on the Torchserve isntance
///
class TorchServeClientBackend : public ClientBackend {
 public:
  /// Create a torchserve client backend which can be used to interact with the
  /// server.
  /// \param url The inference server url and port.
  /// \param protocol The protocol type used.
  /// \param http_headers Map of HTTP headers. The map key/value indicates
  /// the header name/value.
  /// \param verbose Enables the verbose mode.
  /// \param client_backend Returns a new TorchServeClientBackend
  /// object.
  /// \return Error object indicating success or failure.
  static Error Create(
      const std::string& url, const ProtocolType protocol,
      std::shared_ptr<Headers> http_headers, const bool verbose,
      std::unique_ptr<ClientBackend>* client_backend);

  /// See ClientBackend::Infer()
  Error Infer(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs) override;

  /// See ClientBackend::ClientInferStat()
  Error ClientInferStat(InferStat* infer_stat) override;

 private:
  TorchServeClientBackend(std::shared_ptr<Headers> http_headers)
      : ClientBackend(BackendKind::TORCHSERVE), http_headers_(http_headers)
  {
  }

  void ParseInferStat(
      const nic::InferStat& torchserve_infer_stat, InferStat* infer_stat);

  std::unique_ptr<ts::HttpClient> http_client_;

  std::shared_ptr<Headers> http_headers_;
};

//==============================================================
/// TorchServeInferRequestedOutput is a wrapper around
/// InferRequestedOutput object of triton common client library.
///
class TorchServeInferRequestedOutput : public InferRequestedOutput {
 public:
  static Error Create(
      InferRequestedOutput** infer_output, const std::string name);
  /// Returns the raw InferRequestedOutput object required by torchserve
  /// client library.
  nic::InferRequestedOutput* Get() const { return output_.get(); }

 private:
  explicit TorchServeInferRequestedOutput();

  std::unique_ptr<nic::InferRequestedOutput> output_;
};

//==============================================================
/// TorchServeInferResult is a wrapper around InferResult object of
/// torchserve InferResult object.
///
class TorchServeInferResult : public InferResult {
 public:
  explicit TorchServeInferResult(ts::InferResult* result);
  /// See InferResult::Id()
  Error Id(std::string* id) const override;
  /// See InferResult::RequestStatus()
  Error RequestStatus() const override;

 private:
  std::unique_ptr<ts::InferResult> result_;
};

}}  // namespace perfanalyzer::clientbackend
