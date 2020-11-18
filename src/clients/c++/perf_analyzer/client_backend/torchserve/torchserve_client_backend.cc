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

#include "src/clients/c++/perf_analyzer/client_backend/torchserve/torchserve_client_backend.h"
#include "src/clients/c++/examples/json_utils.h"

namespace perfanalyzer { namespace clientbackend {

//==============================================================================

Error
TorchServeClientBackend::Create(
    const std::string& url, const ProtocolType protocol,
    std::shared_ptr<Headers> http_headers, const bool verbose,
    std::unique_ptr<ClientBackend>* client_backend)
{
  if (protocol == ProtocolType::GRPC) {
    return Error(
        "perf_analyzer does not support gRPC protocol with TorchServe");
  }
  std::unique_ptr<TorchServeClientBackend> torchserve_client_backend(
      new TorchServeClientBackend(http_headers));
  RETURN_IF_CB_ERROR(ts::HttpClient::Create(
      &(torchserve_client_backend->http_client_), url, verbose));
  *client_backend = std::move(torchserve_client_backend);
  return Error::Success;
}

Error
TorchServeClientBackend::Infer(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  ts::InferResult* torchserve_result;
  RETURN_IF_CB_ERROR(http_client_->Infer(
      &torchserve_result, options, inputs, outputs, *http_headers_));
  *result = new TorchServeInferResult(torchserve_result);
  return Error::Success;
}

Error
TorchServeClientBackend::ClientInferStat(InferStat* infer_stat)
{
  // Reusing the common library utilities to collect and report the
  // client side statistics.
  nic::InferStat client_infer_stat;
  RETURN_IF_TRITON_ERROR(http_client_->ClientInferStat(&client_infer_stat));
  ParseInferStat(client_infer_stat, infer_stat);
  return Error::Success;
}

void
TorchServeClientBackend::ParseInferStat(
    const nic::InferStat& torchserve_infer_stat, InferStat* infer_stat)
{
  infer_stat->completed_request_count =
      torchserve_infer_stat.completed_request_count;
  infer_stat->cumulative_total_request_time_ns =
      torchserve_infer_stat.cumulative_total_request_time_ns;
  infer_stat->cumulative_send_time_ns =
      torchserve_infer_stat.cumulative_send_time_ns;
  infer_stat->cumulative_receive_time_ns =
      torchserve_infer_stat.cumulative_receive_time_ns;
}

//==============================================================================

TorchServeInferResult::TorchServeInferResult(ts::InferResult* result)
{
  result_.reset(result);
}

Error
TorchServeInferResult::Id(std::string* id) const
{
  id->clear();
  return Error::Success;
}

Error
TorchServeInferResult::RequestStatus() const
{
  RETURN_IF_CB_ERROR(result_->RequestStatus());
  return Error::Success;
}

//==============================================================================

}}  // namespace perfanalyzer::clientbackend
