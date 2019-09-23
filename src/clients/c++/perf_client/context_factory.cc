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

#include "src/clients/c++/perf_client/context_factory.h"

namespace perfclient {
nic::Error
ContextFactory::Create(
    const std::string& url, const ProtocolType protocol,
    const std::map<std::string, std::string>& http_headers,
    const bool streaming, const std::string& model_name,
    const int64_t model_version, std::shared_ptr<ContextFactory>* factory)
{
  factory->reset(new ContextFactory(
      url, protocol, http_headers, streaming, model_name, model_version));

  ni::ServerStatus server_status;
  std::unique_ptr<nic::ServerStatusContext> ctx;
  (*factory)->CreateServerStatusContext(&ctx);
  RETURN_IF_ERROR(ctx->GetServerStatus(&server_status));
  const auto& itr = server_status.model_status().find(model_name);
  if (itr == server_status.model_status().end()) {
    return nic::Error(
        ni::RequestStatusCode::INTERNAL, "unable to find status for model");
  } else {
    if (itr->second.config().has_sequence_batching()) {
      (*factory)->scheduler_type_ = SEQUENCE;
    } else if (itr->second.config().has_ensemble_scheduling()) {
      (*factory)->scheduler_type_ = ENSEMBLE;
    } else if (itr->second.config().has_dynamic_batching()) {
      (*factory)->scheduler_type_ = DYNAMIC;
    } else {
      (*factory)->scheduler_type_ = NONE;
    }
  }
  return nic::Error::Success;
}

nic::Error
ContextFactory::CreateServerStatusContext(
    std::unique_ptr<nic::ServerStatusContext>* ctx)
{
  nic::Error err;
  if (protocol_ == ProtocolType::HTTP) {
    err = nic::ServerStatusHttpContext::Create(ctx, url_, http_headers_, false);
  } else {
    err = nic::ServerStatusGrpcContext::Create(ctx, url_, false);
  }
  return err;
}

nic::Error
ContextFactory::CreateInferContext(std::unique_ptr<nic::InferContext>* ctx)
{
  nic::Error err;
  // Create the context for inference of the specified model,
  // make sure to use an unused correlation id if requested.
  ni::CorrelationID correlation_id = 0;

  if (scheduler_type_ == SEQUENCE) {
    std::lock_guard<std::mutex> lock(correlation_id_mutex_);
    current_correlation_id_++;
    correlation_id = current_correlation_id_;
  }

  if (streaming_) {
    err = nic::InferGrpcStreamContext::Create(
        ctx, correlation_id, url_, model_name_, model_version_, false);
  } else if (protocol_ == ProtocolType::HTTP) {
    err = nic::InferHttpContext::Create(
        ctx, correlation_id, url_, http_headers_, model_name_, model_version_,
        false);
  } else {
    err = nic::InferGrpcContext::Create(
        ctx, correlation_id, url_, model_name_, model_version_, false);
  }
  return err;
}

}  // namespace perfclient
