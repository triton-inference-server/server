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

#ifdef TRTIS_ENABLE_TRACING

#include "src/core/tracing.h"

#include <cppkin.h>

namespace nvidia { namespace inferenceserver {

std::unique_ptr<TraceManager> TraceManager::singleton_;

Status
TraceManager::Create(
    const std::string& trace_name, const std::string& hostname, uint32_t port)
{
  // If trace object is already created then configure has already
  // been called. Can only configure once since the zipkin library we
  // are using doesn't allow reconfiguration.
  if (singleton_ != nullptr) {
    return Status(
        RequestStatusCode::ALREADY_EXISTS, "tracing is already configured");
  }

  if (trace_name.empty()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "trace configuration requires a non-empty trace name");
  }

  if (hostname.empty()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "trace configuration requires a non-empty host name");
  }

  if (port == 0) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "trace configuration requires a non-zero port");
  }

  singleton_.reset(new TraceManager(trace_name, hostname, port));
  return Status::Success;
}

TraceManager::TraceManager(
    const std::string& trace_name, const std::string& hostname, uint32_t port)
    : level_(0 /* disabled */), rate_(1000)
{
  cppkin::CppkinParams cppkin_params;
  cppkin_params.AddParam(cppkin::ConfigTags::HOST_ADDRESS, hostname);
  cppkin_params.AddParam(cppkin::ConfigTags::PORT, port);
  cppkin_params.AddParam(cppkin::ConfigTags::SERVICE_NAME, trace_name);
  cppkin_params.AddParam(cppkin::ConfigTags::SAMPLE_COUNT, 1);
  cppkin_params.AddParam(cppkin::ConfigTags::TRANSPORT_TYPE, "http");
  cppkin_params.AddParam(cppkin::ConfigTags::ENCODING_TYPE, "json");

  cppkin::Init(cppkin_params);
}

TraceManager::~TraceManager()
{
  cppkin::Stop();
}

Status
TraceManager::SetLevel(uint32_t level, uint32_t rate)
{
  if (singleton_ == nullptr) {
    return Status(
        RequestStatusCode::UNAVAILABLE, "tracing is not yet configured");
  }

  singleton_->level_ = level;
  singleton_->rate_ = rate;

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver

#endif  // TRTIS_ENABLE_TRACING
