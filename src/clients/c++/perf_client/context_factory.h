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
#pragma once

#include <string>

#include "src/clients/c++/perf_client/perf_utils.h"

namespace perfclient {

//==============================================================================
/// ContextFactory is a helper class to create client contexts used
/// in perf_client.
///
class ContextFactory {
 public:
  enum ModelSchedulerType { NONE, DYNAMIC, SEQUENCE, ENSEMBLE };
  /// Create a context factory that is responsible to create different types of
  /// contexts that is directly related to the specified model.
  /// \param url The inference server name and port.
  /// \param protocol The protocol type used.
  /// \param http_headers Map of HTTP headers. The map key/value
  /// indicates the header name/value.
  /// \param streaming Whether to use streaming API.
  /// \param model_name The name of the model.
  /// \param model_version The version of the model to use for inference,
  /// or -1 to indicate that the latest (i.e. highest version number)
  /// version should be used.
  /// \param factory Returns a new ContextFactory object.
  /// \return Error object indicating success or failure.
  static nic::Error Create(
      const std::string& url, const ProtocolType protocol,
      const std::map<std::string, std::string>& http_headers,
      const bool streaming, const std::string& model_name,
      const int64_t model_version, std::shared_ptr<ContextFactory>* factory);

  /// Create a ServerStatusContext.
  /// \param ctx Returns a new ServerStatusContext object.
  nic::Error CreateServerStatusContext(
      std::unique_ptr<nic::ServerStatusContext>* ctx);

  /// Create a InferContext.
  /// \param ctx Returns a new InferContext object.
  nic::Error CreateInferContext(std::unique_ptr<nic::InferContext>* ctx);

  /// \return The model name.
  const std::string& ModelName() const { return model_name_; }

  /// \return The model version.
  int64_t ModelVersion() const { return model_version_; }

  /// \return The scheduler type of the model.
  ModelSchedulerType SchedulerType() const { return scheduler_type_; }

 private:
  ContextFactory(
      const std::string& url, const ProtocolType protocol,
      const std::map<std::string, std::string>& http_headers,
      const bool streaming, const std::string& model_name,
      const int64_t model_version)
      : url_(url), protocol_(protocol), http_headers_(http_headers),
        streaming_(streaming), model_name_(model_name),
        model_version_(model_version), current_correlation_id_(0)
  {
  }

  const std::string url_;
  const ProtocolType protocol_;
  const std::map<std::string, std::string> http_headers_;
  const bool streaming_;
  const std::string model_name_;
  const int64_t model_version_;

  ModelSchedulerType scheduler_type_;
  ni::CorrelationID current_correlation_id_;
  std::mutex correlation_id_mutex_;
};

}  // namespace perfclient
