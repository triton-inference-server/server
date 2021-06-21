// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "src/servers/http_server.h"

namespace nvidia { namespace inferenceserver {

// Handle Sagemaker HTTP requests to inference server APIs
class SagemakerAPIServer : public HTTPAPIServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      nvidia::inferenceserver::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& smb_manager,
      const int32_t port, const int thread_cnt,
      std::unique_ptr<HTTPServer>* sagemaker_server);

  class SagemakeInferRequestClass : public InferRequestClass {
   public:
    explicit SagemakeInferRequestClass(
        TRITONSERVER_Server* server, evhtp_request_t* req,
        DataCompressor::Type response_compression_type)
        : InferRequestClass(server, req, response_compression_type)
    {
    }

    void SetResponseHeader(
        const bool has_binary_data, const size_t header_length) override;
  };

 private:
  static std::string GetEnvironmentVariableOrDefault(
      const std::string& variable_name, const std::string& default_value)
  {
    const char* value = getenv(variable_name.c_str());
    return value ? value : default_value;
  }

  explicit SagemakerAPIServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      nvidia::inferenceserver::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const int32_t port, const int thread_cnt)
      : HTTPAPIServer(server, trace_manager, shm_manager, port, thread_cnt),
        ping_regex_(R"(/ping)"), invocations_regex_(R"(/invocations)"),
        ping_mode_("ready"),
        model_name_(SagemakerAPIServer::GetEnvironmentVariableOrDefault(
            "SAGEMAKER_TRITON_DEFAULT_MODEL_NAME",
            "unspecified_SAGEMAKER_TRITON_DEFAULT_MODEL_NAME")),
        model_version_str_("")
  {
  }

  void Handle(evhtp_request_t* req) override;

  std::unique_ptr<InferRequestClass> CreateInferRequest(
      evhtp_request_t* req) override
  {
    return std::unique_ptr<InferRequestClass>(new SagemakeInferRequestClass(
        server_.get(), req, GetResponseCompressionType(req)));
  }
  TRITONSERVER_Error* GetInferenceHeaderLength(
      evhtp_request_t* req, int32_t content_length,
      size_t* header_length) override;

  // Currently the compresssion schema hasn't been defined,
  // assume identity compression type is used for both request and response
  DataCompressor::Type GetRequestCompressionType(evhtp_request_t* req) override
  {
    return DataCompressor::Type::IDENTITY;
  }
  DataCompressor::Type GetResponseCompressionType(evhtp_request_t* req) override
  {
    return DataCompressor::Type::IDENTITY;
  }
  re2::RE2 ping_regex_;
  re2::RE2 invocations_regex_;

  const std::string ping_mode_;

  // For single model mode, assume that only one version of "model" is presented
  const std::string model_name_;
  const std::string model_version_str_;

  static const std::string binary_mime_type_;
};

}}  // namespace nvidia::inferenceserver
