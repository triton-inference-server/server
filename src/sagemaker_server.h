// Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <sys/stat.h>

#include <fstream>
#include <mutex>

#include "common.h"
#include "dirent.h"
#include "http_server.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace server {

// Handle Sagemaker HTTP requests to inference server APIs
class SagemakerAPIServer : public HTTPAPIServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& smb_manager,
      const int32_t port, const std::string address, const int thread_cnt,
      std::unique_ptr<HTTPServer>* sagemaker_server);

  class SagemakeInferRequestClass : public InferRequestClass {
   public:
    explicit SagemakeInferRequestClass(
        TRITONSERVER_Server* server, evhtp_request_t* req,
        DataCompressor::Type response_compression_type,
        const std::shared_ptr<TRITONSERVER_InferenceRequest>& triton_request)
        : InferRequestClass(
              server, req, response_compression_type, triton_request)
    {
    }
    using InferRequestClass::InferResponseComplete;
    static void InferResponseComplete(
        TRITONSERVER_InferenceResponse* response, const uint32_t flags,
        void* userp);

    void SetResponseHeader(
        const bool has_binary_data, const size_t header_length) override;
  };

 private:
  explicit SagemakerAPIServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const int32_t port, const std::string address, const int thread_cnt)
      : HTTPAPIServer(
            server, trace_manager, shm_manager, port, false /* reuse_port */,
            address, "" /* header_forward_pattern */, thread_cnt),
        ping_regex_(R"(/ping)"), invocations_regex_(R"(/invocations)"),
        models_regex_(R"(/models(?:/)?([^/]+)?(/invoke)?)"),
        model_path_regex_(
            R"((\/opt\/ml\/models\/[0-9A-Za-z._]+)\/(model)\/?([0-9A-Za-z._]+)?)"),
        platform_ensemble_regex_(R"(platform:(\s)*\"ensemble\")"),
        ping_mode_(GetEnvironmentVariableOrDefault(
            "SAGEMAKER_TRITON_PING_MODE", "ready")),
        model_name_(GetEnvironmentVariableOrDefault(
            "SAGEMAKER_TRITON_DEFAULT_MODEL_NAME",
            "unspecified_SAGEMAKER_TRITON_DEFAULT_MODEL_NAME")),
        model_version_str_("")
  {
  }

  void ParseSageMakerRequest(
      evhtp_request_t* req,
      std::unordered_map<std::string, std::string>* parse_map,
      const std::string& action);

  void SageMakerMMEHandleInfer(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);

  void SageMakerMMELoadModel(
      evhtp_request_t* req,
      const std::unordered_map<std::string, std::string> parse_map);

  void SageMakerMMEHandleOOMError(
      evhtp_request_t* req, TRITONSERVER_Error* load_err);

  static bool SageMakerMMECheckOOMError(TRITONSERVER_Error* load_err);

  void SageMakerMMEUnloadModel(evhtp_request_t* req, const char* model_name);

  TRITONSERVER_Error* SageMakerMMECheckUnloadedModelIsUnavailable(
      const char* model_name, bool* is_model_unavailable);

  void SageMakerMMEListModel(evhtp_request_t* req);

  void SageMakerMMEGetModel(evhtp_request_t* req, const char* model_name);

  void Handle(evhtp_request_t* req) override;

  std::unique_ptr<InferRequestClass> CreateInferRequest(
      evhtp_request_t* req,
      const std::shared_ptr<TRITONSERVER_InferenceRequest>& triton_request)
      override
  {
    return std::unique_ptr<InferRequestClass>(new SagemakeInferRequestClass(
        server_.get(), req, GetResponseCompressionType(req), triton_request));
  }
  TRITONSERVER_Error* GetInferenceHeaderLength(
      evhtp_request_t* req, int32_t content_length,
      size_t* header_length) override;


  // Currently the compression schema hasn't been defined,
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
  re2::RE2 models_regex_;
  re2::RE2 model_path_regex_;
  re2::RE2 platform_ensemble_regex_;

  const std::string ping_mode_;

  /* For single model mode, assume that only one version of "model" is presented
   */
  const std::string model_name_;
  const std::string model_version_str_;

  static const std::string binary_mime_type_;

  /* Maintain list of loaded models */
  std::unordered_map<std::string, std::string> sagemaker_models_list_;

  /* Mutex to handle concurrent updates */
  std::mutex models_list_mutex_;

  /* Constants */
  const uint32_t UNLOAD_TIMEOUT_SECS_ = 350;
  const uint32_t UNLOAD_SLEEP_MILLISECONDS_ = 500;
  const std::string UNLOAD_EXPECTED_STATE_ = "UNAVAILABLE";
  const std::string UNLOAD_EXPECTED_REASON_ = "unloaded";
};

}}  // namespace triton::server
