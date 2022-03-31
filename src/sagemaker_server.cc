// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "sagemaker_server.h"

namespace triton { namespace server {

const std::string SagemakerAPIServer::binary_mime_type_(
    "application/vnd.sagemaker-triton.binary+json;json-header-size=");

TRITONSERVER_Error*
SagemakerAPIServer::GetInferenceHeaderLength(
    evhtp_request_t* req, int32_t content_length, size_t* header_length)
{
  // Check mime type and set inference header length.
  // Set to content length in case that it is not specified
  *header_length = content_length;
  const char* content_type_c_str =
      evhtp_kv_find(req->headers_in, kContentTypeHeader);
  if (content_type_c_str != NULL) {
    std::string content_type(content_type_c_str);
    size_t pos = content_type.find(binary_mime_type_);
    if (pos != std::string::npos) {
      if (pos != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("expect MIME type for binary data starts with '") +
             binary_mime_type_ + "', got: " + content_type)
                .c_str());
      }

      // Parse
      int32_t parsed_value;
      try {
        parsed_value =
            std::atoi(content_type_c_str + binary_mime_type_.length());
      }
      catch (const std::invalid_argument& ia) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Unable to parse inference header size, got: ") +
             (content_type_c_str + binary_mime_type_.length()))
                .c_str());
      }

      // Check if the content length is in proper range
      if ((parsed_value < 0) || (parsed_value > content_length)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("inference header size should be in range (0, ") +
             std::to_string(content_length) +
             "), got: " + (content_type_c_str + binary_mime_type_.length()))
                .c_str());
      }
      *header_length = parsed_value;
    }
  }
  return nullptr;
}

void
SagemakerAPIServer::SagemakeInferRequestClass::SetResponseHeader(
    bool has_binary_data, size_t header_length)
{
  if (has_binary_data) {
    evhtp_headers_add_header(
        req_->headers_out,
        evhtp_header_new(
            kContentTypeHeader,
            (binary_mime_type_ + std::to_string(header_length)).c_str(), 1, 1));
  } else {
    evhtp_headers_add_header(
        req_->headers_out,
        evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));
  }
}

void
SagemakerAPIServer::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "SageMaker request: " << req->method << " "
                 << req->uri->path->full;

  if (RE2::FullMatch(std::string(req->uri->path->full), ping_regex_)) {
    HandleServerHealth(req, ping_mode_);
    return;
  }

  if (RE2::FullMatch(std::string(req->uri->path->full), invocations_regex_)) {
    HandleInfer(req, model_name_, model_version_str_);
    return;
  }

  std::string multi_model_name, action;
  if (RE2::FullMatch(
          std::string(req->uri->path->full), models_regex_, &multi_model_name, &action)) {

    switch (req->method) {
    case htp_method_GET:
      if (multi_model_name.empty()) {
        LOG_VERBOSE(1) << "SageMaker request: LIST ALL MODELS";
        req->method = htp_method_POST;
        HandleRepositoryIndex(req, "");
        return;
      }
      LOG_VERBOSE(1) << "SageMaker request: GET MODEL";
      return;
    case htp_method_POST:
      if (action == "/invoke") {
        LOG_VERBOSE(1) << "SageMaker request: INVOKE MODEL";
        HandleInfer(req, multi_model_name, model_version_str_);
        return;
      }
      if (action.empty()) {
        LOG_VERBOSE(1) << "SageMaker request: LOAD MODEL";
        req->method = htp_method_POST;
        HandleRepositoryControl(req, "", multi_model_name, "load");
        return;
      }
      break;
    case htp_method_DELETE:
      // UNLOAD MODEL
      LOG_VERBOSE(1) << "SageMaker request: UNLOAD MODEL";
      req->method = htp_method_POST;
      HandleRepositoryControl(req, "", multi_model_name, "unload");
      return;
    default:
      LOG_VERBOSE(1) << "SageMaker error: " << req->method << " "
                     << req->uri->path->full << " - "
                     << static_cast<int>(EVHTP_RES_BADREQ);
      evhtp_send_reply(req, EVHTP_RES_BADREQ);
      return;
    }
  }

  LOG_VERBOSE(1) << "SageMaker error: " << req->method << " "
                 << req->uri->path->full << " - "
                 << static_cast<int>(EVHTP_RES_BADREQ);

  evhtp_send_reply(req, EVHTP_RES_BADREQ);
}


TRITONSERVER_Error*
SagemakerAPIServer::Create(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager, const int32_t port,
    const std::string address, const int thread_cnt,
    std::unique_ptr<HTTPServer>* http_server)
{
  http_server->reset(new SagemakerAPIServer(
      server, trace_manager, shm_manager, port, address, thread_cnt));

  const std::string addr = address + ":" + std::to_string(port);
  LOG_INFO << "Started Sagemaker HTTPService at " << addr;

  return nullptr;
}

}}  // namespace triton::server
