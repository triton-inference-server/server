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
#include "vertex_ai_server.h"

#include <memory>

#include "common.h"

namespace triton { namespace server {

const std::string VertexAiAPIServer::binary_mime_type_(
    "application/vnd.vertex-ai-triton.binary+json;json-header-size=");
const std::string VertexAiAPIServer::redirect_header_(
    "X-Vertex-Ai-Triton-Redirect");

VertexAiAPIServer::VertexAiAPIServer(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager, const int32_t port,
    const std::string address, const int thread_cnt,
    const std::string& prediction_route, const std::string& health_route,
    const std::string& default_model_name)
    : HTTPAPIServer(
          server, trace_manager, shm_manager, port, false /* reuse_port */,
          address, "" /* header_forward_pattern */, thread_cnt),
      prediction_regex_(prediction_route), health_regex_(health_route),
      health_mode_("ready"), model_name_(default_model_name),
      model_version_str_("")
{
}

TRITONSERVER_Error*
VertexAiAPIServer::GetInferenceHeaderLength(
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
VertexAiAPIServer::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "Vertex AI request: " << req->method << " "
                 << req->uri->path->full;

  if (RE2::FullMatch(std::string(req->uri->path->full), health_regex_)) {
    HandleServerHealth(req, health_mode_);
    return;
  }

  if (RE2::FullMatch(std::string(req->uri->path->full), prediction_regex_)) {
    // Secondary regex matching if redirection is requested
    const char* redirect_c_str =
        evhtp_kv_find(req->headers_in, redirect_header_.c_str());
    if (redirect_c_str == nullptr) {
      // Infer the default model
      HandleInfer(req, model_name_, model_version_str_);
      return;
    } else {
      // Endpoint redirection is requested
      // Prepend the header value with "/" to form the regex expected by
      // Triton endpoints
      std::string redirect_endpoint("/");
      redirect_endpoint += redirect_c_str;
      LOG_VERBOSE(1) << "Redirecting Vertex AI request: " << redirect_endpoint;

      // The endpoint handlers in base class expects specific HTTP methods
      // while the Vertex AI endpoint only accepts "POST", so the method will
      // be set to endpoint expected one before invoking the handlers
      if (req->method != htp_method_POST) {
        evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
        return;
      }

      if (redirect_endpoint == "/metrics") {
        req->method = htp_method_GET;
        HandleMetrics(req);
        return;
      }

      if (redirect_endpoint == "/v2/models/stats") {
        // model statistics
        req->method = htp_method_GET;
        HandleModelStats(req);
        return;
      }

      std::string model_name, version, kind;
      if (RE2::FullMatch(
              redirect_endpoint, model_regex_, &model_name, &version, &kind)) {
        if (kind == "ready") {
          // model ready
          req->method = htp_method_GET;
          HandleModelReady(req, model_name, version);
          return;
        } else if (kind == "infer") {
          // model infer
          HandleInfer(req, model_name, version);
          return;
        } else if (kind == "config") {
          // model configuration
          req->method = htp_method_GET;
          HandleModelConfig(req, model_name, version);
          return;
        } else if (kind == "stats") {
          // model statistics
          req->method = htp_method_GET;
          HandleModelStats(req, model_name, version);
          return;
        } else if (kind == "") {
          // model metadata
          req->method = htp_method_GET;
          HandleModelMetadata(req, model_name, version);
          return;
        }
      }

      std::string region, action, rest, repo_name;
      if (redirect_endpoint == "/v2") {
        // server metadata
        req->method = htp_method_GET;
        HandleServerMetadata(req);
        return;
      } else if (RE2::FullMatch(redirect_endpoint, server_regex_, &rest)) {
        // server health
        req->method = htp_method_GET;
        HandleServerHealth(req, rest);
        return;
      } else if (RE2::FullMatch(
                     redirect_endpoint, systemsharedmemory_regex_, &region,
                     &action)) {
        // system shared memory
        if (action == "status") {
          req->method = htp_method_GET;
        }
        HandleSystemSharedMemory(req, region, action);
        return;
      } else if (RE2::FullMatch(
                     redirect_endpoint, cudasharedmemory_regex_, &region,
                     &action)) {
        // cuda shared memory
        if (action == "status") {
          req->method = htp_method_GET;
        }
        HandleCudaSharedMemory(req, region, action);
        return;
      } else if (RE2::FullMatch(
                     redirect_endpoint, modelcontrol_regex_, &repo_name, &kind,
                     &model_name, &action)) {
        // model repository
        if (kind == "index") {
          HandleRepositoryIndex(req, repo_name);
          return;
        } else if (kind.find("models", 0) == 0) {
          HandleRepositoryControl(req, repo_name, model_name, action);
          return;
        }
      }
    }
  }

  LOG_VERBOSE(1) << "Vertex AI error: " << req->method << " "
                 << req->uri->path->full << " - "
                 << static_cast<int>(EVHTP_RES_BADREQ);

  evhtp_send_reply(req, EVHTP_RES_BADREQ);
}

void
VertexAiAPIServer::HandleMetrics(evhtp_request_t* req)
{
  // Mirror of HTTPMetricsServer::Handle()
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  evhtp_res res = EVHTP_RES_BADREQ;

  // Call to metric endpoint should not have any trailing string
  TRITONSERVER_Metrics* metrics = nullptr;
  TRITONSERVER_Error* err = TRITONSERVER_ServerMetrics(server_.get(), &metrics);
  if (err == nullptr) {
    const char* base;
    size_t byte_size;
    err = TRITONSERVER_MetricsFormatted(
        metrics, TRITONSERVER_METRIC_PROMETHEUS, &base, &byte_size);
    if (err == nullptr) {
      res = EVHTP_RES_OK;
      evbuffer_add(req->buffer_out, base, byte_size);
    }
  }

  TRITONSERVER_MetricsDelete(metrics);
  TRITONSERVER_ErrorDelete(err);

  evhtp_send_reply(req, res);
}


TRITONSERVER_Error*
VertexAiAPIServer::Create(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager, const int32_t port,
    const std::string address, const int thread_cnt,
    std::string default_model_name, std::unique_ptr<HTTPServer>* http_server)
{
  auto predict_route = GetEnvironmentVariableOrDefault("AIP_PREDICT_ROUTE", "");
  auto health_route = GetEnvironmentVariableOrDefault("AIP_HEALTH_ROUTE", "");
  if (predict_route.empty())
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "API_PREDICT_ROUTE is not defined for Vertex AI endpoint");
  else if (health_route.empty()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "AIP_HEALTH_ROUTE is not defined for Vertex AI endpoint");
  }

  // Set default model
  {
    TRITONSERVER_Message* model_index_message = nullptr;
    RETURN_IF_ERR(TRITONSERVER_ServerModelIndex(
        server.get(), TRITONSERVER_INDEX_FLAG_READY, &model_index_message));

    // avoid memory leak when return early
    std::shared_ptr<TRITONSERVER_Message> managed_msg(
        model_index_message,
        [](TRITONSERVER_Message* msg) { TRITONSERVER_MessageDelete(msg); });

    const char* buffer;
    size_t byte_size;
    RETURN_IF_ERR(TRITONSERVER_MessageSerializeToJson(
        model_index_message, &buffer, &byte_size));

    triton::common::TritonJson::Value model_index_json;
    RETURN_IF_ERR(model_index_json.Parse(buffer, byte_size));

    if (default_model_name.empty()) {
      if (model_index_json.ArraySize() != 1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "Expect the model repository contains only a single model if "
            "default model is not specified");
      }

      triton::common::TritonJson::Value index_json;
      RETURN_IF_ERR(model_index_json.IndexAsObject(0, &index_json));
      const char* name;
      size_t namelen;
      RETURN_IF_ERR(index_json.MemberAsString("name", &name, &namelen));
      default_model_name = std::string(name, namelen);
    }
    // Check if default model is loaded
    else {
      bool found = false;
      for (size_t idx = 0; idx < model_index_json.ArraySize(); ++idx) {
        triton::common::TritonJson::Value index_json;
        RETURN_IF_ERR(model_index_json.IndexAsObject(idx, &index_json));

        const char* name;
        size_t namelen;
        RETURN_IF_ERR(index_json.MemberAsString("name", &name, &namelen));
        if (default_model_name == std::string(name, namelen)) {
          found = true;
          break;
        }
      }
      if (!found) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Expect the default model '") + default_model_name +
             "' is loaded")
                .c_str());
      }
    }
  }

  http_server->reset(new VertexAiAPIServer(
      server, trace_manager, shm_manager, port, address, thread_cnt,
      predict_route, health_route, default_model_name));

  const std::string addr = address + ":" + std::to_string(port);
  LOG_INFO << "Started Vertex AI HTTPService at " << addr;

  return nullptr;
}

}}  // namespace triton::server
