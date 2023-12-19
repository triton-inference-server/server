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
#include "sagemaker_server.h"

namespace triton { namespace server {

#define HTTP_RESPOND_IF_ERR(REQ, X)                   \
  do {                                                \
    TRITONSERVER_Error* err__ = (X);                  \
    if (err__ != nullptr) {                           \
      EVBufferAddErrorJson((REQ)->buffer_out, err__); \
      evhtp_send_reply((REQ), EVHTP_RES_BADREQ);      \
      TRITONSERVER_ErrorDelete(err__);                \
      return;                                         \
    }                                                 \
  } while (false)

namespace {

void
EVBufferAddErrorJson(evbuffer* buffer, TRITONSERVER_Error* err)
{
  const char* message = TRITONSERVER_ErrorMessage(err);

  triton::common::TritonJson::Value response(
      triton::common::TritonJson::ValueType::OBJECT);
  response.AddStringRef("error", message, strlen(message));

  triton::common::TritonJson::WriteBuffer buffer_json;
  response.Write(&buffer_json);

  evbuffer_add(buffer, buffer_json.Base(), buffer_json.Size());
}

TRITONSERVER_Error*
EVBufferToJson(
    triton::common::TritonJson::Value* document, evbuffer_iovec* v, int* v_idx,
    const size_t length, int n)
{
  size_t offset = 0, remaining_length = length;
  char* json_base;
  std::vector<char> json_buffer;

  // No need to memcpy when number of iovecs is 1
  if ((n > 0) && (v[0].iov_len >= remaining_length)) {
    json_base = static_cast<char*>(v[0].iov_base);
    if (v[0].iov_len > remaining_length) {
      v[0].iov_base = static_cast<void*>(json_base + remaining_length);
      v[0].iov_len -= remaining_length;
      remaining_length = 0;
    } else if (v[0].iov_len == remaining_length) {
      remaining_length = 0;
      *v_idx += 1;
    }
  } else {
    json_buffer.resize(length);
    json_base = json_buffer.data();
    while ((remaining_length > 0) && (*v_idx < n)) {
      char* base = static_cast<char*>(v[*v_idx].iov_base);
      size_t base_size;
      if (v[*v_idx].iov_len > remaining_length) {
        base_size = remaining_length;
        v[*v_idx].iov_base = static_cast<void*>(base + remaining_length);
        v[*v_idx].iov_len -= remaining_length;
        remaining_length = 0;
      } else {
        base_size = v[*v_idx].iov_len;
        remaining_length -= v[*v_idx].iov_len;
        *v_idx += 1;
      }

      memcpy(json_base + offset, base, base_size);
      offset += base_size;
    }
  }

  if (remaining_length != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected size for request JSON, expecting " +
            std::to_string(remaining_length) + " more bytes")
            .c_str());
  }

  RETURN_IF_ERR(document->Parse(json_base, length));

  return nullptr;  // success
}

}  // namespace


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
          std::string(req->uri->path->full), models_regex_, &multi_model_name,
          &action)) {
    switch (req->method) {
      case htp_method_GET:
        if (multi_model_name.empty()) {
          LOG_VERBOSE(1) << "SageMaker request: LIST ALL MODELS";

          SageMakerMMEListModel(req);
          return;
        } else {
          LOG_VERBOSE(1) << "SageMaker request: GET MODEL";

          SageMakerMMEGetModel(req, multi_model_name.c_str());
          return;
        }
      case htp_method_POST:
        if (action == "/invoke") {
          LOG_VERBOSE(1) << "SageMaker request: INVOKE MODEL";

          if (sagemaker_models_list_.find(multi_model_name.c_str()) ==
              sagemaker_models_list_.end()) {
            evhtp_send_reply(req, EVHTP_RES_NOTFOUND); /* 404*/
            return;
          }
          LOG_VERBOSE(1) << "SageMaker MME Custom Invoke Model Path";

          /* Extract targetModel to log the associated archive */
          const char* target_model =
              evhtp_kv_find(req->headers_in, "X-Amzn-SageMaker-Target-Model");

          /* If target_model is not available (e.g., in local testing) use
           * model_name_hash as target_model) */
          if (target_model == nullptr) {
            target_model = multi_model_name.c_str();
          }

          LOG_INFO << "Invoking SageMaker TargetModel: " << target_model;

          SageMakerMMEHandleInfer(req, target_model, model_version_str_);
          return;
        }
        if (action.empty()) {
          LOG_VERBOSE(1) << "SageMaker request: LOAD MODEL";

          std::unordered_map<std::string, std::string> parse_load_map;
          ParseSageMakerRequest(req, &parse_load_map, "load");
          SageMakerMMELoadModel(req, parse_load_map);
          return;
        }
        break;
      case htp_method_DELETE: {
        // UNLOAD MODEL
        LOG_VERBOSE(1) << "SageMaker request: UNLOAD MODEL";
        req->method = htp_method_POST;

        SageMakerMMEUnloadModel(req, multi_model_name.c_str());

        return;
      }
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


void
SagemakerAPIServer::ParseSageMakerRequest(
    evhtp_request_t* req,
    std::unordered_map<std::string, std::string>* parse_map,
    const std::string& action)
{
  struct evbuffer_iovec* v = nullptr;
  int v_idx = 0;
  int n = evbuffer_peek(req->buffer_in, -1, NULL, NULL, 0);
  if (n > 0) {
    v = static_cast<struct evbuffer_iovec*>(
        alloca(sizeof(struct evbuffer_iovec) * n));
    if (evbuffer_peek(req->buffer_in, -1, NULL, v, n) != n) {
      HTTP_RESPOND_IF_ERR(
          req, TRITONSERVER_ErrorNew(
                   TRITONSERVER_ERROR_INTERNAL,
                   "unexpected error getting load model request buffers"));
    }
  }

  std::string model_name_string;
  std::string url_string;

  size_t buffer_len = evbuffer_get_length(req->buffer_in);
  if (buffer_len > 0) {
    triton::common::TritonJson::Value request;
    HTTP_RESPOND_IF_ERR(
        req, EVBufferToJson(&request, v, &v_idx, buffer_len, n));

    triton::common::TritonJson::Value url;
    triton::common::TritonJson::Value model_name;

    if (request.Find("model_name", &model_name)) {
      HTTP_RESPOND_IF_ERR(req, model_name.AsString(&model_name_string));
      LOG_VERBOSE(1) << "Received model_name: " << model_name_string.c_str();
    }

    if ((action == "load") && (request.Find("url", &url))) {
      HTTP_RESPOND_IF_ERR(req, url.AsString(&url_string));
      LOG_VERBOSE(1) << "Received url: " << url_string.c_str();
    }
  }

  if (action == "load") {
    (*parse_map)["url"] = url_string.c_str();
  }
  (*parse_map)["model_name_hash"] = model_name_string.c_str();

  /* Extract target_model, specified in header, to log the associated archive */
  const char* target_model =
      evhtp_kv_find(req->headers_in, "X-Amzn-SageMaker-Target-Model");


  /* If target_model is not available (e.g., in local testing) use
   * model_name_hash as target_model) */
  if (target_model != nullptr) {
    (*parse_map)["target_model"] = target_model;
  } else {
    (*parse_map)["target_model"] = model_name_string.c_str();
  }

  LOG_INFO << "Loading SageMaker TargetModel: " << target_model;

  return;
}

void
SagemakerAPIServer::SagemakeInferRequestClass::InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  // FIXME can't use InferRequestClass object here since it's lifetime
  // is different than response. For response we need to know how to
  // send each output (as json, shm, or binary) and that information
  // has to be maintained in a way that allows us to clean it up
  // appropriately if connection closed or last response sent.
  //
  // But for now userp is the InferRequestClass object and the end of
  // its life is in the ReplyCallback.

  SagemakerAPIServer::SagemakeInferRequestClass* infer_request =
      reinterpret_cast<SagemakerAPIServer::SagemakeInferRequestClass*>(userp);

  auto response_count = infer_request->IncrementResponseCount();

  // Defer to the callback with the final response
  if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
    LOG_ERROR << "[INTERNAL] received a response without FINAL flag";
    return;
  }

  TRITONSERVER_Error* err = nullptr;
  if (response_count != 0) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, std::string(
                                         "expected a single response, got " +
                                         std::to_string(response_count + 1))
                                         .c_str());
  } else if (response == nullptr) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "received an unexpected null response");
  } else {
    err = infer_request->FinalizeResponse(response);
  }

#ifdef TRITON_ENABLE_TRACING
  if (infer_request->trace_ != nullptr) {
    infer_request->trace_->CaptureTimestamp(
        "INFER_RESPONSE_COMPLETE", TraceManager::CaptureTimestamp());
  }
#endif  // TRITON_ENABLE_TRACING

  if (err != nullptr) {
    EVBufferAddErrorJson(infer_request->req_->buffer_out, err);
    // [FIXME] In http_server.cc, error handling is enhanced to reporting
    // different error code according to the Triton error code, holding
    // the change from SageMaker endpoint as it may not fit their SLA.
    infer_request->response_code_ = EVHTP_RES_BADREQ;
    if (SageMakerMMECheckOOMError(err) == true) {
      LOG_VERBOSE(1)
          << "Received an OOM error during INVOKE MODEL. Returning a 507."
          << std::endl;
      infer_request->response_code_ = 507;
    }
    TRITONSERVER_ErrorDelete(err);
  }
  evthr_defer(infer_request->thread_, ReplyCallback, infer_request);

  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceResponseDelete(response),
      "deleting inference response");
}

void
SagemakerAPIServer::SageMakerMMEHandleInfer(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  bool connection_paused = false;

  int64_t requested_model_version;
  auto err = GetModelVersionFromString(
      model_version_str.c_str(), &requested_model_version);

  if (err == nullptr) {
    uint32_t txn_flags;
    err = TRITONSERVER_ServerModelTransactionProperties(
        server_.get(), model_name.c_str(), requested_model_version, &txn_flags,
        nullptr /* voidp */);
    if ((err == nullptr) && (txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "HTTP end point doesn't support models with decoupled "
          "transaction policy");
    }
  }

  // If tracing is enabled see if this request should be traced.
  TRITONSERVER_InferenceTrace* triton_trace = nullptr;
#ifdef TRITON_ENABLE_TRACING
  std::shared_ptr<TraceManager::Trace> trace;
  if (err == nullptr) {
    trace = std::move(trace_manager_->SampleTrace(model_name));
    if (trace != nullptr) {
      triton_trace = trace->trace_;

      // Timestamps from evhtp are capture in 'req'. We record here
      // since this is the first place where we have access to trace
      // manager.
      trace->CaptureTimestamp("HTTP_RECV_START", req->recv_start_ns);
      trace->CaptureTimestamp("HTTP_RECV_END", req->recv_end_ns);
    }
  }
#endif  // TRITON_ENABLE_TRACING

  // Create the inference request object which provides all information needed
  // for an inference.
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  std::shared_ptr<TRITONSERVER_InferenceRequest> irequest_shared = nullptr;
  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestNew(
        &irequest, server_.get(), model_name.c_str(), requested_model_version);
  }
  if (err == nullptr) {
    irequest_shared = std::shared_ptr<TRITONSERVER_InferenceRequest>(
        irequest, [](TRITONSERVER_InferenceRequest* request) {
          LOG_TRITONSERVER_ERROR(
              TRITONSERVER_InferenceRequestDelete(request),
              "deleting HTTP/REST inference request");
        });
  }
  // Decompress request body if it is compressed in supported type
  evbuffer* decompressed_buffer = nullptr;
  if (err == nullptr) {
    auto compression_type = GetRequestCompressionType(req);
    switch (compression_type) {
      case DataCompressor::Type::DEFLATE:
      case DataCompressor::Type::GZIP: {
        decompressed_buffer = evbuffer_new();
        err = DataCompressor::DecompressData(
            compression_type, req->buffer_in, decompressed_buffer);
        break;
      }
      case DataCompressor::Type::UNKNOWN: {
        // Encounter unsupported compressed type,
        // send 415 error with supported types in Accept-Encoding
        evhtp_headers_add_header(
            req->headers_out,
            evhtp_header_new(kAcceptEncodingHTTPHeader, "gzip, deflate", 1, 1));
        evhtp_send_reply(req, EVHTP_RES_UNSUPPORTED);
        return;
      }
      case DataCompressor::Type::IDENTITY:
        // Do nothing
        break;
    }
  }

  // Get the header length
  size_t header_length;
  if (err == nullptr) {
    // Set to body size in case there is no Content-Length to compare with
    int32_t content_length = evbuffer_get_length(req->buffer_in);
    if (decompressed_buffer == nullptr) {
      const char* content_length_c_str =
          evhtp_kv_find(req->headers_in, kContentLengthHeader);
      if (content_length_c_str != nullptr) {
        try {
          content_length = std::atoi(content_length_c_str);
        }
        catch (const std::invalid_argument& ia) {
          err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("Unable to parse ") + kContentLengthHeader +
               ", got: " + content_length_c_str)
                  .c_str());
        }
      }
    } else {
      // The Content-Length doesn't reflect the actual request body size
      // if compression is used, set 'content_length' to the decompressed size
      content_length = evbuffer_get_length(decompressed_buffer);
    }

    if (err == nullptr) {
      err = GetInferenceHeaderLength(req, content_length, &header_length);
    }
  }

  if (err == nullptr) {
    connection_paused = true;

    auto infer_request = CreateInferRequest(req, irequest_shared);
    auto request_release_payload = std::make_unique<RequestReleasePayload>(
        irequest_shared, decompressed_buffer);

#ifdef TRITON_ENABLE_TRACING
    infer_request->trace_ = trace;
#endif  // TRITON_ENABLE_TRACING

    if (err == nullptr) {
      if (header_length != 0) {
        err = EVBufferToInput(
            model_name, irequest,
            (decompressed_buffer == nullptr) ? req->buffer_in
                                             : decompressed_buffer,
            infer_request.get(), header_length);
      } else {
        err = EVBufferToRawInput(
            model_name, irequest,
            (decompressed_buffer == nullptr) ? req->buffer_in
                                             : decompressed_buffer,
            infer_request.get());
      }
    }
    if (err == nullptr) {
      err = TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest, InferRequestClass::InferRequestComplete,
          request_release_payload.get());
      if (err == nullptr) {
        err = TRITONSERVER_InferenceRequestSetResponseCallback(
            irequest, allocator_,
            reinterpret_cast<void*>(&infer_request->alloc_payload_),
            SagemakerAPIServer::SagemakeInferRequestClass::
                InferResponseComplete,
            reinterpret_cast<void*>(infer_request.get()));

        LOG_VERBOSE(1) << std::endl;
      }
      if (err == nullptr) {
        err = TRITONSERVER_ServerInferAsync(
            server_.get(), irequest, triton_trace);
#ifdef TRITON_ENABLE_TRACING
        if (trace != nullptr) {
          trace->trace_ = nullptr;
        }
#endif  // TRITON_ENABLE_TRACING
      }
      if (err == nullptr) {
        infer_request.release();
        request_release_payload.release();
      }
    }
  }

  if (err != nullptr) {
    LOG_VERBOSE(1) << "Infer failed: " << TRITONSERVER_ErrorMessage(err);
    evhtp_headers_add_header(
        req->headers_out,
        evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));

    SageMakerMMEHandleOOMError(req, err);

    if (connection_paused) {
      evhtp_request_resume(req);
    }
    TRITONSERVER_ErrorDelete(err);
#ifdef TRITON_ENABLE_TRACING
    // If HTTP server still owns Triton trace
    if ((trace != nullptr) && (trace->trace_ != nullptr)) {
      TraceManager::TraceRelease(trace->trace_, trace->trace_userp_);
    }
#endif  // TRITON_ENABLE_TRACING
  }
}

TRITONSERVER_Error*
SagemakerAPIServer::SageMakerMMECheckUnloadedModelIsUnavailable(
    const char* model_name, bool* is_model_unavailable)
{
  /* Use the RepositoryIndex API to check if the model state has become
  UNAVAILABLE i.e. model is no longer in the 'in-the-process-of' being
  UNLOADED. Consequently, the reason field should be 'unloaded'.*/
  TRITONSERVER_Message* server_model_index_message = nullptr;
  uint32_t ready_flag = 0;  // value of 1 should be set if only the 'ready'
                            // models are required from the index. In this case,
                            // we need all models.
  TRITONSERVER_ServerModelIndex(
      server_.get(), ready_flag, &server_model_index_message);

  std::shared_ptr<TRITONSERVER_Message> shared_ptr_msg(
      server_model_index_message,
      [](TRITONSERVER_Message* msg) { TRITONSERVER_MessageDelete(msg); });

  const char* index_buffer;
  size_t index_byte_size;

  RETURN_IF_ERR(TRITONSERVER_MessageSerializeToJson(
      server_model_index_message, &index_buffer, &index_byte_size));

  /* Read into json buffer*/
  triton::common::TritonJson::Value server_model_index_json;
  server_model_index_json.Parse(index_buffer, index_byte_size);

  const char* name;
  const char* state;
  const char* reason;
  const char* version;

  size_t name_len;
  size_t state_len;
  size_t reason_len;
  size_t version_len;

  for (size_t id = 0; id < server_model_index_json.ArraySize(); ++id) {
    triton::common::TritonJson::Value index_json;
    server_model_index_json.IndexAsObject(id, &index_json);

    RETURN_IF_ERR(index_json.MemberAsString("name", &name, &name_len));

    if (std::string(name) == std::string(model_name)) {
      RETURN_IF_ERR(index_json.MemberAsString("state", &state, &state_len));

      if (std::string(state) == UNLOAD_EXPECTED_STATE_) {
        RETURN_IF_ERR(
            index_json.MemberAsString("reason", &reason, &reason_len));

        if (std::string(reason) == UNLOAD_EXPECTED_REASON_) {
          *is_model_unavailable = true;

          RETURN_IF_ERR(
              index_json.MemberAsString("version", &version, &version_len));

          LOG_VERBOSE(1) << "Discovered model: " << name
                         << ", version: " << version << " in state: " << state
                         << " for the reason: " << reason;

          break;
        }
      }
    }
  }

  return nullptr;
}

void
SagemakerAPIServer::SageMakerMMEUnloadModel(
    evhtp_request_t* req, const char* model_name_hash)
{
  /* Extract targetModel to log the associated archive */
  const char* target_model =
      evhtp_kv_find(req->headers_in, "X-Amzn-SageMaker-Target-Model");

  /* If target_model is not available (e.g., in local testing) use
   * model_name_hash as target_model) */
  if (target_model == nullptr) {
    target_model = model_name_hash;
  }

  if (sagemaker_models_list_.find(model_name_hash) ==
      sagemaker_models_list_.end()) {
    LOG_VERBOSE(1) << "Model " << target_model << " with model hash "
                   << model_name_hash << " is not loaded." << std::endl;
    evhtp_send_reply(req, EVHTP_RES_NOTFOUND); /* 404*/
    return;
  }

  LOG_INFO << "Unloading SageMaker TargetModel: " << target_model << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  /* Always unload dependents as well - this is required to unload dependents in
   * ensemble */
  TRITONSERVER_Error* unload_err = nullptr;
  unload_err =
      TRITONSERVER_ServerUnloadModelAndDependents(server_.get(), target_model);

  if (unload_err != nullptr) {
    EVBufferAddErrorJson(req->buffer_out, unload_err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);

    LOG_ERROR
        << "Error when unloading SageMaker Model with dependents for model: "
        << target_model << std::endl;

    TRITONSERVER_ErrorDelete(unload_err);
    return;
  }

  /*Note: Model status check is repo-specific and therefore must be run before
   * unregistering the repo, else the model information is lost*/
  bool is_model_unavailable = false;
  int64_t unload_time_in_secs = 0;

  /* Wait for the model to be completely unloaded. SageMaker waits a maximum
  of 360 seconds for the UNLOAD request to timeout. Setting a limit of 350
  seconds for Triton unload. This should be run only if above UNLOAD call has
  succeeded.*/
  if (unload_err == nullptr) {
    LOG_VERBOSE(1) << "Using Model Repository Index during UNLOAD to check for "
                      "status of model hash: "
                   << model_name_hash << " for model: " << target_model;
    while (is_model_unavailable == false &&
           unload_time_in_secs < UNLOAD_TIMEOUT_SECS_) {
      LOG_VERBOSE(1) << "In the loop to wait for model to be unavailable";
      unload_err = SageMakerMMECheckUnloadedModelIsUnavailable(
          target_model, &is_model_unavailable);
      if (unload_err != nullptr) {
        LOG_ERROR << "Error: Received non-zero exit code on checking for "
                     "model unavailability. "
                  << TRITONSERVER_ErrorMessage(unload_err);
        break;
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(UNLOAD_SLEEP_MILLISECONDS_));

      auto end_time = std::chrono::high_resolution_clock::now();

      unload_time_in_secs = std::chrono::duration_cast<std::chrono::seconds>(
                                end_time - start_time)
                                .count();
    }
    LOG_INFO << "UNLOAD for model " << target_model << " completed in "
             << unload_time_in_secs << " seconds.";
    TRITONSERVER_ErrorDelete(unload_err);
  }

  if ((is_model_unavailable == false) &&
      (unload_time_in_secs >= UNLOAD_TIMEOUT_SECS_)) {
    LOG_ERROR << "Error: UNLOAD did not complete within expected "
              << UNLOAD_TIMEOUT_SECS_
              << " seconds. This may "
                 "result in SageMaker UNLOAD timeout.";
  }

  std::string repo_parent_path = sagemaker_models_list_.at(model_name_hash);

  TRITONSERVER_Error* unregister_err = nullptr;

  unregister_err = TRITONSERVER_ServerUnregisterModelRepository(
      server_.get(), repo_parent_path.c_str());

  if (unregister_err != nullptr) {
    EVBufferAddErrorJson(req->buffer_out, unload_err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    LOG_ERROR << "Unable to unregister model repository for path: "
              << repo_parent_path << std::endl;
  } else {
    evhtp_send_reply(req, EVHTP_RES_OK);
  }

  TRITONSERVER_ErrorDelete(unregister_err);

  std::lock_guard<std::mutex> lock(models_list_mutex_);
  sagemaker_models_list_.erase(model_name_hash);
}

void
SagemakerAPIServer::SageMakerMMEGetModel(
    evhtp_request_t* req, const char* model_name)
{
  std::lock_guard<std::mutex> lock(models_list_mutex_);

  if (sagemaker_models_list_.find(model_name) == sagemaker_models_list_.end()) {
    evhtp_send_reply(req, EVHTP_RES_NOTFOUND); /* 404*/
    return;
  }

  triton::common::TritonJson::Value sagemaker_get_json(
      triton::common::TritonJson::ValueType::OBJECT);

  sagemaker_get_json.AddString("modelName", model_name);
  sagemaker_get_json.AddString(
      "modelUrl", sagemaker_models_list_.at(model_name));

  const char* buffer;
  size_t byte_size;

  triton::common::TritonJson::WriteBuffer json_buffer_;
  json_buffer_.Clear();
  sagemaker_get_json.Write(&json_buffer_);

  byte_size = json_buffer_.Size();
  buffer = json_buffer_.Base();

  evbuffer_add(req->buffer_out, buffer, byte_size);
  evhtp_send_reply(req, EVHTP_RES_OK);
}

void
SagemakerAPIServer::SageMakerMMEListModel(evhtp_request_t* req)
{
  std::lock_guard<std::mutex> lock(models_list_mutex_);

  triton::common::TritonJson::Value sagemaker_list_json(
      triton::common::TritonJson::ValueType::OBJECT);

  triton::common::TritonJson::Value models_array(
      sagemaker_list_json, triton::common::TritonJson::ValueType::ARRAY);

  for (auto it = sagemaker_models_list_.begin();
       it != sagemaker_models_list_.end(); it++) {
    triton::common::TritonJson::Value model_url_pair(
        models_array, triton::common::TritonJson::ValueType::OBJECT);

    bool ready = false;
    TRITONSERVER_ServerModelIsReady(
        server_.get(), it->first.c_str(), 1, &ready);

    /* Add to return list only if model is ready to be served */
    if (ready) {
      model_url_pair.AddString("modelName", it->first);
      model_url_pair.AddString("modelUrl", it->second);
    }

    models_array.Append(std::move(model_url_pair));
  }

  sagemaker_list_json.Add("models", std::move(models_array));

  const char* buffer;
  size_t byte_size;

  triton::common::TritonJson::WriteBuffer json_buffer_;
  json_buffer_.Clear();
  sagemaker_list_json.Write(&json_buffer_);

  byte_size = json_buffer_.Size();
  buffer = json_buffer_.Base();

  evbuffer_add(req->buffer_out, buffer, byte_size);
  evhtp_send_reply(req, EVHTP_RES_OK);
}

bool
SagemakerAPIServer::SageMakerMMECheckOOMError(TRITONSERVER_Error* err)
{
  const char* message = TRITONSERVER_ErrorMessage(err);
  std::string error_string(message);

  LOG_VERBOSE(1) << "Logging Verbose Error: " << std::endl
                 << error_string.c_str() << std::endl;

  const std::vector<std::string> error_messages{
      "CUDA out of memory", /* pytorch */
      "CUDA_OUT_OF_MEMORY", /* tensorflow */
      "Out of memory",      /* generic */
      "Out Of Memory",
      "out of memory",
      "MemoryError",
      "OutOfMemory",
      "OOM",
      "Dst tensor is not initialized",
      "Src tensor is not initialized",
      "CNMEM_STATUS_OUT_OF_MEMORY",
      "CUDNN_STATUS_NOT_INITIALIZED",
      "CUBLAS_STATUS_ALLOC_FAILED",
      "CUBLAS_STATUS_NOT_INITIALIZED",
      "Failed to allocate memory",
      "failed to allocate memory",
      "No space left on device"};

  /*
    TODO: Improve the search to do pattern match on whole words only
  */
  for (long unsigned int i = 0; i < error_messages.size(); i++) {
    if (error_string.find(error_messages[i]) != std::string::npos) {
      LOG_VERBOSE(1) << "OOM string '" << error_messages[i].c_str()
                     << "' detected in logs.";
      return true;
    }
  }

  return false;
}

void
SagemakerAPIServer::SageMakerMMEHandleOOMError(
    evhtp_request_t* req, TRITONSERVER_Error* err)
{
  EVBufferAddErrorJson(req->buffer_out, err);

  if (SageMakerMMECheckOOMError(err) == true) {
    /* Return a 507*/
    evhtp_send_reply(req, 507);
    LOG_VERBOSE(1)
        << "Received an OOM error during LOAD MODEL. Returning a 507.";
    return;
  }
  /* Return a 400*/
  evhtp_send_reply(req, EVHTP_RES_BADREQ);
  return;
}


void
SagemakerAPIServer::SageMakerMMELoadModel(
    evhtp_request_t* req,
    const std::unordered_map<std::string, std::string> parse_map)
{
  std::string repo_path = parse_map.at("url");
  std::string model_name_hash = parse_map.at("model_name_hash");
  std::string target_model = parse_map.at("target_model");

  /* Check subdirs for models and find ensemble model within the repo_path
   * If only 1 model, that will be selected as model_subdir
   * Else ensemble model directory is set as model_subdir
   */
  DIR* dir;
  struct dirent* ent;
  int dir_count = 0;
  std::string model_subdir, ensemble_model_subdir;

  if ((dir = opendir(repo_path.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      if ((ent->d_type == DT_DIR) && (!strcmp(ent->d_name, ".") == 0) &&
          (!strcmp(ent->d_name, "..") == 0)) {
        dir_count += 1;
        model_subdir = std::string(ent->d_name);
      }

      if (dir_count >= 2) {
        LOG_VERBOSE(1) << "More than one model detected in archive. "
                          "Checking if it is an ensemble."
                       << std::endl;
      }

      LOG_VERBOSE(1) << "Reading model sub-directory: " << model_subdir.c_str()
                     << std::endl;

      // Read the config.pbtxt file at each path, if available
      std::string ensemble_config_path =
          repo_path + "/" + model_subdir + "/" + "config.pbtxt";
      std::ifstream config_fstream(ensemble_config_path);
      std::stringstream ensemble_config_content;

      if (config_fstream.is_open()) {
        ensemble_config_content << config_fstream.rdbuf();
      } else {
        continue;  // A valid config.pbtxt does not exist at this path, or
                   // cannot be read
      }

      /* Compare matched string with `platform: "ensemble"` or
       * `platform:"ensemble"`. If present, we break, and use the model_subdir
       * to load the ensemble model
       */
      std::string detected_ensemble_regex;
      if (RE2::PartialMatch(
              ensemble_config_content.str(), platform_ensemble_regex_,
              &detected_ensemble_regex)) {
        LOG_INFO << "SageMaker front-end detected an Ensemble config at path: "
                 << ensemble_config_path << std::endl;
        ensemble_model_subdir = model_subdir;
      }

      if (dir_count > 5) {
        LOG_WARNING
            << "Several model directories found. If using ensemble, smaller "
               "ensembles are recommended for better memory management."
            << std::endl;
      }
    }
    closedir(dir);
  }

  if (!strcmp(ensemble_model_subdir.c_str(), "") == 0) {
    model_subdir = ensemble_model_subdir;
  }

  std::vector<const TRITONSERVER_Parameter*> subdir_modelname_map;

  /* Split repo path into three parts:
   * /opt/ml/models/<hash>/model/optional_customer_subdir
   * 1st repo_parent_path: /opt/ml/models/<hash>
   * 2nd subdir: model
   * 3rd customer_subdir: optional_customer_subdir
   */

  std::string repo_parent_path, subdir, customer_subdir;
  RE2::FullMatch(
      repo_path, model_path_regex_, &repo_parent_path, &subdir,
      &customer_subdir);

  std::string config_path = repo_path + "/config.pbtxt";
  struct stat buffer;

  /* If config.pbtxt is at repo root,
   * then repo_parent_path = /opt/ml/models/<hash>/, and model_subdir = model
   * else repo_parent_path = /opt/ml/models/<hash>/model and
   * model_subdir = dir under model/
   */
  if (stat(config_path.c_str(), &buffer) == 0) {
    model_subdir = subdir;
  } else {
    repo_parent_path = repo_path;
  }

  auto param = TRITONSERVER_ParameterNew(
      model_subdir.c_str(), TRITONSERVER_PARAMETER_STRING,
      target_model.c_str());

  if (param != nullptr) {
    subdir_modelname_map.emplace_back(param);
  } else {
    HTTP_RESPOND_IF_ERR(
        req, TRITONSERVER_ErrorNew(
                 TRITONSERVER_ERROR_INTERNAL,
                 "unexpected error on creating Triton parameter"));
  }

  /* Register repository with model mapping */
  TRITONSERVER_Error* err = nullptr;
  err = TRITONSERVER_ServerRegisterModelRepository(
      server_.get(), repo_parent_path.c_str(), subdir_modelname_map.data(),
      subdir_modelname_map.size());

  TRITONSERVER_ParameterDelete(param);

  // If a model_name is reused i.e. model_name is already mapped, return a 409
  if ((err != nullptr) &&
      (TRITONSERVER_ErrorCode(err) == TRITONSERVER_ERROR_ALREADY_EXISTS)) {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_CONFLICT); /* 409 */
    TRITONSERVER_ErrorDelete(err);
    return;
  } else if (err != nullptr) {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    TRITONSERVER_ErrorDelete(err);
    return;
  }

  err = TRITONSERVER_ServerLoadModel(server_.get(), target_model.c_str());

  /* Unlikely after duplicate repo check, but in case Load Model also returns
   * ALREADY_EXISTS error */
  if ((err != nullptr) &&
      (TRITONSERVER_ErrorCode(err) == TRITONSERVER_ERROR_ALREADY_EXISTS)) {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_CONFLICT); /* 409 */
    TRITONSERVER_ErrorDelete(err);
    return;
  } else if (err != nullptr) {
    SageMakerMMEHandleOOMError(req, err);
  } else {
    std::lock_guard<std::mutex> lock(models_list_mutex_);

    /* Use model name hash as expected in SageMaker MME contract */
    sagemaker_models_list_.emplace(model_name_hash, repo_parent_path);
    evhtp_send_reply(req, EVHTP_RES_OK);
  }

  /* Unregister model repository in case of load failure*/
  if (err != nullptr) {
    err = TRITONSERVER_ServerUnregisterModelRepository(
        server_.get(), repo_parent_path.c_str());
    LOG_VERBOSE(1)
        << "Unregistered model repository due to load failure for model: "
        << target_model << std::endl;
  }

  if (err != nullptr) {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    TRITONSERVER_ErrorDelete(err);
  }

  return;
}
}}  // namespace triton::server
