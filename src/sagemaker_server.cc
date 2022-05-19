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

          HandleInfer(req, multi_model_name, model_version_str_);
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
  (*parse_map)["model_name"] = model_name_string.c_str();

  return;
}

void
SagemakerAPIServer::SageMakerMMEUnloadModel(
    evhtp_request_t* req, const char* model_name)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (sagemaker_models_list_.find(model_name) == sagemaker_models_list_.end()) {
    LOG_VERBOSE(1) << "Model " << model_name << "is not loaded." << std::endl;
    evhtp_send_reply(req, EVHTP_RES_NOTFOUND); /* 404*/
    return;
  }

  HandleRepositoryControl(req, "", model_name, "unload");

  std::string repo_path = sagemaker_models_list_.at(model_name);

  std::string repo_parent_path, subdir, customer_subdir;
  RE2::FullMatch(
      repo_path, model_path_regex_, &repo_parent_path, &subdir,
      &customer_subdir);

  TRITONSERVER_Error* unload_err = TRITONSERVER_ServerUnregisterModelRepository(
      server_.get(), repo_parent_path.c_str());

  if (unload_err != nullptr) {
    EVBufferAddErrorJson(req->buffer_out, unload_err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    TRITONSERVER_ErrorDelete(unload_err);
  }

  sagemaker_models_list_.erase(model_name);
}

void
SagemakerAPIServer::SageMakerMMEGetModel(
    evhtp_request_t* req, const char* model_name)
{
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

void
SagemakerAPIServer::SageMakerMMEHandleLoadError(
    evhtp_request_t* req, TRITONSERVER_Error* load_err)
{
  const char* message = TRITONSERVER_ErrorMessage(load_err);
  std::string error_string(message);

  const std::vector<std::string> error_messages{
      "CUDA out of memory", /* pytorch */
      "CUDA_OUT_OF_MEMORY", /* tensorflow */
      "Out of memory",      /* generic */
  };

  EVBufferAddErrorJson(req->buffer_out, load_err);

  for (long unsigned int i = 0; i < error_messages.size(); i++) {
    if (error_string.find(error_messages[i]) != std::string::npos) {
      /* Return a 507*/
      evhtp_send_reply(req, 507);
      LOG_VERBOSE(1)
          << "Received an OOM error during LOAD MODEL. Returning a 507.";
      return;
    } else {
      /* Return a 400*/
      evhtp_send_reply(req, EVHTP_RES_BADREQ);
      return;
    }
  }
}


void
SagemakerAPIServer::SageMakerMMELoadModel(
    evhtp_request_t* req,
    const std::unordered_map<std::string, std::string> parse_map)
{
  std::string repo_path = parse_map.at("url");
  std::string model_name = parse_map.at("model_name");

  /* Error out if there's more than one subdir/version within
   * supplied model repo, as ensemble in MME is not (currently)
   * supported
   */
  DIR* dir;
  struct dirent* ent;
  int dir_count = 0;
  if ((dir = opendir(repo_path.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      if ((ent->d_type == DT_DIR) && (strcmp(ent->d_name, ".") == 0) &&
          (strcmp(ent->d_name, "..") == 0)) {
        dir_count += 1;
      }
      if (dir_count > 1) {
        HTTP_RESPOND_IF_ERR(
            req,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                "More than one version or model directories found. Note that "
                "Ensemble models are not supported in SageMaker MME mode."));
        closedir(dir);
        return;
      }
    }
    closedir(dir);
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

  std::string subdir_path = subdir;
  if (!customer_subdir.empty()) {
    subdir_path = subdir + "/" + customer_subdir;
  }

  auto param = TRITONSERVER_ParameterNew(
      subdir_path.c_str(), TRITONSERVER_PARAMETER_STRING, model_name.c_str());

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

  err = TRITONSERVER_ServerLoadModel(server_.get(), model_name.c_str());

  /* Unlikely after duplicate repo check, but in case Load Model also returns
   * ALREADY_EXISTS error */
  if ((err != nullptr) &&
      (TRITONSERVER_ErrorCode(err) == TRITONSERVER_ERROR_ALREADY_EXISTS)) {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_CONFLICT); /* 409 */
    TRITONSERVER_ErrorDelete(err);
    return;
  } else if (err != nullptr) {
    SageMakerMMEHandleLoadError(req, err);
  } else {
    std::lock_guard<std::mutex> lock(mutex_);

    sagemaker_models_list_.emplace(model_name, repo_path);
    evhtp_send_reply(req, EVHTP_RES_OK);
  }

  /* Unregister model repository in case of load failure*/
  if (err != nullptr) {
    err = TRITONSERVER_ServerUnregisterModelRepository(
        server_.get(), repo_parent_path.c_str());
    LOG_VERBOSE(1)
        << "Unregistered model repository due to load failure for model: "
        << model_name << std::endl;
  }

  if (err != nullptr) {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    TRITONSERVER_ErrorDelete(err);
  }

  return;
}

}}  // namespace triton::server
