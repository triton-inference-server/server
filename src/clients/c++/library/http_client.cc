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

// Include this first to make sure we are a friend of common classes.
#define TRITON_INFERENCE_SERVER_CLIENT_CLASS InferenceServerHttpClient
#include "src/clients/c++/library/common.h"

#include "src/clients/c++/library/http_client.h"

#include <curl/curl.h>
#include <cstdint>
#include <iostream>
#include <queue>

extern "C" {
#include <src/clients/c++/library/cencode.h>
}

#ifdef _WIN32
#define strncasecmp(x, y, z) _strnicmp(x, y, z)
#endif  //_WIN32

namespace nvidia { namespace inferenceserver { namespace client {

namespace {

//==============================================================================

// Global initialization for libcurl. Libcurl requires global
// initialization before any other threads are created and before any
// curl methods are used. The curl_global static object is used to
// perform this initialization.
class CurlGlobal {
 public:
  CurlGlobal();
  ~CurlGlobal();

  const Error& Status() const { return err_; }

 private:
  Error err_;
};

CurlGlobal::CurlGlobal() : err_(Error::Success)
{
  if (curl_global_init(CURL_GLOBAL_ALL) != 0) {
    err_ = Error("global initialization failed");
  }
}

CurlGlobal::~CurlGlobal()
{
  curl_global_cleanup();
}

static CurlGlobal curl_global;

std::string
GetQueryString(const Headers& query_params)
{
  std::string query_string;
  bool first = true;
  for (const auto& pr : query_params) {
    if (first) {
      first = false;
    } else {
      query_string += "&";
    }
    query_string += pr.first + "=" + pr.second;
  }
  return query_string;
}

// Encodes the contents of the provided buffer into base64 string. Note the
// string is not guaranteed to be null-terminated. Must rely on the returned
// encoded size to get the right contents.
void
Base64Encode(
    char* raw_ptr, size_t raw_size, char** encoded_ptr, int* encoded_size)
{
  // Encode the handle object to base64
  base64_encodestate es;
  base64_init_encodestate(&es);
  *encoded_ptr = (char*)malloc(raw_size * 2); /* ~4/3 x raw_size */
  *encoded_size = base64_encode_block(raw_ptr, raw_size, *encoded_ptr, &es);
  int padding_size = base64_encode_blockend(*encoded_ptr + *encoded_size, &es);
  *encoded_size += padding_size;
}

}  // namespace

//==============================================================================

std::string
GetJsonText(const rapidjson::Document& json_dom)
{
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  json_dom.Accept(writer);
  return buffer.GetString();
}

//==============================================================================

class HttpInferRequest : public InferRequest {
 public:
  HttpInferRequest(InferenceServerClient::OnCompleteFn callback = nullptr);
  ~HttpInferRequest();

  // Initialize the request for HTTP transfer. */
  Error InitializeRequest(rapidjson::Document& response_json);

  // Adds the input data to be delivered to the server
  Error AddInput(uint8_t* buf, size_t byte_size);

  // Copy into 'buf' up to 'size' bytes of input data. Return the
  // actual amount copied in 'input_bytes'.
  Error GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes);

 private:
  friend class InferenceServerHttpClient;
  friend class InferResultHttp;

  // Pointer to easy handle that is processing the request
  CURL* easy_handle_;

  // Pointer to the list of the HTTP request header, keep it such that it will
  // be valid during the transfer and can be freed once transfer is completed.
  struct curl_slist* header_list_;

  // Status code for the HTTP request.
  CURLcode http_status_;

  size_t total_input_byte_size_;

  rapidjson::StringBuffer request_json_;

  // Buffer that accumulates the serialized response at the
  // end of the body.
  std::unique_ptr<std::string> infer_response_buffer_;

  // The pointers to the input data.
  std::queue<std::pair<uint8_t*, size_t>> data_buffers_;

  size_t response_json_size_;
};


HttpInferRequest::HttpInferRequest(InferenceServerClient::OnCompleteFn callback)
    : InferRequest(callback), easy_handle_(curl_easy_init()),
      header_list_(nullptr), total_input_byte_size_(0), response_json_size_(0)
{
}

HttpInferRequest::~HttpInferRequest()
{
  if (header_list_ != nullptr) {
    curl_slist_free_all(header_list_);
    header_list_ = nullptr;
  }

  if (easy_handle_ != nullptr) {
    curl_easy_cleanup(easy_handle_);
  }
}

Error
HttpInferRequest::InitializeRequest(rapidjson::Document& request_json)
{
  data_buffers_ = {};
  total_input_byte_size_ = 0;

  request_json_.Clear();
  rapidjson::Writer<rapidjson::StringBuffer> writer(request_json_);
  request_json.Accept(writer);

  // Add the buffer holding the json to be delivered first
  AddInput((uint8_t*)request_json_.GetString(), request_json_.GetSize());

  // Prepare buffer to record the response
  infer_response_buffer_.reset(new std::string());

  return Error::Success;
}

Error
HttpInferRequest::AddInput(uint8_t* buf, size_t byte_size)
{
  data_buffers_.push(std::pair<uint8_t*, size_t>(buf, byte_size));
  total_input_byte_size_ += byte_size;
  return Error::Success;
}

Error
HttpInferRequest::GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes)
{
  *input_bytes = 0;

  if (data_buffers_.empty()) {
    return Error::Success;
  }

  while (!data_buffers_.empty() && size > 0) {
    const size_t csz = (std::min)(data_buffers_.front().second, size);
    if (csz > 0) {
      const uint8_t* input_ptr = data_buffers_.front().first;
      std::copy(input_ptr, input_ptr + csz, buf);
      size -= csz;
      buf += csz;
      *input_bytes += csz;


      data_buffers_.front().first += csz;
      data_buffers_.front().second -= csz;
      if (data_buffers_.front().second == 0) {
        data_buffers_.pop();
      }
    }
  }

  // Set end timestamp if all inputs have been sent.
  if (data_buffers_.empty()) {
    Timer().CaptureTimestamp(RequestTimers::Kind::SEND_END);
  }

  return Error::Success;
}

//==============================================================================

class InferResultHttp : public InferResult {
 public:
  static Error Create(
      InferResult** infer_result,
      std::shared_ptr<HttpInferRequest> infer_request);

  Error RequestStatus() const override;
  Error ModelName(std::string* name) const override;
  Error ModelVersion(std::string* version) const override;
  Error Id(std::string* id) const override;
  Error Shape(const std::string& output_name, std::vector<int64_t>* shape)
      const override;
  Error Datatype(
      const std::string& output_name, std::string* datatype) const override;
  Error RawData(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const override;
  Error StringData(
      const std::string& output_name,
      std::vector<std::string>* string_result) const override;
  std::string DebugString() const override;

 private:
  InferResultHttp(std::shared_ptr<HttpInferRequest> infer_request);

  std::map<std::string, const rapidjson::Value*> output_name_to_result_map_;
  std::map<std::string, std::pair<const uint8_t*, const size_t>>
      output_name_to_buffer_map_;

  rapidjson::Document response_json_;
  std::shared_ptr<HttpInferRequest> infer_request_;
};

Error
InferResultHttp::Create(
    InferResult** infer_result, std::shared_ptr<HttpInferRequest> infer_request)
{
  *infer_result =
      reinterpret_cast<InferResult*>(new InferResultHttp(infer_request));
  return Error::Success;
}

Error
InferResultHttp::ModelName(std::string* name) const
{
  const auto& itr = response_json_.FindMember("model_name");
  if (itr != response_json_.MemberEnd()) {
    *name = std::string(itr->value.GetString(), itr->value.GetStringLength());
  } else {
    return Error("model name was not returned in the response");
  }
  return Error::Success;
}

Error
InferResultHttp::ModelVersion(std::string* version) const
{
  const auto& itr = response_json_.FindMember("model_version");
  if (itr != response_json_.MemberEnd()) {
    *version =
        std::string(itr->value.GetString(), itr->value.GetStringLength());
  } else {
    return Error("model version was not returned in the response");
  }
  return Error::Success;
}

Error
InferResultHttp::Id(std::string* id) const
{
  const auto& itr = response_json_.FindMember("id");
  if (itr != response_json_.MemberEnd()) {
    *id = std::string(itr->value.GetString(), itr->value.GetStringLength());
  } else {
    return Error("model version was not returned in the response");
  }
  return Error::Success;
}

Error
InferResultHttp::Shape(
    const std::string& output_name, std::vector<int64_t>* shape) const
{
  shape->clear();
  auto itr = output_name_to_result_map_.find(output_name);
  if (itr != output_name_to_result_map_.end()) {
    const auto shape_itr = itr->second->FindMember("shape");
    if (shape_itr != itr->second->MemberEnd()) {
      const rapidjson::Value& shape_json = shape_itr->value;
      for (rapidjson::SizeType i = 0; i < shape_json.Size(); i++) {
        shape->push_back(shape_json[i].GetInt());
      }
    } else {
      return Error(
          "The response does not contain shape for output name " + output_name);
    }
  } else {
    return Error(
        "The response does not contain results for output name " + output_name);
  }
  return Error::Success;
}

Error
InferResultHttp::Datatype(
    const std::string& output_name, std::string* datatype) const
{
  auto itr = output_name_to_result_map_.find(output_name);
  if (itr != output_name_to_result_map_.end()) {
    const auto datatype_itr = itr->second->FindMember("datatype");
    if (datatype_itr != itr->second->MemberEnd()) {
      const rapidjson::Value& datatype_json = datatype_itr->value;
      *datatype = std::string(
          datatype_json.GetString(), datatype_json.GetStringLength());
    } else {
      return Error(
          "The response does not contain datatype for output name " +
          output_name);
    }
  } else {
    return Error(
        "The response does not contain datatype or output name " + output_name);
  }
  return Error::Success;
}


Error
InferResultHttp::RawData(
    const std::string& output_name, const uint8_t** buf,
    size_t* byte_size) const
{
  auto itr = output_name_to_buffer_map_.find(output_name);
  if (itr != output_name_to_buffer_map_.end()) {
    *buf = itr->second.first;
    *byte_size = itr->second.second;
  } else {
    return Error(
        "The response does not contain results for output name " + output_name);
  }

  return Error::Success;
}

Error
InferResultHttp::StringData(
    const std::string& output_name,
    std::vector<std::string>* string_result) const
{
  std::string datatype;
  Error err = Datatype(output_name, &datatype);
  if (!err.IsOk()) {
    return err;
  }
  if (datatype.compare("BYTES") != 0) {
    return Error(
        "This function supports tensors with datatype 'BYTES', requested "
        "output tensor '" +
        output_name + "' with datatype '" + datatype + "'");
  }

  const uint8_t* buf;
  size_t byte_size;
  err = RawData(output_name, &buf, &byte_size);
  string_result->clear();
  size_t buf_offset = 0;
  while (byte_size > buf_offset) {
    const uint32_t element_size =
        *(reinterpret_cast<const uint32_t*>(buf + buf_offset));
    string_result->emplace_back(
        reinterpret_cast<const char*>(buf + buf_offset + sizeof(element_size)),
        element_size);
    buf_offset += (sizeof(element_size) + element_size);
  }

  return Error::Success;
}

std::string
InferResultHttp::DebugString() const
{
  return GetJsonText(response_json_);
}

Error
InferResultHttp::RequestStatus() const
{
  const auto& itr = response_json_.FindMember("error");
  if (itr != response_json_.MemberEnd()) {
    return Error(
        std::string(itr->value.GetString(), itr->value.GetStringLength()));
  }

  return Error::Success;
}

InferResultHttp::InferResultHttp(
    std::shared_ptr<HttpInferRequest> infer_request)
    : infer_request_(infer_request)
{
  size_t offset = infer_request->response_json_size_;
  if (offset != 0) {
    response_json_.Parse(
        (char*)infer_request->infer_response_buffer_.get()->c_str(), offset);
  } else {
    response_json_.Parse(
        (char*)infer_request->infer_response_buffer_.get()->c_str());
  }
  const auto& itr = response_json_.FindMember("outputs");
  if (itr != response_json_.MemberEnd()) {
    const rapidjson::Value& outputs = itr->value;
    for (size_t i = 0; i < outputs.Size(); i++) {
      const rapidjson::Value& output = outputs[i];
      const char* output_name = output["name"].GetString();
      output_name_to_result_map_[output_name] = &output;
      const auto& pitr = output.FindMember("parameters");
      if (pitr != output.MemberEnd()) {
        const rapidjson::Value& param = pitr->value;
        const auto& bitr = param.FindMember("binary_data_size");
        if (bitr != param.MemberEnd()) {
          size_t byte_size = bitr->value.GetInt();
          output_name_to_buffer_map_.emplace(
              output_name,
              std::pair<const uint8_t*, const size_t>(
                  (uint8_t*)(infer_request->infer_response_buffer_.get()
                                 ->c_str()) +
                      offset,
                  byte_size));
          offset += byte_size;
        }
      }
    }
  }
}

//==============================================================================

InferenceServerHttpClient::~InferenceServerHttpClient()
{
  exiting_ = true;
  // thread not joinable if AsyncInfer() is not called
  // (it is default constructed thread before the first AsyncInfer() call)
  if (worker_.joinable()) {
    cv_.notify_all();
    worker_.join();
  }

  if (multi_handle_ != nullptr) {
    for (auto& request : ongoing_async_requests_) {
      CURL* easy_handle = request.second->easy_handle_;
      // Just remove, easy_cleanup will be done in ~HttpInferRequest()
      curl_multi_remove_handle(multi_handle_, easy_handle);
    }
    curl_multi_cleanup(multi_handle_);
  }
}

Error
InferenceServerHttpClient::Create(
    std::unique_ptr<InferenceServerHttpClient>* client,
    const std::string& server_url, bool verbose)
{
  client->reset(new InferenceServerHttpClient(server_url, verbose));
  return Error::Success;
}

Error
InferenceServerHttpClient::IsServerLive(
    bool* live, const Headers& headers, const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/health/live");

  long http_code;
  rapidjson::Document response;
  err = Get(request_uri, headers, query_params, &response, &http_code);

  *live = (http_code == 200) ? true : false;

  return err;
}

Error
InferenceServerHttpClient::IsServerReady(
    bool* ready, const Headers& headers, const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/health/live");

  long http_code;
  rapidjson::Document response;
  err = Get(request_uri, headers, query_params, &response, &http_code);

  *ready = (http_code == 200) ? true : false;

  return err;
}

Error
InferenceServerHttpClient::IsModelReady(
    bool* ready, const std::string& model_name,
    const std::string& model_version, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/models/" + model_name);
  if (!model_version.empty()) {
    request_uri = request_uri + "/versions/" + model_version;
  }
  request_uri = request_uri + "/ready";

  long http_code;
  rapidjson::Document response;
  err = Get(request_uri, headers, query_params, &response, &http_code);

  *ready = (http_code == 200) ? true : false;

  return err;
}


Error
InferenceServerHttpClient::ServerMetadata(
    rapidjson::Document* server_metadata, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2");

  long http_code;
  err = Get(request_uri, headers, query_params, server_metadata, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}


Error
InferenceServerHttpClient::ModelMetadata(
    rapidjson::Document* model_metadata, const std::string& model_name,
    const std::string& model_version, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/models/" + model_name);
  if (!model_version.empty()) {
    request_uri = request_uri + "/versions/" + model_version;
  }

  long http_code;
  err = Get(request_uri, headers, query_params, model_metadata, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}


Error
InferenceServerHttpClient::ModelConfig(
    rapidjson::Document* model_config, const std::string& model_name,
    const std::string& model_version, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/models/" + model_name);
  if (!model_version.empty()) {
    request_uri = request_uri + "/versions/" + model_version;
  }
  request_uri = request_uri + "/config";

  long http_code;
  err = Get(request_uri, headers, query_params, model_config, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}


Error
InferenceServerHttpClient::ModelRepositoryIndex(
    rapidjson::Document* repository_index, const Headers& headers,
    const Parameters& query_params)
{
  Error err;
  std::string request_uri(url_ + "/v2/repository/index");

  long http_code;
  err = Get(request_uri, headers, query_params, repository_index, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}

Error
InferenceServerHttpClient::LoadModel(
    const std::string& model_name, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(
      url_ + "/v2/repository/models/" + model_name + "/load");

  rapidjson::Document request(rapidjson::kObjectType);
  rapidjson::Document response(rapidjson::kObjectType);
  long http_code;
  err =
      Post(request_uri, request, headers, query_params, &response, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}

Error
InferenceServerHttpClient::UnloadModel(
    const std::string& model_name, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(
      url_ + "/v2/repository/models/" + model_name + "/unload");

  rapidjson::Document request(rapidjson::kObjectType);
  rapidjson::Document response(rapidjson::kObjectType);
  long http_code;
  err =
      Post(request_uri, request, headers, query_params, &response, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}


Error
InferenceServerHttpClient::ModelInferenceStatistics(
    rapidjson::Document* infer_stat, const std::string& model_name,
    const std::string& model_version, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/models");
  if (!model_name.empty()) {
    request_uri += "/" + model_name;
  }
  if (!model_version.empty()) {
    request_uri += "/versions/" + model_version;
  }
  request_uri += "/stats";

  long http_code;
  err = Get(request_uri, headers, query_params, infer_stat, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}

Error
InferenceServerHttpClient::SystemSharedMemoryStatus(
    rapidjson::Document* status, const std::string& name,
    const Headers& headers, const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/systemsharedmemory");
  if (!name.empty()) {
    request_uri = request_uri + "/region/" + name;
  }
  request_uri = request_uri + "/status";

  long http_code;
  err = Get(request_uri, headers, query_params, status, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}

Error
InferenceServerHttpClient::RegisterSystemSharedMemory(
    const std::string& name, const std::string& key, const size_t byte_size,
    const size_t offset, const Headers& headers, const Parameters& query_params)
{
  Error err;

  std::string request_uri(
      url_ + "/v2/systemsharedmemory/region/" + name + "/register");

  rapidjson::Document request(rapidjson::kObjectType);
  rapidjson::Document::AllocatorType& allocator = request.GetAllocator();
  {
    rapidjson::Value key_json(key.c_str(), allocator);
    request.AddMember("key", key_json, allocator);
    rapidjson::Value offset_json(offset);
    request.AddMember("offet", offset_json, allocator);
    rapidjson::Value byte_size_json(byte_size);
    request.AddMember("byte_size", byte_size_json, allocator);
  }
  rapidjson::Document response(rapidjson::kObjectType);

  long http_code;
  err =
      Post(request_uri, request, headers, query_params, &response, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}

Error
InferenceServerHttpClient::UnregisterSystemSharedMemory(
    const std::string& region_name, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/systemsharedmemory");
  if (!region_name.empty()) {
    request_uri = request_uri + "/region/" + region_name;
  }
  request_uri = request_uri + "/unregister";

  rapidjson::Document request(rapidjson::kObjectType);
  rapidjson::Document response(rapidjson::kObjectType);
  long http_code;
  err =
      Post(request_uri, request, headers, query_params, &response, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}

Error
InferenceServerHttpClient::CudaSharedMemoryStatus(
    rapidjson::Document* status, const std::string& region_name,
    const Headers& headers, const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/cudasharedmemory");
  if (!region_name.empty()) {
    request_uri = request_uri + "/region/" + region_name;
  }
  request_uri = request_uri + "/status";

  long http_code;
  err = Get(request_uri, headers, query_params, status, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}

Error
InferenceServerHttpClient::RegisterCudaSharedMemory(
    const std::string& name, const cudaIpcMemHandle_t& raw_handle,
    const size_t device_id, const size_t byte_size, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(
      url_ + "/v2/cudasharedmemory/region/" + name + "/register");

  rapidjson::Document request(rapidjson::kObjectType);
  rapidjson::Document::AllocatorType& allocator = request.GetAllocator();
  {
    rapidjson::Value raw_handle_json(rapidjson::kObjectType);
    {
      char* encoded_handle = nullptr;
      int encoded_size;
      Base64Encode(
          (char*)((void*)&raw_handle), sizeof(cudaIpcMemHandle_t),
          &encoded_handle, &encoded_size);
      if (encoded_handle == nullptr) {
        return Error("Failed to base64 encode the cudaIpcMemHandle_t");
      }
      const auto encoded_handle_str = std::string(encoded_handle, encoded_size);
      rapidjson::Value b64_json(
          rapidjson::StringRef(encoded_handle_str.c_str()), allocator);
      delete encoded_handle;
      raw_handle_json.AddMember("b64", b64_json, allocator);
    }
    request.AddMember("raw_handle", raw_handle_json, allocator);
    rapidjson::Value device_id_json(device_id);
    request.AddMember("device_id", device_id_json, allocator);
    rapidjson::Value byte_size_json(byte_size);
    request.AddMember("byte_size", byte_size_json, allocator);
  }
  rapidjson::Document response(rapidjson::kObjectType);
  long http_code;
  err =
      Post(request_uri, request, headers, query_params, &response, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}

Error
InferenceServerHttpClient::UnregisterCudaSharedMemory(
    const std::string& name, const Headers& headers,
    const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/cudasharedmemory");
  if (!name.empty()) {
    request_uri = request_uri + "/region/" + name;
  }
  request_uri = request_uri + "/unregister";

  rapidjson::Document request(rapidjson::kObjectType);
  rapidjson::Document response(rapidjson::kObjectType);
  long http_code;
  err =
      Post(request_uri, request, headers, query_params, &response, &http_code);
  if ((http_code != 200) && err.IsOk()) {
    return Error(
        "[INTERNAL] Request failed with missing error message in response");
  }
  return err;
}

Error
InferenceServerHttpClient::Infer(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers, const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/models/" + options.model_name_);
  if (!options.model_version_.empty()) {
    request_uri = request_uri + "/versions/" + options.model_version_;
  }
  request_uri = request_uri + "/infer";

  std::shared_ptr<HttpInferRequest> sync_request(new HttpInferRequest());

  sync_request->Timer().Reset();
  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_START);

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  err = PreRunProcessing(
      request_uri, options, inputs, outputs, headers, query_params,
      sync_request);
  if (!err.IsOk()) {
    return err;
  }

  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_START);

  // Set SEND_END when content length is 0 (because
  // CURLOPT_READFUNCTION will not be called). In that case, we can't
  // measure SEND_END properly (send ends after sending request
  // header).
  if (sync_request->total_input_byte_size_ == 0) {
    sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_END);
  }

  // During this call SEND_END (except in above case), RECV_START, and
  // RECV_END will be set.
  sync_request->http_status_ = curl_easy_perform(sync_request->easy_handle_);

  InferResultHttp::Create(result, sync_request);

  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_END);

  err = UpdateInferStat(sync_request->Timer());
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }

  err = (*result)->RequestStatus();

  return err;
}


Error
InferenceServerHttpClient::AsyncInfer(
    OnCompleteFn callback, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers, const Parameters& query_params)
{
  if (callback == nullptr) {
    return Error(
        "Callback function must be provided along with AsyncInfer() call.");
  }
  std::shared_ptr<HttpInferRequest> async_request;
  if (!multi_handle_) {
    return Error("failed to start HTTP asynchronous client");
  } else if (!worker_.joinable()) {
    worker_ = std::thread(&InferenceServerHttpClient::AsyncTransfer, this);
  }

  std::string request_uri(url_ + "/v2/models/" + options.model_name_);
  if (!options.model_version_.empty()) {
    request_uri = request_uri + "/versions/" + options.model_version_;
  }
  request_uri = request_uri + "/infer";

  HttpInferRequest* raw_async_request =
      new HttpInferRequest(std::move(callback));
  async_request.reset(raw_async_request);

  if (!async_request->easy_handle_) {
    return Error("failed to initialize HTTP client");
  }

  async_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_START);

  Error err = PreRunProcessing(
      request_uri, options, inputs, outputs, headers, query_params,
      async_request);
  if (!err.IsOk()) {
    return err;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);

    auto insert_result = ongoing_async_requests_.emplace(std::make_pair(
        reinterpret_cast<uintptr_t>(async_request->easy_handle_),
        async_request));

    if (!insert_result.second) {
      return Error("Failed to insert new asynchronous request context.");
    }
    async_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_START);
    if (async_request->total_input_byte_size_ == 0) {
      // Set SEND_END here because CURLOPT_READFUNCTION will not be called if
      // content length is 0. In that case, we can't measure SEND_END properly
      // (send ends after sending request header).
      async_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_END);
    }
    curl_multi_add_handle(multi_handle_, async_request->easy_handle_);
  }

  cv_.notify_all();
  return Error::Success;
}

InferenceServerHttpClient::InferenceServerHttpClient(
    const std::string& url, bool verbose)
    : InferenceServerClient(verbose), url_(url),
      multi_handle_(curl_multi_init())
{
}


size_t
InferenceServerHttpClient::InferRequestProvider(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpInferRequest* request = reinterpret_cast<HttpInferRequest*>(userp);

  size_t input_bytes = 0;
  Error err = request->GetNextInput(
      reinterpret_cast<uint8_t*>(contents), size * nmemb, &input_bytes);
  if (!err.IsOk()) {
    std::cerr << "RequestProvider: " << err << std::endl;
    return CURL_READFUNC_ABORT;
  }

  return input_bytes;
}

size_t
InferenceServerHttpClient::InferResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpInferRequest* request = reinterpret_cast<HttpInferRequest*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kInferHeaderContentLengthHTTPHeader);
  if ((idx < byte_size) &&
      !strncasecmp(buf, kInferHeaderContentLengthHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);
      request->response_json_size_ = std::stoi(hdr);
    }
  }

  return byte_size;
}

size_t
InferenceServerHttpClient::InferResponseHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpInferRequest* request = reinterpret_cast<HttpInferRequest*>(userp);

  if (request->Timer().Timestamp(RequestTimers::Kind::RECV_START) == 0) {
    request->Timer().CaptureTimestamp(RequestTimers::Kind::RECV_START);
  }

  uint8_t* buf = reinterpret_cast<uint8_t*>(contents);
  size_t result_bytes = size * nmemb;
  std::copy(
      buf, buf + result_bytes,
      std::back_inserter(*request->infer_response_buffer_));

  // ResponseHandler may be called multiple times so we overwrite
  // RECV_END so that we always have the time of the last.
  request->Timer().CaptureTimestamp(RequestTimers::Kind::RECV_END);

  return result_bytes;
}

void
InferenceServerHttpClient::PrepareRequestJson(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    rapidjson::Document* request_json)
{
  // Populate the request JSON.
  rapidjson::Document::AllocatorType& allocator = request_json->GetAllocator();
  request_json->SetObject();
  {
    rapidjson::Value request_id_json(options.request_id_.c_str(), allocator);
    request_json->AddMember("id", request_id_json, allocator);
    rapidjson::Value parameters_json(rapidjson::kObjectType);
    {
      if (options.sequence_id_ != 0) {
        rapidjson::Value sequence_id_json(options.sequence_id_);
        parameters_json.AddMember("sequence_id", sequence_id_json, allocator);
        rapidjson::Value sequence_start_json(options.sequence_start_);
        parameters_json.AddMember(
            "sequence_start", sequence_start_json, allocator);
        rapidjson::Value sequence_end_json(options.sequence_end_);
        parameters_json.AddMember("sequence_end", sequence_end_json, allocator);
      }

      if (options.priority_ != 0) {
        rapidjson::Value priority_json(options.priority_);
        parameters_json.AddMember("priority", priority_json, allocator);
      }

      if (options.timeout_ != 0) {
        rapidjson::Value timeout_json(options.timeout_);
        parameters_json.AddMember("timeout", timeout_json, allocator);
      }
    }
    request_json->AddMember("parameters", parameters_json, allocator);
  }

  rapidjson::Value inputs_json(rapidjson::kArrayType);
  {
    for (const auto this_input : inputs) {
      rapidjson::Value this_input_json(rapidjson::kObjectType);
      {
        rapidjson::Value name_json(this_input->Name().c_str(), allocator);
        this_input_json.AddMember("name", name_json, allocator);
        rapidjson::Value shape_json(rapidjson::kArrayType);
        {
          for (const auto dim : this_input->Shape()) {
            rapidjson::Value dim_json(dim);
            shape_json.PushBack(dim_json, allocator);
          }
        }
        this_input_json.AddMember("shape", shape_json, allocator);
        rapidjson::Value datatype_json(
            this_input->Datatype().c_str(), allocator);
        this_input_json.AddMember("datatype", datatype_json, allocator);
        rapidjson::Value parameters_json(rapidjson::kObjectType);
        if (this_input->IsSharedMemory()) {
          std::string region_name;
          size_t offset;
          size_t byte_size;
          this_input->SharedMemoryInfo(&region_name, &byte_size, &offset);
          {
            rapidjson::Value shared_memory_region_json(
                region_name.c_str(), allocator);
            parameters_json.AddMember(
                "shared_memory_region", shared_memory_region_json, allocator);
            rapidjson::Value shared_memory_byte_size_json(byte_size);
            parameters_json.AddMember(
                "shared_memory_byte_size", shared_memory_byte_size_json,
                allocator);
            if (offset != 0) {
              rapidjson::Value shared_memory_offset_json(offset);
              parameters_json.AddMember(
                  "shared_memory_offset", shared_memory_offset_json, allocator);
            }
          }
        } else {
          size_t byte_size;
          this_input->ByteSize(&byte_size);
          rapidjson::Value binary_data_size_json(byte_size);
          parameters_json.AddMember(
              "binary_data_size", binary_data_size_json, allocator);
        }
        this_input_json.AddMember("parameters", parameters_json, allocator);
      }
      inputs_json.PushBack(this_input_json, allocator);
    }
  }
  request_json->AddMember("inputs", inputs_json, allocator);

  rapidjson::Value ouputs_json(rapidjson::kArrayType);
  {
    for (const auto this_output : outputs) {
      rapidjson::Value this_output_json(rapidjson::kObjectType);
      {
        rapidjson::Value name_json(this_output->Name().c_str(), allocator);
        this_output_json.AddMember("name", name_json, allocator);
        rapidjson::Value parameters_json(rapidjson::kObjectType);
        size_t class_count = this_output->ClassCount();
        if (class_count != 0) {
          rapidjson::Value classification_json(class_count);
          parameters_json.AddMember(
              "classification", classification_json, allocator);
        }
        if (this_output->IsSharedMemory()) {
          std::string region_name;
          size_t offset;
          size_t byte_size;
          this_output->SharedMemoryInfo(&region_name, &byte_size, &offset);
          {
            rapidjson::Value shared_memory_region_json(
                region_name.c_str(), allocator);
            parameters_json.AddMember(
                "shared_memory_region", shared_memory_region_json, allocator);
            rapidjson::Value shared_memory_byte_size_json(byte_size);
            parameters_json.AddMember(
                "shared_memory_byte_size", shared_memory_byte_size_json,
                allocator);
            if (offset != 0) {
              rapidjson::Value shared_memory_offset_json(offset);
              parameters_json.AddMember(
                  "shared_memory_offset", shared_memory_offset_json, allocator);
            }
          }
        } else {
          rapidjson::Value binary_data_json(true);
          parameters_json.AddMember("binary_data", binary_data_json, allocator);
        }
        this_output_json.AddMember("parameters", parameters_json, allocator);
      }
      ouputs_json.PushBack(this_output_json, allocator);
    }
  }
  request_json->AddMember("outputs", ouputs_json, allocator);
}

Error
InferenceServerHttpClient::PreRunProcessing(
    std::string& request_uri, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers, const Parameters& query_params,
    std::shared_ptr<HttpInferRequest>& request)
{
  rapidjson::Document request_json;
  PrepareRequestJson(options, inputs, outputs, &request_json);

  // Prepare the request object to provide the data for inference.
  std::shared_ptr<HttpInferRequest> http_request =
      std::static_pointer_cast<HttpInferRequest>(request);
  http_request->InitializeRequest(request_json);

  // Add the buffers holding input tensor data
  for (const auto this_input : inputs) {
    if (!this_input->IsSharedMemory()) {
      this_input->PrepareForRequest();
      bool end_of_input = false;
      while (!end_of_input) {
        const uint8_t* buf;
        size_t buf_size;
        this_input->GetNext(&buf, &buf_size, &end_of_input);
        if (buf != nullptr) {
          http_request->AddInput(const_cast<uint8_t*>(buf), buf_size);
        }
      }
    }
  }

  // Prepare curl
  CURL* curl = http_request->easy_handle_;
  if (!curl) {
    return Error("failed to initialize HTTP client");
  }

  if (!query_params.empty()) {
    request_uri = request_uri + "?" + GetQueryString(query_params);
  }

  curl_easy_setopt(curl, CURLOPT_URL, request_uri.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  const long buffer_byte_size = 16 * 1024 * 1024;
  curl_easy_setopt(curl, CURLOPT_UPLOAD_BUFFERSIZE, buffer_byte_size);
  curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, buffer_byte_size);

  // request data provided by InferRequestProvider()
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, InferRequestProvider);
  curl_easy_setopt(curl, CURLOPT_READDATA, http_request.get());

  // response headers handled by InferResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, InferResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, http_request.get());

  // response data handled by InferResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, InferResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, http_request.get());

  const curl_off_t post_byte_size = http_request->total_input_byte_size_;
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, post_byte_size);

  struct curl_slist* list = nullptr;

  std::string infer_hdr{std::string(kInferHeaderContentLengthHTTPHeader) +
                        ": " +
                        std::to_string(http_request->request_json_.GetSize())};
  list = curl_slist_append(list, infer_hdr.c_str());
  list = curl_slist_append(list, "Expect:");
  list = curl_slist_append(list, "Content-Type: application/octet-stream");
  for (const auto& pr : headers) {
    std::string hdr = pr.first + ": " + pr.second;
    list = curl_slist_append(list, hdr.c_str());
  }
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);


  // The list will be freed when the request is destructed
  http_request->header_list_ = list;

  return Error::Success;
}

void
InferenceServerHttpClient::AsyncTransfer()
{
  int place_holder = 0;
  CURLMsg* msg = nullptr;
  do {
    std::vector<std::shared_ptr<HttpInferRequest>> request_list;

    // sleep if no work is available
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
      if (this->exiting_) {
        return true;
      }
      // wake up if an async request has been generated
      return !this->ongoing_async_requests_.empty();
    });
    curl_multi_perform(multi_handle_, &place_holder);
    while ((msg = curl_multi_info_read(multi_handle_, &place_holder))) {
      // update request status
      uintptr_t identifier = reinterpret_cast<uintptr_t>(msg->easy_handle);
      auto itr = ongoing_async_requests_.find(identifier);
      // This shouldn't happen
      if (itr == ongoing_async_requests_.end()) {
        fprintf(
            stderr,
            "Unexpected error: received completed request that"
            " is not in the list of asynchronous requests.\n");
        curl_multi_remove_handle(multi_handle_, msg->easy_handle);
        curl_easy_cleanup(msg->easy_handle);
        continue;
      }
      request_list.emplace_back(itr->second);
      ongoing_async_requests_.erase(identifier);
      curl_multi_remove_handle(multi_handle_, msg->easy_handle);
      std::shared_ptr<HttpInferRequest> async_request = request_list.back();

      if (msg->msg != CURLMSG_DONE) {
        // Something wrong happened.
        fprintf(stderr, "Unexpected error: received CURLMsg=%d\n", msg->msg);
      } else {
        async_request->Timer().CaptureTimestamp(
            RequestTimers::Kind::REQUEST_END);
        Error err = UpdateInferStat(async_request->Timer());
        if (!err.IsOk()) {
          std::cerr << "Failed to update context stat: " << err << std::endl;
        }
      }
      async_request->http_status_ = msg->data.result;
    }
    lock.unlock();

    for (auto& this_request : request_list) {
      InferResult* result;
      InferResultHttp::Create(&result, this_request);
      this_request->callback_(result);
    }
  } while (!exiting_);
}

size_t
InferenceServerHttpClient::ResponseHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  std::string* response_string = reinterpret_cast<std::string*>(userp);
  uint8_t* buf = reinterpret_cast<uint8_t*>(contents);
  size_t result_bytes = size * nmemb;
  std::copy(buf, buf + result_bytes, std::back_inserter(*response_string));
  return result_bytes;
}

Error
InferenceServerHttpClient::Get(
    std::string& request_uri, const Headers& headers,
    const Parameters& query_params, rapidjson::Document* response,
    long* http_code)
{
  if (!query_params.empty()) {
    request_uri = request_uri + "?" + GetQueryString(query_params);
  }

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return Error("failed to initialize HTTP client");
  }

  curl_easy_setopt(curl, CURLOPT_URL, request_uri.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // Response data handled by ResponseHandler()
  std::string response_string;
  response_string.reserve(256);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

  // Add user provided headers...
  struct curl_slist* header_list = nullptr;
  for (const auto& pr : headers) {
    std::string hdr = pr.first + ": " + pr.second;
    header_list = curl_slist_append(header_list, hdr.c_str());
  }

  if (header_list != nullptr) {
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
  }

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);
    return Error("HTTP client failed: " + std::string(curl_easy_strerror(res)));
  }

  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, http_code);

  curl_slist_free_all(header_list);
  curl_easy_cleanup(curl);

  if (!response_string.empty()) {
    response->Parse(response_string.c_str(), response_string.size());
    if (response->HasParseError()) {
      return Error(
          "failed to parse the request JSON buffer: " +
          std::string(GetParseError_En(response->GetParseError())) + " at " +
          std::to_string(response->GetErrorOffset()));
    }

    if (verbose_) {
      std::cout << GetJsonText(*response) << std::endl;
    }

    if (response->IsObject()) {
      const auto& itr = response->FindMember("error");
      if (itr != response->MemberEnd()) {
        return Error(itr->value.GetString());
      }
    }
  }

  return Error::Success;
}

Error
InferenceServerHttpClient::Post(
    std::string& request_uri, const rapidjson::Document& request,
    const Headers& headers, const Parameters& query_params,
    rapidjson::Document* response, long* http_code)
{
  if (!query_params.empty()) {
    request_uri = request_uri + "?" + GetQueryString(query_params);
  }

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return Error("failed to initialize HTTP client");
  }

  // Prepare the string buffer with the request object
  rapidjson::StringBuffer request_data;
  rapidjson::Writer<rapidjson::StringBuffer> writer(request_data);
  request.Accept(writer);

  curl_easy_setopt(curl, CURLOPT_URL, request_uri.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, request_data.GetSize());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_data.GetString());
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // Response data handled by ResponseHandler()
  std::string response_string;
  response_string.reserve(256);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

  // Add user provided headers...
  struct curl_slist* header_list = nullptr;
  for (const auto& pr : headers) {
    std::string hdr = pr.first + ": " + pr.second;
    header_list = curl_slist_append(header_list, hdr.c_str());
  }

  if (header_list != nullptr) {
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header_list);
  }

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_slist_free_all(header_list);
    curl_easy_cleanup(curl);
    return Error("HTTP client failed: " + std::string(curl_easy_strerror(res)));
  }

  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, http_code);

  curl_slist_free_all(header_list);
  curl_easy_cleanup(curl);

  if (!response_string.empty()) {
    response->Parse(response_string.c_str(), response_string.size());
    if (response->HasParseError()) {
      return Error(
          "failed to parse the request JSON buffer: " +
          std::string(GetParseError_En(response->GetParseError())) + " at " +
          std::to_string(response->GetErrorOffset()));
    }
    if (verbose_) {
      std::cout << GetJsonText(*response) << std::endl;
    }

    if (response->IsObject()) {
      const auto& itr = response->FindMember("error");
      if (itr != response->MemberEnd()) {
        return Error(itr->value.GetString());
      }
    }
  }

  return Error::Success;
}

//==============================================================================

}}}  // namespace nvidia::inferenceserver::client
