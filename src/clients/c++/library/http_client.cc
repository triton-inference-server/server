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

#define TRITONJSON_STATUSTYPE nvidia::inferenceserver::client::Error
#define TRITONJSON_STATUSRETURN(M) \
  return nvidia::inferenceserver::client::Error(M)
#define TRITONJSON_STATUSSUCCESS nvidia::inferenceserver::client::Error::Success
#include "src/core/json.h"

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

class HttpInferRequest : public InferRequest {
 public:
  HttpInferRequest(
      InferenceServerClient::OnCompleteFn callback = nullptr,
      const bool verbose = false);
  ~HttpInferRequest();

  // Initialize the request for HTTP transfer. */
  Error InitializeRequest(
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs);

  // Adds the input data to be delivered to the server
  Error AddInput(uint8_t* buf, size_t byte_size);

  // Copy into 'buf' up to 'size' bytes of input data. Return the
  // actual amount copied in 'input_bytes'.
  Error GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes);

 private:
  friend class InferenceServerHttpClient;
  friend class InferResultHttp;

  Error PrepareRequestJson(
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs,
      TritonJson::Value* request_json);

  // Pointer to the list of the HTTP request header, keep it such that it will
  // be valid during the transfer and can be freed once transfer is completed.
  struct curl_slist* header_list_;

  // HTTP response code for the inference request
  long http_code_;

  size_t total_input_byte_size_;

  TritonJson::WriteBuffer request_json_;

  // Buffer that accumulates the response body.
  std::unique_ptr<std::string> infer_response_buffer_;

  // The pointers to the input data.
  std::queue<std::pair<uint8_t*, size_t>> data_buffers_;

  size_t response_json_size_;
};


HttpInferRequest::HttpInferRequest(
    InferenceServerClient::OnCompleteFn callback, const bool verbose)
    : InferRequest(callback, verbose), header_list_(nullptr),
      total_input_byte_size_(0), response_json_size_(0)
{
}

HttpInferRequest::~HttpInferRequest()
{
  if (header_list_ != nullptr) {
    curl_slist_free_all(header_list_);
    header_list_ = nullptr;
  }
}

Error
HttpInferRequest::InitializeRequest(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  data_buffers_ = {};
  total_input_byte_size_ = 0;
  http_code_ = 400;

  TritonJson::Value request_json(TritonJson::ValueType::OBJECT);
  Error err = PrepareRequestJson(options, inputs, outputs, &request_json);
  if (!err.IsOk()) {
    return err;
  }

  request_json_.Clear();
  request_json.Write(&request_json_);

  // Add the buffer holding the json to be delivered first
  AddInput((uint8_t*)request_json_.Base(), request_json_.Size());

  // Prepare buffer to record the response
  infer_response_buffer_.reset(new std::string());

  return Error::Success;
}

Error
HttpInferRequest::PrepareRequestJson(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    TritonJson::Value* request_json)
{
  // Can use string-ref because json is serialized before end of
  // 'options', 'inputs' and 'outputs' lifetime.
  request_json->AddStringRef(
      "id", options.request_id_.c_str(), options.request_id_.size());

  if ((options.sequence_id_ != 0) || (options.priority_ != 0) ||
      (options.timeout_ != 0)) {
    TritonJson::Value parameters_json(
        *request_json, TritonJson::ValueType::OBJECT);
    {
      if (options.sequence_id_ != 0) {
        parameters_json.AddUInt("sequence_id", options.sequence_id_);
        parameters_json.AddBool("sequence_start", options.sequence_start_);
        parameters_json.AddBool("sequence_end", options.sequence_end_);
      }

      if (options.priority_ != 0) {
        parameters_json.AddUInt("priority", options.priority_);
      }

      if (options.timeout_ != 0) {
        parameters_json.AddUInt("timeout", options.timeout_);
      }
    }

    request_json->Add("parameters", std::move(parameters_json));
  }

  if (!inputs.empty()) {
    TritonJson::Value inputs_json(*request_json, TritonJson::ValueType::ARRAY);
    for (const auto io : inputs) {
      TritonJson::Value io_json(*request_json, TritonJson::ValueType::OBJECT);
      io_json.AddStringRef("name", io->Name().c_str(), io->Name().size());
      io_json.AddStringRef(
          "datatype", io->Datatype().c_str(), io->Datatype().size());

      TritonJson::Value shape_json(*request_json, TritonJson::ValueType::ARRAY);
      for (const auto dim : io->Shape()) {
        shape_json.AppendUInt(dim);
      }
      io_json.Add("shape", std::move(shape_json));

      TritonJson::Value ioparams_json(
          *request_json, TritonJson::ValueType::OBJECT);
      if (io->IsSharedMemory()) {
        std::string region_name;
        size_t offset;
        size_t byte_size;
        Error err = io->SharedMemoryInfo(&region_name, &byte_size, &offset);
        if (!err.IsOk()) {
          return err;
        }

        ioparams_json.AddString(
            "shared_memory_region", region_name.c_str(), region_name.size());
        ioparams_json.AddUInt("shared_memory_byte_size", byte_size);
        if (offset != 0) {
          ioparams_json.AddUInt("shared_memory_offset", offset);
        }
      } else {
        size_t byte_size;
        Error err = io->ByteSize(&byte_size);
        if (!err.IsOk()) {
          return err;
        }

        ioparams_json.AddUInt("binary_data_size", byte_size);
      }

      io_json.Add("parameters", std::move(ioparams_json));
      inputs_json.Append(std::move(io_json));
    }

    request_json->Add("inputs", std::move(inputs_json));
  }

  if (!outputs.empty()) {
    TritonJson::Value outputs_json(*request_json, TritonJson::ValueType::ARRAY);
    for (const auto io : outputs) {
      TritonJson::Value io_json(*request_json, TritonJson::ValueType::OBJECT);
      io_json.AddStringRef("name", io->Name().c_str(), io->Name().size());

      TritonJson::Value ioparams_json(
          *request_json, TritonJson::ValueType::OBJECT);

      if (io->ClassificationCount() > 0) {
        ioparams_json.AddUInt("classification", io->ClassificationCount());
      }

      if (io->IsSharedMemory()) {
        std::string region_name;
        size_t offset;
        size_t byte_size;
        Error err = io->SharedMemoryInfo(&region_name, &byte_size, &offset);
        if (!err.IsOk()) {
          return err;
        }

        ioparams_json.AddString(
            "shared_memory_region", region_name.c_str(), region_name.size());
        ioparams_json.AddUInt("shared_memory_byte_size", byte_size);
        if (offset != 0) {
          ioparams_json.AddUInt("shared_memory_offset", offset);
        }
      } else {
        ioparams_json.AddBool("binary_data", true);
      }

      io_json.Add("parameters", std::move(ioparams_json));
      outputs_json.Append(std::move(io_json));
    }

    request_json->Add("outputs", std::move(outputs_json));
  } else {
    return Error(
        "request should include at least one InferRequestedOutput object");
  }

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
  static void Create(
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

  std::map<std::string, TritonJson::Value> output_name_to_result_map_;
  std::map<std::string, std::pair<const uint8_t*, const size_t>>
      output_name_to_buffer_map_;

  Error status_;
  TritonJson::Value response_json_;
  std::shared_ptr<HttpInferRequest> infer_request_;
};

void
InferResultHttp::Create(
    InferResult** infer_result, std::shared_ptr<HttpInferRequest> infer_request)
{
  *infer_result =
      reinterpret_cast<InferResult*>(new InferResultHttp(infer_request));
}

Error
InferResultHttp::ModelName(std::string* name) const
{
  if (!status_.IsOk()) {
    return status_;
  }

  const char* name_str;
  size_t name_strlen;
  Error err =
      response_json_.MemberAsString("model_name", &name_str, &name_strlen);
  if (!err.IsOk()) {
    return Error("model name was not returned in the response");
  }

  name->assign(name_str, name_strlen);
  return Error::Success;
}

Error
InferResultHttp::ModelVersion(std::string* version) const
{
  if (!status_.IsOk()) {
    return status_;
  }

  const char* version_str;
  size_t version_strlen;
  Error err = response_json_.MemberAsString(
      "model_version", &version_str, &version_strlen);
  if (!err.IsOk()) {
    return Error("model version was not returned in the response");
  }

  version->assign(version_str, version_strlen);
  return Error::Success;
}

Error
InferResultHttp::Id(std::string* id) const
{
  if (!status_.IsOk()) {
    return status_;
  }

  const char* id_str;
  size_t id_strlen;
  Error err = response_json_.MemberAsString("id", &id_str, &id_strlen);
  if (!err.IsOk()) {
    return Error("model id was not returned in the response");
  }

  id->assign(id_str, id_strlen);
  return Error::Success;
}

namespace {

Error
ShapeHelper(
    const std::string& result_name, const TritonJson::Value& result_json,
    std::vector<int64_t>* shape)
{
  TritonJson::Value shape_json;
  if (!const_cast<TritonJson::Value&>(result_json).Find("shape", &shape_json)) {
    return Error(
        "The response does not contain shape for output name " + result_name);
  }

  for (size_t i = 0; i < shape_json.ArraySize(); i++) {
    int64_t dim;
    Error err = shape_json.IndexAsInt(i, &dim);
    if (!err.IsOk()) {
      return err;
    }

    shape->push_back(dim);
  }

  return Error::Success;
}

}  // namespace

Error
InferResultHttp::Shape(
    const std::string& output_name, std::vector<int64_t>* shape) const
{
  if (!status_.IsOk()) {
    return status_;
  }

  shape->clear();
  auto itr = output_name_to_result_map_.find(output_name);
  if (itr == output_name_to_result_map_.end()) {
    return Error(
        "The response does not contain results for output name " + output_name);
  }

  return ShapeHelper(output_name, itr->second, shape);
}

Error
InferResultHttp::Datatype(
    const std::string& output_name, std::string* datatype) const
{
  if (!status_.IsOk()) {
    return status_;
  }
  auto itr = output_name_to_result_map_.find(output_name);
  if (itr == output_name_to_result_map_.end()) {
    return Error(
        "The response does not contain results for output name " + output_name);
  }

  const char* dtype_str;
  size_t dtype_strlen;
  Error err = itr->second.MemberAsString("datatype", &dtype_str, &dtype_strlen);
  if (!err.IsOk()) {
    return Error(
        "The response does not contain datatype for output name " +
        output_name);
  }

  datatype->assign(dtype_str, dtype_strlen);
  return Error::Success;
}

Error
InferResultHttp::RawData(
    const std::string& output_name, const uint8_t** buf,
    size_t* byte_size) const
{
  if (!status_.IsOk()) {
    return status_;
  }
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
  if (!status_.IsOk()) {
    return status_;
  }
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
  if (!status_.IsOk()) {
    return status_.Message();
  }

  TritonJson::WriteBuffer buffer;
  Error err = response_json_.Write(&buffer);
  if (!err.IsOk()) {
    return "<failed>";
  }

  return buffer.Contents();
}

Error
InferResultHttp::RequestStatus() const
{
  return status_;
}

InferResultHttp::InferResultHttp(
    std::shared_ptr<HttpInferRequest> infer_request)
    : infer_request_(infer_request)
{
  size_t offset = infer_request->response_json_size_;
  if (offset != 0) {
    if (infer_request->verbose_) {
      std::cout << "inference response: "
                << infer_request->infer_response_buffer_->substr(0, offset)
                << std::endl;
    }
    status_ = response_json_.Parse(
        (char*)infer_request->infer_response_buffer_.get()->c_str(), offset);
  } else {
    if (infer_request->verbose_) {
      std::cout << "inference response: "
                << *infer_request->infer_response_buffer_ << std::endl;
    }
    status_ = response_json_.Parse(
        (char*)infer_request->infer_response_buffer_.get()->c_str());
  }

  // There should be a valid JSON response in all cases. Either the
  // successful infer response or an error response.
  if (status_.IsOk()) {
    if (infer_request->http_code_ != 200) {
      const char* err_str;
      size_t err_strlen;
      if (!response_json_.MemberAsString("error", &err_str, &err_strlen)
               .IsOk()) {
        status_ = Error("inference failed with unknown error");
      } else {
        status_ = Error(std::string(err_str, err_strlen));
      }
    } else {
      TritonJson::Value outputs_json;
      if (response_json_.Find("outputs", &outputs_json)) {
        for (size_t i = 0; i < outputs_json.ArraySize(); i++) {
          TritonJson::Value output_json;
          status_ = outputs_json.IndexAsObject(i, &output_json);
          if (!status_.IsOk()) {
            break;
          }

          const char* name_str;
          size_t name_strlen;
          status_ = output_json.MemberAsString("name", &name_str, &name_strlen);
          if (!status_.IsOk()) {
            break;
          }

          std::string output_name(name_str, name_strlen);

          TritonJson::Value param_json;
          if (output_json.Find("parameters", &param_json)) {
            uint64_t data_size = 0;
            status_ = param_json.MemberAsUInt("binary_data_size", &data_size);
            if (!status_.IsOk()) {
              break;
            }

            output_name_to_buffer_map_.emplace(
                output_name,
                std::pair<const uint8_t*, const size_t>(
                    (uint8_t*)(infer_request->infer_response_buffer_.get()
                                   ->c_str()) +
                        offset,
                    data_size));
            offset += data_size;
          }

          output_name_to_result_map_[output_name] = std::move(output_json);
        }
      }
    }
  }
}

//==============================================================================

Error
InferenceServerHttpClient::Create(
    std::unique_ptr<InferenceServerHttpClient>* client,
    const std::string& server_url, bool verbose)
{
  client->reset(new InferenceServerHttpClient(server_url, verbose));
  return Error::Success;
}

InferenceServerHttpClient::InferenceServerHttpClient(
    const std::string& url, bool verbose)
    : InferenceServerClient(verbose), url_(url),
      easy_handle_(reinterpret_cast<void*>(curl_easy_init())),
      multi_handle_(curl_multi_init())
{
}

InferenceServerHttpClient::~InferenceServerHttpClient()
{
  exiting_ = true;

  // thread not joinable if AsyncInfer() is not called
  // (it is default constructed thread before the first AsyncInfer() call)
  if (worker_.joinable()) {
    cv_.notify_all();
    worker_.join();
  }

  if (easy_handle_ != nullptr) {
    curl_easy_cleanup(reinterpret_cast<CURL*>(easy_handle_));
  }

  if (multi_handle_ != nullptr) {
    for (auto& request : ongoing_async_requests_) {
      CURL* easy_handle = reinterpret_cast<CURL*>(request.first);
      curl_multi_remove_handle(multi_handle_, easy_handle);
      curl_easy_cleanup(easy_handle);
    }
    curl_multi_cleanup(multi_handle_);
  }
}

Error
InferenceServerHttpClient::IsServerLive(
    bool* live, const Headers& headers, const Parameters& query_params)
{
  Error err;

  std::string request_uri(url_ + "/v2/health/live");

  long http_code;
  std::string response;
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
  std::string response;
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
  std::string response;
  err = Get(request_uri, headers, query_params, &response, &http_code);

  *ready = (http_code == 200) ? true : false;

  return err;
}


Error
InferenceServerHttpClient::ServerMetadata(
    std::string* server_metadata, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(url_ + "/v2");
  return Get(request_uri, headers, query_params, server_metadata);
}


Error
InferenceServerHttpClient::ModelMetadata(
    std::string* model_metadata, const std::string& model_name,
    const std::string& model_version, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(url_ + "/v2/models/" + model_name);
  if (!model_version.empty()) {
    request_uri = request_uri + "/versions/" + model_version;
  }

  return Get(request_uri, headers, query_params, model_metadata);
}


Error
InferenceServerHttpClient::ModelConfig(
    std::string* model_config, const std::string& model_name,
    const std::string& model_version, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(url_ + "/v2/models/" + model_name);
  if (!model_version.empty()) {
    request_uri = request_uri + "/versions/" + model_version;
  }
  request_uri = request_uri + "/config";

  return Get(request_uri, headers, query_params, model_config);
}


Error
InferenceServerHttpClient::ModelRepositoryIndex(
    std::string* repository_index, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(url_ + "/v2/repository/index");

  std::string request;  // empty request body
  return Post(request_uri, request, headers, query_params, repository_index);
}

Error
InferenceServerHttpClient::LoadModel(
    const std::string& model_name, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(
      url_ + "/v2/repository/models/" + model_name + "/load");

  std::string request;  // empty request body
  std::string response;
  return Post(request_uri, request, headers, query_params, &response);
}

Error
InferenceServerHttpClient::UnloadModel(
    const std::string& model_name, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(
      url_ + "/v2/repository/models/" + model_name + "/unload");

  std::string request;  // empty request body
  std::string response;
  return Post(request_uri, request, headers, query_params, &response);
}


Error
InferenceServerHttpClient::ModelInferenceStatistics(
    std::string* infer_stat, const std::string& model_name,
    const std::string& model_version, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(url_ + "/v2/models");
  if (!model_name.empty()) {
    request_uri += "/" + model_name;
  }
  if (!model_version.empty()) {
    request_uri += "/versions/" + model_version;
  }
  request_uri += "/stats";

  return Get(request_uri, headers, query_params, infer_stat);
}

Error
InferenceServerHttpClient::SystemSharedMemoryStatus(
    std::string* status, const std::string& name, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(url_ + "/v2/systemsharedmemory");
  if (!name.empty()) {
    request_uri = request_uri + "/region/" + name;
  }
  request_uri = request_uri + "/status";

  return Get(request_uri, headers, query_params, status);
}

Error
InferenceServerHttpClient::RegisterSystemSharedMemory(
    const std::string& name, const std::string& key, const size_t byte_size,
    const size_t offset, const Headers& headers, const Parameters& query_params)
{
  std::string request_uri(
      url_ + "/v2/systemsharedmemory/region/" + name + "/register");

  TritonJson::Value request_json(TritonJson::ValueType::OBJECT);
  {
    request_json.AddStringRef("key", key.c_str(), key.size());
    request_json.AddUInt("offset", offset);
    request_json.AddUInt("byte_size", byte_size);
  }

  TritonJson::WriteBuffer buffer;
  Error err = request_json.Write(&buffer);
  if (!err.IsOk()) {
    return err;
  }

  std::string response;
  return Post(request_uri, buffer.Contents(), headers, query_params, &response);
}

Error
InferenceServerHttpClient::UnregisterSystemSharedMemory(
    const std::string& region_name, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(url_ + "/v2/systemsharedmemory");
  if (!region_name.empty()) {
    request_uri = request_uri + "/region/" + region_name;
  }
  request_uri = request_uri + "/unregister";

  std::string request;  // empty request body
  std::string response;
  return Post(request_uri, request, headers, query_params, &response);
}

Error
InferenceServerHttpClient::CudaSharedMemoryStatus(
    std::string* status, const std::string& region_name, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(url_ + "/v2/cudasharedmemory");
  if (!region_name.empty()) {
    request_uri = request_uri + "/region/" + region_name;
  }
  request_uri = request_uri + "/status";

  return Get(request_uri, headers, query_params, status);
}

Error
InferenceServerHttpClient::RegisterCudaSharedMemory(
    const std::string& name, const cudaIpcMemHandle_t& raw_handle,
    const size_t device_id, const size_t byte_size, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(
      url_ + "/v2/cudasharedmemory/region/" + name + "/register");

  TritonJson::Value request_json(TritonJson::ValueType::OBJECT);
  {
    TritonJson::Value raw_handle_json(
        request_json, TritonJson::ValueType::OBJECT);
    {
      char* encoded_handle = nullptr;
      int encoded_size;
      Base64Encode(
          (char*)((void*)&raw_handle), sizeof(cudaIpcMemHandle_t),
          &encoded_handle, &encoded_size);
      if (encoded_handle == nullptr) {
        return Error("Failed to base64 encode the cudaIpcMemHandle_t");
      }

      raw_handle_json.AddString("b64", encoded_handle, encoded_size);
      delete encoded_handle;
    }
    request_json.Add("raw_handle", std::move(raw_handle_json));
    request_json.AddUInt("device_id", device_id);
    request_json.AddUInt("byte_size", byte_size);
  }

  TritonJson::WriteBuffer buffer;
  Error err = request_json.Write(&buffer);
  if (!err.IsOk()) {
    return err;
  }

  std::string response;
  return Post(request_uri, buffer.Contents(), headers, query_params, &response);
}

Error
InferenceServerHttpClient::UnregisterCudaSharedMemory(
    const std::string& name, const Headers& headers,
    const Parameters& query_params)
{
  std::string request_uri(url_ + "/v2/cudasharedmemory");
  if (!name.empty()) {
    request_uri = request_uri + "/region/" + name;
  }
  request_uri = request_uri + "/unregister";

  std::string request;  // empty request body
  std::string response;
  return Post(request_uri, request, headers, query_params, &response);
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

  std::shared_ptr<HttpInferRequest> sync_request(
      new HttpInferRequest(nullptr /* callback */, verbose_));

  sync_request->Timer().Reset();
  sync_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_START);

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  err = PreRunProcessing(
      easy_handle_, request_uri, options, inputs, outputs, headers,
      query_params, sync_request);
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
  if (curl_easy_perform(easy_handle_) != CURLE_OK) {
    sync_request->http_code_ = 400;
  } else {
    curl_easy_getinfo(
        easy_handle_, CURLINFO_RESPONSE_CODE, &sync_request->http_code_);
  }

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
      new HttpInferRequest(std::move(callback), verbose_);
  async_request.reset(raw_async_request);

  async_request->Timer().CaptureTimestamp(RequestTimers::Kind::REQUEST_START);

  CURL* multi_easy_handle = curl_easy_init();
  Error err = PreRunProcessing(
      reinterpret_cast<void*>(multi_easy_handle), request_uri, options, inputs,
      outputs, headers, query_params, async_request);
  if (!err.IsOk()) {
    curl_easy_cleanup(multi_easy_handle);
    return err;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);

    auto insert_result = ongoing_async_requests_.emplace(std::make_pair(
        reinterpret_cast<uintptr_t>(multi_easy_handle), async_request));
    if (!insert_result.second) {
      curl_easy_cleanup(multi_easy_handle);
      return Error("Failed to insert new asynchronous request context.");
    }

    async_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_START);
    if (async_request->total_input_byte_size_ == 0) {
      // Set SEND_END here because CURLOPT_READFUNCTION will not be called if
      // content length is 0. In that case, we can't measure SEND_END properly
      // (send ends after sending request header).
      async_request->Timer().CaptureTimestamp(RequestTimers::Kind::SEND_END);
    }

    curl_multi_add_handle(multi_handle_, multi_easy_handle);
  }

  cv_.notify_all();
  return Error::Success;
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

Error
InferenceServerHttpClient::PreRunProcessing(
    void* vcurl, std::string& request_uri, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers, const Parameters& query_params,
    std::shared_ptr<HttpInferRequest>& http_request)
{
  CURL* curl = reinterpret_cast<CURL*>(vcurl);

  // Prepare the request object to provide the data for inference.
  Error err = http_request->InitializeRequest(options, inputs, outputs);
  if (!err.IsOk()) {
    return err;
  }

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
                        std::to_string(http_request->request_json_.Size())};
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

  if (verbose_) {
    std::cout << "inference request: " << http_request->request_json_.Contents()
              << std::endl;
  }

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
      uintptr_t identifier = reinterpret_cast<uintptr_t>(msg->easy_handle);
      auto itr = ongoing_async_requests_.find(identifier);
      // This shouldn't happen
      if (itr == ongoing_async_requests_.end()) {
        std::cerr << "Unexpected error: received completed request that is not "
                     "in the list of asynchronous requests"
                  << std::endl;
        curl_multi_remove_handle(multi_handle_, msg->easy_handle);
        curl_easy_cleanup(msg->easy_handle);
        continue;
      }

      long http_code = 400;
      if (msg->data.result == CURLE_OK) {
        curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &http_code);
      }

      request_list.emplace_back(itr->second);
      ongoing_async_requests_.erase(itr);
      curl_multi_remove_handle(multi_handle_, msg->easy_handle);
      curl_easy_cleanup(msg->easy_handle);

      std::shared_ptr<HttpInferRequest> async_request = request_list.back();
      async_request->http_code_ = http_code;

      if (msg->msg != CURLMSG_DONE) {
        // Something wrong happened.
        std::cerr << "Unexpected error: received CURLMsg=" << msg->msg
                  << std::endl;
      } else {
        async_request->Timer().CaptureTimestamp(
            RequestTimers::Kind::REQUEST_END);
        Error err = UpdateInferStat(async_request->Timer());
        if (!err.IsOk()) {
          std::cerr << "Failed to update context stat: " << err << std::endl;
        }
      }
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

namespace {
Error
ParseErrorJson(const std::string& json_str)
{
  TritonJson::Value json;
  Error err = json.Parse(json_str.c_str(), json_str.size());
  if (!err.IsOk()) {
    return err;
  }

  const char* errstr;
  size_t errlen;
  err = json.MemberAsString("error", &errstr, &errlen);
  if (!err.IsOk()) {
    return err;
  }

  return Error(std::move(std::string(errstr, errlen)));
}

}  // namespace

Error
InferenceServerHttpClient::Get(
    std::string& request_uri, const Headers& headers,
    const Parameters& query_params, std::string* response, long* http_code)
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
  curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // Response data handled by ResponseHandler()
  response->clear();
  response->reserve(1024);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);

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

  long lhttp_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &lhttp_code);

  curl_slist_free_all(header_list);
  curl_easy_cleanup(curl);

  if (verbose_) {
    std::cout << *response << std::endl;
  }

  // If http code was requested for return, then just return it,
  // otherwise flag an error if the http code is not 200.
  if (http_code != nullptr) {
    *http_code = lhttp_code;
  } else if (lhttp_code != 200) {
    return ParseErrorJson(*response);
  }

  return Error::Success;
}

Error
InferenceServerHttpClient::Post(
    std::string& request_uri, const std::string& request,
    const Headers& headers, const Parameters& query_params,
    std::string* response)
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
  curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, request.size());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.c_str());
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // Response data handled by ResponseHandler()
  response->clear();
  response->reserve(1024);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);

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

  long http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(header_list);
  curl_easy_cleanup(curl);

  if (verbose_) {
    std::cout << *response << std::endl;
  }

  if (http_code != 200) {
    return ParseErrorJson(*response);
  }

  return Error::Success;
}

//==============================================================================

}}}  // namespace nvidia::inferenceserver::client
