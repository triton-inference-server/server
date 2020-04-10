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

#define DLL_EXPORTING

#include "src/clients/c++/experimental_api_v2/library/http_client.h"

#include <curl/curl.h>
#include <cstdint>
#include <iostream>

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

}  // namespace

//==============================================================================

std::string
GetJsonText(rapidjson::Document& json_dom)
{
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  json_dom.Accept(writer);
  return buffer.GetString();
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
InferenceServerHttpClient::GetServerMetadata(
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
InferenceServerHttpClient::GetModelMetadata(
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
InferenceServerHttpClient::GetModelConfig(
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

InferenceServerHttpClient::InferenceServerHttpClient(
    const std::string& url, bool verbose)
    : url_(url), verbose_(verbose)
{
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

    const auto& itr = response->FindMember("error");
    if (itr != response->MemberEnd()) {
      return Error(itr->value.GetString());
    }
  }

  return Error::Success;
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

}}}  // namespace nvidia::inferenceserver::client
