// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/clients/c++/request_http.h"

#include <curl/curl.h>
#include <google/protobuf/text_format.h>
#include "src/clients/c++/request_common.h"

// MSVC equivalent of POSIX call
#ifdef _MSC_VER
#define strncasecmp _strnicmp
#endif

namespace nvidia { namespace inferenceserver { namespace client {

class HttpRequestImpl;
class InferHttpContextImpl;
using ResponseHandlerUserP = std::pair<InferHttpContextImpl*, HttpRequestImpl*>;

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

CurlGlobal::CurlGlobal() : err_(RequestStatusCode::SUCCESS)
{
  if (curl_global_init(CURL_GLOBAL_ALL) != 0) {
    err_ = Error(RequestStatusCode::INTERNAL, "global initialization failed");
  }
}

CurlGlobal::~CurlGlobal()
{
  curl_global_cleanup();
}

static CurlGlobal curl_global;

}  // namespace

//==============================================================================

class ServerHealthHttpContextImpl : public ServerHealthContext {
 public:
  ServerHealthHttpContextImpl(const std::string& url, bool verbose);
  ServerHealthHttpContextImpl(
      const std::string& url, const std::map<std::string, std::string>& headers,
      bool verbose);

  Error GetReady(bool* ready) override;
  Error GetLive(bool* live) override;

 private:
  Error GetHealth(const std::string& url, bool* health);

  // URL for health endpoint on inference server.
  const std::string url_;

  // Custom HTTP headers
  const std::map<std::string, std::string> headers_;

  // Enable verbose output
  const bool verbose_;
};

ServerHealthHttpContextImpl::ServerHealthHttpContextImpl(
    const std::string& url, bool verbose)
    : url_(url + "/" + kHealthRESTEndpoint), verbose_(verbose)
{
}

ServerHealthHttpContextImpl::ServerHealthHttpContextImpl(
    const std::string& url, const std::map<std::string, std::string>& headers,
    bool verbose)
    : url_(url + "/" + kHealthRESTEndpoint), headers_(headers),
      verbose_(verbose)
{
}

Error
ServerHealthHttpContextImpl::GetHealth(const std::string& url, bool* health)
{
  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // Add custom headers...
  struct curl_slist* header_list = nullptr;
  for (const auto& pr : headers_) {
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
    return Error(
        RequestStatusCode::INTERNAL,
        "HTTP client failed: " + std::string(curl_easy_strerror(res)));
  }

  // Must use long with curl_easy_getinfo
  long http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(header_list);
  curl_easy_cleanup(curl);

  *health = (http_code == 200) ? true : false;

  return Error::Success;
}

Error
ServerHealthHttpContextImpl::GetReady(bool* ready)
{
  return GetHealth(url_ + "/ready", ready);
}

Error
ServerHealthHttpContextImpl::GetLive(bool* live)
{
  return GetHealth(url_ + "/live", live);
}

Error
ServerHealthHttpContext::Create(
    std::unique_ptr<ServerHealthContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ServerHealthContext*>(
      new ServerHealthHttpContextImpl(server_url, verbose)));
  return Error::Success;
}

Error
ServerHealthHttpContext::Create(
    std::unique_ptr<ServerHealthContext>* ctx, const std::string& server_url,
    const std::map<std::string, std::string>& headers, bool verbose)
{
  ctx->reset(static_cast<ServerHealthContext*>(
      new ServerHealthHttpContextImpl(server_url, headers, verbose)));
  return Error::Success;
}

//==============================================================================

class ServerStatusHttpContextImpl : public ServerStatusContext {
 public:
  ServerStatusHttpContextImpl(const std::string& url, bool verbose);
  ServerStatusHttpContextImpl(
      const std::string& url, const std::map<std::string, std::string>& headers,
      bool verbose);
  ServerStatusHttpContextImpl(
      const std::string& url, const std::string& model_name, bool verbose);
  ServerStatusHttpContextImpl(
      const std::string& url, const std::map<std::string, std::string>& headers,
      const std::string& model_name, bool verbose);

  Error GetServerStatus(ServerStatus* status) override;

 private:
  static size_t ResponseHeaderHandler(void*, size_t, size_t, void*);
  static size_t ResponseHandler(void*, size_t, size_t, void*);

  // URL for status endpoint on inference server.
  const std::string url_;

  // Custom HTTP headers
  const std::map<std::string, std::string> headers_;

  // Enable verbose output
  const bool verbose_;

  // RequestStatus received in server response
  RequestStatus request_status_;

  // Serialized ServerStatus response from server.
  std::string response_;
};

ServerStatusHttpContextImpl::ServerStatusHttpContextImpl(
    const std::string& url, bool verbose)
    : url_(url + "/" + kStatusRESTEndpoint), verbose_(verbose)
{
}

ServerStatusHttpContextImpl::ServerStatusHttpContextImpl(
    const std::string& url, const std::map<std::string, std::string>& headers,
    bool verbose)
    : url_(url + "/" + kStatusRESTEndpoint), headers_(headers),
      verbose_(verbose)
{
}

ServerStatusHttpContextImpl::ServerStatusHttpContextImpl(
    const std::string& url, const std::string& model_name, bool verbose)
    : url_(url + "/" + kStatusRESTEndpoint + "/" + model_name),
      verbose_(verbose)
{
}

ServerStatusHttpContextImpl::ServerStatusHttpContextImpl(
    const std::string& url, const std::map<std::string, std::string>& headers,
    const std::string& model_name, bool verbose)
    : url_(url + "/" + kStatusRESTEndpoint + "/" + model_name),
      headers_(headers), verbose_(verbose)
{
}

Error
ServerStatusHttpContextImpl::GetServerStatus(ServerStatus* server_status)
{
  server_status->Clear();
  request_status_.Clear();
  response_.clear();

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  // Request binary representation of the status.
  std::string full_url = url_ + "?format=binary";
  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // Response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);

  // Response data handled by ResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);

  // Add custom headers...
  struct curl_slist* header_list = nullptr;
  for (const auto& pr : headers_) {
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
    return Error(
        RequestStatusCode::INTERNAL,
        "HTTP client failed: " + std::string(curl_easy_strerror(res)));
  }

  // Must use long with curl_easy_getinfo
  long http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(header_list);
  curl_easy_cleanup(curl);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("status request did not return status");
  }

  // If request has failing HTTP status or the request's explicit
  // status is not SUCCESS, then signal an error.
  if ((http_code != 200) ||
      (request_status_.code() != RequestStatusCode::SUCCESS)) {
    return Error(request_status_);
  }

  // Parse the response as a ModelConfigList...
  if (!server_status->ParseFromString(response_)) {
    return Error(RequestStatusCode::INTERNAL, "failed to parse server status");
  }

  if (verbose_) {
    std::cout << server_status->DebugString() << std::endl;
  }

  return Error(request_status_);
}

size_t
ServerStatusHttpContextImpl::ResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  ServerStatusHttpContextImpl* ctx =
      reinterpret_cast<ServerStatusHttpContextImpl*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) && !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);

      if (!google::protobuf::TextFormat::ParseFromString(
              hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

size_t
ServerStatusHttpContextImpl::ResponseHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  ServerStatusHttpContextImpl* ctx =
      reinterpret_cast<ServerStatusHttpContextImpl*>(userp);
  uint8_t* buf = reinterpret_cast<uint8_t*>(contents);
  size_t result_bytes = size * nmemb;
  std::copy(buf, buf + result_bytes, std::back_inserter(ctx->response_));
  return result_bytes;
}

Error
ServerStatusHttpContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx, const std::string& server_url,
    bool verbose)
{
  ctx->reset(static_cast<ServerStatusContext*>(
      new ServerStatusHttpContextImpl(server_url, verbose)));
  return Error::Success;
}

Error
ServerStatusHttpContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx, const std::string& server_url,
    const std::map<std::string, std::string>& headers, bool verbose)
{
  ctx->reset(static_cast<ServerStatusContext*>(
      new ServerStatusHttpContextImpl(server_url, headers, verbose)));
  return Error::Success;
}

Error
ServerStatusHttpContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx, const std::string& server_url,
    const std::string& model_name, bool verbose)
{
  ctx->reset(static_cast<ServerStatusContext*>(
      new ServerStatusHttpContextImpl(server_url, model_name, verbose)));
  return Error::Success;
}

Error
ServerStatusHttpContext::Create(
    std::unique_ptr<ServerStatusContext>* ctx, const std::string& server_url,
    const std::map<std::string, std::string>& headers,
    const std::string& model_name, bool verbose)
{
  ctx->reset(static_cast<ServerStatusContext*>(new ServerStatusHttpContextImpl(
      server_url, headers, model_name, verbose)));
  return Error::Success;
}

//==============================================================================

class ProfileHttpContextImpl : public ProfileContext {
 public:
  ProfileHttpContextImpl(
      const std::string& url, const std::map<std::string, std::string>& headers,
      bool verbose);
  Error StartProfile() override;
  Error StopProfile() override;

 private:
  static size_t ResponseHeaderHandler(void*, size_t, size_t, void*);
  Error SendCommand(const std::string& cmd_str);

  // URL for profile endpoint on inference server.
  const std::string url_;

  // Custom HTTP headers
  const std::map<std::string, std::string> headers_;

  // RequestStatus received in server response
  RequestStatus request_status_;

  // Enable verbose output
  const bool verbose_;
};

ProfileHttpContextImpl::ProfileHttpContextImpl(
    const std::string& url, const std::map<std::string, std::string>& headers,
    bool verbose)
    : url_(url + "/" + kProfileRESTEndpoint), headers_(headers),
      verbose_(verbose)
{
}

Error
ProfileHttpContextImpl::StartProfile()
{
  return SendCommand("start");
}

Error
ProfileHttpContextImpl::StopProfile()
{
  return SendCommand("stop");
}

Error
ProfileHttpContextImpl::SendCommand(const std::string& cmd_str)
{
  request_status_.Clear();

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  // Want binary representation of the status.
  std::string full_url = url_ + "?cmd=" + cmd_str;
  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);

  // Add custom headers...
  struct curl_slist* header_list = nullptr;
  for (const auto& pr : headers_) {
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
    return Error(
        RequestStatusCode::INTERNAL,
        "HTTP client failed: " + std::string(curl_easy_strerror(res)));
  }

  // Must use long with curl_easy_getinfo
  long http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(header_list);
  curl_easy_cleanup(curl);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("profile request did not return status");
  }

  return Error(request_status_);
}

size_t
ProfileHttpContextImpl::ResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  ProfileHttpContextImpl* ctx =
      reinterpret_cast<ProfileHttpContextImpl*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) && !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);

      if (!google::protobuf::TextFormat::ParseFromString(
              hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

Error
ProfileHttpContext::Create(
    std::unique_ptr<ProfileContext>* ctx, const std::string& server_url,
    const std::map<std::string, std::string>& headers, bool verbose)
{
  ctx->reset(static_cast<ProfileContext*>(
      new ProfileHttpContextImpl(server_url, headers, verbose)));
  return Error::Success;
}

//==============================================================================

class ModelControlHttpContextImpl : public ModelControlContext {
 public:
  ModelControlHttpContextImpl(
      const std::string& url, const std::map<std::string, std::string>& headers,
      bool verbose);
  Error Load(const std::string& model_name) override;
  Error Unload(const std::string& model_name) override;

 private:
  static size_t ResponseHeaderHandler(void*, size_t, size_t, void*);
  Error SendRequest(
      const std::string& action_str, const std::string& model_name);

  // URL for control endpoint on inference server.
  const std::string url_;

  // Custom HTTP headers
  const std::map<std::string, std::string> headers_;

  // RequestStatus received in server response
  RequestStatus request_status_;

  // Enable verbose output
  const bool verbose_;
};

ModelControlHttpContextImpl::ModelControlHttpContextImpl(
    const std::string& url, const std::map<std::string, std::string>& headers,
    bool verbose)
    : url_(url + "/" + kModelControlRESTEndpoint), headers_(headers),
      verbose_(verbose)
{
}

Error
ModelControlHttpContextImpl::Load(const std::string& model_name)
{
  return SendRequest("load", model_name);
}

Error
ModelControlHttpContextImpl::Unload(const std::string& model_name)
{
  return SendRequest("unload", model_name);
}

Error
ModelControlHttpContextImpl::SendRequest(
    const std::string& action_str, const std::string& model_name)
{
  request_status_.Clear();

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  // Want binary representation of the status.
  std::string full_url = url_ + "/" + action_str + "/" + model_name;
  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  // use POST method
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, "");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);

  // Add custom headers...
  struct curl_slist* header_list = nullptr;
  for (const auto& pr : headers_) {
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
    return Error(
        RequestStatusCode::INTERNAL,
        "HTTP client failed: " + std::string(curl_easy_strerror(res)));
  }

  // Must use long with curl_easy_getinfo
  long http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(header_list);
  curl_easy_cleanup(curl);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("modelcontrol request did not return status");
  }

  return Error(request_status_);
}

size_t
ModelControlHttpContextImpl::ResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  ModelControlHttpContextImpl* ctx =
      reinterpret_cast<ModelControlHttpContextImpl*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) && !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);

      if (!google::protobuf::TextFormat::ParseFromString(
              hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

Error
ModelControlHttpContext::Create(
    std::unique_ptr<ModelControlContext>* ctx, const std::string& server_url,
    const std::map<std::string, std::string>& headers, bool verbose)
{
  ctx->reset(static_cast<ModelControlContext*>(
      new ModelControlHttpContextImpl(server_url, headers, verbose)));
  return Error::Success;
}

//==============================================================================

class SharedMemoryControlHttpContextImpl : public SharedMemoryControlContext {
 public:
  SharedMemoryControlHttpContextImpl(
      const std::string& url, const std::map<std::string, std::string>& headers,
      bool verbose);
  Error RegisterSharedMemory(
      const std::string& name, const std::string& shm_key, const size_t offset,
      const size_t byte_size) override;
  Error UnregisterSharedMemory(const std::string& name) override;
  Error UnregisterAllSharedMemory() override;

 private:
  static size_t ResponseHeaderHandler(void*, size_t, size_t, void*);
  Error SendRequest(
      const std::string& action_str, const std::string& name,
      const std::string& shm_key, const size_t offset, const size_t byte_size);

  // URL for control endpoint on inference server.
  const std::string url_;

  // Custom HTTP headers
  const std::map<std::string, std::string> headers_;

  // RequestStatus received in server response
  RequestStatus request_status_;

  // Enable verbose output
  const bool verbose_;
};

SharedMemoryControlHttpContextImpl::SharedMemoryControlHttpContextImpl(
    const std::string& url, const std::map<std::string, std::string>& headers,
    bool verbose)
    : url_(url + "/" + kSharedMemoryControlRESTEndpoint), headers_(headers),
      verbose_(verbose)
{
}

Error
SharedMemoryControlHttpContextImpl::RegisterSharedMemory(
    const std::string& name, const std::string& shm_key, const size_t offset,
    const size_t byte_size)
{
  return SendRequest("register", name, shm_key, offset, byte_size);
}

Error
SharedMemoryControlHttpContextImpl::UnregisterSharedMemory(
    const std::string& name)
{
  return SendRequest("unregister", name, "", 0, 0);
}

Error
SharedMemoryControlHttpContextImpl::UnregisterAllSharedMemory()
{
  return SendRequest("unregisterall", "", "", 0, 0);
}

Error
SharedMemoryControlHttpContextImpl::SendRequest(
    const std::string& action_str, const std::string& name,
    const std::string& shm_key, const size_t offset, const size_t byte_size)
{
  request_status_.Clear();

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  // For unregisterall and status only action_str is needed
  std::string full_url = url_ + "/" + action_str;
  if (action_str == "register") {
    full_url += +"/" + name + "/" + shm_key + "/" + std::to_string(offset) +
                "/" + std::to_string(byte_size);
  } else if (action_str == "unregister") {
    full_url += +"/" + name;
  }

  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  // use POST method
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, "");
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // response headers handled by ResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, this);

  // Add custom headers...
  struct curl_slist* header_list = nullptr;
  for (const auto& pr : headers_) {
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
    return Error(
        RequestStatusCode::INTERNAL,
        "HTTP client failed: " + std::string(curl_easy_strerror(res)));
  }

  // Must use long with curl_easy_getinfo
  long http_code;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(header_list);
  curl_easy_cleanup(curl);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg(
        "sharedmemorycontrol request did not return status");
  }

  return Error(request_status_);
}

size_t
SharedMemoryControlHttpContextImpl::ResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  SharedMemoryControlHttpContextImpl* ctx =
      reinterpret_cast<SharedMemoryControlHttpContextImpl*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) && !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);

      if (!google::protobuf::TextFormat::ParseFromString(
              hdr, &ctx->request_status_)) {
        ctx->request_status_.Clear();
      }
    }
  }

  return byte_size;
}

Error
SharedMemoryControlHttpContext::Create(
    std::unique_ptr<SharedMemoryControlContext>* ctx,
    const std::string& server_url,
    const std::map<std::string, std::string>& headers, bool verbose)
{
  ctx->reset(static_cast<SharedMemoryControlContext*>(
      new SharedMemoryControlHttpContextImpl(server_url, headers, verbose)));
  return Error::Success;
}

//==============================================================================

class HttpRequestImpl : public RequestImpl {
 public:
  HttpRequestImpl(
      const uint64_t id,
      const std::vector<std::shared_ptr<InferContext::Input>> inputs,
      InferContext::OnCompleteFn callback = nullptr);
  ~HttpRequestImpl();

  // Initialize the request for HTTP transfer. */
  Error InitializeRequest();

  // Copy into 'buf' up to 'size' bytes of input data. Return the
  // actual amount copied in 'input_bytes'.
  Error GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes);

  // Create a result object for this request.
  Error CreateResult(
      const InferHttpContextImpl& ctx,
      const InferResponseHeader::Output& output, const size_t batch_size);

  // Copy into the context 'size' bytes of result data from
  // 'buf'. Return the actual amount copied in 'result_bytes'.
  Error SetNextRawResult(const uint8_t* buf, size_t size, size_t* result_bytes);

  // Get results from an inference request.
  Error GetResults(InferContext::ResultMap* results);

 private:
  friend class InferHttpContextImpl;

  // Pointer to easy handle that is processing the request
  CURL* easy_handle_;

  // Pointer to the list of the HTTP request header, keep it such that it will
  // be valid during the transfer and can be freed once transfer is completed.
  struct curl_slist* header_list_;

  // Status code for the HTTP request.
  CURLcode http_status_;

  // RequestStatus received in server response.
  RequestStatus request_status_;

  // The partial InferResponseHeader delivered via HTTP header.
  InferResponseHeader response_header_;

  // Buffer that accumulates the serialized InferResponseHeader at the
  // end of the body.
  std::string infer_response_buffer_;

  // The inputs for the request. For asynchronous request, it should
  // be a deep copy of the inputs set by the user in case the user modifies
  // them for another request during the HTTP transfer.
  std::vector<std::shared_ptr<InferContext::Input>> inputs_;

  // The total byte size across all the inputs
  uint64_t total_input_byte_size_;

  // Current positions within input vectors when sending request.
  size_t input_pos_idx_;

  // Current positions within output vectors when processing response.
  size_t result_pos_idx_;

  // Callback data for response handler.
  ResponseHandlerUserP response_handler_userp_;

  // The results of this request, in the order indicated by the
  // response header.
  std::vector<std::unique_ptr<ResultImpl>> ordered_results_;
};

//==============================================================================

class InferHttpContextImpl : public InferContextImpl {
 public:
  InferHttpContextImpl(
      const std::string&, const std::map<std::string, std::string>&,
      const std::string&, int64_t, CorrelationID, bool);
  virtual ~InferHttpContextImpl();

  Error InitHttp(const std::string& server_url);

  Error Run(ResultMap* results) override;
  Error AsyncRun(std::shared_ptr<Request>* async_request) override;
  Error AsyncRun(OnCompleteFn callback) override;
  Error GetAsyncRunResults(
      ResultMap* results, bool* is_ready,
      const std::shared_ptr<Request>& async_request, bool wait) override;

 private:
  Error AsyncRun(
      std::shared_ptr<Request>* async_request,
      InferContext::OnCompleteFn callback);
  static size_t RequestProvider(void*, size_t, size_t, void*);
  static size_t ResponseHeaderHandler(void*, size_t, size_t, void*);
  static size_t ResponseHandler(void*, size_t, size_t, void*);

  void AsyncTransfer();
  Error PreRunProcessing(std::shared_ptr<Request>& request);

  // Custom HTTP headers
  const std::map<std::string, std::string> headers_;

  // curl multi handle for processing asynchronous requests
  CURLM* multi_handle_;

  // URL to POST to
  std::string url_;

  // Serialized InferRequestHeader
  std::string infer_request_str_;

  // Keep an easy handle alive to reuse the connection
  CURL* curl_;
};

//==============================================================================

HttpRequestImpl::HttpRequestImpl(
    const uint64_t id,
    const std::vector<std::shared_ptr<InferContext::Input>> inputs,
    InferContext::OnCompleteFn callback)
    : RequestImpl(id, std::move(callback)), easy_handle_(curl_easy_init()),
      header_list_(nullptr), inputs_(inputs), input_pos_idx_(0),
      result_pos_idx_(0)
{
  if (easy_handle_ != nullptr) {
    SetRunIndex(reinterpret_cast<uintptr_t>(easy_handle_));
  }
}

HttpRequestImpl::~HttpRequestImpl()
{
  if (easy_handle_ != nullptr) {
    curl_easy_cleanup(easy_handle_);
  }
}

Error
HttpRequestImpl::InitializeRequest()
{
  ordered_results_.clear();
  infer_response_buffer_.clear();

  request_status_.Clear();
  response_header_.Clear();

  for (auto& io : inputs_) {
    reinterpret_cast<InputImpl*>(io.get())->PrepareForRequest();
  }

  total_input_byte_size_ = 0;
  input_pos_idx_ = 0;
  result_pos_idx_ = 0;

  return Error::Success;
}


Error
HttpRequestImpl::GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes)
{
  *input_bytes = 0;

  while ((size > 0) && (input_pos_idx_ < inputs_.size())) {
    InputImpl* io = reinterpret_cast<InputImpl*>(inputs_[input_pos_idx_].get());
    size_t ib = 0;
    bool eoi = false;
    Error err = io->GetNext(buf, size, &ib, &eoi);
    if (!err.IsOk()) {
      return err;
    }

    // If input was completely read then move to the next.
    if (eoi) {
      input_pos_idx_++;
    }
    if (ib != 0) {
      *input_bytes += ib;
      size -= ib;
      buf += ib;
    }
  }

  // Sent all input bytes
  if (input_pos_idx_ >= inputs_.size()) {
    Timer().Record(RequestTimers::Kind::SEND_END);
  }

  return Error::Success;
}

Error
HttpRequestImpl::SetNextRawResult(
    const uint8_t* buf, size_t size, size_t* result_bytes)
{
  *result_bytes = 0;

  while ((size > 0) && (result_pos_idx_ < ordered_results_.size())) {
    ResultImpl* io = ordered_results_[result_pos_idx_].get();
    size_t ob = 0;

    // Only try to read raw result for RAW
    if (io->ResultFormat() == InferContext::Result::ResultFormat::RAW) {
      Error err = io->SetNextRawResult(buf, size, false /* inplace */, &ob);
      if (!err.IsOk()) {
        return err;
      }
    }

    // std::cerr << "Output size - ob: " << size << " - " << ob << '\n';
    // If output couldn't accept any more bytes then move to the next.
    if (ob == 0 || io->UsesSharedMemory()) {
      result_pos_idx_++;
    } else {
      *result_bytes += ob;
      size -= ob;
      buf += ob;
    }
  }

  // If there is any bytes left then they belong to the response
  // header, since all the RAW results have been filled.
  if (size > 0) {
    infer_response_buffer_.append(reinterpret_cast<const char*>(buf), size);
    *result_bytes += size;
  }

  return Error::Success;
}

Error
HttpRequestImpl::CreateResult(
    const InferHttpContextImpl& ctx, const InferResponseHeader::Output& output,
    const size_t batch_size)
{
  std::shared_ptr<InferContext::Output> infer_output;
  Error err = ctx.GetOutput(output.name(), &infer_output);
  if (!err.IsOk()) {
    return err;
  }

  std::unique_ptr<ResultImpl> result(new ResultImpl(infer_output, batch_size));

  if (!ctx.UsesSharedMemory(output.name())) {
    result->SetBatch1Shape(output.raw().dims());
    if (IsFixedSizeDataType(infer_output->DType())) {
      result->SetBatchnByteSize(output.raw().batch_byte_size());
    }
    result->SetUsesSharedMemory(false);
  } else {
    result->SetUsesSharedMemory(true);
  }

  ordered_results_.emplace_back(std::move(result));

  return Error::Success;
}

Error
HttpRequestImpl::GetResults(InferContext::ResultMap* results)
{
  InferResponseHeader infer_response;

  if (http_status_ != CURLE_OK) {
    curl_slist_free_all(header_list_);
    ordered_results_.clear();
    return Error(
        RequestStatusCode::INTERNAL,
        "HTTP client failed: " + std::string(curl_easy_strerror(http_status_)));
  }

  // Must use long with curl_easy_getinfo
  long http_code;
  curl_easy_getinfo(easy_handle_, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(header_list_);

  // Should have a request status, if not then create an error status.
  if (request_status_.code() == RequestStatusCode::INVALID) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("infer request did not return status");
  }

  // Should have response header from HTTP header, if not then create
  // an error status.
  if ((request_status_.code() == RequestStatusCode::SUCCESS) &&
      response_header_.model_name().empty()) {
    request_status_.Clear();
    request_status_.set_code(RequestStatusCode::INTERNAL);
    request_status_.set_msg("infer request did not return response header");
  }

  // If request has failing HTTP status or the request's explicit
  // status is not SUCCESS, then signal an error.
  if ((http_code != 200) ||
      (request_status_.code() != RequestStatusCode::SUCCESS)) {
    ordered_results_.clear();
    return Error(request_status_);
  }

  // The infer response header should be available...
  if (infer_response_buffer_.empty()) {
    ordered_results_.clear();
    return Error(
        RequestStatusCode::INTERNAL,
        "infer request did not return result header");
  }

  infer_response.ParseFromString(infer_response_buffer_);

  results->clear();
  for (auto& r : ordered_results_) {
    const std::string& name = r->GetOutput()->Name();
    results->insert(std::make_pair(name, std::move(r)));
  }

  PostRunProcessing(infer_response, results);

  return Error(request_status_);
}

//==============================================================================

InferHttpContextImpl::InferHttpContextImpl(
    const std::string& server_url,
    const std::map<std::string, std::string>& headers,
    const std::string& model_name, int64_t model_version,
    CorrelationID correlation_id, bool verbose)
    : InferContextImpl(model_name, model_version, correlation_id, verbose),
      headers_(headers), multi_handle_(curl_multi_init())
{
  // Process url for HTTP request
  // URL doesn't contain the version portion if using the latest version.
  url_ = server_url + "/" + kInferRESTEndpoint + "/" + model_name;
  if (model_version >= 0) {
    url_ += "/" + std::to_string(model_version);
  }
}

InferHttpContextImpl::~InferHttpContextImpl()
{
  exiting_ = true;
  // thread not joinable if AsyncRun() is not called
  // (it is default constructed thread before the first AsyncRun() call)
  if (worker_.joinable()) {
    cv_.notify_all();
    worker_.join();
  }

  if (multi_handle_ != nullptr) {
    for (auto& request : ongoing_async_requests_) {
      CURL* easy_handle =
          std::static_pointer_cast<HttpRequestImpl>(request.second)
              ->easy_handle_;
      // Just remove, easy_cleanup will be done in ~HttpRequestImpl()
      curl_multi_remove_handle(multi_handle_, easy_handle);
    }
    curl_multi_cleanup(multi_handle_);
  }
}

Error
InferHttpContextImpl::InitHttp(const std::string& server_url)
{
  // Don't let user override the request header.
  if (headers_.find(kInferRequestHTTPHeader) != headers_.end()) {
    return Error(
        RequestStatusCode::INVALID_ARG,
        "HTTP header '" + std::string(kInferRequestHTTPHeader) +
            "' cannot be set");
  }

  std::unique_ptr<ServerStatusContext> sctx;
  Error err = ServerStatusHttpContext::Create(
      &sctx, server_url, headers_, model_name_, verbose_);
  if (err.IsOk()) {
    err = Init(std::move(sctx));
    if (err.IsOk()) {
      // Create request context for synchronous request.
      sync_request_.reset(
          static_cast<InferContext::Request*>(new HttpRequestImpl(0, inputs_)));
    }
  }

  return err;
}

Error
InferHttpContextImpl::Run(ResultMap* results)
{
  std::shared_ptr<HttpRequestImpl> sync_request =
      std::static_pointer_cast<HttpRequestImpl>(sync_request_);

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  Error err = PreRunProcessing(sync_request_);

  if (!err.IsOk()) {
    return err;
  }

  // Take run time
  sync_request->Timer().Reset();
  sync_request->Timer().Record(RequestTimers::Kind::REQUEST_START);
  sync_request->Timer().Record(RequestTimers::Kind::SEND_START);
  if (sync_request->total_input_byte_size_ == 0) {
    // Set SEND_END here because CURLOPT_READFUNCTION will not be called if
    // content length is 0. In that case, we can't measure SEND_END properly
    // (send ends after sending request header).
    sync_request->Timer().Record(RequestTimers::Kind::SEND_END);
  }
  sync_request->http_status_ = curl_easy_perform(sync_request->easy_handle_);
  sync_request->Timer().Record(RequestTimers::Kind::RECEIVE_END);
  sync_request->Timer().Record(RequestTimers::Kind::REQUEST_END);

  err = UpdateStat(sync_request->Timer());
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }
  return sync_request->GetResults(results);
}

Error
InferHttpContextImpl::AsyncRun(std::shared_ptr<Request>* async_request)
{
  return AsyncRun(async_request, nullptr);
}

Error
InferHttpContextImpl::AsyncRun(InferContext::OnCompleteFn callback)
{
  std::shared_ptr<Request> async_request;
  return AsyncRun(&async_request, std::move(callback));
}

Error
InferHttpContextImpl::AsyncRun(
    std::shared_ptr<Request>* async_request,
    InferContext::OnCompleteFn callback)
{
  if (!multi_handle_) {
    return Error(
        RequestStatusCode::INTERNAL,
        "failed to start HTTP asynchronous client");
  } else if (!worker_.joinable()) {
    worker_ = std::thread(&InferHttpContextImpl::AsyncTransfer, this);
  }

  // Make a copy of the current inputs
  std::vector<std::shared_ptr<Input>> inputs;
  for (const auto& io : inputs_) {
    InputImpl* input = reinterpret_cast<InputImpl*>(io.get());
    inputs.emplace_back(std::make_shared<InputImpl>(*input));
  }

  HttpRequestImpl* current_context =
      new HttpRequestImpl(0 /* temp id */, inputs, std::move(callback));
  async_request->reset(static_cast<Request*>(current_context));

  if (!current_context->easy_handle_) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  Error err = PreRunProcessing(*async_request);

  {
    std::lock_guard<std::mutex> lock(mutex_);

    current_context->SetId(async_request_id_++);

    auto insert_result = ongoing_async_requests_.emplace(std::make_pair(
        reinterpret_cast<uintptr_t>(current_context->easy_handle_),
        *async_request));

    if (!insert_result.second) {
      return Error(
          RequestStatusCode::INTERNAL,
          "Failed to insert new asynchronous request context.");
    }

    curl_multi_add_handle(multi_handle_, current_context->easy_handle_);
    current_context->Timer().Reset();
    current_context->Timer().Record(RequestTimers::Kind::REQUEST_START);
    current_context->Timer().Record(RequestTimers::Kind::SEND_START);
    if (current_context->total_input_byte_size_ == 0) {
      current_context->Timer().Record(RequestTimers::Kind::SEND_END);
    }
  }

  cv_.notify_all();
  return Error(RequestStatusCode::SUCCESS);
}

Error
InferHttpContextImpl::GetAsyncRunResults(
    ResultMap* results, bool* is_ready,
    const std::shared_ptr<Request>& async_request, bool wait)
{
  Error err = IsRequestReady(async_request, is_ready, wait);
  if (!err.IsOk() || !(*is_ready)) {
    return err;
  }
  std::shared_ptr<HttpRequestImpl> http_request =
      std::static_pointer_cast<HttpRequestImpl>(async_request);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    ongoing_async_requests_.erase(http_request->RunIndex());
    curl_multi_remove_handle(multi_handle_, http_request->easy_handle_);
  }

  err = UpdateStat(http_request->Timer());
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }
  return http_request->GetResults(results);
}

size_t
InferHttpContextImpl::RequestProvider(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpRequestImpl* request = reinterpret_cast<HttpRequestImpl*>(userp);

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
InferHttpContextImpl::ResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  ResponseHandlerUserP* pr = reinterpret_cast<ResponseHandlerUserP*>(userp);
  InferHttpContextImpl* ctx =
      reinterpret_cast<InferHttpContextImpl*>(pr->first);
  HttpRequestImpl* request = reinterpret_cast<HttpRequestImpl*>(pr->second);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;
  size_t idx;

  // Status header
  idx = strlen(kStatusHTTPHeader);
  if ((idx < byte_size) && !strncasecmp(buf, kStatusHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);
      if (!google::protobuf::TextFormat::ParseFromString(
              hdr, &request->request_status_)) {
        request->request_status_.Clear();
      }
    }
  }

  // Response header
  idx = strlen(kInferResponseHTTPHeader);
  if ((idx < byte_size) && !strncasecmp(buf, kInferResponseHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);
      if (!google::protobuf::TextFormat::ParseFromString(
              hdr, &request->response_header_)) {
        request->response_header_.Clear();
      } else {
        for (const auto& output : request->response_header_.output()) {
          Error err = request->CreateResult(
              *ctx, output, request->response_header_.batch_size());
          if (!err.IsOk()) {
            request->response_header_.Clear();
          }
        }
      }
    }
  }

  return byte_size;
}

size_t
InferHttpContextImpl::ResponseHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpRequestImpl* request = reinterpret_cast<HttpRequestImpl*>(userp);
  size_t result_bytes = 0;

  static auto unset_timepoint = RequestTimers::TimePoint();
  if (request->Timer().receive_start_ == unset_timepoint) {
    request->Timer().Record(RequestTimers::Kind::RECEIVE_START);
  }

  Error err = request->SetNextRawResult(
      reinterpret_cast<uint8_t*>(contents), size * nmemb, &result_bytes);
  if (!err.IsOk()) {
    std::cerr << "ResponseHandler: " << err << std::endl;
    return 0;
  }

  return result_bytes;
}

Error
InferHttpContextImpl::PreRunProcessing(std::shared_ptr<Request>& request)
{
  std::shared_ptr<HttpRequestImpl> http_request =
      std::static_pointer_cast<HttpRequestImpl>(request);

  http_request->InitializeRequest();

  CURL* curl = http_request->easy_handle_;
  if (!curl) {
    return Error(
        RequestStatusCode::INTERNAL, "failed to initialize HTTP client");
  }

  std::string full_url = url_ + "?format=binary";
  curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  // request data provided by RequestProvider()
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, RequestProvider);
  curl_easy_setopt(curl, CURLOPT_READDATA, http_request.get());

  // response headers handled by ResponseHeaderHandler()
  http_request->response_handler_userp_ =
      std::make_pair(this, http_request.get());
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ResponseHeaderHandler);
  curl_easy_setopt(
      curl, CURLOPT_HEADERDATA, &http_request->response_handler_userp_);

  // response data handled by ResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, http_request.get());

  // Create the input metadata for the request now that all input
  // sizes are known. For non-fixed-sized datatypes the
  // per-batch-instance byte-size can be different for different input
  // instances in the batch... so set the batch-byte-size to the total
  // size of the batch (see api.proto).
  infer_request_.mutable_input()->Clear();
  infer_request_.set_id(request->Id());
  for (const auto& io : inputs_) {
    http_request->total_input_byte_size_ +=
        reinterpret_cast<InputImpl*>(io.get())->TotalSendByteSize();

    auto rinput = infer_request_.add_input();
    rinput->set_name(io->Name());

    for (const auto s : io->Shape()) {
      rinput->add_dims(s);
    }
    if (!IsFixedSizeDataType(io->DType())) {
      rinput->set_batch_byte_size(io->TotalByteSize());
    }

    // set shared memory
    if (reinterpret_cast<InputImpl*>(io.get())->IsSharedMemory()) {
      auto rshared_memory = rinput->mutable_shared_memory();
      rshared_memory->set_name(
          reinterpret_cast<InputImpl*>(io.get())->GetSharedMemoryName());
      rshared_memory->set_offset(
          reinterpret_cast<InputImpl*>(io.get())->GetSharedMemoryOffset());
      rshared_memory->set_byte_size(io->TotalByteSize());
    }
  }

  // Set the expected POST size. If you want to POST large amounts of
  // data, consider CURLOPT_POSTFIELDSIZE_LARGE
  curl_easy_setopt(
      curl, CURLOPT_POSTFIELDSIZE, http_request->total_input_byte_size_);

  // Headers to specify input and output tensors
  infer_request_str_.clear();
  infer_request_str_ = std::string(kInferRequestHTTPHeader) + ":" +
                       infer_request_.ShortDebugString();
  struct curl_slist* list = nullptr;
  list = curl_slist_append(list, "Expect:");
  list = curl_slist_append(list, "Content-Type: application/octet-stream");
  list = curl_slist_append(list, infer_request_str_.c_str());
  for (const auto& pr : headers_) {
    std::string hdr = pr.first + ": " + pr.second;
    list = curl_slist_append(list, hdr.c_str());
  }
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

  // The list should be freed after the request
  http_request->header_list_ = list;

  return Error::Success;
}

void
InferHttpContextImpl::AsyncTransfer()
{
  int place_holder = 0;
  CURLMsg* msg = nullptr;
  do {
    std::vector<std::shared_ptr<Request>> request_with_callback;
    bool has_completed = false;
    // sleep if no work is available
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
      if (this->exiting_) {
        return true;
      }
      // wake up if at least one request is not ready
      for (auto& ongoing_async_request : this->ongoing_async_requests_) {
        if (std::static_pointer_cast<HttpRequestImpl>(
                ongoing_async_request.second)
                ->IsReady() == false) {
          return true;
        }
      }
      return false;
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
      std::shared_ptr<HttpRequestImpl> http_request =
          std::static_pointer_cast<HttpRequestImpl>(itr->second);

      if (msg->msg != CURLMSG_DONE) {
        // Something wrong happened.
        fprintf(stderr, "Unexpected error: received CURLMsg=%d\n", msg->msg);
      } else {
        http_request->Timer().Record(RequestTimers::Kind::RECEIVE_END);
        http_request->Timer().Record(RequestTimers::Kind::REQUEST_END);
      }
      http_request->http_status_ = msg->data.result;
      http_request->SetIsReady(true);

      has_completed = true;
      if (http_request->HasCallback()) {
        request_with_callback.emplace_back(itr->second);
      }
    }
    lock.unlock();
    // if it has completed tasks, send signal in case the main thread is waiting
    if (has_completed) {
      cv_.notify_all();
    }
    for (auto& request : request_with_callback) {
      HttpRequestImpl* request_ptr =
          static_cast<HttpRequestImpl*>(request.get());
      request_ptr->callback_(this, std::move(request));
    }
  } while (!exiting_);
}

Error
InferHttpContext::Create(
    std::unique_ptr<InferContext>* ctx, const std::string& server_url,
    const std::string& model_name, int64_t model_version, bool verbose)
{
  std::map<std::string, std::string> headers;
  return Create(
      ctx, 0 /* correlation_id */, server_url, headers, model_name,
      model_version, verbose);
}

Error
InferHttpContext::Create(
    std::unique_ptr<InferContext>* ctx, const std::string& server_url,
    const std::map<std::string, std::string>& headers,
    const std::string& model_name, int64_t model_version, bool verbose)
{
  return Create(
      ctx, 0 /* correlation_id */, server_url, headers, model_name,
      model_version, verbose);
}

Error
InferHttpContext::Create(
    std::unique_ptr<InferContext>* ctx, CorrelationID correlation_id,
    const std::string& server_url, const std::string& model_name,
    int64_t model_version, bool verbose)
{
  std::map<std::string, std::string> headers;
  return Create(
      ctx, correlation_id, server_url, headers, model_name, model_version,
      verbose);
}

Error
InferHttpContext::Create(
    std::unique_ptr<InferContext>* ctx, CorrelationID correlation_id,
    const std::string& server_url,
    const std::map<std::string, std::string>& headers,
    const std::string& model_name, int64_t model_version, bool verbose)
{
  InferHttpContextImpl* ctx_ptr = new InferHttpContextImpl(
      server_url, headers, model_name, model_version, correlation_id, verbose);
  ctx->reset(static_cast<InferContext*>(ctx_ptr));

  Error err = ctx_ptr->InitHttp(server_url);
  if (!err.IsOk()) {
    ctx->reset();
  }

  return err;
}

}}}  // namespace nvidia::inferenceserver::client
