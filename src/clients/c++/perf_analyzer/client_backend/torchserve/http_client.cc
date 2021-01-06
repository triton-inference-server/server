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

#include "src/clients/c++/perf_analyzer/client_backend/torchserve/http_client.h"
#include "src/clients/c++/perf_analyzer/client_backend/torchserve/torchserve_client_backend.h"

#include <chrono>
#include <cstdint>

namespace perfanalyzer { namespace clientbackend { namespace torchserve {

namespace {

constexpr char kContentLengthHTTPHeader[] = "Content-Length";

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


}  // namespace

//==============================================================================

HttpInferRequest::HttpInferRequest()
    : header_list_(nullptr),
      file_ptr_(std::unique_ptr<FILE, Deleter>(nullptr, Deleter()))
{
}

HttpInferRequest::~HttpInferRequest()
{
  if (header_list_ != nullptr) {
    curl_slist_free_all(static_cast<curl_slist*>(header_list_));
    header_list_ = nullptr;
  }
}

Error
HttpInferRequest::InitializeRequest()
{
  http_code_ = 400;
  // Prepare buffer to record the response
  infer_response_buffer_.reset(new std::string());
  return Error::Success;
}

Error
HttpInferRequest::OpenFileData(std::string& file_path)
{
  FILE* pFile = fopen(file_path.c_str(), "rb");
  if (pFile == nullptr) {
    return Error("Failed to open the specified file `" + file_path + "`");
  }
  file_ptr_.reset(pFile);
  return Error::Success;
}

long
HttpInferRequest::FileSize()
{
  long size;
  fseek(file_ptr_.get(), 0, SEEK_END);
  size = ftell(file_ptr_.get());
  rewind(file_ptr_.get());
  return size;
}

Error
HttpInferRequest::CloseFileData()
{
  fclose(file_ptr_.get());
  file_ptr_.reset(nullptr);
  return Error::Success;
}


//==============================================================================

Error
HttpClient::Create(
    std::unique_ptr<HttpClient>* client, const std::string& server_url,
    bool verbose)
{
  client->reset(new HttpClient(server_url, verbose));
  return Error::Success;
}

Error
HttpClient::Infer(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers)
{
  Error err;

  std::string request_uri(url_ + "/predictions/" + options.model_name_);
  if (!options.model_version_.empty()) {
    request_uri += "/" + options.model_version_;
  }

  std::shared_ptr<HttpInferRequest> sync_request(new HttpInferRequest());

  sync_request->Timer().Reset();
  sync_request->Timer().CaptureTimestamp(
      nic::RequestTimers::Kind::REQUEST_START);

  if (!curl_global.Status().IsOk()) {
    return curl_global.Status();
  }

  err = PreRunProcessing(
      easy_handle_, request_uri, options, inputs, outputs, headers,
      sync_request);
  if (!err.IsOk()) {
    return err;
  }

  sync_request->Timer().CaptureTimestamp(nic::RequestTimers::Kind::SEND_START);

  // During this call SEND_END (except in above case), RECV_START, and
  // RECV_END will be set.
  auto curl_status = curl_easy_perform(easy_handle_);
  if (curl_status != CURLE_OK) {
    sync_request->http_code_ = 400;
  } else {
    curl_easy_getinfo(
        easy_handle_, CURLINFO_RESPONSE_CODE, &sync_request->http_code_);
  }

  sync_request->CloseFileData();
  curl_mime_free(mime_handle_);

  InferResult::Create(result, sync_request);

  sync_request->Timer().CaptureTimestamp(nic::RequestTimers::Kind::REQUEST_END);

  nic::Error nic_err = UpdateInferStat(sync_request->Timer());
  if (!nic_err.IsOk()) {
    std::cerr << "Failed to update context stat: " << nic_err << std::endl;
  }

  err = (*result)->RequestStatus();

  return err;
}

size_t
HttpClient::ReadCallback(char* buffer, size_t size, size_t nitems, void* userp)
{
  size_t retcode =
      fread(buffer, size, nitems, ((HttpInferRequest*)userp)->FilePtr());
  if (retcode == 0) {
    ((HttpInferRequest*)userp)
        ->Timer()
        .CaptureTimestamp(nic::RequestTimers::Kind::SEND_END);
  }
  return retcode;
}

int
HttpClient::SeekCallback(void* userp, curl_off_t offset, int origin)
{
  if (fseek(((HttpInferRequest*)userp)->FilePtr(), offset, origin) == 0)
    return CURL_SEEKFUNC_OK;
  else
    return CURL_SEEKFUNC_FAIL;
}

size_t
HttpClient::InferResponseHeaderHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpInferRequest* request = reinterpret_cast<HttpInferRequest*>(userp);

  char* buf = reinterpret_cast<char*>(contents);
  size_t byte_size = size * nmemb;

  size_t idx = strlen(kContentLengthHTTPHeader);
  if ((idx < byte_size) && !strncasecmp(buf, kContentLengthHTTPHeader, idx)) {
    while ((idx < byte_size) && (buf[idx] != ':')) {
      ++idx;
    }

    if (idx < byte_size) {
      std::string hdr(buf + idx + 1, byte_size - idx - 1);
      request->infer_response_buffer_->reserve(std::stoi(hdr));
    }
  }

  return byte_size;
}

size_t
HttpClient::InferResponseHandler(
    void* contents, size_t size, size_t nmemb, void* userp)
{
  HttpInferRequest* request = reinterpret_cast<HttpInferRequest*>(userp);

  if (request->Timer().Timestamp(nic::RequestTimers::Kind::RECV_START) == 0) {
    request->Timer().CaptureTimestamp(nic::RequestTimers::Kind::RECV_START);
  }

  char* buf = reinterpret_cast<char*>(contents);
  size_t result_bytes = size * nmemb;
  request->infer_response_buffer_->append(buf, result_bytes);

  // InferResponseHandler may be called multiple times so we overwrite
  // RECV_END so that we always have the time of the last.
  request->Timer().CaptureTimestamp(nic::RequestTimers::Kind::RECV_END);

  return result_bytes;
}

Error
HttpClient::PreRunProcessing(
    void* vcurl, std::string& request_uri, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers, std::shared_ptr<HttpInferRequest>& http_request)
{
  CURL* curl = reinterpret_cast<CURL*>(vcurl);

  // Prepare the request object to provide the data for inference.
  Error err = http_request->InitializeRequest();
  if (!err.IsOk()) {
    return err;
  }

  std::vector<std::string> input_filepaths;

  curl_easy_setopt(curl, CURLOPT_URL, request_uri.c_str());
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
  curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);

  if (verbose_) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  }

  const long buffer_byte_size = 16 * 1024 * 1024;
  curl_easy_setopt(curl, CURLOPT_UPLOAD_BUFFERSIZE, buffer_byte_size);
  curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, buffer_byte_size);

  // request data provided by InferRequestProvider()
  mime_handle_ = curl_mime_init(easy_handle_);
  // Add the buffers holding input tensor data
  for (const auto input : inputs) {
    TorchServeInferInput* this_input =
        dynamic_cast<TorchServeInferInput*>(input);
    this_input->PrepareForRequest();
    bool end_of_input = false;
    while (!end_of_input) {
      const uint8_t* buf;
      size_t buf_size;
      this_input->GetNext(&buf, &buf_size, &end_of_input);
      std::string file_path(
          reinterpret_cast<const char*>(buf) + 4, buf_size - 4);
      if (buf != nullptr) {
        Error err = http_request->OpenFileData(file_path);
        if (!err.IsOk()) {
          return err;
        }
        if (verbose_) {
          input_filepaths.push_back(file_path);
        }
      }
    }
  }

  long file_size = http_request->FileSize();
  curl_mimepart* part = curl_mime_addpart((curl_mime*)mime_handle_);
  curl_mime_data_cb(
      part, file_size, ReadCallback, SeekCallback, NULL, http_request.get());
  curl_mime_name(part, "data");

  curl_easy_setopt(easy_handle_, CURLOPT_MIMEPOST, (curl_mime*)mime_handle_);

  // response headers handled by InferResponseHeaderHandler()
  curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, InferResponseHeaderHandler);
  curl_easy_setopt(curl, CURLOPT_HEADERDATA, http_request.get());

  // response data handled by InferResponseHandler()
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, InferResponseHandler);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, http_request.get());

  struct curl_slist* list = nullptr;
  for (const auto& pr : headers) {
    std::string hdr = pr.first + ": " + pr.second;
    list = curl_slist_append(list, hdr.c_str());
  }
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

  // The list will be freed when the request is destructed
  http_request->header_list_ = list;

  if (verbose_) {
    std::cout << "inference request : [";
    bool first = true;
    for (const auto& fn : input_filepaths) {
      if (first) {
        first = false;
      } else {
        std::cout << ",";
      }
      std::cout << "\"" << fn << "\"";
    }
    std::cout << "]" << std::endl;
  }

  return Error::Success;
}

HttpClient::HttpClient(const std::string& url, bool verbose)
    : InferenceServerClient(verbose), url_(url),
      easy_handle_(reinterpret_cast<void*>(curl_easy_init()))
{
}

HttpClient::~HttpClient()
{
  exiting_ = true;

  if (easy_handle_ != nullptr) {
    curl_easy_cleanup(reinterpret_cast<CURL*>(easy_handle_));
  }
}

//======================================================================

Error
InferResult::Create(
    InferResult** infer_result, std::shared_ptr<HttpInferRequest> infer_request)
{
  *infer_result =
      reinterpret_cast<InferResult*>(new InferResult(infer_request));
  return Error::Success;
}

Error
InferResult::RequestStatus() const
{
  return status_;
}

InferResult::InferResult(std::shared_ptr<HttpInferRequest> infer_request)
    : infer_request_(infer_request)
{
  if (infer_request->http_code_ != 200) {
    status_ = Error(
        "inference failed with error code " +
        std::to_string(infer_request->http_code_));
  }
}

//======================================================================

}}}  // namespace perfanalyzer::clientbackend::torchserve
