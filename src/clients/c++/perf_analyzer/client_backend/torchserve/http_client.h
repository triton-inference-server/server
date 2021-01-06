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
#pragma once

#include "src/clients/c++/library/common.h"
#include "src/clients/c++/perf_analyzer/client_backend/client_backend.h"
#include "src/clients/c++/perf_analyzer/client_backend/torchserve/torchserve_infer_input.h"

#include <curl/curl.h>
#include <stdio.h>
#include <stdlib.h>

namespace nic = nvidia::inferenceserver::client;

namespace perfanalyzer { namespace clientbackend { namespace torchserve {

class InferResult;
class HttpInferRequest;

using TorchServeOnCompleteFn = std::function<void(InferResult*)>;

//==============================================================================
/// An HttpClient object is used to perform any kind of communication with the
/// torchserve service using libcurl. None of the functions are thread
/// safe.
///
/// \code
///   std::unique_ptr<HttpClient> client;
///   HttpClient::Create(&client, "localhost:8080");
///   ...
///   ...
/// \endcode
///
class HttpClient : public nic::InferenceServerClient {
 public:
  ~HttpClient();

  /// Create a client that can be used to communicate with the server.
  /// \param client Returns a new InferenceServerHttpClient object.
  /// \param server_url The inference server name and port.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<HttpClient>* client, const std::string& server_url,
      const bool verbose);

  /// Run synchronous inference on server.
  /// \param result Returns the result of inference.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs Optional vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the model
  /// config will be returned as default settings.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the
  /// request.
  Error Infer(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>(),
      const Headers& headers = Headers());

 private:
  HttpClient(const std::string& url, bool verbose);
  Error PreRunProcessing(
      void* curl, std::string& request_uri, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs,
      const Headers& headers, std::shared_ptr<HttpInferRequest>& request);

  static size_t ReadCallback(
      char* buffer, size_t size, size_t nitems, void* userp);
  static int SeekCallback(void* userp, curl_off_t offset, int origin);
  static size_t InferResponseHeaderHandler(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t InferResponseHandler(
      void* contents, size_t size, size_t nmemb, void* userp);

  // The server url
  const std::string url_;
  // curl easy handle shared for all synchronous requests.
  void* easy_handle_;
  // The handle to interact with mime API.
  curl_mime* mime_handle_;
};

//======================================================================

class HttpInferRequest {
 public:
  struct Deleter {
    void operator()(FILE* file)
    {
      // Do nothing
    }
  };

  HttpInferRequest();
  ~HttpInferRequest();
  Error InitializeRequest();
  Error OpenFileData(std::string& file_path);
  long FileSize();
  Error CloseFileData();
  nic::RequestTimers& Timer() { return timer_; }
  std::string& DebugString() { return *infer_response_buffer_; }
  FILE* FilePtr() { return file_ptr_.get(); }
  friend HttpClient;
  friend InferResult;

 private:
  // Pointer to the list of the HTTP request header, keep it such that it will
  // be valid during the transfer and can be freed once transfer is completed.
  struct curl_slist* header_list_;
  std::unique_ptr<FILE, Deleter> file_ptr_;
  // HTTP response code for the inference request
  long http_code_;
  // Buffer that accumulates the response body.
  std::unique_ptr<std::string> infer_response_buffer_;
  // The timers for infer request.
  nic::RequestTimers timer_;
};

//======================================================================

class InferResult {
 public:
  static Error Create(
      InferResult** infer_result,
      std::shared_ptr<HttpInferRequest> infer_request);
  Error RequestStatus() const;
  Error Id(std::string* id) const;
  std::string DebugString() const { return infer_request_->DebugString(); }

 private:
  InferResult(std::shared_ptr<HttpInferRequest> infer_request);

  // The status of the inference
  Error status_;
  // The pointer to the HttpInferRequest object
  std::shared_ptr<HttpInferRequest> infer_request_;
};

//======================================================================

}}}  // namespace perfanalyzer::clientbackend::torchserve
