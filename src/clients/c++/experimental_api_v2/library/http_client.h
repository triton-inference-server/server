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

/// \file

#include <map>
#include <memory>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/stringbuffer.h"
#include "src/clients/c++/experimental_api_v2/library/common.h"

namespace nvidia { namespace inferenceserver { namespace client {

/// The key-value map type to be included in the request
/// as custom headers.
typedef std::map<std::string, std::string> Headers;
/// The key-value map type to be included as URL parameters.
typedef std::map<std::string, std::string> Parameters;

/// Returns reader friendly text representation of
/// the DOM object.
/// \param json_dom The json DOM object.
/// \return Formatted string representation of passed JSON.
std::string GetJsonText(const rapidjson::Document& json_dom);

//==============================================================================
/// An InferenceServerHttpClient object is used to perform any kind of
/// communication with the InferenceServer using HTTP protocol.
///
/// \code
///   std::unique_ptr<InferenceServerHttpClient> client;
///   InferenceServerHttpClient::Create(&client, "localhost:8000");
///   bool live;
///   client->IsServerLive(&live);
///   ...
///   ...
/// \endcode
///
class InferenceServerHttpClient : public InferenceServerClient {
 public:
  /// Create a client that can be used to communicate with the server.
  /// \param client Returns a new InferenceServerHttpClient object.
  /// \param server_url The inference server name and port.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<InferenceServerHttpClient>* client,
      const std::string& server_url, bool verbose = false);

  /// Contact the inference server and get its liveness.
  /// \param live Returns whether the server is live or not.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \return Error object indicating success or failure of the request.
  Error IsServerLive(
      bool* live, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get its readiness.
  /// \param ready Returns whether the server is ready or not.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error IsServerReady(
      bool* ready, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the readiness of specified model.
  /// \param ready Returns whether the specified model is ready or not.
  /// \param model_name The name of the model to check for readiness.
  /// \param model_version The version of the model to check for readiness.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error IsModelReady(
      bool* ready, const std::string& model_name,
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get its metadata.
  /// \param server_metadata Returns the server metadata as JSON DOM object.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error GetServerMetadata(
      rapidjson::Document* server_metadata, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the metadata of specified model.
  /// \param model_metadata Returns model metadata as JSON DOM object.
  /// \param model_name The name of the model to get metadata.
  /// \param model_version The version of the model to get metadata.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error GetModelMetadata(
      rapidjson::Document* model_metadata, const std::string& model_name,
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the configuration of specified model.
  /// \param model_config Returns model config as JSON DOM object.
  /// \param model_name The name of the model to get configuration.
  /// \param model_version The version of the model to get configuration.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error GetModelConfig(
      rapidjson::Document* model_config, const std::string& model_name,
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Run synchronous inference on server.
  /// \param result Returns the result of inference.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs Optional vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the model
  /// config will be returned as default settings.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the
  /// request.
  Error Infer(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>(),
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

 private:
  InferenceServerHttpClient(const std::string& url, bool verbose);

  void PrepareRequestJson(
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs,
      rapidjson::Document* request_json);
  Error PreRunProcessing(
      std::string& request_uri, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs,
      const Headers& headers, const Parameters& query_params,
      std::shared_ptr<InferRequest>& request);
  Error Get(
      std::string& request_uri, const Headers& headers,
      const Parameters& query_params, rapidjson::Document* response,
      long* http_code);

  static size_t ResponseHandler(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t InferRequestProvider(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t InferResponseHeaderHandler(
      void* contents, size_t size, size_t nmemb, void* userp);
  static size_t InferResponseHandler(
      void* contents, size_t size, size_t nmemb, void* userp);

  // The server url
  const std::string url_;
  // Enable verbose output
  const bool verbose_;
};

//==============================================================================
/// An InferResultHttp instance is used  to access and interpret the
/// response of an inference request from HTTP endpoint. This object
/// holds data for all requested outputs.
///
class InferResultHttp : public InferResult {
 public:
  /// Create a InferResult instance to interpret server response.
  /// \param infer_result Returns a new InferResult object.
  /// \param response  The response of server for an inference request.
  /// \return Error object indicating success or failure.
  static Error Create(
      InferResult** infer_result, std::unique_ptr<std::string> response,
      size_t json_response_size);

  /// See InferResult::ModelName(std::string* name)
  Error ModelName(std::string* name) const override;

  /// See InferResult::ModelVersion(std::string* version)
  Error ModelVersion(std::string* version) const override;

  /// See InferResult::Id(std::string* id)
  Error Id(std::string* id) const override;

  /// See InferResult::Shape(const std::string& output_name,
  ///  std::vector<int64_t>* shape)
  Error Shape(const std::string& output_name, std::vector<int64_t>* shape)
      const override;

  /// See InferResult::Datatype(
  ///    const std::string& output_name, std::string* datatype)
  Error Datatype(
      const std::string& output_name, std::string* datatype) const override;

  /// See InferResult::RawData(
  ///    const std::string& output_name, const uint8_t** buf,
  ///    size_t* byte_size)
  Error RawData(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const override;

  /// See InferResult::DebugString()
  std::string DebugString() const override;

  /// Returns the status of this request.
  /// \return Error object indicating success or failure of the
  /// request.
  Error RequestStatus() const;

 private:
  InferResultHttp(
      std::unique_ptr<std::string> response, size_t json_response_size);

  std::map<std::string, const rapidjson::Value*> output_name_to_result_map_;
  std::map<std::string, std::pair<const uint8_t*, const size_t>>
      output_name_to_buffer_map_;

  rapidjson::Document response_json_;
  std::unique_ptr<std::string> response_;
};

}}}  // namespace nvidia::inferenceserver::client
