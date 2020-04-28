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

class HttpInferRequest;

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
  ~InferenceServerHttpClient();

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
  /// \param server_metadata Returns the server metadata as rapidJSON DOM
  /// object.
  /// \param headers Optional map specifying additional HTTP headers to
  /// include in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ServerMetadata(
      rapidjson::Document* server_metadata, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the metadata of specified model.
  /// \param model_metadata Returns model metadata as rapidJSON DOM object.
  /// \param model_name The name of the model to get metadata.
  /// \param model_version The version of the model to get metadata.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelMetadata(
      rapidjson::Document* model_metadata, const std::string& model_name,
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the configuration of specified model.
  /// \param model_config Returns model config as rapidJSON DOM object.
  /// \param model_name The name of the model to get configuration.
  /// \param model_version The version of the model to get configuration.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelConfig(
      rapidjson::Document* model_config, const std::string& model_name,
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the index of model repository
  /// contents.
  /// \param repository_index Returns the repository index as rapidJSON DOM
  /// object.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelRepositoryIndex(
      rapidjson::Document* repository_index, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the inference server to load or reload specified model.
  /// \param model_name The name of the model to be loaded or reloaded.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error LoadModel(
      const std::string& model_name, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the inference server to unload specified model.
  /// \param model_name The name of the model to be unloaded.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error UnloadModel(
      const std::string& model_name, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the inference statistics for the
  /// specified model name and version.
  /// \param infer_stat Returns the inference statistics of requested model name
  /// and version as rapidJSON DOM object.
  /// \param model_name The name of the model to get inference statistics. The
  /// default value is an empty string which means statistics of all models will
  /// be returned in the response.
  /// \param model_version The version of the model to get inference statistics.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelInferenceStatistics(
      rapidjson::Document* infer_stat, const std::string& model_name = "",
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the status for requested system
  /// shared memory.
  /// \param status Returns the system shared memory status as rapidJSON DOM
  /// object.
  /// \param region_name The name of the region to query status. The default
  /// value is an empty string, which means that the status of all active system
  /// shared memory will be returned.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error SystemSharedMemoryStatus(
      rapidjson::Document* status, const std::string& region_name = "",
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the server to register a system shared memory with the provided
  /// details.
  /// \param name The name of the region to register.
  /// \param key The key of the underlying memory object that contains the
  /// system shared memory region.
  /// \param byte_size The size of the system shared memory region, in bytes.
  /// \param offset Offset, in bytes, within the underlying memory object to
  /// the start of the system shared memory region. The default value is zero.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request
  Error RegisterSystemSharedMemory(
      const std::string& name, const std::string& key, const size_t byte_size,
      const size_t offset = 0, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the server to unregister a system shared memory with the
  /// specified name.
  /// \param name The name of the region to unregister. The default value is
  /// empty string which means all the system shared memory regions will be
  /// unregistered.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request
  Error UnregisterSystemSharedMemory(
      const std::string& name = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the status for requested CUDA
  /// shared memory.
  /// \param status Returns the CUDA shared memory status as rapidJSON DOM
  /// object.
  /// \param region_name The name of the region to query status. The default
  /// value is an empty string, which means that the status of all active CUDA
  /// shared memory will be returned.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error CudaSharedMemoryStatus(
      rapidjson::Document* status, const std::string& region_name = "",
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the server to register a CUDA shared memory with the provided
  /// details.
  /// \param name The name of the region to register.
  /// \param cuda_shm_handle The cudaIPC handle for the memory object.
  /// \param device_id The GPU device ID on which the cudaIPC handle was
  /// created.
  /// \param byte_size The size of the CUDA shared memory region, in
  /// bytes.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request
  Error RegisterCudaSharedMemory(
      const std::string& name, const cudaIpcMemHandle_t& cuda_shm_handle,
      const size_t device_id, const size_t byte_size,
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the server to unregister a CUDA shared memory with the
  /// specified name.
  /// \param name The name of the region to unregister. The default value is
  /// empty string which means all the CUDA shared memory regions will be
  /// unregistered.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request
  Error UnregisterCudaSharedMemory(
      const std::string& name = "", const Headers& headers = Headers(),
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

  /// Run asynchronous inference on server.
  /// Once the request is completed, the InferResult pointer will be passed to
  /// the provided 'callback' function. Upon the invocation of callback
  /// function, the ownership of InferResult object is transfered to the
  /// function caller. It is then the caller's choice on either retrieving the
  /// results inside the callback function or deferring it to a different thread
  /// so that the client is unblocked. In order to prevent memory leak, user
  /// must ensure this object gets deleted.
  /// Note: InferInput::AppendRaw() or InferInput::SetSharedMemory() calls do
  /// not copy the data buffers but hold the pointers to the data directly.
  /// It is advisable to not to disturb the buffer contents until the respective
  /// callback is invoked.
  /// \param callback The callback function to be invoked on request completion.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs Optional vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the model
  /// config will be returned as default settings.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success
  /// or failure of the request.
  Error AsyncInfer(
      OnCompleteFn callback, const InferOptions& options,
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
      std::shared_ptr<HttpInferRequest>& request);
  void AsyncTransfer();
  Error Get(
      std::string& request_uri, const Headers& headers,
      const Parameters& query_params, rapidjson::Document* response,
      long* http_code);
  Error Post(
      std::string& request_uri, const rapidjson::Document& request,
      const Headers& headers, const Parameters& query_params,
      rapidjson::Document* response, long* http_code);

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

  using AsyncReqMap = std::map<uintptr_t, std::shared_ptr<HttpInferRequest>>;
  // curl multi handle for processing asynchronous requests
  void* multi_handle_;
  // map to record ongoing asynchronous requests with pointer to easy handle
  // or tag id as key
  AsyncReqMap ongoing_async_requests_;
};

}}}  // namespace nvidia::inferenceserver::client
