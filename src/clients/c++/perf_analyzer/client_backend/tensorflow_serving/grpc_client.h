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
#include "src/clients/c++/perf_analyzer/client_backend/tensorflow_serving/tfserve_infer_input.h"


#include <grpc++/grpc++.h>

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

namespace nic = nvidia::inferenceserver::client;

namespace perfanalyzer { namespace clientbackend { namespace tfserving {

struct SslOptions {
  explicit SslOptions() {}
  // File containing the PEM encoding of the server root certificates.
  // If this parameter is empty, the default roots will be used. The
  // default roots can be overridden using the
  // GRPC_DEFAULT_SSL_ROOTS_FILE_PATH environment variable pointing
  // to a file on the file system containing the roots.
  std::string root_certificates;
  // File containing the PEM encoding of the client's private key.
  // This parameter can be empty if the client does not have a
  // private key.
  std::string private_key;
  // File containing the PEM encoding of the client's certificate chain.
  // This parameter can be empty if the client does not have a
  // certificate chain.
  std::string certificate_chain;
};

class InferResult;

using TFServeOnCompleteFn = std::function<void(InferResult*)>;

//==============================================================================
/// An GrpcClient object is used to perform any kind of communication with the
/// TFserving service using gRPC protocol. None of the functions are thread
/// safe.
///
/// \code
///   std::unique_ptr<GrpcClient> client;
///   GrpcClient::Create(&client, "localhost:8500");
///   ...
///   ...
/// \endcode
///
class GrpcClient : public nic::InferenceServerClient {
 public:
  ~GrpcClient();

  /// Create a client that can be used to communicate with the server.
  /// \param client Returns a new InferenceServerGrpcClient object.
  /// \param server_url The inference server name and port.
  /// \param verbose If true generate verbose output when contacting
  /// the inference server.
  /// \param use_ssl If true use encrypted channel to the server.
  /// \param ssl_options Specifies the files required for
  /// SSL encryption and authorization.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<GrpcClient>* client, const std::string& server_url,
      bool verbose = false, bool use_ssl = false,
      const SslOptions& ssl_options = SslOptions());

  /// Contact the inference server and get the metadata of specified model.
  /// \param model_metadata Returns model metadata as ModelMetadataResponse
  /// message.
  /// \param model_name The name of the model to get metadata.
  /// \param model_version The version of the model to get metadata.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the request.
  Error ModelMetadata(
      tensorflow::serving::GetModelMetadataResponse* model_metadata,
      const std::string& model_name, const std::string& model_version = "",
      const Headers& headers = Headers());

  /// Run synchronous inference on server.
  /// \param result Returns the result of inference.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs Optional vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the model
  /// config will be returned as default settings.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \param compression_algorithm The compression algorithm to be used
  /// on the grpc requests.
  /// \return Error object indicating success or failure of the
  /// request.
  Error Infer(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>(),
      const Headers& headers = Headers(),
      const grpc_compression_algorithm compression_algorithm =
          GRPC_COMPRESS_NONE);

  /// Run asynchronous inference on server.
  /// Once the request is completed, the InferResult pointer will be passed to
  /// the provided 'callback' function. Upon the invocation of callback
  /// function, the ownership of InferResult object is transfered to the
  /// function caller. It is then the caller's choice on either retrieving the
  /// results inside the callback function or deferring it to a different thread
  /// so that the client is unblocked. In order to prevent memory leak, user
  /// must ensure this object gets deleted.
  /// \param callback The callback function to be invoked on request completion.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs Optional vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the model
  /// config will be returned as default settings.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \param compression_algorithm The compression algorithm to be used
  /// on the grpc requests.
  /// \return Error object indicating success or failure of the request.
  Error AsyncInfer(
      TFServeOnCompleteFn callback, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>(),
      const Headers& headers = Headers(),
      const grpc_compression_algorithm compression_algorithm =
          GRPC_COMPRESS_NONE);

 private:
  GrpcClient(
      const std::string& url, bool verbose, bool use_ssl,
      const SslOptions& ssl_options);
  Error PreRunProcessing(
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs);
  void AsyncTransfer();
  Error ClearAllInputFields(tensorflow::TensorProto* input_tensor_proto);
  Error PopulateInputData(
      TFServeInferInput* input, tensorflow::TensorProto* input_tensor_proto);
  Error PopulateHalfVal(tensorflow::TensorProto* input_tensor_proto);
  Error PopulateFloatVal(tensorflow::TensorProto* input_tensor_proto);
  Error PopulateDoubleVal(tensorflow::TensorProto* input_tensor_proto);
  Error PopulateIntVal(
      tensorflow::TensorProto* input_tensor_proto, size_t step_size = 4);
  Error PopulateStrVal(tensorflow::TensorProto* input_tensor_proto);
  Error PopulateBoolVal(tensorflow::TensorProto* input_tensor_proto);
  Error PopulateInt64Val(tensorflow::TensorProto* input_tensor_proto);
  Error PopulateUintVal(tensorflow::TensorProto* input_tensor_proto);
  Error PopulateUint64Val(tensorflow::TensorProto* input_tensor_proto);

  // The producer-consumer queue used to communicate asynchronously with
  // the GRPC runtime.
  grpc::CompletionQueue async_request_completion_queue_;

  bool enable_stream_stats_;
  std::mutex stream_mutex_;

  // GRPC end point.
  std::unique_ptr<tensorflow::serving::PredictionService::Stub> stub_;
  // request for GRPC call, one request object can be used for multiple calls
  // since it can be overwritten as soon as the GRPC send finishes.
  tensorflow::serving::PredictRequest infer_request_;
  // A temporary buffer to hold serialized data
  std::string temp_buffer_;
};

//======================================================================

class InferResult {
 public:
  static Error Create(
      InferResult** infer_result,
      std::shared_ptr<tensorflow::serving::PredictResponse> response,
      Error& request_status);


  Error RequestStatus() const;
  Error Id(std::string* id) const;
  std::string DebugString() const { return response_->DebugString(); }

 private:
  InferResult(
      std::shared_ptr<tensorflow::serving::PredictResponse> response,
      Error& request_status);

  std::shared_ptr<tensorflow::serving::PredictResponse> response_;
  Error request_status_;
};

//======================================================================

}}}  // namespace perfanalyzer::clientbackend::tfserving
