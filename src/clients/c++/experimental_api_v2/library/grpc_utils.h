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

#include "src/clients/c++/experimental_api_v2/library/base_utils.h"
#include "src/clients/c++/experimental_api_v2/library/common_utils.h"
#include "src/core/constants.h"
#include "src/core/grpc_service_v2.grpc.pb.h"
#include "src/core/model_config.pb.h"

namespace nvidia { namespace inferenceserver { namespace client {

//==============================================================================
/// An InferInputGrpc object is used to describe the model input for inference
/// using gRPC protocol.
///
class InferInputGrpc : public InferInput {
 public:
  /// Create a InferInputGrpc instance that describes a model input.
  /// \param infer_input Returns a new InferInputGrpc object.
  /// \param name The name of input whose data will be described by this object.
  /// \param dims The shape of the input.
  /// \param datatype The datatype of the input.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::shared_ptr<InferInputGrpc>* infer_input, const std::string& name,
      const std::vector<int64_t>& dims, const std::string& datatype);

  /// See InferInput::GetName(std::string* name)
  Error GetName(std::string* name) const override;

  /// See InferInput::GetDatatype(std::string* datatype)
  Error GetDatatype(std::string* datatype) const override;

  /// See InferInput::GetShape(std::vector<int64_t>* dims)
  Error GetShape(std::vector<int64_t>* dims) const override;

  /// See InferInput::SetShape(const std::vector<int64_t>& dims)
  Error SetShape(const std::vector<int64_t>& dims) override;

  /// See InferInput::SetRaw(const uint8_t* input, size_t input_byte_size)
  Error SetRaw(const uint8_t* input, size_t input_byte_size) override;

  /// See InferInput::SetRaw(const std::vector<uint8_t>& input)
  Error SetRaw(const std::vector<uint8_t>& input) override;

  /// See InferInput::SetFromString(const std::vector<std::string>& input)
  Error SetFromString(const std::vector<std::string>& input) override;

  /// See InferInput::SetSharedMemory(
  ///   const std::string& region_name, const size_t byte_size,
  ///   const size_t offset)
  Error SetSharedMemory(
      const std::string& region_name, const size_t byte_size,
      const size_t offset = 0) override;

  /// Returns the unferlying InferInputTensor message.
  /// \return ModelInferRequest::InferInputTensor
  ModelInferRequest::InferInputTensor GetTensor() { return input_tensor_; }

 private:
  InferInputGrpc(
      const std::string& name, const std::vector<int64_t>& dims,
      const std::string& datatype);

  ModelInferRequest::InferInputTensor input_tensor_;
};

//==============================================================================
/// An InferOutputGrpc object is used to describe the requested output for
/// inference request using gRPC protocol.
///
class InferOutputGrpc : public InferOutput {
 public:
  /// Create a InferOutputGrpc instance that describes a model output being
  /// requested.
  /// \param infer_output Returns a new InferOutputGrpc object.
  /// \param name The name of output being requested.
  /// \param class_count The number of classifications to be requested. The
  /// default value is 0 which means the classification results are not
  /// requested.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::shared_ptr<InferOutputGrpc>* infer_output, const std::string& name,
      const size_t class_count = 0);

  /// See InferOutput::GetName(std::string* name)
  Error GetName(std::string* name) const override;

  /// See InferOutput::SetSharedMemory(
  ///   const std::string& region_name, const size_t byte_size,
  ///   const size_t offset)
  Error SetSharedMemory(
      const std::string& region_name, const size_t byte_size,
      const size_t offset = 0) override;

  /// Returns the unferlying InferRequestedOutputTensor message.
  /// \return ModelInferRequest::InferRequestedOutputTensor
  ModelInferRequest::InferRequestedOutputTensor GetTensor()
  {
    return output_tensor_;
  }

 private:
  InferOutputGrpc(const std::string& name, const size_t class_count = 0);

  ModelInferRequest::InferRequestedOutputTensor output_tensor_;
};


//==============================================================================
/// An InferResultGrpc instance is  used  to access and interpret the
/// response from gRPC inference request.
///
class InferResultGrpc : public InferResult {
 public:
  /// Create a InferResultGrpc instance to interpret server response.
  /// \param infer_result Returns a new InferResultGrpc object.
  /// \param response  The response of server for an inference request.
  /// \return Error object indicating success or failure.
  static Error Create(
      std::shared_ptr<InferResultGrpc>* infer_result,
      std::shared_ptr<ModelInferResponse> response);

  /// See InferResult::GetShape(const std::string& output_name,
  /// std::vector<int64_t>* shape)
  Error GetShape(const std::string& output_name, std::vector<int64_t>* shape)
      const override;

  /// See InferResult::GetDatatype(const std::string& output_name, std::string*
  /// datatype)
  Error GetDatatype(
      const std::string& output_name, std::string* datatype) const override;

  /// See InferResult::GetRaw(const std::string& output_name, const uint8_t**
  /// buf, size_t* byte_size)
  Error GetRaw(
      const std::string& output_name, const uint8_t** buf,
      size_t* byte_size) const override;

  /// Returns the unferlying ModelInferResponse message.
  /// \return pointer to a ModelInferResponse message.
  std::shared_ptr<ModelInferResponse> GetResponse() { return response_; }

 private:
  InferResultGrpc(std::shared_ptr<ModelInferResponse> response);

  std::map<std::string, const ModelInferResponse::InferOutputTensor*>
      output_name_to_result_map_;

  std::shared_ptr<ModelInferResponse> response_;
};

}}}  // namespace nvidia::inferenceserver::client
