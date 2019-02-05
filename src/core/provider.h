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
#pragma once

#include "libevent/include/event2/buffer.h"
#include "src/core/api.pb.h"
#include "src/core/grpc_service.pb.h"
#include "tensorflow/core/lib/core/errors.h"

struct evbuffer;

namespace nvidia { namespace inferenceserver {

class InferenceBackend;

//
// Provide inference request inputs and meta-data
//
class InferRequestProvider {
 public:
  explicit InferRequestProvider(
      const std::string& model_name, const int64_t version)
      : model_name_(model_name), version_(version)
  {
  }

  // Return the requested model name.
  const std::string& ModelName() const { return model_name_; }

  // Return the requested model version, or -1 if no specific version
  // was requested.
  int64_t ModelVersion() const { return version_; }

  // Get the request header for this inference request.
  virtual const InferRequestHeader& RequestHeader() const = 0;

  // Get the next contiguous chunk of bytes for the 'idx'
  // input. Return a pointer to the chunk in 'content' and the length
  // of the chunk in 'content_byte_size'. If there are no more bytes
  // for the input return 'content' = nullptr. If 'force_contiguous'
  // is true then the entire (remaining) input will be returned as a
  // single chunk. In some cases this will require copying the data.
  virtual tensorflow::Status GetNextInputContent(
      int idx, const void** content, size_t* content_byte_size,
      bool force_contiguous) = 0;

 protected:
  tensorflow::Status GetInputBatchByteSize(
      const InferRequestHeader::Input& input, const ModelInput& input_config,
      uint64_t* byte_size);

 private:
  const std::string& model_name_;
  const int64_t version_;
};

//
// Inference input provider for a GRPC inference request
//
class GRPCInferRequestProvider : public InferRequestProvider {
 public:
  // Initialize based on gRPC request
  static tensorflow::Status Create(
      const InferenceBackend& is, const InferRequest& request,
      std::shared_ptr<GRPCInferRequestProvider>* infer_provider);

  const InferRequestHeader& RequestHeader() const override
  {
    return request_.meta_data();
  }

  tensorflow::Status GetNextInputContent(
      int idx, const void** content, size_t* content_byte_size,
      bool force_contiguous) override;

 private:
  GRPCInferRequestProvider(const InferRequest& request, const int64_t version);

  const InferRequest& request_;
  std::vector<bool> content_delivered_;
};

//
// Inference input provider for an HTTP inference request
//
class HTTPInferRequestProvider : public InferRequestProvider {
 public:
  // Initialize based on HTTP request
  static tensorflow::Status Create(
      evbuffer* input_buffer, const InferenceBackend& is,
      const std::string& model_name, const int64_t model_version,
      const std::string& request_header_str,
      std::shared_ptr<HTTPInferRequestProvider>* infer_provider);

  const InferRequestHeader& RequestHeader() const override
  {
    return request_header_;
  }

  tensorflow::Status GetNextInputContent(
      int idx, const void** content, size_t* content_byte_size,
      bool force_contiguous) override;

 private:
  HTTPInferRequestProvider(const std::string& model_name, const int64_t version)
      : InferRequestProvider(model_name, version)
  {
  }

  InferRequestHeader request_header_;
  using Block = std::pair<const char*, size_t>;
  std::vector<std::vector<Block>> contents_;
  std::vector<size_t> contents_idx_;
  std::vector<std::vector<char>> contiguous_buffers_;
};

//
// Provide support for reporting inference response outputs and
// response meta-data
//
class InferResponseProvider {
 public:
  explicit InferResponseProvider(const InferRequestHeader& request_header);

  // Get the full response header for this inference request.
  virtual const InferResponseHeader& ResponseHeader() const = 0;

  // Get a mutuable full response header for this inference request.
  virtual InferResponseHeader* MutableResponseHeader() = 0;

  // Get a buffer to store results for a named output. The output must
  // be listed in the request header.
  virtual tensorflow::Status GetOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) = 0;

  // Return true if this provider requires a named output.
  bool RequiresOutput(const std::string& name);

  // Finialize response based on a servable.
  tensorflow::Status FinalizeResponse(const InferenceBackend& is);

 protected:
  struct Output;

  // Check that 'name' is a valid output. If output is to be buffered,
  // allocate space for it and point to that space with 'content'
  tensorflow::Status CheckAndSetIfBufferedOutput(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape, Output** output);

 protected:
  const InferRequestHeader& request_header_;

  // Map from output name to the InferRequestHeader output information
  // for that output.
  std::unordered_map<std::string, const InferRequestHeader::Output*>
      output_map_;

  // Information about each output.
  struct Output {
    std::string name_;
    std::vector<int64_t> shape_;
    size_t byte_size_;

    // Created buffer for non-RAW results
    std::unique_ptr<char[]> buffer_;
  };

  // Ordered list of outputs as they "added" by
  // GetOutputBuffer().
  std::vector<Output> outputs_;
};

//
// Inference response provider for a GRPC request
//
class GRPCInferResponseProvider : public InferResponseProvider {
 public:
  // Initialize based on gRPC request
  static tensorflow::Status Create(
      const InferRequestHeader& request_header, InferResponse* response,
      std::shared_ptr<GRPCInferResponseProvider>* infer_provider);

  const InferResponseHeader& ResponseHeader() const override;
  InferResponseHeader* MutableResponseHeader() override;
  tensorflow::Status GetOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) override;

 private:
  GRPCInferResponseProvider(
      const InferRequestHeader& request_header, InferResponse* response)
      : InferResponseProvider(request_header), response_(response)
  {
  }

  InferResponse* response_;
};

//
// Inference response provider for an HTTP request
//
class HTTPInferResponseProvider : public InferResponseProvider {
 public:
  static tensorflow::Status Create(
      evbuffer* output_buffer, const InferenceBackend& is,
      const InferRequestHeader& request_header,
      std::shared_ptr<HTTPInferResponseProvider>* infer_provider);

  const InferResponseHeader& ResponseHeader() const override;
  InferResponseHeader* MutableResponseHeader() override;
  tensorflow::Status GetOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) override;

 private:
  HTTPInferResponseProvider(
      evbuffer* output_buffer, const InferRequestHeader& request_header);

  InferResponseHeader response_header_;
  evbuffer* output_buffer_;
};

}}  // namespace nvidia::inferenceserver
