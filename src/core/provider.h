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

#include <event2/buffer.h>
#include "src/core/api.pb.h"
#include "src/core/grpc_service.pb.h"
#include "src/core/model_config.h"
#include "src/core/status.h"
#include "src/core/trtserver.h"

namespace nvidia { namespace inferenceserver {

class InferenceBackend;
class LabelProvider;

//
// SystemMemory used to access data in providers
//
class SystemMemory {
 public:
  // Get the 'idx'-th data block in the buffer. Using index to avoid
  // maintaining internal state such that one buffer can be shared
  // across multiple providers.
  // 'idx' zero base index. Valid indices are continuous.
  // 'byte_size' returns the byte size of the chunk of bytes.
  // Return the pointer to the data block. Returns nullptr if 'idx' is
  // out of range
  virtual const char* BufferAt(size_t idx, size_t* byte_size) const = 0;

  // Return the total byte size of the data buffer
  size_t TotalByteSize() const { return total_byte_size_; }

 protected:
  SystemMemory() : total_byte_size_(0) {}
  size_t total_byte_size_;
};

class SystemMemoryReference : public SystemMemory {
 public:
  // Create a read-only data buffer as a reference to other data buffer
  SystemMemoryReference();

  //\see SystemMemory::BufferAt()
  const char* BufferAt(size_t idx, size_t* byte_size) const override;

  // Add a 'buffer' with 'byte_size' as part of this data buffer
  // Return the index of the buffer
  size_t AddBuffer(const char* buffer, size_t byte_size);

 private:
  using Block = std::pair<const char*, size_t>;
  std::vector<Block> buffer_;
};

class AllocatedSystemMemory : public SystemMemory {
 public:
  // Create a continuous data buffer with 'byte_size'.
  AllocatedSystemMemory(size_t byte_size);

  //\see SystemMemory::BufferAt()
  const char* BufferAt(size_t idx, size_t* byte_size) const override;

  // Return the mutable buffer
  char* MutableBuffer();

 private:
  std::unique_ptr<char[]> buffer_;
};

//
// Provide inference request inputs and meta-data
//
class InferRequestProvider {
 public:
  // Initialize based on map from input name to data. The 'input_buffer' object
  // is mapping from input name to data buffer for that input.
  static Status Create(
      const std::string& model_name, const int64_t model_version,
      const InferRequestHeader& request_header,
      const std::unordered_map<std::string, std::shared_ptr<SystemMemory>>&
          input_buffer,
      std::shared_ptr<InferRequestProvider>* provider);

  // Return the requested model name.
  const std::string& ModelName() const { return model_name_; }

  // Return the requested model version, or -1 if no specific version
  // was requested.
  int64_t ModelVersion() const { return version_; }

  // Get the request header for this inference request that has been
  // validated and normalized so that all inputs have shape and
  // batch-byte-size defined.
  const InferRequestHeader& RequestHeader() const { return request_header_; }

  // Get the next contiguous chunk of bytes for the 'name'd
  // input. Return a pointer to the chunk in 'content'.
  // 'content_byte_size' acts as both input and output. On input
  // 'content_byte_size' is a hint of the maximum chunk size that
  // should be returned in 'content' and must be non-zero unless no
  // additional input is expected. On return 'content_byte_size' gives
  // the actual size of the chunk pointed to by 'content'. If there
  // are no more bytes for the input return 'content' == nullptr. If
  // 'force_contiguous' is true then the entire (remaining) input will
  // be returned as a single chunk. In some cases this will require
  // copying the data.
  virtual Status GetNextInputContent(
      const std::string& name, const void** content, size_t* content_byte_size,
      bool force_contiguous);

  // Retrieve the data buffer of input 'name'.
  Status GetSystemMemory(
      const std::string& name, std::shared_ptr<SystemMemory>* input_buffer);

  // Set content for named inputs. If the input already has content,
  // this content will be in-place of existing content.
  struct InputOverride {
    std::vector<uint8_t> content_;
    DimsList dims_;
    DataType datatype_;
  };

  using InputOverrideMap =
      std::unordered_map<std::string, std::shared_ptr<InputOverride>>;
  const std::shared_ptr<InputOverrideMap>& GetInputOverride() const;
  Status SetInputOverride(const std::shared_ptr<InputOverrideMap>& override);

 protected:
  explicit InferRequestProvider(
      const std::string& model_name, const int64_t version)
      : model_name_(model_name), version_(version)
  {
  }

  // Get the override content for 'name'd input. Return a pointer to
  // the override content in 'content'.  Return the override content
  // byte-size in 'content_byte_size'.  Return true if there is
  // override content (and so 'content' and 'content_byte_size' are
  // valid) or false if there is no override content (and so 'content'
  // and 'content_byte_size' are unchanged).
  bool GetInputOverrideContent(
      const std::string& name, const void** content, size_t* content_byte_size);

  const std::string model_name_;
  const int64_t version_;
  InferRequestHeader request_header_;

  // Input content overrides.
  std::shared_ptr<InputOverrideMap> overrides_;

  // The inputs that have had their override content consumed by a
  // call to GetInputOverrideContent. A given input override will only
  // return the content once and on subsequent calls will return
  // 'content' == nullptr to indicate that all the override content
  // has been consumed.
  std::set<std::string> overrides_consumed_;

  // Placeholder for providing buffer as contiguous block.
  std::vector<std::vector<char>> contiguous_buffers_;

  // Map from input name to the content of the input. The content contains
  // the buffer and index to the next data block for the named input.
  std::unordered_map<
      std::string, std::pair<std::shared_ptr<SystemMemory>, size_t>>
      input_buffer_;
};

//
// Inference input provider that delivers all-zero tensor
// content. This provider is only used internally to replace another
// provider for a request that is cancelled or otherwise doesn't have
// input available.
//
class NULLInferRequestProvider : public InferRequestProvider {
 public:
  explicit NULLInferRequestProvider(const InferRequestHeader& request_header)
      : InferRequestProvider("<NULL>", -1)
  {
    request_header_ = request_header;
  }

  Status GetNextInputContent(
      const std::string& name, const void** content, size_t* content_byte_size,
      bool force_contiguous) override;

 private:
  // A buffer of zero bytes that is used commonly as the NULL input.
  static std::vector<uint8_t> buf_;

  // Mutex to guard buf_
  static std::mutex mu_;
};

//
// Provide support for reporting inference response outputs and
// response meta-data
//
class InferResponseProvider {
 public:
  using SecondaryLabelProvider =
      std::pair<std::string, std::shared_ptr<LabelProvider>>;
  using SecondaryLabelProviderMap =
      std::unordered_map<std::string, SecondaryLabelProvider>;

  InferResponseProvider(
      const InferRequestHeader& request_header,
      const std::shared_ptr<LabelProvider>& label_provider);
  virtual ~InferResponseProvider() = default;

  // Get the full response header for this inference request.
  virtual const InferResponseHeader& ResponseHeader() const = 0;

  // Get a mutuable full response header for this inference request.
  virtual InferResponseHeader* MutableResponseHeader() = 0;

  // Return true if this provider requires a named output.
  bool RequiresOutput(const std::string& name);

  // Get a buffer to store results for a named output. Must be called
  // exactly once for each output that is being returned for the
  // request. The output must be listed in the request header.
  virtual Status AllocateOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) = 0;

  // Get the address and byte-size of an output buffer. Error is
  // returned if the buffer is not already allocated.
  Status OutputBufferContents(
      const std::string& name, const void** content,
      size_t* content_byte_size) const;

  // Get label provider.
  const std::shared_ptr<LabelProvider>& GetLabelProvider() const
  {
    return label_provider_;
  }

  // Get secondary label provider. Return true if the secondary provider for
  // the 'name' is found. False otherwise,
  bool GetSecondaryLabelProvider(
      const std::string& name, SecondaryLabelProvider* provider);

  // Set secondary label provider.
  void SetSecondaryLabelProvider(
      const std::string& name, const SecondaryLabelProvider& provider);

  // Finalize response based on a backend.
  Status FinalizeResponse(const InferenceBackend& is);

 protected:
  struct Output;

  // Check that 'name' is a valid output. If output is to be buffered,
  // allocate space for it and point to that space with 'content'
  Status CheckAndSetIfBufferedOutput(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape, Output** output);

 protected:
  InferRequestHeader request_header_;

  // Map from output name to the InferRequestHeader output information
  // for that output.
  std::unordered_map<std::string, const InferRequestHeader::Output> output_map_;

  // Information about each output.
  struct Output {
    std::string name_;
    std::vector<int64_t> shape_;
    size_t cls_count_;
    void* ptr_;
    size_t byte_size_;

    // Created buffer for non-RAW results
    std::unique_ptr<char[]> buffer_;

    void* release_buffer_;
    void* release_userp_;
  };

  // Ordered list of outputs as they "added" by AllocateOutputBuffer().
  std::vector<Output> outputs_;

  // label provider used to generate classification results.
  std::shared_ptr<LabelProvider> label_provider_;

  // Map from output name to external label provider and name for that provider.
  // This map should only be non-empty if the response provider is for models
  // that doesn't provide labels directly, i.e. ensemble models.
  SecondaryLabelProviderMap secondary_label_provider_map_;
};

//
// Inference response provider for an internal request
//
class InternalInferResponseProvider : public InferResponseProvider {
 public:
  // Create a InternalInferResponseProvider object.
  static Status Create(
      const InferenceBackend& is, const InferRequestHeader& request_header,
      const std::shared_ptr<LabelProvider>& label_provider,
      std::shared_ptr<InternalInferResponseProvider>* infer_provider);

  const InferResponseHeader& ResponseHeader() const override;
  InferResponseHeader* MutableResponseHeader() override;
  Status AllocateOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) override;

  // Retrieve the data buffer of output 'name'.
  Status GetSystemMemory(
      const std::string& name, std::shared_ptr<SystemMemory>* output_buffer);

 private:
  InternalInferResponseProvider(
      const InferRequestHeader& request_header,
      const std::shared_ptr<LabelProvider>& label_provider);

  InferResponseHeader response_header_;
  std::unordered_map<std::string, std::shared_ptr<AllocatedSystemMemory>>
      output_buffer_;
};

//
// Inference response provider for a GRPC request
//
class GRPCInferResponseProvider : public InferResponseProvider {
 public:
  // Initialize based on gRPC request
  static Status Create(
      const InferRequestHeader& request_header, InferResponse* response,
      const std::shared_ptr<LabelProvider>& label_provider,
      std::shared_ptr<GRPCInferResponseProvider>* infer_provider);

  const InferResponseHeader& ResponseHeader() const override;
  InferResponseHeader* MutableResponseHeader() override;
  Status AllocateOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) override;

 private:
  GRPCInferResponseProvider(
      const InferRequestHeader& request_header, InferResponse* response,
      const std::shared_ptr<LabelProvider>& label_provider)
      : InferResponseProvider(request_header, label_provider),
        response_(response)
  {
  }

  InferResponse* response_;
};

//
// Inference response provider for an HTTP request
//
class HTTPInferResponseProvider : public InferResponseProvider {
 public:
  static Status Create(
      evbuffer* output_buffer, const InferenceBackend& is,
      const InferRequestHeader& request_header,
      const std::shared_ptr<LabelProvider>& label_provider,
      std::shared_ptr<HTTPInferResponseProvider>* infer_provider);

  const InferResponseHeader& ResponseHeader() const override;
  InferResponseHeader* MutableResponseHeader() override;
  Status AllocateOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) override;

 private:
  HTTPInferResponseProvider(
      evbuffer* output_buffer, const InferRequestHeader& request_header,
      const std::shared_ptr<LabelProvider>& label_provider);

  InferResponseHeader response_header_;
  evbuffer* output_buffer_;
};

//
// Inference response provider that delegates output buffer allocation
// via a callback.
//
class DelegatingInferResponseProvider : public InferResponseProvider {
 public:
  static Status Create(
      const InferRequestHeader& request_header,
      const std::shared_ptr<LabelProvider>& label_provider,
      TRTSERVER_ResponseAllocator* allocator,
      TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn, void* alloc_userp,
      TRTSERVER_ResponseAllocatorReleaseFn_t release_fn,
      std::shared_ptr<DelegatingInferResponseProvider>* infer_provider);

  ~DelegatingInferResponseProvider();

  const InferResponseHeader& ResponseHeader() const override;
  InferResponseHeader* MutableResponseHeader() override;
  Status AllocateOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape) override;

 private:
  DelegatingInferResponseProvider(
      const InferRequestHeader& request_header,
      const std::shared_ptr<LabelProvider>& label_provider,
      TRTSERVER_ResponseAllocator* allocator,
      TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn, void* alloc_userp,
      TRTSERVER_ResponseAllocatorReleaseFn_t release_fn)
      : InferResponseProvider(request_header, label_provider),
        allocator_(allocator), alloc_fn_(alloc_fn), alloc_userp_(alloc_userp),
        release_fn_(release_fn)
  {
  }

  TRTSERVER_ResponseAllocator* allocator_;
  TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn_;
  void* alloc_userp_;
  TRTSERVER_ResponseAllocatorReleaseFn_t release_fn_;

  InferResponseHeader response_header_;
};

}}  // namespace nvidia::inferenceserver
