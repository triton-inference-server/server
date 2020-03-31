// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#if defined(TRTIS_ENABLE_HTTP) || defined(TRTIS_ENABLE_METRICS)
#include <event2/buffer.h>
#endif
#include "src/core/api.pb.h"
#include "src/core/constants.h"
#include "src/core/grpc_service.pb.h"
#include "src/core/infer_request.h"
#include "src/core/memory.h"
#include "src/core/model_config.h"
#include "src/core/status.h"
#include "src/core/tritonserver.h"
#include "src/core/trtserver.h"

namespace nvidia { namespace inferenceserver {

class InferenceBackend;
class LabelProvider;

//
// Provide inference request inputs and meta-data
//
class InferRequestProvider {
 public:
  static Status Create(
      const std::shared_ptr<InferenceRequest>& irequest,
      std::shared_ptr<InferRequestProvider>* provider);

  // Return the requested model name.
  const std::string& ModelName() const { return irequest_->ModelName(); }

  // Return the requested model version, or -1 if no specific version
  // was requested.
  int64_t ModelVersion() const { return irequest_->RequestedModelVersion(); }

  const std::shared_ptr<InferenceRequest>& Request() const { return irequest_; }

  // Get the next contiguous chunk of bytes for the 'name'd
  // input. Return a pointer to the chunk in 'content'.
  // If there are no more bytes for the input return 'content' == nullptr.
  // 'content_byte_size' acts as both input and output. On input
  // 'content_byte_size' is a hint of the maximum chunk size that
  // should be returned in 'content' and must be non-zero unless no
  // additional input is expected. On return 'content_byte_size' gives
  // the actual size of the chunk pointed to by 'content'.
  // 'memory_type' acts as both input and output. On input 'memory_type'
  // is the buffer memory type preferred by the function caller, it will
  // not affect the function behavior, but it will be propagated to the
  // buffer and the buffer owner may collect such information for other use.
  // On return 'memory_type' gives the actual memory type of the chunk
  // pointed to by 'content'.
  // 'memory_type_id' acts as both input and output. On input 'memory_type_id'
  // is the buffer memory type id preferred by the function caller, it will
  // not affect the function behavior, but it will be propagated to the
  // buffer and the buffer owner may collect such information for other use.
  // On return 'memory_type_id' gives the actual memory type id of the chunk
  // pointed to by 'content'.
  virtual Status GetNextInputContent(
      const std::string& name, const void** content, size_t* content_byte_size,
      TRTSERVER_Memory_Type* memory_type, int64_t* memory_type_id);

  // Retrieve the data buffer of input 'name'. This function will not check
  // input override.
  Status GetMemory(
      const std::string& name, std::shared_ptr<Memory>* input_buffer);

  // Similar to above, but the function caller does not own the Memory object,
  // nor extend its lifetime. This function will check input override.
  Status GetMemoryWithOverride(
      const std::string& name, const Memory** input_buffer);

  // Set content for named inputs. If the input already has content,
  // this content will be used in-place of existing content.
  struct InputOverride {
    std::vector<uint8_t> content_;
    std::vector<int64_t> dims_;
    DataType datatype_;
    // Alternative representation of 'content_' in the form of Memory class
    MemoryReference content_ref_;
  };

  using InputOverrideMap = std::unordered_map<std::string, InputOverride>;
  using InputOverrideMapVec = std::vector<std::shared_ptr<InputOverrideMap>>;
  const InputOverrideMapVec& GetInputOverrides() const;
  Status AddInputOverrides(const std::shared_ptr<InputOverrideMap>& overrides);
  bool HasInputOverride(const std::string& name);
  bool GetInputOverrideShape(
      const std::string& name, std::vector<int64_t>* shape);
  void SetInputOverrideConsumed(const std::string& name, const bool consumed);

 protected:
  explicit InferRequestProvider(
      const std::shared_ptr<InferenceRequest>& irequest)
      : irequest_(irequest)
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

  std::shared_ptr<InferenceRequest> irequest_;

  // Input content overrides. Multiple maps can be provided but a
  // given tensor must not appear in more than one map.
  InputOverrideMapVec overrides_maps_;

  // The inputs that have had their override content consumed by a
  // call to GetInputOverrideContent. A given input override will only
  // return the content once and on subsequent calls will return
  // 'content' == nullptr to indicate that all the override content
  // has been consumed.
  std::set<std::string> overrides_consumed_;
};

//
// Inference input provider that delivers all-zero tensor
// content. This provider is only used internally to replace another
// provider for a request that is cancelled or otherwise doesn't have
// input available.
//
class NULLInferRequestProvider : public InferRequestProvider {
 public:
  explicit NULLInferRequestProvider(
      const std::shared_ptr<InferenceRequest>& irequest)
      : InferRequestProvider(irequest)
  {
  }

  Status GetNextInputContent(
      const std::string& name, const void** content, size_t* content_byte_size,
      TRTSERVER_Memory_Type* memory_type, int64_t* memory_type_id) override;

 private:
  // A buffer of zero bytes that is used commonly as the NULL input.
  static std::vector<uint8_t> buf_;

  // Mutex to guard buf_
  static std::mutex mu_;

  // Record whether an input has been retrieved completely
  std::unordered_map<std::string, size_t> inputs_remaining_bytes_;
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

  static Status Create(
      const std::shared_ptr<InferenceRequest>& irequest,
      const std::shared_ptr<LabelProvider>& label_provider,
      TRTSERVER_ResponseAllocator* allocator,
      TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn, void* alloc_userp,
      TRTSERVER_ResponseAllocatorReleaseFn_t release_fn,
      std::shared_ptr<InferResponseProvider>* infer_provider);

  static Status Create(
      const std::shared_ptr<InferenceRequest>& irequest,
      const std::shared_ptr<LabelProvider>& label_provider,
      TRITONSERVER_ResponseAllocator* allocator,
      TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn, void* alloc_userp,
      TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn,
      std::shared_ptr<InferResponseProvider>* infer_provider);

  ~InferResponseProvider();

  // Get the full response header for this inference request.
  const InferResponseHeader& ResponseHeader() const;

  // Get a mutuable full response header for this inference request.
  InferResponseHeader* MutableResponseHeader();

  // Return true if this provider requires a named output.
  bool RequiresOutput(const std::string& name);

  // Get a buffer to store results for a named output. Must be called
  // exactly once for each output that is being returned for the
  // request. The output must be listed in the request header.
  Status AllocateOutputBuffer(
      const std::string& name, void** content, size_t content_byte_size,
      const std::vector<int64_t>& content_shape,
      const TRTSERVER_Memory_Type preferred_memory_type,
      const int64_t preferred_memory_type_id,
      TRTSERVER_Memory_Type* actual_memory_type,
      int64_t* actual_memory_type_id);

  // Get the address and byte-size of an output buffer. Error is
  // returned if the buffer is not already allocated.
  Status OutputBufferContents(
      const std::string& name, const void** content, size_t* content_byte_size,
      TRTSERVER_Memory_Type* memory_type, int64_t* memory_type_id) const;

  // Get the address and byte-size of an output buffer. Error is
  // returned if the buffer is not already allocated.
  Status OutputBufferContents(
      const std::string& name, const void** content, size_t* content_byte_size,
      TRITONSERVER_Memory_Type* memory_type, int64_t* memory_type_id) const;

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

 private:
  InferResponseProvider(
      const std::shared_ptr<InferenceRequest>& irequest,
      const std::shared_ptr<LabelProvider>& label_provider,
      TRTSERVER_ResponseAllocator* allocator,
      TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn, void* alloc_userp,
      TRTSERVER_ResponseAllocatorReleaseFn_t release_fn);

  InferResponseProvider(
      const std::shared_ptr<InferenceRequest>& irequest,
      const std::shared_ptr<LabelProvider>& label_provider,
      TRITONSERVER_ResponseAllocator* allocator,
      TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn, void* alloc_userp,
      TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn);

  std::shared_ptr<InferenceRequest> irequest_;

  // Map from output name to the InferenceRequest output information
  // for that output.
  std::unordered_map<std::string, const InferenceRequest::RequestedOutput>
      output_map_;

  // Information about each output.
  struct Output {
    std::string name_;
    std::vector<int64_t> shape_;
    size_t cls_count_;
    void* ptr_;
    size_t byte_size_;
    TRTSERVER_Memory_Type memory_type_;
    int64_t memory_type_id_;

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

  TRTSERVER_ResponseAllocator* allocator_;
  TRTSERVER_ResponseAllocatorAllocFn_t alloc_fn_;
  void* alloc_userp_;
  TRTSERVER_ResponseAllocatorReleaseFn_t release_fn_;

  bool using_triton_;
  TRITONSERVER_ResponseAllocator* triton_allocator_;
  TRITONSERVER_ResponseAllocatorAllocFn_t triton_alloc_fn_;
  TRITONSERVER_ResponseAllocatorReleaseFn_t triton_release_fn_;

  InferResponseHeader response_header_;
};

}}  // namespace nvidia::inferenceserver
