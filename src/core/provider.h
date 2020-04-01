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
