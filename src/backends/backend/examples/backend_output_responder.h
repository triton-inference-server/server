// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <list>
#include <string>
#include <vector>
#include "triton/common/tritonbackend.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace backend {

#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
using cudaEvent_t = void*;
#endif  // !TRITON_ENABLE_GPU

//
// BackendOutputResponder
//
class BackendOutputResponder {
 public:
  // The caller can optionally provide 'event' for internal synchronization
  // instead of using 'stream'.
  explicit BackendOutputResponder(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses, const int max_batch_size,
      TRITONBACKEND_MemoryManager* memory_manager, const bool pinned_enabled,
      cudaStream_t stream, cudaEvent_t event = nullptr)
      : need_sync_(false), requests_(requests), request_count_(request_count),
        responses_(responses), max_batch_size_(max_batch_size),
        memory_manager_(memory_manager), pinned_enabled_(pinned_enabled),
        stream_(stream), event_(event), pending_pinned_byte_size_(0)
  {
  }

  ~BackendOutputResponder();

  // Process all responses for a named output tensor.
  void ProcessTensor(
      const std::string& name, const TRITONSERVER_DataType datatype,
      std::vector<int64_t>& batchn_shape, const char* buffer,
      const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id);

  // Finalize processing of all responses for all output
  // tensors. Return true if cudaMemcpyAsync is called, and the caller
  // should call cudaStreamSynchronize (or cudaEventSynchronize on 'event')
  // before using the data.
  bool Finalize();

 private:
  bool FlushPendingPinned(
      const char* tensor_buffer,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id);
  bool SetFixedSizeOutputBuffer(
      TRITONBACKEND_Response** response, TRITONBACKEND_Output* response_output,
      const std::string& output_name, const size_t tensor_byte_size,
      const size_t tensor_offset, const char* tensor_buffer,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id,
      const TRITONSERVER_MemoryType use_pinned_memory_type);

  struct OutputData {
    OutputData(
        const std::string& name, void* buffer, const size_t buffer_byte_size,
        const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id)
        : name_(name), buffer_(buffer), buffer_byte_size_(buffer_byte_size),
          memory_type_(memory_type), memory_type_id_(memory_type_id)
    {
    }
    const std::string name_;
    void* buffer_;
    const size_t buffer_byte_size_;
    const TRITONSERVER_MemoryType memory_type_;
    const int64_t memory_type_id_;
  };

  bool need_sync_;
  TRITONBACKEND_Request** requests_;
  const uint32_t request_count_;
  std::vector<TRITONBACKEND_Response*>* responses_;
  const int max_batch_size_;
  TRITONBACKEND_MemoryManager* memory_manager_;
  const bool pinned_enabled_;
  cudaStream_t stream_;
  cudaEvent_t event_;

  using ResponsesList =
      std::list<std::pair<TRITONBACKEND_Response**, OutputData>>;

  size_t pending_pinned_byte_size_;
  size_t pending_pinned_offset_;
  ResponsesList pending_pinned_outputs_;

  // Pinned memories that need to live over the lifetime of this
  // BackendOutputResponder object.
  std::list<char*> pinned_memories_;

  // Pinned memory buffers and the corresponding response outputs
  // where the final copy to the response is deferred until Finalize()
  // after waiting for all in-flight copies.
  struct DeferredPinned {
    DeferredPinned(
        char* pinned_memory, const size_t pinned_memory_size,
        ResponsesList&& responses)
        : pinned_memory_(pinned_memory),
          pinned_memory_size_(pinned_memory_size),
          responses_(std::move(responses))
    {
    }
    char* pinned_memory_;
    const size_t pinned_memory_size_;
    ResponsesList responses_;
  };

  std::list<DeferredPinned> deferred_pinned_;
};

}}  // namespace triton::backend
