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
#include <memory>
#include <string>
#include <vector>
// FIXME need memory utilties
// #include "src/core/memory.h"
#include "src/backends/backend/tritonbackend.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace nvidia { namespace inferenceserver { namespace backend {

#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
using cudaEvent_t = void*;
#endif  // !TRITON_ENABLE_GPU

//
// ModelInstance
//
class ModelInstance {
 public:
  // GPU device number that indicates that no gpu is available for a
  // context (which is an invalid state since TensorRT requires a
  // GPU).
  static constexpr int NO_GPU_DEVICE = -1;

  // Max batch size value that indicates batching is not supported.
  static constexpr int NO_BATCHING = 0;

  ModelInstance(
      const std::string& name, const int gpu_device, const int max_batch_size,
      const bool enable_pinned_input, const bool enable_pinned_output);

  virtual ~ModelInstance();

  // Create the CUDA stream for data transfer operations. If 'stream' is
  // nullptr, the stream will be created on 'stream_'. Have no effect if GPU
  // support is disabled.
  TRITONSERVER_Error* CreateCudaStream(
      const int cuda_stream_priority = 0, cudaStream_t* stream = nullptr);

  // Run model to execute one or more requests. This function assumes
  // that it is only called by the single runner thread that is
  // assigned to this context. This function takes ownership of
  // 'requests' and is responsible for generating responses and
  // releasing the requests.
  virtual void Run(
      TRITONBACKEND_Model* model, TRITONBACKEND_Request** requests,
      const uint32_t request_count, const uint64_t exec_start_ns) = 0;

  // Name of the model instance
  std::string name_;

  // The GPU index active when this context was created.
  const int gpu_device_;

  // Maximum batch size to allow. This is the minimum of what is
  // supported by the model and what is requested in the
  // configuration.
  const int max_batch_size_;

  // Whether to use indirect pinned buffer for the corresponding data copy type.
  const bool enable_pinned_input_;
  const bool enable_pinned_output_;

  // The stream that executes data transfer operations.
  cudaStream_t stream_;
};

//
// ModelResponder
//
class ModelResponder {
 public:
  // The caller can optionally provide 'event' for internal synchronization
  // instead of using 'stream'.
  explicit ModelResponder(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses, const int max_batch_size,
      const bool pinned_enabled, cudaStream_t stream,
      cudaEvent_t event = nullptr)
      : need_sync_(false), requests_(requests), request_count_(request_count),
        responses_(responses), max_batch_size_(max_batch_size),
        pinned_enabled_(pinned_enabled), stream_(stream), event_(event),
        pending_pinned_byte_size_(0)
  {
  }

  ~ModelResponder();

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
  const bool pinned_enabled_;
  cudaStream_t stream_;
  cudaEvent_t event_;

  using ResponsesList =
      std::list<std::pair<TRITONBACKEND_Response**, OutputData>>;

  size_t pending_pinned_byte_size_;
  size_t pending_pinned_offset_;
  ResponsesList pending_pinned_outputs_;

  // Pinned memories that need to live over the lifetime of this
  // ModelResponder object.
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

//
// ModelInputCollector
//
class ModelInputCollector {
 public:
  // The caller can optionally provide 'event' for internal synchronization
  // instead of using 'stream'.
  explicit ModelInputCollector(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      const bool pinned_enabled, cudaStream_t stream,
      cudaEvent_t event = nullptr)
      : need_sync_(false), requests_(requests), request_count_(request_count),
        responses_(responses), pinned_enabled_(pinned_enabled), stream_(stream),
        event_(event), pending_pinned_byte_size_(0)
  {
  }

  ~ModelInputCollector();

  // Process all requests for a named input tensor.
  void ProcessTensor(
      const char* input_name, char* buffer, const size_t buffer_byte_size,
      const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id);

  // Finalize processing of all requests for all input tensors. Return
  // true if cudaMemcpyAsync is called, and the caller should call
  // should call cudaStreamSynchronize (or cudaEventSynchronize on 'event')
  // before using the data.
  bool Finalize();

 private:
  bool FlushPendingPinned(
      char* tensor_buffer, const size_t tensor_buffer_byte_size,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id);
  bool SetFixedSizeInputTensor(
      TRITONBACKEND_Input* request_input, const size_t tensor_buffer_offset,
      char* tensor_buffer, const size_t tensor_buffer_byte_size,
      const TRITONSERVER_MemoryType tensor_memory_type,
      const int64_t tensor_memory_type_id,
      const TRITONSERVER_MemoryType use_pinned_memory_type,
      TRITONBACKEND_Response** response);

  bool need_sync_;
  TRITONBACKEND_Request** requests_;
  const uint32_t request_count_;
  std::vector<TRITONBACKEND_Response*>* responses_;
  const bool pinned_enabled_;
  cudaStream_t stream_;
  cudaEvent_t event_;

  using RequestsList =
      std::list<std::pair<TRITONBACKEND_Response**, TRITONBACKEND_Input*>>;

  size_t pending_pinned_byte_size_;
  size_t pending_pinned_offset_;
  RequestsList pending_pinned_inputs_;

  // FIXME provide memory utilities to avoid calls to CUDA memory functions
  // Pinned memories that need to live over the lifetime of this
  // ModelInputCollector object.
  std::list<char*> pinned_memories_;

  // Pinned memory buffers and the corresponding request_inputs where
  // the final copy to the tensor is deferred until Finalize() after
  // waiting for all in-flight copies.
  struct DeferredPinned {
    DeferredPinned(
        char* pinned_memory, const size_t pinned_memory_size,
        char* tensor_buffer, const size_t tensor_buffer_offset,
        const TRITONSERVER_MemoryType tensor_memory_type,
        const int64_t tensor_memory_id, RequestsList&& requests)
        : pinned_memory_(pinned_memory),
          pinned_memory_size_(pinned_memory_size),
          tensor_buffer_(tensor_buffer),
          tensor_buffer_offset_(tensor_buffer_offset),
          tensor_memory_type_(tensor_memory_type),
          tensor_memory_id_(tensor_memory_id), requests_(std::move(requests))
    {
    }

    // Holding reference to the pinned memory buffer, which is managed
    // by ModelInputCollector as 'pinned_memory'
    char* pinned_memory_;
    const size_t pinned_memory_size_;
    char* tensor_buffer_;
    const size_t tensor_buffer_offset_;
    const TRITONSERVER_MemoryType tensor_memory_type_;
    const int64_t tensor_memory_id_;
    RequestsList requests_;
  };

  std::list<DeferredPinned> deferred_pinned_;
};

}}}  // namespace nvidia::inferenceserver::backend
