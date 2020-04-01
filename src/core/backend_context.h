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

#include <memory>
#include <string>
#include <vector>
#include "src/core/model_config.h"
#include "src/core/provider.h"
#include "src/core/scheduler.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

class InferenceBackend;

struct InputInfo {
  char* input_buffer_;
  TRTSERVER_Memory_Type memory_type_;
  int64_t memory_type_id_;
  // indirect pinned memory buffers, their locations in 'input_buffer_',
  // and the payloads that are associated with this buffer (for reporting error)
  std::vector<
      std::tuple<std::unique_ptr<AllocatedMemory>, size_t, std::vector<size_t>>>
      indirect_buffers_;
};

struct OutputInfo {
  const char* output_buffer_;
  std::vector<int64_t> output_shape_;
  TRTSERVER_Memory_Type memory_type_;
  int64_t memory_type_id_;
  // indirect pinned memory buffers, the memory references appointing to
  // the destinations in payloads and the payload's index
  std::vector<std::pair<
      std::unique_ptr<AllocatedMemory>,
      std::vector<std::pair<size_t, std::unique_ptr<MutableMemory>>>>>
      indirect_buffers_;
};

struct BackendContext {
 public:
#ifndef TRTIS_ENABLE_GPU
  using cudaStream_t = void*;
#endif  // !TRTIS_ENABLE_GPU

  // GPU device number that indicates that no gpu is available for a
  // context (which is an invalid state since TensorRT requires a
  // GPU).
  static constexpr int NO_GPU_DEVICE = -1;

  // Max batch size value that indicates batching is not supported.
  static constexpr int NO_BATCHING = 0;

  BackendContext(
      const std::string& name, const int gpu_device, const int max_batch_size,
      const bool enable_pinned_input, const bool enable_pinned_output);

  virtual ~BackendContext();

  // Create the CUDA stream for data transfer operations. If 'stream' is
  // nullptr, the stream will be created on 'stream_'. Have no effect if GPU
  // support is disabled.
  Status CreateCudaStream(
      const int cuda_stream_priority = 0, cudaStream_t* stream = nullptr);

  // Run model to execute for one or more requests. This function
  // assumes that it is only called by the single runner thread that
  // is assigned to this context. A non-OK return status indicates
  // an internal error that prevents any of the of requests from
  // completing. If an error is isolate to a single request payload
  // it will be reported in that payload.
  virtual Status Run(
      const InferenceBackend* base,
      std::vector<Scheduler::Payload>* payloads) = 0;

  // Return the contents of a shape tensor. It is the caller's
  // responsibility to call this only for shape tensors that are
  // 1-dimensional, INT32 tensors. A non-OK status indicates that the
  // contents of the tensor could not be peeked.
  virtual Status PeekShapeTensor(
      const InferenceRequest::Input& input, const Scheduler::Payload& payload,
      std::vector<int64_t>* shape);

  // Helper function to batch input data from payloads into 'input_buffer'.
  // 'input_buffer' must be a continuous block that can hold the sum of
  // 'expected_byte_sizes' bytes. On byte size mismatch, the function will
  // set the status of the payload accordingly.
  // Return true if cudaMemcpyAsync is called, and the caller should call
  // cudaStreamSynchronize before using the data. Otherwise, return false.
  bool SetInputBuffer(
      const std::string& name, const std::vector<size_t>& expected_byte_sizes,
      std::vector<Scheduler::Payload>* payloads, InputInfo* input);

  // Overload of SetInputBuffer() which issues the CUDA copies on 'stream'
  // instead of 'stream_'.
  bool SetInputBuffer(
      const std::string& name, const std::vector<size_t>& expected_byte_sizes,
      std::vector<Scheduler::Payload>* payloads, cudaStream_t stream,
      InputInfo* input);

  // Helper function to populate the shape value of specified shape input
  // that corresponds with the batch size. The first shape value is asssumed
  // to be the batch size. Its the user's responsibility to ensure it is called
  // only for the shape tensors.
  // Return true if cudaMemcpyAsync is called, and the caller should call
  // cudaStreamSynchronize before using the data. Otherwise, return false.
  bool SetShapeInputBuffer(
      const std::string& name, const int32_t total_batch_size,
      const int expected_byte_size, const bool support_batching,
      Scheduler::Payload* payload, TRTSERVER_Memory_Type dst_memory_type,
      int64_t dst_memory_type_id, char* input_buffer);

  // Helper function to set output buffer of fixed size data type to
  // payloads Return true if cudaMemcpyAsync is called, and the caller
  // should call cudaStreamSynchronize before using the data. Otherwise,
  // return false.
  bool SetFixedSizeOutputBuffer(
      const std::string& name, const size_t batch1_byte_size,
      OutputInfo* output, std::vector<Scheduler::Payload>* payloads);

  // Helper function to set output buffer Output Shape tensor to payloads. It is
  // callers resposibilty to ensure this method is called only for the shape
  // tensors. Return true if cudaMemcpyAsync is called, and the caller should
  // call cudaStreamSynchronize before using the data. Otherwise, return false.
  bool SetOutputShapeTensorBuffer(
      const std::string& name, const int32_t* content,
      std::vector<int64_t>& content_shape, const bool support_batching,
      TRTSERVER_Memory_Type src_memory_type, int64_t src_memory_type_id,
      std::vector<Scheduler::Payload>* payloads);

  // This function will return the requested input content within a
  // payload in a contiguous chunk. In some cases this will require
  // copying the data. If it happens, 'contiguous_buffer' will be set
  // to hold the contiguous chunk and 'cuda_copy' will be set to
  // indicate whether CUDA copy is conducted.  The data copy can be
  // avoid if the input is already in contiguous chunk and the input
  // is located in memory type and id specified.
  Status GetContiguousInputContent(
      const std::string& name, TRTSERVER_Memory_Type memory_type,
      int64_t memory_type_id, const Scheduler::Payload& payload,
      const char** content, size_t* content_byte_size,
      std::unique_ptr<AllocatedMemory>* contiguous_buffer, bool* cuda_copy);

  // Check if output tensor produced by a model is compatible with the
  // model configuration.  Dimensions with variable size in the model
  // configuration can support any size in the corresponding output
  // tensor dimension.
  //
  // \param supports_batching If True then the configuration expects
  // the model to support batching and so the shape must have the
  // appropriate batch dimension.
  Status CompareOutputDims(
      const std::string& tensor_name, const std::vector<int64_t>& model_shape,
      const DimsList& dims, const bool supports_batching);

  // Meta data for constructing an indirect pinned memory buffer for input
  // <offset in input buffer,
  //  indirect buffer size,
  //  vector of <index of the payload (for status update),
  //             memory block of the provider's input,
  //             index in the memory block>>
  using BufferInfo = std::tuple<
      size_t, size_t, std::vector<std::tuple<size_t, const Memory*, size_t>>>;

  // Meta data for constructing an indirect pinned memory buffer for output
  // <offset in output buffer,
  //  indirect buffer size,
  //  vector of <index of the payload (for status update),
  //             memory block of the provider's output>>
  using OutputBufferInfo = std::tuple<
      size_t, size_t, std::vector<std::pair<size_t, MutableMemory*>>>;

  // Helper function to construct an 'indirect_buffer', and to copy data in
  // 'payloads' to the indirect buffer first, then to copy the indirect buffer
  // to proper location in 'input_buffer', according to 'pinned_buffer_info'.
  bool IssueIndirectInputBufferCopy(
      const std::string& name, const BufferInfo& pinned_buffer_info,
      std::vector<Scheduler::Payload>* payloads, cudaStream_t stream,
      InputInfo* input);

  bool IssueIndirectOutputBufferCopy(
      const std::string& name,
      const BackendContext::OutputBufferInfo& pinned_buffer_info,
      std::vector<Scheduler::Payload>* payloads, cudaStream_t stream,
      OutputInfo* output);

  // Helper function to return whether an indirect buffer is needed in
  // 'need_indirect_buffer', and the memory type that should utilize the
  // indirect buffer in 'candiate_type'.
  void GetIndirectBufferRequirement(
      TRTSERVER_Memory_Type ref_buffer_type, bool is_input,
      TRTSERVER_Memory_Type* candidate_type, bool* need_indirect_buffer);

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

  // The stream where data transfer operations are executed on.
  cudaStream_t stream_;
};

}}  // namespace nvidia::inferenceserver
