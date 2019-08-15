// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include <vector>
#include "src/core/scheduler.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

struct BackendContext {
  // GPU device number that indicates that no gpu is available for a
  // context (which is an invalid state since TensorRT requires a
  // GPU).
  static constexpr int NO_GPU_DEVICE = -1;

  // Max batch size value that indicates batching is not supported.
  static constexpr int NO_BATCHING = 0;

  BackendContext(
      const std::string& name, const int gpu_device, const int max_batch_size);

  virtual ~BackendContext() = default;

  // Helper function to batch input data from payloads into 'input_buffer'.
  // 'input_buffer' must be a continuous block that can hold the sum of
  // 'expected_byte_sizes' bytes. On byte size mismatch, the function will
  // set the status of the payload accordingly.
  // Return true if cudaMemcpyAsync is called, and the caller should call
  // cudaStreamSynchronize before using the data. Otherwise, return false.
  bool SetInputBuffer(
      const std::string& name, const std::vector<size_t>& expected_byte_sizes,
      std::vector<Scheduler::Payload>* payloads,
      TRTSERVER_Memory_Type dst_memory_type, char* input_buffer);

  // Helper function to set output buffer of fixed size data type to payloads
  // Return true if cudaMemcpyAsync is called, and the caller should call
  // cudaStreamSynchronize before using the data. Otherwise, return false.
  bool SetFixedSizeOutputBuffer(
      const std::string& name, const size_t batch1_byte_size,
      const char* content, const std::vector<int64_t>& content_shape,
      TRTSERVER_Memory_Type src_memory_type,
      std::vector<Scheduler::Payload>* payloads);

  // Name of the model instance
  std::string name_;

  // The GPU index active when this context was created.
  int gpu_device_;

  // Maximum batch size to allow. This is the minimum of what is
  // supported by the model and what is requested in the
  // configuration.
  int max_batch_size_;

#ifdef TRTIS_ENABLE_GPU
  // The stream where data transfer operations are executed.
  // nullptr if the context is created on CPU.
  cudaStream_t stream_;
#endif  // TRTIS_ENABLE_GPU
};

}}  // namespace nvidia::inferenceserver
