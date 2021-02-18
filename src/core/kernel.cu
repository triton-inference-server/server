// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/kernel.h"

#include <cuda.h>

#define THREADBLOCK_SIZE 512
__launch_bounds__(THREADBLOCK_SIZE) __global__ void TritonGatherKernel(
    const int8_t** __restrict input_ptr_buffer,
    const size_t* __restrict byte_size_buffer,
    const size_t* __restrict byte_size_offset_buffer,
    int8_t* __restrict output_buffer)
{
  int request_idx = blockIdx.x;
  int lane_id = threadIdx.x;
  const int8_t* request_input_buffer = input_ptr_buffer[request_idx];
  int byte_size = byte_size_buffer[request_idx];
  int byte_size_offset = byte_size_offset_buffer[request_idx];

  int8_t* output_buffer_with_offset = output_buffer + byte_size_offset;
  if (((byte_size % 4) == 0) && (((uint64_t)request_input_buffer % 4) == 0) &&
      (((uint64_t)output_buffer_with_offset % 4) == 0)) {
    int32_t* input_4 = (int32_t*)request_input_buffer;
    int32_t* output_4 = (int32_t*)output_buffer_with_offset;
    int element_count = byte_size / 4;
    for (int elem_id = lane_id; elem_id < element_count;
         elem_id += THREADBLOCK_SIZE) {
      output_4[elem_id] = input_4[elem_id];
    }
  } else {
    for (int elem_id = lane_id; elem_id < byte_size;
         elem_id += THREADBLOCK_SIZE) {
      output_buffer_with_offset[elem_id] =
          __ldg(request_input_buffer + elem_id);
    }
  }
}

void
RunGatherKernel(
    const int8_t** input_ptr_buffer, const size_t* byte_size_buffer,
    const size_t* byte_size_offset_buffer, int8_t* output_buffer,
    size_t request_count, cudaStream_t stream)
{
  TritonGatherKernel<<<request_count, THREADBLOCK_SIZE, 0, stream>>>(
      input_ptr_buffer, byte_size_buffer, byte_size_offset_buffer,
      output_buffer);
}