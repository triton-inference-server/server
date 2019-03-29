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

#if GOOGLE_CUDA

#include <algorithm>
#include "cuda/include/cuda.h"
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"

__global__ void
TRTISExampleAddSubFloat(
    const float* in0, const float* in1, float* sum, float* diff, int cnt)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cnt) {
    sum[tid] = in0[tid] + in1[tid];
    diff[tid] = in0[tid] - in1[tid];
  }
}

void
LaunchTRTISExampleAddSubFloat(
    const float* in0, const float* in1, float* sum, float* diff,
    int element_cnt, const Eigen::GpuDevice& device)
{
  const int block_size = std::min(element_cnt, 1024);
  const int grid_size = (element_cnt + block_size - 1) / block_size;

  TRTISExampleAddSubFloat<<<grid_size, block_size, 0, device.stream()>>>(
      in0, in1, sum, diff, element_cnt);
}

#endif  // GOOGLE_CUDA
