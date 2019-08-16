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
#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include <time.h>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__device__ long store_now[1];

__global__ void
BusyLoopKernel(const int* num_delay_cycles, int* out)
{
  // As shown in
  // https://stackoverflow.com/questions/11217117/equivalent-of-usleep-in-cuda-kernel
  clock_t start = clock();

  for (;;) {
    clock_t now = clock();
    // Adjust for overflow
    clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
    if (cycles >= num_delay_cycles[0]) {
      break;
    }
    // Prevent nvcc optimizations
    store_now[0] = cycles;
  }
}

void
BusyLoopKernelLauncher(
    const Eigen::GpuDevice& device, const int* num_delay_cycles, int* out)
{
  auto stream = device.stream();
  BusyLoopKernel<<<1, 256, 0, stream>>>(num_delay_cycles, out);
}

#endif
