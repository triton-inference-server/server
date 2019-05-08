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

#include <cuda.h>
#include <stdint.h>

namespace nvidia { namespace inferenceserver { namespace custom {
namespace addsub {

__global__ void
VecAddInt32(int32_t* in0, int32_t* in1, int32_t* out, int cnt)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cnt) {
    out[tid] = in0[tid] + in1[tid];
  }
}

__global__ void
VecAddFp32(float* in0, float* in1, float* out, int cnt)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cnt) {
    out[tid] = in0[tid] + in1[tid];
  }
}

__global__ void
VecSubInt32(int32_t* in0, int32_t* in1, int32_t* out, int cnt)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cnt) {
    out[tid] = in0[tid] - in1[tid];
  }
}

__global__ void
VecSubFp32(float* in0, float* in1, float* out, int cnt)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cnt) {
    out[tid] = in0[tid] - in1[tid];
  }
}

}}}}  // namespace nvidia::inferenceserver::custom::addsub
