// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <string>

static const char*
_cudaGetErrorEnum(cudaError_t error)
{
  return cudaGetErrorName(error);
}

template <typename T>
void
check(T result, char const* const func, const char* const file, int const line)
{
  if (result) {
    fprintf(
        stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
        static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This program is used for determining the GPU devices that cannot enable peer
// access. The output of this program will be used in L0_io. This program is
// based on the deviceQuery in the github.com/NVIDIA/cuda-samples repository.
int
main(int argc, char** argv)
{
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf(
        "cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id),
        cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  if (deviceCount >= 2) {
    cudaDeviceProp prop[64];
    // we want to find the first two GPUs that cannot support P2P
    int gpuid[64];
    int gpu_p2p_count = 0;

    for (int i = 0; i < deviceCount; i++) {
      checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));
      // Only boards based on Fermi or later can support P2P
      if (prop[i].major >= 2) {
        // This is an array of P2P capable GPUs
        gpuid[gpu_p2p_count++] = i;
      }
    }

    if (gpu_p2p_count >= 2) {
      for (int i = 0; i < gpu_p2p_count; i++) {
        for (int j = 0; j < gpu_p2p_count; j++) {
          if (gpuid[i] == gpuid[j]) {
            continue;
          }

          int can_access_peer_i_j = 1, can_access_peer_j_i = 1;
          checkCudaErrors(cudaDeviceCanAccessPeer(
              &can_access_peer_i_j, gpuid[i], gpuid[j]));
          checkCudaErrors(cudaDeviceCanAccessPeer(
              &can_access_peer_j_i, gpuid[j], gpuid[i]));
          if (can_access_peer_i_j == 0 && can_access_peer_j_i == 0) {
            printf("%d %d", gpuid[i], gpuid[j]);
            exit(EXIT_SUCCESS);
          }
        }
      }
    }
  }

  printf(
      "Failed to find a GPU pair that are not capable of performing peer "
      "access.\n");
  exit(EXIT_FAILURE);
}
