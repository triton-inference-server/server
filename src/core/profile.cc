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

#include "src/core/profile.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

Status
ProfileStartAll()
{
#ifdef TRTIS_ENABLE_GPU
  int dcnt;
  cudaError_t cuerr = cudaGetDeviceCount(&dcnt);
  if ((cuerr == cudaErrorNoDevice) || (cuerr == cudaErrorInsufficientDriver)) {
    dcnt = 0;
  } else if (cuerr != cudaSuccess) {
    return Status(
        RequestStatusCode::INTERNAL,
        "failed to get device count for profiling: " +
            std::string(cudaGetErrorString(cuerr)));
  }

  for (int i = 0; i < dcnt; i++) {
    cuerr = cudaSetDevice(i);
    if (cuerr != cudaSuccess) {
      return Status(
          RequestStatusCode::INTERNAL,
          "failed to set device for profiling: " +
              std::string(cudaGetErrorString(cuerr)));
    }

    cuerr = cudaProfilerStart();
    if (cuerr != cudaSuccess) {
      return Status(
          RequestStatusCode::INTERNAL,
          "failed to start profiling: " +
              std::string(cudaGetErrorString(cuerr)));
    }
  }
#endif  // TRTIS_ENABLE_GPU

  return Status::Success;
}

Status
ProfileStopAll()
{
#ifdef TRTIS_ENABLE_GPU
  int dcnt;
  cudaError_t cuerr = cudaGetDeviceCount(&dcnt);
  if ((cuerr == cudaErrorNoDevice) || (cuerr == cudaErrorInsufficientDriver)) {
    dcnt = 0;
  } else if (cuerr != cudaSuccess) {
    return Status(
        RequestStatusCode::INTERNAL,
        "failed to get device count for profiling: " +
            std::string(cudaGetErrorString(cuerr)));
  }

  for (int i = 0; i < dcnt; i++) {
    cuerr = cudaSetDevice(i);
    if (cuerr != cudaSuccess) {
      return Status(
          RequestStatusCode::INTERNAL,
          "failed to set device for profiling: " +
              std::string(cudaGetErrorString(cuerr)));
    }

    cuerr = cudaProfilerStop();
    if (cuerr != cudaSuccess) {
      return Status(
          RequestStatusCode::INTERNAL,
          "failed to stop profiling: " +
              std::string(cudaGetErrorString(cuerr)));
    }
  }
#endif  // TRTIS_ENABLE_GPU

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
