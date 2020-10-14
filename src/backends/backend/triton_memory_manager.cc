// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/backend/triton_memory_manager.h"

#include "src/core/pinned_memory_manager.h"
#include "src/core/status.h"
#include "src/core/tritonserver_apis.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#include "src/core/cuda_memory_manager.h"
#endif  // TRITON_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_MemoryManagerAllocate(
    TRITONBACKEND_MemoryManager* manager, void** buffer,
    const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id,
    const uint64_t byte_size)
{
  switch (memory_type) {
    case TRITONSERVER_MEMORY_GPU:
#ifdef TRITON_ENABLE_GPU
    {
      auto status = CudaMemoryManager::Alloc(buffer, byte_size, memory_type_id);
      if (!status.IsOk()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNAVAILABLE, status.Message().c_str());
      }
      break;
    }
#else
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "GPU memory allocation not supported");
#endif  // TRITON_ENABLE_GPU

    case TRITONSERVER_MEMORY_CPU_PINNED:
#ifdef TRITON_ENABLE_GPU
    {
      TRITONSERVER_MemoryType mt = memory_type;
      auto status = PinnedMemoryManager::Alloc(buffer, byte_size, &mt, false);
      if (!status.IsOk()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNAVAILABLE, status.Message().c_str());
      }
      break;
    }
#else
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "Pinned memory allocation not supported");
#endif  // TRITON_ENABLE_GPU

    case TRITONSERVER_MEMORY_CPU: {
      *buffer = malloc(byte_size);
      if (*buffer == nullptr) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNAVAILABLE, "CPU memory allocation failed");
      }
      break;
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_MemoryManagerFree(
    TRITONBACKEND_MemoryManager* manager, void* buffer,
    const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id)
{
  switch (memory_type) {
    case TRITONSERVER_MEMORY_GPU: {
#ifdef TRITON_ENABLE_GPU
      auto status = CudaMemoryManager::Free(buffer, memory_type_id);
      if (!status.IsOk()) {
        return TRITONSERVER_ErrorNew(
            StatusCodeToTritonCode(status.StatusCode()),
            status.Message().c_str());
      }
#endif  // TRITON_ENABLE_GPU
      break;
    }

    case TRITONSERVER_MEMORY_CPU_PINNED: {
#ifdef TRITON_ENABLE_GPU
      auto status = PinnedMemoryManager::Free(buffer);
      if (!status.IsOk()) {
        return TRITONSERVER_ErrorNew(
            StatusCodeToTritonCode(status.StatusCode()),
            status.Message().c_str());
      }
#endif  // TRITON_ENABLE_GPU
      break;
    }

    case TRITONSERVER_MEMORY_CPU:
      free(buffer);
      break;
  }

  return nullptr;  // success
}

}  // extern C

}}  // namespace nvidia::inferenceserver
