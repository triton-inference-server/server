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

#include "src/backends/backend/examples/backend_memory.h"

#include "src/backends/backend/examples/backend_utils.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace nvidia { namespace inferenceserver { namespace backend {

TRITONSERVER_Error*
BackendMemory::Create(
    const TRITONSERVER_MemoryType preferred_memtype, const size_t byte_size,
    BackendMemory** mem)
{
  *mem = nullptr;

  char* ptr = nullptr;
  TRITONSERVER_MemoryType memtype = preferred_memtype;

  // For now only CPU memory types are supported...
  RETURN_ERROR_IF_FALSE(
      (memtype == TRITONSERVER_MEMORY_CPU) ||
          (memtype == TRITONSERVER_MEMORY_CPU_PINNED),
      TRITONSERVER_ERROR_INTERNAL,
      std::string("BackendMemory only supports CPU and CPU_PINNED memory"));

  // If PINNED memory is preferred and GPU is enabled then attempt
  // that memory first.
#ifdef TRITON_ENABLE_GPU
  if (memtype == TRITONSERVER_MEMORY_CPU_PINNED) {
    auto cuerr = cudaHostAlloc((void**)&ptr, byte_size, cudaHostAllocPortable);
    if (cuerr != cudaSuccess) {
      ptr = nullptr;
    }
  }
#endif  // TRITON_ENABLE_GPU

  // Fall-back is non-pinned CPU memory.
  if (ptr == nullptr) {
    memtype = TRITONSERVER_MEMORY_CPU;
    ptr = reinterpret_cast<char*>(malloc(byte_size));
  }

  RETURN_ERROR_IF_TRUE(
      ptr == nullptr, TRITONSERVER_ERROR_INTERNAL,
      std::string("BackendMemory failed to allocate ") +
          std::to_string(byte_size) + " bytes of " +
          TRITONSERVER_MemoryTypeString(memtype));

  *mem = new BackendMemory(memtype, ptr, byte_size);

  return nullptr;  // success
}

BackendMemory::~BackendMemory()
{
  if (memtype_ == TRITONSERVER_MEMORY_CPU) {
    free(buffer_);
  } else if (memtype_ == TRITONSERVER_MEMORY_CPU_PINNED) {
#ifdef TRITON_ENABLE_GPU
    cudaFreeHost(buffer_);
#endif  // TRITON_ENABLE_GPU
  }
}

}}}  // namespace nvidia::inferenceserver::backend
