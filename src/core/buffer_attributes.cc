// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "buffer_attributes.h"

#include <cstring>
#include "src/core/constants.h"

namespace nvidia { namespace inferenceserver {
void
BufferAttributes::SetByteSize(const size_t& byte_size)
{
  byte_size_ = byte_size;
}

void
BufferAttributes::SetMemoryType(const TRITONSERVER_MemoryType& memory_type)
{
  memory_type_ = memory_type;
}

void
BufferAttributes::SetMemoryTypeId(const int64_t& memory_type_id)
{
  memory_type_id_ = memory_type_id;
}

void
BufferAttributes::SetCudaIpcHandle(void* cuda_ipc_handle)
{
  char* lcuda_ipc_handle = reinterpret_cast<char*>(cuda_ipc_handle);
  cuda_ipc_handle_.clear();
  std::copy(
      lcuda_ipc_handle, lcuda_ipc_handle + CUDA_IPC_STRUCT_SIZE,
      std::back_inserter(cuda_ipc_handle_));
}

void*
BufferAttributes::CudaIpcHandle()
{
  if (cuda_ipc_handle_.empty()) {
    return nullptr;
  } else {
    return reinterpret_cast<void*>(cuda_ipc_handle_.data());
  }
}

size_t
BufferAttributes::ByteSize() const
{
  return byte_size_;
}

TRITONSERVER_MemoryType
BufferAttributes::MemoryType() const
{
  return memory_type_;
}

int64_t
BufferAttributes::MemoryTypeId() const
{
  return memory_type_id_;
}

BufferAttributes::BufferAttributes(
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, char* cuda_ipc_handle)
    : byte_size_(byte_size), memory_type_(memory_type),
      memory_type_id_(memory_type)
{
  // cuda ipc handle size
  cuda_ipc_handle_.reserve(CUDA_IPC_STRUCT_SIZE);

  if (cuda_ipc_handle != nullptr) {
    std::copy(
        cuda_ipc_handle, cuda_ipc_handle + CUDA_IPC_STRUCT_SIZE,
        std::back_inserter(cuda_ipc_handle_));
  }
}
}}  // namespace nvidia::inferenceserver
