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

#include <vector>
#include "src/core/tritonserver_apis.h"

#pragma once

namespace nvidia { namespace inferenceserver {
//
// A class to hold information about the buffer allocation.
//
class BufferAttributes {
 public:
  BufferAttributes(
      size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id, char cuda_ipc_handle[64]);
  BufferAttributes()
  {
    memory_type_ = TRITONSERVER_MEMORY_CPU;
    memory_type_id_ = 0;
    cuda_ipc_handle_.reserve(64);
  }

  // Set the buffer byte size
  void SetByteSize(const size_t& byte_size);

  // Set the buffer memory_type
  void SetMemoryType(const TRITONSERVER_MemoryType& memory_type);

  // Set the buffer memory type id
  void SetMemoryTypeId(const int64_t& memory_type_id);

  // Set the cuda ipc handle
  void SetCudaIpcHandle(void* cuda_ipc_handle);

  // Get the cuda ipc handle
  void* CudaIpcHandle();

  // Get the byte size
  size_t ByteSize() const;

  // Get the memory type
  TRITONSERVER_MemoryType MemoryType() const;

  // Get the memory type id
  int64_t MemoryTypeId() const;

 private:
  size_t byte_size_;
  TRITONSERVER_MemoryType memory_type_;
  int64_t memory_type_id_;
  std::vector<char> cuda_ipc_handle_;
};
}}  // namespace nvidia::inferenceserver
