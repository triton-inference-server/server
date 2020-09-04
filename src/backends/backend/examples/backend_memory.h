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
#pragma once

#include <string>
#include "triton/common/tritonbackend.h"
#include "triton/common/tritonserver.h"

namespace triton { namespace backend {

//
// BackendMemory
//
// Utility class for allocating and deallocating memory using
// TRITONBACKEND_MemoryManager.
//
class BackendMemory {
 public:
  // Create a memory allocation using the specified memory type. See
  // TRITONBACKEND_MemoryManagerAllocate for explanation of the
  // returned error codes.
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_MemoryManager* manager,
      const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id,
      const size_t byte_size, BackendMemory** mem);

  // Create a memory allocation using the preferred memory type if
  // possible. If not possible fallback to allocate memory using CPU
  // memory. See TRITONBACKEND_MemoryManagerAllocate for explanation
  // of the returned error codes.
  static TRITONSERVER_Error* CreateWithFallback(
      TRITONBACKEND_MemoryManager* manager,
      const TRITONSERVER_MemoryType preferred_memory_type,
      const int64_t memory_type_id, const size_t byte_size,
      BackendMemory** mem);

  ~BackendMemory();

  TRITONSERVER_MemoryType MemoryType() const { return memtype_; }
  int64_t MemoryTypeId() const { return memtype_id_; }
  char* MemoryPtr() { return buffer_; }
  size_t ByteSize() const { return byte_size_; }

 private:
  BackendMemory(
      TRITONBACKEND_MemoryManager* manager,
      const TRITONSERVER_MemoryType memtype, const int64_t memtype_id,
      char* buffer, const size_t byte_size)
      : manager_(manager), memtype_(memtype), memtype_id_(memtype_id),
        buffer_(buffer), byte_size_(byte_size)
  {
  }

  TRITONBACKEND_MemoryManager* manager_;
  TRITONSERVER_MemoryType memtype_;
  int64_t memtype_id_;
  char* buffer_;
  size_t byte_size_;
};

}}  // namespace triton::backend
