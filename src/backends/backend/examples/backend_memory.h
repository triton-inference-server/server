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
#include "src/core/tritonserver.h"

namespace nvidia { namespace inferenceserver { namespace backend {

//
// BackendMemory
//
// Utility class for allocating and deallocating memory.
//
class BackendMemory {
 public:
  // Create a memory allocation using the preferred memory type if
  // possible. If not possible allocate memory using the next most
  // appropriate memory type.
  static TRITONSERVER_Error* Create(
      const TRITONSERVER_MemoryType preferred_memtype, const size_t byte_size,
      BackendMemory** mem);
  ~BackendMemory();

  TRITONSERVER_MemoryType MemoryType() const { return memtype_; }
  char* MemoryPtr() { return buffer_; }
  size_t ByteSize() const { return byte_size_; }

 private:
  BackendMemory(
      const TRITONSERVER_MemoryType memtype, char* buffer,
      const size_t byte_size)
      : memtype_(memtype), buffer_(buffer), byte_size_(byte_size)
  {
  }

  TRITONSERVER_MemoryType memtype_;
  char* buffer_;
  size_t byte_size_;
};

}}}  // namespace nvidia::inferenceserver::backend
