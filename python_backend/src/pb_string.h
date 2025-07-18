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

#pragma once

#include "shm_manager.h"

namespace triton { namespace backend { namespace python {

struct StringShm {
  bi::managed_external_buffer::handle_t data;
  size_t length;
};

class PbString {
 public:
  static std::unique_ptr<PbString> Create(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      const std::string& string);
  static std::unique_ptr<PbString> Create(
      const std::string& string, char* data_shm,
      bi::managed_external_buffer::handle_t handle);
  static std::unique_ptr<PbString> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);
  static std::unique_ptr<PbString> LoadFromSharedMemory(
      bi::managed_external_buffer::handle_t handle, char* data_shm);
  static std::size_t ShmStructSize(const std::string& string);

  char* MutableString() { return string_shm_ptr_; }
  std::string String()
  {
    return std::string(
        string_shm_ptr_, string_shm_ptr_ + string_container_shm_ptr_->length);
  }
  bi::managed_external_buffer::handle_t ShmHandle();
  std::size_t Size();

 private:
  AllocatedSharedMemory<StringShm> string_container_shm_;
  StringShm* string_container_shm_ptr_;

  AllocatedSharedMemory<char> string_shm_;
  char* string_shm_ptr_;

  bi::managed_external_buffer::handle_t string_handle_;

  PbString(
      AllocatedSharedMemory<StringShm>& string_container_shm,
      AllocatedSharedMemory<char>& string_shm);

  PbString(
      StringShm* string_container_shm, char* string_shm,
      bi::managed_external_buffer::handle_t handle);
};

}}}  // namespace triton::backend::python
