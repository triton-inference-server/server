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

#include "pb_string.h"

namespace triton { namespace backend { namespace python {

std::unique_ptr<PbString>
PbString::Create(
    std::unique_ptr<SharedMemoryManager>& shm_pool, const std::string& string)
{
  AllocatedSharedMemory<StringShm> string_container_shm =
      shm_pool->Construct<StringShm>();
  string_container_shm.data_->length = string.size();

  AllocatedSharedMemory<char> string_shm =
      shm_pool->Construct<char>(string.size());
  std::memcpy(string_shm.data_.get(), string.data(), string.size());
  string_container_shm.data_->data = string_shm.handle_;

  return std::unique_ptr<PbString>(
      new PbString(string_container_shm, string_shm));
}

std::unique_ptr<PbString>
PbString::Create(
    const std::string& string, char* data_shm,
    bi::managed_external_buffer::handle_t handle)
{
  StringShm* string_container_shm = reinterpret_cast<StringShm*>(data_shm);
  string_container_shm->length = string.size();

  char* string_shm = data_shm + sizeof(StringShm);
  std::memcpy(string_shm, string.data(), string.size());

  return std::unique_ptr<PbString>(
      new PbString(string_container_shm, string_shm, handle));
}

std::unique_ptr<PbString>
PbString::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<StringShm> string_container_shm =
      shm_pool->Load<StringShm>(handle);
  AllocatedSharedMemory<char> string_shm =
      shm_pool->Load<char>(string_container_shm.data_->data);

  return std::unique_ptr<PbString>(
      new PbString(string_container_shm, string_shm));
}

std::unique_ptr<PbString>
PbString::LoadFromSharedMemory(
    bi::managed_external_buffer::handle_t handle, char* data_shm)
{
  StringShm* string_container_shm = reinterpret_cast<StringShm*>(data_shm);
  char* string_shm = data_shm + sizeof(StringShm);

  return std::unique_ptr<PbString>(
      new PbString(string_container_shm, string_shm, handle));
}

PbString::PbString(
    AllocatedSharedMemory<StringShm>& string_container_shm,
    AllocatedSharedMemory<char>& string_shm)
    : string_container_shm_(std::move(string_container_shm)),
      string_shm_(std::move(string_shm))
{
  string_shm_ptr_ = string_shm_.data_.get();
  string_container_shm_ptr_ = string_container_shm_.data_.get();
  string_handle_ = string_container_shm_.handle_;
}

PbString::PbString(
    StringShm* string_container_shm, char* string_shm,
    bi::managed_external_buffer::handle_t handle)
{
  string_shm_ptr_ = string_shm;
  string_container_shm_ptr_ = string_container_shm;
  string_handle_ = handle;
}

bi::managed_external_buffer::handle_t
PbString::ShmHandle()
{
  return string_handle_;
}

std::size_t
PbString::ShmStructSize(const std::string& string)
{
  return string.size() + sizeof(StringShm);
}

std::size_t
PbString::Size()
{
  return string_container_shm_ptr_->length + sizeof(StringShm);
}

}}}  // namespace triton::backend::python
