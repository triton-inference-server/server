// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "src/clients/python/shared_memory/shared_memory.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include "src/clients/python/shared_memory/shared_memory_handle.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

//==============================================================================
// Error

nic::Error*
ErrorNew(const char* msg)
{
  return new nic::Error(ni::RequestStatusCode::INTERNAL, std::string(msg));
}

void
ErrorDelete(nic::Error* ctx)
{
  delete ctx;
}

bool
ErrorIsOk(nic::Error* ctx)
{
  return ctx->IsOk();
}

const char*
ErrorMessage(nic::Error* ctx)
{
  return ctx->Message().c_str();
}

const char*
ErrorServerId(nic::Error* ctx)
{
  return ctx->ServerId().c_str();
}

uint64_t
ErrorRequestId(nic::Error* ctx)
{
  return ctx->RequestId();
}

//==============================================================================
// SharedMemoryControlContext

void*
SharedMemoryHandleCreate(void* shm_addr, std::string shm_key, int shm_fd)
{
  SharedMemoryHandle* handle = new SharedMemoryHandle();
  handle->base_addr_ = shm_addr;
  handle->shm_key_ = shm_key;
  handle->shm_fd_ = shm_fd;
  std::cout << handle << '\n';
  return reinterpret_cast<void*>(handle);
}

nic::Error
SharedMemoryRegionMap(
    int shm_fd, size_t offset, size_t byte_size, void** shm_addr)
{
  // map shared memory to process address space
  *shm_addr = mmap(NULL, byte_size, PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (*shm_addr == MAP_FAILED) {
    return nic::Error(
        ni::RequestStatusCode::INVALID_ARG,
        ("unable to read/mmap the shared memory region: " +
         std::to_string(shm_fd))
            .c_str());
  }

  return nic::Error::Success;
}

nic::Error*
SharedMemoryRegionCreate(
    const char* shm_key, size_t byte_size, void** shm_handle)
{
  // get shared memory region descriptor
  int shm_fd = shm_open(shm_key, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  if (shm_fd == -1) {
    return ErrorNew(
        ("unable to get shared memory descriptor for: " + std::string(shm_key))
            .c_str());
  }
  // extend shared memory object as by default it's initialized with size 0
  int res = ftruncate(shm_fd, byte_size);
  if (res == -1) {
    return ErrorNew(
        ("unable to initialize the size for: " + std::string(shm_key)).c_str());
  }

  void* shm_addr = nullptr;
  nic::Error err = SharedMemoryRegionMap(shm_fd, 0, byte_size, &shm_addr);
  if (err.IsOk()) {
    return new nic::Error(err);
  }

  *shm_handle =
      SharedMemoryHandleCreate(shm_addr, std::string(shm_key), shm_fd);

  return nullptr;
}

nic::Error*
SharedMemoryRegionSet(
    void* shm_handle, size_t offset, size_t byte_size, const void* data)
{
  std::cout << shm_handle << '\n';
  void* shm_addr =
      reinterpret_cast<SharedMemoryHandle*>(shm_handle)->base_addr_;
  char* shm_addr_offset = reinterpret_cast<char*>(shm_addr);
  memcpy(shm_addr_offset + offset, data, byte_size);
  return nullptr;
}

nic::Error*
SharedMemoryRegionDestroy(const char* shm_key)
{
  int shm_fd = shm_unlink(shm_key);
  if (shm_fd == -1) {
    return ErrorNew(
        ("unable to unlink the shared memory region: " + std::string(shm_key))
            .c_str());
  }
  return nullptr;
}

//==============================================================================
