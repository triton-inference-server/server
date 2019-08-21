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
#include "src/clients/python/shared_memory_wrapper.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

//==============================================================================
// Error

nic::Error*
ErrorNew(const char* msg)
{
  return new nic::Error(ni::RequestStatusCode::INTERNAL, std::string(msg));
}

//==============================================================================
// SharedMemoryControlContext

nic::Error*
CreateSharedMemoryRegion(
    const char* shm_key, size_t batch_byte_size, int* shm_fd)
{
  // get shared memory region descriptor
  *shm_fd = shm_open(shm_key, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  if (*shm_fd == -1) {
    return ErrorNew(
        ("unable to get shared memory descriptor for: " + std::string(shm_key))
            .c_str());
  }
  // extend shared memory object as by default it's initialized with size 0
  int res = ftruncate(*shm_fd, batch_byte_size);
  if (res == -1) {
    return ErrorNew(
        ("unable to initialize the size for: " + std::string(shm_key)).c_str());
  }

  return nullptr;
}

nic::Error*
OpenSharedMemoryRegion(const char* shm_key, int* shm_fd)
{
  // get shared memory region descriptor
  *shm_fd = shm_open(shm_key, O_RDWR, S_IRUSR | S_IWUSR);
  if (*shm_fd == -1) {
    return ErrorNew(
        ("unable to open the shared memory region: " + std::string(shm_key))
            .c_str());
  }

  return nullptr;
}

nic::Error*
CloseSharedMemoryRegion(int shm_fd)
{
  int tmp = close(shm_fd);
  if (tmp == -1) {
    return ErrorNew(
        ("unable to close the shared memory region: " + std::to_string(shm_fd))
            .c_str());
  }

  return nullptr;
}

nic::Error*
SetSharedMemoryRegionData(
    int shm_fd, size_t offset, size_t batch_byte_size, const void* data)
{
  // map shared memory to process address space
  void* shm_addr =
      mmap(NULL, batch_byte_size, PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (shm_addr == MAP_FAILED) {
    return ErrorNew(
        ("unable to mmap the shared memory region: " + std::to_string(shm_fd))
            .c_str());
  }

  memcpy(shm_addr, data, batch_byte_size);

  return nullptr;
}

nic::Error*
ReadSharedMemoryRegionData(
    int shm_fd, size_t offset, size_t batch_byte_size, const void* data)
{
  // map shared memory to process address space
  void* shm_addr =
      mmap(NULL, batch_byte_size, PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (shm_addr == MAP_FAILED) {
    return ErrorNew(
        ("unable to mmap the shared memory region: " + std::to_string(shm_fd))
            .c_str());
  }

  memcpy(const_cast<void*>(data), shm_addr, batch_byte_size);

  return nullptr;
}

nic::Error*
UnlinkSharedMemoryRegion(const char* shm_key)
{
  int shm_fd = shm_unlink(shm_key);
  if (shm_fd == -1) {
    return ErrorNew(
        ("unable to unlink the shared memory region: " + std::string(shm_key))
            .c_str());
  }
  return nullptr;
}

nic::Error*
UnmapSharedMemory(void* shm_addr, size_t byte_size)
{
  int tmp_fd = munmap(shm_addr, byte_size);
  if (tmp_fd == -1) {
    return ErrorNew("unable to munmap the shared memory region");
  }

  return nullptr;
}

//==============================================================================
