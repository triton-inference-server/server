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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "src/clients/c++/examples/shm_utils.h"


// namespace ni = nvidia::inferenceserver;
// namespace nic = nvidia::inferenceserver::client;

namespace nvidia { namespace inferenceserver { namespace client {

int
CreateSharedMemoryRegion(std::string shm_key, size_t byte_size)
{
  // get shared memory region descriptor
  int shm_fd = shm_open(shm_key.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  if (shm_fd == -1) {
    std::cerr << "error: unable to get shared memory descriptor for "
                 "shared-memory key '" +
                     shm_key + "'"
              << std::endl;
    exit(1);
  }
  // extend shared memory object as by default it's initialized with size 0
  int res = ftruncate(shm_fd, byte_size);
  if (res == -1) {
    std::cerr << "error: unable to initialize shared-memory key '" + shm_key +
                     "' to requested size " + std::to_string(byte_size) +
                     " bytes"
              << std::endl;
    exit(1);
  }
  return shm_fd;
}

void*
MapSharedMemory(int shm_fd, size_t offset, size_t byte_size)
{
  // map shared memory to process address space
  void* shm_addr =
      mmap(NULL, byte_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (shm_addr == MAP_FAILED) {
    std::cerr << "error: unable to process address space or shared-memory "
                 "descriptor: " +
                     std::to_string(shm_fd)
              << std::endl;
    exit(1);
  }
  return shm_addr;
}

void
UnlinkSharedMemoryRegion(std::string shm_key)
{
  int shm_fd = shm_unlink(shm_key.c_str());
  if (shm_fd == -1) {
    std::cerr << "error: unable to unlink shared memory for key '" + shm_key +
                     "'"
              << std::endl;
    exit(1);
  }
}

void
UnmapSharedMemory(void* shm_addr, size_t byte_size)
{
  int tmp_fd = munmap(shm_addr, byte_size);
  if (tmp_fd == -1) {
    std::cerr << "Unable to munmap shared memory region" << std::endl;
    exit(1);
  }
}

}}}  // namespace nvidia::inferenceserver::client
