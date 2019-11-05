// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/clients/c++/library/request_grpc.h"
#include "src/clients/c++/library/request_http.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

namespace nvidia { namespace inferenceserver { namespace client {

// Create a shared memory region of the size 'byte_size' and return the unique
// identifier.
// \param shm_key The string identifier of the shared memory region
// \param byte_size The size in bytes of the shared memory region
// \param shm_id Returns an int descriptor of the created shared memory region
// \return error Returns if an error if unable to open shared memory region.
nic::Error CreateSharedMemoryRegion(
    std::string shm_key, size_t byte_size, int* shm_fd);

// Mmap the shared memory region with the given 'offset' and 'byte_size' and
// return the base address of the region.
// \param shm_id The int descriptor of the created shared memory region
// \param offset The offset of the shared memory block from the start of the
// shared memory region
// \param byte_size The size in bytes of the shared memory region
// \param shm_addr Returns the base address of the shared memory region
// \return error Returns if an error if unable to mmap shared memory region.
nic::Error MapSharedMemory(
    int shm_fd, size_t offset, size_t byte_size, void** shm_addr);

// Destory the shared memory region with the given name.
// \return error Returns if an error if unable to unlink shared memory region.
nic::Error UnlinkSharedMemoryRegion(std::string shm_key);

// Munmap the shared memory region from the base address with the given
// byte_size.
// \return error Returns if an error if unable to unmap shared memory region.
nic::Error UnmapSharedMemory(void* shm_addr, size_t byte_size);

}}}  // namespace nvidia::inferenceserver::client
