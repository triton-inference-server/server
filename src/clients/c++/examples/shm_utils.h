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


// namespace nvidia { namespace inferenceserver {

/// \return true if a TensorFlow shape exactly matches a model
/// configuration shape. Dimensions with variable size are represented
/// by -1 in both the TensorFlow shape and the model configuration
/// shape and these must match as well.
/// \param supports_batching If True then the configuration expects
/// the model to support batching and so the shape must have the
/// appropriate batch dimension.


// }}  // namespace nvidia::inferenceserver

// Create a shared memory region of the size 'byte_size' and return the unique
// identifier.
int CreateSharedMemoryRegion(std::string shm_key, size_t byte_size);

// Mmap the shared memory region with the given 'offset' and 'byte_size' and
// return the base address of the region.
void* MapSharedMemory(int shm_fd, size_t offset, size_t byte_size);

// Destory the shared memory region with the given name.
void UnlinkSharedMemoryRegion(std::string shm_key);

// Munmap the shared memory region from the base address with the given
// byte_size.
void UnmapSharedMemory(void* shm_addr, size_t byte_size);
