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
//
#pragma once

#include <boost/interprocess/managed_external_buffer.hpp>
#include <memory>
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// This is a singleton class responsible for maintaining pinned memory pool
// used by the inference server. Pinned memory allocations and deallocations
// must be requested via functions provided by this class.
class PinnedMemoryManager {
 public:
  // Options to configure pinned memeory manager.
  struct Options {
    Options(uint64_t b = 0) : pinned_memory_pool_byte_size_(b) {}

    uint64_t pinned_memory_pool_byte_size_;
  };

  ~PinnedMemoryManager();

  // Create the pinned memory manager based on 'options' specified.
  // Return true on success, false otherwise.
  static Status Create(const Options& options);

  // Allocate pinned memory with the requested 'size' and return the pointer
  // in 'ptr'. If 'allow_nonpinned_fallback' is true, regular system memory
  // will be allocated as fallback in the case where pinned memory fails to
  // be allocated.
  // Return true on success, false otherwise.
  static Status Alloc(
      void** ptr, uint64_t size, bool allow_nonpinned_fallback = false);

  // Free the memory allocated by the pinned memory manager.
  static Status Free(void* ptr);

 protected:
  PinnedMemoryManager(void* pinned_memory_buffer, uint64_t size);

  Status AllocInternal(
      void** ptr, uint64_t size, bool allow_nonpinned_fallback = false);
  Status FreeInternal(void* ptr);

 private:
  static std::unique_ptr<PinnedMemoryManager> instance_;

  std::mutex info_mtx_;
  std::map<void*, bool> memory_info_;

  void* pinned_memory_buffer_;
  std::mutex buffer_mtx_;
  boost::interprocess::managed_external_buffer managed_pinned_memory_;
};

}}  // namespace nvidia::inferenceserver