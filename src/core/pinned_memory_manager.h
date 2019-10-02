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

#include <memory>
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// This is a singleton class responsible for maintaining pinned memory pool
// used by the inference server. Pinned memory allocations and deallocations
// must be requested via functions provided by this class.
class PinnedMemoryManager {
 public:
  // Options to be checked before allocating pinned memeory.
  struct Options {
    Options(uint64_t b = 0) : pinned_memory_pool_byte_size_(b) {}

    uint64_t pinned_memory_pool_byte_size_;
  };

  ~PinnedMemoryManager() = default;

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

 private:
  PinnedMemoryManager(const Options& options)
      : options_(options), allocated_pinned_memory_byte_size_(0)
  {
  }

  // Helper function to check if the pinned memory should be allocated.
  Status CheckPrerequisite(uint64_t requested_size);

  static std::unique_ptr<PinnedMemoryManager> instance_;

  Options options_;
  std::mutex mtx_;
  uint64_t allocated_pinned_memory_byte_size_;
  std::map<void*, std::pair<bool, uint64_t>> memory_info_;
};

}}  // namespace nvidia::inferenceserver