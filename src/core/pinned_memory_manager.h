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

class PinnedMemoryManager {
 public:
  // Options to be checked before allocating pinned memeory
  struct Options {
    size_t max_total_byte_size;
  };

  ~PinnedMemoryManager() = default;

  static Status Create(const Options& options);

  static Status Alloc(
      void** ptr, size_t size, bool allow_nonpinned_fallback = false);

  static Status Free(void* ptr);

 private:
  PinnedMemoryManager(const Options& options)
      : options_(options), total_pinned_byte_size_(0)
  {
  }
  Status CheckPrerequisite(size_t requested_size);

  static std::unique_ptr<PinnedMemoryManager> instance_;

  Options options_;
  std::mutex mtx_;
  size_t total_pinned_byte_size_;
  std::map<void*, std::pair<bool, size_t>> memory_info_;
};

}}  // namespace nvidia::inferenceserver