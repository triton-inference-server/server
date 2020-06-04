// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <map>
#include <memory>
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// This is a singleton class responsible for maintaining CUDA memory pool
// used by the inference server. CUDA memory allocations and deallocations
// must be requested via functions provided by this class.
class CudaMemoryManager {
 public:
  // Options to configure CUDA memory manager.
  struct Options {
    Options(double cc = 6.0, const std::map<int, uint64_t>& s = {})
        : min_supported_compute_capability_(cc), memory_pool_byte_size_(s)
    {
    }

    // The minimum compute capability of the supported devices.
    double min_supported_compute_capability_;

    // The size of CUDA memory reserved for the specified devices.
    // The memory size will be rounded up to align with
    // the default granularity (512 bytes).
    // No memory will be reserved for devices that is not listed.
    std::map<int, uint64_t> memory_pool_byte_size_;
  };

  ~CudaMemoryManager();

  // Create the memory manager based on 'options' specified.
  // Return Status object indicating success or failure.
  static Status Create(const Options& options);

  // Allocate CUDA memory on GPU 'device_id' with
  // the requested 'size' and return the pointer in 'ptr'.
  // Return Status object indicating success or failure.
  static Status Alloc(void** ptr, uint64_t size, int64_t device_id);

  // Free the memory allocated by the memory manager on 'device_id'.
  // Return Status object indicating success or failure.
  static Status Free(void* ptr, int64_t device_id);

 protected:
  // Provide explicit control on the lifecycle of the CUDA memory manager,
  // for testing only.
  static void Reset();

 private:
  CudaMemoryManager(bool has_allocation) : has_allocation_(has_allocation) {}
  bool has_allocation_;
  static std::unique_ptr<CudaMemoryManager> instance_;
};

}}  // namespace nvidia::inferenceserver
