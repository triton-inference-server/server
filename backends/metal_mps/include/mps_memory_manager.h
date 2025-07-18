// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
// Forward declarations for C++ files
typedef struct id<MTLDevice> MTLDevice;
typedef struct id<MTLBuffer> MTLBuffer;
#endif

namespace triton { namespace backend { namespace metal_mps {

// Buffer pool entry
struct BufferPoolEntry {
  void* buffer;  // id<MTLBuffer>
  size_t size;
  bool in_use;
  uint64_t last_used_time;
};

// Memory statistics
struct MemoryStats {
  size_t total_allocated;
  size_t total_in_use;
  size_t peak_usage;
  size_t allocation_count;
  size_t reuse_count;
};

// MPS Memory Manager - handles Metal buffer allocation and pooling
class MPSMemoryManager {
 public:
  explicit MPSMemoryManager(void* device);  // id<MTLDevice>
  ~MPSMemoryManager();

  // Allocate a new buffer
  void* AllocateBuffer(size_t size);  // Returns id<MTLBuffer>
  
  // Get or create a buffer for CPU data (copies data to GPU)
  void* GetBuffer(const void* cpu_data, size_t size);  // Returns id<MTLBuffer>
  
  // Release a buffer back to the pool
  void ReleaseBuffer(void* buffer);
  
  // Clear all unused buffers from the pool
  void ClearPool();
  
  // Get memory statistics
  MemoryStats GetStats() const;
  
  // Set pool size limits
  void SetMaxPoolSize(size_t max_size);
  void SetMaxBufferAge(uint64_t max_age_ms);

 private:
  // Find a suitable buffer from the pool
  BufferPoolEntry* FindPooledBuffer(size_t size);
  
  // Create a new Metal buffer
  void* CreateMetalBuffer(size_t size);
  
  // Clean up old buffers
  void CleanupOldBuffers();

  void* device_;  // id<MTLDevice>
  
  // Buffer pool organized by size ranges
  std::unordered_map<size_t, std::vector<BufferPoolEntry>> buffer_pools_;
  
  // Mutex for thread safety
  mutable std::mutex mutex_;
  
  // Configuration
  size_t max_pool_size_;
  uint64_t max_buffer_age_ms_;
  
  // Statistics
  MemoryStats stats_;
  
  // Size alignment for Metal buffers
  static constexpr size_t kBufferAlignment = 256;
  
  // Size buckets for pooling
  std::vector<size_t> size_buckets_;
  
  // Helper to round up size to alignment
  size_t AlignSize(size_t size) const;
  
  // Helper to find the appropriate size bucket
  size_t GetSizeBucket(size_t size) const;
};

}}}  // namespace triton::backend::metal_mps