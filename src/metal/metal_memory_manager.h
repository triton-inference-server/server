// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <mutex>
#include <unordered_map>
#include <vector>
#include "metal_device.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* id;
#endif

namespace triton { namespace core { namespace metal {

// Metal buffer wrapper
class MetalBuffer {
 public:
  static std::unique_ptr<MetalBuffer> Create(
      MetalDevice* device, size_t size, bool shared = true);
  
  // Get the underlying Metal buffer
  id GetBuffer() const { return buffer_; }
  
  // Get buffer size
  size_t GetSize() const { return size_; }
  
  // Get CPU-accessible pointer (for shared buffers)
  void* GetContents() const;
  
  // Copy data to buffer
  void CopyFromHost(const void* src, size_t size, size_t offset = 0);
  
  // Copy data from buffer
  void CopyToHost(void* dst, size_t size, size_t offset = 0) const;
  
  // Check if buffer is shared (CPU/GPU accessible)
  bool IsShared() const { return shared_; }
  
  // Get the device this buffer belongs to
  MetalDevice* GetDevice() const { return device_; }

 private:
  MetalBuffer(MetalDevice* device, id buffer, size_t size, bool shared);
  ~MetalBuffer();
  
  MetalDevice* device_;
  id buffer_;  // MTLBuffer instance
  size_t size_;
  bool shared_;
};

// Metal memory allocation statistics
struct MetalMemoryStats {
  size_t total_allocated;
  size_t total_freed;
  size_t current_usage;
  size_t peak_usage;
  size_t allocation_count;
  size_t deallocation_count;
};

// Metal memory manager for a specific device
class MetalMemoryManager {
 public:
  static std::unique_ptr<MetalMemoryManager> Create(MetalDevice* device);
  
  // Allocate memory
  std::unique_ptr<MetalBuffer> Allocate(size_t size, bool shared = true);
  
  // Get memory statistics
  MetalMemoryStats GetStats() const;
  
  // Get available memory on device
  size_t GetAvailableMemory() const;
  
  // Get total memory on device
  size_t GetTotalMemory() const;
  
  // Reset statistics
  void ResetStats();
  
  // Get the device this manager belongs to
  MetalDevice* GetDevice() const { return device_; }

 private:
  MetalMemoryManager(MetalDevice* device);
  
  void UpdateStats(size_t size, bool is_allocation);
  
  MetalDevice* device_;
  mutable std::mutex mutex_;
  MetalMemoryStats stats_;
};

// Global Metal memory manager
class MetalMemoryPool {
 public:
  static MetalMemoryPool& Instance();
  
  // Get or create memory manager for a device
  MetalMemoryManager* GetMemoryManager(int device_id);
  
  // Allocate memory on a specific device
  std::unique_ptr<MetalBuffer> Allocate(int device_id, size_t size, bool shared = true);
  
  // Get total memory stats across all devices
  MetalMemoryStats GetGlobalStats() const;
  
  // Clear all memory managers
  void Clear();

 private:
  MetalMemoryPool() = default;
  ~MetalMemoryPool() = default;
  
  mutable std::mutex mutex_;
  std::unordered_map<int, std::unique_ptr<MetalMemoryManager>> managers_;
  
  // Prevent copying
  MetalMemoryPool(const MetalMemoryPool&) = delete;
  MetalMemoryPool& operator=(const MetalMemoryPool&) = delete;
};

// Utility functions for Metal memory operations
namespace MetalMemoryUtils {
  // Copy memory between Metal buffers
  void CopyBuffer(MetalBuffer* dst, const MetalBuffer* src, 
                  size_t size, size_t dst_offset = 0, size_t src_offset = 0);
  
  // Copy memory between devices
  void CopyBetweenDevices(MetalBuffer* dst, const MetalBuffer* src);
  
  // Zero out a buffer
  void ZeroBuffer(MetalBuffer* buffer);
  
  // Get alignment requirements for Metal buffers
  size_t GetBufferAlignment();
  
  // Align size to Metal requirements
  size_t AlignSize(size_t size);
}

}}}  // namespace triton::core::metal