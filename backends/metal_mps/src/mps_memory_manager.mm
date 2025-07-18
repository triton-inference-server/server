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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mps_memory_manager.h"
#include <chrono>
#include <algorithm>
#include <iostream>

namespace triton { namespace backend { namespace metal_mps {

MPSMemoryManager::MPSMemoryManager(void* device)
    : device_(device), max_pool_size_(1024 * 1024 * 1024),  // 1GB default
      max_buffer_age_ms_(60000)  // 1 minute default
{
  stats_ = {0, 0, 0, 0, 0};
  
  // Initialize size buckets (powers of 2 from 1KB to 1GB)
  size_t bucket = 1024;  // 1KB
  while (bucket <= 1024 * 1024 * 1024) {  // 1GB
    size_buckets_.push_back(bucket);
    bucket *= 2;
  }
}

MPSMemoryManager::~MPSMemoryManager()
{
  ClearPool();
}

void*
MPSMemoryManager::AllocateBuffer(size_t size)
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Try to find a pooled buffer first
  BufferPoolEntry* pooled = FindPooledBuffer(size);
  if (pooled) {
    pooled->in_use = true;
    stats_.reuse_count++;
    stats_.total_in_use += pooled->size;
    return pooled->buffer;
  }
  
  // Create a new buffer
  void* buffer = CreateMetalBuffer(size);
  if (buffer) {
    // Add to pool
    size_t bucket = GetSizeBucket(size);
    BufferPoolEntry entry;
    entry.buffer = buffer;
    entry.size = size;
    entry.in_use = true;
    entry.last_used_time = std::chrono::steady_clock::now().time_since_epoch().count();
    
    buffer_pools_[bucket].push_back(entry);
    
    stats_.total_allocated += size;
    stats_.total_in_use += size;
    stats_.allocation_count++;
    
    if (stats_.total_in_use > stats_.peak_usage) {
      stats_.peak_usage = stats_.total_in_use;
    }
  }
  
  return buffer;
}

void*
MPSMemoryManager::GetBuffer(const void* cpu_data, size_t size)
{
  void* buffer = AllocateBuffer(size);
  if (!buffer) {
    return nullptr;
  }
  
  @autoreleasepool {
    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
    memcpy([mtlBuffer contents], cpu_data, size);
  }
  
  return buffer;
}

void
MPSMemoryManager::ReleaseBuffer(void* buffer)
{
  if (!buffer) return;
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Find the buffer in our pools
  for (auto& pool_pair : buffer_pools_) {
    for (auto& entry : pool_pair.second) {
      if (entry.buffer == buffer && entry.in_use) {
        entry.in_use = false;
        entry.last_used_time = std::chrono::steady_clock::now().time_since_epoch().count();
        stats_.total_in_use -= entry.size;
        return;
      }
    }
  }
}

void
MPSMemoryManager::ClearPool()
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  @autoreleasepool {
    for (auto& pool_pair : buffer_pools_) {
      for (auto& entry : pool_pair.second) {
        if (!entry.in_use && entry.buffer) {
          [(__bridge_transfer id<MTLBuffer>)entry.buffer release];
          stats_.total_allocated -= entry.size;
        }
      }
    }
  }
  
  buffer_pools_.clear();
}

MemoryStats
MPSMemoryManager::GetStats() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void
MPSMemoryManager::SetMaxPoolSize(size_t max_size)
{
  std::lock_guard<std::mutex> lock(mutex_);
  max_pool_size_ = max_size;
  CleanupOldBuffers();
}

void
MPSMemoryManager::SetMaxBufferAge(uint64_t max_age_ms)
{
  std::lock_guard<std::mutex> lock(mutex_);
  max_buffer_age_ms_ = max_age_ms;
  CleanupOldBuffers();
}

BufferPoolEntry*
MPSMemoryManager::FindPooledBuffer(size_t size)
{
  size_t bucket = GetSizeBucket(size);
  auto it = buffer_pools_.find(bucket);
  
  if (it != buffer_pools_.end()) {
    for (auto& entry : it->second) {
      if (!entry.in_use && entry.size >= size) {
        return &entry;
      }
    }
  }
  
  return nullptr;
}

void*
MPSMemoryManager::CreateMetalBuffer(size_t size)
{
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    size_t aligned_size = AlignSize(size);
    
    id<MTLBuffer> buffer = [device newBufferWithLength:aligned_size
                                               options:MTLResourceStorageModeShared];
    
    if (!buffer) {
      std::cerr << "Failed to allocate Metal buffer of size: " << aligned_size << std::endl;
      return nullptr;
    }
    
    return (__bridge_retained void*)buffer;
  }
}

void
MPSMemoryManager::CleanupOldBuffers()
{
  auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  size_t total_pool_size = 0;
  
  @autoreleasepool {
    for (auto& pool_pair : buffer_pools_) {
      auto& entries = pool_pair.second;
      
      entries.erase(
          std::remove_if(entries.begin(), entries.end(),
              [&](BufferPoolEntry& entry) {
                if (!entry.in_use) {
                  // Check age
                  uint64_t age = (now - entry.last_used_time) / 1000000;  // Convert to ms
                  if (age > max_buffer_age_ms_ || total_pool_size + entry.size > max_pool_size_) {
                    [(__bridge_transfer id<MTLBuffer>)entry.buffer release];
                    stats_.total_allocated -= entry.size;
                    return true;
                  }
                  total_pool_size += entry.size;
                }
                return false;
              }),
          entries.end());
    }
  }
}

size_t
MPSMemoryManager::AlignSize(size_t size) const
{
  return ((size + kBufferAlignment - 1) / kBufferAlignment) * kBufferAlignment;
}

size_t
MPSMemoryManager::GetSizeBucket(size_t size) const
{
  for (size_t bucket : size_buckets_) {
    if (size <= bucket) {
      return bucket;
    }
  }
  // For very large sizes, return the exact size
  return size;
}

}}}  // namespace triton::backend::metal_mps