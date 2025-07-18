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

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_memory_manager.h"
#include "metal_command.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace triton { namespace core { namespace metal {

//
// MetalBuffer Implementation
//

MetalBuffer::MetalBuffer(MetalDevice* device, id buffer, size_t size, bool shared)
    : device_(device), buffer_(buffer), size_(size), shared_(shared) {
  if (buffer_) {
    [(id<MTLBuffer>)buffer_ retain];
  }
}

MetalBuffer::~MetalBuffer() {
  if (buffer_) {
    [(id<MTLBuffer>)buffer_ release];
  }
}

std::unique_ptr<MetalBuffer> MetalBuffer::Create(
    MetalDevice* device, size_t size, bool shared) {
  if (!device || size == 0) {
    return nullptr;
  }
  
  id<MTLDevice> mtl_device = (id<MTLDevice>)device->GetDevice();
  MTLResourceOptions options = shared ? 
      MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
  
  id<MTLBuffer> buffer = [mtl_device newBufferWithLength:size options:options];
  if (!buffer) {
    std::cerr << "Failed to allocate Metal buffer of size " << size << std::endl;
    return nullptr;
  }
  
  return std::unique_ptr<MetalBuffer>(new MetalBuffer(device, buffer, size, shared));
}

void* MetalBuffer::GetContents() const {
  if (!shared_) {
    std::cerr << "Cannot get contents of private Metal buffer" << std::endl;
    return nullptr;
  }
  
  id<MTLBuffer> mtl_buffer = (id<MTLBuffer>)buffer_;
  return [mtl_buffer contents];
}

void MetalBuffer::CopyFromHost(const void* src, size_t size, size_t offset) {
  if (!src || size == 0) {
    return;
  }
  
  if (offset + size > size_) {
    std::cerr << "Copy exceeds buffer bounds: offset=" << offset 
              << ", size=" << size << ", buffer_size=" << size_ << std::endl;
    return;
  }
  
  if (shared_) {
    void* contents = GetContents();
    if (contents) {
      std::memcpy(static_cast<char*>(contents) + offset, src, size);
    }
  } else {
    // For private buffers, use a staging buffer and blit encoder
    @autoreleasepool {
      id<MTLDevice> mtl_device = (id<MTLDevice>)device_->GetDevice();
      id<MTLCommandQueue> commandQueue = [mtl_device newCommandQueue];
      if (!commandQueue) {
        std::cerr << "Failed to create command queue for private buffer copy" << std::endl;
        return;
      }
      
      // Create a temporary shared buffer for staging
      id<MTLBuffer> stagingBuffer = [mtl_device newBufferWithBytes:src 
                                                           length:size 
                                                          options:MTLResourceStorageModeShared];
      if (!stagingBuffer) {
        std::cerr << "Failed to create staging buffer for private buffer copy" << std::endl;
        [commandQueue release];
        return;
      }
      
      // Create command buffer and blit encoder
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      if (!commandBuffer) {
        std::cerr << "Failed to create command buffer for private buffer copy" << std::endl;
        [stagingBuffer release];
        [commandQueue release];
        return;
      }
      
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
      if (!blitEncoder) {
        std::cerr << "Failed to create blit encoder for private buffer copy" << std::endl;
        [stagingBuffer release];
        [commandQueue release];
        return;
      }
      
      // Copy from staging buffer to private buffer
      [blitEncoder copyFromBuffer:stagingBuffer
                     sourceOffset:0
                        toBuffer:(id<MTLBuffer>)buffer_
               destinationOffset:offset
                            size:size];
      
      [blitEncoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      
      // Clean up
      [stagingBuffer release];
      [commandQueue release];
    }
  }
}

void MetalBuffer::CopyToHost(void* dst, size_t size, size_t offset) const {
  if (!dst || size == 0) {
    return;
  }
  
  if (offset + size > size_) {
    std::cerr << "Copy exceeds buffer bounds: offset=" << offset 
              << ", size=" << size << ", buffer_size=" << size_ << std::endl;
    return;
  }
  
  if (shared_) {
    void* contents = GetContents();
    if (contents) {
      std::memcpy(dst, static_cast<const char*>(contents) + offset, size);
    }
  } else {
    // For private buffers, use a staging buffer and blit encoder
    @autoreleasepool {
      id<MTLDevice> mtl_device = (id<MTLDevice>)device_->GetDevice();
      id<MTLCommandQueue> commandQueue = [mtl_device newCommandQueue];
      if (!commandQueue) {
        std::cerr << "Failed to create command queue for private buffer copy" << std::endl;
        return;
      }
      
      // Create a temporary shared buffer for staging
      id<MTLBuffer> stagingBuffer = [mtl_device newBufferWithLength:size 
                                                            options:MTLResourceStorageModeShared];
      if (!stagingBuffer) {
        std::cerr << "Failed to create staging buffer for private buffer copy" << std::endl;
        [commandQueue release];
        return;
      }
      
      // Create command buffer and blit encoder
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      if (!commandBuffer) {
        std::cerr << "Failed to create command buffer for private buffer copy" << std::endl;
        [stagingBuffer release];
        [commandQueue release];
        return;
      }
      
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
      if (!blitEncoder) {
        std::cerr << "Failed to create blit encoder for private buffer copy" << std::endl;
        [stagingBuffer release];
        [commandQueue release];
        return;
      }
      
      // Copy from private buffer to staging buffer
      [blitEncoder copyFromBuffer:(id<MTLBuffer>)buffer_
                     sourceOffset:offset
                        toBuffer:stagingBuffer
               destinationOffset:0
                            size:size];
      
      [blitEncoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      
      // Copy from staging buffer to host memory
      void* stagingContents = [stagingBuffer contents];
      if (stagingContents) {
        std::memcpy(dst, stagingContents, size);
      }
      
      // Clean up
      [stagingBuffer release];
      [commandQueue release];
    }
  }
}

//
// MetalMemoryManager Implementation
//

MetalMemoryManager::MetalMemoryManager(MetalDevice* device) 
    : device_(device), stats_{0, 0, 0, 0, 0, 0} {
}

std::unique_ptr<MetalMemoryManager> MetalMemoryManager::Create(MetalDevice* device) {
  if (!device) {
    return nullptr;
  }
  return std::unique_ptr<MetalMemoryManager>(new MetalMemoryManager(device));
}

std::unique_ptr<MetalBuffer> MetalMemoryManager::Allocate(size_t size, bool shared) {
  auto buffer = MetalBuffer::Create(device_, size, shared);
  if (buffer) {
    UpdateStats(size, true);
  }
  return buffer;
}

void MetalMemoryManager::UpdateStats(size_t size, bool is_allocation) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (is_allocation) {
    stats_.total_allocated += size;
    stats_.current_usage += size;
    stats_.allocation_count++;
    
    if (stats_.current_usage > stats_.peak_usage) {
      stats_.peak_usage = stats_.current_usage;
    }
  } else {
    stats_.total_freed += size;
    stats_.current_usage = (stats_.current_usage >= size) ? 
        stats_.current_usage - size : 0;
    stats_.deallocation_count++;
  }
}

MetalMemoryStats MetalMemoryManager::GetStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

size_t MetalMemoryManager::GetAvailableMemory() const {
  return device_->GetAvailableMemory();
}

size_t MetalMemoryManager::GetTotalMemory() const {
  return device_->GetTotalMemory();
}

void MetalMemoryManager::ResetStats() {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_ = {0, 0, 0, 0, 0, 0};
}

//
// MetalMemoryPool Implementation
//

MetalMemoryPool& MetalMemoryPool::Instance() {
  static MetalMemoryPool instance;
  return instance;
}

MetalMemoryManager* MetalMemoryPool::GetMemoryManager(int device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = managers_.find(device_id);
  if (it != managers_.end()) {
    return it->second.get();
  }
  
  // Create new manager
  auto& device_manager = MetalDeviceManager::Instance();
  MetalDevice* device = device_manager.GetDevice(device_id);
  if (!device) {
    std::cerr << "Invalid device ID: " << device_id << std::endl;
    return nullptr;
  }
  
  auto manager = MetalMemoryManager::Create(device);
  if (!manager) {
    return nullptr;
  }
  
  MetalMemoryManager* manager_ptr = manager.get();
  managers_[device_id] = std::move(manager);
  return manager_ptr;
}

std::unique_ptr<MetalBuffer> MetalMemoryPool::Allocate(
    int device_id, size_t size, bool shared) {
  MetalMemoryManager* manager = GetMemoryManager(device_id);
  if (!manager) {
    return nullptr;
  }
  return manager->Allocate(size, shared);
}

MetalMemoryStats MetalMemoryPool::GetGlobalStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  MetalMemoryStats global_stats = {0, 0, 0, 0, 0, 0};
  for (const auto& pair : managers_) {
    MetalMemoryStats device_stats = pair.second->GetStats();
    global_stats.total_allocated += device_stats.total_allocated;
    global_stats.total_freed += device_stats.total_freed;
    global_stats.current_usage += device_stats.current_usage;
    global_stats.peak_usage += device_stats.peak_usage;
    global_stats.allocation_count += device_stats.allocation_count;
    global_stats.deallocation_count += device_stats.deallocation_count;
  }
  
  return global_stats;
}

void MetalMemoryPool::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  managers_.clear();
}

//
// MetalMemoryUtils Implementation
//

void MetalMemoryUtils::CopyBuffer(MetalBuffer* dst, const MetalBuffer* src,
                                  size_t size, size_t dst_offset, size_t src_offset) {
  if (!dst || !src || size == 0) {
    return;
  }
  
  if (dst_offset + size > dst->GetSize() || src_offset + size > src->GetSize()) {
    std::cerr << "Copy exceeds buffer bounds" << std::endl;
    return;
  }
  
  // If both buffers are shared and on the same device, use memcpy
  if (dst->IsShared() && src->IsShared() && dst->GetDevice() == src->GetDevice()) {
    void* dst_ptr = static_cast<char*>(dst->GetContents()) + dst_offset;
    const void* src_ptr = static_cast<const char*>(src->GetContents()) + src_offset;
    std::memcpy(dst_ptr, src_ptr, size);
    return;
  }
  
  // For GPU-to-GPU copy (including private buffers)
  if (dst->GetDevice() == src->GetDevice()) {
    @autoreleasepool {
      id<MTLDevice> mtl_device = (id<MTLDevice>)dst->GetDevice()->GetDevice();
      id<MTLCommandQueue> commandQueue = [mtl_device newCommandQueue];
      if (!commandQueue) {
        std::cerr << "Failed to create command queue for GPU-to-GPU copy" << std::endl;
        return;
      }
      
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      if (!commandBuffer) {
        std::cerr << "Failed to create command buffer for GPU-to-GPU copy" << std::endl;
        [commandQueue release];
        return;
      }
      
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
      if (!blitEncoder) {
        std::cerr << "Failed to create blit encoder for GPU-to-GPU copy" << std::endl;
        [commandQueue release];
        return;
      }
      
      // Copy from source to destination buffer
      [blitEncoder copyFromBuffer:(id<MTLBuffer>)src->GetBuffer()
                     sourceOffset:src_offset
                        toBuffer:(id<MTLBuffer>)dst->GetBuffer()
               destinationOffset:dst_offset
                            size:size];
      
      [blitEncoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      
      [commandQueue release];
    }
  } else {
    // Cross-device copy requires staging through system memory
    std::vector<uint8_t> temp_buffer(size);
    src->CopyToHost(temp_buffer.data(), size, src_offset);
    dst->CopyFromHost(temp_buffer.data(), size, dst_offset);
  }
}

void MetalMemoryUtils::CopyBetweenDevices(MetalBuffer* dst, const MetalBuffer* src) {
  if (!dst || !src) {
    return;
  }
  
  // Metal doesn't support direct GPU-to-GPU copy between different devices
  // Need to go through system memory
  if (dst->GetDevice() != src->GetDevice()) {
    if (src->IsShared() && dst->IsShared()) {
      CopyBuffer(dst, src, std::min(dst->GetSize(), src->GetSize()));
    } else {
      std::cerr << "Cross-device copy requires shared buffers" << std::endl;
    }
  } else {
    CopyBuffer(dst, src, std::min(dst->GetSize(), src->GetSize()));
  }
}

void MetalMemoryUtils::ZeroBuffer(MetalBuffer* buffer) {
  if (!buffer) {
    return;
  }
  
  if (buffer->IsShared()) {
    void* contents = buffer->GetContents();
    if (contents) {
      std::memset(contents, 0, buffer->GetSize());
    }
  } else {
    // For private buffers, use blit encoder's fill operation
    @autoreleasepool {
      id<MTLDevice> mtl_device = (id<MTLDevice>)buffer->GetDevice()->GetDevice();
      id<MTLCommandQueue> commandQueue = [mtl_device newCommandQueue];
      if (!commandQueue) {
        std::cerr << "Failed to create command queue for zero buffer" << std::endl;
        return;
      }
      
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      if (!commandBuffer) {
        std::cerr << "Failed to create command buffer for zero buffer" << std::endl;
        [commandQueue release];
        return;
      }
      
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
      if (!blitEncoder) {
        std::cerr << "Failed to create blit encoder for zero buffer" << std::endl;
        [commandQueue release];
        return;
      }
      
      // Fill buffer with zeros
      [blitEncoder fillBuffer:(id<MTLBuffer>)buffer->GetBuffer()
                        range:NSMakeRange(0, buffer->GetSize())
                        value:0];
      
      [blitEncoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      
      [commandQueue release];
    }
  }
}

size_t MetalMemoryUtils::GetBufferAlignment() {
  // Metal typically requires 256-byte alignment for buffers
  return 256;
}

size_t MetalMemoryUtils::AlignSize(size_t size) {
  const size_t alignment = GetBufferAlignment();
  return (size + alignment - 1) & ~(alignment - 1);
}

}}}  // namespace triton::core::metal