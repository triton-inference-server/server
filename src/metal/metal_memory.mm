// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef TRITON_ENABLE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <unordered_map>

#include "metal_memory.h"
#include "triton/core/tritonserver.h"

// Include the Status definition
#include "../status.h"

namespace triton { namespace core {

//
// MetalBuffer Implementation
//

MetalBuffer::~MetalBuffer()
{
  if (buffer_ != nullptr) {
    // ARC will handle the release
    buffer_ = nullptr;
  }
}

Status
MetalBuffer::Create(
    std::unique_ptr<MetalBuffer>& buffer,
    size_t size,
    MetalMemoryType memory_type,
    int64_t device_id)
{
  @autoreleasepool {
    // Get the Metal device
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices.count == 0) {
      return Status(
          Status::Code::UNAVAILABLE,
          "No Metal devices available");
    }
    
    if (device_id >= devices.count) {
      return Status(
          Status::Code::INVALID_ARG,
          "Invalid Metal device ID: " + std::to_string(device_id));
    }
    
    id<MTLDevice> device = devices[device_id];
    
    // Determine storage mode based on memory type
    MTLResourceOptions options = 0;
    switch (memory_type) {
      case MetalMemoryType::METAL_BUFFER:
        options = MTLResourceStorageModePrivate;
        break;
      case MetalMemoryType::METAL_UNIFIED:
        options = MTLResourceStorageModeShared;
        break;
      case MetalMemoryType::METAL_MANAGED:
        options = MTLResourceStorageModeManaged;
        break;
    }
    
    // Create the Metal buffer
    id<MTLBuffer> mtl_buffer = [device newBufferWithLength:size options:options];
    if (mtl_buffer == nil) {
      return Status(
          Status::Code::INTERNAL,
          "Failed to allocate Metal buffer of size " + std::to_string(size));
    }
    
    // Create the wrapper
    buffer.reset(new MetalBuffer());
    buffer->buffer_ = mtl_buffer;
    buffer->size_ = size;
    buffer->memory_type_ = memory_type;
    buffer->device_id_ = device_id;
    
    return Status::Success;
  }
}

void*
MetalBuffer::Data()
{
  if (buffer_ == nullptr) {
    return nullptr;
  }
  
  // Only shared and managed memory can be accessed from CPU
  if (memory_type_ == MetalMemoryType::METAL_UNIFIED ||
      memory_type_ == MetalMemoryType::METAL_MANAGED) {
    id<MTLBuffer> mtl_buffer = static_cast<id<MTLBuffer>>(buffer_);
    return [mtl_buffer contents];
  }
  
  return nullptr;
}

const void*
MetalBuffer::Data() const
{
  return const_cast<MetalBuffer*>(this)->Data();
}

Status
MetalBuffer::CopyFromHost(const void* src, size_t size, size_t offset)
{
  if (offset + size > size_) {
    return Status(
        Status::Code::INVALID_ARG,
        "Copy size exceeds buffer bounds");
  }
  
  @autoreleasepool {
    id<MTLBuffer> mtl_buffer = static_cast<id<MTLBuffer>>(buffer_);
    
    if (memory_type_ == MetalMemoryType::METAL_UNIFIED ||
        memory_type_ == MetalMemoryType::METAL_MANAGED) {
      // Direct memory access for shared/managed memory
      void* dst = static_cast<char*>([mtl_buffer contents]) + offset;
      std::memcpy(dst, src, size);
      
      // For managed memory, notify that we've modified the buffer
      if (memory_type_ == MetalMemoryType::METAL_MANAGED) {
        [mtl_buffer didModifyRange:NSMakeRange(offset, size)];
      }
    } else {
      // For private memory, we need to use a blit encoder
      // First create a temporary shared buffer
      NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
      id<MTLDevice> device = devices[device_id_];
      
      id<MTLBuffer> temp_buffer = [device newBufferWithBytes:src
                                                       length:size
                                                      options:MTLResourceStorageModeShared];
      if (temp_buffer == nil) {
        return Status(
            Status::Code::INTERNAL,
            "Failed to create temporary buffer for copy");
      }
      
      // Create command buffer and blit encoder
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
      
      [blit copyFromBuffer:temp_buffer
              sourceOffset:0
                  toBuffer:mtl_buffer
         destinationOffset:offset
                      size:size];
      
      [blit endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }
    
    return Status::Success;
  }
}

Status
MetalBuffer::CopyToHost(void* dst, size_t size, size_t offset) const
{
  if (offset + size > size_) {
    return Status(
        Status::Code::INVALID_ARG,
        "Copy size exceeds buffer bounds");
  }
  
  @autoreleasepool {
    id<MTLBuffer> mtl_buffer = static_cast<id<MTLBuffer>>(buffer_);
    
    if (memory_type_ == MetalMemoryType::METAL_UNIFIED ||
        memory_type_ == MetalMemoryType::METAL_MANAGED) {
      // Direct memory access for shared/managed memory
      const void* src = static_cast<const char*>([mtl_buffer contents]) + offset;
      std::memcpy(dst, src, size);
    } else {
      // For private memory, we need to use a blit encoder
      NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
      id<MTLDevice> device = devices[device_id_];
      
      // Create a temporary shared buffer
      id<MTLBuffer> temp_buffer = [device newBufferWithLength:size
                                                      options:MTLResourceStorageModeShared];
      if (temp_buffer == nil) {
        return Status(
            Status::Code::INTERNAL,
            "Failed to create temporary buffer for copy");
      }
      
      // Create command buffer and blit encoder
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
      
      [blit copyFromBuffer:mtl_buffer
              sourceOffset:offset
                  toBuffer:temp_buffer
         destinationOffset:0
                      size:size];
      
      [blit endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      
      // Copy from temporary buffer to host
      std::memcpy(dst, [temp_buffer contents], size);
    }
    
    return Status::Success;
  }
}

Status
MetalBuffer::CopyFrom(const MetalBuffer& src, size_t size, size_t src_offset, size_t dst_offset)
{
  if (src_offset + size > src.size_) {
    return Status(
        Status::Code::INVALID_ARG,
        "Source copy size exceeds buffer bounds");
  }
  
  if (dst_offset + size > size_) {
    return Status(
        Status::Code::INVALID_ARG,
        "Destination copy size exceeds buffer bounds");
  }
  
  @autoreleasepool {
    id<MTLBuffer> src_buffer = static_cast<id<MTLBuffer>>(src.buffer_);
    id<MTLBuffer> dst_buffer = static_cast<id<MTLBuffer>>(buffer_);
    
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    id<MTLDevice> device = devices[device_id_];
    
    // Create command buffer and blit encoder
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    
    [blit copyFromBuffer:src_buffer
            sourceOffset:src_offset
                toBuffer:dst_buffer
       destinationOffset:dst_offset
                    size:size];
    
    [blit endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    
    return Status::Success;
  }
}

Status
MetalBuffer::Synchronize()
{
  @autoreleasepool {
    if (memory_type_ == MetalMemoryType::METAL_MANAGED) {
      // For managed memory, we need to synchronize
      id<MTLBuffer> mtl_buffer = static_cast<id<MTLBuffer>>(buffer_);
      NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
      id<MTLDevice> device = devices[device_id_];
      
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
      
      [blit synchronizeResource:mtl_buffer];
      [blit endEncoding];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
    }
    
    return Status::Success;
  }
}

//
// MetalMemoryManager Implementation
//

struct MetalMemoryManager::Impl {
  Options options;
  std::vector<id<MTLDevice>> devices;
  std::unordered_map<void*, size_t> allocations;
  std::mutex allocation_mutex;
  
  Impl() {
    @autoreleasepool {
      NSArray<id<MTLDevice>>* device_array = MTLCopyAllDevices();
      for (id<MTLDevice> device in device_array) {
        devices.push_back(device);
      }
    }
  }
};

std::unique_ptr<MetalMemoryManager> MetalMemoryManager::instance_;
std::mutex MetalMemoryManager::instance_mu_;

MetalMemoryManager::MetalMemoryManager()
    : impl_(std::make_unique<Impl>())
{
}

MetalMemoryManager::~MetalMemoryManager()
{
  // Clean up any remaining allocations
  std::lock_guard<std::mutex> lock(impl_->allocation_mutex);
  impl_->allocations.clear();
}

Status
MetalMemoryManager::Create(const Options& options)
{
  std::lock_guard<std::mutex> lock(instance_mu_);
  if (instance_ != nullptr) {
    return Status(
        Status::Code::ALREADY_EXISTS,
        "Metal memory manager already initialized");
  }
  
  instance_.reset(new MetalMemoryManager());
  instance_->impl_->options = options;
  
  return Status::Success;
}

void
MetalMemoryManager::Reset()
{
  std::lock_guard<std::mutex> lock(instance_mu_);
  instance_.reset();
}

bool
MetalMemoryManager::IsAvailable()
{
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    return devices.count > 0;
  }
}

size_t
MetalMemoryManager::DeviceCount()
{
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    return devices.count;
  }
}

Status
MetalMemoryManager::Alloc(
    void** ptr,
    size_t size,
    MetalMemoryType memory_type,
    int64_t device_id)
{
  std::lock_guard<std::mutex> lock(instance_mu_);
  if (instance_ == nullptr) {
    return Status(
        Status::Code::INTERNAL,
        "Metal memory manager not initialized");
  }
  
  std::unique_ptr<MetalBuffer> buffer;
  auto status = CreateBuffer(buffer, size, memory_type, device_id);
  if (!status.IsOk()) {
    return status;
  }
  
  *ptr = buffer->Data();
  if (*ptr == nullptr && memory_type != MetalMemoryType::METAL_BUFFER) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to get data pointer from Metal buffer");
  }
  
  // Track allocation
  {
    std::lock_guard<std::mutex> alloc_lock(instance_->impl_->allocation_mutex);
    instance_->impl_->allocations[*ptr] = size;
  }
  
  // Release ownership of the buffer
  buffer.release();
  
  return Status::Success;
}

Status
MetalMemoryManager::Free(void* ptr, int64_t device_id)
{
  std::lock_guard<std::mutex> lock(instance_mu_);
  if (instance_ == nullptr) {
    return Status(
        Status::Code::INTERNAL,
        "Metal memory manager not initialized");
  }
  
  // Remove from tracking
  {
    std::lock_guard<std::mutex> alloc_lock(instance_->impl_->allocation_mutex);
    instance_->impl_->allocations.erase(ptr);
  }
  
  // The actual memory will be freed when the MetalBuffer is destroyed
  // For now, we're just removing it from tracking
  
  return Status::Success;
}

Status
MetalMemoryManager::CreateBuffer(
    std::unique_ptr<MetalBuffer>& buffer,
    size_t size,
    MetalMemoryType memory_type,
    int64_t device_id)
{
  return MetalBuffer::Create(buffer, size, memory_type, device_id);
}

Status
MetalMemoryManager::CopyHostToDevice(
    void* dst, const void* src, size_t size, int64_t device_id)
{
  // For unified memory, just do a memcpy
  std::memcpy(dst, src, size);
  return Status::Success;
}

Status
MetalMemoryManager::CopyDeviceToHost(
    void* dst, const void* src, size_t size, int64_t device_id)
{
  // For unified memory, just do a memcpy
  std::memcpy(dst, src, size);
  return Status::Success;
}

Status
MetalMemoryManager::CopyDeviceToDevice(
    void* dst, const void* src, size_t size,
    int64_t src_device_id, int64_t dst_device_id)
{
  if (src_device_id == dst_device_id) {
    // Same device, just do a memcpy for unified memory
    std::memcpy(dst, src, size);
    return Status::Success;
  }
  
  // Cross-device copy would require more complex handling
  return Status(
      Status::Code::UNIMPLEMENTED,
      "Cross-device Metal memory copy not yet implemented");
}

Status
MetalMemoryManager::GetDeviceName(int64_t device_id, std::string& name)
{
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (device_id >= devices.count) {
      return Status(
          Status::Code::INVALID_ARG,
          "Invalid Metal device ID: " + std::to_string(device_id));
    }
    
    id<MTLDevice> device = devices[device_id];
    name = std::string([[device name] UTF8String]);
    
    return Status::Success;
  }
}

Status
MetalMemoryManager::GetDeviceMemoryInfo(
    int64_t device_id, size_t& total_memory, size_t& available_memory)
{
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (device_id >= devices.count) {
      return Status(
          Status::Code::INVALID_ARG,
          "Invalid Metal device ID: " + std::to_string(device_id));
    }
    
    id<MTLDevice> device = devices[device_id];
    
    // Get recommended working set size
    if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
      total_memory = [device recommendedMaxWorkingSetSize];
    } else {
      // Fallback for older devices
      total_memory = 1024 * 1024 * 1024; // 1GB default
    }
    
    // Metal doesn't provide a direct way to get available memory
    // We'll use 80% of total as an estimate
    available_memory = total_memory * 0.8;
    
    return Status::Success;
  }
}

Status
MetalMemoryManager::DeviceSynchronize(int64_t device_id)
{
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (device_id >= devices.count) {
      return Status(
          Status::Code::INVALID_ARG,
          "Invalid Metal device ID: " + std::to_string(device_id));
    }
    
    // Metal operations are automatically synchronized when we wait for command buffers
    // This is a no-op for now
    return Status::Success;
  }
}

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METAL