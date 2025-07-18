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
#pragma once

#ifdef TRITON_ENABLE_METAL

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "triton/core/tritonserver.h"

// Forward declarations for Objective-C++ types
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLBuffer;
@protocol MTLCommandQueue;
#else
typedef void* id;
#endif

namespace triton { namespace core {

// Forward declarations
class Status;

// Metal-specific memory types that extend TRITONSERVER_MemoryType
enum class MetalMemoryType {
  // Standard Metal buffer
  METAL_BUFFER = 100,
  // Unified memory (shared between CPU and GPU)
  METAL_UNIFIED = 101,
  // Managed memory (automatically migrated)
  METAL_MANAGED = 102
};

// Convert MetalMemoryType to TRITONSERVER_MemoryType
inline TRITONSERVER_MemoryType
ToTritonMemoryType(MetalMemoryType type)
{
  // For now, map all Metal memory types to GPU
  // In the future, we may extend TRITONSERVER_MemoryType
  return TRITONSERVER_MEMORY_GPU;
}

// MetalBuffer wrapper class
class MetalBuffer {
 public:
  MetalBuffer() : buffer_(nullptr), size_(0), memory_type_(MetalMemoryType::METAL_BUFFER) {}
  
  ~MetalBuffer();

  // Create a new Metal buffer
  static Status Create(
      std::unique_ptr<MetalBuffer>& buffer,
      size_t size,
      MetalMemoryType memory_type = MetalMemoryType::METAL_UNIFIED,
      int64_t device_id = 0);

  // Get the raw buffer pointer
  void* Data();
  const void* Data() const;

  // Get the Metal buffer object
#ifdef __OBJC__
  id<MTLBuffer> GetMetalBuffer() const { return buffer_; }
#else
  id GetMetalBuffer() const { return buffer_; }
#endif

  // Get buffer size
  size_t Size() const { return size_; }

  // Get memory type
  MetalMemoryType GetMemoryType() const { return memory_type_; }

  // Copy data from/to the buffer
  Status CopyFromHost(const void* src, size_t size, size_t offset = 0);
  Status CopyToHost(void* dst, size_t size, size_t offset = 0) const;

  // Copy between Metal buffers
  Status CopyFrom(const MetalBuffer& src, size_t size, size_t src_offset = 0, size_t dst_offset = 0);

  // Synchronize buffer (ensure all operations are complete)
  Status Synchronize();

 private:
#ifdef __OBJC__
  id<MTLBuffer> buffer_;
#else
  id buffer_;
#endif
  size_t size_;
  MetalMemoryType memory_type_;
  int64_t device_id_;
};

// Metal Memory Manager singleton
class MetalMemoryManager {
 public:
  // Options to configure Metal memory manager
  struct Options {
    Options() : enable_unified_memory_(true), memory_pool_size_(0) {}

    // Whether to use unified memory by default
    bool enable_unified_memory_;
    
    // Size of memory pool to pre-allocate (0 = no pre-allocation)
    size_t memory_pool_size_;
    
    // Per-device memory limits (device_id -> size_limit)
    std::map<int, size_t> device_memory_limits_;
  };

  ~MetalMemoryManager();

  // Initialize the Metal memory manager
  static Status Create(const Options& options = Options());

  // Reset/cleanup the memory manager
  static void Reset();

  // Check if Metal is available on this system
  static bool IsAvailable();

  // Get the number of available Metal devices
  static size_t DeviceCount();

  // Allocate Metal memory
  static Status Alloc(
      void** ptr,
      size_t size,
      MetalMemoryType memory_type = MetalMemoryType::METAL_UNIFIED,
      int64_t device_id = 0);

  // Free Metal memory
  static Status Free(void* ptr, int64_t device_id = 0);

  // Create a MetalBuffer
  static Status CreateBuffer(
      std::unique_ptr<MetalBuffer>& buffer,
      size_t size,
      MetalMemoryType memory_type = MetalMemoryType::METAL_UNIFIED,
      int64_t device_id = 0);

  // Copy memory between host and device
  static Status CopyHostToDevice(
      void* dst, const void* src, size_t size, int64_t device_id = 0);
  static Status CopyDeviceToHost(
      void* dst, const void* src, size_t size, int64_t device_id = 0);
  static Status CopyDeviceToDevice(
      void* dst, const void* src, size_t size,
      int64_t src_device_id = 0, int64_t dst_device_id = 0);

  // Get device properties
  static Status GetDeviceName(int64_t device_id, std::string& name);
  static Status GetDeviceMemoryInfo(
      int64_t device_id, size_t& total_memory, size_t& available_memory);

  // Synchronize device operations
  static Status DeviceSynchronize(int64_t device_id = 0);

 private:
  MetalMemoryManager();
  
  struct Impl;
  std::unique_ptr<Impl> impl_;
  
  static std::unique_ptr<MetalMemoryManager> instance_;
  static std::mutex instance_mu_;
};

// Metal memory allocation helper
class MetalMemory {
 public:
  // Allocate Metal memory with automatic cleanup
  static Status Alloc(
      std::unique_ptr<MetalBuffer>& buffer,
      size_t size,
      MetalMemoryType memory_type = MetalMemoryType::METAL_UNIFIED,
      int64_t device_id = 0)
  {
    return MetalMemoryManager::CreateBuffer(buffer, size, memory_type, device_id);
  }
};

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METAL