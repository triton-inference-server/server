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

#include "triton_metal_memory.h"
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>

// Include the Status definition
#include "triton/common/error.h"
#include "metal_allocator.h"

namespace triton { namespace core {

namespace {

// Global flag to track Metal initialization
std::atomic<bool> g_metal_initialized{false};

// Global registry for Metal allocators per device
struct MetalAllocatorRegistry {
  std::mutex mutex;
  std::unordered_map<int64_t, std::unique_ptr<triton::server::MetalAllocator>> allocators;
  
  static MetalAllocatorRegistry& Instance() {
    static MetalAllocatorRegistry instance;
    return instance;
  }
  
  triton::server::MetalAllocator* GetAllocator(int64_t device_id) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocators.find(device_id);
    if (it != allocators.end()) {
      return it->second.get();
    }
    
    // Create allocator if not exists
    try {
      auto allocator = std::make_unique<triton::server::MetalAllocator>(device_id);
      auto* ptr = allocator.get();
      allocators[device_id] = std::move(allocator);
      return ptr;
    } catch (...) {
      return nullptr;
    }
  }
  
  size_t GetAllocatedMemory(int64_t device_id) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocators.find(device_id);
    if (it != allocators.end()) {
      return it->second->GetStats().current_usage.load();
    }
    return 0;
  }
};

// Convert Status to TRITONSERVER_Error
TRITONSERVER_Error*
StatusToTritonError(const Status& status)
{
  if (status.IsOk()) {
    return nullptr;
  }
  
  TRITONSERVER_Error_Code code = TRITONSERVER_ERROR_INTERNAL;
  switch (status.StatusCode()) {
    case Status::Code::INVALID:
      code = TRITONSERVER_ERROR_INVALID_ARG;
      break;
    case Status::Code::UNAVAILABLE:
      code = TRITONSERVER_ERROR_UNAVAILABLE;
      break;
    case Status::Code::ALREADY_EXISTS:
      code = TRITONSERVER_ERROR_ALREADY_EXISTS;
      break;
    case Status::Code::NOT_FOUND:
      code = TRITONSERVER_ERROR_NOT_FOUND;
      break;
    case Status::Code::INTERNAL:
    default:
      code = TRITONSERVER_ERROR_INTERNAL;
      break;
  }
  
  return TRITONSERVER_ErrorNew(code, status.Message().c_str());
}

}  // namespace

//
// MetalResponseAllocator Implementation
//

TRITONSERVER_Error*
MetalResponseAllocator::AllocFn(
    TRITONSERVER_ResponseAllocator* allocator,
    const char* tensor_name,
    size_t byte_size,
    TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id,
    void* userp,
    void** buffer,
    void** buffer_userp,
    TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  // Check if Metal is requested or if we're on a Mac where GPU might mean Metal
  if (IsMetalMemoryType(memory_type)) {
    // Determine the actual Metal memory type to use
    MetalMemoryType metal_type = TritonToMetalMemoryType(memory_type);
    
    // Allocate Metal memory
    std::unique_ptr<MetalBuffer> metal_buffer;
    Status status = MetalMemoryManager::CreateBuffer(
        metal_buffer, byte_size, metal_type, memory_type_id);
    
    if (!status.IsOk()) {
      return StatusToTritonError(status);
    }
    
    // Get the data pointer
    *buffer = metal_buffer->Data();
    if (*buffer == nullptr && metal_type != MetalMemoryType::METAL_BUFFER) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "Failed to get data pointer from Metal buffer");
    }
    
    // For private Metal buffers, we need to store the buffer object
    if (metal_type == MetalMemoryType::METAL_BUFFER) {
      *buffer_userp = metal_buffer.release();
      *buffer = *buffer_userp;  // Use buffer_userp as the identifier
    } else {
      *buffer_userp = metal_buffer.release();
    }
    
    // Set actual memory type
    *actual_memory_type = MetalToTritonMemoryType(metal_type);
    *actual_memory_type_id = memory_type_id;
    
    return nullptr;
  }
  
  // Fall back to CPU allocation
  *buffer = malloc(byte_size);
  if (*buffer == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Failed to allocate CPU memory");
  }
  
  *buffer_userp = nullptr;
  *actual_memory_type = TRITONSERVER_MEMORY_CPU;
  *actual_memory_type_id = 0;
  
  return nullptr;
}

TRITONSERVER_Error*
MetalResponseAllocator::ReleaseFn(
    TRITONSERVER_ResponseAllocator* allocator,
    void* buffer,
    void* buffer_userp,
    size_t byte_size,
    TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  if (IsMetalMemoryType(memory_type)) {
    // Delete the MetalBuffer object
    if (buffer_userp != nullptr) {
      MetalBuffer* metal_buffer = static_cast<MetalBuffer*>(buffer_userp);
      delete metal_buffer;
    }
    return nullptr;
  }
  
  // CPU memory
  free(buffer);
  return nullptr;
}

TRITONSERVER_Error*
MetalResponseAllocator::QueryFn(
    TRITONSERVER_ResponseAllocator* allocator,
    void* userp,
    const char* tensor_name,
    size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id)
{
  // Check if Metal is available and initialized
  if (g_metal_initialized.load() && MetalMemoryManager::IsAvailable()) {
    // Prefer unified memory by default for best compatibility
    *memory_type = TRITONSERVER_MEMORY_METAL_UNIFIED;
    *memory_type_id = 0;  // Default device
  } else {
    // Fall back to CPU
    *memory_type = TRITONSERVER_MEMORY_CPU;
    *memory_type_id = 0;
  }
  
  // We don't modify byte_size in the query
  return nullptr;
}

Status
CreateMetalResponseAllocator(
    TRITONSERVER_ResponseAllocator** allocator,
    bool prefer_unified_memory)
{
  TRITONSERVER_Error* err = TRITONSERVER_ResponseAllocatorNew(
      allocator,
      MetalResponseAllocator::AllocFn,
      MetalResponseAllocator::ReleaseFn,
      nullptr);  // No start function needed
  
  if (err != nullptr) {
    Status status = Status(
        Status::Code::INTERNAL,
        TRITONSERVER_ErrorMessage(err));
    TRITONSERVER_ErrorDelete(err);
    return status;
  }
  
  // Set the query function
  err = TRITONSERVER_ResponseAllocatorSetQueryFunction(
      *allocator,
      MetalResponseAllocator::QueryFn);
  
  if (err != nullptr) {
    Status status = Status(
        Status::Code::INTERNAL,
        TRITONSERVER_ErrorMessage(err));
    TRITONSERVER_ErrorDelete(err);
    TRITONSERVER_ResponseAllocatorDelete(*allocator);
    *allocator = nullptr;
    return status;
  }
  
  return Status::Success;
}

//
// Metal utility functions
//

namespace metal {

Status
Initialize(const MetalMemoryManager::Options& options)
{
  if (g_metal_initialized.load()) {
    return Status(
        Status::Code::ALREADY_EXISTS,
        "Metal support already initialized");
  }
  
  if (!MetalMemoryManager::IsAvailable()) {
    return Status(
        Status::Code::UNAVAILABLE,
        "Metal is not available on this system");
  }
  
  Status status = MetalMemoryManager::Create(options);
  if (!status.IsOk()) {
    return status;
  }
  
  g_metal_initialized.store(true);
  return Status::Success;
}

bool
IsInitialized()
{
  return g_metal_initialized.load();
}

void
Shutdown()
{
  if (g_metal_initialized.load()) {
    MetalMemoryManager::Reset();
    g_metal_initialized.store(false);
  }
}

Status
Allocate(
    void** ptr,
    size_t size,
    TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  if (!g_metal_initialized.load()) {
    return Status(
        Status::Code::UNAVAILABLE,
        "Metal support not initialized");
  }
  
  if (IsMetalMemoryType(memory_type)) {
    MetalMemoryType metal_type = TritonToMetalMemoryType(memory_type);
    return MetalMemoryManager::Alloc(ptr, size, metal_type, memory_type_id);
  }
  
  // Fall back to CPU allocation
  *ptr = malloc(size);
  if (*ptr == nullptr) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to allocate CPU memory");
  }
  
  return Status::Success;
}

Status
Free(
    void* ptr,
    TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  if (!g_metal_initialized.load()) {
    return Status(
        Status::Code::UNAVAILABLE,
        "Metal support not initialized");
  }
  
  if (IsMetalMemoryType(memory_type)) {
    return MetalMemoryManager::Free(ptr, memory_type_id);
  }
  
  // CPU memory
  free(ptr);
  return Status::Success;
}

Status
Copy(
    void* dst,
    const void* src,
    size_t size,
    TRITONSERVER_MemoryType dst_memory_type,
    int64_t dst_memory_type_id,
    TRITONSERVER_MemoryType src_memory_type,
    int64_t src_memory_type_id)
{
  if (!g_metal_initialized.load()) {
    return Status(
        Status::Code::UNAVAILABLE,
        "Metal support not initialized");
  }
  
  bool dst_is_metal = IsMetalMemoryType(dst_memory_type);
  bool src_is_metal = IsMetalMemoryType(src_memory_type);
  
  if (!dst_is_metal && !src_is_metal) {
    // CPU to CPU copy
    std::memcpy(dst, src, size);
    return Status::Success;
  }
  
  if (dst_is_metal && !src_is_metal) {
    // Host to Device
    return MetalMemoryManager::CopyHostToDevice(
        dst, src, size, dst_memory_type_id);
  }
  
  if (!dst_is_metal && src_is_metal) {
    // Device to Host
    return MetalMemoryManager::CopyDeviceToHost(
        dst, src, size, src_memory_type_id);
  }
  
  // Device to Device
  return MetalMemoryManager::CopyDeviceToDevice(
        dst, src, size, src_memory_type_id, dst_memory_type_id);
}

Status
GetPreferredMemoryType(
    int64_t device_id,
    TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id)
{
  if (!g_metal_initialized.load()) {
    return Status(
        Status::Code::UNAVAILABLE,
        "Metal support not initialized");
  }
  
  if (device_id >= MetalMemoryManager::DeviceCount()) {
    return Status(
        Status::Code::INVALID,
        "Invalid Metal device ID");
  }
  
  // Prefer unified memory for best compatibility
  *memory_type = TRITONSERVER_MEMORY_METAL_UNIFIED;
  *memory_type_id = device_id;
  
  return Status::Success;
}

Status
GetMemoryStats(
    int64_t device_id,
    size_t* total_memory,
    size_t* available_memory,
    size_t* allocated_memory)
{
  if (!g_metal_initialized.load()) {
    return Status(
        Status::Code::UNAVAILABLE,
        "Metal support not initialized");
  }
  
  Status status = MetalMemoryManager::GetDeviceMemoryInfo(
      device_id, *total_memory, *available_memory);
  
  if (!status.IsOk()) {
    return status;
  }
  
  // Get allocated memory from the allocator registry
  *allocated_memory = MetalAllocatorRegistry::Instance().GetAllocatedMemory(device_id);
  
  return Status::Success;
}

}  // namespace metal

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METAL