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

#include "metal_memory.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace core {

// Forward declarations
class Memory;
class MutableMemory;
class AllocatedMemory;

// Extended TRITONSERVER_MemoryType values for Metal
constexpr TRITONSERVER_MemoryType TRITONSERVER_MEMORY_METAL = 
    static_cast<TRITONSERVER_MemoryType>(100);
constexpr TRITONSERVER_MemoryType TRITONSERVER_MEMORY_METAL_UNIFIED = 
    static_cast<TRITONSERVER_MemoryType>(101);
constexpr TRITONSERVER_MemoryType TRITONSERVER_MEMORY_METAL_MANAGED = 
    static_cast<TRITONSERVER_MemoryType>(102);

// Convert MetalMemoryType to extended TRITONSERVER_MemoryType
inline TRITONSERVER_MemoryType
MetalToTritonMemoryType(MetalMemoryType type)
{
  switch (type) {
    case MetalMemoryType::METAL_BUFFER:
      return TRITONSERVER_MEMORY_METAL;
    case MetalMemoryType::METAL_UNIFIED:
      return TRITONSERVER_MEMORY_METAL_UNIFIED;
    case MetalMemoryType::METAL_MANAGED:
      return TRITONSERVER_MEMORY_METAL_MANAGED;
    default:
      return TRITONSERVER_MEMORY_GPU;
  }
}

// Convert TRITONSERVER_MemoryType to MetalMemoryType
inline MetalMemoryType
TritonToMetalMemoryType(TRITONSERVER_MemoryType type)
{
  switch (type) {
    case TRITONSERVER_MEMORY_METAL:
      return MetalMemoryType::METAL_BUFFER;
    case TRITONSERVER_MEMORY_METAL_UNIFIED:
      return MetalMemoryType::METAL_UNIFIED;
    case TRITONSERVER_MEMORY_METAL_MANAGED:
      return MetalMemoryType::METAL_MANAGED;
    default:
      // Default to unified memory for compatibility
      return MetalMemoryType::METAL_UNIFIED;
  }
}

// Check if a memory type is Metal memory
inline bool
IsMetalMemoryType(TRITONSERVER_MemoryType type)
{
  return type == TRITONSERVER_MEMORY_METAL ||
         type == TRITONSERVER_MEMORY_METAL_UNIFIED ||
         type == TRITONSERVER_MEMORY_METAL_MANAGED ||
         type == TRITONSERVER_MEMORY_GPU;  // GPU might be Metal on macOS
}

// Metal-specific response allocator functions
struct MetalResponseAllocator {
  // Allocate function for Metal memory
  static TRITONSERVER_Error* AllocFn(
      TRITONSERVER_ResponseAllocator* allocator,
      const char* tensor_name,
      size_t byte_size,
      TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id,
      void* userp,
      void** buffer,
      void** buffer_userp,
      TRITONSERVER_MemoryType* actual_memory_type,
      int64_t* actual_memory_type_id);

  // Release function for Metal memory
  static TRITONSERVER_Error* ReleaseFn(
      TRITONSERVER_ResponseAllocator* allocator,
      void* buffer,
      void* buffer_userp,
      size_t byte_size,
      TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id);

  // Query function for Metal memory preferences
  static TRITONSERVER_Error* QueryFn(
      TRITONSERVER_ResponseAllocator* allocator,
      void* userp,
      const char* tensor_name,
      size_t* byte_size,
      TRITONSERVER_MemoryType* memory_type,
      int64_t* memory_type_id);
};

// Helper to create a Metal-aware response allocator
Status CreateMetalResponseAllocator(
    TRITONSERVER_ResponseAllocator** allocator,
    bool prefer_unified_memory = true);

// Utility functions for Metal memory management in Triton
namespace metal {

// Initialize Metal support in Triton
Status Initialize(const MetalMemoryManager::Options& options = MetalMemoryManager::Options());

// Check if Metal is available and initialized
bool IsInitialized();

// Shutdown Metal support
void Shutdown();

// Allocate Metal memory for Triton
Status Allocate(
    void** ptr,
    size_t size,
    TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id);

// Free Metal memory
Status Free(
    void* ptr,
    TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id);

// Copy between different memory types (including Metal)
Status Copy(
    void* dst,
    const void* src,
    size_t size,
    TRITONSERVER_MemoryType dst_memory_type,
    int64_t dst_memory_type_id,
    TRITONSERVER_MemoryType src_memory_type,
    int64_t src_memory_type_id);

// Get the preferred memory type for a given device
Status GetPreferredMemoryType(
    int64_t device_id,
    TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id);

// Get device memory statistics
Status GetMemoryStats(
    int64_t device_id,
    size_t* total_memory,
    size_t* available_memory,
    size_t* allocated_memory);

}  // namespace metal

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METAL