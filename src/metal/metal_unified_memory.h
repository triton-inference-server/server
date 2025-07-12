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

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "metal_memory.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace core {

// Forward declarations
class Status;

// Unified memory usage patterns
enum class UnifiedMemoryPattern {
  // Memory is primarily accessed by CPU
  CPU_DOMINANT = 0,
  // Memory is primarily accessed by GPU
  GPU_DOMINANT = 1,
  // Memory is accessed equally by both
  BALANCED = 2,
  // Memory is used for streaming data
  STREAMING = 3,
  // Memory access pattern is unknown
  UNKNOWN = 4
};

// Memory access statistics
struct MemoryAccessStats {
  std::atomic<size_t> cpu_reads{0};
  std::atomic<size_t> cpu_writes{0};
  std::atomic<size_t> gpu_reads{0};
  std::atomic<size_t> gpu_writes{0};
  std::atomic<size_t> total_accesses{0};
  std::chrono::steady_clock::time_point last_access;
  
  // Calculate the dominant access pattern
  UnifiedMemoryPattern GetPattern() const;
  
  // Reset statistics
  void Reset();
};

// Unified memory allocation metadata
struct UnifiedMemoryAllocation {
  std::unique_ptr<MetalBuffer> buffer;
  size_t size;
  MetalMemoryType current_type;
  UnifiedMemoryPattern pattern;
  MemoryAccessStats stats;
  std::chrono::steady_clock::time_point creation_time;
  bool is_pinned;
  int64_t device_id;
};

// Unified memory optimization configuration
struct UnifiedMemoryConfig {
  // Enable automatic memory placement optimization
  bool enable_auto_placement = true;
  
  // Threshold for changing memory placement (number of accesses)
  size_t placement_change_threshold = 1000;
  
  // Enable prefetching
  bool enable_prefetching = true;
  
  // Prefetch ahead size (bytes)
  size_t prefetch_size = 4 * 1024 * 1024; // 4MB
  
  // Enable memory pressure adaptation
  bool enable_pressure_adaptation = true;
  
  // Memory pressure threshold (percentage)
  float memory_pressure_threshold = 0.8f;
  
  // Enable NUMA optimizations for Mac Studio
  bool enable_numa_optimization = true;
  
  // Cache line size for alignment
  size_t cache_line_size = 128; // Apple Silicon cache line
  
  // Enable zero-copy optimizations
  bool enable_zero_copy = true;
  
  // Memory pool configurations
  struct PoolConfig {
    size_t small_buffer_size = 64 * 1024;      // 64KB
    size_t medium_buffer_size = 1024 * 1024;   // 1MB
    size_t large_buffer_size = 16 * 1024 * 1024; // 16MB
    
    size_t small_pool_count = 128;
    size_t medium_pool_count = 32;
    size_t large_pool_count = 8;
  } pool_config;
};

// Unified Memory Optimization Manager
class UnifiedMemoryOptimizer {
 public:
  ~UnifiedMemoryOptimizer();
  
  // Initialize the optimizer with configuration
  static Status Initialize(const UnifiedMemoryConfig& config = UnifiedMemoryConfig());
  
  // Shutdown the optimizer
  static void Shutdown();
  
  // Allocate optimized unified memory
  static Status AllocateOptimized(
      std::unique_ptr<MetalBuffer>& buffer,
      size_t size,
      UnifiedMemoryPattern pattern = UnifiedMemoryPattern::UNKNOWN,
      int64_t device_id = 0);
  
  // Create a zero-copy tensor buffer
  static Status CreateZeroCopyBuffer(
      std::unique_ptr<MetalBuffer>& buffer,
      void* existing_data,
      size_t size,
      bool is_cpu_data = true,
      int64_t device_id = 0);
  
  // Record memory access for optimization
  static void RecordAccess(
      void* ptr,
      bool is_cpu_access,
      bool is_read,
      size_t size);
  
  // Optimize memory placement based on access patterns
  static Status OptimizePlacement(void* ptr);
  
  // Prefetch memory for upcoming access
  static Status Prefetch(
      void* ptr,
      size_t size,
      bool for_cpu_access);
  
  // Get memory statistics
  static Status GetMemoryStats(
      size_t& total_allocated,
      size_t& unified_memory_used,
      size_t& transfers_eliminated,
      std::unordered_map<UnifiedMemoryPattern, size_t>& pattern_distribution);
  
  // Adapt to memory pressure
  static Status AdaptToMemoryPressure();
  
  // Pin memory to prevent migration
  static Status PinMemory(void* ptr, bool pin = true);
  
  // Get optimal allocation strategy for size and pattern
  static MetalMemoryType GetOptimalMemoryType(
      size_t size,
      UnifiedMemoryPattern pattern);
  
  // Batch allocation for better performance
  static Status BatchAllocate(
      std::vector<std::unique_ptr<MetalBuffer>>& buffers,
      const std::vector<size_t>& sizes,
      UnifiedMemoryPattern pattern = UnifiedMemoryPattern::UNKNOWN,
      int64_t device_id = 0);
  
  // Memory pool operations
  static Status GetPooledBuffer(
      std::unique_ptr<MetalBuffer>& buffer,
      size_t size,
      UnifiedMemoryPattern pattern = UnifiedMemoryPattern::UNKNOWN,
      int64_t device_id = 0);
  
  static Status ReturnToPool(std::unique_ptr<MetalBuffer> buffer);
  
  // NUMA-aware allocation for Mac Studio
  static Status AllocateNUMAOptimized(
      std::unique_ptr<MetalBuffer>& buffer,
      size_t size,
      int numa_node = -1,  // -1 for auto-selection
      int64_t device_id = 0);
  
  // Debug and profiling
  static void EnableProfiling(bool enable);
  static Status DumpProfilingData(const std::string& filename);
  
 private:
  UnifiedMemoryOptimizer();
  
  struct Impl;
  std::unique_ptr<Impl> impl_;
  
  static std::unique_ptr<UnifiedMemoryOptimizer> instance_;
  static std::mutex instance_mutex_;
};

// Helper class for automatic memory access tracking
class ScopedMemoryAccess {
 public:
  ScopedMemoryAccess(void* ptr, size_t size, bool is_cpu, bool is_read)
      : ptr_(ptr), size_(size), is_cpu_(is_cpu), is_read_(is_read) {
    UnifiedMemoryOptimizer::RecordAccess(ptr_, is_cpu_, is_read_, size_);
  }
  
  ~ScopedMemoryAccess() {
    // Could record end of access here if needed
  }
  
 private:
  void* ptr_;
  size_t size_;
  bool is_cpu_;
  bool is_read_;
};

// Zero-copy tensor wrapper
class ZeroCopyTensor {
 public:
  // Create from existing CPU memory
  static Status CreateFromCPUMemory(
      std::unique_ptr<ZeroCopyTensor>& tensor,
      void* data,
      size_t size,
      const std::vector<int64_t>& shape,
      TRITONSERVER_DataType dtype);
  
  // Create from existing GPU memory
  static Status CreateFromGPUMemory(
      std::unique_ptr<ZeroCopyTensor>& tensor,
      void* data,
      size_t size,
      const std::vector<int64_t>& shape,
      TRITONSERVER_DataType dtype,
      int64_t device_id = 0);
  
  // Get data pointer (CPU or GPU)
  void* Data() { return data_; }
  const void* Data() const { return data_; }
  
  // Get Metal buffer if applicable
  MetalBuffer* GetMetalBuffer() { return metal_buffer_.get(); }
  
  // Get tensor properties
  size_t Size() const { return size_; }
  const std::vector<int64_t>& Shape() const { return shape_; }
  TRITONSERVER_DataType DataType() const { return dtype_; }
  
  // Check if tensor is on CPU or GPU
  bool IsCPU() const { return is_cpu_; }
  bool IsGPU() const { return !is_cpu_; }
  
 private:
  void* data_;
  size_t size_;
  std::vector<int64_t> shape_;
  TRITONSERVER_DataType dtype_;
  bool is_cpu_;
  std::unique_ptr<MetalBuffer> metal_buffer_;
};

// Memory transfer elimination tracker
class TransferEliminationTracker {
 public:
  // Record an eliminated transfer
  static void RecordEliminatedTransfer(size_t size, bool cpu_to_gpu);
  
  // Get statistics
  static void GetStatistics(
      size_t& total_eliminated_transfers,
      size_t& total_bytes_saved,
      size_t& cpu_to_gpu_eliminated,
      size_t& gpu_to_cpu_eliminated);
  
  // Reset statistics
  static void Reset();
  
 private:
  static std::atomic<size_t> eliminated_transfers_;
  static std::atomic<size_t> bytes_saved_;
  static std::atomic<size_t> cpu_to_gpu_eliminated_;
  static std::atomic<size_t> gpu_to_cpu_eliminated_;
  static std::mutex stats_mutex_;
};

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METAL