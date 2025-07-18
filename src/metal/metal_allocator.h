// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __OBJC__
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#else
// Forward declarations for C++ compilation units
#ifdef __APPLE__
typedef struct objc_object* id;
typedef id MTLDevice;
typedef id MTLCommandQueue;
typedef id MTLBuffer;
typedef id MTLHeap;
#endif
#endif

#include "triton/core/tritonserver.h"

namespace triton { namespace server {

// Forward declarations
class MetalMemoryPool;
class MetalAllocationStrategy;

// Statistics for memory allocation tracking
struct MetalAllocationStats {
  std::atomic<size_t> total_allocated{0};
  std::atomic<size_t> total_freed{0};
  std::atomic<size_t> current_usage{0};
  std::atomic<size_t> peak_usage{0};
  std::atomic<size_t> allocation_count{0};
  std::atomic<size_t> free_count{0};
  std::atomic<size_t> pool_hits{0};
  std::atomic<size_t> pool_misses{0};
  std::atomic<size_t> fragmentation_bytes{0};
  std::atomic<size_t> gc_runs{0};
  std::atomic<size_t> heap_allocations{0};
  std::atomic<size_t> unified_allocations{0};
};

// Configuration for memory pools
struct MetalPoolConfig {
  // Size classes for pooled allocations (in bytes)
  std::vector<size_t> size_classes = {
      256,      // 256B
      1024,     // 1KB
      4096,     // 4KB
      16384,    // 16KB
      65536,    // 64KB
      262144,   // 256KB
      1048576,  // 1MB
      4194304,  // 4MB
      16777216, // 16MB
      67108864  // 64MB
  };
  
  // Number of pre-allocated buffers per size class
  std::vector<size_t> initial_pool_sizes = {
      64,  // 256B x 64 = 16KB
      32,  // 1KB x 32 = 32KB
      16,  // 4KB x 16 = 64KB
      8,   // 16KB x 8 = 128KB
      4,   // 64KB x 4 = 256KB
      2,   // 256KB x 2 = 512KB
      2,   // 1MB x 2 = 2MB
      1,   // 4MB x 1 = 4MB
      1,   // 16MB x 1 = 16MB
      1    // 64MB x 1 = 64MB
  };
  
  // Maximum pool sizes per size class
  std::vector<size_t> max_pool_sizes = {
      256, // 256B x 256 = 64KB
      128, // 1KB x 128 = 128KB
      64,  // 4KB x 64 = 256KB
      32,  // 16KB x 32 = 512KB
      16,  // 64KB x 16 = 1MB
      8,   // 256KB x 8 = 2MB
      4,   // 1MB x 4 = 4MB
      2,   // 4MB x 2 = 8MB
      2,   // 16MB x 2 = 32MB
      1    // 64MB x 1 = 64MB
  };
  
  // Enable unified memory for large allocations
  bool use_unified_memory = true;
  size_t unified_memory_threshold = 16 * 1024 * 1024; // 16MB
  
  // Garbage collection settings
  bool enable_gc = true;
  std::chrono::seconds gc_interval{30};
  double gc_fragmentation_threshold = 0.3; // 30% fragmentation triggers GC
  
  // Memory pressure thresholds
  double high_memory_watermark = 0.9; // 90% of available memory
  double low_memory_watermark = 0.7;  // 70% of available memory
};

// Allocation metadata
struct MetalAllocation {
  void* buffer;
  size_t size;
  size_t actual_size;  // May be larger due to alignment/pooling
  bool is_pooled;
  bool is_unified;
  int64_t device_id;
  std::chrono::steady_clock::time_point allocation_time;
#ifdef __APPLE__
#ifdef __OBJC__
  id<MTLBuffer> mtl_buffer;
  id<MTLHeap> mtl_heap;  // If allocated from heap
#else
  id mtl_buffer;
  id mtl_heap;
#endif
#endif
};

// Metal memory allocator interface
class MetalAllocator {
 public:
  // Constructor
  explicit MetalAllocator(
      int64_t device_id, const MetalPoolConfig& config = MetalPoolConfig());
  
  // Destructor
  ~MetalAllocator();
  
  // Allocate memory
  TRITONSERVER_Error* Allocate(
      size_t byte_size, void** buffer, MetalAllocation** allocation);
  
  // Allocate memory with specific requirements
  TRITONSERVER_Error* AllocateWithRequirements(
      size_t byte_size, size_t alignment, bool prefer_unified, 
      void** buffer, MetalAllocation** allocation);
  
  // Free memory
  TRITONSERVER_Error* Free(MetalAllocation* allocation);
  
  // Query available memory
  TRITONSERVER_Error* QueryAvailableMemory(
      size_t* available_bytes, size_t* total_bytes);
  
  // Get allocation statistics
  const MetalAllocationStats& GetStats() const { return stats_; }
  
  // Reset statistics
  void ResetStats();
  
  // Force garbage collection
  void RunGarbageCollection();
  
  // Get device ID
  int64_t DeviceId() const { return device_id_; }
  
  // Check if unified memory is supported
  bool SupportsUnifiedMemory() const { return supports_unified_memory_; }
  
  // Set allocation strategy
  void SetAllocationStrategy(std::unique_ptr<MetalAllocationStrategy> strategy);
  
  // Print memory usage report
  std::string GetMemoryUsageReport() const;

 private:
  // Initialize Metal resources
  TRITONSERVER_Error* Initialize();
  
  // Allocate from pool
  TRITONSERVER_Error* AllocateFromPool(
      size_t byte_size, void** buffer, MetalAllocation** allocation);
  
  // Allocate from heap
  TRITONSERVER_Error* AllocateFromHeap(
      size_t byte_size, size_t alignment, bool use_unified,
      void** buffer, MetalAllocation** allocation);
  
  // Return buffer to pool
  void ReturnToPool(MetalAllocation* allocation);
  
  // Background garbage collection thread
  void GarbageCollectionThread();
  
  // Update statistics
  void UpdateAllocationStats(size_t size, bool from_pool);
  void UpdateFreeStats(size_t size, bool to_pool);
  
  // Member variables
  int64_t device_id_;
  MetalPoolConfig config_;
  MetalAllocationStats stats_;
  
#ifdef __APPLE__
#ifdef __OBJC__
  id<MTLDevice> device_;
  std::vector<id<MTLHeap>> heaps_;
#else
  id device_;
  std::vector<id> heaps_;
#endif
  std::mutex heap_mutex_;
#endif
  
  // Memory pools organized by size class
  std::vector<std::unique_ptr<MetalMemoryPool>> pools_;
  std::mutex pool_mutex_;
  
  // Active allocations tracking
  std::unordered_map<void*, std::unique_ptr<MetalAllocation>> allocations_;
  std::mutex allocation_mutex_;
  
  // Allocation strategy
  std::unique_ptr<MetalAllocationStrategy> strategy_;
  
  // Garbage collection
  std::thread gc_thread_;
  std::condition_variable gc_cv_;
  std::mutex gc_mutex_;
  bool gc_running_;
  bool gc_shutdown_;
  
  // System capabilities
  bool supports_unified_memory_;
  size_t max_buffer_size_;
  size_t total_memory_;
};

// Memory pool for a specific size class
class MetalMemoryPool {
 public:
  MetalMemoryPool(
      size_t buffer_size, size_t initial_count, size_t max_count,
      int64_t device_id);
  ~MetalMemoryPool();
  
  // Try to get a buffer from the pool
  bool TryGetBuffer(void** buffer, MetalAllocation** allocation);
  
  // Return a buffer to the pool
  bool ReturnBuffer(MetalAllocation* allocation);
  
  // Get pool statistics
  size_t GetFreeCount() const { return free_buffers_.size(); }
  size_t GetTotalCount() const { return total_count_; }
  size_t GetBufferSize() const { return buffer_size_; }
  size_t GetFragmentation() const;
  
  // Grow the pool
  TRITONSERVER_Error* Grow(size_t additional_count);
  
  // Shrink the pool (for GC)
  size_t Shrink(size_t target_free_count);

 private:
  // Allocate new buffers
  TRITONSERVER_Error* AllocateBuffers(size_t count);
  
  size_t buffer_size_;
  size_t max_count_;
  size_t total_count_;
  int64_t device_id_;
  
  std::deque<std::unique_ptr<MetalAllocation>> free_buffers_;
  std::mutex mutex_;
  
#ifdef __APPLE__
#ifdef __OBJC__
  id<MTLDevice> device_;
  id<MTLHeap> heap_;  // Dedicated heap for this pool
#else
  id device_;
  id heap_;
#endif
#endif
};

// Allocation strategy interface
class MetalAllocationStrategy {
 public:
  virtual ~MetalAllocationStrategy() = default;
  
  // Decide allocation method based on size and current state
  virtual bool ShouldUsePool(size_t byte_size, const MetalAllocationStats& stats) = 0;
  virtual bool ShouldUseUnifiedMemory(size_t byte_size, const MetalAllocationStats& stats) = 0;
  virtual size_t GetAlignment(size_t byte_size) = 0;
};

// Default allocation strategy
class DefaultMetalAllocationStrategy : public MetalAllocationStrategy {
 public:
  explicit DefaultMetalAllocationStrategy(const MetalPoolConfig& config)
      : config_(config) {}
  
  bool ShouldUsePool(size_t byte_size, const MetalAllocationStats& stats) override;
  bool ShouldUseUnifiedMemory(size_t byte_size, const MetalAllocationStats& stats) override;
  size_t GetAlignment(size_t byte_size) override;

 private:
  MetalPoolConfig config_;
};

// Triton response allocator using Metal
class MetalResponseAllocator {
 public:
  explicit MetalResponseAllocator(std::shared_ptr<MetalAllocator> allocator);
  ~MetalResponseAllocator();
  
  // Get the TRITONSERVER allocator
  TRITONSERVER_ResponseAllocator* GetAllocator() { return allocator_; }

 private:
  // Static callback functions
  static TRITONSERVER_Error* ResponseAlloc(
      TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
      size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
      int64_t preferred_memory_type_id, void* userp, void** buffer,
      void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
      int64_t* actual_memory_type_id);
  
  static TRITONSERVER_Error* ResponseRelease(
      TRITONSERVER_ResponseAllocator* allocator, void* buffer, 
      void* buffer_userp, size_t byte_size,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id);
  
  static TRITONSERVER_Error* ResponseStart(
      TRITONSERVER_ResponseAllocator* allocator, void* userp);
  
  std::shared_ptr<MetalAllocator> metal_allocator_;
  TRITONSERVER_ResponseAllocator* allocator_;
};

// Memory type for Metal (to be added to TRITONSERVER_MemoryType enum)
// For now, we'll use GPU type with Metal-specific device IDs
constexpr int64_t METAL_DEVICE_ID_OFFSET = 1000;

}}  // namespace triton::server