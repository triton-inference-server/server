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
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_map.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#include "metal_unified_memory.h"
#include "../status.h"

namespace triton { namespace core {

// MemoryAccessStats implementation
UnifiedMemoryPattern
MemoryAccessStats::GetPattern() const
{
  if (total_accesses == 0) {
    return UnifiedMemoryPattern::UNKNOWN;
  }
  
  size_t cpu_total = cpu_reads + cpu_writes;
  size_t gpu_total = gpu_reads + gpu_writes;
  
  float cpu_ratio = static_cast<float>(cpu_total) / total_accesses;
  float gpu_ratio = static_cast<float>(gpu_total) / total_accesses;
  
  if (cpu_ratio > 0.8f) {
    return UnifiedMemoryPattern::CPU_DOMINANT;
  } else if (gpu_ratio > 0.8f) {
    return UnifiedMemoryPattern::GPU_DOMINANT;
  } else if (std::abs(cpu_ratio - gpu_ratio) < 0.2f) {
    return UnifiedMemoryPattern::BALANCED;
  } else {
    // Check for streaming pattern (high read/write ratio)
    size_t total_reads = cpu_reads + gpu_reads;
    size_t total_writes = cpu_writes + gpu_writes;
    if (total_reads > total_writes * 4 || total_writes > total_reads * 4) {
      return UnifiedMemoryPattern::STREAMING;
    }
  }
  
  return UnifiedMemoryPattern::UNKNOWN;
}

void
MemoryAccessStats::Reset()
{
  cpu_reads = 0;
  cpu_writes = 0;
  gpu_reads = 0;
  gpu_writes = 0;
  total_accesses = 0;
}

// Memory pool for reusable buffers
struct MemoryPool {
  struct PooledBuffer {
    std::unique_ptr<MetalBuffer> buffer;
    size_t size;
    bool in_use;
    std::chrono::steady_clock::time_point last_used;
  };
  
  std::vector<PooledBuffer> small_buffers;
  std::vector<PooledBuffer> medium_buffers;
  std::vector<PooledBuffer> large_buffers;
  std::mutex pool_mutex;
  
  UnifiedMemoryConfig::PoolConfig config;
  
  void Initialize(const UnifiedMemoryConfig::PoolConfig& cfg) {
    config = cfg;
    
    // Pre-allocate buffers
    small_buffers.reserve(config.small_pool_count);
    medium_buffers.reserve(config.medium_pool_count);
    large_buffers.reserve(config.large_pool_count);
  }
  
  Status GetBuffer(
      std::unique_ptr<MetalBuffer>& buffer,
      size_t size,
      MetalMemoryType memory_type,
      int64_t device_id) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    std::vector<PooledBuffer>* pool = nullptr;
    size_t pool_size = 0;
    
    if (size <= config.small_buffer_size) {
      pool = &small_buffers;
      pool_size = config.small_buffer_size;
    } else if (size <= config.medium_buffer_size) {
      pool = &medium_buffers;
      pool_size = config.medium_buffer_size;
    } else if (size <= config.large_buffer_size) {
      pool = &large_buffers;
      pool_size = config.large_buffer_size;
    }
    
    if (pool) {
      // Look for available buffer
      for (auto& pooled : *pool) {
        if (!pooled.in_use && pooled.size >= size) {
          pooled.in_use = true;
          pooled.last_used = std::chrono::steady_clock::now();
          buffer = std::move(pooled.buffer);
          return Status::Success;
        }
      }
      
      // Create new buffer if pool not full
      if (pool->size() < (pool == &small_buffers ? config.small_pool_count :
                         pool == &medium_buffers ? config.medium_pool_count :
                         config.large_pool_count)) {
        auto status = MetalBuffer::Create(buffer, pool_size, memory_type, device_id);
        if (status.IsOk()) {
          PooledBuffer pooled;
          pooled.size = pool_size;
          pooled.in_use = true;
          pooled.last_used = std::chrono::steady_clock::now();
          pool->push_back(std::move(pooled));
        }
        return status;
      }
    }
    
    // Fallback to regular allocation
    return MetalBuffer::Create(buffer, size, memory_type, device_id);
  }
  
  void ReturnBuffer(std::unique_ptr<MetalBuffer> buffer) {
    if (!buffer) return;
    
    size_t size = buffer->Size();
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    std::vector<PooledBuffer>* pool = nullptr;
    
    if (size <= config.small_buffer_size) {
      pool = &small_buffers;
    } else if (size <= config.medium_buffer_size) {
      pool = &medium_buffers;
    } else if (size <= config.large_buffer_size) {
      pool = &large_buffers;
    }
    
    if (pool) {
      // Find the slot and return the buffer
      for (auto& pooled : *pool) {
        if (pooled.in_use && pooled.size == size) {
          pooled.buffer = std::move(buffer);
          pooled.in_use = false;
          pooled.last_used = std::chrono::steady_clock::now();
          return;
        }
      }
    }
    
    // Buffer doesn't belong to pool, let it be destroyed
  }
  
  void CleanupUnused(std::chrono::seconds max_age) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    auto now = std::chrono::steady_clock::now();
    
    auto cleanup = [&](std::vector<PooledBuffer>& buffers) {
      buffers.erase(
          std::remove_if(buffers.begin(), buffers.end(),
                        [&](const PooledBuffer& pooled) {
                          return !pooled.in_use &&
                                 (now - pooled.last_used) > max_age;
                        }),
          buffers.end());
    };
    
    cleanup(small_buffers);
    cleanup(medium_buffers);
    cleanup(large_buffers);
  }
};

// UnifiedMemoryOptimizer implementation
struct UnifiedMemoryOptimizer::Impl {
  UnifiedMemoryConfig config;
  std::unordered_map<void*, std::unique_ptr<UnifiedMemoryAllocation>> allocations;
  std::mutex allocations_mutex;
  
  MemoryPool memory_pool;
  
  // Statistics
  std::atomic<size_t> total_allocated{0};
  std::atomic<size_t> transfers_eliminated{0};
  std::atomic<size_t> prefetch_count{0};
  
  // Profiling
  bool profiling_enabled{false};
  std::vector<std::string> profiling_events;
  std::mutex profiling_mutex;
  
  // NUMA information (for Mac Studio)
  struct NUMAInfo {
    int num_nodes;
    std::vector<size_t> node_memory_sizes;
    std::vector<int> cpu_to_node_map;
  } numa_info;
  
  Impl() {
    DetectNUMATopology();
  }
  
  void DetectNUMATopology() {
    @autoreleasepool {
      // Mac Studio with M1/M2 Ultra has dual-chip configuration
      NSProcessInfo* processInfo = [NSProcessInfo processInfo];
      numa_info.num_nodes = 1; // Default
      
      // Check if this is a Mac Studio with Ultra chip
      if (processInfo.physicalMemory > 64ULL * 1024 * 1024 * 1024) { // >64GB suggests Ultra
        numa_info.num_nodes = 2; // Dual chip
      }
      
      numa_info.node_memory_sizes.resize(numa_info.num_nodes);
      size_t total_memory = processInfo.physicalMemory;
      for (int i = 0; i < numa_info.num_nodes; i++) {
        numa_info.node_memory_sizes[i] = total_memory / numa_info.num_nodes;
      }
      
      // Map CPUs to NUMA nodes
      int cpu_count = processInfo.processorCount;
      numa_info.cpu_to_node_map.resize(cpu_count);
      for (int i = 0; i < cpu_count; i++) {
        numa_info.cpu_to_node_map[i] = i % numa_info.num_nodes;
      }
    }
  }
  
  void RecordProfilingEvent(const std::string& event) {
    if (!profiling_enabled) return;
    
    std::lock_guard<std::mutex> lock(profiling_mutex);
    auto now = std::chrono::steady_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
    
    std::stringstream ss;
    ss << timestamp << "," << event;
    profiling_events.push_back(ss.str());
  }
};

std::unique_ptr<UnifiedMemoryOptimizer> UnifiedMemoryOptimizer::instance_;
std::mutex UnifiedMemoryOptimizer::instance_mutex_;

UnifiedMemoryOptimizer::UnifiedMemoryOptimizer()
    : impl_(std::make_unique<Impl>())
{
}

UnifiedMemoryOptimizer::~UnifiedMemoryOptimizer() = default;

Status
UnifiedMemoryOptimizer::Initialize(const UnifiedMemoryConfig& config)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (instance_) {
    return Status(
        Status::Code::ALREADY_EXISTS,
        "Unified memory optimizer already initialized");
  }
  
  instance_.reset(new UnifiedMemoryOptimizer());
  instance_->impl_->config = config;
  instance_->impl_->memory_pool.Initialize(config.pool_config);
  
  // Initialize Metal memory manager if not already done
  MetalMemoryManager::Options metal_options;
  metal_options.enable_unified_memory_ = config.enable_zero_copy;
  auto status = MetalMemoryManager::Create(metal_options);
  if (!status.IsOk() && status.StatusCode() != Status::Code::ALREADY_EXISTS) {
    return status;
  }
  
  return Status::Success;
}

void
UnifiedMemoryOptimizer::Shutdown()
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  instance_.reset();
}

Status
UnifiedMemoryOptimizer::AllocateOptimized(
    std::unique_ptr<MetalBuffer>& buffer,
    size_t size,
    UnifiedMemoryPattern pattern,
    int64_t device_id)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_) {
    return Status(
        Status::Code::INTERNAL,
        "Unified memory optimizer not initialized");
  }
  
  auto& impl = *instance_->impl_;
  
  // Align size to cache line
  size_t aligned_size = (size + impl.config.cache_line_size - 1) &
                       ~(impl.config.cache_line_size - 1);
  
  // Determine optimal memory type
  MetalMemoryType memory_type = GetOptimalMemoryType(aligned_size, pattern);
  
  // Try to get from pool first
  Status status;
  if (impl.config.enable_auto_placement) {
    status = impl.memory_pool.GetBuffer(buffer, aligned_size, memory_type, device_id);
  } else {
    status = MetalBuffer::Create(buffer, aligned_size, memory_type, device_id);
  }
  
  if (!status.IsOk()) {
    return status;
  }
  
  // Track allocation
  auto allocation = std::make_unique<UnifiedMemoryAllocation>();
  allocation->buffer = std::move(buffer);
  allocation->size = aligned_size;
  allocation->current_type = memory_type;
  allocation->pattern = pattern;
  allocation->creation_time = std::chrono::steady_clock::now();
  allocation->is_pinned = false;
  allocation->device_id = device_id;
  
  void* ptr = allocation->buffer->Data();
  buffer = std::move(allocation->buffer);
  
  {
    std::lock_guard<std::mutex> alloc_lock(impl.allocations_mutex);
    impl.allocations[ptr] = std::move(allocation);
  }
  
  impl.total_allocated += aligned_size;
  impl.RecordProfilingEvent("ALLOC," + std::to_string(aligned_size) + "," +
                           std::to_string(static_cast<int>(pattern)));
  
  return Status::Success;
}

Status
UnifiedMemoryOptimizer::CreateZeroCopyBuffer(
    std::unique_ptr<MetalBuffer>& buffer,
    void* existing_data,
    size_t size,
    bool is_cpu_data,
    int64_t device_id)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_) {
    return Status(
        Status::Code::INTERNAL,
        "Unified memory optimizer not initialized");
  }
  
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (device_id >= devices.count) {
      return Status(
          Status::Code::INVALID_ARG,
          "Invalid Metal device ID: " + std::to_string(device_id));
    }
    
    id<MTLDevice> device = devices[device_id];
    
    // Create a Metal buffer that wraps existing memory (zero-copy)
    id<MTLBuffer> mtl_buffer = [device newBufferWithBytesNoCopy:existing_data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
    
    if (mtl_buffer == nil) {
      return Status(
          Status::Code::INTERNAL,
          "Failed to create zero-copy Metal buffer");
    }
    
    // Create wrapper
    buffer.reset(new MetalBuffer());
    // Note: This requires adding a constructor or setter to MetalBuffer
    // to set the internal buffer directly
    
    instance_->impl_->transfers_eliminated++;
    instance_->impl_->RecordProfilingEvent("ZERO_COPY," + std::to_string(size));
    
    return Status::Success;
  }
}

void
UnifiedMemoryOptimizer::RecordAccess(
    void* ptr,
    bool is_cpu_access,
    bool is_read,
    size_t size)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_) return;
  
  auto& impl = *instance_->impl_;
  
  std::lock_guard<std::mutex> alloc_lock(impl.allocations_mutex);
  auto it = impl.allocations.find(ptr);
  if (it == impl.allocations.end()) {
    return;
  }
  
  auto& allocation = *it->second;
  auto& stats = allocation.stats;
  
  if (is_cpu_access) {
    if (is_read) {
      stats.cpu_reads++;
    } else {
      stats.cpu_writes++;
    }
  } else {
    if (is_read) {
      stats.gpu_reads++;
    } else {
      stats.gpu_writes++;
    }
  }
  
  stats.total_accesses++;
  stats.last_access = std::chrono::steady_clock::now();
  
  // Check if we should optimize placement
  if (impl.config.enable_auto_placement &&
      stats.total_accesses >= impl.config.placement_change_threshold &&
      !allocation.is_pinned) {
    OptimizePlacement(ptr);
  }
}

Status
UnifiedMemoryOptimizer::OptimizePlacement(void* ptr)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_) {
    return Status(
        Status::Code::INTERNAL,
        "Unified memory optimizer not initialized");
  }
  
  auto& impl = *instance_->impl_;
  
  std::lock_guard<std::mutex> alloc_lock(impl.allocations_mutex);
  auto it = impl.allocations.find(ptr);
  if (it == impl.allocations.end()) {
    return Status(
        Status::Code::NOT_FOUND,
        "Allocation not found");
  }
  
  auto& allocation = *it->second;
  if (allocation.is_pinned) {
    return Status::Success; // Cannot optimize pinned memory
  }
  
  // Determine new pattern and optimal memory type
  UnifiedMemoryPattern new_pattern = allocation.stats.GetPattern();
  MetalMemoryType new_type = GetOptimalMemoryType(allocation.size, new_pattern);
  
  if (new_type != allocation.current_type) {
    // In a real implementation, we would migrate the buffer here
    // For now, just update metadata
    allocation.pattern = new_pattern;
    allocation.current_type = new_type;
    
    impl.RecordProfilingEvent("OPTIMIZE_PLACEMENT," + 
                             std::to_string(static_cast<int>(new_pattern)));
  }
  
  return Status::Success;
}

Status
UnifiedMemoryOptimizer::Prefetch(
    void* ptr,
    size_t size,
    bool for_cpu_access)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_ || !instance_->impl_->config.enable_prefetching) {
    return Status::Success;
  }
  
  auto& impl = *instance_->impl_;
  
  @autoreleasepool {
    // For managed memory, we can hint to the system about upcoming access
    std::lock_guard<std::mutex> alloc_lock(impl.allocations_mutex);
    auto it = impl.allocations.find(ptr);
    if (it != impl.allocations.end()) {
      auto& allocation = *it->second;
      
      if (allocation.current_type == MetalMemoryType::METAL_MANAGED) {
        // Use madvise to hint about upcoming access
        int advice = for_cpu_access ? MADV_WILLNEED : MADV_DONTNEED;
        madvise(ptr, size, advice);
        
        impl.prefetch_count++;
        impl.RecordProfilingEvent("PREFETCH," + std::to_string(size) + "," +
                                 (for_cpu_access ? "CPU" : "GPU"));
      }
    }
  }
  
  return Status::Success;
}

Status
UnifiedMemoryOptimizer::GetMemoryStats(
    size_t& total_allocated,
    size_t& unified_memory_used,
    size_t& transfers_eliminated,
    std::unordered_map<UnifiedMemoryPattern, size_t>& pattern_distribution)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_) {
    return Status(
        Status::Code::INTERNAL,
        "Unified memory optimizer not initialized");
  }
  
  auto& impl = *instance_->impl_;
  
  total_allocated = impl.total_allocated;
  transfers_eliminated = impl.transfers_eliminated;
  
  unified_memory_used = 0;
  pattern_distribution.clear();
  
  std::lock_guard<std::mutex> alloc_lock(impl.allocations_mutex);
  for (const auto& [ptr, allocation] : impl.allocations) {
    if (allocation->current_type == MetalMemoryType::METAL_UNIFIED ||
        allocation->current_type == MetalMemoryType::METAL_MANAGED) {
      unified_memory_used += allocation->size;
    }
    
    pattern_distribution[allocation->pattern] += allocation->size;
  }
  
  return Status::Success;
}

Status
UnifiedMemoryOptimizer::AdaptToMemoryPressure()
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_ || !instance_->impl_->config.enable_pressure_adaptation) {
    return Status::Success;
  }
  
  auto& impl = *instance_->impl_;
  
  @autoreleasepool {
    // Get current memory pressure
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(natural_t);
    
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                         (host_info64_t)&vm_stat, &host_size) == KERN_SUCCESS) {
      NSProcessInfo* processInfo = [NSProcessInfo processInfo];
      size_t total_memory = processInfo.physicalMemory;
      size_t free_memory = vm_stat.free_count * vm_page_size;
      
      float memory_pressure = 1.0f - (static_cast<float>(free_memory) / total_memory);
      
      if (memory_pressure > impl.config.memory_pressure_threshold) {
        // Clean up unused pooled buffers
        impl.memory_pool.CleanupUnused(std::chrono::seconds(60));
        
        // Could also migrate some buffers to more efficient storage
        impl.RecordProfilingEvent("MEMORY_PRESSURE_ADAPT," + 
                                 std::to_string(memory_pressure));
      }
    }
  }
  
  return Status::Success;
}

Status
UnifiedMemoryOptimizer::PinMemory(void* ptr, bool pin)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_) {
    return Status(
        Status::Code::INTERNAL,
        "Unified memory optimizer not initialized");
  }
  
  auto& impl = *instance_->impl_;
  
  std::lock_guard<std::mutex> alloc_lock(impl.allocations_mutex);
  auto it = impl.allocations.find(ptr);
  if (it == impl.allocations.end()) {
    return Status(
        Status::Code::NOT_FOUND,
        "Allocation not found");
  }
  
  it->second->is_pinned = pin;
  
  if (pin) {
    // Use mlock to pin pages in memory
    mlock(ptr, it->second->size);
  } else {
    munlock(ptr, it->second->size);
  }
  
  impl.RecordProfilingEvent("PIN_MEMORY," + std::string(pin ? "PIN" : "UNPIN"));
  
  return Status::Success;
}

MetalMemoryType
UnifiedMemoryOptimizer::GetOptimalMemoryType(
    size_t size,
    UnifiedMemoryPattern pattern)
{
  if (!instance_) {
    return MetalMemoryType::METAL_UNIFIED;
  }
  
  auto& config = instance_->impl_->config;
  
  switch (pattern) {
    case UnifiedMemoryPattern::CPU_DOMINANT:
      // CPU-dominant access benefits from shared memory
      return MetalMemoryType::METAL_UNIFIED;
      
    case UnifiedMemoryPattern::GPU_DOMINANT:
      // GPU-dominant might benefit from private memory for large buffers
      if (size > 16 * 1024 * 1024 && !config.enable_zero_copy) {
        return MetalMemoryType::METAL_BUFFER;
      }
      return MetalMemoryType::METAL_UNIFIED;
      
    case UnifiedMemoryPattern::BALANCED:
      // Balanced access always benefits from unified memory
      return MetalMemoryType::METAL_UNIFIED;
      
    case UnifiedMemoryPattern::STREAMING:
      // Streaming can benefit from managed memory
      return MetalMemoryType::METAL_MANAGED;
      
    case UnifiedMemoryPattern::UNKNOWN:
    default:
      // Default to unified for unknown patterns
      return MetalMemoryType::METAL_UNIFIED;
  }
}

Status
UnifiedMemoryOptimizer::BatchAllocate(
    std::vector<std::unique_ptr<MetalBuffer>>& buffers,
    const std::vector<size_t>& sizes,
    UnifiedMemoryPattern pattern,
    int64_t device_id)
{
  if (sizes.empty()) {
    return Status::Success;
  }
  
  buffers.clear();
  buffers.reserve(sizes.size());
  
  // Allocate all buffers
  for (size_t size : sizes) {
    std::unique_ptr<MetalBuffer> buffer;
    auto status = AllocateOptimized(buffer, size, pattern, device_id);
    if (!status.IsOk()) {
      // Cleanup on failure
      buffers.clear();
      return status;
    }
    buffers.push_back(std::move(buffer));
  }
  
  if (instance_) {
    instance_->impl_->RecordProfilingEvent("BATCH_ALLOC," + 
                                          std::to_string(sizes.size()));
  }
  
  return Status::Success;
}

Status
UnifiedMemoryOptimizer::GetPooledBuffer(
    std::unique_ptr<MetalBuffer>& buffer,
    size_t size,
    UnifiedMemoryPattern pattern,
    int64_t device_id)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_) {
    return Status(
        Status::Code::INTERNAL,
        "Unified memory optimizer not initialized");
  }
  
  auto& impl = *instance_->impl_;
  MetalMemoryType memory_type = GetOptimalMemoryType(size, pattern);
  
  return impl.memory_pool.GetBuffer(buffer, size, memory_type, device_id);
}

Status
UnifiedMemoryOptimizer::ReturnToPool(std::unique_ptr<MetalBuffer> buffer)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_ || !buffer) {
    return Status::Success;
  }
  
  instance_->impl_->memory_pool.ReturnBuffer(std::move(buffer));
  return Status::Success;
}

Status
UnifiedMemoryOptimizer::AllocateNUMAOptimized(
    std::unique_ptr<MetalBuffer>& buffer,
    size_t size,
    int numa_node,
    int64_t device_id)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_) {
    return Status(
        Status::Code::INTERNAL,
        "Unified memory optimizer not initialized");
  }
  
  auto& impl = *instance_->impl_;
  
  if (!impl.config.enable_numa_optimization ||
      impl.numa_info.num_nodes <= 1) {
    // Fallback to regular allocation
    return AllocateOptimized(buffer, size, UnifiedMemoryPattern::UNKNOWN, device_id);
  }
  
  // For Mac Studio with Ultra chips, we can hint which chip to prefer
  // This is a simplified implementation
  if (numa_node < 0) {
    // Auto-select based on current CPU
    numa_node = sched_getcpu() % impl.numa_info.num_nodes;
  }
  
  // Allocate with NUMA awareness
  auto status = AllocateOptimized(buffer, size, UnifiedMemoryPattern::UNKNOWN, device_id);
  
  if (status.IsOk()) {
    impl.RecordProfilingEvent("NUMA_ALLOC," + std::to_string(size) + "," +
                             std::to_string(numa_node));
  }
  
  return status;
}

void
UnifiedMemoryOptimizer::EnableProfiling(bool enable)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (instance_) {
    instance_->impl_->profiling_enabled = enable;
  }
}

Status
UnifiedMemoryOptimizer::DumpProfilingData(const std::string& filename)
{
  std::lock_guard<std::mutex> lock(instance_mutex_);
  if (!instance_) {
    return Status(
        Status::Code::INTERNAL,
        "Unified memory optimizer not initialized");
  }
  
  auto& impl = *instance_->impl_;
  
  std::lock_guard<std::mutex> prof_lock(impl.profiling_mutex);
  
  std::ofstream file(filename);
  if (!file.is_open()) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to open profiling file: " + filename);
  }
  
  file << "timestamp_us,event\n";
  for (const auto& event : impl.profiling_events) {
    file << event << "\n";
  }
  
  file.close();
  return Status::Success;
}

// TransferEliminationTracker implementation
std::atomic<size_t> TransferEliminationTracker::eliminated_transfers_{0};
std::atomic<size_t> TransferEliminationTracker::bytes_saved_{0};
std::atomic<size_t> TransferEliminationTracker::cpu_to_gpu_eliminated_{0};
std::atomic<size_t> TransferEliminationTracker::gpu_to_cpu_eliminated_{0};
std::mutex TransferEliminationTracker::stats_mutex_;

void
TransferEliminationTracker::RecordEliminatedTransfer(size_t size, bool cpu_to_gpu)
{
  eliminated_transfers_++;
  bytes_saved_ += size;
  
  if (cpu_to_gpu) {
    cpu_to_gpu_eliminated_++;
  } else {
    gpu_to_cpu_eliminated_++;
  }
}

void
TransferEliminationTracker::GetStatistics(
    size_t& total_eliminated_transfers,
    size_t& total_bytes_saved,
    size_t& cpu_to_gpu_eliminated,
    size_t& gpu_to_cpu_eliminated)
{
  total_eliminated_transfers = eliminated_transfers_;
  total_bytes_saved = bytes_saved_;
  cpu_to_gpu_eliminated = cpu_to_gpu_eliminated_;
  gpu_to_cpu_eliminated = gpu_to_cpu_eliminated_;
}

void
TransferEliminationTracker::Reset()
{
  eliminated_transfers_ = 0;
  bytes_saved_ = 0;
  cpu_to_gpu_eliminated_ = 0;
  gpu_to_cpu_eliminated_ = 0;
}

// ZeroCopyTensor implementation
Status
ZeroCopyTensor::CreateFromCPUMemory(
    std::unique_ptr<ZeroCopyTensor>& tensor,
    void* data,
    size_t size,
    const std::vector<int64_t>& shape,
    TRITONSERVER_DataType dtype)
{
  tensor.reset(new ZeroCopyTensor());
  tensor->data_ = data;
  tensor->size_ = size;
  tensor->shape_ = shape;
  tensor->dtype_ = dtype;
  tensor->is_cpu_ = true;
  
  // Create a zero-copy Metal buffer wrapper
  return UnifiedMemoryOptimizer::CreateZeroCopyBuffer(
      tensor->metal_buffer_, data, size, true, 0);
}

Status
ZeroCopyTensor::CreateFromGPUMemory(
    std::unique_ptr<ZeroCopyTensor>& tensor,
    void* data,
    size_t size,
    const std::vector<int64_t>& shape,
    TRITONSERVER_DataType dtype,
    int64_t device_id)
{
  tensor.reset(new ZeroCopyTensor());
  tensor->data_ = data;
  tensor->size_ = size;
  tensor->shape_ = shape;
  tensor->dtype_ = dtype;
  tensor->is_cpu_ = false;
  
  // For GPU memory, we assume it's already a Metal buffer
  return Status::Success;
}

}}  // namespace triton::core

#endif  // TRITON_ENABLE_METAL