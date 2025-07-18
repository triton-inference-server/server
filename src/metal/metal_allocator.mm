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

#include "metal_allocator.h"

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#endif

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace triton { namespace server {

namespace {

// Helper to align size to power of 2
size_t AlignSize(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

// Helper to find appropriate size class
size_t FindSizeClass(size_t size, const std::vector<size_t>& size_classes) {
  auto it = std::lower_bound(size_classes.begin(), size_classes.end(), size);
  if (it != size_classes.end()) {
    return std::distance(size_classes.begin(), it);
  }
  return size_classes.size(); // Too large for any pool
}

}  // namespace

//
// MetalAllocator Implementation
//

MetalAllocator::MetalAllocator(int64_t device_id, const MetalPoolConfig& config)
    : device_id_(device_id), config_(config), gc_running_(true), gc_shutdown_(false),
      supports_unified_memory_(false), max_buffer_size_(0), total_memory_(0)
{
  auto err = Initialize();
  if (err != nullptr) {
    TRITONSERVER_ErrorDelete(err);
    throw std::runtime_error("Failed to initialize Metal allocator");
  }
}

MetalAllocator::~MetalAllocator()
{
  // Shutdown garbage collection
  {
    std::lock_guard<std::mutex> lock(gc_mutex_);
    gc_shutdown_ = true;
    gc_cv_.notify_all();
  }
  if (gc_thread_.joinable()) {
    gc_thread_.join();
  }

  // Clean up all allocations
  {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    allocations_.clear();
  }

  // Clean up pools
  {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    pools_.clear();
  }

#ifdef __APPLE__
  // Clean up Metal resources
  {
    std::lock_guard<std::mutex> lock(heap_mutex_);
    heaps_.clear();
    device_ = nil;
  }
#endif
}

TRITONSERVER_Error* MetalAllocator::Initialize()
{
#ifdef __APPLE__
  @autoreleasepool {
    // Get Metal device
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices.count == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE, "No Metal devices found");
    }
    
    // Select device based on device_id
    if (device_id_ >= devices.count) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, 
          ("Invalid Metal device ID: " + std::to_string(device_id_)).c_str());
    }
    
    device_ = devices[device_id_];
    
    // Check device capabilities
    supports_unified_memory_ = device_.hasUnifiedMemory;
    max_buffer_size_ = device_.maxBufferLength;
    
    // Get total memory (approximation)
    if (@available(macOS 10.15, *)) {
      total_memory_ = device_.recommendedMaxWorkingSetSize;
    } else {
      // Fallback: use system memory as approximation
      total_memory_ = 8ULL * 1024 * 1024 * 1024; // 8GB default
    }
    
    // Create initial heaps
    if (@available(macOS 10.13, *)) {
      MTLHeapDescriptor* heap_desc = [[MTLHeapDescriptor alloc] init];
      heap_desc.storageMode = supports_unified_memory_ ? 
          MTLStorageModeShared : MTLStorageModePrivate;
      heap_desc.size = 256 * 1024 * 1024; // 256MB initial heap
      
      id<MTLHeap> initial_heap = [device_ newHeapWithDescriptor:heap_desc];
      if (initial_heap) {
        heaps_.push_back(initial_heap);
      }
    }
  }
#endif

  // Initialize memory pools
  size_t num_size_classes = config_.size_classes.size();
  pools_.reserve(num_size_classes);
  
  for (size_t i = 0; i < num_size_classes; ++i) {
    pools_.emplace_back(std::make_unique<MetalMemoryPool>(
        config_.size_classes[i],
        config_.initial_pool_sizes[i],
        config_.max_pool_sizes[i],
        device_id_));
  }
  
  // Initialize allocation strategy
  strategy_ = std::make_unique<DefaultMetalAllocationStrategy>(config_);
  
  // Start garbage collection thread
  if (config_.enable_gc) {
    gc_thread_ = std::thread(&MetalAllocator::GarbageCollectionThread, this);
  }
  
  return nullptr;
}

TRITONSERVER_Error* MetalAllocator::Allocate(
    size_t byte_size, void** buffer, MetalAllocation** allocation)
{
  if (byte_size == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "Cannot allocate 0 bytes");
  }
  
  // Check if we should use pool
  if (strategy_->ShouldUsePool(byte_size, stats_)) {
    auto err = AllocateFromPool(byte_size, buffer, allocation);
    if (err == nullptr) {
      return nullptr;
    }
    TRITONSERVER_ErrorDelete(err);
  }
  
  // Fall back to heap allocation
  size_t alignment = strategy_->GetAlignment(byte_size);
  bool use_unified = strategy_->ShouldUseUnifiedMemory(byte_size, stats_);
  
  return AllocateFromHeap(byte_size, alignment, use_unified, buffer, allocation);
}

TRITONSERVER_Error* MetalAllocator::AllocateWithRequirements(
    size_t byte_size, size_t alignment, bool prefer_unified,
    void** buffer, MetalAllocation** allocation)
{
  if (byte_size == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "Cannot allocate 0 bytes");
  }
  
  return AllocateFromHeap(byte_size, alignment, prefer_unified, buffer, allocation);
}

TRITONSERVER_Error* MetalAllocator::AllocateFromPool(
    size_t byte_size, void** buffer, MetalAllocation** allocation)
{
  size_t size_class_idx = FindSizeClass(byte_size, config_.size_classes);
  if (size_class_idx >= pools_.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE, "Size too large for pool allocation");
  }
  
  std::lock_guard<std::mutex> lock(pool_mutex_);
  auto& pool = pools_[size_class_idx];
  
  if (pool->TryGetBuffer(buffer, allocation)) {
    // Track allocation
    {
      std::lock_guard<std::mutex> alloc_lock(allocation_mutex_);
      allocations_[*buffer] = std::unique_ptr<MetalAllocation>(*allocation);
    }
    
    UpdateAllocationStats((*allocation)->actual_size, true);
    return nullptr;
  }
  
  // Try to grow the pool
  auto err = pool->Grow(1);
  if (err != nullptr) {
    return err;
  }
  
  // Retry allocation
  if (pool->TryGetBuffer(buffer, allocation)) {
    // Track allocation
    {
      std::lock_guard<std::mutex> alloc_lock(allocation_mutex_);
      allocations_[*buffer] = std::unique_ptr<MetalAllocation>(*allocation);
    }
    
    UpdateAllocationStats((*allocation)->actual_size, true);
    return nullptr;
  }
  
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNAVAILABLE, "Failed to allocate from pool");
}

TRITONSERVER_Error* MetalAllocator::AllocateFromHeap(
    size_t byte_size, size_t alignment, bool use_unified,
    void** buffer, MetalAllocation** allocation)
{
#ifdef __APPLE__
  @autoreleasepool {
    size_t aligned_size = AlignSize(byte_size, alignment);
    
    id<MTLBuffer> mtl_buffer = nil;
    id<MTLHeap> mtl_heap = nil;
    
    // Try heap allocation first for private memory
    if (!use_unified && !heaps_.empty()) {
      std::lock_guard<std::mutex> lock(heap_mutex_);
      
      for (auto& heap : heaps_) {
        if (@available(macOS 10.13, *)) {
          if ([heap maxAvailableSizeWithAlignment:alignment] >= aligned_size) {
            mtl_buffer = [heap newBufferWithLength:aligned_size 
                                          options:MTLResourceStorageModePrivate];
            if (mtl_buffer) {
              mtl_heap = heap;
              break;
            }
          }
        }
      }
      
      // Create new heap if needed
      if (!mtl_buffer && heaps_.size() < 10) {  // Limit heap count
        MTLHeapDescriptor* heap_desc = [[MTLHeapDescriptor alloc] init];
        heap_desc.storageMode = MTLStorageModePrivate;
        heap_desc.size = std::max(aligned_size * 2, size_t(256 * 1024 * 1024));
        
        id<MTLHeap> new_heap = [device_ newHeapWithDescriptor:heap_desc];
        if (new_heap) {
          mtl_buffer = [new_heap newBufferWithLength:aligned_size 
                                            options:MTLResourceStorageModePrivate];
          if (mtl_buffer) {
            heaps_.push_back(new_heap);
            mtl_heap = new_heap;
          }
        }
      }
    }
    
    // Fall back to direct allocation
    if (!mtl_buffer) {
      MTLResourceOptions options = use_unified ? 
          MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
      
      mtl_buffer = [device_ newBufferWithLength:aligned_size options:options];
      if (!mtl_buffer) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNAVAILABLE, 
            "Failed to allocate Metal buffer");
      }
    }
    
    // Create allocation record
    auto alloc = std::make_unique<MetalAllocation>();
    alloc->buffer = use_unified ? [mtl_buffer contents] : nullptr;
    alloc->size = byte_size;
    alloc->actual_size = aligned_size;
    alloc->is_pooled = false;
    alloc->is_unified = use_unified;
    alloc->device_id = device_id_;
    alloc->allocation_time = std::chrono::steady_clock::now();
    alloc->mtl_buffer = mtl_buffer;
    alloc->mtl_heap = mtl_heap;
    
    *buffer = alloc->buffer;
    *allocation = alloc.get();
    
    // Track allocation
    {
      std::lock_guard<std::mutex> lock(allocation_mutex_);
      allocations_[alloc->buffer] = std::move(alloc);
    }
    
    UpdateAllocationStats(aligned_size, false);
    
    if (use_unified) {
      stats_.unified_allocations.fetch_add(1);
    } else {
      stats_.heap_allocations.fetch_add(1);
    }
    
    return nullptr;
  }
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, 
      "Metal allocator requires macOS");
#endif
}

TRITONSERVER_Error* MetalAllocator::Free(MetalAllocation* allocation)
{
  if (!allocation) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "Null allocation");
  }
  
  void* buffer = allocation->buffer;
  
  // Find and remove from tracked allocations
  std::unique_ptr<MetalAllocation> alloc_ptr;
  {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    auto it = allocations_.find(buffer);
    if (it == allocations_.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_NOT_FOUND, "Allocation not found");
    }
    alloc_ptr = std::move(it->second);
    allocations_.erase(it);
  }
  
  UpdateFreeStats(alloc_ptr->actual_size, alloc_ptr->is_pooled);
  
  // Return to pool if pooled
  if (alloc_ptr->is_pooled) {
    ReturnToPool(alloc_ptr.release());
  }
  // Otherwise, destructor will handle cleanup
  
  return nullptr;
}

void MetalAllocator::ReturnToPool(MetalAllocation* allocation)
{
  size_t size_class_idx = FindSizeClass(allocation->size, config_.size_classes);
  if (size_class_idx >= pools_.size()) {
    // Should not happen for pooled allocations
    delete allocation;
    return;
  }
  
  std::lock_guard<std::mutex> lock(pool_mutex_);
  auto& pool = pools_[size_class_idx];
  
  if (!pool->ReturnBuffer(allocation)) {
    // Pool is full, delete the allocation
    delete allocation;
  }
}

TRITONSERVER_Error* MetalAllocator::QueryAvailableMemory(
    size_t* available_bytes, size_t* total_bytes)
{
#ifdef __APPLE__
  @autoreleasepool {
    *total_bytes = total_memory_;
    
    // Calculate used memory
    size_t used_memory = stats_.current_usage.load();
    
    // Add heap usage
    {
      std::lock_guard<std::mutex> lock(heap_mutex_);
      for (auto& heap : heaps_) {
        if (@available(macOS 10.13, *)) {
          used_memory += [heap currentAllocatedSize];
        }
      }
    }
    
    *available_bytes = (*total_bytes > used_memory) ? 
        (*total_bytes - used_memory) : 0;
    
    return nullptr;
  }
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, 
      "Metal allocator requires macOS");
#endif
}

void MetalAllocator::UpdateAllocationStats(size_t size, bool from_pool)
{
  stats_.total_allocated.fetch_add(size);
  stats_.current_usage.fetch_add(size);
  stats_.allocation_count.fetch_add(1);
  
  if (from_pool) {
    stats_.pool_hits.fetch_add(1);
  } else {
    stats_.pool_misses.fetch_add(1);
  }
  
  // Update peak usage
  size_t current = stats_.current_usage.load();
  size_t peak = stats_.peak_usage.load();
  while (current > peak && !stats_.peak_usage.compare_exchange_weak(peak, current)) {
    // Retry
  }
}

void MetalAllocator::UpdateFreeStats(size_t size, bool to_pool)
{
  stats_.total_freed.fetch_add(size);
  stats_.current_usage.fetch_sub(size);
  stats_.free_count.fetch_add(1);
}

void MetalAllocator::ResetStats()
{
  // Reset atomic members individually
  stats_.total_allocated.store(0);
  stats_.total_freed.store(0);
  stats_.current_usage.store(0);
  stats_.peak_usage.store(0);
  stats_.allocation_count.store(0);
  stats_.free_count.store(0);
  stats_.pool_hits.store(0);
  stats_.pool_misses.store(0);
  stats_.fragmentation_bytes.store(0);
  stats_.gc_runs.store(0);
  stats_.heap_allocations.store(0);
  stats_.unified_allocations.store(0);
}

void MetalAllocator::RunGarbageCollection()
{
  stats_.gc_runs.fetch_add(1);
  
  // Calculate fragmentation and decide on GC actions
  size_t total_pool_memory = 0;
  size_t free_pool_memory = 0;
  
  {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    for (auto& pool : pools_) {
      size_t pool_total = pool->GetTotalCount() * pool->GetBufferSize();
      size_t pool_free = pool->GetFreeCount() * pool->GetBufferSize();
      total_pool_memory += pool_total;
      free_pool_memory += pool_free;
      
      // Shrink oversized pools
      if (pool->GetFreeCount() > pool->GetTotalCount() / 2) {
        pool->Shrink(pool->GetTotalCount() / 4);
      }
    }
  }
  
  // Update fragmentation stats
  stats_.fragmentation_bytes.store(free_pool_memory);
  
#ifdef __APPLE__
  // Compact heaps
  {
    std::lock_guard<std::mutex> lock(heap_mutex_);
    for (auto& heap : heaps_) {
      if (@available(macOS 10.13, *)) {
        [heap setPurgeableState:MTLPurgeableStateEmpty];
        [heap setPurgeableState:MTLPurgeableStateNonVolatile];
      }
    }
  }
#endif
}

void MetalAllocator::GarbageCollectionThread()
{
  while (gc_running_) {
    std::unique_lock<std::mutex> lock(gc_mutex_);
    if (gc_cv_.wait_for(lock, config_.gc_interval, 
                        [this] { return gc_shutdown_; })) {
      break;
    }
    
    RunGarbageCollection();
  }
}

std::string MetalAllocator::GetMemoryUsageReport() const
{
  std::stringstream ss;
  ss << "Metal Memory Allocator Report (Device " << device_id_ << ")\n";
  ss << "=====================================\n";
  ss << std::fixed << std::setprecision(2);
  
  // Basic stats
  ss << "Total Allocated: " << (stats_.total_allocated.load() / (1024.0 * 1024.0)) << " MB\n";
  ss << "Total Freed: " << (stats_.total_freed.load() / (1024.0 * 1024.0)) << " MB\n";
  ss << "Current Usage: " << (stats_.current_usage.load() / (1024.0 * 1024.0)) << " MB\n";
  ss << "Peak Usage: " << (stats_.peak_usage.load() / (1024.0 * 1024.0)) << " MB\n";
  ss << "\n";
  
  // Allocation stats
  ss << "Allocations: " << stats_.allocation_count.load() << "\n";
  ss << "Frees: " << stats_.free_count.load() << "\n";
  ss << "Pool Hits: " << stats_.pool_hits.load() << "\n";
  ss << "Pool Misses: " << stats_.pool_misses.load() << "\n";
  
  double hit_rate = (stats_.pool_hits.load() + stats_.pool_misses.load() > 0) ?
      (100.0 * stats_.pool_hits.load()) / (stats_.pool_hits.load() + stats_.pool_misses.load()) : 0.0;
  ss << "Pool Hit Rate: " << hit_rate << "%\n";
  ss << "\n";
  
  // Memory type stats
  ss << "Heap Allocations: " << stats_.heap_allocations.load() << "\n";
  ss << "Unified Allocations: " << stats_.unified_allocations.load() << "\n";
  ss << "GC Runs: " << stats_.gc_runs.load() << "\n";
  ss << "Fragmentation: " << (stats_.fragmentation_bytes.load() / (1024.0 * 1024.0)) << " MB\n";
  ss << "\n";
  
  // Pool details
  ss << "Memory Pools:\n";
  for (size_t i = 0; i < pools_.size(); ++i) {
    const auto& pool = pools_[i];
    ss << "  Size Class " << config_.size_classes[i] << " bytes: "
       << pool->GetFreeCount() << "/" << pool->GetTotalCount() << " free\n";
  }
  
  return ss.str();
}

void MetalAllocator::SetAllocationStrategy(std::unique_ptr<MetalAllocationStrategy> strategy)
{
  strategy_ = std::move(strategy);
}

//
// MetalMemoryPool Implementation
//

MetalMemoryPool::MetalMemoryPool(
    size_t buffer_size, size_t initial_count, size_t max_count, int64_t device_id)
    : buffer_size_(buffer_size), max_count_(max_count), total_count_(0),
      device_id_(device_id)
{
#ifdef __APPLE__
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (device_id < devices.count) {
      device_ = devices[device_id];
      
      // Create dedicated heap for pool
      if (@available(macOS 10.13, *)) {
        MTLHeapDescriptor* heap_desc = [[MTLHeapDescriptor alloc] init];
        heap_desc.storageMode = device_.hasUnifiedMemory ? 
            MTLStorageModeShared : MTLStorageModePrivate;
        heap_desc.size = buffer_size * max_count;
        
        heap_ = [device_ newHeapWithDescriptor:heap_desc];
      }
    }
  }
#endif
  
  // Pre-allocate initial buffers
  AllocateBuffers(initial_count);
}

MetalMemoryPool::~MetalMemoryPool()
{
  std::lock_guard<std::mutex> lock(mutex_);
  free_buffers_.clear();
  
#ifdef __APPLE__
  heap_ = nil;
  device_ = nil;
#endif
}

bool MetalMemoryPool::TryGetBuffer(void** buffer, MetalAllocation** allocation)
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (free_buffers_.empty()) {
    return false;
  }
  
  auto alloc = std::move(free_buffers_.front());
  free_buffers_.pop_front();
  
  *buffer = alloc->buffer;
  *allocation = alloc.release();
  
  return true;
}

bool MetalMemoryPool::ReturnBuffer(MetalAllocation* allocation)
{
  if (!allocation || allocation->size > buffer_size_) {
    return false;
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Check if pool is full
  if (free_buffers_.size() >= max_count_) {
    return false;
  }
  
  // Reset allocation time
  allocation->allocation_time = std::chrono::steady_clock::now();
  
  free_buffers_.push_back(std::unique_ptr<MetalAllocation>(allocation));
  return true;
}

TRITONSERVER_Error* MetalMemoryPool::Grow(size_t additional_count)
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  size_t new_total = total_count_ + additional_count;
  if (new_total > max_count_) {
    additional_count = max_count_ - total_count_;
    if (additional_count == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE, "Pool is at maximum capacity");
    }
  }
  
  return AllocateBuffers(additional_count);
}

size_t MetalMemoryPool::Shrink(size_t target_free_count)
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  size_t removed = 0;
  while (free_buffers_.size() > target_free_count && !free_buffers_.empty()) {
    free_buffers_.pop_back();
    total_count_--;
    removed++;
  }
  
  return removed;
}

TRITONSERVER_Error* MetalMemoryPool::AllocateBuffers(size_t count)
{
#ifdef __APPLE__
  @autoreleasepool {
    for (size_t i = 0; i < count; ++i) {
      id<MTLBuffer> mtl_buffer = nil;
      
      // Try heap allocation first
      if (heap_ && @available(macOS 10.13, *)) {
        mtl_buffer = [heap_ newBufferWithLength:buffer_size_ 
                                       options:MTLResourceStorageModePrivate];
      }
      
      // Fall back to direct allocation
      if (!mtl_buffer) {
        MTLResourceOptions options = device_.hasUnifiedMemory ? 
            MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
        
        mtl_buffer = [device_ newBufferWithLength:buffer_size_ options:options];
        if (!mtl_buffer) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNAVAILABLE, 
              "Failed to allocate Metal buffer for pool");
        }
      }
      
      auto alloc = std::make_unique<MetalAllocation>();
      alloc->buffer = device_.hasUnifiedMemory ? [mtl_buffer contents] : nullptr;
      alloc->size = buffer_size_;
      alloc->actual_size = buffer_size_;
      alloc->is_pooled = true;
      alloc->is_unified = device_.hasUnifiedMemory;
      alloc->device_id = device_id_;
      alloc->allocation_time = std::chrono::steady_clock::now();
      alloc->mtl_buffer = mtl_buffer;
      alloc->mtl_heap = heap_;
      
      free_buffers_.push_back(std::move(alloc));
      total_count_++;
    }
    
    return nullptr;
  }
#else
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, 
      "Metal allocator requires macOS");
#endif
}

size_t MetalMemoryPool::GetFragmentation() const
{
  std::lock_guard<std::mutex> lock(
      const_cast<std::mutex&>(mutex_));  // Const cast for const method
  return free_buffers_.size() * buffer_size_;
}

//
// DefaultMetalAllocationStrategy Implementation
//

bool DefaultMetalAllocationStrategy::ShouldUsePool(
    size_t byte_size, const MetalAllocationStats& stats)
{
  // Don't use pool if it's too large
  if (config_.size_classes.empty() || 
      byte_size > config_.size_classes.back()) {
    return false;
  }
  
  // Use pool for sizes that fit in a size class
  return true;
}

bool DefaultMetalAllocationStrategy::ShouldUseUnifiedMemory(
    size_t byte_size, const MetalAllocationStats& stats)
{
  if (!config_.use_unified_memory) {
    return false;
  }
  
  // Use unified memory for large allocations
  return byte_size >= config_.unified_memory_threshold;
}

size_t DefaultMetalAllocationStrategy::GetAlignment(size_t byte_size)
{
  // Metal typically requires 256-byte alignment for buffers
  // Larger alignments for larger buffers
  if (byte_size >= 1024 * 1024) {  // 1MB
    return 4096;  // Page alignment
  } else if (byte_size >= 16384) {  // 16KB
    return 256;
  } else {
    return 16;  // Minimum alignment
  }
}

//
// MetalResponseAllocator Implementation
//

MetalResponseAllocator::MetalResponseAllocator(
    std::shared_ptr<MetalAllocator> allocator)
    : metal_allocator_(allocator), allocator_(nullptr)
{
  TRITONSERVER_Error* err = TRITONSERVER_ResponseAllocatorNew(
      &allocator_, ResponseAlloc, ResponseRelease, ResponseStart);
  
  if (err == nullptr) {
    // Set this object as the user data for the allocator
    err = TRITONSERVER_ResponseAllocatorSetUserData(allocator_, this);
  }
  
  if (err != nullptr) {
    TRITONSERVER_ErrorDelete(err);
    throw std::runtime_error("Failed to create response allocator");
  }
}

MetalResponseAllocator::~MetalResponseAllocator()
{
  if (allocator_ != nullptr) {
    TRITONSERVER_ResponseAllocatorDelete(allocator_);
  }
}

TRITONSERVER_Error* MetalResponseAllocator::ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  auto* self = reinterpret_cast<MetalResponseAllocator*>(userp);
  
  // Check if Metal allocation is requested
  if (preferred_memory_type == TRITONSERVER_MEMORY_GPU &&
      preferred_memory_type_id >= METAL_DEVICE_ID_OFFSET) {
    
    MetalAllocation* allocation;
    auto err = self->metal_allocator_->Allocate(byte_size, buffer, &allocation);
    if (err != nullptr) {
      return err;
    }
    
    *buffer_userp = allocation;
    *actual_memory_type = TRITONSERVER_MEMORY_GPU;
    *actual_memory_type_id = preferred_memory_type_id;
    
    return nullptr;
  }
  
  // Fall back to CPU allocation
  *buffer = malloc(byte_size);
  if (*buffer == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE, "Failed to allocate CPU memory");
  }
  
  *buffer_userp = nullptr;
  *actual_memory_type = TRITONSERVER_MEMORY_CPU;
  *actual_memory_type_id = 0;
  
  return nullptr;
}

TRITONSERVER_Error* MetalResponseAllocator::ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer,
    void* buffer_userp, size_t byte_size,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
{
  if (memory_type == TRITONSERVER_MEMORY_GPU &&
      memory_type_id >= METAL_DEVICE_ID_OFFSET && buffer_userp != nullptr) {
    
    // Get the response allocator instance
    auto* self = reinterpret_cast<MetalResponseAllocator*>(
        TRITONSERVER_ResponseAllocatorUserData(allocator));
    
    // The allocation object contains the info needed to free it
    auto* allocation = reinterpret_cast<MetalAllocation*>(buffer_userp);
    
    // Properly free the Metal allocation
    TRITONSERVER_Error* err = self->metal_allocator_->Free(allocation);
    if (err != nullptr) {
      return err;
    }
    
    return nullptr;
  }
  
  // CPU memory
  free(buffer);
  return nullptr;
}

TRITONSERVER_Error* MetalResponseAllocator::ResponseStart(
    TRITONSERVER_ResponseAllocator* allocator, void* userp)
{
  // No special start actions needed
  return nullptr;
}

}}  // namespace triton::server