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

#include "src/metal/metal_allocator.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <random>
#include <thread>
#include <vector>

namespace triton { namespace server {

class MetalAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
#ifdef __APPLE__
    // Check if Metal is available
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices.count == 0) {
      GTEST_SKIP() << "No Metal devices available";
    }
#else
    GTEST_SKIP() << "Metal tests require macOS";
#endif
  }
};

TEST_F(MetalAllocatorTest, BasicAllocation)
{
  MetalAllocator allocator(0);
  
  void* buffer = nullptr;
  MetalAllocation* allocation = nullptr;
  
  // Test basic allocation
  auto err = allocator.Allocate(1024, &buffer, &allocation);
  ASSERT_EQ(err, nullptr);
  ASSERT_NE(buffer, nullptr);
  ASSERT_NE(allocation, nullptr);
  EXPECT_EQ(allocation->size, 1024);
  EXPECT_GE(allocation->actual_size, 1024);
  
  // Test statistics
  auto stats = allocator.GetStats();
  EXPECT_GT(stats.total_allocated.load(), 0);
  EXPECT_GT(stats.current_usage.load(), 0);
  EXPECT_EQ(stats.allocation_count.load(), 1);
  
  // Free allocation
  err = allocator.Free(allocation);
  ASSERT_EQ(err, nullptr);
  
  // Check stats after free
  stats = allocator.GetStats();
  EXPECT_EQ(stats.current_usage.load(), 0);
  EXPECT_EQ(stats.free_count.load(), 1);
}

TEST_F(MetalAllocatorTest, PoolAllocation)
{
  MetalPoolConfig config;
  config.size_classes = {256, 1024, 4096};
  config.initial_pool_sizes = {10, 5, 2};
  config.max_pool_sizes = {20, 10, 5};
  
  MetalAllocator allocator(0, config);
  
  // Test pool hit
  void* buffer = nullptr;
  MetalAllocation* allocation = nullptr;
  
  auto err = allocator.Allocate(512, &buffer, &allocation);
  ASSERT_EQ(err, nullptr);
  EXPECT_TRUE(allocation->is_pooled);
  EXPECT_EQ(allocation->actual_size, 1024);  // Should use 1KB pool
  
  auto stats = allocator.GetStats();
  EXPECT_EQ(stats.pool_hits.load(), 1);
  EXPECT_EQ(stats.pool_misses.load(), 0);
  
  // Free and reallocate - should reuse from pool
  err = allocator.Free(allocation);
  ASSERT_EQ(err, nullptr);
  
  void* buffer2 = nullptr;
  MetalAllocation* allocation2 = nullptr;
  
  err = allocator.Allocate(512, &buffer2, &allocation2);
  ASSERT_EQ(err, nullptr);
  
  stats = allocator.GetStats();
  EXPECT_EQ(stats.pool_hits.load(), 2);
  
  err = allocator.Free(allocation2);
  ASSERT_EQ(err, nullptr);
}

TEST_F(MetalAllocatorTest, LargeAllocation)
{
  MetalPoolConfig config;
  config.use_unified_memory = true;
  config.unified_memory_threshold = 1024 * 1024;  // 1MB
  
  MetalAllocator allocator(0, config);
  
  // Test large allocation
  size_t large_size = 16 * 1024 * 1024;  // 16MB
  void* buffer = nullptr;
  MetalAllocation* allocation = nullptr;
  
  auto err = allocator.Allocate(large_size, &buffer, &allocation);
  ASSERT_EQ(err, nullptr);
  EXPECT_FALSE(allocation->is_pooled);
  
  if (allocator.SupportsUnifiedMemory()) {
    EXPECT_TRUE(allocation->is_unified);
  }
  
  auto stats = allocator.GetStats();
  EXPECT_EQ(stats.pool_misses.load(), 1);
  
  err = allocator.Free(allocation);
  ASSERT_EQ(err, nullptr);
}

TEST_F(MetalAllocatorTest, AlignmentRequirements)
{
  MetalAllocator allocator(0);
  
  // Test various alignments
  std::vector<size_t> alignments = {16, 256, 4096};
  
  for (size_t alignment : alignments) {
    void* buffer = nullptr;
    MetalAllocation* allocation = nullptr;
    
    auto err = allocator.AllocateWithRequirements(
        1024, alignment, false, &buffer, &allocation);
    ASSERT_EQ(err, nullptr);
    
    // Check alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer);
    EXPECT_EQ(addr % alignment, 0) << "Buffer not aligned to " << alignment;
    
    err = allocator.Free(allocation);
    ASSERT_EQ(err, nullptr);
  }
}

TEST_F(MetalAllocatorTest, ConcurrentAllocations)
{
  MetalAllocator allocator(0);
  const int num_threads = 4;
  const int allocations_per_thread = 100;
  
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};
  
  auto worker = [&]() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(256, 4096);
    
    std::vector<std::pair<void*, MetalAllocation*>> allocations;
    
    for (int i = 0; i < allocations_per_thread; ++i) {
      void* buffer = nullptr;
      MetalAllocation* allocation = nullptr;
      size_t size = size_dist(gen);
      
      auto err = allocator.Allocate(size, &buffer, &allocation);
      if (err == nullptr) {
        allocations.push_back({buffer, allocation});
        success_count.fetch_add(1);
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
      
      // Random delay
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    // Free all allocations
    for (auto& [buf, alloc] : allocations) {
      auto err = allocator.Free(alloc);
      if (err != nullptr) {
        TRITONSERVER_ErrorDelete(err);
      }
    }
  };
  
  // Start threads
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }
  
  // Wait for completion
  for (auto& t : threads) {
    t.join();
  }
  
  EXPECT_GT(success_count.load(), 0);
  
  auto stats = allocator.GetStats();
  EXPECT_EQ(stats.current_usage.load(), 0);  // All freed
}

TEST_F(MetalAllocatorTest, MemoryPressure)
{
  MetalPoolConfig config;
  config.enable_gc = true;
  config.gc_interval = std::chrono::seconds(1);
  
  MetalAllocator allocator(0, config);
  
  // Allocate a lot of memory
  std::vector<std::pair<void*, MetalAllocation*>> allocations;
  size_t total_allocated = 0;
  const size_t target_size = 100 * 1024 * 1024;  // 100MB
  
  while (total_allocated < target_size) {
    void* buffer = nullptr;
    MetalAllocation* allocation = nullptr;
    size_t size = 1024 * 1024;  // 1MB chunks
    
    auto err = allocator.Allocate(size, &buffer, &allocation);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
      break;
    }
    
    allocations.push_back({buffer, allocation});
    total_allocated += size;
  }
  
  EXPECT_GT(allocations.size(), 0);
  
  // Query available memory
  size_t available, total;
  auto err = allocator.QueryAvailableMemory(&available, &total);
  ASSERT_EQ(err, nullptr);
  EXPECT_GT(total, 0);
  EXPECT_LT(available, total);
  
  // Free half the allocations
  size_t half = allocations.size() / 2;
  for (size_t i = 0; i < half; ++i) {
    err = allocator.Free(allocations[i].second);
    ASSERT_EQ(err, nullptr);
  }
  allocations.erase(allocations.begin(), allocations.begin() + half);
  
  // Wait for GC
  std::this_thread::sleep_for(std::chrono::seconds(2));
  
  auto stats = allocator.GetStats();
  EXPECT_GT(stats.gc_runs.load(), 0);
  
  // Free remaining
  for (auto& [buf, alloc] : allocations) {
    err = allocator.Free(alloc);
    ASSERT_EQ(err, nullptr);
  }
}

TEST_F(MetalAllocatorTest, ResponseAllocator)
{
  auto metal_allocator = std::make_shared<MetalAllocator>(0);
  MetalResponseAllocator response_allocator(metal_allocator);
  
  auto* triton_allocator = response_allocator.GetAllocator();
  ASSERT_NE(triton_allocator, nullptr);
  
  // Test allocation through response allocator
  void* buffer = nullptr;
  void* buffer_userp = nullptr;
  TRITONSERVER_MemoryType actual_memory_type;
  int64_t actual_memory_type_id;
  
  auto err = TRITONSERVER_ResponseAllocatorAllocate(
      triton_allocator, "test_tensor", 1024,
      TRITONSERVER_MEMORY_GPU, METAL_DEVICE_ID_OFFSET,
      &response_allocator, &buffer, &buffer_userp,
      &actual_memory_type, &actual_memory_type_id);
  
  ASSERT_EQ(err, nullptr);
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(actual_memory_type, TRITONSERVER_MEMORY_GPU);
  EXPECT_EQ(actual_memory_type_id, METAL_DEVICE_ID_OFFSET);
  
  // Test release
  err = TRITONSERVER_ResponseAllocatorRelease(
      triton_allocator, buffer, buffer_userp, 1024,
      actual_memory_type, actual_memory_type_id);
  
  ASSERT_EQ(err, nullptr);
}

// Stress test
TEST_F(MetalAllocatorTest, StressTest)
{
  MetalPoolConfig config;
  config.enable_gc = true;
  
  MetalAllocator allocator(0, config);
  
  const int num_iterations = 1000;
  const int max_concurrent = 50;
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(256, 1024 * 1024);
  std::uniform_int_distribution<> action_dist(0, 2);
  
  std::vector<std::pair<void*, MetalAllocation*>> active_allocations;
  
  for (int i = 0; i < num_iterations; ++i) {
    int action = action_dist(gen);
    
    if (action < 2 && active_allocations.size() < max_concurrent) {
      // Allocate
      void* buffer = nullptr;
      MetalAllocation* allocation = nullptr;
      size_t size = size_dist(gen);
      
      auto err = allocator.Allocate(size, &buffer, &allocation);
      if (err == nullptr) {
        active_allocations.push_back({buffer, allocation});
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    } else if (!active_allocations.empty()) {
      // Free random allocation
      std::uniform_int_distribution<> idx_dist(0, active_allocations.size() - 1);
      int idx = idx_dist(gen);
      
      auto err = allocator.Free(active_allocations[idx].second);
      if (err != nullptr) {
        TRITONSERVER_ErrorDelete(err);
      }
      
      active_allocations.erase(active_allocations.begin() + idx);
    }
    
    // Occasionally force GC
    if (i % 100 == 0) {
      allocator.RunGarbageCollection();
    }
  }
  
  // Clean up remaining allocations
  for (auto& [buf, alloc] : active_allocations) {
    auto err = allocator.Free(alloc);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
    }
  }
  
  auto stats = allocator.GetStats();
  EXPECT_EQ(stats.current_usage.load(), 0);
  
  // Print final report
  std::cout << allocator.GetMemoryUsageReport() << std::endl;
}

// Benchmark test
TEST_F(MetalAllocatorTest, BenchmarkPoolVsHeap)
{
  const int num_allocations = 10000;
  const size_t alloc_size = 4096;
  
  // Test with pool
  {
    MetalPoolConfig config;
    config.size_classes = {4096};
    config.initial_pool_sizes = {100};
    config.max_pool_sizes = {1000};
    
    MetalAllocator allocator(0, config);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_allocations; ++i) {
      void* buffer = nullptr;
      MetalAllocation* allocation = nullptr;
      
      auto err = allocator.Allocate(alloc_size, &buffer, &allocation);
      if (err == nullptr) {
        allocator.Free(allocation);
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto pool_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
    
    std::cout << "Pool allocations: " << pool_duration << " us ("
              << pool_duration / num_allocations << " us/allocation)" << std::endl;
  }
  
  // Test without pool
  {
    MetalPoolConfig config;
    config.size_classes = {};  // No pools
    
    MetalAllocator allocator(0, config);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_allocations; ++i) {
      void* buffer = nullptr;
      MetalAllocation* allocation = nullptr;
      
      auto err = allocator.Allocate(alloc_size, &buffer, &allocation);
      if (err == nullptr) {
        allocator.Free(allocation);
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto heap_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
    
    std::cout << "Heap allocations: " << heap_duration << " us ("
              << heap_duration / num_allocations << " us/allocation)" << std::endl;
  }
}

}}  // namespace triton::server

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}