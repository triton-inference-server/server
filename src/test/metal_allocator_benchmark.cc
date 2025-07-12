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

#include <benchmark/benchmark.h>
#include <algorithm>
#include <random>
#include <vector>

namespace triton { namespace server {

// Benchmark small allocations (256B - 4KB)
static void BM_SmallAllocations(benchmark::State& state)
{
  MetalPoolConfig config;
  config.size_classes = {256, 512, 1024, 2048, 4096};
  config.initial_pool_sizes = {100, 50, 50, 25, 25};
  config.max_pool_sizes = {1000, 500, 500, 250, 250};
  
  MetalAllocator allocator(0, config);
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(256, 4096);
  
  for (auto _ : state) {
    void* buffer = nullptr;
    MetalAllocation* allocation = nullptr;
    size_t size = size_dist(gen);
    
    auto err = allocator.Allocate(size, &buffer, &allocation);
    if (err == nullptr) {
      allocator.Free(allocation);
    } else {
      TRITONSERVER_ErrorDelete(err);
      state.SkipWithError("Allocation failed");
    }
  }
  
  state.SetBytesProcessed(state.iterations() * 2048);  // Average size
}
BENCHMARK(BM_SmallAllocations);

// Benchmark medium allocations (4KB - 1MB)
static void BM_MediumAllocations(benchmark::State& state)
{
  MetalPoolConfig config;
  config.size_classes = {4096, 16384, 65536, 262144, 1048576};
  config.initial_pool_sizes = {20, 10, 5, 2, 1};
  config.max_pool_sizes = {100, 50, 25, 10, 5};
  
  MetalAllocator allocator(0, config);
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(4096, 1048576);
  
  for (auto _ : state) {
    void* buffer = nullptr;
    MetalAllocation* allocation = nullptr;
    size_t size = size_dist(gen);
    
    auto err = allocator.Allocate(size, &buffer, &allocation);
    if (err == nullptr) {
      allocator.Free(allocation);
    } else {
      TRITONSERVER_ErrorDelete(err);
      state.SkipWithError("Allocation failed");
    }
  }
  
  state.SetBytesProcessed(state.iterations() * 524288);  // Average size
}
BENCHMARK(BM_MediumAllocations);

// Benchmark large allocations (1MB - 64MB)
static void BM_LargeAllocations(benchmark::State& state)
{
  MetalPoolConfig config;
  config.use_unified_memory = true;
  config.unified_memory_threshold = 1024 * 1024;
  
  MetalAllocator allocator(0, config);
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(1048576, 67108864);
  
  for (auto _ : state) {
    void* buffer = nullptr;
    MetalAllocation* allocation = nullptr;
    size_t size = size_dist(gen);
    
    auto err = allocator.Allocate(size, &buffer, &allocation);
    if (err == nullptr) {
      allocator.Free(allocation);
    } else {
      TRITONSERVER_ErrorDelete(err);
      state.SkipWithError("Allocation failed");
    }
  }
  
  state.SetBytesProcessed(state.iterations() * 33554432);  // Average size
}
BENCHMARK(BM_LargeAllocations);

// Benchmark pool hit rate
static void BM_PoolHitRate(benchmark::State& state)
{
  MetalPoolConfig config;
  config.size_classes = {1024, 4096, 16384};
  config.initial_pool_sizes = {100, 50, 25};
  config.max_pool_sizes = {200, 100, 50};
  
  MetalAllocator allocator(0, config);
  
  // Pre-warm pools
  std::vector<std::pair<void*, MetalAllocation*>> warmup_allocs;
  for (size_t size : config.size_classes) {
    for (int i = 0; i < 10; ++i) {
      void* buffer = nullptr;
      MetalAllocation* allocation = nullptr;
      allocator.Allocate(size, &buffer, &allocation);
      warmup_allocs.push_back({buffer, allocation});
    }
  }
  for (auto& [buf, alloc] : warmup_allocs) {
    allocator.Free(alloc);
  }
  
  // Benchmark with high pool hit rate
  size_t allocation_size = 4096;
  
  for (auto _ : state) {
    void* buffer = nullptr;
    MetalAllocation* allocation = nullptr;
    
    auto err = allocator.Allocate(allocation_size, &buffer, &allocation);
    if (err == nullptr) {
      allocator.Free(allocation);
    } else {
      TRITONSERVER_ErrorDelete(err);
      state.SkipWithError("Allocation failed");
    }
  }
  
  auto stats = allocator.GetStats();
  state.counters["pool_hit_rate"] = 
      (double)stats.pool_hits.load() / 
      (stats.pool_hits.load() + stats.pool_misses.load());
}
BENCHMARK(BM_PoolHitRate);

// Benchmark concurrent allocations
static void BM_ConcurrentAllocations(benchmark::State& state)
{
  MetalPoolConfig config;
  config.size_classes = {1024, 4096, 16384};
  config.initial_pool_sizes = {200, 100, 50};
  config.max_pool_sizes = {1000, 500, 250};
  
  MetalAllocator allocator(0, config);
  
  const int num_threads = state.range(0);
  
  for (auto _ : state) {
    std::vector<std::thread> threads;
    std::atomic<int> allocations_done{0};
    
    auto worker = [&]() {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> size_dist(1024, 16384);
      
      for (int i = 0; i < 100; ++i) {
        void* buffer = nullptr;
        MetalAllocation* allocation = nullptr;
        size_t size = size_dist(gen);
        
        auto err = allocator.Allocate(size, &buffer, &allocation);
        if (err == nullptr) {
          allocations_done.fetch_add(1);
          allocator.Free(allocation);
        } else {
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
    
    state.counters["allocations"] = allocations_done.load();
  }
}
BENCHMARK(BM_ConcurrentAllocations)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

// Benchmark memory fragmentation
static void BM_Fragmentation(benchmark::State& state)
{
  MetalPoolConfig config;
  config.enable_gc = true;
  config.gc_interval = std::chrono::seconds(60);  // Don't interfere
  
  MetalAllocator allocator(0, config);
  
  // Create fragmentation pattern
  std::vector<std::pair<void*, MetalAllocation*>> allocations;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(1024, 1048576);
  
  for (auto _ : state) {
    state.PauseTiming();
    
    // Allocate many buffers
    for (int i = 0; i < 100; ++i) {
      void* buffer = nullptr;
      MetalAllocation* allocation = nullptr;
      size_t size = size_dist(gen);
      
      auto err = allocator.Allocate(size, &buffer, &allocation);
      if (err == nullptr) {
        allocations.push_back({buffer, allocation});
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
    }
    
    // Free every other allocation (create fragmentation)
    for (size_t i = 0; i < allocations.size(); i += 2) {
      allocator.Free(allocations[i].second);
    }
    
    // Remove freed allocations
    allocations.erase(
        std::remove_if(allocations.begin(), allocations.end(),
                       [&, idx = 0](const auto& p) mutable {
                         return idx++ % 2 == 0;
                       }),
        allocations.end());
    
    state.ResumeTiming();
    
    // Try to allocate large buffer (may need defragmentation)
    void* buffer = nullptr;
    MetalAllocation* allocation = nullptr;
    size_t large_size = 10 * 1024 * 1024;  // 10MB
    
    auto err = allocator.Allocate(large_size, &buffer, &allocation);
    if (err == nullptr) {
      allocator.Free(allocation);
    } else {
      TRITONSERVER_ErrorDelete(err);
    }
    
    state.PauseTiming();
    
    // Clean up
    for (auto& [buf, alloc] : allocations) {
      allocator.Free(alloc);
    }
    allocations.clear();
    
    state.ResumeTiming();
  }
  
  auto stats = allocator.GetStats();
  state.counters["fragmentation_bytes"] = stats.fragmentation_bytes.load();
}
BENCHMARK(BM_Fragmentation);

// Benchmark garbage collection impact
static void BM_GarbageCollection(benchmark::State& state)
{
  MetalPoolConfig config;
  config.enable_gc = true;
  config.gc_interval = std::chrono::milliseconds(100);  // Aggressive GC
  
  MetalAllocator allocator(0, config);
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(1024, 65536);
  
  for (auto _ : state) {
    std::vector<std::pair<void*, MetalAllocation*>> allocations;
    
    // Allocate and free in a pattern that triggers GC
    for (int i = 0; i < 1000; ++i) {
      void* buffer = nullptr;
      MetalAllocation* allocation = nullptr;
      size_t size = size_dist(gen);
      
      auto err = allocator.Allocate(size, &buffer, &allocation);
      if (err == nullptr) {
        allocations.push_back({buffer, allocation});
      } else {
        TRITONSERVER_ErrorDelete(err);
      }
      
      // Free old allocations
      if (allocations.size() > 100) {
        for (int j = 0; j < 50; ++j) {
          allocator.Free(allocations[j].second);
        }
        allocations.erase(allocations.begin(), allocations.begin() + 50);
      }
    }
    
    // Clean up
    for (auto& [buf, alloc] : allocations) {
      allocator.Free(alloc);
    }
  }
  
  auto stats = allocator.GetStats();
  state.counters["gc_runs"] = stats.gc_runs.load();
}
BENCHMARK(BM_GarbageCollection);

// Custom allocation strategy benchmark
class AggressivePoolStrategy : public MetalAllocationStrategy {
 public:
  bool ShouldUsePool(size_t byte_size, const MetalAllocationStats& stats) override {
    // Always try to use pool for sizes up to 1MB
    return byte_size <= 1048576;
  }
  
  bool ShouldUseUnifiedMemory(size_t byte_size, const MetalAllocationStats& stats) override {
    // Only use unified for very large allocations
    return byte_size >= 64 * 1024 * 1024;
  }
  
  size_t GetAlignment(size_t byte_size) override {
    return 256;  // Fixed alignment
  }
};

static void BM_CustomStrategy(benchmark::State& state)
{
  MetalPoolConfig config;
  config.size_classes = {4096, 65536, 1048576};
  config.initial_pool_sizes = {100, 50, 10};
  config.max_pool_sizes = {1000, 500, 100};
  
  MetalAllocator allocator(0, config);
  allocator.SetAllocationStrategy(std::make_unique<AggressivePoolStrategy>());
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(1024, 2097152);
  
  for (auto _ : state) {
    void* buffer = nullptr;
    MetalAllocation* allocation = nullptr;
    size_t size = size_dist(gen);
    
    auto err = allocator.Allocate(size, &buffer, &allocation);
    if (err == nullptr) {
      allocator.Free(allocation);
    } else {
      TRITONSERVER_ErrorDelete(err);
      state.SkipWithError("Allocation failed");
    }
  }
  
  auto stats = allocator.GetStats();
  state.counters["pool_utilization"] = 
      (double)stats.pool_hits.load() / 
      (stats.pool_hits.load() + stats.pool_misses.load());
}
BENCHMARK(BM_CustomStrategy);

}}  // namespace triton::server

BENCHMARK_MAIN();