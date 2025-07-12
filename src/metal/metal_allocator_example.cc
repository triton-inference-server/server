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

#include <chrono>
#include <iostream>
#include <vector>

using namespace triton::server;

void PrintError(TRITONSERVER_Error* err, const std::string& msg)
{
  if (err != nullptr) {
    std::cerr << msg << ": " << TRITONSERVER_ErrorMessage(err) << std::endl;
    TRITONSERVER_ErrorDelete(err);
  }
}

void DemoBasicUsage()
{
  std::cout << "\n=== Basic Metal Allocator Usage ===" << std::endl;
  
  // Create allocator with default configuration
  MetalAllocator allocator(0);
  
  // Allocate various sizes
  std::vector<std::pair<size_t, std::string>> test_sizes = {
      {256, "256B"},
      {1024, "1KB"},
      {65536, "64KB"},
      {1048576, "1MB"},
      {16777216, "16MB"}
  };
  
  std::vector<std::pair<void*, MetalAllocation*>> allocations;
  
  for (const auto& [size, desc] : test_sizes) {
    void* buffer = nullptr;
    MetalAllocation* allocation = nullptr;
    
    auto err = allocator.Allocate(size, &buffer, &allocation);
    if (err == nullptr) {
      std::cout << "Allocated " << desc << " - "
                << (allocation->is_pooled ? "pooled" : "heap") << ", "
                << (allocation->is_unified ? "unified" : "private") << " memory"
                << std::endl;
      allocations.push_back({buffer, allocation});
    } else {
      PrintError(err, "Failed to allocate " + desc);
    }
  }
  
  // Print statistics
  std::cout << "\nAllocation Statistics:" << std::endl;
  std::cout << allocator.GetMemoryUsageReport() << std::endl;
  
  // Free allocations
  for (auto& [buf, alloc] : allocations) {
    auto err = allocator.Free(alloc);
    if (err != nullptr) {
      PrintError(err, "Failed to free allocation");
    }
  }
  
  std::cout << "\nAfter freeing all allocations:" << std::endl;
  auto stats = allocator.GetStats();
  std::cout << "Current usage: " << stats.current_usage.load() << " bytes" << std::endl;
}

void DemoPoolEfficiency()
{
  std::cout << "\n=== Pool Efficiency Demo ===" << std::endl;
  
  // Configure pools for specific sizes
  MetalPoolConfig config;
  config.size_classes = {1024, 4096, 16384};
  config.initial_pool_sizes = {50, 25, 10};
  config.max_pool_sizes = {100, 50, 20};
  
  MetalAllocator allocator(0, config);
  
  // Measure allocation/deallocation performance
  const int iterations = 1000;
  const size_t alloc_size = 4096;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < iterations; ++i) {
    void* buffer = nullptr;
    MetalAllocation* allocation = nullptr;
    
    auto err = allocator.Allocate(alloc_size, &buffer, &allocation);
    if (err == nullptr) {
      allocator.Free(allocation);
    } else {
      PrintError(err, "Allocation failed");
      break;
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  std::cout << "Performed " << iterations << " allocations/deallocations in "
            << duration.count() << " microseconds" << std::endl;
  std::cout << "Average: " << duration.count() / iterations << " us per operation" << std::endl;
  
  auto stats = allocator.GetStats();
  double hit_rate = (stats.pool_hits.load() + stats.pool_misses.load() > 0) ?
      (100.0 * stats.pool_hits.load()) / (stats.pool_hits.load() + stats.pool_misses.load()) : 0.0;
  std::cout << "Pool hit rate: " << hit_rate << "%" << std::endl;
}

void DemoResponseAllocator()
{
  std::cout << "\n=== Response Allocator Demo ===" << std::endl;
  
  auto metal_allocator = std::make_shared<MetalAllocator>(0);
  MetalResponseAllocator response_allocator(metal_allocator);
  
  auto* triton_allocator = response_allocator.GetAllocator();
  
  // Simulate response allocation
  void* buffer = nullptr;
  void* buffer_userp = nullptr;
  TRITONSERVER_MemoryType actual_memory_type;
  int64_t actual_memory_type_id;
  
  std::cout << "Allocating response buffer for tensor output..." << std::endl;
  
  // Note: This is a simplified call - normally done through TRITONSERVER APIs
  // The ResponseAlloc function would be called internally by Triton
  auto err = MetalResponseAllocator::ResponseAlloc(
      triton_allocator, "output_tensor", 1048576,
      TRITONSERVER_MEMORY_GPU, METAL_DEVICE_ID_OFFSET,
      &response_allocator, &buffer, &buffer_userp,
      &actual_memory_type, &actual_memory_type_id);
  
  if (err == nullptr) {
    std::cout << "Successfully allocated response buffer" << std::endl;
    std::cout << "Memory type: " << 
        (actual_memory_type == TRITONSERVER_MEMORY_GPU ? "GPU" : "CPU") << std::endl;
    std::cout << "Memory type ID: " << actual_memory_type_id << std::endl;
    
    // Release the buffer
    err = MetalResponseAllocator::ResponseRelease(
        triton_allocator, buffer, buffer_userp, 1048576,
        actual_memory_type, actual_memory_type_id);
    
    if (err != nullptr) {
      PrintError(err, "Failed to release response buffer");
    }
  } else {
    PrintError(err, "Failed to allocate response buffer");
  }
}

void DemoMemoryPressure()
{
  std::cout << "\n=== Memory Pressure Demo ===" << std::endl;
  
  MetalPoolConfig config;
  config.enable_gc = true;
  config.gc_interval = std::chrono::seconds(2);
  
  MetalAllocator allocator(0, config);
  
  // Query initial memory state
  size_t available, total;
  auto err = allocator.QueryAvailableMemory(&available, &total);
  if (err == nullptr) {
    std::cout << "Initial memory - Total: " << total / (1024.0 * 1024.0) << " MB, "
              << "Available: " << available / (1024.0 * 1024.0) << " MB" << std::endl;
  } else {
    PrintError(err, "Failed to query memory");
  }
  
  // Allocate significant memory
  std::vector<std::pair<void*, MetalAllocation*>> allocations;
  size_t total_allocated = 0;
  const size_t chunk_size = 4 * 1024 * 1024;  // 4MB chunks
  const size_t target_allocation = 100 * 1024 * 1024;  // 100MB total
  
  std::cout << "\nAllocating memory in 4MB chunks..." << std::endl;
  
  while (total_allocated < target_allocation) {
    void* buffer = nullptr;
    MetalAllocation* allocation = nullptr;
    
    err = allocator.Allocate(chunk_size, &buffer, &allocation);
    if (err == nullptr) {
      allocations.push_back({buffer, allocation});
      total_allocated += chunk_size;
    } else {
      PrintError(err, "Allocation failed");
      break;
    }
  }
  
  std::cout << "Allocated " << allocations.size() << " chunks ("
            << total_allocated / (1024.0 * 1024.0) << " MB)" << std::endl;
  
  // Query memory after allocation
  err = allocator.QueryAvailableMemory(&available, &total);
  if (err == nullptr) {
    std::cout << "After allocation - Available: " 
              << available / (1024.0 * 1024.0) << " MB" << std::endl;
  }
  
  // Free half the allocations
  std::cout << "\nFreeing half the allocations..." << std::endl;
  size_t half = allocations.size() / 2;
  for (size_t i = 0; i < half; ++i) {
    allocator.Free(allocations[i].second);
  }
  allocations.erase(allocations.begin(), allocations.begin() + half);
  
  // Wait for garbage collection
  std::cout << "Waiting for garbage collection..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(3));
  
  // Force GC
  allocator.RunGarbageCollection();
  
  // Check final state
  auto stats = allocator.GetStats();
  std::cout << "\nFinal statistics:" << std::endl;
  std::cout << "GC runs: " << stats.gc_runs.load() << std::endl;
  std::cout << "Fragmentation: " << stats.fragmentation_bytes.load() / (1024.0 * 1024.0) 
            << " MB" << std::endl;
  
  // Clean up remaining allocations
  for (auto& [buf, alloc] : allocations) {
    allocator.Free(alloc);
  }
}

int main(int argc, char** argv)
{
#ifdef __APPLE__
  @autoreleasepool {
    std::cout << "Metal Memory Allocator Example" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // Check for Metal devices
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (devices.count == 0) {
      std::cerr << "No Metal devices found!" << std::endl;
      return 1;
    }
    
    std::cout << "Found " << devices.count << " Metal device(s):" << std::endl;
    for (NSUInteger i = 0; i < devices.count; ++i) {
      id<MTLDevice> device = devices[i];
      std::cout << "  Device " << i << ": " << device.name.UTF8String << std::endl;
      
      if (@available(macOS 10.15, *)) {
        std::cout << "    - Max working set: " 
                  << device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0) 
                  << " GB" << std::endl;
      }
      
      std::cout << "    - Unified memory: " 
                << (device.hasUnifiedMemory ? "Yes" : "No") << std::endl;
      std::cout << "    - Max buffer size: " 
                << device.maxBufferLength / (1024.0 * 1024.0 * 1024.0) 
                << " GB" << std::endl;
    }
    
    try {
      DemoBasicUsage();
      DemoPoolEfficiency();
      DemoResponseAllocator();
      DemoMemoryPressure();
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return 1;
    }
    
    std::cout << "\nExample completed successfully!" << std::endl;
    return 0;
  }
#else
  std::cerr << "This example requires macOS with Metal support." << std::endl;
  return 1;
#endif
}