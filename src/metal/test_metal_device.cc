// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include <chrono>
#include "metal_device.h"
#include "metal_memory_manager.h"
#include "metal_backend_utils.h"

using namespace triton::core::metal;

// Test basic device enumeration
void TestDeviceEnumeration() {
  std::cout << "\n=== Testing Device Enumeration ===" << std::endl;
  
  if (!MetalBackendUtils::Initialize()) {
    std::cerr << "Failed to initialize Metal backend" << std::endl;
    return;
  }
  
  std::cout << MetalBackendUtils::GetAllDeviceProperties() << std::endl;
}

// Test memory allocation
void TestMemoryAllocation() {
  std::cout << "\n=== Testing Memory Allocation ===" << std::endl;
  
  auto& pool = MetalMemoryPool::Instance();
  
  // Test allocation on each device
  size_t device_count = MetalBackendUtils::GetDeviceCount();
  for (size_t device_id = 0; device_id < device_count; ++device_id) {
    std::cout << "\nDevice " << device_id << ":" << std::endl;
    
    // Get memory info before allocation
    auto mem_info_before = MetalBackendUtils::GetDeviceMemoryInfo(device_id);
    std::cout << "  Memory before allocation:" << std::endl;
    std::cout << "    Total: " << (mem_info_before.total_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "    Available: " << (mem_info_before.available_bytes / (1024 * 1024)) << " MB" << std::endl;
    
    // Allocate various sizes
    std::vector<size_t> sizes = {1 * 1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024};
    std::vector<std::unique_ptr<MetalBuffer>> buffers;
    
    for (size_t size : sizes) {
      auto buffer = pool.Allocate(device_id, size, true);
      if (buffer) {
        std::cout << "  Allocated " << (size / (1024 * 1024)) << " MB successfully" << std::endl;
        buffers.push_back(std::move(buffer));
      } else {
        std::cout << "  Failed to allocate " << (size / (1024 * 1024)) << " MB" << std::endl;
      }
    }
    
    // Get memory stats
    auto manager = pool.GetMemoryManager(device_id);
    if (manager) {
      auto stats = manager->GetStats();
      std::cout << "  Allocation statistics:" << std::endl;
      std::cout << "    Total allocated: " << (stats.total_allocated / (1024 * 1024)) << " MB" << std::endl;
      std::cout << "    Current usage: " << (stats.current_usage / (1024 * 1024)) << " MB" << std::endl;
      std::cout << "    Peak usage: " << (stats.peak_usage / (1024 * 1024)) << " MB" << std::endl;
      std::cout << "    Allocations: " << stats.allocation_count << std::endl;
    }
  }
}

// Test data transfer
void TestDataTransfer() {
  std::cout << "\n=== Testing Data Transfer ===" << std::endl;
  
  auto& pool = MetalMemoryPool::Instance();
  const size_t data_size = 1024 * 1024;  // 1 MB
  
  // Create test data
  std::vector<float> host_data(data_size / sizeof(float));
  for (size_t i = 0; i < host_data.size(); ++i) {
    host_data[i] = static_cast<float>(i);
  }
  
  // Allocate buffer on device 0
  auto buffer = pool.Allocate(0, data_size, true);
  if (!buffer) {
    std::cerr << "Failed to allocate buffer" << std::endl;
    return;
  }
  
  // Copy data to device
  auto start = std::chrono::high_resolution_clock::now();
  buffer->CopyFromHost(host_data.data(), data_size);
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double bandwidth = (data_size / (1024.0 * 1024.0)) / (duration.count() / 1000000.0);
  std::cout << "  Host to device transfer: " << bandwidth << " MB/s" << std::endl;
  
  // Copy data back
  std::vector<float> result_data(host_data.size());
  start = std::chrono::high_resolution_clock::now();
  buffer->CopyToHost(result_data.data(), data_size);
  end = std::chrono::high_resolution_clock::now();
  
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  bandwidth = (data_size / (1024.0 * 1024.0)) / (duration.count() / 1000000.0);
  std::cout << "  Device to host transfer: " << bandwidth << " MB/s" << std::endl;
  
  // Verify data
  bool correct = true;
  for (size_t i = 0; i < host_data.size(); ++i) {
    if (host_data[i] != result_data[i]) {
      correct = false;
      break;
    }
  }
  std::cout << "  Data verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
}

// Test device affinity
void TestDeviceAffinity() {
  std::cout << "\n=== Testing Device Affinity ===" << std::endl;
  
  auto& manager = MetalDeviceManager::Instance();
  
  // Test thread-local device affinity
  std::vector<std::thread> threads;
  const int num_threads = 4;
  
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([i, &manager]() {
      // Set affinity to device (i % device_count)
      int device_id = i % manager.GetDeviceCount();
      manager.SetThreadDeviceAffinity(device_id);
      
      // Verify affinity
      int affinity = manager.GetThreadDeviceAffinity();
      std::cout << "  Thread " << i << " affinity set to device " << affinity << std::endl;
      
      // Get default device (should respect affinity)
      MetalDevice* device = manager.GetDefaultDevice();
      if (device) {
        std::cout << "  Thread " << i << " default device: " << device->GetName() << std::endl;
      }
      
      // Clear affinity
      manager.ClearThreadDeviceAffinity();
    });
  }
  
  for (auto& thread : threads) {
    thread.join();
  }
}

// Test device selection
void TestDeviceSelection() {
  std::cout << "\n=== Testing Device Selection ===" << std::endl;
  
  auto& manager = MetalDeviceManager::Instance();
  
  // Test best device selection
  std::cout << "  Selecting best discrete device:" << std::endl;
  MetalDevice* discrete = manager.SelectBestDevice(true);
  if (discrete) {
    std::cout << "    Selected: " << discrete->GetName() << 
                 " (ID: " << discrete->GetDeviceId() << ")" << std::endl;
  }
  
  std::cout << "  Selecting best integrated device:" << std::endl;
  MetalDevice* integrated = manager.SelectBestDevice(false);
  if (integrated) {
    std::cout << "    Selected: " << integrated->GetName() << 
                 " (ID: " << integrated->GetDeviceId() << ")" << std::endl;
  }
  
  // Test model-based selection
  std::vector<size_t> model_sizes = {
    100 * 1024 * 1024,    // 100 MB
    1024 * 1024 * 1024,   // 1 GB
    4 * 1024ULL * 1024 * 1024  // 4 GB
  };
  
  for (size_t size : model_sizes) {
    int device_id = MetalBackendUtils::SelectDeviceForModel(size, true);
    std::cout << "  Model size " << (size / (1024 * 1024)) << " MB -> Device " << device_id << std::endl;
  }
}

// Test instance group configuration
void TestInstanceGroupConfig() {
  std::cout << "\n=== Testing Instance Group Configuration ===" << std::endl;
  
  // Test parsing
  std::vector<std::string> configs = {
    "0:1:default",
    "1:4:high_throughput",
    "0:2:low_latency"
  };
  
  for (const auto& config_str : configs) {
    auto config = MetalInstanceGroupConfig::Parse(config_str);
    std::cout << "  Parsed '" << config_str << "' -> " << config.ToString() << std::endl;
    std::cout << "    Device: " << config.device_id << 
                 ", Count: " << config.count << 
                 ", Profile: " << config.profile << std::endl;
  }
}

// Test command queue
void TestCommandQueue() {
  std::cout << "\n=== Testing Command Queue ===" << std::endl;
  
  auto& manager = MetalDeviceManager::Instance();
  MetalDevice* device = manager.GetDefaultDevice();
  if (!device) {
    std::cerr << "No default device available" << std::endl;
    return;
  }
  
  auto queue = device->CreateCommandQueue();
  if (!queue) {
    std::cerr << "Failed to create command queue" << std::endl;
    return;
  }
  
  std::cout << "  Command queue created successfully" << std::endl;
  
  // Test synchronization
  std::cout << "  Testing synchronization..." << std::endl;
  queue->WaitUntilCompleted();
  std::cout << "  Synchronization completed" << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "Metal Device Management Test Suite" << std::endl;
  std::cout << "==================================" << std::endl;
  
  // Set verbose logging
  MetalLogger::SetLogLevel(MetalLogger::Verbose);
  
  // Check Metal support
  if (!MetalDeviceUtils::IsMetalSupported()) {
    std::cerr << "Metal is not supported on this system" << std::endl;
    return 1;
  }
  
  std::cout << "Metal is supported. Version: " << MetalDeviceUtils::GetMetalVersion() << std::endl;
  
  // Run tests
  TestDeviceEnumeration();
  TestMemoryAllocation();
  TestDataTransfer();
  TestDeviceAffinity();
  TestDeviceSelection();
  TestInstanceGroupConfig();
  TestCommandQueue();
  
  std::cout << "\n=== All Tests Completed ===" << std::endl;
  
  return 0;
}