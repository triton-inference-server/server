// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Example demonstrating how to use the Metal memory abstraction layer

#ifdef TRITON_ENABLE_METAL

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

#include "metal_memory.h"
#include "triton_metal_memory.h"

using namespace triton::core;

void ExampleBasicUsage()
{
  std::cout << "\n=== Basic Metal Memory Usage ===" << std::endl;
  
  // Check if Metal is available
  if (!MetalMemoryManager::IsAvailable()) {
    std::cout << "Metal is not available on this system" << std::endl;
    return;
  }
  
  std::cout << "Metal devices found: " << MetalMemoryManager::DeviceCount() << std::endl;
  
  // Initialize Metal memory manager
  MetalMemoryManager::Options options;
  options.enable_unified_memory_ = true;
  
  Status status = MetalMemoryManager::Create(options);
  if (!status.IsOk()) {
    std::cerr << "Failed to initialize Metal memory manager: " 
              << status.Message() << std::endl;
    return;
  }
  
  // Get device information
  for (size_t i = 0; i < MetalMemoryManager::DeviceCount(); ++i) {
    std::string device_name;
    size_t total_memory, available_memory;
    
    MetalMemoryManager::GetDeviceName(i, device_name);
    MetalMemoryManager::GetDeviceMemoryInfo(i, total_memory, available_memory);
    
    std::cout << "Device " << i << ": " << device_name << std::endl;
    std::cout << "  Total memory: " << (total_memory / (1024.0 * 1024.0 * 1024.0)) 
              << " GB" << std::endl;
    std::cout << "  Available memory: " << (available_memory / (1024.0 * 1024.0 * 1024.0)) 
              << " GB" << std::endl;
  }
  
  // Clean up
  MetalMemoryManager::Reset();
}

void ExampleUnifiedMemory()
{
  std::cout << "\n=== Unified Memory Example ===" << std::endl;
  
  if (!MetalMemoryManager::IsAvailable()) {
    std::cout << "Metal is not available" << std::endl;
    return;
  }
  
  MetalMemoryManager::Create();
  
  // Create a unified memory buffer (accessible from both CPU and GPU)
  const size_t size = 1024 * 1024;  // 1 MB
  std::unique_ptr<MetalBuffer> buffer;
  
  Status status = MetalBuffer::Create(
      buffer, size, MetalMemoryType::METAL_UNIFIED);
  
  if (!status.IsOk()) {
    std::cerr << "Failed to create buffer: " << status.Message() << std::endl;
    return;
  }
  
  std::cout << "Created unified memory buffer of size: " << size << " bytes" << std::endl;
  
  // Direct CPU access to unified memory
  float* data = static_cast<float*>(buffer->Data());
  size_t num_elements = size / sizeof(float);
  
  // Initialize data on CPU
  for (size_t i = 0; i < num_elements; ++i) {
    data[i] = static_cast<float>(i);
  }
  
  // Data is automatically available on GPU without explicit copy
  std::cout << "Initialized " << num_elements << " float elements" << std::endl;
  
  // Verify data
  bool correct = true;
  for (size_t i = 0; i < num_elements && i < 10; ++i) {
    if (data[i] != static_cast<float>(i)) {
      correct = false;
      break;
    }
  }
  
  std::cout << "Data verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
  
  MetalMemoryManager::Reset();
}

void ExampleDataTransfers()
{
  std::cout << "\n=== Data Transfer Example ===" << std::endl;
  
  if (!MetalMemoryManager::IsAvailable()) {
    std::cout << "Metal is not available" << std::endl;
    return;
  }
  
  MetalMemoryManager::Create();
  
  const size_t size = 1024 * 1024;  // 1 MB
  std::vector<uint8_t> host_data(size);
  
  // Initialize host data
  for (size_t i = 0; i < size; ++i) {
    host_data[i] = static_cast<uint8_t>(i % 256);
  }
  
  // Create a private GPU buffer
  std::unique_ptr<MetalBuffer> gpu_buffer;
  Status status = MetalBuffer::Create(
      gpu_buffer, size, MetalMemoryType::METAL_BUFFER);
  
  if (!status.IsOk()) {
    std::cerr << "Failed to create GPU buffer: " << status.Message() << std::endl;
    return;
  }
  
  // Measure host to device transfer time
  auto start = std::chrono::high_resolution_clock::now();
  status = gpu_buffer->CopyFromHost(host_data.data(), size);
  auto end = std::chrono::high_resolution_clock::now();
  
  if (!status.IsOk()) {
    std::cerr << "Failed to copy to GPU: " << status.Message() << std::endl;
    return;
  }
  
  auto h2d_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double h2d_bandwidth = (size / (1024.0 * 1024.0)) / (h2d_time / 1e6);
  
  std::cout << "Host to Device transfer:" << std::endl;
  std::cout << "  Time: " << h2d_time << " µs" << std::endl;
  std::cout << "  Bandwidth: " << h2d_bandwidth << " MB/s" << std::endl;
  
  // Measure device to host transfer time
  std::vector<uint8_t> result(size);
  start = std::chrono::high_resolution_clock::now();
  status = gpu_buffer->CopyToHost(result.data(), size);
  end = std::chrono::high_resolution_clock::now();
  
  if (!status.IsOk()) {
    std::cerr << "Failed to copy from GPU: " << status.Message() << std::endl;
    return;
  }
  
  auto d2h_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double d2h_bandwidth = (size / (1024.0 * 1024.0)) / (d2h_time / 1e6);
  
  std::cout << "Device to Host transfer:" << std::endl;
  std::cout << "  Time: " << d2h_time << " µs" << std::endl;
  std::cout << "  Bandwidth: " << d2h_bandwidth << " MB/s" << std::endl;
  
  // Verify data
  bool correct = true;
  for (size_t i = 0; i < size; ++i) {
    if (result[i] != host_data[i]) {
      correct = false;
      break;
    }
  }
  
  std::cout << "Data verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
  
  MetalMemoryManager::Reset();
}

void ExampleTritonIntegration()
{
  std::cout << "\n=== Triton Integration Example ===" << std::endl;
  
  if (!MetalMemoryManager::IsAvailable()) {
    std::cout << "Metal is not available" << std::endl;
    return;
  }
  
  // Initialize Metal support in Triton
  Status status = metal::Initialize();
  if (!status.IsOk()) {
    std::cerr << "Failed to initialize Metal support: " 
              << status.Message() << std::endl;
    return;
  }
  
  std::cout << "Metal support initialized in Triton" << std::endl;
  
  // Create a response allocator for Metal memory
  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  status = CreateMetalResponseAllocator(&allocator, true);
  
  if (!status.IsOk()) {
    std::cerr << "Failed to create Metal response allocator: " 
              << status.Message() << std::endl;
    metal::Shutdown();
    return;
  }
  
  std::cout << "Created Metal-aware response allocator" << std::endl;
  
  // Simulate allocation for a tensor
  void* buffer = nullptr;
  void* buffer_userp = nullptr;
  TRITONSERVER_MemoryType actual_memory_type;
  int64_t actual_memory_type_id;
  
  TRITONSERVER_Error* err = MetalResponseAllocator::AllocFn(
      allocator,
      "output_tensor",
      1024 * 1024,  // 1 MB
      TRITONSERVER_MEMORY_METAL_UNIFIED,
      0,  // device 0
      nullptr,
      &buffer,
      &buffer_userp,
      &actual_memory_type,
      &actual_memory_type_id);
  
  if (err != nullptr) {
    std::cerr << "Allocation failed: " << TRITONSERVER_ErrorMessage(err) << std::endl;
    TRITONSERVER_ErrorDelete(err);
  } else {
    std::cout << "Successfully allocated Metal memory for tensor" << std::endl;
    std::cout << "  Memory type: " << actual_memory_type << std::endl;
    std::cout << "  Device ID: " << actual_memory_type_id << std::endl;
    
    // Release the memory
    err = MetalResponseAllocator::ReleaseFn(
        allocator,
        buffer,
        buffer_userp,
        1024 * 1024,
        actual_memory_type,
        actual_memory_type_id);
    
    if (err != nullptr) {
      std::cerr << "Release failed: " << TRITONSERVER_ErrorMessage(err) << std::endl;
      TRITONSERVER_ErrorDelete(err);
    } else {
      std::cout << "Successfully released Metal memory" << std::endl;
    }
  }
  
  // Clean up
  TRITONSERVER_ResponseAllocatorDelete(allocator);
  metal::Shutdown();
}

int main()
{
  std::cout << "Metal Memory Abstraction Layer Examples" << std::endl;
  std::cout << "======================================" << std::endl;
  
  ExampleBasicUsage();
  ExampleUnifiedMemory();
  ExampleDataTransfers();
  ExampleTritonIntegration();
  
  std::cout << "\nAll examples completed!" << std::endl;
  
  return 0;
}

#else

int main()
{
  std::cout << "This example requires TRITON_ENABLE_METAL to be defined." << std::endl;
  std::cout << "Metal support is only available on macOS." << std::endl;
  return 1;
}

#endif  // TRITON_ENABLE_METAL