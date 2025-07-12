// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Test program for Metal private buffer operations

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_memory_manager.h"
#include "metal_device.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>

using namespace triton::core::metal;

void TestPrivateBufferCopyFromHost() {
  std::cout << "Testing CopyFromHost for private buffers..." << std::endl;
  
  // Get device
  auto& device_manager = MetalDeviceManager::Instance();
  MetalDevice* device = device_manager.GetDevice(0);
  if (!device) {
    std::cerr << "Failed to get Metal device" << std::endl;
    return;
  }
  
  // Create private buffer
  auto buffer = MetalBuffer::Create(device, 1024, false); // false = private buffer
  assert(buffer != nullptr);
  assert(!buffer->IsShared());
  
  // Prepare test data
  std::vector<float> test_data(256);
  for (size_t i = 0; i < test_data.size(); ++i) {
    test_data[i] = static_cast<float>(i);
  }
  
  // Copy data to private buffer
  buffer->CopyFromHost(test_data.data(), test_data.size() * sizeof(float), 0);
  
  // Read back to verify
  std::vector<float> read_data(256);
  buffer->CopyToHost(read_data.data(), read_data.size() * sizeof(float), 0);
  
  // Verify
  bool success = true;
  for (size_t i = 0; i < test_data.size(); ++i) {
    if (test_data[i] != read_data[i]) {
      std::cerr << "Mismatch at index " << i << ": expected " 
                << test_data[i] << ", got " << read_data[i] << std::endl;
      success = false;
      break;
    }
  }
  
  if (success) {
    std::cout << "✓ CopyFromHost test passed!" << std::endl;
  } else {
    std::cout << "✗ CopyFromHost test failed!" << std::endl;
  }
}

void TestPrivateBufferCopyToHost() {
  std::cout << "Testing CopyToHost for private buffers..." << std::endl;
  
  // Get device
  auto& device_manager = MetalDeviceManager::Instance();
  MetalDevice* device = device_manager.GetDevice(0);
  if (!device) {
    std::cerr << "Failed to get Metal device" << std::endl;
    return;
  }
  
  // Create buffers
  auto src_buffer = MetalBuffer::Create(device, 512, true);  // shared
  auto dst_buffer = MetalBuffer::Create(device, 512, false); // private
  
  // Initialize source buffer
  float* src_data = static_cast<float*>(src_buffer->GetContents());
  for (int i = 0; i < 128; ++i) {
    src_data[i] = static_cast<float>(i * 2);
  }
  
  // Copy to private buffer
  MetalMemoryUtils::CopyBuffer(dst_buffer.get(), src_buffer.get(), 512);
  
  // Read from private buffer
  std::vector<float> read_data(128);
  dst_buffer->CopyToHost(read_data.data(), 512, 0);
  
  // Verify
  bool success = true;
  for (int i = 0; i < 128; ++i) {
    if (src_data[i] != read_data[i]) {
      std::cerr << "Mismatch at index " << i << ": expected " 
                << src_data[i] << ", got " << read_data[i] << std::endl;
      success = false;
      break;
    }
  }
  
  if (success) {
    std::cout << "✓ CopyToHost test passed!" << std::endl;
  } else {
    std::cout << "✗ CopyToHost test failed!" << std::endl;
  }
}

void TestGPUtoGPUCopy() {
  std::cout << "Testing GPU-to-GPU copy..." << std::endl;
  
  // Get device
  auto& device_manager = MetalDeviceManager::Instance();
  MetalDevice* device = device_manager.GetDevice(0);
  if (!device) {
    std::cerr << "Failed to get Metal device" << std::endl;
    return;
  }
  
  // Test 1: Private to Private
  {
    auto src = MetalBuffer::Create(device, 1024, false);
    auto dst = MetalBuffer::Create(device, 1024, false);
    
    // Initialize source
    std::vector<uint8_t> test_data(1024);
    for (size_t i = 0; i < test_data.size(); ++i) {
      test_data[i] = static_cast<uint8_t>(i & 0xFF);
    }
    src->CopyFromHost(test_data.data(), 1024);
    
    // Copy GPU to GPU
    MetalMemoryUtils::CopyBuffer(dst.get(), src.get(), 1024);
    
    // Verify
    std::vector<uint8_t> result(1024);
    dst->CopyToHost(result.data(), 1024);
    
    bool success = (std::memcmp(test_data.data(), result.data(), 1024) == 0);
    std::cout << "  Private->Private: " << (success ? "✓ PASS" : "✗ FAIL") << std::endl;
  }
  
  // Test 2: Shared to Private
  {
    auto src = MetalBuffer::Create(device, 512, true);
    auto dst = MetalBuffer::Create(device, 512, false);
    
    // Initialize source
    uint32_t* src_data = static_cast<uint32_t*>(src->GetContents());
    for (int i = 0; i < 128; ++i) {
      src_data[i] = i * i;
    }
    
    // Copy
    MetalMemoryUtils::CopyBuffer(dst.get(), src.get(), 512);
    
    // Verify
    std::vector<uint32_t> result(128);
    dst->CopyToHost(result.data(), 512);
    
    bool success = true;
    for (int i = 0; i < 128; ++i) {
      if (src_data[i] != result[i]) {
        success = false;
        break;
      }
    }
    std::cout << "  Shared->Private: " << (success ? "✓ PASS" : "✗ FAIL") << std::endl;
  }
  
  // Test 3: Private to Shared
  {
    auto src = MetalBuffer::Create(device, 256, false);
    auto dst = MetalBuffer::Create(device, 256, true);
    
    // Initialize source
    std::vector<int16_t> test_data(128);
    for (size_t i = 0; i < test_data.size(); ++i) {
      test_data[i] = static_cast<int16_t>(i - 64);
    }
    src->CopyFromHost(test_data.data(), 256);
    
    // Copy
    MetalMemoryUtils::CopyBuffer(dst.get(), src.get(), 256);
    
    // Verify
    int16_t* dst_data = static_cast<int16_t*>(dst->GetContents());
    bool success = true;
    for (size_t i = 0; i < test_data.size(); ++i) {
      if (dst_data[i] != test_data[i]) {
        success = false;
        break;
      }
    }
    std::cout << "  Private->Shared: " << (success ? "✓ PASS" : "✗ FAIL") << std::endl;
  }
}

void TestZeroBuffer() {
  std::cout << "Testing ZeroBuffer for private buffers..." << std::endl;
  
  // Get device
  auto& device_manager = MetalDeviceManager::Instance();
  MetalDevice* device = device_manager.GetDevice(0);
  if (!device) {
    std::cerr << "Failed to get Metal device" << std::endl;
    return;
  }
  
  // Create private buffer
  auto buffer = MetalBuffer::Create(device, 1024, false);
  
  // Fill with non-zero data
  std::vector<uint8_t> initial_data(1024, 0xFF);
  buffer->CopyFromHost(initial_data.data(), 1024);
  
  // Zero the buffer
  MetalMemoryUtils::ZeroBuffer(buffer.get());
  
  // Read back
  std::vector<uint8_t> result(1024);
  buffer->CopyToHost(result.data(), 1024);
  
  // Verify all zeros
  bool success = true;
  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i] != 0) {
      std::cerr << "Non-zero value " << (int)result[i] 
                << " at index " << i << std::endl;
      success = false;
      break;
    }
  }
  
  if (success) {
    std::cout << "✓ ZeroBuffer test passed!" << std::endl;
  } else {
    std::cout << "✗ ZeroBuffer test failed!" << std::endl;
  }
}

void TestPartialCopies() {
  std::cout << "Testing partial copies with offsets..." << std::endl;
  
  // Get device
  auto& device_manager = MetalDeviceManager::Instance();
  MetalDevice* device = device_manager.GetDevice(0);
  if (!device) {
    std::cerr << "Failed to get Metal device" << std::endl;
    return;
  }
  
  // Create private buffer
  auto buffer = MetalBuffer::Create(device, 1024, false);
  
  // Test partial write
  std::vector<float> write_data(64);
  for (size_t i = 0; i < write_data.size(); ++i) {
    write_data[i] = static_cast<float>(i * 3.14f);
  }
  
  // Write to offset 256
  buffer->CopyFromHost(write_data.data(), 256, 256);
  
  // Read from offset 256
  std::vector<float> read_data(64);
  buffer->CopyToHost(read_data.data(), 256, 256);
  
  // Verify
  bool success = true;
  for (size_t i = 0; i < write_data.size(); ++i) {
    if (std::abs(write_data[i] - read_data[i]) > 0.0001f) {
      std::cerr << "Mismatch at index " << i << std::endl;
      success = false;
      break;
    }
  }
  
  if (success) {
    std::cout << "✓ Partial copy test passed!" << std::endl;
  } else {
    std::cout << "✗ Partial copy test failed!" << std::endl;
  }
}

int main() {
  @autoreleasepool {
    std::cout << "=== Metal Private Buffer Operations Test ===" << std::endl;
    std::cout << std::endl;
    
    TestPrivateBufferCopyFromHost();
    std::cout << std::endl;
    
    TestPrivateBufferCopyToHost();
    std::cout << std::endl;
    
    TestGPUtoGPUCopy();
    std::cout << std::endl;
    
    TestZeroBuffer();
    std::cout << std::endl;
    
    TestPartialCopies();
    std::cout << std::endl;
    
    std::cout << "=== Test Complete ===" << std::endl;
  }
  
  return 0;
}