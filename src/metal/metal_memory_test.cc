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

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cstring>

#include "metal_memory.h"
#include "triton_metal_memory.h"

namespace triton { namespace core {

class MetalMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize Metal memory manager
    if (MetalMemoryManager::IsAvailable()) {
      MetalMemoryManager::Options options;
      options.enable_unified_memory_ = true;
      ASSERT_TRUE(MetalMemoryManager::Create(options).IsOk());
    }
  }

  void TearDown() override {
    MetalMemoryManager::Reset();
  }
};

TEST_F(MetalMemoryTest, Availability)
{
  // Test Metal availability
  bool available = MetalMemoryManager::IsAvailable();
  if (!available) {
    GTEST_SKIP() << "Metal is not available on this system";
  }
  
  EXPECT_GT(MetalMemoryManager::DeviceCount(), 0);
}

TEST_F(MetalMemoryTest, BufferAllocation)
{
  if (!MetalMemoryManager::IsAvailable()) {
    GTEST_SKIP() << "Metal is not available on this system";
  }
  
  // Test different buffer sizes
  std::vector<size_t> sizes = {1024, 1024 * 1024, 16 * 1024 * 1024};
  
  for (size_t size : sizes) {
    std::unique_ptr<MetalBuffer> buffer;
    Status status = MetalBuffer::Create(
        buffer, size, MetalMemoryType::METAL_UNIFIED);
    
    ASSERT_TRUE(status.IsOk()) << "Failed to create buffer of size " << size;
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->Size(), size);
    EXPECT_EQ(buffer->GetMemoryType(), MetalMemoryType::METAL_UNIFIED);
    
    // For unified memory, we should be able to get a data pointer
    void* data = buffer->Data();
    EXPECT_NE(data, nullptr);
  }
}

TEST_F(MetalMemoryTest, MemoryTypes)
{
  if (!MetalMemoryManager::IsAvailable()) {
    GTEST_SKIP() << "Metal is not available on this system";
  }
  
  const size_t size = 1024;
  
  // Test unified memory
  {
    std::unique_ptr<MetalBuffer> buffer;
    Status status = MetalBuffer::Create(
        buffer, size, MetalMemoryType::METAL_UNIFIED);
    
    ASSERT_TRUE(status.IsOk());
    EXPECT_EQ(buffer->GetMemoryType(), MetalMemoryType::METAL_UNIFIED);
    EXPECT_NE(buffer->Data(), nullptr);
  }
  
  // Test managed memory
  {
    std::unique_ptr<MetalBuffer> buffer;
    Status status = MetalBuffer::Create(
        buffer, size, MetalMemoryType::METAL_MANAGED);
    
    ASSERT_TRUE(status.IsOk());
    EXPECT_EQ(buffer->GetMemoryType(), MetalMemoryType::METAL_MANAGED);
    EXPECT_NE(buffer->Data(), nullptr);
  }
  
  // Test private memory
  {
    std::unique_ptr<MetalBuffer> buffer;
    Status status = MetalBuffer::Create(
        buffer, size, MetalMemoryType::METAL_BUFFER);
    
    ASSERT_TRUE(status.IsOk());
    EXPECT_EQ(buffer->GetMemoryType(), MetalMemoryType::METAL_BUFFER);
    // Private memory should not have CPU-accessible pointer
    EXPECT_EQ(buffer->Data(), nullptr);
  }
}

TEST_F(MetalMemoryTest, HostToDeviceCopy)
{
  if (!MetalMemoryManager::IsAvailable()) {
    GTEST_SKIP() << "Metal is not available on this system";
  }
  
  const size_t size = 1024;
  std::vector<uint8_t> host_data(size);
  
  // Fill with test pattern
  for (size_t i = 0; i < size; ++i) {
    host_data[i] = static_cast<uint8_t>(i % 256);
  }
  
  // Test copy to unified memory
  {
    std::unique_ptr<MetalBuffer> buffer;
    Status status = MetalBuffer::Create(
        buffer, size, MetalMemoryType::METAL_UNIFIED);
    
    ASSERT_TRUE(status.IsOk());
    
    // Copy data
    status = buffer->CopyFromHost(host_data.data(), size);
    ASSERT_TRUE(status.IsOk());
    
    // Verify data
    const uint8_t* device_data = static_cast<const uint8_t*>(buffer->Data());
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(device_data[i], host_data[i]);
    }
  }
  
  // Test copy to private memory and back
  {
    std::unique_ptr<MetalBuffer> buffer;
    Status status = MetalBuffer::Create(
        buffer, size, MetalMemoryType::METAL_BUFFER);
    
    ASSERT_TRUE(status.IsOk());
    
    // Copy data to device
    status = buffer->CopyFromHost(host_data.data(), size);
    ASSERT_TRUE(status.IsOk());
    
    // Copy back to host
    std::vector<uint8_t> result(size);
    status = buffer->CopyToHost(result.data(), size);
    ASSERT_TRUE(status.IsOk());
    
    // Verify data
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(result[i], host_data[i]);
    }
  }
}

TEST_F(MetalMemoryTest, DeviceToDeviceCopy)
{
  if (!MetalMemoryManager::IsAvailable()) {
    GTEST_SKIP() << "Metal is not available on this system";
  }
  
  const size_t size = 1024;
  std::vector<uint8_t> host_data(size);
  
  // Fill with test pattern
  for (size_t i = 0; i < size; ++i) {
    host_data[i] = static_cast<uint8_t>(i % 256);
  }
  
  // Create source buffer
  std::unique_ptr<MetalBuffer> src_buffer;
  Status status = MetalBuffer::Create(
      src_buffer, size, MetalMemoryType::METAL_UNIFIED);
  ASSERT_TRUE(status.IsOk());
  
  // Copy data to source
  status = src_buffer->CopyFromHost(host_data.data(), size);
  ASSERT_TRUE(status.IsOk());
  
  // Create destination buffer
  std::unique_ptr<MetalBuffer> dst_buffer;
  status = MetalBuffer::Create(
      dst_buffer, size, MetalMemoryType::METAL_BUFFER);
  ASSERT_TRUE(status.IsOk());
  
  // Copy between buffers
  status = dst_buffer->CopyFrom(*src_buffer, size);
  ASSERT_TRUE(status.IsOk());
  
  // Verify by copying back
  std::vector<uint8_t> result(size);
  status = dst_buffer->CopyToHost(result.data(), size);
  ASSERT_TRUE(status.IsOk());
  
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(result[i], host_data[i]);
  }
}

TEST_F(MetalMemoryTest, PartialCopy)
{
  if (!MetalMemoryManager::IsAvailable()) {
    GTEST_SKIP() << "Metal is not available on this system";
  }
  
  const size_t size = 1024;
  const size_t offset = 256;
  const size_t copy_size = 512;
  
  std::unique_ptr<MetalBuffer> buffer;
  Status status = MetalBuffer::Create(
      buffer, size, MetalMemoryType::METAL_UNIFIED);
  ASSERT_TRUE(status.IsOk());
  
  // Clear buffer
  std::memset(buffer->Data(), 0, size);
  
  // Create test data
  std::vector<uint8_t> test_data(copy_size);
  for (size_t i = 0; i < copy_size; ++i) {
    test_data[i] = static_cast<uint8_t>(i % 256);
  }
  
  // Copy with offset
  status = buffer->CopyFromHost(test_data.data(), copy_size, offset);
  ASSERT_TRUE(status.IsOk());
  
  // Verify
  const uint8_t* device_data = static_cast<const uint8_t*>(buffer->Data());
  
  // Before offset should be zero
  for (size_t i = 0; i < offset; ++i) {
    EXPECT_EQ(device_data[i], 0);
  }
  
  // Copied region should match
  for (size_t i = 0; i < copy_size; ++i) {
    EXPECT_EQ(device_data[offset + i], test_data[i]);
  }
  
  // After copied region should be zero
  for (size_t i = offset + copy_size; i < size; ++i) {
    EXPECT_EQ(device_data[i], 0);
  }
}

TEST_F(MetalMemoryTest, MemoryManagerAllocation)
{
  if (!MetalMemoryManager::IsAvailable()) {
    GTEST_SKIP() << "Metal is not available on this system";
  }
  
  const size_t size = 1024;
  void* ptr = nullptr;
  
  // Test allocation
  Status status = MetalMemoryManager::Alloc(
      &ptr, size, MetalMemoryType::METAL_UNIFIED);
  ASSERT_TRUE(status.IsOk());
  ASSERT_NE(ptr, nullptr);
  
  // Write and read test
  std::memset(ptr, 42, size);
  uint8_t* bytes = static_cast<uint8_t*>(ptr);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(bytes[i], 42);
  }
  
  // Free
  status = MetalMemoryManager::Free(ptr);
  ASSERT_TRUE(status.IsOk());
}

TEST_F(MetalMemoryTest, DeviceInfo)
{
  if (!MetalMemoryManager::IsAvailable()) {
    GTEST_SKIP() << "Metal is not available on this system";
  }
  
  size_t device_count = MetalMemoryManager::DeviceCount();
  ASSERT_GT(device_count, 0);
  
  for (size_t i = 0; i < device_count; ++i) {
    std::string name;
    Status status = MetalMemoryManager::GetDeviceName(i, name);
    ASSERT_TRUE(status.IsOk());
    EXPECT_FALSE(name.empty());
    
    size_t total_memory, available_memory;
    status = MetalMemoryManager::GetDeviceMemoryInfo(
        i, total_memory, available_memory);
    ASSERT_TRUE(status.IsOk());
    EXPECT_GT(total_memory, 0);
    EXPECT_GT(available_memory, 0);
    EXPECT_LE(available_memory, total_memory);
  }
}

TEST_F(MetalMemoryTest, ErrorHandling)
{
  if (!MetalMemoryManager::IsAvailable()) {
    GTEST_SKIP() << "Metal is not available on this system";
  }
  
  // Test invalid device ID
  {
    std::unique_ptr<MetalBuffer> buffer;
    Status status = MetalBuffer::Create(
        buffer, 1024, MetalMemoryType::METAL_UNIFIED, 999);
    EXPECT_FALSE(status.IsOk());
  }
  
  // Test copy beyond buffer bounds
  {
    std::unique_ptr<MetalBuffer> buffer;
    Status status = MetalBuffer::Create(
        buffer, 1024, MetalMemoryType::METAL_UNIFIED);
    ASSERT_TRUE(status.IsOk());
    
    std::vector<uint8_t> data(2048);
    status = buffer->CopyFromHost(data.data(), 2048);
    EXPECT_FALSE(status.IsOk());
  }
  
  // Test partial copy beyond bounds
  {
    std::unique_ptr<MetalBuffer> buffer;
    Status status = MetalBuffer::Create(
        buffer, 1024, MetalMemoryType::METAL_UNIFIED);
    ASSERT_TRUE(status.IsOk());
    
    std::vector<uint8_t> data(512);
    status = buffer->CopyFromHost(data.data(), 512, 768);
    EXPECT_FALSE(status.IsOk());
  }
}

TEST_F(MetalMemoryTest, TritonIntegration)
{
  if (!MetalMemoryManager::IsAvailable()) {
    GTEST_SKIP() << "Metal is not available on this system";
  }
  
  // Initialize Metal support
  Status status = metal::Initialize();
  ASSERT_TRUE(status.IsOk());
  EXPECT_TRUE(metal::IsInitialized());
  
  // Test memory allocation through Triton interface
  const size_t size = 1024;
  void* ptr = nullptr;
  
  status = metal::Allocate(
      &ptr, size, TRITONSERVER_MEMORY_METAL_UNIFIED, 0);
  ASSERT_TRUE(status.IsOk());
  ASSERT_NE(ptr, nullptr);
  
  // Test memory operations
  std::memset(ptr, 0xAB, size);
  uint8_t* bytes = static_cast<uint8_t*>(ptr);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(bytes[i], 0xAB);
  }
  
  // Free memory
  status = metal::Free(ptr, TRITONSERVER_MEMORY_METAL_UNIFIED, 0);
  ASSERT_TRUE(status.IsOk());
  
  // Test preferred memory type
  TRITONSERVER_MemoryType mem_type;
  int64_t mem_type_id;
  status = metal::GetPreferredMemoryType(0, &mem_type, &mem_type_id);
  ASSERT_TRUE(status.IsOk());
  EXPECT_EQ(mem_type, TRITONSERVER_MEMORY_METAL_UNIFIED);
  EXPECT_EQ(mem_type_id, 0);
  
  // Shutdown
  metal::Shutdown();
  EXPECT_FALSE(metal::IsInitialized());
}

TEST_F(MetalMemoryTest, ResponseAllocator)
{
  if (!MetalMemoryManager::IsAvailable()) {
    GTEST_SKIP() << "Metal is not available on this system";
  }
  
  // Initialize Metal support
  Status status = metal::Initialize();
  ASSERT_TRUE(status.IsOk());
  
  // Create Metal response allocator
  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  status = CreateMetalResponseAllocator(&allocator, true);
  ASSERT_TRUE(status.IsOk());
  ASSERT_NE(allocator, nullptr);
  
  // Test allocation
  void* buffer = nullptr;
  void* buffer_userp = nullptr;
  TRITONSERVER_MemoryType actual_memory_type;
  int64_t actual_memory_type_id;
  
  TRITONSERVER_Error* err = MetalResponseAllocator::AllocFn(
      allocator,
      "test_tensor",
      1024,
      TRITONSERVER_MEMORY_METAL_UNIFIED,
      0,
      nullptr,
      &buffer,
      &buffer_userp,
      &actual_memory_type,
      &actual_memory_type_id);
  
  EXPECT_EQ(err, nullptr);
  EXPECT_NE(buffer, nullptr);
  EXPECT_EQ(actual_memory_type, TRITONSERVER_MEMORY_METAL_UNIFIED);
  
  // Test release
  err = MetalResponseAllocator::ReleaseFn(
      allocator,
      buffer,
      buffer_userp,
      1024,
      actual_memory_type,
      actual_memory_type_id);
  
  EXPECT_EQ(err, nullptr);
  
  // Clean up
  TRITONSERVER_ResponseAllocatorDelete(allocator);
  metal::Shutdown();
}

}  // namespace core
}  // namespace triton

#endif  // TRITON_ENABLE_METAL