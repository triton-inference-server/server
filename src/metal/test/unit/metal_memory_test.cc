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

#include "src/metal/test/utils/metal_test_fixtures.h"
#include "src/metal/metal_memory.h"
#include "src/metal/metal_memory_manager.h"

#include <random>

namespace triton { namespace server { namespace test {

class MetalMemoryUnitTest : public MetalMemoryTest {
};

TEST_F(MetalMemoryUnitTest, BasicAllocation)
{
  size_t size = 1024 * 1024;  // 1MB
  
  auto buffer = memory_manager_->AllocateBuffer(size);
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->GetSize(), size);
  EXPECT_NE(buffer->GetData(), nullptr);
  
  // Should be able to write to the buffer
  float* data = static_cast<float*>(buffer->GetData());
  for (size_t i = 0; i < size / sizeof(float); ++i) {
    data[i] = static_cast<float>(i);
  }
  
  // Verify data
  for (size_t i = 0; i < size / sizeof(float); ++i) {
    EXPECT_EQ(data[i], static_cast<float>(i));
  }
}

TEST_F(MetalMemoryUnitTest, ZeroSizeAllocation)
{
  auto buffer = memory_manager_->AllocateBuffer(0);
  EXPECT_EQ(buffer, nullptr);
}

TEST_F(MetalMemoryUnitTest, LargeAllocation)
{
  size_t size = 256 * 1024 * 1024;  // 256MB
  
  auto buffer = memory_manager_->AllocateBuffer(size);
  if (buffer) {
    EXPECT_EQ(buffer->GetSize(), size);
    EXPECT_NE(buffer->GetData(), nullptr);
  } else {
    // May fail on systems with limited memory
    std::cout << "Large allocation failed (expected on low-memory systems)\n";
  }
}

TEST_F(MetalMemoryUnitTest, MultipleAllocations)
{
  const int num_buffers = 100;
  const size_t buffer_size = 1024 * 1024;  // 1MB each
  
  std::vector<std::shared_ptr<MetalBuffer>> buffers;
  
  for (int i = 0; i < num_buffers; ++i) {
    auto buffer = memory_manager_->AllocateBuffer(buffer_size);
    ASSERT_NE(buffer, nullptr);
    buffers.push_back(buffer);
  }
  
  // Verify all buffers are unique
  for (int i = 0; i < num_buffers; ++i) {
    for (int j = i + 1; j < num_buffers; ++j) {
      EXPECT_NE(buffers[i]->GetData(), buffers[j]->GetData());
    }
  }
}

TEST_F(MetalMemoryUnitTest, BufferModes)
{
  size_t size = 1024 * 1024;
  
  // Test different buffer modes
  std::vector<MetalBufferMode> modes = {
    MetalBufferMode::Shared,
    MetalBufferMode::Private,
    MetalBufferMode::Managed
  };
  
  for (auto mode : modes) {
    auto buffer = memory_manager_->AllocateBuffer(size, mode);
    if (buffer) {
      EXPECT_EQ(buffer->GetMode(), mode);
      EXPECT_EQ(buffer->GetSize(), size);
      
      if (GetConfig().verbose) {
        std::cout << "Allocated buffer with mode: ";
        switch (mode) {
          case MetalBufferMode::Shared:
            std::cout << "Shared\n";
            break;
          case MetalBufferMode::Private:
            std::cout << "Private\n";
            break;
          case MetalBufferMode::Managed:
            std::cout << "Managed\n";
            break;
        }
      }
    }
  }
}

TEST_F(MetalMemoryUnitTest, BufferCopy)
{
  size_t size = 1024 * 1024;
  TestDataGenerator gen;
  auto test_data = gen.GenerateFloatData(size / sizeof(float));
  
  // Create source buffer
  auto src_buffer = memory_manager_->AllocateBuffer(size);
  ASSERT_NE(src_buffer, nullptr);
  
  // Fill with test data
  memcpy(src_buffer->GetData(), test_data.data(), size);
  
  // Create destination buffer
  auto dst_buffer = memory_manager_->AllocateBuffer(size);
  ASSERT_NE(dst_buffer, nullptr);
  
  // Copy buffer
  auto err = memory_manager_->CopyBuffer(src_buffer.get(), dst_buffer.get());
  ASSERT_TRITON_OK(err);
  
  // Verify copy
  EXPECT_TRUE(ValidateBuffer(
      dst_buffer->GetMetalBuffer(), test_data.data(), size));
}

TEST_F(MetalMemoryUnitTest, BufferSubregionCopy)
{
  size_t size = 1024 * 1024;
  size_t offset = 256 * 1024;
  size_t copy_size = 512 * 1024;
  
  TestDataGenerator gen;
  auto test_data = gen.GenerateFloatData(size / sizeof(float));
  
  // Create buffers
  auto src_buffer = memory_manager_->AllocateBuffer(size);
  auto dst_buffer = memory_manager_->AllocateBuffer(size);
  ASSERT_NE(src_buffer, nullptr);
  ASSERT_NE(dst_buffer, nullptr);
  
  // Fill source with test data
  memcpy(src_buffer->GetData(), test_data.data(), size);
  
  // Copy subregion
  auto err = memory_manager_->CopyBufferRegion(
      src_buffer.get(), offset, dst_buffer.get(), offset, copy_size);
  ASSERT_TRITON_OK(err);
  
  // Verify copy
  float* dst_data = static_cast<float*>(dst_buffer->GetData());
  float* expected_data = test_data.data() + offset / sizeof(float);
  
  for (size_t i = 0; i < copy_size / sizeof(float); ++i) {
    EXPECT_EQ(dst_data[offset / sizeof(float) + i], expected_data[i]);
  }
}

TEST_F(MetalMemoryUnitTest, MemorySync)
{
  size_t size = 1024 * 1024;
  
  // Create managed buffer
  auto buffer = memory_manager_->AllocateBuffer(
      size, MetalBufferMode::Managed);
  
  if (buffer) {
    // Write to CPU side
    float* data = static_cast<float*>(buffer->GetData());
    for (size_t i = 0; i < size / sizeof(float); ++i) {
      data[i] = static_cast<float>(i);
    }
    
    // Sync to GPU
    auto err = buffer->SyncToGPU();
    ASSERT_TRITON_OK(err);
    
    // Modify and sync back
    data[0] = 999.0f;
    err = buffer->SyncToCPU();
    ASSERT_TRITON_OK(err);
  }
}

TEST_F(MetalMemoryUnitTest, MemoryUsageTracking)
{
  auto initial_stats = memory_manager_->GetStats();
  
  size_t alloc_size = 10 * 1024 * 1024;  // 10MB
  std::vector<std::shared_ptr<MetalBuffer>> buffers;
  
  // Allocate buffers
  for (int i = 0; i < 5; ++i) {
    auto buffer = memory_manager_->AllocateBuffer(alloc_size);
    ASSERT_NE(buffer, nullptr);
    buffers.push_back(buffer);
  }
  
  auto stats = memory_manager_->GetStats();
  EXPECT_GE(stats.total_allocated, 50 * 1024 * 1024);
  EXPECT_GE(stats.current_usage, 50 * 1024 * 1024);
  EXPECT_EQ(stats.allocation_count, 5);
  
  // Free some buffers
  buffers.resize(2);
  
  stats = memory_manager_->GetStats();
  EXPECT_LE(stats.current_usage, 20 * 1024 * 1024);
  
  if (GetConfig().verbose) {
    std::cout << "Memory stats:\n";
    std::cout << "  Total allocated: " << stats.total_allocated / (1024*1024) << " MB\n";
    std::cout << "  Current usage: " << stats.current_usage / (1024*1024) << " MB\n";
    std::cout << "  Peak usage: " << stats.peak_usage / (1024*1024) << " MB\n";
    std::cout << "  Allocations: " << stats.allocation_count << "\n";
    std::cout << "  Deallocations: " << stats.deallocation_count << "\n";
  }
}

TEST_F(MetalMemoryUnitTest, MemoryAlignment)
{
  std::vector<size_t> alignments = {16, 256, 4096, 16384};
  
  for (size_t alignment : alignments) {
    auto buffer = memory_manager_->AllocateAlignedBuffer(1024, alignment);
    ASSERT_NE(buffer, nullptr);
    
    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer->GetData());
    EXPECT_EQ(addr % alignment, 0) 
        << "Buffer not aligned to " << alignment << " bytes";
  }
}

TEST_F(MetalMemoryUnitTest, MemoryPooling)
{
  // Configure memory pool
  MetalMemoryPoolConfig pool_config;
  pool_config.enable_pooling = true;
  pool_config.pool_sizes = {1024, 4096, 16384, 65536};
  pool_config.max_pool_size = 10 * 1024 * 1024;
  
  memory_manager_->ConfigurePool(pool_config);
  
  // Test pool hits
  auto stats_before = memory_manager_->GetStats();
  
  // Allocate and free same size multiple times
  for (int i = 0; i < 10; ++i) {
    auto buffer = memory_manager_->AllocateBuffer(4096);
    ASSERT_NE(buffer, nullptr);
    // Buffer goes out of scope and returns to pool
  }
  
  auto stats_after = memory_manager_->GetStats();
  
  // Should have pool hits after first allocation
  EXPECT_GT(stats_after.pool_hits, 0);
  
  if (GetConfig().verbose) {
    std::cout << "Pool statistics:\n";
    std::cout << "  Pool hits: " << stats_after.pool_hits << "\n";
    std::cout << "  Pool misses: " << stats_after.pool_misses << "\n";
  }
}

TEST_F(MetalMemoryUnitTest, MemoryDefragmentation)
{
  // Allocate many small buffers
  std::vector<std::shared_ptr<MetalBuffer>> buffers;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(1024, 10240);
  
  for (int i = 0; i < 100; ++i) {
    size_t size = size_dist(gen);
    auto buffer = memory_manager_->AllocateBuffer(size);
    if (buffer) {
      buffers.push_back(buffer);
    }
  }
  
  auto stats_before = memory_manager_->GetStats();
  
  // Free every other buffer (create fragmentation)
  for (size_t i = 0; i < buffers.size(); i += 2) {
    buffers[i].reset();
  }
  
  // Defragment
  auto err = memory_manager_->Defragment();
  ASSERT_TRITON_OK(err);
  
  auto stats_after = memory_manager_->GetStats();
  
  if (GetConfig().verbose) {
    std::cout << "Defragmentation results:\n";
    std::cout << "  Fragmentation before: " << stats_before.fragmentation << "%\n";
    std::cout << "  Fragmentation after: " << stats_after.fragmentation << "%\n";
  }
}

TEST_F(MetalMemoryUnitTest, MemoryMigration)
{
  size_t size = 1024 * 1024;
  TestDataGenerator gen;
  auto test_data = gen.GenerateFloatData(size / sizeof(float));
  
  // Create shared buffer
  auto shared_buffer = memory_manager_->AllocateBuffer(
      size, MetalBufferMode::Shared);
  ASSERT_NE(shared_buffer, nullptr);
  
  // Fill with data
  memcpy(shared_buffer->GetData(), test_data.data(), size);
  
  // Migrate to private buffer
  auto private_buffer = memory_manager_->MigrateBuffer(
      shared_buffer.get(), MetalBufferMode::Private);
  ASSERT_NE(private_buffer, nullptr);
  
  EXPECT_EQ(private_buffer->GetMode(), MetalBufferMode::Private);
  EXPECT_EQ(private_buffer->GetSize(), size);
}

TEST_F(MetalMemoryUnitTest, ConcurrentMemoryOperations)
{
  const int num_threads = 8;
  const int ops_per_thread = 100;
  
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};
  std::atomic<int> error_count{0};
  
  auto worker = [&](int thread_id) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(1024, 102400);
    std::uniform_int_distribution<> action_dist(0, 2);
    
    std::vector<std::shared_ptr<MetalBuffer>> local_buffers;
    
    for (int i = 0; i < ops_per_thread; ++i) {
      int action = action_dist(gen);
      
      if (action < 2 && local_buffers.size() < 10) {
        // Allocate
        size_t size = size_dist(gen);
        auto buffer = memory_manager_->AllocateBuffer(size);
        if (buffer) {
          local_buffers.push_back(buffer);
          success_count.fetch_add(1);
        } else {
          error_count.fetch_add(1);
        }
      } else if (!local_buffers.empty()) {
        // Free
        local_buffers.pop_back();
        success_count.fetch_add(1);
      }
    }
  };
  
  // Start threads
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker, i);
  }
  
  // Wait for completion
  for (auto& t : threads) {
    t.join();
  }
  
  EXPECT_GT(success_count.load(), 0);
  
  if (GetConfig().verbose) {
    std::cout << "Concurrent operations:\n";
    std::cout << "  Successful: " << success_count.load() << "\n";
    std::cout << "  Errors: " << error_count.load() << "\n";
  }
}

TEST_F(MetalMemoryUnitTest, MemoryPressureHandling)
{
  // Register pressure handler
  bool pressure_received = false;
  memory_manager_->RegisterPressureHandler([&pressure_received]() {
    pressure_received = true;
    return true;  // Successfully handled
  });
  
  // Try to allocate a lot of memory
  std::vector<std::shared_ptr<MetalBuffer>> buffers;
  const size_t chunk_size = 100 * 1024 * 1024;  // 100MB chunks
  
  while (!pressure_received && buffers.size() < 100) {
    auto buffer = memory_manager_->AllocateBuffer(chunk_size);
    if (!buffer) {
      break;
    }
    buffers.push_back(buffer);
  }
  
  if (GetConfig().verbose) {
    std::cout << "Allocated " << buffers.size() * 100 << " MB before pressure\n";
    std::cout << "Pressure received: " << (pressure_received ? "Yes" : "No") << "\n";
  }
}

}}}  // namespace triton::server::test