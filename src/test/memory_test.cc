// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "gtest/gtest.h"

#include <cuda_runtime_api.h>
#include "src/core/cuda_memory_manager.h"
#include "src/core/cuda_utils.h"
#include "src/core/memory.h"
#include "src/core/pinned_memory_manager.h"

namespace ni = nvidia::inferenceserver;

namespace {

#define CHECK_POINTER_ATTRIBUTES(ptr__, type__, device__)                   \
  do {                                                                      \
    cudaPointerAttributes attr;                                             \
    auto cuerr = cudaPointerGetAttributes(&attr, ptr__);                    \
    ASSERT_TRUE(cuerr == cudaSuccess)                                       \
        << "Failed to get CUDA pointer attributes: "                        \
        << cudaGetErrorString(cuerr);                                       \
    EXPECT_TRUE(attr.type == type__)                                        \
        << "Expect pointer with type " << type__ << ", got: " << attr.type; \
    if (attr.type == cudaMemoryTypeDevice) {                                \
      EXPECT_TRUE(attr.device == device__)                                  \
          << "Expect allocation on CUDA device " << device__                \
          << ", got: " << attr.device;                                      \
    }                                                                       \
  } while (false)

// Wrapper of CudaMemoryManager class to expose Reset() for unit testing
class TestingCudaMemoryManager : public ni::CudaMemoryManager {
 public:
  static void Reset() { CudaMemoryManager::Reset(); }
};

class CudaMemoryManagerTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Default memory manager options
    options_.min_supported_compute_capability_ = 6.0;
    options_.memory_pool_byte_size_ = {{0, 1 << 10}};
  }

  void TearDown() override { TestingCudaMemoryManager::Reset(); }

  ni::CudaMemoryManager::Options options_;
};

TEST_F(CudaMemoryManagerTest, Init)
{
  // Set to reserve too much memory
  {
    double cc = 6.0;
    std::map<int, uint64_t> s{{0, uint64_t(1) << 40 /* 1024 GB */}};
    const ni::CudaMemoryManager::Options options{cc, s};
    auto status = ni::CudaMemoryManager::Create(options);
    EXPECT_FALSE(status.IsOk()) << "Expect creation error";
  }

  {
    double cc = 6.0;
    std::map<int, uint64_t> s{{0, 1 << 10 /* 1024 bytes */}};
    const ni::CudaMemoryManager::Options options{cc, s};
    auto status = ni::CudaMemoryManager::Create(options);
    EXPECT_TRUE(status.IsOk()) << status.Message();
  }
}

TEST_F(CudaMemoryManagerTest, AllocSuccess)
{
  auto status = ni::CudaMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  void* ptr = nullptr;
  status = ni::CudaMemoryManager::Alloc(&ptr, 1024, 0);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(ptr) << "Expect pointer to allocated buffer";
  // check if returned pointer is CUDA pointer
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeDevice, 0);
}

TEST_F(CudaMemoryManagerTest, AllocFail)
{
  auto status = ni::CudaMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  void* ptr = nullptr;
  status = ni::CudaMemoryManager::Alloc(&ptr, 2048, 0);
  ASSERT_FALSE(status.IsOk()) << "Unexpected successful allocation";
}

TEST_F(CudaMemoryManagerTest, MultipleAlloc)
{
  auto status = ni::CudaMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  void* first_ptr = nullptr;
  status = ni::CudaMemoryManager::Alloc(&first_ptr, 600, 0);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(first_ptr) << "Expect pointer to allocated buffer";
  // check if returned pointer is CUDA pointer
  CHECK_POINTER_ATTRIBUTES(first_ptr, cudaMemoryTypeDevice, 0);

  // 512 + 600 > 1024
  void* second_ptr = nullptr;
  status = ni::CudaMemoryManager::Alloc(&second_ptr, 512, 0);
  ASSERT_FALSE(status.IsOk()) << "Unexpected successful allocation";

  // Free the first pointer and retry the second one
  status = ni::CudaMemoryManager::Free(first_ptr, 0);
  EXPECT_TRUE(status.IsOk()) << status.Message();
  status = ni::CudaMemoryManager::Alloc(&second_ptr, 512, 0);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(second_ptr) << "Expect pointer to allocated buffer";
  // check if returned pointer is CUDA pointer
  CHECK_POINTER_ATTRIBUTES(second_ptr, cudaMemoryTypeDevice, 0);
}

TEST_F(CudaMemoryManagerTest, MultipleDevice)
{
  std::set<int> supported_gpus;
  auto status = ni::GetSupportedGPUs(
      &supported_gpus, options_.min_supported_compute_capability_);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_GE(supported_gpus.size(), size_t(2))
      << "Test requires at least two supported CUDA devices";

  {
    double cc = 6.0;
    std::map<int, uint64_t> s;
    // Only enough memory is only reserved in one of the devices
    s[*supported_gpus.begin()] = 32;
    s[*(++supported_gpus.begin())] = 1024;
    const ni::CudaMemoryManager::Options options{cc, s};
    status = ni::CudaMemoryManager::Create(options);
    ASSERT_TRUE(status.IsOk()) << status.Message();
  }

  void* ptr = nullptr;
  // Allocation on small device
  int small_device = *supported_gpus.begin();
  status = ni::CudaMemoryManager::Alloc(&ptr, 1024, small_device);
  ASSERT_FALSE(status.IsOk()) << "Unexpected successful allocation";

  // Allocation on large device
  int large_device = *(++supported_gpus.begin());
  status = ni::CudaMemoryManager::Alloc(&ptr, 1024, large_device);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(ptr) << "Expect pointer to allocated buffer";
  // check if returned pointer is CUDA pointer
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeDevice, large_device);

  // Free allocation ...
  status = ni::CudaMemoryManager::Free(ptr, small_device);
  EXPECT_FALSE(status.IsOk()) << "Unexpected deallocation on wrong device";
  status = ni::CudaMemoryManager::Free(ptr, large_device);
  EXPECT_TRUE(status.IsOk()) << status.Message();
}

class AllocatedMemoryTest : public ::testing::Test {
 protected:
  // Per-test-suite set-up.
  static void SetUpTestSuite()
  {
    // Pinned memory manager
    {
      ni::PinnedMemoryManager::Options options{1024};
      auto status = ni::PinnedMemoryManager::Create(options);
      ASSERT_TRUE(status.IsOk()) << status.Message();
    }
  }

  // Set up CUDA memory manager per test for special fallback case
  void SetUp() override
  {
    ni::CudaMemoryManager::Options options{6.0, {{0, 1 << 10}}};
    auto status = ni::CudaMemoryManager::Create(options);
    ASSERT_TRUE(status.IsOk()) << status.Message();
  }

  void TearDown() override { TestingCudaMemoryManager::Reset(); }
};

TEST_F(AllocatedMemoryTest, AllocGPU)
{
  size_t expect_size = 512, actual_size;
  TRITONSERVER_MemoryType expect_type = TRITONSERVER_MEMORY_GPU, actual_type;
  int64_t expect_id = 0, actual_id;
  ni::AllocatedMemory memory(expect_size, expect_type, expect_id);

  auto ptr = memory.BufferAt(0, &actual_size, &actual_type, &actual_id);
  EXPECT_EQ(expect_size, actual_size)
      << "Expect size: " << expect_size << ", got: " << actual_size;
  EXPECT_EQ(expect_type, actual_type)
      << "Expect type: " << expect_type << ", got: " << actual_type;
  EXPECT_EQ(expect_id, actual_id)
      << "Expect id: " << expect_id << ", got: " << actual_id;

  // Sanity check on the pointer property
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeDevice, expect_id);
}

TEST_F(AllocatedMemoryTest, AllocPinned)
{
  size_t expect_size = 512, actual_size;
  TRITONSERVER_MemoryType expect_type = TRITONSERVER_MEMORY_CPU_PINNED,
                          actual_type;
  int64_t expect_id = 0, actual_id;
  ni::AllocatedMemory memory(expect_size, expect_type, expect_id);

  auto ptr = memory.BufferAt(0, &actual_size, &actual_type, &actual_id);
  EXPECT_EQ(expect_size, actual_size)
      << "Expect size: " << expect_size << ", got: " << actual_size;
  EXPECT_EQ(expect_type, actual_type)
      << "Expect type: " << expect_type << ", got: " << actual_type;
  EXPECT_EQ(expect_id, actual_id)
      << "Expect id: " << expect_id << ", got: " << actual_id;

  // Sanity check on the pointer property
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeHost, expect_id);
}

TEST_F(AllocatedMemoryTest, AllocFallback)
{
  // Each allocation uses half of the target reserved memory
  size_t expect_size = 600, actual_size;
  TRITONSERVER_MemoryType expect_type = TRITONSERVER_MEMORY_GPU, actual_type;
  int64_t expect_id = 0, actual_id;

  // First allocation
  ni::AllocatedMemory cuda_memory(expect_size, expect_type, expect_id);

  auto ptr = cuda_memory.BufferAt(0, &actual_size, &actual_type, &actual_id);
  EXPECT_EQ(expect_size, actual_size)
      << "Expect size: " << expect_size << ", got: " << actual_size;
  EXPECT_EQ(expect_type, actual_type)
      << "Expect type: " << expect_type << ", got: " << actual_type;
  EXPECT_EQ(expect_id, actual_id)
      << "Expect id: " << expect_id << ", got: " << actual_id;

  // Sanity check on the pointer property
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeDevice, expect_id);

  // Second allocation, should trigger fallback from CUDA -> pinned memory
  ni::AllocatedMemory pinned_memory(expect_size, expect_type, expect_id);

  ptr = pinned_memory.BufferAt(0, &actual_size, &actual_type, &actual_id);
  EXPECT_EQ(expect_size, actual_size)
      << "Expect size: " << expect_size << ", got: " << actual_size;
  EXPECT_EQ(TRITONSERVER_MEMORY_CPU_PINNED, actual_type)
      << "Expect type: " << TRITONSERVER_MEMORY_CPU_PINNED
      << ", got: " << actual_type;

  // Sanity check on the pointer property
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeHost, expect_id);

  // Third allocation, CUDA -> pinned -> non-pinned
  ni::AllocatedMemory system_memory(expect_size, expect_type, expect_id);

  ptr = system_memory.BufferAt(0, &actual_size, &actual_type, &actual_id);
  EXPECT_EQ(expect_size, actual_size)
      << "Expect size: " << expect_size << ", got: " << actual_size;
  EXPECT_EQ(TRITONSERVER_MEMORY_CPU, actual_type)
      << "Expect type: " << TRITONSERVER_MEMORY_CPU_PINNED
      << ", got: " << actual_type;

  // Sanity check on the pointer property
  cudaPointerAttributes attr;
  EXPECT_EQ(cudaPointerGetAttributes(&attr, ptr), cudaErrorInvalidValue)
      << "Expect cudaErrorInvalidValue is returned for non-pinned memory";

  // Note: After CUDA 11.0, we can verify non-pinned memory with the macro,
  // but before that, only check cudaErrorInvalidValue is returned.
  //
  // CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeUnregistered, expect_id);
}

TEST_F(AllocatedMemoryTest, AllocFallbackNoCuda)
{
  // Test fallback in the case where CUDA memory manager is not properly created
  TestingCudaMemoryManager::Reset();

  size_t expect_size = 600, actual_size;
  TRITONSERVER_MemoryType expect_type = TRITONSERVER_MEMORY_GPU, actual_type;
  int64_t expect_id = 0, actual_id;

  // CUDA memory allocation should trigger fallback to allocate pinned memory
  ni::AllocatedMemory pinned_memory(expect_size, expect_type, expect_id);

  auto ptr = pinned_memory.BufferAt(0, &actual_size, &actual_type, &actual_id);
  EXPECT_EQ(expect_size, actual_size)
      << "Expect size: " << expect_size << ", got: " << actual_size;
  EXPECT_EQ(TRITONSERVER_MEMORY_CPU_PINNED, actual_type)
      << "Expect type: " << TRITONSERVER_MEMORY_CPU_PINNED
      << ", got: " << actual_type;

  // Sanity check on the pointer property
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeHost, expect_id);
}

TEST_F(AllocatedMemoryTest, Release)
{
  // Similar to above, but verify that the memory will be released once
  // out of scope
  // Each allocation uses half of the target reserved memory
  size_t expect_size = 600, actual_size;
  TRITONSERVER_MemoryType expect_type = TRITONSERVER_MEMORY_GPU, actual_type;
  int64_t expect_id = 0, actual_id;

  {
    // First allocation
    ni::AllocatedMemory cuda_memory(expect_size, expect_type, expect_id);

    auto ptr = cuda_memory.BufferAt(0, &actual_size, &actual_type, &actual_id);
    EXPECT_EQ(expect_size, actual_size)
        << "Expect size: " << expect_size << ", got: " << actual_size;
    EXPECT_EQ(expect_type, actual_type)
        << "Expect type: " << expect_type << ", got: " << actual_type;
    EXPECT_EQ(expect_id, actual_id)
        << "Expect id: " << expect_id << ", got: " << actual_id;

    // Sanity check on the pointer property
    CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeDevice, expect_id);

    // Second allocation, should trigger fallback from CUDA -> pinned memory
    ni::AllocatedMemory pinned_memory(expect_size, expect_type, expect_id);

    ptr = pinned_memory.BufferAt(0, &actual_size, &actual_type, &actual_id);
    EXPECT_EQ(expect_size, actual_size)
        << "Expect size: " << expect_size << ", got: " << actual_size;
    EXPECT_EQ(TRITONSERVER_MEMORY_CPU_PINNED, actual_type)
        << "Expect type: " << TRITONSERVER_MEMORY_CPU_PINNED
        << ", got: " << actual_type;

    // Sanity check on the pointer property
    CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeHost, expect_id);
  }

  // Third allocation, should not trigger fallback
  ni::AllocatedMemory memory(expect_size, expect_type, expect_id);

  auto ptr = memory.BufferAt(0, &actual_size, &actual_type, &actual_id);
  EXPECT_EQ(expect_size, actual_size)
      << "Expect size: " << expect_size << ", got: " << actual_size;
  EXPECT_EQ(expect_type, actual_type)
      << "Expect type: " << expect_type << ", got: " << actual_type;

  // Sanity check on the pointer property
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeDevice, expect_id);
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
