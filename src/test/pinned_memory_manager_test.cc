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
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include "src/core/pinned_memory_manager.h"
#include "src/core/tritonserver_apis.h"

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

#define STORE_RESULT_AND_RETURN_IF_ERROR(metadata__, idx__, status__) \
  do {                                                                \
    if (!status__.IsOk()) {                                           \
      std::lock_guard<std::mutex> lk(metadata__->mtx_);               \
      metadata__->results_[idx__] = status__.AsString();              \
      return;                                                         \
    }                                                                 \
  } while (false)

struct MemoryWorkMetadata {
  MemoryWorkMetadata(size_t thread_count)
      : thread_count_(thread_count), ready_count_(0), results_(thread_count, "")
  {
  }
  size_t thread_count_;
  size_t ready_count_;
  std::vector<std::string> results_;
  std::mutex mtx_;
  std::condition_variable cv_;
};

void
RunMemoryWork(
    size_t idx, size_t alloc_size, bool allow_nonpinned_fallback,
    MemoryWorkMetadata* metadata)
{
  // Prepare variable to hold input / output
  std::unique_ptr<char> input(new char[alloc_size]);
  std::unique_ptr<char> output(new char[alloc_size]);

  // Wait until all threads are issued
  {
    std::unique_lock<std::mutex> lk(metadata->mtx_);
    metadata->ready_count_++;
    if (metadata->ready_count_ != metadata->thread_count_) {
      while (metadata->ready_count_ != metadata->thread_count_) {
        metadata->cv_.wait(lk);
      }
    }
    metadata->cv_.notify_one();
  }

  // Simulate receive input data -> alloc and write to input buffer
  // -> alloc and write to output buffer -> return output data
  TRITONSERVER_MemoryType allocated_type = TRITONSERVER_MEMORY_GPU;
  void* input_buffer = nullptr;
  STORE_RESULT_AND_RETURN_IF_ERROR(
      metadata, idx,
      ni::PinnedMemoryManager::Alloc(
          &input_buffer, alloc_size, &allocated_type,
          allow_nonpinned_fallback));
  if ((!allow_nonpinned_fallback) &&
      (allocated_type != TRITONSERVER_MEMORY_CPU_PINNED)) {
    ni::Status status(
        ni::Status::Code::INVALID_ARG, "returned memory buffer is not pinned");
    STORE_RESULT_AND_RETURN_IF_ERROR(metadata, idx, status);
  }
  memcpy(input_buffer, input.get(), alloc_size);
  void* output_buffer = nullptr;
  STORE_RESULT_AND_RETURN_IF_ERROR(
      metadata, idx,
      ni::PinnedMemoryManager::Alloc(
          &output_buffer, alloc_size, &allocated_type,
          allow_nonpinned_fallback));
  if ((!allow_nonpinned_fallback) &&
      (allocated_type != TRITONSERVER_MEMORY_CPU_PINNED)) {
    ni::Status status(
        ni::Status::Code::INVALID_ARG, "returned memory buffer is not pinned");
    STORE_RESULT_AND_RETURN_IF_ERROR(metadata, idx, status);
  }
  memcpy(output_buffer, input_buffer, alloc_size);
  memcpy(output.get(), output_buffer, alloc_size);
  for (size_t offset = 0; offset < alloc_size; offset++) {
    if (input.get()[offset] != output.get()[offset]) {
      std::lock_guard<std::mutex> lk(metadata->mtx_);
      metadata->results_[idx] =
          std::string("mismatch between input and output for work idx ") +
          std::to_string(idx);
      return;
    }
  }
}

// Wrapper of PinnedMemoryManager class to expose Reset() for unit testing
class TestingPinnedMemoryManager : public ni::PinnedMemoryManager {
 public:
  static void Reset() { PinnedMemoryManager::Reset(); }
};

class PinnedMemoryManagerTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Default memory manager options
    options_.pinned_memory_pool_byte_size_ = 1 << 10;
  }

  void TearDown() override { TestingPinnedMemoryManager::Reset(); }

  ni::PinnedMemoryManager::Options options_;
};

TEST_F(PinnedMemoryManagerTest, InitOOM)
{
  // Set to reserve too much memory
  options_.pinned_memory_pool_byte_size_ = uint64_t(1) << 40 /* 1024 GB */;
  auto status = ni::PinnedMemoryManager::Create(options_);
  // For pinned memory manager, it will still be created for "CPU fallback"
  // allocation even if it fails to create pinned memory pool
  EXPECT_TRUE(status.IsOk()) << status.Message();
}

TEST_F(PinnedMemoryManagerTest, InitSuccess)
{
  auto status = ni::PinnedMemoryManager::Create(options_);
  EXPECT_TRUE(status.IsOk()) << status.Message();
}

TEST_F(PinnedMemoryManagerTest, InitZeroByte)
{
  options_.pinned_memory_pool_byte_size_ = 0;
  auto status = ni::PinnedMemoryManager::Create(options_);
  EXPECT_TRUE(status.IsOk()) << status.Message();

  void* ptr = nullptr;
  TRITONSERVER_MemoryType allocated_type = TRITONSERVER_MEMORY_GPU;
  status = ni::PinnedMemoryManager::Alloc(
      &ptr, 1, &allocated_type, false /* allow_nonpinned_fallback */);
  ASSERT_FALSE(status.IsOk()) << "Unexpected successful allocation";
}

TEST_F(PinnedMemoryManagerTest, AllocSuccess)
{
  auto status = ni::PinnedMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  void* ptr = nullptr;
  TRITONSERVER_MemoryType allocated_type = TRITONSERVER_MEMORY_GPU;
  status = ni::PinnedMemoryManager::Alloc(
      &ptr, 512, &allocated_type, false /* allow_nonpinned_fallback */);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(ptr) << "Expect pointer to allocated buffer";
  ASSERT_TRUE(allocated_type == TRITONSERVER_MEMORY_CPU_PINNED)
      << "Expect pointer to pinned memory";
  // check if returned pointer is pinned memory pointer
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeHost, 0);
}

TEST_F(PinnedMemoryManagerTest, AllocFallbackSuccess)
{
  auto status = ni::PinnedMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  void* ptr = nullptr;
  TRITONSERVER_MemoryType allocated_type = TRITONSERVER_MEMORY_GPU;
  status = ni::PinnedMemoryManager::Alloc(
      &ptr, 2048, &allocated_type, true /* allow_nonpinned_fallback */);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(ptr) << "Expect pointer to allocated buffer";
  ASSERT_TRUE(allocated_type == TRITONSERVER_MEMORY_CPU)
      << "Expect pointer to non-pinned memory";
  // check if returned pointer is non-pinned memory pointer
  CHECK_POINTER_ATTRIBUTES(ptr, cudaMemoryTypeUnregistered, 0);
}

TEST_F(PinnedMemoryManagerTest, AllocFail)
{
  auto status = ni::PinnedMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  void* ptr = nullptr;
  TRITONSERVER_MemoryType allocated_type = TRITONSERVER_MEMORY_GPU;
  status = ni::PinnedMemoryManager::Alloc(
      &ptr, 2048, &allocated_type, false /* allow_nonpinned_fallback */);
  ASSERT_FALSE(status.IsOk()) << "Unexpected successful allocation";
}

TEST_F(PinnedMemoryManagerTest, MultipleAlloc)
{
  auto status = ni::PinnedMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  void* first_ptr = nullptr;
  TRITONSERVER_MemoryType allocated_type = TRITONSERVER_MEMORY_GPU;
  status = ni::PinnedMemoryManager::Alloc(
      &first_ptr, 600, &allocated_type, false /* allow_nonpinned_fallback */);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(first_ptr) << "Expect pointer to allocated buffer";
  ASSERT_TRUE(allocated_type == TRITONSERVER_MEMORY_CPU_PINNED)
      << "Expect pointer to pinned memory";
  // check if returned pointer is pinned memory pointer
  CHECK_POINTER_ATTRIBUTES(first_ptr, cudaMemoryTypeHost, 0);

  // 512 + 600 > 1024
  void* second_ptr = nullptr;
  status = ni::PinnedMemoryManager::Alloc(
      &second_ptr, 512, &allocated_type, false /* allow_nonpinned_fallback */);
  ASSERT_FALSE(status.IsOk()) << "Unexpected successful allocation";

  // Free the first pointer and retry the second one
  status = ni::PinnedMemoryManager::Free(first_ptr);
  EXPECT_TRUE(status.IsOk()) << status.Message();
  status = ni::PinnedMemoryManager::Alloc(
      &second_ptr, 512, &allocated_type, false /* allow_nonpinned_fallback */);
  ASSERT_TRUE(status.IsOk()) << status.Message();
  ASSERT_TRUE(second_ptr) << "Expect pointer to allocated buffer";
  ASSERT_TRUE(allocated_type == TRITONSERVER_MEMORY_CPU_PINNED)
      << "Expect pointer to pinned memory";
  // check if returned pointer is pinned memory pointer
  CHECK_POINTER_ATTRIBUTES(second_ptr, cudaMemoryTypeHost, 0);
}

TEST_F(PinnedMemoryManagerTest, ParallelAlloc)
{
  options_.pinned_memory_pool_byte_size_ = uint64_t(1) << 28 /* 256 MB */;
  auto status = ni::PinnedMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  // Create threads to perform operations on allocated memory in parallel
  // Seems like for 1 MB alloc size (2 MB for both input and output),
  // 100 threads is a good amount for pool manager not to use CPU fallback.
  size_t thread_count = 100;
  size_t allocated_size = 1 << 20 /* 1 MB */;
  MemoryWorkMetadata metadata(thread_count);
  std::vector<std::thread> threads;
  for (size_t idx = 0; idx < thread_count; idx++) {
    threads.emplace_back(
        std::thread(RunMemoryWork, idx, allocated_size, false, &metadata));
  }
  for (size_t idx = 0; idx < thread_count; idx++) {
    threads[idx].join();
    EXPECT_TRUE(metadata.results_[idx].empty()) << metadata.results_[idx];
  }
}


TEST_F(PinnedMemoryManagerTest, ParallelAllocFallback)
{
  options_.pinned_memory_pool_byte_size_ = uint64_t(1) << 28 /* 256 MB */;
  auto status = ni::PinnedMemoryManager::Create(options_);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  // Create threads to perform operations on allocated memory in parallel
  size_t thread_count = 128;
  size_t allocated_size = 1 << 24 /* 4 MB */;
  MemoryWorkMetadata metadata(thread_count);
  std::vector<std::thread> threads;
  for (size_t idx = 0; idx < thread_count; idx++) {
    threads.emplace_back(
        std::thread(RunMemoryWork, idx, allocated_size, true, &metadata));
  }
  for (size_t idx = 0; idx < thread_count; idx++) {
    threads[idx].join();
    EXPECT_TRUE(metadata.results_[idx].empty()) << metadata.results_[idx];
  }
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
