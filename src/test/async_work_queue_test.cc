// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <future>
#include <random>
#include <limits>
#include <chrono>
#include "src/core/async_work_queue.h"

namespace ni = nvidia::inferenceserver;

namespace {

// Wrapper of AsyncWorkQueue class to expose Reset() for unit testing
class TestingAsyncWorkQueue : public ni::AsyncWorkQueue {
 public:
  static void Reset() { AsyncWorkQueue::Reset(); }
};

class AsyncWorkQueueTest : public ::testing::Test {
 protected:
  void TearDown() override { TestingAsyncWorkQueue::Reset(); }
};

TEST_F(AsyncWorkQueueTest, InitZeroWorker)
{
  auto status = ni::AsyncWorkQueue::Initialize(0);
  EXPECT_FALSE(status.IsOk()) << "Expect error when initialized with 0 worker";
}

TEST_F(AsyncWorkQueueTest, InitOneWorker)
{
  auto status = ni::AsyncWorkQueue::Initialize(1);
  EXPECT_TRUE(status.IsOk()) << status.Message();
}

TEST_F(AsyncWorkQueueTest, InitFourWorker)
{
  auto status = ni::AsyncWorkQueue::Initialize(1);
  EXPECT_TRUE(status.IsOk()) << status.Message();
}

TEST_F(AsyncWorkQueueTest, InitTwice)
{
  auto status = ni::AsyncWorkQueue::Initialize(4);
  EXPECT_TRUE(status.IsOk()) << status.Message();
  auto status = ni::AsyncWorkQueue::Initialize(2);
  EXPECT_FALSE(status.IsOk()) << "Expect error from initializing twice";
}

TEST_F(AsyncWorkQueueTest, WorkerCountUninitialized)
{
  EXPECT_EQ(ni::AsyncWorkQueue::WorkerCount(), 0) << "Expect 0 worker count for uninitialized queue";
}

TEST_F(AsyncWorkQueueTest, WorkerCountInitialized)
{
  auto status = ni::AsyncWorkQueue::Initialize(4);
  EXPECT_TRUE(status.IsOk()) << status.Message();
  EXPECT_EQ(ni::AsyncWorkQueue::WorkerCount(), 4) << "Expect 4 worker count for initialized queue";
}


TEST_F(AsyncWorkQueueTest, RunTasksInParallel)
{
  auto AddTwoFn = [](const std::vector<int>& lhs, const std::vector<int>& rhs,
    std::promise<std::vector<int>>* res)
  {
    std::vector<int> lres;
    lres.reserve(lhs.size());
    for (size_t idx = 0; idx < lhs.size(); idx++) {
      lres.push_back(lhs[idx] + rhs[idx]);
    }
    res->set_value(lres);
  };

  size_t task_count = 8;
  std::vector<std::vector<int>> operands;
  std::vector<std::vector<int>> expected_results;
  {
    size_t element_count = 1 << 20;
    auto RandHalfIntFn = std::bind(std::uniform_int_distribution<>{std::numeric_limits<int>::min() / 2, std::numeric_limits<int>::max() / 2}, std::default_random_engine{});
    for (size_t tc = 0; tc < task_count + 1; tc++) {
      results.push_back(std::vector<int>());
      operands.push_back(std::vector<int>());
      operands.back().reserve(element_count);
      for (size_t ec = 0; ec < element_count; ec++) {
        operands.back().push_back(RandHalfIntFn());
      }
    }
  }

  // Get serialized time as baseline and store expected results
  uint64_t serialized_duration = 0;
  {
    std::vector<std::promise<std::vector<int>>> res(task_count);

    auto start_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count();

    for (size_t count = 0; count < task_count; count++) {
      AddTwoFn(operands[count], operands[count+1], &res[count]);
    }

    auto end_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count();
    
    for (size_t count = 0; count < task_count; count++) {
      expected_results[count].swap(res[count].get_future().get());
    }
    serialized_duration = end_ts - start_ts;
  }

  auto status = ni::AsyncWorkQueue::Initialize(4);
  ASSERT_TRUE(status.IsOk()) << status.Message();

  uint64_t parallelized_duration = 0;
  {
    std::vector<std::promise<std::vector<int>>> res(task_count);

    auto start_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count();

    for (size_t count = 0; count < task_count; count++) {
      ni::AsyncWorkQueue::AddTask([&AddTwoFn, &operands, &res, count]() mutable {
        AddTwoFn(operands[count], operands[count+1], &res[count]);
      });
    }
    for (size_t count = 0; count < task_count; count++) {
      res[count].get_future().wait();
    }

    auto end_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now().time_since_epoch())
              .count();
    
    parallelized_duration = end_ts - start_ts;
    EXPECT_LT(parallelized_duration, serialized_duration / 3) << "Expected parallelized work was completed within 1/3 of serialized time";
    for (size_t count = 0; count < task_count; count++) {
      auto res = std::move(res[count].get_future().get());
      EXPECT_EQ(res, expected_results[count]) << "Mismatched parallelized result";
    }
  }
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
