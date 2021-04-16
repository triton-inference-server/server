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

#include <chrono>
#include <condition_variable>
#include <future>
#include <limits>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include "triton/common/async_work_queue.h"

namespace tc = triton::common;

namespace {

// Wrapper of AsyncWorkQueue class to expose Reset() for unit testing
class TestingAsyncWorkQueue : public tc::AsyncWorkQueue {
 public:
  static void Reset() { AsyncWorkQueue::Reset(); }
};

class AsyncWorkQueueTest : public ::testing::Test {
 protected:
  void TearDown() override { TestingAsyncWorkQueue::Reset(); }
};

TEST_F(AsyncWorkQueueTest, InitZeroWorker)
{
  auto error = tc::AsyncWorkQueue::Initialize(0);
  EXPECT_FALSE(error.IsOk()) << "Expect error when initialized with 0 worker";
}

TEST_F(AsyncWorkQueueTest, InitOneWorker)
{
  auto error = tc::AsyncWorkQueue::Initialize(1);
  EXPECT_TRUE(error.IsOk()) << error.Message();
}

TEST_F(AsyncWorkQueueTest, InitFourWorker)
{
  auto error = tc::AsyncWorkQueue::Initialize(1);
  EXPECT_TRUE(error.IsOk()) << error.Message();
}

TEST_F(AsyncWorkQueueTest, InitTwice)
{
  auto error = tc::AsyncWorkQueue::Initialize(4);
  EXPECT_TRUE(error.IsOk()) << error.Message();
  error = tc::AsyncWorkQueue::Initialize(2);
  EXPECT_FALSE(error.IsOk()) << "Expect error from initializing twice";
}

TEST_F(AsyncWorkQueueTest, WorkerCountUninitialized)
{
  EXPECT_EQ(tc::AsyncWorkQueue::WorkerCount(), (size_t)0)
      << "Expect 0 worker count for uninitialized queue";
}

TEST_F(AsyncWorkQueueTest, WorkerCountInitialized)
{
  auto error = tc::AsyncWorkQueue::Initialize(4);
  EXPECT_TRUE(error.IsOk()) << error.Message();
  EXPECT_EQ(tc::AsyncWorkQueue::WorkerCount(), (size_t)4)
      << "Expect 4 worker count for initialized queue";
}


TEST_F(AsyncWorkQueueTest, RunTasksInParallel)
{
  auto AddTwoFn = [](const std::vector<int>& lhs, const std::vector<int>& rhs,
                     std::promise<std::vector<int>>* res) {
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
    // Use large element count to reduce the async work queue overhead
    size_t element_count = 1 << 24;
    auto RandHalfIntFn = std::bind(
        std::uniform_int_distribution<>{std::numeric_limits<int>::min() / 2,
                                        std::numeric_limits<int>::max() / 2},
        std::default_random_engine{});
    for (size_t tc = 0; tc < task_count + 1; tc++) {
      expected_results.push_back(std::vector<int>());
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

    auto start_ts =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();

    for (size_t count = 0; count < task_count; count++) {
      AddTwoFn(operands[count], operands[count + 1], &res[count]);
    }

    auto end_ts =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();

    for (size_t count = 0; count < task_count; count++) {
      expected_results[count] = std::move(res[count].get_future().get());
    }
    serialized_duration = end_ts - start_ts;
  }

  auto error = tc::AsyncWorkQueue::Initialize(4);
  ASSERT_TRUE(error.IsOk()) << error.Message();

  uint64_t parallelized_duration = 0;
  {
    std::vector<std::promise<std::vector<int>>> ps(task_count);
    std::vector<std::future<std::vector<int>>> fs;
    for (auto& p : ps) {
      fs.emplace_back(std::move(p.get_future()));
    }

    auto start_ts =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();

    for (size_t count = 0; count < task_count; count++) {
      tc::AsyncWorkQueue::AddTask([&AddTwoFn, &operands, &ps, count]() mutable {
        AddTwoFn(operands[count], operands[count + 1], &ps[count]);
      });
    }
    for (size_t count = 0; count < task_count; count++) {
      fs[count].wait();
    }

    auto end_ts =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();

    parallelized_duration = end_ts - start_ts;
    // FIXME manual testing shows parallelized time is between 30% to 33.3% for
    // 128 M total elements
    EXPECT_LT(parallelized_duration, serialized_duration / 3)
        << "Expected parallelized work was completed within 1/3 of serialized "
           "time";
    for (size_t count = 0; count < task_count; count++) {
      auto res = std::move(fs[count].get());
      EXPECT_EQ(res, expected_results[count])
          << "Mismatched parallelized result";
    }
  }
}

TEST_F(AsyncWorkQueueTest, RunTasksFIFO)
{
  auto CaptureTimestampFn = [](std::promise<uint64_t>* res) {
    res->set_value(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  };

  size_t task_count = 8;
  std::vector<std::promise<uint64_t>> ps(task_count);

  auto error = tc::AsyncWorkQueue::Initialize(2);
  ASSERT_TRUE(error.IsOk()) << error.Message();

  std::vector<std::promise<void>> barrier(2);
  tc::AsyncWorkQueue::AddTask([&barrier]() mutable {
    barrier[0].get_future().get();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });
  tc::AsyncWorkQueue::AddTask([&barrier]() mutable {
    barrier[1].get_future().get();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  });
  for (size_t count = 0; count < task_count; count++) {
    tc::AsyncWorkQueue::AddTask([count, &CaptureTimestampFn, &ps]() mutable {
      CaptureTimestampFn(&ps[count]);
    });
  }

  // Signal to start the work
  barrier[0].set_value();
  barrier[1].set_value();

  uint64_t prev_ts = 0;
  for (size_t count = 0; count < task_count; count++) {
    uint64_t curr_ts = ps[count].get_future().get();
    EXPECT_LT(prev_ts, curr_ts)
        << "Expected async work is processed in FIFO order";
  }
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}