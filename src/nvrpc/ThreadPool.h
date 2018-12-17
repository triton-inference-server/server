// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
//
//
// Original Source: https://github.com/progschj/ThreadPool
//
// Original License:
//
// Copyright (c) 2012 Jakob Progsch, Václav Zeman
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
//   1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
//
//   2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
//
//   3. This notice may not be removed or altered from any source
//   distribution.
//
// Modifications:
//   * Header-only file was split into .h/.cc files
//   * Added an extra safety check (lines 30-31) in the construction (.cc file).
//   * Added CPU affinity options to the constructor
//   * Added Size() method to get thread count
//
#pragma once

#include <future>
#include <queue>

namespace nvrpc {

/**
 * @brief Manages a Pool of Threads that consume a shared work Queue
 *
 * ThreadPool is the primary resoruce class for handling threads used throughout
 * the examples and tests.  The library is entirely a BYO-resources; however,
 * this implemenation is provided as a convenience class.  Many thanks to the
 * original authors for a beautifully designed class.
 */
class ThreadPool {
 public:
  /**
   * @brief Construct a new Thread Pool
   * @param nThreads Number of Worker Threads
   */
  ThreadPool(size_t nThreads);

  ~ThreadPool();

  /**
   * @brief Enqueue Work to the ThreadPool by passing a Lambda Function
   *
   * Variadic template allows for an arbituary number of arguments to be passed
   * the captured lambda function.  Captures are still allowed and used
   * throughout the examples.
   *
   * The queue can grow larger than the number of threads.  A single worker
   * thread executues pulls a lambda function off the queue and executes it to
   * completion.  These are synchronous executions in an async messaging
   * library.  These synchronous pools can be swapped for truely async workers
   * using libevent or asio.  Happy to accept PRs to improve the async
   * abilities.
   *
   * @tparam F
   * @tparam Args
   * @param f
   * @param args
   * @return std::future<typename std::result_of<F(Args...)>::type>
   */
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;

  /**
   * @brief Number of Threads in the Pool
   */
  int Size();

 private:
  void InitThread();

  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::queue<std::function<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

// add new work item to the pool
template <class F, class... Args>
auto
ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    // don't allow enqueueing after stopping the pool
    if (stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");

    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
}

}  // end namespace nvrpc
