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
#pragma once

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "src/core/status.h"
#include "src/core/sync_queue.h"

namespace nvidia { namespace inferenceserver {

class ThreadPool {
 public:
  ThreadPool(int thread_count) : thread_count_(thread_count)
  {
    worker_threads_.reserve(thread_count);
    futures_.reserve(thread_count);
  }

  ~ThreadPool();

  // Add task thread to queue.
  Status AddTask(std::thread task_data, std::promise<Status> promise);

  // Run task threads remaining in queue on the worker threads.
  Status CompleteQueue();

 private:
  // Wait for all pending worker threads to finish.
  Status AwaitCompletion();

  // Get Id of next available worker thread. If all workers are occupied then
  // wait for them to finish and start from the beginning.
  Status GetNextAvailableId(int* worker_id, bool await_available);

  int thread_count_;
  std::vector<std::unique_ptr<std::thread>> worker_threads_;
  std::vector<std::future<Status>> futures_;
  SyncQueue<std::promise<Status>> promises_;
  SyncQueue<std::thread> queue_;
};

}}  // namespace nvidia::inferenceserver
