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

class AsyncWorkQueue {
 public:
  // Should only be called once to set the number of worker threads.
  static void SetWorkerCount(size_t thread_count)
  {
    GetSingleton()->exit_ = false;
    for (size_t id = 0; id < thread_count; id++)
      GetSingleton()->worker_threads_.push_back(std::unique_ptr<std::thread>(
          new std::thread([id] { Initialize(id); })));
  }

  // Add task thread to queue.
  static void AddTask(const std::function<Status(void*)> task, void* task_data)
  {
    std::lock_guard<std::mutex> lock(GetSingleton()->mutex_);
    GetSingleton()->task_queue_.Put(task);
    GetSingleton()->data_queue_.Put(task_data);
    GetSingleton()->queue_pending.notify_one();
  }

  // Wait till queue is empty and return vector of status
  static void GetResults(std::vector<Status>* status_queue)
  {
    while (!GetSingleton()->task_queue_.Empty()) {
    }

    GetSingleton()->queue_pending.notify_one();
    std::lock_guard<std::mutex> lock(GetSingleton()->mutex_);

    while (!GetSingleton()->status_queue_.Empty()) {
      auto status = std::move(GetSingleton()->status_queue_.Get());
      status_queue->push_back(status);
    }
  }

 private:
  AsyncWorkQueue(){};
  ~AsyncWorkQueue()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      exit_ = true;
      queue_pending.notify_one();
    }

    for (auto& worker_thread : worker_threads_) {
      if (worker_thread->joinable()) {
        worker_thread->join();
      }
    }
  }

  static AsyncWorkQueue* GetSingleton()
  {
    static AsyncWorkQueue singleton;
    return &singleton;
  }

  static void Initialize(size_t worker_id)
  {
    std::function<Status(void*)> task;
    void* task_data;

    while (true) {
      {
        std::unique_lock<std::mutex> lock(GetSingleton()->mutex_);
        GetSingleton()->queue_pending.wait(lock, [&]() {
          return GetSingleton()->exit_ || !GetSingleton()->task_queue_.Empty();
        });

        if (GetSingleton()->exit_)
          return;

        task = GetSingleton()->task_queue_.Get();
        task_data = GetSingleton()->data_queue_.Get();
      }
      Status status = task(task_data);
      GetSingleton()->status_queue_.Put(status);
    }
  }

  std::vector<std::unique_ptr<std::thread>> worker_threads_;
  SyncQueue<Status> status_queue_;
  SyncQueue<std::function<Status(void*)>> task_queue_;
  SyncQueue<void*> data_queue_;
  std::condition_variable queue_pending;
  std::mutex mutex_;
  bool exit_;
};

}}  // namespace nvidia::inferenceserver
