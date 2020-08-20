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
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "src/core/thread_pool.h"

namespace nvidia { namespace inferenceserver {

ThreadPool::~ThreadPool()
{
  for (auto& worker_thread : worker_threads_) {
    if (worker_thread->joinable()) {
      worker_thread->join();
    }
  }
}

Status
ThreadPool::AwaitCompletion()
{
  for (size_t i = 0; i < worker_threads_.size(); i++) {
    if (worker_threads_[i]->joinable()) {
      worker_threads_[i]->join();
      RETURN_IF_ERROR(futures_[i].get());
    }
  }
  return Status::Success;
}

Status
ThreadPool::GetNextAvailableId(int* worker_id, bool await_available)
{
  *worker_id = -1;
  for (size_t i = 0; i < worker_threads_.size(); i++) {
    if (!worker_threads_[i]->joinable()) {
      *worker_id = i;
    }
  }
  if ((*worker_id == -1) && await_available) {
    RETURN_IF_ERROR(AwaitCompletion());
    *worker_id = 0;
  }

  return Status::Success;
}

Status
ThreadPool::AddTask(std::thread task_data, std::promise<Status> promise)
{
  int worker_id;
  RETURN_IF_ERROR(GetNextAvailableId(&worker_id, false));
  if (worker_id == -1) {
    queue_.Put(std::move(task_data));
    promises_.Put(std::move(promise));
  } else {
    futures_[worker_id] = promise.get_future();
    worker_threads_[worker_id].reset(&task_data);
  }
  return Status::Success;
}

Status
ThreadPool::CompleteQueue()
{
  while (!queue_.Empty()) {
    auto task_data = std::move(queue_.Get());
    auto promise = promises_.Get();
    int worker_id;
    RETURN_IF_ERROR(GetNextAvailableId(&worker_id, true));
    futures_[worker_id] = promise.get_future();
    worker_threads_[worker_id].reset(&task_data);
  }
  RETURN_IF_ERROR(AwaitCompletion());

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
