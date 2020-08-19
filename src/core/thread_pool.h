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

#include "src/core/cuda_utils.h"
#include "src/core/status.h"
#include "src/core/sync_queue.h"

namespace nvidia { namespace inferenceserver {

class WorkerThreadPool {
 public:
  WorkerThreadPool(int thread_count) : thread_count_(thread_count)
  {
    worker_threads_.reserve(thread_count);
    futures_.reserve(thread_count);
  }

  ~WorkerThreadPool();

  // Wait for all pending worker threads to finish.
  Status AwaitCompletion();

  // Get Id of next available worker thread. If all workers are occupied then
  // wait for them to finish and start from the beginning.
  Status GetNextAvailableId(int* worker_id);

  // Add CopyBuffer task to queue
  Status AddTask(
      const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
      const int64_t src_memory_type_id,
      const TRITONSERVER_MemoryType dst_memory_type,
      const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
      void* dst, cudaStream_t cuda_stream, bool* cuda_used);

  // Run CopyBuffer on the worker threads. Must add to queue using AddTask
  // before calling ProcessQueue.
  Status ProcessQueue();

 private:
  /// A struct that stores the parameters for the CopyBuffer operation.
  struct CopyBufferData {
    CopyBufferData(
        const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
        const int64_t src_memory_type_id,
        const TRITONSERVER_MemoryType dst_memory_type,
        const int64_t dst_memory_type_id, const size_t byte_size,
        const void* src, void* dst, cudaStream_t cuda_stream, bool* cuda_used)
        : msg_(msg), src_memory_type_(src_memory_type),
          src_memory_type_id_(src_memory_type_id),
          dst_memory_type_(dst_memory_type),
          dst_memory_type_id_(dst_memory_type_id), byte_size_(byte_size),
          src_(src), dst_(dst), cuda_stream_(cuda_stream), cuda_used_(cuda_used)
    {
    }

    const std::string& msg_;
    const TRITONSERVER_MemoryType src_memory_type_;
    const int64_t src_memory_type_id_;
    const TRITONSERVER_MemoryType dst_memory_type_;
    const int64_t dst_memory_type_id_;
    const size_t byte_size_;
    const void* src_;
    void* dst_;
    cudaStream_t cuda_stream_;
    bool* cuda_used_;
    std::promise<Status> status_;
  };

  // Helper around CopyBuffer
  static void CopyBufferHandler(CopyBufferData* data);

  int thread_count_;
  std::vector<std::thread> worker_threads_;
  std::vector<std::future<Status>> futures_;
  std::condition_variable cv_;
  SyncQueue<std::unique_ptr<CopyBufferData>> queue_;
};

}}  // namespace nvidia::inferenceserver
