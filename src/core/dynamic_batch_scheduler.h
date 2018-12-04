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
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

// Scheduler that implements dynamic batching.
class DynamicBatchScheduler : public Scheduler {
 public:
  // Create a scheduler to support a given number of runners and a run
  // function to call when a request is scheduled.
  DynamicBatchScheduler(
    const ModelConfig& config, const uint32_t runner_cnt,
    StandardRunFunc OnSchedule);
  ~DynamicBatchScheduler();

  // \see Scheduler::Enqueue()
  void Enqueue(
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(tensorflow::Status)> OnComplete) override;

 private:
  void SchedulerThread(const uint32_t runner_id, const int nice);
  uint64_t GetDynamicBatch();

  // Function the scheduler will call to schedule a payload(s) for
  // execution.
  const StandardRunFunc OnSchedule_;

  // The number of scheduler threads.
  const uint32_t scheduler_thread_cnt_;

  // The number of scheduler threads currently idle.
  uint32_t idle_scheduler_thread_cnt_;

  // True if dynamic batching is enabled.
  bool dynamic_batching_enabled_;

  // Mutex and condvar protecting the scheduling queue.
  std::mutex mu_;
  std::condition_variable cv_;

  // Queue holding inference requests for the model represented by
  // this servable.
  std::deque<Scheduler::Payload> queue_;

  std::vector<std::unique_ptr<std::thread>> scheduler_threads_;
  std::atomic<bool> scheduler_threads_exit_;

  size_t max_preferred_batch_size_;
  std::set<int32_t> preferred_batch_sizes_;
  uint64_t pending_batch_delay_ns_;
  size_t pending_batch_size_;
  size_t pending_batch_queue_cnt_;
};

}}  // namespace nvidia::inferenceserver
