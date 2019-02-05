// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <sys/time.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

class NULLInferRequestProvider;

// Scheduler that implements batching for a sequence of correlated
// inferences.
class SequenceBatchScheduler : public Scheduler {
 public:
  // Create a scheduler to support a given number of runners and a run
  // function to call when a request is scheduled.
  SequenceBatchScheduler(
      const ModelConfig& config, const uint32_t runner_cnt,
      StandardRunFunc OnSchedule);

  // \see Scheduler::Enqueue()
  void Enqueue(
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(tensorflow::Status)> OnComplete) override;

 private:
  // Scheduler payload for each request.
  struct SequencePayload : public Scheduler::Payload {
    SequencePayload() = default;
    SequencePayload(const SequencePayload& payload) = delete;
    SequencePayload(SequencePayload&& payload) : Payload(std::move(payload)) {}
    SequencePayload(
        std::unique_ptr<ModelInferStats::ScopedTimer>& queue_timer,
        const std::shared_ptr<ModelInferStats>& stats,
        const std::shared_ptr<InferRequestProvider>& request_provider,
        const std::shared_ptr<InferResponseProvider>& response_provider,
        const std::function<void(tensorflow::Status)> complete_function)
        : Payload(
              queue_timer, stats, request_provider, response_provider,
              complete_function)
    {
    }
  };

  // Queued requests for a model instance that will be sent through
  // that instance together in a batch.
  class SequenceBatch {
   public:
    SequenceBatch(
        const uint32_t runner_id, const size_t batch_size,
        const ModelConfig& config, StandardRunFunc OnSchedule);
    ~SequenceBatch();

    // Return the index within the batch that has no queued
    // requests. If there are multiple such indices return the lowest
    // numbered one.
    bool GetFreeSlot(uint32_t* slot);

    // Enqueue a payload into the appropriate queue for the requested
    // slot.
    void Enqueue(
        const uint32_t slot, const CorrelationID correlation_id,
        std::unique_ptr<ModelInferStats::ScopedTimer>& queue_timer,
        const std::shared_ptr<ModelInferStats>& stats,
        const std::shared_ptr<InferRequestProvider>& request_provider,
        const std::shared_ptr<InferResponseProvider>& response_provider,
        std::function<void(tensorflow::Status)> OnComplete);

   private:
    void SchedulerThread(const uint32_t runner_id, const int nice);

    // Function to call to execute this batch of requests.
    const StandardRunFunc OnSchedule_;

    // The thread scheduling payloads queued in this batch.
    std::unique_ptr<std::thread> scheduler_thread_;
    bool scheduler_thread_exit_;
    bool scheduler_idle_;

    // Mutex protecting correlation IDs, queues, max-active-slots.
    std::mutex mu_;
    std::condition_variable cv_;

    // The request provider to use when an inference is issuing and
    // there is no request available in a slot.
    std::shared_ptr<NULLInferRequestProvider> null_request_provider_;

    // The correlation ID of the requests using a batch slot or 0
    // (zero) if the slot is currently unused.
    std::vector<CorrelationID> correlation_ids_;

    // Queues holding inference requests. There are 'batch_size'
    // queues, one for each batch slot where requests assigned to that
    // slot are enqueued to wait for inferencing.
    std::vector<std::deque<SequencePayload>> queues_;

    // The maximum active slot. A value of -1 indicates that no slots
    // are active in the bundle.
    int32_t max_active_slot_;
  };

 private:
  // The SequenceBatch's being managed by this scheduler.
  std::vector<std::shared_ptr<SequenceBatch>> batches_;

  // The target location for requests for a given correlation ID. The
  // target is either a SequenceBatch or a backlog queue.
  struct SequenceTarget {
    // Return true if this target is a backlog queue, false if this
    // target is a SequenceBatch_slot.
    bool IsBacklog() const { return sequence_batch_ == nullptr; }

    // If 'sequence_batch_' is non-null then the target is 'slot_'
    // within 'sequence_batch_'.
    std::shared_ptr<SequenceBatch> sequence_batch_;
    uint32_t slot_;

    // If 'sequence_batch_' is null then the target is a backlog
    // queue.
    std::deque<SequencePayload> backlog_;
  };

  // Map from a request's correlation ID to the SequenceBatch+slot or backlog
  // queue assigned to that correlation ID.
  using SequenceTargetMap = std::unordered_map<CorrelationID, SequenceTarget>;
  SequenceTargetMap sequence_to_target_map_;

  // Ordered list of correlation IDs in the backlog. When a slot
  // becomes available the first item from the backlog, if any, is
  // used to fill that slot.
  std::deque<CorrelationID> backlog_sequence_ids_;

  // Mutex protecting correlation IDs -> SequenceBatch maps.
  std::mutex mu_;
};

}}  // namespace nvidia::inferenceserver
