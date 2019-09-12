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
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/provider.h"
#include "src/core/scheduler.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// Scheduler that implements batching across sequences of correlated
// inferences.
class SequenceBatchScheduler : public Scheduler {
 public:
  SequenceBatchScheduler() = default;
  ~SequenceBatchScheduler();

  // Create a scheduler to support a given number of runners and a run
  // function to call when a request is scheduled.
  static Status Create(
      const ModelConfig& config, const uint32_t runner_cnt,
      StandardInitFunc OnInit, StandardRunFunc OnSchedule,
      std::unique_ptr<Scheduler>* scheduler);

  // \see Scheduler::Enqueue()
  void Enqueue(
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(const Status&)> OnComplete) override;

  // A batch-slot combination. The batch is represented by the index
  // into 'batches_'.
  struct BatchSlot {
    BatchSlot() = default;
    BatchSlot(const BatchSlot&) = default;
    BatchSlot(size_t b, uint32_t s) : batcher_idx_(b), slot_(s) {}
    size_t batcher_idx_;
    uint32_t slot_;
  };

  // Show that a batch slot is no longer being used.
  bool ReleaseBatchSlot(
      const BatchSlot& batch_slot, std::deque<Scheduler::Payload>* payloads);

  // For debugging/testing, batcher reports how many waiting requests
  // and returns true if the batcher should continue waiting.
  bool DelayScheduler(
      const uint32_t batcher_idx, const size_t cnt, const size_t total);

 private:
  void ReaperThread(const int nice);

  Status CreateControlTensors(
      const ModelConfig& config,
      std::shared_ptr<InferRequestProvider::InputOverrideMap>*
          start_input_overrides,
      std::shared_ptr<InferRequestProvider::InputOverrideMap>*
          continue_input_overrides,
      std::shared_ptr<InferRequestProvider::InputOverrideMap>*
          notready_input_overrides);

  // Queued requests for a model instance that will be sent through
  // that instance together in a batch.
  class SequenceBatch {
   public:
    SequenceBatch(
        SequenceBatchScheduler* base, const uint32_t batcher_idx,
        const size_t batch_size, const ModelConfig& config,
        StandardInitFunc OnInit, StandardRunFunc OnSchedule,
        const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
            start_input_overrides,
        const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
            continue_input_overrides,
        const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
            notready_input_overrides,
        std::promise<bool>* is_initialized);
    ~SequenceBatch();

    // Enqueue a payload into the appropriate queue for the requested
    // slot.
    void Enqueue(
        const uint32_t slot, const CorrelationID correlation_id,
        const std::shared_ptr<ModelInferStats>& stats,
        const std::shared_ptr<InferRequestProvider>& request_provider,
        const std::shared_ptr<InferResponseProvider>& response_provider,
        std::function<void(const Status&)> OnComplete);

   private:
    void SchedulerThread(const int nice, std::promise<bool>* is_initialized);

    // Function the scheduler will call to initialize a runner.
    const StandardInitFunc OnInit_;

    // Function to call to execute this batch of requests.
    const StandardRunFunc OnSchedule_;

    // The controlling scheduler.
    SequenceBatchScheduler* const base_;

    // The index of this batcher within the controlling scheduler.
    const uint32_t batcher_idx_;

    // The thread scheduling payloads queued in this batch.
    std::unique_ptr<std::thread> scheduler_thread_;
    bool scheduler_thread_exit_;
    bool scheduler_idle_;

    // Mutex protecting correlation queues, etc.
    std::mutex mu_;
    std::condition_variable cv_;

    // The request header needed to create a null provider to use when
    // an inference is issuing and there is no request available in a
    // slot.
    InferRequestHeader null_request_header_;

    // Queues holding inference requests. There are 'batch_size'
    // queues, one for each batch slot where requests assigned to that
    // slot are enqueued to wait for inferencing.
    std::vector<std::deque<Scheduler::Payload>> queues_;

    // The maximum active slot. A value of -1 indicates that no slots
    // are active in the backend.
    int32_t max_active_slot_;

    // Is each batch slot active or not. An empty queue for a batch
    // slot does not mean its empty... it could just not have any
    // requests pending at the moment.
    std::vector<CorrelationID> slot_correlation_ids_;

    // The control values, delivered as input tensors, that should be
    // used when starting a sequence, continuing a sequence, and
    // showing that a sequence has not input available.
    std::shared_ptr<InferRequestProvider::InputOverrideMap>
        start_input_overrides_;
    std::shared_ptr<InferRequestProvider::InputOverrideMap>
        continue_input_overrides_;
    std::shared_ptr<InferRequestProvider::InputOverrideMap>
        notready_input_overrides_;
  };

 private:
  struct BatchSlotCompare {
    bool operator()(const BatchSlot& a, const BatchSlot& b) const
    {
      return a.slot_ > b.slot_;
    }
  };

  // The max_sequence_idle_microseconds value for this scheduler.
  uint64_t max_sequence_idle_microseconds_;

  // Mutex
  std::mutex mu_;

  // The reaper thread
  std::unique_ptr<std::thread> reaper_thread_;
  std::condition_variable reaper_cv_;
  bool reaper_thread_exit_;

  // The SequenceBatchs being managed by this scheduler.
  std::vector<std::shared_ptr<SequenceBatch>> batchers_;

  // Map from a request's correlation ID to the BatchSlot assigned to
  // that correlation ID.
  using BatchSlotMap = std::unordered_map<CorrelationID, BatchSlot>;
  BatchSlotMap sequence_to_batchslot_map_;

  // Map from a request's correlation ID to the backlog queue
  // collecting requests for that correlation ID.
  using BacklogMap = std::unordered_map<
      CorrelationID, std::shared_ptr<std::deque<Scheduler::Payload>>>;
  BacklogMap sequence_to_backlog_map_;

  // The ordered backlog of sequences waiting for a free slot.
  std::deque<std::shared_ptr<std::deque<Scheduler::Payload>>> backlog_queues_;

  // The batch/slot locations ready to accept a new sequence. Ordered
  // from lowest slot-number to highest so that all batches grow at
  // the same rate and attempt to remain as small as possible.
  std::priority_queue<BatchSlot, std::vector<BatchSlot>, BatchSlotCompare>
      ready_batch_slots_;

  // For each correlation ID the most recently seen timestamp, in
  // microseconds, for a request using that correlation ID.
  std::unordered_map<CorrelationID, uint64_t> correlation_id_timestamps_;

  // Used for debugging/testing.
  size_t backlog_delay_cnt_;
  std::vector<size_t> queue_request_cnts_;
};

}}  // namespace nvidia::inferenceserver
