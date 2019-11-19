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

class SequenceBatch;

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
      StandardInitFunc OnInit, StandardWarmupFunc OnWarmup,
      StandardRunFunc OnSchedule, std::unique_ptr<Scheduler>* scheduler);

  // \see Scheduler::Enqueue()
  void Enqueue(
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(const Status&)> OnComplete) override;

  // A batcher-sequence_slot combination. The batcher is represented
  // by the index into 'batchers_'.
  struct BatcherSequenceSlot {
    BatcherSequenceSlot() = default;
    BatcherSequenceSlot(const BatcherSequenceSlot&) = default;
    BatcherSequenceSlot(size_t b, uint32_t s) : batcher_idx_(b), seq_slot_(s) {}
    size_t batcher_idx_;
    uint32_t seq_slot_;
  };

  // Show that a sequence slot is no longer being used.
  bool ReleaseSequenceSlot(
      const BatcherSequenceSlot& seq_slot,
      std::deque<Scheduler::Payload>* payloads);

  // For debugging/testing, batcher reports how many waiting requests
  // and returns true if the batcher should continue waiting.
  bool DelayScheduler(
      const uint32_t batcher_idx, const size_t cnt, const size_t total);

 private:
  void ReaperThread(const int nice);

  Status CreateBooleanControlTensors(
      const ModelConfig& config,
      std::shared_ptr<InferRequestProvider::InputOverrideMap>*
          start_input_overrides,
      std::shared_ptr<InferRequestProvider::InputOverrideMap>*
          end_input_overrides,
      std::shared_ptr<InferRequestProvider::InputOverrideMap>*
          startend_input_overrides,
      std::shared_ptr<InferRequestProvider::InputOverrideMap>*
          continue_input_overrides,
      std::shared_ptr<InferRequestProvider::InputOverrideMap>*
          notready_input_overrides);

 private:
  struct BatcherSequenceSlotCompare {
    bool operator()(
        const BatcherSequenceSlot& a, const BatcherSequenceSlot& b) const
    {
      return a.seq_slot_ > b.seq_slot_;
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

  // Map from a request's correlation ID to the BatcherSequenceSlot
  // assigned to that correlation ID.
  using BatcherSequenceSlotMap =
      std::unordered_map<CorrelationID, BatcherSequenceSlot>;
  BatcherSequenceSlotMap sequence_to_batcherseqslot_map_;

  // Map from a request's correlation ID to the backlog queue
  // collecting requests for that correlation ID.
  using BacklogMap = std::unordered_map<
      CorrelationID, std::shared_ptr<std::deque<Scheduler::Payload>>>;
  BacklogMap sequence_to_backlog_map_;

  // The ordered backlog of sequences waiting for a free sequenceslot.
  std::deque<std::shared_ptr<std::deque<Scheduler::Payload>>> backlog_queues_;

  // The batcher/sequence-slot locations ready to accept a new
  // sequence. Ordered from lowest sequence-slot-number to highest so
  // that all batchers grow at the same rate and attempt to remain as
  // small as possible.
  std::priority_queue<
      BatcherSequenceSlot, std::vector<BatcherSequenceSlot>,
      BatcherSequenceSlotCompare>
      ready_batcher_seq_slots_;

  // For each correlation ID the most recently seen timestamp, in
  // microseconds, for a request using that correlation ID.
  std::unordered_map<CorrelationID, uint64_t> correlation_id_timestamps_;

  // Used for debugging/testing.
  size_t backlog_delay_cnt_;
  std::vector<size_t> queue_request_cnts_;
};

// Base class for a scheduler that implements a particular scheduling
// strategy for a model instance.
class SequenceBatch {
 public:
  SequenceBatch(
      SequenceBatchScheduler* base, const uint32_t batcher_idx,
      const size_t seq_slot_cnt,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          start_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          end_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          startend_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          continue_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          notready_input_overrides);
  virtual ~SequenceBatch() = default;

  // Enqueue a payload into the appropriate queue for the requested
  // sequence slot.
  virtual void Enqueue(
      const uint32_t seq_slot, const CorrelationID correlation_id,
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(const Status&)> OnComplete) = 0;

 protected:
  bool CreateCorrelationIDControl(const ModelConfig& config);
  void SetControlTensors(
      const InferRequestHeader& request_header,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const int32_t seq_slot, const CorrelationID corr_id);

  // The controlling scheduler.
  SequenceBatchScheduler* const base_;

  // The index of this batcher within the controlling scheduler.
  const uint32_t batcher_idx_;

  // The number of candidate sequence slots.
  const size_t seq_slot_cnt_;

  // The control values, delivered as input tensors, that should be
  // used when starting a sequence, continuing a sequence, ending a
  // sequence, and showing that a sequence has not input available.
  std::shared_ptr<InferRequestProvider::InputOverrideMap>
      start_input_overrides_;
  std::shared_ptr<InferRequestProvider::InputOverrideMap> end_input_overrides_;
  std::shared_ptr<InferRequestProvider::InputOverrideMap>
      startend_input_overrides_;
  std::shared_ptr<InferRequestProvider::InputOverrideMap>
      continue_input_overrides_;
  std::shared_ptr<InferRequestProvider::InputOverrideMap>
      notready_input_overrides_;

  // The name of the model input to which each sequence slot's
  // correlation ID should be delivered. Empty if the model does not
  // specify the CONTROL_SEQUENCE_CORRID control.
  std::string correlation_id_tensor_;

  // For each sequence slot the override map that provides the
  // correlation ID for that slot. Empty if model does not specify
  // the CONTROL_SEQUENCE_CORRID control.
  std::vector<InferRequestProvider::InputOverride*> seq_slot_corrid_overrides_;
  std::vector<std::shared_ptr<InferRequestProvider::InputOverrideMap>>
      seq_slot_corrid_overrides_maps_;
};

// Scheduler that implements the Direct sequence scheduling strategy
// for a model instance.
class DirectSequenceBatch : public SequenceBatch {
 public:
  DirectSequenceBatch(
      SequenceBatchScheduler* base, const uint32_t batcher_idx,
      const size_t seq_slot_cnt, const ModelConfig& config,
      Scheduler::StandardInitFunc OnInit,
      Scheduler::StandardWarmupFunc OnWarmup,
      Scheduler::StandardRunFunc OnSchedule,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          start_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          end_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          startend_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          continue_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          notready_input_overrides,
      std::promise<bool>* is_initialized);
  ~DirectSequenceBatch();

  // Enqueue a payload into the appropriate queue for the requested
  // sequence slot.
  void Enqueue(
      const uint32_t seq_slot, const CorrelationID correlation_id,
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(const Status&)> OnComplete) override;

 private:
  void SchedulerThread(const int nice, std::promise<bool>* is_initialized);

  // Function the scheduler will call to initialize a runner.
  const Scheduler::StandardInitFunc OnInit_;

  // Function the scheduler will call to warmup a runner.
  const Scheduler::StandardWarmupFunc OnWarmup_;

  // Function to call to execute this batch of requests.
  const Scheduler::StandardRunFunc OnSchedule_;

  // The thread scheduling payloads queued in this batch.
  std::unique_ptr<std::thread> scheduler_thread_;
  bool scheduler_thread_exit_;
  bool scheduler_idle_;

  // Mutex protecting correlation queues, etc.
  std::mutex mu_;
  std::condition_variable cv_;

  // The request header needed to create a null provider to use when
  // an inference is issuing and there is no request available in a
  // sequence slot.
  InferRequestHeader null_request_header_;

  // Queues holding inference requests. There are 'seq_slot_cnt'
  // queues, one for each sequence slot where requests assigned to
  // that slot are enqueued to wait for inferencing.
  std::vector<std::deque<Scheduler::Payload>> queues_;

  // Is each sequence slot active or not? A zero value indicates
  // inactive, a non-zero value indicates active and is the
  // correlation ID of the sequence active in the slot. An empty
  // queue for a sequence slot does not mean it's inactive... it
  // could just not have any requests pending at the moment.
  std::vector<CorrelationID> seq_slot_correlation_ids_;

  // The maximum active sequence slot. A value of -1 indicates that
  // no slots are active in the backend.
  int32_t max_active_seq_slot_;
};

// Scheduler that implements the oldest-first sequence scheduling
// strategy for a model instance.
class OldestSequenceBatch : public SequenceBatch {
 public:
  OldestSequenceBatch(
      SequenceBatchScheduler* base, const uint32_t batcher_idx,
      const size_t seq_slot_cnt, const ModelConfig& config,
      Scheduler::StandardInitFunc OnInit,
      Scheduler::StandardWarmupFunc OnWarmup,
      Scheduler::StandardRunFunc OnSchedule,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          start_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          end_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          startend_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          continue_input_overrides,
      const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
          notready_input_overrides,
      std::promise<bool>* is_initialized);
  ~OldestSequenceBatch();

  // Enqueue a payload into the appropriate queue for the requested
  // sequence slot.
  void Enqueue(
      const uint32_t seq_slot, const CorrelationID correlation_id,
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(const Status&)> OnComplete) override;

 private:
  void CompleteAndNext(
      const uint32_t seq_slot, std::function<void(const Status&)> OnComplete,
      const Status& status);

  // The dynamic batcher for this scheduler
  std::unique_ptr<Scheduler> dynamic_batcher_;

  // Mutex protecting queues, etc.
  std::mutex mu_;

  // For each sequence slot, true if there is a request for that
  // sequence in-flight in the dynamic batcher. Used to ensure that at
  // most one request from each sequence can be scheduled at a time.
  std::vector<bool> in_flight_;

  // Queues holding inference requests. There are 'seq_slot_cnt'
  // queues, one for each sequence slot where requests assigned to
  // that slot are enqueued to wait for inferencing.
  std::vector<std::deque<Scheduler::Payload>> queues_;
};

}}  // namespace nvidia::inferenceserver
