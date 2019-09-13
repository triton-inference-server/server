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

#include "src/core/sequence_batch_scheduler.h"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"
#include "src/core/provider.h"
#include "src/core/server_status.h"

namespace nvidia { namespace inferenceserver {

Status
SequenceBatchScheduler::Create(
    const ModelConfig& config, const uint32_t runner_cnt,
    StandardInitFunc OnInit, StandardRunFunc OnSchedule,
    std::unique_ptr<Scheduler>* scheduler)
{
  std::unique_ptr<SequenceBatchScheduler> sched(new SequenceBatchScheduler());

  // For debugging and testing,
  const char* dstr = getenv("TRTSERVER_BACKLOG_DELAY_SCHEDULER");
  sched->backlog_delay_cnt_ = 0;
  if (dstr != nullptr) {
    sched->backlog_delay_cnt_ = atoi(dstr);
    LOG_INFO << "Delaying scheduler until " << sched->backlog_delay_cnt_
             << " backlog queued payloads...";
  }
  sched->queue_request_cnts_.resize(runner_cnt, 0);

  // Max sequence idle...
  sched->max_sequence_idle_microseconds_ =
      config.sequence_batching().max_sequence_idle_microseconds();

  // Get the batch size to allow for each runner. This is at least 1
  // even if the model doesn't support batching.
  size_t batch_size = std::max(1, config.max_batch_size());

  // Based on the model configuration create input tensors for control
  // signals indicating sequence start, sequence continue, and
  // sequence not ready.
  std::shared_ptr<InferRequestProvider::InputOverrideMap> start;
  std::shared_ptr<InferRequestProvider::InputOverrideMap> cont;
  std::shared_ptr<InferRequestProvider::InputOverrideMap> notready;
  RETURN_IF_ERROR(
      sched->CreateControlTensors(config, &start, &cont, &notready));

  // Create one SequenceBatch object for each requested runner. The
  // SequenceBatch object has a thread that manages the batch of
  // requests.
  for (uint32_t c = 0; c < runner_cnt; ++c) {
    std::promise<bool> init_state;
    std::shared_ptr<SequenceBatch> sb = std::make_shared<SequenceBatch>(
        sched.get(), c, batch_size, config, OnInit, OnSchedule, start, cont,
        notready, &init_state);

    if (init_state.get_future().get()) {
      sched->batchers_.push_back(sb);
      // All slots in the batch are initially ready for a new sequence.
      for (size_t b = 0; b < batch_size; ++b) {
        sched->ready_batch_slots_.push(SequenceBatchScheduler::BatchSlot(c, b));
      }
    }
  }
  if (sched->batchers_.empty()) {
    return Status(
        RequestStatusCode::INTERNAL,
        "Initialization failed for all sequence-batch scheduler threads");
  }

  // Create a reaper thread that watches for idle sequences. Run the
  // reaper a lower priority.
  SequenceBatchScheduler* raw = sched.release();

  raw->reaper_thread_exit_ = false;
  raw->reaper_thread_.reset(
      new std::thread([raw]() { raw->ReaperThread(10 /* nice */); }));

  scheduler->reset(raw);

  return Status::Success;
}

SequenceBatchScheduler::~SequenceBatchScheduler()
{
  // Signal the reaper thread to exit...
  {
    std::unique_lock<std::mutex> lock(mu_);
    reaper_thread_exit_ = true;
  }

  reaper_cv_.notify_one();
  if ((reaper_thread_ != nullptr) && reaper_thread_->joinable()) {
    reaper_thread_->join();
  }
}

Status
SequenceBatchScheduler::CreateControlTensors(
    const ModelConfig& config,
    std::shared_ptr<InferRequestProvider::InputOverrideMap>*
        start_input_overrides,
    std::shared_ptr<InferRequestProvider::InputOverrideMap>*
        continue_input_overrides,
    std::shared_ptr<InferRequestProvider::InputOverrideMap>*
        notready_input_overrides)
{
  // Currently only batch-size 1 requests are supported so only need
  // to provide control vectors of that size.
  *start_input_overrides =
      std::make_shared<InferRequestProvider::InputOverrideMap>();
  *continue_input_overrides =
      std::make_shared<InferRequestProvider::InputOverrideMap>();
  *notready_input_overrides =
      std::make_shared<InferRequestProvider::InputOverrideMap>();

  std::string tensor_name;
  DataType tensor_datatype;
  int32_t int32_false_value, int32_true_value;
  float fp32_false_value, fp32_true_value;

  // START
  {
    RETURN_IF_ERROR(GetSequenceControlProperties(
        config.sequence_batching(), config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        true /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value));
    uint8_t* false_p =
        ((tensor_datatype == DataType::TYPE_INT32)
             ? reinterpret_cast<uint8_t*>(&int32_false_value)
             : reinterpret_cast<uint8_t*>(&fp32_false_value));
    uint8_t* true_p =
        ((tensor_datatype == DataType::TYPE_INT32)
             ? reinterpret_cast<uint8_t*>(&int32_true_value)
             : reinterpret_cast<uint8_t*>(&fp32_true_value));

    auto false_override =
        std::make_shared<InferRequestProvider::InputOverride>();
    false_override->content_.assign(false_p, false_p + sizeof(float));
    false_override->dims_.Add(1);
    false_override->datatype_ = tensor_datatype;

    auto true_override =
        std::make_shared<InferRequestProvider::InputOverride>();
    true_override->content_.assign(true_p, true_p + sizeof(float));
    true_override->dims_.Add(1);
    true_override->datatype_ = tensor_datatype;

    (*start_input_overrides)
        ->insert(std::make_pair(tensor_name, true_override));
    (*continue_input_overrides)
        ->insert(std::make_pair(tensor_name, false_override));
    (*notready_input_overrides)
        ->insert(std::make_pair(tensor_name, false_override));
  }

  // READY
  {
    RETURN_IF_ERROR(GetSequenceControlProperties(
        config.sequence_batching(), config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY,
        true /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value));
    uint8_t* false_p =
        ((tensor_datatype == DataType::TYPE_INT32)
             ? reinterpret_cast<uint8_t*>(&int32_false_value)
             : reinterpret_cast<uint8_t*>(&fp32_false_value));
    uint8_t* true_p =
        ((tensor_datatype == DataType::TYPE_INT32)
             ? reinterpret_cast<uint8_t*>(&int32_true_value)
             : reinterpret_cast<uint8_t*>(&fp32_true_value));

    auto false_override =
        std::make_shared<InferRequestProvider::InputOverride>();
    false_override->content_.assign(false_p, false_p + sizeof(float));
    false_override->dims_.Add(1);
    false_override->datatype_ = tensor_datatype;

    auto true_override =
        std::make_shared<InferRequestProvider::InputOverride>();
    true_override->content_.assign(true_p, true_p + sizeof(float));
    true_override->dims_.Add(1);
    true_override->datatype_ = tensor_datatype;

    (*start_input_overrides)
        ->insert(std::make_pair(tensor_name, true_override));
    (*continue_input_overrides)
        ->insert(std::make_pair(tensor_name, true_override));
    (*notready_input_overrides)
        ->insert(std::make_pair(tensor_name, false_override));
  }

  return Status::Success;
}

void
SequenceBatchScheduler::Enqueue(
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(const Status&)> OnComplete)
{
  // Queue timer starts at the beginning of the queueing and
  // scheduling process
  stats->CaptureTimestamp(ModelInferStats::TimestampKind::kQueueStart);

  const auto& request_header = request_provider->RequestHeader();

  // For now the request must have batch-size 1 since the sequence
  // batcher does not yet support requests that are statically
  // batched.
  if (request_header.batch_size() != 1) {
    OnComplete(Status(
        RequestStatusCode::INVALID_ARG,
        "inference request to model '" + request_provider->ModelName() +
            "' must specify batch-size 1 due to requirements of sequence "
            "batcher"));
    return;
  }

  // A request must have a correlation ID to be processed correctly by
  // this scheduler. A value of 0 (zero) indicates that the request
  // doesn't have a correlation ID.
  const CorrelationID correlation_id = request_header.correlation_id();
  if (correlation_id == 0) {
    OnComplete(Status(
        RequestStatusCode::INVALID_ARG,
        "inference request to model '" + request_provider->ModelName() +
            "' must specify a non-zero correlation ID"));
    return;
  }

  BatchSlot* target = nullptr;

  const bool seq_start =
      ((request_header.flags() & InferRequestHeader::FLAG_SEQUENCE_START) != 0);
  const bool seq_end =
      ((request_header.flags() & InferRequestHeader::FLAG_SEQUENCE_END) != 0);

  std::unique_lock<std::mutex> lock(mu_);

  auto sb_itr = sequence_to_batchslot_map_.find(correlation_id);
  auto bl_itr = sequence_to_backlog_map_.find(correlation_id);

  // If this request is not starting a new sequence its correlation ID
  // should already be known with a target in either a slot or in the
  // backlog. If it doesn't then the sequence wasn't started correctly
  // or there has been a correlation ID conflict. In either case fail
  // this request.
  if (!seq_start && (sb_itr == sequence_to_batchslot_map_.end()) &&
      (bl_itr == sequence_to_backlog_map_.end())) {
    OnComplete(Status(
        RequestStatusCode::INVALID_ARG,
        "inference request for sequence " + std::to_string(correlation_id) +
            " to model '" + request_provider->ModelName() +
            "' must specify the START flag on the first request of the "
            "sequence"));
    return;
  }

  // Record the timestamp of this request for the correlation ID. The
  // reaper thread will check to make sure that
  // max_sequence_idle_microseconds value is not exceed for any
  // sequence, and if it is it will release the slot (if any)
  // allocated to that sequence.
  {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t now_us = TIMESPEC_TO_NANOS(now) / 1000;
    correlation_id_timestamps_[correlation_id] = now_us;
  }

  // If this request starts a new sequence but the correlation ID
  // already has an in-progress sequence then that previous sequence
  // did not end correctly, or there is a correlation ID conflict. In
  // this case we continue the new sequence (in either backlog or
  // slot). It is ok for a backlog/slot to have multiple starts... as
  // long as it has a single end. The previous sequence that was not
  // correctly ended will have its existing requests handled and then
  // the new sequence will start.
  if (seq_start && ((sb_itr != sequence_to_batchslot_map_.end()) ||
                    (bl_itr != sequence_to_backlog_map_.end()))) {
    LOG_WARNING
        << "sequence " << correlation_id << " for model '"
        << request_provider->ModelName()
        << "' has a conflict. The previous sequence did not end before this "
           "sequence start. Previous sequence will be terminated early.";
  }

  // This request already has an assigned slot...
  if (sb_itr != sequence_to_batchslot_map_.end()) {
    target = &sb_itr->second;
  }
  // This request already has a queue in the backlog...
  else if (bl_itr != sequence_to_backlog_map_.end()) {
    LOG_VERBOSE(1)
        << "Enqueuing sequence inference request into backlog for model '"
        << request_provider->ModelName();

    bl_itr->second->emplace_back(
        stats, request_provider, response_provider, OnComplete);
    // If the sequence is ending then forget correlation ID
    // connection to this backlog queue. If another sequence starts
    // with the same correlation ID it will be collected in another
    // backlog queue.
    if (seq_end) {
      sequence_to_backlog_map_.erase(bl_itr);
    }
    return;
  }
  // This request does not have an assigned backlog or slot. By the
  // above checks it must be starting. If there is a free slot
  // available then assign this sequence to that slot...
  else if (!ready_batch_slots_.empty()) {
    target = &sequence_to_batchslot_map_[correlation_id];
    *target = ready_batch_slots_.top();
    ready_batch_slots_.pop();
  }
  // Last option is to assign this request to the backlog...
  else {
    LOG_VERBOSE(1)
        << "Enqueuing sequence inference request into new backlog for model '"
        << request_provider->ModelName();

    auto backlog = std::make_shared<std::deque<Scheduler::Payload>>();
    backlog_queues_.push_back(backlog);
    backlog->emplace_back(
        stats, request_provider, response_provider, OnComplete);
    if (!seq_end) {
      sequence_to_backlog_map_[correlation_id] = std::move(backlog);
    }
    return;
  }

  // Need to grab the target contents before the erase below since
  // that can free it.
  const size_t batcher_idx = target->batcher_idx_;
  const uint32_t slot = target->slot_;

  // At this point the request has been assigned to a slot. If the
  // sequence is ending then stop tracking the correlation.
  if (seq_end) {
    sequence_to_batchslot_map_.erase(correlation_id);
  }

  // Enqueue request into batcher and slot.  No need to hold the lock
  // while enqueuing in a specific batcher.
  lock.unlock();

  LOG_VERBOSE(1) << "Enqueuing sequence inference request for model '"
                 << request_provider->ModelName() << "' into batcher "
                 << batcher_idx << ", slot " << slot;

  batchers_[batcher_idx]->Enqueue(
      slot, correlation_id, stats, request_provider, response_provider,
      OnComplete);
}

bool
SequenceBatchScheduler::ReleaseBatchSlot(
    const BatchSlot& batch_slot, std::deque<Scheduler::Payload>* payloads)
{
  std::unique_lock<std::mutex> lock(mu_);

  // If there is a backlogged sequence and it is requested, return it
  // so that it can use the newly available slot.
  if (!backlog_queues_.empty()) {
    auto& backlog = backlog_queues_.front();
    *payloads = std::move(*backlog);
    backlog_queues_.pop_front();
    if (!payloads->empty()) {  // should never be empty...
      const auto& request_provider = payloads->back().request_provider_;
      const auto& request_header = request_provider->RequestHeader();
      const CorrelationID correlation_id = request_header.correlation_id();

      // If the last queue entry is not an END request then the entire
      // sequence is not contained in the backlog. In that case must
      // update backlog and batchslot maps so that future requests get
      // directed to the batch slot instead of the backlog.
      const bool seq_end =
          ((request_header.flags() & InferRequestHeader::FLAG_SEQUENCE_END) !=
           0);
      if (!seq_end) {
        // Since the correlation ID is being actively collected in the
        // backlog, there should not be any in-flight sequences with
        // that same correlation ID that have an assigned slot.
        if (sequence_to_batchslot_map_.find(correlation_id) !=
            sequence_to_batchslot_map_.end()) {
          LOG_ERROR << "internal: backlog sequence " << correlation_id
                    << " conflicts with in-flight sequence for model '"
                    << request_provider->ModelName() << "'";
        }

        sequence_to_backlog_map_.erase(correlation_id);
        sequence_to_batchslot_map_[correlation_id] = batch_slot;
      }

      LOG_VERBOSE(1) << "Reusing slot in batcher " << batch_slot.batcher_idx_
                     << ", slot " << batch_slot.slot_ << " for sequence "
                     << correlation_id;
      return false;
    }
  }

  // There is no backlogged sequence so just release the batch slot
  LOG_VERBOSE(1) << "Freeing slot in batcher " << batch_slot.batcher_idx_
                 << ", slot " << batch_slot.slot_;

  ready_batch_slots_.push(batch_slot);
  return true;
}

bool
SequenceBatchScheduler::DelayScheduler(
    const uint32_t batcher_idx, const size_t cnt, const size_t total)
{
  std::unique_lock<std::mutex> lock(mu_);
  queue_request_cnts_[batcher_idx] = cnt;

  size_t seen = 0;
  for (auto c : queue_request_cnts_) {
    seen += c;
  }

  if (seen < total) {
    return true;
  }

  if (backlog_delay_cnt_ > 0) {
    size_t backlog_seen = 0;
    for (const auto& q : backlog_queues_) {
      backlog_seen += q->size();
    }

    if (backlog_seen < backlog_delay_cnt_) {
      return true;
    }
  }

  return false;
}

void
SequenceBatchScheduler::ReaperThread(const int nice)
{
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting sequence-batch reaper thread at nice " << nice
                   << "...";
  } else {
    LOG_VERBOSE(1) << "Starting sequence-batch reaper thread at default nice "
                      "(requested nice "
                   << nice << " failed)...";
  }

  const uint64_t backlog_idle_wait_microseconds = 50 * 1000;

  while (!reaper_thread_exit_) {
    std::unique_lock<std::mutex> lock(mu_);

    uint64_t wait_microseconds = max_sequence_idle_microseconds_;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t now_us = TIMESPEC_TO_NANOS(now) / 1000;

    for (auto cid_itr = correlation_id_timestamps_.cbegin();
         cid_itr != correlation_id_timestamps_.cend();) {
      int64_t remaining_microseconds =
          (int64_t)max_sequence_idle_microseconds_ - (now_us - cid_itr->second);
      if (remaining_microseconds > 0) {
        wait_microseconds =
            std::min(wait_microseconds, (uint64_t)remaining_microseconds + 1);
        ++cid_itr;
        continue;
      }

      const CorrelationID idle_correlation_id = cid_itr->first;
      LOG_VERBOSE(1) << "Max sequence idle exceeded for sequence "
                     << idle_correlation_id;

      auto idle_sb_itr = sequence_to_batchslot_map_.find(idle_correlation_id);

      // If the idle correlation ID has an assigned slot, then release
      // that assignment so it becomes available for another
      // sequence. An assignment is released by enqueuing a payload
      // with null providers and null completion callback. The
      // scheduler thread will interpret the payload as meaning it
      // should release the slot but otherwise do nothing with the
      // payload.
      if (idle_sb_itr != sequence_to_batchslot_map_.end()) {
        // Need to grab the contents before the erase below since that
        // can free it.
        const size_t batcher_idx = idle_sb_itr->second.batcher_idx_;
        const uint32_t slot = idle_sb_itr->second.slot_;

        LOG_VERBOSE(1) << "reaper enqueuing force-end in batcher "
                       << batcher_idx << ", slot " << slot << " for sequence "
                       << idle_correlation_id;


        sequence_to_batchslot_map_.erase(idle_correlation_id);

        batchers_[batcher_idx]->Enqueue(
            slot, idle_correlation_id, nullptr, nullptr, nullptr, nullptr);
        cid_itr = correlation_id_timestamps_.erase(cid_itr);
      } else {
        // If the idle correlation ID is in the backlog, then just
        // need to increase the timeout so that we revisit it again
        // in the future to check if it is assigned to a slot.
        auto idle_bl_itr = sequence_to_backlog_map_.find(idle_correlation_id);
        if (idle_bl_itr != sequence_to_backlog_map_.end()) {
          LOG_VERBOSE(1) << "reaper found idle sequence in backlog so "
                            "extending timeout for sequence "
                         << idle_correlation_id;
          wait_microseconds =
              std::min(wait_microseconds, backlog_idle_wait_microseconds);
          ++cid_itr;
        } else {
          LOG_VERBOSE(1) << "ignoring stale idle for sequence "
                         << idle_correlation_id;
          cid_itr = correlation_id_timestamps_.erase(cid_itr);
        }
      }
    }

    // Wait until the next idle timeout needs to be checked
    if (wait_microseconds > 0) {
      LOG_VERBOSE(1) << "Sequence-batch reaper sleeping for "
                     << wait_microseconds << "us...";
      std::chrono::microseconds wait_timeout(wait_microseconds);
      reaper_cv_.wait_for(lock, wait_timeout);
    }
  }

  LOG_VERBOSE(1) << "Stopping sequence-batch reaper thread...";
}


SequenceBatchScheduler::SequenceBatch::SequenceBatch(
    SequenceBatchScheduler* base, const uint32_t batcher_idx,
    const size_t batch_size, const ModelConfig& config, StandardInitFunc OnInit,
    StandardRunFunc OnSchedule,
    const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
        start_input_overrides,
    const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
        continue_input_overrides,
    const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
        notready_input_overrides,
    std::promise<bool>* is_initialized)
    : OnInit_(OnInit), OnSchedule_(OnSchedule), base_(base),
      batcher_idx_(batcher_idx), scheduler_thread_exit_(false),
      scheduler_idle_(false), queues_(batch_size), max_active_slot_(-1),
      slot_correlation_ids_(batch_size, 0),
      start_input_overrides_(start_input_overrides),
      continue_input_overrides_(continue_input_overrides),
      notready_input_overrides_(notready_input_overrides)
{
  // Create a scheduler thread associated with 'batcher_idx' that
  // executes the queued payloads.
  const int nice = GetCpuNiceLevel(config);
  scheduler_thread_.reset(new std::thread([this, nice, is_initialized]() {
    SchedulerThread(nice, is_initialized);
  }));
}

SequenceBatchScheduler::SequenceBatch::~SequenceBatch()
{
  // Signal the scheduler thread to exit...
  {
    std::unique_lock<std::mutex> lock(mu_);
    scheduler_thread_exit_ = true;
  }

  cv_.notify_one();
  scheduler_thread_->join();
}

void
SequenceBatchScheduler::SequenceBatch::Enqueue(
    const uint32_t slot, const CorrelationID correlation_id,
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(const Status&)> OnComplete)
{
  bool wake_runner = false;

  {
    std::lock_guard<std::mutex> lock(mu_);

    // All requests in this SequenceBatch must have the same shape for
    // all inputs (since they are going to be executed together in a
    // batch). If this is the first request into this SequenceBatch
    // then grab a copy of the request header that is needed to create
    // NULL version request providers that can stand in as
    // representative when inference is issuing and there is no
    // request available in one or more slots.
    if ((max_active_slot_ == -1) && (request_provider != nullptr)) {
      null_request_header_ = request_provider->RequestHeader();
    }

    queues_[slot].emplace_back(
        stats, request_provider, response_provider, OnComplete);

    slot_correlation_ids_[slot] = correlation_id;
    max_active_slot_ = std::max(max_active_slot_, static_cast<int32_t>(slot));

    // If runner is idle then wake it to service this request. We do
    // the actual wake outside of the lock to avoid having the woken
    // thread immediately block on the lock
    wake_runner = scheduler_idle_;
  }

  if (wake_runner) {
    cv_.notify_one();
  }
}

void
SequenceBatchScheduler::SequenceBatch::SchedulerThread(
    const int nice, std::promise<bool>* is_initialized)
{
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting sequence-batch scheduler thread "
                   << batcher_idx_ << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting sequence-batch scheduler thread "
                   << batcher_idx_ << " at default nice (requested nice "
                   << nice << " failed)...";
  }

  // Initialize using the thread. If error then just exit this thread
  // now... that means the corresponding model instance will not have
  // any runner and so will not get used for execution.
  Status init_status = OnInit_(batcher_idx_);
  if (!init_status.IsOk()) {
    LOG_ERROR << "Initialization failed for sequence-batch scheduler thread "
              << batcher_idx_ << ": " << init_status.Message();
    is_initialized->set_value(false);
    return;
  } else {
    is_initialized->set_value(true);
  }

  // For debugging and testing, delay start of thread until queues
  // contain the specified number of entries (across all
  // SequenceBatchs in the scheduler).
  const char* dstr = getenv("TRTSERVER_DELAY_SCHEDULER");
  size_t delay_cnt = 0;
  if (dstr != nullptr) {
    delay_cnt = atoi(dstr);
    LOG_INFO << "Delaying scheduler thread " << batcher_idx_ << " until "
             << delay_cnt << " queued payloads...";
  }

  const uint64_t default_wait_microseconds = 500 * 1000;

  while (!scheduler_thread_exit_) {
    auto payloads = std::make_shared<std::vector<Scheduler::Payload>>();
    uint64_t wait_microseconds = 0;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);

      bool adjust_max_active_slot = false;

      if (delay_cnt > 0) {
        wait_microseconds = 10 * 1000;
        // Debugging/testing... wait until queues together contain at
        // least 'delay_cnt' items...
        size_t total_size = 0;
        for (const auto& q : queues_) {
          total_size += q.size();
        }
        if (!base_->DelayScheduler(batcher_idx_, total_size, delay_cnt)) {
          delay_cnt = 0;
        }
        LOG_INFO << "Delaying scheduler thread " << batcher_idx_ << " until "
                 << delay_cnt
                 << " queued payloads, current total = " << total_size;
      } else {
        // Make sure there is at least one request that needs to be
        // handled. Find the largest slot index that has a payload
        // available...
        int32_t max_slot = max_active_slot_;
        while ((max_slot >= 0) && queues_[max_slot].empty()) {
          max_slot--;
        }

        if (max_slot < 0) {
          wait_microseconds = default_wait_microseconds;
        } else {
          // Collect payloads from slot 0 to max_slot.
          for (int32_t slot = 0; slot <= max_slot; ++slot) {
            bool end_of_sequence = false;
            bool use_null_provider = false;
            std::deque<Scheduler::Payload>& queue = queues_[slot];

            // If 'slot' doesn't have any requests then change the
            // request provider to send dummy/null input tensors for
            // this slot. We need this so that other payloads stay in
            // the correct slot.
            if (queue.empty()) {
              use_null_provider = true;
            } else {
              // If the payload has no request provider then the
              // sequence is being forcibly ended (e.g. because it has
              // been idle to long). Use a null provider for the slot
              // since there isn't an actual payload but also handle
              // as if it were the end of the sequence.
              Scheduler::Payload& slot_payload = queue.front();
              if (slot_payload.request_provider_ == nullptr) {
                use_null_provider = true;
                end_of_sequence = true;
                queue.pop_front();
              }
            }

            // Use null-provider if necessary otherwise the next
            // payload in the queue...
            if (use_null_provider) {
              auto null_request_provider =
                  std::make_shared<NULLInferRequestProvider>(
                      null_request_header_);
              null_request_provider->SetInputOverride(
                  notready_input_overrides_);

              payloads->emplace_back(
                  nullptr, null_request_provider, nullptr, nullptr);
            } else {
              Scheduler::Payload& slot_payload = queue.front();
              const auto& request_provider = slot_payload.request_provider_;
              const auto& request_header = request_provider->RequestHeader();

              // If this is the first payload in a sequence then send
              // the appropriate sequence start indicator to the
              // backend.
              if ((request_header.flags() &
                   InferRequestHeader::FLAG_SEQUENCE_START) != 0) {
                request_provider->SetInputOverride(start_input_overrides_);
              } else {
                request_provider->SetInputOverride(continue_input_overrides_);
              }

              payloads->emplace_back(
                  slot_payload.stats_, request_provider,
                  slot_payload.response_provider_,
                  slot_payload.complete_function_);

              queue.pop_front();

              if ((request_header.flags() &
                   InferRequestHeader::FLAG_SEQUENCE_END) != 0) {
                end_of_sequence = true;
              }
            }

            // If the sequence has ended then attempt to refill the
            // slot with a sequence from the backlog. If there is no
            // backlog show that the slot is no longer active, and if
            // it is currently the maximum active slot note that we
            // need to adjust max_active_slot_ once all slots are
            // processed (we defer processing because multiple slots
            // could have ending sequences).
            if (end_of_sequence) {
              LOG_VERBOSE(1) << "Ending sequence in batcher " << batcher_idx_
                             << ", slot " << slot;

              // Should never be anything in a queue after the END
              // marker. If it happens that means we will clobber
              // that request if/when we swap in a backlog sequence
              // in ReleaseBatchSlot below.
              if (!queue.empty()) {
                LOG_ERROR << "internal: unexpected requests after sequence "
                             "end in slot "
                          << slot;
              }

              SequenceBatchScheduler::BatchSlot batch_slot(batcher_idx_, slot);
              bool released = base_->ReleaseBatchSlot(batch_slot, &queue);
              if (released) {
                slot_correlation_ids_[slot] = 0;
                if (slot == max_active_slot_) {
                  adjust_max_active_slot = true;
                }
              }
            }
          }
        }
      }

      // If one or more sequences ended, and one of them was in
      // max_active_slot_, then need to find the new max_active_slot_.
      if (adjust_max_active_slot) {
        while ((max_active_slot_ >= 0) &&
               (slot_correlation_ids_[max_active_slot_] == 0)) {
          max_active_slot_--;
        }
      }

      // If no requests are to be handled, wait for notification or
      // for the specified timeout before checking the queues again.
      if (wait_microseconds > 0) {
        scheduler_idle_ = true;
        std::chrono::microseconds wait_timeout(wait_microseconds);
        cv_.wait_for(lock, wait_timeout);
        scheduler_idle_ = false;
      }
    }

    if ((payloads != nullptr) && !payloads->empty()) {
      auto OnCompleteQueuedPayloads = [payloads](const Status& rstatus) {
        // Payloads that don't have a completion function don't have
        // anywhere to report their errors. Those errors could have
        // caused other payloads to have issues (due to mis-alignment
        // within the batch, etc.). So if any such payload has an
        // error we just fail all payloads.
        Status status = rstatus;
        if (status.IsOk()) {
          for (auto& payload : *payloads) {
            if (payload.complete_function_ == nullptr) {
              if (!payload.status_.IsOk()) {
                status = payload.status_;
                break;
              }
            }
          }
        }

        // Complete each payload by calling the competion function.
        bool found_success = false;
        for (auto& payload : *payloads) {
          const Status& final_status = status.IsOk() ? payload.status_ : status;

          // All the payloads executed together, so count 1 execution
          // in the first successful payload. Other payloads stay at 0
          // executions.
          if (!found_success && final_status.IsOk() &&
              (payload.stats_ != nullptr)) {
            payload.stats_->SetModelExecutionCount(1);
            found_success = true;
          }

          if (payload.complete_function_ != nullptr) {
            payload.complete_function_(final_status);
          }
        }
      };

      // Run the backend...
      OnSchedule_(batcher_idx_, payloads.get(), OnCompleteQueuedPayloads);
    }
  }  // end runner loop

  LOG_VERBOSE(1) << "Stopping sequence-batch scheduler thread " << batcher_idx_
                 << "...";
}

}}  // namespace nvidia::inferenceserver
