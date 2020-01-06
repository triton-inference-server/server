// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "src/core/dynamic_batch_scheduler.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"
#include "src/core/provider.h"
#include "src/core/server_status.h"

namespace nvidia { namespace inferenceserver {

Status
SequenceBatchScheduler::Create(
    const ModelConfig& config, const uint32_t runner_cnt,
    Scheduler::StandardInitFunc OnInit, Scheduler::StandardWarmupFunc OnWarmup,
    Scheduler::StandardRunFunc OnSchedule,
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

  // Get the number of candidate sequence slots to allow for each
  // runner. This is at least 1 even if the model doesn't support
  // batching.
  const size_t model_batch_size = std::max(1, config.max_batch_size());
  size_t seq_slot_cnt = model_batch_size;
  if (config.sequence_batching().has_oldest()) {
    seq_slot_cnt =
        config.sequence_batching().oldest().max_candidate_sequences();
  }

  // Based on the model configuration create input tensors for control
  // signals indicating sequence start, sequence continue, and
  // sequence not ready.
  std::shared_ptr<InferRequestProvider::InputOverrideMap> start;
  std::shared_ptr<InferRequestProvider::InputOverrideMap> end;
  std::shared_ptr<InferRequestProvider::InputOverrideMap> startend;
  std::shared_ptr<InferRequestProvider::InputOverrideMap> cont;
  std::shared_ptr<InferRequestProvider::InputOverrideMap> notready;
  RETURN_IF_ERROR(sched->CreateBooleanControlTensors(
      config, &start, &end, &startend, &cont, &notready));

  // Create one SequenceBatch object for each requested runner. The
  // SequenceBatch object has a thread that manages the batch of
  // requests.
  for (uint32_t c = 0; c < runner_cnt; ++c) {
    std::promise<bool> init_state;
    std::shared_ptr<SequenceBatch> sb;

    // Create the SequenceBatch derivative that handles the requested
    // scheduling strategy.
    if (config.sequence_batching().has_oldest()) {
      sb.reset(new OldestSequenceBatch(
          sched.get(), c, seq_slot_cnt, config, OnInit, OnWarmup, OnSchedule,
          start, end, startend, cont, notready, &init_state));
    } else {
      sb.reset(new DirectSequenceBatch(
          sched.get(), c, seq_slot_cnt, config, OnInit, OnWarmup, OnSchedule,
          start, end, startend, cont, notready, &init_state));
    }

    if (init_state.get_future().get()) {
      sched->batchers_.push_back(sb);
      // All sequence slots in the batcher are initially ready for a
      // new sequence.
      for (size_t b = 0; b < seq_slot_cnt; ++b) {
        sched->ready_batcher_seq_slots_.push(
            SequenceBatchScheduler::BatcherSequenceSlot(c, b));
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
SequenceBatchScheduler::CreateBooleanControlTensors(
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
        notready_input_overrides)
{
  // Currently only batch-size 1 requests are supported so only need
  // to provide control vectors of that size.
  *start_input_overrides =
      std::make_shared<InferRequestProvider::InputOverrideMap>();
  *end_input_overrides =
      std::make_shared<InferRequestProvider::InputOverrideMap>();
  *startend_input_overrides =
      std::make_shared<InferRequestProvider::InputOverrideMap>();
  *continue_input_overrides =
      std::make_shared<InferRequestProvider::InputOverrideMap>();
  *notready_input_overrides =
      std::make_shared<InferRequestProvider::InputOverrideMap>();

  std::string tensor_name;
  DataType tensor_datatype;
  int32_t int32_false_value, int32_true_value;
  float fp32_false_value, fp32_true_value;

  // START, optional
  {
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        config.sequence_batching(), config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START,
        false /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value));
    if (!tensor_name.empty()) {
      uint8_t* false_p =
          ((tensor_datatype == DataType::TYPE_INT32)
               ? reinterpret_cast<uint8_t*>(&int32_false_value)
               : reinterpret_cast<uint8_t*>(&fp32_false_value));
      uint8_t* true_p =
          ((tensor_datatype == DataType::TYPE_INT32)
               ? reinterpret_cast<uint8_t*>(&int32_true_value)
               : reinterpret_cast<uint8_t*>(&fp32_true_value));

      InferRequestProvider::InputOverride false_override;
      false_override.content_.assign(false_p, false_p + sizeof(float));
      false_override.dims_.Add(1);
      false_override.datatype_ = tensor_datatype;

      InferRequestProvider::InputOverride true_override;
      true_override.content_.assign(true_p, true_p + sizeof(float));
      true_override.dims_.Add(1);
      true_override.datatype_ = tensor_datatype;

      (*start_input_overrides)
          ->insert(std::make_pair(tensor_name, true_override));
      (*end_input_overrides)
          ->insert(std::make_pair(tensor_name, false_override));
      (*startend_input_overrides)
          ->insert(std::make_pair(tensor_name, true_override));
      (*continue_input_overrides)
          ->insert(std::make_pair(tensor_name, false_override));
      (*notready_input_overrides)
          ->insert(std::make_pair(tensor_name, false_override));
    }
  }

  // END, optional
  {
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        config.sequence_batching(), config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_END,
        false /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value));
    if (!tensor_name.empty()) {
      uint8_t* false_p =
          ((tensor_datatype == DataType::TYPE_INT32)
               ? reinterpret_cast<uint8_t*>(&int32_false_value)
               : reinterpret_cast<uint8_t*>(&fp32_false_value));
      uint8_t* true_p =
          ((tensor_datatype == DataType::TYPE_INT32)
               ? reinterpret_cast<uint8_t*>(&int32_true_value)
               : reinterpret_cast<uint8_t*>(&fp32_true_value));

      InferRequestProvider::InputOverride false_override;
      false_override.content_.assign(false_p, false_p + sizeof(float));
      false_override.dims_.Add(1);
      false_override.datatype_ = tensor_datatype;

      InferRequestProvider::InputOverride true_override;
      true_override.content_.assign(true_p, true_p + sizeof(float));
      true_override.dims_.Add(1);
      true_override.datatype_ = tensor_datatype;

      (*start_input_overrides)
          ->insert(std::make_pair(tensor_name, false_override));
      (*end_input_overrides)
          ->insert(std::make_pair(tensor_name, true_override));
      (*startend_input_overrides)
          ->insert(std::make_pair(tensor_name, true_override));
      (*continue_input_overrides)
          ->insert(std::make_pair(tensor_name, false_override));
      (*notready_input_overrides)
          ->insert(std::make_pair(tensor_name, false_override));
    }
  }

  // READY, optional
  {
    RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
        config.sequence_batching(), config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY,
        false /* required */, &tensor_name, &tensor_datatype, &fp32_false_value,
        &fp32_true_value, &int32_false_value, &int32_true_value));
    if (!tensor_name.empty()) {
      uint8_t* false_p =
          ((tensor_datatype == DataType::TYPE_INT32)
               ? reinterpret_cast<uint8_t*>(&int32_false_value)
               : reinterpret_cast<uint8_t*>(&fp32_false_value));
      uint8_t* true_p =
          ((tensor_datatype == DataType::TYPE_INT32)
               ? reinterpret_cast<uint8_t*>(&int32_true_value)
               : reinterpret_cast<uint8_t*>(&fp32_true_value));

      InferRequestProvider::InputOverride false_override;
      false_override.content_.assign(false_p, false_p + sizeof(float));
      false_override.dims_.Add(1);
      false_override.datatype_ = tensor_datatype;

      InferRequestProvider::InputOverride true_override;
      true_override.content_.assign(true_p, true_p + sizeof(float));
      true_override.dims_.Add(1);
      true_override.datatype_ = tensor_datatype;

      (*start_input_overrides)
          ->insert(std::make_pair(tensor_name, true_override));
      (*end_input_overrides)
          ->insert(std::make_pair(tensor_name, true_override));
      (*startend_input_overrides)
          ->insert(std::make_pair(tensor_name, true_override));
      (*continue_input_overrides)
          ->insert(std::make_pair(tensor_name, true_override));
      (*notready_input_overrides)
          ->insert(std::make_pair(tensor_name, false_override));
    }
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
#ifdef TRTIS_ENABLE_STATS
  // Queue timer starts at the beginning of the queueing and
  // scheduling process
  stats->CaptureTimestamp(ModelInferStats::TimestampKind::kQueueStart);
#endif  // TRTIS_ENABLE_STATS

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

  BatcherSequenceSlot* target = nullptr;

  const bool seq_start =
      ((request_header.flags() & InferRequestHeader::FLAG_SEQUENCE_START) != 0);
  const bool seq_end =
      ((request_header.flags() & InferRequestHeader::FLAG_SEQUENCE_END) != 0);

  std::unique_lock<std::mutex> lock(mu_);

  auto sb_itr = sequence_to_batcherseqslot_map_.find(correlation_id);
  auto bl_itr = sequence_to_backlog_map_.find(correlation_id);

  // If this request is not starting a new sequence its correlation ID
  // should already be known with a target in either a sequence slot
  // or in the backlog. If it doesn't then the sequence wasn't started
  // correctly or there has been a correlation ID conflict. In either
  // case fail this request.
  if (!seq_start && (sb_itr == sequence_to_batcherseqslot_map_.end()) &&
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
  // sequence, and if it is it will release the sequence slot (if any)
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
  // sequence slot). It is ok for a backlog/slot to have multiple
  // starts... as long as it has a single end. The previous sequence
  // that was not correctly ended will have its existing requests
  // handled and then the new sequence will start.
  if (seq_start && ((sb_itr != sequence_to_batcherseqslot_map_.end()) ||
                    (bl_itr != sequence_to_backlog_map_.end()))) {
    LOG_WARNING
        << "sequence " << correlation_id << " for model '"
        << request_provider->ModelName()
        << "' has a conflict. The previous sequence did not end before this "
           "sequence start. Previous sequence will be terminated early.";
  }

  // This request already has an assigned slot...
  if (sb_itr != sequence_to_batcherseqslot_map_.end()) {
    target = &sb_itr->second;
  }
  // This request already has a queue in the backlog...
  else if (bl_itr != sequence_to_backlog_map_.end()) {
    LOG_VERBOSE(1) << "Enqueuing CORRID " << correlation_id
                   << " into existing backlog: "
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
  // This request does not have an assigned backlog or sequence
  // slot. By the above checks it must be starting. If there is a free
  // sequence slot available then assign this sequence to that slot...
  else if (!ready_batcher_seq_slots_.empty()) {
    target = &sequence_to_batcherseqslot_map_[correlation_id];
    *target = ready_batcher_seq_slots_.top();
    ready_batcher_seq_slots_.pop();
  }
  // Last option is to assign this request to the backlog...
  else {
    LOG_VERBOSE(1) << "Enqueuing CORRID " << correlation_id
                   << " into new backlog: " << request_provider->ModelName();

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
  const uint32_t seq_slot = target->seq_slot_;

  // At this point the request has been assigned to a sequence
  // slot. If the sequence is ending then stop tracking the
  // correlation.
  if (seq_end) {
    sequence_to_batcherseqslot_map_.erase(correlation_id);
  }

  // Enqueue request into batcher and sequence slot.  Don't hold the
  // lock while enqueuing in a specific batcher.
  lock.unlock();

  LOG_VERBOSE(1) << "Enqueuing CORRID " << correlation_id << " into batcher "
                 << batcher_idx << ", sequence slot " << seq_slot << ": "
                 << request_provider->ModelName();

  batchers_[batcher_idx]->Enqueue(
      seq_slot, correlation_id, stats, request_provider, response_provider,
      OnComplete);
}

bool
SequenceBatchScheduler::ReleaseSequenceSlot(
    const BatcherSequenceSlot& batcher_seq_slot,
    std::deque<Scheduler::Payload>* payloads)
{
  std::unique_lock<std::mutex> lock(mu_);

  // If there is a backlogged sequence and it is requested, return it
  // so that it can use the newly available sequence slot.
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
      // update backlog and batcherseqslot maps so that future
      // requests get directed to the batcher sequence-slot instead of
      // the backlog.
      const bool seq_end =
          ((request_header.flags() & InferRequestHeader::FLAG_SEQUENCE_END) !=
           0);
      if (!seq_end) {
        // Since the correlation ID is being actively collected in the
        // backlog, there should not be any in-flight sequences with
        // that same correlation ID that have an assigned slot.
        if (sequence_to_batcherseqslot_map_.find(correlation_id) !=
            sequence_to_batcherseqslot_map_.end()) {
          LOG_ERROR << "internal: backlog sequence " << correlation_id
                    << " conflicts with in-flight sequence for model '"
                    << request_provider->ModelName() << "'";
        }

        sequence_to_backlog_map_.erase(correlation_id);
        sequence_to_batcherseqslot_map_[correlation_id] = batcher_seq_slot;
      }

      LOG_VERBOSE(1) << "CORRID " << correlation_id << " reusing batcher "
                     << batcher_seq_slot.batcher_idx_ << ", slot "
                     << batcher_seq_slot.seq_slot_ << ": "
                     << request_provider->ModelName();
      return false;
    }
  }

  // There is no backlogged sequence so just release the batch slot
  LOG_VERBOSE(1) << "Freeing slot in batcher " << batcher_seq_slot.batcher_idx_
                 << ", slot " << batcher_seq_slot.seq_slot_;

  ready_batcher_seq_slots_.push(batcher_seq_slot);
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
    uint64_t wait_microseconds = max_sequence_idle_microseconds_;
    BatcherSequenceSlotMap force_end_sequences;

    {
      std::unique_lock<std::mutex> lock(mu_);

      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      uint64_t now_us = TIMESPEC_TO_NANOS(now) / 1000;

      for (auto cid_itr = correlation_id_timestamps_.cbegin();
           cid_itr != correlation_id_timestamps_.cend();) {
        int64_t remaining_microseconds =
            (int64_t)max_sequence_idle_microseconds_ -
            (now_us - cid_itr->second);
        if (remaining_microseconds > 0) {
          wait_microseconds =
              std::min(wait_microseconds, (uint64_t)remaining_microseconds + 1);
          ++cid_itr;
          continue;
        }

        const CorrelationID idle_correlation_id = cid_itr->first;
        LOG_VERBOSE(1) << "Reaper: CORRID " << idle_correlation_id
                       << ": max sequence idle exceeded";

        auto idle_sb_itr =
            sequence_to_batcherseqslot_map_.find(idle_correlation_id);

        // If the idle correlation ID has an assigned sequence slot,
        // then release that assignment so it becomes available for
        // another sequence. Release is done by enqueuing and must be
        // done outside the lock, so just collect needed info here.
        if (idle_sb_itr != sequence_to_batcherseqslot_map_.end()) {
          force_end_sequences[idle_correlation_id] = idle_sb_itr->second;

          sequence_to_batcherseqslot_map_.erase(idle_correlation_id);
          cid_itr = correlation_id_timestamps_.erase(cid_itr);
        } else {
          // If the idle correlation ID is in the backlog, then just
          // need to increase the timeout so that we revisit it again in
          // the future to check if it is assigned to a sequence slot.
          auto idle_bl_itr = sequence_to_backlog_map_.find(idle_correlation_id);
          if (idle_bl_itr != sequence_to_backlog_map_.end()) {
            LOG_VERBOSE(1) << "Reaper: found idle CORRID "
                           << idle_correlation_id;
            wait_microseconds =
                std::min(wait_microseconds, backlog_idle_wait_microseconds);
            ++cid_itr;
          } else {
            LOG_VERBOSE(1) << "Reaper: ignoring stale idle CORRID "
                           << idle_correlation_id;
            cid_itr = correlation_id_timestamps_.erase(cid_itr);
          }
        }
      }
    }

    // Enqueue force-ends outside of the lock.
    for (const auto& pr : force_end_sequences) {
      const CorrelationID idle_correlation_id = pr.first;
      const size_t batcher_idx = pr.second.batcher_idx_;
      const uint32_t seq_slot = pr.second.seq_slot_;

      LOG_VERBOSE(1) << "Reaper: force-ending CORRID " << idle_correlation_id
                     << " in batcher " << batcher_idx << ", slot " << seq_slot;

      // A slot assignment is released by enqueuing a payload with
      // null providers and null completion callback. The scheduler
      // thread will interpret the payload as meaning it should
      // release the sequence slot but otherwise do nothing with the
      // payload.
      batchers_[batcher_idx]->Enqueue(
          seq_slot, idle_correlation_id, nullptr, nullptr, nullptr, nullptr);
    }

    // Wait until the next idle timeout needs to be checked
    if (wait_microseconds > 0) {
      std::unique_lock<std::mutex> lock(mu_);
      LOG_VERBOSE(1) << "Reaper: sleeping for " << wait_microseconds << "us...";
      std::chrono::microseconds wait_timeout(wait_microseconds);
      reaper_cv_.wait_for(lock, wait_timeout);
    }
  }

  LOG_VERBOSE(1) << "Stopping sequence-batch reaper thread...";
}

SequenceBatch::SequenceBatch(
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
        notready_input_overrides)
    : base_(base), batcher_idx_(batcher_idx), seq_slot_cnt_(seq_slot_cnt),
      start_input_overrides_(start_input_overrides),
      end_input_overrides_(end_input_overrides),
      startend_input_overrides_(startend_input_overrides),
      continue_input_overrides_(continue_input_overrides),
      notready_input_overrides_(notready_input_overrides)
{
}

bool
SequenceBatch::CreateCorrelationIDControl(const ModelConfig& config)
{
  // If model wants CORRID control then get the name of the input
  // tensor and initialize the override structure for each sequence
  // slot that is used to communicate the correlation ID.
  DataType correlation_id_datatype;
  Status corrid_status = GetTypedSequenceControlProperties(
      config.sequence_batching(), config.name(),
      ModelSequenceBatching::Control::CONTROL_SEQUENCE_CORRID,
      false /* required */, &correlation_id_tensor_, &correlation_id_datatype);
  if (!corrid_status.IsOk()) {
    LOG_ERROR << "Failed validating CORRID control for sequence-batch "
                 "scheduler thread "
              << batcher_idx_ << ": " << corrid_status.Message();
    return false;
  }

  if (!correlation_id_tensor_.empty()) {
    if ((correlation_id_datatype != TYPE_UINT64) &&
        (correlation_id_datatype != TYPE_INT64) &&
        (correlation_id_datatype != TYPE_UINT32) &&
        (correlation_id_datatype != TYPE_INT32)) {
      LOG_ERROR << "unexpected control data type, expected TYPE_UINT64, "
                   "TYPE_INT64, TYPE_UINT32 or TYPE_INT32 for "
                << ModelSequenceBatching_Control_Kind_Name(
                       ModelSequenceBatching::Control::CONTROL_SEQUENCE_CORRID)
                << " for " << config.name();
      return false;
    }

    for (size_t b = 0; b < seq_slot_cnt_; ++b) {
      seq_slot_corrid_overrides_maps_.emplace_back(
          new InferRequestProvider::InputOverrideMap());
      std::shared_ptr<InferRequestProvider::InputOverrideMap>& ovr_map =
          seq_slot_corrid_overrides_maps_.back();
      InferRequestProvider::InputOverride& ovr =
          (*ovr_map)[correlation_id_tensor_];
      ovr.dims_.Add(1);
      ovr.datatype_ = correlation_id_datatype;
      ovr.content_.resize(GetDataTypeByteSize(correlation_id_datatype));

      seq_slot_corrid_overrides_.push_back(&ovr);
    }
  }

  return true;
}

void
SequenceBatch::SetControlTensors(
    const InferRequestHeader& request_header,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const int32_t seq_slot, const CorrelationID corr_id)
{
  // Set the start, end, and ready control tensors
  // appropriately...
  if ((request_header.flags() & (InferRequestHeader::FLAG_SEQUENCE_START |
                                 InferRequestHeader::FLAG_SEQUENCE_END)) ==
      (InferRequestHeader::FLAG_SEQUENCE_START |
       InferRequestHeader::FLAG_SEQUENCE_END)) {
    request_provider->AddInputOverrides(startend_input_overrides_);
  } else if (
      (request_header.flags() & InferRequestHeader::FLAG_SEQUENCE_START) != 0) {
    request_provider->AddInputOverrides(start_input_overrides_);
  } else if (
      (request_header.flags() & InferRequestHeader::FLAG_SEQUENCE_END) != 0) {
    request_provider->AddInputOverrides(end_input_overrides_);
  } else {
    request_provider->AddInputOverrides(continue_input_overrides_);
  }

  // Set correlation ID control tensor if requested by the model.
  if (!correlation_id_tensor_.empty()) {
    const uint8_t* corrid_p = reinterpret_cast<const uint8_t*>(&corr_id);
    std::vector<uint8_t>& content =
        seq_slot_corrid_overrides_[seq_slot]->content_;
    content.assign(corrid_p, corrid_p + content.size());
    request_provider->AddInputOverrides(
        seq_slot_corrid_overrides_maps_[seq_slot]);
  }
}

DirectSequenceBatch::DirectSequenceBatch(
    SequenceBatchScheduler* base, const uint32_t batcher_idx,
    const size_t seq_slot_cnt, const ModelConfig& config,
    Scheduler::StandardInitFunc OnInit, Scheduler::StandardWarmupFunc OnWarmup,
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
    std::promise<bool>* is_initialized)
    : SequenceBatch(
          base, batcher_idx, seq_slot_cnt, start_input_overrides,
          end_input_overrides, startend_input_overrides,
          continue_input_overrides, notready_input_overrides),
      OnInit_(OnInit), OnWarmup_(OnWarmup), OnSchedule_(OnSchedule),
      scheduler_thread_exit_(false), scheduler_idle_(false),
      queues_(seq_slot_cnt), seq_slot_correlation_ids_(seq_slot_cnt, 0),
      max_active_seq_slot_(-1)
{
  // Initialize to handle CORRID control. If error just exit
  // now... that means the corresponding model instance will not have
  // any runner and so will not get used for execution.
  if (!CreateCorrelationIDControl(config)) {
    is_initialized->set_value(false);
    return;
  }

  // Create a scheduler thread associated with 'batcher_idx' that
  // executes the queued payloads.
  const int nice = GetCpuNiceLevel(config);
  scheduler_thread_.reset(new std::thread([this, nice, is_initialized]() {
    SchedulerThread(nice, is_initialized);
  }));
}

DirectSequenceBatch::~DirectSequenceBatch()
{
  // Signal the scheduler thread to exit...
  {
    std::unique_lock<std::mutex> lock(mu_);
    scheduler_thread_exit_ = true;
  }

  cv_.notify_one();

  // It is possible for the scheduler thread to be the last holder of
  // a backend object, and when that scheduler thread releases the
  // object the scheduler thread itself will destroy this
  // SequenceBatch object. So we need to check to make sure the
  // scheduler thread does not join it against itself and instead
  // detach it so there is not a problem when its thread object is
  // destroyed.
  if (scheduler_thread_->get_id() != std::this_thread::get_id()) {
    scheduler_thread_->join();
  } else {
    scheduler_thread_->detach();
  }
}

void
DirectSequenceBatch::Enqueue(
    const uint32_t seq_slot, const CorrelationID correlation_id,
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
    // request available in one or more sequence slots.
    if ((max_active_seq_slot_ == -1) && (request_provider != nullptr)) {
      null_request_header_ = request_provider->RequestHeader();
    }

    queues_[seq_slot].emplace_back(
        stats, request_provider, response_provider, OnComplete);

    seq_slot_correlation_ids_[seq_slot] = correlation_id;
    max_active_seq_slot_ =
        std::max(max_active_seq_slot_, static_cast<int32_t>(seq_slot));

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
DirectSequenceBatch::SchedulerThread(
    const int nice, std::promise<bool>* is_initialized)
{
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting Direct sequence-batch scheduler thread "
                   << batcher_idx_ << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting Direct sequence-batch scheduler thread "
                   << batcher_idx_ << " at default nice (requested nice "
                   << nice << " failed)...";
  }

  // Initialize using the thread. If error then just exit this thread
  // now... that means the corresponding model instance will not have
  // any runner and so will not get used for execution.
  Status startup_status = OnInit_(batcher_idx_);

  // Run warmup function if initialization succeed.
  if (startup_status.IsOk()) {
    startup_status = OnWarmup_(batcher_idx_);
  }
  if (!startup_status.IsOk()) {
    LOG_ERROR
        << "Initialization failed for Direct sequence-batch scheduler thread "
        << batcher_idx_ << ": " << startup_status.Message();
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

  // For testing this scheduler thread to be the last to release the
  // backend object.
  uint64_t backend_release_wait_milliseconds = 0;
  {
    const char* dstr = getenv("TRTSERVER_DELAY_SCHEDULER_BACKEND_RELEASE");
    if (dstr != nullptr) {
      backend_release_wait_milliseconds = atoi(dstr);
      LOG_INFO << "Delaying scheduler backend release for " << batcher_idx_
               << ": " << backend_release_wait_milliseconds << "ms";
    }
  }

  const uint64_t default_wait_microseconds = 500 * 1000;

  while (!scheduler_thread_exit_) {
    auto payloads = std::make_shared<std::vector<Scheduler::Payload>>();
    uint64_t wait_microseconds = 0;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);

      bool adjust_max_active_seq_slot = false;

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
        // handled. Find the largest sequence slot index that has a
        // payload available...
        int32_t max_seq_slot = max_active_seq_slot_;
        while ((max_seq_slot >= 0) && queues_[max_seq_slot].empty()) {
          max_seq_slot--;
        }

        if (max_seq_slot < 0) {
          wait_microseconds = default_wait_microseconds;
        } else {
          // Collect payloads from slot 0 to max_seq_slot.
          for (int32_t seq_slot = 0; seq_slot <= max_seq_slot; ++seq_slot) {
            bool end_of_sequence = false;
            bool use_null_provider = false;
            std::deque<Scheduler::Payload>& queue = queues_[seq_slot];

            // If 'seq_slot' doesn't have any requests then change the
            // request provider to send dummy/null input tensors for
            // this slot. We need this so that other payloads stay in
            // the correct slot.
            if (queue.empty()) {
              use_null_provider = true;
            } else {
              // If the payload has no request provider then the
              // sequence is being forcibly ended (e.g. because it has
              // been idle too long). Use a null provider for the
              // sequence slot since there isn't an actual payload but
              // also handle as if it were the end of the sequence.
              Scheduler::Payload& seq_slot_payload = queue.front();
              if (seq_slot_payload.request_provider_ == nullptr) {
                use_null_provider = true;
                end_of_sequence = true;
                queue.pop_front();
              }
            }

            // Use null-provider if necessary otherwise use the next
            // payload in the queue...
            if (use_null_provider) {
              auto null_request_provider =
                  std::make_shared<NULLInferRequestProvider>(
                      null_request_header_);
              null_request_provider->AddInputOverrides(
                  notready_input_overrides_);

              payloads->emplace_back(
                  nullptr, null_request_provider, nullptr, nullptr);
            } else {
              Scheduler::Payload& seq_slot_payload = queue.front();
              const auto& request_provider = seq_slot_payload.request_provider_;
              const auto& request_header = request_provider->RequestHeader();

              // Set the control tensor values in the request provider.
              SetControlTensors(
                  request_header, request_provider, seq_slot,
                  seq_slot_correlation_ids_[seq_slot]);

              payloads->emplace_back(
                  seq_slot_payload.stats_, request_provider,
                  seq_slot_payload.response_provider_,
                  seq_slot_payload.complete_function_);

              queue.pop_front();

              if ((request_header.flags() &
                   InferRequestHeader::FLAG_SEQUENCE_END) != 0) {
                end_of_sequence = true;
              }
            }

            // If the sequence has ended then attempt to refill the
            // sequence slot with a sequence from the backlog. If
            // there is no backlog show that the slot is no longer
            // active, and if it is currently the maximum active slot
            // note that we need to adjust 'max_active_seq_slot_' once
            // all slots are processed (we defer processing because
            // multiple slots could have ending sequences).
            if (end_of_sequence) {
              LOG_VERBOSE(1)
                  << "End sequence CORRID "
                  << seq_slot_correlation_ids_[seq_slot] << " in batcher "
                  << batcher_idx_ << ", slot " << seq_slot;

              // Should never be anything in a queue after the END
              // marker. If it happens that means we will clobber
              // that request if/when we swap in a backlog sequence
              // in ReleaseSequenceSlot below.
              if (!queue.empty()) {
                LOG_ERROR << "internal: unexpected requests after sequence "
                             "end in slot "
                          << seq_slot;
              }

              SequenceBatchScheduler::BatcherSequenceSlot batcher_seq_slot(
                  batcher_idx_, seq_slot);
              bool released =
                  base_->ReleaseSequenceSlot(batcher_seq_slot, &queue);
              if (released) {
                seq_slot_correlation_ids_[seq_slot] = 0;
                if (seq_slot == max_active_seq_slot_) {
                  adjust_max_active_seq_slot = true;
                }
              }
            }
          }
        }
      }

      // If one or more sequences ended, and one of them was in
      // 'max_active_seq_slot_', then need to find the new
      // 'max_active_seq_slot_'.
      if (adjust_max_active_seq_slot) {
        while ((max_active_seq_slot_ >= 0) &&
               (seq_slot_correlation_ids_[max_active_seq_slot_] == 0)) {
          max_active_seq_slot_--;
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
#ifdef TRTIS_ENABLE_STATS
        bool found_success = false;
#endif  // TRTIS_ENABLE_STATS
        for (auto& payload : *payloads) {
          const Status& final_status = status.IsOk() ? payload.status_ : status;

#ifdef TRTIS_ENABLE_STATS
          // All the payloads executed together, so count 1 execution
          // in the first successful payload. Other payloads stay at 0
          // executions.
          if (!found_success && final_status.IsOk() &&
              (payload.stats_ != nullptr)) {
            payload.stats_->SetModelExecutionCount(1);
            found_success = true;
          }
#endif  // TRTIS_ENABLE_STATS

          if (payload.complete_function_ != nullptr) {
            payload.complete_function_(final_status);
          }
        }
      };

      // Run the backend...
      OnSchedule_(batcher_idx_, payloads.get(), OnCompleteQueuedPayloads);

      // For testing we introduce a delay here to make the
      // "SequenceBatchScheduler destroyed by this thread" case
      // described in the comment below reproducible.
      if (backend_release_wait_milliseconds > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(backend_release_wait_milliseconds));
      }
    }

    // At the end of this scope 'payloads' will be destroyed.  A
    // handle to the backend is held through the
    // payload.complete_function_. If the server is exiting or the
    // backend is unloaded, it could be that this handle is the last
    // one for the backend and so destroying 'payloads' will cause the
    // backend to be deleted which in turn will call this thread's
    // SequenceBatchScheduler to be destroyed by this thread
    // itself. In that case it is important that this thread not
    // reference the object after this point since the object will be
    // invalid. The while statement above uses a local atomic which is
    // set to false by the destructor (and so the while look will
    // exit) and the logging below uses only local variables... so
    // this code is ok.
  }  // end runner loop

  LOG_VERBOSE(1) << "Stopping Direct sequence-batch scheduler thread "
                 << batcher_idx_ << "...";
}

OldestSequenceBatch::OldestSequenceBatch(
    SequenceBatchScheduler* base, const uint32_t batcher_idx,
    const size_t seq_slot_cnt, const ModelConfig& config,
    Scheduler::StandardInitFunc OnInit, Scheduler::StandardWarmupFunc OnWarmup,
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
    std::promise<bool>* is_initialized)
    : SequenceBatch(
          base, batcher_idx, seq_slot_cnt, start_input_overrides,
          end_input_overrides, startend_input_overrides,
          continue_input_overrides, notready_input_overrides),
      in_flight_(seq_slot_cnt, false), queues_(seq_slot_cnt)
{
  // Initialize to handle CORRID control. If error just exit
  // now... that means the corresponding model instance will not have
  // any runner and so will not get used for execution.
  if (!CreateCorrelationIDControl(config)) {
    is_initialized->set_value(false);
    return;
  }

  // Create a dynamic batcher use to batch together sequences for
  // inference.
  std::set<int32_t> preferred_batch_sizes;
  for (const auto size :
       config.sequence_batching().oldest().preferred_batch_size()) {
    preferred_batch_sizes.insert(size);
  }

  Status status = DynamicBatchScheduler::Create(
      batcher_idx_, 1 /* runner_cnt */, GetCpuNiceLevel(config), OnInit,
      OnWarmup, OnSchedule, true /* dynamic_batching_enabled */,
      false /* enforce_equal_shape_batch */, true /* preserve_ordering */,
      preferred_batch_sizes,
      config.sequence_batching().oldest().max_queue_delay_microseconds(),
      &dynamic_batcher_);
  if (!status.IsOk()) {
    LOG_ERROR << "failed creating dynamic sequence batcher for OldestFirst "
              << batcher_idx_ << ": " << status.Message();
    is_initialized->set_value(false);
    return;
  }

  is_initialized->set_value(true);
}

OldestSequenceBatch::~OldestSequenceBatch() {}

void
OldestSequenceBatch::CompleteAndNext(
    const uint32_t seq_slot, std::function<void(const Status&)> OnComplete,
    const Status& status)
{
  // If there is a completion function, call it to communicate the
  // completion of the sequence's inference request.
  if (OnComplete != nullptr) {
    OnComplete(status);
  }

  std::lock_guard<std::mutex> lock(mu_);

  // We may enqueue 1 or more pending inferences triggered by the
  // completion. If the sequence has a pending inference then it needs
  // to be send to dynamic batcher since the "previous" inference just
  // completed. If this next inference ends up being the end of the
  // sequence (either from the END flag or because the sequence is
  // being force-ended) then we try to fill the now-free sequence slot
  // from the backlog and then send the first inference from that
  // sequence to the dynamic batcher...
  std::deque<Scheduler::Payload>& queue = queues_[seq_slot];
  bool retry = true;
  while (retry) {
    retry = false;

    bool release_seq_slot = false;
    in_flight_[seq_slot] = false;

    // If the next sequence inference is ready in the queue then enqueue
    // it in the dynamic batcher now.
    if (!queue.empty()) {
      Scheduler::Payload& payload = queue.front();
      const auto& request_provider = payload.request_provider_;

      // If the request provider is null then this inference request is
      // from the reaper thread indicating a timed-out sequence. Mark
      // that the sequence slot should be released but otherwise do
      // nothing.
      if (request_provider == nullptr) {
        LOG_VERBOSE(1) << "force-end sequence in batcher " << batcher_idx_
                       << ", slot " << seq_slot;
        release_seq_slot = true;
      } else {
        const auto& request_header = request_provider->RequestHeader();
        const CorrelationID correlation_id = request_header.correlation_id();

        // After handling the last inference in a sequence we must
        // release the sequence slot to make it available to another
        // sequence.
        if ((request_header.flags() & InferRequestHeader::FLAG_SEQUENCE_END) !=
            0) {
          LOG_VERBOSE(1) << "end sequence CORRID " << correlation_id
                         << " in batcher " << batcher_idx_ << ", slot "
                         << seq_slot;
          release_seq_slot = true;
        }

        // Add the appropriate control tensor values to the request.
        SetControlTensors(
            request_header, request_provider, seq_slot, correlation_id);

        LOG_VERBOSE(1) << "issue to dynamic batcher CORRID " << correlation_id
                       << " in batcher " << batcher_idx_ << ", slot "
                       << seq_slot;
        in_flight_[seq_slot] = true;

        auto on_complete_fn = std::bind(
            &OldestSequenceBatch::CompleteAndNext, this, seq_slot,
            payload.complete_function_, std::placeholders::_1);
        dynamic_batcher_->Enqueue(
            payload.stats_, request_provider, payload.response_provider_,
            on_complete_fn);
      }

      queue.pop_front();
    }

    // If releasing the sequence slot then the sequence queue should be
    // empty and we can now assign a new sequence to the queue (from the
    // backlog).
    if (release_seq_slot) {
      // Should never be anything in a queue after the END marker. If it
      // happens that means we will clobber that request if/when we swap
      // in a backlog sequence in ReleaseSequenceSlot below.
      if (!queue.empty()) {
        LOG_ERROR << "internal: unexpected requests after sequence end in slot "
                  << seq_slot;
      }

      SequenceBatchScheduler::BatcherSequenceSlot batcher_seq_slot(
          batcher_idx_, seq_slot);
      const bool released =
          base_->ReleaseSequenceSlot(batcher_seq_slot, &queue);
      if (!released) {
        LOG_VERBOSE(1) << "Enqueued new sequence containing " << queue.size()
                       << " requests into OldestFirst batcher " << batcher_idx_
                       << ", slot " << seq_slot;

        // If an inference is already in-flight in the dynamic batcher
        // in this sequence slot then can't process the new queue
        // inferences right now, because the in-flight request is
        // using slot resources like the CORRID override map.
        if (!in_flight_[seq_slot]) {
          retry = true;
        }
      }
    }
  }
}

void
OldestSequenceBatch::Enqueue(
    const uint32_t seq_slot, const CorrelationID correlation_id,
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(const Status&)> OnComplete)
{
  // Queue the new request... if there isn't already a request in
  // flight then send one to the dynamic batcher immediately.
  bool in_flight;
  {
    std::lock_guard<std::mutex> lock(mu_);

    std::deque<Scheduler::Payload>& queue = queues_[seq_slot];
    queue.emplace_back(stats, request_provider, response_provider, OnComplete);
    in_flight = in_flight_[seq_slot];
  }

  if (!in_flight) {
    CompleteAndNext(seq_slot, nullptr, Status::Success);
  }
}

}}  // namespace nvidia::inferenceserver
