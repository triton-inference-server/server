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
#include "src/core/provider.h"
#include "src/core/server_status.h"
#include "src/core/utils.h"

namespace nvidia { namespace inferenceserver {

tensorflow::Status
SequenceBatchScheduler::Create(
    const ModelConfig& config, const uint32_t runner_cnt,
    StandardRunFunc OnSchedule, std::unique_ptr<Scheduler>* scheduler)
{
  std::unique_ptr<SequenceBatchScheduler> sched(new SequenceBatchScheduler());

  // Get the batch size to allow for each runner. This is at least 1
  // even if the model doesn't support batching.
  size_t batch_size = std::max(1, config.max_batch_size());

  // Based on the model configuration create input tensors for control
  // signals indicating sequence start, sequence continue, and
  // sequence not ready.
  std::shared_ptr<InferRequestProvider::InputOverrideMap> start;
  std::shared_ptr<InferRequestProvider::InputOverrideMap> cont;
  std::shared_ptr<InferRequestProvider::InputOverrideMap> notready;
  TF_RETURN_IF_ERROR(
      sched->CreateControlTensors(config, &start, &cont, &notready));

  // Create one SequenceBatch object for each requested runner. The
  // SequenceBatch object has a thread that manages the batch of
  // requests.
  for (uint32_t c = 0; c < runner_cnt; ++c) {
    std::shared_ptr<SequenceBatch> sb = std::make_shared<SequenceBatch>(
        c, batch_size, config, OnSchedule, start, cont, notready);
    sched->batches_.push_back(sb);
  }

  scheduler->reset(sched.release());

  return tensorflow::Status::OK();
}

tensorflow::Status
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
  int32_t false_value, true_value;

  // START
  {
    TF_RETURN_IF_ERROR(GetSequenceControlProperties(
        config.sequence_batching(), config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_START, &tensor_name,
        &false_value, &true_value));
    uint8_t* false_p = reinterpret_cast<uint8_t*>(&false_value);
    uint8_t* true_p = reinterpret_cast<uint8_t*>(&true_value);
    std::vector<uint8_t> false_vec(false_p, false_p + sizeof(false_value));
    std::vector<uint8_t> true_vec(true_p, true_p + sizeof(true_value));

    (*start_input_overrides)->insert(std::make_pair(tensor_name, true_vec));
    (*continue_input_overrides)->insert(std::make_pair(tensor_name, false_vec));
    (*notready_input_overrides)->insert(std::make_pair(tensor_name, false_vec));
  }

  // READY
  {
    TF_RETURN_IF_ERROR(GetSequenceControlProperties(
        config.sequence_batching(), config.name(),
        ModelSequenceBatching::Control::CONTROL_SEQUENCE_READY, &tensor_name,
        &false_value, &true_value));
    uint8_t* false_p = reinterpret_cast<uint8_t*>(&false_value);
    uint8_t* true_p = reinterpret_cast<uint8_t*>(&true_value);
    std::vector<uint8_t> false_vec(false_p, false_p + sizeof(false_value));
    std::vector<uint8_t> true_vec(true_p, true_p + sizeof(true_value));

    (*start_input_overrides)->insert(std::make_pair(tensor_name, true_vec));
    (*continue_input_overrides)->insert(std::make_pair(tensor_name, true_vec));
    (*notready_input_overrides)->insert(std::make_pair(tensor_name, false_vec));
  }

  return tensorflow::Status::OK();
}

void
SequenceBatchScheduler::Enqueue(
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(tensorflow::Status)> OnComplete)
{
  // Queue timer starts at the beginning of the queueing and scheduling process
  std::unique_ptr<ModelInferStats::ScopedTimer> queue_timer(
      new ModelInferStats::ScopedTimer());
  stats->StartQueueTimer(queue_timer.get());

  const auto& request_header = request_provider->RequestHeader();

  // For now the request must have batch-size 1 since the sequence
  // batcher does not yet support requests that statically batch.
  if (request_header.batch_size() != 1) {
    OnComplete(tensorflow::errors::InvalidArgument(
        "inference request to model '", request_provider->ModelName(),
        "' must specify batch-size 1 due to requirements of sequence batcher"));
    return;
  }

  // A request must have a correlation ID to be processed correctly by
  // this scheduler. A value of 0 (zero) indicates that the request
  // doesn't have a correlation ID.
  const CorrelationID correlation_id = request_header.correlation_id();
  if (correlation_id == 0) {
    OnComplete(tensorflow::errors::InvalidArgument(
        "inference request to model '", request_provider->ModelName(),
        "' must specify a non-zero correlation ID"));
    return;
  }

  SequenceTarget* target = nullptr;

  std::unique_lock<std::mutex> lock(mu_);

  // If the request's correlation_id is new, then attempt to find a
  // free slot to use for that ID. If one doesn't exist then put the
  // request onto the backlog queue where it must wait for a slot to
  // come free. If a free slot is found assign this and subsequent
  // requests with this correlation ID to that same
  // SequenceBatch+slot.
  auto sb_itr = sequence_to_target_map_.find(correlation_id);
  if (sb_itr == sequence_to_target_map_.end()) {
    bool found_slot = false;
    std::shared_ptr<SequenceBatch> isb;
    uint32_t islot;

    // Look through the slots of the batches in order to find the
    // first free slot. This method favors keeping the requests in the
    // minimum number of model instances and creating large
    // batches... another option would be to distribute the requests
    // across all the instances.
    for (const std::shared_ptr<SequenceBatch>& bsb : batches_) {
      found_slot = bsb->GetFreeSlot(&islot);
      if (found_slot) {
        isb = bsb;
        break;
      }
    }

    target = &sequence_to_target_map_[correlation_id];
    if (found_slot) {
      target->sequence_batch_ = isb;
      target->slot_ = islot;
    } else {
      backlog_sequence_ids_.push_back(correlation_id);
    }
  } else {
    // Correlation ID is known...
    target = &sb_itr->second;
  }

  // If correlation ID doesn't have a slot just add this request to
  // the backlog for that correlation ID.
  if (target->IsBacklog()) {
    target->backlog_.emplace_back(
        queue_timer, stats, request_provider, response_provider, OnComplete);
    return;
  }

  std::shared_ptr<SequenceBatch> sb = target->sequence_batch_;
  const uint32_t slot = target->slot_;

  sb->Enqueue(
      slot, correlation_id, queue_timer, stats, request_provider,
      response_provider, OnComplete);
}

SequenceBatchScheduler::SequenceBatch::SequenceBatch(
    const uint32_t runner_id, const size_t batch_size,
    const ModelConfig& config, StandardRunFunc OnSchedule,
    const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
        start_input_overrides,
    const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
        continue_input_overrides,
    const std::shared_ptr<InferRequestProvider::InputOverrideMap>&
        notready_input_overrides)
    : OnSchedule_(OnSchedule), scheduler_thread_exit_(false),
      scheduler_idle_(false), correlation_ids_(batch_size, 0),
      first_sequence_request_(batch_size, false), queues_(batch_size),
      max_active_slot_(-1), start_input_overrides_(start_input_overrides),
      continue_input_overrides_(continue_input_overrides),
      notready_input_overrides_(notready_input_overrides)
{
  // Create a scheduler thread associated with 'runner_id' that
  // executes the queued payloads.
  const int nice = GetCpuNiceLevel(config);
  scheduler_thread_.reset(new std::thread(
      [this, runner_id, nice]() { SchedulerThread(runner_id, nice); }));
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

bool
SequenceBatchScheduler::SequenceBatch::GetFreeSlot(uint32_t* slot)
{
  std::unique_lock<std::mutex> lock(mu_);

  // A slot is free if it doesn't have a correlation ID assigned to it
  // and there are no requests in the queue.
  for (size_t i = 0; i < queues_.size(); ++i) {
    if ((correlation_ids_[i] == 0) && queues_[i].empty()) {
      *slot = i;
      return true;
    }
  }

  return false;
}

void
SequenceBatchScheduler::SequenceBatch::Enqueue(
    const uint32_t slot, const CorrelationID correlation_id,
    std::unique_ptr<ModelInferStats::ScopedTimer>& queue_timer,
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(tensorflow::Status)> OnComplete)
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
    if (max_active_slot_ == -1) {
      null_request_header_ = request_provider->RequestHeader();
    }

    // Show that the slot has a new sequence if this is the first
    // request for the correlation ID. The runner thread will use this
    // to notify the backend that this is the start of a new sequence.
    if (correlation_ids_[slot] == 0) {
      first_sequence_request_[slot] = true;
    }

    correlation_ids_[slot] = correlation_id;
    queues_[slot].emplace_back(
        queue_timer, stats, request_provider, response_provider, OnComplete);
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
    const uint32_t runner_id, const int nice)
{
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting sequence-batch scheduler thread " << runner_id
                   << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting sequence-batch scheduler thread " << runner_id
                   << " at default nice (requested nice " << nice
                   << " failed)...";
  }

  // For debugging, delay start of thread until the queue contains the
  // specified number of entries.
  const char* dstr = getenv("TRTSERVER_DELAY_SCHEDULER");
  size_t delay_cnt = 0;
  if (dstr != nullptr) {
    delay_cnt = atoi(dstr);
    LOG_INFO << "Delaying scheduler thread " << runner_id << " until "
             << delay_cnt << " queued payloads...";
  }

  const uint64_t default_wait_microseconds = 500 * 1000;

  while (!scheduler_thread_exit_) {
    auto payloads = std::make_shared<std::vector<Scheduler::Payload>>();
    uint64_t wait_microseconds = 0;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (delay_cnt > 0) {
        wait_microseconds = 10 * 1000;
        // Debugging... wait until queues together contain at least
        // 'delay_cnt' items...
        size_t total_size = 0;
        for (const auto& q : queues_) {
          total_size += q.size();
        }
        if (total_size >= delay_cnt) {
          delay_cnt = 0;
        }
      } else {
        // Make sure there is at least one request that needs to be
        // handled.
        bool have_request = false;
        for (int32_t slot = 0; slot <= max_active_slot_; ++slot) {
          if (!queues_[slot].empty()) {
            have_request = true;
            break;
          }
        }

        if (!have_request) {
          wait_microseconds = default_wait_microseconds;
        } else {
          // Collect payloads from slot 0 to max_active_slot_.
          for (int32_t slot = 0; slot <= max_active_slot_; ++slot) {
            // If 'slot' is not active or doesn't have any requests
            // then change the request provider to send dummy/null
            // input tensors for this slot. We need this so that other
            // payloads stay in the correct slot.
            std::deque<SequencePayload>& queue = queues_[slot];

            if ((correlation_ids_[slot] == 0) || queue.empty()) {
              auto null_request_provider =
                  std::make_shared<NULLInferRequestProvider>(
                      null_request_header_);
              null_request_provider->SetInputContentOverride(
                  notready_input_overrides_);

              std::unique_ptr<ModelInferStats::ScopedTimer> queue_timer;
              payloads->emplace_back(
                  queue_timer, nullptr, null_request_provider, nullptr,
                  nullptr);
            } else {
              // If this is the first payload in a sequence then send
              // the appropriate sequence start indicator to the
              // backend.
              SequencePayload& slot_payload = queue.front();
              if (first_sequence_request_[slot]) {
                slot_payload.request_provider_->SetInputContentOverride(
                    start_input_overrides_);
                first_sequence_request_[slot] = false;
              } else {
                slot_payload.request_provider_->SetInputContentOverride(
                    continue_input_overrides_);
              }

              payloads->emplace_back(
                  slot_payload.queue_timer_, slot_payload.stats_,
                  slot_payload.request_provider_,
                  slot_payload.response_provider_,
                  slot_payload.complete_function_);

              queue.pop_front();
            }
          }
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
      auto OnCompleteQueuedPayloads = [payloads](tensorflow::Status status) {
        // Payloads that don't have a completion function don't have
        // anywhere to report their errors. Those errors could have
        // caused other payloads to have issues (due to mis-alignment
        // within the batch, etc.). So if any such payload has an
        // error we just fail all payloads.
        if (status.ok()) {
          for (auto& payload : *payloads) {
            if (payload.complete_function_ == nullptr) {
              const tensorflow::Status& no_complete_status =
                  payload.status_.ok() ? payload.compute_status_
                                       : payload.status_;
              if (!no_complete_status.ok()) {
                status = no_complete_status;
                break;
              }
            }
          }
        }

        bool found_success = false;
        for (auto& payload : *payloads) {
          const tensorflow::Status& final_status =
              status.ok() ? (payload.status_.ok() ? payload.compute_status_
                                                  : payload.status_)
                          : status;

          // All the payloads executed together, so count 1 execution in
          // the first successful payload. Other payloads stay at 0
          // executions.
          if (!found_success && final_status.ok() &&
              (payload.stats_ != nullptr)) {
            payload.stats_->SetModelExecutionCount(1);
            found_success = true;
          }

          if (payload.complete_function_ != nullptr) {
            payload.complete_function_(final_status);
          }
        }
      };

      OnSchedule_(runner_id, payloads.get(), OnCompleteQueuedPayloads);
    }
  }  // end runner loop

  LOG_VERBOSE(1) << "Stopping sequence-batch scheduler thread " << runner_id
                 << "...";
}

}}  // namespace nvidia::inferenceserver
