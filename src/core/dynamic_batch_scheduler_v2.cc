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

#include "src/core/dynamic_batch_scheduler_v2.h"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/nvtx.h"
#include "src/core/server.h"

namespace nvidia { namespace inferenceserver {

DynamicBatchSchedulerV2::DynamicBatchSchedulerV2(
    const void* triton_model, const StandardSchedFuncV2& OnSchedule,
    const bool dynamic_batching_enabled, const int32_t max_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering,
    const std::set<int32_t>& preferred_batch_sizes,
    const uint64_t max_queue_delay_microseconds,
    const inference::ModelQueuePolicy& default_queue_policy,
    const uint32_t priority_levels, const ModelQueuePolicyMap& queue_policy_map)
    : triton_model_((const TritonModel*)triton_model), OnSchedule_(OnSchedule),
      dynamic_batching_enabled_(dynamic_batching_enabled),
      queue_(default_queue_policy, priority_levels, queue_policy_map),
      staged_queue_(default_queue_policy, 0, queue_policy_map), delay_cnt_(0),
      max_batch_size_((size_t)std::max(1, max_batch_size)),
      preferred_batch_sizes_(preferred_batch_sizes),
      pending_batch_delay_ns_(max_queue_delay_microseconds * 1000),
      pending_batch_size_(0), staged_batch_size_(0), queued_batch_size_(0),
      next_preferred_batch_size_(0),
      enforce_equal_shape_tensors_(enforce_equal_shape_tensors),
      preserve_ordering_(preserve_ordering), signal_exit_(false)
{
  max_preferred_batch_size_ = 0;
  for (const auto size : preferred_batch_sizes_) {
    max_preferred_batch_size_ =
        std::max(max_preferred_batch_size_, (size_t)size);
  }

  rate_limiter_ = triton_model_->Server()->GetRateLimiter();
}

Status
DynamicBatchSchedulerV2::Create(
    const void* triton_model, const int nice,
    const StandardSchedFuncV2& OnSchedule, const bool dynamic_batching_enabled,
    const int32_t max_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering,
    const std::set<int32_t>& preferred_batch_sizes,
    const uint64_t max_queue_delay_microseconds,
    std::unique_ptr<Scheduler>* scheduler)
{
  inference::ModelDynamicBatching batcher_config;
  batcher_config.set_preserve_ordering(preserve_ordering);
  for (const auto& bs : preferred_batch_sizes) {
    batcher_config.add_preferred_batch_size(bs);
  }
  batcher_config.set_max_queue_delay_microseconds(max_queue_delay_microseconds);

  return Create(
      triton_model, nice, OnSchedule, dynamic_batching_enabled, max_batch_size,
      enforce_equal_shape_tensors, batcher_config, scheduler);
}

Status
DynamicBatchSchedulerV2::Create(
    const void* triton_model, const int nice,
    const StandardSchedFuncV2& OnSchedule, const bool dynamic_batching_enabled,
    const int32_t max_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const inference::ModelDynamicBatching& batcher_config,
    std::unique_ptr<Scheduler>* scheduler)
{
  std::set<int32_t> preferred_batch_sizes;
  for (const auto size : batcher_config.preferred_batch_size()) {
    preferred_batch_sizes.insert(size);
  }

  DynamicBatchSchedulerV2* dyna_sched = new DynamicBatchSchedulerV2(
      triton_model, OnSchedule, dynamic_batching_enabled, max_batch_size,
      enforce_equal_shape_tensors, batcher_config.preserve_ordering(),
      preferred_batch_sizes, batcher_config.max_queue_delay_microseconds(),
      batcher_config.default_queue_policy(), batcher_config.priority_levels(),
      batcher_config.priority_queue_policy());
  std::unique_ptr<DynamicBatchSchedulerV2> sched(dyna_sched);

  std::promise<bool> init_state;
  sched->sched_thread_.reset(new std::thread([dyna_sched, nice, &init_state]() {
    dyna_sched->SchedulerThread(nice, &init_state);
  }));
  if (!init_state.get_future().get()) {
    if (sched->sched_thread_->joinable()) {
      sched->sched_thread_->join();
    }
  }

  // Check if the Scheduler thread is initialized properly

  // For debugging/testing, delay start of threads until the queue
  // contains the specified number of entries.
  sched->delay_cnt_ = 0;
  {
    const char* dstr = getenv("TRITONSERVER_DELAY_SCHEDULER");
    if (dstr != nullptr) {
      sched->delay_cnt_ = atoi(dstr);
      LOG_VERBOSE(1) << "Delaying scheduler thread for model "
                     << sched->triton_model_->Name() << "  until "
                     << sched->delay_cnt_ << " queued requests...";
    }
  }

  scheduler->reset(sched.release());

  return Status::Success;
}

DynamicBatchSchedulerV2::~DynamicBatchSchedulerV2()
{
  signal_exit_ = true;
  // Signal the scheduler threads to exit and then wait for them...
  {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.notify_all();
  }
  if (sched_thread_->joinable()) {
    sched_thread_->join();
  }
}

Status
DynamicBatchSchedulerV2::Enqueue(std::unique_ptr<InferenceRequest>& request)
{
  // Queue timer starts at the beginning of the queueing and
  // scheduling process
  request->CaptureQueueStartNs();
  INFER_TRACE_ACTIVITY(
      request->Trace(), TRITONSERVER_TRACE_QUEUE_START,
      request->QueueStartNs());

  Status enqueue_status;
  bool wake_runner = false;
  {
    std::lock_guard<std::mutex> lock(mu_);

    queued_batch_size_ += std::max(1U, request->BatchSize());

    // Assuming no error is returned, this call takes ownership of
    // 'request' and so we can't use it after this point.
    RETURN_IF_ERROR(queue_.Enqueue(request->Priority(), request));

    // We may wake up runner less often if we don't enforce equal shape within
    // a batch, otherwise must always wake up runner to check it
    if (enforce_equal_shape_tensors_.empty()) {
      wake_runner &=
          ((staged_batch_size_ + queued_batch_size_) >=
           next_preferred_batch_size_);
    }
  }

  if (wake_runner) {
    cv_.notify_one();
  }

  return Status::Success;
}

void
DynamicBatchSchedulerV2::SchedulerThread(
    const int nice, std::promise<bool>* is_initialized)
{
  // I need to take the requests off the queue, communicate with the rate
  // limiter and in the end ask backend thread pool to execute the inference.

  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting dynamic-batch scheduler thread for model "
                   << triton_model_->Name() << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting dynamic-batch scheduler thread for model "
                   << triton_model_->Name()
                   << " at default nice (requested nice " << nice
                   << " failed)...";
  }

  // Initialize using the thread. If error then just exit this thread
  // now... that means the corresponding model instance will not have
  // any runner and so will not get used for execution.
  // Status startup_status = OnInit_();

  // Run warmup function if initialization succeed.
  // if (startup_status.IsOk()) {
  //  startup_status = OnWarmup_();
  //}

  // if (!startup_status.IsOk()) {
  if (false) {
    // LOG_ERROR << "Initialization failed for dynamic-batch scheduler thread
    // for model "
    //          << triton_model_->Name() << ": " << startup_status.Message();
    is_initialized->set_value(false);
    return;
  } else {
    is_initialized->set_value(true);
  }

  const uint64_t default_wait_microseconds = 500 * 1000;

  while (!signal_exit_) {
    NVTX_RANGE(nvtx_, "DynamicBatchSchedulerV2 " + runner_id);

    bool request_model_instance = false;
    uint64_t wait_microseconds = 0;

    if (delay_cnt_ > 0) {
      // Debugging/testing... wait until queue contains 'delay_cnt'
      // items...
      wait_microseconds = 10 * 1000;
      if (queue_.Size() >= delay_cnt_) {
        delay_cnt_ = 0;
      }
      LOG_VERBOSE(1) << "Delaying scheduler thread for model "
                     << triton_model_->Name() << " until " << delay_cnt_
                     << " queued requests, current total = " << queue_.Size();
    } else if (queue_.Empty()) {
      wait_microseconds = default_wait_microseconds;
    } else if (dynamic_batching_enabled_) {
      // TODO: Enable Dynamic Batching!!! The requests are deferred and not
      // necessarily executed right away. We need to optimize below so
      // that we don't requests for too many model_instances from the
      // rate_limiter. Intelligently mixing the two queues looks like the
      // way ahead.
  
      /*
      // Use dynamic batching to get request(s) to execute.
      wait_microseconds = GetDynamicBatch();
      // Get requests that are rejected from searching dynamic batch.
      queue_.ReleaseRejectedRequests(&rejected_requests);

      // Extract batch only if there is pending batch
      auto pending_batch_queue_cnt = queue_.PendingBatchCount();
      if ((wait_microseconds == 0) && (pending_batch_queue_cnt != 0)) {
        for (size_t idx = 0; idx < pending_batch_queue_cnt; ++idx) {
          std::unique_ptr<InferenceRequest> request;
          auto status = queue_.Dequeue(&request);
          if (status.IsOk()) {
            std::lock_guard<std::mutex> lock(staged_queue_mtx_);
            staged_batch_size_ += std::max(1U, request->BatchSize());
            staged_queue_.Enqueue(0, std::move(request));
          } else {
            // The queue is empty which conflicts with pending batch count.
            // Send the current batch if any and reset related variables.
            LOG_ERROR << "Failed to retrieve request from scheduler queue: "
                      << status.Message();
            queue_.ResetCursor();
            queued_batch_size_ = 0;
            pending_batch_size_ = 0;
            break;
          }
        }

        queued_batch_size_ -= pending_batch_size_;
        // Set next preferred to be 0 so that enqueue thread will wake up
        // runners when new request arrives. In the case where the queue
        // becomes empty, this helps the runners to set up proper wait time
        // instead of waiting for the default timer or actual next preferred
        // batch size is reached.
        next_preferred_batch_size_ = 0;

        pending_batch_size_ = 0;
        required_equal_inputs_.clear();
      }
      */
    } else {
      // No batching... stage next request
      std::unique_ptr<InferenceRequest> request;
      auto status = queue_.Dequeue(&request);
      if (status.IsOk()) {
        request_model_instance = true;
        std::lock_guard<std::mutex> lock(staged_queue_mtx_);
        staged_queue_.Enqueue(0, request);
      } else {
        LOG_ERROR << "Failed to retrieve request from scheduler queue: "
                  << status.Message();
      }
    }

    // If no requests are to be handled, wait for notification or
    // for the specified timeout before checking the queue again.
    if (wait_microseconds > 0) {
      std::chrono::microseconds wait_timeout(wait_microseconds);
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait_for(lock, wait_timeout);
    }

    if (request_model_instance) {
      auto sched_cb = [this](RateLimiter::ModelInstance* instance) {
        std::vector<std::unique_ptr<InferenceRequest>> requests;
        std::shared_ptr<
            std::vector<std::deque<std::unique_ptr<InferenceRequest>>>>
            rejected_requests;

        if (dynamic_batching_enabled_) {
          // TODO: See above comment regarding the dynamic batching. Need
          // to recheck for any new requests in queue/staged_queue that
          // will allow to form better batches..
          /*
                    // Use dynamic batching to get request(s) to execute.
                    wait_microseconds = GetDynamicBatch();

                    // Get requests that are rejected from searching dynamic
             batch. queue_.ReleaseRejectedRequests(&rejected_requests);

                    // Extract batch only if there is pending batch
                    auto pending_batch_queue_cnt = queue_.PendingBatchCount();
                    if ((wait_microseconds == 0) && (pending_batch_queue_cnt !=
             0)) { requests.reserve(pending_batch_queue_cnt); for (size_t idx =
             0; idx < pending_batch_queue_cnt; ++idx) {
                        std::unique_ptr<InferenceRequest> request;
                        auto status = queue_.Dequeue(&request);
                        if (status.IsOk()) {
                          requests.emplace_back(std::move(request));
                        } else {
                          // The queue is empty which conflicts with pending
             batch count.
                          // Send the current batch if any and reset related
             variables. LOG_ERROR << "Failed to retrieve request from scheduler
             queue: "
                                    << status.Message();
                          queue_.ResetCursor();
                          queued_batch_size_ = 0;
                          pending_batch_size_ = 0;
                          break;
                        }
                      }
                      if (preserve_ordering_ && !requests.empty()) {
                        std::lock_guard<std::mutex> lock(completion_queue_mtx_);
                        for (auto& request : requests) {
                          completion_queue_.emplace_back();
                          auto queue_slot = &completion_queue_.back();
                          request->SetResponseDelegator(
                              [this, queue_slot](
                                  std::unique_ptr<InferenceResponse>&& response,
                                  const uint32_t flags) {
                                {
                                  std::lock_guard<std::mutex>
             lock(completion_queue_mtx_);
                                  queue_slot->emplace_back(std::move(response),
             flags);
                                }
                                FinalizeResponses();
                              });
                        }
                      }

                      queued_batch_size_ -= pending_batch_size_;
                      // Set next preferred to be 0 so that enqueue thread will
             wake up
                      // runners when new request arrives. In the case where the
             queue
                      // becomes empty, this helps the runners to set up proper
             wait time
                      // instead of waiting for the default timer or actual next
             preferred
                      // batch size is reached.
                      next_preferred_batch_size_ = 0;

                      pending_batch_size_ = 0;
                      required_equal_inputs_.clear();
                    }
          */
        } else {
          std::unique_ptr<InferenceRequest> request;
          auto status = staged_queue_.Dequeue(&request);
          if (status.IsOk()) {
            requests.emplace_back(std::move(request));
            if (preserve_ordering_) {
              std::lock_guard<std::mutex> lock(completion_queue_mtx_);
              for (auto& request : requests) {
                completion_queue_.emplace_back();
                auto queue_slot = &completion_queue_.back();
                request->SetResponseDelegator(
                    [this, queue_slot](
                        std::unique_ptr<InferenceResponse>&& response,
                        const uint32_t flags) {
                      {
                        std::lock_guard<std::mutex> lock(completion_queue_mtx_);
                        queue_slot->emplace_back(std::move(response), flags);
                      }
                      FinalizeResponses();
                    });
              }
            }
          } else {
            LOG_ERROR << "Failed to retrieve request from scheduler queue: "
                      << status.Message();
          }
        }
        auto completion_cb = [instance]() {
          instance->Release(true /* executed */);
        };
        if (!requests.empty()) {
          OnSchedule_(
              const_cast<TritonModelInstance*>(instance->RawInstance()),
              std::move(requests), completion_cb);
        } else {
          instance->Release(false /* executed */);
        }
      };
      rate_limiter_->RequestModelInstance(sched_cb, triton_model_);
    }

/*
    // Finish rejected requests if any
    if (rejected_requests != nullptr) {
      static Status rejected_status =
          Status(Status::Code::UNAVAILABLE, "Request timeout expired");
      for (auto& rejected_queue : *rejected_requests) {
        for (auto& rejected_request : rejected_queue) {
          InferenceRequest::RespondIfError(
              rejected_request, rejected_status, true);
        }
      }
    }
*/    
  }

  LOG_VERBOSE(1) << "Stopping dynamic-batch scheduler thread for model"
                 << triton_model_->Name() << "...";
}

uint64_t
DynamicBatchSchedulerV2::GetDynamicBatch(PriorityQueue& queue)
{
  // 'queue_mtx_' mutex must be held when this function is called. queue_
  // must not be empty.

  // Examine the new requests. If adding these new requests to the
  // pending batch allows a preferred batch size then execute it
  // immediately. Stop examining requests if the maximum preferred
  // batch size would be exceeded or if the shape of the next request
  // does not match the shape of the pending batch.
  bool send_now = false;
  if (!queue.IsCursorValid()) {
    queue.ResetCursor();
    pending_batch_size_ = 0;
  }
  size_t best_preferred_batch_size = 0;
  queued_batch_size_ -= queue_.ApplyPolicyAtCursor();
  while (!queue_.CursorEnd()) {
    const auto batch_size = std::max(1U, queue_.RequestAtCursor()->BatchSize());

    // If there is no pending batch, then this request is starting a
    // new batch.
    if ((staged_batch_size_ + queue_.PendingBatchCount()) == 0) {
      // Get the shape of the new batch that is being started...
      if (!enforce_equal_shape_tensors_.empty()) {
        if (!InitRequiredEqualInputs(
                 queue_.RequestAtCursor(), enforce_equal_shape_tensors_,
                 &required_equal_inputs_)
                 .IsOk()) {
          send_now = true;
          break;
        }
      }
    } else {
      // There is a pending batch and adding this request would make
      // the batch size larger than all of the preferred batch sizes,
      // so mark the cursor at this point. Not sending the pending batch so that
      // we can examine the queue delay of requests that fits in a batch.
      if (((pending_batch_size_ + batch_size) > max_preferred_batch_size_) &&
          (best_preferred_batch_size == 0)) {
        best_preferred_batch_size = pending_batch_size_;
        queue_.MarkCursor();
      }
      if ((pending_batch_size_ + batch_size) > max_batch_size_) {
        send_now = true;
        break;
      }

      // There is a pending batch and it has a different shape then
      // this request, so send the pending batch as it is.
      if (!enforce_equal_shape_tensors_.empty() &&
          !CompareWithRequiredEqualInputs(
              queue_.RequestAtCursor(), required_equal_inputs_)) {
        send_now = true;
        break;
      }
    }

    pending_batch_size_ += batch_size;
    queue_.AdvanceCursor();
    queued_batch_size_ -= queue_.ApplyPolicyAtCursor();

    if (preferred_batch_sizes_.find(pending_batch_size_) !=
        preferred_batch_sizes_.end()) {
      best_preferred_batch_size = pending_batch_size_;
      queue_.MarkCursor();
    }
  }

  // Obatin the age of the oldest pending request to compare with the maximum
  // batch queuing delay
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  uint64_t now_ns = TIMESPEC_TO_NANOS(now);
  uint64_t delay_ns = now_ns - queue_.OldestEnqueueTime();
  bool delay_is_exceeded = (delay_ns >= pending_batch_delay_ns_);

  // If we found a preferred batch size and the queue delay hasn't been
  // exceeded, then execute that.
  if ((best_preferred_batch_size != 0) && !delay_is_exceeded) {
    pending_batch_size_ = best_preferred_batch_size;
    queue_.SetCursorToMark();
    return 0;
  }

  // No request in pending batch happens when all queued requests have expired
  // timeout and the policies are REJECT
  if (queue_.PendingBatchCount() == 0) {
    return 0;
  }

  // If the delay has been exceeded, or if the current batch can't grow
  // any larger then just immediately execute whatever is pending.
  if (send_now || delay_is_exceeded ||
      (pending_batch_size_ >= max_preferred_batch_size_)) {
    return 0;
  }

  // Set the next preferred batch size given the pending batch size
  auto next_preferred_batch_size_it =
      preferred_batch_sizes_.upper_bound(pending_batch_size_);
  if (next_preferred_batch_size_it != preferred_batch_sizes_.end()) {
    next_preferred_batch_size_ = *next_preferred_batch_size_it;
  } else {
    next_preferred_batch_size_ =
        preferred_batch_sizes_.empty() ? 0 : *preferred_batch_sizes_.begin();
  }

  uint64_t wait_ns = pending_batch_delay_ns_ - delay_ns;
  // Note that taking request timeout into consideration allows us to reset
  // pending batch as soon as it is invalidated. But the cost is that in edge
  // case where the timeout will be expired one by one, the thread will be
  // waken frequently.
  if (queue_.ClosestTimeout() != 0) {
    if (now_ns <= queue_.ClosestTimeout()) {
      wait_ns = std::min(queue_.ClosestTimeout() - now_ns, wait_ns);
    } else {
      // A request in pending batch is timed-out, wait for 1 us to force the
      // thread to reset the pending batch right the way.
      wait_ns = 1000;
    }
  }

  // Return non-zero wait microseconds to cause this thread to wait
  // until the queue delay or the closest timeout has expired.
  // Another thread may be awaken due to incoming request to handle the pending
  // batch before this thread wakes and that is ok. But if no other request
  // comes in then this thread will wake and revisit the pending batch
  // (and at that time will then see the delay has been exceeded and will send
  // the batch).
  return wait_ns / 1000;
}

void
DynamicBatchSchedulerV2::FinalizeResponses()
{
  // Need exclusive access of the function to ensure responses are sent
  // in order
  static std::mutex finalize_mtx;
  std::lock_guard<std::mutex> lock(finalize_mtx);
  // Finalize the completed payloads in-order as far as possible
  std::deque<std::pair<std::unique_ptr<InferenceResponse>, const uint32_t>>
      responses;
  {
    std::lock_guard<std::mutex> queue_lock(completion_queue_mtx_);
    while (!completion_queue_.empty() && !completion_queue_.front().empty()) {
      bool response_complete = false;
      for (auto& response_pair : completion_queue_.front()) {
        // Assuming FINAL flag is set only in the last response of the request
        response_complete =
            ((response_pair.second & TRITONSERVER_RESPONSE_COMPLETE_FINAL) !=
             0);
        responses.emplace_back(std::move(response_pair));
      }
      if (response_complete) {
        completion_queue_.pop_front();
      } else {
        completion_queue_.front().clear();
      }
    }
  }

  for (auto& response : responses) {
    InferenceResponse::Send(std::move(response.first), response.second);
  }
}

}}  // namespace nvidia::inferenceserver
