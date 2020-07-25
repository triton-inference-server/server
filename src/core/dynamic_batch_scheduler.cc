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

#include "src/core/dynamic_batch_scheduler.h"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/nvtx.h"

namespace nvidia { namespace inferenceserver {

DynamicBatchScheduler::DynamicBatchScheduler(
    const uint32_t runner_id_start, const uint32_t runner_cnt,
    const StandardInitFunc& OnInit, const StandardWarmupFunc& OnWarmup,
    const StandardRunFunc& OnSchedule, const bool dynamic_batching_enabled,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering,
    const std::set<int32_t>& preferred_batch_sizes,
    const uint64_t max_queue_delay_microseconds,
    const ModelQueuePolicy& default_queue_policy,
    const uint32_t priority_levels, const ModelQueuePolicyMap& queue_policy_map)
    : OnInit_(OnInit), OnWarmup_(OnWarmup), OnSchedule_(OnSchedule),
      dynamic_batching_enabled_(dynamic_batching_enabled),
      scheduler_thread_cnt_(runner_cnt), idle_scheduler_thread_cnt_(0),
      queue_(default_queue_policy, priority_levels, queue_policy_map),
      preferred_batch_sizes_(preferred_batch_sizes),
      pending_batch_delay_ns_(max_queue_delay_microseconds * 1000),
      pending_batch_size_(0), queued_batch_size_(0),
      next_preferred_batch_size_(0),
      enforce_equal_shape_tensors_(enforce_equal_shape_tensors),
      preserve_ordering_(preserve_ordering)
{
  max_preferred_batch_size_ = 0;
  for (const auto size : preferred_batch_sizes_) {
    max_preferred_batch_size_ =
        std::max(max_preferred_batch_size_, (size_t)size);
  }
}

Status
DynamicBatchScheduler::Create(
    const uint32_t runner_id_start, const uint32_t runner_cnt, const int nice,
    const StandardInitFunc& OnInit, const StandardWarmupFunc& OnWarmup,
    const StandardRunFunc& OnSchedule, const bool dynamic_batching_enabled,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering,
    const std::set<int32_t>& preferred_batch_sizes,
    const uint64_t max_queue_delay_microseconds,
    std::unique_ptr<Scheduler>* scheduler)
{
  return Create(
      runner_id_start, runner_cnt, nice, OnInit, OnWarmup, OnSchedule,
      dynamic_batching_enabled, enforce_equal_shape_tensors, preserve_ordering,
      preferred_batch_sizes, max_queue_delay_microseconds, ModelQueuePolicy(),
      0, ModelQueuePolicyMap(), scheduler);
}

Status
DynamicBatchScheduler::Create(
    const uint32_t runner_id_start, const uint32_t runner_cnt, const int nice,
    const StandardInitFunc& OnInit, const StandardWarmupFunc& OnWarmup,
    const StandardRunFunc& OnSchedule, const bool dynamic_batching_enabled,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering,
    const std::set<int32_t>& preferred_batch_sizes,
    const uint64_t max_queue_delay_microseconds,
    const ModelQueuePolicy& default_queue_policy,
    const uint32_t priority_levels, const ModelQueuePolicyMap& queue_policy_map,
    std::unique_ptr<Scheduler>* scheduler)
{
  DynamicBatchScheduler* dyna_sched = new DynamicBatchScheduler(
      runner_id_start, runner_cnt, OnInit, OnWarmup, OnSchedule,
      dynamic_batching_enabled, enforce_equal_shape_tensors, preserve_ordering,
      preferred_batch_sizes, max_queue_delay_microseconds, default_queue_policy,
      priority_levels, queue_policy_map);
  std::unique_ptr<DynamicBatchScheduler> sched(dyna_sched);

  // Create one scheduler thread for each requested runner. Associate
  // each scheduler thread with a runner.
  for (uint32_t c = 0; c < sched->scheduler_thread_cnt_; ++c) {
    const uint32_t runner_id = runner_id_start + c;
    std::promise<bool> init_state;
    auto thread_exit = std::make_shared<std::atomic<bool>>(false);
    sched->scheduler_threads_exit_.emplace_back(thread_exit);
    sched->scheduler_threads_.emplace_back(new std::thread(
        [dyna_sched, runner_id, nice, thread_exit, &init_state]() {
          dyna_sched->SchedulerThread(
              runner_id, nice, thread_exit, &init_state);
        }));
    if (!init_state.get_future().get()) {
      if (sched->scheduler_threads_.back()->joinable()) {
        sched->scheduler_threads_.back()->join();
      }
      sched->scheduler_threads_exit_.pop_back();
      sched->scheduler_threads_.pop_back();
    }
  }

  if (sched->scheduler_threads_.empty()) {
    return Status(
        Status::Code::INTERNAL,
        "Initialization failed for all dynamic-batch scheduler threads");
  }

  scheduler->reset(sched.release());

  return Status::Success;
}

DynamicBatchScheduler::~DynamicBatchScheduler()
{
  // Signal the scheduler threads to exit and then wait for them...
  {
    std::unique_lock<std::mutex> lock(mu_);
    for (auto& ex : scheduler_threads_exit_) {
      ex->store(true);
    }

    cv_.notify_all();
  }

  // It is possible for (one of) the scheduler threads to be the last
  // holder of a backend object, and when that scheduler thread
  // releases the object the scheduler thread itself will destroy the
  // DynamicBatchScheduler object. So we need to check for a scheduler
  // thread and not join it against itself. Instead we detach it so
  // there is not a problem when its thread object is destroyed.
  for (auto& thd : scheduler_threads_) {
    if (thd->get_id() != std::this_thread::get_id()) {
      if (thd->joinable()) {
        thd->join();
      }
    } else {
      thd->detach();
    }
  }
}

Status
DynamicBatchScheduler::Enqueue(std::unique_ptr<InferenceRequest>& request)
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

    // If there are any idle runners and the queued batch size is greater or
    // equal to next preferred batch size, then wake one up to service this
    // request. We do the actual wake outside of the lock to avoid having the
    // woken thread immediately block on the lock
    wake_runner = (idle_scheduler_thread_cnt_ > 0);

    // We may wake up runner less often if we don't enforce equal shape within
    // a batch, otherwise must always wake up runner to check it
    if (enforce_equal_shape_tensors_.empty()) {
      wake_runner &= (queued_batch_size_ >= next_preferred_batch_size_);
    }
  }

  if (wake_runner) {
    cv_.notify_one();
  }

  return Status::Success;
}

void
DynamicBatchScheduler::SchedulerThread(
    const uint32_t runner_id, const int nice,
    const std::shared_ptr<std::atomic<bool>>& rthread_exit,
    std::promise<bool>* is_initialized)
{
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting dynamic-batch scheduler thread " << runner_id
                   << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting dynamic-batch scheduler thread " << runner_id
                   << " at default nice (requested nice " << nice
                   << " failed)...";
  }

  // Initialize using the thread. If error then just exit this thread
  // now... that means the corresponding model instance will not have
  // any runner and so will not get used for execution.
  Status startup_status = OnInit_(runner_id);

  // Run warmup function if initialization succeed.
  if (startup_status.IsOk()) {
    startup_status = OnWarmup_(runner_id);
  }

  if (!startup_status.IsOk()) {
    LOG_ERROR << "Initialization failed for dynamic-batch scheduler thread "
              << runner_id << ": " << startup_status.Message();
    is_initialized->set_value(false);
    return;
  } else {
    is_initialized->set_value(true);
  }

  // For testing this scheduler thread to be the last to release the
  // backend object.
  uint64_t backend_release_wait_milliseconds = 0;
  {
    const char* dstr = getenv("TRITONSERVER_DELAY_SCHEDULER_BACKEND_RELEASE");
    if (dstr != nullptr) {
      backend_release_wait_milliseconds = atoi(dstr);
      LOG_VERBOSE(1) << "Delaying scheduler backend release for " << runner_id
                     << ": " << backend_release_wait_milliseconds << "ms";
    }
  }

  // For debugging/testing, delay start of threads until the queue
  // contains the specified number of entries.
  size_t delay_cnt = 0;
  {
    const char* dstr = getenv("TRITONSERVER_DELAY_SCHEDULER");
    if (dstr != nullptr) {
      delay_cnt = atoi(dstr);
      LOG_VERBOSE(1) << "Delaying scheduler thread " << runner_id << " until "
                     << delay_cnt << " queued requests...";
    }
  }

  // Make a local copy of the atomic used to signal the thread to
  // exit. See comment at end of function for explanation.
  std::shared_ptr<std::atomic<bool>> thread_exit = rthread_exit;

  const uint64_t default_wait_microseconds = 500 * 1000;

  while (!thread_exit->load()) {
    NVTX_RANGE(nvtx_, "DynamicBatchScheduler " + runner_id);

    std::vector<std::unique_ptr<InferenceRequest>> requests;
    std::shared_ptr<std::vector<std::deque<std::unique_ptr<InferenceRequest>>>>
        rejected_requests;
    bool wake_thread = false;
    uint64_t wait_microseconds = 0;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (delay_cnt > 0) {
        // Debugging/testing... wait until queue contains 'delay_cnt'
        // items...
        wait_microseconds = 10 * 1000;
        if (queue_.Size() >= delay_cnt) {
          delay_cnt = 0;
        }
        LOG_VERBOSE(1) << "Delaying scheduler thread " << runner_id << " until "
                       << delay_cnt
                       << " queued requests, current total = " << queue_.Size();
      } else if (queue_.Empty()) {
        wait_microseconds = default_wait_microseconds;
      } else if (dynamic_batching_enabled_) {
        // Use dynamic batching to get request(s) to execute.
        wait_microseconds = GetDynamicBatch(runner_id);

        // Get requests that are rejected from searching dynamic batch.
        queue_.ReleaseRejectedRequests(&rejected_requests);

        // Extract batch only if there is pending batch
        auto pending_batch_queue_cnt = queue_.PendingBatchCount();
        if ((wait_microseconds == 0) && (pending_batch_queue_cnt != 0)) {
          requests.reserve(pending_batch_queue_cnt);
          for (size_t idx = 0; idx < pending_batch_queue_cnt; ++idx) {
            std::unique_ptr<InferenceRequest> request;
            auto status = queue_.Dequeue(&request);
            if (status.IsOk()) {
              requests.emplace_back(std::move(request));
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
                      std::lock_guard<std::mutex> lock(completion_queue_mtx_);
                      queue_slot->emplace_back(std::move(response), flags);
                    }
                    FinalizeResponses();
                  });
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

          // If there are still requests in the queue after removing
          // the pending batch and if there are any idle threads then
          // wake one up to service the requests remaining in the
          // queue. We need this special wake logic for the dynamic
          // batching case because we may delay handling requests in
          // the queue and so idle the threads that would normally be
          // handling those requests. We do the actual wake outside of
          // the lock to avoid having the woken thread immediately
          // block on the lock.
          wake_thread = !queue_.Empty() && (idle_scheduler_thread_cnt_ > 0);
        }
      } else {
        // No batching... execute next request
        std::unique_ptr<InferenceRequest> request;
        auto status = queue_.Dequeue(&request);
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

      // If no requests are to be handled, wait for notification or
      // for the specified timeout before checking the queue again.
      if (wait_microseconds > 0) {
        idle_scheduler_thread_cnt_++;
        std::chrono::microseconds wait_timeout(wait_microseconds);
        cv_.wait_for(lock, wait_timeout);
        idle_scheduler_thread_cnt_--;
      }
    }

    if (wake_thread) {
      cv_.notify_one();
    }

    if (!requests.empty()) {
      OnSchedule_(runner_id, std::move(requests));

      // For testing we introduce a delay here to make the
      // "DynamicBatchScheduler destroyed by this thread" case
      // described in the comment below reproducible.
      if (backend_release_wait_milliseconds > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(backend_release_wait_milliseconds));
      }
    }

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

    // FIXME, this isn't really true anymore so needs to be revisited.
    //
    // At the end of this scope 'requests' will be destroyed.  A
    // handle to the backend is held by the request. If the server is
    // exiting or the backend is unloaded, it could be that this
    // handle is the last one for the backend and so destroying
    // 'requests' will cause the backend to be deleted which in turn
    // will call this thread's DynamicBatchScheduler to be destroyed
    // by this thread itself. In that case it is important that this
    // thread not reference the object after this point since the
    // object will be invalid. The while statement above uses a local
    // atomic which is set to false by the destructor (and so the
    // while loop will exit) and the logging below uses only local
    // variables... so this code is ok.
  }  // end runner loop

  LOG_VERBOSE(1) << "Stopping dynamic-batch scheduler thread " << runner_id
                 << "...";
}

uint64_t
DynamicBatchScheduler::GetDynamicBatch(const int64_t runner_id)
{
  // 'mu_' mutex must be held when this function is called. queue_
  // must not be empty.

  // Examine the new requests. If adding these new requests to the
  // pending batch allows a preferred batch size then execute it
  // immediately. Stop examining requests if the maximum preferred
  // batch size would be exceeded or if the shape of the next request
  // does not match the shape of the pending batch.
  bool send_now = false;
  if (!queue_.IsCursorValid()) {
    queue_.ResetCursor();
    pending_batch_size_ = 0;
  }
  size_t best_preferred_batch_size = 0;
  queued_batch_size_ -= queue_.ApplyPolicyAtCursor();
  while (!queue_.CursorEnd()) {
    const auto batch_size = std::max(1U, queue_.RequestAtCursor()->BatchSize());

    // If there is no pending batch, then this request is starting a
    // new batch.
    if (queue_.PendingBatchCount() == 0) {
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
      // the batch size too large, so send the pending batch as it is.
      if ((pending_batch_size_ + batch_size) > max_preferred_batch_size_) {
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

  // If we found a preferred batch size then execute that.
  if (best_preferred_batch_size != 0) {
    pending_batch_size_ = best_preferred_batch_size;
    queue_.SetCursorToMark();
    return 0;
  }

  // No request in pending batch happens when all queued requests have expired
  // timeout and the policies are REJECT
  if (queue_.PendingBatchCount() == 0) {
    return 0;
  }

  // If there is no batch queuing delay or if the current batch can't
  // grow any larger then just immediately execute whatever is
  // pending.
  if (send_now || (pending_batch_delay_ns_ == 0) ||
      (pending_batch_size_ >= max_preferred_batch_size_)) {
    return 0;
  }

  // Compare the age of the oldest pending request to the maximum
  // batch queuing delay and execute now if queuing delay is
  // exceeded. If queuing delay not exceeded create a timer to wakeup
  // a thread to check again at the maximum allowed delay.
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  uint64_t now_ns = TIMESPEC_TO_NANOS(now);
  uint64_t delay_ns = now_ns - queue_.OldestEnqueueTime();

  if (delay_ns >= pending_batch_delay_ns_) {
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
DynamicBatchScheduler::FinalizeResponses()
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
