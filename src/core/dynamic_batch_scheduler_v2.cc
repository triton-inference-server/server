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
    const void* triton_model, const uint32_t runner_id_start,
    const uint32_t runner_cnt, const StandardInitFunc& OnInit,
    const StandardWarmupFunc& OnWarmup, const StandardRunFunc& OnSchedule,
    const bool dynamic_batching_enabled, const int32_t max_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering,
    const std::set<int32_t>& preferred_batch_sizes,
    const uint64_t max_queue_delay_microseconds,
    const inference::ModelQueuePolicy& default_queue_policy,
    const uint32_t priority_levels, const ModelQueuePolicyMap& queue_policy_map)
    : triton_model_((const TritonModel*)triton_model),
      runner_id_start_(runner_id_start), runner_cnt_(runner_cnt),
      OnInit_(OnInit), OnWarmup_(OnWarmup), OnSchedule_(OnSchedule),
      dynamic_batching_enabled_(dynamic_batching_enabled),
      scheduler_thread_cnt_(runner_cnt), idle_scheduler_thread_cnt_(0),
      queue_(default_queue_policy, priority_levels, queue_policy_map),
      max_batch_size_((size_t)std::max(1, max_batch_size)),
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

  rate_limiter_ = triton_model_->Server()->GetRateLimiter();
}

Status
DynamicBatchSchedulerV2::Create(
    const void* triton_model, const uint32_t runner_id_start,
    const uint32_t runner_cnt, const int nice, const StandardInitFunc& OnInit,
    const StandardWarmupFunc& OnWarmup, const StandardRunFunc& OnSchedule,
    const bool dynamic_batching_enabled, const int32_t max_batch_size,
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
      triton_model, runner_id_start, runner_cnt, nice, OnInit, OnWarmup,
      OnSchedule, dynamic_batching_enabled, max_batch_size,
      enforce_equal_shape_tensors, batcher_config, scheduler);
}

Status
DynamicBatchSchedulerV2::Create(
    const void* triton_model, const uint32_t runner_id_start,
    const uint32_t runner_cnt, const int nice, const StandardInitFunc& OnInit,
    const StandardWarmupFunc& OnWarmup, const StandardRunFunc& OnSchedule,
    const bool dynamic_batching_enabled, const int32_t max_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const inference::ModelDynamicBatching& batcher_config,
    std::unique_ptr<Scheduler>* scheduler)
{
  std::set<int32_t> preferred_batch_sizes;
  for (const auto size : batcher_config.preferred_batch_size()) {
    preferred_batch_sizes.insert(size);
  }

  DynamicBatchSchedulerV2* dyna_sched = new DynamicBatchSchedulerV2(
      triton_model, runner_id_start, runner_cnt, OnInit, OnWarmup, OnSchedule,
      dynamic_batching_enabled, max_batch_size, enforce_equal_shape_tensors,
      batcher_config.preserve_ordering(), preferred_batch_sizes,
      batcher_config.max_queue_delay_microseconds(),
      batcher_config.default_queue_policy(), batcher_config.priority_levels(),
      batcher_config.priority_queue_policy());
  std::unique_ptr<DynamicBatchSchedulerV2> sched(dyna_sched);

  // Create one scheduler thread for each requested runner. Associate
  // each scheduler thread with a runner.
  for (uint32_t c = 0; c < sched->scheduler_thread_cnt_; ++c) {
    const uint32_t runner_id = runner_id_start + c;
    std::promise<bool> init_state;
    auto thread_exit = std::make_shared<std::atomic<bool>>(false);
    auto thread_context = std::make_shared<SchedulerThreadContext>(thread_exit);
    thread_context->thread_.reset(new std::thread(
        [dyna_sched, runner_id, nice, thread_context, &init_state]() {
          dyna_sched->SchedulerThread(
              runner_id, nice, thread_context, &init_state);
        }));
    sched->sched_thread_contexts_.push_back(thread_context);
    if (!init_state.get_future().get()) {
      if (sched->sched_thread_contexts_.back()->thread_->joinable()) {
        sched->sched_thread_contexts_.back()->thread_->join();
      }
      sched->sched_thread_contexts_.pop_back();
    }
  }

  if (sched->sched_thread_contexts_.empty()) {
    return Status(
        Status::Code::INTERNAL,
        "Initialization failed for all dynamic-batch scheduler threads");
  }

  // For debugging/testing, delay start of threads until the queue
  // contains the specified number of entries.
  sched->delay_cnt_ = 0;
  {
    const char* dstr = getenv("TRITONSERVER_DELAY_SCHEDULER");
    if (dstr != nullptr) {
      sched->delay_cnt_ = atoi(dstr);
      LOG_VERBOSE(1) << "Delaying scheduler thread [ " << runner_id_start
                     << " - " << (runner_id_start + runner_cnt - 1)
                     << " ] until " << sched->delay_cnt_
                     << " queued requests...";
    }
  }

  scheduler->reset(sched.release());

  return Status::Success;
}

DynamicBatchSchedulerV2::~DynamicBatchSchedulerV2()
{
  // Signal the scheduler threads to exit and then wait for them...
  {
    // std::unique_lock<std::mutex> lock(queue_mtx_);
    uint32_t count = 0;
    for (auto& context : sched_thread_contexts_) {
      context->exit_->store(true);
      context->ready_cv_.notify_all();

      // It is possible for (one of) the scheduler threads to be the last
      // holder of a backend object, and when that scheduler thread
      // releases the object the scheduler thread itself will destroy the
      // DynamicBatchSchedulerV2 object. So we need to check for a scheduler
      // thread and not join it against itself. Instead we detach it so
      // there is not a problem when its thread object is destroyed.
      if (context->thread_->get_id() != std::this_thread::get_id()) {
        if (context->thread_->joinable()) {
          context->thread_->join();
        }
      } else {
        context->thread_->detach();
      }
      count++;
    }
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

  {
    std::lock_guard<std::mutex> lock(queue_mtx_);

    queued_batch_size_ += std::max(1U, request->BatchSize());

    // Assuming no error is returned, this call takes ownership of
    // 'request' and so we can't use it after this point.
    RETURN_IF_ERROR(queue_.Enqueue(request->Priority(), request));

    if (delay_cnt_ > 0) {
      if (queue_.Size() >= delay_cnt_) {
        delay_cnt_ = 0;
      } else {
        LOG_VERBOSE(1) << "Delaying scheduler threads [ " << runner_id_start_
                       << " - " << (runner_id_start_ + runner_cnt_ - 1)
                       << " ] until " << delay_cnt_
                       << " queued requests, current total = " << queue_.Size();
        return Status::Success;
      }
    }
  }

  auto callback_fn = [this](RateLimiter::ModelInstance* instance) {
    NotificationFunction(instance);
  };

  rate_limiter_->RequestModelInstance(callback_fn, triton_model_);

  return Status::Success;
}

void
DynamicBatchSchedulerV2::NotificationFunction(
    RateLimiter::ModelInstance* instance)
{
  if (dynamic_batching_enabled_ &&
      (!rate_limiter_->IgnoreResourcesAndPriority())) {
    PushInstanceToQueue(instance);
  }

  // Signal the scheduler thread to make progress
  if (instance->RawInstance()->Index() >= sched_thread_contexts_.size()) {
    LOG_ERROR << "Instance Index " << instance->RawInstance()->Index()
              << " should be less than " << sched_thread_contexts_.size();
  }

  // Note that we are notifying a specific scheduler thread for
  // a given instance index. This is done to benefit from the
  // locality of the thread with instance.
  // This requires a distinct runner thread for each instance.
  // TODO: When TensorRT gets migrated to new backend API, this
  // condition might not be true.
  auto& sched_thread_context =
      sched_thread_contexts_[instance->RawInstance()->Index()];
  sched_thread_context->allocated_instance_ = instance;
  sched_thread_context->ready_.store(true);
  sched_thread_context->ready_cv_.notify_all();
}

void
DynamicBatchSchedulerV2::SchedulerThread(
    const uint32_t runner_id, const int nice,
    const std::shared_ptr<SchedulerThreadContext>& rthread_context,
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

  // Make a local copy of the atomic used to signal the thread to
  // exit. See comment at end of function for explanation.
  std::shared_ptr<SchedulerThreadContext> thread_context = rthread_context;

  while (!thread_context->exit_->load()) {
    NVTX_RANGE(nvtx_, "DynamicBatchSchedulerV2 " + runner_id);

    // The thread will wait till rate limiter marks the thread
    // ready and allocates a model instance to run.
    if (!thread_context->ready_) {
      std::unique_lock<std::mutex> lock(thread_context->ready_mu_);
      thread_context->ready_cv_.wait(lock, [&thread_context]() {
        return (thread_context->ready_.load() || thread_context->exit_->load());
      });
    }

    // With dynamic batching enable the priority order across
    // the instances must be preserved. This serialization step
    // can be ignored if rate_limiter is configured to ignore the
    // resource and priority.
    if (dynamic_batching_enabled_ &&
        (!rate_limiter_->IgnoreResourcesAndPriority())) {
      std::unique_lock<std::mutex> lock(sync_mu_);
      sync_cv_.wait(lock, [this, &thread_context]() {
        return (ProceedOk(thread_context->allocated_instance_));
      });
    }

    std::vector<std::unique_ptr<InferenceRequest>> requests;
    std::shared_ptr<std::vector<std::deque<std::unique_ptr<InferenceRequest>>>>
        rejected_requests;
    uint64_t wait_microseconds = 0;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(queue_mtx_);
      if (queue_.Empty()) {
        // Release the allocated instance
        thread_context->ready_ = false;
        if (thread_context->allocated_instance_ != nullptr) {
          thread_context->allocated_instance_->Release(false /*executed*/);
          thread_context->allocated_instance_ = nullptr;
        }
        continue;
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
    }

    // If wait for notification or for the specified timeout before checking
    // the queue again.
    if (wait_microseconds > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(wait_microseconds));
      continue;
    }
    // If finally going to run then pop the instance from the queue
    // queue and give chance to other
    if (!rate_limiter_->IgnoreResourcesAndPriority()) {
      PopInstanceFromQueue();
      sync_cv_.notify_all();
    }

    if (!requests.empty()) {
      OnSchedule_(runner_id, std::move(requests));

      // For testing we introduce a delay here to make the
      // "DynamicBatchSchedulerV2 destroyed by this thread" case
      // described in the comment below reproducible.
      if (backend_release_wait_milliseconds > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(backend_release_wait_milliseconds));
      }

      // Release the allocated instance
      thread_context->ready_ = false;
      thread_context->allocated_instance_->Release(true /* executed */);
      thread_context->allocated_instance_ = nullptr;
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
    // will call this thread's DynamicBatchSchedulerV2 to be destroyed
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
DynamicBatchSchedulerV2::GetDynamicBatch(const int64_t runner_id)
{
  // 'queue_mtx_' mutex must be held when this function is called. queue_
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

bool
DynamicBatchSchedulerV2::ProceedOk(
    const RateLimiter::ModelInstance* model_instance)
{
  std::unique_lock<std::mutex> lock(alloc_instances_queue_mtx_);
  return (alloc_instances_queue_.front() == model_instance);
}

void
DynamicBatchSchedulerV2::PushInstanceToQueue(
    const RateLimiter::ModelInstance* model_instance)
{
  std::unique_lock<std::mutex> lock(alloc_instances_queue_mtx_);
  alloc_instances_queue_.push(model_instance);
}

void
DynamicBatchSchedulerV2::PopInstanceFromQueue()
{
  std::unique_lock<std::mutex> lock(alloc_instances_queue_mtx_);
  alloc_instances_queue_.pop();
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
