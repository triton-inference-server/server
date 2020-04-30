// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/scheduler_utils.h"

#include <cassert>
#include "src/core/constants.h"
#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

Status
InitRequiredEqualInputs(
    const std::unique_ptr<InferenceRequest>& request,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    RequiredEqualInputs* required_equal_inputs)
{
  required_equal_inputs->clear();

  for (const auto& pr : request->ImmutableInputs()) {
    const InferenceRequest::Input* input = pr.second;
    const auto itr = enforce_equal_shape_tensors.find(input->Name());
    if (itr != enforce_equal_shape_tensors.end()) {
      required_equal_inputs->emplace(
          std::piecewise_construct, std::forward_as_tuple(input->Name()),
          std::forward_as_tuple(input, itr->second));
    }
  }

  return Status::Success;
}

bool
CompareWithRequiredEqualInputs(
    const std::unique_ptr<InferenceRequest>& request,
    const RequiredEqualInputs& required_equal_inputs)
{
  for (const auto& pr : request->ImmutableInputs()) {
    const InferenceRequest::Input* input = pr.second;
    const auto itr = required_equal_inputs.find(input->Name());
    if (itr != required_equal_inputs.end()) {
      // Make sure shape of input tensors is equal.
      if (!CompareDims(itr->second.first->Shape(), input->Shape())) {
        return false;
      }

      // If necessary compare the contents as well...
      if (itr->second.second) {
        const auto& d1 = itr->second.first->Data();
        const auto& d2 = input->Data();

        // For now being conservative and assuming that content
        // comparison is for shape tensors which are likely to always
        // be in a single buffer.
        if ((d1->BufferCount() != 1) || (d2->BufferCount() != 1)) {
          return false;
        }

        size_t d1_byte_size, d2_byte_size;
        TRITONSERVER_MemoryType d1_memory_type, d2_memory_type;
        int64_t d1_memory_id, d2_memory_id;
        const char* d1_buffer = d1->BufferAt(
            0 /* idx */, &d1_byte_size, &d1_memory_type, &d1_memory_id);
        const char* d2_buffer = d2->BufferAt(
            0 /* idx */, &d2_byte_size, &d2_memory_type, &d2_memory_id);

        // Tensor must be same size and in in CPU memory so that it
        // can be easily compared. If not return false conservatively.
        if ((d1_byte_size != d2_byte_size) || (d1_buffer == nullptr) ||
            (d2_buffer == nullptr) ||
            (d1_memory_type == TRITONSERVER_MEMORY_GPU) ||
            (d2_memory_type == TRITONSERVER_MEMORY_GPU)) {
          return false;
        }

        if (strncmp(d1_buffer, d2_buffer, d1_byte_size) != 0) {
          return false;
        }
      }
    }
  }

  return true;
}

Status
PriorityQueue::PolicyQueue::Enqueue(std::unique_ptr<InferenceRequest>& request)
{
  if ((max_queue_size_ != 0) && (Size() >= max_queue_size_)) {
    return Status(Status::Code::UNAVAILABLE, "Exceeds maximum queue size");
  }

  queue_.emplace_back(std::move(request));
  auto timeout_us = default_timeout_us_;
  if (allow_timeout_override_) {
    auto override_timeout_us = queue_.back()->TimeoutMicroseconds();
    if (override_timeout_us != 0 && override_timeout_us < timeout_us) {
      timeout_us = override_timeout_us;
    }
  }
  if (timeout_us != 0) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    timeout_timestamp_ns_.emplace_back(
        TIMESPEC_TO_NANOS(now) + timeout_us * 1000);
  } else {
    timeout_timestamp_ns_.emplace_back(0);
  }

  return Status::Success;
}

Status
PriorityQueue::PolicyQueue::Dequeue(std::unique_ptr<InferenceRequest>* request)
{
  if (!queue_.empty()) {
    *request = std::move(queue_.front());
    queue_.pop_front();
    timeout_timestamp_ns_.pop_front();
  } else {
    *request = std::move(delayed_queue_.front());
    delayed_queue_.pop_front();
  }

  return Status::Success;
}

bool
PriorityQueue::PolicyQueue::ApplyPolicy(
    size_t idx, size_t* rejected_count, size_t* rejected_batch_size)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  auto now_nanoseconds = TIMESPEC_TO_NANOS(now);
  if (idx < queue_.size()) {
    size_t curr_idx = idx;
    while (curr_idx < queue_.size()) {
      if ((timeout_timestamp_ns_[curr_idx] != 0) &&
          (now_nanoseconds > timeout_timestamp_ns_[curr_idx])) {
        if (timeout_action_ == ModelQueuePolicy::DELAY) {
          delayed_queue_.emplace_back(std::move(queue_[curr_idx]));
        } else {
          rejected_queue_.emplace_back(std::move(queue_[curr_idx]));
          *rejected_count += 1;
          *rejected_batch_size += rejected_queue_.back()->BatchSize();
        }
        curr_idx++;
      } else {
        break;
      }
    }

    // Use range erasure on deque as all erasure functions are linear,
    // this implies in the edge case where this function is always called on
    // 'bad' index can be O(n^2). However, for data structures that are O(1)
    // erasure, the traversal may not be as efficient due to cache miss
    // (elements not stored contiguously).
    queue_.erase(queue_.begin() + idx, queue_.begin() + curr_idx);
    timeout_timestamp_ns_.erase(
        timeout_timestamp_ns_.begin() + idx,
        timeout_timestamp_ns_.begin() + curr_idx);

    // Current idx is pointing to an item with unexpired timeout
    if (idx < queue_.size()) {
      return true;
    }
  }
  // At this point, idx is pointing to an item with expired timeout.
  // If the item is in delayed queue, then return true. Otherwise, false
  // meaning the queue has no item with this 'idx'.
  return ((idx - queue_.size()) < delayed_queue_.size());
}

void
PriorityQueue::PolicyQueue::ReleaseRejectedQueue(
    std::deque<std::unique_ptr<InferenceRequest>>* requests)
{
  rejected_queue_.swap(*requests);
}

const std::unique_ptr<InferenceRequest>&
PriorityQueue::PolicyQueue::At(size_t idx) const
{
  if (idx < queue_.size()) {
    return queue_[idx];
  } else {
    return delayed_queue_[idx - queue_.size()];
  }
}

uint64_t
PriorityQueue::PolicyQueue::TimeoutAt(size_t idx)
{
  if (idx < queue_.size()) {
    return timeout_timestamp_ns_[idx];
  } else {
    return 0;
  }
}

PriorityQueue::PriorityQueue()
    : size_(0), front_priority_level_(0), last_priority_level_(0)
{
  ModelQueuePolicy default_policy;
  queues_.emplace(0, PolicyQueue(default_policy));
  front_priority_level_ = queues_.begin()->first;
  ResetCursor();
}

PriorityQueue::PriorityQueue(
    const ModelQueuePolicy& default_queue_policy, uint32_t priority_levels,
    const ModelQueuePolicyMap queue_policy_map)
    : size_(0), last_priority_level_(priority_levels)
{
  if (priority_levels == 0) {
    queues_.emplace(0, PolicyQueue(default_queue_policy));
  } else {
    for (uint32_t level = 1; level <= priority_levels; level++) {
      auto it = queue_policy_map.find(level);
      if (it == queue_policy_map.end()) {
        queues_.emplace(level, PolicyQueue(default_queue_policy));
      } else {
        queues_.emplace(level, PolicyQueue(it->second));
      }
    }
  }
  front_priority_level_ = queues_.begin()->first;
  ResetCursor();
}

Status
PriorityQueue::Enqueue(
    uint32_t priority_level, std::unique_ptr<InferenceRequest>& request)
{
  auto status = queues_[priority_level].Enqueue(request);
  if (status.IsOk()) {
    size_++;
    front_priority_level_ = std::min(front_priority_level_, priority_level);
    // Invalidate the pending batch cursor if the enqueued item is placed
    // within the pending batch. At the same priority level the request is
    // guaranteed to be after pending batch if the batch hasn't reached
    // delayed queue.
    if ((priority_level < pending_cursor_.curr_it_->first) ||
        ((priority_level == pending_cursor_.curr_it_->first) &&
         (pending_cursor_.at_delayed_queue_))) {
      pending_cursor_.valid_ = false;
    }
  }

  return status;
}

Status
PriorityQueue::Dequeue(std::unique_ptr<InferenceRequest>* request)
{
  pending_cursor_.valid_ = false;
  while (true) {
    if (!queues_[front_priority_level_].Empty()) {
      RETURN_IF_ERROR(queues_[front_priority_level_].Dequeue(request));
      size_--;
      return Status::Success;
    } else if (front_priority_level_ != last_priority_level_) {
      front_priority_level_++;
      continue;
    }

    // Control reach here if the queue for last priority level is also
    // empty, then return error below.
    break;
  }

  return Status(Status::Code::UNAVAILABLE, "dequeue on empty queue");
}

void
PriorityQueue::ReleaseRejectedRequests(
    std::shared_ptr<std::vector<std::deque<std::unique_ptr<InferenceRequest>>>>*
        requests)
{
  auto res = std::make_shared<
      std::vector<std::deque<std::unique_ptr<InferenceRequest>>>>(
      queues_.size());
  size_t idx = 0;
  for (auto& queue : queues_) {
    queue.second.ReleaseRejectedQueue(&((*res)[idx]));
    idx++;
  }

  requests->swap(res);
}

bool
PriorityQueue::IsCursorValid()
{
  if (pending_cursor_.valid_) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return TIMESPEC_TO_NANOS(now) <
           pending_cursor_.pending_batch_closest_timeout_ns_;
  }
  return false;
}

PriorityQueue::Cursor::Cursor(PriorityQueues::iterator start_it)
    : curr_it_(start_it), queue_idx_(0), at_delayed_queue_(false),
      pending_batch_closest_timeout_ns_(0),
      pending_batch_oldest_enqueue_time_ns_(0), pending_batch_count_(0),
      valid_(true)
{
}

size_t
PriorityQueue::ApplyPolicyAtCursor()
{
  size_t rejected_batch_size = 0;
  size_t rejected_count = 0;
  while (pending_cursor_.curr_it_ != queues_.end()) {
    if (!(pending_cursor_.curr_it_->second.ApplyPolicy(
            pending_cursor_.queue_idx_, &rejected_count,
            &rejected_batch_size))) {
      if (size_ > pending_cursor_.pending_batch_count_ + rejected_count) {
        pending_cursor_.curr_it_++;
        pending_cursor_.queue_idx_ = 0;
        continue;
      }
    }
    // Control reach here if the cursor points to a request that is candidate
    // for pending batch, or if all requests are in pending batch.
    break;
  }
  size_ -= rejected_count;
  return rejected_batch_size;
}

void
PriorityQueue::AdvanceCursor()
{
  if (pending_cursor_.pending_batch_count_ >= size_) {
    return;
  }

  const auto& timeout_ns =
      pending_cursor_.curr_it_->second.TimeoutAt(pending_cursor_.queue_idx_);
  if (timeout_ns != 0) {
    if (pending_cursor_.pending_batch_closest_timeout_ns_ != 0) {
      pending_cursor_.pending_batch_closest_timeout_ns_ = std::min(
          pending_cursor_.pending_batch_closest_timeout_ns_, timeout_ns);
    } else {
      pending_cursor_.pending_batch_closest_timeout_ns_ = timeout_ns;
    }
  }

  // FIXME stats
  uint64_t curr_enqueue_time_ns = 0; /* TIMESPEC_TO_NANOS(
      pending_cursor_.curr_it_->second.At(pending_cursor_.queue_idx_)
          .stats_->Timestamp(ModelInferStats::TimestampKind::kQueueStart)); */
  if (pending_cursor_.pending_batch_oldest_enqueue_time_ns_ != 0) {
    pending_cursor_.pending_batch_oldest_enqueue_time_ns_ = std::min(
        pending_cursor_.pending_batch_oldest_enqueue_time_ns_,
        curr_enqueue_time_ns);
  } else {
    pending_cursor_.pending_batch_oldest_enqueue_time_ns_ =
        curr_enqueue_time_ns;
  }
  ++pending_cursor_.queue_idx_;
  ++pending_cursor_.pending_batch_count_;
  // pending batch includes delayed request if (queue_idx_ - 1) points to
  // delayed queue.
  pending_cursor_.at_delayed_queue_ =
      (pending_cursor_.queue_idx_ >
       pending_cursor_.curr_it_->second.UnexpiredSize());
}

}}  // namespace nvidia::inferenceserver
