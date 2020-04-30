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
#pragma once

#include <deque>
#include <unordered_map>
#include "src/core/infer_stats.h"
#include "src/core/model_config.h"
#include "src/core/scheduler.h"

namespace nvidia { namespace inferenceserver {

using RequiredEqualInputs = std::unordered_map<
    std::string,
    std::pair<const InferenceRequest::Input*, bool /* compare contents */>>;

Status InitRequiredEqualInputs(
    const std::unique_ptr<InferenceRequest>& request,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    RequiredEqualInputs* required_equal_inputs);

bool CompareWithRequiredEqualInputs(
    const std::unique_ptr<InferenceRequest>& request,
    const RequiredEqualInputs& required_equal_inputs);

//
// PriorityQueue
//
using ModelQueuePolicyMap =
    ::google::protobuf::Map<::google::protobuf::uint32, ModelQueuePolicy>;

class PriorityQueue {
 public:
  // Construct a queue with no priority level with default queue policy,
  // which will behave the same as regular queue.
  PriorityQueue();

  // Construct a queue with 'priority_levels', the priority starts from 1.
  // Different priority level may follow different queue policies given by
  // 'queue_policy_map', otherwise, the 'default_queue_policy' will be used.
  PriorityQueue(
      const ModelQueuePolicy& default_queue_policy, uint32_t priority_levels,
      const ModelQueuePolicyMap queue_policy_map);

  // Enqueue a request with priority set to 'priority_level'. If
  // Status::Success is returned then the queue has taken ownership of
  // the request object and so 'request' will be nullptr. If
  // non-success is returned then the caller still retains ownership
  // of 'request'.
  Status Enqueue(
      uint32_t priority_level, std::unique_ptr<InferenceRequest>& request);

  // Dequeue the request at the front of the queue.
  Status Dequeue(std::unique_ptr<InferenceRequest>* request);

  // Retrieve the requests that are rejected based on the queue policies.
  void ReleaseRejectedRequests(
      std::shared_ptr<
          std::vector<std::deque<std::unique_ptr<InferenceRequest>>>>*
          requests);

  // Return the number of requests in the queue, rejected requests are
  // not included.
  size_t Size() { return size_; }

  // Is the queue is empty? Rejected requests are not included.
  bool Empty() { return Size() == 0; }

  // Reset the cursor such that it is representing an empty pending batch.
  void ResetCursor() { pending_cursor_ = Cursor(queues_.begin()); }

  // Record the current cursor. The cursor can be restored to recorded state
  // by invoking SetCursorToMark(). Note that Enqueue(), Dequeue(), and
  // ResetCursor() will invalidate the marker, it is the function caller's
  // responsibility to ensure the marker is valid before calling
  // SetCursorToMark().
  void MarkCursor() { current_mark_ = pending_cursor_; }

  // Apply the queue policy and alter the underlying queue accordingly. After
  // the function returns, the cursor may be at its end to indicate that
  // there no request after the pending batch.
  // Returns the total batch size of the newly rejected requests.
  size_t ApplyPolicyAtCursor();

  // Return the request at the cursor.
  const std::unique_ptr<InferenceRequest>& RequestAtCursor()
  {
    return pending_cursor_.curr_it_->second.At(pending_cursor_.queue_idx_);
  }

  // Advance the cursor for pending batch. This function will not trigger the
  // queue policy. No effect if the cursor already reach the end of the queue.
  void AdvanceCursor();

  // Whether the cursor reaches its end,
  bool CursorEnd() { return pending_cursor_.pending_batch_count_ == size_; }

  // Restore the cursor state to the marker.
  void SetCursorToMark() { pending_cursor_ = current_mark_; }

  // Whether the cursor is still valid. The cursor is valid only if the pending
  // batch is unchanged.
  bool IsCursorValid();

  // Return the oldest queued time of requests in pending batch.
  uint64_t OldestEnqueueTime()
  {
    return pending_cursor_.pending_batch_oldest_enqueue_time_ns_;
  }

  // Return the closest timeout of requests in pending batch.
  uint64_t ClosestTimeout()
  {
    return pending_cursor_.pending_batch_closest_timeout_ns_;
  }

  // Return the number of requests in pending batch.
  size_t PendingBatchCount() { return pending_cursor_.pending_batch_count_; }

 private:
  class PolicyQueue {
   public:
    // Construct a policy queue with default policy, which will behave the same
    // as regular queue.
    PolicyQueue()
        : timeout_action_(ModelQueuePolicy::REJECT), default_timeout_us_(0),
          allow_timeout_override_(false), max_queue_size_(0)
    {
    }

    // Construct a policy queue with given 'policy'.
    PolicyQueue(const ModelQueuePolicy& policy)
        : timeout_action_(policy.timeout_action()),
          default_timeout_us_(policy.default_timeout_microseconds()),
          allow_timeout_override_(policy.allow_timeout_override()),
          max_queue_size_(policy.max_queue_size())
    {
    }

    // Enqueue a request and set up its timeout accordingly. If
    // Status::Success is returned then the queue has taken ownership
    // of the request object and so 'request' will be nullptr. If
    // non-success is returned then the caller still retains ownership
    // of 'request'.
    Status Enqueue(std::unique_ptr<InferenceRequest>& request);

    // Dequeue the request at the front of the queue.
    Status Dequeue(std::unique_ptr<InferenceRequest>* request);

    // Apply the queue policy to the request at 'idx'.
    // 'rejected_count' will be incremented by the number of the newly rejected
    // requets after applying the policy.
    // 'rejected_batch_size' will be incremented by the total batch size of the
    // newly rejected requests after applying the policy.
    // Return true if the 'idx' still points to a request after applying the
    // policy, false otherwise.
    bool ApplyPolicy(
        size_t idx, size_t* rejected_count, size_t* rejected_batch_size);

    // Return the rejected requests held by the queue.
    void ReleaseRejectedQueue(
        std::deque<std::unique_ptr<InferenceRequest>>* requests);

    // Return the request at 'idx'.
    const std::unique_ptr<InferenceRequest>& At(size_t idx) const;

    // Return the timeout timestamp of the request at 'idx', in ns. A value of 0
    // indicates that the request doesn't specify a timeout.
    uint64_t TimeoutAt(size_t idx);

    // Return whether the queue is empty, rejected requests are not included.
    bool Empty() { return Size() == 0; }

    // Return the number of requests in the queue, rejected requests are not
    // included.
    size_t Size() { return queue_.size() + delayed_queue_.size(); }

    // Return the number of unexpired requests in the queue
    size_t UnexpiredSize() { return queue_.size(); }

   private:
    // Variables that define the policy for the queue
    const ModelQueuePolicy::TimeoutAction timeout_action_;
    const uint64_t default_timeout_us_;
    const bool allow_timeout_override_;
    const uint32_t max_queue_size_;

    std::deque<uint64_t> timeout_timestamp_ns_;
    std::deque<std::unique_ptr<InferenceRequest>> queue_;
    std::deque<std::unique_ptr<InferenceRequest>> delayed_queue_;
    std::deque<std::unique_ptr<InferenceRequest>> rejected_queue_;
  };
  using PriorityQueues = std::map<uint32_t, PolicyQueue>;

  // Cursor for tracking pending batch, the cursor points to the item after
  // the pending batch.
  struct Cursor {
    Cursor() = default;
    Cursor(PriorityQueues::iterator start_it);

    Cursor(const Cursor& rhs)
        : curr_it_(rhs.curr_it_), queue_idx_(rhs.queue_idx_),
          at_delayed_queue_(rhs.at_delayed_queue_),
          pending_batch_closest_timeout_ns_(
              rhs.pending_batch_closest_timeout_ns_),
          pending_batch_oldest_enqueue_time_ns_(
              rhs.pending_batch_oldest_enqueue_time_ns_),
          pending_batch_count_(rhs.pending_batch_count_), valid_(rhs.valid_)
    {
    }

    PriorityQueues::iterator curr_it_;
    size_t queue_idx_;
    bool at_delayed_queue_;
    uint64_t pending_batch_closest_timeout_ns_;
    uint64_t pending_batch_oldest_enqueue_time_ns_;
    size_t pending_batch_count_;
    bool valid_;
  };

  PriorityQueues queues_;
  size_t size_;

  // Keep track of the priority level that the first request in the queue
  // is at to avoid traversing 'queues_'
  uint32_t front_priority_level_;
  uint32_t last_priority_level_;

  Cursor pending_cursor_;
  Cursor current_mark_;
};

}}  // namespace nvidia::inferenceserver
