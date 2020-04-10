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

#include <condition_variable>
#include <functional>
#include <list>

namespace nvidia { namespace inferenceserver {

//
// Abstract class that acts like condition variable, which allows various
// types of strategies for inter-thread communication
//
template <class TritonMutex>
class CondVar {
 public:
  // Wait until 'pred' is evaluated to true. 'lock' will be unlocked during
  // the wait and it will be reaquired when the function returns.
  virtual void Wait(
      std::unique_lock<TritonMutex>& lock, std::function<bool()> pred) = 0;

  // Wait until 'rel_time_us' has passed or be notified.
  virtual void WaitFor(
      std::unique_lock<TritonMutex>& lock, uint64_t rel_time_us) = 0;

  // Notify all waiting threads to re-evaluate their predicates.
  virtual void NotifyAll() = 0;

  // Notify one of the waiting threads to re-evaluate its predicate.
  virtual void NotifyOne() = 0;
};

template <class TritonMutex>
class BusyWaitCondVar : public CondVar<TritonMutex> {
 public:
  BusyWaitCondVar() = default;

  void Wait(
      std::unique_lock<TritonMutex>& lock, std::function<bool()> pred) override
  {
    lock.unlock();
    while (true) {
      while (!pred()) {
        // busy-loop, which can be considered as always waking up spuriously
      }
      lock.lock();
      // must ensure the predicate still holds after acquiring the lock as
      // multiple waiting threads may get out of the busy-loop simultaneously.
      if (pred()) {
        break;
      }
      lock.unlock();
    }
  }

  void WaitFor(
      std::unique_lock<TritonMutex>& lock, uint64_t rel_time_us) override
  {
    lock.unlock();
    bool notified = false;
    signals_.push_back(&notified);
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t now_ns = TIMESPEC_TO_NANOS(now);
    uint64_t end_time_ns = now_ns + rel_time_us * 1000;
    while ((now_ns < end_time_ns) && (!notified)) {
      // busy-loop
      clock_gettime(CLOCK_MONOTONIC, &now);
      now_ns = TIMESPEC_TO_NANOS(now);
    }
    lock.lock();
  }

  void NotifyAll() override
  {
    while (!signals_.empty()) {
      *(signals_.front()) = true;
      signals_.pop_front();
    }
  }
  void NotifyOne() override
  {
    if (!signals_.empty()) {
      *(signals_.front()) = true;
      signals_.pop_front();
    }
  }

 private:
  std::list<bool*> signals_;
};

template <class TritonMutex>
class StdCondVar : public CondVar<TritonMutex> {
 public:
  StdCondVar() = default;

  void Wait(
      std::unique_lock<TritonMutex>& lock, std::function<bool()> pred) override
  {
    cv_.wait(lock, pred);
  }

  void WaitFor(
      std::unique_lock<TritonMutex>& lock, uint64_t rel_time_us) override
  {
    cv_.wait_for(lock, std::chrono::microseconds(rel_time_us));
  }

  void NotifyAll() override { cv_.notify_all(); }

  void NotifyOne() override { cv_.notify_one(); }

 private:
  std::condition_variable cv_;
};

}}  // namespace nvidia::inferenceserver
