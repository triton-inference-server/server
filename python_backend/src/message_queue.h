// Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/thread/thread_time.hpp>
#include <cstddef>

#include "pb_utils.h"
#include "shm_manager.h"
#ifdef TRITON_PB_STUB
#include "pb_stub_log.h"
#endif

namespace triton { namespace backend { namespace python {
namespace bi = boost::interprocess;

/// Struct holding the representation of a message queue inside the shared
/// memory.
/// \param size Total size of the message queue. Considered invalid after
/// MessageQueue::LoadFromSharedMemory. Check DLIS-8378 for additional details.
/// \param mutex Handle of the mutex variable protecting index.
/// \param index Used element index.
/// \param sem_empty Semaphore object counting the number of empty buffer slots.
/// \param sem_full Semaphore object counting the number of used buffer slots.
struct MessageQueueShm {
  bi::interprocess_semaphore sem_empty{0};
  bi::interprocess_semaphore sem_full{0};
  bi::interprocess_mutex mutex;
  std::size_t size;
  bi::managed_external_buffer::handle_t buffer;
  int head;
  int tail;
};

template <typename T>
class MessageQueue {
 public:
  /// Create a new MessageQueue in the shared memory.
  static std::unique_ptr<MessageQueue<T>> Create(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      uint32_t message_queue_size)
  {
    AllocatedSharedMemory<MessageQueueShm> mq_shm =
        shm_pool->Construct<MessageQueueShm>();
    mq_shm.data_->size = message_queue_size;

    AllocatedSharedMemory<T> mq_buffer_shm =
        shm_pool->Construct<T>(message_queue_size /* count */);
    mq_shm.data_->buffer = mq_buffer_shm.handle_;
    mq_shm.data_->head = 0;
    mq_shm.data_->tail = 0;

    new (&(mq_shm.data_->mutex)) bi::interprocess_mutex{};
    new (&(mq_shm.data_->sem_empty))
        bi::interprocess_semaphore{message_queue_size};
    new (&(mq_shm.data_->sem_full)) bi::interprocess_semaphore{0};

    return std::unique_ptr<MessageQueue<T>>(
        new MessageQueue<T>(mq_shm, mq_buffer_shm));
  }

  /// Load an already existing message queue from the shared memory.
  static std::unique_ptr<MessageQueue<T>> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t message_queue_handle)
  {
    AllocatedSharedMemory<MessageQueueShm> mq_shm =
        shm_pool->Load<MessageQueueShm>(message_queue_handle);
    AllocatedSharedMemory<T> mq_shm_buffer =
        shm_pool->Load<T>(mq_shm.data_->buffer);

    return std::unique_ptr<MessageQueue<T>>(
        new MessageQueue(mq_shm, mq_shm_buffer));
  }

  /// Push a message inside the message queue.
  /// \param message The shared memory handle of the message.
  void Push(T message)
  {
    while (true) {
      try {
        SemEmptyMutable()->wait();
        break;
      }
      catch (bi::interprocess_exception& ex) {
      }
    }

    {
      bi::scoped_lock<bi::interprocess_mutex> lock{*MutexMutable()};
      int head_idx = Head();
      // Additional check to avoid out of bounds read/write. Check DLIS-8378 for
      // additional details.
      if (head_idx < 0 || static_cast<uint32_t>(head_idx) >= Size()) {
        std::string error_msg =
            "internal error: message queue head index out of bounds. Expects "
            "positive integer less than the size of message queue " +
            std::to_string(Size()) + " but got " + std::to_string(head_idx);
#ifdef TRITON_PB_STUB
        LOG_ERROR << error_msg;
#else
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, error_msg.c_str());
#endif
        return;
      }
      Buffer()[head_idx] = message;
      HeadIncrement();
    }
    SemFullMutable()->post();
  }

  void Push(T message, int const& duration, bool& success)
  {
    boost::system_time timeout =
        boost::get_system_time() + boost::posix_time::milliseconds(duration);

    while (true) {
      try {
        if (!SemEmptyMutable()->timed_wait(timeout)) {
          success = false;
          return;
        } else {
          break;
        }
      }
      catch (bi::interprocess_exception& ex) {
      }
    }

    {
      timeout =
          boost::get_system_time() + boost::posix_time::milliseconds(duration);
      bi::scoped_lock<bi::interprocess_mutex> lock{*MutexMutable(), timeout};
      if (!lock) {
        SemEmptyMutable()->post();
        success = false;
        return;
      }
      success = true;

      int head_idx = Head();
      // Additional check to avoid out of bounds read/write. Check DLIS-8378 for
      // additional details.
      if (head_idx < 0 || static_cast<uint32_t>(head_idx) >= Size()) {
        std::string error_msg =
            "internal error: message queue head index out of bounds. Expects "
            "positive integer less than the size of message queue " +
            std::to_string(Size()) + " but got " + std::to_string(head_idx);
#ifdef TRITON_PB_STUB
        LOG_ERROR << error_msg;
#else
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, error_msg.c_str());
#endif
        return;
      }
      Buffer()[head_idx] = message;
      HeadIncrement();
    }
    SemFullMutable()->post();
  }

  /// Pop a message from the message queue. This call will block until there
  /// is a message inside the message queue to return.
  /// \return the handle of the new message.
  T Pop()
  {
    T message;

    while (true) {
      try {
        SemFullMutable()->wait();
        break;
      }
      catch (bi::interprocess_exception& ex) {
      }
    }

    {
      bi::scoped_lock<bi::interprocess_mutex> lock{*MutexMutable()};

      message = Buffer()[Tail()];
      TailIncrement();
    }
    SemEmptyMutable()->post();

    return message;
  }

  T Pop(int const& duration, bool& success)
  {
    T message = 0;
    boost::system_time timeout =
        boost::get_system_time() + boost::posix_time::milliseconds(duration);

    while (true) {
      try {
        if (!SemFullMutable()->timed_wait(timeout)) {
          success = false;
          return message;
        } else {
          break;
        }
      }
      catch (bi::interprocess_exception& ex) {
      }
    }

    {
      timeout =
          boost::get_system_time() + boost::posix_time::milliseconds(duration);
      bi::scoped_lock<bi::interprocess_mutex> lock{*MutexMutable(), timeout};
      if (!lock) {
        SemFullMutable()->post();
        success = false;
        return message;
      }
      success = true;

      message = Buffer()[Tail()];
      TailIncrement();
    }
    SemEmptyMutable()->post();

    return message;
  }

  /// Resets the semaphores for the message queue. This function is useful for
  /// when the stub process may have exited unexpectedly and the semaphores need
  /// to be restarted so that the message queue is in a proper state.
  void ResetSemaphores()
  {
    new (SemFullMutable()) bi::interprocess_semaphore(0);
    new (SemEmptyMutable()) bi::interprocess_semaphore(Size());
    new (MutexMutable()) bi::interprocess_mutex;
    mq_shm_ptr_->tail = 0;
    mq_shm_ptr_->head = 0;
  }

  /// Get the shared memory handle of MessageQueue
  bi::managed_external_buffer::handle_t ShmHandle() { return mq_handle_; }

  /// Release the ownership of this object in shared memory.
  void Release()
  {
    if (mq_shm_.data_ != nullptr) {
      mq_shm_.data_.release();
    }

    if (mq_buffer_shm_.data_ != nullptr) {
      mq_buffer_shm_.data_.release();
    }
  }

 private:
  uint32_t Size() { return size_; }
  const bi::interprocess_mutex& Mutex() { return mq_shm_ptr_->mutex; }
  bi::interprocess_mutex* MutexMutable() { return &(mq_shm_ptr_->mutex); }
  int& Head() { return mq_shm_ptr_->head; }
  int& Tail() { return mq_shm_ptr_->tail; }
  T* Buffer() { return mq_buffer_shm_ptr_; }
  const bi::interprocess_semaphore& SemEmpty()
  {
    return mq_shm_ptr_->sem_empty;
  }
  bi::interprocess_semaphore* SemEmptyMutable()
  {
    return &(mq_shm_ptr_->sem_empty);
  }
  const bi::interprocess_semaphore& SemFull() { return mq_shm_ptr_->sem_full; }
  bi::interprocess_semaphore* SemFullMutable()
  {
    return &(mq_shm_ptr_->sem_full);
  }

  void HeadIncrement() { mq_shm_ptr_->head = (mq_shm_ptr_->head + 1) % Size(); }
  void TailIncrement() { mq_shm_ptr_->tail = (mq_shm_ptr_->tail + 1) % Size(); }

  AllocatedSharedMemory<MessageQueueShm> mq_shm_;
  AllocatedSharedMemory<T> mq_buffer_shm_;

  MessageQueueShm* mq_shm_ptr_;
  T* mq_buffer_shm_ptr_;
  bi::managed_external_buffer::handle_t mq_handle_;
  uint32_t size_;

  /// Create/load a Message queue.
  /// \param mq_shm Message queue representation in shared memory.
  MessageQueue(
      AllocatedSharedMemory<MessageQueueShm>& mq_shm,
      AllocatedSharedMemory<T>& mq_buffer_shm)
      : mq_shm_(std::move(mq_shm)), mq_buffer_shm_(std::move(mq_buffer_shm))
  {
    mq_buffer_shm_ptr_ = mq_buffer_shm_.data_.get();
    mq_shm_ptr_ = mq_shm_.data_.get();
    mq_handle_ = mq_shm_.handle_;
    size_ = mq_shm_ptr_->size;
  }
};
}}}  // namespace triton::backend::python
