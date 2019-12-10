// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/clients/c++/perf_client/concurrency_manager.h"
#include <queue>

ConcurrencyManager::~ConcurrencyManager()
{
  // The destruction of derived class should wait for all the request generator
  // threads to finish
  StopWorkerThreads();
}


nic::Error
ConcurrencyManager::Create(
    const bool async, const int32_t batch_size, const size_t max_threads,
    const size_t max_concurrency, const size_t sequence_length,
    const size_t string_length, const std::string& string_data,
    const bool zero_input,
    const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
    std::vector<std::string>& user_data,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const std::shared_ptr<ContextFactory>& factory,
    std::unique_ptr<LoadManager>* manager)
{
  std::unique_ptr<ConcurrencyManager> local_manager(new ConcurrencyManager(
      async, input_shapes, batch_size, max_threads, max_concurrency,
      sequence_length, shared_memory_type, output_shm_size, factory));

  local_manager->threads_config_.reserve(max_threads);

  RETURN_IF_ERROR(local_manager->InitManagerInputs(
      string_length, string_data, zero_input, user_data));

  if (local_manager->shared_memory_type_ !=
      SharedMemoryType::NO_SHARED_MEMORY) {
    RETURN_IF_ERROR(local_manager->InitSharedMemory());
  }

  *manager = std::move(local_manager);

  return nic::Error::Success;
}

ConcurrencyManager::ConcurrencyManager(
    const bool async,
    const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
    const int32_t batch_size, const size_t max_threads,
    const size_t max_concurrency, const size_t sequence_length,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const std::shared_ptr<ContextFactory>& factory)
    : LoadManager(
          async, input_shapes, batch_size, max_threads, sequence_length,
          shared_memory_type, output_shm_size, factory),
      max_concurrency_(max_concurrency)
{
  if (on_sequence_model_) {
    for (uint64_t i = 0; i < max_concurrency_; i++) {
      sequence_stat_.emplace_back(new SequenceStat(0));
    }
  }
}

nic::Error
ConcurrencyManager::ChangeConcurrencyLevel(
    const size_t concurrent_request_count)
{
  // Always prefer to create new threads if the maximum limit has not been met
  while ((concurrent_request_count > threads_.size()) &&
         (threads_.size() < max_threads_)) {
    // Launch new thread for inferencing
    threads_stat_.emplace_back(new ThreadStat());
    threads_config_.emplace_back(new ThreadConfig(threads_config_.size()));

    // Worker maintains concurrency in different ways.
    // For sequence models, multiple contexts must be created for multiple
    // concurrent sequences.
    // For non-sequence models, one context can send out multiple requests
    // at the same time. Thus it uses one single context as every infer context
    // creates a worker thread implicitly.
    // While operating in synchronous mode, each context can send only one
    // request at a time, hence the number of worker threads should be equal to
    // the requested concurrency levels.
    threads_.emplace_back(
        &ConcurrencyManager::Infer, this, threads_stat_.back(),
        threads_config_.back());
  }

  // Compute the new concurrency level for each thread (take floor)
  // and spread the remaining value
  size_t avg_concurrency = concurrent_request_count / threads_.size();
  size_t threads_add_one = concurrent_request_count % threads_.size();

  active_threads_ = 0;
  for (size_t i = 0; i < threads_stat_.size(); i++) {
    threads_config_[i]->concurrency_ =
        avg_concurrency + (i < threads_add_one ? 1 : 0);
    if (threads_config_[i]->concurrency_) {
      active_threads_++;
    }
  }

  // Make sure all threads will check their updated concurrency level
  wake_signal_.notify_all();

  std::cout << "Request concurrency: " << concurrent_request_count << std::endl;
  return nic::Error::Success;
}

// Function for worker threads.
// If the model is non-sequence model, each worker uses only one context
// to maintain concurrency assigned to worker.
// If the model is sequence model, each worker has to use multiples contexts
// to maintain (sequence) concurrency assigned to worker.
void
ConcurrencyManager::Infer(
    std::shared_ptr<ThreadStat> thread_stat,
    std::shared_ptr<ThreadConfig> thread_config)
{
  std::vector<std::unique_ptr<InferContextMetaData>> ctxs;
  uint32_t seq_id = 0, ctx_id = 0;
  std::queue<int> free_ctx_ids;

  // Reserve the vectors in case of sequence models. In non-sequence or
  // synchronous mode only one context will be opened hence no need of
  // reserving.
  if (on_sequence_model_ && async_) {
    thread_stat->contexts_stat_.reserve(max_concurrency_);
    ctxs.reserve(max_concurrency_);
  }

  // Variable that can be used across InferContexts
  std::unique_ptr<nic::InferContext::Options> options(nullptr);

  // Variable used to signal request completion
  bool notified = false;
  std::mutex cb_mtx;
  std::condition_variable cb_cv;

  std::atomic<int> total_ongoing_requests(0);

  // run inferencing until receiving exit signal to maintain server load.
  do {
    // Only interact with synchronous mechanism if the worker should wait
    if (thread_config->concurrency_ == 0) {
      // Wait if no request should be sent and it is not exiting
      std::unique_lock<std::mutex> lock(wake_mutex_);
      wake_signal_.wait(lock, [&thread_config]() {
        return early_exit || (thread_config->concurrency_ > 0);
      });
    }

    size_t num_reqs = thread_config->concurrency_;

    // If the model is non-sequence model, use one InferContext to maintain
    // concurrency for this thread.
    size_t active_ctx_cnt = on_sequence_model_ ? num_reqs : 1;

    while (active_ctx_cnt > ctxs.size()) {
      free_ctx_ids.push(ctxs.size());
      ctxs.emplace_back(new InferContextMetaData());
      thread_stat->contexts_stat_.emplace_back();
      if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
        thread_stat->status_ = PrepareInfer(&(ctxs.back()->ctx_), &options);
      } else {
        thread_stat->status_ =
            PrepareSharedMemoryInfer(&(ctxs.back()->ctx_), &options);
      }
      if (!thread_stat->status_.IsOk()) {
        return;
      }
    }

    // Create async requests such that the number of ongoing requests
    // matches the concurrency level
    // Non-sequence model is 'num_reqs' * 1 ctx
    // Sequence model is 1 request of 1 sequence * 'active_ctx_cnt' ctxs
    while (total_ongoing_requests < (int)num_reqs) {
      // Update the inputs if required for non-sequence
      if (using_json_data_ && (!on_sequence_model_)) {
        int step_id = (thread_config->non_sequence_data_step_id_ %
                       data_loader_->GetTotalStepsNonSequence()) *
                      batch_size_;
        thread_config->non_sequence_data_step_id_ += active_threads_;
        // There will be only one ctx in non-sequence case
        thread_stat->status_ =
            UpdateInputs(ctxs[0]->ctx_->Inputs(), 0, step_id);
        if (!thread_stat->status_.IsOk()) {
          return;
        }
      }

      uint32_t flags = 0;
      if (on_sequence_model_) {
        flags = 0;

        size_t offset = 0;
        for (size_t i = 0; i < thread_config->thread_id_; i++) {
          offset += threads_config_[i]->concurrency_;
        }

        // Find the next available context id to use for this request
        {
          std::lock_guard<std::mutex> lk(cb_mtx);
          ctx_id = free_ctx_ids.front();
          free_ctx_ids.pop();
        }
        seq_id = offset + ctx_id;

        {
          std::lock_guard<std::mutex> guard(sequence_stat_[seq_id]->mtx_);
          if (sequence_stat_[seq_id]->remaining_queries_ == 0) {
            flags |= ni::InferRequestHeader::FLAG_SEQUENCE_START;
            InitNewSequence(seq_id);
          }
          if (sequence_stat_[seq_id]->remaining_queries_ == 1) {
            flags |= ni::InferRequestHeader::FLAG_SEQUENCE_END;
          }
          options->SetFlag(
              ni::InferRequestHeader::FLAG_SEQUENCE_START,
              flags & ni::InferRequestHeader::FLAG_SEQUENCE_START);
          options->SetFlag(
              ni::InferRequestHeader::FLAG_SEQUENCE_END,
              flags & ni::InferRequestHeader::FLAG_SEQUENCE_END);

          // Override the correlation ID.
          options->SetCorrelationId(sequence_stat_[seq_id]->corr_id_);
          ctxs[ctx_id]->ctx_->SetRunOptions(*options);

          // Update the inputs if required
          if (using_json_data_) {
            int step_id = data_loader_->GetTotalSteps(
                              sequence_stat_[seq_id]->data_stream_id_) -
                          sequence_stat_[seq_id]->remaining_queries_;
            thread_stat->status_ = UpdateInputs(
                ctxs[ctx_id]->ctx_->Inputs(),
                sequence_stat_[seq_id]->data_stream_id_, step_id);
            if (!thread_stat->status_.IsOk()) {
              return;
            }
          }
          sequence_stat_[seq_id]->remaining_queries_--;
        }
      }
      if (async_) {
        struct timespec start_time_async;
        clock_gettime(CLOCK_MONOTONIC, &start_time_async);
        thread_stat->status_ = ctxs[ctx_id]->ctx_->AsyncRun(
            [&notified, &cb_mtx, &cb_cv, &ctxs, &thread_stat,
             &total_ongoing_requests, &free_ctx_ids, start_time_async, flags,
             ctx_id](
                nic::InferContext* ctx,
                std::shared_ptr<nic::InferContext::Request> request) {
              std::map<std::string, std::unique_ptr<nic::InferContext::Result>>
                  results;
              ctxs[ctx_id]->ctx_->GetAsyncRunResults(request, &results);
              struct timespec end_time_async;
              clock_gettime(CLOCK_MONOTONIC, &end_time_async);
              {
                // Add the request timestamp to thread Timestamp vector with
                // proper locking
                std::lock_guard<std::mutex> lock(thread_stat->mu_);
                thread_stat->request_timestamps_.emplace_back(std::make_tuple(
                    start_time_async, end_time_async, flags,
                    false /* delayed */));
                ctxs[ctx_id]->ctx_->GetStat(
                    &(thread_stat->contexts_stat_[ctx_id]));
              }

              // avoid competition over 'cb_mtx'
              {
                std::lock_guard<std::mutex> lk(cb_mtx);
                free_ctx_ids.push(ctx_id);
                notified = true;
              }
              total_ongoing_requests--;

              cb_cv.notify_all();
            });
        if (!thread_stat->status_.IsOk()) {
          return;
        }
      } else {
        std::map<std::string, std::unique_ptr<nic::InferContext::Result>>
            results;
        struct timespec start_time_sync, end_time_sync;
        clock_gettime(CLOCK_MONOTONIC, &start_time_sync);
        thread_stat->status_ = ctxs[ctx_id]->ctx_->Run(&results);
        if (!thread_stat->status_.IsOk()) {
          return;
        }
        clock_gettime(CLOCK_MONOTONIC, &end_time_sync);
        {
          // Add the request timestamp to thread Timestamp vector with proper
          // locking
          std::lock_guard<std::mutex> lock(thread_stat->mu_);
          thread_stat->request_timestamps_.emplace_back(std::make_tuple(
              start_time_sync, end_time_sync, flags, false /* delayed */));
          ctxs[ctx_id]->ctx_->GetStat(&(thread_stat->contexts_stat_[ctx_id]));
        }
        free_ctx_ids.push(ctx_id);
      }
      total_ongoing_requests++;
    }

    if (async_) {
      {
        // If async, then wait for signal from callback.
        std::unique_lock<std::mutex> lk(cb_mtx);
        cb_cv.wait(lk, [&notified] {
          if (notified) {
            notified = false;
            return true;
          }
          return false;
        });
      }
    } else {
      // If synchronous, then all the requests have already been completed.
      total_ongoing_requests = 0;
    }

    if (early_exit) {
      if (async_) {
        // Wait for all callbacks to complete.
        // Loop to ensure all the inflight requests have been completed.
        while (total_ongoing_requests != 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
      }
      // end loop
      break;
    }
  } while (true);
}
