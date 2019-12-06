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
    const std::string& user_data, const SharedMemoryType shared_memory_type,
    const size_t output_shm_size,
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
  std::unique_ptr<InferContextMetaData> ctx(new InferContextMetaData());
  uint32_t seq_id = -1;

  std::unique_ptr<nic::InferContext::Options> options(nullptr);
  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    thread_stat->status_ = PrepareInfer(&(ctx->ctx_), &options);
  } else {
    thread_stat->status_ = PrepareSharedMemoryInfer(&(ctx->ctx_), &options);
  }
  if (!thread_stat->status_.IsOk()) {
    return;
  }

  // Variable used to signal request completion
  bool notified = false;
  std::mutex cb_mtx;
  std::condition_variable cb_cv;

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

    auto& request_cnt = ctx->inflight_request_cnt_;
    while (request_cnt < num_reqs) {
      // Update the inputs if required for non-sequence
      if (using_json_data_ && (!on_sequence_model_)) {
        int step_id =
            (thread_config->non_sequence_step_id_ % max_non_sequence_step_id_) *
            batch_size_;
        thread_config->non_sequence_step_id_ += active_threads_;
        thread_stat->status_ = UpdateInputs(ctx->ctx_->Inputs(), 0, step_id);
        if (!thread_stat->status_.IsOk()) {
          return;
        }
      }

      uint32_t flags = 0;
      if (on_sequence_model_) {
        flags = 0;
        // Select the next sequence id in the range of the worker thread
        size_t offset = 0;
        for (size_t i = 0; i < thread_config->thread_id_; i++) {
          offset += threads_config_[i]->concurrency_;
        }
        seq_id = offset + ((seq_id + 1) % num_reqs);
        // No need to lock because different worker threads dispatch distinct
        // sequences
        if (sequence_stat_[seq_id]->remaining_queries_ == 0) {
          flags |= ni::InferRequestHeader::FLAG_SEQUENCE_START;
          // Update the correlation id to the next available unique id
          sequence_stat_[seq_id]->corr_id_ = next_corr_id_++;
          if (!using_json_data_) {
            size_t new_length = GetRandomLength(0.2);
            sequence_stat_[seq_id]->remaining_queries_ =
                new_length == 0 ? 1 : new_length;
          } else {
            sequence_stat_[seq_id]->data_stream_id_ =
                sequence_stat_[seq_id]->corr_id_ % json_data_stream_cnt_;
            sequence_stat_[seq_id]->remaining_queries_ =
                json_step_num_[sequence_stat_[seq_id]->data_stream_id_];
          }
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
        ctx->ctx_->SetRunOptions(*options);

        // Update the inputs if required
        if (using_json_data_) {
          int step_id =
              json_step_num_[sequence_stat_[seq_id]->data_stream_id_] -
              sequence_stat_[seq_id]->remaining_queries_;
          thread_stat->status_ = UpdateInputs(
              ctx->ctx_->Inputs(), sequence_stat_[seq_id]->data_stream_id_,
              step_id);
          if (!thread_stat->status_.IsOk()) {
            return;
          }
        }
      }
      if (async_) {
        struct timespec start_time_async;
        clock_gettime(CLOCK_MONOTONIC, &start_time_async);
        thread_stat->status_ = ctx->ctx_->AsyncRun(
            [&notified, &cb_mtx, &cb_cv, &ctx, &thread_stat, start_time_async,
             flags](
                nic::InferContext* ctx,
                std::shared_ptr<nic::InferContext::Request> request) {
              std::map<std::string, std::unique_ptr<nic::InferContext::Result>>
                  results;
              ctx->ctx_->GetAsyncRunResults(request, &results);
              struct timespec end_time_async;
              clock_gettime(CLOCK_MONOTONIC, &end_time_async);
              {
                // Add the request timestamp to thread Timestamp vector with
                // proper locking
                std::lock_guard<std::mutex> lock(thread_stat->mu_);
                thread_stat->request_timestamps_.emplace_back(std::make_tuple(
                    start_time_async, end_time_async, flags,
                    false /* delayed */));
                ctx->ctx_->GetStat(&(thread_stat->context_stat_));
              }
              ctx->inflight_request_cnt_--;
              // avoid competition over 'cb_mtx'
              if (!notified) {
                {
                  std::lock_guard<std::mutex> lk(cb_mtx);
                  notified = true;
                }
                cb_cv.notify_all();
              }
            });
        if (!thread_stat->status_.IsOk()) {
          return;
        }
        ctx->inflight_request_cnt_++;
      } else {
        std::map<std::string, std::unique_ptr<nic::InferContext::Result>>
            results;
        struct timespec start_time_sync, end_time_sync;
        clock_gettime(CLOCK_MONOTONIC, &start_time_sync);
        thread_stat->status_ = ctx->ctx_->Run(&results);
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
          ctx->ctx_->GetStat(&(thread_stat->context_stat_));
        }
        request_cnt++;
      }
      if (on_sequence_model_) {
        sequence_stat_[seq_id]->remaining_queries_--;
      }
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
      ctx->inflight_request_cnt_ = 0;
    }

    if (early_exit) {
      if (async_) {
        // Wait for all callbacks to complete.
        // Loop to ensure all the inflight requests have been completed.
        while (ctx->inflight_request_cnt_ != 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
      }
      // end loop
      break;
    }
  } while (true);
}
