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

#include "src/clients/c++/experimental_api_v2/perf_client/concurrency_manager.h"
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
    const bool zero_input, std::vector<std::string>& user_data,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<TritonClientFactory>& factory,
    std::unique_ptr<LoadManager>* manager)
{
  std::unique_ptr<ConcurrencyManager> local_manager(new ConcurrencyManager(
      async, batch_size, max_threads, max_concurrency, sequence_length,
      shared_memory_type, output_shm_size, parser, factory));

  local_manager->threads_config_.reserve(max_threads);

  RETURN_IF_ERROR(local_manager->InitManagerInputs(
      string_length, string_data, zero_input, user_data));

  // if (local_manager->shared_memory_type_ !=
  //    SharedMemoryType::NO_SHARED_MEMORY) {
  //  RETURN_IF_ERROR(local_manager->InitSharedMemory());
  //}

  *manager = std::move(local_manager);

  return nic::Error::Success;
}

ConcurrencyManager::ConcurrencyManager(
    const bool async, const int32_t batch_size, const size_t max_threads,
    const size_t max_concurrency, const size_t sequence_length,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<TritonClientFactory>& factory)
    : LoadManager(
          async, batch_size, max_threads, sequence_length, shared_memory_type,
          output_shm_size, parser, factory),
      execute_(true), max_concurrency_(max_concurrency)
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
  if (on_sequence_model_ && async_) {
    execute_ = false;
    // Wait to see all threads are paused.
    for (auto& thread_config : threads_config_) {
      while (!thread_config->is_paused_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  }
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

  if (on_sequence_model_ && async_) {
    execute_ = true;
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

  // Variable used to signal request completion
  bool notified = false;
  std::mutex cb_mtx;
  std::condition_variable cb_cv;

  std::atomic<int> total_ongoing_requests(0);

  // run inferencing until receiving exit signal to maintain server load.
  do {
    if (on_sequence_model_ && async_) {
      if (!execute_) {
        // Ensures the clean exit of the sequences
        while (total_ongoing_requests != 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        // Wait if no request should be sent and it is not exiting
        thread_config->is_paused_ = true;
        std::unique_lock<std::mutex> lock(wake_mutex_);
        wake_signal_.wait(lock, [this]() { return early_exit || execute_; });
      }
    }

    thread_config->is_paused_ = false;

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
      thread_stat->status_ =
          factory_->CreateTritonClient(&(ctxs.back()->infer_client_));
      ctxs.back()->options_.reset(new nic::InferOptions(parser_->ModelName()));
      ctxs.back()->options_->model_version_ = parser_->ModelVersion();
      thread_stat->contexts_stat_.emplace_back();
      if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
        thread_stat->status_ = PrepareInfer(ctxs.back().get());
      } else {
        // thread_stat->status_ =
        //    PrepareSharedMemoryInfer(&(ctxs.back()->ctx_), &options);
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
        thread_stat->status_ = UpdateInputs(ctxs[ctx_id]->inputs_, 0, step_id);
        if (!thread_stat->status_.IsOk()) {
          return;
        }
      }

      struct timespec start_time_sync, end_time_sync;
      clock_gettime(CLOCK_MONOTONIC, &start_time_sync);
      nic::InferResult* results;
      thread_stat->status_ = ctxs[ctx_id]->infer_client_->Infer(
          &results, *(ctxs[ctx_id]->options_), ctxs[ctx_id]->inputs_,
          ctxs[ctx_id]->outputs_);
      if (results != nullptr) {
        delete results;
      }
      if (!thread_stat->status_.IsOk()) {
        return;
      }
      clock_gettime(CLOCK_MONOTONIC, &end_time_sync);
      {
        // Add the request timestamp to thread Timestamp vector with proper
        // locking
        std::lock_guard<std::mutex> lock(thread_stat->mu_);
        thread_stat->request_timestamps_.emplace_back(std::make_tuple(
            start_time_sync, end_time_sync, 0, false /* delayed */));
        thread_stat->status_ = ctxs[ctx_id]->infer_client_->ClientInferStat(
            &(thread_stat->contexts_stat_[ctx_id]));
        if (!thread_stat->status_.IsOk()) {
          return;
        }
      }
      free_ctx_ids.push(ctx_id);
    }
    total_ongoing_requests++;
    //  }

    total_ongoing_requests = 0;

    if (early_exit || (!thread_stat->cb_status_.IsOk())) {
      // if (async_) {
      // Wait for all callbacks to complete.
      // Loop to ensure all the inflight requests have been completed.
      //  while (total_ongoing_requests != 0) {
      //    std::this_thread::sleep_for(std::chrono::milliseconds(500));
      //  }
      //}
      // end loop
      break;
    }
  } while (true);
}
