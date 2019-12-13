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

#include "src/clients/c++/perf_client/request_rate_manager.h"

RequestRateManager::~RequestRateManager()
{
  // The destruction of derived class should wait for all the request generator
  // threads to finish
  StopWorkerThreads();
}

nic::Error
RequestRateManager::Create(
    const bool async, const uint64_t measurement_window_ms,
    Distribution request_distribution, const int32_t batch_size,
    const size_t max_threads, const uint32_t num_of_sequences,
    const size_t sequence_length, const size_t string_length,
    const std::string& string_data, const bool zero_input,
    const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
    std::vector<std::string>& user_data,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const std::shared_ptr<ContextFactory>& factory,
    std::unique_ptr<LoadManager>* manager)
{
  std::unique_ptr<RequestRateManager> local_manager(new RequestRateManager(
      async, input_shapes, request_distribution, batch_size,
      measurement_window_ms, max_threads, num_of_sequences, sequence_length,
      shared_memory_type, output_shm_size, factory));

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

RequestRateManager::RequestRateManager(
    const bool async,
    const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
    Distribution request_distribution, int32_t batch_size,
    const uint64_t measurement_window_ms, const size_t max_threads,
    const uint32_t num_of_sequences, const size_t sequence_length,
    const SharedMemoryType shared_memory_type, const size_t output_shm_size,
    const std::shared_ptr<ContextFactory>& factory)
    : LoadManager(
          async, input_shapes, batch_size, max_threads, sequence_length,
          shared_memory_type, output_shm_size, factory),
      request_distribution_(request_distribution), execute_(false)
{
  if (on_sequence_model_) {
    for (uint64_t i = 0; i < num_of_sequences; i++) {
      sequence_stat_.emplace_back(new SequenceStat(next_corr_id_++));
    }
  }
  gen_duration_.reset(
      new std::chrono::nanoseconds(2 * measurement_window_ms * 1000 * 1000));
}

nic::Error
RequestRateManager::ChangeRequestRate(const double request_rate)
{
  PauseWorkers();
  // Can safely update the schedule
  GenerateSchedule(request_rate);
  ResumeWorkers();

  return nic::Error::Success;
}

nic::Error
RequestRateManager::ResetWorkers()
{
  PauseWorkers();
  ResumeWorkers();

  return nic::Error::Success;
}

void
RequestRateManager::GenerateSchedule(const double request_rate)
{
  std::function<std::chrono::nanoseconds(std::mt19937&)> distribution;
  if (request_distribution_ == Distribution::POISSON) {
    distribution = ScheduleDistribution<Distribution::POISSON>(request_rate);
  } else if (request_distribution_ == Distribution::CONSTANT) {
    distribution = ScheduleDistribution<Distribution::CONSTANT>(request_rate);
  } else {
    return;
  }
  schedule_.clear();
  schedule_.emplace_back(0);

  std::mt19937 schedule_rng;
  while (schedule_.back() < *gen_duration_) {
    std::chrono::nanoseconds next_timestamp(
        schedule_.back() + distribution(schedule_rng));
    schedule_.emplace_back(next_timestamp);
  }
  std::cout << "Request Rate: " << request_rate
            << " inference requests per seconds" << std::endl;
}

void
RequestRateManager::PauseWorkers()
{
  // Pause all the threads
  execute_ = false;

  if (threads_.empty()) {
    while (threads_.size() < max_threads_) {
      // Launch new thread for inferencing
      threads_stat_.emplace_back(new ThreadStat());
      threads_config_.emplace_back(
          new ThreadConfig(threads_.size(), max_threads_));

      // Worker threads share the responsibility to generate the inferences at
      // a particular schedule.
      threads_.emplace_back(
          &RequestRateManager::Infer, this, threads_stat_.back(),
          threads_config_.back());
    }
  }

  // Wait to see all threads are paused.
  for (auto& thread_config : threads_config_) {
    while (!thread_config->is_paused_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void
RequestRateManager::ResumeWorkers()
{
  // Reset all the thread counters
  for (auto& thread_config : threads_config_) {
    thread_config->index_ = thread_config->id_;
    thread_config->rounds_ = 0;
  }

  // Update the start_time_ to point to current time
  start_time_ = std::chrono::steady_clock::now();

  // Wake up all the threads to begin execution
  execute_ = true;
  wake_signal_.notify_all();
}


void
RequestRateManager::Infer(
    std::shared_ptr<ThreadStat> thread_stat,
    std::shared_ptr<ThreadConfig> thread_config)
{
  std::shared_ptr<InferContextMetaData> ctx(new InferContextMetaData());
  thread_stat->contexts_stat_.emplace_back();

  std::unique_ptr<nic::InferContext::Options> options(nullptr);
  if (shared_memory_type_ == SharedMemoryType::NO_SHARED_MEMORY) {
    thread_stat->status_ = PrepareInfer(&(ctx->ctx_), &options);
  } else {
    thread_stat->status_ = PrepareSharedMemoryInfer(&(ctx->ctx_), &options);
  }
  if (!thread_stat->status_.IsOk()) {
    return;
  }

  // run inferencing until receiving exit signal to maintain server load.
  do {
    // Should wait till main thread signals execution start
    if (!execute_) {
      // Ensures the clean measurements after thread is woken up.
      while (ctx->inflight_request_cnt_ != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
      // Wait if no request should be sent and it is not exiting
      thread_config->is_paused_ = true;
      std::unique_lock<std::mutex> lock(wake_mutex_);
      wake_signal_.wait(lock, [this]() { return early_exit || execute_; });
    }

    thread_config->is_paused_ = false;

    uint32_t seq_id = 0, flags = 0;

    // Sleep if required
    std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    std::chrono::nanoseconds wait_time =
        (schedule_[thread_config->index_] +
         (thread_config->rounds_ * (*gen_duration_))) -
        (now - start_time_);

    thread_config->index_ = (thread_config->index_ + thread_config->stride_);
    // Loop around the schedule to keep running
    thread_config->rounds_ += (thread_config->index_ / schedule_.size());
    thread_config->index_ = thread_config->index_ % schedule_.size();

    bool delayed = false;
    if (wait_time.count() < 0) {
      delayed = true;
    } else {
      std::this_thread::sleep_for(wait_time);
    }

    // Update the inputs if required
    if (using_json_data_ && (!on_sequence_model_)) {
      int step_id = (thread_config->non_sequence_data_step_id_ %
                     data_loader_->GetTotalStepsNonSequence()) *
                    batch_size_;
      thread_config->non_sequence_data_step_id_ += max_threads_;
      thread_stat->status_ = UpdateInputs(ctx->ctx_->Inputs(), 0, step_id);
      if (!thread_stat->status_.IsOk()) {
        return;
      }
    }

    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    if (on_sequence_model_) {
      flags = 0;
      // Select one of the sequence at random for this request
      seq_id = rand() % sequence_stat_.size();
      // Need lock to protect the order of dispatch across worker threads.
      // This also helps in reporting the realistic latencies.
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
      ctx->ctx_->SetRunOptions(*options);

      // Update the inputs if required
      if (using_json_data_) {
        int step_id = data_loader_->GetTotalSteps(
                          sequence_stat_[seq_id]->data_stream_id_) -
                      sequence_stat_[seq_id]->remaining_queries_;
        thread_stat->status_ = UpdateInputs(
            ctx->ctx_->Inputs(), sequence_stat_[seq_id]->data_stream_id_,
            step_id);
        if (!thread_stat->status_.IsOk()) {
          return;
        }
      }

      Request(ctx, flags, delayed, start_time, thread_stat);
      sequence_stat_[seq_id]->remaining_queries_--;
    } else {
      Request(ctx, flags, delayed, start_time, thread_stat);
    }

    if (early_exit) {
      if (on_sequence_model_) {
        // Finish off all the ongoing sequences for graceful exit
        for (size_t i = thread_config->id_; i < sequence_stat_.size();
             i += thread_config->stride_) {
          std::lock_guard<std::mutex> guard(sequence_stat_[i]->mtx_);
          if (sequence_stat_[i]->remaining_queries_ != 0) {
            uint32_t flags = 0;
            flags |= ni::InferRequestHeader::FLAG_SEQUENCE_END;
            options->SetFlag(
                ni::InferRequestHeader::FLAG_SEQUENCE_START,
                flags & ni::InferRequestHeader::FLAG_SEQUENCE_START);
            options->SetFlag(
                ni::InferRequestHeader::FLAG_SEQUENCE_END,
                flags & ni::InferRequestHeader::FLAG_SEQUENCE_END);

            // Override the correlation ID.
            options->SetCorrelationId(sequence_stat_[i]->corr_id_);
            ctx->ctx_->SetRunOptions(*options);
            Request(ctx, flags, false /* delayed */, start_time, thread_stat);
            sequence_stat_[i]->remaining_queries_ = 0;
          }
        }
      }
      if (async_) {
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


void
RequestRateManager::Request(
    std::shared_ptr<InferContextMetaData> context, const uint32_t flags,
    const bool delayed, const struct timespec start_time,
    std::shared_ptr<ThreadStat> thread_stat)
{
  if (async_) {
    thread_stat->status_ = context->ctx_->AsyncRun(
        [context, start_time, flags, delayed, thread_stat](
            nic::InferContext* ctx,
            std::shared_ptr<nic::InferContext::Request> request) {
          std::map<std::string, std::unique_ptr<nic::InferContext::Result>>
              results;
          context->ctx_->GetAsyncRunResults(request, &results);
          struct timespec end_time_async;
          clock_gettime(CLOCK_MONOTONIC, &end_time_async);
          {
            // Add the request timestamp to thread Timestamp vector with
            // proper locking
            std::lock_guard<std::mutex> lock(thread_stat->mu_);
            thread_stat->request_timestamps_.emplace_back(
                std::make_tuple(start_time, end_time_async, flags, delayed));
            context->ctx_->GetStat(&(thread_stat->contexts_stat_[0]));
          }
          context->inflight_request_cnt_--;
        });
    if (!thread_stat->status_.IsOk()) {
      return;
    }
    context->inflight_request_cnt_++;
  } else {
    std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
    struct timespec end_time_sync;
    thread_stat->status_ = context->ctx_->Run(&results);
    if (!thread_stat->status_.IsOk()) {
      return;
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time_sync);
    {
      // Add the request timestamp to thread Timestamp vector with proper
      // locking
      std::lock_guard<std::mutex> lock(thread_stat->mu_);
      thread_stat->request_timestamps_.emplace_back(
          std::make_tuple(start_time, end_time_sync, flags, delayed));
      context->ctx_->GetStat(&(thread_stat->contexts_stat_[0]));
    }
  }
}
