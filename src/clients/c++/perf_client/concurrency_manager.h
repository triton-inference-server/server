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
#pragma once

#include "src/clients/c++/perf_client/context_factory.h"
#include "src/clients/c++/perf_client/load_manager.h"
#include "src/clients/c++/perf_client/perf_utils.h"

#include <condition_variable>
#include <thread>


namespace perfclient {
//==============================================================================
/// ConcurrencyManager is a helper class to send inference requests to inference
/// server consistently, based on the specified setting, so that the perf_client
/// can measure performance under different concurrency.
///
/// An instance of concurrency manager will be created at the beginning of the
/// perf client and it will be used to simulate different load level in respect
/// to number of concurrent infer requests and to collect per-request statistic.
///
/// Detail:
/// Concurrency Manager will maintain the number of concurrent requests by
/// spawning worker threads that keep sending randomly generated requests to the
/// server. The worker threads will record the start time and end
/// time of each request into a shared vector.
///
class ConcurrencyManager : public LoadManager {
 public:
  ~ConcurrencyManager();

  /// Create a concurrency manager that is responsible to maintain specified
  /// load on inference server.
  /// \param batch_size The batch size used for each request.
  /// \param max_threads The maximum number of working threads to be spawned.
  /// \param sequence_length The base length of each sequence.
  /// \param zero_input Whether to fill the input tensors with zero.
  /// \param factory The ContextFactory object used to create InferContext.
  /// \param manger Returns a new ConcurrencyManager object.
  /// \return Error object indicating success or failure.
  static nic::Error Create(
      const int32_t batch_size, const size_t max_threads,
      const size_t sequence_length, const bool zero_input,
      const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
      const std::string& data_directory,
      const std::shared_ptr<ContextFactory>& factory,
      std::unique_ptr<LoadManager>* manager);

  /// @ See LoadManager.ChangeConurrencyLevel()
  nic::Error ChangeConcurrencyLevel(
      const size_t concurrent_request_count) override;

  /// @ See LoadManager.CheckHealth()
  nic::Error CheckHealth() override;

  /// @ See LoadManager.SwapTimestamps()
  nic::Error SwapTimestamps(TimestampVector& new_timestamps) override;

  /// @ See LoadManager.GetAccumulatedContextStat()
  nic::Error GetAccumulatedContextStat(
      nic::InferContext::Stat* contexts_stat) override;

  /// @ See LoadManager.BatchSize()
  size_t BatchSize() const override { return batch_size_; }

 public:
  struct RequestMetaData {
    RequestMetaData(
        const std::shared_ptr<nic::InferContext::Request> request,
        const struct timespec start_time, const uint32_t flags)
        : request_(std::move(request)), start_time_(start_time), flags_(flags)
    {
    }

    const std::shared_ptr<nic::InferContext::Request> request_;
    const struct timespec start_time_;
    const uint32_t flags_;
  };

  struct InferContextMetaData {
    InferContextMetaData() : inflight_request_cnt_(0) {}
    InferContextMetaData(InferContextMetaData&&) = delete;
    InferContextMetaData(const InferContextMetaData&) = delete;

    std::unique_ptr<nic::InferContext> ctx_;
    size_t inflight_request_cnt_;
    // mutex to guard 'completed_requests_' which will be acessed by
    // both the main thread and callback thread
    std::mutex mtx_;
    std::vector<RequestMetaData> completed_requests_;
  };

 private:
  ConcurrencyManager(
      const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
      const int32_t batch_size, const size_t max_threads,
      const size_t sequence_length,
      const std::shared_ptr<ContextFactory>& factory);

  /// Function for worker that sends async inference requests.
  /// \param err Returns the status of the worker
  /// \param stats Returns the statistic of the InferContexts
  /// \param concurrency The concurrency level that the worker should produce.
  void AsyncInfer(
      std::shared_ptr<nic::Error> err,
      std::shared_ptr<std::vector<nic::InferContext::Stat>> stats,
      std::shared_ptr<size_t> concurrency);

  /// Helper function to prepare the InferContext for sending inference request.
  /// \param ctx Returns a new InferContext.
  /// \param options Returns the options used by 'ctx'.
  nic::Error PrepareInfer(
      std::unique_ptr<nic::InferContext>* ctx,
      std::unique_ptr<nic::InferContext::Options>* options);

  /// Generate random sequence length based on 'offset_ratio' and
  /// 'sequence_length_'. (1 +/- 'offset_ratio') * 'sequence_length_'
  /// \param offset_ratio The offset ratio of the generated length
  /// \return random sequence length
  size_t GetRandomLength(double offset_ratio);

  size_t batch_size_;
  size_t max_threads_;
  size_t sequence_length_;

  bool on_sequence_model_;

  std::shared_ptr<ContextFactory> factory_;

  // User provided input shape
  std::unordered_map<std::string, std::vector<int64_t>> input_shapes_;

  // User provided input data, it will be preferred over synthetic data
  std::unordered_map<std::string, std::vector<char>> input_data_;

  // Placeholder for generated input data, which will be used for all inputs
  std::vector<uint8_t> input_buf_;

  // Note: early_exit signal is kept global
  std::vector<std::thread> threads_;
  std::vector<std::shared_ptr<nic::Error>> threads_status_;
  std::vector<std::shared_ptr<std::vector<nic::InferContext::Stat>>>
      threads_contexts_stat_;
  std::vector<std::shared_ptr<size_t>> threads_concurrency_;

  // Use condition variable to pause/continue worker threads
  std::condition_variable wake_signal_;
  std::mutex wake_mutex_;

  // Pointer to a vector of request timestamps <start_time, end_time>
  // Request latency will be end_time - start_time
  std::shared_ptr<TimestampVector> request_timestamps_;
  // Mutex to avoid race condition on adding elements into the timestamp vector
  // and on updating context statistic.
  std::mutex status_report_mutex_;
};

}  // namespace perfclient
