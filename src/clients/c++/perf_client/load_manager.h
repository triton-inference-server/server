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
#include "src/clients/c++/perf_client/perf_utils.h"

#include <condition_variable>
#include <thread>

class LoadManager {
 public:
  virtual ~LoadManager();

  /// Check if the load manager is working as expected.
  /// \return Error object indicating success or failure.
  nic::Error CheckHealth();

  /// Swap the content of the timestamp vector recorded by the load
  /// manager with a new timestamp vector
  /// \param new_timestamps The timestamp vector to be swapped.
  /// \return Error object indicating success or failure.
  nic::Error SwapTimestamps(TimestampVector& new_timestamps);

  /// Get the sum of all contexts' stat
  /// \param contexts_stat Returned the accumulated stat from all contexts
  /// in load manager
  nic::Error GetAccumulatedContextStat(nic::InferContext::Stat* contexts_stat);

  /// \return the batch size used for the inference requests
  size_t BatchSize() const { return batch_size_; }

  /// Resets all worker thread states to beginning of schedule.
  /// \return Error object indicating success or failure.
  virtual nic::Error ResetWorkers()
  {
    return nic::Error(
        ni::RequestStatusCode::INTERNAL,
        "resetting worker threads not supported for this load manager.");
  }

  struct InferContextMetaData {
    InferContextMetaData() : inflight_request_cnt_(0) {}
    InferContextMetaData(InferContextMetaData&&) = delete;
    InferContextMetaData(const InferContextMetaData&) = delete;

    std::unique_ptr<nic::InferContext> ctx_;
    std::atomic<size_t> inflight_request_cnt_;
  };


 protected:
  LoadManager(
      const bool async,
      const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
      const int32_t batch_size, const size_t max_threads,
      const size_t sequence_length, const SharedMemoryType shared_memory_type,
      const size_t output_shm_size,
      const std::shared_ptr<ContextFactory>& factory);

  /// Helper funtion to retrieve the input data for the inferences
  /// \param string_length The length of the random strings to be generated
  /// for string inputs.
  /// \param string_data The string to be used as string inputs for model.
  /// \param zero_input Whether to use zero for model inputs.
  /// \param data_directory The path to the directory containing the input data
  /// in binary or text files.
  /// \return Error object indicating success or failure.
  nic::Error InitManagerInputs(
      const size_t string_length, const std::string& string_data,
      const bool zero_input, const std::string& data_directory);

  /// Helper function to allocate and prepare shared memory.
  /// from shared memory.
  /// \return Error object indicating success or failure.
  nic::Error InitSharedMemory();

  /// Helper function to prepare the InferContext for sending inference request.
  /// \param ctx Returns a new InferContext.
  /// \param options Returns the options used by 'ctx'.
  nic::Error PrepareInfer(
      std::unique_ptr<nic::InferContext>* ctx,
      std::unique_ptr<nic::InferContext::Options>* options);

  /// Helper function to prepare the InferContext for sending inference request
  /// in shared memory.
  /// \param ctx Returns a new InferContext.
  /// \param options
  /// Returns the options used by 'ctx'.
  nic::Error PrepareSharedMemoryInfer(
      std::unique_ptr<nic::InferContext>* ctx,
      std::unique_ptr<nic::InferContext::Options>* options);

  /// Generate random sequence length based on 'offset_ratio' and
  /// 'sequence_length_'. (1 +/- 'offset_ratio') * 'sequence_length_'
  /// \param offset_ratio The offset ratio of the generated length
  /// \return random sequence length
  size_t GetRandomLength(double offset_ratio);

  /// Stops all the worker threads generating the request load.
  void StopWorkerThreads();

 private:
  /// Helper function to access data for the specified input
  /// \param input The target input
  /// Returns the pointer to the memory holding data
  nic::Error GetInputData(
      std::shared_ptr<nic::InferContext::Input> input, const uint8_t** data,
      size_t* batch1_size);

 protected:
  bool async_;
  // User provided input shape
  std::unordered_map<std::string, std::vector<int64_t>> input_shapes_;
  size_t batch_size_;
  size_t max_threads_;

  size_t sequence_length_;
  SharedMemoryType shared_memory_type_;
  size_t output_shm_size_;
  bool on_sequence_model_;

  std::shared_ptr<ContextFactory> factory_;

  // User provided input data, it will be preferred over synthetic data
  std::unordered_map<std::string, std::vector<char>> input_data_;
  std::unordered_map<std::string, std::vector<char>> input_string_data_;

  // Placeholder for generated input data, which will be used for all inputs
  // except string
  std::vector<uint8_t> input_buf_;

  std::unique_ptr<nic::SharedMemoryControlContext> shared_memory_ctx_;

  // Map from shared memory key to its starting address and size
  std::unordered_map<std::string, std::pair<uint8_t*, size_t>>
      shared_memory_regions_;

  struct ThreadStat {
    ThreadStat() : status_(ni::RequestStatusCode::SUCCESS) {}

    // The status of the worker thread
    nic::Error status_;
    // The statistics of the InferContext
    std::vector<nic::InferContext::Stat> contexts_stat_;
    //  The concurrency level that the worker should produce
    size_t concurrency_;
    // A vector of request timestamps <start_time, end_time>
    // Request latency will be end_time - start_time
    TimestampVector request_timestamps_;
    // A lock to protect thread data
    std::mutex mu_;
  };

  // Worker threads that loads the server with inferences
  std::vector<std::thread> threads_;
  // Contains the statistics on the current working threads
  std::vector<std::shared_ptr<ThreadStat>> threads_stat_;

  // Use condition variable to pause/continue worker threads
  std::condition_variable wake_signal_;
  std::mutex wake_mutex_;
};
