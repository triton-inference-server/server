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

#include "src/clients/c++/experimental_api_v2/perf_client/data_loader.h"
#include "src/clients/c++/experimental_api_v2/perf_client/model_parser.h"
#include "src/clients/c++/experimental_api_v2/perf_client/perf_utils.h"

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
  nic::Error GetAccumulatedClientStat(nic::InferStat* contexts_stat);

  /// \return the batch size used for the inference requests
  size_t BatchSize() const { return batch_size_; }

  /// Resets all worker thread states to beginning of schedule.
  /// \return Error object indicating success or failure.
  virtual nic::Error ResetWorkers()
  {
    return nic::Error(
        "resetting worker threads not supported for this load manager.");
  }

  struct InferContextMetaData {
    explicit InferContextMetaData() : inflight_request_cnt_(0) {}
    InferContextMetaData(InferContextMetaData&&) = delete;
    InferContextMetaData(const InferContextMetaData&) = delete;
    ~InferContextMetaData()
    {
      for (const auto input : inputs_) {
        delete input;
      }
      for (const auto output : outputs_) {
        delete output;
      }
    }

    std::unique_ptr<TritonClientWrapper> infer_client_;
    std::vector<nic::InferInput*> inputs_;
    std::vector<const nic::InferRequestedOutput*> outputs_;
    std::unique_ptr<nic::InferOptions> options_;
    std::atomic<size_t> inflight_request_cnt_;
  };


 protected:
  LoadManager(
      const bool async, const int32_t batch_size, const size_t max_threads,
      const size_t sequence_length, const SharedMemoryType shared_memory_type,
      const size_t output_shm_size, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<TritonClientFactory>& factory);

  /// Helper funtion to retrieve the input data for the inferences
  /// \param string_length The length of the random strings to be generated
  /// for string inputs.
  /// \param string_data The string to be used as string inputs for model.
  /// \param zero_input Whether to use zero for model inputs.
  /// \param user_data The vector containing path/paths to user-provided data
  /// that can be a directory or path to a json data file.
  /// \return Error object indicating success or failure.
  nic::Error InitManagerInputs(
      const size_t string_length, const std::string& string_data,
      const bool zero_input, std::vector<std::string>& user_data);

  /// Helper function to allocate and prepare shared memory.
  /// from shared memory.
  /// \return Error object indicating success or failure.
  nic::Error InitSharedMemory();

  /// Helper function to prepare the InferContext for sending inference request.
  /// \param ctx Returns a new InferContext.
  /// \param options Returns the options used by 'ctx'.
  /// \return Error object indicating success or failure.
  nic::Error PrepareInfer(InferContextMetaData* ctx);


  /// Helper function to prepare the InferContext for sending inference
  /// request
  /// in shared memory.
  /// \param ctx Returns a new InferContext.
  /// \param options
  /// Returns the options used by 'ctx'.
  nic::Error PrepareSharedMemoryInfer(InferContextMetaData* ctx);


  /// Updates the input data to use for inference request
  /// \param inputs The inputs to the model
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return Error object indicating success or failure.
  nic::Error UpdateInputs(
      std::vector<nic::InferInput*>& inputs, int stream_index, int step_index);

  void InitNewSequence(int sequence_id);

  /// Generate random sequence length based on 'offset_ratio' and
  /// 'sequence_length_'. (1 +/- 'offset_ratio') * 'sequence_length_'
  /// \param offset_ratio The offset ratio of the generated length
  /// \return random sequence length
  size_t GetRandomLength(double offset_ratio);

  /// Stops all the worker threads generating the request load.
  void StopWorkerThreads();

 private:
  /// Helper function to update the inputs
  /// \param inputs The inputs to the model
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return Error object indicating success or failure.
  nic::Error SetInputs(
      const std::vector<nic::InferInput*>& inputs, const int stream_index,
      const int step_index);

  /// Helper function to update the shared memory inputs
  /// \param inputs The inputs to the model
  /// \param stream_index The data stream to use for next data
  /// \param step_index The step index to use for next data
  /// \return Error object indicating success or failure.
  nic::Error SetInputsSharedMemory(
      const std::vector<nic::InferInput*>& inputs, const int stream_index,
      const int step_index);

 protected:
  bool async_;
  size_t batch_size_;
  size_t max_threads_;
  size_t sequence_length_;
  SharedMemoryType shared_memory_type_;
  size_t output_shm_size_;
  bool on_sequence_model_;

  std::shared_ptr<ModelParser> parser_;
  std::shared_ptr<TritonClientFactory> factory_;

  bool using_json_data_;

  std::unique_ptr<DataLoader> data_loader_;
  std::unique_ptr<TritonClientWrapper> client_;

  // Map from shared memory key to its starting address and size
  std::unordered_map<std::string, std::pair<uint8_t*, size_t>>
      shared_memory_regions_;

  // Holds the running status of the thread.
  struct ThreadStat {
    ThreadStat() {}

    // The status of the worker thread
    nic::Error status_;
    // The status of the callback thread for async requests
    nic::Error cb_status_;
    // The statistics of the InferContext
    std::vector<nic::InferStat> contexts_stat_;
    // The concurrency level that the worker should produce
    size_t concurrency_;
    // A vector of request timestamps <start_time, end_time>
    // Request latency will be end_time - start_time
    TimestampVector request_timestamps_;
    // A lock to protect thread data
    std::mutex mu_;
  };

  // Holds the status of the inflight sequence
  struct SequenceStat {
    SequenceStat(uint64_t seq_id)
        : seq_id_(seq_id), data_stream_id_(0), remaining_queries_(0)
    {
    }
    // The unique correlation id allocated to the sequence
    uint64_t seq_id_;
    // The data stream id providing data for the sequence
    uint64_t data_stream_id_;
    // The number of queries remaining to complete the sequence
    size_t remaining_queries_;
    // A lock to protect sequence data
    std::mutex mtx_;
  };

  std::vector<std::shared_ptr<SequenceStat>> sequence_stat_;
  std::atomic<uint64_t> next_seq_id_;

  // Worker threads that loads the server with inferences
  std::vector<std::thread> threads_;
  // Contains the statistics on the current working threads
  std::vector<std::shared_ptr<ThreadStat>> threads_stat_;

  // Use condition variable to pause/continue worker threads
  std::condition_variable wake_signal_;
  std::mutex wake_mutex_;
};
