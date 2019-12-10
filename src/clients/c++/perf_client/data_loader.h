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

#include <fstream>
#include "src/clients/c++/perf_client/perf_utils.h"

class DataLoader {
 public:
  DataLoader(size_t batch_size);

  /// Returns the total number of data steps that can be supported by a
  /// non-sequence model.
  size_t GetTotalStepsNonSequence() { return max_non_sequence_step_id_; }

  /// Returns the total number of data streams available.
  size_t GetDataStreamsCount() { return data_stream_cnt_; }

  /// Returns the total data steps supported for a requested data stream
  /// id.
  /// \param stream_id The target stream id
  size_t GetTotalSteps(size_t stream_id)
  {
    if (stream_id < data_stream_cnt_) {
      return step_num_[stream_id];
    }
    return 0;
  }

  /// Reads the input data from the specified data directory
  /// \param inputs The vector of inputs to the target model.
  /// \param data_directory The path to the directory containing the data
  nic::Error ReadDataFromDir(
      std::vector<std::shared_ptr<nic::InferContext::Input>> inputs,
      const std::string& data_directory);

  /// Reads the input data from the specified json file and append to the
  /// stream buffers.
  /// \param inputs The vector of inputs to the target model.
  /// \param json_file The json file containing the user-provided input
  /// data.
  /// Returns error object indicating status
  nic::Error ReadDataFromJSON(
      std::vector<std::shared_ptr<nic::InferContext::Input>> inputs,
      const std::string& json_file);

  /// Generates the input data to use with the inference requests
  /// \param inputs The vector of inputs to the target model.
  /// \param zero_input Whether or not to use zero value for buffer
  /// initialization.
  /// \param string_length The length of the string to generate for
  /// tensor inputs.
  /// \param string_data The user provided string to use to populate
  /// string tensors
  /// Returns error object indicating status
  nic::Error GenerateData(
      std::vector<std::shared_ptr<nic::InferContext::Input>> inputs,
      const bool zero_input, const size_t string_length,
      const std::string& string_data);

  /// Helper function to access data for the specified input
  /// \param input The target input
  /// \param stream_id The data stream_id to use for retrieving input data.
  /// \param step_id The data step_id to use for retrieving input data.
  /// \param data Returns the pointer to the data for the requested input.
  /// \param batch1_size Returns the size of the input data in bytes.
  /// Returns error object indicating status
  nic::Error GetInputData(
      std::shared_ptr<nic::InferContext::Input> input, const int stream_id,
      const int step_id, const uint8_t** data_ptr, size_t* batch1_size);


 private:
  /// Helper function to read data for the specified input from json
  /// \param step the DOM for current step
  /// \param inputs The inputs to the model
  /// \param stream_index the stream index the data should be exported to.
  /// \param step_index the step index the data should be exported to.
  /// Returns error object indicating status
  nic::Error ReadInputTensorData(
      const rapidjson::Value& step,
      std::vector<std::shared_ptr<nic::InferContext::Input>>& inputs,
      int stream_index, int step_index);


  // The batch_size_ for the data
  size_t batch_size_;
  // The total number of data streams available.
  size_t data_stream_cnt_;
  // A vector containing the supported step number for respective stream ids.
  std::vector<size_t> step_num_;
  // The maximum supported data step id for non-sequence model.
  size_t max_non_sequence_step_id_;

  // User provided input data, it will be preferred over synthetic data
  std::unordered_map<std::string, std::vector<char>> input_data_;

  // Placeholder for generated input data, which will be used for all inputs
  // except string
  std::vector<uint8_t> input_buf_;
};