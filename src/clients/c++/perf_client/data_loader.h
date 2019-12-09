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

  size_t GetTotalStepsNonSequence() { return max_non_sequence_step_id_; }

  size_t GetDataStreamsCount() { return data_stream_cnt_; }

  size_t GetTotalSteps(size_t stream_id)
  {
    if (stream_id < data_stream_cnt_) {
      return step_num_[stream_id];
    }
    return 0;
  }
  /// Helper function to access data for the specified input
  /// \param input The target input
  /// Returns the pointer to the memory holding data
  nic::Error GetInputData(
      std::shared_ptr<nic::InferContext::Input> input, const uint8_t** data,
      size_t* batch1_size, const int step_id = 0, const int sequence_id = 0);

  /// Reads the input data from the specified data directory
  /// \param input The input for which the data is to be read
  /// \param data_directory The path to the directory containing the data
  /// \param index The index to allocate for the read data
  nic::Error ReadDataFromDir(
      std::vector<std::shared_ptr<nic::InferContext::Input>> inputs,
      const std::string& data_directory);

  /// Reads the input data from the specified json file
  /// \param input The input for which the data is to be read
  /// \param data_directory The path to the directory containing the data
  /// \param index The index to allocate for the read data
  nic::Error ReadDataFromJSON(
      std::vector<std::shared_ptr<nic::InferContext::Input>> inputs,
      const std::string& data_directory);

  /// Generates the input data for specified input
  /// \param input The target input
  nic::Error GenerateData(
      std::vector<std::shared_ptr<nic::InferContext::Input>> inputs,
      const bool zero_input, const size_t string_length,
      const std::string& string_data);

 private:
  size_t batch_size_;
  size_t data_stream_cnt_;
  std::vector<size_t> step_num_;
  size_t max_non_sequence_step_id_;

  // User provided input data, it will be preferred over synthetic data
  std::unordered_map<std::string, std::vector<char>> input_data_;

  // Placeholder for generated input data, which will be used for all inputs
  // except string
  std::vector<uint8_t> input_buf_;
};