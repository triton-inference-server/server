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

#include "src/clients/c++/perf_client/request_rate_manager.h"

//==============================================================================
/// CustomLoadManager is a helper class to send inference requests to
/// inference server in accordance with  user provided time intervals. This
/// load manager can be used to model certain patterns of interest.
///
class CustomLoadManager : public RequestRateManager {
 public:
  ~CustomLoadManager() = default;

  /// Create an object of realistic load manager that is responsible to maintain
  /// specified load on inference server.
  /// \param async Whether to use asynchronous or synchronous API for infer
  /// request.
  /// \param measurement_window_ms The time window for measurements.
  /// \param request_intervals_file The path to the file to use to pick up the
  /// time intervals between the successive requests.
  /// \param batch_size The batch size used for each request.
  /// \param max_threads The maximum number of working threads to be spawned.
  /// \param num_of_sequences The number of concurrent sequences that must be
  /// maintained on the server.
  /// \param sequence_length The base length of each sequence.
  /// \param zero_input Whether to fill the input tensors with zero.
  /// \param input_shapes The shape of the input tensors.
  /// \param user_data The vector containing path/paths to user-provided data
  /// that can be a directory or path to a json data file.
  /// \param shared_memory_type The type of shared memory to use for inputs.
  /// \param output_shm_size The size of the shared memory to allocate for the
  /// output.
  /// \param factory The ContextFactory object used to create
  /// InferContext.
  /// \param manager Returns a new ConcurrencyManager object.
  /// \return Error object indicating success or failure.
  static nic::Error Create(
      const bool async, const uint64_t measurement_window_ms,
      const std::string& request_intervals_file, const int32_t batch_size,
      const size_t max_threads, const uint32_t num_of_sequences,
      const size_t sequence_length, const size_t string_length,
      const std::string& string_data, const bool zero_input,
      const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
      std::vector<std::string>& user_data,
      const SharedMemoryType shared_memory_type, const size_t output_shm_size,
      const std::shared_ptr<ContextFactory>& factory,
      std::unique_ptr<LoadManager>* manager);


  /// Initializes the load manager with the provided file containing request
  /// intervals
  /// \return Error object indicating success or failure.
  nic::Error InitCustomIntervals();

  /// Computes the request rate from the time interval file. Fails with an error
  /// if the file is not present or is empty.
  /// \param request_rate Returns request rate as computed from the time
  /// interval file.
  /// \return Error object indicating success or failure.
  nic::Error GetCustomRequestRate(double* request_rate);

 private:
  CustomLoadManager(
      const bool async,
      const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
      const std::string& request_intervals_file, const int32_t batch_size,
      const uint64_t measurement_window_ms, const size_t max_threads,
      const uint32_t num_of_sequences, const size_t sequence_length,
      const SharedMemoryType shared_memory_type, const size_t output_shm_size,
      const std::shared_ptr<ContextFactory>& factory);

  std::string request_intervals_file_;
  std::vector<std::chrono::nanoseconds> custom_intervals_;
};
