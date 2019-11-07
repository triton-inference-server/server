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


//==============================================================================
/// RealisticLoadManager is a helper class to send inference requests to
/// inference server in accordance with a Poisson distribution. This
/// distribution models the real-world traffic patterns.
///
/// An instance of this load manager will be created at the beginning of the
/// perf client and it will be used to simulate load with different target
/// requests per second values and to collect per-request statistic.
///
/// Detail:
/// Realistic Load Manager will try to follow a pre-computed schedule while
/// issuing requests to the server. The manager might spawn additional worker
/// thread to meet the timeline imposed by the schedule. The worker threads will
/// record the start time and end time of each request into a shared vector
/// which will be used to report the observed latencies in serving requests.
/// Additionally, they will report a vector of the number of requests missed
/// their schedule.
///
class RealisticLoadManager : public LoadManager {
 public:
  ~RealisticLoadManager();

  /// Create an object of realistic load manager that is responsible to maintain
  /// specified load on inference server.
  /// \param async Whether to use asynchronous or synchronous API for infer
  /// request.
  /// \param batch_size The batch size used for each request.
  /// \param max_threads The maximum number of working threads to be spawned.
  /// \param sequence_length The base length of each sequence.
  /// \param zero_input Whether to fill the input tensors with zero.
  /// \param factory The ContextFactory object used to create InferContext.
  /// \param manager Returns a new ConcurrencyManager object.
  /// \return Error object indicating success or failure.
  static nic::Error Create(
      const bool async, const int32_t batch_size, const size_t max_threads,
      const size_t sequence_length,
      const size_t string_length, const std::string& string_data,
      const bool zero_input,
      const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
      const std::string& data_directory,
      const std::shared_ptr<ContextFactory>& factory,
      std::unique_ptr<LoadManager>* manager);
