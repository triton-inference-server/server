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

#include "src/clients/c++/perf_client/perf_utils.h"

namespace perfclient {

class LoadManager {
 public:
  /// Virtual destructor for well defined cleanup
  virtual ~LoadManager(){};

  /// Adjust the number of concurrent requests to be the same as
  /// 'concurrent_request_count' (by creating threads or by pausing threads)
  /// \parm concurent_request_count The number of concurrent requests.
  /// \return Error object indicating success or failure.
  virtual nic::Error ChangeConcurrencyLevel(
      const size_t concurrent_request_count) = 0;

  /// Check if the load manager is working as expected.
  /// \return Error object indicating success or failure.
  virtual nic::Error CheckHealth() = 0;

  /// Swap the content of the timestamp vector recorded by the load
  /// manager with a new timestamp vector
  /// \param new_timestamps The timestamp vector to be swapped.
  /// \return Error object indicating success or failure.
  virtual nic::Error SwapTimestamps(TimestampVector& new_timestamps) = 0;

  /// Get the sum of all contexts' stat
  /// \param contexts_stat Returned the accumulated stat from all contexts
  /// in load manager
  virtual nic::Error GetAccumulatedContextStat(
      nic::InferContext::Stat* contexts_stat) = 0;

  /// \return the batch size used for the inference requests
  virtual size_t BatchSize() const = 0;
};

}  // namespace perfclient
