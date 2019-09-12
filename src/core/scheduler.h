// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/server_status.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferRequestProvider;
class InferResponseProvider;
class ModelInferStats;

// Scheduler interface.
class Scheduler {
 public:
  virtual ~Scheduler() {}

  // The data associated with each request being scheduled.
  struct Payload {
    Payload() = default;
    Payload(const Payload& payload) = delete;
    Payload(Payload&& payload)
        : stats_(std::move(payload.stats_)),
          request_provider_(std::move(payload.request_provider_)),
          response_provider_(std::move(payload.response_provider_)),
          complete_function_(std::move(payload.complete_function_)),
          status_(payload.status_)
    {
    }
    Payload(
        const std::shared_ptr<ModelInferStats>& stats,
        const std::shared_ptr<InferRequestProvider>& request_provider,
        const std::shared_ptr<InferResponseProvider>& response_provider,
        const std::function<void(const Status&)> complete_function)
        : stats_(stats), request_provider_(request_provider),
          response_provider_(response_provider),
          complete_function_(complete_function), status_(Status::Success)
    {
    }

    std::shared_ptr<ModelInferStats> stats_;
    std::shared_ptr<InferRequestProvider> request_provider_;
    std::shared_ptr<InferResponseProvider> response_provider_;
    std::function<void(const Status&)> complete_function_;
    Status status_;
  };

  // The prototype for the initialization function that will be called
  // by the "standard" schedulers created based on a model's
  // scheduling_choice settings. The init function is called once by
  // the runner that will later execute payloads for 'runner_idx'. A
  // non-OK error status indicates an initialization error that
  // prevents scheduler from using the runner.
  using StandardInitFunc = std::function<Status(uint32_t runner_idx)>;

  // The prototype for the run function that will be called by the
  // "standard" schedulers created based on a model's
  // scheduling_choice settings. The run function must accept a
  // 'runner_idx' indicating which runner should execute the
  // 'payloads'. When the execution completes the runner must call the
  // 'OnRunComplete' function with error status. A non-OK error status
  // indicates an internal error that prevents any of the of
  // 'payloads' requests from completing. If an error is isolated to a
  // single request in 'payloads' it will be reported in that payload.
  using StandardRunFunc = std::function<void(
      uint32_t runner_idx, std::vector<Payload>* payloads,
      std::function<void(const Status&)> OnRunComplete)>;

  // Enqueue a request with the scheduler.
  virtual void Enqueue(
      const std::shared_ptr<ModelInferStats>& stats,
      const std::shared_ptr<InferRequestProvider>& request_provider,
      const std::shared_ptr<InferResponseProvider>& response_provider,
      std::function<void(const Status&)> OnComplete) = 0;
};

}}  // namespace nvidia::inferenceserver
