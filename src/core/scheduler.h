// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <functional>
#include "src/core/api.pb.h"
#include "src/core/infer_request.h"
#include "src/core/infer_stats.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// Scheduler interface.
class Scheduler {
 public:
  virtual ~Scheduler() {}

  // The prototype for the initialization function that will be called
  // by the "standard" schedulers created based on a model's
  // scheduling_choice settings. The init function is called once by
  // the runner that will later execute requests for 'runner_idx'. A
  // non-OK error status indicates an initialization error that
  // prevents scheduler from using the runner.
  using StandardInitFunc = std::function<Status(uint32_t runner_idx)>;

  // The prototype for the warmup function that will be called by the
  // "standard" schedulers created based on a model's
  // scheduling_choice settings. The warmup function is called once by
  // the runner that will later execute requests for 'runner_idx'. A
  // non-OK error status indicates an error that prevents scheduler
  // from sending warmup requests to the runner.
  using StandardWarmupFunc = std::function<Status(uint32_t runner_idx)>;

  // The prototype for the run function that will be called by the
  // "standard" schedulers created based on a model's
  // scheduling_choice settings. The run function must accept a
  // 'runner_idx' indicating which runner should execute the
  // 'requests'. Ownership of the 'requests' is transferred to the
  // runner which is responsible for generating responses and
  // releasing the requests.
  using StandardRunFunc = std::function<void(
      uint32_t runner_idx,
      std::vector<std::unique_ptr<InferenceRequest>>&& requests)>;

  // The prototype for the shape-tensor peek function that can be
  // called by the "standard" schedulers created based on a model's
  // scheduling_choice settings. The peek function can be called to
  // get the contents of a shape tensor. A non-OK error status
  // indicates that the peek failed.
  using StandardShapeTensorPeekFunc = std::function<Status(
      uint32_t runner_idx, const InferenceRequest::Input& input,
      const std::unique_ptr<InferenceRequest>& request,
      std::vector<int64_t>* shape)>;

  // Enqueue a request with the scheduler. If Status::Success is returned
  // then the backend has taken ownership of the request object and so
  // 'request' will be nullptr. If non-success is returned then the
  // caller still retains ownership of 'request'.
  virtual Status Enqueue(std::unique_ptr<InferenceRequest>& request) = 0;
};

}}  // namespace nvidia::inferenceserver
