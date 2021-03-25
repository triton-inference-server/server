// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "src/backends/backend/triton_model.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferenceRequest;
class TritonModel;
class TritonModelInstance;


class TritonBackendThreadPool {
 public:
  // The prototype for the initialization function that will be called
  // by the "standard" backend threads created based on a model's
  // scheduling_choice settings. The init function is called once by
  // the runner that will later execute requests for 'instance'. A
  // non-OK error status indicates an initialization error that
  // prevents scheduler from using the runner.
  using StandardInitFuncV2 = std::function<Status(void* instance)>;

  // The prototype for the warmup function that will be called by the
  // "standard" backend threads created based on a model's
  // scheduling_choice settings. The warmup function is called once by
  // the runner that will later execute requests for 'instance'. A
  // non-OK error status indicates an error that prevents scheduler
  // from sending warmup requests to the runner.
  using StandardWarmupFuncV2 = std::function<Status(void* instance)>;

  // The prototype for the run function that will be called by the
  // "standard" schedulers created based on a model's
  // scheduling_choice settings. The run function must accept a
  // 'instance' indicating which model instance should execute the
  // 'requests'. Ownership of the 'requests' is transferred to the
  // runner which is responsible for generating responses and
  // releasing the requests.
  using StandardRunFuncV2 = std::function<void(
      void* instance,
      std::vector<std::unique_ptr<InferenceRequest>>&& requests)>;

  static Status CreateThreadPool(
      const TritonModel* triton_model, const StandardInitFuncV2 OnInit,
      const StandardRunFuncV2 OnRun,
      std::unique_ptr<TritonBackendThreadPool>* threadpool);
  ~TritonBackendThreadPool();

  Status SubmitRequest(
      TritonModelInstance* instance,
      std::vector<std::unique_ptr<InferenceRequest>>&& requests,
      std::function<void()> callback_fn);

 private:
  TritonBackendThreadPool(
      const int nice, const StandardInitFuncV2& OnInit,
      const StandardRunFuncV2& OnRun);

  struct BackendThreadContext {
    BackendThreadContext() {}

    std::mutex mtx_;
    std::condition_variable cv_;
    std::unique_ptr<std::vector<std::unique_ptr<InferenceRequest>>> requests_;
    std::unique_ptr<std::thread> thread_;
  };

  void BackendThread(
      TritonModelInstance* instance, BackendThreadContext* rthread_context,
      std::promise<bool>* is_initialized);

  int nice_;
  StandardInitFuncV2 OnInit_;
  StandardWarmupFuncV2 OnWarmup_;
  StandardRunFuncV2 OnRun_;
  bool signal_exit_;

  std::map<TritonModelInstance*, std::unique_ptr<BackendThreadContext>>
      backend_thread_contexts_;
};

}}  // namespace nvidia::inferenceserver