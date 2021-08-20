// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "src/backends/backend/triton_model_instance.h"
#include "src/core/status.h"


namespace nvidia { namespace inferenceserver {

class Payload {
 public:
  enum Operation { INFER_RUN = 0, INIT = 1, WARM_UP = 2, EXIT = 3 };
  enum State {
    UNINITIALIZED = 0,
    READY = 1,
    REQUESTED = 2,
    SCHEDULED = 3,
    EXECUTING = 4,
    RELEASED = 5
  };

  Payload();
  void Reset(const Operation op_type, TritonModelInstance* instance = nullptr);
  Operation GetOpType() { return op_type_; }
  std::mutex* GetExecMutex() { return exec_mu_.get(); }
  size_t RequestCount() { return requests_.size(); }
  size_t BatchSize();
  void ReserveRequests(size_t size);
  void AddRequest(std::unique_ptr<InferenceRequest> request);
  void SetCallback(std::function<void()> OnCallback);
  void Callback();
  void SetInstance(TritonModelInstance* model_instance);
  TritonModelInstance* GetInstance() { return instance_; }

  State GetState() { return state_; }
  void SetState(State state);
  void Execute(bool* should_exit);
  Status Wait();
  void Release();

 private:
  Operation op_type_;
  std::vector<std::unique_ptr<InferenceRequest>> requests_;
  std::function<void()> OnCallback_;
  TritonModelInstance* instance_;
  State state_;
  std::unique_ptr<std::promise<Status>> status_;
  std::unique_ptr<std::mutex> exec_mu_;
};

}}  // namespace nvidia::inferenceserver