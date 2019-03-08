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

#include "src/core/backend.h"
#include "src/core/model_config.pb.h"
#include "src/core/scheduler.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

class EnsembleBundle : public InferenceBackend {
 public:
  EnsembleBundle() = default;
  EnsembleBundle(EnsembleBundle&&) = default;

  tensorflow::Status Init(
      const tensorflow::StringPiece& path, const ModelConfig& config,
      uint64_t inference_server);

 private:
  // Run model on the context associated with 'runner_idx' to
  // execute for one or more requests.
  void Run(
      uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
      std::function<void(tensorflow::Status)> OnCompleteQueuedPayloads);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(EnsembleBundle);
  friend std::ostream& operator<<(std::ostream&, const EnsembleBundle&);

  // [TODO] remove this after DLIS-290 is done,
  // it can be directly pass to scheduler on creation
  uintptr_t inference_server_;
};

std::ostream& operator<<(std::ostream& out, const EnsembleBundle& pb);

}}  // namespace nvidia::inferenceserver
