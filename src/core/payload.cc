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

#include "src/core/payload.h"

#include "src/core/infer_request.h"

namespace nvidia { namespace inferenceserver {

Payload::Payload()
    : op_type_(Operation::INFER_RUN),
      requests_(std::vector<std::unique_ptr<InferenceRequest>>()),
      OnCallback_([]() {}), instance_(nullptr), state_(State::UNINITIALIZED)
{
  exec_mu_.reset(new std::mutex());
}

void
Payload::Reset(const Operation op_type, TritonModelInstance* instance)
{
  op_type_ = op_type;
  requests_.clear();
  OnCallback_ = []() {};
  instance_ = instance;
  state_ = State::UNINITIALIZED;
  OnCallback_ = []() {};
  status_.reset(new std::promise<Status>());
}

void
Payload::Release()
{
  op_type_ = Operation::INFER_RUN;
  requests_.clear();
  OnCallback_ = []() {};
  instance_ = nullptr;
  state_ = State::RELEASED;
  OnCallback_ = []() {};
}

size_t
Payload::BatchSize()
{
  size_t batch_size = 0;
  for (const auto& request : requests_) {
    batch_size += std::max(1U, request->BatchSize());
  }
  return batch_size;
}

void
Payload::ReserveRequests(size_t size)
{
  requests_.reserve(size);
}

void
Payload::AddRequest(std::unique_ptr<InferenceRequest> request)
{
  requests_.push_back(std::move(request));
}

void
Payload::SetCallback(std::function<void()> OnCallback)
{
  OnCallback_ = OnCallback;
}

void
Payload::SetInstance(TritonModelInstance* model_instance)
{
  instance_ = model_instance;
}

void
Payload::SetState(Payload::State state)
{
  state_ = state;
}

Status
Payload::Wait()
{
  return status_->get_future().get();
}

void
Payload::Callback()
{
  OnCallback_();
}

void
Payload::Execute(bool* should_exit)
{
  *should_exit = false;

  Status status;
  switch (op_type_) {
    case Operation::INFER_RUN:
      instance_->Schedule(std::move(requests_), OnCallback_);
      break;
    case Operation::INIT:
      status = instance_->Initialize();
      break;
    case Operation::WARM_UP:
      status = instance_->WarmUp();
      break;
    case Operation::EXIT:
      *should_exit = true;
  }

  status_->set_value(status);
}

}}  // namespace nvidia::inferenceserver
