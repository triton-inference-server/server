// Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "infer_payload.h"

namespace triton { namespace backend { namespace python {

InferPayload::InferPayload(
    const bool is_decoupled,
    std::function<void(std::unique_ptr<InferResponse>)> callback)
    : is_decoupled_(is_decoupled), is_promise_set_(false), callback_(callback),
      request_address_(reinterpret_cast<intptr_t>(nullptr))
{
  promise_.reset(new std::promise<std::unique_ptr<InferResponse>>());
}

void
InferPayload::SetValue(std::unique_ptr<InferResponse> infer_response)
{
  {
    // Only set value to the promise with the first response. Call the callback
    // function to send decoupled response to the stub.
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_promise_set_) {
      is_promise_set_ = true;
      promise_->set_value(std::move(infer_response));
      return;
    }
  }
  Callback(std::move(infer_response));
}

void
InferPayload::SetFuture(
    std::future<std::unique_ptr<InferResponse>>& response_future)
{
  response_future = promise_->get_future();
}

bool
InferPayload::IsDecoupled()
{
  return is_decoupled_;
}

bool
InferPayload::IsPromiseSet()
{
  return is_promise_set_;
}

void
InferPayload::Callback(std::unique_ptr<InferResponse> infer_response)
{
  return callback_(std::move(infer_response));
}

void
InferPayload::SetResponseAllocUserp(
    const ResponseAllocatorUserp& response_alloc_userp)
{
  response_alloc_userp_ =
      std::make_shared<ResponseAllocatorUserp>(response_alloc_userp);
}

std::shared_ptr<ResponseAllocatorUserp>
InferPayload::ResponseAllocUserp()
{
  return response_alloc_userp_;
}

void
InferPayload::SetRequestAddress(intptr_t request_address)
{
  std::unique_lock<std::mutex> lock(request_address_mutex_);
  request_address_ = request_address;
}

void
InferPayload::SetRequestCancellationFunc(
    const std::function<void(intptr_t)>& request_cancel_func)
{
  request_cancel_func_ = request_cancel_func;
}

void
InferPayload::SafeCancelRequest()
{
  std::unique_lock<std::mutex> lock(request_address_mutex_);
  if (request_address_ == 0L) {
    return;
  }

  if (request_cancel_func_) {
    request_cancel_func_(request_address_);
  }
}

}}}  // namespace triton::backend::python
