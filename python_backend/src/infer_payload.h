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

#pragma once

#include <functional>
#include <queue>

#include "infer_response.h"
#include "pb_preferred_memory.h"

namespace triton { namespace backend { namespace python {

struct ResponseAllocatorUserp {
  ResponseAllocatorUserp(
      void* shm_pool, const PreferredMemory& preferred_memory)
      : shm_pool(shm_pool), preferred_memory(preferred_memory)
  {
  }
  void* shm_pool;
  PreferredMemory preferred_memory;
};

class InferPayload : public std::enable_shared_from_this<InferPayload> {
 public:
  InferPayload(
      const bool is_decouple,
      std::function<void(std::unique_ptr<InferResponse>)> callback);

  /// GetPtr should be only called when the InferPayload object is constructed
  /// using a shared pointer. Calling this function in any other circumstance
  /// is undefined behaviour until C++17.
  std::shared_ptr<InferPayload> GetPtr() { return shared_from_this(); }
  void SetValue(std::unique_ptr<InferResponse> infer_response);
  void SetFuture(std::future<std::unique_ptr<InferResponse>>& response_future);
  bool IsDecoupled();
  bool IsPromiseSet();
  void Callback(std::unique_ptr<InferResponse> infer_response);
  void SetResponseAllocUserp(
      const ResponseAllocatorUserp& response_alloc_userp);
  std::shared_ptr<ResponseAllocatorUserp> ResponseAllocUserp();
  void SetRequestAddress(intptr_t request_address);
  void SetRequestCancellationFunc(
      const std::function<void(intptr_t)>& request_cancel_func);
  void SafeCancelRequest();

 private:
  std::unique_ptr<std::promise<std::unique_ptr<InferResponse>>> promise_;
  bool is_decoupled_;
  std::mutex mutex_;
  bool is_promise_set_;
  std::function<void(std::unique_ptr<InferResponse>)> callback_;
  std::shared_ptr<ResponseAllocatorUserp> response_alloc_userp_;
  std::mutex request_address_mutex_;
  intptr_t request_address_;
  std::function<void(intptr_t)> request_cancel_func_;
};

}}}  // namespace triton::backend::python
