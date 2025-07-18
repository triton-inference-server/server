// Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>
#include <mutex>

#include "infer_response.h"
#include "pb_cancel.h"
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {

class ResponseSender {
 public:
  ResponseSender(
      intptr_t request_address, intptr_t response_factory_address,
      bool const* is_decoupled,
      const std::set<std::string>& requested_output_names,
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      const std::shared_ptr<PbCancel>& pb_cancel);
  intptr_t ResponseFactory() { return response_factory_address_; }
  ~ResponseSender();
  void Send(std::shared_ptr<InferResponse> response, const uint32_t flags);
  bool IsCancelled();
  void UpdateStateAndCounters(InferResponse* response, const uint32_t flags);

  // Can be useful at stopping the model from sending any more responses.
  void Close();
  bool IsClosed();

 private:
  void DeleteResponseFactory();

  intptr_t request_address_;
  intptr_t response_factory_address_;
  bool const* is_decoupled_;
  std::set<std::string> requested_output_names_;
  std::unique_ptr<SharedMemoryManager>& shm_pool_;
  std::shared_ptr<PbCancel> pb_cancel_;

  std::mutex mu_;
  bool closed_;
  size_t number_of_response_sent_;

  std::atomic<bool> response_factory_deleted_;
};
}}}  // namespace triton::backend::python
