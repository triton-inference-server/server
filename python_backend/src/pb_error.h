// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "pb_string.h"
#include "pb_utils.h"

namespace triton { namespace backend { namespace python {

struct PbErrorShm {
  TRITONSERVER_Error_Code code;
  bi::managed_external_buffer::handle_t message_shm_handle;
};

class PbError {
 public:
  PbError(
      const std::string& message,
      TRITONSERVER_Error_Code code = TRITONSERVER_ERROR_INTERNAL)
      : code_(code), message_(message)
  {
  }
  DISALLOW_COPY_AND_ASSIGN(PbError);

  TRITONSERVER_Error_Code Code();
  const std::string& Message();

  void SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool);
  bi::managed_external_buffer::handle_t ShmHandle();

  static std::shared_ptr<PbError> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);

 private:
  PbError(
      std::shared_ptr<PbString>&& message_shm,
      AllocatedSharedMemory<PbErrorShm>&& error_shm,
      TRITONSERVER_Error_Code code, std::string&& message);

  std::shared_ptr<PbString> message_shm_;
  AllocatedSharedMemory<PbErrorShm> error_shm_;
  bi::managed_external_buffer::handle_t shm_handle_;

  TRITONSERVER_Error_Code code_;
  std::string message_;
};

}}};  // namespace triton::backend::python
