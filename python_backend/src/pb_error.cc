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

#include "pb_error.h"

namespace triton { namespace backend { namespace python {

TRITONSERVER_Error_Code
PbError::Code()
{
  return code_;
}

const std::string&
PbError::Message()
{
  return message_;
}

bi::managed_external_buffer::handle_t
PbError::ShmHandle()
{
  return shm_handle_;
}

void
PbError::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  message_shm_ = PbString::Create(shm_pool, message_);
  error_shm_ = shm_pool->Construct<PbErrorShm>();
  error_shm_.data_->code = code_;
  error_shm_.data_->message_shm_handle = message_shm_->ShmHandle();
  shm_handle_ = error_shm_.handle_;
}

std::shared_ptr<PbError>
PbError::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t shm_handle)
{
  AllocatedSharedMemory<PbErrorShm> error_shm =
      shm_pool->Load<PbErrorShm>(shm_handle);
  std::unique_ptr<PbString> message_shm = PbString::LoadFromSharedMemory(
      shm_pool, error_shm.data_->message_shm_handle);

  TRITONSERVER_Error_Code code = error_shm.data_->code;
  std::string message = message_shm->String();

  return std::shared_ptr<PbError>(new PbError(
      std::move(message_shm), std::move(error_shm), code, std::move(message)));
}

PbError::PbError(
    std::shared_ptr<PbString>&& message_shm,
    AllocatedSharedMemory<PbErrorShm>&& error_shm, TRITONSERVER_Error_Code code,
    std::string&& message)
    : message_shm_(std::move(message_shm)), error_shm_(std::move(error_shm)),
      code_(code), message_(std::move(message))
{
}

}}}  // namespace triton::backend::python
