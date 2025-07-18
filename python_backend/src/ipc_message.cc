// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ipc_message.h"

#include <memory>

namespace triton { namespace backend { namespace python {
std::unique_ptr<IPCMessage>
IPCMessage::Create(
    const std::unique_ptr<SharedMemoryManager>& shm_pool, bool inline_response)
{
  AllocatedSharedMemory<IPCMessageShm> ipc_message_shm =
      shm_pool->Construct<IPCMessageShm>();

  ipc_message_shm.data_->inline_response = inline_response;
  AllocatedSharedMemory<bi::interprocess_mutex> response_mutex_shm;
  AllocatedSharedMemory<bi::interprocess_condition> response_cond_shm;
  if (inline_response) {
    response_mutex_shm = std::move(shm_pool->Construct<bi::interprocess_mutex>(
        1 /* count */, true /* aligned */));
    response_cond_shm =
        std::move(shm_pool->Construct<bi::interprocess_condition>(
            1 /* count */, true /* aligned */));

    ipc_message_shm.data_->response_mutex = response_mutex_shm.handle_;
    ipc_message_shm.data_->response_cond = response_cond_shm.handle_;
    new (response_mutex_shm.data_.get()) bi::interprocess_mutex{};
    new (response_cond_shm.data_.get()) bi::interprocess_condition{};
  }

  return std::unique_ptr<IPCMessage>(
      new IPCMessage(ipc_message_shm, response_mutex_shm, response_cond_shm));
}

std::unique_ptr<IPCMessage>
IPCMessage::Create(
    IPCMessageShm* ipc_message_shm,
    bi::managed_external_buffer::handle_t& message_handle)
{
  return std::unique_ptr<IPCMessage>(
      new IPCMessage(ipc_message_shm, message_handle));
}

AllocatedSharedMemory<IPCMessageShm>&
IPCMessage::GetAllocatedSharedMemory()
{
  return ipc_message_shm_;
}

std::unique_ptr<IPCMessage>
IPCMessage::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t message_handle)
{
  AllocatedSharedMemory<IPCMessageShm> ipc_message_shm =
      shm_pool->Load<IPCMessageShm>(message_handle);

  AllocatedSharedMemory<bi::interprocess_mutex> response_mutex_shm;
  AllocatedSharedMemory<bi::interprocess_condition> response_cond_shm;
  if (ipc_message_shm.data_->inline_response) {
    response_mutex_shm = shm_pool->Load<bi::interprocess_mutex>(
        ipc_message_shm.data_->response_mutex);
    response_cond_shm = shm_pool->Load<bi::interprocess_condition>(
        ipc_message_shm.data_->response_cond);
  }

  return std::unique_ptr<IPCMessage>(
      new IPCMessage(ipc_message_shm, response_mutex_shm, response_cond_shm));
}

PYTHONSTUB_CommandType&
IPCMessage::Command()
{
  return ipc_message_shm_ptr_->command;
}

bi::managed_external_buffer::handle_t&
IPCMessage::Args()
{
  return ipc_message_shm_ptr_->args;
}

bool&
IPCMessage::InlineResponse()
{
  return ipc_message_shm_ptr_->inline_response;
}

bi::interprocess_condition*
IPCMessage::ResponseCondition()
{
  return response_cond_shm_ptr_;
}

bi::interprocess_mutex*
IPCMessage::ResponseMutex()
{
  return response_mutex_shm_ptr_;
}

bi::managed_external_buffer::handle_t&
IPCMessage::ResponseHandle()
{
  return ipc_message_shm_ptr_->response_handle;
}

bi::managed_external_buffer::handle_t
IPCMessage::ShmHandle()
{
  return ipc_message_handle_;
}

IPCMessage::IPCMessage(
    AllocatedSharedMemory<IPCMessageShm>& ipc_message_shm,
    AllocatedSharedMemory<bi::interprocess_mutex>& response_mutex_shm,
    AllocatedSharedMemory<bi::interprocess_condition>& response_cond_shm)
    : ipc_message_shm_(std::move(ipc_message_shm)),
      response_mutex_shm_(std::move(response_mutex_shm)),
      response_cond_shm_(std::move(response_cond_shm))
{
  ipc_message_shm_ptr_ = ipc_message_shm_.data_.get();
  response_mutex_shm_ptr_ = response_mutex_shm_.data_.get();
  response_cond_shm_ptr_ = response_cond_shm_.data_.get();
  ipc_message_handle_ = ipc_message_shm_.handle_;
}

IPCMessage::IPCMessage(
    IPCMessageShm* ipc_message_shm,
    bi::managed_external_buffer::handle_t& handle)
{
  ipc_message_handle_ = handle;
  ipc_message_shm_ptr_ = ipc_message_shm;
}

}}};  // namespace triton::backend::python
