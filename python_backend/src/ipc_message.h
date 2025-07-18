// Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

#include "shm_manager.h"


namespace triton { namespace backend { namespace python {

namespace bi = boost::interprocess;

typedef enum PYTHONSTUB_commandtype_enum {
  PYTHONSTUB_ExecuteRequest,
  PYTHONSTUB_ExecuteResponse,
  PYTHONSTUB_InitializeRequest,
  PYTHONSTUB_InitializeResponse,
  PYTHONSTUB_CUDAPoolInitializeRequest,
  PYTHONSTUB_FinalizeRequest,
  PYTHONSTUB_FinalizeResponse,
  PYTHONSTUB_LoadGPUBuffers,
  PYTHONSTUB_InferExecRequest,
  PYTHONSTUB_InferStreamExecRequest,
  PYTHONSTUB_InferExecResponse,
  PYTHONSTUB_InferStreamExecResponse,
  PYTHONSTUB_ResponseSend,
  PYTHONSTUB_ResponseClose,
  PYTHONSTUB_AutoCompleteRequest,
  PYTHONSTUB_AutoCompleteResponse,
  PYTHONSTUB_LogRequest,
  PYTHONSTUB_BLSDecoupledInferPayloadCleanup,
  PYTHONSTUB_DecoupledResponseFactoryCleanup,
  PYTHONSTUB_MetricFamilyRequestNew,
  PYTHONSTUB_MetricFamilyRequestDelete,
  PYTHONSTUB_MetricRequestNew,
  PYTHONSTUB_MetricRequestDelete,
  PYTHONSTUB_MetricRequestValue,
  PYTHONSTUB_MetricRequestIncrement,
  PYTHONSTUB_MetricRequestSet,
  PYTHONSTUB_MetricRequestObserve,
  PYTHONSTUB_LoadModelRequest,
  PYTHONSTUB_UnloadModelRequest,
  PYTHONSTUB_ModelReadinessRequest,
  PYTHONSTUB_IsRequestCancelled,
  PYTHONSTUB_CancelBLSInferRequest
} PYTHONSTUB_CommandType;

///
/// Shared memory representation of IPCMessage
///
/// \param command determines the IPC command that is going to be passed.
/// \param args determines the shared memory handle for the input parameters.
/// \param inline_response determines whether this is a response of another IPC
/// message. If this parameter is set, it must provide the handle of the
/// corresponding request in \param response_handle.
/// \param response_handle determines the request handle.
/// \param response_mutex stores the handle for the mutex for the response
/// object.
/// \param response_cond stores the handle for the condition variable
/// for the response object.
struct IPCMessageShm {
  PYTHONSTUB_CommandType command;
  bi::managed_external_buffer::handle_t args;
  bool inline_response = false;
  bi::managed_external_buffer::handle_t response_handle;
  bi::managed_external_buffer::handle_t response_mutex;
  bi::managed_external_buffer::handle_t response_cond;
};

class IPCMessage {
 public:
  static std::unique_ptr<IPCMessage> Create(
      const std::unique_ptr<SharedMemoryManager>& shm_pool,
      bool inline_response);

  static std::unique_ptr<IPCMessage> Create(
      IPCMessageShm* ipc_message_shm,
      bi::managed_external_buffer::handle_t& message_handle);
  static std::unique_ptr<IPCMessage> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t message_handle);

  PYTHONSTUB_CommandType& Command();
  bool& InlineResponse();
  bi::managed_external_buffer::handle_t& ResponseHandle();
  bi::interprocess_condition* ResponseCondition();
  bi::interprocess_mutex* ResponseMutex();
  bi::managed_external_buffer::handle_t& Args();
  bi::managed_external_buffer::handle_t ShmHandle();
  AllocatedSharedMemory<IPCMessageShm>& GetAllocatedSharedMemory();

 private:
  AllocatedSharedMemory<IPCMessageShm> ipc_message_shm_;
  IPCMessageShm* ipc_message_shm_ptr_;

  AllocatedSharedMemory<bi::interprocess_mutex> response_mutex_shm_;
  bi::interprocess_mutex* response_mutex_shm_ptr_;

  AllocatedSharedMemory<bi::interprocess_condition> response_cond_shm_;
  bi::interprocess_condition* response_cond_shm_ptr_;

  bi::managed_external_buffer::handle_t ipc_message_handle_;

  /// Create/load a IPCMessage shm object.
  /// \param ipc_message_shm IPCMessage representation in shared memory.
  /// \param response_mutex_shm response mutex.
  /// \param response_condition_shm response condition.
  IPCMessage(
      AllocatedSharedMemory<IPCMessageShm>& ipc_message_shm,
      AllocatedSharedMemory<bi::interprocess_mutex>& response_mutex_shm,
      AllocatedSharedMemory<bi::interprocess_condition>& response_cond_shm);

  IPCMessage(
      IPCMessageShm* ipc_message_shm,
      bi::managed_external_buffer::handle_t& handle);
};

}}};  // namespace triton::backend::python
