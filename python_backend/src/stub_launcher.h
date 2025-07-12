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

#include <atomic>
#include <boost/asio.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/functional/hash.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/thread/thread_time.hpp>
#include <future>
#include <sstream>
#include <string>
#include <vector>

#include "ipc_message.h"
#include "memory_manager.h"
#include "message_queue.h"
#include "pb_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace python {

class ModelState;

class StubLauncher {
 public:
  StubLauncher(const std::string stub_process_kind);
  StubLauncher(
      const std::string stub_process_kind,
      const std::string model_instance_name, const int32_t device_id,
      const std::string kind);

  // Initialize stub process
  TRITONSERVER_Error* Initialize(ModelState* model_state);

  // Stub process setup
  TRITONSERVER_Error* Setup();

  // Launch stub process
  TRITONSERVER_Error* Launch();

  // Auto-complete stub process
  void AutocompleteStubProcess();

  // Model instance stub process
  TRITONSERVER_Error* ModelInstanceStubProcess();

  // Check if Stub PID is active
  bool StubActive();

  // Health mutex
  bi::interprocess_mutex* HealthMutex() { return health_mutex_; }

  // Stub message queue
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>&
  StubMessageQueue()
  {
    return stub_message_queue_;
  }

  // Parent message queue
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>&
  ParentMessageQueue()
  {
    return parent_message_queue_;
  }

  // Stub to parent message queue
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>&
  StubToParentMessageQueue()
  {
    return stub_to_parent_mq_;
  }

  // Parent to stub message queue
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>&
  ParentToStubMessageQueue()
  {
    return parent_to_stub_mq_;
  }

  // Memory Manager
  std::unique_ptr<MemoryManager>& GetMemoryManager() { return memory_manager_; }

  // IPC control
  std::unique_ptr<IPCControlShm, std::function<void(IPCControlShm*)>>&
  IpcControl()
  {
    return ipc_control_;
  }

  // Shared memory pool
  std::unique_ptr<SharedMemoryManager>& ShmPool() { return shm_pool_; }

  // Get auto-complete model configuration
  common::TritonJson::Value& AutoCompleteConfig()
  {
    return auto_complete_config_;
  }

  // Update health variable
  void UpdateHealth();

  // Is Healthy
  bool IsHealthy() { return is_healthy_; }

  // Destruct Stub process
  void TerminateStub();

  // Reset log queue and bls decoupled queue pointers
  void ClearQueues();

  // Kill stub process
  void KillStubProcess();

  // Get a message from the stub process
  TRITONSERVER_Error* ReceiveMessageFromStub(
      bi::managed_external_buffer::handle_t& message);

  // Wait for stub process
  void WaitForStubProcess();

#ifndef _WIN32
  // FIXME [DLIS-5969]: Enable for Windows when custom execution environments
  // are supported.
  TRITONSERVER_Error* GetPythonEnvironment(ModelState* model_state);
#endif
#ifdef TRITON_ENABLE_GPU
  // Share CUDA memory pool with stub process
  void ShareCUDAMemoryPool(
      TRITONBACKEND_MemoryManager* triton_mem_manager, const int32_t device_id);
#endif  // TRITON_ENABLE_GPU

 private:
#ifdef _WIN32
  STARTUPINFO startup_info_;
  DWORD parent_pid_;
  PROCESS_INFORMATION stub_pid_;
#else
  pid_t parent_pid_;
  pid_t stub_pid_;
#endif
  bool is_initialized_;
  bool is_decoupled_;
  bool is_healthy_;
  std::string shm_region_name_;
  std::string model_repository_path_;
  std::string model_path_;
  std::string runtime_modeldir_;
  const std::string stub_process_kind_;
  std::string model_name_;
  const std::string model_instance_name_;
  const int32_t device_id_;
  const std::string kind_;
  uint64_t model_version_;

  std::string python_lib_;
  int64_t shm_default_byte_size_;
  int64_t shm_growth_byte_size_;
  int64_t shm_message_queue_size_;

  // Path to python execution environment
  std::string path_to_libpython_;
  std::string path_to_activate_;
  std::string python_execution_env_;

  common::TritonJson::WriteBuffer model_config_buffer_;
  common::TritonJson::Value auto_complete_config_;

  bi::interprocess_mutex* health_mutex_;
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>
      stub_message_queue_;
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>
      parent_message_queue_;
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>
      stub_to_parent_mq_;
  std::unique_ptr<MessageQueue<bi::managed_external_buffer::handle_t>>
      parent_to_stub_mq_;
  std::unique_ptr<MemoryManager> memory_manager_;
  std::unique_ptr<IPCControlShm, std::function<void(IPCControlShm*)>>
      ipc_control_;
  bi::managed_external_buffer::handle_t ipc_control_handle_;
  std::unique_ptr<SharedMemoryManager> shm_pool_;
#ifdef TRITON_ENABLE_GPU
  std::mutex cuda_shm_pool_mutex_;
  std::unordered_map<int32_t, bool> tried_sharing_cuda_pool_map_;
#endif  // TRITON_ENABLE_GPU
};
}}}  // namespace triton::backend::python
