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

#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "message_queue.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_memory.h"
#include "triton/core/tritonserver.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU


namespace triton { namespace backend { namespace python {

class MemoryRecord {
 public:
  virtual const std::function<void(void*)>& ReleaseCallback() = 0;
  virtual void* MemoryId() = 0;
  virtual ~MemoryRecord() = default;
};

#ifdef TRITON_ENABLE_GPU
class BackendMemoryRecord : public MemoryRecord {
 public:
  BackendMemoryRecord(std::unique_ptr<BackendMemory> backend_memory);
  const std::function<void(void*)>& ReleaseCallback() override;
  void* MemoryId() override;
  ~BackendMemoryRecord() { backend_memory_.reset(); }

 private:
  std::unique_ptr<BackendMemory> backend_memory_;
  std::function<void(void*)> release_callback_;
};
#endif

/// Memory manager class is used primarily for managing the lifetime of GPU
/// tensors in BLS. It mainly consists of a background thread that monitors a
/// message queue in shared memory. Whenever a GPU tensor is created, it will
/// be pushed to the memory manager. The stub process must send a message to the
/// message queue asking the memory manager to deallocate the GPU tensor.
class MemoryManager {
 public:
  MemoryManager(std::unique_ptr<MessageQueue<intptr_t>>&& memory_message_queue);
  intptr_t AddRecord(std::unique_ptr<MemoryRecord>&& memory_record);
  TRITONSERVER_Error* ResetCounter();
  ~MemoryManager();

 private:
  std::thread thread_;
  std::unordered_map<intptr_t, std::unique_ptr<MemoryRecord>> records_;
  std::unique_ptr<MessageQueue<intptr_t>> message_queue_;
  void QueueMonitorThread();
  std::mutex mu_;
};
}}};  // namespace triton::backend::python
