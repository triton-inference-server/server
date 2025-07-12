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

#include "pb_utils.h"
#include "shm_manager.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_memory.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace backend { namespace python {

//
// Represents a memory object in shared memory.
//
struct MemoryShm {
  // If the memory type is a GPU pointer, the offset of the GPU pointer from the
  // base address. For CPU memory type this field contains garbage data. This
  // field will only be used when the memory is not allocated from the CUDA
  // shared memory pool.
  uint64_t gpu_pointer_offset;
  bool use_cuda_shared_pool;
  // The offset of the memory from the base address of the CUDA shared memory
  // pool.
  uint64_t cuda_pool_offset;

  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;
  uint64_t byte_size;
  uint64_t memory_release_id;
};

class PbMemory {
 public:
  static std::unique_ptr<PbMemory> Create(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
      uint64_t byte_size, char* data, bool copy_gpu = true);

  static std::unique_ptr<PbMemory> Create(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
      uint64_t byte_size, char* data, char* data_shm,
      bi::managed_external_buffer::handle_t handle, bool copy_gpu = true);

#ifndef TRITON_PB_STUB
  static std::unique_ptr<PbMemory> Create(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      std::unique_ptr<BackendMemory>&& backend_memory, bool copy_gpu = true);
#endif

#ifdef TRITON_ENABLE_GPU
  void SetCudaIpcHandle(cudaIpcMemHandle_t* cuda_ipc_handle);

  void UpdateCUDAOffset(std::unique_ptr<CUDAMemoryPoolManager>& cuda_pool);
#endif

  // Copy the destination buffer to the source buffer.
  static void CopyBuffer(
      std::unique_ptr<PbMemory>& dst, std::unique_ptr<PbMemory>& src);

  static std::unique_ptr<PbMemory> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t memory_handle,
      bool open_cuda_handle);
  static std::unique_ptr<PbMemory> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle, char* data_shm,
      bool open_cuda_handle);
  static uint64_t ShmStructSize(
      TRITONSERVER_MemoryType memory_type, uint64_t byte_size);

  bi::managed_external_buffer::handle_t ShmHandle();

  /// Get the total byte size of the tensor.
  uint64_t ByteSize() const;

  /// Get the triton memory type.
  /// \return the memory type of the tensor.
  TRITONSERVER_MemoryType MemoryType() const;

  /// Get the pointer.
  /// \return The location to the memory where the data is stored.
  char* DataPtr() const;

  /// Get the memory type id.
  /// \return The memory type id of the tensor.
  int64_t MemoryTypeId() const;

  /// Get the shm data
  /// \return The memory type id of the tensor.
  char* ShmData() const;

  /// Set the memory release id
  void SetMemoryReleaseId(uint64_t memory_release_id);

  /// Memory Release ID
  uint64_t MemoryReleaseId();

  void SetMemoryReleaseCallback(std::function<void(void)> release_callback);

  bool UseCUDASharedPool() const
  {
    return memory_shm_ptr_->use_cuda_shared_pool;
  }

  ~PbMemory();

#ifndef TRITON_PB_STUB
  void SetBackendMemory(std::unique_ptr<BackendMemory>&& backend_memory)
  {
    backend_memory_ = std::move(backend_memory);
  };

  std::unique_ptr<BackendMemory> GetBackendMemory()
  {
    return std::move(backend_memory_);
  };
#endif

 private:
  AllocatedSharedMemory<char> memory_shm_;
  MemoryShm* memory_shm_ptr_;

#ifndef TRITON_PB_STUB
  std::unique_ptr<BackendMemory> backend_memory_;
#endif

  std::function<void()> release_callback_;

  // Refers to the pointer that can hold the data. For CPU pointers this will be
  // the same as memory_data_shm_ptr_.
  char* data_ptr_;

  bi::managed_external_buffer::handle_t memory_shm_handle_;
  bool opened_cuda_ipc_handle_;

#ifdef TRITON_ENABLE_GPU
  /// Calculate the pointer offset from the base address.
  /// \return The offset of a device pointer.
  /// \throws PythonBackendException if the tensor is stored in CPU.
  uint64_t GetGPUPointerOffset();

  /// Get the GPU start address.
  /// \return The start address of a device pointer.
  /// \throws PythonBackendException if the tensor is stored in CPU.
  void* GetGPUStartAddress();

#endif

  static void FillShmData(
      std::unique_ptr<CUDAMemoryPoolManager>& cuda_pool,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
      uint64_t byte_size, char* data, char* data_shm,
      bi::managed_external_buffer::handle_t handle, bool copy_gpu = true);

  PbMemory(
      AllocatedSharedMemory<char>& memory_shm, char* data,
      bool opened_cuda_ipc_handle);

  PbMemory(
      char* memory_shm, char* data,
      bi::managed_external_buffer::handle_t handle,
      bool opened_cuda_ipc_handle);
};
}}}  // namespace triton::backend::python
