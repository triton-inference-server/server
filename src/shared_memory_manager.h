// Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include "triton/core/tritonserver.h"

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "triton/common/triton_json.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace triton { namespace server {

class SharedMemoryManager {
 public:
  SharedMemoryManager() = default;
  ~SharedMemoryManager();

  /// Add a shared memory block representing shared memory in system
  /// (CPU) memory to the manager. Return TRITONSERVER_ERROR_ALREADY_EXISTS
  /// if a shared memory block of the same name already exists in the manager.
  /// \param name The name of the memory block.
  /// \param shm_key The name of the posix shared memory object
  /// containing the block of memory.
  /// \param offset The offset within the shared memory object to the
  /// start of the block.
  /// \param byte_size The size, in bytes of the block.
  /// \return a TRITONSERVER_Error indicating success or failure.
  TRITONSERVER_Error* RegisterSystemSharedMemory(
      const std::string& name, const std::string& shm_key, const size_t offset,
      const size_t byte_size);

#ifdef TRITON_ENABLE_GPU
  /// Add a shared memory block representing shared memory in CUDA
  /// (GPU) memory to the manager. Return TRITONSERVER_ERROR_ALREADY_EXISTS
  /// if a shared memory block of the same name already exists in the manager.
  /// \param name The name of the memory block.
  /// \param cuda_shm_handle The unique memory handle to the cuda shared
  /// memory block.
  /// \param byte_size The size, in bytes of the block.
  /// \param device id The GPU number the shared memory region is in.
  /// \return a TRITONSERVER_Error indicating success or failure.
  TRITONSERVER_Error* RegisterCUDASharedMemory(
      const std::string& name, const cudaIpcMemHandle_t* cuda_shm_handle,
      const size_t byte_size, const int device_id);
#endif  // TRITON_ENABLE_GPU

  /// Get the access information for the shared memory block
  /// with the specified name. Return TRITONSERVER_ERROR_NOT_FOUND
  /// if named block doesn't exist.
  /// \param name The name of the shared memory block to get.
  /// \param offset The offset in the block
  /// \param shm_mapped_addr Returns the pointer to the shared
  /// memory block with the specified name and offset
  /// \param memory_type Returns the type of the memory
  /// \param device_id Returns the device id associated with the
  /// memory block
  /// \return a TRITONSERVER_Error indicating success or failure.
  TRITONSERVER_Error* GetMemoryInfo(
      const std::string& name, size_t offset, void** shm_mapped_addr,
      TRITONSERVER_MemoryType* memory_type, int64_t* device_id);

#ifdef TRITON_ENABLE_GPU
  /// Get the CUDA memory handle associated with the block name.
  /// Return TRITONSERVER_ERROR_NOT_FOUND if named block doesn't exist.
  /// \param name The name of the shared memory block to get.
  /// \param cuda_mem_handle Returns the cuda memory handle with the memory
  /// block.
  /// \return a TRITONSERVER_Error indicating success or failure.
  TRITONSERVER_Error* GetCUDAHandle(
      const std::string& name, cudaIpcMemHandle_t** cuda_mem_handle);
#endif

  /// Populates the status of active system/CUDA shared memory regions
  /// in the status JSON. If 'name' is empty then return status of all
  /// active system/CUDA shared memory regions as specified by 'memory_type'.
  /// \param name The name of the shared memory block to get the status of.
  /// \param memory_type The type of memory to get the status of.
  /// \param shm_status Returns status of active shared memory blocks in JSON.
  /// \return a TRITONSERVER_Error indicating success or failure.
  TRITONSERVER_Error* GetStatus(
      const std::string& name, TRITONSERVER_MemoryType memory_type,
      triton::common::TritonJson::Value* shm_status);

  /// Removes the named shared memory block of the specified type from
  /// the manager. Any future attempt to get the details of this block
  /// will result in an array till another block with the same name is
  /// added to the manager.
  /// \param name The name of the shared memory block to remove.
  /// \param memory_type The type of memory to unregister.
  /// \return a TRITONSERVER_Error indicating success or failure.
  TRITONSERVER_Error* Unregister(
      const std::string& name, TRITONSERVER_MemoryType memory_type);

  /// Unregister all shared memory blocks of specified type from the manager.
  /// \param memory_type The type of memory to unregister.
  /// \return a TRITONSERVER_Error indicating success or failure.
  TRITONSERVER_Error* UnregisterAll(TRITONSERVER_MemoryType memory_type);

 private:
  /// A helper function to remove the named shared memory blocks of
  /// specified type
  TRITONSERVER_Error* UnregisterHelper(
      const std::string& name, TRITONSERVER_MemoryType memory_type);

  /// A struct that records the shared memory regions registered by the shared
  /// memory manager.
  struct SharedMemoryInfo {
    SharedMemoryInfo(
        const std::string& name, const std::string& shm_key,
        const size_t offset, const size_t byte_size, int shm_fd,
        void* mapped_addr, const TRITONSERVER_MemoryType kind,
        const int64_t device_id)
        : name_(name), shm_key_(shm_key), offset_(offset),
          byte_size_(byte_size), shm_fd_(shm_fd), mapped_addr_(mapped_addr),
          kind_(kind), device_id_(device_id)
    {
    }

    std::string name_;
    std::string shm_key_;
    size_t offset_;
    size_t byte_size_;
    int shm_fd_;
    void* mapped_addr_;
    TRITONSERVER_MemoryType kind_;
    int64_t device_id_;
  };

#ifdef TRITON_ENABLE_GPU
  struct CUDASharedMemoryInfo : SharedMemoryInfo {
    CUDASharedMemoryInfo(
        const std::string& name, const std::string& shm_key,
        const size_t offset, const size_t byte_size, int shm_fd,
        void* mapped_addr, const TRITONSERVER_MemoryType kind,
        const int64_t device_id, const cudaIpcMemHandle_t* cuda_ipc_handle)
        : SharedMemoryInfo(
              name, shm_key, offset, byte_size, shm_fd, mapped_addr, kind,
              device_id),
          cuda_ipc_handle_(*cuda_ipc_handle)
    {
    }

    cudaIpcMemHandle_t cuda_ipc_handle_;
  };
#endif

  using SharedMemoryStateMap =
      std::map<std::string, std::unique_ptr<SharedMemoryInfo>>;
  // A map between the name and the details of the associated
  // shared memory block
  SharedMemoryStateMap shared_memory_map_;
  // A mutex to protect the concurrent access to shared_memory_map_
  std::mutex mu_;
};
}}  // namespace triton::server
