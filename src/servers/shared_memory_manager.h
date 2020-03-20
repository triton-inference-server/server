// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <mutex>
#include <unordered_map>
#include "src/core/server_status.pb.h"
#include "src/core/trtserver.h"

#ifdef TRTIS_ENABLE_GRPC_V2
#include "src/core/grpc_service_v2.grpc.pb.h"
#endif

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

class SharedMemoryManager {
 public:
  SharedMemoryManager() = default;
  ~SharedMemoryManager();

  /// Add a shared memory block representing shared memory in system
  /// (CPU) memory to the manager. Return TRTSERVER_ERROR_ALREADY_EXISTS
  /// if a shared memory block of the same name already exists in the manager.
  /// \param name The name of the memory block.
  /// \param shm_key The name of the posix shared memory object
  /// containing the block of memory.
  /// \param offset The offset within the shared memory object to the
  /// start of the block.
  /// \param byte_size The size, in bytes of the block.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* RegisterSystemSharedMemory(
      const std::string& name, const std::string& shm_key, const size_t offset,
      const size_t byte_size);

#ifdef TRTIS_ENABLE_GPU
  /// Add a shared memory block representing shared memory in CUDA
  /// (GPU) memory to the manager. Return TRTSERVER_ERROR_ALREADY_EXISTS
  /// if a shared memory block of the same name already exists in the manager.
  /// \param name The name of the memory block.
  /// \param cuda_shm_handle The unique memory handle to the cuda shared
  /// memory block.
  /// \param byte_size The size, in bytes of the block.
  /// \param device id The GPU number the shared memory region is in.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* RegisterCUDASharedMemory(
      const std::string& name, const cudaIpcMemHandle_t* cuda_shm_handle,
      const size_t byte_size, const int device_id);
#endif  // TRTIS_ENABLE_GPU

  /// Get the access information for the shared memory block
  /// with the specified name. Return TRTSERVER_ERROR_NOT_FOUND
  /// if named block doesn't exist.
  /// \param name The name of the shared memory block to get.
  /// \param offset The offset in the block
  /// \param shm_mapped_addr Returns the pointer to the shared
  /// memory block with the specified name and offset
  /// \param memory_type Returns the type of the memory
  /// \param device_id Returns the device id associated with the
  /// memory block
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* GetMemoryInfo(
      const std::string& name, size_t offset, void** shm_mapped_addr,
      TRTSERVER_Memory_Type* memory_type, int64_t* device_id);

  /// Removes the named shared memory block from the manager. Any future
  /// attempt to get the details of this block will result in an array
  /// till another block with the same name is added to the manager.
  /// \param name The name of the shared memory block to remove.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* Unregister(const std::string& name);

  /// Unregister all shared memory blocks from the manager.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* UnregisterAll();

  /// Populates the status of active shared memory regions in the
  /// specified protobuf message.
  /// \param status Returns status of active shared meeory blocks
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* GetStatus(SharedMemoryStatus* status);

#ifdef TRTIS_ENABLE_GRPC_V2
  /// Populates the status of active system shared memory regions
  /// in the response protobuf. If 'name' is missing then return status of
  /// all active system shared memory regions.
  /// \param name The name of the shared memory block to get the status of.
  /// \param shm_status Returns status of active shared meeory blocks
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* GetStatusV2(
      const std::string& name, SystemSharedMemoryStatusResponse*& shm_status);

  /// Populates the status of active CUDA shared memory regions
  /// in the response protobuf. If 'name' is missing then return status of
  /// all active CUDA shared memory regions.
  /// \param name The name of the shared memory block to get the status of.
  /// \param shm_status Returns status of active shared meeory blocks.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* GetStatusV2(
      const std::string& name, CudaSharedMemoryStatusResponse*& shm_status);
#endif  // TRTIS_ENABLE_GRPC_V2

#ifdef TRTIS_ENABLE_HTTP_V2
  /// Populates the status of active system/CUDA shared memory regions
  /// in the status protobuf. If 'name' is missing then return status of all
  /// active system/CUDA shared memory regions as specified by 'memory_type'.
  /// \param name The name of the shared memory block to get the status of.
  /// \param shm_status Returns status of active shared meeory blocks.
  /// \param memory_type The type of memory to get the status of.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* GetStatusV2(
      const std::string& name, SharedMemoryStatus* shm_status,
      TRTSERVER_Memory_Type memory_type);
#endif  // TRTIS_ENABLE_HTTP_V2

#if defined(TRTIS_ENABLE_GRPC) || defined(TRTIS_ENABLE_GRPC_V2)
  /// Removes the named shared memory block of the specified type from
  /// the manager. Any future attempt to get the details of this block
  /// will result in an array till another block with the same name is
  /// added to the manager.
  /// \param name The name of the shared memory block to remove.
  /// \param memory_type The type of memory to unregister.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* UnregisterV2(
      const std::string& name, TRTSERVER_Memory_Type memory_type);

  /// Unregister all shared memory blocks of specified type from the manager.
  /// \param memory_type The type of memory to unregister.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* UnregisterAllV2(TRTSERVER_Memory_Type memory_type);
#endif  // TRTIS_ENABLE_GRPC_V2 || TRTIS_ENABLE_HTTP_V2

 private:
  /// A helper function to remove the named shared memory blocks.
  TRTSERVER_Error* UnregisterHelper(const std::string& name);

#ifdef TRTIS_ENABLE_GRPC_V2
  /// A helper function to remove the named shared memory blocks of
  /// specified type
  TRTSERVER_Error* UnregisterHelperV2(
      const std::string& name, TRTSERVER_Memory_Type memory_type);
#endif  // TRTIS_ENABLE_GRPC_V2

  /// A struct that records the shared memory regions registered by the shared
  /// memory manager.
  struct SharedMemoryInfo {
    SharedMemoryInfo(
        const std::string& name, const std::string& shm_key,
        const size_t offset, const size_t byte_size, int shm_fd,
        void* mapped_addr, const TRTSERVER_Memory_Type kind,
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
    TRTSERVER_Memory_Type kind_;
    int64_t device_id_;
  };

  using SharedMemoryStateMap =
      std::map<std::string, std::unique_ptr<SharedMemoryInfo>>;
  // A map between the name and the details of the associated
  // shared memory block
  SharedMemoryStateMap shared_memory_map_;
  // A mutex to protect the concurrent access to shared_memory_map_
  std::mutex mu_;
};

}}  // namespace nvidia::inferenceserver
