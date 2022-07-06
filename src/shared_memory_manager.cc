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

#include "shared_memory_manager.h"

// Not supporting shared memory for now
#ifdef _WIN32
namespace triton { namespace server {
SharedMemoryManager::~SharedMemoryManager() {}

TRITONSERVER_Error*
SharedMemoryManager::RegisterSystemSharedMemory(
    const std::string& name, const std::string& shm_key, const size_t offset,
    const size_t byte_size)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED,
      std::string("Shared memory feature is currently not supported on Windows")
          .c_str());
}

#ifdef TRITON_ENABLE_GPU
TRITONSERVER_Error*
SharedMemoryManager::RegisterCUDASharedMemory(
    const std::string& name, const cudaIpcMemHandle_t* cuda_shm_handle,
    const size_t byte_size, const int device_id)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED,
      std::string("Shared memory feature is currently not supported on Windows")
          .c_str());
}

TRITONSERVER_Error*
SharedMemoryManager::GetCUDAHandle(
    const std::string& name, cudaIpcMemHandle_t** cuda_mem_handle)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED,
      std::string("Shared memory feature is currently not supported on Windows")
          .c_str());
}
#endif  // TRITON_ENABLE_GPU

TRITONSERVER_Error*
SharedMemoryManager::GetMemoryInfo(
    const std::string& name, size_t offset, void** shm_mapped_addr,
    TRITONSERVER_MemoryType* memory_type, int64_t* device_id)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED,
      std::string("Shared memory feature is currently not supported on Windows")
          .c_str());
}

TRITONSERVER_Error*
SharedMemoryManager::GetStatus(
    const std::string& name, TRITONSERVER_MemoryType memory_type,
    triton::common::TritonJson::Value* shm_status)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED,
      std::string("Shared memory feature is currently not supported on Windows")
          .c_str());
}

TRITONSERVER_Error*
SharedMemoryManager::Unregister(
    const std::string& name, TRITONSERVER_MemoryType memory_type)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED,
      std::string("Shared memory feature is currently not supported on Windows")
          .c_str());
}

TRITONSERVER_Error*
SharedMemoryManager::UnregisterAll(TRITONSERVER_MemoryType memory_type)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED,
      std::string("Shared memory feature is currently not supported on Windows")
          .c_str());
}

TRITONSERVER_Error*
SharedMemoryManager::UnregisterHelper(
    const std::string& name, TRITONSERVER_MemoryType memory_type)
{
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED,
      std::string("Shared memory feature is currently not supported on Windows")
          .c_str());
}
}}  // namespace triton::server
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "common.h"
#include "triton/common/logging.h"

namespace triton { namespace server {

namespace {

TRITONSERVER_Error*
OpenSharedMemoryRegion(const std::string& shm_key, int* shm_fd)
{
  // get shared memory region descriptor
  *shm_fd = shm_open(shm_key.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
  if (*shm_fd == -1) {
    LOG_VERBOSE(1) << "shm_open failed, errno: " << errno;
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string("Unable to open shared memory region: '" + shm_key + "'")
            .c_str());
  }

  return nullptr;
}

TRITONSERVER_Error*
MapSharedMemory(
    const int shm_fd, const size_t offset, const size_t byte_size,
    void** mapped_addr)
{
  // map shared memory to process address space
  *mapped_addr = mmap(NULL, byte_size, PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (*mapped_addr == MAP_FAILED) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, std::string(
                                         "unable to process address space" +
                                         std::string(std::strerror(errno)))
                                         .c_str());
  }

  return nullptr;
}

TRITONSERVER_Error*
CloseSharedMemoryRegion(int shm_fd)
{
  int status = close(shm_fd);
  if (status == -1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "unable to close shared memory descriptor, errno: " +
            std::string(std::strerror(errno)))
            .c_str());
  }

  return nullptr;
}

TRITONSERVER_Error*
UnmapSharedMemory(void* mapped_addr, size_t byte_size)
{
  int status = munmap(mapped_addr, byte_size);
  if (status == -1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "unable to munmap shared memory region, errno: " +
            std::string(std::strerror(errno)))
            .c_str());
  }

  return nullptr;
}

#ifdef TRITON_ENABLE_GPU
TRITONSERVER_Error*
OpenCudaIPCRegion(
    const cudaIpcMemHandle_t* cuda_shm_handle, void** data_ptr, int device_id)
{
  // Set to device curres
  cudaSetDevice(device_id);

  // Open CUDA IPC handle and read data from it
  cudaError_t err = cudaIpcOpenMemHandle(
      data_ptr, *cuda_shm_handle, cudaIpcMemLazyEnablePeerAccess);
  if (err != cudaSuccess) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, std::string(
                                         "failed to open CUDA IPC handle: " +
                                         std::string(cudaGetErrorString(err)))
                                         .c_str());
  }

  return nullptr;
}
#endif  // TRITON_ENABLE_GPU

}  // namespace

SharedMemoryManager::~SharedMemoryManager()
{
  UnregisterAll(TRITONSERVER_MEMORY_CPU);
  UnregisterAll(TRITONSERVER_MEMORY_GPU);
}

TRITONSERVER_Error*
SharedMemoryManager::RegisterSystemSharedMemory(
    const std::string& name, const std::string& shm_key, const size_t offset,
    const size_t byte_size)
{
  std::lock_guard<std::mutex> lock(mu_);

  if (shared_memory_map_.find(name) != shared_memory_map_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS,
        std::string("shared memory region '" + name + "' already in manager")
            .c_str());
  }

  // register
  void* mapped_addr;
  int shm_fd = -1;

  // don't re-open if shared memory is already open
  for (auto itr = shared_memory_map_.begin(); itr != shared_memory_map_.end();
       ++itr) {
    if (itr->second->shm_key_ == shm_key) {
      shm_fd = itr->second->shm_fd_;
      break;
    }
  }

  // open and set new shm_fd if new shared memory key
  if (shm_fd == -1) {
    RETURN_IF_ERR(OpenSharedMemoryRegion(shm_key, &shm_fd));
  }

  // Mmap and then close the shared memory descriptor
  TRITONSERVER_Error* err_mmap =
      MapSharedMemory(shm_fd, offset, byte_size, &mapped_addr);
  TRITONSERVER_Error* err_close = CloseSharedMemoryRegion(shm_fd);
  if (err_mmap != nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "failed to register shared memory region '" + name +
            "': " + TRITONSERVER_ErrorMessage(err_mmap))
            .c_str());
  }

  if (err_close != nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "failed to register shared memory region '" + name +
            "': " + TRITONSERVER_ErrorMessage(err_close))
            .c_str());
  }

  shared_memory_map_.insert(std::make_pair(
      name, std::unique_ptr<SharedMemoryInfo>(new SharedMemoryInfo(
                name, shm_key, offset, byte_size, shm_fd, mapped_addr,
                TRITONSERVER_MEMORY_CPU, 0))));

  return nullptr;  // success
}

#ifdef TRITON_ENABLE_GPU
TRITONSERVER_Error*
SharedMemoryManager::RegisterCUDASharedMemory(
    const std::string& name, const cudaIpcMemHandle_t* cuda_shm_handle,
    const size_t byte_size, const int device_id)
{
  // Serialize all operations that write/read current shared memory regions
  std::lock_guard<std::mutex> lock(mu_);

  // If name is already in shared_memory_map_ then return error saying already
  // registered
  if (shared_memory_map_.find(name) != shared_memory_map_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_ALREADY_EXISTS,
        std::string("shared memory region '" + name + "' already in manager")
            .c_str());
  }

  // register
  void* mapped_addr;

  // Get CUDA shared memory base address
  TRITONSERVER_Error* err =
      OpenCudaIPCRegion(cuda_shm_handle, &mapped_addr, device_id);
  if (err != nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "failed to register CUDA shared memory region '" + name +
            "': " + TRITONSERVER_ErrorMessage(err))
            .c_str());
  }

  shared_memory_map_.insert(std::make_pair(
      name, std::unique_ptr<CUDASharedMemoryInfo>(new CUDASharedMemoryInfo(
                name, "", 0, byte_size, 0, mapped_addr, TRITONSERVER_MEMORY_GPU,
                device_id, cuda_shm_handle))));

  return nullptr;  // success
}
#endif  // TRITON_ENABLE_GPU

TRITONSERVER_Error*
SharedMemoryManager::GetMemoryInfo(
    const std::string& name, size_t offset, void** shm_mapped_addr,
    TRITONSERVER_MemoryType* memory_type, int64_t* device_id)
{
  // protect shared_memory_map_ from concurrent access
  std::lock_guard<std::mutex> lock(mu_);

  auto it = shared_memory_map_.find(name);
  if (it == shared_memory_map_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        std::string("Unable to find shared memory region: '" + name + "'")
            .c_str());
  }
  if (it->second->kind_ == TRITONSERVER_MEMORY_CPU) {
    *shm_mapped_addr =
        (void*)((uint8_t*)it->second->mapped_addr_ + it->second->offset_ + offset);
  } else {
    *shm_mapped_addr = (void*)((uint8_t*)it->second->mapped_addr_ + offset);
  }

  *memory_type = it->second->kind_;
  *device_id = it->second->device_id_;

  return nullptr;
}

#ifdef TRITON_ENABLE_GPU
TRITONSERVER_Error*
SharedMemoryManager::GetCUDAHandle(
    const std::string& name, cudaIpcMemHandle_t** cuda_mem_handle)
{
  // protect shared_memory_map_ from concurrent access
  std::lock_guard<std::mutex> lock(mu_);

  auto it = shared_memory_map_.find(name);
  if (it == shared_memory_map_.end()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        std::string("Unable to find shared memory region: '" + name + "'")
            .c_str());
  }
  CUDASharedMemoryInfo& shm_info =
      reinterpret_cast<CUDASharedMemoryInfo&>(*(it->second));
  *cuda_mem_handle = &(shm_info.cuda_ipc_handle_);

  return nullptr;
}
#endif

TRITONSERVER_Error*
SharedMemoryManager::GetStatus(
    const std::string& name, TRITONSERVER_MemoryType memory_type,
    triton::common::TritonJson::Value* shm_status)
{
  std::lock_guard<std::mutex> lock(mu_);

  if (name.empty()) {
    for (const auto& shm_info : shared_memory_map_) {
      if (shm_info.second->kind_ == memory_type) {
        triton::common::TritonJson::Value shm_region(
            *shm_status, triton::common::TritonJson::ValueType::OBJECT);
        RETURN_IF_ERR(shm_region.AddString(
            "name", shm_info.first.c_str(), shm_info.first.size()));
        if (memory_type == TRITONSERVER_MEMORY_CPU) {
          RETURN_IF_ERR(shm_region.AddString(
              "key", shm_info.second->shm_key_.c_str(),
              shm_info.second->shm_key_.size()));
          RETURN_IF_ERR(shm_region.AddUInt("offset", shm_info.second->offset_));
        } else {
          RETURN_IF_ERR(
              shm_region.AddUInt("device_id", shm_info.second->device_id_));
        }
        RETURN_IF_ERR(
            shm_region.AddUInt("byte_size", shm_info.second->byte_size_));
        RETURN_IF_ERR(shm_status->Append(std::move(shm_region)));
      }
    }
  } else {
    auto it = shared_memory_map_.find(name);
    if (it == shared_memory_map_.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_NOT_FOUND,
          std::string(
              "Unable to find system shared memory region: '" + name + "'")
              .c_str());
    }

    if (it->second->kind_ != memory_type) {
      if (it->second->kind_ == TRITONSERVER_MEMORY_GPU) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_NOT_FOUND,
            std::string(
                "The region named '" + name +
                "' is registered as CUDA shared "
                "memory, not system shared memory")
                .c_str());
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_NOT_FOUND,
            std::string(
                "The region named '" + name +
                "' is registered as system shared "
                "memory, not CUDA shared memory")
                .c_str());
      }
    }

    triton::common::TritonJson::Value shm_region(
        *shm_status, triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_ERR(shm_region.AddString(
        "name", it->second->name_.c_str(), it->second->name_.size()));
    if (memory_type == TRITONSERVER_MEMORY_CPU) {
      RETURN_IF_ERR(shm_region.AddString(
          "key", it->second->shm_key_.c_str(), it->second->shm_key_.size()));
      RETURN_IF_ERR(shm_region.AddUInt("offset", it->second->offset_));
    } else {
      RETURN_IF_ERR(shm_region.AddUInt("device_id", it->second->device_id_));
    }
    RETURN_IF_ERR(shm_region.AddUInt("byte_size", it->second->byte_size_));
    RETURN_IF_ERR(shm_status->Append(std::move(shm_region)));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
SharedMemoryManager::Unregister(
    const std::string& name, TRITONSERVER_MemoryType memory_type)
{
  // Serialize all operations that write/read current shared memory regions
  std::lock_guard<std::mutex> lock(mu_);

  return UnregisterHelper(name, memory_type);
}

TRITONSERVER_Error*
SharedMemoryManager::UnregisterAll(TRITONSERVER_MemoryType memory_type)
{
  std::lock_guard<std::mutex> lock(mu_);
  std::string error_message = "Failed to unregister the following ";
  std::vector<std::string> unregister_fails;
  if (memory_type == TRITONSERVER_MEMORY_CPU) {
    // Serialize all operations that write/read current shared memory regions
    error_message += "system shared memory regions: ";
    for (auto it = shared_memory_map_.cbegin(), next_it = it;
         it != shared_memory_map_.cend(); it = next_it) {
      ++next_it;
      if (it->second->kind_ == TRITONSERVER_MEMORY_CPU) {
        TRITONSERVER_Error* err = UnregisterHelper(it->first, memory_type);
        if (err != nullptr) {
          unregister_fails.push_back(it->first);
        }
      }
    }
  } else if (memory_type == TRITONSERVER_MEMORY_GPU) {
    error_message += "cuda shared memory regions: ";
    for (auto it = shared_memory_map_.cbegin(), next_it = it;
         it != shared_memory_map_.cend(); it = next_it) {
      ++next_it;
      if (it->second->kind_ == TRITONSERVER_MEMORY_GPU) {
        TRITONSERVER_Error* err = UnregisterHelper(it->first, memory_type);
        ;
        if (err != nullptr) {
          unregister_fails.push_back(it->first);
        }
      }
    }
  }

  if (!unregister_fails.empty()) {
    for (auto unreg_fail : unregister_fails) {
      error_message += unreg_fail + " ,";
    }
    LOG_ERROR << error_message;
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, error_message.c_str());
  }

  return nullptr;
}

TRITONSERVER_Error*
SharedMemoryManager::UnregisterHelper(
    const std::string& name, TRITONSERVER_MemoryType memory_type)
{
  // Must hold the lock on register_mu_ while calling this function.
  auto it = shared_memory_map_.find(name);
  if (it != shared_memory_map_.end() && it->second->kind_ == memory_type) {
    if (it->second->kind_ == TRITONSERVER_MEMORY_CPU) {
      RETURN_IF_ERR(
          UnmapSharedMemory(it->second->mapped_addr_, it->second->byte_size_));
    } else {
#ifdef TRITON_ENABLE_GPU
      cudaError_t err = cudaIpcCloseMemHandle(it->second->mapped_addr_);
      if (err != cudaSuccess) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "failed to close CUDA IPC handle: " +
                std::string(cudaGetErrorString(err)))
                .c_str());
      }
#else
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "failed to unregister CUDA shared memory region: '" + name +
              "', GPUs not supported")
              .c_str());
#endif  // TRITON_ENABLE_GPU
    }

    // Remove region information from shared_memory_map_
    shared_memory_map_.erase(it);
  }

  return nullptr;
}

}}  // namespace triton::server
#endif
