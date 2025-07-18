// Copyright 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pb_memory.h"

#include <sstream>

namespace triton { namespace backend { namespace python {

std::unique_ptr<PbMemory>
PbMemory::Create(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
    uint64_t byte_size, char* data, bool copy_gpu)
{
  size_t requested_byte_size = sizeof(MemoryShm);
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    requested_byte_size += sizeof(cudaIpcMemHandle_t);
#endif
  } else {
    requested_byte_size += byte_size;
  }

  AllocatedSharedMemory<char> memory_shm =
      shm_pool->Construct<char>(requested_byte_size);

  PbMemory::FillShmData(
      shm_pool->GetCUDAMemoryPoolManager(), memory_type, memory_type_id,
      byte_size, data, memory_shm.data_.get(), memory_shm.handle_, copy_gpu);

  if (memory_type == TRITONSERVER_MEMORY_CPU) {
    data = memory_shm.data_.get() + sizeof(MemoryShm);
  }

  std::unique_ptr<PbMemory> pb_memory(
      new PbMemory(memory_shm, data, false /* opened_cuda_ipc_handle */));

#ifdef TRITON_ENABLE_GPU
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
    pb_memory->memory_shm_ptr_->gpu_pointer_offset =
        pb_memory->GetGPUPointerOffset();
  }
#endif
  return pb_memory;
}

#ifndef TRITON_PB_STUB
std::unique_ptr<PbMemory>
PbMemory::Create(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    std::unique_ptr<BackendMemory>&& backend_memory, bool copy_gpu)
{
  std::unique_ptr<PbMemory> pb_memory = PbMemory::Create(
      shm_pool, backend_memory->MemoryType(), backend_memory->MemoryTypeId(),
      backend_memory->ByteSize(), backend_memory->MemoryPtr(), copy_gpu);
  pb_memory->backend_memory_ = std::move(backend_memory);

  return pb_memory;
}
#endif

std::unique_ptr<PbMemory>
PbMemory::Create(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
    uint64_t byte_size, char* data, char* data_shm,
    bi::managed_external_buffer::handle_t handle, bool copy_gpu)
{
  PbMemory::FillShmData(
      shm_pool->GetCUDAMemoryPoolManager(), memory_type, memory_type_id,
      byte_size, data, data_shm, handle, copy_gpu);

  if (memory_type == TRITONSERVER_MEMORY_CPU) {
    data = data_shm + sizeof(MemoryShm);
  }

  std::unique_ptr<PbMemory> pb_memory(
      new PbMemory(data_shm, data, handle, false /* opened_cuda_ipc_handle */));

#ifdef TRITON_ENABLE_GPU
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
    pb_memory->memory_shm_ptr_->gpu_pointer_offset =
        pb_memory->GetGPUPointerOffset();
  }
#endif

  return pb_memory;
}

void
PbMemory::CopyBuffer(
    std::unique_ptr<PbMemory>& dst, std::unique_ptr<PbMemory>& src)
{
  if (src->ByteSize() != dst->ByteSize()) {
    throw PythonBackendException(
        "Failed to copy memory buffers. Source and destination byte size do "
        "not match: " +
        std::to_string(dst->ByteSize()) +
        " != " + std::to_string(src->ByteSize()));
  }

  if (src->MemoryType() == TRITONSERVER_MEMORY_CPU &&
      dst->MemoryType() == TRITONSERVER_MEMORY_CPU) {
    std::memcpy(dst->DataPtr(), src->DataPtr(), dst->ByteSize());
    return;
  }

#ifdef TRITON_ENABLE_GPU
  cudaMemcpyKind kind = cudaMemcpyHostToDevice;

  if (src->MemoryType() == TRITONSERVER_MEMORY_CPU &&
      dst->MemoryType() == TRITONSERVER_MEMORY_GPU) {
    kind = cudaMemcpyHostToDevice;
  } else if (
      src->MemoryType() == TRITONSERVER_MEMORY_GPU &&
      dst->MemoryType() == TRITONSERVER_MEMORY_CPU) {
    kind = cudaMemcpyDeviceToHost;
  } else if (
      src->MemoryType() == TRITONSERVER_MEMORY_GPU &&
      dst->MemoryType() == TRITONSERVER_MEMORY_GPU) {
    kind = cudaMemcpyDeviceToDevice;
  }

  cudaError_t err;
  if ((kind == cudaMemcpyDeviceToDevice) &&
      (src->MemoryTypeId() != dst->MemoryTypeId())) {
    err = cudaMemcpyPeer(
        dst->DataPtr(), dst->MemoryTypeId(), src->DataPtr(),
        src->MemoryTypeId(), src->ByteSize());

  } else {
    err = cudaMemcpy(dst->DataPtr(), src->DataPtr(), src->ByteSize(), kind);
  }

  if (err != cudaSuccess) {
    throw PythonBackendException(
        std::string(
            "failed to copy data: " + std::string(cudaGetErrorString(err)))
            .c_str());
  }

  if (kind == cudaMemcpyDeviceToDevice) {
    // Synchronize the default stream for d2d copies.
    // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
    err = cudaStreamSynchronize(0);
    if (err != cudaSuccess) {
      throw PythonBackendException(
          std::string(
              "failed to synchronize the default CUDA stream. error: " +
              std::string(cudaGetErrorString(err)))
              .c_str());
    }
  }
#endif
}

void
PbMemory::FillShmData(
    std::unique_ptr<CUDAMemoryPoolManager>& cuda_pool,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
    uint64_t byte_size, char* data, char* data_shm,
    bi::managed_external_buffer::handle_t handle, bool copy_gpu)
{
  char* memory_data_shm = data_shm + sizeof(MemoryShm);
  MemoryShm* memory_shm_ptr = reinterpret_cast<MemoryShm*>(data_shm);
  memory_shm_ptr->memory_release_id = 0;
  bool use_cuda_shared_pool = false;

  if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    if (data != nullptr) {
      if (copy_gpu) {
        ScopedSetDevice scoped_set_device(memory_type_id);
        THROW_IF_CUDA_ERROR(cudaIpcGetMemHandle(
            reinterpret_cast<cudaIpcMemHandle_t*>(memory_data_shm), data));
      }
      if (cuda_pool->UseCudaSharedPool(memory_type_id) &&
          IsUsingCUDAPool(cuda_pool, memory_type_id, data)) {
        use_cuda_shared_pool = true;
        memory_shm_ptr->cuda_pool_offset =
            data -
            reinterpret_cast<char*>(cuda_pool->CUDAPoolAddress(memory_type_id));
      }
    }
#endif  // TRITON_ENABLE_GPU
  } else {
    if (data != nullptr) {
      std::copy(data, data + byte_size, memory_data_shm);
    }
  }

  memory_shm_ptr->byte_size = byte_size;
  memory_shm_ptr->memory_type_id = memory_type_id;
  memory_shm_ptr->memory_type = memory_type;
  memory_shm_ptr->use_cuda_shared_pool = use_cuda_shared_pool;
}

std::unique_ptr<PbMemory>
PbMemory::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle, char* data_shm,
    bool open_cuda_handle)
{
  MemoryShm* memory_shm_ptr = reinterpret_cast<MemoryShm*>(data_shm);
  char* memory_data_shm = data_shm + sizeof(MemoryShm);
  char* data_ptr = nullptr;
  bool opened_cuda_ipc_handle = false;
  if (memory_shm_ptr->memory_type == TRITONSERVER_MEMORY_GPU &&
      open_cuda_handle) {
#ifdef TRITON_ENABLE_GPU
    if (memory_shm_ptr->use_cuda_shared_pool) {
      // When CUDA shared memory pool is used, the stub will retrieve the
      // data pointer using the offset.
      data_ptr =
          (reinterpret_cast<char*>(
               shm_pool->GetCUDAMemoryPoolManager()->CUDAPoolAddress(
                   memory_shm_ptr->memory_type_id)) +
           memory_shm_ptr->cuda_pool_offset);
    } else {
      cudaIpcMemHandle_t* cuda_handle =
          reinterpret_cast<cudaIpcMemHandle_t*>(memory_data_shm);

      // The pointer opened by the cudaIpcOpenMemHandle will refer to the base
      // address. We need to manually correct the offset.
      void* data_ptr_base;
      CUDAHandler& cuda_handler = CUDAHandler::getInstance();
      cuda_handler.OpenCudaHandle(
          memory_shm_ptr->memory_type_id, cuda_handle, &data_ptr_base);

      data_ptr =
          (reinterpret_cast<char*>(data_ptr_base) +
           memory_shm_ptr->gpu_pointer_offset);
      opened_cuda_ipc_handle = true;
    }

#endif  // TRITON_ENABLE_GPU
  } else {
    data_ptr = memory_data_shm;
  }

  // This check only validates CPU shared memory access.
  if (memory_shm_ptr->memory_type != TRITONSERVER_MEMORY_GPU &&
      (data_ptr + memory_shm_ptr->byte_size >
       (char*)shm_pool->GetBaseAddress() + shm_pool->GetCurrentCapacity())) {
    std::ostringstream oss;
    oss << "0x" << std::hex
        << (reinterpret_cast<uintptr_t>(data_ptr) + memory_shm_ptr->byte_size);
    throw PythonBackendException(
        std::string("Attempted to access out of bounds memory address ") +
        oss.str());
  }

  return std::unique_ptr<PbMemory>(new PbMemory(
      data_shm, data_ptr, handle,
      opened_cuda_ipc_handle /* opened_cuda_ipc_handle */));
}

std::unique_ptr<PbMemory>
PbMemory::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle, bool open_cuda_handle)
{
  AllocatedSharedMemory<char> memory_shm = shm_pool->Load<char>(handle);
  MemoryShm* memory_shm_ptr =
      reinterpret_cast<MemoryShm*>(memory_shm.data_.get());
  char* memory_data_shm = memory_shm.data_.get() + sizeof(MemoryShm);

  char* data_ptr = nullptr;
  bool opened_cuda_ipc_handle = false;
  if (memory_shm_ptr->memory_type == TRITONSERVER_MEMORY_GPU) {
    if (memory_shm_ptr->byte_size > 0 && open_cuda_handle) {
#ifdef TRITON_ENABLE_GPU
      if (memory_shm_ptr->use_cuda_shared_pool) {
        // When CUDA shared memory pool is used, the stub will retrieve the
        // data pointer using the offset.
        data_ptr =
            (reinterpret_cast<char*>(
                 shm_pool->GetCUDAMemoryPoolManager()->CUDAPoolAddress(
                     memory_shm_ptr->memory_type_id)) +
             memory_shm_ptr->cuda_pool_offset);
      } else {
        cudaIpcMemHandle_t* cuda_handle =
            reinterpret_cast<cudaIpcMemHandle_t*>(memory_data_shm);

        // The pointer opened by the cudaIpcOpenMemHandle will refer to the base
        // address. We need to manually correct the offset.
        void* data_ptr_base;
        CUDAHandler& cuda_handler = CUDAHandler::getInstance();
        cuda_handler.OpenCudaHandle(
            memory_shm_ptr->memory_type_id, cuda_handle, &data_ptr_base);

        data_ptr =
            (reinterpret_cast<char*>(data_ptr_base) +
             memory_shm_ptr->gpu_pointer_offset);
        opened_cuda_ipc_handle = true;
      }
#endif
    }
  } else {
    data_ptr = memory_data_shm;
  }

  // This check only validates CPU shared memory access.
  if (memory_shm_ptr->memory_type != TRITONSERVER_MEMORY_GPU &&
      (data_ptr + memory_shm_ptr->byte_size >
       (char*)shm_pool->GetBaseAddress() + shm_pool->GetCurrentCapacity())) {
    std::ostringstream oss;
    oss << "0x" << std::hex
        << (reinterpret_cast<uintptr_t>(data_ptr) + memory_shm_ptr->byte_size);
    throw PythonBackendException(
        std::string("Attempted to access out of bounds memory address ") +
        oss.str());
  }

  return std::unique_ptr<PbMemory>(new PbMemory(
      memory_shm, data_ptr,
      opened_cuda_ipc_handle /* opened_cuda_ipc_handle */));
}

PbMemory::PbMemory(
    AllocatedSharedMemory<char>& memory_shm, char* data,
    bool opened_cuda_ipc_handle)
    : memory_shm_(std::move(memory_shm)), data_ptr_(data),
      opened_cuda_ipc_handle_(opened_cuda_ipc_handle)
{
  memory_shm_ptr_ = reinterpret_cast<MemoryShm*>(memory_shm_.data_.get());
  memory_shm_handle_ = memory_shm_.handle_;
}

PbMemory::PbMemory(
    char* memory_shm, char* data, bi::managed_external_buffer::handle_t handle,
    bool opened_cuda_ipc_handle)
{
  memory_shm_ptr_ = reinterpret_cast<MemoryShm*>(memory_shm);
  data_ptr_ = data;
  opened_cuda_ipc_handle_ = opened_cuda_ipc_handle;
  memory_shm_handle_ = handle;
}

bi::managed_external_buffer::handle_t
PbMemory::ShmHandle()
{
  return memory_shm_handle_;
}

#ifdef TRITON_ENABLE_GPU
void*
PbMemory::GetGPUStartAddress()
{
  if (memory_shm_ptr_->memory_type == TRITONSERVER_MEMORY_GPU) {
    CUDAHandler& cuda_api = CUDAHandler::getInstance();
    CUdeviceptr start_address = 0;

    // Skip this step for empty tensor as the CUDA API 'cuPointerGetAttribute'
    // we use in this function does not accept nullptr.
    if (data_ptr_) {
      cuda_api.PointerGetAttribute(
          &start_address, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
          reinterpret_cast<CUdeviceptr>(data_ptr_));
    }

    return reinterpret_cast<void*>(start_address);
  }

  throw PythonBackendException(
      "Calling GetGPUStartAddress function on CPU memory.");
}

uint64_t
PbMemory::GetGPUPointerOffset()
{
  uint64_t offset;
  if (memory_shm_ptr_->memory_type == TRITONSERVER_MEMORY_GPU) {
    offset = data_ptr_ - reinterpret_cast<char*>(GetGPUStartAddress());
  } else {
    throw PythonBackendException(
        "Calling GetGPUPointerOffset function on CPU tensor.");
  }
  return offset;
}
#endif

TRITONSERVER_MemoryType
PbMemory::MemoryType() const
{
  return memory_shm_ptr_->memory_type;
}

void
PbMemory::SetMemoryReleaseId(uint64_t memory_release_id)
{
  memory_shm_ptr_->memory_release_id = memory_release_id;
}

int64_t
PbMemory::MemoryTypeId() const
{
  return memory_shm_ptr_->memory_type_id;
}

uint64_t
PbMemory::ByteSize() const
{
  return memory_shm_ptr_->byte_size;
}

char*
PbMemory::ShmData() const
{
  return reinterpret_cast<char*>(memory_shm_ptr_) + sizeof(MemoryShm);
}

char*
PbMemory::DataPtr() const
{
  return data_ptr_;
}

uint64_t
PbMemory::ShmStructSize(TRITONSERVER_MemoryType memory_type, uint64_t byte_size)
{
  uint64_t total_memory_size = sizeof(MemoryShm);
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    total_memory_size += sizeof(cudaIpcMemHandle_t);
#endif
  } else {
    total_memory_size += byte_size;
  }

  return total_memory_size;
}

#ifdef TRITON_ENABLE_GPU
void
PbMemory::SetCudaIpcHandle(cudaIpcMemHandle_t* cuda_ipc_handle)
{
  *(reinterpret_cast<cudaIpcMemHandle_t*>(ShmData())) = *(cuda_ipc_handle);
}

void
PbMemory::UpdateCUDAOffset(std::unique_ptr<CUDAMemoryPoolManager>& cuda_pool)
{
  if (cuda_pool->UseCudaSharedPool(MemoryTypeId()) &&
      IsUsingCUDAPool(cuda_pool, MemoryTypeId(), DataPtr())) {
    memory_shm_ptr_->cuda_pool_offset =
        DataPtr() -
        reinterpret_cast<char*>(cuda_pool->CUDAPoolAddress(MemoryTypeId()));
    memory_shm_ptr_->use_cuda_shared_pool = true;
  }
}
#endif

PbMemory::~PbMemory()
{
  if (opened_cuda_ipc_handle_) {
#ifdef TRITON_ENABLE_GPU
    CUDAHandler& cuda_handler = CUDAHandler::getInstance();
    cuda_handler.CloseCudaHandle(
        memory_shm_ptr_->memory_type_id, GetGPUStartAddress());
#endif
  }

  if (release_callback_) {
    release_callback_();
  }
}

void
PbMemory::SetMemoryReleaseCallback(std::function<void(void)> release_callback)
{
  if (!release_callback_) {
    release_callback_ = release_callback;
  } else {
    throw PythonBackendException("Release callback is already set.");
  }
}

uint64_t
PbMemory::MemoryReleaseId()
{
  return memory_shm_ptr_->memory_release_id;
}

}}}  // namespace triton::backend::python
