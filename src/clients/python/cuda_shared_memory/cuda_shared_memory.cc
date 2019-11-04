// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/clients/python/cuda_shared_memory/cuda_shared_memory.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include "src/clients/python/shared_memory/shared_memory_handle.h"

//==============================================================================
// SharedMemoryControlContext

namespace {

void*
CudaSharedMemoryHandleCreate(
    std::string trtis_shm_name, cudaIpcMemHandle_t cuda_shm_handle,
    void* base_addr, size_t byte_size, int device_id)
{
  SharedMemoryHandle* handle = new SharedMemoryHandle();
  handle->trtis_shm_name_ = trtis_shm_name;
  handle->cuda_shm_handle_ = cuda_shm_handle;
  handle->base_addr_ = base_addr;
  handle->byte_size_ = byte_size;
  handle->device_id_ = device_id;
  handle->offset_ = 0;
  handle->shm_key_ = "";
  handle->shm_fd_ = 0;
  return reinterpret_cast<void*>(handle);
}

}  // namespace

int
CudaSharedMemoryRegionCreate(
    const char* trtis_shm_name, size_t byte_size, int device_id,
    void** cuda_shm_handle)
{
  // remember previous device and set to new device
  int previous_device;
  cudaGetDevice(&previous_device);
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    cudaSetDevice(previous_device);
    return -1;
  }

  void* base_addr;
  cudaIpcMemHandle_t cuda_handle;

  // Allocate data and create cuda IPC handle for data on the gpu
  err = cudaMalloc(&base_addr, byte_size);
  err = cudaIpcGetMemHandle(&cuda_handle, base_addr);
  if (err != cudaSuccess) {
    cudaSetDevice(previous_device);
    return -2;
  }

  // create a handle for the shared memory region
  *cuda_shm_handle = CudaSharedMemoryHandleCreate(
      std::string(trtis_shm_name), cuda_handle, base_addr, byte_size,
      device_id);

  // Set device to previous GPU
  cudaSetDevice(previous_device);

  return 0;
}

int
CudaSharedMemoryRegionSet(
    void* cuda_shm_handle, size_t offset, size_t byte_size, const void* data)
{
  // remember previous device and set to new device
  int previous_device;
  cudaGetDevice(&previous_device);
  cudaError_t err = cudaSetDevice(
      reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle)->device_id_);
  if (err != cudaSuccess) {
    cudaSetDevice(previous_device);
    return -1;
  }

  // Copy data into cuda shared memory
  void* base_addr =
      reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle)->base_addr_;
  err = cudaMemcpy(
      reinterpret_cast<uint8_t*>(base_addr) + offset, data, byte_size,
      cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaSetDevice(previous_device);
    return -3;
  }

  // Set device to previous GPU
  cudaSetDevice(previous_device);

  return 0;
}

int
CudaSharedMemoryRegionDestroy(void* cuda_shm_handle)
{
  // remember previous device and set to new device
  int previous_device;
  cudaGetDevice(&previous_device);
  cudaError_t err = cudaSetDevice(
      reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle)->device_id_);
  if (err != cudaSuccess) {
    cudaSetDevice(previous_device);
    return -1;
  }

  SharedMemoryHandle* shm_hand =
      reinterpret_cast<SharedMemoryHandle*>(cuda_shm_handle);

  // Free GPU device memory
  err = cudaFree(shm_hand->base_addr_);
  if (err != cudaSuccess) {
    cudaSetDevice(previous_device);
    return -4;
  }

  // Set device to previous GPU
  cudaSetDevice(previous_device);

  return 0;
}

//==============================================================================
