// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
//
#include "src/core/cuda_memory_manager.h"

#include <cnmem.h>
#include <set>
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"

namespace {

#define RETURN_IF_CNMEM_ERROR(S)                                             \
  do {                                                                       \
    auto status__ = (S);                                                     \
    if (status__ != CNMEM_STATUS_SUCCESS) {                                  \
      std::string msg = std::string(cnmemGetErrorString(status__));          \
      return Status(                                                         \
          RequestStatusCode::INTERNAL, "CUDA memory manager error " +        \
                                           std::to_string(status__) + ": " + \
                                           std::string(msg));                \
    }                                                                        \
  } while (false)

}  // namespace

namespace nvidia { namespace inferenceserver {

std::unique_ptr<CUDAMemoryManager> CUDAMemoryManager::instance_;

CUDAMemoryManager::~CUDAMemoryManager()
{
  auto status = cnmemFinalize();
  if (status != CNMEM_STATUS_SUCCESS) {
    LOG_ERROR << "Failed to finalize CUDA memory manager: [" << status << "] "
              << cnmemGetErrorString(status);
  }
}

Status
CUDAMemoryManager::Create(const Options& options)
{
  std::set<int> supported_gpus;
  RETURN_IF_ERROR(GetSupportedGPUs(
      &supported_gpus, options.min_supported_compute_capability_));
  std::vector<cnmemDevice_t> devices;
  for (auto gpu : supported_gpus) {
    devices.emplace_back();
    auto& device = devices.back();
    memset(&device, 0, sizeof(device));
    device.device = gpu;
    device.size = options.memory_pool_byte_size_;
  }
  RETURN_IF_CNMEM_ERROR(
      cnmemInit(devices.size(), devices.data(), CNMEM_FLAGS_CANNOT_GROW));
  return Status::Success;
}

Status
CUDAMemoryManager::Alloc(void** ptr, uint64_t size)
{
  return Status(
      RequestStatusCode::UNSUPPORTED,
      "CUDAMemoryManager::Alloc() not implemented");
}

Status
CUDAMemoryManager::Free(void* ptr)
{
  return Status(
      RequestStatusCode::UNSUPPORTED,
      "CUDAMemoryManager::Free() not implemented");
}

}}  // namespace nvidia::inferenceserver