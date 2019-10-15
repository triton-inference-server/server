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
//

#include "src/core/pinned_memory_manager.h"

#include <sstream>
#include "src/core/logging.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRTIS_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

namespace {

std::string
PointerToString(void* ptr)
{
  std::stringstream ss;
  ss << ptr;
  return ss.str();
}

}  // namespace

std::unique_ptr<PinnedMemoryManager> PinnedMemoryManager::instance_;

PinnedMemoryManager::PinnedMemoryManager(
    void* pinned_memory_buffer, uint64_t size)
    : pinned_memory_buffer_(pinned_memory_buffer)
{
  if (pinned_memory_buffer_ != nullptr) {
    managed_pinned_memory_ = boost::interprocess::managed_external_buffer(
        boost::interprocess::create_only_t{}, pinned_memory_buffer_, size);
  }
}

PinnedMemoryManager::~PinnedMemoryManager()
{
  // Clean up
  for (const auto& memory_info : memory_info_) {
    const auto& is_pinned = memory_info.second;
    if (!is_pinned) {
      free(memory_info.first);
    }
  }
#ifdef TRTIS_ENABLE_GPU
  if (pinned_memory_buffer_ != nullptr) {
    cudaFreeHost(pinned_memory_buffer_);
  }
#endif  // TRTIS_ENABLE_GPU
}

Status
PinnedMemoryManager::AllocInternal(
    void** ptr, uint64_t size, bool allow_nonpinned_fallback)
{
  auto status = Status::Success;
  if (pinned_memory_buffer_ != nullptr) {
    std::lock_guard<std::mutex> lk(buffer_mtx_);
    *ptr = managed_pinned_memory_.allocate(size, std::nothrow_t{});
    if (*ptr == nullptr) {
      status = Status(
          RequestStatusCode::INTERNAL,
          "failed to allocate pinned system memory");
    }
  } else {
    status = Status(
        RequestStatusCode::INTERNAL,
        "failed to allocate pinned system memory: no pinned memory pool");
  }

  bool is_pinned = true;
  if ((!status.IsOk()) && allow_nonpinned_fallback) {
    static bool warning_logged = false;
    if (!warning_logged) {
      LOG_WARNING << status.Message()
                  << ", falling back to non-pinned system memory";
      warning_logged = true;
    }
    *ptr = malloc(size);
    is_pinned = false;
    if (*ptr == nullptr) {
      status = Status(
          RequestStatusCode::INTERNAL,
          "failed to allocate non-pinned system memory");
    } else {
      status = Status::Success;
    }
  }

  // keep track of allocated buffer or clean up
  {
    std::lock_guard<std::mutex> lk(info_mtx_);
    if (status.IsOk()) {
      auto res = memory_info_.emplace(*ptr, is_pinned);
      if (!res.second) {
        status = Status(
            RequestStatusCode::INTERNAL,
            "unexpected memory address collision, '" + PointerToString(*ptr) +
                "' has been managed");
      }
      LOG_VERBOSE(1) << (is_pinned ? "" : "non-")
                     << "pinned memory allocation: "
                     << "size " << size << ", addr " << *ptr;
    }
  }

  if ((!status.IsOk()) && (*ptr != nullptr)) {
    if (is_pinned) {
      std::lock_guard<std::mutex> lk(buffer_mtx_);
      managed_pinned_memory_.deallocate(*ptr);
    } else {
      free(*ptr);
    }
  }

  return status;
}

Status
PinnedMemoryManager::FreeInternal(void* ptr)
{
  bool is_pinned = true;
  {
    std::lock_guard<std::mutex> lk(info_mtx_);
    auto it = memory_info_.find(ptr);
    if (it != memory_info_.end()) {
      is_pinned = it->second;
      LOG_VERBOSE(1) << (is_pinned ? "" : "non-")
                     << "pinned memory deallocation: "
                     << "addr " << ptr;
      memory_info_.erase(it);
    } else {
      return Status(
          RequestStatusCode::INTERNAL, "unexpected memory address '" +
                                           PointerToString(ptr) +
                                           "' is not being managed");
    }
  }

  if (is_pinned) {
    std::lock_guard<std::mutex> lk(buffer_mtx_);
    managed_pinned_memory_.deallocate(ptr);
  } else {
    free(ptr);
  }
  return Status::Success;
}

Status
PinnedMemoryManager::Create(const Options& options)
{
  if (instance_ != nullptr) {
    return Status(
        RequestStatusCode::ALREADY_EXISTS,
        "PinnedMemoryManager has been created");
  }

  void* buffer = nullptr;
#ifdef TRTIS_ENABLE_GPU
  auto err = cudaHostAlloc(
      &buffer, options.pinned_memory_pool_byte_size_, cudaHostAllocPortable);
  if (err != cudaSuccess) {
    buffer = nullptr;
    LOG_ERROR << "failed to allocate pinned system memory: "
              << std::string(cudaGetErrorString(err));
  } else {
    LOG_VERBOSE(1) << "Pinned memory pool is created at '"
                   << PointerToString(buffer) << "' with size "
                   << options.pinned_memory_pool_byte_size_;
  }
#endif  // TRTIS_ENABLE_GPU
  instance_.reset(
      new PinnedMemoryManager(buffer, options.pinned_memory_pool_byte_size_));
  return Status::Success;
}

Status
PinnedMemoryManager::Alloc(
    void** ptr, uint64_t size, bool allow_nonpinned_fallback)
{
  if (instance_ == nullptr) {
    return Status(
        RequestStatusCode::UNAVAILABLE,
        "PinnedMemoryManager has not been created");
  }

  return instance_->AllocInternal(ptr, size, allow_nonpinned_fallback);
}

Status
PinnedMemoryManager::Free(void* ptr)
{
  if (instance_ == nullptr) {
    return Status(
        RequestStatusCode::UNAVAILABLE,
        "PinnedMemoryManager has not been created");
  }

  return instance_->FreeInternal(ptr);
}

}}  // namespace nvidia::inferenceserver