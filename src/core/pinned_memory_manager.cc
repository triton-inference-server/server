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

Status
PinnedMemoryManager::Create(const Options& options)
{
  if (instance_ != nullptr) {
    return Status(
        RequestStatusCode::ALREADY_EXISTS,
        "PinnedMemoryManager has been created");
  }

  instance_.reset(new PinnedMemoryManager(options));
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

  auto status = Status::Success;
  {
    // only holds the lock on accessing manager member
    std::lock_guard<std::mutex> lk(instance_->mtx_);

    status = instance_->CheckPrerequisite(size);
    // treat as if the operation will succeed to avoid over-subscription
    instance_->allocated_pinned_memory_byte_size_ += size;
  }

  // allocate buffer
  bool is_pinned = true;
  if (status.IsOk()) {
#ifdef TRTIS_ENABLE_GPU
    auto err = cudaHostAlloc(ptr, size, cudaHostAllocPortable);
    if (err != cudaSuccess) {
      // set to nullptr on error to avoid freeing invalid pointer
      *ptr = nullptr;
      status = Status(
          RequestStatusCode::INTERNAL,
          "failed to allocate pinned system memory: " +
              std::string(cudaGetErrorString(err)));
    }
#else
    *ptr = nullptr;
    status = Status(
        RequestStatusCode::INTERNAL,
        "failed to allocate pinned system memory: " +
            "TRTIS_ENABLE_GPU is not set");
#endif  // TRTIS_ENABLE_GPU
  }

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

  // manage allocated buffer or clean up
  {
    std::lock_guard<std::mutex> lk(instance_->mtx_);
    if (status.IsOk()) {
      auto res = instance_->memory_info_.emplace(
          *ptr, std::make_pair(is_pinned, size));
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
    // In either case, we need to adjust back the pinned byte size
    if ((!status.IsOk()) || (!is_pinned)) {
      instance_->allocated_pinned_memory_byte_size_ -= size;
    }
  }

  if ((!status.IsOk()) && (*ptr != nullptr)) {
#ifdef TRTIS_ENABLE_GPU
    if (is_pinned) {
      cudaFreeHost(*ptr);
    } else {
      free(*ptr);
    }
#else
    free(*ptr);
#endif  // TRTIS_ENABLE_GPU
  }

  return status;
}

Status
PinnedMemoryManager::Free(void* ptr)
{
  if (instance_ == nullptr) {
    return Status(
        RequestStatusCode::UNAVAILABLE,
        "PinnedMemoryManager has not been created");
  }

  bool is_pinned = true;
  {
    std::lock_guard<std::mutex> lk(instance_->mtx_);
    auto it = instance_->memory_info_.find(ptr);
    if (it != instance_->memory_info_.end()) {
      is_pinned = it->second.first;
      const auto& size = it->second.second;
      if (is_pinned) {
        instance_->allocated_pinned_memory_byte_size_ -= size;
      }
      instance_->memory_info_.erase(it);
      LOG_VERBOSE(1) << (is_pinned ? "" : "non-")
                     << "pinned memory deallocation: "
                     << "addr " << ptr;
    } else {
      return Status(
          RequestStatusCode::INTERNAL, "unexpected memory address '" +
                                           PointerToString(ptr) +
                                           "' is not being managed");
    }
  }

  if (is_pinned) {
#ifdef TRTIS_ENABLE_GPU
    cudaFreeHost(ptr);
#else
    return Status(
        RequestStatusCode::INTERNAL,
        "unexpected pinned system memory is managed while " +
            "TRTIS_ENABLE_GPU is not set"));
#endif  // TRTIS_ENABLE_GPU
  } else {
    free(ptr);
  }
  return Status::Success;
}

Status
PinnedMemoryManager::CheckPrerequisite(uint64_t requested_size)
{
  std::string error_message;
#ifdef TRTIS_ENABLE_GPU
  if ((allocated_pinned_memory_byte_size_ + requested_size) >
      options_.pinned_memory_pool_byte_size_) {
    error_message =
        ("pinned memory poll exceeded (" +
         std::to_string(options_.pinned_memory_pool_byte_size_) + " < " +
         std::to_string(requested_size) + " + " +
         std::to_string(allocated_pinned_memory_byte_size_) + ")");
  }
#else
  error_message = "TRTIS_ENABLE_GPU is not set";
#endif  // TRTIS_ENABLE_GPU

  if (!error_message.empty()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "reject pinned memory allocation request: " + error_message);
  }
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver