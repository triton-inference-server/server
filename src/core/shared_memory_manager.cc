// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/shared_memory_manager.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <deque>
#include <exception>
#include <future>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/server_status.h"

namespace nvidia { namespace inferenceserver {

namespace {

Status
OpenSharedMemoryRegion(const std::string& shm_key, int* shm_fd)
{
  // get shared memory region descriptor
  *shm_fd = shm_open(shm_key.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
  if (*shm_fd == -1) {
    LOG_VERBOSE(1) << "shm_open failed, errno: " << errno;
    return Status(
        RequestStatusCode::INTERNAL,
        "Unable to open shared memory region: '" + shm_key + "'");
  }

  return Status::Success;
}

Status
MapSharedMemory(
    const int shm_fd, const size_t offset, const size_t byte_size,
    void** mapped_addr)
{
  // map shared memory to process address space
  *mapped_addr = mmap(NULL, byte_size, PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (*mapped_addr == MAP_FAILED) {
    LOG_VERBOSE(1) << "mmap failed, errno: " << errno;
    return Status(
        RequestStatusCode::INTERNAL, "Unable to process address space");
  }

  return Status::Success;
}

Status
CloseSharedMemoryRegion(int shm_fd)
{
  int tmp = close(shm_fd);
  if (tmp == -1) {
    return Status(
        RequestStatusCode::INTERNAL, "Unable to close shared memory region");
  }

  return Status::Success;
}

Status
UnmapSharedMemory(void* mapped_addr, size_t byte_size)
{
  int tmp_fd = munmap(mapped_addr, byte_size);
  if (tmp_fd == -1) {
    return Status(
        RequestStatusCode::INTERNAL, "Unable to munmap shared memory region");
  }

  return Status::Success;
}

}  // namespace

Status
SharedMemoryManager::RegisterSharedMemory(
    const std::string& name, const std::string& shm_key, const size_t offset,
    const size_t byte_size)
{
  // Serialize all operations that write/read current shared memory regions
  std::lock_guard<std::mutex> lock(register_mu_);

  // If key is already in shared_memory_map_ then return error saying already
  // registered
  if (shared_memory_map_.find(name) != shared_memory_map_.end()) {
    return Status(
        RequestStatusCode::ALREADY_EXISTS,
        "shared memory region '" + name + "' is already registered");
  }

  // register (or re-register)
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
    RETURN_IF_ERROR(OpenSharedMemoryRegion(shm_key, &shm_fd));
  }

  Status status = MapSharedMemory(shm_fd, offset, byte_size, &mapped_addr);
  if (!status.IsOk()) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "failed to register shared memory region '" + name + "'");
  }

  shared_memory_map_.insert(std::make_pair(
      name, std::unique_ptr<SharedMemoryInfo>(new SharedMemoryInfo(
                name, shm_key, offset, byte_size, shm_fd, mapped_addr))));

  return Status::Success;
}

Status
SharedMemoryManager::UnregisterSharedMemory(const std::string& name)
{
  // Serialize all operations that write/read current shared memory regions
  std::lock_guard<std::mutex> lock(register_mu_);

  auto it = shared_memory_map_.find(name);
  if (it != shared_memory_map_.end()) {
    RETURN_IF_ERROR(
        UnmapSharedMemory(it->second->mapped_addr_, it->second->byte_size_));

    // remove region info from shared_memory_map_
    shared_memory_map_.erase(it);

    // if no other region with same shm_key then close
    bool last_one = true;
    for (auto itr = shared_memory_map_.begin(); itr != shared_memory_map_.end();
         ++itr) {
      if (itr->second->shm_key_ == it->second->shm_key_) {
        last_one = false;
        break;
      }
    }
    if (last_one) {
      RETURN_IF_ERROR(CloseSharedMemoryRegion(it->second->shm_fd_));
    }
  }

  return Status::Success;
}

Status
SharedMemoryManager::UnregisterAllSharedMemory()
{
  // Serialize all operations that write/read current shared memory regions
  std::lock_guard<std::mutex> lock(register_mu_);

  std::string error_message =
      "Failed to unregister the following shared memory regions: ";
  std::vector<std::string> unregister_fails;
  for (const auto& shm_info : shared_memory_map_) {
    Status unregister_status = UnregisterSharedMemory(shm_info.first);
    if (!unregister_status.IsOk()) {
      unregister_fails.push_back(shm_info.first);
    }
  }

  if (!unregister_fails.empty()) {
    for (auto unreg_fail : unregister_fails) {
      error_message += unreg_fail + " ,";
    }
    LOG_ERROR << error_message;
  }

  return Status::Success;
}

Status
SharedMemoryManager::GetSharedMemoryStatus(
    std::vector<SharedMemoryInfo*>* active_shm_regions)
{
  active_shm_regions->clear();
  for (const auto& shm_info : shared_memory_map_) {
    active_shm_regions->push_back(shm_info.second.get());
  }

  return Status::Success;
}

SharedMemoryManager::SharedMemoryManager(
    const std::shared_ptr<ServerStatusManager>& status_manager)
    : status_manager_(status_manager)
{
}

SharedMemoryManager::~SharedMemoryManager()
{
  UnregisterAllSharedMemory();
}

Status
SharedMemoryManager::Create(
    const std::shared_ptr<ServerStatusManager>& status_manager,
    std::unique_ptr<SharedMemoryManager>* shared_memory_manager)
{
  // Not setting the smart pointer directly to simplify clean up
  std::unique_ptr<SharedMemoryManager> tmp_manager(
      new SharedMemoryManager(status_manager));
  *shared_memory_manager = std::move(tmp_manager);

  return Status::Success;
}

Status
SharedMemoryManager::SharedMemoryAddress(
    const std::string& name, size_t offset, size_t byte_size,
    void** shm_mapped_addr)
{
  auto it = shared_memory_map_.find(name);
  if (it == shared_memory_map_.end()) {
    return Status(
        RequestStatusCode::INTERNAL,
        "Unable to find shared memory region: '" + name + "'");
  }

  *shm_mapped_addr =
      (void*)((uint8_t*)it->second->mapped_addr_ + it->second->offset_ + offset);
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
