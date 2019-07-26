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

Status
open_shm_region(const std::string& shm_key, int* shm_fd)
{
  // get shared memory region descriptor
  *shm_fd = shm_open(shm_key.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
  if (*shm_fd == -1) {
    return Status(
        RequestStatusCode::INTERNAL, "Unable to get shared memory descriptor");
  }

  return Status::Success;
}

Status
get_shm_addr(
    const int shm_fd, const size_t offset, const size_t byte_size,
    void** shm_addr)
{
  // map shared memory to process address space
  *shm_addr = mmap(NULL, byte_size, PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (*shm_addr == MAP_FAILED) {
    return Status(
        RequestStatusCode::INTERNAL, "Unable to process address space");
  }

  return Status::Success;
}

Status
shm_close(int shm_fd)
{
  int tmp = close(shm_fd);
  if (tmp == -1) {
    return Status(
        RequestStatusCode::INTERNAL, "Unable to close shared memory region");
  }

  return Status::Success;
}

Status
mmap_cleanup(void* shm_addr, size_t byte_size)
{
  int tmp_fd = munmap(shm_addr, byte_size);
  if (tmp_fd == -1) {
    return Status(
        RequestStatusCode::INTERNAL, "Unable to munmap shared memory region");
  }

  return Status::Success;
}

Status
SharedMemoryManager::RegisterSharedMemory(
    const std::string& name, const std::string& shm_key, const size_t offset,
    const size_t byte_size)
{
  LOG_VERBOSE(1) << "Register() shared memory region: '" << name << "'";
  Status status = Status::Success;

  // check if key is in shared_memory_map_ then remove
  if (shared_memory_map_.find(name) != shared_memory_map_.end()) {
    UnregisterSharedMemory(name);
  }

  // register (or re-register)
  try {
    void* tmp_addr;
    int shm_fd;

    // don't re-open if shared memory is alreay open
    for (auto itr = shared_memory_map_.begin(); itr != shared_memory_map_.end();
         ++itr) {
      if (itr->second->shm_key_ == shm_key)
        shm_fd = itr->second->shm_fd_;
      else {
        RETURN_IF_ERROR(open_shm_region(shm_key, &shm_fd));
      }
    }

    RETURN_IF_ERROR(get_shm_addr(shm_fd, offset, byte_size, &tmp_addr));
    std::unique_ptr<SharedMemoryInfo> shm_info(new SharedMemoryInfo(
        name, shm_key, offset, byte_size, shm_fd, tmp_addr));
    shared_memory_map_.insert(std::make_pair(
        name, std::unique_ptr<SharedMemoryInfo>(new SharedMemoryInfo(
                  name, shm_key, offset, byte_size, shm_fd, tmp_addr))));
  }
  catch (std::exception& ex) {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "Unable to register shared memory region successfully.");
  }

  return Status::Success;
}

Status
SharedMemoryManager::UnregisterSharedMemory(const std::string& name)
{
  LOG_VERBOSE(1) << "Unregister() '" << name << "'";

  auto it = shared_memory_map_.find(name);
  if (it != shared_memory_map_.end()) {
    void* tmp_addr;
    RETURN_IF_ERROR(get_shm_addr(
        it->second->shm_fd_, it->second->offset_, it->second->byte_size_,
        &tmp_addr));
    RETURN_IF_ERROR(mmap_cleanup(tmp_addr, it->second->byte_size_));

    // if no other region with same shm_key then close
    bool last_one = true;
    for (auto itr = shared_memory_map_.begin(); itr != shared_memory_map_.end();
         ++itr) {
      if (itr->second->shm_key_ == it->second->shm_key_)
        last_one = false;
    }
    if (last_one) {
      RETURN_IF_ERROR(shm_close(it->second->shm_fd_));
    }

    // remove region info from shared_memory_map_
    shared_memory_map_.erase(it);
  } else {
    return Status(
        RequestStatusCode::INVALID_ARG,
        "Cannot unregister shared memory region that has not been registered.");
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

void
SharedMemoryManager::Create(
    const std::shared_ptr<ServerStatusManager>& status_manager,
    std::unique_ptr<SharedMemoryManager>* shared_memory_manager)
{
  // Not setting the smart pointer directly to simplify clean up
  std::unique_ptr<SharedMemoryManager> tmp_manager(
      new SharedMemoryManager(status_manager));
  *shared_memory_manager = std::move(tmp_manager);
}

Status
SharedMemoryManager::RegisterUnregisterSharedMemory(
    const std::string& name, const std::string& shm_key, const size_t offset,
    const size_t byte_size, ActionType type)
{
  // Serialize all operations that change model state
  std::lock_guard<std::mutex> lock(register_mu_);

  // Update SharedMemoryInfo related to file system accordingly
  std::set<std::string> added, deleted, modified;
  {
    std::lock_guard<std::mutex> lk(shm_info_mu_);
    if (type == ActionType::UNREGISTER) {
      UnregisterSharedMemory(name);
    } else {
      RegisterSharedMemory(name, shm_key, offset, byte_size);
    }
  }

  return Status::Success;
}

Status
SharedMemoryManager::UnregisterAllSharedMemory()
{
  for (const auto& shm_info : shared_memory_map_) {
    Status unregister_status = UnregisterSharedMemory(shm_info.first);
    if (!unregister_status.IsOk()) {
      return Status(
          RequestStatusCode::INTERNAL,
          "Failed to gracefully unregister all shared memory regions: " +
              unregister_status.Message());
    }
  }
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
