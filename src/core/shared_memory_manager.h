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
#pragma once

#include <mutex>
#include "src/core/grpc_service.pb.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferenceServer;
class InferenceBackend;
class ServerStatusManager;

/// An object to manage the registered shared memory regions in the server.
class SharedMemoryManager {
 public:
  /// A struct that records the shared memory regions registered by the shared
  /// memory manager.
  struct SharedMemoryInfo {
    SharedMemoryInfo(
        const std::string& name, const std::string& shm_key,
        const size_t offset, const size_t byte_size, int shm_fd, void* shm_addr)
        : name_(name), shm_key_(shm_key), offset_(offset),
          byte_size_(byte_size), shm_fd_(shm_fd), shm_addr_(shm_addr)
    {
    }

    std::string name_;
    std::string shm_key_;
    size_t offset_;
    size_t byte_size_;
    int shm_fd_;
    void* shm_addr_;
  };

  using SharedMemoryStateMap =
      std::map<std::string, std::unique_ptr<SharedMemoryInfo>>;

  enum ActionType { NO_ACTION, REGISTER, UNREGISTER };

  ~SharedMemoryManager();

  /// Register or Unregister a specified shared memory region.
  /// \param name The user-given name for the shared memory region to be
  /// registered.
  /// \param shm_key The unique name of the location in shared memory being
  /// registered.
  /// \param offset The offset into the shared memory region.
  /// \param byte_size The size, in bytes of the tensor data.
  /// \parm type The type action to be performed. If the action is REGISTER and
  /// the shared memory region has been registered, the shared memory region
  /// will be re-registered.
  /// \return error status. Return "NOT_FOUND" if it tries to unregister a
  /// shared memory region that hasn't been registered.
  Status RegisterUnregisterSharedMemory(
      const std::string& name, const std::string& shm_key, const size_t offset,
      const size_t byte_size, ActionType type);

  /// Helper function that (Re)registers shared memory region if valid
  Status RegisterSharedMemory(
      const std::string& name, const std::string& shm_key, const size_t offset,
      const size_t byte_size);

  /// Helper function that Unregisters shared memory region if it is registered
  Status UnregisterSharedMemory(const std::string& name);

  /// Unregisters all shared memory regions. This function can be called by the
  /// user. It should be called before shutting down the shared memory manager.
  Status UnregisterAllSharedMemory();

  /// updates the list of all registered shared memory regions.
  Status GetLiveSharedMemory(std::vector<std::string>& active_shm_regions);

  /// Creates a SharedMemoryManager object that uses the given status_manager
  static Status Create(
      const std::shared_ptr<ServerStatusManager>& status_manager,
      std::unique_ptr<SharedMemoryManager>* shared_memory_manager);

 private:
  SharedMemoryManager(
      const std::shared_ptr<ServerStatusManager>& status_manager);

  std::mutex shm_info_mu_;
  std::mutex register_mu_;

  std::shared_ptr<ServerStatusManager> status_manager_;
  SharedMemoryStateMap shared_memory_map_;
};

}}  // namespace nvidia::inferenceserver
