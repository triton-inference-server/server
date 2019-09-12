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

#include "src/servers/shared_memory_block_manager.h"

namespace nvidia { namespace inferenceserver {

SharedMemoryBlockManager::~SharedMemoryBlockManager()
{
  TRTSERVER_Error* err = Clear();
  if (err != nullptr) {
    LOG_ERROR << TRTSERVER_ErrorMessage(err);
  }
}

TRTSERVER_Error*
SharedMemoryBlockManager::Create(
    TRTSERVER_SharedMemoryBlock** smb, const std::string& name,
    const std::string& shm_key, const size_t offset, const size_t byte_size)
{
  *smb = nullptr;

  if (blocks_.find(name) != blocks_.end()) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_ALREADY_EXISTS,
        std::string("shared memory block '" + name + "' already in manager")
            .c_str());
  }

  RETURN_IF_ERR(TRTSERVER_SharedMemoryBlockCpuNew(
      smb, name.c_str(), shm_key.c_str(), offset, byte_size));
  blocks_.emplace(name, *smb);

  return nullptr;  // success
}

TRTSERVER_Error*
SharedMemoryBlockManager::Get(
    TRTSERVER_SharedMemoryBlock** smb, const std::string& name)
{
  *smb = nullptr;

  auto itr = blocks_.find(name);
  if (itr == blocks_.end()) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_NOT_FOUND,
        std::string("shared memory block '" + name + "' not found in manager")
            .c_str());
  }

  *smb = itr->second;

  return nullptr;  // success
}

TRTSERVER_Error*
SharedMemoryBlockManager::Find(
    TRTSERVER_SharedMemoryBlock** smb, const std::string& name)
{
  *smb = nullptr;

  auto itr = blocks_.find(name);
  if (itr != blocks_.end()) {
    *smb = itr->second;
  }

  return nullptr;  // success
}

TRTSERVER_Error*
SharedMemoryBlockManager::Remove(
    TRTSERVER_SharedMemoryBlock** smb, const std::string& name)
{
  *smb = nullptr;

  auto itr = blocks_.find(name);
  if (itr != blocks_.end()) {
    *smb = itr->second;
    blocks_.erase(itr);
  }

  return nullptr;  // success
}

TRTSERVER_Error*
SharedMemoryBlockManager::Clear()
{
  std::string failed_blocks;

  auto it = blocks_.begin();
  while (it != blocks_.cend()) {
    TRTSERVER_Error* err = TRTSERVER_SharedMemoryBlockDelete(it->second);
    if (err != nullptr) {
      if (failed_blocks.empty()) {
        failed_blocks = it->first;
      } else {
        failed_blocks += ", " + it->first;
      }
      ++it;
    } else {
      blocks_.erase(it++);
    }
  }

  if (!failed_blocks.empty()) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INTERNAL,
        std::string("failed to delete shared memory blocks: " + failed_blocks)
            .c_str());
  }

  return nullptr;  // success
}

}}  // namespace nvidia::inferenceserver
