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
#pragma once

#include <unordered_map>
#include "src/core/trtserver.h"
#include "src/servers/common.h"

namespace nvidia { namespace inferenceserver {

///
/// Manage TRTSERVER_SharedMemoryBlock created by trtserver
///
class SharedMemoryBlockManager {
 public:
  SharedMemoryBlockManager() = default;
  ~SharedMemoryBlockManager();

  /// Add a shared memory block representing shared memory in system
  /// (CPU) memory to the manager. Return
  /// TRTSERVER_ERROR_ALREADY_EXISTS if a shared memory block of the
  /// same name already exists in the manager.
  /// \param smb Returns the shared memory block.
  /// \param name The name of the memory block.
  /// \param shm_key The name of the posix shared memory object
  /// containing the block of memory.
  /// \param offset The offset within the shared memory object to the
  /// start of the block.
  /// \param byte_size The size, in bytes of the block.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* Create(
      TRTSERVER_SharedMemoryBlock** smb, const std::string& name,
      const std::string& shm_key, const size_t offset, const size_t byte_size);

  /// Get a named shared memory block. Return
  /// TRTSERVER_ERROR_NOT_FOUND if named block doesn't exist.
  /// \param smb Returns the shared memory block.
  /// \param name The name of the shared memory block to get.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* Get(
      TRTSERVER_SharedMemoryBlock** smb, const std::string& name);

  /// Find a named shared memory block. Return 'smb' == nullptr if the
  /// named block doesn't exist.
  /// \param smb Returns the shared memory block.
  /// \param name The name of the shared memory block to find.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* Find(
      TRTSERVER_SharedMemoryBlock** smb, const std::string& name);

  /// Remove from manager and return a named shared memory block
  /// Ownership of 'smb' is transferred to the caller which is
  /// responsible for deleting the object. Return 'smb' == nullptr if
  /// the named block doesn't exist.
  /// \param smb Returns the shared memory block.
  /// \param name The name of the shared memory block to remove.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* Remove(
      TRTSERVER_SharedMemoryBlock** smb, const std::string& name);

  /// Remove all shared memory blocks from the manager.
  /// \return a TRTSERVER_Error indicating success or failure.
  TRTSERVER_Error* Clear();

 private:
  std::unordered_map<std::string, TRTSERVER_SharedMemoryBlock*> blocks_;
};

}}  // namespace nvidia::inferenceserver
