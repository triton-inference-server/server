// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "pb_string.h"
#include "pb_utils.h"

namespace triton { namespace backend { namespace python {

enum class CorrelationIdDataType { UINT64, STRING };

struct CorrelationIdShm {
  bi::managed_external_buffer::handle_t id_string_shm_handle;
  uint64_t id_uint;
  CorrelationIdDataType id_type;
};

class CorrelationId {
 public:
  CorrelationId();
  CorrelationId(const std::string& id_string);
  CorrelationId(uint64_t id_uint);
  CorrelationId(const CorrelationId& rhs);
  CorrelationId(std::unique_ptr<CorrelationId>& correlation_id_shm);
  CorrelationId& operator=(const CorrelationId& rhs);

  /// Save CorrelationId object to shared memory.
  /// \param shm_pool Shared memory pool to save the CorrelationId object.
  void SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool);

  /// Create a CorrelationId object from shared memory.
  /// \param shm_pool Shared memory pool
  /// \param handle Shared memory handle of the CorrelationId.
  /// \return Returns the CorrelationId in the specified handle
  /// location.
  static std::unique_ptr<CorrelationId> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);

  // Function that help determine exact type of Correlation Id
  CorrelationIdDataType Type() const { return id_type_; }

  // Get the value of the CorrelationId based on the type
  const std::string& StringValue() const { return id_string_; }
  uint64_t UnsignedIntValue() const { return id_uint_; }

  bi::managed_external_buffer::handle_t ShmHandle() { return shm_handle_; }

 private:
  // The private constructor for creating a CorrelationId object from shared
  // memory.
  CorrelationId(
      AllocatedSharedMemory<CorrelationIdShm>& correlation_id_shm,
      std::unique_ptr<PbString>& id_string_shm);

  std::string id_string_;
  uint64_t id_uint_;
  CorrelationIdDataType id_type_;

  // Shared Memory Data Structures
  AllocatedSharedMemory<CorrelationIdShm> correlation_id_shm_;
  CorrelationIdShm* correlation_id_shm_ptr_;
  bi::managed_external_buffer::handle_t shm_handle_;
  std::unique_ptr<PbString> id_string_shm_;
};

}}};  // namespace triton::backend::python
