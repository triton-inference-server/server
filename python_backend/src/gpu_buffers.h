// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pb_string.h"
#include "pb_utils.h"
#include "scoped_defer.h"

namespace triton { namespace backend { namespace python {

/// \param success indicating whether the process of fetching the GPU buffers
/// was successful.
/// \param error if success is equal to false, the error object will be set.
/// \param buffers list of buffers elements.
/// \param buffer_count the number of buffers.
struct GPUBuffersShm {
  bool success;
  bi::managed_external_buffer::handle_t error;
  bi::managed_external_buffer::handle_t buffers;
  uint32_t buffer_count;
};

/// Helper class to facilitate transfer of metadata associated
/// the GPU buffers in shared memory.
class GPUBuffersHelper {
 public:
  GPUBuffersHelper();
  void AddBuffer(const bi::managed_external_buffer::handle_t& handle);
  void Complete(std::unique_ptr<SharedMemoryManager>& shm_pool);
  void SetError(
      std::unique_ptr<SharedMemoryManager>& shm_pool, const std::string& error);
  bi::managed_external_buffer::handle_t ShmHandle();

 private:
  AllocatedSharedMemory<GPUBuffersShm> gpu_buffers_shm_;
  std::vector<bi::managed_external_buffer::handle_t> buffers_;
  AllocatedSharedMemory<bi::managed_external_buffer::handle_t>
      buffers_handle_shm_;
  std::unique_ptr<PbString> error_shm_;
  bool completed_;
};

}}};  // namespace triton::backend::python
