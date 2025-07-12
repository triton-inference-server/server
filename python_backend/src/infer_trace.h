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

struct InferenceTraceShm {
  bi::managed_external_buffer::handle_t trace_context_shm_handle;
  // The address of the 'TRITONSERVER_InferTrace' object.
  void* triton_trace;
};

//
// Inference Trace
//
class InferenceTrace {
 public:
  InferenceTrace(void* triton_trace, const std::string& ctxt)
      : triton_trace_(triton_trace), trace_context_(ctxt)
  {
  }
  InferenceTrace() : triton_trace_(nullptr), trace_context_("") {}
  InferenceTrace(const InferenceTrace& rhs);
  InferenceTrace(std::unique_ptr<InferenceTrace>& trace_shm);
  InferenceTrace& operator=(const InferenceTrace& rhs);
  /// Save InferenceTrace object to shared memory.
  /// \param shm_pool Shared memory pool to save the InferenceTrace object.
  void SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool);

  /// Create a InferenceTrace object from shared memory.
  /// \param shm_pool Shared memory pool
  /// \param handle Shared memory handle of the InferenceTrace.
  /// \return Returns the InferenceTrace in the specified handle
  /// location.
  static std::unique_ptr<InferenceTrace> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);

  void* TritonTrace() { return triton_trace_; }
  const std::string& Context() const { return trace_context_; }

  bi::managed_external_buffer::handle_t ShmHandle() { return shm_handle_; }

 private:
  // The private constructor for creating a InferenceTrace object from shared
  // memory.
  InferenceTrace(
      AllocatedSharedMemory<InferenceTraceShm>& infer_trace_shm,
      std::unique_ptr<PbString>& trace_context_shm);

  void* triton_trace_;
  std::string trace_context_;

  // Shared Memory Data Structures
  AllocatedSharedMemory<InferenceTraceShm> infer_trace_shm_;
  InferenceTraceShm* infer_trace_shm_ptr_;
  bi::managed_external_buffer::handle_t shm_handle_;
  std::unique_ptr<PbString> trace_context_shm_;
};

}}};  // namespace triton::backend::python
