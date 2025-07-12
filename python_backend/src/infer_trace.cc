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

#include "infer_trace.h"

namespace triton { namespace backend { namespace python {

InferenceTrace::InferenceTrace(const InferenceTrace& rhs)
{
  triton_trace_ = rhs.triton_trace_;
  trace_context_ = rhs.trace_context_;
}

InferenceTrace&
InferenceTrace::operator=(const InferenceTrace& rhs)
{
  triton_trace_ = rhs.triton_trace_;
  trace_context_ = rhs.trace_context_;
  return *this;
}

InferenceTrace::InferenceTrace(std::unique_ptr<InferenceTrace>& trace_shm)
{
  triton_trace_ = trace_shm->triton_trace_;
  trace_context_ = trace_shm->trace_context_;
}

void
InferenceTrace::SaveToSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<InferenceTraceShm> infer_trace_shm =
      shm_pool->Construct<InferenceTraceShm>();
  infer_trace_shm_ptr_ = infer_trace_shm.data_.get();

  infer_trace_shm_ptr_->triton_trace = triton_trace_;

  std::unique_ptr<PbString> trace_context_shm =
      PbString::Create(shm_pool, trace_context_);

  infer_trace_shm_ptr_->trace_context_shm_handle =
      trace_context_shm->ShmHandle();

  // Save the references to shared memory.
  trace_context_shm_ = std::move(trace_context_shm);
  infer_trace_shm_ = std::move(infer_trace_shm);
  shm_handle_ = infer_trace_shm_.handle_;
}

std::unique_ptr<InferenceTrace>
InferenceTrace::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<InferenceTraceShm> infer_trace_shm =
      shm_pool->Load<InferenceTraceShm>(handle);
  InferenceTraceShm* infer_trace_shm_ptr = infer_trace_shm.data_.get();

  std::unique_ptr<PbString> trace_context_shm = PbString::LoadFromSharedMemory(
      shm_pool, infer_trace_shm_ptr->trace_context_shm_handle);

  return std::unique_ptr<InferenceTrace>(
      new InferenceTrace(infer_trace_shm, trace_context_shm));
}

InferenceTrace::InferenceTrace(
    AllocatedSharedMemory<InferenceTraceShm>& infer_trace_shm,
    std::unique_ptr<PbString>& trace_context_shm)
    : infer_trace_shm_(std::move(infer_trace_shm)),
      trace_context_shm_(std::move(trace_context_shm))
{
  infer_trace_shm_ptr_ = infer_trace_shm_.data_.get();
  shm_handle_ = infer_trace_shm_.handle_;
  triton_trace_ = infer_trace_shm_ptr_->triton_trace;
  trace_context_ = trace_context_shm_->String();
}

}}};  // namespace triton::backend::python
