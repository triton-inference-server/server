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

#include "gpu_buffers.h"

#include "pb_string.h"

namespace triton { namespace backend { namespace python {
GPUBuffersHelper::GPUBuffersHelper()
{
  completed_ = false;
}

void
GPUBuffersHelper::AddBuffer(const bi::managed_external_buffer::handle_t& handle)
{
  if (completed_) {
    throw PythonBackendException(
        "It is not possible to add buffers after 'Complete' has been called on "
        "a GPUBuffersHelper.");
  }

  buffers_.emplace_back(handle);
}

void
GPUBuffersHelper::SetError(
    std::unique_ptr<SharedMemoryManager>& shm_pool, const std::string& error)
{
  error_shm_ = PbString::Create(shm_pool, error);
}

void
GPUBuffersHelper::Complete(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  if (completed_) {
    throw PythonBackendException(
        "Complete has already been called. Complete should only be called "
        "once.");
  }
  gpu_buffers_shm_ = shm_pool->Construct<GPUBuffersShm>();
  if (!error_shm_) {
    buffers_handle_shm_ =
        shm_pool->Construct<bi::managed_external_buffer::handle_t>(
            buffers_.size());
    gpu_buffers_shm_.data_->buffer_count = buffers_.size();
    gpu_buffers_shm_.data_->success = true;
    gpu_buffers_shm_.data_->buffers = buffers_handle_shm_.handle_;
    for (size_t i = 0; i < buffers_.size(); ++i) {
      buffers_handle_shm_.data_.get()[i] = buffers_[i];
    }
  } else {
    gpu_buffers_shm_.data_->success = false;
    gpu_buffers_shm_.data_->error = error_shm_->ShmHandle();
  }
  completed_ = true;
}


bi::managed_external_buffer::handle_t
GPUBuffersHelper::ShmHandle()
{
  return gpu_buffers_shm_.handle_;
}

}}}  // namespace triton::backend::python
