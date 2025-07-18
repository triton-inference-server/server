// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pb_bls_cancel.h"

#include "pb_stub.h"
#include "pb_stub_log.h"

namespace triton { namespace backend { namespace python {

void
PbBLSCancel::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  cancel_shm_ = shm_pool->Construct<CancelBLSRequestMessage>();
  new (&(cancel_shm_.data_->mu)) bi::interprocess_mutex;
  new (&(cancel_shm_.data_->cv)) bi::interprocess_condition;
  cancel_shm_.data_->waiting_on_stub = false;
  cancel_shm_.data_->infer_payload_id = infer_playload_id_;
  cancel_shm_.data_->is_cancelled = is_cancelled_;
}

bi::managed_external_buffer::handle_t
PbBLSCancel::ShmHandle()
{
  return cancel_shm_.handle_;
}

CancelBLSRequestMessage*
PbBLSCancel::ShmPayload()
{
  return cancel_shm_.data_.get();
}

void
PbBLSCancel::Cancel()
{
  // Release the GIL. Python objects are not accessed during the check.
  py::gil_scoped_release gil_release;

  std::unique_lock<std::mutex> lk(mu_);
  // The cancelled flag can only move from false to true, not the other way, so
  // it is checked on each query until cancelled and then implicitly cached.
  if (is_cancelled_) {
    return;
  }
  if (!updating_) {
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    if (!stub->StubToParentServiceActive()) {
      LOG_ERROR << "Cannot communicate with parent service";
      return;
    }

    stub->EnqueueCancelBLSRequest(this);
    updating_ = true;
  }
  cv_.wait(lk, [this] { return !updating_; });
}

void
PbBLSCancel::ReportIsCancelled(bool is_cancelled)
{
  {
    std::lock_guard<std::mutex> lk(mu_);
    is_cancelled_ = is_cancelled;
    updating_ = false;
  }
  cv_.notify_all();
}

}}}  // namespace triton::backend::python
