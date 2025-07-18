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

#include "correlation_id.h"

namespace triton { namespace backend { namespace python {

CorrelationId::CorrelationId()
    : id_string_(""), id_uint_(0), id_type_(CorrelationIdDataType::UINT64)
{
}

CorrelationId::CorrelationId(const std::string& id_string)
    : id_string_(id_string), id_uint_(0),
      id_type_(CorrelationIdDataType::STRING)
{
}

CorrelationId::CorrelationId(uint64_t id_uint)
    : id_string_(""), id_uint_(id_uint), id_type_(CorrelationIdDataType::UINT64)
{
}

CorrelationId::CorrelationId(const CorrelationId& rhs)
{
  id_uint_ = rhs.id_uint_;
  id_type_ = rhs.id_type_;
  id_string_ = rhs.id_string_;
}

CorrelationId::CorrelationId(std::unique_ptr<CorrelationId>& correlation_id_shm)
{
  id_uint_ = correlation_id_shm->id_uint_;
  id_type_ = correlation_id_shm->id_type_;
  id_string_ = correlation_id_shm->id_string_;
}

CorrelationId&
CorrelationId::operator=(const CorrelationId& rhs)
{
  id_uint_ = rhs.id_uint_;
  id_type_ = rhs.id_type_;
  id_string_ = rhs.id_string_;
  return *this;
}

void
CorrelationId::SaveToSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<CorrelationIdShm> correlation_id_shm =
      shm_pool->Construct<CorrelationIdShm>();
  correlation_id_shm_ptr_ = correlation_id_shm.data_.get();

  std::unique_ptr<PbString> id_string_shm =
      PbString::Create(shm_pool, id_string_);

  correlation_id_shm_ptr_->id_uint = id_uint_;
  correlation_id_shm_ptr_->id_string_shm_handle = id_string_shm->ShmHandle();
  correlation_id_shm_ptr_->id_type = id_type_;

  // Save the references to shared memory.
  correlation_id_shm_ = std::move(correlation_id_shm);
  id_string_shm_ = std::move(id_string_shm);
  shm_handle_ = correlation_id_shm_.handle_;
}

std::unique_ptr<CorrelationId>
CorrelationId::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<CorrelationIdShm> correlation_id_shm =
      shm_pool->Load<CorrelationIdShm>(handle);
  CorrelationIdShm* correlation_id_shm_ptr = correlation_id_shm.data_.get();

  std::unique_ptr<PbString> id_string_shm = PbString::LoadFromSharedMemory(
      shm_pool, correlation_id_shm_ptr->id_string_shm_handle);

  return std::unique_ptr<CorrelationId>(
      new CorrelationId(correlation_id_shm, id_string_shm));
}

CorrelationId::CorrelationId(
    AllocatedSharedMemory<CorrelationIdShm>& correlation_id_shm,
    std::unique_ptr<PbString>& id_string_shm)
    : correlation_id_shm_(std::move(correlation_id_shm)),
      id_string_shm_(std::move(id_string_shm))
{
  correlation_id_shm_ptr_ = correlation_id_shm_.data_.get();
  shm_handle_ = correlation_id_shm_.handle_;
  id_string_ = id_string_shm_->String();
  id_uint_ = correlation_id_shm_ptr_->id_uint;
  id_type_ = correlation_id_shm_ptr_->id_type;
}

}}};  // namespace triton::backend::python
