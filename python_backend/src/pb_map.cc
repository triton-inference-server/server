// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pb_map.h"

namespace triton { namespace backend { namespace python {

std::unique_ptr<PbMap>
PbMap::Create(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    std::unordered_map<std::string, std::string>& map)
{
  std::vector<std::unique_ptr<PbString>> strings;
  AllocatedSharedMemory<DictShm> dict_shm = shm_pool->Construct<DictShm>();
  dict_shm.data_->length = map.size();

  AllocatedSharedMemory<PairShm> pair_shms =
      shm_pool->Construct<PairShm>(map.size());
  dict_shm.data_->values = pair_shms.handle_;

  size_t i = 0;
  for (auto& pair : map) {
    auto key = PbString::Create(shm_pool, pair.first);
    auto value = PbString::Create(shm_pool, pair.second);

    (pair_shms.data_.get())[i].key = key->ShmHandle();
    (pair_shms.data_.get())[i].value = value->ShmHandle();

    strings.emplace_back(std::move(key));
    strings.emplace_back(std::move(value));
    i++;
  }

  return std::unique_ptr<PbMap>(new PbMap(strings, dict_shm, pair_shms, map));
}

const std::unordered_map<std::string, std::string>&
PbMap::UnorderedMap()
{
  return map_;
}

bi::managed_external_buffer::handle_t
PbMap::ShmHandle()
{
  return dict_handle_;
}

std::unique_ptr<PbMap>
PbMap::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<DictShm> dict_shm = shm_pool->Load<DictShm>(handle);
  AllocatedSharedMemory<PairShm> pair_shms =
      shm_pool->Load<PairShm>(dict_shm.data_->values);

  std::vector<std::unique_ptr<PbString>> pb_strings;
  std::unordered_map<std::string, std::string> map;
  for (size_t i = 0; i < dict_shm.data_->length; i++) {
    std::unique_ptr<PbString> key = PbString::LoadFromSharedMemory(
        shm_pool, (pair_shms.data_.get())[i].key);

    std::unique_ptr<PbString> value = PbString::LoadFromSharedMemory(
        shm_pool, (pair_shms.data_.get())[i].value);

    map.insert({key->String(), value->String()});
    pb_strings.emplace_back(std::move(key));
    pb_strings.emplace_back(std::move(value));
  }

  return std::unique_ptr<PbMap>(
      new PbMap(pb_strings, dict_shm, pair_shms, map));
}

PbMap::PbMap(
    std::vector<std::unique_ptr<PbString>>& strings,
    AllocatedSharedMemory<DictShm>& dict_shm,
    AllocatedSharedMemory<PairShm>& pair_shms,
    std::unordered_map<std::string, std::string>& map)
    : strings_(std::move(strings)), dict_shm_(std::move(dict_shm)),
      pair_shms_(std::move(pair_shms)), map_(std::move(map))
{
  dict_handle_ = dict_shm.handle_;
}

}}}  // namespace triton::backend::python
