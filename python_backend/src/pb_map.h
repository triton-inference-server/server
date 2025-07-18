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

#pragma once

#include <unordered_map>

#include "pb_string.h"
#include "shm_manager.h"

namespace triton { namespace backend { namespace python {

struct PairShm {
  bi::managed_external_buffer::handle_t key;
  bi::managed_external_buffer::handle_t value;
};

struct DictShm {
  uint32_t length;
  // `values` point to the location where there are `length` of Pair objects.
  bi::managed_external_buffer::handle_t values;
};


class PbMap {
 public:
  static std::unique_ptr<PbMap> Create(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      std::unordered_map<std::string, std::string>& map);
  static std::unique_ptr<PbMap> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);
  const std::unordered_map<std::string, std::string>& UnorderedMap();
  bi::managed_external_buffer::handle_t ShmHandle();

 private:
  PbMap(
      std::vector<std::unique_ptr<PbString>>& strings,
      AllocatedSharedMemory<DictShm>& dict_shm,
      AllocatedSharedMemory<PairShm>& pair_shms,
      std::unordered_map<std::string, std::string>& map);

  std::vector<std::unique_ptr<PbString>> strings_;
  AllocatedSharedMemory<DictShm> dict_shm_;
  AllocatedSharedMemory<PairShm> pair_shms_;
  bi::managed_external_buffer::handle_t dict_handle_;
  std::unordered_map<std::string, std::string> map_;
};
}}}  // namespace triton::backend::python
