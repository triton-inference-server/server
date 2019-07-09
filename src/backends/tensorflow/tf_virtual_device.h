// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class VirtualDeviceTracker {
 public:
  static Status Create(
      const std::vector<std::vector<float>>& memory_limit_mb,
      VirtualDeviceTracker** device_tracker);

  Status GetNextDeviceId(const int gpu_device, int* vgpu_device);

  ~VirtualDeviceTracker() = default;

 private:
  VirtualDeviceTracker(const std::vector<std::vector<float>>& memory_limit_mb)
      : per_device_memory_(memory_limit_mb)
  {
    // Initialize virtual device counter
    for (size_t gpu_idx = 0; gpu_idx < memory_limit_mb.size(); gpu_idx++) {
      virtual_device_ids_.emplace(
          std::piecewise_construct, std::forward_as_tuple(gpu_idx),
          std::forward_as_tuple(0));
    }
  }
  std::unordered_map<int, std::atomic<size_t>> virtual_device_ids_;
  const std::vector<std::vector<float>>& per_device_memory_;
};

}}  // namespace nvidia::inferenceserver
