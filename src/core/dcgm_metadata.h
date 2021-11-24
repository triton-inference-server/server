// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dcgm_agent.h>

struct DcgmMetadata {
  // DCGM handles for initialization and destruction
  dcgmHandle_t dcgm_handle_ = 0;
  dcgmGpuGrp_t groupId_ = 0;
  // DCGM Flags
  bool standalone_ = false;
  // DCGM Fields
  size_t field_count_ = 0;
  std::vector<unsigned short> fields_;
  // GPU Device Mapping
  std::map<uint32_t, uint32_t> cuda_ids_to_dcgm_ids_;
  std::vector<uint32_t> available_cuda_gpu_ids_;
  // Stop attempting metrics if they fail multiple consecutive
  // times for a device.
  const int fail_threshold_ = 3;
  // DCGM Failure Tracking
  std::vector<int> power_limit_fail_cnt_;
  std::vector<int> power_usage_fail_cnt_;
  std::vector<int> energy_fail_cnt_;
  std::vector<int> util_fail_cnt_;
  std::vector<int> mem_fail_cnt_;
  // DCGM Energy Tracking
  std::vector<unsigned long long> last_energy_;
};
