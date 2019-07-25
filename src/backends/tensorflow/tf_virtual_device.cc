// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/tensorflow/tf_virtual_device.h"

#include <unordered_map>
#include <vector>

namespace nvidia { namespace inferenceserver {

std::unique_ptr<VirtualDeviceTracker> VirtualDeviceTracker::instance_;

VirtualDeviceTracker::VirtualDeviceTracker(
    const std::map<int, std::vector<float>>& memory_limit_mb)
{
  // Initialize virtual device counter
  int base_index = 0;
  for (auto const& allocation : memory_limit_mb) {
    int gpu_idx = allocation.first;
    auto mem_limits_for_device = allocation.second;
    virtual_device_base_index_[gpu_idx] = base_index;

    // default 1 vgpu case
    if (mem_limits_for_device.empty()) {
      num_virtual_per_physical_[gpu_idx] = 1;
    } else {
      num_virtual_per_physical_[gpu_idx] = (mem_limits_for_device.size());
    }

    virtual_device_ids_.emplace(
        std::piecewise_construct, std::forward_as_tuple(gpu_idx),
        std::forward_as_tuple(0));
    base_index += num_virtual_per_physical_[gpu_idx];
  }
}

Status
VirtualDeviceTracker::Init(
    const std::map<int, std::vector<float>>& memory_limit_mb)
{
  if (memory_limit_mb.empty()) {
    return Status::Success;
  }

  // Initialize it once
  static std::once_flag instance_initialized;

  std::call_once(
      instance_initialized,
      [](const std::map<int, std::vector<float>>& memory_limit_mb) {
        instance_.reset(new VirtualDeviceTracker(memory_limit_mb));
      },
      memory_limit_mb);

  return Status::Success;
}

Status
VirtualDeviceTracker::GetNextVirtualDevice(
    const int gpu_device, int* vgpu_device)
{
  // Check for instantiation
  if (!instance_) {
    return Status(
        RequestStatusCode::INTERNAL,
        "VirtualDeviceTracker has not been initialized");
  }

  // Check if physical device index has a mapping
  if (instance_->virtual_device_ids_.find(gpu_device) ==
      instance_->virtual_device_ids_.end()) {
    return Status(
        RequestStatusCode::INTERNAL, "Invalid physical device ID " +
                                         std::to_string(gpu_device) +
                                         " while creating model instance");
  }

  // Get device tracker and next device id
  *vgpu_device = instance_->NextDeviceId(gpu_device);
  return Status::Success;
}

bool
VirtualDeviceTracker::HasVirtualDevice()
{
  return (instance_ != nullptr);
}

int
VirtualDeviceTracker::NextDeviceId(const int gpu_device)
{
  // Read and atomically increment virtual device id to use for creating model
  int num_vgpus_on_device = num_virtual_per_physical_[gpu_device];
  int base_index = virtual_device_base_index_[gpu_device];
  return ((virtual_device_ids_[gpu_device]++) % num_vgpus_on_device) +
         base_index;
}

}}  // namespace nvidia::inferenceserver