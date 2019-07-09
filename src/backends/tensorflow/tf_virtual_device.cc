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

Status
VirtualDeviceTracker::Init(
    const std::vector<std::vector<float>>& memory_limit_mb)
{
  // Instantiate tracker object
  auto device_tracker = GetInstance();

  // Initialize virtual device counter
  for (size_t gpu_idx = 0; gpu_idx < memory_limit_mb.size(); gpu_idx++) {
    device_tracker->num_virtual_per_physical_.push_back(
        memory_limit_mb[gpu_idx].size());
    device_tracker->virtual_device_ids_.emplace(
        std::piecewise_construct, std::forward_as_tuple(gpu_idx),
        std::forward_as_tuple(0));
  }
  return Status::Success;
}

Status
VirtualDeviceTracker::GetNextVirtualDevice(
    const int gpu_device, int* vgpu_device)
{
  // Get the tracker
  auto device_tracker = GetInstance();
  // Check physical device index
  if (gpu_device < 0 || static_cast<unsigned int>(gpu_device) >=
                            device_tracker->num_virtual_per_physical_.size()) {
    return Status(
        RequestStatusCode::INTERNAL, "Encountered invalid physical device ID " +
                                         std::to_string(gpu_device) +
                                         " while creating model instance");
  }

  // Get device tracker and next device id
  *vgpu_device = device_tracker->NextDeviceId(gpu_device);
  return Status::Success;
}

int
VirtualDeviceTracker::NextDeviceId(const int gpu_device)
{
  // Read and atomically increment virtual device id to use for creating model
  int num_vgpus_on_device = num_virtual_per_physical_[gpu_device];
  return (virtual_device_ids_[gpu_device]++) % num_vgpus_on_device;
}

VirtualDeviceTracker*
VirtualDeviceTracker::GetInstance()
{
  static VirtualDeviceTracker virtual_device_tracker;
  return &virtual_device_tracker;
}

}}  // namespace nvidia::inferenceserver