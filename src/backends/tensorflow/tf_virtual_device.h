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
#pragma once

#include "src/core/constants.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// This is a singleton class responsible for maintaining device ids
// in the creation of tensorflow virtual devices. This class should
// be initialized with a mapping from physical gpus to the number of
// virtual devices required on that physical gpu. Instantiation check
// can be done via HasVirtualDevice(). After initialization, the id
// of the next device to set as a default for a model instance can
// be obtained with GetNextVirtualDevice.

class VirtualDeviceTracker {
 public:
  // Creates the VirtualDeviceTracker and records a pointer to it.
  // Initializes the device tracker with the number of virtual gpus
  // for each physical gpu.
  static Status Init(const std::map<int, std::vector<float>>& memory_limit_mb);

  // Gets the device ID of the next available virtual device on the physical
  // device indexed by gpu_device (currently round robin). Updates internal
  // state of device id counter. Returns an error status if gpu_device is out of
  // bounds or if no VirtualDeviceTracker has been initialized
  static Status GetNextVirtualDevice(const int gpu_device, int* vgpu_device);

  // Returns True if the VirtualDevice Tracker has been initialized, else false.
  static bool HasVirtualDevice();

  ~VirtualDeviceTracker() = default;

 private:
  DISALLOW_COPY_AND_ASSIGN(VirtualDeviceTracker);

  // Currently just implements round robin id generation
  // but can potentially be replaced with something smarter
  int NextDeviceId(const int gpu_device);

  VirtualDeviceTracker(
      const std::map<int, std::vector<float>>& memory_limit_mb);

  std::unordered_map<int, std::atomic<size_t>> virtual_device_ids_;
  std::unordered_map<int, size_t> num_virtual_per_physical_;
  std::unordered_map<int, int> virtual_device_base_index_;
  static std::unique_ptr<VirtualDeviceTracker> instance_;
};

}}  // namespace nvidia::inferenceserver
