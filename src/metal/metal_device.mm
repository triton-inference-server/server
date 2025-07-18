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

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_device.h"
#include <iostream>
#include <sstream>

namespace triton { namespace core { namespace metal {

// Thread-local storage for device affinity
thread_local int MetalDeviceManager::thread_device_affinity_ = -1;

//
// MetalDevice Implementation
//

MetalDevice::MetalDevice(id device) : device_(device), device_id_(-1) {
  if (device_) {
    [(id<MTLDevice>)device_ retain];
    InitializeCapabilities();
  }
}

MetalDevice::~MetalDevice() {
  if (device_) {
    [(id<MTLDevice>)device_ release];
  }
}

std::unique_ptr<MetalDevice> MetalDevice::Create(id device) {
  if (!device) {
    return nullptr;
  }
  return std::unique_ptr<MetalDevice>(new MetalDevice(device));
}

void MetalDevice::InitializeCapabilities() {
  id<MTLDevice> mtl_device = (id<MTLDevice>)device_;
  
  // Basic device information
  capabilities_.name = [mtl_device.name UTF8String];
  capabilities_.registry_id = mtl_device.registryID;
  
  // Device characteristics
  capabilities_.is_low_power = mtl_device.isLowPower;
  capabilities_.is_removable = mtl_device.isRemovable;
  
  // Determine if integrated (unified memory architecture)
#if defined(__MAC_13_0) || defined(__IPHONE_16_0)
  if (@available(macOS 13.0, iOS 16.0, *)) {
    capabilities_.is_integrated = mtl_device.hasUnifiedMemory;
  } else
#endif
  {
    // For older systems, assume integrated if low power
    capabilities_.is_integrated = capabilities_.is_low_power;
  }
  
  // Feature set detection
  capabilities_.feature_set = MetalFeatureSet::Unknown;
  
  // Check Mac feature sets
#if TARGET_OS_OSX
  if ([mtl_device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily2_v1]) {
    capabilities_.feature_set = MetalFeatureSet::Mac2;
  } else if ([mtl_device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily1_v4]) {
    capabilities_.feature_set = MetalFeatureSet::Mac1;
  }
#endif
  
  // Check iOS/Apple Silicon feature sets
#if defined(__MAC_11_0) || defined(__IPHONE_14_0)
  if (@available(macOS 11.0, iOS 14.0, *)) {
    if ([mtl_device supportsFamily:MTLGPUFamilyApple8]) {
      capabilities_.feature_set = MetalFeatureSet::Apple8;
    } else if ([mtl_device supportsFamily:MTLGPUFamilyApple7]) {
      capabilities_.feature_set = MetalFeatureSet::Apple7;
    } else if ([mtl_device supportsFamily:MTLGPUFamilyApple6]) {
      capabilities_.feature_set = MetalFeatureSet::Apple6;
    } else if ([mtl_device supportsFamily:MTLGPUFamilyApple5]) {
      capabilities_.feature_set = MetalFeatureSet::Apple5;
    } else if ([mtl_device supportsFamily:MTLGPUFamilyApple4]) {
      capabilities_.feature_set = MetalFeatureSet::Apple4;
    } else if ([mtl_device supportsFamily:MTLGPUFamilyApple3]) {
      capabilities_.feature_set = MetalFeatureSet::Apple3;
    } else if ([mtl_device supportsFamily:MTLGPUFamilyApple2]) {
      capabilities_.feature_set = MetalFeatureSet::Apple2;
    } else if ([mtl_device supportsFamily:MTLGPUFamilyApple1]) {
      capabilities_.feature_set = MetalFeatureSet::Apple1;
    }
  }
#endif
  
  // Compute capabilities
  capabilities_.max_threads_per_threadgroup = mtl_device.maxThreadsPerThreadgroup.width;
  capabilities_.max_buffer_length = mtl_device.maxBufferLength;
  
  // Texture capabilities
#if defined(__MAC_10_13) || defined(__IPHONE_11_0)
  if (@available(macOS 10.13, iOS 11.0, *)) {
    capabilities_.max_texture_dimension_1d = 16384;  // Default for most devices
    capabilities_.max_texture_dimension_2d = 16384;
    capabilities_.max_texture_dimension_3d = 2048;
  }
#endif
  
  // Advanced feature support
#if defined(__MAC_11_0) || defined(__IPHONE_14_0)
  if (@available(macOS 11.0, iOS 14.0, *)) {
    capabilities_.supports_raytracing = [mtl_device supportsRaytracing];
    capabilities_.supports_function_pointers = [mtl_device supportsFunctionPointers];
    capabilities_.supports_dynamic_libraries = [mtl_device supportsDynamicLibraries];
  } else
#endif
  {
    capabilities_.supports_raytracing = false;
    capabilities_.supports_function_pointers = false;
    capabilities_.supports_dynamic_libraries = false;
  }
  
  // Memory information
  capabilities_.recommended_max_working_set_size = mtl_device.recommendedMaxWorkingSetSize;
}

bool MetalDevice::SupportsFeatureSet(MetalFeatureSet feature_set) const {
  return static_cast<int>(capabilities_.feature_set) >= static_cast<int>(feature_set);
}

size_t MetalDevice::GetAvailableMemory() const {
  std::lock_guard<std::mutex> lock(mutex_);
  id<MTLDevice> mtl_device = (id<MTLDevice>)device_;
  
#if defined(__MAC_10_13) || defined(__IPHONE_11_0)
  if (@available(macOS 10.13, iOS 11.0, *)) {
    return mtl_device.currentAllocatedSize;
  }
#endif
  
  // Fallback: return a portion of recommended working set
  return capabilities_.recommended_max_working_set_size * 3 / 4;
}

size_t MetalDevice::GetTotalMemory() const {
  // Metal doesn't provide direct access to total GPU memory
  // Use recommended working set size as an approximation
  return capabilities_.recommended_max_working_set_size;
}

bool MetalDevice::IsSystemDefault() const {
  id<MTLDevice> mtl_device = (id<MTLDevice>)device_;
  id<MTLDevice> default_device = MTLCreateSystemDefaultDevice();
  bool is_default = (mtl_device == default_device);
  [default_device release];
  return is_default;
}

std::unique_ptr<MetalCommandQueue> MetalDevice::CreateCommandQueue() {
  id<MTLDevice> mtl_device = (id<MTLDevice>)device_;
  id<MTLCommandQueue> queue = [mtl_device newCommandQueue];
  if (!queue) {
    return nullptr;
  }
  return MetalCommandQueue::Create(queue);
}

//
// MetalCommandQueue Implementation
//

MetalCommandQueue::MetalCommandQueue(id queue) : queue_(queue) {
  if (queue_) {
    [(id<MTLCommandQueue>)queue_ retain];
  }
}

MetalCommandQueue::~MetalCommandQueue() {
  if (queue_) {
    [(id<MTLCommandQueue>)queue_ release];
  }
}

std::unique_ptr<MetalCommandQueue> MetalCommandQueue::Create(id queue) {
  if (!queue) {
    return nullptr;
  }
  return std::unique_ptr<MetalCommandQueue>(new MetalCommandQueue(queue));
}

id MetalCommandQueue::CreateCommandBuffer() {
  id<MTLCommandQueue> mtl_queue = (id<MTLCommandQueue>)queue_;
  return [mtl_queue commandBuffer];
}

void MetalCommandQueue::WaitUntilCompleted() {
  id<MTLCommandQueue> mtl_queue = (id<MTLCommandQueue>)queue_;
  id<MTLCommandBuffer> cmd_buffer = [mtl_queue commandBuffer];
  [cmd_buffer commit];
  [cmd_buffer waitUntilCompleted];
}

//
// MetalDeviceManager Implementation
//

MetalDeviceManager& MetalDeviceManager::Instance() {
  static MetalDeviceManager instance;
  return instance;
}

void MetalDeviceManager::Initialize() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!devices_.empty()) {
    return;  // Already initialized
  }
  
  EnumerateDevices();
}

void MetalDeviceManager::EnumerateDevices() {
  NSArray<id<MTLDevice>>* devices = nil;
  
#if TARGET_OS_OSX
  // On macOS, use MTLCopyAllDevices to get all GPUs
  devices = MTLCopyAllDevices();
#else
  // On iOS/tvOS, only system default device is available
  id<MTLDevice> default_device = MTLCreateSystemDefaultDevice();
  if (default_device) {
    devices = @[default_device];
  }
#endif
  
  if (!devices || devices.count == 0) {
    std::cerr << "No Metal devices found on this system" << std::endl;
    return;
  }
  
  int device_id = 0;
  for (id<MTLDevice> device in devices) {
    auto metal_device = MetalDevice::Create(device);
    if (metal_device) {
      metal_device->SetDeviceId(device_id++);
      registry_id_map_[metal_device->GetCapabilities().registry_id] = metal_device.get();
      devices_.push_back(std::move(metal_device));
      
      std::cout << "Found Metal device " << (device_id - 1) << ": " 
                << devices_.back()->GetName() 
                << " (Feature Set: " << MetalDeviceUtils::FeatureSetToString(devices_.back()->GetCapabilities().feature_set) << ")"
                << std::endl;
    }
  }
  
#if TARGET_OS_OSX
  [devices release];
#endif
}

size_t MetalDeviceManager::GetDeviceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return devices_.size();
}

MetalDevice* MetalDeviceManager::GetDevice(int device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
    return nullptr;
  }
  return devices_[device_id].get();
}

MetalDevice* MetalDeviceManager::GetDefaultDevice() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.empty()) {
    return nullptr;
  }
  
  // Check thread affinity first
  if (thread_device_affinity_ >= 0 && thread_device_affinity_ < static_cast<int>(devices_.size())) {
    return devices_[thread_device_affinity_].get();
  }
  
  // Return the first device that is the system default
  for (const auto& device : devices_) {
    if (device->IsSystemDefault()) {
      return device.get();
    }
  }
  
  // Fallback to first device
  return devices_[0].get();
}

MetalDevice* MetalDeviceManager::GetDeviceByRegistryId(size_t registry_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = registry_id_map_.find(registry_id);
  if (it != registry_id_map_.end()) {
    return it->second;
  }
  return nullptr;
}

MetalDevice* MetalDeviceManager::SelectBestDevice(bool prefer_discrete) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.empty()) {
    return nullptr;
  }
  
  MetalDevice* best_device = nullptr;
  int best_score = -1;
  
  for (const auto& device : devices_) {
    int score = 0;
    const auto& caps = device->GetCapabilities();
    
    // Feature set score (higher is better)
    score += static_cast<int>(caps.feature_set) * 100;
    
    // Memory score
    score += static_cast<int>(caps.recommended_max_working_set_size / (1024 * 1024 * 1024));  // GB
    
    // Discrete vs integrated preference
    if (prefer_discrete && !caps.is_integrated) {
      score += 1000;
    } else if (!prefer_discrete && caps.is_integrated) {
      score += 500;
    }
    
    // Advanced features
    if (caps.supports_raytracing) score += 50;
    if (caps.supports_function_pointers) score += 30;
    if (caps.supports_dynamic_libraries) score += 20;
    
    if (score > best_score) {
      best_score = score;
      best_device = device.get();
    }
  }
  
  return best_device;
}

void MetalDeviceManager::SetThreadDeviceAffinity(int device_id) {
  if (device_id >= 0 && device_id < static_cast<int>(devices_.size())) {
    thread_device_affinity_ = device_id;
  }
}

int MetalDeviceManager::GetThreadDeviceAffinity() const {
  return thread_device_affinity_;
}

void MetalDeviceManager::ClearThreadDeviceAffinity() {
  thread_device_affinity_ = -1;
}

//
// MetalDeviceUtils Implementation
//

std::string MetalDeviceUtils::FeatureSetToString(MetalFeatureSet feature_set) {
  switch (feature_set) {
    case MetalFeatureSet::Unknown: return "Unknown";
    case MetalFeatureSet::Apple1: return "Apple1";
    case MetalFeatureSet::Apple2: return "Apple2";
    case MetalFeatureSet::Apple3: return "Apple3";
    case MetalFeatureSet::Apple4: return "Apple4";
    case MetalFeatureSet::Apple5: return "Apple5";
    case MetalFeatureSet::Apple6: return "Apple6";
    case MetalFeatureSet::Apple7: return "Apple7";
    case MetalFeatureSet::Apple8: return "Apple8";
    case MetalFeatureSet::Mac1: return "Mac1";
    case MetalFeatureSet::Mac2: return "Mac2";
    default: return "Unknown";
  }
}

std::string MetalDeviceUtils::GetMemoryTypeString() {
  return "METAL";
}

bool MetalDeviceUtils::IsMetalSupported() {
#if TARGET_OS_OSX || TARGET_OS_IOS || TARGET_OS_TV
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  bool supported = (device != nil);
  if (device) {
    [device release];
  }
  return supported;
#else
  return false;
#endif
}

std::string MetalDeviceUtils::GetMetalVersion() {
  std::stringstream ss;
  
#if defined(__MAC_14_0)
  if (@available(macOS 14.0, *)) {
    ss << "Metal 3.1";
  } else
#endif
#if defined(__MAC_13_0)
  if (@available(macOS 13.0, *)) {
    ss << "Metal 3.0";
  } else
#endif
#if defined(__MAC_12_0)
  if (@available(macOS 12.0, *)) {
    ss << "Metal 2.4";
  } else
#endif
#if defined(__MAC_11_0)
  if (@available(macOS 11.0, *)) {
    ss << "Metal 2.3";
  } else
#endif
  {
    ss << "Metal 2.0+";
  }
  
  return ss.str();
}

}}}  // namespace triton::core::metal