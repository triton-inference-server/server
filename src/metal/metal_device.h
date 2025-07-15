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

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#else
// Forward declarations for non-Objective-C++ files
typedef struct objc_object* id;
#endif
#endif

namespace triton { namespace core { namespace metal {

// Forward declarations
class MetalCommandQueue;
class MetalMemoryManager;

// Enum for Metal feature sets
enum class MetalFeatureSet {
  Unknown = 0,
  Apple1 = 1,
  Apple2 = 2,
  Apple3 = 3,
  Apple4 = 4,
  Apple5 = 5,
  Apple6 = 6,
  Apple7 = 7,
  Apple8 = 8,
  Mac1 = 100,
  Mac2 = 101
};

// Metal device capabilities
struct MetalDeviceCapabilities {
  std::string name;
  MetalFeatureSet feature_set;
  size_t max_threads_per_threadgroup;
  size_t max_buffer_length;
  size_t max_texture_dimension_1d;
  size_t max_texture_dimension_2d;
  size_t max_texture_dimension_3d;
  bool supports_raytracing;
  bool supports_function_pointers;
  bool supports_dynamic_libraries;
  bool is_low_power;
  bool is_removable;
  bool is_integrated;
  size_t recommended_max_working_set_size;
  size_t registry_id;
};

// MetalDevice class encapsulates a Metal device
class MetalDevice {
 public:
  // Factory method to create a MetalDevice
  static std::unique_ptr<MetalDevice> Create(id device);
  
  // Get the device ID (index in the device list)
  int GetDeviceId() const { return device_id_; }
  
  // Set the device ID
  void SetDeviceId(int id) { device_id_ = id; }
  
  // Get device capabilities
  const MetalDeviceCapabilities& GetCapabilities() const { return capabilities_; }
  
  // Get the underlying Metal device object
  id GetDevice() const { return device_; }
  
  // Create a command queue for this device
  std::unique_ptr<MetalCommandQueue> CreateCommandQueue();
  
  // Check if device supports a specific feature set
  bool SupportsFeatureSet(MetalFeatureSet feature_set) const;
  
  // Get available memory (in bytes)
  size_t GetAvailableMemory() const;
  
  // Get total memory (in bytes)
  size_t GetTotalMemory() const;
  
  // Get device name
  const std::string& GetName() const { return capabilities_.name; }
  
  // Check if this is the default system device
  bool IsSystemDefault() const;

 private:
  MetalDevice(id device);
  ~MetalDevice();
  
  // Initialize device capabilities
  void InitializeCapabilities();
  
  id device_;  // MTLDevice instance
  int device_id_;
  MetalDeviceCapabilities capabilities_;
  mutable std::mutex mutex_;
};

// MetalCommandQueue encapsulates a Metal command queue
class MetalCommandQueue {
 public:
  static std::unique_ptr<MetalCommandQueue> Create(id device);
  
  // Get the underlying Metal command queue
  id GetQueue() const { return queue_; }
  
  // Create a command buffer
  id CreateCommandBuffer();
  
  // Wait for all commands to complete
  void WaitUntilCompleted();

 private:
  MetalCommandQueue(id queue);
  ~MetalCommandQueue();
  
  id queue_;  // MTLCommandQueue instance
};

// MetalDeviceManager manages all Metal devices in the system
class MetalDeviceManager {
 public:
  // Get the singleton instance
  static MetalDeviceManager& Instance();
  
  // Initialize the device manager
  void Initialize();
  
  // Get the number of available devices
  size_t GetDeviceCount() const;
  
  // Get a device by index
  MetalDevice* GetDevice(int device_id) const;
  
  // Get the default device
  MetalDevice* GetDefaultDevice() const;
  
  // Get device by registry ID
  MetalDevice* GetDeviceByRegistryId(size_t registry_id) const;
  
  // Select the best device based on criteria
  MetalDevice* SelectBestDevice(bool prefer_discrete = true) const;
  
  // Get all devices
  const std::vector<std::unique_ptr<MetalDevice>>& GetDevices() const {
    return devices_;
  }
  
  // Set device affinity for current thread
  void SetThreadDeviceAffinity(int device_id);
  
  // Get device affinity for current thread
  int GetThreadDeviceAffinity() const;
  
  // Clear device affinity for current thread
  void ClearThreadDeviceAffinity();

 private:
  MetalDeviceManager() = default;
  ~MetalDeviceManager() = default;
  
  // Enumerate all available Metal devices
  void EnumerateDevices();
  
  std::vector<std::unique_ptr<MetalDevice>> devices_;
  std::unordered_map<size_t, MetalDevice*> registry_id_map_;
  mutable std::mutex mutex_;
  
  // Thread-local device affinity
  thread_local static int thread_device_affinity_;
  
  // Prevent copying
  MetalDeviceManager(const MetalDeviceManager&) = delete;
  MetalDeviceManager& operator=(const MetalDeviceManager&) = delete;
};

// Helper functions for Metal device management
namespace MetalDeviceUtils {
  // Convert feature set to string
  std::string FeatureSetToString(MetalFeatureSet feature_set);
  
  // Get the memory type string for logging
  std::string GetMemoryTypeString();
  
  // Check if Metal is supported on this system
  bool IsMetalSupported();
  
  // Get Metal runtime version
  std::string GetMetalVersion();
}

}}}  // namespace triton::core::metal