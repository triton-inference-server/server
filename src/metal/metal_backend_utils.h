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

#include <string>
#include <vector>
#include "metal_device.h"

// Include the actual Triton headers for proper type definitions
#include "triton/core/tritonserver.h"

namespace triton { namespace core { namespace metal {

// Forward declarations
class MetalBuffer;

// Metal-specific memory type extension
// Using high values to avoid conflicts with future Triton enums
const TRITONSERVER_MemoryType TRITONSERVER_MEMORY_METAL = 
    static_cast<TRITONSERVER_MemoryType>(100);

// Metal-specific instance group kind
const TRITONSERVER_InstanceGroupKind TRITONSERVER_INSTANCEGROUPKIND_METAL = 
    static_cast<TRITONSERVER_InstanceGroupKind>(100);

// Backend utility functions for Metal integration
class MetalBackendUtils {
 public:
  // Convert Triton memory type to string
  static std::string MemoryTypeString(TRITONSERVER_MemoryType type);
  
  // Check if memory type is Metal
  static bool IsMetalMemoryType(TRITONSERVER_MemoryType type);
  
  // Convert instance group kind to string
  static std::string InstanceGroupKindString(TRITONSERVER_InstanceGroupKind kind);
  
  // Check if instance group kind is Metal
  static bool IsMetalInstanceGroupKind(TRITONSERVER_InstanceGroupKind kind);
  
  // Parse device string (e.g., "metal:0", "metal:1")
  static bool ParseDeviceString(const std::string& device_str, int& device_id);
  
  // Get device ID from instance group device ID
  // Maps negative IDs to thread affinity, positive to actual device
  static int GetDeviceIdFromInstanceGroup(int instance_group_device_id);
  
  // Set thread device affinity based on instance group
  static void SetThreadDeviceAffinity(int instance_group_device_id);
  
  // Get recommended device for model based on memory requirements
  static int SelectDeviceForModel(size_t model_size_bytes, bool prefer_discrete = true);
  
  // Initialize Metal backend (call once at startup)
  static bool Initialize();
  
  // Check if Metal is available and initialized
  static bool IsAvailable();
  
  // Get device count
  static size_t GetDeviceCount();
  
  // Get device properties as string (for logging)
  static std::string GetDeviceProperties(int device_id);
  
  // Get all device properties as string (for server info)
  static std::string GetAllDeviceProperties();
  
  // Validate device ID
  static bool IsValidDeviceId(int device_id);
  
  // Get memory info for device
  struct MemoryInfo {
    size_t total_bytes;
    size_t available_bytes;
    size_t used_bytes;
  };
  static MemoryInfo GetDeviceMemoryInfo(int device_id);
  
  // Convert between Metal buffer and Triton memory representation
  struct TritonMetalMemory {
    void* data_ptr;
    size_t byte_size;
    int device_id;
    bool is_shared;
  };
  
  // Create Triton-compatible memory info from Metal buffer
  static TritonMetalMemory GetTritonMemory(MetalBuffer* buffer);

 private:
  static bool initialized_;
};

// Instance group configuration for Metal
struct MetalInstanceGroupConfig {
  int device_id;
  size_t count;
  std::string profile;  // "default", "low_latency", "high_throughput"
  
  // Parse from config string
  static MetalInstanceGroupConfig Parse(const std::string& config_str);
  
  // Convert to string
  std::string ToString() const;
};

// Metal-specific error codes
enum class MetalErrorCode {
  Success = 0,
  DeviceNotFound = 1,
  OutOfMemory = 2,
  InvalidArgument = 3,
  NotSupported = 4,
  InternalError = 5
};

// Convert error code to string
std::string MetalErrorString(MetalErrorCode code);

// Logging utilities
class MetalLogger {
 public:
  enum Level {
    Verbose = 0,
    Info = 1,
    Warning = 2,
    Error = 3
  };
  
  static void Log(Level level, const std::string& message);
  static void SetLogLevel(Level level);
  
 private:
  static Level log_level_;
};

}}}  // namespace triton::core::metal