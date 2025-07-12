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

#include "metal_backend_utils.h"
#include "metal_memory_manager.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>

namespace triton { namespace core { namespace metal {

bool MetalBackendUtils::initialized_ = false;
MetalLogger::Level MetalLogger::log_level_ = MetalLogger::Info;

//
// MetalBackendUtils Implementation
//

std::string MetalBackendUtils::MemoryTypeString(TRITONSERVER_MemoryType type) {
  switch (type) {
    case TRITONSERVER_MEMORY_CPU:
      return "CPU";
    case TRITONSERVER_MEMORY_CPU_PINNED:
      return "CPU_PINNED";
    case TRITONSERVER_MEMORY_GPU:
      return "GPU";
    case TRITONSERVER_MEMORY_METAL:
      return "METAL";
    default:
      return "UNKNOWN";
  }
}

bool MetalBackendUtils::IsMetalMemoryType(TRITONSERVER_MemoryType type) {
  return type == TRITONSERVER_MEMORY_METAL;
}

std::string MetalBackendUtils::InstanceGroupKindString(TRITONSERVER_InstanceGroupKind kind) {
  switch (kind) {
    case TRITONSERVER_INSTANCEGROUPKIND_AUTO:
      return "AUTO";
    case TRITONSERVER_INSTANCEGROUPKIND_CPU:
      return "CPU";
    case TRITONSERVER_INSTANCEGROUPKIND_GPU:
      return "GPU";
    case TRITONSERVER_INSTANCEGROUPKIND_METAL:
      return "METAL";
    case TRITONSERVER_INSTANCEGROUPKIND_MODEL:
      return "MODEL";
    default:
      return "UNKNOWN";
  }
}

bool MetalBackendUtils::IsMetalInstanceGroupKind(TRITONSERVER_InstanceGroupKind kind) {
  return kind == TRITONSERVER_INSTANCEGROUPKIND_METAL;
}

bool MetalBackendUtils::ParseDeviceString(const std::string& device_str, int& device_id) {
  if (device_str.empty()) {
    return false;
  }
  
  // Parse "metal:N" format
  if (device_str.find("metal:") == 0) {
    try {
      device_id = std::stoi(device_str.substr(6));
      return true;
    } catch (...) {
      return false;
    }
  }
  
  // Try to parse as plain number
  try {
    device_id = std::stoi(device_str);
    return true;
  } catch (...) {
    return false;
  }
}

int MetalBackendUtils::GetDeviceIdFromInstanceGroup(int instance_group_device_id) {
  // Negative device IDs indicate CPU affinity in Triton
  // Map them to thread-local device selection
  if (instance_group_device_id < 0) {
    auto& manager = MetalDeviceManager::Instance();
    int affinity = manager.GetThreadDeviceAffinity();
    if (affinity >= 0) {
      return affinity;
    }
    // Default to device 0 if no affinity set
    return 0;
  }
  return instance_group_device_id;
}

void MetalBackendUtils::SetThreadDeviceAffinity(int instance_group_device_id) {
  if (instance_group_device_id >= 0) {
    auto& manager = MetalDeviceManager::Instance();
    manager.SetThreadDeviceAffinity(instance_group_device_id);
  }
}

int MetalBackendUtils::SelectDeviceForModel(size_t model_size_bytes, bool prefer_discrete) {
  auto& manager = MetalDeviceManager::Instance();
  
  // Find device with enough memory
  MetalDevice* best_device = nullptr;
  int best_device_id = -1;
  
  for (size_t i = 0; i < manager.GetDeviceCount(); ++i) {
    MetalDevice* device = manager.GetDevice(i);
    if (!device) continue;
    
    size_t available = device->GetAvailableMemory();
    if (available >= model_size_bytes * 1.2) {  // 20% overhead
      if (!best_device || 
          (prefer_discrete && !device->GetCapabilities().is_integrated) ||
          (!prefer_discrete && device->GetCapabilities().is_integrated)) {
        best_device = device;
        best_device_id = i;
      }
    }
  }
  
  if (best_device_id < 0 && manager.GetDeviceCount() > 0) {
    // Fallback to device with most memory
    MetalDevice* selected = manager.SelectBestDevice(prefer_discrete);
    if (selected) {
      best_device_id = selected->GetDeviceId();
    }
  }
  
  return best_device_id >= 0 ? best_device_id : 0;
}

bool MetalBackendUtils::Initialize() {
  if (initialized_) {
    return true;
  }
  
  if (!MetalDeviceUtils::IsMetalSupported()) {
    MetalLogger::Log(MetalLogger::Error, "Metal is not supported on this system");
    return false;
  }
  
  MetalLogger::Log(MetalLogger::Info, "Initializing Metal backend");
  MetalLogger::Log(MetalLogger::Info, "Metal version: " + MetalDeviceUtils::GetMetalVersion());
  
  auto& manager = MetalDeviceManager::Instance();
  manager.Initialize();
  
  size_t device_count = manager.GetDeviceCount();
  if (device_count == 0) {
    MetalLogger::Log(MetalLogger::Error, "No Metal devices found");
    return false;
  }
  
  MetalLogger::Log(MetalLogger::Info, 
      "Found " + std::to_string(device_count) + " Metal device(s)");
  
  initialized_ = true;
  return true;
}

bool MetalBackendUtils::IsAvailable() {
  return initialized_ && MetalDeviceUtils::IsMetalSupported();
}

size_t MetalBackendUtils::GetDeviceCount() {
  if (!initialized_) {
    return 0;
  }
  return MetalDeviceManager::Instance().GetDeviceCount();
}

std::string MetalBackendUtils::GetDeviceProperties(int device_id) {
  auto& manager = MetalDeviceManager::Instance();
  MetalDevice* device = manager.GetDevice(device_id);
  if (!device) {
    return "Device " + std::to_string(device_id) + " not found";
  }
  
  const auto& caps = device->GetCapabilities();
  std::stringstream ss;
  
  ss << "Device " << device_id << ": " << caps.name << "\n";
  ss << "  Feature Set: " << MetalDeviceUtils::FeatureSetToString(caps.feature_set) << "\n";
  ss << "  Registry ID: " << caps.registry_id << "\n";
  ss << "  Type: " << (caps.is_integrated ? "Integrated" : "Discrete") << "\n";
  ss << "  Low Power: " << (caps.is_low_power ? "Yes" : "No") << "\n";
  ss << "  Removable: " << (caps.is_removable ? "Yes" : "No") << "\n";
  ss << "  Max Threads per Threadgroup: " << caps.max_threads_per_threadgroup << "\n";
  ss << "  Max Buffer Length: " << (caps.max_buffer_length / (1024 * 1024)) << " MB\n";
  ss << "  Recommended Working Set: " << 
        (caps.recommended_max_working_set_size / (1024 * 1024)) << " MB\n";
  ss << "  Advanced Features:\n";
  ss << "    Ray Tracing: " << (caps.supports_raytracing ? "Yes" : "No") << "\n";
  ss << "    Function Pointers: " << (caps.supports_function_pointers ? "Yes" : "No") << "\n";
  ss << "    Dynamic Libraries: " << (caps.supports_dynamic_libraries ? "Yes" : "No") << "\n";
  
  return ss.str();
}

std::string MetalBackendUtils::GetAllDeviceProperties() {
  std::stringstream ss;
  size_t count = GetDeviceCount();
  
  ss << "Metal Devices (" << count << "):\n";
  ss << "=====================================\n";
  
  for (size_t i = 0; i < count; ++i) {
    ss << GetDeviceProperties(i);
    if (i < count - 1) {
      ss << "-------------------------------------\n";
    }
  }
  
  return ss.str();
}

bool MetalBackendUtils::IsValidDeviceId(int device_id) {
  return device_id >= 0 && 
         device_id < static_cast<int>(MetalDeviceManager::Instance().GetDeviceCount());
}

MetalBackendUtils::MemoryInfo MetalBackendUtils::GetDeviceMemoryInfo(int device_id) {
  MemoryInfo info = {0, 0, 0};
  
  auto& manager = MetalDeviceManager::Instance();
  MetalDevice* device = manager.GetDevice(device_id);
  if (!device) {
    return info;
  }
  
  info.total_bytes = device->GetTotalMemory();
  info.available_bytes = device->GetAvailableMemory();
  info.used_bytes = info.total_bytes - info.available_bytes;
  
  return info;
}

MetalBackendUtils::TritonMetalMemory MetalBackendUtils::GetTritonMemory(MetalBuffer* buffer) {
  TritonMetalMemory mem = {nullptr, 0, -1, false};
  
  if (buffer) {
    mem.data_ptr = buffer->GetContents();  // May be null for private buffers
    mem.byte_size = buffer->GetSize();
    mem.device_id = buffer->GetDevice()->GetDeviceId();
    mem.is_shared = buffer->IsShared();
  }
  
  return mem;
}

//
// MetalInstanceGroupConfig Implementation
//

MetalInstanceGroupConfig MetalInstanceGroupConfig::Parse(const std::string& config_str) {
  MetalInstanceGroupConfig config = {0, 1, "default"};
  
  // Simple format: "device_id:count:profile"
  std::stringstream ss(config_str);
  std::string token;
  int index = 0;
  
  while (std::getline(ss, token, ':')) {
    switch (index) {
      case 0:
        try {
          config.device_id = std::stoi(token);
        } catch (...) {
          config.device_id = 0;
        }
        break;
      case 1:
        try {
          config.count = std::stoull(token);
        } catch (...) {
          config.count = 1;
        }
        break;
      case 2:
        config.profile = token;
        break;
    }
    index++;
  }
  
  return config;
}

std::string MetalInstanceGroupConfig::ToString() const {
  return std::to_string(device_id) + ":" + 
         std::to_string(count) + ":" + 
         profile;
}

//
// Error handling
//

std::string MetalErrorString(MetalErrorCode code) {
  switch (code) {
    case MetalErrorCode::Success:
      return "Success";
    case MetalErrorCode::DeviceNotFound:
      return "Metal device not found";
    case MetalErrorCode::OutOfMemory:
      return "Out of Metal memory";
    case MetalErrorCode::InvalidArgument:
      return "Invalid argument";
    case MetalErrorCode::NotSupported:
      return "Operation not supported";
    case MetalErrorCode::InternalError:
      return "Internal Metal error";
    default:
      return "Unknown Metal error";
  }
}

//
// MetalLogger Implementation
//

void MetalLogger::Log(Level level, const std::string& message) {
  if (level < log_level_) {
    return;
  }
  
  // Get current time
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  
  // Format timestamp
  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
  
  // Level string
  std::string level_str;
  switch (level) {
    case Verbose: level_str = "VERBOSE"; break;
    case Info: level_str = "INFO"; break;
    case Warning: level_str = "WARNING"; break;
    case Error: level_str = "ERROR"; break;
  }
  
  // Output
  std::ostream& out = (level >= Warning) ? std::cerr : std::cout;
  out << "[" << ss.str() << "] [METAL " << level_str << "] " << message << std::endl;
}

void MetalLogger::SetLogLevel(Level level) {
  log_level_ = level;
}

}}}  // namespace triton::core::metal