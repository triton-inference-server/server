// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef TRITON_ENABLE_METAL

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "metal_unified_memory.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace metal {

// Configuration for unified memory in backends
struct UnifiedBackendConfig {
  // Enable unified memory optimizations
  bool enable_unified_memory = true;
  
  // Pattern hints for different tensor types
  std::unordered_map<std::string, triton::core::UnifiedMemoryPattern> tensor_patterns = {
    {"input", triton::core::UnifiedMemoryPattern::CPU_DOMINANT},
    {"output", triton::core::UnifiedMemoryPattern::GPU_DOMINANT},
    {"weights", triton::core::UnifiedMemoryPattern::GPU_DOMINANT},
    {"intermediate", triton::core::UnifiedMemoryPattern::BALANCED}
  };
  
  // Zero-copy threshold (tensors larger than this use zero-copy)
  size_t zero_copy_threshold = 1024 * 1024; // 1MB
  
  // Enable batch optimization
  bool enable_batch_optimization = true;
  
  // Enable profiling
  bool enable_profiling = false;
};

// Base class for Metal-enabled backends with unified memory support
class UnifiedMetalBackend : public BackendModel {
 public:
  UnifiedMetalBackend(TRITONBACKEND_Model* triton_model);
  virtual ~UnifiedMetalBackend();
  
  // Initialize unified memory support
  TRITONSERVER_Error* InitializeUnifiedMemory(const UnifiedBackendConfig& config);
  
  // Create optimized input tensor
  TRITONSERVER_Error* CreateOptimizedInputTensor(
      const std::string& name,
      TRITONSERVER_DataType datatype,
      const std::vector<int64_t>& shape,
      std::unique_ptr<triton::core::MetalBuffer>& buffer,
      bool zero_copy_if_possible = true);
  
  // Create optimized output tensor
  TRITONSERVER_Error* CreateOptimizedOutputTensor(
      const std::string& name,
      TRITONSERVER_DataType datatype,
      const std::vector<int64_t>& shape,
      std::unique_ptr<triton::core::MetalBuffer>& buffer);
  
  // Get zero-copy tensor from request
  TRITONSERVER_Error* GetZeroCopyInputTensor(
      TRITONBACKEND_Request* request,
      const std::string& name,
      std::unique_ptr<triton::core::ZeroCopyTensor>& tensor);
  
  // Set zero-copy output tensor
  TRITONSERVER_Error* SetZeroCopyOutputTensor(
      TRITONBACKEND_Response* response,
      const std::string& name,
      std::unique_ptr<triton::core::ZeroCopyTensor> tensor);
  
  // Batch allocation optimization
  TRITONSERVER_Error* BatchAllocateTensors(
      const std::vector<std::string>& names,
      const std::vector<TRITONSERVER_DataType>& datatypes,
      const std::vector<std::vector<int64_t>>& shapes,
      std::vector<std::unique_ptr<triton::core::MetalBuffer>>& buffers);
  
  // Get memory usage statistics
  void GetMemoryStatistics(
      size_t& total_allocated,
      size_t& unified_memory_used,
      size_t& transfers_eliminated);
  
 protected:
  // Helper to determine tensor pattern
  triton::core::UnifiedMemoryPattern GetTensorPattern(const std::string& tensor_name);
  
  // Helper to calculate tensor size
  size_t CalculateTensorSize(
      TRITONSERVER_DataType datatype,
      const std::vector<int64_t>& shape);
  
 private:
  UnifiedBackendConfig config_;
  bool unified_memory_initialized_;
};

// Helper class for automatic memory access tracking in backends
class BackendMemoryAccessTracker {
 public:
  BackendMemoryAccessTracker(
      void* data,
      size_t size,
      bool is_input,
      const std::string& tensor_name)
      : access_(data, size, is_input, true) {
    // Could log tensor-specific access patterns
  }
  
 private:
  triton::core::ScopedMemoryAccess access_;
};

// Example implementation for a specific backend
class UnifiedTensorFlowBackend : public UnifiedMetalBackend {
 public:
  UnifiedTensorFlowBackend(TRITONBACKEND_Model* triton_model)
      : UnifiedMetalBackend(triton_model) {}
  
  TRITONSERVER_Error* Execute(
      TRITONBACKEND_Request** requests,
      const uint32_t request_count);
  
 private:
  TRITONSERVER_Error* ProcessBatch(
      const std::vector<TRITONBACKEND_Request*>& requests,
      std::vector<TRITONBACKEND_Response*>& responses);
};

// Example implementation for PyTorch backend
class UnifiedPyTorchBackend : public UnifiedMetalBackend {
 public:
  UnifiedPyTorchBackend(TRITONBACKEND_Model* triton_model)
      : UnifiedMetalBackend(triton_model) {}
  
  TRITONSERVER_Error* Execute(
      TRITONBACKEND_Request** requests,
      const uint32_t request_count);
  
 private:
  TRITONSERVER_Error* ProcessBatchOptimized(
      const std::vector<TRITONBACKEND_Request*>& requests,
      std::vector<TRITONBACKEND_Response*>& responses);
};

// Utility functions for backend integration
namespace utils {

// Convert Triton memory type to unified memory pattern
inline triton::core::UnifiedMemoryPattern
GetPatternFromMemoryType(TRITONSERVER_MemoryType memory_type) {
  switch (memory_type) {
    case TRITONSERVER_MEMORY_CPU:
    case TRITONSERVER_MEMORY_CPU_PINNED:
      return triton::core::UnifiedMemoryPattern::CPU_DOMINANT;
    case TRITONSERVER_MEMORY_GPU:
      return triton::core::UnifiedMemoryPattern::GPU_DOMINANT;
    default:
      return triton::core::UnifiedMemoryPattern::UNKNOWN;
  }
}

// Check if zero-copy is possible for given tensor
inline bool CanUseZeroCopy(
    size_t tensor_size,
    TRITONSERVER_MemoryType memory_type,
    const UnifiedBackendConfig& config) {
  return config.enable_unified_memory &&
         tensor_size >= config.zero_copy_threshold &&
         (memory_type == TRITONSERVER_MEMORY_CPU ||
          memory_type == TRITONSERVER_MEMORY_CPU_PINNED);
}

// Create optimized memory configuration from model config
inline UnifiedBackendConfig
CreateUnifiedConfigFromModelConfig(
    common::TritonJson::Value& model_config) {
  UnifiedBackendConfig config;
  
  // Parse optimization parameters from model config
  common::TritonJson::Value optimization;
  if (model_config.Find("optimization", &optimization)) {
    common::TritonJson::Value unified_memory;
    if (optimization.Find("unified_memory", &unified_memory)) {
      unified_memory.MemberAsBool("enable", &config.enable_unified_memory);
      unified_memory.MemberAsBool("batch_optimization", &config.enable_batch_optimization);
      unified_memory.MemberAsBool("profiling", &config.enable_profiling);
      
      int64_t threshold;
      if (unified_memory.MemberAsInt("zero_copy_threshold", &threshold)) {
        config.zero_copy_threshold = static_cast<size_t>(threshold);
      }
      
      // Parse tensor patterns
      common::TritonJson::Value patterns;
      if (unified_memory.Find("tensor_patterns", &patterns)) {
        for (size_t i = 0; i < patterns.ArraySize(); ++i) {
          common::TritonJson::Value pattern;
          patterns.IndexAsObject(i, &pattern);
          
          std::string name, type;
          pattern.MemberAsString("name", &name);
          pattern.MemberAsString("pattern", &type);
          
          if (type == "cpu_dominant") {
            config.tensor_patterns[name] = triton::core::UnifiedMemoryPattern::CPU_DOMINANT;
          } else if (type == "gpu_dominant") {
            config.tensor_patterns[name] = triton::core::UnifiedMemoryPattern::GPU_DOMINANT;
          } else if (type == "balanced") {
            config.tensor_patterns[name] = triton::core::UnifiedMemoryPattern::BALANCED;
          } else if (type == "streaming") {
            config.tensor_patterns[name] = triton::core::UnifiedMemoryPattern::STREAMING;
          }
        }
      }
    }
  }
  
  return config;
}

} // namespace utils

}}}  // namespace triton::backend::metal

#endif  // TRITON_ENABLE_METAL