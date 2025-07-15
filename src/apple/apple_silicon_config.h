// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Apple Silicon Configuration
//
// This file defines configuration structures for Apple Silicon optimizations
// following NVIDIA Triton's configuration patterns.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include "triton/core/tritonserver.h"

namespace triton {
namespace apple {

// Configuration for AMX (Apple Matrix coprocessor)
struct AMXConfig {
  bool enabled = true;
  size_t tile_m = 32;  // M dimension tile size
  size_t tile_n = 32;  // N dimension tile size
  size_t tile_k = 32;  // K dimension tile size
  size_t alignment = 64;  // Memory alignment in bytes
  
  // Thresholds for operation routing
  size_t min_size_threshold = 1024;  // Minimum size for AMX activation
  size_t direct_compute_threshold = 1000;  // Below this, use direct computation
  
  // Performance tuning
  bool auto_tune = false;
  int num_threads = 0;  // 0 = auto-detect
};

// Configuration for ANE (Apple Neural Engine)
struct ANEConfig {
  bool enabled = true;
  std::string precision = "fp16";  // fp16, fp32, int8
  std::string power_mode = "high_performance";  // high_performance, balanced, low_power
  
  // Memory management
  size_t cache_size_mb = 256;
  bool enable_model_partitioning = true;
  
  // Performance tuning
  bool enable_graph_optimization = true;
  bool enable_quantization = false;
  int optimization_level = 2;  // 0-3
};

// Configuration for Metal GPU backend
struct MetalConfig {
  bool enabled = true;
  int64_t device_id = 0;
  
  // Memory pool configuration
  struct PoolConfig {
    size_t initial_size = 256 * 1024 * 1024;  // 256MB
    size_t max_size = 2ULL * 1024 * 1024 * 1024;  // 2GB
    size_t chunk_size = 64 * 1024 * 1024;  // 64MB
    bool enable_gc = true;
    double gc_threshold = 0.8;  // Trigger GC at 80% usage
    int64_t gc_delay_ms = 1000;
  } pool;
  
  // Command buffer configuration
  size_t command_buffer_count = 16;
  size_t max_command_buffer_size = 16 * 1024 * 1024;  // 16MB
  
  // Performance settings
  bool prefer_unified_memory = true;
  bool enable_profiling = false;
};

// Configuration for Winograd convolution
struct WinogradConfig {
  bool enabled = true;
  size_t max_channels = 256;  // Maximum channels for stack allocation
  bool auto_select = true;  // Auto-select between Winograd and direct
  
  // Thresholds for auto-selection
  size_t min_batch_size = 1;
  size_t min_spatial_size = 8;
  float memory_threshold_mb = 100.0;
};

// Model-specific optimization configuration
struct ModelOptimizationConfig {
  std::string backend = "auto";  // auto, amx, ane, metal, cpu
  std::string precision = "auto";  // auto, fp32, fp16, int8
  int optimization_level = -1;  // -1 = use global setting
  
  // Backend-specific overrides
  std::map<std::string, std::string> backend_params;
};

// Main Apple Silicon configuration
struct AppleSiliconConfig {
  // Hardware components
  AMXConfig amx;
  ANEConfig ane;
  MetalConfig metal;
  WinogradConfig winograd;
  
  // Global settings
  int optimization_level = 2;  // 0=none, 1=basic, 2=standard, 3=aggressive
  bool verbose_logging = false;
  bool enable_profiling = false;
  
  // Model-specific configurations
  std::map<std::string, ModelOptimizationConfig> model_configs;
  
  // Parse from backend config string
  TRITONSERVER_Error* ParseFromBackendConfig(const std::string& config_json);
  
  // Parse from command line format: key=value,key2=value2
  TRITONSERVER_Error* ParseFromCommandLine(const std::string& config_str);
  
  // Convert to JSON for serialization
  std::string ToJSON() const;
  
  // Apply environment variable overrides
  void ApplyEnvironmentOverrides();
  
  // Validate configuration
  TRITONSERVER_Error* Validate() const;
  
  // Get model-specific config (with fallback to defaults)
  ModelOptimizationConfig GetModelConfig(const std::string& model_name) const;
};

// Global configuration instance (following Triton patterns)
AppleSiliconConfig& GetAppleSiliconConfig();

// Parse backend config from Triton server
TRITONSERVER_Error* ParseAppleSiliconBackendConfig(
    TRITONBACKEND_Backend* backend,
    AppleSiliconConfig& config);

}  // namespace apple
}  // namespace triton