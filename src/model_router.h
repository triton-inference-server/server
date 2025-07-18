// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "triton/core/tritonserver.h"

namespace triton { namespace core {

// Forward declarations
class ModelRoutingPolicy;
class BackendCapabilities;
class ModelProfile;
class RoutingMetrics;

//
// Routing decision structure
//
struct RoutingDecision {
  enum class BackendType {
    CPU,
    GPU,
    NEURAL_ENGINE,
    METAL_MPS,
    MIXED  // For ensemble or partitioned execution
  };

  BackendType primary_backend;
  std::vector<BackendType> secondary_backends;  // For fallback or partitioning
  
  // Routing rationale
  std::string reason;
  float confidence_score;  // 0.0 to 1.0
  
  // Performance expectations
  float expected_latency_ms;
  float expected_throughput;
  float expected_memory_usage_mb;
  float expected_power_usage_w;
  
  // Load balancing info
  int instance_id;  // Which instance of the backend to use
  float load_factor;  // Current load on selected backend
};

//
// Model characteristics used for routing decisions
//
struct ModelCharacteristics {
  // Model properties
  std::string model_name;
  std::string model_version;
  std::string framework;  // TensorFlow, PyTorch, ONNX, etc.
  
  // Model complexity metrics
  uint64_t parameter_count;
  uint64_t flop_count;
  uint64_t memory_footprint_bytes;
  
  // Model structure
  bool has_dynamic_shapes;
  bool has_control_flow;
  bool has_custom_ops;
  bool is_quantized;
  int quantization_bits;  // 8, 16, etc.
  
  // Input/output characteristics
  std::vector<std::pair<std::string, std::vector<int64_t>>> input_shapes;
  std::vector<std::pair<std::string, TRITONSERVER_DataType>> input_types;
  std::vector<std::pair<std::string, std::vector<int64_t>>> output_shapes;
  
  // Optimization hints
  bool supports_batching;
  int32_t optimal_batch_size;
  bool supports_gpu_acceleration;
  bool supports_neural_engine;
  bool supports_int8_acceleration;
};

//
// Runtime context for routing decisions
//
struct RoutingContext {
  // Request properties
  int32_t batch_size;
  std::unordered_map<std::string, std::vector<int64_t>> actual_input_shapes;
  
  // Performance requirements
  float max_latency_ms;  // SLA requirement
  float min_throughput;
  
  // Resource constraints
  float max_memory_mb;
  float max_power_w;
  
  // Optimization preference
  enum class OptimizationGoal {
    MINIMIZE_LATENCY,
    MAXIMIZE_THROUGHPUT,
    MINIMIZE_POWER,
    BALANCED
  };
  OptimizationGoal optimization_goal = OptimizationGoal::BALANCED;
  
  // A/B testing
  std::string experiment_id;
  float routing_override_percentage;  // For gradual rollout
};

//
// Backend capabilities description
//
class BackendCapabilities {
 public:
  BackendCapabilities(const std::string& backend_name);
  
  // Hardware capabilities
  bool SupportsGPU() const { return supports_gpu_; }
  bool SupportsNeuralEngine() const { return supports_neural_engine_; }
  bool SupportsInt8() const { return supports_int8_; }
  bool SupportsFP16() const { return supports_fp16_; }
  
  // Performance characteristics
  float GetPeakThroughputGFLOPS() const { return peak_throughput_gflops_; }
  float GetMemoryBandwidthGBps() const { return memory_bandwidth_gbps_; }
  float GetPowerEfficiencyGFLOPSPerWatt() const { return power_efficiency_; }
  
  // Current resource usage
  float GetCurrentUtilization() const { return current_utilization_.load(); }
  float GetAvailableMemoryMB() const { return available_memory_mb_.load(); }
  
  void UpdateUtilization(float utilization);
  void UpdateAvailableMemory(float memory_mb);
  
 private:
  std::string backend_name_;
  
  // Static capabilities
  bool supports_gpu_;
  bool supports_neural_engine_;
  bool supports_int8_;
  bool supports_fp16_;
  
  float peak_throughput_gflops_;
  float memory_bandwidth_gbps_;
  float power_efficiency_;
  
  // Dynamic state
  std::atomic<float> current_utilization_{0.0f};
  std::atomic<float> available_memory_mb_{0.0f};
};

//
// Model performance profile
//
class ModelProfile {
 public:
  ModelProfile(const std::string& model_name);
  
  // Record performance observation
  void RecordExecution(
      RoutingDecision::BackendType backend,
      int32_t batch_size,
      float latency_ms,
      float memory_usage_mb,
      float power_usage_w);
  
  // Get performance predictions
  float PredictLatency(
      RoutingDecision::BackendType backend,
      int32_t batch_size) const;
  
  float PredictMemoryUsage(
      RoutingDecision::BackendType backend,
      int32_t batch_size) const;
  
  float PredictPowerUsage(
      RoutingDecision::BackendType backend,
      int32_t batch_size) const;
  
  // Get confidence in predictions
  float GetPredictionConfidence(
      RoutingDecision::BackendType backend) const;
  
 private:
  struct PerformanceRecord {
    int32_t batch_size;
    float latency_ms;
    float memory_usage_mb;
    float power_usage_w;
    std::chrono::steady_clock::time_point timestamp;
  };
  
  std::string model_name_;
  mutable std::mutex mutex_;
  
  // Performance history per backend
  std::unordered_map<RoutingDecision::BackendType,
                     std::vector<PerformanceRecord>> performance_history_;
  
  // Simple linear regression for prediction
  struct RegressionModel {
    float slope;
    float intercept;
    float r_squared;  // Confidence metric
  };
  
  mutable std::unordered_map<RoutingDecision::BackendType,
                             RegressionModel> latency_models_;
  mutable std::unordered_map<RoutingDecision::BackendType,
                             RegressionModel> memory_models_;
  mutable std::unordered_map<RoutingDecision::BackendType,
                             RegressionModel> power_models_;
  
  void UpdateRegressionModels() const;
};

//
// Routing policy interface
//
class ModelRoutingPolicy {
 public:
  virtual ~ModelRoutingPolicy() = default;
  
  // Make routing decision
  virtual RoutingDecision Route(
      const ModelCharacteristics& model,
      const RoutingContext& context,
      const std::vector<std::shared_ptr<BackendCapabilities>>& backends,
      const std::shared_ptr<ModelProfile>& profile) = 0;
  
  // Policy name for logging
  virtual std::string GetName() const = 0;
};

//
// Built-in routing policies
//

// Latency-optimized routing
class LatencyOptimizedPolicy : public ModelRoutingPolicy {
 public:
  RoutingDecision Route(
      const ModelCharacteristics& model,
      const RoutingContext& context,
      const std::vector<std::shared_ptr<BackendCapabilities>>& backends,
      const std::shared_ptr<ModelProfile>& profile) override;
  
  std::string GetName() const override { return "LatencyOptimized"; }
};

// Throughput-optimized routing
class ThroughputOptimizedPolicy : public ModelRoutingPolicy {
 public:
  RoutingDecision Route(
      const ModelCharacteristics& model,
      const RoutingContext& context,
      const std::vector<std::shared_ptr<BackendCapabilities>>& backends,
      const std::shared_ptr<ModelProfile>& profile) override;
  
  std::string GetName() const override { return "ThroughputOptimized"; }
};

// Power-efficient routing
class PowerEfficientPolicy : public ModelRoutingPolicy {
 public:
  RoutingDecision Route(
      const ModelCharacteristics& model,
      const RoutingContext& context,
      const std::vector<std::shared_ptr<BackendCapabilities>>& backends,
      const std::shared_ptr<ModelProfile>& profile) override;
  
  std::string GetName() const override { return "PowerEfficient"; }
};

// Adaptive routing based on model characteristics
class AdaptiveRoutingPolicy : public ModelRoutingPolicy {
 public:
  RoutingDecision Route(
      const ModelCharacteristics& model,
      const RoutingContext& context,
      const std::vector<std::shared_ptr<BackendCapabilities>>& backends,
      const std::shared_ptr<ModelProfile>& profile) override;
  
  std::string GetName() const override { return "Adaptive"; }
  
 private:
  // Heuristics for backend selection
  float ScoreBackendForModel(
      const ModelCharacteristics& model,
      const RoutingContext& context,
      RoutingDecision::BackendType backend,
      const std::shared_ptr<BackendCapabilities>& capabilities,
      const std::shared_ptr<ModelProfile>& profile);
};

//
// Main routing engine
//
class ModelRouter {
 public:
  static ModelRouter& GetInstance();
  
  // Initialize router with available backends
  TRITONSERVER_Error* Initialize(
      const std::vector<std::string>& backend_names);
  
  // Register a model for routing
  TRITONSERVER_Error* RegisterModel(
      const std::string& model_name,
      const ModelCharacteristics& characteristics);
  
  // Make routing decision
  RoutingDecision RouteRequest(
      const std::string& model_name,
      const RoutingContext& context);
  
  // Update performance profile after execution
  void RecordExecution(
      const std::string& model_name,
      const RoutingDecision& decision,
      float actual_latency_ms,
      float actual_memory_mb,
      float actual_power_w);
  
  // Policy management
  void SetRoutingPolicy(std::shared_ptr<ModelRoutingPolicy> policy);
  void SetRoutingPolicyForModel(
      const std::string& model_name,
      std::shared_ptr<ModelRoutingPolicy> policy);
  
  // A/B testing support
  void EnableABTesting(
      const std::string& experiment_id,
      std::shared_ptr<ModelRoutingPolicy> control_policy,
      std::shared_ptr<ModelRoutingPolicy> treatment_policy,
      float treatment_percentage);
  
  // Configuration
  struct Config {
    // Profiling settings
    bool enable_profiling = true;
    int profile_warmup_iterations = 10;
    int profile_history_size = 100;
    
    // Load balancing
    bool enable_load_balancing = true;
    float max_backend_utilization = 0.8f;
    
    // Fallback behavior
    bool enable_fallback = true;
    float fallback_latency_threshold_ms = 100.0f;
    
    // A/B testing
    bool enable_ab_testing = false;
    
    // Monitoring
    bool enable_metrics = true;
    std::chrono::seconds metrics_interval{60};
  };
  
  void SetConfig(const Config& config) { config_ = config; }
  const Config& GetConfig() const { return config_; }
  
  // Metrics and monitoring
  std::shared_ptr<RoutingMetrics> GetMetrics() const { return metrics_; }
  
 private:
  ModelRouter() = default;
  ~ModelRouter() = default;
  ModelRouter(const ModelRouter&) = delete;
  ModelRouter& operator=(const ModelRouter&) = delete;
  
  // Configuration
  Config config_;
  
  // Registered models and their characteristics
  std::mutex models_mutex_;
  std::unordered_map<std::string, ModelCharacteristics> model_characteristics_;
  std::unordered_map<std::string, std::shared_ptr<ModelProfile>> model_profiles_;
  
  // Available backends
  std::mutex backends_mutex_;
  std::unordered_map<RoutingDecision::BackendType,
                     std::shared_ptr<BackendCapabilities>> backend_capabilities_;
  
  // Routing policies
  std::mutex policy_mutex_;
  std::shared_ptr<ModelRoutingPolicy> default_policy_;
  std::unordered_map<std::string, std::shared_ptr<ModelRoutingPolicy>> model_policies_;
  
  // A/B testing state
  struct ABTestState {
    std::string experiment_id;
    std::shared_ptr<ModelRoutingPolicy> control_policy;
    std::shared_ptr<ModelRoutingPolicy> treatment_policy;
    float treatment_percentage;
    std::atomic<uint64_t> control_count{0};
    std::atomic<uint64_t> treatment_count{0};
  };
  std::shared_ptr<ABTestState> ab_test_state_;
  
  // Metrics collection
  std::shared_ptr<RoutingMetrics> metrics_;
  
  // Helper methods
  std::shared_ptr<BackendCapabilities> GetBackendCapabilities(
      RoutingDecision::BackendType type);
  
  bool ShouldUseTreatmentPolicy(const std::string& model_name);
  
  void CollectMetrics(
      const std::string& model_name,
      const RoutingDecision& decision,
      float latency_ms);
};

//
// Routing metrics for monitoring
//
class RoutingMetrics {
 public:
  RoutingMetrics();
  
  // Record routing decision
  void RecordRoutingDecision(
      const std::string& model_name,
      const RoutingDecision& decision,
      const std::string& policy_name);
  
  // Record execution results
  void RecordExecution(
      const std::string& model_name,
      RoutingDecision::BackendType backend,
      float latency_ms,
      bool success);
  
  // Get metrics
  struct ModelMetrics {
    uint64_t total_requests;
    std::unordered_map<RoutingDecision::BackendType, uint64_t> backend_counts;
    std::unordered_map<RoutingDecision::BackendType, float> avg_latency_ms;
    std::unordered_map<RoutingDecision::BackendType, float> p99_latency_ms;
    std::unordered_map<RoutingDecision::BackendType, float> success_rate;
  };
  
  ModelMetrics GetModelMetrics(const std::string& model_name) const;
  std::vector<std::string> GetModelNames() const;
  
  // Export metrics in Prometheus format
  std::string ExportPrometheusMetrics() const;
  
 private:
  mutable std::mutex mutex_;
  
  struct ExecutionRecord {
    RoutingDecision::BackendType backend;
    float latency_ms;
    bool success;
    std::chrono::steady_clock::time_point timestamp;
  };
  
  // Per-model metrics
  std::unordered_map<std::string, std::vector<ExecutionRecord>> execution_history_;
  std::unordered_map<std::string, std::unordered_map<RoutingDecision::BackendType, uint64_t>> routing_counts_;
  std::unordered_map<std::string, std::unordered_map<std::string, uint64_t>> policy_counts_;
};

}}  // namespace triton::core