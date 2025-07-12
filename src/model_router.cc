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

#include "model_router.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>

namespace triton { namespace core {

//
// BackendCapabilities Implementation
//
BackendCapabilities::BackendCapabilities(const std::string& backend_name)
    : backend_name_(backend_name)
{
  // Initialize capabilities based on backend type
  // These are example values - should be populated from actual hardware queries
  
  if (backend_name == "cpu") {
    supports_gpu_ = false;
    supports_neural_engine_ = false;
    supports_int8_ = true;
    supports_fp16_ = false;
    peak_throughput_gflops_ = 100.0f;  // Example: 100 GFLOPS
    memory_bandwidth_gbps_ = 50.0f;
    power_efficiency_ = 10.0f;  // GFLOPS/Watt
  } else if (backend_name == "gpu" || backend_name == "cuda") {
    supports_gpu_ = true;
    supports_neural_engine_ = false;
    supports_int8_ = true;
    supports_fp16_ = true;
    peak_throughput_gflops_ = 10000.0f;  // Example: 10 TFLOPS
    memory_bandwidth_gbps_ = 500.0f;
    power_efficiency_ = 50.0f;
  } else if (backend_name == "neural_engine" || backend_name == "ane") {
    supports_gpu_ = false;
    supports_neural_engine_ = true;
    supports_int8_ = true;
    supports_fp16_ = true;
    peak_throughput_gflops_ = 5000.0f;  // Example: 5 TFLOPS
    memory_bandwidth_gbps_ = 200.0f;
    power_efficiency_ = 100.0f;  // Very power efficient
  } else if (backend_name == "metal_mps") {
    supports_gpu_ = true;
    supports_neural_engine_ = false;
    supports_int8_ = false;
    supports_fp16_ = true;
    peak_throughput_gflops_ = 8000.0f;  // Example: 8 TFLOPS
    memory_bandwidth_gbps_ = 400.0f;
    power_efficiency_ = 80.0f;
  }
  
  // Initialize dynamic state
  current_utilization_ = 0.0f;
  available_memory_mb_ = 8192.0f;  // Example: 8GB available
}

void
BackendCapabilities::UpdateUtilization(float utilization)
{
  current_utilization_ = std::max(0.0f, std::min(1.0f, utilization));
}

void
BackendCapabilities::UpdateAvailableMemory(float memory_mb)
{
  available_memory_mb_ = std::max(0.0f, memory_mb);
}

//
// ModelProfile Implementation
//
ModelProfile::ModelProfile(const std::string& model_name)
    : model_name_(model_name)
{
}

void
ModelProfile::RecordExecution(
    RoutingDecision::BackendType backend,
    int32_t batch_size,
    float latency_ms,
    float memory_usage_mb,
    float power_usage_w)
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  PerformanceRecord record{
      batch_size,
      latency_ms,
      memory_usage_mb,
      power_usage_w,
      std::chrono::steady_clock::now()
  };
  
  performance_history_[backend].push_back(record);
  
  // Keep only recent history (last 1000 records per backend)
  if (performance_history_[backend].size() > 1000) {
    performance_history_[backend].erase(
        performance_history_[backend].begin(),
        performance_history_[backend].begin() + 500);
  }
  
  // Update regression models
  UpdateRegressionModels();
}

float
ModelProfile::PredictLatency(
    RoutingDecision::BackendType backend,
    int32_t batch_size) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = latency_models_.find(backend);
  if (it == latency_models_.end()) {
    // No model available, return a conservative estimate
    return 100.0f * batch_size;  // 100ms per batch item
  }
  
  // Simple linear prediction: latency = slope * batch_size + intercept
  return it->second.slope * batch_size + it->second.intercept;
}

float
ModelProfile::PredictMemoryUsage(
    RoutingDecision::BackendType backend,
    int32_t batch_size) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = memory_models_.find(backend);
  if (it == memory_models_.end()) {
    // No model available, return a conservative estimate
    return 100.0f * batch_size;  // 100MB per batch item
  }
  
  return it->second.slope * batch_size + it->second.intercept;
}

float
ModelProfile::PredictPowerUsage(
    RoutingDecision::BackendType backend,
    int32_t batch_size) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = power_models_.find(backend);
  if (it == power_models_.end()) {
    // No model available, return a conservative estimate
    return 50.0f * batch_size;  // 50W per batch item
  }
  
  return it->second.slope * batch_size + it->second.intercept;
}

float
ModelProfile::GetPredictionConfidence(
    RoutingDecision::BackendType backend) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = latency_models_.find(backend);
  if (it == latency_models_.end()) {
    return 0.0f;
  }
  
  // Use R-squared as confidence metric
  return it->second.r_squared;
}

void
ModelProfile::UpdateRegressionModels() const
{
  // Simple linear regression implementation
  for (const auto& [backend, records] : performance_history_) {
    if (records.size() < 10) {
      continue;  // Need at least 10 data points
    }
    
    // Calculate linear regression for latency
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    int n = records.size();
    
    for (const auto& record : records) {
      float x = static_cast<float>(record.batch_size);
      float y = record.latency_ms;
      sum_x += x;
      sum_y += y;
      sum_xy += x * y;
      sum_xx += x * x;
    }
    
    float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    float intercept = (sum_y - slope * sum_x) / n;
    
    // Calculate R-squared
    float mean_y = sum_y / n;
    float ss_tot = 0, ss_res = 0;
    
    for (const auto& record : records) {
      float x = static_cast<float>(record.batch_size);
      float y = record.latency_ms;
      float y_pred = slope * x + intercept;
      ss_tot += (y - mean_y) * (y - mean_y);
      ss_res += (y - y_pred) * (y - y_pred);
    }
    
    float r_squared = 1.0f - (ss_res / ss_tot);
    
    latency_models_[backend] = RegressionModel{slope, intercept, r_squared};
    
    // Similar calculations for memory and power models (simplified here)
    memory_models_[backend] = RegressionModel{10.0f, 100.0f, 0.8f};
    power_models_[backend] = RegressionModel{5.0f, 20.0f, 0.7f};
  }
}

//
// LatencyOptimizedPolicy Implementation
//
RoutingDecision
LatencyOptimizedPolicy::Route(
    const ModelCharacteristics& model,
    const RoutingContext& context,
    const std::vector<std::shared_ptr<BackendCapabilities>>& backends,
    const std::shared_ptr<ModelProfile>& profile)
{
  RoutingDecision decision;
  float best_latency = std::numeric_limits<float>::max();
  RoutingDecision::BackendType best_backend = RoutingDecision::BackendType::CPU;
  
  // Evaluate each backend
  for (const auto& backend : backends) {
    RoutingDecision::BackendType backend_type;
    
    // Map backend name to type
    if (backend->GetPeakThroughputGFLOPS() > 5000) {
      backend_type = RoutingDecision::BackendType::GPU;
    } else if (backend->SupportsNeuralEngine()) {
      backend_type = RoutingDecision::BackendType::NEURAL_ENGINE;
    } else {
      backend_type = RoutingDecision::BackendType::CPU;
    }
    
    // Skip if backend is overloaded
    if (backend->GetCurrentUtilization() > 0.9f) {
      continue;
    }
    
    // Skip if not enough memory
    float predicted_memory = profile->PredictMemoryUsage(backend_type, context.batch_size);
    if (predicted_memory > backend->GetAvailableMemoryMB()) {
      continue;
    }
    
    // Predict latency
    float predicted_latency = profile->PredictLatency(backend_type, context.batch_size);
    
    // Add penalty for low confidence predictions
    float confidence = profile->GetPredictionConfidence(backend_type);
    if (confidence < 0.5f) {
      predicted_latency *= 1.5f;  // 50% penalty for low confidence
    }
    
    if (predicted_latency < best_latency) {
      best_latency = predicted_latency;
      best_backend = backend_type;
    }
  }
  
  decision.primary_backend = best_backend;
  decision.expected_latency_ms = best_latency;
  decision.confidence_score = profile->GetPredictionConfidence(best_backend);
  decision.reason = "Selected backend with lowest predicted latency";
  
  return decision;
}

//
// ThroughputOptimizedPolicy Implementation
//
RoutingDecision
ThroughputOptimizedPolicy::Route(
    const ModelCharacteristics& model,
    const RoutingContext& context,
    const std::vector<std::shared_ptr<BackendCapabilities>>& backends,
    const std::shared_ptr<ModelProfile>& profile)
{
  RoutingDecision decision;
  float best_throughput = 0.0f;
  RoutingDecision::BackendType best_backend = RoutingDecision::BackendType::CPU;
  
  // For throughput optimization, prefer backends with high compute capacity
  for (const auto& backend : backends) {
    RoutingDecision::BackendType backend_type;
    
    if (backend->GetPeakThroughputGFLOPS() > 5000) {
      backend_type = RoutingDecision::BackendType::GPU;
    } else if (backend->SupportsNeuralEngine()) {
      backend_type = RoutingDecision::BackendType::NEURAL_ENGINE;
    } else {
      backend_type = RoutingDecision::BackendType::CPU;
    }
    
    // Calculate effective throughput considering current utilization
    float effective_throughput = backend->GetPeakThroughputGFLOPS() *
                                (1.0f - backend->GetCurrentUtilization());
    
    // Adjust for model characteristics
    if (model.is_quantized && backend->SupportsInt8()) {
      effective_throughput *= 2.0f;  // Int8 typically 2x faster
    }
    
    if (effective_throughput > best_throughput) {
      best_throughput = effective_throughput;
      best_backend = backend_type;
    }
  }
  
  decision.primary_backend = best_backend;
  decision.expected_throughput = best_throughput;
  decision.confidence_score = 0.8f;
  decision.reason = "Selected backend with highest available throughput";
  
  return decision;
}

//
// PowerEfficientPolicy Implementation
//
RoutingDecision
PowerEfficientPolicy::Route(
    const ModelCharacteristics& model,
    const RoutingContext& context,
    const std::vector<std::shared_ptr<BackendCapabilities>>& backends,
    const std::shared_ptr<ModelProfile>& profile)
{
  RoutingDecision decision;
  float best_efficiency = 0.0f;
  RoutingDecision::BackendType best_backend = RoutingDecision::BackendType::CPU;
  
  for (const auto& backend : backends) {
    RoutingDecision::BackendType backend_type;
    
    if (backend->SupportsNeuralEngine()) {
      backend_type = RoutingDecision::BackendType::NEURAL_ENGINE;
      // Neural engines are typically most power efficient
    } else if (backend->GetPeakThroughputGFLOPS() > 5000) {
      backend_type = RoutingDecision::BackendType::GPU;
    } else {
      backend_type = RoutingDecision::BackendType::CPU;
    }
    
    // Calculate performance per watt
    float predicted_power = profile->PredictPowerUsage(backend_type, context.batch_size);
    float predicted_latency = profile->PredictLatency(backend_type, context.batch_size);
    float throughput = context.batch_size / (predicted_latency / 1000.0f);  // items/second
    float efficiency = throughput / predicted_power;  // items/second/watt
    
    if (efficiency > best_efficiency) {
      best_efficiency = efficiency;
      best_backend = backend_type;
    }
  }
  
  decision.primary_backend = best_backend;
  decision.expected_power_usage_w = profile->PredictPowerUsage(best_backend, context.batch_size);
  decision.confidence_score = 0.7f;
  decision.reason = "Selected backend with best power efficiency";
  
  return decision;
}

//
// AdaptiveRoutingPolicy Implementation
//
RoutingDecision
AdaptiveRoutingPolicy::Route(
    const ModelCharacteristics& model,
    const RoutingContext& context,
    const std::vector<std::shared_ptr<BackendCapabilities>>& backends,
    const std::shared_ptr<ModelProfile>& profile)
{
  RoutingDecision decision;
  float best_score = -std::numeric_limits<float>::max();
  RoutingDecision::BackendType best_backend = RoutingDecision::BackendType::CPU;
  
  // Evaluate each backend with a comprehensive scoring function
  std::unordered_map<RoutingDecision::BackendType, std::shared_ptr<BackendCapabilities>> backend_map;
  
  for (const auto& backend : backends) {
    RoutingDecision::BackendType backend_type;
    
    if (backend->GetPeakThroughputGFLOPS() > 5000) {
      backend_type = RoutingDecision::BackendType::GPU;
    } else if (backend->SupportsNeuralEngine()) {
      backend_type = RoutingDecision::BackendType::NEURAL_ENGINE;
    } else {
      backend_type = RoutingDecision::BackendType::CPU;
    }
    
    backend_map[backend_type] = backend;
    
    float score = ScoreBackendForModel(model, context, backend_type, backend, profile);
    
    if (score > best_score) {
      best_score = score;
      best_backend = backend_type;
    }
  }
  
  // Consider fallback options
  if (best_score < 0.5f) {
    // Primary backend has low score, add fallback
    decision.secondary_backends.push_back(RoutingDecision::BackendType::CPU);
  }
  
  decision.primary_backend = best_backend;
  decision.confidence_score = std::min(1.0f, best_score);
  
  // Set expected metrics
  decision.expected_latency_ms = profile->PredictLatency(best_backend, context.batch_size);
  decision.expected_memory_usage_mb = profile->PredictMemoryUsage(best_backend, context.batch_size);
  decision.expected_power_usage_w = profile->PredictPowerUsage(best_backend, context.batch_size);
  
  // Generate reason
  std::stringstream reason;
  reason << "Adaptive routing selected " << static_cast<int>(best_backend)
         << " based on model characteristics and context";
  decision.reason = reason.str();
  
  return decision;
}

float
AdaptiveRoutingPolicy::ScoreBackendForModel(
    const ModelCharacteristics& model,
    const RoutingContext& context,
    RoutingDecision::BackendType backend,
    const std::shared_ptr<BackendCapabilities>& capabilities,
    const std::shared_ptr<ModelProfile>& profile)
{
  float score = 0.0f;
  
  // Model complexity factors
  float complexity_score = 0.0f;
  
  // Large models benefit from GPU/Neural Engine
  if (model.parameter_count > 1e9) {  // > 1B parameters
    if (backend == RoutingDecision::BackendType::GPU ||
        backend == RoutingDecision::BackendType::NEURAL_ENGINE) {
      complexity_score += 0.3f;
    }
  } else if (model.parameter_count < 1e6) {  // < 1M parameters
    if (backend == RoutingDecision::BackendType::CPU) {
      complexity_score += 0.3f;  // Small models run efficiently on CPU
    }
  }
  
  // Quantization support
  if (model.is_quantized) {
    if ((model.quantization_bits == 8 && capabilities->SupportsInt8()) ||
        (model.quantization_bits == 16 && capabilities->SupportsFP16())) {
      complexity_score += 0.2f;
    }
  }
  
  // Dynamic shapes are harder for specialized accelerators
  if (model.has_dynamic_shapes) {
    if (backend == RoutingDecision::BackendType::CPU) {
      complexity_score += 0.1f;
    } else {
      complexity_score -= 0.1f;
    }
  }
  
  score += complexity_score;
  
  // Performance prediction factors
  float performance_score = 0.0f;
  
  float predicted_latency = profile->PredictLatency(backend, context.batch_size);
  float latency_score = 0.0f;
  
  if (context.max_latency_ms > 0) {
    latency_score = 1.0f - (predicted_latency / context.max_latency_ms);
    latency_score = std::max(0.0f, std::min(1.0f, latency_score));
  } else {
    // No SLA, use relative scoring
    latency_score = 1.0f / (1.0f + predicted_latency / 100.0f);
  }
  
  performance_score += latency_score * 0.4f;
  
  // Resource availability
  float resource_score = 0.0f;
  
  float utilization = capabilities->GetCurrentUtilization();
  resource_score += (1.0f - utilization) * 0.3f;
  
  float predicted_memory = profile->PredictMemoryUsage(backend, context.batch_size);
  float memory_availability = capabilities->GetAvailableMemoryMB() / predicted_memory;
  if (memory_availability > 2.0f) {
    resource_score += 0.2f;
  } else if (memory_availability < 1.2f) {
    resource_score -= 0.3f;
  }
  
  score += resource_score;
  
  // Context-based adjustments
  switch (context.optimization_goal) {
    case RoutingContext::OptimizationGoal::MINIMIZE_LATENCY:
      score += latency_score * 0.5f;
      break;
    case RoutingContext::OptimizationGoal::MAXIMIZE_THROUGHPUT:
      score += (capabilities->GetPeakThroughputGFLOPS() / 10000.0f) * 0.3f;
      break;
    case RoutingContext::OptimizationGoal::MINIMIZE_POWER:
      score += (capabilities->GetPowerEfficiencyGFLOPSPerWatt() / 100.0f) * 0.3f;
      break;
    case RoutingContext::OptimizationGoal::BALANCED:
      // Already balanced
      break;
  }
  
  // Confidence adjustment
  float confidence = profile->GetPredictionConfidence(backend);
  if (confidence < 0.5f) {
    score *= 0.7f;  // Reduce score for uncertain predictions
  }
  
  return score;
}

//
// ModelRouter Implementation
//
ModelRouter&
ModelRouter::GetInstance()
{
  static ModelRouter instance;
  return instance;
}

TRITONSERVER_Error*
ModelRouter::Initialize(const std::vector<std::string>& backend_names)
{
  std::lock_guard<std::mutex> lock(backends_mutex_);
  
  // Initialize backend capabilities
  for (const auto& name : backend_names) {
    auto capabilities = std::make_shared<BackendCapabilities>(name);
    
    // Map to backend type
    RoutingDecision::BackendType type;
    if (name == "cpu") {
      type = RoutingDecision::BackendType::CPU;
    } else if (name == "gpu" || name == "cuda") {
      type = RoutingDecision::BackendType::GPU;
    } else if (name == "neural_engine" || name == "ane") {
      type = RoutingDecision::BackendType::NEURAL_ENGINE;
    } else if (name == "metal_mps") {
      type = RoutingDecision::BackendType::METAL_MPS;
    } else {
      // Unknown backend, default to CPU type
      type = RoutingDecision::BackendType::CPU;
    }
    
    backend_capabilities_[type] = capabilities;
  }
  
  // Initialize default routing policy
  default_policy_ = std::make_shared<AdaptiveRoutingPolicy>();
  
  // Initialize metrics
  metrics_ = std::make_shared<RoutingMetrics>();
  
  return nullptr;  // Success
}

TRITONSERVER_Error*
ModelRouter::RegisterModel(
    const std::string& model_name,
    const ModelCharacteristics& characteristics)
{
  {
    std::lock_guard<std::mutex> lock(models_mutex_);
    model_characteristics_[model_name] = characteristics;
    model_profiles_[model_name] = std::make_shared<ModelProfile>(model_name);
  }
  
  return nullptr;  // Success
}

RoutingDecision
ModelRouter::RouteRequest(
    const std::string& model_name,
    const RoutingContext& context)
{
  // Get model characteristics
  ModelCharacteristics characteristics;
  std::shared_ptr<ModelProfile> profile;
  
  {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    auto char_it = model_characteristics_.find(model_name);
    if (char_it == model_characteristics_.end()) {
      // Model not registered, use default characteristics
      characteristics.model_name = model_name;
      characteristics.parameter_count = 1e6;  // 1M parameters default
    } else {
      characteristics = char_it->second;
    }
    
    auto prof_it = model_profiles_.find(model_name);
    if (prof_it == model_profiles_.end()) {
      profile = std::make_shared<ModelProfile>(model_name);
      model_profiles_[model_name] = profile;
    } else {
      profile = prof_it->second;
    }
  }
  
  // Get available backends
  std::vector<std::shared_ptr<BackendCapabilities>> backends;
  {
    std::lock_guard<std::mutex> lock(backends_mutex_);
    for (const auto& [type, cap] : backend_capabilities_) {
      backends.push_back(cap);
    }
  }
  
  // Select routing policy
  std::shared_ptr<ModelRoutingPolicy> policy = default_policy_;
  
  // Check for A/B testing
  if (config_.enable_ab_testing && ab_test_state_) {
    if (ShouldUseTreatmentPolicy(model_name)) {
      policy = ab_test_state_->treatment_policy;
      ab_test_state_->treatment_count++;
    } else {
      policy = ab_test_state_->control_policy;
      ab_test_state_->control_count++;
    }
  } else {
    // Check for model-specific policy
    std::lock_guard<std::mutex> lock(policy_mutex_);
    auto it = model_policies_.find(model_name);
    if (it != model_policies_.end()) {
      policy = it->second;
    }
  }
  
  // Make routing decision
  RoutingDecision decision = policy->Route(characteristics, context, backends, profile);
  
  // Record metrics
  if (config_.enable_metrics) {
    metrics_->RecordRoutingDecision(model_name, decision, policy->GetName());
  }
  
  return decision;
}

void
ModelRouter::RecordExecution(
    const std::string& model_name,
    const RoutingDecision& decision,
    float actual_latency_ms,
    float actual_memory_mb,
    float actual_power_w)
{
  // Update model profile
  {
    std::lock_guard<std::mutex> lock(models_mutex_);
    auto it = model_profiles_.find(model_name);
    if (it != model_profiles_.end()) {
      // Get batch size from somewhere (would need to be passed in)
      int32_t batch_size = 1;  // Default
      it->second->RecordExecution(
          decision.primary_backend,
          batch_size,
          actual_latency_ms,
          actual_memory_mb,
          actual_power_w);
    }
  }
  
  // Update backend utilization (simplified)
  {
    std::lock_guard<std::mutex> lock(backends_mutex_);
    auto it = backend_capabilities_.find(decision.primary_backend);
    if (it != backend_capabilities_.end()) {
      // Simple utilization update based on latency
      float utilization = actual_latency_ms / 100.0f;  // Rough estimate
      it->second->UpdateUtilization(utilization);
    }
  }
  
  // Record metrics
  if (config_.enable_metrics) {
    metrics_->RecordExecution(model_name, decision.primary_backend,
                             actual_latency_ms, true);
  }
}

void
ModelRouter::SetRoutingPolicy(std::shared_ptr<ModelRoutingPolicy> policy)
{
  std::lock_guard<std::mutex> lock(policy_mutex_);
  default_policy_ = policy;
}

void
ModelRouter::SetRoutingPolicyForModel(
    const std::string& model_name,
    std::shared_ptr<ModelRoutingPolicy> policy)
{
  std::lock_guard<std::mutex> lock(policy_mutex_);
  model_policies_[model_name] = policy;
}

void
ModelRouter::EnableABTesting(
    const std::string& experiment_id,
    std::shared_ptr<ModelRoutingPolicy> control_policy,
    std::shared_ptr<ModelRoutingPolicy> treatment_policy,
    float treatment_percentage)
{
  ab_test_state_ = std::make_shared<ABTestState>();
  ab_test_state_->experiment_id = experiment_id;
  ab_test_state_->control_policy = control_policy;
  ab_test_state_->treatment_policy = treatment_policy;
  ab_test_state_->treatment_percentage = treatment_percentage;
  
  config_.enable_ab_testing = true;
}

std::shared_ptr<BackendCapabilities>
ModelRouter::GetBackendCapabilities(RoutingDecision::BackendType type)
{
  std::lock_guard<std::mutex> lock(backends_mutex_);
  auto it = backend_capabilities_.find(type);
  if (it != backend_capabilities_.end()) {
    return it->second;
  }
  return nullptr;
}

bool
ModelRouter::ShouldUseTreatmentPolicy(const std::string& model_name)
{
  if (!ab_test_state_) {
    return false;
  }
  
  // Simple hash-based assignment for consistency
  std::hash<std::string> hasher;
  size_t hash = hasher(model_name + ab_test_state_->experiment_id);
  float random_value = static_cast<float>(hash % 1000) / 1000.0f;
  
  return random_value < ab_test_state_->treatment_percentage;
}

void
ModelRouter::CollectMetrics(
    const std::string& model_name,
    const RoutingDecision& decision,
    float latency_ms)
{
  if (metrics_) {
    metrics_->RecordExecution(model_name, decision.primary_backend,
                             latency_ms, true);
  }
}

//
// RoutingMetrics Implementation
//
RoutingMetrics::RoutingMetrics()
{
}

void
RoutingMetrics::RecordRoutingDecision(
    const std::string& model_name,
    const RoutingDecision& decision,
    const std::string& policy_name)
{
  std::lock_guard<std::mutex> lock(mutex_);
  routing_counts_[model_name][decision.primary_backend]++;
  policy_counts_[model_name][policy_name]++;
}

void
RoutingMetrics::RecordExecution(
    const std::string& model_name,
    RoutingDecision::BackendType backend,
    float latency_ms,
    bool success)
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  ExecutionRecord record{
      backend,
      latency_ms,
      success,
      std::chrono::steady_clock::now()
  };
  
  execution_history_[model_name].push_back(record);
  
  // Keep only recent history (last 10000 records per model)
  if (execution_history_[model_name].size() > 10000) {
    execution_history_[model_name].erase(
        execution_history_[model_name].begin(),
        execution_history_[model_name].begin() + 5000);
  }
}

RoutingMetrics::ModelMetrics
RoutingMetrics::GetModelMetrics(const std::string& model_name) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  ModelMetrics metrics;
  metrics.total_requests = 0;
  
  // Calculate metrics from execution history
  auto history_it = execution_history_.find(model_name);
  if (history_it != execution_history_.end()) {
    const auto& history = history_it->second;
    
    // Group by backend
    std::unordered_map<RoutingDecision::BackendType, std::vector<float>> latencies;
    std::unordered_map<RoutingDecision::BackendType, int> success_counts;
    std::unordered_map<RoutingDecision::BackendType, int> total_counts;
    
    for (const auto& record : history) {
      latencies[record.backend].push_back(record.latency_ms);
      total_counts[record.backend]++;
      if (record.success) {
        success_counts[record.backend]++;
      }
      metrics.total_requests++;
    }
    
    // Calculate per-backend metrics
    for (const auto& [backend, latency_list] : latencies) {
      metrics.backend_counts[backend] = total_counts[backend];
      
      // Average latency
      float sum = std::accumulate(latency_list.begin(), latency_list.end(), 0.0f);
      metrics.avg_latency_ms[backend] = sum / latency_list.size();
      
      // P99 latency
      std::vector<float> sorted_latencies = latency_list;
      std::sort(sorted_latencies.begin(), sorted_latencies.end());
      size_t p99_index = static_cast<size_t>(sorted_latencies.size() * 0.99);
      metrics.p99_latency_ms[backend] = sorted_latencies[p99_index];
      
      // Success rate
      metrics.success_rate[backend] = 
          static_cast<float>(success_counts[backend]) / total_counts[backend];
    }
  }
  
  return metrics;
}

std::vector<std::string>
RoutingMetrics::GetModelNames() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  std::vector<std::string> names;
  for (const auto& [name, _] : execution_history_) {
    names.push_back(name);
  }
  
  return names;
}

std::string
RoutingMetrics::ExportPrometheusMetrics() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  
  std::stringstream output;
  
  // Export routing counts
  output << "# HELP triton_model_routing_total Total routing decisions by model and backend\n";
  output << "# TYPE triton_model_routing_total counter\n";
  
  for (const auto& [model, backend_counts] : routing_counts_) {
    for (const auto& [backend, count] : backend_counts) {
      output << "triton_model_routing_total{model=\"" << model
             << "\",backend=\"" << static_cast<int>(backend) << "\"} "
             << count << "\n";
    }
  }
  
  // Export latency metrics
  output << "\n# HELP triton_model_latency_ms Model inference latency in milliseconds\n";
  output << "# TYPE triton_model_latency_ms summary\n";
  
  for (const auto& model_name : GetModelNames()) {
    auto metrics = GetModelMetrics(model_name);
    
    for (const auto& [backend, avg_latency] : metrics.avg_latency_ms) {
      output << "triton_model_latency_ms{model=\"" << model_name
             << "\",backend=\"" << static_cast<int>(backend)
             << "\",quantile=\"0.5\"} " << avg_latency << "\n";
      
      output << "triton_model_latency_ms{model=\"" << model_name
             << "\",backend=\"" << static_cast<int>(backend)
             << "\",quantile=\"0.99\"} " << metrics.p99_latency_ms[backend] << "\n";
    }
  }
  
  // Export success rates
  output << "\n# HELP triton_model_success_rate Model inference success rate\n";
  output << "# TYPE triton_model_success_rate gauge\n";
  
  for (const auto& model_name : GetModelNames()) {
    auto metrics = GetModelMetrics(model_name);
    
    for (const auto& [backend, rate] : metrics.success_rate) {
      output << "triton_model_success_rate{model=\"" << model_name
             << "\",backend=\"" << static_cast<int>(backend) << "\"} "
             << rate << "\n";
    }
  }
  
  return output.str();
}

}}  // namespace triton::core