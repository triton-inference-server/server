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

#include <gtest/gtest.h>
#include "../model_router.h"
#include "../model_router_config.h"

namespace triton { namespace core {

class ModelRouterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize router for each test
    auto& router = ModelRouter::GetInstance();
    std::vector<std::string> backends = {"cpu", "gpu", "neural_engine"};
    router.Initialize(backends);
  }
  
  ModelCharacteristics CreateTestModel(
      const std::string& name,
      uint64_t param_count,
      bool supports_gpu = true,
      bool is_quantized = false)
  {
    ModelCharacteristics model;
    model.model_name = name;
    model.parameter_count = param_count;
    model.supports_gpu_acceleration = supports_gpu;
    model.is_quantized = is_quantized;
    model.quantization_bits = is_quantized ? 8 : 0;
    model.supports_batching = true;
    model.optimal_batch_size = 32;
    return model;
  }
};

//
// Basic functionality tests
//
TEST_F(ModelRouterTest, RegisterAndRouteModel) {
  auto& router = ModelRouter::GetInstance();
  
  // Register a model
  auto model = CreateTestModel("test_model", 10e6);
  ASSERT_EQ(nullptr, router.RegisterModel("test_model", model));
  
  // Route a request
  RoutingContext context;
  context.batch_size = 8;
  
  auto decision = router.RouteRequest("test_model", context);
  
  // Verify we got a valid decision
  EXPECT_GE(decision.confidence_score, 0.0f);
  EXPECT_LE(decision.confidence_score, 1.0f);
  EXPECT_FALSE(decision.reason.empty());
}

TEST_F(ModelRouterTest, UnregisteredModelRouting) {
  auto& router = ModelRouter::GetInstance();
  
  // Try routing for unregistered model
  RoutingContext context;
  context.batch_size = 4;
  
  auto decision = router.RouteRequest("unknown_model", context);
  
  // Should still get a valid decision with default characteristics
  EXPECT_GE(decision.confidence_score, 0.0f);
  EXPECT_FALSE(decision.reason.empty());
}

//
// Policy tests
//
TEST_F(ModelRouterTest, LatencyOptimizedPolicy) {
  auto& router = ModelRouter::GetInstance();
  router.SetRoutingPolicy(std::make_shared<LatencyOptimizedPolicy>());
  
  auto model = CreateTestModel("latency_test", 50e6);
  router.RegisterModel("latency_test", model);
  
  RoutingContext context;
  context.batch_size = 1;
  context.max_latency_ms = 10.0f;
  context.optimization_goal = RoutingContext::OptimizationGoal::MINIMIZE_LATENCY;
  
  auto decision = router.RouteRequest("latency_test", context);
  
  // Latency policy should pick fastest backend
  EXPECT_GT(decision.expected_latency_ms, 0.0f);
  EXPECT_EQ("Selected backend with lowest predicted latency", decision.reason);
}

TEST_F(ModelRouterTest, ThroughputOptimizedPolicy) {
  auto& router = ModelRouter::GetInstance();
  router.SetRoutingPolicy(std::make_shared<ThroughputOptimizedPolicy>());
  
  auto model = CreateTestModel("throughput_test", 100e6);
  router.RegisterModel("throughput_test", model);
  
  RoutingContext context;
  context.batch_size = 32;
  context.optimization_goal = RoutingContext::OptimizationGoal::MAXIMIZE_THROUGHPUT;
  
  auto decision = router.RouteRequest("throughput_test", context);
  
  // Throughput policy should pick highest capacity backend
  EXPECT_GT(decision.expected_throughput, 0.0f);
  EXPECT_EQ("Selected backend with highest available throughput", decision.reason);
}

TEST_F(ModelRouterTest, PowerEfficientPolicy) {
  auto& router = ModelRouter::GetInstance();
  router.SetRoutingPolicy(std::make_shared<PowerEfficientPolicy>());
  
  auto model = CreateTestModel("power_test", 5e6, true, true);
  router.RegisterModel("power_test", model);
  
  RoutingContext context;
  context.batch_size = 8;
  context.optimization_goal = RoutingContext::OptimizationGoal::MINIMIZE_POWER;
  
  auto decision = router.RouteRequest("power_test", context);
  
  // Power policy should pick most efficient backend
  EXPECT_GT(decision.expected_power_usage_w, 0.0f);
  EXPECT_EQ("Selected backend with best power efficiency", decision.reason);
}

TEST_F(ModelRouterTest, AdaptivePolicy) {
  auto& router = ModelRouter::GetInstance();
  router.SetRoutingPolicy(std::make_shared<AdaptiveRoutingPolicy>());
  
  // Test with different model characteristics
  auto small_model = CreateTestModel("small_model", 1e6);
  auto large_model = CreateTestModel("large_model", 1e9);
  auto quantized_model = CreateTestModel("quantized_model", 10e6, true, true);
  
  router.RegisterModel("small_model", small_model);
  router.RegisterModel("large_model", large_model);
  router.RegisterModel("quantized_model", quantized_model);
  
  RoutingContext context;
  context.batch_size = 16;
  
  auto small_decision = router.RouteRequest("small_model", context);
  auto large_decision = router.RouteRequest("large_model", context);
  auto quantized_decision = router.RouteRequest("quantized_model", context);
  
  // Adaptive policy should make different decisions based on model characteristics
  EXPECT_FALSE(small_decision.reason.empty());
  EXPECT_FALSE(large_decision.reason.empty());
  EXPECT_FALSE(quantized_decision.reason.empty());
}

//
// Profile and learning tests
//
TEST_F(ModelRouterTest, PerformanceProfiling) {
  auto& router = ModelRouter::GetInstance();
  
  auto model = CreateTestModel("profile_test", 20e6);
  router.RegisterModel("profile_test", model);
  
  RoutingContext context;
  context.batch_size = 8;
  
  // Initial routing
  auto initial_decision = router.RouteRequest("profile_test", context);
  float initial_confidence = initial_decision.confidence_score;
  
  // Simulate multiple executions
  for (int i = 0; i < 20; i++) {
    auto decision = router.RouteRequest("profile_test", context);
    
    // Record execution with consistent performance
    router.RecordExecution(
        "profile_test",
        decision,
        25.0f + i * 0.1f,  // Slight variation in latency
        200.0f,
        50.0f
    );
  }
  
  // After profiling, confidence should improve
  auto profiled_decision = router.RouteRequest("profile_test", context);
  EXPECT_GE(profiled_decision.confidence_score, initial_confidence);
}

//
// A/B testing tests
//
TEST_F(ModelRouterTest, ABTesting) {
  auto& router = ModelRouter::GetInstance();
  
  auto control = std::make_shared<LatencyOptimizedPolicy>();
  auto treatment = std::make_shared<ThroughputOptimizedPolicy>();
  
  router.EnableABTesting("test_experiment", control, treatment, 0.5f);
  
  auto model = CreateTestModel("ab_test_model", 30e6);
  router.RegisterModel("ab_test_model", model);
  
  RoutingContext context;
  context.batch_size = 16;
  
  // Run multiple requests
  int latency_policy_count = 0;
  int throughput_policy_count = 0;
  
  for (int i = 0; i < 100; i++) {
    auto decision = router.RouteRequest("ab_test_model", context);
    
    // Check which policy was likely used based on reason
    if (decision.reason.find("latency") != std::string::npos) {
      latency_policy_count++;
    } else if (decision.reason.find("throughput") != std::string::npos) {
      throughput_policy_count++;
    }
  }
  
  // Should be roughly 50/50 split
  EXPECT_GT(latency_policy_count, 30);
  EXPECT_GT(throughput_policy_count, 30);
  EXPECT_LT(std::abs(latency_policy_count - throughput_policy_count), 30);
}

//
// Configuration tests
//
TEST_F(ModelRouterTest, ModelSpecificPolicies) {
  auto& router = ModelRouter::GetInstance();
  
  // Set different policies for different models
  router.SetRoutingPolicyForModel("latency_model",
                                 std::make_shared<LatencyOptimizedPolicy>());
  router.SetRoutingPolicyForModel("throughput_model",
                                 std::make_shared<ThroughputOptimizedPolicy>());
  
  auto model1 = CreateTestModel("latency_model", 10e6);
  auto model2 = CreateTestModel("throughput_model", 10e6);
  
  router.RegisterModel("latency_model", model1);
  router.RegisterModel("throughput_model", model2);
  
  RoutingContext context;
  context.batch_size = 8;
  
  auto decision1 = router.RouteRequest("latency_model", context);
  auto decision2 = router.RouteRequest("throughput_model", context);
  
  EXPECT_EQ("Selected backend with lowest predicted latency", decision1.reason);
  EXPECT_EQ("Selected backend with highest available throughput", decision2.reason);
}

TEST_F(ModelRouterTest, RouterConfiguration) {
  auto& router = ModelRouter::GetInstance();
  
  ModelRouter::Config config;
  config.enable_profiling = true;
  config.profile_warmup_iterations = 5;
  config.enable_load_balancing = true;
  config.max_backend_utilization = 0.7f;
  config.enable_fallback = true;
  config.fallback_latency_threshold_ms = 50.0f;
  
  router.SetConfig(config);
  
  auto retrieved_config = router.GetConfig();
  EXPECT_EQ(config.enable_profiling, retrieved_config.enable_profiling);
  EXPECT_EQ(config.profile_warmup_iterations, retrieved_config.profile_warmup_iterations);
  EXPECT_EQ(config.max_backend_utilization, retrieved_config.max_backend_utilization);
}

//
// Backend capability tests
//
TEST_F(ModelRouterTest, BackendCapabilities) {
  BackendCapabilities cpu_backend("cpu");
  BackendCapabilities gpu_backend("gpu");
  BackendCapabilities ne_backend("neural_engine");
  
  // Test CPU capabilities
  EXPECT_FALSE(cpu_backend.SupportsGPU());
  EXPECT_FALSE(cpu_backend.SupportsNeuralEngine());
  EXPECT_TRUE(cpu_backend.SupportsInt8());
  EXPECT_FALSE(cpu_backend.SupportsFP16());
  
  // Test GPU capabilities
  EXPECT_TRUE(gpu_backend.SupportsGPU());
  EXPECT_FALSE(gpu_backend.SupportsNeuralEngine());
  EXPECT_TRUE(gpu_backend.SupportsInt8());
  EXPECT_TRUE(gpu_backend.SupportsFP16());
  
  // Test Neural Engine capabilities
  EXPECT_FALSE(ne_backend.SupportsGPU());
  EXPECT_TRUE(ne_backend.SupportsNeuralEngine());
  EXPECT_TRUE(ne_backend.SupportsInt8());
  EXPECT_TRUE(ne_backend.SupportsFP16());
  
  // Test dynamic updates
  cpu_backend.UpdateUtilization(0.5f);
  EXPECT_FLOAT_EQ(0.5f, cpu_backend.GetCurrentUtilization());
  
  cpu_backend.UpdateAvailableMemory(4096.0f);
  EXPECT_FLOAT_EQ(4096.0f, cpu_backend.GetAvailableMemoryMB());
}

//
// Model profile tests
//
TEST_F(ModelRouterTest, ModelProfile) {
  ModelProfile profile("test_model");
  
  // Record some executions
  for (int batch_size = 1; batch_size <= 32; batch_size *= 2) {
    float latency = 10.0f + batch_size * 2.0f;  // Linear relationship
    float memory = 100.0f + batch_size * 10.0f;
    float power = 50.0f + batch_size * 5.0f;
    
    profile.RecordExecution(
        RoutingDecision::BackendType::GPU,
        batch_size,
        latency,
        memory,
        power
    );
  }
  
  // Test predictions
  float predicted_latency = profile.PredictLatency(
      RoutingDecision::BackendType::GPU, 16);
  float predicted_memory = profile.PredictMemoryUsage(
      RoutingDecision::BackendType::GPU, 16);
  float predicted_power = profile.PredictPowerUsage(
      RoutingDecision::BackendType::GPU, 16);
  
  // Predictions should be reasonable
  EXPECT_GT(predicted_latency, 0.0f);
  EXPECT_LT(predicted_latency, 1000.0f);
  EXPECT_GT(predicted_memory, 0.0f);
  EXPECT_LT(predicted_memory, 10000.0f);
  EXPECT_GT(predicted_power, 0.0f);
  EXPECT_LT(predicted_power, 1000.0f);
  
  // Confidence should be low for backends with no data
  float confidence = profile.GetPredictionConfidence(
      RoutingDecision::BackendType::CPU);
  EXPECT_EQ(0.0f, confidence);
}

//
// Metrics tests
//
TEST_F(ModelRouterTest, RoutingMetrics) {
  RoutingMetrics metrics;
  
  // Record some routing decisions
  RoutingDecision decision1;
  decision1.primary_backend = RoutingDecision::BackendType::GPU;
  metrics.RecordRoutingDecision("model1", decision1, "latency");
  
  RoutingDecision decision2;
  decision2.primary_backend = RoutingDecision::BackendType::CPU;
  metrics.RecordRoutingDecision("model1", decision2, "power");
  
  // Record executions
  metrics.RecordExecution("model1", RoutingDecision::BackendType::GPU, 25.0f, true);
  metrics.RecordExecution("model1", RoutingDecision::BackendType::GPU, 30.0f, true);
  metrics.RecordExecution("model1", RoutingDecision::BackendType::CPU, 50.0f, true);
  metrics.RecordExecution("model1", RoutingDecision::BackendType::CPU, 55.0f, false);
  
  // Get metrics
  auto model_metrics = metrics.GetModelMetrics("model1");
  
  EXPECT_EQ(4, model_metrics.total_requests);
  EXPECT_EQ(2, model_metrics.backend_counts[RoutingDecision::BackendType::GPU]);
  EXPECT_EQ(2, model_metrics.backend_counts[RoutingDecision::BackendType::CPU]);
  
  // Check average latencies
  EXPECT_FLOAT_EQ(27.5f, model_metrics.avg_latency_ms[RoutingDecision::BackendType::GPU]);
  EXPECT_FLOAT_EQ(52.5f, model_metrics.avg_latency_ms[RoutingDecision::BackendType::CPU]);
  
  // Check success rates
  EXPECT_FLOAT_EQ(1.0f, model_metrics.success_rate[RoutingDecision::BackendType::GPU]);
  EXPECT_FLOAT_EQ(0.5f, model_metrics.success_rate[RoutingDecision::BackendType::CPU]);
  
  // Test Prometheus export
  std::string prometheus_metrics = metrics.ExportPrometheusMetrics();
  EXPECT_FALSE(prometheus_metrics.empty());
  EXPECT_NE(prometheus_metrics.find("triton_model_routing_total"), std::string::npos);
  EXPECT_NE(prometheus_metrics.find("triton_model_latency_ms"), std::string::npos);
  EXPECT_NE(prometheus_metrics.find("triton_model_success_rate"), std::string::npos);
}

//
// Edge case tests
//
TEST_F(ModelRouterTest, ResourceConstraints) {
  auto& router = ModelRouter::GetInstance();
  
  auto model = CreateTestModel("resource_test", 1e9);  // 1B params
  router.RegisterModel("resource_test", model);
  
  RoutingContext context;
  context.batch_size = 64;
  context.max_memory_mb = 100.0f;  // Very limited memory
  
  auto decision = router.RouteRequest("resource_test", context);
  
  // Should still get a valid decision
  EXPECT_GE(decision.confidence_score, 0.0f);
  EXPECT_FALSE(decision.reason.empty());
}

TEST_F(ModelRouterTest, DynamicShapeModels) {
  auto& router = ModelRouter::GetInstance();
  
  ModelCharacteristics dynamic_model;
  dynamic_model.model_name = "dynamic_shape_model";
  dynamic_model.has_dynamic_shapes = true;
  dynamic_model.has_control_flow = true;
  dynamic_model.parameter_count = 50e6;
  
  router.RegisterModel("dynamic_shape_model", dynamic_model);
  
  RoutingContext context;
  context.batch_size = 1;
  context.actual_input_shapes["input"] = {1, 512, 512, 3};
  
  auto decision = router.RouteRequest("dynamic_shape_model", context);
  
  // Should handle dynamic shapes appropriately
  EXPECT_GE(decision.confidence_score, 0.0f);
  EXPECT_FALSE(decision.reason.empty());
}

}}  // namespace triton::core