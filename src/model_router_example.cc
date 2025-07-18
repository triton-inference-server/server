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
#include "model_router_config.h"

#include <iostream>
#include <chrono>
#include <random>
#include <thread>

namespace triton { namespace core {

//
// Example 1: Basic routing setup
//
void BasicRoutingExample()
{
  std::cout << "\n=== Basic Routing Example ===" << std::endl;
  
  // Get router instance
  auto& router = ModelRouter::GetInstance();
  
  // Initialize with available backends
  std::vector<std::string> backends = {"cpu", "gpu", "neural_engine", "metal_mps"};
  router.Initialize(backends);
  
  // Register a model
  ModelCharacteristics resnet50;
  resnet50.model_name = "resnet50";
  resnet50.framework = "ONNX";
  resnet50.parameter_count = 25e6;  // 25M parameters
  resnet50.flop_count = 4e9;  // 4 GFLOPs
  resnet50.memory_footprint_bytes = 100e6;  // 100MB
  resnet50.supports_batching = true;
  resnet50.optimal_batch_size = 32;
  resnet50.supports_gpu_acceleration = true;
  resnet50.supports_neural_engine = true;
  
  router.RegisterModel("resnet50", resnet50);
  
  // Create routing context
  RoutingContext context;
  context.batch_size = 8;
  context.max_latency_ms = 50.0f;  // 50ms SLA
  context.optimization_goal = RoutingContext::OptimizationGoal::MINIMIZE_LATENCY;
  
  // Route request
  auto decision = router.RouteRequest("resnet50", context);
  
  std::cout << "Routing decision for ResNet50:" << std::endl;
  std::cout << "  Primary backend: " << static_cast<int>(decision.primary_backend) << std::endl;
  std::cout << "  Expected latency: " << decision.expected_latency_ms << " ms" << std::endl;
  std::cout << "  Confidence: " << decision.confidence_score << std::endl;
  std::cout << "  Reason: " << decision.reason << std::endl;
}

//
// Example 2: Different optimization goals
//
void OptimizationGoalsExample()
{
  std::cout << "\n=== Optimization Goals Example ===" << std::endl;
  
  auto& router = ModelRouter::GetInstance();
  
  // Register a larger model
  ModelCharacteristics bert;
  bert.model_name = "bert_large";
  bert.framework = "PyTorch";
  bert.parameter_count = 340e6;  // 340M parameters
  bert.flop_count = 50e9;  // 50 GFLOPs
  bert.memory_footprint_bytes = 1.3e9;  // 1.3GB
  bert.has_dynamic_shapes = true;
  bert.supports_gpu_acceleration = true;
  
  router.RegisterModel("bert_large", bert);
  
  // Test different optimization goals
  std::vector<RoutingContext::OptimizationGoal> goals = {
      RoutingContext::OptimizationGoal::MINIMIZE_LATENCY,
      RoutingContext::OptimizationGoal::MAXIMIZE_THROUGHPUT,
      RoutingContext::OptimizationGoal::MINIMIZE_POWER,
      RoutingContext::OptimizationGoal::BALANCED
  };
  
  for (auto goal : goals) {
    RoutingContext context;
    context.batch_size = 16;
    context.optimization_goal = goal;
    
    auto decision = router.RouteRequest("bert_large", context);
    
    std::cout << "\nOptimization goal: " << static_cast<int>(goal) << std::endl;
    std::cout << "  Selected backend: " << static_cast<int>(decision.primary_backend) << std::endl;
    std::cout << "  Expected metrics:" << std::endl;
    std::cout << "    Latency: " << decision.expected_latency_ms << " ms" << std::endl;
    std::cout << "    Memory: " << decision.expected_memory_usage_mb << " MB" << std::endl;
    std::cout << "    Power: " << decision.expected_power_usage_w << " W" << std::endl;
  }
}

//
// Example 3: Policy comparison
//
void PolicyComparisonExample()
{
  std::cout << "\n=== Policy Comparison Example ===" << std::endl;
  
  auto& router = ModelRouter::GetInstance();
  
  // Register a quantized model
  ModelCharacteristics mobilenet;
  mobilenet.model_name = "mobilenet_v2";
  mobilenet.framework = "TensorFlow";
  mobilenet.parameter_count = 3.5e6;  // 3.5M parameters
  mobilenet.is_quantized = true;
  mobilenet.quantization_bits = 8;
  mobilenet.supports_int8_acceleration = true;
  
  router.RegisterModel("mobilenet_v2", mobilenet);
  
  // Test different policies
  std::vector<std::shared_ptr<ModelRoutingPolicy>> policies = {
      std::make_shared<LatencyOptimizedPolicy>(),
      std::make_shared<ThroughputOptimizedPolicy>(),
      std::make_shared<PowerEfficientPolicy>(),
      std::make_shared<AdaptiveRoutingPolicy>()
  };
  
  RoutingContext context;
  context.batch_size = 4;
  
  for (const auto& policy : policies) {
    router.SetRoutingPolicy(policy);
    auto decision = router.RouteRequest("mobilenet_v2", context);
    
    std::cout << "\nPolicy: " << policy->GetName() << std::endl;
    std::cout << "  Backend: " << static_cast<int>(decision.primary_backend) << std::endl;
    std::cout << "  Confidence: " << decision.confidence_score << std::endl;
  }
}

//
// Example 4: A/B testing
//
void ABTestingExample()
{
  std::cout << "\n=== A/B Testing Example ===" << std::endl;
  
  auto& router = ModelRouter::GetInstance();
  
  // Enable A/B testing
  auto control_policy = std::make_shared<LatencyOptimizedPolicy>();
  auto treatment_policy = std::make_shared<AdaptiveRoutingPolicy>();
  
  router.EnableABTesting(
      "adaptive_vs_latency_test",
      control_policy,
      treatment_policy,
      0.3f  // 30% get treatment
  );
  
  // Simulate multiple requests
  RoutingContext context;
  context.batch_size = 8;
  
  int control_count = 0;
  int treatment_count = 0;
  
  for (int i = 0; i < 100; i++) {
    auto decision = router.RouteRequest("resnet50", context);
    
    // In real implementation, we'd check which policy was used
    // For this example, we'll estimate based on backend selection
    if (decision.confidence_score > 0.7f) {
      treatment_count++;
    } else {
      control_count++;
    }
  }
  
  std::cout << "A/B test results after 100 requests:" << std::endl;
  std::cout << "  Control group: ~" << control_count << " requests" << std::endl;
  std::cout << "  Treatment group: ~" << treatment_count << " requests" << std::endl;
}

//
// Example 5: Performance profiling and adaptation
//
void PerformanceProfilingExample()
{
  std::cout << "\n=== Performance Profiling Example ===" << std::endl;
  
  auto& router = ModelRouter::GetInstance();
  
  // Simulate execution and profiling
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> latency_dist(20.0f, 5.0f);
  std::normal_distribution<float> memory_dist(200.0f, 50.0f);
  std::normal_distribution<float> power_dist(100.0f, 20.0f);
  
  RoutingContext context;
  context.batch_size = 16;
  
  // Simulate 50 executions
  for (int i = 0; i < 50; i++) {
    auto decision = router.RouteRequest("resnet50", context);
    
    // Simulate execution with some variability
    float actual_latency = std::max(5.0f, latency_dist(gen));
    float actual_memory = std::max(50.0f, memory_dist(gen));
    float actual_power = std::max(20.0f, power_dist(gen));
    
    // Record execution results
    router.RecordExecution(
        "resnet50",
        decision,
        actual_latency,
        actual_memory,
        actual_power
    );
    
    if (i % 10 == 0) {
      std::cout << "After " << (i + 1) << " executions:" << std::endl;
      std::cout << "  Predicted latency: " << decision.expected_latency_ms << " ms" << std::endl;
      std::cout << "  Actual latency: " << actual_latency << " ms" << std::endl;
      std::cout << "  Confidence: " << decision.confidence_score << std::endl;
    }
  }
}

//
// Example 6: Configuration from file
//
void ConfigurationExample()
{
  std::cout << "\n=== Configuration Example ===" << std::endl;
  
  // Create configuration using builder
  auto config = ModelRouterConfigBuilder()
      .SetDefaultPolicy("adaptive")
      .AddRoutingRule("resnet*", "throughput")
      .AddRoutingRule("bert_*", "latency")
      .AddRoutingRule("*_mobile", "power")
      .SetBackendWeight("neural_engine", 1.5f)
      .SetBackendWeight("gpu", 1.2f)
      .SetBackendWeight("cpu", 0.8f)
      .EnableProfiling(true)
      .EnableLoadBalancing(true)
      .Build();
  
  // Apply configuration
  auto& router = ModelRouter::GetInstance();
  config.ApplyToRouter(router);
  
  std::cout << "Applied routing configuration:" << std::endl;
  std::cout << "  ResNet models -> Throughput policy" << std::endl;
  std::cout << "  BERT models -> Latency policy" << std::endl;
  std::cout << "  Mobile models -> Power policy" << std::endl;
  std::cout << "  Backend weights: Neural Engine=1.5, GPU=1.2, CPU=0.8" << std::endl;
}

//
// Example 7: Metrics export
//
void MetricsExportExample()
{
  std::cout << "\n=== Metrics Export Example ===" << std::endl;
  
  auto& router = ModelRouter::GetInstance();
  
  // Simulate some traffic
  std::vector<std::string> models = {"resnet50", "bert_base", "mobilenet_v2"};
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> model_dist(0, models.size() - 1);
  std::uniform_int_distribution<> batch_dist(1, 32);
  
  for (int i = 0; i < 100; i++) {
    RoutingContext context;
    context.batch_size = batch_dist(gen);
    
    std::string model = models[model_dist(gen)];
    auto decision = router.RouteRequest(model, context);
    
    // Simulate execution
    float latency = 10.0f + static_cast<float>(context.batch_size) * 2.0f;
    router.RecordExecution(model, decision, latency, 100.0f, 50.0f);
  }
  
  // Export metrics
  auto metrics = router.GetMetrics();
  std::cout << "\nPrometheus metrics:\n" << metrics->ExportPrometheusMetrics() << std::endl;
}

//
// Benchmark: Routing overhead
//
void BenchmarkRoutingOverhead()
{
  std::cout << "\n=== Routing Overhead Benchmark ===" << std::endl;
  
  auto& router = ModelRouter::GetInstance();
  
  RoutingContext context;
  context.batch_size = 8;
  
  const int num_iterations = 10000;
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < num_iterations; i++) {
    auto decision = router.RouteRequest("resnet50", context);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  double avg_routing_time = static_cast<double>(duration.count()) / num_iterations;
  
  std::cout << "Routing overhead benchmark results:" << std::endl;
  std::cout << "  Total iterations: " << num_iterations << std::endl;
  std::cout << "  Average routing time: " << avg_routing_time << " μs" << std::endl;
  std::cout << "  Routing throughput: " << (1000000.0 / avg_routing_time) 
            << " decisions/second" << std::endl;
}

//
// Main function to run all examples
//
void RunAllExamples()
{
  std::cout << "Model Router Examples and Benchmarks" << std::endl;
  std::cout << "====================================" << std::endl;
  
  // Run examples
  BasicRoutingExample();
  OptimizationGoalsExample();
  PolicyComparisonExample();
  ABTestingExample();
  PerformanceProfilingExample();
  ConfigurationExample();
  MetricsExportExample();
  
  // Run benchmarks
  BenchmarkRoutingOverhead();
  
  std::cout << "\nAll examples completed!" << std::endl;
}

}}  // namespace triton::core

// Standalone main for testing
int main()
{
  triton::core::RunAllExamples();
  return 0;
}