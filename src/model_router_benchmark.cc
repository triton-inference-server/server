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

#include <benchmark/benchmark.h>
#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <vector>

namespace triton { namespace core {

// Initialize router once for all benchmarks
class RouterBenchmarkFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) override {
    auto& router = ModelRouter::GetInstance();
    
    // Initialize if not already done
    static bool initialized = false;
    if (!initialized) {
      std::vector<std::string> backends = {"cpu", "gpu", "neural_engine", "metal_mps"};
      router.Initialize(backends);
      
      // Register various models
      RegisterBenchmarkModels();
      
      initialized = true;
    }
  }
  
 private:
  void RegisterBenchmarkModels() {
    auto& router = ModelRouter::GetInstance();
    
    // Small model (MobileNet-like)
    ModelCharacteristics small_model;
    small_model.model_name = "small_model";
    small_model.parameter_count = 4e6;
    small_model.flop_count = 0.5e9;
    small_model.memory_footprint_bytes = 16e6;
    small_model.is_quantized = true;
    small_model.quantization_bits = 8;
    router.RegisterModel("small_model", small_model);
    
    // Medium model (ResNet-like)
    ModelCharacteristics medium_model;
    medium_model.model_name = "medium_model";
    medium_model.parameter_count = 25e6;
    medium_model.flop_count = 4e9;
    medium_model.memory_footprint_bytes = 100e6;
    medium_model.supports_gpu_acceleration = true;
    router.RegisterModel("medium_model", medium_model);
    
    // Large model (BERT-like)
    ModelCharacteristics large_model;
    large_model.model_name = "large_model";
    large_model.parameter_count = 340e6;
    large_model.flop_count = 50e9;
    large_model.memory_footprint_bytes = 1.3e9;
    large_model.has_dynamic_shapes = true;
    router.RegisterModel("large_model", large_model);
    
    // Extra large model (GPT-like)
    ModelCharacteristics xl_model;
    xl_model.model_name = "xl_model";
    xl_model.parameter_count = 1.5e9;
    xl_model.flop_count = 500e9;
    xl_model.memory_footprint_bytes = 6e9;
    xl_model.has_control_flow = true;
    router.RegisterModel("xl_model", xl_model);
  }
};

//
// Benchmark: Basic routing decision
//
BENCHMARK_DEFINE_F(RouterBenchmarkFixture, BasicRouting)(benchmark::State& state) {
  auto& router = ModelRouter::GetInstance();
  
  RoutingContext context;
  context.batch_size = state.range(0);
  
  for (auto _ : state) {
    auto decision = router.RouteRequest("medium_model", context);
    benchmark::DoNotOptimize(decision);
  }
  
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(RouterBenchmarkFixture, BasicRouting)
    ->Arg(1)->Arg(8)->Arg(32)->Arg(64);

//
// Benchmark: Routing with different policies
//
BENCHMARK_DEFINE_F(RouterBenchmarkFixture, PolicyComparison)(benchmark::State& state) {
  auto& router = ModelRouter::GetInstance();
  
  std::vector<std::shared_ptr<ModelRoutingPolicy>> policies = {
      std::make_shared<LatencyOptimizedPolicy>(),
      std::make_shared<ThroughputOptimizedPolicy>(),
      std::make_shared<PowerEfficientPolicy>(),
      std::make_shared<AdaptiveRoutingPolicy>()
  };
  
  auto policy = policies[state.range(0)];
  router.SetRoutingPolicy(policy);
  
  RoutingContext context;
  context.batch_size = 16;
  
  for (auto _ : state) {
    auto decision = router.RouteRequest("medium_model", context);
    benchmark::DoNotOptimize(decision);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel(policy->GetName());
}
BENCHMARK_REGISTER_F(RouterBenchmarkFixture, PolicyComparison)
    ->Arg(0)->Arg(1)->Arg(2)->Arg(3);

//
// Benchmark: Routing with profiling
//
BENCHMARK_DEFINE_F(RouterBenchmarkFixture, RoutingWithProfiling)(benchmark::State& state) {
  auto& router = ModelRouter::GetInstance();
  
  RoutingContext context;
  context.batch_size = 16;
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> latency_dist(20.0f, 5.0f);
  
  for (auto _ : state) {
    state.PauseTiming();
    // Simulate some profiling data
    for (int i = 0; i < 10; i++) {
      auto decision = router.RouteRequest("medium_model", context);
      router.RecordExecution(
          "medium_model", decision,
          latency_dist(gen), 200.0f, 50.0f);
    }
    state.ResumeTiming();
    
    auto decision = router.RouteRequest("medium_model", context);
    benchmark::DoNotOptimize(decision);
  }
  
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(RouterBenchmarkFixture, RoutingWithProfiling);

//
// Benchmark: Concurrent routing requests
//
void ConcurrentRoutingBenchmark(benchmark::State& state) {
  auto& router = ModelRouter::GetInstance();
  
  const int num_threads = state.range(0);
  std::vector<std::thread> threads;
  std::atomic<int64_t> total_requests{0};
  std::atomic<bool> stop{false};
  
  auto worker = [&]() {
    RoutingContext context;
    context.batch_size = 8;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> model_dist(0, 3);
    std::vector<std::string> models = {
        "small_model", "medium_model", "large_model", "xl_model"
    };
    
    while (!stop.load()) {
      auto decision = router.RouteRequest(models[model_dist(gen)], context);
      benchmark::DoNotOptimize(decision);
      total_requests.fetch_add(1);
    }
  };
  
  for (auto _ : state) {
    state.PauseTiming();
    total_requests = 0;
    stop = false;
    
    // Start worker threads
    for (int i = 0; i < num_threads; i++) {
      threads.emplace_back(worker);
    }
    state.ResumeTiming();
    
    // Run for 1 second
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    state.PauseTiming();
    stop = true;
    for (auto& t : threads) {
      t.join();
    }
    threads.clear();
    state.ResumeTiming();
    
    state.SetIterationTime(1.0);
  }
  
  state.SetItemsProcessed(total_requests.load());
  state.counters["throughput"] = benchmark::Counter(
      total_requests.load(), benchmark::Counter::kIsRate);
}
BENCHMARK(ConcurrentRoutingBenchmark)
    ->Arg(1)->Arg(4)->Arg(8)->Arg(16)
    ->UseManualTime()
    ->Unit(benchmark::kSecond);

//
// Benchmark: Model characteristics impact
//
BENCHMARK_DEFINE_F(RouterBenchmarkFixture, ModelSizeImpact)(benchmark::State& state) {
  auto& router = ModelRouter::GetInstance();
  
  std::vector<std::string> models = {
      "small_model", "medium_model", "large_model", "xl_model"
  };
  
  std::string model = models[state.range(0)];
  
  RoutingContext context;
  context.batch_size = 16;
  context.optimization_goal = RoutingContext::OptimizationGoal::BALANCED;
  
  for (auto _ : state) {
    auto decision = router.RouteRequest(model, context);
    benchmark::DoNotOptimize(decision);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel(model);
}
BENCHMARK_REGISTER_F(RouterBenchmarkFixture, ModelSizeImpact)
    ->Arg(0)->Arg(1)->Arg(2)->Arg(3);

//
// Benchmark: A/B testing overhead
//
BENCHMARK_DEFINE_F(RouterBenchmarkFixture, ABTestingOverhead)(benchmark::State& state) {
  auto& router = ModelRouter::GetInstance();
  
  bool enable_ab = state.range(0);
  
  if (enable_ab) {
    router.EnableABTesting(
        "benchmark_test",
        std::make_shared<LatencyOptimizedPolicy>(),
        std::make_shared<AdaptiveRoutingPolicy>(),
        0.5f
    );
  }
  
  RoutingContext context;
  context.batch_size = 16;
  
  for (auto _ : state) {
    auto decision = router.RouteRequest("medium_model", context);
    benchmark::DoNotOptimize(decision);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel(enable_ab ? "With A/B" : "Without A/B");
}
BENCHMARK_REGISTER_F(RouterBenchmarkFixture, ABTestingOverhead)
    ->Arg(0)->Arg(1);

//
// Benchmark: Metrics collection overhead
//
BENCHMARK_DEFINE_F(RouterBenchmarkFixture, MetricsOverhead)(benchmark::State& state) {
  auto& router = ModelRouter::GetInstance();
  
  ModelRouter::Config config;
  config.enable_metrics = state.range(0);
  router.SetConfig(config);
  
  RoutingContext context;
  context.batch_size = 16;
  
  for (auto _ : state) {
    auto decision = router.RouteRequest("medium_model", context);
    
    // Simulate execution recording
    router.RecordExecution("medium_model", decision, 25.0f, 200.0f, 50.0f);
    
    benchmark::DoNotOptimize(decision);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.SetLabel(config.enable_metrics ? "With Metrics" : "Without Metrics");
}
BENCHMARK_REGISTER_F(RouterBenchmarkFixture, MetricsOverhead)
    ->Arg(0)->Arg(1);

//
// Benchmark: Backend capability queries
//
static void BM_BackendCapabilities(benchmark::State& state) {
  BackendCapabilities gpu_backend("gpu");
  
  for (auto _ : state) {
    // Simulate typical capability queries during routing
    bool supports_gpu = gpu_backend.SupportsGPU();
    bool supports_int8 = gpu_backend.SupportsInt8();
    float throughput = gpu_backend.GetPeakThroughputGFLOPS();
    float utilization = gpu_backend.GetCurrentUtilization();
    float memory = gpu_backend.GetAvailableMemoryMB();
    
    benchmark::DoNotOptimize(supports_gpu);
    benchmark::DoNotOptimize(supports_int8);
    benchmark::DoNotOptimize(throughput);
    benchmark::DoNotOptimize(utilization);
    benchmark::DoNotOptimize(memory);
  }
  
  state.SetItemsProcessed(state.iterations() * 5);  // 5 queries per iteration
}
BENCHMARK(BM_BackendCapabilities);

//
// Benchmark: Model profile predictions
//
static void BM_ModelProfilePredictions(benchmark::State& state) {
  ModelProfile profile("benchmark_model");
  
  // Add some training data
  for (int batch_size = 1; batch_size <= 64; batch_size *= 2) {
    for (int i = 0; i < 20; i++) {
      float latency = 10.0f + batch_size * 1.5f + (rand() % 10);
      profile.RecordExecution(
          RoutingDecision::BackendType::GPU,
          batch_size, latency, 100.0f, 50.0f);
    }
  }
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> batch_dist(1, 64);
  
  for (auto _ : state) {
    int batch_size = batch_dist(gen);
    
    float latency = profile.PredictLatency(
        RoutingDecision::BackendType::GPU, batch_size);
    float memory = profile.PredictMemoryUsage(
        RoutingDecision::BackendType::GPU, batch_size);
    float power = profile.PredictPowerUsage(
        RoutingDecision::BackendType::GPU, batch_size);
    float confidence = profile.GetPredictionConfidence(
        RoutingDecision::BackendType::GPU);
    
    benchmark::DoNotOptimize(latency);
    benchmark::DoNotOptimize(memory);
    benchmark::DoNotOptimize(power);
    benchmark::DoNotOptimize(confidence);
  }
  
  state.SetItemsProcessed(state.iterations() * 4);  // 4 predictions per iteration
}
BENCHMARK(BM_ModelProfilePredictions);

//
// Benchmark: Routing decision memory allocation
//
static void BM_RoutingDecisionAllocation(benchmark::State& state) {
  for (auto _ : state) {
    RoutingDecision decision;
    decision.primary_backend = RoutingDecision::BackendType::GPU;
    decision.secondary_backends.push_back(RoutingDecision::BackendType::CPU);
    decision.reason = "Benchmark routing decision";
    decision.confidence_score = 0.95f;
    decision.expected_latency_ms = 25.0f;
    decision.expected_throughput = 1000.0f;
    decision.expected_memory_usage_mb = 200.0f;
    decision.expected_power_usage_w = 150.0f;
    decision.instance_id = 0;
    decision.load_factor = 0.7f;
    
    benchmark::DoNotOptimize(decision);
  }
  
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_RoutingDecisionAllocation);

//
// Benchmark: End-to-end routing latency
//
BENCHMARK_DEFINE_F(RouterBenchmarkFixture, EndToEndLatency)(benchmark::State& state) {
  auto& router = ModelRouter::GetInstance();
  
  // Different scenarios
  std::vector<std::tuple<std::string, int, RoutingContext::OptimizationGoal>> scenarios = {
      {"small_model", 1, RoutingContext::OptimizationGoal::MINIMIZE_LATENCY},
      {"medium_model", 8, RoutingContext::OptimizationGoal::BALANCED},
      {"large_model", 32, RoutingContext::OptimizationGoal::MAXIMIZE_THROUGHPUT},
      {"xl_model", 64, RoutingContext::OptimizationGoal::MINIMIZE_POWER}
  };
  
  auto [model, batch_size, goal] = scenarios[state.range(0) % scenarios.size()];
  
  RoutingContext context;
  context.batch_size = batch_size;
  context.optimization_goal = goal;
  
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    
    auto decision = router.RouteRequest(model, context);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    state.SetIterationTime(elapsed.count() / 1e9);
    benchmark::DoNotOptimize(decision);
  }
  
  state.SetItemsProcessed(state.iterations());
  state.counters["latency_us"] = benchmark::Counter(
      state.iterations(), benchmark::Counter::kAvgIterations,
      benchmark::Counter::OneK::kIs1000);
}
BENCHMARK_REGISTER_F(RouterBenchmarkFixture, EndToEndLatency)
    ->Arg(0)->Arg(1)->Arg(2)->Arg(3)
    ->UseManualTime();

}}  // namespace triton::core

BENCHMARK_MAIN();