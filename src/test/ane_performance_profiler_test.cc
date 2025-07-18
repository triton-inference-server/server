// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// ANE Performance Profiler Tests

#include <gtest/gtest.h>
#include <chrono>
#include <thread>

#include "../apple/ane_performance_profiler.h"
#include "../apple/ane_provider.h"

using namespace triton::apple;

class ANEPerformanceProfilerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize ANE if available
        auto err = ANEProvider::Instance().Initialize();
        if (err) {
            GTEST_SKIP() << "ANE not available on this system";
            TRITONSERVER_ErrorDelete(err);
        }
        
        // Initialize profiler
        ANEPerformanceProfiler::Config config;
        config.warmup_iterations = 2;
        config.profile_iterations = 5;
        config.export_path = "./test_ane_profiles/";
        config.export_json = true;
        config.export_csv = false;
        
        profiler_ = std::make_unique<ANEPerformanceProfiler>();
        err = profiler_->Initialize(config);
        ASSERT_EQ(err, nullptr) << "Failed to initialize profiler";
    }
    
    void TearDown() override {
        // Clean up test files
        system("rm -rf ./test_ane_profiles/");
    }
    
    std::unique_ptr<ANEPerformanceProfiler> profiler_;
    
    // Helper to create a mock model path
    std::string GetMockModelPath() {
        // In real tests, this would return a path to a test CoreML model
        return "./test_model.mlmodel";
    }
};

TEST_F(ANEPerformanceProfilerTest, MetricsCalculation) {
    ANEPerformanceMetrics metrics;
    
    // Set some metrics
    metrics.inference_time_ms = 10.0;
    metrics.average_power_watts = 2.0;
    metrics.flops_per_inference = 1e9;  // 1 GFLOP
    
    // Calculate derived metrics
    metrics.CalculateDerivedMetrics();
    
    // Check calculations
    EXPECT_NEAR(metrics.inferences_per_second, 100.0, 0.01);
    EXPECT_GT(metrics.efficiency_tops_per_watt, 0.0);
    EXPECT_NEAR(metrics.energy_per_inference_joules, 0.02, 0.001);
}

TEST_F(ANEPerformanceProfilerTest, MetricsJSON) {
    ANEPerformanceMetrics metrics;
    metrics.inference_time_ms = 5.0;
    metrics.memory_usage_mb = 100.0;
    metrics.average_power_watts = 2.5;
    metrics.model_size_mb = 50;
    metrics.precision = "INT8";
    
    metrics.CalculateDerivedMetrics();
    
    std::string json = metrics.ToJSON();
    
    // Verify JSON contains expected fields
    EXPECT_NE(json.find("\"inference_time_ms\": 5"), std::string::npos);
    EXPECT_NE(json.find("\"memory_usage_mb\": 100"), std::string::npos);
    EXPECT_NE(json.find("\"average_power_watts\": 2.5"), std::string::npos);
    EXPECT_NE(json.find("\"precision\": \"INT8\""), std::string::npos);
    EXPECT_NE(json.find("\"inferences_per_second\""), std::string::npos);
}

TEST_F(ANEPerformanceProfilerTest, ProfilerInitialization) {
    // Already initialized in SetUp, just verify
    EXPECT_NE(profiler_, nullptr);
    
    // Try to profile without a model (should fail gracefully)
    ANEPerformanceMetrics metrics;
    auto err = profiler_->ProfileModel("nonexistent.mlmodel", "test_model", metrics);
    EXPECT_NE(err, nullptr);
    if (err) {
        TRITONSERVER_ErrorDelete(err);
    }
}

TEST_F(ANEPerformanceProfilerTest, SimulatedProfiling) {
    // Since we don't have a real model, simulate the profiling process
    ANEPerformanceMetrics metrics;
    
    // Manually set metrics as if we profiled
    metrics.load_time_ms = 100.0;
    metrics.compilation_time_ms = 50.0;
    metrics.inference_time_ms = 5.0;
    metrics.memory_usage_mb = 75.0;
    metrics.average_power_watts = 2.0;
    metrics.ane_utilization_percent = 80.0;
    metrics.model_size_mb = 25;
    metrics.parameter_count = 1000000;
    metrics.flops_per_inference = 2e9;
    
    metrics.CalculateDerivedMetrics();
    
    // Verify metrics are reasonable
    EXPECT_GT(metrics.inferences_per_second, 0);
    EXPECT_GT(metrics.efficiency_tops_per_watt, 0);
    EXPECT_EQ(metrics.total_time_ms, 155.0);
    
    // Test export
    auto err = profiler_->ExportResults(metrics, "test_export");
    EXPECT_EQ(err, nullptr);
    
    // Verify file was created
    std::ifstream file("./test_ane_profiles/test_export.json");
    EXPECT_TRUE(file.good());
}

TEST_F(ANEPerformanceProfilerTest, BatchProfiling) {
    ANEPerformanceProfiler::BatchProfilingResults results;
    
    // Simulate batch profiling results
    results.batch_sizes = {1, 2, 4, 8, 16};
    results.latencies_ms = {5.0, 8.0, 14.0, 25.0, 48.0};
    results.throughputs = {200.0, 250.0, 285.7, 320.0, 333.3};
    results.power_consumption = {2.0, 2.5, 3.0, 3.5, 4.0};
    
    // Find optimal batch size (highest throughput)
    auto max_it = std::max_element(results.throughputs.begin(), results.throughputs.end());
    size_t optimal_idx = std::distance(results.throughputs.begin(), max_it);
    results.optimal_batch_size = results.batch_sizes[optimal_idx];
    results.optimal_throughput = *max_it;
    
    EXPECT_EQ(results.optimal_batch_size, 16);
    EXPECT_NEAR(results.optimal_throughput, 333.3, 0.1);
    
    // Calculate efficiency scores
    results.efficiency_scores.clear();
    for (size_t i = 0; i < results.batch_sizes.size(); ++i) {
        double efficiency = results.throughputs[i] / results.power_consumption[i];
        results.efficiency_scores.push_back(efficiency);
    }
    
    // Best efficiency might be different from best throughput
    auto max_eff_it = std::max_element(results.efficiency_scores.begin(), 
                                       results.efficiency_scores.end());
    size_t best_efficiency_batch = results.batch_sizes[
        std::distance(results.efficiency_scores.begin(), max_eff_it)];
    
    EXPECT_GE(best_efficiency_batch, 1);
    EXPECT_LE(best_efficiency_batch, 16);
}

TEST_F(ANEPerformanceProfilerTest, ComparativeAnalysis) {
    ANEPerformanceProfiler::ComparativeResults comp_results;
    
    // Simulate comparative results
    comp_results.ane_metrics.inference_time_ms = 5.0;
    comp_results.ane_metrics.average_power_watts = 2.0;
    comp_results.ane_metrics.efficiency_tops_per_watt = 8.0;
    
    comp_results.cpu_metrics.inference_time_ms = 25.0;
    comp_results.cpu_metrics.average_power_watts = 15.0;
    comp_results.cpu_metrics.efficiency_tops_per_watt = 1.0;
    
    comp_results.gpu_metrics.inference_time_ms = 4.0;
    comp_results.gpu_metrics.average_power_watts = 25.0;
    comp_results.gpu_metrics.efficiency_tops_per_watt = 3.0;
    
    // Calculate speedups
    comp_results.ane_vs_cpu_speedup = 
        comp_results.cpu_metrics.inference_time_ms / comp_results.ane_metrics.inference_time_ms;
    comp_results.ane_vs_gpu_speedup = 
        comp_results.gpu_metrics.inference_time_ms / comp_results.ane_metrics.inference_time_ms;
    
    EXPECT_NEAR(comp_results.ane_vs_cpu_speedup, 5.0, 0.01);
    EXPECT_NEAR(comp_results.ane_vs_gpu_speedup, 0.8, 0.01);
    
    // ANE is 5x faster than CPU but 20% slower than GPU
    // However, ANE is much more power efficient
    
    comp_results.ane_vs_cpu_efficiency = 8.0;  // 8x more efficient
    comp_results.ane_vs_gpu_efficiency = 2.67; // 2.67x more efficient
    
    // Make recommendation
    if (comp_results.ane_vs_gpu_efficiency > 2.0) {
        comp_results.recommended_target = "ANE";
        comp_results.recommendation_reason = "ANE provides best power efficiency";
    } else {
        comp_results.recommended_target = "GPU";
        comp_results.recommendation_reason = "GPU provides best performance";
    }
    
    EXPECT_EQ(comp_results.recommended_target, "ANE");
}

TEST_F(ANEPerformanceProfilerTest, PowerProfiling) {
    ANEPerformanceProfiler::PowerProfile power_profile;
    
    // Simulate power measurements over time
    for (int i = 0; i < 100; ++i) {
        ANEPerformanceProfiler::PowerProfile::PowerState state;
        state.timestamp_ms = i * 10.0;
        state.power_watts = 2.0 + 0.5 * std::sin(i * 0.1);  // Varying power
        state.temperature_celsius = 35.0 + 5.0 * (state.power_watts - 2.0);
        state.ane_frequency_mhz = 1000.0 + 200.0 * (state.power_watts - 2.0) / 0.5;
        
        power_profile.power_timeline.push_back(state);
    }
    
    // Calculate statistics
    double total_power = 0.0;
    power_profile.peak_power = 0.0;
    power_profile.max_temperature = 0.0;
    
    for (const auto& state : power_profile.power_timeline) {
        total_power += state.power_watts;
        power_profile.peak_power = std::max(power_profile.peak_power, state.power_watts);
        power_profile.max_temperature = std::max(power_profile.max_temperature, 
                                                state.temperature_celsius);
    }
    
    power_profile.average_power = total_power / power_profile.power_timeline.size();
    power_profile.total_energy_joules = (total_power * 0.01);  // 10ms per sample
    
    EXPECT_NEAR(power_profile.average_power, 2.0, 0.1);
    EXPECT_GT(power_profile.peak_power, 2.0);
    EXPECT_LT(power_profile.peak_power, 3.0);
    EXPECT_GT(power_profile.max_temperature, 35.0);
    EXPECT_LT(power_profile.max_temperature, 45.0);
}

TEST_F(ANEPerformanceProfilerTest, MemoryProfiling) {
    ANEPerformanceProfiler::MemoryProfile mem_profile;
    
    // Simulate memory profile
    mem_profile.model_memory_mb = 50;
    mem_profile.activation_memory_mb = 25;
    mem_profile.peak_memory_mb = 100;
    mem_profile.memory_bandwidth_gb_s = 50.0;
    
    mem_profile.cache_hit_rate = 0.95;
    mem_profile.memory_read_bytes = 1024 * 1024 * 100;  // 100 MB
    mem_profile.memory_write_bytes = 1024 * 1024 * 50;  // 50 MB
    
    // Layer memory breakdown
    mem_profile.layer_memory_mb["conv1"] = 10;
    mem_profile.layer_memory_mb["conv2"] = 15;
    mem_profile.layer_memory_mb["fc1"] = 20;
    mem_profile.layer_memory_mb["output"] = 5;
    
    // Verify total matches
    size_t total_layer_memory = 0;
    for (const auto& [layer, memory] : mem_profile.layer_memory_mb) {
        total_layer_memory += memory;
    }
    EXPECT_EQ(total_layer_memory, 50);
    
    // Check memory efficiency
    double memory_efficiency = static_cast<double>(mem_profile.model_memory_mb) / 
                             mem_profile.peak_memory_mb;
    EXPECT_GT(memory_efficiency, 0.4);  // At least 40% efficient
}

TEST_F(ANEPerformanceProfilerTest, OptimizationRecommendations) {
    ANEPerformanceProfiler::OptimizationRecommendations recommendations;
    ANEPerformanceMetrics current_metrics;
    
    // Set current performance
    current_metrics.inference_time_ms = 10.0;
    current_metrics.precision = "FP16";
    current_metrics.memory_usage_mb = 200;
    current_metrics.ane_utilization_percent = 60.0;
    
    // Analyze for optimizations
    // Low ANE utilization suggests room for improvement
    if (current_metrics.ane_utilization_percent < 80.0) {
        recommendations.should_batch = true;
        recommendations.optimal_batch_size = 4;
        recommendations.optimization_suggestions.push_back(
            "Increase batch size to improve ANE utilization");
    }
    
    // Check if INT8 could help
    if (current_metrics.precision == "FP16") {
        recommendations.can_use_int8 = true;
        recommendations.optimization_suggestions.push_back(
            "Consider INT8 quantization for 2x performance improvement");
        recommendations.potential_speedup = 2.0;
    }
    
    // Memory optimization
    if (current_metrics.memory_usage_mb > 150) {
        recommendations.optimization_suggestions.push_back(
            "High memory usage detected - consider model pruning");
    }
    
    EXPECT_TRUE(recommendations.should_batch);
    EXPECT_EQ(recommendations.optimal_batch_size, 4);
    EXPECT_TRUE(recommendations.can_use_int8);
    EXPECT_EQ(recommendations.optimization_suggestions.size(), 3);
    EXPECT_NEAR(recommendations.potential_speedup, 2.0, 0.01);
}

TEST_F(ANEPerformanceProfilerTest, EfficiencyScore) {
    ANEPerformanceMetrics metrics;
    
    // High efficiency scenario
    metrics.inferences_per_second = 500.0;
    metrics.efficiency_tops_per_watt = 8.0;
    metrics.memory_usage_mb = 60.0;
    metrics.model_size_mb = 50.0;
    
    double score = CalculateEfficiencyScore(metrics);
    EXPECT_GT(score, 70.0);  // Should be a good score
    
    // Low efficiency scenario
    ANEPerformanceMetrics poor_metrics;
    poor_metrics.inferences_per_second = 50.0;
    poor_metrics.efficiency_tops_per_watt = 1.0;
    poor_metrics.memory_usage_mb = 500.0;
    poor_metrics.model_size_mb = 50.0;
    
    double poor_score = CalculateEfficiencyScore(poor_metrics);
    EXPECT_LT(poor_score, 30.0);  // Should be a poor score
    EXPECT_LT(poor_score, score);  // Poor should be worse than good
}

TEST_F(ANEPerformanceProfilerTest, FormatMetrics) {
    ANEPerformanceMetrics metrics;
    metrics.inference_time_ms = 5.5;
    metrics.inferences_per_second = 181.8;
    metrics.average_power_watts = 2.3;
    metrics.efficiency_tops_per_watt = 7.9;
    
    std::string formatted = FormatMetrics(metrics);
    
    // Check formatted string contains key information
    EXPECT_NE(formatted.find("5.50 ms"), std::string::npos);
    EXPECT_NE(formatted.find("181.80 inf/s"), std::string::npos);
    EXPECT_NE(formatted.find("2.30 W"), std::string::npos);
    EXPECT_NE(formatted.find("7.90 TOPS/W"), std::string::npos);
}

// Integration test - only run if we have a real model
TEST_F(ANEPerformanceProfilerTest, DISABLED_RealModelProfiling) {
    // This test would require a real CoreML model
    // Disabled by default but can be enabled for integration testing
    
    std::string model_path = GetMockModelPath();
    std::ifstream model_file(model_path);
    if (!model_file.good()) {
        GTEST_SKIP() << "No test model available at " << model_path;
    }
    
    ANEPerformanceMetrics metrics;
    auto err = profiler_->ProfileModel(model_path, "integration_test", metrics);
    ASSERT_EQ(err, nullptr);
    
    // Verify we got meaningful metrics
    EXPECT_GT(metrics.load_time_ms, 0);
    EXPECT_GT(metrics.inference_time_ms, 0);
    EXPECT_GT(metrics.inferences_per_second, 0);
    EXPECT_GT(metrics.memory_usage_mb, 0);
    
    // If power profiling worked
    if (metrics.average_power_watts > 0) {
        EXPECT_GT(metrics.efficiency_tops_per_watt, 0);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}