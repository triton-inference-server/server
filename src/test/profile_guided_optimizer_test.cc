// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Profile-Guided Optimizer Tests

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <random>
#include <fstream>

#include "../apple/profile_guided_optimizer.h"
#include "../apple/amx_provider.h"

using namespace triton::apple;

class ProfileGuidedOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize PGO with test configuration
        ProfileGuidedOptimizer::Config config;
        config.enabled = true;
        config.auto_tune = true;
        config.warmup_iterations = 5;
        config.profile_sample_rate = 2;  // Profile every other operation
        config.exploration_probability = 0.2;
        config.profile_save_path = "./test_pgo_profile.json";
        config.persistent_profiles = false;  // Don't persist in tests
        
        auto err = ProfileGuidedOptimizer::Instance().Initialize(config);
        ASSERT_EQ(err, nullptr) << "Failed to initialize PGO";
        
        // Initialize AMX if available
        AMXProvider::Instance().Initialize();
    }
    
    void TearDown() override {
        ProfileGuidedOptimizer::Instance().Shutdown();
        
        // Clean up test files
        std::remove("./test_pgo_profile.json");
    }
    
    // Simulate execution with varying performance
    void SimulateExecution(ExecutionProfile::ExecutionTarget target, 
                          double& time_ms, double& power_watts) {
        // Base performance characteristics
        double base_time = 10.0;
        double base_power = 5.0;
        
        // Add target-specific performance
        switch (target) {
            case ExecutionProfile::CPU:
                time_ms = base_time * 1.0;
                power_watts = base_power * 1.0;
                break;
            case ExecutionProfile::AMX:
                time_ms = base_time * 0.5;  // 2x faster
                power_watts = base_power * 0.7;
                break;
            case ExecutionProfile::METAL:
                time_ms = base_time * 0.3;  // 3.3x faster
                power_watts = base_power * 2.0;  // More power
                break;
            case ExecutionProfile::ANE:
                time_ms = base_time * 0.2;  // 5x faster
                power_watts = base_power * 0.4;  // Very efficient
                break;
            default:
                time_ms = base_time;
                power_watts = base_power;
        }
        
        // Add some noise
        static std::mt19937 gen(42);
        std::normal_distribution<double> noise(0.0, 0.1);
        time_ms *= (1.0 + noise(gen));
        power_watts *= (1.0 + noise(gen));
        
        // Simulate actual execution time
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
};

TEST_F(ProfileGuidedOptimizerTest, BasicProfiling) {
    auto& pgo = ProfileGuidedOptimizer::Instance();
    
    // Create a profile for GEMM operation
    std::vector<size_t> dimensions = {1024, 1024, 1024};
    auto profile = pgo.GetProfile("GEMM", dimensions);
    
    ASSERT_NE(profile, nullptr);
    EXPECT_EQ(profile->operation_type, "GEMM");
    EXPECT_EQ(profile->dimensions, dimensions);
    EXPECT_GT(profile->compute_intensity, 0);
}

TEST_F(ProfileGuidedOptimizerTest, TargetSelection) {
    auto& pgo = ProfileGuidedOptimizer::Instance();
    
    std::vector<size_t> dimensions = {512, 512, 512};
    auto profile = pgo.GetProfile("GEMM", dimensions);
    
    // During warmup, should cycle through targets
    std::set<ExecutionProfile::ExecutionTarget> seen_targets;
    
    for (int i = 0; i < 10; ++i) {
        auto target = pgo.GetRecommendedTarget(profile);
        seen_targets.insert(target);
        
        // Simulate execution
        double time_ms, power_watts;
        SimulateExecution(target, time_ms, power_watts);
        pgo.UpdateProfile(profile, target, time_ms, power_watts);
    }
    
    // Should have tried multiple targets during warmup
    EXPECT_GT(seen_targets.size(), 1);
}

TEST_F(ProfileGuidedOptimizerTest, ConvergenceToBestTarget) {
    auto& pgo = ProfileGuidedOptimizer::Instance();
    
    std::vector<size_t> dimensions = {256, 256, 256};
    auto profile = pgo.GetProfile("GEMM", dimensions);
    
    // Run many iterations
    std::unordered_map<ExecutionProfile::ExecutionTarget, int> target_counts;
    
    for (int i = 0; i < 50; ++i) {
        auto target = pgo.GetRecommendedTarget(profile);
        target_counts[target]++;
        
        double time_ms, power_watts;
        SimulateExecution(target, time_ms, power_watts);
        pgo.UpdateProfile(profile, target, time_ms, power_watts);
    }
    
    // After warmup, should converge to best target (ANE in our simulation)
    EXPECT_EQ(profile->best_target, ExecutionProfile::ANE);
    EXPECT_GT(profile->confidence, 0.5);
    
    // Most executions should use the best target
    int best_count = target_counts[ExecutionProfile::ANE];
    int total_count = 0;
    for (const auto& [_, count] : target_counts) {
        total_count += count;
    }
    EXPECT_GT(static_cast<double>(best_count) / total_count, 0.6);
}

TEST_F(ProfileGuidedOptimizerTest, ProfilePersistence) {
    auto& pgo = ProfileGuidedOptimizer::Instance();
    
    // Create and train a profile
    std::vector<size_t> dimensions = {128, 128, 128};
    auto profile = pgo.GetProfile("Conv2D", dimensions);
    
    // Train with specific target
    for (int i = 0; i < 20; ++i) {
        double time_ms = 5.0 + (i % 3) * 0.1;
        double power_watts = 3.0;
        pgo.UpdateProfile(profile, ExecutionProfile::AMX, time_ms, power_watts);
    }
    
    // Save profiles
    auto err = pgo.SaveProfiles("./test_profile.json");
    ASSERT_EQ(err, nullptr);
    
    // Reset PGO
    pgo.ResetProfiles();
    
    // Load profiles
    err = pgo.LoadProfiles("./test_profile.json");
    ASSERT_EQ(err, nullptr);
    
    // Verify profile was restored
    auto loaded_profile = pgo.GetProfile("Conv2D", dimensions);
    ASSERT_NE(loaded_profile, nullptr);
    
    auto& metrics = loaded_profile->target_metrics[ExecutionProfile::AMX];
    EXPECT_EQ(metrics.execution_count, 20);
    EXPECT_NEAR(metrics.avg_time_ms, 5.1, 0.1);
    
    // Clean up
    std::remove("./test_profile.json");
}

TEST_F(ProfileGuidedOptimizerTest, GEMMProfiling) {
    auto& pgo = ProfileGuidedOptimizer::Instance();
    
    ProfileGuidedOptimizer::GEMMProfile gemm;
    gemm.M = 1024;
    gemm.N = 1024;
    gemm.K = 1024;
    gemm.transA = false;
    gemm.transB = false;
    gemm.alpha = 1.0f;
    gemm.beta = 0.0f;
    
    // Allocate dummy data
    std::vector<float> A(gemm.M * gemm.K, 1.0f);
    std::vector<float> B(gemm.K * gemm.N, 1.0f);
    std::vector<float> C(gemm.M * gemm.N, 0.0f);
    
    // Profile multiple times
    std::vector<ExecutionProfile::ExecutionTarget> targets;
    
    for (int i = 0; i < 10; ++i) {
        auto target = pgo.ProfileGEMM(gemm, A.data(), B.data(), C.data(), false);
        targets.push_back(target);
    }
    
    // Should have profiled different targets initially
    std::set<ExecutionProfile::ExecutionTarget> unique_targets(targets.begin(), targets.end());
    EXPECT_GT(unique_targets.size(), 1);
    
    // Get statistics
    auto stats = pgo.GetStatistics();
    EXPECT_GT(stats.total_operations, 0);
    EXPECT_GT(stats.operation_counts["GEMM"], 0);
}

TEST_F(ProfileGuidedOptimizerTest, ScopedProfiler) {
    auto& pgo = ProfileGuidedOptimizer::Instance();
    
    std::vector<size_t> dimensions = {64, 64, 64};
    auto profile = pgo.GetProfile("MatMul", dimensions);
    
    // Use scoped profiler
    {
        auto profiler = pgo.CreateScopedProfiler("MatMul", dimensions, 
                                                ExecutionProfile::CPU);
        
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
        profiler->SetPower(10.0);
    }  // Profiler destructor updates profile
    
    // Check that profile was updated
    ASSERT_GT(profile->target_metrics.size(), 0);
    auto& metrics = profile->target_metrics[ExecutionProfile::CPU];
    EXPECT_EQ(metrics.execution_count, 1);
    EXPECT_GT(metrics.avg_time_ms, 4.0);  // At least 4ms
    EXPECT_EQ(metrics.power_watts, 10.0);
}

TEST_F(ProfileGuidedOptimizerTest, ExplorationBehavior) {
    auto& pgo = ProfileGuidedOptimizer::Instance();
    
    // Set high exploration probability
    pgo.SetExplorationProbability(0.5);
    
    std::vector<size_t> dimensions = {100, 100, 100};
    auto profile = pgo.GetProfile("TestOp", dimensions);
    
    // Train heavily on one target
    for (int i = 0; i < 30; ++i) {
        pgo.UpdateProfile(profile, ExecutionProfile::CPU, 10.0, 5.0);
    }
    
    // Even with CPU being established, should still explore
    int exploration_count = 0;
    for (int i = 0; i < 100; ++i) {
        auto target = pgo.GetRecommendedTarget(profile);
        if (target != ExecutionProfile::CPU) {
            exploration_count++;
        }
    }
    
    // Should have explored other targets ~50% of the time
    EXPECT_GT(exploration_count, 30);
    EXPECT_LT(exploration_count, 70);
}

TEST_F(ProfileGuidedOptimizerTest, EfficiencyScore) {
    // Test efficiency score calculation
    double score1 = CalculateEfficiencyScore(10.0, 5.0, 1e9);  // 1 GFLOP in 10ms at 5W
    double score2 = CalculateEfficiencyScore(5.0, 10.0, 1e9);  // 1 GFLOP in 5ms at 10W
    
    // First should have better efficiency (same performance, less power)
    EXPECT_GT(score1, score2);
}

TEST_F(ProfileGuidedOptimizerTest, OperationComplexity) {
    // Test GEMM complexity
    size_t gemm_complexity = EstimateOperationComplexity("GEMM", {100, 200, 300});
    EXPECT_EQ(gemm_complexity, 2 * 100 * 200 * 300);  // 2*M*N*K
    
    // Test Conv2D complexity
    size_t conv_complexity = EstimateOperationComplexity("Conv2D", 
        {1, 28, 28, 32, 64, 3, 3});  // batch, h, w, in_c, out_c, kh, kw
    EXPECT_GT(conv_complexity, 0);
    
    // Test Transformer complexity
    size_t transformer_complexity = EstimateOperationComplexity("Transformer_BERT",
        {4, 128, 768});  // batch, seq_len, hidden_dim
    EXPECT_GT(transformer_complexity, 0);
}

TEST_F(ProfileGuidedOptimizerTest, Statistics) {
    auto& pgo = ProfileGuidedOptimizer::Instance();
    
    // Perform various operations
    for (int i = 0; i < 20; ++i) {
        auto profile1 = pgo.GetProfile("GEMM", {64, 64, 64});
        pgo.UpdateProfile(profile1, ExecutionProfile::AMX, 5.0, 3.0);
        
        auto profile2 = pgo.GetProfile("Conv2D", {1, 32, 32, 16, 32, 3, 3});
        pgo.UpdateProfile(profile2, ExecutionProfile::METAL, 8.0, 10.0);
    }
    
    auto stats = pgo.GetStatistics();
    EXPECT_EQ(stats.total_operations, 40);
    EXPECT_EQ(stats.target_usage[ExecutionProfile::AMX], 20);
    EXPECT_EQ(stats.target_usage[ExecutionProfile::METAL], 20);
    EXPECT_EQ(stats.operation_counts["GEMM"], 20);
    EXPECT_EQ(stats.operation_counts["Conv2D"], 20);
}

TEST_F(ProfileGuidedOptimizerTest, PrintSummary) {
    auto& pgo = ProfileGuidedOptimizer::Instance();
    
    // Create some profiles with varied performance
    for (int size = 64; size <= 512; size *= 2) {
        auto profile = pgo.GetProfile("GEMM", {size, size, size});
        
        // Simulate CPU being slow
        for (int i = 0; i < 5; ++i) {
            pgo.UpdateProfile(profile, ExecutionProfile::CPU, 20.0, 15.0);
        }
        
        // Simulate AMX being fast
        for (int i = 0; i < 10; ++i) {
            pgo.UpdateProfile(profile, ExecutionProfile::AMX, 8.0, 5.0);
        }
    }
    
    // This should print a nice summary
    testing::internal::CaptureStdout();
    pgo.PrintSummary();
    std::string output = testing::internal::GetCapturedStdout();
    
    // Verify summary contains expected sections
    EXPECT_NE(output.find("Profile-Guided Optimization Summary"), std::string::npos);
    EXPECT_NE(output.find("Global Statistics"), std::string::npos);
    EXPECT_NE(output.find("Target Usage"), std::string::npos);
    EXPECT_NE(output.find("Operation Distribution"), std::string::npos);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}