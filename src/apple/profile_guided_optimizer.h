// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Profile-Guided Optimization (PGO) for Apple Silicon

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tritonserver.h"

namespace triton {
namespace apple {

// Forward declarations
class AMXProvider;
class MetalBackend;
class ANEProvider;

// Execution profile for an operation
struct ExecutionProfile {
    enum ExecutionTarget {
        CPU,
        AMX,
        METAL,
        ANE,
        HYBRID_AMX_METAL,
        HYBRID_AMX_ANE,
        HYBRID_METAL_ANE
    };
    
    struct TargetMetrics {
        size_t execution_count = 0;
        double total_time_ms = 0.0;
        double min_time_ms = std::numeric_limits<double>::max();
        double max_time_ms = 0.0;
        double avg_time_ms = 0.0;
        double variance_ms = 0.0;
        double power_watts = 0.0;
        double efficiency_score = 0.0;  // Combined metric
        
        void Update(double time_ms, double power = 0.0);
        double GetScore() const;  // Calculate efficiency score
    };
    
    // Operation characteristics
    std::string operation_type;  // GEMM, Conv2D, etc.
    size_t input_size;
    size_t output_size;
    size_t compute_intensity;  // FLOPs
    std::vector<size_t> dimensions;
    
    // Performance data per target
    std::unordered_map<ExecutionTarget, TargetMetrics> target_metrics;
    
    // Best target based on current data
    ExecutionTarget best_target = CPU;
    double confidence = 0.0;  // 0-1 confidence in best target
    
    // Update best target based on metrics
    void UpdateBestTarget();
};

// Profile-Guided Optimizer
class ProfileGuidedOptimizer {
public:
    // Configuration
    struct Config {
        bool enabled = true;
        bool auto_tune = true;  // Automatically try different targets
        size_t warmup_iterations = 10;  // Iterations before making decisions
        size_t profile_sample_rate = 100;  // Profile every N executions
        double exploration_probability = 0.1;  // Probability of trying non-optimal
        std::string profile_save_path = "./triton_pgo_profile.bin";
        bool persistent_profiles = true;  // Save/load profiles across runs
        
        // Thresholds
        double switch_threshold = 1.2;  // Switch if >20% better
        double confidence_threshold = 0.8;  // Minimum confidence to switch
    };
    
    // Singleton instance
    static ProfileGuidedOptimizer& Instance();
    
    ~ProfileGuidedOptimizer();
    
    // Initialize with configuration
    TRITONSERVER_Error* Initialize(const Config& config = Config());
    
    // Shutdown and save profiles
    void Shutdown();
    
    // ======================
    // Profile Management
    // ======================
    
    // Get or create profile for an operation
    ExecutionProfile* GetProfile(
        const std::string& operation_type,
        const std::vector<size_t>& dimensions);
    
    // Update profile with execution results
    void UpdateProfile(
        ExecutionProfile* profile,
        ExecutionProfile::ExecutionTarget target,
        double execution_time_ms,
        double power_watts = 0.0);
    
    // Get recommended execution target
    ExecutionProfile::ExecutionTarget GetRecommendedTarget(
        const ExecutionProfile* profile) const;
    
    // ======================
    // Operation Profiling
    // ======================
    
    // Profile GEMM operation
    struct GEMMProfile {
        size_t M, N, K;
        bool transA, transB;
        float alpha, beta;
    };
    
    ExecutionProfile::ExecutionTarget ProfileGEMM(
        const GEMMProfile& gemm,
        const float* A, const float* B, float* C,
        bool force_profile = false);
    
    // Profile Convolution operation  
    struct ConvProfile {
        size_t batch_size;
        size_t height, width;
        size_t in_channels, out_channels;
        size_t kernel_h, kernel_w;
        size_t stride_h, stride_w;
        size_t pad_h, pad_w;
    };
    
    ExecutionProfile::ExecutionTarget ProfileConvolution(
        const ConvProfile& conv,
        const float* input,
        const float* kernel,
        float* output,
        bool force_profile = false);
    
    // Profile Transformer operation
    struct TransformerProfile {
        size_t batch_size;
        size_t seq_length;
        size_t hidden_dim;
        size_t num_heads;
        std::string model_type;  // BERT, GPT, etc.
    };
    
    ExecutionProfile::ExecutionTarget ProfileTransformer(
        const TransformerProfile& transformer,
        const void* input,
        void* output,
        bool force_profile = false);
    
    // ======================
    // Auto-tuning
    // ======================
    
    // Auto-tune a model (try different configurations)
    TRITONSERVER_Error* AutoTuneModel(
        const std::string& model_name,
        const std::string& model_path,
        size_t num_iterations = 100);
    
    // Get auto-tuning recommendations
    struct TuningRecommendation {
        std::string operation;
        ExecutionProfile::ExecutionTarget recommended_target;
        double expected_speedup;
        double confidence;
        std::string rationale;
    };
    
    std::vector<TuningRecommendation> GetTuningRecommendations(
        const std::string& model_name) const;
    
    // ======================
    // Profile Persistence
    // ======================
    
    // Save profiles to disk
    TRITONSERVER_Error* SaveProfiles(const std::string& path = "");
    
    // Load profiles from disk
    TRITONSERVER_Error* LoadProfiles(const std::string& path = "");
    
    // Export profiles as JSON for analysis
    TRITONSERVER_Error* ExportProfilesJSON(const std::string& path);
    
    // ======================
    // Runtime Adaptation
    // ======================
    
    // Enable/disable runtime adaptation
    void SetAdaptiveMode(bool enabled) { adaptive_mode_ = enabled; }
    
    // Set exploration probability (0-1)
    void SetExplorationProbability(double prob) { 
        config_.exploration_probability = std::max(0.0, std::min(1.0, prob));
    }
    
    // Force re-profiling of all operations
    void ResetProfiles();
    
    // ======================
    // Statistics and Monitoring
    // ======================
    
    struct GlobalStats {
        size_t total_operations = 0;
        size_t profiled_operations = 0;
        size_t target_switches = 0;
        double total_time_saved_ms = 0.0;
        double total_power_saved_watts = 0.0;
        
        std::unordered_map<ExecutionProfile::ExecutionTarget, size_t> target_usage;
        std::unordered_map<std::string, size_t> operation_counts;
    };
    
    GlobalStats GetStatistics() const;
    
    // Print profile summary
    void PrintSummary() const;
    
    // ======================
    // Integration Helpers
    // ======================
    
    // Scoped profiler for automatic timing
    class ScopedProfiler {
    public:
        ScopedProfiler(ExecutionProfile* profile, 
                      ExecutionProfile::ExecutionTarget target);
        ~ScopedProfiler();
        
        void SetPower(double watts) { power_watts_ = watts; }
        
    private:
        ExecutionProfile* profile_;
        ExecutionProfile::ExecutionTarget target_;
        std::chrono::steady_clock::time_point start_time_;
        double power_watts_ = 0.0;
    };
    
    // Create scoped profiler
    std::unique_ptr<ScopedProfiler> CreateScopedProfiler(
        const std::string& operation_type,
        const std::vector<size_t>& dimensions,
        ExecutionProfile::ExecutionTarget target);
    
private:
    ProfileGuidedOptimizer();
    ProfileGuidedOptimizer(const ProfileGuidedOptimizer&) = delete;
    ProfileGuidedOptimizer& operator=(const ProfileGuidedOptimizer&) = delete;
    
    // Configuration
    Config config_;
    bool initialized_ = false;
    bool adaptive_mode_ = true;
    
    // Profile storage
    using ProfileKey = std::string;
    std::unordered_map<ProfileKey, std::unique_ptr<ExecutionProfile>> profiles_;
    mutable std::mutex profiles_mutex_;
    
    // Statistics
    mutable GlobalStats stats_;
    mutable std::mutex stats_mutex_;
    
    // Random number generation for exploration
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};
    
    // Helper methods
    ProfileKey GenerateKey(const std::string& op_type, 
                          const std::vector<size_t>& dims) const;
    
    bool ShouldExplore() const;
    
    ExecutionProfile::ExecutionTarget SelectExplorationTarget(
        const ExecutionProfile* profile) const;
    
    void UpdateStatistics(ExecutionProfile::ExecutionTarget target);
    
    // Profile execution on different targets
    void ProfileOnTarget(ExecutionProfile* profile,
                        ExecutionProfile::ExecutionTarget target,
                        std::function<void()> execute_fn);
    
    // Estimate power consumption
    double EstimatePower(ExecutionProfile::ExecutionTarget target,
                        size_t compute_intensity) const;
};

// ======================
// Utility Functions
// ======================

// Convert execution target to string
std::string ExecutionTargetToString(ExecutionProfile::ExecutionTarget target);

// Parse execution target from string
ExecutionProfile::ExecutionTarget StringToExecutionTarget(const std::string& str);

// Calculate efficiency score (performance per watt)
double CalculateEfficiencyScore(double time_ms, double power_watts, 
                               size_t compute_intensity);

// Estimate operation complexity
size_t EstimateOperationComplexity(const std::string& op_type,
                                   const std::vector<size_t>& dimensions);

} // namespace apple
} // namespace triton