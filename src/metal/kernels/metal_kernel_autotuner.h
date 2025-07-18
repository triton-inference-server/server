// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "metal_kernel_library.h"
#include "../metal_performance_monitor.h"

namespace triton {
namespace metal {
namespace kernels {

// Auto-tuning result for a specific configuration
struct AutoTuneResult {
    KernelConfig config;
    double execution_time_ms;
    double memory_bandwidth_gb_s;
    double power_usage_watts;
    size_t flops;
    bool is_valid;
    std::string error_message;
    
    // Performance score (higher is better)
    double score() const {
        if (!is_valid) return 0.0;
        // Balance between speed, bandwidth utilization, and power efficiency
        double speed_score = 1000.0 / execution_time_ms;
        double bandwidth_score = memory_bandwidth_gb_s / 100.0; // Normalize to ~1
        double power_score = 10.0 / power_usage_watts; // Lower power is better
        return speed_score * 0.5 + bandwidth_score * 0.3 + power_score * 0.2;
    }
};

// Auto-tuning strategy
enum class TuningStrategy {
    EXHAUSTIVE,      // Try all possible configurations
    GENETIC,         // Use genetic algorithm
    BAYESIAN,        // Bayesian optimization
    GRID_SEARCH,     // Grid search with pruning
    RANDOM_SEARCH,   // Random sampling
    ADAPTIVE         // Start with grid, refine with Bayesian
};

// Auto-tuning constraints
struct TuningConstraints {
    size_t max_iterations = 100;
    std::chrono::milliseconds max_time{5000}; // 5 seconds default
    double min_improvement_threshold = 0.01; // 1% improvement
    bool profile_power = true;
    bool profile_memory = true;
    size_t warmup_iterations = 3;
    size_t timing_iterations = 10;
};

// Configuration space for tuning
struct ConfigurationSpace {
    // Thread configuration
    std::vector<std::array<size_t, 3>> threadgroup_sizes;
    std::vector<std::array<size_t, 3>> grid_sizes;
    
    // Memory configuration
    std::vector<size_t> shared_memory_sizes;
    std::vector<bool> use_simdgroup_options;
    
    // Algorithm variants
    std::vector<std::string> kernel_variants;
    
    // Precision options
    std::vector<DataType> precision_options;
    
    // Tiling parameters
    std::vector<std::unordered_map<std::string, int>> tiling_configs;
    
    // Generate all combinations
    std::vector<KernelConfig> generate_configs() const;
    
    // Generate subset based on heuristics
    std::vector<KernelConfig> generate_smart_configs(
        const MetalTensorDescriptor& input_desc) const;
};

// Kernel auto-tuner
class MetalKernelAutoTuner {
public:
    MetalKernelAutoTuner();
    ~MetalKernelAutoTuner();
    
    // Auto-tune a kernel for specific input/output tensors
    AutoTuneResult tune(
        MetalKernel* kernel,
        const std::vector<MetalTensorDescriptor>& inputs,
        const std::vector<MetalTensorDescriptor>& outputs,
        const ConfigurationSpace& search_space,
        const TuningConstraints& constraints = {},
        TuningStrategy strategy = TuningStrategy::ADAPTIVE);
    
    // Auto-tune with actual data
    AutoTuneResult tune_with_data(
        MetalKernel* kernel,
        const std::vector<id<MTLBuffer>>& input_buffers,
        const std::vector<id<MTLBuffer>>& output_buffers,
        const std::vector<MetalTensorDescriptor>& input_descs,
        const std::vector<MetalTensorDescriptor>& output_descs,
        const ConfigurationSpace& search_space,
        const TuningConstraints& constraints = {},
        TuningStrategy strategy = TuningStrategy::ADAPTIVE);
    
    // Load/save tuning cache
    void load_cache(const std::string& cache_file);
    void save_cache(const std::string& cache_file) const;
    
    // Get cached result if available
    std::optional<AutoTuneResult> get_cached_result(
        const std::string& kernel_name,
        const std::vector<MetalTensorDescriptor>& inputs,
        const std::vector<MetalTensorDescriptor>& outputs) const;
    
    // Clear cache
    void clear_cache() { cache_.clear(); }
    
    // Set performance monitor
    void set_performance_monitor(
        std::shared_ptr<MetalPerformanceMonitor> monitor) {
        performance_monitor_ = monitor;
    }
    
private:
    // Tuning strategies implementation
    AutoTuneResult tune_exhaustive(
        MetalKernel* kernel,
        const std::vector<id<MTLBuffer>>& inputs,
        const std::vector<id<MTLBuffer>>& outputs,
        const std::vector<KernelConfig>& configs,
        const TuningConstraints& constraints);
    
    AutoTuneResult tune_genetic(
        MetalKernel* kernel,
        const std::vector<id<MTLBuffer>>& inputs,
        const std::vector<id<MTLBuffer>>& outputs,
        const ConfigurationSpace& search_space,
        const TuningConstraints& constraints);
    
    AutoTuneResult tune_bayesian(
        MetalKernel* kernel,
        const std::vector<id<MTLBuffer>>& inputs,
        const std::vector<id<MTLBuffer>>& outputs,
        const ConfigurationSpace& search_space,
        const TuningConstraints& constraints);
    
    // Benchmark a single configuration
    AutoTuneResult benchmark_config(
        MetalKernel* kernel,
        const std::vector<id<MTLBuffer>>& inputs,
        const std::vector<id<MTLBuffer>>& outputs,
        const KernelConfig& config,
        const TuningConstraints& constraints);
    
    // Calculate FLOPS for a kernel based on its type and configuration
    uint64_t calculate_kernel_flops(
        MetalKernel* kernel,
        const std::vector<id<MTLBuffer>>& inputs,
        const std::vector<id<MTLBuffer>>& outputs,
        const KernelConfig& config) const;
    
    // Generate cache key
    std::string generate_cache_key(
        const std::string& kernel_name,
        const std::vector<MetalTensorDescriptor>& inputs,
        const std::vector<MetalTensorDescriptor>& outputs) const;
    
    // Genetic algorithm helpers
    struct Individual {
        KernelConfig config;
        double fitness;
    };
    
    std::vector<Individual> create_initial_population(
        const ConfigurationSpace& space, size_t population_size);
    
    Individual crossover(const Individual& parent1, const Individual& parent2);
    void mutate(Individual& individual, const ConfigurationSpace& space);
    
    // Bayesian optimization helpers
    class GaussianProcess;
    std::unique_ptr<GaussianProcess> gp_model_;
    
    KernelConfig suggest_next_config(
        const std::vector<std::pair<KernelConfig, double>>& observations,
        const ConfigurationSpace& space);
    
    // Member variables
    std::unordered_map<std::string, AutoTuneResult> cache_;
    mutable std::mutex cache_mutex_;
    std::shared_ptr<MetalPerformanceMonitor> performance_monitor_;
    
    // Metal resources
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
};

// Global auto-tuner instance
class AutoTunerManager {
public:
    static AutoTunerManager& Instance();
    
    MetalKernelAutoTuner* GetAutoTuner() { return &auto_tuner_; }
    
    // Enable/disable auto-tuning globally
    void SetEnabled(bool enabled) { enabled_ = enabled; }
    bool IsEnabled() const { return enabled_; }
    
    // Set global cache file
    void SetCacheFile(const std::string& file) { 
        cache_file_ = file;
        if (!file.empty()) {
            auto_tuner_.load_cache(file);
        }
    }
    
    // Save cache periodically
    void SaveCache() {
        if (!cache_file_.empty()) {
            auto_tuner_.save_cache(cache_file_);
        }
    }
    
private:
    AutoTunerManager() = default;
    ~AutoTunerManager() { SaveCache(); }
    
    MetalKernelAutoTuner auto_tuner_;
    bool enabled_ = true;
    std::string cache_file_;
};

// Helper to create common configuration spaces
ConfigurationSpace create_gemm_config_space(
    size_t m, size_t n, size_t k, DataType dtype);

ConfigurationSpace create_conv_config_space(
    size_t batch, size_t height, size_t width, 
    size_t channels, size_t filters, 
    size_t kernel_h, size_t kernel_w);

ConfigurationSpace create_reduction_config_space(
    const std::vector<size_t>& shape, int axis);

} // namespace kernels
} // namespace metal
} // namespace triton