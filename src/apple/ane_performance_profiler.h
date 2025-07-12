// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// ANE Performance Profiler - Detailed performance analysis for Apple Neural Engine

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <CoreML/CoreML.h>
#include <MetricKit/MetricKit.h>
#endif

#include "tritonserver.h"
#include "ane_provider.h"

namespace triton {
namespace apple {

// Performance metrics for ANE operations
struct ANEPerformanceMetrics {
    // Timing metrics
    double load_time_ms = 0.0;
    double compilation_time_ms = 0.0;
    double inference_time_ms = 0.0;
    double total_time_ms = 0.0;
    
    // Throughput metrics
    double inferences_per_second = 0.0;
    double tokens_per_second = 0.0;  // For transformer models
    double images_per_second = 0.0;  // For vision models
    
    // Resource utilization
    double ane_utilization_percent = 0.0;
    double memory_usage_mb = 0.0;
    double peak_memory_mb = 0.0;
    size_t ane_compute_units_used = 0;
    
    // Power metrics
    double average_power_watts = 0.0;
    double peak_power_watts = 0.0;
    double energy_per_inference_joules = 0.0;
    double efficiency_tops_per_watt = 0.0;
    
    // Model characteristics
    size_t model_size_mb = 0;
    size_t parameter_count = 0;
    size_t flops_per_inference = 0;
    std::string precision = "FP16";  // FP16, INT8, INT4
    
    // Latency breakdown
    struct LatencyBreakdown {
        double preprocessing_ms = 0.0;
        double ane_execution_ms = 0.0;
        double postprocessing_ms = 0.0;
        double memory_transfer_ms = 0.0;
        
        std::unordered_map<std::string, double> layer_timings_ms;
    } latency_breakdown;
    
    // Calculate derived metrics
    void CalculateDerivedMetrics();
    
    // Export as JSON
    std::string ToJSON() const;
};

// ANE Performance Profiler
class ANEPerformanceProfiler {
public:
    // Configuration
    struct Config {
        bool enabled = true;
        bool detailed_profiling = false;  // Layer-by-layer profiling
        bool power_profiling = true;      // Measure power consumption
        bool memory_profiling = true;     // Track memory usage
        size_t warmup_iterations = 10;    // Warmup before profiling
        size_t profile_iterations = 100;  // Iterations for profiling
        std::string export_path = "./ane_profile_results/";
        bool export_json = true;
        bool export_csv = true;
        bool real_time_monitoring = false;  // Live performance monitoring
    };
    
    ANEPerformanceProfiler();
    ~ANEPerformanceProfiler();
    
    // Initialize profiler
    TRITONSERVER_Error* Initialize(const Config& config = Config());
    
    // ======================
    // Model Profiling
    // ======================
    
    // Profile a CoreML model comprehensively
    TRITONSERVER_Error* ProfileModel(
        const std::string& model_path,
        const std::string& model_name,
        ANEPerformanceMetrics& metrics);
    
    // Profile with specific input
    TRITONSERVER_Error* ProfileModelWithInput(
        const std::string& model_name,
        const void* input_data,
        size_t input_size,
        ANEPerformanceMetrics& metrics);
    
    // Profile transformer model
    TRITONSERVER_Error* ProfileTransformer(
        const std::string& model_name,
        size_t batch_size,
        size_t sequence_length,
        ANEPerformanceMetrics& metrics);
    
    // Profile vision model
    TRITONSERVER_Error* ProfileVisionModel(
        const std::string& model_name,
        size_t batch_size,
        size_t height,
        size_t width,
        size_t channels,
        ANEPerformanceMetrics& metrics);
    
    // ======================
    // Real-time Monitoring
    // ======================
    
    // Start real-time monitoring
    TRITONSERVER_Error* StartMonitoring(const std::string& model_name);
    
    // Stop monitoring and get results
    TRITONSERVER_Error* StopMonitoring(ANEPerformanceMetrics& metrics);
    
    // Get current metrics (during monitoring)
    ANEPerformanceMetrics GetCurrentMetrics() const;
    
    // ======================
    // Comparative Analysis
    // ======================
    
    struct ComparativeResults {
        ANEPerformanceMetrics ane_metrics;
        ANEPerformanceMetrics cpu_metrics;
        ANEPerformanceMetrics gpu_metrics;  // If Metal is available
        
        double ane_vs_cpu_speedup;
        double ane_vs_gpu_speedup;
        double ane_vs_cpu_efficiency;
        double ane_vs_gpu_efficiency;
        
        std::string recommended_target;
        std::string recommendation_reason;
    };
    
    // Compare ANE performance against CPU/GPU
    TRITONSERVER_Error* ComparativeProfile(
        const std::string& model_path,
        ComparativeResults& results);
    
    // ======================
    // Optimization Analysis
    // ======================
    
    struct OptimizationRecommendations {
        bool can_use_int8 = false;
        bool can_use_int4 = false;
        bool should_batch = false;
        size_t optimal_batch_size = 1;
        bool should_use_dynamic_shapes = false;
        
        std::vector<std::string> bottlenecks;
        std::vector<std::string> optimization_suggestions;
        
        double potential_speedup;
        double potential_power_savings;
    };
    
    // Analyze model for optimization opportunities
    TRITONSERVER_Error* AnalyzeOptimizations(
        const std::string& model_name,
        const ANEPerformanceMetrics& current_metrics,
        OptimizationRecommendations& recommendations);
    
    // ======================
    // Power Profiling
    // ======================
    
    struct PowerProfile {
        struct PowerState {
            double timestamp_ms;
            double power_watts;
            double temperature_celsius;
            double ane_frequency_mhz;
        };
        
        std::vector<PowerState> power_timeline;
        double average_power;
        double peak_power;
        double total_energy_joules;
        
        // Thermal information
        double max_temperature;
        double thermal_throttle_events;
        
        // Efficiency curves
        std::vector<std::pair<double, double>> performance_per_watt_curve;
    };
    
    // Detailed power profiling
    TRITONSERVER_Error* ProfilePower(
        const std::string& model_name,
        size_t duration_seconds,
        PowerProfile& profile);
    
    // ======================
    // Export and Visualization
    // ======================
    
    // Export profiling results
    TRITONSERVER_Error* ExportResults(
        const ANEPerformanceMetrics& metrics,
        const std::string& filename);
    
    // Export comparative results
    TRITONSERVER_Error* ExportComparison(
        const ComparativeResults& results,
        const std::string& filename);
    
    // Generate performance report
    TRITONSERVER_Error* GenerateReport(
        const std::string& model_name,
        const ANEPerformanceMetrics& metrics,
        const std::string& report_path);
    
    // Generate visualization HTML
    TRITONSERVER_Error* GenerateVisualization(
        const std::string& model_name,
        const ANEPerformanceMetrics& metrics,
        const PowerProfile& power_profile,
        const std::string& output_path);
    
    // ======================
    // Batch Profiling
    // ======================
    
    struct BatchProfilingResults {
        std::vector<size_t> batch_sizes;
        std::vector<double> latencies_ms;
        std::vector<double> throughputs;
        std::vector<double> power_consumption;
        std::vector<double> efficiency_scores;
        
        size_t optimal_batch_size;
        double optimal_throughput;
        std::string analysis;
    };
    
    // Profile different batch sizes
    TRITONSERVER_Error* ProfileBatchSizes(
        const std::string& model_name,
        const std::vector<size_t>& batch_sizes,
        BatchProfilingResults& results);
    
    // ======================
    // Memory Analysis
    // ======================
    
    struct MemoryProfile {
        size_t model_memory_mb;
        size_t activation_memory_mb;
        size_t peak_memory_mb;
        size_t memory_bandwidth_gb_s;
        
        // Memory access patterns
        double cache_hit_rate;
        size_t memory_read_bytes;
        size_t memory_write_bytes;
        
        // Per-layer memory usage
        std::unordered_map<std::string, size_t> layer_memory_mb;
    };
    
    // Profile memory usage
    TRITONSERVER_Error* ProfileMemory(
        const std::string& model_name,
        MemoryProfile& profile);
    
    // ======================
    // Debugging Support
    // ======================
    
    // Enable verbose logging
    void SetVerbose(bool verbose) { verbose_ = verbose; }
    
    // Get last error details
    std::string GetLastError() const { return last_error_; }
    
    // Validate model for ANE compatibility
    TRITONSERVER_Error* ValidateModel(
        const std::string& model_path,
        std::vector<std::string>& warnings,
        std::vector<std::string>& errors);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    Config config_;
    bool initialized_ = false;
    bool monitoring_ = false;
    bool verbose_ = false;
    std::string last_error_;
    
    // Helper methods
    TRITONSERVER_Error* MeasureInferenceTime(
        const std::string& model_name,
        const void* input_data,
        size_t input_size,
        size_t iterations,
        double& avg_time_ms,
        double& std_dev_ms);
    
    TRITONSERVER_Error* MeasurePower(
        std::function<void()> workload,
        double& avg_power_watts,
        double& peak_power_watts);
    
    TRITONSERVER_Error* MeasureMemory(
        const std::string& model_name,
        size_t& current_mb,
        size_t& peak_mb);
    
    void LogMetrics(const ANEPerformanceMetrics& metrics);
    
#ifdef __APPLE__
    // Platform-specific implementations
    TRITONSERVER_Error* GetANEUtilization(double& utilization_percent);
    TRITONSERVER_Error* GetThermalState(double& temperature_celsius);
    TRITONSERVER_Error* GetPowerMetrics(double& current_watts);
#endif
};

// ======================
// Utility Functions
// ======================

// Estimate TOPS (Tera Operations Per Second) for a model
double EstimateModelTOPS(const std::string& model_path);

// Calculate efficiency score (0-100)
double CalculateEfficiencyScore(const ANEPerformanceMetrics& metrics);

// Format metrics for display
std::string FormatMetrics(const ANEPerformanceMetrics& metrics);

// Compare two performance metrics
std::string CompareMetrics(const ANEPerformanceMetrics& metrics1,
                          const ANEPerformanceMetrics& metrics2,
                          const std::string& label1 = "Model 1",
                          const std::string& label2 = "Model 2");

} // namespace apple
} // namespace triton