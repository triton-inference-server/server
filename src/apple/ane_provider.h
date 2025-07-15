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

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __OBJC__
#ifdef __APPLE__
#include <CoreML/CoreML.h>
#endif
#else
// Forward declarations for C++ compilation units
#ifdef __APPLE__
typedef struct objc_object* id;
#endif
#endif

#include "triton/core/tritonserver.h"

namespace triton {
namespace apple {

// Forward declarations
class ANEModelOptimizer;
class ANETransformerEngine;

// ANE capability information
struct ANECapabilities {
    bool has_ane = false;
    size_t ane_version = 0;  // 1 for M1, 2 for M2, etc.
    size_t compute_units = 0;
    size_t peak_tops = 0;  // Peak INT8 operations per second
    size_t max_batch_size = 0;
    size_t max_sequence_length = 0;
    bool supports_fp16 = false;
    bool supports_int8 = false;
    bool supports_int4 = false;
    bool supports_dynamic_shapes = false;
    bool supports_transformer_engine = false;
    size_t memory_bandwidth_gb_s = 0;
};

// ANE optimization options
struct ANEOptimizationOptions {
    // Quantization
    enum class QuantizationMode {
        NONE,
        INT8_SYMMETRIC,
        INT8_ASYMMETRIC,
        INT4,
        MIXED  // Different precision per layer
    };
    QuantizationMode quantization = QuantizationMode::INT8_SYMMETRIC;
    
    // Compute units
    enum class ComputeUnits {
        ALL,         // Use all available units
        CPU_AND_ANE, // Use both CPU and ANE
        ANE_ONLY     // Force ANE only
    };
    ComputeUnits compute_units = ComputeUnits::ALL;
    
    // Optimization level
    enum class OptimizationLevel {
        O0,  // No optimization
        O1,  // Basic optimizations
        O2,  // Aggressive optimizations
        O3   // Maximum optimizations (may affect accuracy)
    };
    OptimizationLevel optimization_level = OptimizationLevel::O2;
    
    // Model-specific options
    bool enable_transformer_engine = true;
    bool enable_kernel_fusion = true;
    bool enable_memory_compression = true;
    bool enable_weight_pruning = false;
    float pruning_threshold = 0.01f;
    
    // Performance targets
    float target_latency_ms = -1.0f;  // -1 for no target
    float target_accuracy = 0.99f;
    size_t max_model_size_mb = 0;  // 0 for no limit
};

// ANE model metadata
struct ANEModelMetadata {
    std::string model_name;
    std::string model_version;
    size_t input_size;
    size_t output_size;
    size_t parameter_count;
    size_t flops;
    std::vector<std::string> supported_operations;
    std::vector<std::string> unsupported_operations;
    float estimated_latency_ms;
    float estimated_power_watts;
    bool fully_compatible;
};

// ANE performance metrics
struct ANEMetrics {
    size_t total_inferences = 0;
    double total_time_ms = 0.0;
    double min_latency_ms = std::numeric_limits<double>::max();
    double max_latency_ms = 0.0;
    double avg_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    size_t tops_achieved = 0;
    double power_usage_watts = 0.0;
    double memory_usage_mb = 0.0;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
};

// ANE provider for Neural Engine acceleration
class ANEProvider {
public:
    // Singleton instance
    static ANEProvider& Instance();
    
    // Initialize ANE support
    TRITONSERVER_Error* Initialize();
    
    // Check if ANE is available
    bool IsAvailable() const { return capabilities_.has_ane; }
    
    // Get capabilities
    const ANECapabilities& GetCapabilities() const { return capabilities_; }
    
    // ======================
    // Model Optimization
    // ======================
    
    // Optimize a model for ANE execution
    TRITONSERVER_Error* OptimizeModel(
        const std::string& model_path,
        const std::string& optimized_path,
        const ANEOptimizationOptions& options = ANEOptimizationOptions());
    
    // Analyze model compatibility with ANE
    TRITONSERVER_Error* AnalyzeModel(
        const std::string& model_path,
        ANEModelMetadata& metadata);
    
    // Convert ONNX model to CoreML for ANE
    TRITONSERVER_Error* ConvertONNXToCoreML(
        const std::string& onnx_path,
        const std::string& coreml_path,
        const ANEOptimizationOptions& options = ANEOptimizationOptions());
    
    // ======================
    // Model Execution
    // ======================
    
    // Load optimized model
    TRITONSERVER_Error* LoadModel(
        const std::string& model_path,
        const std::string& model_name);
    
    // Unload model
    TRITONSERVER_Error* UnloadModel(const std::string& model_name);
    
    // Execute inference
    TRITONSERVER_Error* Execute(
        const std::string& model_name,
        const void* input_data,
        size_t input_size,
        void* output_data,
        size_t output_size);
    
    // Batch inference
    TRITONSERVER_Error* ExecuteBatch(
        const std::string& model_name,
        const std::vector<const void*>& input_batch,
        const std::vector<size_t>& input_sizes,
        std::vector<void*>& output_batch,
        const std::vector<size_t>& output_sizes);
    
    // ======================
    // Performance Management
    // ======================
    
    // Get performance metrics
    ANEMetrics GetMetrics(const std::string& model_name = "") const;
    void ResetMetrics(const std::string& model_name = "");
    
    // Enable/disable ANE
    void SetEnabled(bool enabled) { enabled_ = enabled; }
    bool IsEnabled() const { return enabled_ && capabilities_.has_ane; }
    
    // Set power mode
    enum class PowerMode {
        HIGH_PERFORMANCE,  // Maximum performance
        BALANCED,         // Balance performance and efficiency
        LOW_POWER        // Maximize battery life
    };
    void SetPowerMode(PowerMode mode) { power_mode_ = mode; }
    
    // Cache management
    void ClearCache();
    void SetCacheSize(size_t size_mb) { cache_size_mb_ = size_mb; }
    
    // ======================
    // Advanced Features
    // ======================
    
    // Get the model optimizer
    ANEModelOptimizer* GetOptimizer() { return optimizer_.get(); }
    
    // Get the transformer engine
    class ANETransformerEngine* GetTransformerEngine() { return transformer_engine_.get(); }
    
    // Profile model on ANE
    TRITONSERVER_Error* ProfileModel(
        const std::string& model_name,
        const void* sample_input,
        size_t input_size,
        int num_iterations = 100);
    
private:
    ANEProvider();
    ~ANEProvider();
    
    // Detect ANE capabilities
    void DetectCapabilities();
    
    // CoreML model wrapper
    struct CoreMLModel {
#ifdef __APPLE__
        MLModel* model = nullptr;
        MLModelConfiguration* config = nullptr;
#endif
        ANEModelMetadata metadata;
        ANEMetrics metrics;
        std::chrono::steady_clock::time_point last_used;
    };
    
    // Update metrics after inference
    void UpdateMetrics(
        const std::string& model_name,
        double inference_time_ms);
    
    // Member variables
    ANECapabilities capabilities_;
    std::atomic<bool> enabled_{true};
    PowerMode power_mode_ = PowerMode::BALANCED;
    size_t cache_size_mb_ = 256;
    
    // Model management
    std::unordered_map<std::string, std::unique_ptr<CoreMLModel>> models_;
    mutable std::mutex models_mutex_;
    
    // Components
    std::unique_ptr<ANEModelOptimizer> optimizer_;
    std::unique_ptr<class ANETransformerEngine> transformer_engine_;
    
    // Global metrics
    mutable std::mutex metrics_mutex_;
    ANEMetrics global_metrics_;
    
    // Prevent copying
    ANEProvider(const ANEProvider&) = delete;
    ANEProvider& operator=(const ANEProvider&) = delete;
};

// ANE model optimizer
class ANEModelOptimizer {
public:
    ANEModelOptimizer();
    ~ANEModelOptimizer();
    
    // Optimization passes
    TRITONSERVER_Error* OptimizeGraph(
        const std::string& input_path,
        const std::string& output_path,
        const ANEOptimizationOptions& options);
    
    // Quantization
    TRITONSERVER_Error* QuantizeModel(
        const std::string& model_path,
        const std::string& quantized_path,
        ANEOptimizationOptions::QuantizationMode mode,
        const std::string& calibration_data_path = "");
    
    // Operation fusion
    TRITONSERVER_Error* FuseOperations(
        const std::string& model_path,
        const std::string& fused_path);
    
    // Model partitioning for hybrid execution
    struct ModelPartition {
        std::vector<std::string> ane_ops;
        std::vector<std::string> cpu_ops;
        std::vector<std::string> gpu_ops;
        std::vector<std::pair<std::string, std::string>> transfer_edges;
    };
    
    ModelPartition PartitionModel(
        const std::string& model_path,
        const ANEOptimizationOptions& options);
    
    // Weight pruning
    TRITONSERVER_Error* PruneWeights(
        const std::string& model_path,
        const std::string& pruned_path,
        float sparsity_target);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Helper functions
bool DetectANESupport();
std::string GetANEInfoString();
size_t GetANEPeakTOPS();

// ANE backend factory
class ANEBackendFactory {
public:
    static TRITONSERVER_Error* CreateBackend(
        void** backend,
        const char* name,
        const uint64_t version);
};

} // namespace apple
} // namespace triton