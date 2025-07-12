// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Example demonstrating the Metal kernel auto-tuner

#include <iostream>
#include <memory>
#include "metal_kernel_autotuner.h"
#include "gemm_kernel.h"

using namespace triton::metal::kernels;

void demo_gemm_autotuning() {
    std::cout << "=== GEMM Kernel Auto-Tuning Demo ===" << std::endl;
    
    // Create GEMM kernel
    auto gemm_kernel = std::make_unique<GEMMKernel>();
    
    // Define problem size
    size_t M = 1024;
    size_t N = 1024;
    size_t K = 1024;
    
    // Create tensor descriptors
    std::vector<MetalTensorDescriptor> inputs = {
        MetalTensorDescriptor({M, K}, DataType::FLOAT32),  // Matrix A
        MetalTensorDescriptor({K, N}, DataType::FLOAT32)   // Matrix B
    };
    
    std::vector<MetalTensorDescriptor> outputs = {
        MetalTensorDescriptor({M, N}, DataType::FLOAT32)   // Matrix C
    };
    
    // Create configuration space
    auto config_space = create_gemm_config_space(M, N, K, DataType::FLOAT32);
    
    // Set tuning constraints
    TuningConstraints constraints;
    constraints.max_iterations = 50;
    constraints.max_time = std::chrono::seconds(10);
    constraints.warmup_iterations = 3;
    constraints.timing_iterations = 10;
    constraints.profile_power = true;
    constraints.profile_memory = true;
    
    // Get auto-tuner instance
    auto& tuner_manager = AutoTunerManager::Instance();
    auto* auto_tuner = tuner_manager.GetAutoTuner();
    
    // Enable cache
    tuner_manager.SetCacheFile("gemm_tuning_cache.json");
    
    std::cout << "Starting auto-tuning for GEMM " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Max iterations: " << constraints.max_iterations << std::endl;
    std::cout << "Max time: " << constraints.max_time.count() << "ms" << std::endl;
    
    // Run auto-tuning
    auto result = auto_tuner->tune(
        gemm_kernel.get(),
        inputs,
        outputs,
        config_space,
        constraints,
        TuningStrategy::ADAPTIVE
    );
    
    if (result.is_valid) {
        std::cout << "\nBest configuration found:" << std::endl;
        std::cout << "  Thread group: " 
                  << result.config.threadgroup_size[0] << "x"
                  << result.config.threadgroup_size[1] << "x"
                  << result.config.threadgroup_size[2] << std::endl;
        std::cout << "  Shared memory: " << result.config.shared_memory_size << " bytes" << std::endl;
        std::cout << "  Use SIMD group: " << (result.config.use_simdgroup ? "Yes" : "No") << std::endl;
        std::cout << "  Execution time: " << result.execution_time_ms << " ms" << std::endl;
        std::cout << "  Memory bandwidth: " << result.memory_bandwidth_gb_s << " GB/s" << std::endl;
        std::cout << "  Power usage: " << result.power_usage_watts << " W" << std::endl;
        std::cout << "  Performance score: " << result.score() << std::endl;
        
        // Calculate GFLOPS
        double flops = 2.0 * M * N * K;  // 2 ops per multiply-add
        double gflops = (flops / 1e9) / (result.execution_time_ms / 1000.0);
        std::cout << "  Estimated GFLOPS: " << gflops << std::endl;
    } else {
        std::cout << "Auto-tuning failed: " << result.error_message << std::endl;
    }
    
    // Save cache
    tuner_manager.SaveCache();
}

void demo_conv_autotuning() {
    std::cout << "\n=== Convolution Kernel Auto-Tuning Demo ===" << std::endl;
    
    // Define convolution problem
    size_t batch = 32;
    size_t height = 224;
    size_t width = 224;
    size_t in_channels = 64;
    size_t out_channels = 128;
    size_t kernel_h = 3;
    size_t kernel_w = 3;
    
    // Create configuration space
    auto config_space = create_conv_config_space(
        batch, height, width, in_channels, out_channels, kernel_h, kernel_w);
    
    std::cout << "Convolution configuration space created:" << std::endl;
    std::cout << "  Input: " << batch << "x" << in_channels << "x" << height << "x" << width << std::endl;
    std::cout << "  Kernel: " << out_channels << "x" << in_channels << "x" << kernel_h << "x" << kernel_w << std::endl;
    std::cout << "  Algorithm variants: " << config_space.kernel_variants.size() << std::endl;
    std::cout << "  Thread configurations: " << config_space.threadgroup_sizes.size() << std::endl;
    
    // Note: Actual convolution kernel implementation would be needed here
}

void demo_cached_results() {
    std::cout << "\n=== Cached Results Demo ===" << std::endl;
    
    auto& tuner_manager = AutoTunerManager::Instance();
    auto* auto_tuner = tuner_manager.GetAutoTuner();
    
    // Load existing cache
    auto_tuner->load_cache("gemm_tuning_cache.json");
    
    // Check if we have cached results
    std::vector<MetalTensorDescriptor> inputs = {
        MetalTensorDescriptor({1024, 1024}, DataType::FLOAT32),
        MetalTensorDescriptor({1024, 1024}, DataType::FLOAT32)
    };
    
    std::vector<MetalTensorDescriptor> outputs = {
        MetalTensorDescriptor({1024, 1024}, DataType::FLOAT32)
    };
    
    auto cached_result = auto_tuner->get_cached_result("gemm", inputs, outputs);
    
    if (cached_result.has_value()) {
        std::cout << "Found cached result for GEMM 1024x1024x1024:" << std::endl;
        std::cout << "  Execution time: " << cached_result->execution_time_ms << " ms" << std::endl;
        std::cout << "  Performance score: " << cached_result->score() << std::endl;
    } else {
        std::cout << "No cached result found for GEMM 1024x1024x1024" << std::endl;
    }
}

int main() {
    std::cout << "Metal Kernel Auto-Tuner Example" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // Demo GEMM auto-tuning
    demo_gemm_autotuning();
    
    // Demo convolution configuration space
    demo_conv_autotuning();
    
    // Demo cached results
    demo_cached_results();
    
    return 0;
}