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

#include "metal_kernel_autotuner.h"

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include <json/json.h>

#import <Metal/Metal.h>

namespace triton {
namespace metal {
namespace kernels {

// Configuration space implementation
std::vector<KernelConfig> ConfigurationSpace::generate_configs() const {
    std::vector<KernelConfig> configs;
    
    // Generate all combinations (can be very large!)
    for (const auto& threadgroup : threadgroup_sizes) {
        for (const auto& grid : grid_sizes) {
            for (size_t shared_mem : shared_memory_sizes) {
                for (bool use_simdgroup : use_simdgroup_options) {
                    for (const auto& variant : kernel_variants) {
                        for (DataType precision : precision_options) {
                            for (const auto& tiling : tiling_configs) {
                                KernelConfig config;
                                config.threadgroup_size = threadgroup;
                                config.grid_size = grid;
                                config.shared_memory_size = shared_mem;
                                config.use_simdgroup = use_simdgroup;
                                config.kernel_variant = variant;
                                config.precision = precision;
                                config.int_params = tiling;
                                configs.push_back(config);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return configs;
}

std::vector<KernelConfig> ConfigurationSpace::generate_smart_configs(
    const MetalTensorDescriptor& input_desc) const {
    std::vector<KernelConfig> configs;
    
    // Use heuristics to generate a smaller set of promising configurations
    size_t total_size = input_desc.size();
    
    // Select thread group sizes based on tensor size
    std::vector<std::array<size_t, 3>> smart_threadgroups;
    if (total_size < 1024) {
        smart_threadgroups = {{32, 1, 1}, {64, 1, 1}};
    } else if (total_size < 1024 * 1024) {
        smart_threadgroups = {{256, 1, 1}, {16, 16, 1}, {32, 8, 1}};
    } else {
        smart_threadgroups = {{256, 1, 1}, {16, 16, 1}, {32, 32, 1}};
    }
    
    // Generate configs with smart defaults
    for (const auto& threadgroup : smart_threadgroups) {
        KernelConfig config;
        config.threadgroup_size = threadgroup;
        
        // Calculate grid size based on total work
        config.grid_size = {
            (total_size + threadgroup[0] - 1) / threadgroup[0],
            1, 1
        };
        
        // Use simdgroup for larger operations
        config.use_simdgroup = total_size > 4096;
        
        // Default shared memory based on operation size
        config.shared_memory_size = std::min<size_t>(16384, total_size * 4);
        
        // Add precision variants
        for (DataType precision : precision_options) {
            config.precision = precision;
            configs.push_back(config);
        }
    }
    
    return configs;
}

// MetalKernelAutoTuner implementation
MetalKernelAutoTuner::MetalKernelAutoTuner() {
    device_ = MTLCreateSystemDefaultDevice();
    if (device_) {
        command_queue_ = [device_ newCommandQueue];
    }
}

MetalKernelAutoTuner::~MetalKernelAutoTuner() {
    command_queue_ = nil;
    device_ = nil;
}

AutoTuneResult MetalKernelAutoTuner::tune(
    MetalKernel* kernel,
    const std::vector<MetalTensorDescriptor>& inputs,
    const std::vector<MetalTensorDescriptor>& outputs,
    const ConfigurationSpace& search_space,
    const TuningConstraints& constraints,
    TuningStrategy strategy) {
    
    // Check cache first
    std::string cache_key = generate_cache_key(kernel->name(), inputs, outputs);
    auto cached = get_cached_result(kernel->name(), inputs, outputs);
    if (cached.has_value()) {
        return cached.value();
    }
    
    // Create temporary buffers for tuning
    std::vector<id<MTLBuffer>> input_buffers;
    std::vector<id<MTLBuffer>> output_buffers;
    
    for (const auto& input : inputs) {
        size_t bytes = input.bytes();
        id<MTLBuffer> buffer = [device_ newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
        if (!buffer) {
            return {KernelConfig(), 0, 0, 0, 0, false, "Failed to allocate input buffer"};
        }
        
        // Initialize with random data
        float* data = (float*)buffer.contents;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < bytes / sizeof(float); ++i) {
            data[i] = dist(gen);
        }
        
        input_buffers.push_back(buffer);
    }
    
    for (const auto& output : outputs) {
        size_t bytes = output.bytes();
        id<MTLBuffer> buffer = [device_ newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
        if (!buffer) {
            return {KernelConfig(), 0, 0, 0, 0, false, "Failed to allocate output buffer"};
        }
        output_buffers.push_back(buffer);
    }
    
    // Run tuning
    auto result = tune_with_data(kernel, input_buffers, output_buffers,
                                inputs, outputs, search_space, constraints, strategy);
    
    // Cache result
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_[cache_key] = result;
    }
    
    return result;
}

AutoTuneResult MetalKernelAutoTuner::tune_with_data(
    MetalKernel* kernel,
    const std::vector<id<MTLBuffer>>& input_buffers,
    const std::vector<id<MTLBuffer>>& output_buffers,
    const std::vector<MetalTensorDescriptor>& input_descs,
    const std::vector<MetalTensorDescriptor>& output_descs,
    const ConfigurationSpace& search_space,
    const TuningConstraints& constraints,
    TuningStrategy strategy) {
    
    AutoTuneResult best_result;
    best_result.is_valid = false;
    
    switch (strategy) {
        case TuningStrategy::EXHAUSTIVE: {
            auto configs = search_space.generate_configs();
            best_result = tune_exhaustive(kernel, input_buffers, output_buffers,
                                        configs, constraints);
            break;
        }
        
        case TuningStrategy::GRID_SEARCH: {
            auto configs = search_space.generate_smart_configs(input_descs[0]);
            best_result = tune_exhaustive(kernel, input_buffers, output_buffers,
                                        configs, constraints);
            break;
        }
        
        case TuningStrategy::GENETIC: {
            best_result = tune_genetic(kernel, input_buffers, output_buffers,
                                     search_space, constraints);
            break;
        }
        
        case TuningStrategy::BAYESIAN: {
            best_result = tune_bayesian(kernel, input_buffers, output_buffers,
                                      search_space, constraints);
            break;
        }
        
        case TuningStrategy::RANDOM_SEARCH: {
            auto all_configs = search_space.generate_configs();
            std::random_device rd;
            std::mt19937 gen(rd());
            
            // Sample subset of configurations
            size_t sample_size = std::min(constraints.max_iterations, all_configs.size());
            std::shuffle(all_configs.begin(), all_configs.end(), gen);
            all_configs.resize(sample_size);
            
            best_result = tune_exhaustive(kernel, input_buffers, output_buffers,
                                        all_configs, constraints);
            break;
        }
        
        case TuningStrategy::ADAPTIVE:
        default: {
            // Start with smart grid search
            auto configs = search_space.generate_smart_configs(input_descs[0]);
            best_result = tune_exhaustive(kernel, input_buffers, output_buffers,
                                        configs, constraints);
            
            // If time permits, refine with Bayesian optimization
            auto elapsed = std::chrono::steady_clock::now();
            if (constraints.max_iterations > configs.size()) {
                // Use remaining iterations for Bayesian optimization
                TuningConstraints refined_constraints = constraints;
                refined_constraints.max_iterations -= configs.size();
                
                auto bayesian_result = tune_bayesian(kernel, input_buffers, output_buffers,
                                                   search_space, refined_constraints);
                if (bayesian_result.score() > best_result.score()) {
                    best_result = bayesian_result;
                }
            }
            break;
        }
    }
    
    return best_result;
}

AutoTuneResult MetalKernelAutoTuner::tune_exhaustive(
    MetalKernel* kernel,
    const std::vector<id<MTLBuffer>>& inputs,
    const std::vector<id<MTLBuffer>>& outputs,
    const std::vector<KernelConfig>& configs,
    const TuningConstraints& constraints) {
    
    AutoTuneResult best_result;
    best_result.is_valid = false;
    double best_score = 0.0;
    
    auto start_time = std::chrono::steady_clock::now();
    size_t iterations = 0;
    
    for (const auto& config : configs) {
        // Check time limit
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time);
        if (elapsed > constraints.max_time) {
            break;
        }
        
        // Check iteration limit
        if (iterations >= constraints.max_iterations) {
            break;
        }
        
        // Benchmark this configuration
        auto result = benchmark_config(kernel, inputs, outputs, config, constraints);
        
        if (result.is_valid && result.score() > best_score) {
            best_result = result;
            best_score = result.score();
        }
        
        iterations++;
    }
    
    return best_result;
}

uint64_t MetalKernelAutoTuner::calculate_kernel_flops(
    MetalKernel* kernel,
    const std::vector<id<MTLBuffer>>& inputs,
    const std::vector<id<MTLBuffer>>& outputs,
    const KernelConfig& config) const {
    
    uint64_t flops = 0;
    std::string kernel_name = kernel->name();
    
    // GEMM kernels: 2*M*N*K operations
    if (kernel_name.find("gemm") != std::string::npos ||
        kernel_name.find("matmul") != std::string::npos) {
        // Assume inputs are matrices A[M,K] and B[K,N]
        if (inputs.size() >= 2) {
            // Extract dimensions from buffer sizes and data type
            size_t bytes_a = [inputs[0] length];
            size_t bytes_b = [inputs[1] length];
            size_t bytes_c = outputs.empty() ? 0 : [outputs[0] length];
            
            // Assuming float32 for now
            size_t elem_size = sizeof(float);
            if (config.precision == DataType::FLOAT16) {
                elem_size = sizeof(uint16_t);
            }
            
            // Estimate dimensions based on common patterns
            // This is a heuristic - real kernels should provide exact dimensions
            size_t total_elems_a = bytes_a / elem_size;
            size_t total_elems_b = bytes_b / elem_size;
            size_t total_elems_c = bytes_c / elem_size;
            
            // For square matrices: M=N=K=sqrt(total_elems)
            if (total_elems_a == total_elems_b && total_elems_a > 0) {
                size_t dim = static_cast<size_t>(std::sqrt(total_elems_a));
                flops = 2ULL * dim * dim * dim;  // 2*M*N*K for GEMM
            } else if (total_elems_c > 0) {
                // Try to deduce from output size
                size_t mn = total_elems_c;
                size_t k = total_elems_a / std::sqrt(mn);  // Rough estimate
                flops = 2ULL * mn * k;
            }
        }
    }
    // Convolution kernels
    else if (kernel_name.find("conv") != std::string::npos) {
        // Conv2D: 2 * batch * out_h * out_w * in_channels * out_channels * kernel_h * kernel_w
        // This is a simplified estimate
        if (inputs.size() >= 2) {
            size_t input_bytes = [inputs[0] length];
            size_t weight_bytes = [inputs[1] length];
            
            size_t elem_size = sizeof(float);
            if (config.precision == DataType::FLOAT16) {
                elem_size = sizeof(uint16_t);
            }
            
            size_t input_elems = input_bytes / elem_size;
            size_t weight_elems = weight_bytes / elem_size;
            
            // Rough estimate: operations proportional to input size * weight size
            flops = 2ULL * input_elems * weight_elems / 1000;  // Divided by 1000 as a rough normalization
        }
    }
    // Element-wise operations
    else if (kernel_name.find("add") != std::string::npos ||
             kernel_name.find("mul") != std::string::npos ||
             kernel_name.find("relu") != std::string::npos ||
             kernel_name.find("sigmoid") != std::string::npos) {
        // One operation per element
        if (!outputs.empty()) {
            size_t output_bytes = [outputs[0] length];
            size_t elem_size = sizeof(float);
            if (config.precision == DataType::FLOAT16) {
                elem_size = sizeof(uint16_t);
            }
            flops = output_bytes / elem_size;
            
            // Sigmoid/tanh have more operations per element
            if (kernel_name.find("sigmoid") != std::string::npos ||
                kernel_name.find("tanh") != std::string::npos) {
                flops *= 10;  // Rough estimate for transcendental functions
            }
        }
    }
    // Reduction operations
    else if (kernel_name.find("reduce") != std::string::npos ||
             kernel_name.find("sum") != std::string::npos ||
             kernel_name.find("mean") != std::string::npos) {
        if (!inputs.empty()) {
            size_t input_bytes = [inputs[0] length];
            size_t elem_size = sizeof(float);
            if (config.precision == DataType::FLOAT16) {
                elem_size = sizeof(uint16_t);
            }
            // N-1 operations for N elements
            flops = (input_bytes / elem_size) - 1;
        }
    }
    // Batch normalization
    else if (kernel_name.find("batch_norm") != std::string::npos ||
             kernel_name.find("layer_norm") != std::string::npos) {
        if (!inputs.empty()) {
            size_t input_bytes = [inputs[0] length];
            size_t elem_size = sizeof(float);
            if (config.precision == DataType::FLOAT16) {
                elem_size = sizeof(uint16_t);
            }
            // Roughly 5 operations per element (subtract mean, divide by std, scale, shift)
            flops = 5ULL * (input_bytes / elem_size);
        }
    }
    // Softmax
    else if (kernel_name.find("softmax") != std::string::npos) {
        if (!inputs.empty()) {
            size_t input_bytes = [inputs[0] length];
            size_t elem_size = sizeof(float);
            if (config.precision == DataType::FLOAT16) {
                elem_size = sizeof(uint16_t);
            }
            size_t num_elems = input_bytes / elem_size;
            // exp + sum + division for each element
            flops = 3ULL * num_elems;
        }
    }
    // Attention mechanisms
    else if (kernel_name.find("attention") != std::string::npos) {
        if (!inputs.empty()) {
            size_t input_bytes = [inputs[0] length];
            size_t elem_size = sizeof(float);
            if (config.precision == DataType::FLOAT16) {
                elem_size = sizeof(uint16_t);
            }
            size_t seq_len = static_cast<size_t>(std::sqrt(input_bytes / elem_size));
            // Rough estimate: O(seq_len^2 * hidden_dim)
            flops = 4ULL * seq_len * seq_len * (input_bytes / elem_size) / seq_len;
        }
    }
    
    return flops;
}

AutoTuneResult MetalKernelAutoTuner::benchmark_config(
    MetalKernel* kernel,
    const std::vector<id<MTLBuffer>>& inputs,
    const std::vector<id<MTLBuffer>>& outputs,
    const KernelConfig& config,
    const TuningConstraints& constraints) {
    
    AutoTuneResult result;
    result.config = config;
    result.is_valid = false;
    
    try {
        // Warmup iterations
        for (size_t i = 0; i < constraints.warmup_iterations; ++i) {
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            kernel->encode(encoder, inputs, outputs, config);
            
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            
            if (command_buffer.error) {
                result.error_message = "Kernel execution failed during warmup";
                return result;
            }
        }
        
        // Timing iterations
        double total_time = 0.0;
        double total_bandwidth = 0.0;
        double total_power = 0.0;
        
        for (size_t i = 0; i < constraints.timing_iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            kernel->encode(encoder, inputs, outputs, config);
            
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            
            auto end = std::chrono::high_resolution_clock::now();
            
            if (command_buffer.error) {
                result.error_message = "Kernel execution failed during timing";
                return result;
            }
            
            // Calculate metrics
            std::chrono::duration<double, std::milli> elapsed = end - start;
            total_time += elapsed.count();
            
            // Get performance metrics if monitor is available
            if (performance_monitor_ && constraints.profile_memory) {
                auto metrics = performance_monitor_->GetCurrentMetrics();
                total_bandwidth += metrics.memory_bandwidth_gb_s;
                total_power += metrics.power_usage_watts;
            }
        }
        
        // Average metrics
        result.execution_time_ms = total_time / constraints.timing_iterations;
        result.memory_bandwidth_gb_s = total_bandwidth / constraints.timing_iterations;
        result.power_usage_watts = total_power / constraints.timing_iterations;
        result.is_valid = true;
        
        // Calculate FLOPS based on kernel type and configuration
        result.flops = calculate_kernel_flops(kernel, inputs, outputs, config);
        
    } catch (const std::exception& e) {
        result.error_message = e.what();
    }
    
    return result;
}

std::string MetalKernelAutoTuner::generate_cache_key(
    const std::string& kernel_name,
    const std::vector<MetalTensorDescriptor>& inputs,
    const std::vector<MetalTensorDescriptor>& outputs) const {
    
    std::stringstream ss;
    ss << kernel_name << "_";
    
    // Include input shapes and types
    for (const auto& input : inputs) {
        for (size_t dim : input.shape()) {
            ss << dim << "x";
        }
        ss << "_" << static_cast<int>(input.dtype()) << "_";
    }
    
    // Include output shapes and types
    for (const auto& output : outputs) {
        for (size_t dim : output.shape()) {
            ss << dim << "x";
        }
        ss << "_" << static_cast<int>(output.dtype()) << "_";
    }
    
    // Include device info
    if (device_) {
        ss << [device_ name].UTF8String;
    }
    
    return ss.str();
}

std::optional<AutoTuneResult> MetalKernelAutoTuner::get_cached_result(
    const std::string& kernel_name,
    const std::vector<MetalTensorDescriptor>& inputs,
    const std::vector<MetalTensorDescriptor>& outputs) const {
    
    std::string key = generate_cache_key(kernel_name, inputs, outputs);
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }
    
    return std::nullopt;
}

void MetalKernelAutoTuner::load_cache(const std::string& cache_file) {
    std::ifstream file(cache_file);
    if (!file.is_open()) {
        return;
    }
    
    Json::Value root;
    file >> root;
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    
    for (const auto& key : root.getMemberNames()) {
        const auto& entry = root[key];
        AutoTuneResult result;
        
        // Parse config
        const auto& config_json = entry["config"];
        result.config.threadgroup_size = {
            config_json["threadgroup"][0].asUInt(),
            config_json["threadgroup"][1].asUInt(),
            config_json["threadgroup"][2].asUInt()
        };
        result.config.grid_size = {
            config_json["grid"][0].asUInt(),
            config_json["grid"][1].asUInt(),
            config_json["grid"][2].asUInt()
        };
        result.config.shared_memory_size = config_json["shared_memory"].asUInt();
        result.config.use_simdgroup = config_json["use_simdgroup"].asBool();
        
        // Parse metrics
        result.execution_time_ms = entry["execution_time_ms"].asDouble();
        result.memory_bandwidth_gb_s = entry["memory_bandwidth_gb_s"].asDouble();
        result.power_usage_watts = entry["power_usage_watts"].asDouble();
        result.flops = entry["flops"].asUInt64();
        result.is_valid = entry["is_valid"].asBool();
        result.error_message = entry["error_message"].asString();
        
        cache_[key] = result;
    }
}

void MetalKernelAutoTuner::save_cache(const std::string& cache_file) const {
    Json::Value root;
    
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (const auto& [key, result] : cache_) {
            Json::Value entry;
            
            // Save config
            Json::Value config_json;
            Json::Value threadgroup(Json::arrayValue);
            threadgroup.append(result.config.threadgroup_size[0]);
            threadgroup.append(result.config.threadgroup_size[1]);
            threadgroup.append(result.config.threadgroup_size[2]);
            config_json["threadgroup"] = threadgroup;
            
            Json::Value grid(Json::arrayValue);
            grid.append(result.config.grid_size[0]);
            grid.append(result.config.grid_size[1]);
            grid.append(result.config.grid_size[2]);
            config_json["grid"] = grid;
            
            config_json["shared_memory"] = result.config.shared_memory_size;
            config_json["use_simdgroup"] = result.config.use_simdgroup;
            entry["config"] = config_json;
            
            // Save metrics
            entry["execution_time_ms"] = result.execution_time_ms;
            entry["memory_bandwidth_gb_s"] = result.memory_bandwidth_gb_s;
            entry["power_usage_watts"] = result.power_usage_watts;
            entry["flops"] = static_cast<Json::UInt64>(result.flops);
            entry["is_valid"] = result.is_valid;
            entry["error_message"] = result.error_message;
            
            root[key] = entry;
        }
    }
    
    std::ofstream file(cache_file);
    file << root;
}

// AutoTunerManager implementation
AutoTunerManager& AutoTunerManager::Instance() {
    static AutoTunerManager instance;
    return instance;
}

// Helper functions for creating configuration spaces
ConfigurationSpace create_gemm_config_space(
    size_t m, size_t n, size_t k, DataType dtype) {
    
    ConfigurationSpace space;
    
    // Thread group sizes for GEMM
    space.threadgroup_sizes = {
        {32, 32, 1},
        {16, 16, 1},
        {8, 8, 1},
        {64, 4, 1},
        {4, 64, 1}
    };
    
    // Precision options
    space.precision_options = {dtype};
    if (dtype == DataType::FLOAT32) {
        space.precision_options.push_back(DataType::FLOAT16);
    }
    
    // Tiling configurations
    space.tiling_configs = {
        {{"TILE_M", 32}, {"TILE_N", 32}, {"TILE_K", 8}},
        {{"TILE_M", 64}, {"TILE_N", 64}, {"TILE_K", 16}},
        {{"TILE_M", 128}, {"TILE_N", 128}, {"TILE_K", 32}}
    };
    
    // Shared memory sizes
    space.shared_memory_sizes = {0, 16384, 32768, 49152};
    
    // SIMD options
    space.use_simdgroup_options = {false, true};
    
    // Kernel variants
    space.kernel_variants = {"basic", "tiled", "simdgroup"};
    
    return space;
}

ConfigurationSpace create_conv_config_space(
    size_t batch, size_t height, size_t width,
    size_t channels, size_t filters,
    size_t kernel_h, size_t kernel_w) {
    
    ConfigurationSpace space;
    
    // Thread group sizes for convolution
    if (kernel_h == 1 && kernel_w == 1) {
        // 1x1 convolution
        space.threadgroup_sizes = {
            {32, 8, 1},
            {16, 16, 1},
            {8, 32, 1}
        };
    } else {
        // General convolution
        space.threadgroup_sizes = {
            {8, 8, 1},
            {16, 8, 1},
            {8, 16, 1},
            {4, 4, 4}
        };
    }
    
    // Tiling for different algorithms
    space.tiling_configs = {
        {{"TILE_H", 4}, {"TILE_W", 4}},
        {{"TILE_H", 8}, {"TILE_W", 8}}
    };
    
    // Memory and precision options
    space.shared_memory_sizes = {0, 16384, 32768};
    space.precision_options = {DataType::FLOAT32, DataType::FLOAT16};
    space.use_simdgroup_options = {false, true};
    
    // Algorithm variants
    if (kernel_h == 3 && kernel_w == 3) {
        space.kernel_variants = {"direct", "winograd", "im2col"};
    } else {
        space.kernel_variants = {"direct", "im2col"};
    }
    
    return space;
}

} // namespace kernels
} // namespace metal
} // namespace triton