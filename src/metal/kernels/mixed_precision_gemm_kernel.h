// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Mixed precision GEMM kernel for Metal

#pragma once

#include "metal_kernel_library.h"

namespace triton {
namespace metal {
namespace kernels {

// Mixed precision GEMM kernel implementation
class MixedPrecisionGEMMKernel : public MetalKernel {
public:
    MixedPrecisionGEMMKernel();
    ~MixedPrecisionGEMMKernel() override = default;
    
    void encode(id<MTLComputeCommandEncoder> encoder,
                const std::vector<id<MTLBuffer>>& inputs,
                const std::vector<id<MTLBuffer>>& outputs,
                const KernelConfig& config) override;
    
    KernelConfig suggest_config(const std::vector<MetalTensorDescriptor>& inputs,
                               const std::vector<MetalTensorDescriptor>& outputs) const override;
    
    bool validate(const std::vector<MetalTensorDescriptor>& inputs,
                 const std::vector<MetalTensorDescriptor>& outputs) const override;
    
    // GEMM-specific configuration
    void set_alpha(float alpha) { alpha_ = alpha; }
    void set_beta(float beta) { beta_ = beta; }
    void set_transpose_a(bool transpose) { transpose_a_ = transpose; }
    void set_transpose_b(bool transpose) { transpose_b_ = transpose; }
    
    // Quantization scales for INT8
    void set_scale_a(float scale) { scale_a_ = scale; }
    void set_scale_b(float scale) { scale_b_ = scale; }
    void set_scale_c(float scale) { scale_c_ = scale; }
    
    // Set precision mode
    enum class PrecisionMode {
        FP32,           // Standard float32
        FP16,           // Half precision
        INT8,           // 8-bit integer
        MIXED_FP16_INT8,// FP16 x INT8
        AUTO            // Choose based on hardware
    };
    
    void set_precision_mode(PrecisionMode mode) { precision_mode_ = mode; }
    
private:
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    bool transpose_a_ = false;
    bool transpose_b_ = false;
    
    // Quantization parameters
    float scale_a_ = 1.0f;
    float scale_b_ = 1.0f;
    float scale_c_ = 1.0f;
    
    PrecisionMode precision_mode_ = PrecisionMode::AUTO;
    
    // Different kernel variants
    id<MTLComputePipelineState> fp32_pipeline_;
    id<MTLComputePipelineState> fp16_pipeline_;
    id<MTLComputePipelineState> fp16_tiled_pipeline_;
    id<MTLComputePipelineState> fp16_simdgroup_pipeline_;
    id<MTLComputePipelineState> int8_pipeline_;
    id<MTLComputePipelineState> mixed_fp16_int8_pipeline_;
    
    // Select optimal kernel based on inputs and config
    id<MTLComputePipelineState> select_pipeline(
        const std::vector<MetalTensorDescriptor>& input_descs,
        const KernelConfig& config) const;
    
    // Get optimal precision for the current hardware
    PrecisionMode get_optimal_precision(
        const std::vector<MetalTensorDescriptor>& inputs,
        size_t m, size_t n, size_t k) const;
};

// Quantization helper for preparing INT8 data
class QuantizationHelper {
public:
    // Quantize FP32 to INT8 with symmetric quantization
    static void quantize_symmetric(
        const float* input,
        char* output,
        size_t count,
        float& scale);
    
    // Quantize FP32 to INT8 with asymmetric quantization
    static void quantize_asymmetric(
        const float* input,
        char* output,
        size_t count,
        float& scale,
        int& zero_point);
    
    // Dequantize INT8 to FP32
    static void dequantize(
        const char* input,
        float* output,
        size_t count,
        float scale,
        int zero_point = 0);
    
    // Compute optimal quantization scale
    static float compute_scale(
        const float* data,
        size_t count,
        bool symmetric = true);
};

// Performance profiler for mixed precision kernels
class MixedPrecisionProfiler {
public:
    struct ProfileResult {
        double fp32_time_ms;
        double fp16_time_ms;
        double int8_time_ms;
        double mixed_time_ms;
        
        double fp32_tflops;
        double fp16_tflops;
        double int8_tops;
        double mixed_tflops;
        
        double fp32_power_w;
        double fp16_power_w;
        double int8_power_w;
        double mixed_power_w;
        
        MixedPrecisionGEMMKernel::PrecisionMode recommended_mode;
        std::string recommendation_reason;
    };
    
    // Profile different precision modes for given problem size
    static ProfileResult profile_gemm(
        size_t m, size_t n, size_t k,
        int num_iterations = 100);
    
    // Get hardware capabilities
    static bool supports_fp16();
    static bool supports_int8();
    static size_t get_fp16_tflops();
    static size_t get_int8_tops();
};

} // namespace kernels
} // namespace metal
} // namespace triton