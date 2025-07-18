// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Winograd Convolution for 3x3 filters on Apple Silicon

#pragma once

#include <memory>
#include <vector>
#include "triton/core/tritonserver.h"

namespace triton {
namespace apple {

// Winograd F(2x2, 3x3) - Outputs 2x2 tile from 3x3 kernel
// Requires 4x4 input tile, reduces 9 multiplications to 16
class WinogradConv3x3 {
public:
    // Configuration for Winograd convolution
    struct Config {
        size_t batch_size;
        size_t height;
        size_t width;
        size_t in_channels;
        size_t out_channels;
        size_t stride = 1;
        size_t padding = 1;
        bool use_amx = true;     // Use AMX for transform matrices
        bool use_metal = false;  // Use Metal for large batches
    };
    
    WinogradConv3x3();
    ~WinogradConv3x3();
    
    // Initialize Winograd convolution with configuration
    TRITONSERVER_Error* Initialize(const Config& config);
    
    // Execute Winograd convolution
    TRITONSERVER_Error* Execute(
        const float* input,
        const float* kernel,
        float* output,
        const float* bias = nullptr);
    
    // Execute with pre-transformed kernel (for multiple executions)
    TRITONSERVER_Error* ExecuteWithTransformedKernel(
        const float* input,
        const float* transformed_kernel,
        float* output,
        const float* bias = nullptr);
    
    // Transform kernel offline (for reuse)
    TRITONSERVER_Error* TransformKernel(
        const float* kernel,
        float* transformed_kernel);
    
    // Get required workspace size
    size_t GetWorkspaceSize() const;
    
    // Get transformed kernel size
    size_t GetTransformedKernelSize() const;
    
    // Profile Winograd vs direct convolution
    struct ProfileResult {
        double winograd_time_ms;
        double direct_time_ms;
        double speedup;
        size_t memory_usage_mb;
    };
    
    ProfileResult Profile(int iterations = 100);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // Direct convolution for testing/comparison
    void DirectConv3x3(const float* input, const float* kernel, float* output);
    
    // Transformation functions
    void TransformInputTiles(const float* input, float* transformed);
    void TransformOutputTiles(const float* transformed, float* output);
    void AddBias(float* output, const float* bias);
    void TransformInputTileNEON(const float* tile, float* transformed);
    
    // Transformation matrices
    static constexpr float G[4][3] = {
        { 1.0f,  0.0f,  0.0f},
        { 0.5f,  0.5f,  0.5f},
        { 0.5f, -0.5f,  0.5f},
        { 0.0f,  0.0f,  1.0f}
    };
    
    static constexpr float BT[4][4] = {
        { 1.0f,  0.0f, -1.0f,  0.0f},
        { 0.0f,  1.0f,  1.0f,  0.0f},
        { 0.0f, -1.0f,  1.0f,  0.0f},
        { 0.0f,  1.0f,  0.0f, -1.0f}
    };
    
    static constexpr float AT[2][4] = {
        { 1.0f,  1.0f,  1.0f,  0.0f},
        { 0.0f,  1.0f, -1.0f, -1.0f}
    };
};

// Winograd F(4x4, 3x3) - Outputs 4x4 tile from 3x3 kernel
// More efficient for larger tiles but requires more memory
class WinogradConv3x3_F4x4 {
public:
    struct Config {
        size_t batch_size;
        size_t height;
        size_t width;
        size_t in_channels;
        size_t out_channels;
        size_t stride = 1;
        size_t padding = 1;
        bool use_amx = true;
        bool use_metal = false;
        bool use_neon = true;  // ARM NEON optimizations
    };
    
    WinogradConv3x3_F4x4();
    ~WinogradConv3x3_F4x4();
    
    TRITONSERVER_Error* Initialize(const Config& config);
    
    TRITONSERVER_Error* Execute(
        const float* input,
        const float* kernel,
        float* output,
        const float* bias = nullptr);
    
    // Fused Winograd + ReLU
    TRITONSERVER_Error* ExecuteWithReLU(
        const float* input,
        const float* kernel,
        float* output,
        const float* bias = nullptr);
    
    // Mixed precision execution (FP16)
    TRITONSERVER_Error* ExecuteFP16(
        const __fp16* input,
        const __fp16* kernel,
        __fp16* output,
        const __fp16* bias = nullptr);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // F(4x4, 3x3) transformation matrices
    static constexpr float G_4x4[6][3] = {
        { 1.0f/4.0f,     0.0f,      0.0f},
        {-1.0f/6.0f, -1.0f/6.0f, -1.0f/6.0f},
        {-1.0f/6.0f,  1.0f/6.0f, -1.0f/6.0f},
        { 1.0f/24.0f, 1.0f/12.0f, 1.0f/6.0f},
        { 1.0f/24.0f,-1.0f/12.0f, 1.0f/6.0f},
        {     0.0f,      0.0f,      1.0f}
    };
};

// Auto-selector for best Winograd configuration
class WinogradAutoSelector {
public:
    enum class WinogradType {
        NONE,      // Direct convolution
        F2x2_3x3,  // Winograd F(2x2, 3x3)
        F4x4_3x3   // Winograd F(4x4, 3x3)
    };
    
    struct SelectionResult {
        WinogradType type;
        float expected_speedup;
        size_t memory_overhead_mb;
        std::string reason;
    };
    
    // Select best Winograd configuration based on problem size
    static SelectionResult SelectOptimal(
        size_t batch_size,
        size_t height,
        size_t width,
        size_t in_channels,
        size_t out_channels,
        size_t stride = 1);
    
    // Benchmark all configurations
    static void BenchmarkAll(
        size_t batch_size,
        size_t height,
        size_t width,
        size_t in_channels,
        size_t out_channels);
};

// Utility functions for Winograd convolution
namespace winograd_utils {
    
    // Check if problem size is suitable for Winograd
    bool IsSuitableForWinograd(
        size_t height,
        size_t width,
        size_t stride);
    
    // Calculate number of tiles
    void CalculateTiles(
        size_t height,
        size_t width,
        size_t tile_size,
        size_t& tiles_h,
        size_t& tiles_w);
    
    // Pad input for Winograd tiling
    TRITONSERVER_Error* PadInput(
        const float* input,
        float* padded_input,
        size_t batch_size,
        size_t height,
        size_t width,
        size_t channels,
        size_t pad_h,
        size_t pad_w);
}

} // namespace apple
} // namespace triton