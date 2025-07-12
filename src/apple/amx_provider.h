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
#include <vector>

#include "triton/core/tritonserver.h"

namespace triton {
namespace apple {

// Forward declarations
class AMXKernelLibrary;

// AMX capability flags
struct AMXCapabilities {
    bool has_amx = false;
    bool has_amx2 = false;  // M2 and later
    bool has_amx_fp16 = false;
    bool has_amx_int8 = false;
    bool has_amx_bf16 = false;  // M3 and later
    size_t max_tile_rows = 0;
    size_t max_tile_cols = 0;
    size_t tile_palette_size = 0;
};

// AMX operation types
enum class AMXOpType {
    GEMM_FP32,
    GEMM_FP16,
    GEMM_INT8,
    GEMM_BF16,
    GEMMV_FP32,    // Matrix-vector
    OUTER_PRODUCT,
    FMA,           // Fused multiply-add
    LOAD_STORE
};

// AMX execution configuration
struct AMXConfig {
    AMXOpType op_type = AMXOpType::GEMM_FP32;
    bool use_tiles = true;
    size_t tile_m = 32;  // AMX uses 32x32 tiles
    size_t tile_n = 32;
    size_t tile_k = 32;
    bool zero_accumulator = true;
    bool transpose_a = false;
    bool transpose_b = false;
};

// AMX performance metrics
struct AMXMetrics {
    size_t total_operations = 0;
    double total_time_ms = 0.0;
    double gflops = 0.0;
    double memory_bandwidth_gb_s = 0.0;
    size_t cache_misses = 0;
    double power_usage_watts = 0.0;
};

// AMX provider for CPU-side matrix acceleration
class AMXProvider {
public:
    // Singleton instance
    static AMXProvider& Instance();
    
    // Initialize AMX support
    TRITONSERVER_Error* Initialize();
    
    // Check if AMX is available
    bool IsAvailable() const { return capabilities_.has_amx; }
    
    // Get capabilities
    const AMXCapabilities& GetCapabilities() const { return capabilities_; }
    
    // Execute operations
    TRITONSERVER_Error* ExecuteGEMM(
        const float* A, const float* B, float* C,
        size_t M, size_t N, size_t K,
        float alpha = 1.0f, float beta = 0.0f,
        const AMXConfig& config = AMXConfig());
    
    TRITONSERVER_Error* ExecuteGEMMBatch(
        const float* const* A, const float* const* B, float* const* C,
        size_t batch_size, size_t M, size_t N, size_t K,
        float alpha = 1.0f, float beta = 0.0f,
        const AMXConfig& config = AMXConfig());
    
    // Mixed precision operations
    TRITONSERVER_Error* ExecuteGEMM_FP16(
        const uint16_t* A, const uint16_t* B, uint16_t* C,
        size_t M, size_t N, size_t K,
        float alpha = 1.0f, float beta = 0.0f);
    
    TRITONSERVER_Error* ExecuteGEMM_INT8(
        const int8_t* A, const int8_t* B, int32_t* C,
        size_t M, size_t N, size_t K,
        int32_t alpha = 1, int32_t beta = 0);
    
    // Vector operations
    TRITONSERVER_Error* ExecuteGEMV(
        const float* A, const float* x, float* y,
        size_t M, size_t N,
        float alpha = 1.0f, float beta = 0.0f);
    
    // Get performance metrics
    AMXMetrics GetMetrics() const;
    void ResetMetrics();
    
    // Enable/disable AMX
    void SetEnabled(bool enabled) { enabled_ = enabled; }
    bool IsEnabled() const { return enabled_ && capabilities_.has_amx; }
    
    // Performance tuning
    void SetAutoTuning(bool enable) { auto_tuning_ = enable; }
    AMXConfig GetOptimalConfig(size_t M, size_t N, size_t K) const;
    
private:
    AMXProvider();
    ~AMXProvider();
    
    // Detect AMX capabilities
    void DetectCapabilities();
    
    // AMX instruction wrappers (using inline assembly)
    void AMXStart();
    void AMXStop();
    void AMXLoadConfig(const void* config);
    void AMXLoadTile(void* tile, const void* src, size_t stride);
    void AMXStoreTile(void* dst, const void* tile, size_t stride);
    void AMXFMAOuter(void* dst, const void* a, const void* b);
    void AMXFMA32(void* dst, const void* a, const void* b);
    void AMXFMA16(void* dst, const void* a, const void* b);
    
    // Tiled implementation
    TRITONSERVER_Error* ExecuteTiledGEMM(
        const float* A, const float* B, float* C,
        size_t M, size_t N, size_t K,
        float alpha, float beta,
        const AMXConfig& config);
    
    // Update metrics
    void UpdateMetrics(size_t ops, double time_ms);
    
    // Member variables
    AMXCapabilities capabilities_;
    std::atomic<bool> enabled_{true};
    std::atomic<bool> auto_tuning_{false};
    mutable std::mutex metrics_mutex_;
    AMXMetrics metrics_;
    std::unique_ptr<AMXKernelLibrary> kernel_library_;
    
    // Prevent copying
    AMXProvider(const AMXProvider&) = delete;
    AMXProvider& operator=(const AMXProvider&) = delete;
};

// AMX kernel library
class AMXKernelLibrary {
public:
    AMXKernelLibrary();
    ~AMXKernelLibrary();
    
    // Optimized kernels
    void sgemm_amx(
        const float* A, const float* B, float* C,
        size_t M, size_t N, size_t K,
        float alpha, float beta);
    
    void hgemm_amx(
        const uint16_t* A, const uint16_t* B, uint16_t* C,
        size_t M, size_t N, size_t K,
        float alpha, float beta);
    
    void igemm_amx(
        const int8_t* A, const int8_t* B, int32_t* C,
        size_t M, size_t N, size_t K);
    
    // Convolution kernels
    void conv2d_amx(
        const float* input, const float* kernel, float* output,
        size_t batch, size_t height, size_t width,
        size_t in_channels, size_t out_channels,
        size_t kernel_h, size_t kernel_w,
        size_t stride_h, size_t stride_w,
        size_t pad_h, size_t pad_w);
    
    // Activation functions using AMX
    void relu_amx(const float* input, float* output, size_t size);
    void sigmoid_amx(const float* input, float* output, size_t size);
    void tanh_amx(const float* input, float* output, size_t size);
    
private:
    // Implementation details hidden
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Helper functions
bool DetectAMXSupport();
std::string GetAMXInfoString();
size_t GetAMXPeakGFLOPS();

// Integration with Triton backend
class AMXBackendFactory {
public:
    static TRITONSERVER_Error* CreateBackend(
        TRITONSERVER_Backend** backend,
        const char* name,
        const uint64_t version);
};

} // namespace apple
} // namespace triton