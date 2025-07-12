// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Winograd Convolution Implementation for Apple Silicon

#include "winograd_conv3x3.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#endif

#include "amx_provider.h"
#include "amx_kernels.h"
#include "../metal/metal_backend_utils.h"

namespace triton {
namespace apple {

// ===========================
// WinogradConv3x3 F(2x2, 3x3)
// ===========================

struct WinogradConv3x3::Impl {
    Config config;
    
    // Workspace buffers
    std::vector<float> input_transform_buffer;
    std::vector<float> kernel_transform_buffer;
    std::vector<float> output_transform_buffer;
    std::vector<float> temp_buffer;
    
    // Tile information
    size_t tiles_h = 0;
    size_t tiles_w = 0;
    size_t total_tiles = 0;
    
    // Precomputed sizes
    size_t transformed_kernel_size = 0;
    size_t workspace_size = 0;
    
    // Performance tracking
    mutable double total_transform_time = 0.0;
    mutable double total_compute_time = 0.0;
    mutable size_t execution_count = 0;
};

WinogradConv3x3::WinogradConv3x3() : impl_(std::make_unique<Impl>()) {}

WinogradConv3x3::~WinogradConv3x3() = default;

TRITONSERVER_Error* WinogradConv3x3::Initialize(const Config& config) {
    impl_->config = config;
    
    // Winograd F(2x2, 3x3) requires stride 1
    if (config.stride != 1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "Winograd F(2x2, 3x3) only supports stride 1");
    }
    
    // Calculate number of tiles
    size_t padded_h = config.height + 2 * config.padding;
    size_t padded_w = config.width + 2 * config.padding;
    
    impl_->tiles_h = (padded_h - 2) / 2;  // Output tile is 2x2
    impl_->tiles_w = (padded_w - 2) / 2;
    impl_->total_tiles = impl_->tiles_h * impl_->tiles_w;
    
    // Allocate workspace buffers
    size_t batch_tiles = config.batch_size * impl_->total_tiles;
    
    // Input transform: (batch * tiles) * channels * 16
    impl_->input_transform_buffer.resize(batch_tiles * config.in_channels * 16);
    
    // Kernel transform: out_channels * in_channels * 16
    impl_->transformed_kernel_size = config.out_channels * config.in_channels * 16;
    impl_->kernel_transform_buffer.resize(impl_->transformed_kernel_size);
    
    // Output transform: (batch * tiles) * out_channels * 16
    impl_->output_transform_buffer.resize(batch_tiles * config.out_channels * 16);
    
    // Temp buffer for intermediate computations
    impl_->temp_buffer.resize(std::max(
        batch_tiles * config.out_channels * 16,
        config.out_channels * config.in_channels * 9
    ));
    
    impl_->workspace_size = (impl_->input_transform_buffer.size() +
                            impl_->output_transform_buffer.size() +
                            impl_->temp_buffer.size()) * sizeof(float);
    
    return nullptr;
}

TRITONSERVER_Error* WinogradConv3x3::Execute(
    const float* input,
    const float* kernel,
    float* output,
    const float* bias) {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Transform kernel G * g * G^T
    TransformKernel(kernel, impl_->kernel_transform_buffer.data());
    
    auto kernel_transform_end = std::chrono::high_resolution_clock::now();
    
    // Step 2: Execute with transformed kernel
    auto err = ExecuteWithTransformedKernel(
        input, 
        impl_->kernel_transform_buffer.data(),
        output,
        bias);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Update timing stats
    impl_->total_transform_time += std::chrono::duration<double, std::milli>(
        kernel_transform_end - start).count();
    impl_->total_compute_time += std::chrono::duration<double, std::milli>(
        end - kernel_transform_end).count();
    impl_->execution_count++;
    
    return err;
}

TRITONSERVER_Error* WinogradConv3x3::ExecuteWithTransformedKernel(
    const float* input,
    const float* transformed_kernel,
    float* output,
    const float* bias) {
    
    const auto& cfg = impl_->config;
    
    // Clear output
    std::memset(output, 0, cfg.batch_size * cfg.height * cfg.width * 
                          cfg.out_channels * sizeof(float));
    
    // Process each batch
    for (size_t b = 0; b < cfg.batch_size; ++b) {
        // Step 1: Transform input tiles B^T * d * B
        TransformInputTiles(
            input + b * cfg.height * cfg.width * cfg.in_channels,
            impl_->input_transform_buffer.data() + b * impl_->total_tiles * cfg.in_channels * 16
        );
        
        // Step 2: Compute in Winograd domain
        // For each of 16 Winograd points
        for (size_t w = 0; w < 16; ++w) {
            // Extract slices for this Winograd point
            float* output_slice = impl_->output_transform_buffer.data() + 
                                 b * impl_->total_tiles * cfg.out_channels * 16 +
                                 w * cfg.out_channels;
            
            const float* input_slice = impl_->input_transform_buffer.data() +
                                      b * impl_->total_tiles * cfg.in_channels * 16 +
                                      w * cfg.in_channels;
            
            const float* kernel_slice = transformed_kernel + w * cfg.out_channels * cfg.in_channels;
            
            // Perform batched GEMM for all tiles
            if (cfg.use_amx && AMXProvider::Instance().IsEnabled()) {
                // Use AMX for small GEMMs
                AMXProvider::Instance().ExecuteGEMM(
                    kernel_slice,              // A: out_channels x in_channels
                    input_slice,               // B: in_channels x tiles
                    output_slice,              // C: out_channels x tiles
                    cfg.out_channels,          // M
                    impl_->total_tiles,        // N
                    cfg.in_channels,           // K
                    1.0f, 0.0f                // alpha, beta
                );
            } else {
                // Fallback to BLAS
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    cfg.out_channels, impl_->total_tiles, cfg.in_channels,
                    1.0f,
                    kernel_slice, cfg.in_channels,
                    input_slice, impl_->total_tiles,
                    0.0f,
                    output_slice, impl_->total_tiles
                );
            }
        }
        
        // Step 3: Transform output A^T * result * A
        TransformOutputTiles(
            impl_->output_transform_buffer.data() + b * impl_->total_tiles * cfg.out_channels * 16,
            output + b * cfg.height * cfg.width * cfg.out_channels
        );
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        AddBias(output, bias);
    }
    
    return nullptr;
}

TRITONSERVER_Error* WinogradConv3x3::TransformKernel(
    const float* kernel,
    float* transformed_kernel) {
    
    const auto& cfg = impl_->config;
    
    // Transform each kernel: G * g * G^T
    // kernel shape: [out_channels, in_channels, 3, 3]
    // transformed shape: [16, out_channels, in_channels]
    
    for (size_t oc = 0; oc < cfg.out_channels; ++oc) {
        for (size_t ic = 0; ic < cfg.in_channels; ++ic) {
            // Extract 3x3 kernel
            float g[3][3];
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    g[i][j] = kernel[(oc * cfg.in_channels + ic) * 9 + i * 3 + j];
                }
            }
            
            // Compute G * g
            float Gg[4][3];
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    Gg[i][j] = 0;
                    for (size_t k = 0; k < 3; ++k) {
                        Gg[i][j] += G[i][k] * g[k][j];
                    }
                }
            }
            
            // Compute (G * g) * G^T
            float GgGT[4][4];
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    GgGT[i][j] = 0;
                    for (size_t k = 0; k < 3; ++k) {
                        GgGT[i][j] += Gg[i][k] * G[j][k];  // G^T
                    }
                }
            }
            
            // Store in transformed kernel buffer
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    size_t w_idx = i * 4 + j;
                    transformed_kernel[w_idx * cfg.out_channels * cfg.in_channels +
                                     oc * cfg.in_channels + ic] = GgGT[i][j];
                }
            }
        }
    }
    
    return nullptr;
}

void WinogradConv3x3::TransformInputTiles(const float* input, float* transformed) {
    const auto& cfg = impl_->config;
    
    // Process each tile
    size_t tile_idx = 0;
    for (size_t ty = 0; ty < impl_->tiles_h; ++ty) {
        for (size_t tx = 0; tx < impl_->tiles_w; ++tx) {
            // Extract 4x4 input tile
            float tile[4][4][256];  // Max 256 channels for stack allocation
            
            for (size_t c = 0; c < cfg.in_channels; ++c) {
                for (size_t i = 0; i < 4; ++i) {
                    for (size_t j = 0; j < 4; ++j) {
                        size_t y = ty * 2 + i;
                        size_t x = tx * 2 + j;
                        
                        // Handle padding
                        if (y < cfg.padding || y >= cfg.height + cfg.padding ||
                            x < cfg.padding || x >= cfg.width + cfg.padding) {
                            tile[i][j][c] = 0;
                        } else {
                            size_t input_y = y - cfg.padding;
                            size_t input_x = x - cfg.padding;
                            tile[i][j][c] = input[(input_y * cfg.width + input_x) * 
                                                  cfg.in_channels + c];
                        }
                    }
                }
            }
            
            // Transform tile: B^T * d * B
            // Use NEON if available
#ifdef __APPLE__
            if (cfg.in_channels % 4 == 0) {
                TransformInputTileNEON(tile, transformed + tile_idx * cfg.in_channels * 16);
            } else
#endif
            {
                // Scalar fallback
                for (size_t c = 0; c < cfg.in_channels; ++c) {
                    // Compute B^T * d
                    float BTd[4][4];
                    for (size_t i = 0; i < 4; ++i) {
                        for (size_t j = 0; j < 4; ++j) {
                            BTd[i][j] = 0;
                            for (size_t k = 0; k < 4; ++k) {
                                BTd[i][j] += BT[i][k] * tile[k][j][c];
                            }
                        }
                    }
                    
                    // Compute (B^T * d) * B
                    float BTdB[4][4];
                    for (size_t i = 0; i < 4; ++i) {
                        for (size_t j = 0; j < 4; ++j) {
                            BTdB[i][j] = 0;
                            for (size_t k = 0; k < 4; ++k) {
                                BTdB[i][j] += BTd[i][k] * BT[j][k];  // B^T
                            }
                        }
                    }
                    
                    // Store transformed tile
                    for (size_t i = 0; i < 4; ++i) {
                        for (size_t j = 0; j < 4; ++j) {
                            size_t w_idx = i * 4 + j;
                            transformed[w_idx * cfg.in_channels * impl_->total_tiles +
                                       c * impl_->total_tiles + tile_idx] = BTdB[i][j];
                        }
                    }
                }
            }
            
            tile_idx++;
        }
    }
}

void WinogradConv3x3::TransformOutputTiles(const float* transformed, float* output) {
    const auto& cfg = impl_->config;
    
    // Process each tile
    size_t tile_idx = 0;
    for (size_t ty = 0; ty < impl_->tiles_h; ++ty) {
        for (size_t tx = 0; tx < impl_->tiles_w; ++tx) {
            // Extract transformed tile for all channels
            float tile[4][4][256];  // Max 256 channels
            
            for (size_t c = 0; c < cfg.out_channels; ++c) {
                for (size_t i = 0; i < 4; ++i) {
                    for (size_t j = 0; j < 4; ++j) {
                        size_t w_idx = i * 4 + j;
                        tile[i][j][c] = transformed[w_idx * cfg.out_channels * impl_->total_tiles +
                                                   c * impl_->total_tiles + tile_idx];
                    }
                }
            }
            
            // Transform tile: A^T * result * A
            for (size_t c = 0; c < cfg.out_channels; ++c) {
                // Compute A^T * result
                float ATr[2][4];
                for (size_t i = 0; i < 2; ++i) {
                    for (size_t j = 0; j < 4; ++j) {
                        ATr[i][j] = 0;
                        for (size_t k = 0; k < 4; ++k) {
                            ATr[i][j] += AT[i][k] * tile[k][j][c];
                        }
                    }
                }
                
                // Compute (A^T * result) * A
                float ATrA[2][2];
                for (size_t i = 0; i < 2; ++i) {
                    for (size_t j = 0; j < 2; ++j) {
                        ATrA[i][j] = 0;
                        for (size_t k = 0; k < 4; ++k) {
                            ATrA[i][j] += ATr[i][k] * AT[j][k];  // A^T
                        }
                    }
                }
                
                // Store output tile
                for (size_t i = 0; i < 2; ++i) {
                    for (size_t j = 0; j < 2; ++j) {
                        size_t y = ty * 2 + i;
                        size_t x = tx * 2 + j;
                        if (y < cfg.height && x < cfg.width) {
                            output[(y * cfg.width + x) * cfg.out_channels + c] += ATrA[i][j];
                        }
                    }
                }
            }
            
            tile_idx++;
        }
    }
}

void WinogradConv3x3::AddBias(float* output, const float* bias) {
    const auto& cfg = impl_->config;
    size_t spatial_size = cfg.height * cfg.width;
    
#ifdef __APPLE__
    // Use NEON for bias addition
    for (size_t c = 0; c < cfg.out_channels; ++c) {
        float32x4_t bias_vec = vdupq_n_f32(bias[c]);
        
        for (size_t i = 0; i < cfg.batch_size * spatial_size; i += 4) {
            float32x4_t out = vld1q_f32(&output[i * cfg.out_channels + c]);
            out = vaddq_f32(out, bias_vec);
            vst1q_f32(&output[i * cfg.out_channels + c], out);
        }
    }
#else
    // Scalar fallback
    for (size_t b = 0; b < cfg.batch_size; ++b) {
        for (size_t i = 0; i < spatial_size; ++i) {
            for (size_t c = 0; c < cfg.out_channels; ++c) {
                output[(b * spatial_size + i) * cfg.out_channels + c] += bias[c];
            }
        }
    }
#endif
}

#ifdef __APPLE__
void WinogradConv3x3::TransformInputTileNEON(
    const float tile[4][4][256], 
    float* transformed) {
    
    const auto& cfg = impl_->config;
    
    // NEON-optimized input transformation
    // Process 4 channels at a time
    for (size_t c = 0; c < cfg.in_channels; c += 4) {
        // Load tile data for 4 channels
        float32x4_t d[4][4];
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                d[i][j] = vld1q_f32(&tile[i][j][c]);
            }
        }
        
        // Compute B^T * d using NEON
        float32x4_t BTd[4][4];
        
        // Row 0: d0 - d2
        BTd[0][0] = vsubq_f32(d[0][0], d[2][0]);
        BTd[0][1] = vsubq_f32(d[0][1], d[2][1]);
        BTd[0][2] = vsubq_f32(d[0][2], d[2][2]);
        BTd[0][3] = vsubq_f32(d[0][3], d[2][3]);
        
        // Row 1: d1 + d2
        BTd[1][0] = vaddq_f32(d[1][0], d[2][0]);
        BTd[1][1] = vaddq_f32(d[1][1], d[2][1]);
        BTd[1][2] = vaddq_f32(d[1][2], d[2][2]);
        BTd[1][3] = vaddq_f32(d[1][3], d[2][3]);
        
        // Row 2: -d1 + d2
        BTd[2][0] = vsubq_f32(d[2][0], d[1][0]);
        BTd[2][1] = vsubq_f32(d[2][1], d[1][1]);
        BTd[2][2] = vsubq_f32(d[2][2], d[1][2]);
        BTd[2][3] = vsubq_f32(d[2][3], d[1][3]);
        
        // Row 3: d1 - d3
        BTd[3][0] = vsubq_f32(d[1][0], d[3][0]);
        BTd[3][1] = vsubq_f32(d[1][1], d[3][1]);
        BTd[3][2] = vsubq_f32(d[1][2], d[3][2]);
        BTd[3][3] = vsubq_f32(d[1][3], d[3][3]);
        
        // Compute (B^T * d) * B
        float32x4_t result[4][4];
        
        // Similar pattern for columns
        for (size_t i = 0; i < 4; ++i) {
            result[i][0] = vsubq_f32(BTd[i][0], BTd[i][2]);
            result[i][1] = vaddq_f32(BTd[i][1], BTd[i][2]);
            result[i][2] = vsubq_f32(BTd[i][2], BTd[i][1]);
            result[i][3] = vsubq_f32(BTd[i][1], BTd[i][3]);
        }
        
        // Store results
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                size_t w_idx = i * 4 + j;
                vst1q_f32(&transformed[w_idx * cfg.in_channels * impl_->total_tiles +
                                      c * impl_->total_tiles], result[i][j]);
            }
        }
    }
}
#endif

size_t WinogradConv3x3::GetWorkspaceSize() const {
    return impl_->workspace_size;
}

size_t WinogradConv3x3::GetTransformedKernelSize() const {
    return impl_->transformed_kernel_size * sizeof(float);
}

WinogradConv3x3::ProfileResult WinogradConv3x3::Profile(int iterations) {
    ProfileResult result;
    const auto& cfg = impl_->config;
    
    // Allocate test data
    std::vector<float> input(cfg.batch_size * cfg.height * cfg.width * cfg.in_channels);
    std::vector<float> kernel(cfg.out_channels * cfg.in_channels * 9);
    std::vector<float> output_winograd(cfg.batch_size * cfg.height * cfg.width * cfg.out_channels);
    std::vector<float> output_direct(cfg.batch_size * cfg.height * cfg.width * cfg.out_channels);
    
    // Initialize with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& v : input) v = dist(gen);
    for (auto& v : kernel) v = dist(gen);
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        Execute(input.data(), kernel.data(), output_winograd.data());
    }
    
    // Benchmark Winograd
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        Execute(input.data(), kernel.data(), output_winograd.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    result.winograd_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    
    // Benchmark direct convolution
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        DirectConv3x3(input.data(), kernel.data(), output_direct.data());
    }
    end = std::chrono::high_resolution_clock::now();
    result.direct_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    
    // Calculate speedup
    result.speedup = result.direct_time_ms / result.winograd_time_ms;
    
    // Memory usage
    result.memory_usage_mb = impl_->workspace_size / (1024.0 * 1024.0);
    
    // Print detailed metrics
    std::cout << "\nWinograd Profile Results:\n";
    std::cout << "  Configuration: " << cfg.batch_size << "x" << cfg.height << "x" 
              << cfg.width << "x" << cfg.in_channels << "→" << cfg.out_channels << "\n";
    std::cout << "  Winograd time: " << result.winograd_time_ms << " ms\n";
    std::cout << "  Direct time: " << result.direct_time_ms << " ms\n";
    std::cout << "  Speedup: " << result.speedup << "x\n";
    std::cout << "  Memory overhead: " << result.memory_usage_mb << " MB\n";
    
    if (impl_->execution_count > 0) {
        std::cout << "  Transform time: " << (impl_->total_transform_time / impl_->execution_count) 
                  << " ms (" << (100.0 * impl_->total_transform_time / 
                                (impl_->total_transform_time + impl_->total_compute_time)) 
                  << "%)\n";
        std::cout << "  Compute time: " << (impl_->total_compute_time / impl_->execution_count) 
                  << " ms\n";
    }
    
    return result;
}

void WinogradConv3x3::DirectConv3x3(
    const float* input,
    const float* kernel,
    float* output) {
    
    const auto& cfg = impl_->config;
    
    // Clear output
    std::memset(output, 0, cfg.batch_size * cfg.height * cfg.width * 
                          cfg.out_channels * sizeof(float));
    
    // Direct convolution implementation
    for (size_t b = 0; b < cfg.batch_size; ++b) {
        for (size_t oc = 0; oc < cfg.out_channels; ++oc) {
            for (size_t y = 0; y < cfg.height; ++y) {
                for (size_t x = 0; x < cfg.width; ++x) {
                    float sum = 0;
                    
                    for (size_t ic = 0; ic < cfg.in_channels; ++ic) {
                        for (size_t ky = 0; ky < 3; ++ky) {
                            for (size_t kx = 0; kx < 3; ++kx) {
                                int in_y = y + ky - cfg.padding;
                                int in_x = x + kx - cfg.padding;
                                
                                if (in_y >= 0 && in_y < cfg.height && 
                                    in_x >= 0 && in_x < cfg.width) {
                                    
                                    float in_val = input[(b * cfg.height * cfg.width +
                                                         in_y * cfg.width + in_x) * 
                                                        cfg.in_channels + ic];
                                    
                                    float k_val = kernel[(oc * cfg.in_channels + ic) * 9 +
                                                        ky * 3 + kx];
                                    
                                    sum += in_val * k_val;
                                }
                            }
                        }
                    }
                    
                    output[(b * cfg.height * cfg.width + y * cfg.width + x) * 
                           cfg.out_channels + oc] = sum;
                }
            }
        }
    }
}

// ===========================
// WinogradAutoSelector
// ===========================

WinogradAutoSelector::SelectionResult WinogradAutoSelector::SelectOptimal(
    size_t batch_size,
    size_t height,
    size_t width,
    size_t in_channels,
    size_t out_channels,
    size_t stride) {
    
    SelectionResult result;
    
    // Stride > 1 not supported by Winograd
    if (stride > 1) {
        result.type = WinogradType::NONE;
        result.expected_speedup = 1.0f;
        result.memory_overhead_mb = 0;
        result.reason = "Stride > 1 not supported by Winograd";
        return result;
    }
    
    // Small spatial dimensions not efficient
    if (height < 8 || width < 8) {
        result.type = WinogradType::NONE;
        result.expected_speedup = 1.0f;
        result.memory_overhead_mb = 0;
        result.reason = "Spatial dimensions too small for Winograd";
        return result;
    }
    
    // Calculate tiles for each configuration
    size_t tiles_2x2 = ((height - 2) / 2) * ((width - 2) / 2);
    size_t tiles_4x4 = ((height - 4) / 4) * ((width - 4) / 4);
    
    // Memory overhead calculation
    size_t mem_2x2 = batch_size * tiles_2x2 * (in_channels + out_channels) * 16 * sizeof(float);
    size_t mem_4x4 = batch_size * tiles_4x4 * (in_channels + out_channels) * 36 * sizeof(float);
    
    // Heuristics based on problem size
    if (batch_size * in_channels * out_channels > 1000000) {
        // Large problem - F(4x4) more efficient
        result.type = WinogradType::F4x4_3x3;
        result.expected_speedup = 2.5f;
        result.memory_overhead_mb = mem_4x4 / (1024.0f * 1024.0f);
        result.reason = "Large problem size favors F(4x4, 3x3)";
    } else if (in_channels < 32 || out_channels < 32) {
        // Small channel count - direct might be better
        result.type = WinogradType::NONE;
        result.expected_speedup = 0.8f;
        result.memory_overhead_mb = 0;
        result.reason = "Channel count too small for efficient Winograd";
    } else {
        // Medium size - F(2x2) is good balance
        result.type = WinogradType::F2x2_3x3;
        result.expected_speedup = 1.8f;
        result.memory_overhead_mb = mem_2x2 / (1024.0f * 1024.0f);
        result.reason = "F(2x2, 3x3) provides good balance";
    }
    
    // Memory constraint check
    if (result.memory_overhead_mb > 100) {
        result.type = WinogradType::NONE;
        result.expected_speedup = 1.0f;
        result.memory_overhead_mb = 0;
        result.reason = "Memory overhead too high";
    }
    
    return result;
}

void WinogradAutoSelector::BenchmarkAll(
    size_t batch_size,
    size_t height,
    size_t width,
    size_t in_channels,
    size_t out_channels) {
    
    std::cout << "\nWinograd Configuration Benchmark\n";
    std::cout << "Problem: " << batch_size << "x" << height << "x" << width 
              << "x" << in_channels << "→" << out_channels << "\n\n";
    
    // Test F(2x2, 3x3)
    {
        WinogradConv3x3 conv_2x2;
        WinogradConv3x3::Config config;
        config.batch_size = batch_size;
        config.height = height;
        config.width = width;
        config.in_channels = in_channels;
        config.out_channels = out_channels;
        
        auto err = conv_2x2.Initialize(config);
        if (!err) {
            auto result = conv_2x2.Profile(50);
            std::cout << "F(2x2, 3x3): " << result.speedup << "x speedup, "
                      << result.memory_usage_mb << " MB overhead\n";
        }
    }
    
    // Test F(4x4, 3x3) when implemented
    std::cout << "F(4x4, 3x3): Not yet implemented\n";
    
    // Show recommendation
    auto selection = SelectOptimal(batch_size, height, width, in_channels, out_channels);
    std::cout << "\nRecommendation: ";
    switch (selection.type) {
        case WinogradType::NONE:
            std::cout << "Direct convolution";
            break;
        case WinogradType::F2x2_3x3:
            std::cout << "Winograd F(2x2, 3x3)";
            break;
        case WinogradType::F4x4_3x3:
            std::cout << "Winograd F(4x4, 3x3)";
            break;
    }
    std::cout << "\nReason: " << selection.reason << "\n";
}

// ===========================
// Utility Functions
// ===========================

namespace winograd_utils {

bool IsSuitableForWinograd(size_t height, size_t width, size_t stride) {
    return stride == 1 && height >= 4 && width >= 4;
}

void CalculateTiles(size_t height, size_t width, size_t tile_size,
                   size_t& tiles_h, size_t& tiles_w) {
    // For output tile size, input needs to be larger
    size_t input_tile_size = tile_size + 2;  // For 3x3 kernel
    
    tiles_h = (height - input_tile_size + tile_size) / tile_size;
    tiles_w = (width - input_tile_size + tile_size) / tile_size;
}

TRITONSERVER_Error* PadInput(
    const float* input,
    float* padded_input,
    size_t batch_size,
    size_t height,
    size_t width,
    size_t channels,
    size_t pad_h,
    size_t pad_w) {
    
    size_t padded_h = height + 2 * pad_h;
    size_t padded_w = width + 2 * pad_w;
    
    // Clear padded buffer
    std::memset(padded_input, 0, batch_size * padded_h * padded_w * channels * sizeof(float));
    
    // Copy input data with padding
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                for (size_t c = 0; c < channels; ++c) {
                    size_t src_idx = ((b * height + y) * width + x) * channels + c;
                    size_t dst_idx = ((b * padded_h + y + pad_h) * padded_w + x + pad_w) * channels + c;
                    padded_input[dst_idx] = input[src_idx];
                }
            }
        }
    }
    
    return nullptr;
}

} // namespace winograd_utils

} // namespace apple
} // namespace triton