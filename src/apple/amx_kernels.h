// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// AMX Optimized Kernel Library

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace triton {
namespace apple {
namespace amx {

// AMX data types
enum class DataType {
    FP32,
    FP16,
    BF16,
    INT8,
    INT16,
    INT32
};

// Transpose options
enum class TransposeOp {
    NoTrans,
    Trans,
    ConjTrans  // For complex numbers
};

// Activation types
enum class ActivationType {
    None,
    ReLU,
    ReLU6,
    Sigmoid,
    Tanh,
    GELU,
    Swish
};

// Convolution parameters
struct ConvParams {
    size_t stride_h = 1;
    size_t stride_w = 1;
    size_t pad_h = 0;
    size_t pad_w = 0;
    size_t dilation_h = 1;
    size_t dilation_w = 1;
    size_t groups = 1;
    ActivationType activation = ActivationType::None;
};

// Pooling parameters
struct PoolParams {
    size_t kernel_h;
    size_t kernel_w;
    size_t stride_h = 1;
    size_t stride_w = 1;
    size_t pad_h = 0;
    size_t pad_w = 0;
};

// AMX kernel functions
namespace kernels {

// ======================
// Matrix Operations
// ======================

// General matrix multiply: C = alpha * op(A) * op(B) + beta * C
void gemm(
    TransposeOp trans_a,
    TransposeOp trans_b,
    size_t m, size_t n, size_t k,
    float alpha,
    const float* a, size_t lda,
    const float* b, size_t ldb,
    float beta,
    float* c, size_t ldc);

// FP16 GEMM
void hgemm(
    TransposeOp trans_a,
    TransposeOp trans_b,
    size_t m, size_t n, size_t k,
    float alpha,
    const uint16_t* a, size_t lda,
    const uint16_t* b, size_t ldb,
    float beta,
    uint16_t* c, size_t ldc);

// INT8 GEMM with INT32 accumulation
void igemm(
    TransposeOp trans_a,
    TransposeOp trans_b,
    size_t m, size_t n, size_t k,
    int32_t alpha,
    const int8_t* a, size_t lda,
    const int8_t* b, size_t ldb,
    int32_t beta,
    int32_t* c, size_t ldc);

// Batched GEMM
void gemm_batch(
    TransposeOp trans_a,
    TransposeOp trans_b,
    size_t m, size_t n, size_t k,
    float alpha,
    const float* const* a_array, size_t lda,
    const float* const* b_array, size_t ldb,
    float beta,
    float* const* c_array, size_t ldc,
    size_t batch_count);

// Matrix-vector multiply: y = alpha * A * x + beta * y
void gemv(
    TransposeOp trans,
    size_t m, size_t n,
    float alpha,
    const float* a, size_t lda,
    const float* x, size_t incx,
    float beta,
    float* y, size_t incy);

// ======================
// Convolution Operations
// ======================

// 2D Convolution
void conv2d(
    const float* input,
    const float* kernel,
    const float* bias,
    float* output,
    size_t batch, size_t in_h, size_t in_w, size_t in_c,
    size_t out_c, size_t kernel_h, size_t kernel_w,
    const ConvParams& params);

// Depthwise convolution
void depthwise_conv2d(
    const float* input,
    const float* kernel,
    const float* bias,
    float* output,
    size_t batch, size_t in_h, size_t in_w, size_t channels,
    size_t kernel_h, size_t kernel_w,
    const ConvParams& params);

// 1x1 convolution (optimized)
void conv2d_1x1(
    const float* input,
    const float* kernel,
    const float* bias,
    float* output,
    size_t batch, size_t in_h, size_t in_w, size_t in_c,
    size_t out_c,
    const ConvParams& params);

// Winograd convolution for 3x3 filters
void conv2d_winograd_3x3(
    const float* input,
    const float* kernel,
    const float* bias,
    float* output,
    size_t batch, size_t in_h, size_t in_w, size_t in_c,
    size_t out_c,
    const ConvParams& params);

// ======================
// Pooling Operations
// ======================

void max_pool2d(
    const float* input,
    float* output,
    size_t batch, size_t in_h, size_t in_w, size_t channels,
    const PoolParams& params);

void avg_pool2d(
    const float* input,
    float* output,
    size_t batch, size_t in_h, size_t in_w, size_t channels,
    const PoolParams& params);

void global_avg_pool2d(
    const float* input,
    float* output,
    size_t batch, size_t in_h, size_t in_w, size_t channels);

// ======================
// Activation Functions
// ======================

void relu(const float* input, float* output, size_t size);
void relu6(const float* input, float* output, size_t size);
void sigmoid(const float* input, float* output, size_t size);
void tanh(const float* input, float* output, size_t size);
void gelu(const float* input, float* output, size_t size);
void swish(const float* input, float* output, size_t size);
void leaky_relu(const float* input, float* output, size_t size, float alpha = 0.01f);

// Fused operations
void relu_inplace(float* data, size_t size);
void add_relu(const float* a, const float* b, float* c, size_t size);
void bias_add_relu(const float* input, const float* bias, float* output,
                   size_t batch, size_t channels);

// ======================
// Normalization
// ======================

void batch_norm(
    const float* input,
    const float* scale,
    const float* bias,
    const float* mean,
    const float* variance,
    float* output,
    size_t batch, size_t channels, size_t spatial_size,
    float epsilon = 1e-5f);

void layer_norm(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    size_t batch, size_t seq_len, size_t hidden_size,
    float epsilon = 1e-5f);

void group_norm(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    size_t batch, size_t channels, size_t spatial_size,
    size_t num_groups,
    float epsilon = 1e-5f);

// ======================
// Transformer Operations
// ======================

// Scaled dot-product attention
void attention(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    size_t batch, size_t num_heads, size_t seq_len, size_t head_dim,
    float scale = -1.0f,  // Default: 1/sqrt(head_dim)
    const float* mask = nullptr);

// Multi-head attention (fused)
void multihead_attention(
    const float* query,
    const float* key,
    const float* value,
    const float* w_q,
    const float* w_k,
    const float* w_v,
    const float* w_o,
    const float* bias_q,
    const float* bias_k,
    const float* bias_v,
    const float* bias_o,
    float* output,
    size_t batch, size_t seq_len, size_t hidden_dim, size_t num_heads,
    const float* mask = nullptr);

// Feed-forward network
void ffn(
    const float* input,
    const float* w1,
    const float* w2,
    const float* bias1,
    const float* bias2,
    float* output,
    size_t batch, size_t seq_len, size_t hidden_dim, size_t ffn_dim,
    ActivationType activation = ActivationType::ReLU);

// ======================
// Utility Operations
// ======================

// Element-wise operations
void add(const float* a, const float* b, float* c, size_t size);
void multiply(const float* a, const float* b, float* c, size_t size);
void scale(const float* input, float* output, size_t size, float scale);
void axpy(size_t n, float alpha, const float* x, float* y);  // y = alpha*x + y

// Reduction operations
float reduce_sum(const float* input, size_t size);
float reduce_mean(const float* input, size_t size);
float reduce_max(const float* input, size_t size);
float reduce_min(const float* input, size_t size);

// Memory operations
void copy(const float* src, float* dst, size_t size);
void fill(float* data, size_t size, float value);
void transpose(const float* input, float* output, size_t rows, size_t cols);

} // namespace kernels

// ======================
// Performance Utilities
// ======================

// Auto-tuning support
struct TuningResult {
    size_t tile_m;
    size_t tile_n;
    size_t tile_k;
    bool use_amx2;
    double gflops;
};

TuningResult autotune_gemm(
    size_t m, size_t n, size_t k,
    DataType dtype = DataType::FP32);

// Profiling support
class KernelProfiler {
public:
    void start();
    void stop();
    double get_time_ms() const;
    double get_gflops() const;
    void reset();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Memory management
void* allocate_aligned(size_t size, size_t alignment = 64);
void free_aligned(void* ptr);

// Check if size is AMX-friendly (multiple of tile size)
bool is_amx_friendly_size(size_t m, size_t n, size_t k);

// Pad dimensions to AMX tile boundaries
void get_padded_dims(size_t m, size_t n, size_t k,
                    size_t& padded_m, size_t& padded_n, size_t& padded_k);

} // namespace amx
} // namespace apple
} // namespace triton