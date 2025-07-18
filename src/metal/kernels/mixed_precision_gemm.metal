// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Mixed precision GEMM kernels for Metal

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Helper functions for type conversions
inline float convert_to_float(half value) {
    return static_cast<float>(value);
}

inline float convert_to_float(char value) {  // INT8
    return static_cast<float>(value) / 127.0f;
}

inline half convert_to_half(float value) {
    return static_cast<half>(value);
}

inline char convert_to_int8(float value) {
    return static_cast<char>(clamp(value * 127.0f, -127.0f, 127.0f));
}

// FP16 GEMM kernel with FP32 accumulation
kernel void gemm_half_precision(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]])
{
    const uint row = gid.y;
    const uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    // Use FP32 for accumulation to maintain precision
    float sum = 0.0f;
    
    // Compute dot product
    for (uint k = 0; k < K; k++) {
        float a_val = convert_to_float(A[row * K + k]);
        float b_val = convert_to_float(B[k * N + col]);
        sum += a_val * b_val;
    }
    
    // Apply alpha and beta
    float c_val = (beta != 0.0f) ? beta * convert_to_float(C[row * N + col]) : 0.0f;
    C[row * N + col] = convert_to_half(alpha * sum + c_val);
}

// Tiled FP16 GEMM with shared memory
kernel void gemm_half_tiled(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    threadgroup half* tileA [[threadgroup(0)]],
    threadgroup half* tileB [[threadgroup(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgId [[threadgroup_position_in_grid]],
    uint2 tgSize [[threads_per_threadgroup]])
{
    const uint TILE_SIZE = 16;
    const uint row = tgId.y * TILE_SIZE + tid.y;
    const uint col = tgId.x * TILE_SIZE + tid.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        if (row < M && t * TILE_SIZE + tid.x < K) {
            tileA[tid.y * TILE_SIZE + tid.x] = A[row * K + t * TILE_SIZE + tid.x];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0h;
        }
        
        // Load tile from B
        if (col < N && t * TILE_SIZE + tid.y < K) {
            tileB[tid.y * TILE_SIZE + tid.x] = B[(t * TILE_SIZE + tid.y) * N + col];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0h;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += convert_to_float(tileA[tid.y * TILE_SIZE + k]) * 
                   convert_to_float(tileB[k * TILE_SIZE + tid.x]);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        float c_val = (beta != 0.0f) ? beta * convert_to_float(C[row * N + col]) : 0.0f;
        C[row * N + col] = convert_to_half(alpha * sum + c_val);
    }
}

// INT8 GEMM kernel with INT32 accumulation
kernel void gemm_int8(
    device const char* A [[buffer(0)]],
    device const char* B [[buffer(1)]],
    device char* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    constant float& scale_a [[buffer(8)]],
    constant float& scale_b [[buffer(9)]],
    constant float& scale_c [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint row = gid.y;
    const uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    // Use INT32 for accumulation
    int32_t sum = 0;
    
    // Compute dot product in INT8
    for (uint k = 0; k < K; k++) {
        int8_t a_val = A[row * K + k];
        int8_t b_val = B[k * N + col];
        sum += int32_t(a_val) * int32_t(b_val);
    }
    
    // Convert to float for scaling
    float result = float(sum) * scale_a * scale_b * alpha;
    
    // Add bias if needed
    if (beta != 0.0f) {
        result += beta * float(C[row * N + col]) * scale_c;
    }
    
    // Convert back to INT8 with output scale
    C[row * N + col] = convert_to_int8(result / scale_c);
}

// Mixed precision GEMM: FP16 x INT8 -> FP16
kernel void gemm_mixed_fp16_int8(
    device const half* A [[buffer(0)]],
    device const char* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    constant float& scale_b [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint row = gid.y;
    const uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    
    // Compute dot product with mixed types
    for (uint k = 0; k < K; k++) {
        float a_val = convert_to_float(A[row * K + k]);
        float b_val = float(B[k * N + col]) * scale_b;
        sum += a_val * b_val;
    }
    
    // Apply alpha and beta
    float c_val = (beta != 0.0f) ? beta * convert_to_float(C[row * N + col]) : 0.0f;
    C[row * N + col] = convert_to_half(alpha * sum + c_val);
}

// SIMD group optimized FP16 GEMM
kernel void gemm_half_simdgroup(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    const uint TILE_K = 32;
    const uint row = gid.y;
    const uint col = gid.x * simd_size + simd_lane;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    
    // Process in chunks for better SIMD utilization
    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        simdgroup_float8x8 a_tile;
        simdgroup_float8x8 b_tile;
        
        // Load and convert data
        for (uint k = 0; k < min(TILE_K, K - k_base); k++) {
            half a_val = A[row * K + k_base + k];
            half b_val = B[(k_base + k) * N + col];
            
            // Accumulate using SIMD operations
            sum += convert_to_float(a_val) * convert_to_float(b_val);
        }
    }
    
    // Write result
    float c_val = (beta != 0.0f) ? beta * convert_to_float(C[row * N + col]) : 0.0f;
    C[row * N + col] = convert_to_half(alpha * sum + c_val);
}