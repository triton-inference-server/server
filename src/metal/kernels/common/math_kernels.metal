#include <metal_stdlib>
using namespace metal;

// GEMM kernels with various optimizations

// Basic GEMM kernel
kernel void gemm_basic(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    
    const uint row = gid.y;
    const uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    const uint idx = row * N + col;
    C[idx] = alpha * sum + beta * C[idx];
}

// GEMM with shared memory tiling
kernel void gemm_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]]) {
    
    const uint TILE_SIZE = 16;
    const uint row = gid.y;
    const uint col = gid.x;
    const uint localRow = tid.y;
    const uint localCol = tid.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (uint tileIdx = 0; tileIdx < (K + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {
        // Load A tile
        const uint aRow = row;
        const uint aCol = tileIdx * TILE_SIZE + localCol;
        if (aRow < M && aCol < K) {
            tileA[localRow * TILE_SIZE + localCol] = A[aRow * K + aCol];
        } else {
            tileA[localRow * TILE_SIZE + localCol] = 0.0f;
        }
        
        // Load B tile
        const uint bRow = tileIdx * TILE_SIZE + localRow;
        const uint bCol = col;
        if (bRow < K && bCol < N) {
            tileB[localRow * TILE_SIZE + localCol] = B[bRow * N + bCol];
        } else {
            tileB[localRow * TILE_SIZE + localCol] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[localRow * TILE_SIZE + k] * tileB[k * TILE_SIZE + localCol];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        const uint idx = row * N + col;
        C[idx] = alpha * sum + beta * C[idx];
    }
}

// GEMM optimized for Apple Silicon with simdgroup operations
kernel void gemm_simdgroup(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
    
    const uint SIMD_SIZE = 32;
    const uint row = gid.y;
    const uint col_base = gid.x * SIMD_SIZE;
    
    if (row >= M) return;
    
    float4 sum = float4(0.0f);
    
    // Process 4 columns per thread
    for (uint k = 0; k < K; ++k) {
        float a_val = A[row * K + k];
        
        for (uint c = 0; c < 4; ++c) {
            uint col = col_base + simd_lane_id + c * 8;
            if (col < N) {
                float b_val = B[k * N + col];
                sum[c] += a_val * b_val;
            }
        }
    }
    
    // Write results
    for (uint c = 0; c < 4; ++c) {
        uint col = col_base + simd_lane_id + c * 8;
        if (col < N) {
            const uint idx = row * N + col;
            C[idx] = alpha * sum[c] + beta * C[idx];
        }
    }
}

// Half precision GEMM
kernel void gemm_half(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant half& alpha [[buffer(6)]],
    constant half& beta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    
    const uint row = gid.y;
    const uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;  // Use float for accumulation
    
    for (uint k = 0; k < K; ++k) {
        sum += float(A[row * K + k]) * float(B[k * N + col]);
    }
    
    const uint idx = row * N + col;
    C[idx] = half(float(alpha) * sum + float(beta) * float(C[idx]));
}

// GEMV (Matrix-Vector multiplication)
kernel void gemv(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant float& alpha [[buffer(5)]],
    constant float& beta [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid >= M) return;
    
    float sum = 0.0f;
    for (uint j = 0; j < N; ++j) {
        sum += A[gid * N + j] * x[j];
    }
    
    y[gid] = alpha * sum + beta * y[gid];
}

// Batched GEMM
kernel void gemm_batched(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant float& alpha [[buffer(7)]],
    constant float& beta [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint batch = gid.z;
    const uint row = gid.y;
    const uint col = gid.x;
    
    if (batch >= batch_size || row >= M || col >= N) return;
    
    const uint batch_offset_a = batch * M * K;
    const uint batch_offset_b = batch * K * N;
    const uint batch_offset_c = batch * M * N;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        sum += A[batch_offset_a + row * K + k] * B[batch_offset_b + k * N + col];
    }
    
    const uint idx = batch_offset_c + row * N + col;
    C[idx] = alpha * sum + beta * C[idx];
}

// Element-wise operations
kernel void add_tensor(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        c[gid] = a[gid] + b[gid];
    }
}

kernel void mul_tensor(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        c[gid] = a[gid] * b[gid];
    }
}

kernel void add_scalar(
    device const float* a [[buffer(0)]],
    constant float& scalar [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        c[gid] = a[gid] + scalar;
    }
}

kernel void mul_scalar(
    device const float* a [[buffer(0)]],
    constant float& scalar [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        c[gid] = a[gid] * scalar;
    }
}

// AXPY: y = alpha * x + y
kernel void axpy(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        y[gid] = alpha * x[gid] + y[gid];
    }
}

// Dot product with reduction
kernel void dot_product(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device atomic_float* result [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    // Load and compute local dot product
    float local_sum = 0.0f;
    if (gid < size) {
        local_sum = a[gid] * b[gid];
    }
    
    // Store in shared memory
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride && gid + stride < size) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        atomic_fetch_add_explicit(result, shared[0], memory_order_relaxed);
    }
}