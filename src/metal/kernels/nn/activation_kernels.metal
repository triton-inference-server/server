#include <metal_stdlib>
using namespace metal;

// Activation function kernels

// ReLU
kernel void relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        output[gid] = max(input[gid], 0.0f);
    }
}

// ReLU6
kernel void relu6(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        output[gid] = min(max(input[gid], 0.0f), 6.0f);
    }
}

// Leaky ReLU
kernel void leaky_relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        float x = input[gid];
        output[gid] = x >= 0 ? x : alpha * x;
    }
}

// PReLU (Parametric ReLU)
kernel void prelu(
    device const float* input [[buffer(0)]],
    device const float* slope [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        float x = input[gid];
        uint channel = (gid / size) % channels;
        output[gid] = x >= 0 ? x : slope[channel] * x;
    }
}

// Sigmoid
kernel void sigmoid(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        output[gid] = 1.0f / (1.0f + exp(-input[gid]));
    }
}

// Tanh
kernel void tanh_activation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        output[gid] = tanh(input[gid]);
    }
}

// GELU (Gaussian Error Linear Unit)
kernel void gelu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        float x = input[gid];
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608f;
        output[gid] = 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    }
}

// GELU fast approximation
kernel void gelu_fast(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        float x = input[gid];
        // Fast approximation: x * sigmoid(1.702 * x)
        output[gid] = x / (1.0f + exp(-1.702f * x));
    }
}

// Swish (SiLU)
kernel void swish(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        float x = input[gid];
        output[gid] = x / (1.0f + exp(-x));
    }
}

// ELU (Exponential Linear Unit)
kernel void elu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        float x = input[gid];
        output[gid] = x >= 0 ? x : alpha * (exp(x) - 1.0f);
    }
}

// SELU (Scaled Exponential Linear Unit)
kernel void selu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    const float alpha = 1.6732632423543772f;
    const float scale = 1.0507009873554805f;
    
    if (gid < size) {
        float x = input[gid];
        output[gid] = scale * (x >= 0 ? x : alpha * (exp(x) - 1.0f));
    }
}

// Softmax (1D version for simplicity)
kernel void softmax_1d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device atomic_float* max_val [[buffer(2)]],
    device atomic_float* sum_exp [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    // Step 1: Find maximum value
    float local_max = -INFINITY;
    if (gid < size) {
        local_max = input[gid];
    }
    
    // Reduce to find max
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride && gid + stride < size) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        atomic_fetch_max_explicit(max_val, shared[0], memory_order_relaxed);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Step 2: Compute exp(x - max) and sum
    float max_value = atomic_load_explicit(max_val, memory_order_relaxed);
    float local_exp = 0.0f;
    
    if (gid < size) {
        local_exp = exp(input[gid] - max_value);
        output[gid] = local_exp;  // Temporary storage
    }
    
    // Reduce to find sum
    shared[tid] = local_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride && gid + stride < size) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        atomic_fetch_add_explicit(sum_exp, shared[0], memory_order_relaxed);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Step 3: Normalize
    float sum_value = atomic_load_explicit(sum_exp, memory_order_relaxed);
    if (gid < size) {
        output[gid] = output[gid] / sum_value;
    }
}

// Softmax 2D (along last dimension)
kernel void softmax_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint2& shape [[buffer(2)]],  // [rows, cols]
    uint2 gid [[thread_position_in_grid]]) {
    
    const uint row = gid.y;
    const uint col = gid.x;
    const uint rows = shape.x;
    const uint cols = shape.y;
    
    if (row >= rows) return;
    
    // Each row handles its own softmax
    if (col == 0) {
        const uint offset = row * cols;
        
        // Find max
        float max_val = -INFINITY;
        for (uint c = 0; c < cols; ++c) {
            max_val = max(max_val, input[offset + c]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (uint c = 0; c < cols; ++c) {
            float exp_val = exp(input[offset + c] - max_val);
            output[offset + c] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (uint c = 0; c < cols; ++c) {
            output[offset + c] /= sum_exp;
        }
    }
}

// LogSoftmax
kernel void log_softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint2& shape [[buffer(2)]],  // [batch, classes]
    uint2 gid [[thread_position_in_grid]]) {
    
    const uint batch = gid.y;
    const uint idx = gid.x;
    const uint batch_size = shape.x;
    const uint num_classes = shape.y;
    
    if (batch >= batch_size || idx >= num_classes) return;
    
    const uint offset = batch * num_classes;
    
    // Find max
    float max_val = -INFINITY;
    for (uint c = 0; c < num_classes; ++c) {
        max_val = max(max_val, input[offset + c]);
    }
    
    // Compute log sum exp
    float sum_exp = 0.0f;
    for (uint c = 0; c < num_classes; ++c) {
        sum_exp += exp(input[offset + c] - max_val);
    }
    float log_sum_exp = max_val + log(sum_exp);
    
    // Compute log softmax
    output[offset + idx] = input[offset + idx] - log_sum_exp;
}

// Hardswish
kernel void hardswish(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        float x = input[gid];
        if (x <= -3.0f) {
            output[gid] = 0.0f;
        } else if (x >= 3.0f) {
            output[gid] = x;
        } else {
            output[gid] = x * (x + 3.0f) / 6.0f;
        }
    }
}

// Mish
kernel void mish(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        float x = input[gid];
        output[gid] = x * tanh(log(1.0f + exp(x)));
    }
}

// Fused activation functions for better performance
kernel void conv2d_relu_fused(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint4& input_shape [[buffer(4)]],
    constant uint4& weight_shape [[buffer(5)]],
    constant uint4& output_shape [[buffer(6)]],
    constant uint2& stride [[buffer(7)]],
    constant uint2& padding [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint k = gid.y;
    const uint out_idx = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint H = input_shape.z;
    const uint W = input_shape.w;
    
    const uint K = weight_shape.x;
    const uint R = weight_shape.z;
    const uint S = weight_shape.w;
    
    const uint P = output_shape.z;
    const uint Q = output_shape.w;
    
    if (n >= N || k >= K || out_idx >= P * Q) return;
    
    const uint p = out_idx / Q;
    const uint q = out_idx % Q;
    
    float sum = bias ? bias[k] : 0.0f;
    
    // Convolution
    for (uint c = 0; c < C; ++c) {
        for (uint r = 0; r < R; ++r) {
            for (uint s = 0; s < S; ++s) {
                int h = p * stride.x - padding.x + r;
                int w = q * stride.y - padding.y + s;
                
                if (h >= 0 && h < int(H) && w >= 0 && w < int(W)) {
                    uint input_idx = n * C * H * W + c * H * W + h * W + w;
                    uint weight_idx = k * C * R * S + c * R * S + r * S + s;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Fused ReLU
    uint output_idx = n * K * P * Q + k * P * Q + p * Q + q;
    output[output_idx] = max(sum, 0.0f);
}