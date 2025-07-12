#include <metal_stdlib>
using namespace metal;

// Normalization kernels

// Batch Normalization (inference mode)
kernel void batch_norm_inference(
    device const float* input [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* variance [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant uint4& shape [[buffer(7)]],  // [N, C, H, W]
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint c = gid.y;
    const uint spatial_idx = gid.x;
    
    const uint N = shape.x;
    const uint C = shape.y;
    const uint spatial_size = shape.z * shape.w;
    
    if (n >= N || c >= C || spatial_idx >= spatial_size) return;
    
    const uint idx = n * C * spatial_size + c * spatial_size + spatial_idx;
    
    float x = input[idx];
    float m = mean[c];
    float v = variance[c];
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;
    
    // Normalize: y = gamma * (x - mean) / sqrt(var + eps) + beta
    output[idx] = g * (x - m) / sqrt(v + epsilon) + b;
}

// Batch Normalization (training mode with running stats update)
kernel void batch_norm_training(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* running_mean [[buffer(2)]],
    device float* running_var [[buffer(3)]],
    device const float* gamma [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device atomic_float* batch_mean [[buffer(6)]],
    device atomic_float* batch_var [[buffer(7)]],
    threadgroup float* shared_mean [[threadgroup(0)]],
    threadgroup float* shared_var [[threadgroup(1)]],
    constant float& momentum [[buffer(8)]],
    constant float& epsilon [[buffer(9)]],
    constant uint4& shape [[buffer(10)]],  // [N, C, H, W]
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint c = gid.y;
    const uint idx = gid.x;
    
    const uint N = shape.x;
    const uint C = shape.y;
    const uint spatial_size = shape.z * shape.w;
    const uint channel_size = N * spatial_size;
    
    if (c >= C) return;
    
    // Step 1: Compute mean for this channel
    float local_sum = 0.0f;
    uint count = 0;
    
    for (uint n = 0; n < N; ++n) {
        for (uint s = tid; s < spatial_size; s += tg_size) {
            uint input_idx = n * C * spatial_size + c * spatial_size + s;
            local_sum += input[input_idx];
            count++;
        }
    }
    
    // Reduce within threadgroup
    shared_mean[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mean[tid] += shared_mean[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        atomic_fetch_add_explicit(&batch_mean[c], shared_mean[0], memory_order_relaxed);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    float mean = atomic_load_explicit(&batch_mean[c], memory_order_relaxed) / float(channel_size);
    
    // Step 2: Compute variance
    float local_var = 0.0f;
    
    for (uint n = 0; n < N; ++n) {
        for (uint s = tid; s < spatial_size; s += tg_size) {
            uint input_idx = n * C * spatial_size + c * spatial_size + s;
            float diff = input[input_idx] - mean;
            local_var += diff * diff;
        }
    }
    
    // Reduce variance
    shared_var[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_var[tid] += shared_var[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        atomic_fetch_add_explicit(&batch_var[c], shared_var[0], memory_order_relaxed);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    float var = atomic_load_explicit(&batch_var[c], memory_order_relaxed) / float(channel_size);
    
    // Step 3: Normalize and apply affine transform
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;
    float inv_std = 1.0f / sqrt(var + epsilon);
    
    for (uint n = 0; n < N; ++n) {
        for (uint s = tid; s < spatial_size; s += tg_size) {
            uint idx = n * C * spatial_size + c * spatial_size + s;
            output[idx] = g * (input[idx] - mean) * inv_std + b;
        }
    }
    
    // Update running statistics (only one thread per channel)
    if (tid == 0) {
        running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
        running_var[c] = momentum * running_var[c] + (1.0f - momentum) * var;
    }
}

// Layer Normalization
kernel void layer_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    constant uint2& shape [[buffer(5)]],  // [batch_size, normalized_size]
    threadgroup float* shared [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint batch = gid.y;
    const uint batch_size = shape.x;
    const uint norm_size = shape.y;
    
    if (batch >= batch_size) return;
    
    // Step 1: Compute mean
    float local_sum = 0.0f;
    for (uint i = tid; i < norm_size; i += tg_size) {
        local_sum += input[batch * norm_size + i];
    }
    
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to get mean
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = shared[0] / float(norm_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Compute variance
    float local_var = 0.0f;
    for (uint i = tid; i < norm_size; i += tg_size) {
        float diff = input[batch * norm_size + i] - mean;
        local_var += diff * diff;
    }
    
    shared[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to get variance
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float var = shared[0] / float(norm_size);
    float inv_std = 1.0f / sqrt(var + epsilon);
    
    // Step 3: Normalize and apply affine transform
    for (uint i = tid; i < norm_size; i += tg_size) {
        uint idx = batch * norm_size + i;
        float normalized = (input[idx] - mean) * inv_std;
        float g = gamma ? gamma[i] : 1.0f;
        float b = beta ? beta[i] : 0.0f;
        output[idx] = g * normalized + b;
    }
}

// Instance Normalization
kernel void instance_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    constant uint4& shape [[buffer(5)]],  // [N, C, H, W]
    threadgroup float* shared [[threadgroup(0)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint n = gid.z;
    const uint c = gid.y;
    
    const uint N = shape.x;
    const uint C = shape.y;
    const uint H = shape.z;
    const uint W = shape.w;
    const uint spatial_size = H * W;
    
    if (n >= N || c >= C) return;
    
    const uint instance_offset = n * C * spatial_size + c * spatial_size;
    
    // Compute mean for this instance
    float local_sum = 0.0f;
    for (uint i = tid; i < spatial_size; i += tg_size) {
        local_sum += input[instance_offset + i];
    }
    
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce mean
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = shared[0] / float(spatial_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute variance
    float local_var = 0.0f;
    for (uint i = tid; i < spatial_size; i += tg_size) {
        float diff = input[instance_offset + i] - mean;
        local_var += diff * diff;
    }
    
    shared[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce variance
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float var = shared[0] / float(spatial_size);
    float inv_std = 1.0f / sqrt(var + epsilon);
    
    // Normalize
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;
    
    for (uint i = tid; i < spatial_size; i += tg_size) {
        uint idx = instance_offset + i;
        output[idx] = g * (input[idx] - mean) * inv_std + b;
    }
}

// Group Normalization
kernel void group_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    constant uint4& shape [[buffer(5)]],     // [N, C, H, W]
    constant uint& num_groups [[buffer(6)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint n = gid.z;
    const uint g = gid.y;  // group index
    
    const uint N = shape.x;
    const uint C = shape.y;
    const uint H = shape.z;
    const uint W = shape.w;
    const uint spatial_size = H * W;
    
    if (n >= N || g >= num_groups) return;
    
    const uint channels_per_group = C / num_groups;
    const uint group_size = channels_per_group * spatial_size;
    
    // Compute mean for this group
    float local_sum = 0.0f;
    for (uint c = 0; c < channels_per_group; ++c) {
        uint channel = g * channels_per_group + c;
        for (uint s = tid; s < spatial_size; s += tg_size) {
            uint idx = n * C * spatial_size + channel * spatial_size + s;
            local_sum += input[idx];
        }
    }
    
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce mean
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = shared[0] / float(group_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute variance
    float local_var = 0.0f;
    for (uint c = 0; c < channels_per_group; ++c) {
        uint channel = g * channels_per_group + c;
        for (uint s = tid; s < spatial_size; s += tg_size) {
            uint idx = n * C * spatial_size + channel * spatial_size + s;
            float diff = input[idx] - mean;
            local_var += diff * diff;
        }
    }
    
    shared[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce variance
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float var = shared[0] / float(group_size);
    float inv_std = 1.0f / sqrt(var + epsilon);
    
    // Normalize
    for (uint c = 0; c < channels_per_group; ++c) {
        uint channel = g * channels_per_group + c;
        float g_val = gamma ? gamma[channel] : 1.0f;
        float b_val = beta ? beta[channel] : 0.0f;
        
        for (uint s = tid; s < spatial_size; s += tg_size) {
            uint idx = n * C * spatial_size + channel * spatial_size + s;
            output[idx] = g_val * (input[idx] - mean) * inv_std + b_val;
        }
    }
}

// Local Response Normalization (LRN)
kernel void lrn_across_channels(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint4& shape [[buffer(2)]],     // [N, C, H, W]
    constant uint& local_size [[buffer(3)]], // normalization window size
    constant float& alpha [[buffer(4)]],
    constant float& beta [[buffer(5)]],
    constant float& k [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint c = gid.y;
    const uint spatial_idx = gid.x;
    
    const uint N = shape.x;
    const uint C = shape.y;
    const uint spatial_size = shape.z * shape.w;
    
    if (n >= N || c >= C || spatial_idx >= spatial_size) return;
    
    const int radius = local_size / 2;
    const int c_start = max(0, int(c) - radius);
    const int c_end = min(int(C), int(c) + radius + 1);
    
    float sum_sq = 0.0f;
    for (int i = c_start; i < c_end; ++i) {
        uint idx = n * C * spatial_size + i * spatial_size + spatial_idx;
        float val = input[idx];
        sum_sq += val * val;
    }
    
    uint idx = n * C * spatial_size + c * spatial_size + spatial_idx;
    float scale = k + alpha * sum_sq;
    output[idx] = input[idx] / pow(scale, beta);
}

// RMS Normalization (used in some transformer variants)
kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    constant uint2& shape [[buffer(4)]],  // [batch_size, normalized_size]
    threadgroup float* shared [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint batch = gid.y;
    const uint batch_size = shape.x;
    const uint norm_size = shape.y;
    
    if (batch >= batch_size) return;
    
    // Compute RMS
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < norm_size; i += tg_size) {
        float val = input[batch * norm_size + i];
        local_sum_sq += val * val;
    }
    
    shared[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float rms = sqrt(shared[0] / float(norm_size) + epsilon);
    
    // Normalize
    for (uint i = tid; i < norm_size; i += tg_size) {
        uint idx = batch * norm_size + i;
        float g = gamma ? gamma[i] : 1.0f;
        output[idx] = g * input[idx] / rms;
    }
}