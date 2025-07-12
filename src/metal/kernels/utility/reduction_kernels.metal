#include <metal_stdlib>
using namespace metal;

// Reduction kernels

// Sum reduction along last dimension
kernel void reduce_sum_last_dim(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    constant uint2& shape [[buffer(2)]],  // [outer_size, reduce_size]
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint outer_idx = gid.y;
    const uint outer_size = shape.x;
    const uint reduce_size = shape.y;
    
    if (outer_idx >= outer_size) return;
    
    // Each thread accumulates partial sum
    float local_sum = 0.0f;
    for (uint i = tid; i < reduce_size; i += tg_size) {
        local_sum += input[outer_idx * reduce_size + i];
    }
    
    // Store in shared memory
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        output[outer_idx] = shared[0];
    }
}

// Generic sum reduction
kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    constant uint* input_shape [[buffer(2)]],
    constant uint* output_shape [[buffer(3)]],
    constant uint& reduce_dim [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    // Calculate strides
    uint input_strides[6];
    uint output_strides[6];
    input_strides[ndim - 1] = 1;
    output_strides[ndim - 1] = 1;
    
    for (int i = ndim - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }
    
    // Get output position
    uint output_idx = gid.z;
    if (output_idx >= output_strides[0] * output_shape[0]) return;
    
    // Decompose output index
    uint output_coords[6] = {0};
    uint idx = output_idx;
    for (uint i = 0; i < ndim; ++i) {
        if (i != reduce_dim) {
            output_coords[i] = idx / output_strides[i];
            idx %= output_strides[i];
        }
    }
    
    // Sum over reduction dimension
    float local_sum = 0.0f;
    for (uint r = tid; r < input_shape[reduce_dim]; r += tg_size) {
        uint input_idx = 0;
        for (uint i = 0; i < ndim; ++i) {
            uint coord = (i == reduce_dim) ? r : output_coords[i];
            input_idx += coord * input_strides[i];
        }
        local_sum += input[input_idx];
    }
    
    // Reduce within threadgroup
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Atomic add to output
    if (tid == 0) {
        atomic_fetch_add_explicit(&output[output_idx], shared[0], memory_order_relaxed);
    }
}

// Mean reduction
kernel void reduce_mean(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    constant uint2& shape [[buffer(2)]],  // [outer_size, reduce_size]
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint outer_idx = gid.y;
    const uint outer_size = shape.x;
    const uint reduce_size = shape.y;
    
    if (outer_idx >= outer_size) return;
    
    // Sum reduction
    float local_sum = 0.0f;
    for (uint i = tid; i < reduce_size; i += tg_size) {
        local_sum += input[outer_idx * reduce_size + i];
    }
    
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute mean
    if (tid == 0) {
        output[outer_idx] = shared[0] / float(reduce_size);
    }
}

// Max reduction
kernel void reduce_max(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    constant uint2& shape [[buffer(2)]],  // [outer_size, reduce_size]
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint outer_idx = gid.y;
    const uint outer_size = shape.x;
    const uint reduce_size = shape.y;
    
    if (outer_idx >= outer_size) return;
    
    // Find local maximum
    float local_max = -INFINITY;
    for (uint i = tid; i < reduce_size; i += tg_size) {
        local_max = max(local_max, input[outer_idx * reduce_size + i]);
    }
    
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction for maximum
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[outer_idx] = shared[0];
    }
}

// Min reduction
kernel void reduce_min(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    constant uint2& shape [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint outer_idx = gid.y;
    const uint outer_size = shape.x;
    const uint reduce_size = shape.y;
    
    if (outer_idx >= outer_size) return;
    
    float local_min = INFINITY;
    for (uint i = tid; i < reduce_size; i += tg_size) {
        local_min = min(local_min, input[outer_idx * reduce_size + i]);
    }
    
    shared[tid] = local_min;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = min(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[outer_idx] = shared[0];
    }
}

// Product reduction
kernel void reduce_prod(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    constant uint2& shape [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint outer_idx = gid.y;
    const uint outer_size = shape.x;
    const uint reduce_size = shape.y;
    
    if (outer_idx >= outer_size) return;
    
    float local_prod = 1.0f;
    for (uint i = tid; i < reduce_size; i += tg_size) {
        local_prod *= input[outer_idx * reduce_size + i];
    }
    
    shared[tid] = local_prod;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] *= shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[outer_idx] = shared[0];
    }
}

// Argmax reduction
kernel void reduce_argmax(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    threadgroup float* shared_val [[threadgroup(0)]],
    threadgroup int* shared_idx [[threadgroup(1)]],
    constant uint2& shape [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint outer_idx = gid.y;
    const uint outer_size = shape.x;
    const uint reduce_size = shape.y;
    
    if (outer_idx >= outer_size) return;
    
    float local_max = -INFINITY;
    int local_idx = 0;
    
    for (uint i = tid; i < reduce_size; i += tg_size) {
        float val = input[outer_idx * reduce_size + i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }
    
    shared_val[tid] = local_max;
    shared_idx[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (shared_val[tid + stride] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[outer_idx] = shared_idx[0];
    }
}

// Variance reduction (using Welford's algorithm for numerical stability)
kernel void reduce_variance(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* mean_output [[buffer(2)]],
    threadgroup float* shared_mean [[threadgroup(0)]],
    threadgroup float* shared_m2 [[threadgroup(1)]],
    threadgroup uint* shared_count [[threadgroup(2)]],
    constant uint2& shape [[buffer(3)]],
    constant bool& unbiased [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint outer_idx = gid.y;
    const uint outer_size = shape.x;
    const uint reduce_size = shape.y;
    
    if (outer_idx >= outer_size) return;
    
    // Welford's algorithm
    float mean = 0.0f;
    float m2 = 0.0f;
    uint count = 0;
    
    for (uint i = tid; i < reduce_size; i += tg_size) {
        float val = input[outer_idx * reduce_size + i];
        count++;
        float delta = val - mean;
        mean += delta / float(count);
        float delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    shared_mean[tid] = mean;
    shared_m2[tid] = m2;
    shared_count[tid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction of Welford's algorithm
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < tg_size) {
            float mean1 = shared_mean[tid];
            float mean2 = shared_mean[tid + stride];
            uint count1 = shared_count[tid];
            uint count2 = shared_count[tid + stride];
            
            uint total_count = count1 + count2;
            if (total_count > 0) {
                float delta = mean2 - mean1;
                float combined_mean = (count1 * mean1 + count2 * mean2) / float(total_count);
                float combined_m2 = shared_m2[tid] + shared_m2[tid + stride] + 
                                   delta * delta * count1 * count2 / float(total_count);
                
                shared_mean[tid] = combined_mean;
                shared_m2[tid] = combined_m2;
                shared_count[tid] = total_count;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        float variance = shared_m2[0] / float(unbiased ? reduce_size - 1 : reduce_size);
        output[outer_idx] = variance;
        if (mean_output) {
            mean_output[outer_idx] = shared_mean[0];
        }
    }
}

// L2 norm reduction
kernel void reduce_l2_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    constant uint2& shape [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint outer_idx = gid.y;
    const uint outer_size = shape.x;
    const uint reduce_size = shape.y;
    
    if (outer_idx >= outer_size) return;
    
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < reduce_size; i += tg_size) {
        float val = input[outer_idx * reduce_size + i];
        local_sum_sq += val * val;
    }
    
    shared[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[outer_idx] = sqrt(shared[0]);
    }
}

// Cumulative sum (prefix sum)
kernel void cumsum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    constant uint2& shape [[buffer(2)]],  // [batch_size, sequence_length]
    constant bool& exclusive [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint batch = gid.y;
    const uint batch_size = shape.x;
    const uint seq_len = shape.y;
    
    if (batch >= batch_size) return;
    
    const uint offset = batch * seq_len;
    
    // Simple sequential scan for now
    // For larger sequences, would use parallel prefix sum algorithm
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < seq_len; ++i) {
            if (exclusive) {
                output[offset + i] = sum;
                sum += input[offset + i];
            } else {
                sum += input[offset + i];
                output[offset + i] = sum;
            }
        }
    }
}

// Any (logical OR reduction)
kernel void reduce_any(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    threadgroup int* shared [[threadgroup(0)]],
    constant uint2& shape [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint outer_idx = gid.y;
    const uint outer_size = shape.x;
    const uint reduce_size = shape.y;
    
    if (outer_idx >= outer_size) return;
    
    int local_any = 0;
    for (uint i = tid; i < reduce_size; i += tg_size) {
        if (input[outer_idx * reduce_size + i] != 0.0f) {
            local_any = 1;
            break;
        }
    }
    
    shared[tid] = local_any;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = shared[tid] || shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[outer_idx] = shared[0];
    }
}

// All (logical AND reduction)
kernel void reduce_all(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    threadgroup int* shared [[threadgroup(0)]],
    constant uint2& shape [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    const uint outer_idx = gid.y;
    const uint outer_size = shape.x;
    const uint reduce_size = shape.y;
    
    if (outer_idx >= outer_size) return;
    
    int local_all = 1;
    for (uint i = tid; i < reduce_size; i += tg_size) {
        if (input[outer_idx * reduce_size + i] == 0.0f) {
            local_all = 0;
            break;
        }
    }
    
    shared[tid] = local_all;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = shared[tid] && shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[outer_idx] = shared[0];
    }
}