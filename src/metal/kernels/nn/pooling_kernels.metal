#include <metal_stdlib>
using namespace metal;

// Pooling kernels

// 2D Max Pooling
kernel void maxpool2d_nchw(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint4& input_shape [[buffer(2)]],   // [N, C, H, W]
    constant uint4& output_shape [[buffer(3)]],  // [N, C, P, Q]
    constant uint2& kernel_size [[buffer(4)]],   // [pool_h, pool_w]
    constant uint2& stride [[buffer(5)]],        // [stride_h, stride_w]
    constant uint2& padding [[buffer(6)]],       // [pad_h, pad_w]
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint c = gid.y;
    const uint out_idx = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint H = input_shape.z;
    const uint W = input_shape.w;
    
    const uint P = output_shape.z;
    const uint Q = output_shape.w;
    
    if (n >= N || c >= C || out_idx >= P * Q) return;
    
    const uint p = out_idx / Q;
    const uint q = out_idx % Q;
    
    float max_val = -INFINITY;
    
    for (uint kh = 0; kh < kernel_size.x; ++kh) {
        for (uint kw = 0; kw < kernel_size.y; ++kw) {
            int h = p * stride.x - padding.x + kh;
            int w = q * stride.y - padding.y + kw;
            
            if (h >= 0 && h < int(H) && w >= 0 && w < int(W)) {
                uint input_idx = n * C * H * W + c * H * W + h * W + w;
                max_val = max(max_val, input[input_idx]);
            }
        }
    }
    
    uint output_idx = n * C * P * Q + c * P * Q + p * Q + q;
    output[output_idx] = max_val;
}

// 2D Max Pooling with indices (for unpooling)
kernel void maxpool2d_with_indices_nchw(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device int* indices [[buffer(2)]],
    constant uint4& input_shape [[buffer(3)]],
    constant uint4& output_shape [[buffer(4)]],
    constant uint2& kernel_size [[buffer(5)]],
    constant uint2& stride [[buffer(6)]],
    constant uint2& padding [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint c = gid.y;
    const uint out_idx = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint H = input_shape.z;
    const uint W = input_shape.w;
    
    const uint P = output_shape.z;
    const uint Q = output_shape.w;
    
    if (n >= N || c >= C || out_idx >= P * Q) return;
    
    const uint p = out_idx / Q;
    const uint q = out_idx % Q;
    
    float max_val = -INFINITY;
    int max_idx = -1;
    
    for (uint kh = 0; kh < kernel_size.x; ++kh) {
        for (uint kw = 0; kw < kernel_size.y; ++kw) {
            int h = p * stride.x - padding.x + kh;
            int w = q * stride.y - padding.y + kw;
            
            if (h >= 0 && h < int(H) && w >= 0 && w < int(W)) {
                uint input_idx = n * C * H * W + c * H * W + h * W + w;
                float val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = h * W + w;  // Store flattened HW index
                }
            }
        }
    }
    
    uint output_idx = n * C * P * Q + c * P * Q + p * Q + q;
    output[output_idx] = max_val;
    indices[output_idx] = max_idx;
}

// 2D Average Pooling
kernel void avgpool2d_nchw(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint4& input_shape [[buffer(2)]],
    constant uint4& output_shape [[buffer(3)]],
    constant uint2& kernel_size [[buffer(4)]],
    constant uint2& stride [[buffer(5)]],
    constant uint2& padding [[buffer(6)]],
    constant bool& count_include_pad [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint c = gid.y;
    const uint out_idx = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint H = input_shape.z;
    const uint W = input_shape.w;
    
    const uint P = output_shape.z;
    const uint Q = output_shape.w;
    
    if (n >= N || c >= C || out_idx >= P * Q) return;
    
    const uint p = out_idx / Q;
    const uint q = out_idx % Q;
    
    float sum = 0.0f;
    uint count = 0;
    
    for (uint kh = 0; kh < kernel_size.x; ++kh) {
        for (uint kw = 0; kw < kernel_size.y; ++kw) {
            int h = p * stride.x - padding.x + kh;
            int w = q * stride.y - padding.y + kw;
            
            if (h >= 0 && h < int(H) && w >= 0 && w < int(W)) {
                uint input_idx = n * C * H * W + c * H * W + h * W + w;
                sum += input[input_idx];
                count++;
            } else if (count_include_pad) {
                count++;
            }
        }
    }
    
    uint output_idx = n * C * P * Q + c * P * Q + p * Q + q;
    output[output_idx] = count > 0 ? sum / float(count) : 0.0f;
}

// Global Average Pooling
kernel void global_avgpool2d_nchw(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint4& input_shape [[buffer(2)]],  // [N, C, H, W]
    uint2 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.y;
    const uint c = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint H = input_shape.z;
    const uint W = input_shape.w;
    
    if (n >= N || c >= C) return;
    
    float sum = 0.0f;
    const uint spatial_size = H * W;
    const uint offset = n * C * spatial_size + c * spatial_size;
    
    for (uint i = 0; i < spatial_size; ++i) {
        sum += input[offset + i];
    }
    
    output[n * C + c] = sum / float(spatial_size);
}

// Global Max Pooling
kernel void global_maxpool2d_nchw(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint4& input_shape [[buffer(2)]],  // [N, C, H, W]
    uint2 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.y;
    const uint c = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint H = input_shape.z;
    const uint W = input_shape.w;
    
    if (n >= N || c >= C) return;
    
    float max_val = -INFINITY;
    const uint spatial_size = H * W;
    const uint offset = n * C * spatial_size + c * spatial_size;
    
    for (uint i = 0; i < spatial_size; ++i) {
        max_val = max(max_val, input[offset + i]);
    }
    
    output[n * C + c] = max_val;
}

// 3D Max Pooling
kernel void maxpool3d_ncdhw(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],   // [N, C, D, H, W]
    constant uint* output_shape [[buffer(3)]],  // [N, C, D', H', W']
    constant uint3& kernel_size [[buffer(4)]],  // [pool_d, pool_h, pool_w]
    constant uint3& stride [[buffer(5)]],       // [stride_d, stride_h, stride_w]
    constant uint3& padding [[buffer(6)]],      // [pad_d, pad_h, pad_w]
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint c = gid.y;
    const uint out_idx = gid.x;
    
    const uint N = input_shape[0];
    const uint C = input_shape[1];
    const uint D = input_shape[2];
    const uint H = input_shape[3];
    const uint W = input_shape[4];
    
    const uint D_out = output_shape[2];
    const uint H_out = output_shape[3];
    const uint W_out = output_shape[4];
    const uint spatial_out = D_out * H_out * W_out;
    
    if (n >= N || c >= C || out_idx >= spatial_out) return;
    
    const uint d_out = out_idx / (H_out * W_out);
    const uint h_out = (out_idx / W_out) % H_out;
    const uint w_out = out_idx % W_out;
    
    float max_val = -INFINITY;
    
    for (uint kd = 0; kd < kernel_size.x; ++kd) {
        for (uint kh = 0; kh < kernel_size.y; ++kh) {
            for (uint kw = 0; kw < kernel_size.z; ++kw) {
                int d = d_out * stride.x - padding.x + kd;
                int h = h_out * stride.y - padding.y + kh;
                int w = w_out * stride.z - padding.z + kw;
                
                if (d >= 0 && d < int(D) && h >= 0 && h < int(H) && w >= 0 && w < int(W)) {
                    uint input_idx = n * C * D * H * W + c * D * H * W + 
                                   d * H * W + h * W + w;
                    max_val = max(max_val, input[input_idx]);
                }
            }
        }
    }
    
    uint output_idx = n * C * spatial_out + c * spatial_out + out_idx;
    output[output_idx] = max_val;
}

// Adaptive Average Pooling 2D
kernel void adaptive_avgpool2d_nchw(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint4& input_shape [[buffer(2)]],   // [N, C, H, W]
    constant uint4& output_shape [[buffer(3)]],  // [N, C, OH, OW]
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint c = gid.y;
    const uint out_idx = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint IH = input_shape.z;
    const uint IW = input_shape.w;
    
    const uint OH = output_shape.z;
    const uint OW = output_shape.w;
    
    if (n >= N || c >= C || out_idx >= OH * OW) return;
    
    const uint oh = out_idx / OW;
    const uint ow = out_idx % OW;
    
    // Calculate input region for this output position
    const uint ih_start = oh * IH / OH;
    const uint ih_end = (oh + 1) * IH / OH;
    const uint iw_start = ow * IW / OW;
    const uint iw_end = (ow + 1) * IW / OW;
    
    float sum = 0.0f;
    uint count = 0;
    
    for (uint ih = ih_start; ih < ih_end; ++ih) {
        for (uint iw = iw_start; iw < iw_end; ++iw) {
            uint input_idx = n * C * IH * IW + c * IH * IW + ih * IW + iw;
            sum += input[input_idx];
            count++;
        }
    }
    
    uint output_idx = n * C * OH * OW + c * OH * OW + oh * OW + ow;
    output[output_idx] = count > 0 ? sum / float(count) : 0.0f;
}

// ROI Pooling
kernel void roi_pool2d(
    device const float* input [[buffer(0)]],
    device const float* rois [[buffer(1)]],     // [num_rois, 5] format: [batch_idx, x1, y1, x2, y2]
    device float* output [[buffer(2)]],
    constant uint4& input_shape [[buffer(3)]],  // [N, C, H, W]
    constant uint2& output_size [[buffer(4)]],  // [pooled_h, pooled_w]
    constant float& spatial_scale [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint roi_idx = gid.z;
    const uint c = gid.y;
    const uint out_idx = gid.x;
    
    const uint C = input_shape.y;
    const uint H = input_shape.z;
    const uint W = input_shape.w;
    
    const uint pooled_h = output_size.x;
    const uint pooled_w = output_size.y;
    
    if (c >= C || out_idx >= pooled_h * pooled_w) return;
    
    const uint ph = out_idx / pooled_w;
    const uint pw = out_idx % pooled_w;
    
    // Get ROI coordinates
    const uint roi_offset = roi_idx * 5;
    const int batch_idx = int(rois[roi_offset]);
    const float x1 = rois[roi_offset + 1] * spatial_scale;
    const float y1 = rois[roi_offset + 2] * spatial_scale;
    const float x2 = rois[roi_offset + 3] * spatial_scale;
    const float y2 = rois[roi_offset + 4] * spatial_scale;
    
    const float roi_w = max(x2 - x1, 1.0f);
    const float roi_h = max(y2 - y1, 1.0f);
    
    const float bin_h = roi_h / float(pooled_h);
    const float bin_w = roi_w / float(pooled_w);
    
    const int h_start = int(floor(y1 + ph * bin_h));
    const int h_end = int(ceil(y1 + (ph + 1) * bin_h));
    const int w_start = int(floor(x1 + pw * bin_w));
    const int w_end = int(ceil(x1 + (pw + 1) * bin_w));
    
    const int h_start_clamp = max(h_start, 0);
    const int h_end_clamp = min(h_end, int(H));
    const int w_start_clamp = max(w_start, 0);
    const int w_end_clamp = min(w_end, int(W));
    
    float max_val = -INFINITY;
    
    for (int h = h_start_clamp; h < h_end_clamp; ++h) {
        for (int w = w_start_clamp; w < w_end_clamp; ++w) {
            uint input_idx = batch_idx * C * H * W + c * H * W + h * W + w;
            max_val = max(max_val, input[input_idx]);
        }
    }
    
    uint output_idx = roi_idx * C * pooled_h * pooled_w + 
                     c * pooled_h * pooled_w + ph * pooled_w + pw;
    output[output_idx] = max_val;
}

// Unpooling (Max Unpooling)
kernel void max_unpool2d_nchw(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint4& input_shape [[buffer(3)]],   // [N, C, H, W] (pooled)
    constant uint4& output_shape [[buffer(4)]],  // [N, C, H', W'] (unpooled)
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint c = gid.y;
    const uint idx = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint H = input_shape.z;
    const uint W = input_shape.w;
    
    const uint H_out = output_shape.z;
    const uint W_out = output_shape.w;
    
    if (n >= N || c >= C || idx >= H * W) return;
    
    const uint input_idx = n * C * H * W + c * H * W + idx;
    const int hw_idx = indices[input_idx];
    
    if (hw_idx >= 0) {
        const uint h_out = hw_idx / W_out;
        const uint w_out = hw_idx % W_out;
        const uint output_idx = n * C * H_out * W_out + c * H_out * W_out + 
                               h_out * W_out + w_out;
        output[output_idx] = input[input_idx];
    }
}