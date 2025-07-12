#include <metal_stdlib>
using namespace metal;

// 2D Convolution kernels

// Basic 2D convolution
kernel void conv2d_nchw(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint4& input_shape [[buffer(4)]],   // [N, C, H, W]
    constant uint4& weight_shape [[buffer(5)]],  // [K, C, R, S]
    constant uint4& output_shape [[buffer(6)]],  // [N, K, P, Q]
    constant uint2& stride [[buffer(7)]],        // [stride_h, stride_w]
    constant uint2& padding [[buffer(8)]],       // [pad_h, pad_w]
    constant uint2& dilation [[buffer(9)]],      // [dilation_h, dilation_w]
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;                    // batch index
    const uint k = gid.y;                    // output channel
    const uint out_idx = gid.x;              // linearized output position
    
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
    
    // Convolution loop
    for (uint c = 0; c < C; ++c) {
        for (uint r = 0; r < R; ++r) {
            for (uint s = 0; s < S; ++s) {
                int h = p * stride.x - padding.x + r * dilation.x;
                int w = q * stride.y - padding.y + s * dilation.y;
                
                if (h >= 0 && h < int(H) && w >= 0 && w < int(W)) {
                    uint input_idx = n * C * H * W + c * H * W + h * W + w;
                    uint weight_idx = k * C * R * S + c * R * S + r * S + s;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    uint output_idx = n * K * P * Q + k * P * Q + p * Q + q;
    output[output_idx] = sum;
}

// Depthwise convolution
kernel void depthwise_conv2d_nchw(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint4& input_shape [[buffer(4)]],   // [N, C, H, W]
    constant uint2& kernel_size [[buffer(5)]],   // [R, S]
    constant uint4& output_shape [[buffer(6)]],  // [N, C, P, Q]
    constant uint2& stride [[buffer(7)]],        // [stride_h, stride_w]
    constant uint2& padding [[buffer(8)]],       // [pad_h, pad_w]
    constant uint& depth_multiplier [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint c = gid.y;
    const uint out_idx = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint H = input_shape.z;
    const uint W = input_shape.w;
    
    const uint R = kernel_size.x;
    const uint S = kernel_size.y;
    
    const uint P = output_shape.z;
    const uint Q = output_shape.w;
    
    if (n >= N || c >= C || out_idx >= P * Q) return;
    
    const uint p = out_idx / Q;
    const uint q = out_idx % Q;
    
    float sum = bias ? bias[c] : 0.0f;
    
    // Depthwise convolution
    for (uint r = 0; r < R; ++r) {
        for (uint s = 0; s < S; ++s) {
            int h = p * stride.x - padding.x + r;
            int w = q * stride.y - padding.y + s;
            
            if (h >= 0 && h < int(H) && w >= 0 && w < int(W)) {
                uint input_idx = n * C * H * W + c * H * W + h * W + w;
                uint weight_idx = c * R * S + r * S + s;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    uint output_idx = n * C * P * Q + c * P * Q + p * Q + q;
    output[output_idx] = sum;
}

// 1x1 convolution optimized
kernel void conv2d_1x1_nchw(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint4& input_shape [[buffer(4)]],   // [N, C, H, W]
    constant uint& K [[buffer(5)]],              // output channels
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint k = gid.y;
    const uint spatial_idx = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint spatial_size = input_shape.z * input_shape.w;
    
    if (n >= N || k >= K || spatial_idx >= spatial_size) return;
    
    float sum = bias ? bias[k] : 0.0f;
    
    // 1x1 convolution is just a matrix multiplication
    for (uint c = 0; c < C; ++c) {
        uint input_idx = n * C * spatial_size + c * spatial_size + spatial_idx;
        uint weight_idx = k * C + c;
        sum += input[input_idx] * weight[weight_idx];
    }
    
    uint output_idx = n * K * spatial_size + k * spatial_size + spatial_idx;
    output[output_idx] = sum;
}

// Winograd convolution for 3x3 kernels (F(2,3))
kernel void winograd_transform_input(
    device const float* input [[buffer(0)]],
    device float* transformed [[buffer(1)]],
    constant uint4& input_shape [[buffer(2)]],   // [N, C, H, W]
    constant uint2& tile_size [[buffer(3)]],     // [tile_h, tile_w]
    uint3 gid [[thread_position_in_grid]]) {
    
    // Winograd F(2,3) uses 4x4 input tiles to produce 2x2 output tiles
    const float BT[4][4] = {
        { 1.0f,  0.0f, -1.0f,  0.0f},
        { 0.0f,  1.0f,  1.0f,  0.0f},
        { 0.0f, -1.0f,  1.0f,  0.0f},
        { 0.0f,  1.0f,  0.0f, -1.0f}
    };
    
    const uint n = gid.z;
    const uint c = gid.y;
    const uint tile_idx = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint H = input_shape.z;
    const uint W = input_shape.w;
    
    const uint tiles_h = (H + 1) / 2;  // Output tile size is 2x2
    const uint tiles_w = (W + 1) / 2;
    const uint total_tiles = tiles_h * tiles_w;
    
    if (n >= N || c >= C || tile_idx >= total_tiles) return;
    
    const uint tile_h = tile_idx / tiles_w;
    const uint tile_w = tile_idx % tiles_w;
    
    // Load 4x4 input tile
    float tile[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            int h = tile_h * 2 + i - 1;  // Account for padding
            int w = tile_w * 2 + j - 1;
            
            if (h >= 0 && h < int(H) && w >= 0 && w < int(W)) {
                tile[i][j] = input[n * C * H * W + c * H * W + h * W + w];
            } else {
                tile[i][j] = 0.0f;
            }
        }
    }
    
    // Transform: V = BT * d * B
    float temp[4][4];
    
    // temp = tile * B
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            temp[i][j] = 0.0f;
            for (uint k = 0; k < 4; ++k) {
                temp[i][j] += tile[i][k] * BT[j][k];  // B = BT^T
            }
        }
    }
    
    // transformed = BT * temp
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            float val = 0.0f;
            for (uint k = 0; k < 4; ++k) {
                val += BT[i][k] * temp[k][j];
            }
            
            // Store in transformed format: [16, N, C, tiles]
            uint idx = (i * 4 + j) * N * C * total_tiles + 
                      n * C * total_tiles + c * total_tiles + tile_idx;
            transformed[idx] = val;
        }
    }
}

// 3D Convolution
kernel void conv3d_ncdhw(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint* input_shape [[buffer(4)]],   // [N, C, D, H, W]
    constant uint* weight_shape [[buffer(5)]],  // [K, C, T, R, S]
    constant uint* output_shape [[buffer(6)]],  // [N, K, D', H', W']
    constant uint3& stride [[buffer(7)]],       // [stride_d, stride_h, stride_w]
    constant uint3& padding [[buffer(8)]],      // [pad_d, pad_h, pad_w]
    constant uint3& dilation [[buffer(9)]],     // [dilation_d, dilation_h, dilation_w]
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint k = gid.y;
    const uint out_idx = gid.x;
    
    const uint N = input_shape[0];
    const uint C = input_shape[1];
    const uint D = input_shape[2];
    const uint H = input_shape[3];
    const uint W = input_shape[4];
    
    const uint K = weight_shape[0];
    const uint T = weight_shape[2];
    const uint R = weight_shape[3];
    const uint S = weight_shape[4];
    
    const uint D_out = output_shape[2];
    const uint H_out = output_shape[3];
    const uint W_out = output_shape[4];
    const uint spatial_out = D_out * H_out * W_out;
    
    if (n >= N || k >= K || out_idx >= spatial_out) return;
    
    // Decompose output index
    const uint d_out = out_idx / (H_out * W_out);
    const uint h_out = (out_idx / W_out) % H_out;
    const uint w_out = out_idx % W_out;
    
    float sum = bias ? bias[k] : 0.0f;
    
    // 3D convolution loop
    for (uint c = 0; c < C; ++c) {
        for (uint t = 0; t < T; ++t) {
            for (uint r = 0; r < R; ++r) {
                for (uint s = 0; s < S; ++s) {
                    int d = d_out * stride.x - padding.x + t * dilation.x;
                    int h = h_out * stride.y - padding.y + r * dilation.y;
                    int w = w_out * stride.z - padding.z + s * dilation.z;
                    
                    if (d >= 0 && d < int(D) && h >= 0 && h < int(H) && w >= 0 && w < int(W)) {
                        uint input_idx = n * C * D * H * W + c * D * H * W + 
                                       d * H * W + h * W + w;
                        uint weight_idx = k * C * T * R * S + c * T * R * S + 
                                        t * R * S + r * S + s;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    uint output_idx = n * K * spatial_out + k * spatial_out + out_idx;
    output[output_idx] = sum;
}

// Group convolution
kernel void group_conv2d_nchw(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint4& input_shape [[buffer(4)]],   // [N, C, H, W]
    constant uint4& weight_shape [[buffer(5)]],  // [K, C/G, R, S]
    constant uint4& output_shape [[buffer(6)]],  // [N, K, P, Q]
    constant uint2& stride [[buffer(7)]],
    constant uint2& padding [[buffer(8)]],
    constant uint& groups [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    const uint n = gid.z;
    const uint k = gid.y;
    const uint out_idx = gid.x;
    
    const uint N = input_shape.x;
    const uint C = input_shape.y;
    const uint H = input_shape.z;
    const uint W = input_shape.w;
    
    const uint K = weight_shape.x;
    const uint C_per_group = C / groups;
    const uint K_per_group = K / groups;
    const uint R = weight_shape.z;
    const uint S = weight_shape.w;
    
    const uint P = output_shape.z;
    const uint Q = output_shape.w;
    
    if (n >= N || k >= K || out_idx >= P * Q) return;
    
    const uint p = out_idx / Q;
    const uint q = out_idx % Q;
    const uint g = k / K_per_group;  // Group index
    
    float sum = bias ? bias[k] : 0.0f;
    
    // Group convolution
    for (uint c = 0; c < C_per_group; ++c) {
        uint c_in = g * C_per_group + c;
        for (uint r = 0; r < R; ++r) {
            for (uint s = 0; s < S; ++s) {
                int h = p * stride.x - padding.x + r;
                int w = q * stride.y - padding.y + s;
                
                if (h >= 0 && h < int(H) && w >= 0 && w < int(W)) {
                    uint input_idx = n * C * H * W + c_in * H * W + h * W + w;
                    uint weight_idx = k * C_per_group * R * S + c * R * S + r * S + s;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    uint output_idx = n * K * P * Q + k * P * Q + p * Q + q;
    output[output_idx] = sum;
}