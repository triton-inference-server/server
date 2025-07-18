#include <metal_stdlib>
using namespace metal;

// Transformation and utility kernels

// Transpose 2D
kernel void transpose_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint2& shape [[buffer(2)]],  // [rows, cols]
    uint2 gid [[thread_position_in_grid]]) {
    
    const uint row = gid.y;
    const uint col = gid.x;
    const uint rows = shape.x;
    const uint cols = shape.y;
    
    if (row >= rows || col >= cols) return;
    
    // input[row][col] -> output[col][row]
    uint input_idx = row * cols + col;
    uint output_idx = col * rows + row;
    output[output_idx] = input[input_idx];
}

// Permute/Transpose for arbitrary dimensions (up to 6D)
kernel void permute(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],
    constant uint* output_shape [[buffer(3)]],
    constant uint* permutation [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid >= output_shape[0]) return;  // Total size check
    
    // Convert linear index to multi-dimensional output coordinates
    uint output_coords[6] = {0};
    uint idx = gid;
    for (int i = ndim - 1; i >= 0; --i) {
        output_coords[i] = idx % output_shape[i];
        idx /= output_shape[i];
    }
    
    // Map to input coordinates using permutation
    uint input_coords[6] = {0};
    for (uint i = 0; i < ndim; ++i) {
        input_coords[permutation[i]] = output_coords[i];
    }
    
    // Convert input coordinates to linear index
    uint input_idx = 0;
    uint stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        input_idx += input_coords[i] * stride;
        stride *= input_shape[i];
    }
    
    output[gid] = input[input_idx];
}

// Reshape (simple copy)
kernel void reshape(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        output[gid] = input[gid];
    }
}

// Concatenate along a dimension
kernel void concat(
    device const float* const* inputs [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* input_sizes [[buffer(2)]],  // Size along concat dimension for each input
    constant uint& num_inputs [[buffer(3)]],
    constant uint& concat_dim [[buffer(4)]],
    constant uint& pre_concat_size [[buffer(5)]],   // Product of dims before concat_dim
    constant uint& post_concat_size [[buffer(6)]],  // Product of dims after concat_dim
    uint gid [[thread_position_in_grid]]) {
    
    // Decompose output index
    uint pre_idx = gid / (post_concat_size * input_sizes[num_inputs]);  // Total concat size
    uint concat_idx = (gid / post_concat_size) % input_sizes[num_inputs];
    uint post_idx = gid % post_concat_size;
    
    // Find which input tensor and offset within it
    uint input_idx = 0;
    uint offset = 0;
    for (uint i = 0; i < num_inputs; ++i) {
        if (concat_idx < offset + input_sizes[i]) {
            input_idx = i;
            concat_idx -= offset;
            break;
        }
        offset += input_sizes[i];
    }
    
    // Compute input index
    uint idx = pre_idx * input_sizes[input_idx] * post_concat_size +
               concat_idx * post_concat_size + post_idx;
    
    output[gid] = inputs[input_idx][idx];
}

// Split along a dimension
kernel void split(
    device const float* input [[buffer(0)]],
    device float* const* outputs [[buffer(1)]],
    constant uint* split_sizes [[buffer(2)]],
    constant uint& num_outputs [[buffer(3)]],
    constant uint& split_dim [[buffer(4)]],
    constant uint& pre_split_size [[buffer(5)]],
    constant uint& post_split_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    
    // Decompose input index
    uint pre_idx = gid / (post_split_size * split_sizes[num_outputs]);
    uint split_idx = (gid / post_split_size) % split_sizes[num_outputs];
    uint post_idx = gid % post_split_size;
    
    // Find which output tensor
    uint output_idx = 0;
    uint offset = 0;
    for (uint i = 0; i < num_outputs; ++i) {
        if (split_idx < offset + split_sizes[i]) {
            output_idx = i;
            split_idx -= offset;
            break;
        }
        offset += split_sizes[i];
    }
    
    // Compute output index
    uint idx = pre_idx * split_sizes[output_idx] * post_split_size +
               split_idx * post_split_size + post_idx;
    
    outputs[output_idx][idx] = input[gid];
}

// Slice extraction
kernel void slice(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],
    constant uint* starts [[buffer(3)]],
    constant uint* ends [[buffer(4)]],
    constant uint* steps [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    
    // Convert output linear index to coordinates
    uint output_coords[6] = {0};
    uint idx = gid;
    uint output_strides[6];
    
    // Compute output shape and strides
    output_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        uint dim_size = (ends[i + 1] - starts[i + 1] + steps[i + 1] - 1) / steps[i + 1];
        output_strides[i] = output_strides[i + 1] * dim_size;
    }
    
    for (int i = ndim - 1; i >= 0; --i) {
        uint dim_size = (ends[i] - starts[i] + steps[i] - 1) / steps[i];
        output_coords[i] = idx % dim_size;
        idx /= dim_size;
    }
    
    // Map to input coordinates
    uint input_idx = 0;
    uint input_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        uint input_coord = starts[i] + output_coords[i] * steps[i];
        input_idx += input_coord * input_stride;
        input_stride *= input_shape[i];
    }
    
    output[gid] = input[input_idx];
}

// Padding
kernel void pad_constant(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],
    constant uint* output_shape [[buffer(3)]],
    constant uint* pad_before [[buffer(4)]],
    constant uint* pad_after [[buffer(5)]],
    constant float& pad_value [[buffer(6)]],
    constant uint& ndim [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    
    // Convert output index to coordinates
    uint output_coords[6] = {0};
    uint idx = gid;
    for (int i = ndim - 1; i >= 0; --i) {
        output_coords[i] = idx % output_shape[i];
        idx /= output_shape[i];
    }
    
    // Check if within input bounds
    bool in_bounds = true;
    uint input_idx = 0;
    uint input_stride = 1;
    
    for (int i = ndim - 1; i >= 0; --i) {
        if (output_coords[i] < pad_before[i] || 
            output_coords[i] >= pad_before[i] + input_shape[i]) {
            in_bounds = false;
            break;
        }
        uint input_coord = output_coords[i] - pad_before[i];
        input_idx += input_coord * input_stride;
        input_stride *= input_shape[i];
    }
    
    output[gid] = in_bounds ? input[input_idx] : pad_value;
}

// Pad with reflection
kernel void pad_reflect(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],
    constant uint* output_shape [[buffer(3)]],
    constant uint* pad_before [[buffer(4)]],
    constant uint* pad_after [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    
    // Convert output index to coordinates
    uint output_coords[6] = {0};
    uint idx = gid;
    for (int i = ndim - 1; i >= 0; --i) {
        output_coords[i] = idx % output_shape[i];
        idx /= output_shape[i];
    }
    
    // Map to input coordinates with reflection
    uint input_idx = 0;
    uint input_stride = 1;
    
    for (int i = ndim - 1; i >= 0; --i) {
        int coord = int(output_coords[i]) - int(pad_before[i]);
        
        // Reflect at boundaries
        if (coord < 0) {
            coord = -coord;
        } else if (coord >= int(input_shape[i])) {
            coord = 2 * int(input_shape[i]) - coord - 2;
        }
        
        // Clamp to valid range
        coord = clamp(coord, 0, int(input_shape[i]) - 1);
        
        input_idx += uint(coord) * input_stride;
        input_stride *= input_shape[i];
    }
    
    output[gid] = input[input_idx];
}

// Type casting kernels
kernel void cast_float_to_half(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        output[gid] = half(input[gid]);
    }
}

kernel void cast_half_to_float(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        output[gid] = float(input[gid]);
    }
}

kernel void cast_float_to_int(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        output[gid] = int(input[gid]);
    }
}

kernel void cast_int_to_float(
    device const int* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid < size) {
        output[gid] = float(input[gid]);
    }
}

// Gather operation
kernel void gather(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& gather_dim_size [[buffer(4)]],
    constant uint& indices_size [[buffer(5)]],
    constant uint& inner_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    
    const uint batch = gid / (indices_size * inner_size);
    const uint idx = (gid / inner_size) % indices_size;
    const uint inner = gid % inner_size;
    
    if (batch >= batch_size) return;
    
    int gather_idx = indices[batch * indices_size + idx];
    
    // Handle negative indices
    if (gather_idx < 0) {
        gather_idx += gather_dim_size;
    }
    
    // Clamp to valid range
    gather_idx = clamp(gather_idx, 0, int(gather_dim_size) - 1);
    
    uint input_idx = batch * gather_dim_size * inner_size + 
                    gather_idx * inner_size + inner;
    output[gid] = input[input_idx];
}

// Scatter operation
kernel void scatter_add(
    device const float* updates [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device atomic_float* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& scatter_dim_size [[buffer(4)]],
    constant uint& indices_size [[buffer(5)]],
    constant uint& inner_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    
    const uint batch = gid / (indices_size * inner_size);
    const uint idx = (gid / inner_size) % indices_size;
    const uint inner = gid % inner_size;
    
    if (batch >= batch_size) return;
    
    int scatter_idx = indices[batch * indices_size + idx];
    
    // Handle negative indices
    if (scatter_idx < 0) {
        scatter_idx += scatter_dim_size;
    }
    
    // Skip invalid indices
    if (scatter_idx < 0 || scatter_idx >= int(scatter_dim_size)) {
        return;
    }
    
    uint output_idx = batch * scatter_dim_size * inner_size + 
                     scatter_idx * inner_size + inner;
    
    atomic_fetch_add_explicit(&output[output_idx], updates[gid], memory_order_relaxed);
}

// One-hot encoding
kernel void one_hot(
    device const int* indices [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_classes [[buffer(2)]],
    constant float& on_value [[buffer(3)]],
    constant float& off_value [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    
    const uint idx = gid.y;
    const uint class_idx = gid.x;
    
    if (class_idx >= num_classes) return;
    
    int index = indices[idx];
    
    // Handle negative indices
    if (index < 0) {
        index += num_classes;
    }
    
    output[idx * num_classes + class_idx] = 
        (index == int(class_idx)) ? on_value : off_value;
}

// Tile/Repeat operation
kernel void tile(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint* input_shape [[buffer(2)]],
    constant uint* repeats [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    
    // Convert output index to input index
    uint input_idx = 0;
    uint idx = gid;
    uint input_stride = 1;
    
    for (int i = ndim - 1; i >= 0; --i) {
        uint output_coord = idx % (input_shape[i] * repeats[i]);
        uint input_coord = output_coord % input_shape[i];
        
        input_idx += input_coord * input_stride;
        input_stride *= input_shape[i];
        idx /= (input_shape[i] * repeats[i]);
    }
    
    output[gid] = input[input_idx];
}