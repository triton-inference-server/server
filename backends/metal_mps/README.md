# Triton Metal Performance Shaders (MPS) Backend

This backend enables Triton Inference Server to run deep learning models using Apple's Metal Performance Shaders (MPS) framework on macOS devices with Apple Silicon or AMD GPUs.

## Features

- **Metal Performance Shaders Integration**: Leverages Apple's optimized GPU compute framework
- **Graph-based Execution**: Uses MPSGraph for efficient model execution
- **Memory Pooling**: Efficient Metal buffer management with reuse pooling
- **Type Support**: Supports common data types (FP32, FP16, INT8, etc.)
- **Operation Coverage**: Implements common DNN operations (Conv2D, MatMul, Pooling, Activations, etc.)

## Supported Operations

The MPS backend currently supports the following operations:

### Tensor Operations
- Element-wise: Add, Subtract, Multiply, Divide
- Matrix operations: MatMul, GEMM

### Neural Network Layers
- Convolution: Conv2D with various padding and stride options
- Pooling: MaxPool2D, AvgPool2D
- Normalization: BatchNorm

### Activation Functions
- ReLU, Sigmoid, Tanh, Softmax

### Shape Operations
- Reshape, Transpose, Concat, Split

### Reduction Operations
- ReduceMean, ReduceSum, ReduceMax, ReduceMin

## Model Format

The backend supports models in the following formats:
- **ONNX** (.onnx): Standard ONNX models
- **MPS** (.mps): Custom format (currently same as ONNX)

## Building

### Prerequisites

- macOS 11.0 or later
- Xcode 12.0 or later
- CMake 3.17 or later
- Metal Performance Shaders framework (included with macOS)

### Build Instructions

```bash
cd backends/metal_mps
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install
```

## Model Configuration

Example `config.pbtxt` for an MPS model:

```
name: "mps_model"
backend: "metal_mps"
max_batch_size: 8

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

## Performance Optimization

### Memory Management
- The backend uses a buffer pooling system to minimize allocation overhead
- Buffers are reused when possible based on size buckets
- Configure pool size with environment variables:
  - `MPS_MAX_POOL_SIZE`: Maximum memory pool size (default: 1GB)
  - `MPS_MAX_BUFFER_AGE`: Maximum buffer age in ms (default: 60000)

### Graph Optimization
- MPSGraph automatically performs optimizations like kernel fusion
- The backend compiles graphs with optimization level 1 by default

### Best Practices
1. Use batch sizes that are powers of 2 for optimal performance
2. Prefer NCHW layout for CNN models
3. Enable graph caching for repeated inference requests

## Limitations

- Currently only supports models that can be represented as static graphs
- Dynamic shapes are not yet supported
- Some ONNX operators may not have MPS equivalents

## Testing

Run the test client to verify the backend is working:

```bash
cd test
python3 test_client.py
```

## Debugging

Enable verbose logging:
```bash
export TRITON_MPS_VERBOSE=1
```

Check Metal device capabilities:
```bash
system_profiler SPDisplaysDataType
```

## Future Enhancements

- Dynamic shape support
- Additional ONNX operator coverage
- Custom MPS kernel support
- Multi-GPU support
- Quantization support (INT8/INT4)
- Graph serialization/deserialization

## License

This backend is released under the same license as Triton Inference Server.