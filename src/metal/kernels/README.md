# Metal Kernel Library

A comprehensive Metal compute kernel library for Triton inference server, providing optimized implementations of common deep learning operations for Apple Silicon.

## Features

### Supported Operations

#### Math Operations
- **GEMM** (General Matrix Multiply)
  - Basic, tiled, and SIMD-optimized variants
  - Support for alpha/beta scaling
  - Batched GEMM
  - Half precision support
- **GEMV** (Matrix-Vector multiplication)
- **Element-wise operations**: Add, Sub, Mul, Div, Pow, Exp, Log, Sqrt
- **Linear algebra**: Dot product, AXPY

#### Neural Network Operations
- **Convolution**
  - 2D and 3D convolution
  - Depthwise convolution
  - Group convolution
  - 1x1 optimized convolution
  - Winograd convolution (3x3)
- **Activation Functions**
  - ReLU, ReLU6, Leaky ReLU, PReLU
  - Sigmoid, Tanh
  - GELU, Swish, ELU, SELU
  - Softmax, LogSoftmax
  - Hardswish, Mish
- **Pooling**
  - Max pooling (2D/3D)
  - Average pooling
  - Global pooling
  - Adaptive pooling
  - ROI pooling
- **Normalization**
  - Batch normalization
  - Layer normalization
  - Instance normalization
  - Group normalization
  - Local response normalization (LRN)
  - RMS normalization

#### Utility Operations
- **Tensor transformations**
  - Transpose, Permute
  - Reshape, Concat, Split
  - Slice, Pad
  - Gather, Scatter
  - One-hot encoding
  - Tile/Repeat
- **Reductions**
  - Sum, Mean, Max, Min, Product
  - Argmax, Argmin
  - Variance, L2 norm
  - Cumulative sum
  - Any, All
- **Type casting**
  - Float32 ↔ Float16
  - Float ↔ Int
  - Quantization support

## Architecture

### Key Components

1. **MetalKernelLibrary**: Main entry point for kernel management
2. **MetalKernel**: Base class for all kernel implementations
3. **MetalKernelCompiler**: Runtime shader compilation
4. **MetalKernelCache**: Compiled kernel caching system
5. **MetalTensorDescriptor**: Tensor metadata and layout information

### Performance Features

- **Optimized for Apple Silicon**: Leverages Apple GPU architecture
- **Multiple kernel variants**: Automatically selects optimal implementation
- **Shared memory usage**: Tiled algorithms for better cache utilization
- **SIMD operations**: Utilizes Metal's simdgroup operations
- **Half precision support**: FP16 compute for better performance
- **Kernel fusion**: Fused operations to reduce memory bandwidth
- **Dynamic dispatch**: Runtime kernel selection based on input sizes

## Usage

### Basic Example

```cpp
#include "metal_kernel_library.h"

// Initialize the library
MetalKernelLibrary& library = MetalKernelLibrary::instance();
library.initialize(device);

// Create tensor descriptors
std::vector<MetalTensorDescriptor> inputs = {
    MetalTensorDescriptor({1024, 1024}, DataType::FLOAT32),
    MetalTensorDescriptor({1024, 1024}, DataType::FLOAT32)
};
std::vector<MetalTensorDescriptor> outputs = {
    MetalTensorDescriptor({1024, 1024}, DataType::FLOAT32)
};

// Get GEMM kernel
auto kernel = library.get_kernel(KernelType::GEMM, inputs, outputs);

// Get optimal configuration
auto config = kernel->suggest_config(inputs, outputs);

// Execute kernel
kernel->encode(encoder, {bufferA, bufferB}, {bufferC}, config);
```

### Advanced Features

#### Custom Kernel Registration
```cpp
// Create custom kernel
class MyCustomKernel : public MetalKernel {
    // Implementation...
};

// Register with library
library.register_custom_kernel("my_kernel", 
    std::make_shared<MyCustomKernel>());
```

#### Performance Profiling
```cpp
// Enable profiling
library.enable_profiling(true);

// Set callback for metrics
library.set_profiling_callback([](const std::string& name, 
                                 const KernelMetrics& metrics) {
    std::cout << name << ": " << metrics.execution_time_ms << " ms" << std::endl;
});
```

#### Kernel Configuration
```cpp
KernelConfig config;
config.threadgroup_size = MTLSizeMake(16, 16, 1);
config.use_half_precision = true;
config.use_simd_group = true;
config.shared_memory_size = 16384; // 16KB
```

## Building

### Requirements
- macOS 11.0 or later
- Xcode 12.0 or later
- CMake 3.18 or later
- Metal shader compiler (included with Xcode)

### Build Instructions
```bash
mkdir build && cd build
cmake .. -DTRITON_BUILD_TESTS=ON -DTRITON_BUILD_BENCHMARKS=ON
make -j8
```

### Running Tests
```bash
./test/metal_kernel_tests
```

### Running Benchmarks
```bash
./benchmark/metal_kernel_benchmarks
```

## Performance Optimization Guidelines

1. **Choose appropriate data layout**: NCHW vs NHWC based on operation
2. **Use half precision when possible**: 2x throughput on modern GPUs
3. **Batch operations**: Better GPU utilization
4. **Minimize memory transfers**: Keep data on GPU
5. **Use kernel fusion**: Combine operations when possible
6. **Profile and tune**: Use built-in profiling to identify bottlenecks

## Integration with Triton

The kernel library is designed to integrate seamlessly with Triton's Metal backend:

```cpp
// In Triton backend
auto kernel_lib = MetalKernelLibrary::instance();
kernel_lib.initialize(metal_device_);

// During inference
auto kernel = kernel_lib.get_kernel(op_type, input_descs, output_descs);
kernel->encode(encoder, input_buffers, output_buffers, config);
```

## Extending the Library

### Adding New Kernels

1. Define kernel type in `KernelType` enum
2. Create kernel class inheriting from `MetalKernel`
3. Implement Metal shader in `.metal` file
4. Register kernel in `MetalKernelLibrary::register_builtin_kernels()`
5. Add tests and benchmarks

### Optimization Checklist

- [ ] Implement basic functionality
- [ ] Add tiled/shared memory variant
- [ ] Add SIMD-optimized variant
- [ ] Add half precision support
- [ ] Profile and tune threadgroup sizes
- [ ] Add specialized variants for common sizes
- [ ] Implement kernel fusion opportunities

## License

See LICENSE file in the repository root.