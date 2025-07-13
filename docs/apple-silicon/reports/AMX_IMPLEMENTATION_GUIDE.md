# AMX (Apple Matrix Extensions) Implementation Guide

## Overview

This document describes the implementation of AMX-optimized kernels in the Triton Inference Server for Apple Silicon processors. The AMX coprocessor provides hardware-accelerated matrix operations on M1, M2, and M3 chips.

## Architecture

### AMX Hardware Capabilities

- **M1 Series**: 
  - 32x32 tile operations
  - FP32, FP16, INT8 support
  - ~2 TFLOPS peak performance

- **M2 Series**:
  - Enhanced AMX2 instruction set
  - Improved FP16 performance
  - ~3.6 TFLOPS peak performance

- **M3 Series**:
  - AMX2 with BF16 support
  - Further optimizations
  - ~4.5 TFLOPS peak performance

### Implementation Structure

```
src/apple/
├── amx_provider.h           # Main AMX provider interface
├── amx_provider.cc          # Provider implementation
├── amx_kernel_library.cc    # Kernel registry and implementations
└── amx_kernel_advanced.cc   # Advanced AMX instruction patterns
```

## Key Components

### 1. AMXProvider Class

The singleton provider manages AMX resources and operations:

```cpp
class AMXProvider {
    // Capability detection
    AMXCapabilities capabilities_;
    
    // Kernel library
    std::unique_ptr<AMXKernelLibrary> kernel_library_;
    
    // Performance metrics
    AMXMetrics metrics_;
};
```

### 2. AMXKernelLibrary

Provides optimized kernels for various operations:

- **GEMM Operations**:
  - `sgemm_amx`: Single-precision matrix multiplication
  - `hgemm_amx`: Half-precision (FP16) matrix multiplication
  - `igemm_amx`: Integer (INT8) matrix multiplication

- **Convolution Kernels**:
  - `conv2d_amx`: 2D convolution with im2col transformation
  - Winograd convolution for 3x3 kernels

- **Activation Functions**:
  - `relu_amx`: ReLU activation
  - `sigmoid_amx`: Sigmoid activation
  - `tanh_amx`: Tanh activation

### 3. AMX Instruction Usage

The implementation uses inline assembly for AMX instructions:

```cpp
// Load data into X register
__asm__ volatile(".word 0x00201000 | (reg << 5)" :: : "memory");

// Perform FP32 matrix multiply
__asm__ volatile(".word 0x0080180c" ::: "memory");

// Store result from Z register
__asm__ volatile(".word 0x00201420 | (reg << 5)" :: : "memory");
```

## Optimization Strategies

### 1. Tiling Strategy

AMX operates on fixed-size tiles:
- FP32: 32x32 tiles
- FP16: 64x64 tiles
- INT8: 128x128 tiles

The kernels automatically tile larger matrices for optimal performance.

### 2. Memory Layout

- Use row-major layout for matrices
- Ensure 64-byte alignment for AMX operations
- Minimize data movement between main memory and AMX registers

### 3. Pipeline Optimization

- Overlap data loading with computation
- Use double buffering for continuous operation
- Minimize AMX state transitions

## Usage Example

```cpp
// Initialize AMX provider
auto& provider = AMXProvider::Instance();
auto err = provider.Initialize();

// Perform FP32 GEMM
float* A = ...;  // M x K matrix
float* B = ...;  // K x N matrix
float* C = ...;  // M x N matrix

err = provider.ExecuteGEMM(A, B, C, M, N, K, alpha, beta);

// Get performance metrics
auto metrics = provider.GetMetrics();
std::cout << "GFLOPS: " << metrics.gflops << std::endl;
```

## Performance Considerations

### 1. When to Use AMX

AMX is beneficial for:
- Large matrix multiplications (> 128x128)
- Batch operations
- Convolution layers in neural networks
- Linear algebra operations

### 2. When to Avoid AMX

AMX may not be optimal for:
- Small matrices (< 32x32)
- Element-wise operations
- Memory-bound operations
- Irregular access patterns

### 3. Performance Tuning

- Use the `GetOptimalConfig()` method for automatic configuration
- Enable auto-tuning for workload-specific optimization
- Monitor metrics to identify bottlenecks

## Integration with Triton

### 1. Backend Integration

AMX operations are integrated into Triton backends:
- ONNX Runtime backend
- TensorFlow backend
- PyTorch backend

### 2. Model Optimization

Models can be optimized for AMX:
- Quantization to INT8 for higher throughput
- Layer fusion to reduce memory transfers
- Batch size optimization

### 3. Memory Management

- Uses unified memory for efficient data transfer
- Integrates with Metal allocator for GPU interop
- Supports zero-copy operations where possible

## Testing and Validation

### 1. Unit Tests

```bash
# Run AMX kernel tests
./test_amx_kernels

# Run accuracy tests
./test_amx_accuracy
```

### 2. Performance Benchmarks

```bash
# Run performance benchmarks
./apple_silicon_benchmarks --amx
```

### 3. Validation Suite

- Accuracy validation against reference implementations
- Performance regression testing
- Hardware compatibility testing

## Troubleshooting

### Common Issues

1. **AMX Not Available**
   - Check CPU model supports AMX
   - Verify macOS version (11.0+)
   - Check security settings

2. **Performance Issues**
   - Verify proper alignment
   - Check tile sizes
   - Monitor thermal throttling

3. **Accuracy Issues**
   - Validate numerical precision
   - Check for overflow in INT8 operations
   - Verify alpha/beta scaling

## Future Enhancements

1. **Planned Features**
   - Dynamic kernel selection
   - Multi-threaded AMX operations
   - Advanced fusion patterns

2. **Research Areas**
   - Sparse matrix support
   - Mixed-precision operations
   - Custom operator development

## References

- Apple Developer Documentation (limited public information)
- Reverse engineering research papers
- Open-source AMX investigations
- Performance analysis tools

## Contributing

To contribute AMX optimizations:
1. Follow the kernel implementation pattern
2. Add comprehensive tests
3. Document performance improvements
4. Submit benchmarks with PRs