# Apple Silicon Optimizations for NVIDIA Triton Inference Server

## Overview

This document describes the Apple Silicon optimizations implemented for NVIDIA Triton Inference Server, enabling high-performance inference on macOS systems with M1, M2, M3, and future Apple Silicon processors.

## Architecture

The Apple Silicon optimizations are organized into three main components:

### 1. AMX (Apple Matrix Coprocessor) Provider
- Location: `src/apple/amx_*`
- Leverages Apple's Accelerate framework for optimal matrix operations
- Automatically utilizes AMX hardware when available
- Supports FP32, FP16, BF16 (M3+), and INT8 operations

### 2. ANE (Apple Neural Engine) Provider  
- Location: `src/apple/ane_*`
- Integrates with CoreML for neural network acceleration
- Supports model optimization and quantization
- Includes specialized transformer optimizations

### 3. Metal Backend Extensions
- Location: `src/metal/*`
- Unified memory support for zero-copy operations
- Advanced memory pooling with size-classed allocation
- Comprehensive performance monitoring

## Building with Apple Silicon Support

### Prerequisites
- macOS 11.0 or later
- Xcode 12.0 or later
- CMake 3.18 or later

### Build Configuration

```bash
cmake -DTRITON_ENABLE_METAL=ON \
      -DTRITON_ENABLE_APPLE_OPTIMIZATIONS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make -j$(sysctl -n hw.ncpu)
```

### CMake Options
- `TRITON_ENABLE_METAL`: Enable Metal GPU support (default: ON on macOS)
- `TRITON_ENABLE_APPLE_OPTIMIZATIONS`: Enable AMX/ANE optimizations (default: ON on macOS)
- `TRITON_ENABLE_TESTS`: Build unit tests (default: ON)
- `TRITON_ENABLE_BENCHMARKS`: Build performance benchmarks (default: OFF)

## Usage

### AMX Provider

The AMX provider automatically accelerates matrix operations when available:

```cpp
// AMX operations are used transparently through Triton's backend API
// No code changes required - operations are automatically accelerated
```

### ANE Provider

To enable ANE acceleration for a model:

1. Add to model configuration:
```
optimization { 
  execution_accelerators {
    gpu_execution_accelerator : [
      {
        name : "apple_neural_engine"
        parameters { key: "precision" value: "fp16" }
        parameters { key: "power_mode" value: "high_performance" }
      }
    ]
  }
}
```

2. The ANE provider will automatically:
   - Convert compatible operations to CoreML format
   - Optimize the computation graph
   - Schedule operations on ANE when beneficial

### Metal Memory Management

The Metal allocator provides unified memory support:

```cpp
// Request Metal GPU memory with unified memory hint
auto* allocator = GetMetalAllocator(device_id);
allocator->AllocateUnified(size, &buffer);
```

## Performance Optimization Guide

### 1. Matrix Operations (AMX)
- Use tile-friendly dimensions (multiples of 32)
- Batch operations when possible
- Leverage unified memory to avoid copies

### 2. Neural Networks (ANE)
- Use FP16 precision when accuracy permits
- Enable graph optimization passes
- Consider model partitioning for large models

### 3. Memory Management
- Use Metal unified memory for CPU-GPU shared data
- Enable memory pooling for repeated allocations
- Monitor memory pressure with performance tools

## Supported Operations

### AMX Accelerated
- GEMM (all precisions)
- Convolution (optimized for common sizes)
- Activation functions (ReLU, GELU, etc.)
- Layer normalization
- Winograd convolution (3x3)

### ANE Accelerated  
- Standard CNN layers
- Transformer blocks
- Attention mechanisms
- Common activation functions
- Quantized operations

## Performance Monitoring

### Metrics Available
- GPU utilization percentage
- Memory bandwidth usage
- Power consumption
- Thermal state
- Operation timings

### Prometheus Integration
```cpp
// Metrics are automatically exported when TRITON_ENABLE_METRICS=ON
// Available at http://localhost:8002/metrics
```

### Example Metrics
```
# GPU Utilization
metal_gpu_utilization_percent{device="0"} 85.5

# Memory Bandwidth  
metal_memory_bandwidth_gbps{device="0"} 120.3

# Power Usage
metal_power_usage_watts{device="0"} 15.2
```

## Limitations and Known Issues

1. **AMX Direct Access**: Direct AMX instructions are not publicly documented. This implementation uses Apple's Accelerate framework which provides optimal performance.

2. **ANE Model Support**: Not all model architectures can run on ANE. The provider will automatically fall back to GPU/CPU for unsupported operations.

3. **Memory Limits**: Unified memory is limited by system RAM. Large models may require partitioning.

4. **Winograd Convolution**: Currently limited to 256 input channels for F(2x2, 3x3). F(4x4, 3x3) is not yet implemented.

## Troubleshooting

### Build Issues
- Ensure Xcode command line tools are installed: `xcode-select --install`
- Check CMake version: `cmake --version` (must be 3.18+)
- Verify Metal support: `system_profiler SPDisplaysDataType`

### Runtime Issues
- Check device capabilities: Use `GetAMXCapabilities()` and `GetANECapabilities()`
- Monitor memory pressure: Use Activity Monitor or `metal_memory_stats`
- Enable debug logging: Set `TRITON_METAL_DEBUG=1`

### Performance Issues
- Profile with Instruments: Use Metal System Trace
- Check thermal throttling: Monitor with `metal_performance_monitor`
- Verify operation routing: Enable `TRITON_APPLE_VERBOSE=1`

## Future Enhancements

1. **Extended Winograd**: Implement F(4x4, 3x3) and F(6x6, 3x3) variants
2. **Dynamic Quantization**: Runtime quantization for ANE
3. **Multi-ANE Support**: Utilize multiple Neural Engine cores
4. **Sparse Operations**: Leverage M3+ sparse acceleration
5. **Custom Kernels**: Metal shader generation for specialized operations

## Contributing

When contributing Apple Silicon optimizations:

1. Follow Triton coding standards
2. Add comprehensive unit tests
3. Include performance benchmarks
4. Update documentation
5. Test on multiple Apple Silicon generations

## References

- [Apple Accelerate Framework](https://developer.apple.com/documentation/accelerate)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [Triton Inference Server](https://github.com/triton-inference-server/server)