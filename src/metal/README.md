# Metal Memory Allocator for Triton

This directory contains a high-performance memory allocator for Apple Metal, designed to efficiently manage GPU memory for Triton Inference Server on macOS.

## Features

### Memory Pool Management
- **Size-based pooling**: Pre-allocated pools for common buffer sizes (256B to 64MB)
- **Dynamic pool growth**: Pools expand based on demand up to configured limits
- **Pool hit optimization**: Reuses freed buffers to minimize allocation overhead
- **Configurable size classes**: Customize pool sizes for your workload

### Allocation Strategies
- **Small buffer optimization**: Pooled allocation for buffers up to 64MB
- **Large buffer handling**: Direct heap allocation with MTLHeap support
- **Unified memory support**: Automatic detection and use of unified memory on Apple Silicon
- **Alignment requirements**: Proper alignment for Metal performance (256-byte default)

### Memory Management
- **Garbage collection**: Background thread periodically reclaims unused memory
- **Fragmentation management**: Monitors and reduces memory fragmentation
- **Memory pressure handling**: Responds to system memory pressure events
- **Statistics tracking**: Comprehensive metrics for allocation patterns and usage

### Integration with Triton
- **Response allocator**: Compatible with TRITONSERVER_ResponseAllocator interface
- **Backend support**: Can be used by any Triton backend requiring Metal memory
- **Memory type negotiation**: Supports preferred and fallback memory types
- **Zero-copy transfers**: Efficient data movement with unified memory

## Architecture

### Key Components

1. **MetalAllocator**: Main allocator class managing pools and heap allocations
2. **MetalMemoryPool**: Per-size-class pool for efficient buffer reuse
3. **MetalAllocationStrategy**: Pluggable strategy for allocation decisions
4. **MetalResponseAllocator**: Triton response allocator implementation

### Memory Hierarchy

```
┌─────────────────────────────────────┐
│        MetalAllocator               │
├─────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────────┐ │
│ │ Memory Pools│ │  MTLHeap Array  │ │
│ ├─────────────┤ ├─────────────────┤ │
│ │ 256B Pool   │ │ Heap 0 (256MB)  │ │
│ │ 1KB Pool    │ │ Heap 1 (256MB)  │ │
│ │ 4KB Pool    │ │ Heap 2 (512MB)  │ │
│ │ ...         │ │ ...             │ │
│ │ 64MB Pool   │ │ Heap N          │ │
│ └─────────────┘ └─────────────────┘ │
└─────────────────────────────────────┘
```

### Allocation Flow

1. Request comes to `MetalAllocator::Allocate()`
2. Strategy determines if pool allocation is suitable
3. For pooled allocation:
   - Find appropriate size class
   - Try to get buffer from pool
   - If pool empty, grow pool or fall back to heap
4. For heap allocation:
   - Try existing MTLHeaps
   - Create new heap if needed
   - Allocate MTLBuffer from heap
5. Track allocation metadata
6. Return buffer pointer

## Usage

### Basic Usage

```cpp
#include "metal_allocator.h"

// Create allocator with default config
MetalAllocator allocator(0);  // Device ID 0

// Allocate memory
void* buffer = nullptr;
MetalAllocation* allocation = nullptr;
auto err = allocator.Allocate(1024 * 1024, &buffer, &allocation);
if (err == nullptr) {
    // Use buffer...
    
    // Free when done
    allocator.Free(allocation);
} else {
    // Handle error
    TRITONSERVER_ErrorDelete(err);
}
```

### Custom Configuration

```cpp
MetalPoolConfig config;
config.size_classes = {1024, 4096, 16384};  // Custom size classes
config.initial_pool_sizes = {100, 50, 25};   // Initial pool sizes
config.use_unified_memory = true;            // Enable unified memory
config.unified_memory_threshold = 16 * 1024 * 1024;  // 16MB threshold

MetalAllocator allocator(0, config);
```

### Response Allocator Integration

```cpp
auto metal_allocator = std::make_shared<MetalAllocator>(0);
MetalResponseAllocator response_allocator(metal_allocator);

// Use with TRITONSERVER APIs
TRITONSERVER_ResponseAllocator* allocator = response_allocator.GetAllocator();
```

### Custom Allocation Strategy

```cpp
class MyStrategy : public MetalAllocationStrategy {
    bool ShouldUsePool(size_t byte_size, const MetalAllocationStats& stats) override {
        // Custom logic for pool usage
        return byte_size <= 1024 * 1024;  // Pool for sizes up to 1MB
    }
    
    bool ShouldUseUnifiedMemory(size_t byte_size, const MetalAllocationStats& stats) override {
        // Custom logic for unified memory
        return byte_size >= 64 * 1024 * 1024;  // Unified for 64MB+
    }
    
    size_t GetAlignment(size_t byte_size) override {
        return 256;  // Metal-friendly alignment
    }
};

allocator.SetAllocationStrategy(std::make_unique<MyStrategy>());
```

## Configuration Options

### MetalPoolConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `size_classes` | vector<size_t> | 256B-64MB | Buffer sizes for pools |
| `initial_pool_sizes` | vector<size_t> | 64-1 | Initial buffers per pool |
| `max_pool_sizes` | vector<size_t> | 256-1 | Maximum buffers per pool |
| `use_unified_memory` | bool | true | Enable unified memory |
| `unified_memory_threshold` | size_t | 16MB | Threshold for unified memory |
| `enable_gc` | bool | true | Enable garbage collection |
| `gc_interval` | chrono::seconds | 30s | GC run interval |
| `gc_fragmentation_threshold` | double | 0.3 | Fragmentation trigger (30%) |
| `high_memory_watermark` | double | 0.9 | High memory threshold (90%) |
| `low_memory_watermark` | double | 0.7 | Low memory threshold (70%) |

## Performance Characteristics

### Allocation Performance
- **Pooled allocations**: ~1-5 microseconds
- **Heap allocations**: ~10-50 microseconds
- **Large allocations**: ~50-200 microseconds

### Memory Overhead
- **Pool overhead**: ~5-10% for active pools
- **Heap overhead**: <1% for large allocations
- **Metadata overhead**: 64-128 bytes per allocation

### Optimization Tips
1. **Size your pools** based on your workload's allocation patterns
2. **Use unified memory** for large buffers on Apple Silicon
3. **Monitor statistics** to tune pool configurations
4. **Enable GC** for long-running services
5. **Align allocations** to 256 bytes for best performance

## Building

### Requirements
- macOS 10.13 or later
- Xcode with Metal SDK
- CMake 3.18 or later
- C++17 compiler

### Build Commands

```bash
cd triton-inference-server
mkdir build && cd build
cmake -DTRITON_ENABLE_TESTS=ON ..
make triton-metal-allocator
```

### Running Tests

```bash
# Unit tests
./src/test/metal_allocator_test

# Benchmarks
./src/test/metal_allocator_benchmark

# Example program
./src/metal/metal_allocator_example
```

## Debugging

### Environment Variables
- `METAL_ALLOCATOR_DEBUG`: Enable debug logging
- `METAL_ALLOCATOR_STATS`: Print statistics on exit
- `METAL_DEVICE_ID`: Override default Metal device

### Memory Leak Detection
The allocator tracks all active allocations and reports leaks on destruction.

### Performance Profiling
Use Instruments with the Metal System Trace template to profile allocation patterns.

## Future Enhancements

1. **Multi-device support**: Allocate across multiple Metal devices
2. **NUMA awareness**: Optimize for unified memory architecture
3. **Compression**: Automatic compression for idle buffers
4. **Persistent memory**: Support for memory-mapped files
5. **Advanced statistics**: Histogram of allocation sizes and lifetimes

## License

This code is released under the same license as Triton Inference Server.