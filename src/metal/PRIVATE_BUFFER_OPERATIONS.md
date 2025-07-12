# Metal Private Buffer Operations Implementation

## Overview

This document describes the implementation of private buffer operations for Metal memory management in the Triton server. Private buffers in Metal are GPU-only accessible memory that provides better performance but requires special handling for host-GPU data transfers.

## Implementation Details

### 1. CopyFromHost for Private Buffers

The `CopyFromHost` method for private buffers uses a staging buffer approach:

```cpp
void MetalBuffer::CopyFromHost(const void* src, size_t size, size_t offset)
```

**Implementation Steps:**
1. Create a temporary shared staging buffer with the source data
2. Create a command buffer and blit encoder
3. Use the blit encoder to copy from staging buffer to private buffer
4. Wait for completion
5. Clean up resources

**Key Features:**
- Automatic command queue management
- Proper error handling
- Memory bounds checking
- Synchronous execution for simplicity

### 2. CopyToHost for Private Buffers

The `CopyToHost` method reverses the staging buffer approach:

```cpp
void MetalBuffer::CopyToHost(void* dst, size_t size, size_t offset) const
```

**Implementation Steps:**
1. Create a temporary shared staging buffer
2. Create a command buffer and blit encoder
3. Copy from private buffer to staging buffer using blit encoder
4. Copy from staging buffer to host memory
5. Clean up resources

### 3. GPU-to-GPU Copy

The `CopyBuffer` utility function handles various buffer copy scenarios:

```cpp
void MetalMemoryUtils::CopyBuffer(MetalBuffer* dst, const MetalBuffer* src, 
                                  size_t size, size_t dst_offset, size_t src_offset)
```

**Supported Operations:**
- Shared to Shared: Direct memcpy (fastest)
- Private to Private: Metal blit encoder
- Shared to Private: Metal blit encoder
- Private to Shared: Metal blit encoder
- Cross-device: Staging through system memory

### 4. ZeroBuffer for Private Buffers

The `ZeroBuffer` function uses Metal's fill operation:

```cpp
void MetalMemoryUtils::ZeroBuffer(MetalBuffer* buffer)
```

**Implementation:**
- For shared buffers: Direct memset
- For private buffers: Metal blit encoder's fillBuffer command

## Performance Considerations

1. **Staging Buffer Overhead**: Each host-GPU transfer creates a temporary staging buffer. For frequent transfers, consider maintaining a pool of staging buffers.

2. **Synchronous Operations**: Current implementation waits for completion. For better performance, consider asynchronous operations with callbacks.

3. **Command Queue Reuse**: Each operation creates a new command queue. Consider maintaining a thread-local or global command queue pool.

4. **Memory Alignment**: Metal buffers are aligned to 256 bytes by default, which is handled by the existing alignment utilities.

## Usage Examples

### Basic Private Buffer Usage
```cpp
// Create a private buffer
auto buffer = MetalBuffer::Create(device, 1024, false);

// Write data to private buffer
std::vector<float> data(256);
buffer->CopyFromHost(data.data(), data.size() * sizeof(float));

// Read data from private buffer
std::vector<float> result(256);
buffer->CopyToHost(result.data(), result.size() * sizeof(float));
```

### GPU-to-GPU Copy
```cpp
auto src = MetalBuffer::Create(device, 1024, false);
auto dst = MetalBuffer::Create(device, 1024, false);

// Copy entire buffer
MetalMemoryUtils::CopyBuffer(dst.get(), src.get(), 1024);

// Copy with offsets
MetalMemoryUtils::CopyBuffer(dst.get(), src.get(), 512, 256, 0);
```

### Zero Initialization
```cpp
auto buffer = MetalBuffer::Create(device, 4096, false);
MetalMemoryUtils::ZeroBuffer(buffer.get());
```

## Error Handling

The implementation includes comprehensive error checking:
- Null pointer validation
- Buffer bounds checking
- Metal API error handling
- Resource allocation failures

Error messages are logged to stderr with descriptive information about the failure.

## Future Improvements

1. **Asynchronous Operations**: Add support for non-blocking transfers with completion callbacks
2. **Staging Buffer Pool**: Implement a pool to reduce allocation overhead
3. **Command Queue Pool**: Reuse command queues across operations
4. **Performance Metrics**: Add timing and throughput measurements
5. **Batch Operations**: Support for multiple transfers in a single command buffer

## Testing

A comprehensive test suite is provided in `test_private_buffer_ops.mm` that covers:
- Host to private buffer transfers
- Private buffer to host transfers
- GPU-to-GPU copies (all combinations)
- Zero buffer operations
- Partial copies with offsets

Run the test with:
```bash
./build_and_test_private_buffer.sh
```