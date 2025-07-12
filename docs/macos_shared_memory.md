# macOS Shared Memory Support for Triton Inference Server

## Overview

This document describes the changes made to support shared memory functionality on macOS (Darwin) in the Triton Inference Server.

## Key Differences Between Linux and macOS

### 1. Shared Memory Naming Conventions

**Linux:**
- Accepts shared memory names with multiple slashes
- Names can be arbitrary paths like `/dev/shm/my_region` or `//double/slash`
- No strict naming requirements

**macOS:**
- Shared memory names must start with exactly one slash
- Names cannot contain any other slashes after the initial one
- Invalid: `/path/to/memory`, `//double_slash`
- Valid: `/my_memory_region`, `/test_region_123`

### 2. Shared Memory Location

**Linux:**
- Shared memory regions are typically visible in `/dev/shm/`
- Can be inspected as files in the filesystem

**macOS:**
- No `/dev/shm` directory
- Shared memory regions are managed by the kernel
- Not directly visible in the filesystem

## Implementation Changes

### 1. Name Normalization Function

Added `NormalizeSharedMemoryName()` function that:
- Removes all leading slashes from the input name
- Replaces any internal slashes with underscores
- Prepends a single slash to create a valid macOS shared memory name

Example transformations:
- `/simple_name` → `/simple_name`
- `//double_slash` → `/double_slash`
- `/path/with/slashes` → `/path_with_slashes`
- `no_leading_slash` → `/no_leading_slash`

### 2. Region Tracking for Cleanup

On macOS, we maintain a registry of created shared memory regions to ensure proper cleanup:
- `created_shm_regions`: Set of normalized shared memory names
- `RegisterSharedMemoryRegion()`: Adds a region to the tracking set
- `UnlinkSharedMemoryRegion()`: Removes and unlinks a region
- `CleanupAllSharedMemoryRegions()`: Unlinks all tracked regions

### 3. Automatic Cleanup

- Added cleanup in `SharedMemoryManager` destructor
- Unlink shared memory regions when unregistering on macOS
- Helps prevent resource leaks even if the process terminates unexpectedly

## Platform-Specific Code Sections

All macOS-specific code is wrapped in `#ifdef __APPLE__` blocks:

```cpp
#ifdef __APPLE__
    // macOS-specific code
#else
    // Linux/other platform code
#endif
```

## Testing

### Unit Test Compilation
```bash
# Compile the test program
g++ -std=c++11 test_macos_shm.cc -o test_macos_shm

# Run the test
./test_macos_shm
```

### Expected Test Output
```
Testing shared memory name normalization for macOS:
Original: '/simple_name' -> Normalized: '/simple_name'
Original: '//double_slash' -> Normalized: '/double_slash'
Original: '/path/with/slashes' -> Normalized: '/path_with_slashes'
Original: 'no_leading_slash' -> Normalized: '/no_leading_slash'
Original: '///multiple///slashes///' -> Normalized: '/multiple___slashes___'

Testing shared memory operations:
Created shared memory region: /test_shm_region
Mapped shared memory successfully
Wrote data: Hello from macOS shared memory!
Read back: Hello from macOS shared memory!
Cleanup completed successfully
```

## Limitations

1. **Name Transformation**: Client applications must be aware that shared memory names may be transformed on macOS. The original name provided to the API is preserved for user reference, but the actual system name may differ.

2. **CUDA Shared Memory**: The CUDA shared memory implementation remains unchanged as it uses CUDA IPC handles rather than POSIX shared memory.

3. **Performance**: The additional name normalization and tracking overhead is minimal and only affects registration/unregistration operations, not data access.

## Integration with Python Client

The Python client library (`tritonclient.utils.shared_memory`) should work without modification as long as:
1. The client provides valid shared memory names
2. The server handles the platform-specific normalization internally

## Error Handling

Enhanced error messages include both the original and normalized shared memory names for debugging:
```
LOG_VERBOSE(1) << "shm_open failed for key '" << normalized_key 
               << "' (original: '" << shm_key << "'), errno: " << errno;
```

## Future Improvements

1. Consider adding a configuration option to disable automatic unlinking on macOS
2. Implement persistent shared memory region tracking across server restarts
3. Add metrics for shared memory usage and cleanup operations