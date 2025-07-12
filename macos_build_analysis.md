# Triton Inference Server Build System Analysis for macOS/Apple Silicon Support

## Overview

This document analyzes the Triton Inference Server build system and identifies the changes needed to support macOS and Apple Silicon (M1/M2/M3) platforms.

## Current Build System Architecture

### 1. Build Infrastructure

**Primary Build Tools:**
- CMake (minimum version 3.18, uses 3.28.3 in containers)
- Python build script (`build.py`) that orchestrates the build process
- Docker-based containerized builds for Linux/Windows
- Direct builds also supported

**Supported Platforms:**
- Linux (Ubuntu 24.04, RHEL/CentOS)
- Windows (Windows 10)
- NVIDIA GPU support (CUDA 12.8)
- Limited ARM support (aarch64 for Linux/Jetson)

### 2. Major Dependencies

#### Core Dependencies (from CMakeLists.txt):
- **protobuf** - For gRPC/protocol buffers
- **grpc** - For gRPC endpoint support
- **libevent/libevhtp** - For HTTP server functionality
- **re2** - Regular expression library
- **boost** (1.80.0) - Various utilities including interprocess, stacktrace
- **opentelemetry-cpp** - For tracing support (Linux only)
- **googletest** - For unit testing

#### NVIDIA-Specific Dependencies:
- **CUDA Toolkit** (12.8) - GPU compute platform
- **cuDNN** (9.7.1.26) - Deep learning primitives
- **TensorRT** (10.8.0.43) - Inference optimization
- **DCGM** (3.3.6) - GPU management
- **NVTX** - NVIDIA profiling tools
- **NCCL** - Multi-GPU communication

#### Cloud Storage Dependencies:
- **aws-sdk-cpp** - For S3 support
- **google-cloud-cpp** - For GCS support
- **Azure Storage SDK** - For Azure blob storage

#### Backend-Specific Dependencies:
- **PyTorch** - ML framework backend
- **ONNX Runtime** - ONNX model support
- **OpenVINO** - Intel inference engine
- **TensorFlow** - ML framework backend
- **DALI** - Data loading library
- **vLLM** - LLM serving
- **TensorRT-LLM** - NVIDIA LLM optimization

### 3. Platform-Specific Code

#### Conditional Compilation:
```cpp
#ifdef _WIN32  // Windows-specific code
#ifdef __linux__  // Linux-specific code
#ifdef __APPLE__  // macOS-specific code (not currently used)
```

**Files with platform-specific code:**
- `command_line_parser.cc` - getopt implementation for Windows
- `http_server.cc` - Socket handling differences
- `main.cc` - WSA initialization for Windows
- `shared_memory_manager.cc` - Shared memory implementation
- `triton_signal.cc` - Signal handling differences
- `tracer.cc` - OpenTelemetry support (disabled on Windows)

### 4. Build Process Flow

1. **Container Build Path** (default):
   - Create base container with dependencies
   - Build Triton components inside container
   - Create final runtime container

2. **Direct Build Path**:
   - Install dependencies locally
   - Run CMake build
   - Install artifacts

## Required Changes for macOS/Apple Silicon Support

### 1. Build System Modifications

#### CMakeLists.txt Changes:
```cmake
# Add macOS platform detection
if(APPLE)
  set(TRITON_PLATFORM_MACOS ON)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(TRITON_PLATFORM_APPLE_SILICON ON)
  endif()
endif()

# Disable unsupported features on macOS
if(APPLE)
  set(TRITON_ENABLE_GPU OFF CACHE BOOL "GPU not supported on macOS" FORCE)
  set(TRITON_ENABLE_MALI_GPU OFF CACHE BOOL "Mali GPU not supported on macOS" FORCE)
  set(TRITON_ENABLE_METRICS_GPU OFF CACHE BOOL "GPU metrics not supported on macOS" FORCE)
  set(TRITON_ENABLE_NVTX OFF CACHE BOOL "NVTX not supported on macOS" FORCE)
  # Consider disabling some backends initially
  set(TRITON_ENABLE_TENSORRT OFF CACHE BOOL "TensorRT not supported on macOS" FORCE)
  set(TRITON_ENABLE_TENSORRTLLM OFF CACHE BOOL "TensorRT-LLM not supported on macOS" FORCE)
endif()

# macOS-specific library paths
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
endif()
```

#### build.py Modifications:
```python
def target_platform():
    platform_string = platform.system().lower()
    if platform_string == "darwin":
        return "macos"
    # ... existing code

def target_machine():
    machine = platform.machine().lower()
    if machine == "arm64" and target_platform() == "macos":
        return "apple_silicon"
    # ... existing code
```

### 2. Dependency Management

#### Replace/Adapt Linux-specific Dependencies:
1. **libevhtp** - May need patches for macOS compatibility
2. **gperftools** - Use macOS-compatible profiling tools
3. **jemalloc** - Consider using system allocator or building from source
4. **CUDA/GPU libraries** - Remove or provide CPU-only alternatives

#### Use Homebrew/MacPorts for Dependencies:
```bash
# Example homebrew formula additions
brew install protobuf grpc boost re2 libevent openssl cmake
```

### 3. Platform-Specific Code Additions

#### Add macOS Support to Signal Handling:
```cpp
#ifdef __APPLE__
#include <signal.h>
// macOS-specific signal handling
#endif
```

#### Shared Memory Implementation:
```cpp
#ifdef __APPLE__
// Use POSIX shared memory APIs
// Handle macOS-specific limitations
#endif
```

#### Dynamic Library Loading:
```cpp
#ifdef __APPLE__
// Use dlopen with .dylib extension
// Handle @rpath, @loader_path
#endif
```

### 4. Backend Modifications

#### Priority Backends for macOS:
1. **Python** - Should work with minimal changes
2. **ONNX Runtime** - Has macOS support
3. **OpenVINO** - May need adaptation
4. **PyTorch** - Has macOS/Metal support

#### Backends Requiring Major Work:
1. **TensorRT** - NVIDIA-specific, needs replacement
2. **DALI** - GPU-focused, needs CPU alternative
3. **FasterTransformer** - CUDA-dependent
4. **vLLM** - May need Metal backend

### 5. Build Script Changes

#### Create macOS Build Script:
```bash
#!/bin/bash
# macos_build.sh

# Install dependencies via Homebrew
brew install cmake protobuf grpc boost re2 libevent

# Set macOS-specific environment
export MACOSX_DEPLOYMENT_TARGET=11.0

# Configure CMake for macOS
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRITON_ENABLE_GPU=OFF \
  -DTRITON_ENABLE_STATS=ON \
  -DTRITON_ENABLE_METRICS=ON \
  -DTRITON_ENABLE_LOGGING=ON \
  -DTRITON_ENABLE_HTTP=ON \
  -DTRITON_ENABLE_GRPC=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local/tritonserver

# Build
cmake --build build -j$(sysctl -n hw.ncpu)
```

### 6. Testing Adaptations

#### Disable GPU-specific Tests:
- Skip CUDA-related tests
- Adapt performance benchmarks
- Add macOS-specific test cases

### 7. Metal/CoreML Integration (Future)

For optimal performance on Apple Silicon:
1. Add Metal Performance Shaders backend
2. Integrate CoreML for optimized inference
3. Support Apple Neural Engine

## Implementation Priority

1. **Phase 1**: Basic CPU-only build
   - Core server functionality
   - HTTP/gRPC endpoints
   - Python backend support

2. **Phase 2**: Backend expansion
   - ONNX Runtime support
   - PyTorch CPU support
   - Basic performance optimization

3. **Phase 3**: Apple Silicon optimization
   - Metal backend development
   - CoreML integration
   - Performance tuning

## Challenges and Considerations

1. **No NVIDIA GPU Support**: All GPU-specific features must be disabled or replaced
2. **Library Compatibility**: Some Linux-specific libraries may need replacements
3. **Performance**: Without GPU acceleration, performance optimization is crucial
4. **Docker Limitations**: Docker on macOS has different networking/filesystem behavior
5. **Code Signing**: macOS may require code signing for distribution

## Recommendations

1. Start with a minimal build configuration
2. Focus on CPU-only inference initially
3. Leverage existing macOS ML frameworks (CoreML, Metal Performance Shaders)
4. Create automated CI/CD for macOS builds
5. Engage with the community for testing and feedback

This analysis provides a roadmap for adding macOS/Apple Silicon support to Triton Inference Server. The implementation will require significant effort but is technically feasible with the modular architecture of Triton.