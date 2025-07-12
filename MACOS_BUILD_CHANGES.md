# macOS/Darwin Platform Support Implementation

This document describes all changes made to enable macOS/Darwin platform detection and build support in the Triton Inference Server build system.

## Summary of Changes

### 1. Created `cmake/MacOS.cmake`
- New file containing macOS-specific CMake configuration
- Detects macOS version and architecture (Intel vs Apple Silicon)
- Sets appropriate compiler flags for Apple Clang
- Configures RPATH settings for dynamic libraries on macOS
- Disables unsupported features (CUDA, TensorRT, NVTX, Mali GPU)
- Adds Homebrew and MacPorts library search paths
- Defines `TRITON_MACOS` and `TRITON_APPLE_SILICON` compile definitions

### 2. Updated `CMakeLists.txt` (main)
- Added platform detection for Darwin/macOS at line 126-137
- Includes the new `cmake/MacOS.cmake` file when macOS is detected
- Updated protobuf config directory handling for macOS (line 186-187)
- Updated OpenTelemetry config directory handling for macOS (line 197-198)
- Disabled OpenTelemetry dependency on macOS (line 222)

### 3. Updated `src/CMakeLists.txt`
- Added platform detection for macOS (lines 166-174)
- Added macOS-specific compiler options with Apple Clang support (lines 142-161)
  - Uses `-stdlib=libc++` and `-fvisibility=hidden`
  - Defines `TRITON_MACOS=1`
- Updated RPATH settings for macOS to use `@loader_path` (lines 197-217)
- Modified platform-specific library linking (lines 320-333)
  - macOS links only with `dl` (no `rt` library)
- Disabled OpenTelemetry support on macOS (lines 81, 465, 580)
- Disabled Python frontend build on macOS (line 851)

### 4. Updated `build.py`
- Modified `target_platform()` function to recognize Darwin as "macos" (lines 132-134)
- Updated `enable_all()` function to handle macOS platform (lines 2352-2378)
  - CPU-only backends (no CUDA/GPU support)
  - Disabled GPU-related features
  - Disabled OpenTelemetry tracing
  - Limited endpoints to HTTP and gRPC only
- Updated help text for `--target-platform` to include "macos" (line 2516)
- Added check to prevent container builds on macOS (lines 1699-1701)

## Key Features and Limitations

### Supported on macOS:
- CPU-only inference backends (ONNX Runtime, PyTorch, Python, OpenVINO)
- HTTP and gRPC endpoints
- Local and Redis caching
- Cloud storage (GCS, S3, Azure)
- Logging, statistics, and CPU metrics

### Not Supported on macOS:
- GPU acceleration (CUDA, TensorRT)
- NVTX profiling
- Mali GPU support
- OpenTelemetry tracing (temporary limitation)
- Container-based builds (must use `--no-container-build`)
- SageMaker and Vertex AI endpoints
- Python frontend package (temporary limitation)

## Build Instructions for macOS

To build Triton on macOS, use the following command:

```bash
./build.py --no-container-build --build-dir=<build_directory> --enable-all
```

Or for a minimal build:

```bash
./build.py --no-container-build --build-dir=<build_directory> \
  --backend=onnxruntime --backend=pytorch --endpoint=http --endpoint=grpc
```

## Platform Detection Logic

The build system now properly detects:
1. **System**: Darwin → "macos"
2. **Architecture**: arm64 → Apple Silicon, x86_64 → Intel Mac
3. **Compiler**: Automatically uses Apple Clang with appropriate flags

## Future Enhancements

1. Enable OpenTelemetry support once macOS compatibility is resolved
2. Add Metal Performance Shaders backend for GPU acceleration
3. Enable Python frontend package build
4. Consider adding CoreML backend for optimized inference
5. Support for containerized builds using Docker Desktop for Mac

## Testing

All changes maintain backward compatibility with existing Linux and Windows builds. The macOS support is cleanly isolated through platform detection, ensuring no impact on other platforms.