# Triton Inference Server macOS Dependencies Mapping

This document provides a comprehensive inventory of Linux-specific dependencies in Triton and their macOS equivalents.

## Table of Contents
- [System Libraries](#system-libraries)
- [CUDA and GPU Libraries](#cuda-and-gpu-libraries)
- [Build Tools and Compilers](#build-tools-and-compilers)
- [Third-Party Libraries](#third-party-libraries)
- [Python Dependencies](#python-dependencies)
- [System Calls and Headers](#system-calls-and-headers)
- [Docker-Specific Dependencies](#docker-specific-dependencies)
- [Summary of Changes Required](#summary-of-changes-required)

## System Libraries

| Linux Library | Purpose | macOS Equivalent | Installation | Notes |
|--------------|---------|------------------|--------------|-------|
| `libdl.so` | Dynamic library loading | Built into macOS | N/A | Use `dlopen()` directly |
| `librt.so` | Real-time extensions | Built into macOS | N/A | Functions available in system libraries |
| `libpthread.so` | POSIX threads | Built into macOS | N/A | Use `-pthread` flag |
| `libc.so` | C standard library | `libSystem.dylib` | Built-in | Automatically linked |
| `libstdc++.so` | C++ standard library | `libc++.dylib` | Built-in | Use `-stdlib=libc++` |
| `libssl.so` | OpenSSL crypto | `libssl.dylib` | `brew install openssl` | May need explicit linking |
| `libcurl.so` | HTTP/HTTPS client | `libcurl.dylib` | Built-in or `brew install curl` | |
| `libarchive.so` | Archive handling | `libarchive.dylib` | `brew install libarchive` | |
| `libz.so` | Compression | `libz.dylib` | Built-in | |

## CUDA and GPU Libraries

| Linux Library | Purpose | macOS Equivalent | Installation | Notes |
|--------------|---------|------------------|--------------|-------|
| `libcuda.so` | CUDA driver | **None** | N/A | **Must be removed/stubbed** |
| `libcudart.so` | CUDA runtime | **None** | N/A | **Must be removed/stubbed** |
| `libcublas.so` | CUDA BLAS | **None** | N/A | Use Apple Accelerate.framework |
| `libcudnn.so` | CUDA DNN | **None** | N/A | Use CoreML or Metal Performance Shaders |
| `libnccl.so` | NVIDIA collective comm | **None** | N/A | **Must be removed** |
| `libnvinfer.so` | TensorRT | **None** | N/A | **Must be removed** |
| `libcupti.so` | CUDA profiling | **None** | N/A | **Must be removed** |
| `libcufile.so` | GPU Direct Storage | **None** | N/A | **Must be removed** |
| DCGM | GPU management | **None** | N/A | **Must be removed** |

### Apple GPU Alternative
For GPU acceleration on macOS:
- Use Metal Performance Shaders (MPS)
- Use CoreML for inference
- Use Apple's Accelerate.framework for BLAS operations

## Build Tools and Compilers

| Linux Tool | Purpose | macOS Equivalent | Installation | Notes |
|-----------|---------|------------------|--------------|-------|
| `gcc/g++` | GNU compiler | `clang/clang++` | Xcode Command Line Tools | Built-in with Xcode |
| `cmake` | Build system | `cmake` | `brew install cmake` | Same version requirements |
| `make` | Build automation | `make` | Built-in | GNU make available |
| `autoconf` | Configure scripts | `autoconf` | `brew install autoconf` | |
| `automake` | Makefile generation | `automake` | `brew install automake` | |
| `libtool` | Library tool | `libtool` | `brew install libtool` | |
| `pkg-config` | Package info | `pkg-config` | `brew install pkg-config` | |
| `ccache` | Compilation cache | `ccache` | `brew install ccache` | |

## Third-Party Libraries

| Linux Package | Purpose | macOS Equivalent | Installation | Notes |
|--------------|---------|------------------|--------------|-------|
| `protobuf` | Serialization | `protobuf` | `brew install protobuf` | |
| `grpc` | RPC framework | `grpc` | `brew install grpc` | |
| `libevent` | Event notification | `libevent` | `brew install libevent` | |
| `libevhtp` | HTTP API | `libevhtp` | `brew install libevhtp` | May need manual build |
| `opencv` | Computer vision | `opencv` | `brew install opencv` | |
| `rapidjson` | JSON parsing | `rapidjson` | `brew install rapidjson` | |
| `boost` | C++ libraries | `boost` | `brew install boost` | |
| `re2` | Regular expressions | `re2` | `brew install re2` | |
| `aws-sdk-cpp` | AWS S3 support | `aws-sdk-cpp` | `brew install aws-sdk-cpp` | |
| `google-cloud-cpp` | GCS support | Build from source | Manual build required | |

## Python Dependencies

All Python packages listed in `requirements.txt` are platform-independent:
- `fastapi==0.115.6`
- `httpx==0.27.2`
- `openai==1.60.0`
- `partial-json-parser`
- `starlette>=0.40.0`
- `grpcio-tools<1.68`
- `numpy<2`
- `pillow`

These can be installed identically on macOS using pip.

## System Calls and Headers

| Linux Header/Call | Purpose | macOS Equivalent | Changes Required |
|------------------|---------|------------------|------------------|
| `<sys/mman.h>` | Memory mapping | Same | No change |
| `<sys/stat.h>` | File statistics | Same | No change |
| `<linux/*.h>` | Linux-specific | **None** | Must be conditionally compiled out |
| `mmap()` | Memory mapping | Same | No change |
| `shm_open()` | Shared memory | Same | Different behavior, needs testing |
| `/dev/shm` | Shared memory | `/tmp` or POSIX shm | Code changes required |
| `epoll` | Event polling | `kqueue` | Requires code changes |

## Docker-Specific Dependencies

These are found in Dockerfiles and need alternative solutions:

### APT Packages (Debian/Ubuntu)
```bash
# Development tools
apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim
```

**macOS equivalent:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install additional tools via Homebrew
brew install git wget curl vim
```

### YUM/DNF Packages (RHEL)
```bash
yum install -y \
    autoconf \
    automake \
    libtool
```

**macOS equivalent:**
```bash
brew install autoconf automake libtool
```

## Summary of Changes Required

### 1. Must Remove/Disable
- All CUDA-related code (`TRITON_ENABLE_GPU=OFF`)
- TensorRT backend
- DCGM metrics
- GPU memory tracking
- NVTX tracing

### 2. Code Modifications Required
- Replace `epoll` with `kqueue` for event handling
- Update shared memory paths from `/dev/shm` to POSIX shared memory
- Conditional compilation for Linux-specific headers
- Update dynamic library extensions from `.so` to `.dylib`
- Handle `@rpath` and `@loader_path` for dynamic linking

### 3. Build System Changes
- Update CMake to detect macOS platform
- Modify library search paths
- Update compiler flags for clang
- Remove CUDA-related CMake configurations
- Add Framework linking for Apple libraries

### 4. Alternative Implementations
- Replace CUDA BLAS with Apple's Accelerate.framework
- Consider CoreML or ONNX Runtime for inference
- Use Metal Performance Shaders for GPU acceleration
- Implement CPU-only versions of GPU-accelerated features

### 5. Can Be Kept As-Is
- Most POSIX system calls
- Standard C/C++ libraries
- Python dependencies
- Network libraries (with minor linking adjustments)
- File I/O operations

## Installation Script for macOS

```bash
#!/bin/bash
# Install macOS dependencies for Triton

# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install build dependencies
brew install \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    ccache

# Install libraries
brew install \
    protobuf \
    grpc \
    libevent \
    opencv \
    rapidjson \
    boost \
    re2 \
    openssl \
    libarchive

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r python/openai/requirements.txt
```

## Next Steps

1. Create a `MACOS_BUILD_GUIDE.md` with step-by-step build instructions
2. Implement platform detection in CMake files
3. Create stub implementations for GPU-only features
4. Test each component individually
5. Update CI/CD to support macOS builds