# ONNX Runtime Backend for macOS

This document describes the macOS implementation of the ONNX Runtime backend for Triton Inference Server.

## Overview

The ONNX Runtime backend has been fully adapted to work on macOS, supporting both Intel and Apple Silicon Macs. This implementation includes:

- Native macOS build support
- Homebrew integration for easy dependency management
- CoreML execution provider support for Apple Silicon optimization
- Comprehensive testing framework
- Full compatibility with Triton's backend API

## Features

### Platform Support
- **Intel Macs**: Full CPU inference support
- **Apple Silicon (M1/M2/M3)**: CPU inference with optional CoreML acceleration
- **macOS Versions**: Tested on macOS 11.0 (Big Sur) and later

### Execution Providers
- **CPUExecutionProvider**: Default provider for all platforms
- **CoreMLExecutionProvider**: Available on macOS for hardware acceleration (optional)

### Key Capabilities
- Dynamic model loading
- Batch inference support
- Multiple model instances
- Thread pool configuration
- Memory optimization for macOS

## Prerequisites

### System Requirements
- macOS 11.0 or later
- Xcode Command Line Tools
- Homebrew package manager

### Required Dependencies
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install cmake
brew install onnxruntime
brew install python@3.11

# Install Python packages
pip3 install onnx numpy protobuf
```

## Building the Backend

### Quick Build
```bash
# Clone the repository
git clone https://github.com/triton-inference-server/onnxruntime_backend.git
cd onnxruntime_backend

# Run the macOS build script
./build_macos.sh --clean --build-type=Release

# Or use the comprehensive setup script
./setup_and_test_macos.sh
```

### Manual Build
```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRITON_ENABLE_GPU=OFF \
  -DTRITON_ENABLE_ONNXRUNTIME_COREML=ON

# Build
make -j$(sysctl -n hw.ncpu)

# Install
sudo make install
```

### Build Options
- `TRITON_ENABLE_ONNXRUNTIME_COREML`: Enable CoreML provider (ON by default on macOS)
- `TRITON_BUILD_ONNXRUNTIME_VERSION`: Specify ONNX Runtime version
- `TRITON_ONNXRUNTIME_INCLUDE_PATHS`: Custom ONNX Runtime include paths
- `TRITON_ONNXRUNTIME_LIB_PATHS`: Custom ONNX Runtime library paths

## Installation

The backend is installed to `/usr/local/tritonserver/backends/onnxruntime` by default. The installation includes:

```
/usr/local/tritonserver/backends/onnxruntime/
├── libtriton_onnxruntime.dylib    # Backend shared library
├── libonnxruntime.dylib            # ONNX Runtime library (if bundled)
└── test_backend.sh                 # Validation script
```

## Configuration

### Model Configuration

Create a `config.pbtxt` file for your ONNX model:

```protobuf
name: "my_onnx_model"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 224, 224, 3 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

# Enable CoreML on macOS
optimization {
  execution_accelerators {
    cpu_execution_accelerator [
      {
        name: "coreml"
        parameters { key: "enable_coreml" value: { string_value: "true" } }
      }
    ]
  }
}
```

### Backend Configuration

Configure backend behavior via command-line arguments:

```bash
tritonserver \
  --model-repository=/path/to/models \
  --backend-config=onnxruntime,enable-global-threadpool=true \
  --backend-config=onnxruntime,intra_op_thread_count=4 \
  --backend-config=onnxruntime,inter_op_thread_count=2
```

## Testing

### Quick Test
```bash
# Run the test script
/usr/local/tritonserver/backends/onnxruntime/test_backend.sh
```

### Comprehensive Test
```bash
# Run the full test suite
./setup_and_test_macos.sh
```

### Manual Testing
```bash
# Build test program
cd build/test
cmake ../../test
make

# Run test
./test_macos_onnx_backend /path/to/libtriton_onnxruntime.dylib
```

## Performance Optimization

### Apple Silicon Optimization
- Enable CoreML provider for Metal Performance Shaders acceleration
- Use `enable_coreml=true` in model configuration
- CoreML automatically utilizes Neural Engine when beneficial

### CPU Optimization
- Configure thread pools based on CPU cores
- Use `intra_op_thread_count` for parallelism within operations
- Use `inter_op_thread_count` for parallelism between operations

### Memory Optimization
- macOS unified memory architecture benefits from proper buffer management
- Use appropriate batch sizes to maximize throughput
- Monitor memory pressure with Activity Monitor

## Troubleshooting

### Common Issues

1. **Library not found errors**
   ```bash
   # Check library dependencies
   otool -L /path/to/libtriton_onnxruntime.dylib
   
   # Fix library paths if needed
   install_name_tool -add_rpath @loader_path /path/to/libtriton_onnxruntime.dylib
   ```

2. **ONNX Runtime not found**
   ```bash
   # Reinstall via Homebrew
   brew reinstall onnxruntime
   
   # Or specify custom paths in CMake
   -DTRITON_ONNXRUNTIME_INCLUDE_PATHS=/custom/path/include
   -DTRITON_ONNXRUNTIME_LIB_PATHS=/custom/path/lib
   ```

3. **CoreML provider not available**
   - Ensure macOS 11.0+ is installed
   - Rebuild with `-DTRITON_ENABLE_ONNXRUNTIME_COREML=ON`
   - Check Console.app for CoreML-related errors

### Debug Build
```bash
# Build with debug symbols
./build_macos.sh --build-type=Debug

# Run with verbose logging
TRITONSERVER_LOG_VERBOSE=1 tritonserver --model-repository=/path/to/models
```

## Integration with Triton Server

1. Build Triton Server for macOS (see main build documentation)
2. Copy the backend to Triton's backend directory:
   ```bash
   cp -r /usr/local/tritonserver/backends/onnxruntime \
         /path/to/triton/backends/
   ```
3. Start Triton with ONNX models:
   ```bash
   tritonserver --model-repository=/path/to/onnx/models
   ```

## Limitations

- GPU support is limited to Metal Performance Shaders via CoreML
- TensorRT execution provider is not available on macOS
- OpenVINO execution provider is not available on macOS
- Some ONNX operators may not be accelerated by CoreML

## Contributing

When contributing macOS-specific changes:
1. Test on both Intel and Apple Silicon Macs
2. Ensure Homebrew compatibility
3. Update documentation for macOS-specific features
4. Add platform-specific tests

## License

This implementation maintains the same license as the original ONNX Runtime backend. See LICENSE file for details.