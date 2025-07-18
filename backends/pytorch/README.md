# PyTorch Backend for Triton Inference Server (macOS)

This is a complete implementation of the PyTorch backend for Triton Inference Server with full macOS support.

## Features

- Full LibTorch integration for macOS (both Intel and Apple Silicon)
- Support for TorchScript models (.pt files)
- **MPS (Metal Performance Shaders) support for Apple Silicon GPUs**
- CPU and GPU execution optimized for macOS
- Automatic handling of library dependencies
- Comprehensive test suite

## Directory Structure

```
pytorch/
├── CMakeLists.txt              # CMake build configuration
├── src/
│   ├── pytorch.cc              # Main backend implementation
│   └── libtriton_pytorch.ldscript  # Linux version script
├── cmake/
│   └── TritonPyTorchBackendConfig.cmake.in
├── test/
│   ├── create_test_model.py   # Test model generator
│   ├── test_client.py          # Test client
│   └── models/                 # Test model repository
│       └── pytorch_simple/
│           ├── config.pbtxt    # Model configuration
│           └── 1/
│               └── model.pt    # TorchScript model (generated)
├── download_libtorch_macos.sh  # LibTorch download script
├── build_macos.sh              # Build script for macOS
├── build_macos_mps.sh          # Build script with MPS support
├── test_pytorch_backend_macos.sh  # Comprehensive test script
├── MPS_INTEGRATION.md          # MPS integration documentation
└── test/
    ├── test_mps_backend.py     # MPS-specific tests
    └── models/
        └── pytorch_mps_example/  # MPS example configuration
```

## Prerequisites

1. **Triton Server built for macOS** (from the main repository)
2. **CMake 3.17+**
3. **Python 3.8+** with PyTorch (for creating test models)
4. **Xcode Command Line Tools**

## Quick Start

1. **Build the PyTorch backend:**
   
   For CPU-only:
   ```bash
   ./build_macos.sh
   ```
   
   For MPS (Metal Performance Shaders) support:
   ```bash
   ./build_macos_mps.sh
   ```
   
   These scripts will:
   - Download LibTorch for your macOS architecture (Intel/Apple Silicon)
   - Configure and build the backend
   - Install it to the local directory
   - Enable MPS support if available (Apple Silicon)

2. **Run the comprehensive test:**
   ```bash
   ./test_pytorch_backend_macos.sh
   ```
   This will:
   - Build the backend
   - Create a test model
   - Start Triton server
   - Run inference tests
   - Verify results

## Manual Installation

1. **Download LibTorch:**
   ```bash
   ./download_libtorch_macos.sh
   ```

2. **Build the backend:**
   ```bash
   mkdir build && cd build
   cmake .. -DTORCH_PATH=../libtorch -DCMAKE_INSTALL_PREFIX=../install
   make -j$(sysctl -n hw.ncpu)
   make install
   ```

3. **Copy to Triton backends directory:**
   ```bash
   cp install/backends/pytorch/*.dylib /path/to/triton/build/backends/pytorch/
   ```

## Creating PyTorch Models

Models must be saved as TorchScript:

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x, y):
        return x + y, x - y

model = MyModel()
model.eval()

# Trace the model
example_inputs = (torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224))
traced_model = torch.jit.trace(model, example_inputs)

# Save as TorchScript
traced_model.save("model.pt")
```

## Model Configuration

Create a `config.pbtxt` file:

```protobuf
name: "my_pytorch_model"
backend: "pytorch"
max_batch_size: 8

input [
  {
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  },
  {
    name: "INPUT1"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  },
  {
    name: "OUTPUT1"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU  # Use KIND_GPU for MPS on Apple Silicon
  }
]

# Optional: Explicit MPS configuration
optimization {
  execution_accelerators {
    gpu_execution_accelerator [
      {
        name: "mps"
      }
    ]
  }
}
```

## Supported Data Types

- `TYPE_BOOL` → `torch.bool`
- `TYPE_UINT8` → `torch.uint8`
- `TYPE_INT8` → `torch.int8`
- `TYPE_INT16` → `torch.int16`
- `TYPE_INT32` → `torch.int32`
- `TYPE_INT64` → `torch.int64`
- `TYPE_FP16` → `torch.float16`
- `TYPE_FP32` → `torch.float32`
- `TYPE_FP64` → `torch.float64`

## macOS-Specific Considerations

1. **Library Paths**: The backend automatically fixes LibTorch library paths using `install_name_tool`
2. **OpenMP**: Disabled by default on macOS to avoid conflicts
3. **MPS Support**: GPU acceleration available on Apple Silicon through Metal Performance Shaders
4. **Architecture**: Automatically detects and downloads the correct LibTorch for Intel/Apple Silicon
5. **Device Selection**: Automatically selects MPS when available and configured

## Troubleshooting

### LibTorch Download Issues
If automatic download fails, manually download LibTorch:
- **Apple Silicon**: https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.1.0.zip
- **Intel**: https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.1.0.zip

Extract to the `libtorch` directory.

### Library Loading Errors
The download script automatically fixes library paths. If you still have issues:
```bash
cd libtorch/lib
for lib in *.dylib; do
    install_name_tool -id "@loader_path/$(basename $lib)" $lib
    # Fix dependencies
    otool -L $lib | grep @rpath | while read -r dep _; do
        install_name_tool -change "$dep" "@loader_path/$(basename $dep)" $lib
    done
done
```

### Model Loading Errors
Ensure your model is saved with the same PyTorch version as LibTorch (2.1.0 in this implementation).

## Testing

Run the test client manually:
```bash
# Start Triton server first
tritonserver --model-repository=test/models --backend-directory=/path/to/backends

# In another terminal
cd test
python3 test_client.py --verbose
```

## Performance Optimization

For best performance on macOS:
1. Use batch sizes that match your CPU core count
2. Enable CPU affinity if needed
3. **Use MPS (Metal Performance Shaders) on Apple Silicon:**
   - Set `kind: KIND_GPU` in instance_group
   - Use model warmup to pre-compile Metal shaders
   - Monitor GPU usage with Activity Monitor
4. **MPS-specific optimizations:**
   - Use FP32 for best compatibility
   - Batch operations for better GPU utilization
   - See `MPS_INTEGRATION.md` for detailed guidelines

## MPS (Metal Performance Shaders) Support

The backend now includes full MPS support for Apple Silicon Macs:

- **Automatic Detection**: MPS availability is detected at runtime
- **Seamless Integration**: Use `KIND_GPU` to enable MPS
- **Memory Management**: Efficient CPU-MPS tensor transfers
- **Fallback Support**: Automatic fallback to CPU if MPS is unavailable

For detailed MPS integration information, see [MPS_INTEGRATION.md](MPS_INTEGRATION.md).

### Testing MPS Support

```bash
# Test MPS availability and performance
python3 test/test_mps_backend.py

# Build and test with MPS
./build_macos_mps.sh
```

## License

Same as Triton Inference Server - BSD 3-Clause License