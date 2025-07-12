# Triton CoreML Backend

This backend enables Triton Inference Server to run Apple CoreML models on macOS systems, leveraging the Neural Engine, GPU (via Metal), and CPU for optimal performance.

## Features

- **Multiple Model Formats**: Supports both `.mlmodel` and `.mlpackage` formats
- **Hardware Acceleration**: Automatic routing to Neural Engine, GPU (Metal), or CPU
- **Flexible Compute Units**: Configurable compute unit selection (CPU, GPU, Neural Engine, or combinations)
- **Power Efficiency**: Options for power-efficient inference
- **Model Versioning**: Full support for Triton's model versioning system
- **Asynchronous Execution**: Efficient handling of concurrent requests

## Requirements

- macOS 11.0 or later (macOS 12.0+ recommended for best Neural Engine support)
- Xcode Command Line Tools
- CMake 3.17 or later
- Triton Inference Server dependencies

## Building

### Quick Build

```bash
./build_macos.sh
```

### Build Options

```bash
./build_macos.sh --help
```

Available options:
- `--build-type <Debug|Release>`: Build type (default: Release)
- `--install-prefix <path>`: Install prefix (default: /opt/tritonserver)
- `--no-gpu`: Disable GPU support
- `--no-stats`: Disable statistics collection
- `--build-dir <path>`: Build directory (default: build)

### Manual Build

```bash
mkdir build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/tritonserver
make -j$(sysctl -n hw.ncpu)
sudo make install
```

## Model Repository Structure

CoreML models should be organized following Triton's standard model repository structure:

```
model_repository/
└── my_coreml_model/
    ├── config.pbtxt
    └── 1/
        └── model.mlmodel  # or model.mlpackage
```

## Model Configuration

### Basic Configuration

```protobuf
name: "my_coreml_model"
platform: "coreml"
max_batch_size: 1
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
```

### Advanced Configuration with Parameters

```protobuf
name: "my_coreml_model"
platform: "coreml"
max_batch_size: 1

# Backend-specific parameters
parameters: {
  key: "compute_units"
  value: {
    string_value: "ALL"  # Options: CPU_ONLY, CPU_AND_GPU, ALL, CPU_AND_NE
  }
}
parameters: {
  key: "use_neural_engine"
  value: {
    string_value: "true"  # Enable/disable Neural Engine
  }
}
parameters: {
  key: "prefer_power_efficiency"
  value: {
    string_value: "false"  # Prefer power efficiency over performance
  }
}

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
```

## Compute Units Options

The `compute_units` parameter controls which hardware units CoreML can use:

- `CPU_ONLY`: Use only the CPU
- `CPU_AND_GPU`: Use CPU and GPU (Metal)
- `ALL`: Use all available compute units (CPU, GPU, and Neural Engine)
- `CPU_AND_NE`: Use CPU and Neural Engine (macOS 11.0+)

## Supported Data Types

Currently supported data types:
- `TYPE_FP32` (Float32)
- `TYPE_FP64` (Float64)
- `TYPE_INT32` (Int32)

## Examples

### Image Classification Model

```protobuf
name: "resnet50"
platform: "coreml"
max_batch_size: 1
parameters: {
  key: "compute_units"
  value: { string_value: "ALL" }
}
input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [ 224, 224, 3 ]
  }
]
output [
  {
    name: "classLabelProbs"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
```

### Object Detection Model

```protobuf
name: "yolov5"
platform: "coreml"
max_batch_size: 1
parameters: {
  key: "compute_units"
  value: { string_value: "CPU_AND_GPU" }
}
parameters: {
  key: "use_neural_engine"
  value: { string_value: "false" }
}
input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [ 640, 640, 3 ]
  }
]
output [
  {
    name: "confidence"
    data_type: TYPE_FP32
    dims: [ 25200, 85 ]
  }
]
```

## Performance Optimization

1. **Neural Engine Usage**: For models that support it, the Neural Engine provides the best performance/watt ratio
2. **Model Compilation**: CoreML models are compiled on first load and cached for subsequent runs
3. **Compute Unit Selection**: Choose appropriate compute units based on your model and hardware:
   - Use `CPU_ONLY` for models with operations not supported by GPU/NE
   - Use `ALL` for maximum performance when all operations are supported
   - Use `CPU_AND_GPU` if Neural Engine compatibility is uncertain

## Limitations

1. **Batch Size**: CoreML doesn't natively support batching. Batch size > 1 will process requests sequentially
2. **Data Types**: Limited to data types supported by CoreML (primarily float32, float64, int32)
3. **Platform**: macOS only (iOS support could be added with modifications)
4. **Model Format**: Only `.mlmodel` and `.mlpackage` formats are supported

## Troubleshooting

### Model Loading Errors

If you encounter model loading errors:
1. Verify the model file exists in the correct location
2. Ensure the model is a valid CoreML model (test with Xcode)
3. Check that input/output names in config.pbtxt match the model

### Performance Issues

1. Check compute unit configuration
2. Monitor system thermal state
3. Ensure models are optimized for CoreML (use coremltools for conversion)

### Memory Issues

1. CoreML models are loaded into memory
2. Large models may require significant RAM
3. Consider model quantization for memory-constrained systems

## Development

### Adding New Features

The backend is designed to be extensible. Key areas for enhancement:
1. Additional data type support
2. Custom preprocessing/postprocessing
3. Model encryption support
4. Performance profiling integration

### Testing

Run the test script to verify the backend is working correctly:

```bash
cd test
python create_test_model.py
../test_coreml_backend_macos.sh
```

## License

This backend is released under the same license as Triton Inference Server. See the LICENSE file for details.