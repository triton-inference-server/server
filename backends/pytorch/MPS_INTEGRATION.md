# PyTorch MPS Integration for Triton Inference Server

This document describes the Metal Performance Shaders (MPS) integration for the PyTorch backend in Triton Inference Server.

## Overview

The PyTorch backend now supports Apple's Metal Performance Shaders (MPS) for accelerated inference on macOS devices with Apple Silicon (M1, M2, M3, etc.) or AMD GPUs. This integration enables efficient execution of PyTorch models on Mac hardware using the unified memory architecture.

## Features

### 1. Automatic Device Detection
- The backend automatically detects MPS availability at runtime
- Falls back to CPU if MPS is not available
- Logs device information during initialization

### 2. Device Management
- Support for `KIND_GPU` instance group configuration (maps to MPS on macOS)
- Explicit MPS device selection through optimization settings
- Proper device assignment for model and tensors

### 3. Memory Management
- Efficient CPU-to-MPS tensor transfers
- Automatic synchronization of MPS operations
- Zero-copy optimization where possible using unified memory
- Proper handling of output tensors (automatic transfer back to CPU)

### 4. Execution Optimization
- Batch processing support on MPS
- Automatic gradient computation disabling for inference
- Model evaluation mode enforcement

## Configuration

### Basic MPS Configuration

```protobuf
name: "my_pytorch_model"
backend: "pytorch"
max_batch_size: 8

instance_group [
  {
    count: 1
    kind: KIND_GPU  # Will use MPS on macOS if available
  }
]
```

### Explicit MPS Configuration

```protobuf
name: "my_pytorch_model"
backend: "pytorch"
max_batch_size: 8

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

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

### Model Warmup

For optimal performance, configure model warmup:

```protobuf
model_warmup [
  {
    name: "warmup"
    batch_size: 1
    inputs {
      key: "input_name"
      value: {
        data_type: TYPE_FP32
        dims: [ 3, 224, 224 ]
        zero_data: true
      }
    }
  }
]
```

## Usage

### 1. Model Preparation

Save your PyTorch model as TorchScript:

```python
import torch
import torchvision.models as models

# Create and load your model
model = models.resnet50(pretrained=True)
model.eval()

# Trace the model
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Save the model
traced_model.save("model.pt")
```

### 2. Model Repository Structure

```
model_repository/
└── my_model/
    ├── config.pbtxt
    └── 1/
        └── model.pt
```

### 3. Running with MPS

The backend will automatically use MPS when:
- Running on macOS with Apple Silicon or supported AMD GPU
- PyTorch is built with MPS support
- `KIND_GPU` is specified in the instance group

## Performance Considerations

1. **First Inference**: The first inference may be slower due to Metal shader compilation. Use model warmup to mitigate this.

2. **Batch Size**: Experiment with different batch sizes to find optimal performance for your model and hardware.

3. **Memory Usage**: MPS uses unified memory, which can be more efficient but has different characteristics than discrete GPU memory.

4. **Data Types**: MPS performs best with FP32. FP16 support varies by model and operation.

## Debugging

### Check MPS Availability

The backend logs MPS availability during initialization:

```
I0101 00:00:00.000000 12345 pytorch.cc:65] MPS (Metal Performance Shaders) is available on this system
I0101 00:00:00.000000 12345 pytorch.cc:70] PyTorch was built with MPS support
```

### Device Assignment

The backend logs device assignment for each model instance:

```
I0101 00:00:00.000000 12345 pytorch.cc:308] Loaded PyTorch model from: /path/to/model.pt on device: mps
I0101 00:00:00.000000 12345 pytorch.cc:638] PyTorch instance initialized on device: mps
```

### Troubleshooting

1. **MPS Not Available**: Ensure you're running on supported hardware and have PyTorch built with MPS support.

2. **Performance Issues**: 
   - Enable model warmup
   - Check batch size optimization
   - Monitor memory usage with Activity Monitor

3. **Accuracy Issues**: Some operations may have different numerical precision on MPS. Validate model outputs against CPU execution.

## Limitations

1. Not all PyTorch operations are supported on MPS. Unsupported ops will fail with an error.
2. Dynamic shapes may have performance implications on MPS.
3. Multi-GPU configurations are not supported on MPS (macOS limitation).

## Future Enhancements

1. **Memory Pool Integration**: Direct integration with Metal memory allocator for improved memory management.
2. **Stream Management**: Explicit control over Metal command queues for better concurrency.
3. **Profiling Support**: Integration with Metal performance tools for detailed profiling.
4. **Mixed Precision**: Automatic mixed precision support for MPS.

## Example Code

See the `test/models/pytorch_mps_example/` directory for a complete example configuration.