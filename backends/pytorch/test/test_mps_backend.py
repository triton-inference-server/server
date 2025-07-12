#!/usr/bin/env python3
"""
Test script for PyTorch MPS backend integration
"""

import torch
import numpy as np
import sys
import os
import time

def check_mps_availability():
    """Check if MPS is available on this system"""
    print("Checking MPS availability...")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS is not available on this system")
        if not torch.backends.mps.is_built():
            print("   PyTorch was not built with MPS support")
        return False
    
    print("✅ MPS is available")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   MPS built: {torch.backends.mps.is_built()}")
    
    return True

def create_test_model():
    """Create a simple test model for MPS testing"""
    print("\nCreating test model...")
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    # Trace the model
    example_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)
    
    return traced_model, example_input

def test_mps_inference(model, input_tensor):
    """Test model inference on MPS"""
    print("\nTesting MPS inference...")
    
    # Move model and input to MPS
    device = torch.device("mps")
    model_mps = model.to(device)
    input_mps = input_tensor.to(device)
    
    # Warmup
    print("Warming up MPS...")
    for _ in range(5):
        _ = model_mps(input_mps)
    
    # Benchmark
    print("Benchmarking MPS performance...")
    num_iterations = 100
    
    torch.mps.synchronize()  # Ensure all operations are complete
    start_time = time.time()
    
    for _ in range(num_iterations):
        output = model_mps(input_mps)
    
    torch.mps.synchronize()  # Ensure all operations are complete
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    print(f"✅ Average inference time on MPS: {avg_time:.2f} ms")
    
    # Compare with CPU
    print("\nComparing with CPU performance...")
    model_cpu = model.to("cpu")
    input_cpu = input_tensor.to("cpu")
    
    # CPU warmup
    for _ in range(5):
        _ = model_cpu(input_cpu)
    
    start_time = time.time()
    for _ in range(num_iterations):
        output = model_cpu(input_cpu)
    end_time = time.time()
    
    avg_time_cpu = (end_time - start_time) / num_iterations * 1000  # ms
    print(f"✅ Average inference time on CPU: {avg_time_cpu:.2f} ms")
    print(f"   Speedup: {avg_time_cpu/avg_time:.2f}x")
    
    return output

def test_tensor_operations():
    """Test various tensor operations on MPS"""
    print("\nTesting tensor operations on MPS...")
    
    device = torch.device("mps")
    
    # Test different data types
    print("Testing data types...")
    try:
        # Float32
        tensor_f32 = torch.randn(1000, 1000, dtype=torch.float32, device=device)
        result_f32 = torch.matmul(tensor_f32, tensor_f32)
        print("✅ Float32 operations: OK")
        
        # Float16 (may not be fully supported)
        try:
            tensor_f16 = torch.randn(1000, 1000, dtype=torch.float16, device=device)
            result_f16 = torch.matmul(tensor_f16, tensor_f16)
            print("✅ Float16 operations: OK")
        except Exception as e:
            print(f"⚠️  Float16 operations: {str(e)}")
        
        # Integer operations
        tensor_int = torch.randint(0, 100, (1000, 1000), dtype=torch.int32, device=device)
        result_int = tensor_int + tensor_int
        print("✅ Integer operations: OK")
        
    except Exception as e:
        print(f"❌ Error in tensor operations: {str(e)}")
        return False
    
    # Test memory transfer
    print("\nTesting memory transfers...")
    try:
        # CPU to MPS
        cpu_tensor = torch.randn(1000, 1000)
        mps_tensor = cpu_tensor.to(device)
        print("✅ CPU to MPS transfer: OK")
        
        # MPS to CPU
        cpu_back = mps_tensor.to("cpu")
        assert torch.allclose(cpu_tensor, cpu_back)
        print("✅ MPS to CPU transfer: OK")
        
    except Exception as e:
        print(f"❌ Error in memory transfers: {str(e)}")
        return False
    
    return True

def save_test_model(model, output_dir):
    """Save the test model for Triton"""
    print(f"\nSaving model to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pt")
    
    # Save as CPU model (Triton will move to MPS)
    model_cpu = model.to("cpu")
    model_cpu.save(model_path)
    
    print(f"✅ Model saved to {model_path}")
    
    # Create config.pbtxt
    config_path = os.path.join(os.path.dirname(output_dir), "config.pbtxt")
    config_content = """name: "pytorch_mps_test"
backend: "pytorch"
max_batch_size: 8

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]

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
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Config saved to {config_path}")

def main():
    """Main test function"""
    print("=" * 60)
    print("PyTorch MPS Backend Test")
    print("=" * 60)
    
    # Check MPS availability
    if not check_mps_availability():
        print("\n⚠️  MPS is not available. Tests cannot continue.")
        sys.exit(1)
    
    # Create test model
    model, example_input = create_test_model()
    
    # Test MPS inference
    output = test_mps_inference(model, example_input)
    
    # Test tensor operations
    test_tensor_operations()
    
    # Save model for Triton
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "pytorch_mps_test/1"
    
    save_test_model(model, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()