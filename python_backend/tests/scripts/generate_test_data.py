#!/usr/bin/env python3
"""
Generate test data for Triton Python Backend tests
"""

import os
import json
import numpy as np
import struct
import argparse
from pathlib import Path

class TestDataGenerator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_tensor_data(self, name, shape, dtype='float32', pattern='random'):
        """Generate tensor data with various patterns"""
        
        if dtype == 'float32':
            np_dtype = np.float32
        elif dtype == 'float64':
            np_dtype = np.float64
        elif dtype == 'int32':
            np_dtype = np.int32
        elif dtype == 'int64':
            np_dtype = np.int64
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        # Generate data based on pattern
        if pattern == 'random':
            data = np.random.randn(*shape).astype(np_dtype)
        elif pattern == 'zeros':
            data = np.zeros(shape, dtype=np_dtype)
        elif pattern == 'ones':
            data = np.ones(shape, dtype=np_dtype)
        elif pattern == 'sequential':
            data = np.arange(np.prod(shape)).reshape(shape).astype(np_dtype)
        elif pattern == 'gradient':
            # Create a gradient pattern
            data = np.linspace(0, 1, np.prod(shape)).reshape(shape).astype(np_dtype)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Save in multiple formats
        # 1. NumPy format
        np_path = self.output_dir / f"{name}.npy"
        np.save(np_path, data)
        
        # 2. Binary format
        bin_path = self.output_dir / f"{name}.bin"
        data.tofile(bin_path)
        
        # 3. JSON format (for small tensors)
        if data.size < 1000:
            json_path = self.output_dir / f"{name}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'shape': list(shape),
                    'dtype': dtype,
                    'data': data.tolist()
                }, f, indent=2)
        
        # 4. Metadata
        meta_path = self.output_dir / f"{name}.meta"
        with open(meta_path, 'w') as f:
            json.dump({
                'name': name,
                'shape': list(shape),
                'dtype': dtype,
                'pattern': pattern,
                'size_bytes': data.nbytes,
                'checksum': int(np.sum(data))
            }, f, indent=2)
        
        return data
    
    def generate_test_cases(self):
        """Generate various test cases"""
        
        test_cases = []
        
        # 1. Simple vectors
        for size in [10, 100, 1000]:
            name = f"vector_{size}"
            data = self.generate_tensor_data(name, (size,), 'float32', 'random')
            test_cases.append({
                'name': name,
                'expected_output': data + 1  # Simple add operation
            })
        
        # 2. Matrices
        for rows, cols in [(10, 10), (100, 50), (32, 64)]:
            name = f"matrix_{rows}x{cols}"
            data = self.generate_tensor_data(name, (rows, cols), 'float32', 'random')
            test_cases.append({
                'name': name,
                'expected_output': data * 2  # Simple multiply operation
            })
        
        # 3. Batched data
        for batch_size in [1, 8, 16, 32]:
            name = f"batch_{batch_size}"
            data = self.generate_tensor_data(name, (batch_size, 10), 'float32', 'sequential')
            test_cases.append({
                'name': name,
                'expected_output': data  # Identity operation
            })
        
        # 4. Different data types
        for dtype in ['float32', 'int32', 'int64']:
            name = f"dtype_{dtype}"
            data = self.generate_tensor_data(name, (50,), dtype, 'ones')
            test_cases.append({
                'name': name,
                'expected_output': data * 2
            })
        
        # 5. Edge cases
        # Empty tensor
        self.generate_tensor_data('empty', (0,), 'float32', 'zeros')
        
        # Single element
        self.generate_tensor_data('scalar', (1,), 'float32', 'ones')
        
        # Large tensor
        self.generate_tensor_data('large', (1000, 1000), 'float32', 'gradient')
        
        # Save test case manifest
        manifest_path = self.output_dir / 'test_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(test_cases, f, indent=2)
        
        return test_cases
    
    def generate_model_configs(self):
        """Generate sample model configurations"""
        
        configs_dir = self.output_dir / 'configs'
        configs_dir.mkdir(exist_ok=True)
        
        # 1. Simple model config
        simple_config = """
name: "simple_model"
backend: "python"
max_batch_size: 8

input [
  {
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]

output [
  {
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
"""
        with open(configs_dir / 'simple_model.pbtxt', 'w') as f:
            f.write(simple_config)
        
        # 2. Multi-input model config
        multi_input_config = """
name: "multi_input_model"
backend: "python"
max_batch_size: 16

input [
  {
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  },
  {
    name: "INPUT1"
    data_type: TYPE_INT32
    dims: [ 5 ]
  }
]

output [
  {
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [ 15 ]
  }
]
"""
        with open(configs_dir / 'multi_input_model.pbtxt', 'w') as f:
            f.write(multi_input_config)
        
        # 3. Dynamic batching config
        dynamic_batch_config = """
name: "dynamic_batch_model"
backend: "python"
max_batch_size: 32

input [
  {
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [ 224, 224, 3 ]
  }
]

output [
  {
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100000
  preserve_ordering: true
}

instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]
"""
        with open(configs_dir / 'dynamic_batch_model.pbtxt', 'w') as f:
            f.write(dynamic_batch_config)
    
    def generate_performance_data(self):
        """Generate data for performance testing"""
        
        perf_dir = self.output_dir / 'performance'
        perf_dir.mkdir(exist_ok=True)
        
        # Different sizes for performance testing
        sizes = [
            (100,),           # Small vector
            (1000,),          # Medium vector
            (10000,),         # Large vector
            (100, 100),       # Small matrix
            (1000, 1000),     # Large matrix
            (32, 224, 224, 3) # Batch of images
        ]
        
        for i, shape in enumerate(sizes):
            name = f"perf_test_{i}"
            self.generate_tensor_data(f"performance/{name}", shape, 'float32', 'random')

def main():
    parser = argparse.ArgumentParser(description='Generate test data for Triton Python Backend')
    parser.add_argument('--output-dir', default='tests/data', 
                       help='Output directory for test data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Generate test data
    generator = TestDataGenerator(args.output_dir)
    
    print("Generating test cases...")
    test_cases = generator.generate_test_cases()
    print(f"Generated {len(test_cases)} test cases")
    
    print("Generating model configurations...")
    generator.generate_model_configs()
    
    print("Generating performance test data...")
    generator.generate_performance_data()
    
    print(f"\nTest data generated in: {args.output_dir}")
    print("Files created:")
    for file in sorted(Path(args.output_dir).rglob('*')):
        if file.is_file():
            print(f"  - {file.relative_to(args.output_dir)}")

if __name__ == '__main__':
    main()