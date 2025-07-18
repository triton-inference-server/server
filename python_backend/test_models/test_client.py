#!/usr/bin/env python3
"""
Simple test client to verify Python backend functionality.
This doesn't use Triton's client library but simulates the basic flow.
"""

import numpy as np
import json

def test_model():
    """Test the simple_python model."""
    print("Testing simple_python model...")
    
    # Create test input
    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    print(f"Input data: {input_data}")
    
    # Expected output (input + 1)
    expected_output = input_data + 1
    print(f"Expected output: {expected_output}")
    
    # In a real scenario, this would send a request to Triton server
    # For now, we'll just print what would happen
    print("\nModel configuration:")
    with open('../simple_python/config.pbtxt', 'r') as f:
        print(f.read())
    
    print("\nModel implementation:")
    with open('../simple_python/1/model.py', 'r') as f:
        # Print first 20 lines
        for i, line in enumerate(f):
            if i < 20:
                print(line.rstrip())
            else:
                print("...")
                break

if __name__ == "__main__":
    test_model()