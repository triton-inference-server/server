#!/usr/bin/env python3
"""
Test client for PyTorch backend
"""

import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import sys
import argparse

def test_pytorch_model(client, model_name="pytorch_simple", verbose=False):
    """Test the PyTorch model with sample inputs"""
    
    # Create test inputs
    input0_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input1_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Create input objects
    inputs = []
    inputs.append(client.InferInput("INPUT0", input0_data.shape, "FP32"))
    inputs.append(client.InferInput("INPUT1", input1_data.shape, "FP32"))
    
    # Set data
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)
    
    # Create output objects
    outputs = []
    outputs.append(client.InferRequestedOutput("OUTPUT0"))
    outputs.append(client.InferRequestedOutput("OUTPUT1"))
    
    # Perform inference
    print(f"Running inference on model '{model_name}'...")
    try:
        results = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        
        # Get outputs
        output0_data = results.as_numpy("OUTPUT0")
        output1_data = results.as_numpy("OUTPUT1")
        
        # Verify results
        expected_add = input0_data + input1_data
        expected_sub = input0_data - input1_data
        
        add_correct = np.allclose(output0_data, expected_add, rtol=1e-5)
        sub_correct = np.allclose(output1_data, expected_sub, rtol=1e-5)
        
        print(f"Addition result correct: {add_correct}")
        print(f"Subtraction result correct: {sub_correct}")
        
        if verbose:
            print(f"\nInput shapes: {input0_data.shape}, {input1_data.shape}")
            print(f"Output shapes: {output0_data.shape}, {output1_data.shape}")
            print(f"\nSample values (first 3 elements):")
            print(f"Input0: {input0_data.flat[:3]}")
            print(f"Input1: {input1_data.flat[:3]}")
            print(f"Output0 (add): {output0_data.flat[:3]}")
            print(f"Output1 (sub): {output1_data.flat[:3]}")
        
        if add_correct and sub_correct:
            print("\nTEST PASSED!")
            return True
        else:
            print("\nTEST FAILED!")
            return False
            
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test PyTorch backend')
    parser.add_argument('--protocol', type=str, default='http', choices=['http', 'grpc'],
                        help='Protocol to use (default: http)')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Triton server host (default: localhost)')
    parser.add_argument('--port', type=int, default=None,
                        help='Triton server port (default: 8000 for http, 8001 for grpc)')
    parser.add_argument('--model', type=str, default='pytorch_simple',
                        help='Model name (default: pytorch_simple)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set default port based on protocol
    if args.port is None:
        args.port = 8000 if args.protocol == 'http' else 8001
    
    # Create client
    try:
        if args.protocol == 'http':
            client = httpclient.InferenceServerClient(f"{args.host}:{args.port}")
        else:
            client = grpcclient.InferenceServerClient(f"{args.host}:{args.port}")
        
        # Check if server is alive
        if not client.is_server_live():
            print(f"Error: Triton server at {args.host}:{args.port} is not responding")
            return 1
        
        # Check if server is ready
        if not client.is_server_ready():
            print(f"Error: Triton server at {args.host}:{args.port} is not ready")
            return 1
        
        # Check if model is ready
        if not client.is_model_ready(args.model):
            print(f"Error: Model '{args.model}' is not loaded")
            print("Available models:")
            for model in client.get_model_repository_index():
                print(f"  - {model['name']}")
            return 1
        
        # Run test
        success = test_pytorch_model(client, args.model, args.verbose)
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())