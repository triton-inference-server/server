#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Test client for CoreML backend in Triton Inference Server.
"""

import argparse
import numpy as np
import time
import sys

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
except ImportError:
    print("tritonclient is not installed. Please install with:")
    print("pip install tritonclient[grpc]")
    sys.exit(1)

def test_simple_model(client, model_name="coreml_simple", verbose=False):
    """Test the simple CoreML model."""
    print(f"\nTesting {model_name}...")
    
    # Check if model is ready
    if not client.is_model_ready(model_name):
        print(f"Model {model_name} is not ready!")
        return False
    
    # Get model metadata
    metadata = client.get_model_metadata(model_name)
    if verbose:
        print(f"Model metadata: {metadata}")
    
    # Get model config
    config = client.get_model_config(model_name)
    if verbose:
        print(f"Model config: {config}")
    
    # Create input data
    input_data = np.random.randn(1, 10).astype(np.float32)
    
    # Create input tensor
    inputs = []
    inputs.append(grpcclient.InferInput('input', input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)
    
    # Create output tensor
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('output'))
    
    # Perform inference
    print(f"Input shape: {input_data.shape}")
    print(f"Input data (first 5 values): {input_data[0][:5]}")
    
    start_time = time.time()
    try:
        response = client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        inference_time = time.time() - start_time
        
        # Get output
        output_data = response.as_numpy('output')
        print(f"Output shape: {output_data.shape}")
        print(f"Output data: {output_data[0]}")
        print(f"Inference time: {inference_time*1000:.2f} ms")
        
        return True
        
    except InferenceServerException as e:
        print(f"Inference failed: {e}")
        return False

def test_performance(client, model_name="coreml_simple", num_requests=100):
    """Test performance of the CoreML model."""
    print(f"\nPerformance testing {model_name}...")
    print(f"Number of requests: {num_requests}")
    
    # Prepare input data
    input_data = np.random.randn(1, 10).astype(np.float32)
    
    # Create input tensor
    inputs = []
    inputs.append(grpcclient.InferInput('input', input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)
    
    # Create output tensor
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('output'))
    
    # Warm up
    print("Warming up...")
    for _ in range(10):
        client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    
    # Performance test
    print("Running performance test...")
    latencies = []
    
    for i in range(num_requests):
        start_time = time.time()
        try:
            response = client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_requests} requests")
                
        except InferenceServerException as e:
            print(f"Request {i+1} failed: {e}")
    
    # Calculate statistics
    if latencies:
        latencies = np.array(latencies)
        print(f"\nPerformance Statistics:")
        print(f"  Mean latency: {np.mean(latencies):.2f} ms")
        print(f"  Min latency: {np.min(latencies):.2f} ms")
        print(f"  Max latency: {np.max(latencies):.2f} ms")
        print(f"  Std dev: {np.std(latencies):.2f} ms")
        print(f"  P50: {np.percentile(latencies, 50):.2f} ms")
        print(f"  P90: {np.percentile(latencies, 90):.2f} ms")
        print(f"  P95: {np.percentile(latencies, 95):.2f} ms")
        print(f"  P99: {np.percentile(latencies, 99):.2f} ms")
        print(f"  Throughput: {1000.0 / np.mean(latencies):.2f} requests/second")

def main():
    parser = argparse.ArgumentParser(description='Test CoreML backend')
    parser.add_argument('--url', type=str, default='localhost:8001',
                        help='Inference server URL (default: localhost:8001)')
    parser.add_argument('--model', type=str, default='coreml_simple',
                        help='Model name (default: coreml_simple)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--performance', action='store_true',
                        help='Run performance test')
    parser.add_argument('--requests', type=int, default=100,
                        help='Number of requests for performance test (default: 100)')
    
    args = parser.parse_args()
    
    # Create client
    try:
        client = grpcclient.InferenceServerClient(url=args.url)
    except Exception as e:
        print(f"Failed to create client: {e}")
        return
    
    # Check server health
    if not client.is_server_live():
        print("Server is not live!")
        return
    
    if not client.is_server_ready():
        print("Server is not ready!")
        return
    
    print(f"Connected to Triton at {args.url}")
    
    # Get server metadata
    server_metadata = client.get_server_metadata()
    print(f"Server: {server_metadata.name} v{server_metadata.version}")
    
    # Test model
    success = test_simple_model(client, args.model, args.verbose)
    
    if success and args.performance:
        test_performance(client, args.model, args.requests)

if __name__ == '__main__':
    main()