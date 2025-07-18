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

import sys
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import time

def test_mps_inference(model_name, server_url='localhost:8000'):
    """Test inference with MPS backend"""
    
    try:
        # Create client
        client = httpclient.InferenceServerClient(url=server_url)
        
        # Check if model is loaded
        if not client.is_model_ready(model_name):
            print(f"Model {model_name} is not ready")
            return False
        
        # Get model metadata
        metadata = client.get_model_metadata(model_name)
        print(f"\nModel: {model_name}")
        print(f"Metadata: {metadata}")
        
        # Get model config
        config = client.get_model_config(model_name)
        print(f"Config: {config}")
        
        # Prepare input data based on model
        if model_name == "mps_simple":
            input_shape = [1, 3, 224, 224]
            output_shape = [1, 64, 56, 56]
        elif model_name == "mps_resnet_block":
            input_shape = [1, 64, 56, 56]
            output_shape = [1, 64, 56, 56]
        else:
            print(f"Unknown model: {model_name}")
            return False
        
        # Create input data
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Create input tensor
        inputs = []
        inputs.append(httpclient.InferInput('input', input_shape, "FP32"))
        inputs[0].set_data_from_numpy(input_data)
        
        # Request output
        outputs = []
        outputs.append(httpclient.InferRequestedOutput('output'))
        
        # Run inference
        print(f"\nRunning inference on {model_name}...")
        start_time = time.time()
        
        response = client.infer(
            model_name,
            inputs,
            outputs=outputs
        )
        
        inference_time = (time.time() - start_time) * 1000
        print(f"Inference time: {inference_time:.2f} ms")
        
        # Get output
        output_data = response.as_numpy('output')
        print(f"Output shape: {output_data.shape}")
        print(f"Output stats - min: {output_data.min():.4f}, max: {output_data.max():.4f}, mean: {output_data.mean():.4f}")
        
        # Verify output shape
        expected_shape = tuple(output_shape)
        if output_data.shape != expected_shape:
            print(f"ERROR: Expected output shape {expected_shape}, got {output_data.shape}")
            return False
        
        print(f"\n✓ Test passed for {model_name}")
        return True
        
    except InferenceServerException as e:
        print(f"Inference failed: {e}")
        return False

def benchmark_model(model_name, num_requests=100, server_url='localhost:8000'):
    """Benchmark model performance"""
    
    try:
        client = httpclient.InferenceServerClient(url=server_url)
        
        if not client.is_model_ready(model_name):
            print(f"Model {model_name} is not ready")
            return
        
        # Prepare input
        if model_name == "mps_simple":
            input_shape = [1, 3, 224, 224]
        elif model_name == "mps_resnet_block":
            input_shape = [1, 64, 56, 56]
        else:
            print(f"Unknown model: {model_name}")
            return
        
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        inputs = []
        inputs.append(httpclient.InferInput('input', input_shape, "FP32"))
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = []
        outputs.append(httpclient.InferRequestedOutput('output'))
        
        # Warmup
        print(f"\nWarming up {model_name}...")
        for _ in range(10):
            client.infer(model_name, inputs, outputs=outputs)
        
        # Benchmark
        print(f"Benchmarking {model_name} with {num_requests} requests...")
        latencies = []
        
        for i in range(num_requests):
            start_time = time.time()
            client.infer(model_name, inputs, outputs=outputs)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")
        
        # Calculate statistics
        latencies = np.array(latencies)
        print(f"\nBenchmark Results for {model_name}:")
        print(f"  Average latency: {latencies.mean():.2f} ms")
        print(f"  Median latency: {np.median(latencies):.2f} ms")
        print(f"  P95 latency: {np.percentile(latencies, 95):.2f} ms")
        print(f"  P99 latency: {np.percentile(latencies, 99):.2f} ms")
        print(f"  Min latency: {latencies.min():.2f} ms")
        print(f"  Max latency: {latencies.max():.2f} ms")
        print(f"  Throughput: {1000 / latencies.mean():.2f} requests/sec")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")

def main():
    """Main test function"""
    
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    else:
        server_url = 'localhost:8000'
    
    print(f"Testing MPS backend with server at {server_url}")
    
    # Test models
    models_to_test = ['mps_simple', 'mps_resnet_block']
    
    all_passed = True
    for model_name in models_to_test:
        if not test_mps_inference(model_name, server_url):
            all_passed = False
    
    if all_passed:
        print("\n" + "="*50)
        print("All tests passed! Running benchmarks...")
        print("="*50)
        
        for model_name in models_to_test:
            benchmark_model(model_name, num_requests=100, server_url=server_url)
    else:
        print("\nSome tests failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()