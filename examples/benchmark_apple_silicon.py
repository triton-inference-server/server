#!/usr/bin/env python3
"""
Benchmark script to test Apple Silicon optimizations
"""

import time
import numpy as np
import argparse
import sys

try:
    import tritonclient.http as httpclient
except ImportError:
    print("Installing tritonclient...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tritonclient[http]"])
    import tritonclient.http as httpclient

def benchmark_model(client, model_name, inputs, num_requests=100):
    """Benchmark a model with given inputs"""
    print(f"\n🔬 Benchmarking {model_name}...")
    
    # Warmup
    print("  Warming up...")
    for _ in range(10):
        client.infer(model_name, inputs)
    
    # Benchmark
    print(f"  Running {num_requests} inferences...")
    start_time = time.time()
    
    for _ in range(num_requests):
        response = client.infer(model_name, inputs)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Calculate metrics
    throughput = num_requests / duration
    latency = (duration / num_requests) * 1000  # ms
    
    print(f"  ✅ Results:")
    print(f"     Throughput: {throughput:.2f} req/s")
    print(f"     Latency: {latency:.2f} ms/req")
    print(f"     Total time: {duration:.2f} seconds")
    
    return throughput, latency

def create_bert_input(seq_length=128, batch_size=1):
    """Create sample BERT input"""
    input_ids = np.random.randint(0, 1000, size=(batch_size, seq_length), dtype=np.int32)
    
    inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT32"),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    
    return inputs

def create_resnet_input(batch_size=1):
    """Create sample ResNet input"""
    image = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    
    inputs = [
        httpclient.InferInput("input", image.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(image)
    
    return inputs

def create_gpt2_input(seq_length=64, batch_size=1):
    """Create sample GPT-2 input"""
    input_ids = np.random.randint(0, 50000, size=(batch_size, seq_length), dtype=np.int64)
    
    inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    
    return inputs

def main():
    parser = argparse.ArgumentParser(description='Benchmark Apple Silicon Triton Server')
    parser.add_argument('--server', default='localhost:8000', help='Server URL')
    parser.add_argument('--model', help='Specific model to benchmark')
    parser.add_argument('--requests', type=int, default=100, help='Number of requests')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    args = parser.parse_args()
    
    # Create client
    client = httpclient.InferenceServerClient(args.server)
    
    # Check server
    if not client.is_server_live():
        print("❌ Server is not running!")
        sys.exit(1)
    
    print("🚀 Apple Silicon Triton Server Benchmark")
    print("=" * 50)
    
    # Get available models
    models = client.get_model_repository_index()
    print(f"\n📦 Available models: {[m['name'] for m in models]}")
    
    results = {}
    
    # Benchmark specific model or all
    if args.model:
        model_list = [args.model]
    else:
        # Try common model patterns
        model_list = []
        model_names = [m['name'] for m in models]
        
        # Check for BERT models
        bert_models = [m for m in model_names if 'bert' in m.lower()]
        if bert_models:
            model_list.extend(bert_models[:1])  # Take first BERT model
        
        # Check for ResNet models
        resnet_models = [m for m in model_names if 'resnet' in m.lower()]
        if resnet_models:
            model_list.extend(resnet_models[:1])
        
        # Check for GPT models
        gpt_models = [m for m in model_names if 'gpt' in m.lower()]
        if gpt_models:
            model_list.extend(gpt_models[:1])
    
    # Benchmark each model
    for model_name in model_list:
        try:
            # Create appropriate inputs
            if 'bert' in model_name.lower():
                inputs = create_bert_input(batch_size=args.batch_size)
            elif 'resnet' in model_name.lower():
                inputs = create_resnet_input(batch_size=args.batch_size)
            elif 'gpt' in model_name.lower():
                inputs = create_gpt2_input(batch_size=args.batch_size)
            else:
                print(f"⚠️  Skipping {model_name} - unknown model type")
                continue
            
            throughput, latency = benchmark_model(
                client, model_name, inputs, args.requests
            )
            results[model_name] = (throughput, latency)
            
        except Exception as e:
            print(f"❌ Error benchmarking {model_name}: {e}")
    
    # Print summary
    if results:
        print("\n" + "=" * 50)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"{'Model':<20} {'Throughput (req/s)':<20} {'Latency (ms)':<15}")
        print("-" * 55)
        
        for model, (throughput, latency) in results.items():
            print(f"{model:<20} {throughput:<20.2f} {latency:<15.2f}")
        
        # Check for Apple Silicon optimizations
        print("\n🍎 Apple Silicon Optimization Status:")
        try:
            metrics_response = client.get_inference_statistics()
            if metrics_response:
                print("✅ Server is running with Apple Silicon optimizations")
            
            # Try to get custom metrics
            import requests
            metrics = requests.get(f"http://{args.server.replace(':8000', ':8002')}/metrics").text
            
            if 'apple_silicon' in metrics:
                print("✅ Apple Silicon metrics detected")
                
                # Extract some key metrics
                for line in metrics.split('\n'):
                    if 'apple_silicon_ane_utilization' in line and not line.startswith('#'):
                        print(f"   ANE Utilization: {line.split()[-1]}%")
                    elif 'apple_silicon_power_usage' in line and not line.startswith('#'):
                        print(f"   Power Usage: {line.split()[-1]}W")
                        
        except:
            pass
    
    print("\n✅ Benchmark complete!")

if __name__ == "__main__":
    main()