#!/usr/bin/env python3
import time
import numpy as np
import sys
import json

try:
    import tritonclient.http as httpclient
    from transformers import BertTokenizer
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tritonclient[http]", "transformers"])
    import tritonclient.http as httpclient
    from transformers import BertTokenizer

def benchmark_transformer(model_name, text_samples, runs=100):
    """Benchmark transformer model"""
    print(f"\n🔬 Benchmarking {model_name}...")
    
    # Initialize client and tokenizer
    client = httpclient.InferenceServerClient("localhost:8000")
    tokenizer = BertTokenizer.from_pretrained("models/tokenizer")
    
    # Prepare inputs
    max_length = 128
    all_input_ids = []
    all_attention_masks = []
    
    for text in text_samples:
        tokens = tokenizer(text, return_tensors="np", max_length=max_length, 
                          padding="max_length", truncation=True)
        all_input_ids.append(tokens["input_ids"])
        all_attention_masks.append(tokens["attention_mask"])
    
    input_ids = np.vstack(all_input_ids).astype(np.int32)
    attention_mask = np.vstack(all_attention_masks).astype(np.int32)
    
    # Create Triton inputs
    inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT32"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT32")
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)
    
    # Warmup
    print("  Warming up...")
    for _ in range(5):
        client.infer(model_name, inputs)
    
    # Benchmark
    print(f"  Running {runs} inferences...")
    latencies = []
    start_time = time.time()
    
    for _ in range(runs):
        t0 = time.time()
        response = client.infer(model_name, inputs)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)  # ms
    
    end_time = time.time()
    
    # Get results from last inference
    last_hidden = response.as_numpy("last_hidden_state")
    pooler = response.as_numpy("pooler_output")
    
    # Calculate metrics
    total_time = end_time - start_time
    throughput = runs / total_time
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    tokens_per_second = (runs * len(text_samples) * max_length) / total_time
    
    print(f"\n  ✅ Results for {model_name}:")
    print(f"     Throughput: {throughput:.2f} req/s")
    print(f"     Avg Latency: {avg_latency:.2f} ms")
    print(f"     P99 Latency: {p99_latency:.2f} ms")
    print(f"     Tokens/sec: {tokens_per_second:.0f}")
    print(f"     Output shape: {last_hidden.shape}")
    
    return {
        "throughput": throughput,
        "avg_latency": avg_latency,
        "p99_latency": p99_latency,
        "tokens_per_second": tokens_per_second
    }

def main():
    print("🤖 Apple Silicon Transformer Benchmark")
    print("=" * 50)
    
    # Test samples
    text_samples = [
        "The Apple Silicon M1 chip delivers incredible performance.",
        "Neural Engine acceleration makes transformer models fly.",
        "Triton Inference Server now supports Apple Silicon natively.",
        "This is a test of the emergency broadcast system."
    ]
    
    # Check server
    client = httpclient.InferenceServerClient("localhost:8000")
    if not client.is_server_live():
        print("❌ Server not running! Start with: ./QUICK_START.sh")
        sys.exit(1)
    
    results = {}
    
    # Test ANE-optimized model
    try:
        results["ANE (CoreML)"] = benchmark_transformer("bert_ane", text_samples)
    except Exception as e:
        print(f"❌ Error with ANE model: {e}")
    
    # Test PyTorch with MPS
    try:
        results["Metal (PyTorch)"] = benchmark_transformer("bert_pytorch", text_samples)
    except Exception as e:
        print(f"❌ Error with PyTorch model: {e}")
    
    # Print comparison
    if len(results) > 1:
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE COMPARISON")
        print("=" * 50)
        print(f"{'Backend':<20} {'Throughput':<15} {'Latency':<15} {'Tokens/sec':<15}")
        print("-" * 65)
        
        for backend, metrics in results.items():
            print(f"{backend:<20} {metrics['throughput']:<15.2f} {metrics['avg_latency']:<15.2f} {metrics['tokens_per_second']:<15.0f}")
        
        # Calculate speedup
        if "Metal (PyTorch)" in results and "ANE (CoreML)" in results:
            speedup = results["ANE (CoreML)"]["throughput"] / results["Metal (PyTorch)"]["throughput"]
            efficiency = results["ANE (CoreML)"]["tokens_per_second"] / results["Metal (PyTorch)"]["tokens_per_second"]
            print(f"\n🚀 ANE Speedup: {speedup:.2f}x")
            print(f"⚡ Token Processing Efficiency: {efficiency:.2f}x")
    
    # Try to get Apple Silicon metrics
    try:
        import requests
        metrics = requests.get("http://localhost:8002/metrics").text
        
        print("\n🍎 Apple Silicon Metrics:")
        for line in metrics.split('\n'):
            if 'apple_silicon_ane_utilization' in line and not line.startswith('#'):
                print(f"   ANE Utilization: {line.split()[-1]}%")
            elif 'apple_silicon_power_usage' in line and not line.startswith('#'):
                print(f"   Power Usage: {line.split()[-1]}W")
            elif 'apple_silicon_memory_bandwidth' in line and not line.startswith('#'):
                print(f"   Memory Bandwidth: {line.split()[-1]} GB/s")
    except:
        pass
    
    print("\n✅ Benchmark complete!")
    print("\n💡 Tips:")
    print("   - ANE (Neural Engine) is optimized for transformers")
    print("   - Use batch size 1-8 for best ANE performance")
    print("   - Monitor Activity Monitor to see ANE usage")
    print("   - Check Console.app for ANE performance logs")

if __name__ == "__main__":
    main()
