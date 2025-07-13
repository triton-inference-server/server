#!/bin/bash
# Setup and run transformer model with Apple Silicon optimizations

set -e

echo "🤖 Apple Silicon Transformer Demo Setup"
echo "======================================"

# Create directories
mkdir -p models/bert_ane/1
mkdir -p models/bert_metal/1
mkdir -p models/bert_pytorch/1
mkdir -p test_data

# Function to download and convert BERT model
setup_bert_model() {
    echo "📥 Setting up BERT models..."
    
    # Create model conversion script
    cat > convert_bert_to_coreml.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import numpy as np

print("📦 Installing required packages...")
import subprocess

packages = ["transformers", "torch", "coremltools", "tokenizers"]
for package in packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import torch
from transformers import BertModel, BertTokenizer
import coremltools as ct

print("📥 Downloading BERT model...")
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# Create example input
max_length = 128
example_text = "Hello, this is a test of Apple Silicon optimizations!"
inputs = tokenizer(example_text, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)

print("🔄 Converting to CoreML (for ANE)...")
# Trace the model
traced_model = torch.jit.trace(model, (inputs["input_ids"], inputs["attention_mask"]))

# Convert to CoreML with ANE optimization
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, max_length), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, max_length), dtype=np.int32)
    ],
    outputs=[
        ct.TensorType(name="last_hidden_state"),
        ct.TensorType(name="pooler_output")
    ],
    compute_units=ct.ComputeUnit.ALL,  # Use ANE when available
    convert_to="mlprogram"
)

# Save the model
os.makedirs("models/bert_ane/1", exist_ok=True)
mlmodel.save("models/bert_ane/1/model.mlpackage")

print("✅ CoreML model saved (optimized for ANE)")

# Also save PyTorch version
print("💾 Saving PyTorch version...")
torch.jit.save(traced_model, "models/bert_pytorch/1/model.pt")

# Save tokenizer
tokenizer.save_pretrained("models/tokenizer")

print("✅ Models ready!")
EOF

    python3 convert_bert_to_coreml.py
}

# Create Triton model configurations
create_model_configs() {
    echo "📝 Creating model configurations..."
    
    # BERT ANE (CoreML) configuration
    cat > models/bert_ane/config.pbtxt << 'EOF'
name: "bert_ane"
platform: "coreml"
max_batch_size: 8
input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ 128 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [ 128 ]
  }
]
output [
  {
    name: "last_hidden_state"
    data_type: TYPE_FP32
    dims: [ 128, 768 ]
  },
  {
    name: "pooler_output"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]
parameters: {
  key: "compute_units"
  value: { string_value: "ALL" }
}
parameters: {
  key: "enable_ane"
  value: { string_value: "true" }
}
parameters: {
  key: "optimization_hints"
  value: { string_value: "transformer" }
}
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
optimization {
  execution_accelerators {
    cpu_execution_accelerator: "neural_engine"
  }
}
EOF

    # BERT PyTorch configuration (with MPS)
    cat > models/bert_pytorch/config.pbtxt << 'EOF'
name: "bert_pytorch"
platform: "pytorch"
max_batch_size: 8
input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ 128 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [ 128 ]
  }
]
output [
  {
    name: "last_hidden_state"
    data_type: TYPE_FP32
    dims: [ 128, 768 ]
  },
  {
    name: "pooler_output"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]
parameters: {
  key: "device"
  value: { string_value: "mps" }
}
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
EOF

    # Create a Python wrapper for PyTorch model
    cat > models/bert_pytorch/1/model.py << 'EOF'
import torch
import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        
        # Load the model
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = torch.jit.load(
            f"{args['model_repository']}/1/model.pt",
            map_location=self.device
        )
        self.model.eval()
    
    def execute(self, requests):
        responses = []
        
        for request in requests:
            # Get inputs
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids")
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask")
            
            # Convert to PyTorch tensors
            input_ids_t = torch.from_numpy(input_ids.as_numpy()).to(self.device)
            attention_mask_t = torch.from_numpy(attention_mask.as_numpy()).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_ids_t, attention_mask_t)
            
            # Create output tensors
            last_hidden = pb_utils.Tensor("last_hidden_state", 
                                         outputs[0].cpu().numpy())
            pooler = pb_utils.Tensor("pooler_output", 
                                    outputs[1].cpu().numpy())
            
            responses.append(pb_utils.InferenceResponse([last_hidden, pooler]))
        
        return responses
EOF
}

# Create benchmark client
create_transformer_client() {
    cat > test_transformer.py << 'EOF'
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
EOF
    chmod +x test_transformer.py
}

# Main execution
echo "🔧 Step 1: Setting up transformer models..."
# Check if models already exist
if [ -f "models/bert_ane/1/model.mlpackage/Data/com.apple.CoreML/model.mlmodel" ] && [ -f "models/bert_pytorch/1/model.pt" ]; then
    echo "✅ Models already exist, skipping conversion"
else
    setup_bert_model
fi

echo ""
echo "🔧 Step 2: Creating model configurations..."
create_model_configs

echo ""
echo "🔧 Step 3: Creating benchmark client..."
create_transformer_client

echo ""
echo "🚀 Step 4: Starting Triton Server with Apple Silicon optimizations..."

# Kill any existing server
pkill -f tritonserver 2>/dev/null || true
sleep 2

# Start server with all optimizations
./build/tritonserver \
    --model-repository=$(pwd)/models \
    --backend-config=coreml,enable_ane=true \
    --backend-config=coreml,compute_precision=mixed \
    --backend-config=pytorch,device=mps \
    --backend-config=model_router,enable=true \
    --backend-config=model_router,policy=performance \
    --backend-config=profile_guided,enable=true \
    --allow-metrics=true \
    --log-verbose=1 \
    > triton_transformer.log 2>&1 &

SERVER_PID=$!

echo "   Server PID: $SERVER_PID"
echo "   Waiting for server to start..."

# Wait for server
for i in {1..30}; do
    if curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
        echo "   ✅ Server ready!"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

if ! curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
    echo "❌ Server failed to start. Check triton_transformer.log"
    tail -20 triton_transformer.log
    exit 1
fi

echo ""
echo "🎯 Step 5: Running transformer benchmark..."
echo ""

python3 test_transformer.py

echo ""
echo "📊 Additional Commands:"
echo "   Monitor ANE usage: watch -n 1 'powermetrics --samplers tasks | grep -A20 Neural'"
echo "   View server logs: tail -f triton_transformer.log"
echo "   Stop server: kill $SERVER_PID"
echo ""
echo "🎉 Transformer demo complete! The ANE should show significant speedup."