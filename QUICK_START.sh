#!/bin/bash
# Quick Start Script for Apple Silicon Triton Server

set -e

echo "🚀 Apple Silicon Triton Server - Quick Start"
echo "==========================================="

# Check if we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ Error: This script is for macOS only"
    exit 1
fi

# Check for Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "⚠️  Warning: Not running on Apple Silicon. Performance will be limited."
fi

# Function to create a simple test model
create_test_model() {
    echo "📦 Creating test model repository..."
    mkdir -p models/simple_add/1
    
    cat > models/simple_add/config.pbtxt << 'EOF'
name: "simple_add"
platform: "python"
max_batch_size: 8
input [
  {
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [ -1 ]
  },
  {
    name: "INPUT1"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
output [
  {
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
EOF

    cat > models/simple_add/1/model.py << 'EOF'
import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1").as_numpy()
            out = in_0 + in_1
            out_tensor = pb_utils.Tensor("OUTPUT", out)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
EOF
}

# Function to create test client
create_test_client() {
    cat > test_client.py << 'EOF'
#!/usr/bin/env python3
import sys
import numpy as np

try:
    import tritonclient.http as httpclient
except ImportError:
    print("Installing tritonclient...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tritonclient[http]"])
    import tritonclient.http as httpclient

# Create client
client = httpclient.InferenceServerClient("localhost:8000")

# Check server health
if client.is_server_live():
    print("✅ Server is running!")
else:
    print("❌ Server is not responding")
    sys.exit(1)

# Prepare inputs
input0_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
input1_data = np.array([4.0, 5.0, 6.0], dtype=np.float32)

inputs = [
    httpclient.InferInput("INPUT0", input0_data.shape, "FP32"),
    httpclient.InferInput("INPUT1", input1_data.shape, "FP32")
]
inputs[0].set_data_from_numpy(input0_data)
inputs[1].set_data_from_numpy(input1_data)

# Request output
outputs = [httpclient.InferRequestedOutput("OUTPUT")]

# Run inference
print(f"🔄 Running inference: {input0_data} + {input1_data}")
response = client.infer("simple_add", inputs, outputs=outputs)

# Get result
result = response.as_numpy("OUTPUT")
print(f"✅ Result: {result}")
print(f"🎉 Inference successful! Server is working correctly.")

# Show server stats
print("\n📊 Server Statistics:")
stats = client.get_inference_statistics()
if stats:
    for model in stats.get('model_stats', []):
        print(f"  Model: {model.get('name', 'unknown')}")
        print(f"  Inference Count: {model.get('inference_count', 0)}")
EOF
    chmod +x test_client.py
}

# Main execution
echo ""
echo "1️⃣  Checking build status..."
if [ ! -f "build/tritonserver" ]; then
    echo "   Building Triton Server..."
    if [ -f "build_macos.sh" ]; then
        ./build_macos.sh
    else
        mkdir -p build && cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j$(sysctl -n hw.ncpu)
        cd ..
    fi
else
    echo "   ✅ Build found!"
fi

echo ""
echo "2️⃣  Setting up test environment..."
create_test_model
create_test_client

echo ""
echo "3️⃣  Starting Triton Server..."
echo "   Server will start in background. Logs in triton_server.log"

# Kill any existing server
pkill -f tritonserver 2>/dev/null || true
sleep 2

# Start server with Apple Silicon optimizations
./build/tritonserver \
    --model-repository=$(pwd)/models \
    --backend-config=python,shm-region-prefix-name=apple_triton \
    --backend-config=coreml,enable_ane=true \
    --backend-config=metal_mps,enable_gpu=true \
    --log-verbose=1 \
    > triton_server.log 2>&1 &

SERVER_PID=$!
echo "   Server PID: $SERVER_PID"

# Wait for server to start
echo "   Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
        echo "   ✅ Server is ready!"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Check if server started successfully
if ! curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
    echo "❌ Server failed to start. Check triton_server.log for details."
    tail -20 triton_server.log
    exit 1
fi

echo ""
echo "4️⃣  Testing server..."
python3 test_client.py

echo ""
echo "5️⃣  Server Info:"
echo "   HTTP endpoint: http://localhost:8000"
echo "   gRPC endpoint: localhost:8001" 
echo "   Metrics endpoint: http://localhost:8002/metrics"
echo "   Model repository: $(pwd)/models"
echo "   Server PID: $SERVER_PID"
echo "   Logs: triton_server.log"

echo ""
echo "📚 Next Steps:"
echo "   - Add your models to the models/ directory"
echo "   - Check APPLE_SILICON_USAGE_GUIDE.md for detailed instructions"
echo "   - Run 'kill $SERVER_PID' to stop the server"
echo "   - View logs with 'tail -f triton_server.log'"

echo ""
echo "🎉 Apple Silicon Triton Server is running!"
echo ""
echo "Try these commands:"
echo "  curl http://localhost:8000/v2/models/simple_add"
echo "  curl http://localhost:8002/metrics | grep apple_silicon"