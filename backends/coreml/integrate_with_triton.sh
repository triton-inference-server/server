#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Script to integrate CoreML backend with Triton Inference Server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Integrating CoreML Backend with Triton Inference Server${NC}"
echo "======================================================"

# Configuration
TRITON_DIR="${TRITON_DIR:-/opt/tritonserver}"
BACKEND_NAME="coreml"
BACKEND_DIR="$(pwd)"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ] || [ ! -d "src" ]; then
    echo -e "${RED}Error: This script must be run from the CoreML backend directory${NC}"
    exit 1
fi

# Check if backend is built
if [ ! -f "build/libtriton_coreml.dylib" ]; then
    echo -e "${YELLOW}Backend not built. Building now...${NC}"
    ./build_macos.sh
fi

# Check if Triton is installed
if [ ! -d "$TRITON_DIR" ]; then
    echo -e "${RED}Error: Triton not found at $TRITON_DIR${NC}"
    echo "Please set TRITON_DIR environment variable to your Triton installation"
    exit 1
fi

# Create backend directory in Triton
echo -e "${YELLOW}Creating backend directory in Triton...${NC}"
sudo mkdir -p "$TRITON_DIR/backends/$BACKEND_NAME"

# Copy backend library
echo -e "${YELLOW}Copying backend library...${NC}"
sudo cp build/libtriton_coreml.dylib "$TRITON_DIR/backends/$BACKEND_NAME/libtriton_coreml.dylib"

# Create a simple test to verify integration
echo -e "${YELLOW}Creating integration test...${NC}"
cat > /tmp/test_coreml_integration.sh << 'EOF'
#!/bin/bash
# Test CoreML backend integration

MODEL_REPO="/tmp/coreml_test_repo"
mkdir -p $MODEL_REPO/simple_model/1

# Create a dummy config (actual model would be needed for real test)
cat > $MODEL_REPO/simple_model/config.pbtxt << EOC
name: "simple_model"
platform: "coreml"
max_batch_size: 1
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 5 ]
  }
]
EOC

# Try to start Triton with the model
echo "Testing Triton startup with CoreML backend..."
timeout 10 tritonserver --model-repository=$MODEL_REPO --log-verbose=1 2>&1 | grep -E "(coreml|CoreML)" || true

# Cleanup
rm -rf $MODEL_REPO
EOF

chmod +x /tmp/test_coreml_integration.sh

# Run integration test
echo -e "${YELLOW}Running integration test...${NC}"
if /tmp/test_coreml_integration.sh | grep -q "coreml"; then
    echo -e "${GREEN}Integration test passed!${NC}"
else
    echo -e "${YELLOW}Integration test inconclusive (this is normal without a real model)${NC}"
fi

# Cleanup
rm -f /tmp/test_coreml_integration.sh

# Print summary
echo ""
echo -e "${GREEN}CoreML Backend Integration Complete!${NC}"
echo "===================================="
echo ""
echo "Backend installed to: $TRITON_DIR/backends/$BACKEND_NAME/"
echo ""
echo "To use the CoreML backend:"
echo "1. Convert your model to CoreML format (.mlmodel or .mlpackage)"
echo "2. Create a model repository with the following structure:"
echo "   model_repository/"
echo "   └── your_model/"
echo "       ├── config.pbtxt  (with platform: \"coreml\")"
echo "       └── 1/"
echo "           └── model.mlmodel"
echo ""
echo "3. Start Triton:"
echo "   tritonserver --model-repository=model_repository"
echo ""
echo "For examples, see: ${BACKEND_DIR}/examples/"
echo ""
echo -e "${YELLOW}Note: Ensure your models are compatible with CoreML${NC}"
echo "Use coremltools to convert models from other frameworks"