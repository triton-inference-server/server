#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Test script for CoreML backend on macOS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Testing CoreML Backend for Triton Inference Server${NC}"
echo "=================================================="

# Configuration
TRITON_DIR="${TRITON_DIR:-/opt/tritonserver}"
MODEL_REPO="test/models"
BACKEND_DIR="$(pwd)/build/libtriton_coreml.dylib"

# Check if backend is built
if [ ! -f "$BACKEND_DIR" ]; then
    echo -e "${RED}Error: Backend not built. Run ./build_macos.sh first${NC}"
    exit 1
fi

# Check if Triton is installed
if [ ! -d "$TRITON_DIR" ]; then
    echo -e "${YELLOW}Warning: Triton not found at $TRITON_DIR${NC}"
    echo "Set TRITON_DIR environment variable if Triton is installed elsewhere"
    echo ""
fi

# Create test models if they don't exist
if [ ! -d "$MODEL_REPO/coreml_simple/1/model.mlmodel" ]; then
    echo -e "${YELLOW}Creating test models...${NC}"
    cd test
    python3 create_test_model.py
    cd ..
fi

# Copy backend to Triton backends directory (if Triton is available)
if [ -d "$TRITON_DIR/backends" ]; then
    echo -e "${YELLOW}Installing backend to Triton...${NC}"
    sudo mkdir -p "$TRITON_DIR/backends/coreml"
    sudo cp "$BACKEND_DIR" "$TRITON_DIR/backends/coreml/"
    echo -e "${GREEN}Backend installed successfully${NC}"
fi

# Start Triton server with test model repository
if command -v tritonserver &> /dev/null; then
    echo -e "${YELLOW}Starting Triton Inference Server...${NC}"
    
    # Kill any existing Triton process
    pkill -f tritonserver || true
    sleep 2
    
    # Start Triton in background
    tritonserver --model-repository="$MODEL_REPO" \
                 --backend-directory="$(dirname $BACKEND_DIR)" \
                 --log-verbose=1 &
    
    TRITON_PID=$!
    echo "Triton PID: $TRITON_PID"
    
    # Wait for server to be ready
    echo -e "${YELLOW}Waiting for Triton to be ready...${NC}"
    sleep 5
    
    # Check if server is ready
    max_attempts=30
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/v2/health/ready | grep -q "true"; then
            echo -e "${GREEN}Triton is ready!${NC}"
            break
        fi
        echo "Waiting for server... (attempt $((attempt+1))/$max_attempts)"
        sleep 2
        attempt=$((attempt+1))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        echo -e "${RED}Error: Triton failed to start${NC}"
        kill $TRITON_PID 2>/dev/null || true
        exit 1
    fi
    
    # Run tests
    echo -e "${YELLOW}Running inference tests...${NC}"
    cd test
    
    # Basic test
    echo -e "${YELLOW}Running basic inference test...${NC}"
    python3 test_client.py --verbose
    
    # Performance test
    echo -e "${YELLOW}Running performance test...${NC}"
    python3 test_client.py --performance --requests 50
    
    cd ..
    
    # Cleanup
    echo -e "${YELLOW}Stopping Triton...${NC}"
    kill $TRITON_PID 2>/dev/null || true
    wait $TRITON_PID 2>/dev/null || true
    
    echo -e "${GREEN}All tests completed successfully!${NC}"
else
    echo -e "${YELLOW}Warning: tritonserver not found in PATH${NC}"
    echo "Tests requiring a running server were skipped"
    echo ""
    echo "To run full tests:"
    echo "1. Install Triton Inference Server"
    echo "2. Add tritonserver to your PATH"
    echo "3. Run this script again"
fi

echo ""
echo -e "${GREEN}CoreML backend test completed${NC}"