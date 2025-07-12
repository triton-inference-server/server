#!/bin/bash
# Comprehensive test script for PyTorch backend on macOS

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVER_BUILD_DIR="/Volumes/Untitled/coder/server/build"
MODEL_REPO="${SCRIPT_DIR}/test/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== PyTorch Backend Test for macOS ===${NC}"

# Step 1: Build the backend
echo -e "\n${GREEN}Step 1: Building PyTorch backend...${NC}"
"${SCRIPT_DIR}/build_macos.sh"

# Step 2: Copy backend to server build directory
echo -e "\n${GREEN}Step 2: Installing backend to server build directory...${NC}"
BACKEND_INSTALL_DIR="${SERVER_BUILD_DIR}/backends/pytorch"
mkdir -p "${BACKEND_INSTALL_DIR}"

# Copy the built backend library
if [ -f "${SCRIPT_DIR}/install/backends/pytorch/libtriton-pytorch-backend.dylib" ]; then
    cp "${SCRIPT_DIR}/install/backends/pytorch/"*.dylib "${BACKEND_INSTALL_DIR}/"
    echo -e "${GREEN}Backend library copied successfully${NC}"
else
    echo -e "${RED}Error: Backend library not found!${NC}"
    exit 1
fi

# Step 3: Check if test model exists
echo -e "\n${GREEN}Step 3: Checking test model...${NC}"
if [ ! -f "${MODEL_REPO}/pytorch_simple/1/model.pt" ]; then
    echo -e "${YELLOW}Test model not found. Creating...${NC}"
    cd "${SCRIPT_DIR}/test"
    if command -v python3 &> /dev/null && python3 -c "import torch" &> /dev/null; then
        python3 create_test_model.py
    else
        echo -e "${RED}Error: Python3 with PyTorch is required to create test model${NC}"
        echo -e "${YELLOW}Install with: pip3 install torch${NC}"
        exit 1
    fi
fi

# Step 4: Start Triton server
echo -e "\n${GREEN}Step 4: Starting Triton server...${NC}"
SERVER_LOG="/tmp/triton_pytorch_test.log"
"${SERVER_BUILD_DIR}/bin/tritonserver" \
    --model-repository="${MODEL_REPO}" \
    --backend-directory="${SERVER_BUILD_DIR}/backends" \
    --log-verbose=1 \
    > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Wait for server to start
echo -n "Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
        echo -e " ${GREEN}Ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Check if server started successfully
if ! curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
    echo -e " ${RED}Failed!${NC}"
    echo -e "${RED}Server failed to start. Last 50 lines of log:${NC}"
    tail -50 "${SERVER_LOG}"
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
fi

# Step 5: Run tests
echo -e "\n${GREEN}Step 5: Running tests...${NC}"

# Check if tritonclient is installed
if ! python3 -c "import tritonclient.http" &> /dev/null; then
    echo -e "${YELLOW}tritonclient not installed. Installing...${NC}"
    pip3 install tritonclient[all]
fi

# Run the test client
cd "${SCRIPT_DIR}/test"
python3 test_client.py --verbose

TEST_RESULT=$?

# Step 6: Cleanup
echo -e "\n${GREEN}Step 6: Cleaning up...${NC}"
kill ${SERVER_PID} 2>/dev/null || true
wait ${SERVER_PID} 2>/dev/null || true

# Show server log if test failed
if [ ${TEST_RESULT} -ne 0 ]; then
    echo -e "\n${RED}Test failed! Server log:${NC}"
    tail -100 "${SERVER_LOG}"
else
    echo -e "\n${GREEN}All tests passed!${NC}"
fi

# Cleanup log file
rm -f "${SERVER_LOG}"

exit ${TEST_RESULT}