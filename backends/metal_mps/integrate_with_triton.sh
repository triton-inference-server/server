#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Script to integrate MPS backend with Triton server

set -e

echo "Integrating MPS backend with Triton..."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if build exists
if [ ! -d "${SCRIPT_DIR}/build/install" ]; then
    echo "Error: Backend not built. Run ./build_macos.sh first"
    exit 1
fi

# Find Triton backends directory
TRITON_BACKENDS_DIR="${TRITON_BACKENDS_DIR:-/opt/tritonserver/backends}"

if [ ! -d "${TRITON_BACKENDS_DIR}" ]; then
    echo "Warning: Triton backends directory not found at ${TRITON_BACKENDS_DIR}"
    echo "Please set TRITON_BACKENDS_DIR environment variable"
    echo ""
    echo "Example locations:"
    echo "  /opt/tritonserver/backends"
    echo "  /usr/local/tritonserver/backends"
    echo "  ~/tritonserver/backends"
    exit 1
fi

# Copy backend
echo "Copying MPS backend to ${TRITON_BACKENDS_DIR}..."
sudo cp -r "${SCRIPT_DIR}/build/install/backends/metal_mps" "${TRITON_BACKENDS_DIR}/"

# Verify installation
if [ -f "${TRITON_BACKENDS_DIR}/metal_mps/libtriton_metal_mps.so" ] || [ -f "${TRITON_BACKENDS_DIR}/metal_mps/libtriton_metal_mps.dylib" ]; then
    echo "✓ MPS backend successfully installed!"
    echo ""
    echo "To test the backend:"
    echo "1. Create test models:"
    echo "   cd ${SCRIPT_DIR}/test"
    echo "   python3 create_test_model.py"
    echo ""
    echo "2. Copy models to Triton model repository:"
    echo "   cp -r ${SCRIPT_DIR}/test/models/* /path/to/model_repository/"
    echo ""
    echo "3. Start Triton server:"
    echo "   tritonserver --model-repository=/path/to/model_repository"
    echo ""
    echo "4. Run test client:"
    echo "   python3 ${SCRIPT_DIR}/test/test_client.py"
else
    echo "Error: Backend installation failed"
    exit 1
fi