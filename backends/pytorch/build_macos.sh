#!/bin/bash
# Build PyTorch backend for macOS

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building PyTorch backend for macOS${NC}"

# Check if LibTorch is available
if [ ! -d "${SCRIPT_DIR}/libtorch" ]; then
    echo -e "${YELLOW}LibTorch not found. Running download script...${NC}"
    "${SCRIPT_DIR}/download_libtorch_macos.sh"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRITON_ENABLE_GPU=OFF \
    -DTORCH_PATH="${SCRIPT_DIR}/libtorch" \
    -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}/install"

# Build
echo -e "${GREEN}Building...${NC}"
make -j$(sysctl -n hw.ncpu)

# Install
echo -e "${GREEN}Installing...${NC}"
make install

echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}Backend installed to: ${SCRIPT_DIR}/install/backends/pytorch${NC}"

# Create test model if Python and PyTorch are available
if command -v python3 &> /dev/null; then
    if python3 -c "import torch" &> /dev/null; then
        echo -e "${GREEN}Creating test model...${NC}"
        cd "${SCRIPT_DIR}/test"
        python3 create_test_model.py
    else
        echo -e "${YELLOW}PyTorch Python package not found. Skipping test model creation.${NC}"
        echo -e "${YELLOW}Install with: pip3 install torch${NC}"
    fi
else
    echo -e "${YELLOW}Python3 not found. Skipping test model creation.${NC}"
fi

echo -e "${GREEN}Setup complete!${NC}"