#!/bin/bash
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

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building CoreML Backend for Triton Inference Server${NC}"
echo "======================================================"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script can only run on macOS${NC}"
    exit 1
fi

# Default values
BUILD_TYPE="Release"
INSTALL_PREFIX="/opt/tritonserver"
ENABLE_GPU="ON"
ENABLE_STATS="ON"
BUILD_DIR="build"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --no-gpu)
            ENABLE_GPU="OFF"
            shift
            ;;
        --no-stats)
            ENABLE_STATS="OFF"
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --build-type <Debug|Release>    Build type (default: Release)"
            echo "  --install-prefix <path>         Install prefix (default: /opt/tritonserver)"
            echo "  --no-gpu                        Disable GPU support"
            echo "  --no-stats                      Disable statistics collection"
            echo "  --build-dir <path>              Build directory (default: build)"
            echo "  --help                          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Build type: ${BUILD_TYPE}"
echo "  Install prefix: ${INSTALL_PREFIX}"
echo "  GPU support: ${ENABLE_GPU}"
echo "  Statistics: ${ENABLE_STATS}"
echo "  Build directory: ${BUILD_DIR}"
echo ""

# Check for required tools
echo -e "${YELLOW}Checking requirements...${NC}"

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake is not installed${NC}"
    echo "Install with: brew install cmake"
    exit 1
fi

if ! command -v ninja &> /dev/null; then
    echo -e "${YELLOW}Warning: ninja is not installed. Using make instead.${NC}"
    echo "Install ninja for faster builds: brew install ninja"
    CMAKE_GENERATOR="Unix Makefiles"
else
    CMAKE_GENERATOR="Ninja"
fi

# Check Xcode command line tools
if ! xcode-select -p &> /dev/null; then
    echo -e "${RED}Error: Xcode command line tools are not installed${NC}"
    echo "Install with: xcode-select --install"
    exit 1
fi

echo -e "${GREEN}All requirements satisfied${NC}"
echo ""

# Create build directory
echo -e "${YELLOW}Creating build directory...${NC}"
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Configure
echo -e "${YELLOW}Configuring CoreML backend...${NC}"
cmake .. \
    -G "${CMAKE_GENERATOR}" \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DTRITON_ENABLE_GPU=${ENABLE_GPU} \
    -DTRITON_ENABLE_STATS=${ENABLE_STATS} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
echo -e "${YELLOW}Building CoreML backend...${NC}"
if [[ "${CMAKE_GENERATOR}" == "Ninja" ]]; then
    ninja -v
else
    make -j$(sysctl -n hw.ncpu) VERBOSE=1
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "To install the backend, run:"
echo "  cd ${BUILD_DIR} && sudo ${CMAKE_GENERATOR,,} install"
echo ""
echo "The backend library will be installed to:"
echo "  ${INSTALL_PREFIX}/backends/coreml/libtriton_coreml.dylib"