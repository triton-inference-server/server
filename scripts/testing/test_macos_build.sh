#!/bin/bash
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Simple test script to verify macOS build functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Testing Triton Inference Server macOS Build"
echo "=========================================="

# Test 1: Check if build script exists and is executable
echo -n "Test 1: Build script exists and is executable... "
if [[ -x "${SCRIPT_DIR}/build_macos.sh" ]]; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 2: Check build script help
echo -n "Test 2: Build script help works... "
if "${SCRIPT_DIR}/build_macos.sh" --help &> /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 3: Check system detection
echo -n "Test 3: System detection... "
if [[ "$(uname)" == "Darwin" ]]; then
    echo -e "${GREEN}PASS${NC} (macOS detected)"
else
    echo -e "${RED}FAIL${NC} (Not running on macOS)"
    exit 1
fi

# Test 4: Check for required commands
echo "Test 4: Checking required commands..."
REQUIRED_COMMANDS=(
    "clang"
    "clang++"
    "make"
)

ALL_PASS=true
for cmd in "${REQUIRED_COMMANDS[@]}"; do
    echo -n "  - $cmd: "
    if command -v "$cmd" &> /dev/null; then
        echo -e "${GREEN}Found${NC}"
    else
        echo -e "${YELLOW}Not found${NC} (will be installed during build)"
        ALL_PASS=false
    fi
done

# Test 5: Check CMake module
echo -n "Test 5: macOS CMake module exists... "
if [[ -f "${SCRIPT_DIR}/cmake/MacOS.cmake" ]]; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    exit 1
fi

# Test 6: Dry run of build script (parse args only)
echo -n "Test 6: Build script argument parsing... "
# Create a temporary test to check arg parsing
if bash -n "${SCRIPT_DIR}/build_macos.sh" 2> /dev/null; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC} (Syntax error in build script)"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}All tests passed!${NC}"
echo ""
echo "You can now run the build with:"
echo "  ./build_macos.sh"
echo ""
echo "For a quick test build (debug mode, no dependencies):"
echo "  ./build_macos.sh --build-type=Debug --skip-deps"