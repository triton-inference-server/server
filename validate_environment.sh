#!/bin/bash
# Validate Apple Silicon development environment

echo "======================================"
echo "Apple Silicon Environment Validation"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ISSUES=0

# Check OS
echo -n "Checking operating system... "
if [[ "$(uname)" == "Darwin" ]]; then
    echo -e "${GREEN}✓ macOS${NC}"
else
    echo -e "${RED}✗ Not macOS${NC}"
    ((ISSUES++))
fi

# Check architecture
echo -n "Checking CPU architecture... "
if [[ "$(uname -m)" == "arm64" ]]; then
    echo -e "${GREEN}✓ Apple Silicon (arm64)${NC}"
    
    # Get chip info
    CHIP=$(sysctl -n machdep.cpu.brand_string)
    echo "  Chip: $CHIP"
else
    echo -e "${YELLOW}⚠ Not Apple Silicon ($(uname -m))${NC}"
    echo "  Note: Some features may not be available"
fi

# Check Xcode Command Line Tools
echo -n "Checking Xcode Command Line Tools... "
if xcode-select -p &> /dev/null; then
    echo -e "${GREEN}✓ Installed${NC}"
    echo "  Path: $(xcode-select -p)"
else
    echo -e "${RED}✗ Not installed${NC}"
    echo "  Run: xcode-select --install"
    ((ISSUES++))
fi

# Check compiler
echo -n "Checking C++ compiler... "
if command -v clang++ &> /dev/null; then
    echo -e "${GREEN}✓ Found${NC}"
    echo "  Version: $(clang++ --version | head -n1)"
else
    echo -e "${RED}✗ Not found${NC}"
    ((ISSUES++))
fi

# Check CMake
echo -n "Checking CMake... "
if command -v cmake &> /dev/null; then
    echo -e "${GREEN}✓ Found${NC}"
    echo "  Version: $(cmake --version | head -n1)"
else
    echo -e "${YELLOW}⚠ Not found${NC}"
    echo "  Install with: brew install cmake"
fi

# Check Python (for visualization)
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓ Found${NC}"
    echo "  Version: $(python3 --version)"
else
    echo -e "${YELLOW}⚠ Not found${NC}"
    echo "  Required for benchmark visualization"
fi

# Check frameworks
echo ""
echo "Checking required frameworks:"

check_framework() {
    local framework=$1
    local path="/System/Library/Frameworks/${framework}.framework"
    echo -n "  $framework... "
    
    if [ -d "$path" ]; then
        echo -e "${GREEN}✓ Found${NC}"
    else
        echo -e "${RED}✗ Not found${NC}"
        ((ISSUES++))
    fi
}

check_framework "Accelerate"
check_framework "Metal"
check_framework "MetalPerformanceShaders"
check_framework "CoreML"

# Check for Triton headers
echo ""
echo -n "Checking Triton headers... "
if [ -f "include/triton/core/tritonserver.h" ] || [ -f "../core/include/triton/core/tritonserver.h" ]; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${YELLOW}⚠ Not found${NC}"
    echo "  Make sure you're in the Triton server directory"
fi

# Check our implementation files
echo ""
echo "Checking Apple Silicon implementation files:"

check_file() {
    local file=$1
    echo -n "  $(basename $file)... "
    
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ Found${NC}"
    else
        echo -e "${RED}✗ Not found${NC}"
        ((ISSUES++))
    fi
}

check_file "src/apple/amx_provider.h"
check_file "src/apple/amx_provider.cc"
check_file "src/metal/metal_device.h"
check_file "src/apple/winograd_conv3x3.h"
check_file "src/apple/profile_guided_optimizer.h"

# Summary
echo ""
echo "======================================"
if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✓ Environment is ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run the test suite: ./test_apple_silicon_optimizations.sh"
    echo "2. Build the demo: make -f Makefile.demo"
    echo "3. Run benchmarks: ./build/src/benchmarks/apple_silicon_benchmarks"
else
    echo -e "${RED}✗ Found $ISSUES issues${NC}"
    echo ""
    echo "Please fix the issues above before proceeding."
fi
echo "======================================"