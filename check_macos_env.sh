#!/bin/bash
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Environment checker for Triton macOS build

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "Triton Inference Server - macOS Environment Check"
echo "================================================="
echo ""

# System Information
echo -e "${BLUE}System Information:${NC}"
echo "  OS: $(sw_vers -productName) $(sw_vers -productVersion)"
echo "  Architecture: $(uname -m)"
echo "  Cores: $(sysctl -n hw.ncpu)"
echo "  Memory: $(echo "scale=2; $(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc) GB"
echo ""

# Check Xcode
echo -e "${BLUE}Development Tools:${NC}"
echo -n "  Xcode Command Line Tools: "
if xcode-select -p &> /dev/null; then
    echo -e "${GREEN}Installed${NC} ($(xcode-select -p))"
else
    echo -e "${RED}Not installed${NC} - Run: xcode-select --install"
fi

# Check compilers
echo -n "  Clang: "
if command -v clang &> /dev/null; then
    VERSION=$(clang --version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    echo -e "${GREEN}Found${NC} (version $VERSION)"
else
    echo -e "${RED}Not found${NC}"
fi

echo -n "  Clang++: "
if command -v clang++ &> /dev/null; then
    echo -e "${GREEN}Found${NC}"
else
    echo -e "${RED}Not found${NC}"
fi

# Check Homebrew
echo -n "  Homebrew: "
if command -v brew &> /dev/null; then
    VERSION=$(brew --version | head -n1 | cut -d' ' -f2)
    echo -e "${GREEN}Found${NC} (version $VERSION)"
    HOMEBREW_PREFIX=$(brew --prefix)
else
    echo -e "${RED}Not found${NC} - Install from https://brew.sh"
    HOMEBREW_PREFIX=""
fi

# Check CMake
echo -n "  CMake: "
if command -v cmake &> /dev/null; then
    VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    MAJOR=$(echo "$VERSION" | cut -d. -f1)
    MINOR=$(echo "$VERSION" | cut -d. -f2)
    if [[ $MAJOR -gt 3 ]] || ([[ $MAJOR -eq 3 ]] && [[ $MINOR -ge 18 ]]); then
        echo -e "${GREEN}Found${NC} (version $VERSION)"
    else
        echo -e "${YELLOW}Found but outdated${NC} (version $VERSION, need 3.18+)"
    fi
else
    echo -e "${YELLOW}Not found${NC} - Will be installed by build script"
fi

# Check Python
echo -n "  Python3: "
if command -v python3 &> /dev/null; then
    VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}Found${NC} (version $VERSION)"
else
    echo -e "${YELLOW}Not found${NC} - Optional, needed for Python backends"
fi

echo ""

# Check key dependencies
echo -e "${BLUE}Key Dependencies:${NC}"
if [[ -n "$HOMEBREW_PREFIX" ]]; then
    DEPS=(protobuf grpc libevent rapidjson boost re2 openssl libarchive)
    for dep in "${DEPS[@]}"; do
        echo -n "  $dep: "
        if brew list "$dep" &> /dev/null 2>&1; then
            echo -e "${GREEN}Installed${NC}"
        else
            echo -e "${YELLOW}Not installed${NC} - Will be installed by build script"
        fi
    done
else
    echo "  ${YELLOW}Cannot check - Homebrew not installed${NC}"
fi

echo ""

# Check environment
echo -e "${BLUE}Environment:${NC}"
echo "  PATH includes Homebrew: "
if [[ -n "$HOMEBREW_PREFIX" ]] && echo "$PATH" | grep -q "$HOMEBREW_PREFIX/bin"; then
    echo -e "    ${GREEN}Yes${NC}"
else
    echo -e "    ${YELLOW}No${NC} - You may need to add Homebrew to PATH"
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo "    Run: echo 'eval \"\$(/opt/homebrew/bin/brew shellenv)\"' >> ~/.zprofile"
    fi
fi

# Check for incompatible software
echo ""
echo -e "${BLUE}Checking for conflicts:${NC}"
echo -n "  MacPorts: "
if command -v port &> /dev/null; then
    echo -e "${YELLOW}Found${NC} - May conflict with Homebrew"
else
    echo -e "${GREEN}Not found${NC} (good)"
fi

# Summary
echo ""
echo -e "${BLUE}Summary:${NC}"

READY=true

if ! xcode-select -p &> /dev/null; then
    echo -e "  ${RED}✗${NC} Xcode Command Line Tools required"
    READY=false
fi

if ! command -v brew &> /dev/null; then
    echo -e "  ${RED}✗${NC} Homebrew required"
    READY=false
fi

if [[ "$READY" == "true" ]]; then
    echo -e "  ${GREEN}✓${NC} System is ready for Triton build"
    echo ""
    echo "You can now run: ./build_macos.sh"
else
    echo -e "  ${RED}✗${NC} Please install missing requirements first"
fi

echo ""