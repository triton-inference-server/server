#!/bin/bash
# Build script for PyTorch backend with MPS support on macOS

set -e

echo "=========================================="
echo "Building PyTorch Backend with MPS Support"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script is for macOS only${NC}"
    exit 1
fi

# Set build directory
BUILD_DIR="${BUILD_DIR:-build_mps}"
INSTALL_DIR="${INSTALL_DIR:-/opt/tritonserver}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --torch-path)
            TORCH_PATH="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--torch-path PATH] [--build-dir DIR] [--install-dir DIR] [--clean]"
            exit 1
            ;;
    esac
done

# Clean build directory if requested
if [ "$CLEAN_BUILD" = "1" ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake
echo -e "${GREEN}Configuring CMake...${NC}"
cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
    -DTRITON_ENABLE_GPU=OFF
    -DTRITON_ENABLE_MPS=ON
    -DTRITON_ENABLE_STATS=ON
)

if [ -n "$TORCH_PATH" ]; then
    cmake_args+=(-DTORCH_PATH="$TORCH_PATH")
fi

echo "CMake arguments: ${cmake_args[@]}"
cmake "${cmake_args[@]}" ..

# Build
echo -e "${GREEN}Building PyTorch backend...${NC}"
make -j$(sysctl -n hw.ncpu)

# Check if MPS was detected
echo -e "${GREEN}Checking MPS support in build...${NC}"
if grep -q "MPS.*support enabled" CMakeCache.txt; then
    echo -e "${GREEN}✓ MPS support is enabled${NC}"
else
    echo -e "${YELLOW}⚠ MPS support might not be enabled${NC}"
fi

# Install
echo -e "${GREEN}Installing PyTorch backend...${NC}"
make install

# Create test script
echo -e "${GREEN}Creating test script...${NC}"
cat > test_mps_installation.py << 'EOF'
#!/usr/bin/env python3
import torch
import sys

print("PyTorch MPS Installation Test")
print("=" * 40)
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    # Test MPS device
    device = torch.device("mps")
    x = torch.randn(10, 10).to(device)
    y = torch.randn(10, 10).to(device)
    z = torch.matmul(x, y)
    print(f"MPS computation test: {'PASSED' if z.shape == (10, 10) else 'FAILED'}")
else:
    print("MPS is not available on this system")
    sys.exit(1)
EOF

chmod +x test_mps_installation.py

# Run test
echo -e "${GREEN}Testing MPS installation...${NC}"
python3 test_mps_installation.py

echo -e "${GREEN}=========================================="
echo -e "Build completed successfully!"
echo -e "Installation directory: $INSTALL_DIR"
echo -e "==========================================${NC}"

# Print next steps
echo ""
echo "Next steps:"
echo "1. Ensure LibTorch libraries are in your library path"
echo "2. Copy your model to the model repository"
echo "3. Configure your model with KIND_GPU for MPS support"
echo "4. Start Triton server with the PyTorch backend"