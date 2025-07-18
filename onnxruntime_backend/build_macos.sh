#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Build script for ONNX Runtime backend on macOS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local/tritonserver}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.16.3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-type=*)
            BUILD_TYPE="${1#*=}"
            ;;
        --install-prefix=*)
            INSTALL_PREFIX="${1#*=}"
            ;;
        --onnxruntime-version=*)
            ONNXRUNTIME_VERSION="${1#*=}"
            ;;
        --clean)
            rm -rf "${BUILD_DIR}"
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --build-type=<Debug|Release>     Build type (default: Release)"
            echo "  --install-prefix=<path>          Installation prefix (default: /usr/local/tritonserver)"
            echo "  --onnxruntime-version=<version>  ONNX Runtime version (default: 1.16.3)"
            echo "  --clean                          Clean build directory"
            echo "  --help                           Show this help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

print_info "Building ONNX Runtime backend for macOS"
print_info "Build type: ${BUILD_TYPE}"
print_info "Install prefix: ${INSTALL_PREFIX}"
print_info "ONNX Runtime version: ${ONNXRUNTIME_VERSION}"

# Check for required tools
check_tool() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is required but not installed"
        return 1
    fi
    return 0
}

print_info "Checking required tools..."
MISSING_TOOLS=0
check_tool cmake || MISSING_TOOLS=1
check_tool git || MISSING_TOOLS=1
check_tool make || MISSING_TOOLS=1

if [ $MISSING_TOOLS -eq 1 ]; then
    print_error "Please install missing tools and try again"
    exit 1
fi

# Check for ONNX Runtime
print_info "Checking for ONNX Runtime..."
ONNXRUNTIME_FOUND=0
ONNXRUNTIME_INCLUDE=""
ONNXRUNTIME_LIB=""

# Check Homebrew locations
if [ -f "/opt/homebrew/include/onnxruntime/onnxruntime_cxx_api.h" ]; then
    ONNXRUNTIME_INCLUDE="/opt/homebrew/include/onnxruntime"
    ONNXRUNTIME_LIB="/opt/homebrew/lib"
    ONNXRUNTIME_FOUND=1
    print_success "Found ONNX Runtime in /opt/homebrew (Apple Silicon)"
elif [ -f "/usr/local/include/onnxruntime/onnxruntime_cxx_api.h" ]; then
    ONNXRUNTIME_INCLUDE="/usr/local/include/onnxruntime"
    ONNXRUNTIME_LIB="/usr/local/lib"
    ONNXRUNTIME_FOUND=1
    print_success "Found ONNX Runtime in /usr/local (Intel)"
fi

if [ $ONNXRUNTIME_FOUND -eq 0 ]; then
    print_error "ONNX Runtime not found!"
    print_info "Please install ONNX Runtime using:"
    print_info "  brew install onnxruntime"
    print_info ""
    print_info "Or download and install manually from:"
    print_info "  https://github.com/microsoft/onnxruntime/releases"
    exit 1
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
print_info "Configuring with CMake..."
cmake_args=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
    -DTRITON_BUILD_ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION}"
    -DTRITON_ONNXRUNTIME_INCLUDE_PATHS="${ONNXRUNTIME_INCLUDE}"
    -DTRITON_ONNXRUNTIME_LIB_PATHS="${ONNXRUNTIME_LIB}"
    -DTRITON_ENABLE_GPU=OFF
    -DTRITON_ENABLE_ONNXRUNTIME_COREML=ON
    -DTRITON_ENABLE_STATS=ON
)

# Use the macOS-specific CMakeLists.txt
cp "${SCRIPT_DIR}/CMakeLists.txt.macos" "${SCRIPT_DIR}/CMakeLists.txt.backup" 2>/dev/null || true
cp "${SCRIPT_DIR}/CMakeLists.txt" "${SCRIPT_DIR}/CMakeLists.txt.original" 2>/dev/null || true
cp "${SCRIPT_DIR}/CMakeLists.txt.macos" "${SCRIPT_DIR}/CMakeLists.txt"

cmake "${cmake_args[@]}" ..

# Build
print_info "Building..."
cmake --build . --config "${BUILD_TYPE}" -j$(sysctl -n hw.ncpu)

# Install
print_info "Installing to ${INSTALL_PREFIX}..."
cmake --install . --config "${BUILD_TYPE}"

# Restore original CMakeLists.txt
cp "${SCRIPT_DIR}/CMakeLists.txt.original" "${SCRIPT_DIR}/CMakeLists.txt" 2>/dev/null || true

print_success "ONNX Runtime backend built successfully!"
print_info "Backend installed to: ${INSTALL_PREFIX}/backends/onnxruntime"

# Create a simple test script
cat > "${INSTALL_PREFIX}/backends/onnxruntime/test_backend.sh" << 'EOF'
#!/bin/bash
# Test if the ONNX Runtime backend can be loaded

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_LIB="${BACKEND_DIR}/libtriton_onnxruntime.dylib"

if [ ! -f "${BACKEND_LIB}" ]; then
    echo "ERROR: Backend library not found: ${BACKEND_LIB}"
    exit 1
fi

# Check if the library can be loaded
if otool -L "${BACKEND_LIB}" &> /dev/null; then
    echo "SUCCESS: Backend library can be loaded"
    echo "Dependencies:"
    otool -L "${BACKEND_LIB}" | grep -E "(onnxruntime|triton)" | sed 's/^/  /'
else
    echo "ERROR: Failed to load backend library"
    exit 1
fi

# Check for required symbols
echo ""
echo "Checking for required backend symbols..."
if nm -g "${BACKEND_LIB}" | grep -q "TRITONBACKEND_Initialize"; then
    echo "SUCCESS: Found TRITONBACKEND_Initialize"
else
    echo "ERROR: Missing TRITONBACKEND_Initialize"
    exit 1
fi

if nm -g "${BACKEND_LIB}" | grep -q "TRITONBACKEND_ModelInitialize"; then
    echo "SUCCESS: Found TRITONBACKEND_ModelInitialize"
else
    echo "ERROR: Missing TRITONBACKEND_ModelInitialize"
    exit 1
fi

echo ""
echo "Backend validation passed!"
EOF

chmod +x "${INSTALL_PREFIX}/backends/onnxruntime/test_backend.sh"

print_info "Run ${INSTALL_PREFIX}/backends/onnxruntime/test_backend.sh to test the backend"