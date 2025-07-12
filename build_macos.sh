#!/bin/bash
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#########################################
# Triton Inference Server Build Script for macOS
#########################################
#
# This script automates the build process for Triton Inference Server on macOS.
# It handles dependency checking, installation, and compilation with proper
# configuration for both Intel and Apple Silicon Macs.
#
# Usage:
#   ./build_macos.sh [options]
#
# Options:
#   --build-type=<Debug|Release>    Build type (default: Release)
#   --install-prefix=<path>         Installation directory (default: /usr/local)
#   --clean                         Clean build (removes build directory)
#   --verbose                       Enable verbose output
#   --help                          Show this help message
#   --skip-deps                     Skip dependency installation
#   --parallel=<n>                  Number of parallel build jobs (default: auto)
#   --enable-http                   Enable HTTP endpoint (default: ON)
#   --enable-grpc                   Enable gRPC endpoint (default: ON)
#   --enable-metrics                Enable metrics (default: ON)
#   --enable-logging                Enable logging (default: ON)
#   --enable-stats                  Enable statistics (default: ON)
#   --enable-tracing                Enable tracing (default: OFF)
#   --enable-ensemble               Enable ensemble support (default: OFF)
#   --enable-s3                     Enable S3 support (default: OFF)
#   --enable-gcs                    Enable GCS support (default: OFF)
#   --enable-azure                  Enable Azure Storage support (default: OFF)
#   --run-tests                     Run tests after build
#   --ccache                        Use ccache for compilation
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Default values
BUILD_TYPE="Release"
INSTALL_PREFIX="/usr/local"
CLEAN_BUILD=0
VERBOSE=0
SKIP_DEPS=0
PARALLEL_JOBS=""
RUN_TESTS=0
USE_CCACHE=0

# Feature flags (matching CMake options)
ENABLE_HTTP="ON"
ENABLE_GRPC="ON"
ENABLE_METRICS="ON"
ENABLE_LOGGING="ON"
ENABLE_STATS="ON"
ENABLE_TRACING="OFF"
ENABLE_ENSEMBLE="OFF"
ENABLE_S3="OFF"
ENABLE_GCS="OFF"
ENABLE_AZURE="OFF"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#########################################
# Helper Functions
#########################################

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "Triton Inference Server Build Script for macOS"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --build-type=<Debug|Release>    Build type (default: Release)"
    echo "  --install-prefix=<path>         Installation directory (default: /usr/local)"
    echo "  --clean                         Clean build (removes build directory)"
    echo "  --verbose                       Enable verbose output"
    echo "  --help                          Show this help message"
    echo "  --skip-deps                     Skip dependency installation"
    echo "  --parallel=<n>                  Number of parallel build jobs (default: auto)"
    echo "  --enable-http                   Enable HTTP endpoint (default: ON)"
    echo "  --enable-grpc                   Enable gRPC endpoint (default: ON)"
    echo "  --enable-metrics                Enable metrics (default: ON)"
    echo "  --enable-logging                Enable logging (default: ON)"
    echo "  --enable-stats                  Enable statistics (default: ON)"
    echo "  --enable-tracing                Enable tracing (default: OFF)"
    echo "  --enable-ensemble               Enable ensemble support (default: OFF)"
    echo "  --enable-s3                     Enable S3 support (default: OFF)"
    echo "  --enable-gcs                    Enable GCS support (default: OFF)"
    echo "  --enable-azure                  Enable Azure Storage support (default: OFF)"
    echo "  --run-tests                     Run tests after build"
    echo "  --ccache                        Use ccache for compilation"
    echo ""
    echo "Examples:"
    echo "  # Basic build"
    echo "  $0"
    echo ""
    echo "  # Debug build with verbose output"
    echo "  $0 --build-type=Debug --verbose"
    echo ""
    echo "  # Clean build with custom install prefix"
    echo "  $0 --clean --install-prefix=/opt/triton"
    echo ""
    echo "  # Build with cloud storage support"
    echo "  $0 --enable-s3 --enable-gcs"
}

#########################################
# Parse Command Line Arguments
#########################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-type=*)
            BUILD_TYPE="${1#*=}"
            if [[ ! "$BUILD_TYPE" =~ ^(Debug|Release)$ ]]; then
                print_error "Invalid build type: $BUILD_TYPE. Must be Debug or Release."
                exit 1
            fi
            ;;
        --install-prefix=*)
            INSTALL_PREFIX="${1#*=}"
            ;;
        --clean)
            CLEAN_BUILD=1
            ;;
        --verbose)
            VERBOSE=1
            ;;
        --skip-deps)
            SKIP_DEPS=1
            ;;
        --parallel=*)
            PARALLEL_JOBS="${1#*=}"
            ;;
        --enable-http)
            ENABLE_HTTP="ON"
            ;;
        --disable-http)
            ENABLE_HTTP="OFF"
            ;;
        --enable-grpc)
            ENABLE_GRPC="ON"
            ;;
        --disable-grpc)
            ENABLE_GRPC="OFF"
            ;;
        --enable-metrics)
            ENABLE_METRICS="ON"
            ;;
        --disable-metrics)
            ENABLE_METRICS="OFF"
            ;;
        --enable-logging)
            ENABLE_LOGGING="ON"
            ;;
        --disable-logging)
            ENABLE_LOGGING="OFF"
            ;;
        --enable-stats)
            ENABLE_STATS="ON"
            ;;
        --disable-stats)
            ENABLE_STATS="OFF"
            ;;
        --enable-tracing)
            ENABLE_TRACING="ON"
            ;;
        --disable-tracing)
            ENABLE_TRACING="OFF"
            ;;
        --enable-ensemble)
            ENABLE_ENSEMBLE="ON"
            ;;
        --disable-ensemble)
            ENABLE_ENSEMBLE="OFF"
            ;;
        --enable-s3)
            ENABLE_S3="ON"
            ;;
        --disable-s3)
            ENABLE_S3="OFF"
            ;;
        --enable-gcs)
            ENABLE_GCS="ON"
            ;;
        --disable-gcs)
            ENABLE_GCS="OFF"
            ;;
        --enable-azure)
            ENABLE_AZURE="ON"
            ;;
        --disable-azure)
            ENABLE_AZURE="OFF"
            ;;
        --run-tests)
            RUN_TESTS=1
            ;;
        --ccache)
            USE_CCACHE=1
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

#########################################
# System Detection
#########################################

detect_system() {
    print_info "Detecting system configuration..."
    
    # Check if running on macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        print_error "This script is designed for macOS only. Detected: $(uname)"
        exit 1
    fi
    
    # Get macOS version
    MACOS_VERSION=$(sw_vers -productVersion)
    print_info "macOS version: $MACOS_VERSION"
    
    # Check minimum macOS version (11.0 Big Sur)
    MACOS_MAJOR=$(echo "$MACOS_VERSION" | cut -d. -f1)
    if [[ $MACOS_MAJOR -lt 11 ]]; then
        print_error "macOS 11.0 or later is required. Current version: $MACOS_VERSION"
        exit 1
    fi
    
    # Detect architecture
    ARCH=$(uname -m)
    print_info "Architecture: $ARCH"
    
    if [[ "$ARCH" == "arm64" ]]; then
        print_info "Detected Apple Silicon Mac"
        HOMEBREW_PREFIX="/opt/homebrew"
    else
        print_info "Detected Intel Mac"
        HOMEBREW_PREFIX="/usr/local"
    fi
    
    # Detect number of cores for parallel builds
    if [[ -z "$PARALLEL_JOBS" ]]; then
        PARALLEL_JOBS=$(sysctl -n hw.ncpu)
        print_info "Using $PARALLEL_JOBS parallel build jobs"
    fi
}

#########################################
# Dependency Checking Functions
#########################################

check_xcode() {
    print_info "Checking for Xcode Command Line Tools..."
    
    if ! xcode-select -p &> /dev/null; then
        print_warning "Xcode Command Line Tools not found"
        print_info "Installing Xcode Command Line Tools..."
        xcode-select --install
        
        # Wait for installation
        print_info "Please complete the Xcode Command Line Tools installation and press Enter to continue..."
        read -r
        
        # Verify installation
        if ! xcode-select -p &> /dev/null; then
            print_error "Xcode Command Line Tools installation failed"
            exit 1
        fi
    fi
    
    XCODE_PATH=$(xcode-select -p)
    print_success "Xcode Command Line Tools found at: $XCODE_PATH"
    
    # Check compiler
    if ! command -v clang++ &> /dev/null; then
        print_error "clang++ compiler not found"
        exit 1
    fi
    
    CLANG_VERSION=$(clang++ --version | head -n1)
    print_info "Compiler: $CLANG_VERSION"
}

check_homebrew() {
    print_info "Checking for Homebrew..."
    
    if ! command -v brew &> /dev/null; then
        print_warning "Homebrew not found"
        print_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon
        if [[ "$ARCH" == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    fi
    
    BREW_VERSION=$(brew --version | head -n1)
    print_success "Homebrew found: $BREW_VERSION"
    
    # Update Homebrew
    print_info "Updating Homebrew..."
    brew update
}

check_cmake() {
    print_info "Checking for CMake..."
    
    if ! command -v cmake &> /dev/null; then
        print_warning "CMake not found"
        return 1
    fi
    
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
    CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
    
    # Require CMake 3.18 or later
    if [[ $CMAKE_MAJOR -lt 3 ]] || ([[ $CMAKE_MAJOR -eq 3 ]] && [[ $CMAKE_MINOR -lt 18 ]]); then
        print_warning "CMake version $CMAKE_VERSION is too old. Version 3.18 or later is required."
        return 1
    fi
    
    print_success "CMake found: version $CMAKE_VERSION"
    return 0
}

install_dependencies() {
    if [[ $SKIP_DEPS -eq 1 ]]; then
        print_info "Skipping dependency installation (--skip-deps)"
        return
    fi
    
    print_info "Installing dependencies..."
    
    # Core build tools
    local BUILD_DEPS=(
        cmake
        autoconf
        automake
        libtool
        pkg-config
    )
    
    # Optional tools
    if [[ $USE_CCACHE -eq 1 ]]; then
        BUILD_DEPS+=(ccache)
    fi
    
    # Core libraries
    local LIB_DEPS=(
        protobuf
        libevent
        rapidjson
        boost
        re2
        openssl
        libarchive
    )
    
    # Conditional dependencies
    if [[ "$ENABLE_GRPC" == "ON" ]]; then
        LIB_DEPS+=(grpc)
    fi
    
    if [[ "$ENABLE_S3" == "ON" ]]; then
        LIB_DEPS+=(aws-sdk-cpp)
    fi
    
    # Install dependencies
    for dep in "${BUILD_DEPS[@]}" "${LIB_DEPS[@]}"; do
        if brew list "$dep" &> /dev/null; then
            print_info "$dep is already installed"
        else
            print_info "Installing $dep..."
            brew install "$dep"
        fi
    done
    
    # Special handling for GCS support
    if [[ "$ENABLE_GCS" == "ON" ]]; then
        print_warning "Google Cloud C++ SDK needs to be built from source for GCS support"
        print_info "Please refer to: https://github.com/googleapis/google-cloud-cpp"
    fi
    
    # Install Python dependencies
    print_info "Installing Python dependencies..."
    if command -v python3 &> /dev/null; then
        python3 -m pip install --upgrade pip
        if [[ -f "${PROJECT_ROOT}/python/openai/requirements.txt" ]]; then
            python3 -m pip install -r "${PROJECT_ROOT}/python/openai/requirements.txt"
        fi
    else
        print_warning "Python 3 not found. Python dependencies not installed."
    fi
    
    print_success "Dependencies installed successfully"
}

#########################################
# Environment Setup
#########################################

setup_environment() {
    print_info "Setting up build environment..."
    
    # Set compiler
    export CC=clang
    export CXX=clang++
    
    # Set paths for Homebrew
    export PATH="${HOMEBREW_PREFIX}/bin:${PATH}"
    export LDFLAGS="-L${HOMEBREW_PREFIX}/lib"
    export CPPFLAGS="-I${HOMEBREW_PREFIX}/include"
    export PKG_CONFIG_PATH="${HOMEBREW_PREFIX}/lib/pkgconfig"
    
    # Set OpenSSL paths (Homebrew doesn't link it by default)
    if [[ -d "${HOMEBREW_PREFIX}/opt/openssl" ]]; then
        export LDFLAGS="${LDFLAGS} -L${HOMEBREW_PREFIX}/opt/openssl/lib"
        export CPPFLAGS="${CPPFLAGS} -I${HOMEBREW_PREFIX}/opt/openssl/include"
        export PKG_CONFIG_PATH="${HOMEBREW_PREFIX}/opt/openssl/lib/pkgconfig:${PKG_CONFIG_PATH}"
    fi
    
    # Set up ccache if requested
    if [[ $USE_CCACHE -eq 1 ]] && command -v ccache &> /dev/null; then
        export CC="ccache ${CC}"
        export CXX="ccache ${CXX}"
        print_info "Using ccache for compilation"
    fi
    
    # Verbose output
    if [[ $VERBOSE -eq 1 ]]; then
        export VERBOSE=1
        CMAKE_VERBOSE="-DCMAKE_VERBOSE_MAKEFILE=ON"
    else
        CMAKE_VERBOSE=""
    fi
    
    print_success "Environment configured"
}

#########################################
# Build Functions
#########################################

prepare_build_directory() {
    BUILD_DIR="${PROJECT_ROOT}/build"
    
    if [[ $CLEAN_BUILD -eq 1 ]]; then
        print_info "Performing clean build..."
        if [[ -d "$BUILD_DIR" ]]; then
            rm -rf "$BUILD_DIR"
            print_success "Build directory cleaned"
        fi
    fi
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
}

configure_cmake() {
    print_info "Configuring CMake..."
    
    # Base CMake arguments
    local CMAKE_ARGS=(
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
        "-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "${CMAKE_VERBOSE}"
    )
    
    # macOS specific settings
    CMAKE_ARGS+=(
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0"
        "-DCMAKE_CXX_STANDARD=17"
    )
    
    # Disable unsupported features on macOS
    CMAKE_ARGS+=(
        "-DTRITON_ENABLE_GPU=OFF"
        "-DTRITON_ENABLE_TENSORRT=OFF"
        "-DTRITON_ENABLE_NVTX=OFF"
        "-DTRITON_ENABLE_METRICS_GPU=OFF"
        "-DTRITON_ENABLE_MALI_GPU=OFF"
    )
    
    # Feature flags
    CMAKE_ARGS+=(
        "-DTRITON_ENABLE_HTTP=${ENABLE_HTTP}"
        "-DTRITON_ENABLE_GRPC=${ENABLE_GRPC}"
        "-DTRITON_ENABLE_METRICS=${ENABLE_METRICS}"
        "-DTRITON_ENABLE_METRICS_CPU=ON"
        "-DTRITON_ENABLE_LOGGING=${ENABLE_LOGGING}"
        "-DTRITON_ENABLE_STATS=${ENABLE_STATS}"
        "-DTRITON_ENABLE_TRACING=${ENABLE_TRACING}"
        "-DTRITON_ENABLE_ENSEMBLE=${ENABLE_ENSEMBLE}"
        "-DTRITON_ENABLE_S3=${ENABLE_S3}"
        "-DTRITON_ENABLE_GCS=${ENABLE_GCS}"
        "-DTRITON_ENABLE_AZURE_STORAGE=${ENABLE_AZURE}"
    )
    
    # Include the macOS CMake module
    CMAKE_ARGS+=(
        "-DCMAKE_MODULE_PATH=${PROJECT_ROOT}/cmake"
    )
    
    print_info "CMake configuration:"
    printf '%s\n' "${CMAKE_ARGS[@]}" | sed 's/^/  /'
    
    # Run CMake
    if ! cmake "${CMAKE_ARGS[@]}" "${PROJECT_ROOT}"; then
        print_error "CMake configuration failed"
        exit 1
    fi
    
    print_success "CMake configuration completed"
}

build_triton() {
    print_info "Building Triton Inference Server..."
    
    # Build command
    local BUILD_CMD=(cmake --build . --config "${BUILD_TYPE}")
    
    if [[ -n "$PARALLEL_JOBS" ]]; then
        BUILD_CMD+=(--parallel "$PARALLEL_JOBS")
    fi
    
    if [[ $VERBOSE -eq 1 ]]; then
        BUILD_CMD+=(--verbose)
    fi
    
    # Run build
    if ! "${BUILD_CMD[@]}"; then
        print_error "Build failed"
        exit 1
    fi
    
    print_success "Build completed successfully"
}

install_triton() {
    print_info "Installing Triton Inference Server..."
    
    # Check if we need sudo
    if [[ -w "$INSTALL_PREFIX" ]]; then
        cmake --install . --config "${BUILD_TYPE}"
    else
        print_info "Installation requires sudo privileges"
        sudo cmake --install . --config "${BUILD_TYPE}"
    fi
    
    print_success "Installation completed to: $INSTALL_PREFIX"
}

run_tests() {
    if [[ $RUN_TESTS -eq 0 ]]; then
        return
    fi
    
    print_info "Running tests..."
    
    # Check if ctest is available
    if ! command -v ctest &> /dev/null; then
        print_warning "ctest not found. Skipping tests."
        return
    fi
    
    # Run tests
    if ctest --output-on-failure -C "${BUILD_TYPE}"; then
        print_success "All tests passed"
    else
        print_error "Some tests failed"
        exit 1
    fi
}

#########################################
# Validation Functions
#########################################

validate_build() {
    print_info "Validating build..."
    
    # Check if the main executable was built
    local TRITON_EXEC="${BUILD_DIR}/src/tritonserver"
    if [[ ! -f "$TRITON_EXEC" ]]; then
        print_error "Triton server executable not found at: $TRITON_EXEC"
        exit 1
    fi
    
    # Check executable architecture
    local EXEC_ARCH=$(lipo -info "$TRITON_EXEC" 2>/dev/null | awk '{print $NF}')
    print_info "Built for architecture: $EXEC_ARCH"
    
    # Verify dynamic libraries
    print_info "Checking dynamic library dependencies..."
    otool -L "$TRITON_EXEC" | head -20
    
    # Check for common issues
    if otool -L "$TRITON_EXEC" | grep -q "libcuda"; then
        print_warning "Binary has CUDA dependencies which won't work on macOS"
    fi
    
    print_success "Build validation completed"
}

print_summary() {
    echo ""
    echo "========================================"
    echo "Build Summary"
    echo "========================================"
    echo "Build Type:        $BUILD_TYPE"
    echo "Install Prefix:    $INSTALL_PREFIX"
    echo "Architecture:      $ARCH"
    echo "macOS Version:     $MACOS_VERSION"
    echo "Parallel Jobs:     $PARALLEL_JOBS"
    echo ""
    echo "Features:"
    echo "  HTTP:            $ENABLE_HTTP"
    echo "  gRPC:            $ENABLE_GRPC"
    echo "  Metrics:         $ENABLE_METRICS"
    echo "  Logging:         $ENABLE_LOGGING"
    echo "  Statistics:      $ENABLE_STATS"
    echo "  Tracing:         $ENABLE_TRACING"
    echo "  Ensemble:        $ENABLE_ENSEMBLE"
    echo "  S3:              $ENABLE_S3"
    echo "  GCS:             $ENABLE_GCS"
    echo "  Azure Storage:   $ENABLE_AZURE"
    echo ""
    echo "Build Directory:   $BUILD_DIR"
    echo "========================================"
    echo ""
    
    print_success "Triton Inference Server built successfully!"
    echo ""
    echo "To run Triton Server:"
    echo "  ${INSTALL_PREFIX}/bin/tritonserver --model-repository=<model_repo_path>"
    echo ""
    echo "For more options:"
    echo "  ${INSTALL_PREFIX}/bin/tritonserver --help"
}

#########################################
# Main Execution
#########################################

main() {
    print_info "Starting Triton Inference Server build for macOS"
    echo ""
    
    # System detection
    detect_system
    
    # Check prerequisites
    check_xcode
    check_homebrew
    
    # Install/check dependencies
    if ! check_cmake; then
        install_dependencies
    else
        install_dependencies
    fi
    
    # Setup environment
    setup_environment
    
    # Prepare build
    prepare_build_directory
    
    # Configure
    configure_cmake
    
    # Build
    build_triton
    
    # Validate
    validate_build
    
    # Install
    install_triton
    
    # Run tests if requested
    run_tests
    
    # Print summary
    print_summary
}

# Run main function
main "$@"