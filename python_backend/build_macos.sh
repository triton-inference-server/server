#!/bin/bash

# Build script for Python backend on macOS

set -e

# Clean previous build
if [ -d "build" ]; then
    echo "Cleaning previous build..."
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_INSTALL_PREFIX:PATH=$(pwd)/install \
    -DTRITON_ENABLE_GPU=OFF \
    -DTRITON_BACKEND_REPO_TAG=main \
    -DTRITON_COMMON_REPO_TAG=main \
    -DTRITON_CORE_REPO_TAG=main \
    -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
make -j$(sysctl -n hw.ncpu)

# Install
echo "Installing..."
make install

echo "Build complete!"
echo "Python backend installed to: $(pwd)/install/backends/python"