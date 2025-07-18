#!/bin/bash
# Download and set up LibTorch for macOS

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIBTORCH_DIR="${SCRIPT_DIR}/libtorch"

# Check if LibTorch is already downloaded
if [ -d "${LIBTORCH_DIR}" ]; then
    echo "LibTorch already exists at ${LIBTORCH_DIR}"
    exit 0
fi

echo "Downloading LibTorch for macOS..."

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" == "arm64" ]; then
    # Apple Silicon
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.1.0.zip"
    echo "Detected Apple Silicon (arm64)"
else
    # Intel
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.1.0.zip"
    echo "Detected Intel (x86_64)"
fi

# Download LibTorch
TEMP_FILE="/tmp/libtorch.zip"
echo "Downloading from: ${LIBTORCH_URL}"
curl -L "${LIBTORCH_URL}" -o "${TEMP_FILE}"

# Extract
echo "Extracting LibTorch..."
unzip -q "${TEMP_FILE}" -d "${SCRIPT_DIR}"

# Clean up
rm "${TEMP_FILE}"

echo "LibTorch successfully downloaded to ${LIBTORCH_DIR}"

# Fix library paths for macOS
echo "Fixing library paths for macOS..."
cd "${LIBTORCH_DIR}/lib"

# Function to fix single library
fix_lib() {
    local lib=$1
    if [ -f "$lib" ]; then
        echo "Fixing $lib"
        # Get all dependencies
        local deps=$(otool -L "$lib" | grep -E "^\s*@rpath" | awk '{print $1}')
        for dep in $deps; do
            local dep_name=$(basename "$dep")
            if [ -f "${LIBTORCH_DIR}/lib/${dep_name}" ]; then
                install_name_tool -change "$dep" "@loader_path/${dep_name}" "$lib" 2>/dev/null || true
            fi
        done
        # Fix the library's own install name if it uses @rpath
        local lib_id=$(otool -D "$lib" | tail -n 1)
        if [[ "$lib_id" == *"@rpath"* ]]; then
            install_name_tool -id "@loader_path/$(basename "$lib")" "$lib" 2>/dev/null || true
        fi
    fi
}

# Fix all dylib files
for lib in *.dylib; do
    fix_lib "$lib"
done

echo "Library paths fixed"
echo "Setup complete!"