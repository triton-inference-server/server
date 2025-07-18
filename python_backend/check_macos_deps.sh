#!/bin/bash

# Script to check macOS dependencies for Python backend

echo "Checking macOS dependencies for Triton Python Backend..."
echo "=================================================="

# Check for required tools
check_command() {
    if command -v $1 &> /dev/null; then
        echo "✓ $1 found: $(command -v $1)"
        if [ "$2" = "version" ]; then
            $1 --version 2>&1 | head -n 1
        fi
    else
        echo "✗ $1 NOT FOUND - Please install it"
        return 1
    fi
}

# Check build tools
echo -e "\nBuild Tools:"
check_command cmake version
check_command make version
check_command clang version

# Check Python
echo -e "\nPython:"
check_command python3 version
if command -v python3 &> /dev/null; then
    echo "  Python location: $(which python3)"
    echo "  Python version: $(python3 --version)"
    
    # Check for Python development files
    python3 -c "import sysconfig; print(f'  Include dir: {sysconfig.get_path(\"include\")}')"
    python3 -c "import sysconfig; print(f'  Library dir: {sysconfig.get_config_var(\"LIBDIR\")}')"
fi

# Check for Homebrew (common on macOS)
echo -e "\nPackage Managers:"
if command -v brew &> /dev/null; then
    echo "✓ Homebrew found: $(brew --version | head -n 1)"
else
    echo "ℹ Homebrew not found (optional but recommended)"
fi

# Check for required libraries
echo -e "\nRequired Libraries:"

# Check for Boost
if [ -d "/opt/homebrew/include/boost" ] || [ -d "/usr/local/include/boost" ]; then
    echo "✓ Boost headers found"
else
    echo "✗ Boost headers NOT FOUND - Install with: brew install boost"
fi

# Check for libarchive
if pkg-config --exists libarchive 2>/dev/null || [ -f "/opt/homebrew/lib/libarchive.dylib" ] || [ -f "/usr/local/lib/libarchive.dylib" ]; then
    echo "✓ libarchive found"
else
    echo "✗ libarchive NOT FOUND - Install with: brew install libarchive"
fi

# Check for numpy (required for Python models)
echo -e "\nPython Packages:"
if python3 -c "import numpy" 2>/dev/null; then
    echo "✓ numpy found"
else
    echo "✗ numpy NOT FOUND - Install with: pip3 install numpy"
fi

# System info
echo -e "\nSystem Information:"
echo "  macOS version: $(sw_vers -productVersion)"
echo "  Architecture: $(uname -m)"
echo "  Xcode version: $(xcodebuild -version 2>/dev/null | head -n 1 || echo 'Xcode not installed')"

# Check dynamic library path
echo -e "\nDynamic Library Path:"
echo "  DYLD_LIBRARY_PATH: ${DYLD_LIBRARY_PATH:-'(not set)'}"
echo "  DYLD_FALLBACK_LIBRARY_PATH: ${DYLD_FALLBACK_LIBRARY_PATH:-'(not set)'}"

echo -e "\n=================================================="
echo "Dependency check complete!"