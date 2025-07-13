#!/bin/bash
# CMake wrapper to handle version compatibility

# Get the real cmake path
REAL_CMAKE=$(which cmake)

# Check if we're building a dependency that needs patching
if [[ "$*" =~ "libevent" ]] || [[ "$PWD" =~ "libevent" ]]; then
    # For libevent, we need to patch CMakeLists.txt first
    if [[ -f "CMakeLists.txt" ]] && grep -q "cmake_minimum_required.*VERSION.*3\.[0-4]" CMakeLists.txt; then
        echo "Patching libevent CMakeLists.txt for CMake 4.0.3 compatibility..."
        sed -i.bak 's/cmake_minimum_required.*VERSION.*3\.[0-4]/cmake_minimum_required(VERSION 3.10/' CMakeLists.txt
    fi
fi

# Execute the real cmake
exec "$REAL_CMAKE" "$@"
