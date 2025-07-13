#!/bin/bash
# Pre-build hook to fix CMake issues

# Fix any downloaded dependencies
if [[ -d "build/_deps" ]]; then
    echo "Fixing CMake files in downloaded dependencies..."
    find build/_deps -name "CMakeLists.txt" -type f | while read -r file; do
        if grep -q "cmake_minimum_required.*VERSION.*3\.[0-4]" "$file"; then
            echo "Patching: $file"
            sed -i.bak 's/cmake_minimum_required.*VERSION.*3\.[0-4]/cmake_minimum_required(VERSION 3.10/' "$file"
        fi
    done
fi

# Fix any third-party dependencies
if [[ -d "third-party" ]]; then
    echo "Fixing CMake files in third-party..."
    find third-party -name "CMakeLists.txt" -type f | while read -r file; do
        if grep -q "cmake_minimum_required.*VERSION.*3\.[0-4]" "$file"; then
            echo "Patching: $file"
            sed -i.bak 's/cmake_minimum_required.*VERSION.*3\.[0-4]/cmake_minimum_required(VERSION 3.10/' "$file"
        fi
    done
fi
