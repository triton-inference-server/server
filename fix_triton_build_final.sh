#!/bin/bash
# Fix Triton build issues on macOS with proper include paths

echo "🔧 Fixing Triton build issues..."

# Fix the main CMakeLists.txt to prioritize vendored libraries
cat > /tmp/cmake_patch.txt << 'EOF'
# Remove any system include paths that might interfere
list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/opt/homebrew/include")
list(REMOVE_ITEM CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/opt/homebrew/include")

# Set include directories with proper order
target_include_directories(triton-core BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add vendored protobuf includes with highest priority
target_include_directories(triton-core SYSTEM BEFORE PRIVATE 
    ${CMAKE_BINARY_DIR}/../../../third-party/protobuf/include
)

# Add other system includes after
if(${Protobuf_INCLUDE_DIRS})
    target_include_directories(triton-core SYSTEM PRIVATE ${Protobuf_INCLUDE_DIRS})
endif()

# Boost should come last to avoid conflicts
if(${Boost_INCLUDE_DIRS})
    target_include_directories(triton-core SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
endif()
EOF

# Backup and patch the CMakeLists.txt
cd /Volumes/Untitled/coder/server/build/_deps/repo-core-src/src
cp CMakeLists.txt CMakeLists.txt.bak

# Find the target_include_directories section and replace it
sed -i '' '/target_include_directories(/,/^$/d' CMakeLists.txt
sed -i '' "/^)$/r /tmp/cmake_patch.txt" CMakeLists.txt

# Also fix test CMakeLists.txt if it exists
if [ -f "test/CMakeLists.txt" ]; then
    echo "Fixing test CMakeLists.txt..."
    cd test
    # Remove homebrew includes from test builds
    sed -i '' 's|/opt/homebrew/include||g' CMakeLists.txt
    cd ..
fi

# Clean build artifacts
echo "🧹 Cleaning build artifacts..."
cd /Volumes/Untitled/coder/server/build
rm -rf _deps/repo-core-build/triton-core/CMakeFiles
rm -f _deps/repo-core-build/triton-core/CMakeCache.txt

# Reconfigure
echo "🔄 Reconfiguring build..."
cd _deps/repo-core-build/triton-core
cmake . \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_PREFIX_PATH="/Volumes/Untitled/coder/server/build/third-party" \
    -DProtobuf_DIR="/Volumes/Untitled/coder/server/build/third-party/protobuf/lib/cmake/protobuf" \
    -Dabsl_DIR="/Volumes/Untitled/coder/server/build/third-party/absl/lib/cmake/absl"

echo "✅ Build fixes applied. Run 'make triton-core' to continue building."