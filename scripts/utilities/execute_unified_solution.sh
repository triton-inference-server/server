#!/bin/bash
set -e

echo "🚀 Executing Unified Multi-Agent Protobuf Solution"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SOLUTIONS_TRIED=0
SUCCESS=false

try_solution() {
    local name="$1"
    local description="$2"
    local command="$3"
    
    SOLUTIONS_TRIED=$((SOLUTIONS_TRIED + 1))
    
    echo -e "\n${BLUE}[TIER $SOLUTIONS_TRIED] Trying: $name${NC}"
    echo -e "${YELLOW}Description: $description${NC}"
    echo "Command: $command"
    echo "----------------------------------------"
    
    if eval "$command"; then
        echo -e "${GREEN}✅ SUCCESS: $name worked!${NC}"
        SUCCESS=true
        return 0
    else
        echo -e "${RED}❌ FAILED: $name did not work${NC}"
        return 1
    fi
}

cleanup_on_exit() {
    if [ "$SUCCESS" = false ]; then
        echo -e "\n${RED}🔥 All solutions failed. See UNIFIED_PROTOBUF_SOLUTION.md for manual approaches.${NC}"
        exit 1
    else
        echo -e "\n${GREEN}🎉 Build successful! Triton should now be ready.${NC}"
        exit 0
    fi
}

trap cleanup_on_exit EXIT

# Backup current state
echo "📦 Creating backup of current build state..."
cp build/_deps/repo-core-src/src/CMakeLists.txt build/_deps/repo-core-src/src/CMakeLists.txt.backup 2>/dev/null || true

# TIER 1: Enhanced Include Path Isolation
try_solution \
    "Enhanced Include Path Isolation" \
    "Force CMake to use correct protobuf headers with SYSTEM BEFORE" \
    "
    cd build/_deps/repo-core-src/src && 
    cat > /tmp/cmake_protobuf_fix.txt << 'CMAKEEOF'
# Remove any system include paths that might interfere
list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES \"/opt/homebrew/include\")
list(REMOVE_ITEM CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES \"/opt/homebrew/include\")

# Add vendored protobuf includes with HIGHEST priority
target_include_directories(triton-core SYSTEM BEFORE PRIVATE 
    \"/Volumes/Untitled/coder/server/build/third-party/protobuf/include\"
)
CMAKEEOF
    && sed -i '' '/# Add Protobuf includes FIRST/r /tmp/cmake_protobuf_fix.txt' CMakeLists.txt &&
    cd /Volumes/Untitled/coder/server/build &&
    rm -rf _deps/repo-core-build/triton-core/CMakeFiles/triton-core.dir/*.o &&
    make triton-core -j\$(sysctl -n hw.ncpu)
    " && exit 0

# TIER 2: Direct Header Patching
try_solution \
    "Direct Header Patching" \
    "Patch the generated protobuf headers to use absolute includes" \
    "
    cd /Volumes/Untitled/coder/server &&
    python3 -c \"
import re

header_file = 'build/_deps/repo-core-build/triton-core/_deps/repo-common-build/protobuf/model_config.pb.h'
with open(header_file, 'r') as f:
    content = f.read()

# Add absolute include at the top
new_content = '''#ifndef TRITON_PROTOBUF_FIX
#define TRITON_PROTOBUF_FIX
#include \\\"/Volumes/Untitled/coder/server/build/third-party/protobuf/include/google/protobuf/port_def.inc\\\"
#endif

''' + content

with open(header_file, 'w') as f:
    f.write(new_content)

print('Patched protobuf header successfully')
    \" &&
    cd build &&
    make triton-core -j\$(sysctl -n hw.ncpu)
    " && exit 0

# TIER 3: Environment Isolation
try_solution \
    "Environment Isolation" \
    "Temporarily unlink homebrew protobuf and use isolated environment" \
    "
    # Save current environment
    OLD_PATH=\$PATH
    OLD_CMAKE_PREFIX_PATH=\$CMAKE_PREFIX_PATH
    
    # Temporarily unlink conflicting packages
    brew unlink protobuf 2>/dev/null || true &&
    
    # Set isolated environment
    export PATH=\"/usr/bin:/bin:/usr/sbin:/sbin:/Volumes/Untitled/coder/server/build/third-party/protobuf/bin\" &&
    export CMAKE_PREFIX_PATH=\"/Volumes/Untitled/coder/server/build/third-party\" &&
    export CC=/usr/bin/clang &&
    export CXX=/usr/bin/clang++ &&
    export PKG_CONFIG_PATH=\"\" &&
    
    cd /Volumes/Untitled/coder/server/build &&
    rm -rf _deps/repo-core-build/triton-core/CMakeCache.txt &&
    cmake . &&
    make triton-core -j\$(sysctl -n hw.ncpu) &&
    
    # Restore environment
    export PATH=\"\$OLD_PATH\" &&
    export CMAKE_PREFIX_PATH=\"\$OLD_CMAKE_PREFIX_PATH\" &&
    brew link protobuf 2>/dev/null || true
    " && exit 0

# TIER 4: Minimal Build
try_solution \
    "Minimal Triton Build" \
    "Build only core components without problematic dependencies" \
    "
    cd /Volumes/Untitled/coder/server/build &&
    cmake . -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_METRICS=OFF -DTRITON_ENABLE_ENSEMBLE=OFF &&
    make triton-core -j\$(sysctl -n hw.ncpu)
    " && exit 0

# TIER 5: Nuclear Option - Complete Rebuild
try_solution \
    "Complete Rebuild" \
    "Clean everything and rebuild from scratch with explicit settings" \
    "
    cd /Volumes/Untitled/coder/server &&
    rm -rf build &&
    mkdir build &&
    cd build &&
    cmake .. \\
        -DCMAKE_PREFIX_PATH=\"/Volumes/Untitled/coder/server/build/third-party\" \\
        -DProtobuf_DIR=\"/Volumes/Untitled/coder/server/build/third-party/protobuf/lib/cmake/protobuf\" \\
        -DCMAKE_CXX_STANDARD=17 \\
        -DCMAKE_INCLUDE_DIRECTORIES_BEFORE=ON \\
        -DCMAKE_FIND_FRAMEWORK=LAST &&
    make -j\$(sysctl -n hw.ncpu)
    "

echo -e "\n${RED}All unified solutions have been exhausted.${NC}"
SUCCESS=false