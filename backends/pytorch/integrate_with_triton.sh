#!/bin/bash
# Script to integrate PyTorch backend with Triton server build

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVER_DIR="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Integrating PyTorch backend with Triton server${NC}"

# Add PyTorch backend option to server CMakeLists.txt if not already present
CMAKE_FILE="${SERVER_DIR}/CMakeLists.txt"
if ! grep -q "TRITON_ENABLE_PYTORCH_BACKEND" "${CMAKE_FILE}"; then
    echo -e "${YELLOW}Adding PyTorch backend option to CMakeLists.txt${NC}"
    
    # Find a good place to insert (after other backend options)
    # We'll add it after TRITON_ENABLE_ENSEMBLE
    sed -i.bak '/option(TRITON_ENABLE_ENSEMBLE/a\
option(TRITON_ENABLE_PYTORCH_BACKEND "Build PyTorch backend" OFF)' "${CMAKE_FILE}"
    
    echo -e "${GREEN}CMake option added${NC}"
else
    echo -e "${GREEN}PyTorch backend option already present in CMakeLists.txt${NC}"
fi

# Create a cmake module for finding/building PyTorch backend
BACKEND_CMAKE="${SERVER_DIR}/cmake/pytorch_backend.cmake"
cat > "${BACKEND_CMAKE}" << 'EOF'
# PyTorch Backend Integration

if(TRITON_ENABLE_PYTORCH_BACKEND)
  message(STATUS "Building PyTorch backend")
  
  # Check if backends/pytorch exists
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/backends/pytorch/CMakeLists.txt")
    add_subdirectory(backends/pytorch)
  else()
    message(WARNING "PyTorch backend source not found at backends/pytorch")
  endif()
endif()
EOF

echo -e "${GREEN}Created ${BACKEND_CMAKE}${NC}"

# Add include to main CMakeLists.txt if not already present
if ! grep -q "pytorch_backend.cmake" "${CMAKE_FILE}"; then
    echo -e "${YELLOW}Adding include for pytorch_backend.cmake${NC}"
    
    # Add after the project() command
    sed -i.bak '/^project(tritonserver/a\
\
# Include PyTorch backend if enabled\
include(cmake/pytorch_backend.cmake)' "${CMAKE_FILE}"
    
    echo -e "${GREEN}Include added${NC}"
fi

echo -e "\n${GREEN}Integration complete!${NC}"
echo -e "${GREEN}To build with PyTorch backend, use:${NC}"
echo -e "  cmake .. -DTRITON_ENABLE_PYTORCH_BACKEND=ON"
echo -e "\n${GREEN}Or add to your build script:${NC}"
echo -e "  -DTRITON_ENABLE_PYTORCH_BACKEND=ON"