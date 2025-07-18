# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.17)

project(tritonpytorchbackend LANGUAGES C CXX)

# C++17 is required for PyTorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(TRITON_ENABLE_GPU "Enable GPU support" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

# Dependencies
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

# Find LibTorch
if(APPLE)
  # macOS specific LibTorch handling
  if(NOT DEFINED TORCH_PATH)
    # Try to find LibTorch in common locations
    set(TORCH_SEARCH_PATHS
      "${CMAKE_CURRENT_SOURCE_DIR}/libtorch"
      "/opt/libtorch"
      "/usr/local/libtorch"
      "$ENV{HOME}/libtorch"
    )
    
    foreach(path ${TORCH_SEARCH_PATHS})
      if(EXISTS "${path}/share/cmake/Torch/TorchConfig.cmake")
        set(TORCH_PATH ${path})
        message(STATUS "Found LibTorch at: ${TORCH_PATH}")
        break()
      endif()
    endforeach()
    
    if(NOT DEFINED TORCH_PATH)
      message(FATAL_ERROR "Could not find LibTorch. Please set TORCH_PATH or download LibTorch to one of the search paths.")
    endif()
  endif()
  
  set(CMAKE_PREFIX_PATH ${TORCH_PATH})
  
  # macOS specific flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
  
  # Disable OpenMP on macOS as it can cause issues
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAT_PARALLEL_OPENMP=0")
  
  # Enable MPS support
  option(TRITON_ENABLE_MPS "Enable Metal Performance Shaders support" ON)
  if(TRITON_ENABLE_MPS)
    message(STATUS "MPS (Metal Performance Shaders) support enabled")
    add_definitions(-DUSE_MPS)
  endif()
else()
  # Linux LibTorch handling
  if(NOT DEFINED TORCH_PATH)
    set(TORCH_PATH "/opt/pytorch/pytorch")
  endif()
  set(CMAKE_PREFIX_PATH ${TORCH_PATH})
endif()

find_package(Torch REQUIRED)

# PyTorch backend shared library
set(PYTORCH_BACKEND_SRCS
  src/pytorch.cc
)

add_library(triton-pytorch-backend SHARED ${PYTORCH_BACKEND_SRCS})

target_include_directories(
  triton-pytorch-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(triton-pytorch-backend PRIVATE cxx_std_17)
target_compile_options(
  triton-pytorch-backend
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -Wno-unused-parameter>
)

target_link_libraries(
  triton-pytorch-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-backend-utils    # from repo-backend
    ${TORCH_LIBRARIES}
)

if(APPLE)
  # macOS specific linking
  set_target_properties(
    triton-pytorch-backend
    PROPERTIES
      LINK_FLAGS "-undefined dynamic_lookup"
      BUILD_RPATH "@loader_path;${TORCH_INSTALL_PREFIX}/lib"
      INSTALL_RPATH "@loader_path;${TORCH_INSTALL_PREFIX}/lib"
  )
  
  # Find and link Metal framework for MPS support
  if(TRITON_ENABLE_MPS)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders)
    find_library(METALPERFORMANCESHADERSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)
    
    if(METAL_FRAMEWORK)
      message(STATUS "Found Metal framework: ${METAL_FRAMEWORK}")
      target_link_libraries(triton-pytorch-backend PRIVATE ${METAL_FRAMEWORK})
    endif()
    
    if(METALPERFORMANCESHADERS_FRAMEWORK)
      message(STATUS "Found MetalPerformanceShaders framework: ${METALPERFORMANCESHADERS_FRAMEWORK}")
      target_link_libraries(triton-pytorch-backend PRIVATE ${METALPERFORMANCESHADERS_FRAMEWORK})
    endif()
    
    if(METALPERFORMANCESHADERSGRAPH_FRAMEWORK)
      message(STATUS "Found MetalPerformanceShadersGraph framework: ${METALPERFORMANCESHADERSGRAPH_FRAMEWORK}")
      target_link_libraries(triton-pytorch-backend PRIVATE ${METALPERFORMANCESHADERSGRAPH_FRAMEWORK})
    endif()
  endif()
else()
  # Linux specific
  set_target_properties(
    triton-pytorch-backend
    PROPERTIES
      LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_pytorch.ldscript
      LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/libtriton_pytorch.ldscript"
  )
endif()

# Create version script for Linux
if(NOT APPLE)
  configure_file(src/libtriton_pytorch.ldscript libtriton_pytorch.ldscript COPYONLY)
endif()

# Install
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonPyTorchBackend)

install(
  TARGETS triton-pytorch-backend
  EXPORT triton-pytorch-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/pytorch
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/pytorch
)

# Install LibTorch libraries on macOS
if(APPLE)
  # Get LibTorch library directory
  get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARIES}" DIRECTORY)
  
  # Install LibTorch libraries
  file(GLOB TORCH_LIBS "${TORCH_LIB_DIR}/*.dylib")
  install(
    FILES ${TORCH_LIBS}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/pytorch
  )
endif()

# Export configuration
install(
  EXPORT triton-pytorch-backend-targets
  FILE TritonPyTorchBackendTargets.cmake
  NAMESPACE TritonPyTorchBackend::
  DESTINATION ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TritonPyTorchBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonPyTorchBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/TritonPyTorchBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)