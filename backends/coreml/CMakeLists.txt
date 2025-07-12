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

project(tritoncoremlbackend LANGUAGES C CXX OBJCXX)

# C++17 is required for modern C++ features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Objective-C++ settings
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Ensure we're building on macOS
if(NOT APPLE)
  message(FATAL_ERROR "CoreML backend can only be built on macOS")
endif()

# Options
option(TRITON_ENABLE_GPU "Enable GPU support (Metal)" ON)
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

# Find required Apple frameworks
find_library(COREML_FRAMEWORK CoreML REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal)
find_library(METAL_PERFORMANCE_SHADERS_FRAMEWORK MetalPerformanceShaders)

message(STATUS "CoreML Framework: ${COREML_FRAMEWORK}")
message(STATUS "Foundation Framework: ${FOUNDATION_FRAMEWORK}")
if(METAL_FRAMEWORK)
  message(STATUS "Metal Framework: ${METAL_FRAMEWORK}")
endif()

# CoreML backend shared library
set(COREML_BACKEND_SRCS
  src/coreml.cc
)

add_library(triton-coreml-backend SHARED ${COREML_BACKEND_SRCS})

# Set Objective-C++ language for the source file
set_source_files_properties(src/coreml.cc PROPERTIES LANGUAGE OBJCXX)

target_include_directories(
  triton-coreml-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(triton-coreml-backend PRIVATE cxx_std_17)

# macOS specific flags
target_compile_options(
  triton-coreml-backend
  PRIVATE
    -fobjc-arc  # Enable ARC for Objective-C++
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-deprecated-declarations
)

# Link libraries
target_link_libraries(
  triton-coreml-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-backend-utils    # from repo-backend
    ${COREML_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Optionally link Metal frameworks
if(TRITON_ENABLE_GPU AND METAL_FRAMEWORK)
  target_link_libraries(
    triton-coreml-backend
    PRIVATE
      ${METAL_FRAMEWORK}
  )
  if(METAL_PERFORMANCE_SHADERS_FRAMEWORK)
    target_link_libraries(
      triton-coreml-backend
      PRIVATE
        ${METAL_PERFORMANCE_SHADERS_FRAMEWORK}
    )
  endif()
endif()

# macOS specific linking
set_target_properties(
  triton-coreml-backend
  PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup"
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path"
)

# Version script not needed for macOS
# Output name
set_target_properties(
  triton-coreml-backend
  PROPERTIES
    OUTPUT_NAME "triton_coreml"
)

# Install
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonCoreMLBackend)

install(
  TARGETS triton-coreml-backend
  EXPORT triton-coreml-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/coreml
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/coreml
)

# Export configuration
install(
  EXPORT triton-coreml-backend-targets
  FILE TritonCoreMLBackendTargets.cmake
  NAMESPACE TritonCoreMLBackend::
  DESTINATION ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TritonCoreMLBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonCoreMLBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/TritonCoreMLBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

# Print configuration summary
message(STATUS "")
message(STATUS "CoreML Backend Configuration Summary")
message(STATUS "====================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "GPU support (Metal): ${TRITON_ENABLE_GPU}")
message(STATUS "Statistics collection: ${TRITON_ENABLE_STATS}")
message(STATUS "")