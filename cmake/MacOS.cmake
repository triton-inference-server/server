# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# macOS/Darwin Platform Configuration
#

# Detect macOS version
if(APPLE)
  execute_process(
    COMMAND sw_vers -productVersion
    OUTPUT_VARIABLE MACOS_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "Detected macOS version: ${MACOS_VERSION}")
  
  # Detect if running on Apple Silicon
  execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE MACOS_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(MACOS_ARCH STREQUAL "arm64")
    set(TRITON_MACOS_APPLE_SILICON ON)
    message(STATUS "Detected Apple Silicon (${MACOS_ARCH})")
  else()
    set(TRITON_MACOS_APPLE_SILICON OFF)
    message(STATUS "Detected Intel Mac (${MACOS_ARCH})")
  endif()
endif()

# macOS-specific compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
  # Base flags for Apple Clang
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  
  # Warning flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
  
  # macOS-specific flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
  
  # Apple Silicon optimizations
  if(TRITON_MACOS_APPLE_SILICON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
  endif()
  
  # Debug flags
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
  
  # Release flags
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
  
  # Minimum macOS version
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment version")
  endif()
endif()

# macOS Framework paths
set(CMAKE_FRAMEWORK_PATH
  /System/Library/Frameworks
  /Library/Frameworks
  ~/Library/Frameworks
)

# RPATH configuration for macOS
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Set RPATH to find libraries relative to executable
set(CMAKE_INSTALL_RPATH "@loader_path/../lib;@loader_path/../lib64")

# Library suffix for macOS
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
set(CMAKE_SHARED_MODULE_SUFFIX ".so")

# Disable features not supported on macOS
if(APPLE)
  # GPU features not supported on macOS (CUDA is NVIDIA-specific)
  if(TRITON_ENABLE_GPU)
    message(WARNING "GPU support (CUDA) is not available on macOS. Disabling TRITON_ENABLE_GPU.")
    set(TRITON_ENABLE_GPU OFF CACHE BOOL "GPU support not available on macOS" FORCE)
  endif()
  
  if(TRITON_ENABLE_MALI_GPU)
    message(WARNING "Mali GPU support is not available on macOS. Disabling TRITON_ENABLE_MALI_GPU.")
    set(TRITON_ENABLE_MALI_GPU OFF CACHE BOOL "Mali GPU support not available on macOS" FORCE)
  endif()
  
  if(TRITON_ENABLE_TENSORRT)
    message(WARNING "TensorRT is not available on macOS. Disabling TRITON_ENABLE_TENSORRT.")
    set(TRITON_ENABLE_TENSORRT OFF CACHE BOOL "TensorRT not available on macOS" FORCE)
  endif()
  
  if(TRITON_ENABLE_NVTX)
    message(WARNING "NVTX is not available on macOS. Disabling TRITON_ENABLE_NVTX.")
    set(TRITON_ENABLE_NVTX OFF CACHE BOOL "NVTX not available on macOS" FORCE)
  endif()
  
  # Metrics GPU requires GPU support
  if(TRITON_ENABLE_METRICS_GPU)
    message(WARNING "GPU metrics require GPU support. Disabling TRITON_ENABLE_METRICS_GPU.")
    set(TRITON_ENABLE_METRICS_GPU OFF CACHE BOOL "GPU metrics not available on macOS" FORCE)
  endif()
endif()

# macOS-specific library search paths
if(APPLE)
  # Homebrew paths (both Intel and Apple Silicon)
  if(TRITON_MACOS_APPLE_SILICON)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
    set(HOMEBREW_PREFIX "/opt/homebrew")
  else()
    list(APPEND CMAKE_PREFIX_PATH "/usr/local")
    set(HOMEBREW_PREFIX "/usr/local")
  endif()
  
  # Add common Homebrew library paths
  list(APPEND CMAKE_LIBRARY_PATH "${HOMEBREW_PREFIX}/lib")
  list(APPEND CMAKE_INCLUDE_PATH "${HOMEBREW_PREFIX}/include")
  
  # MacPorts paths (if used)
  if(EXISTS "/opt/local")
    list(APPEND CMAKE_PREFIX_PATH "/opt/local")
    list(APPEND CMAKE_LIBRARY_PATH "/opt/local/lib")
    list(APPEND CMAKE_INCLUDE_PATH "/opt/local/include")
  endif()
endif()

# Thread library configuration for macOS
if(APPLE)
  set(CMAKE_THREAD_LIBS_INIT "-pthread")
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_WIN32_THREADS_INIT 0)
  set(CMAKE_USE_PTHREADS_INIT 1)
endif()

# Handle dynamic library loading differences
if(APPLE)
  add_compile_definitions(TRITON_MACOS=1)
  if(TRITON_MACOS_APPLE_SILICON)
    add_compile_definitions(TRITON_APPLE_SILICON=1)
  endif()
endif()

# Export variables for use in other CMake files
set(TRITON_MACOS ${APPLE} PARENT_SCOPE)
set(TRITON_MACOS_APPLE_SILICON ${TRITON_MACOS_APPLE_SILICON} PARENT_SCOPE)

message(STATUS "macOS platform configuration complete")