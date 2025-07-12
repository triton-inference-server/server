# Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Allow compatibility with older CMake versions in dependencies
cmake_policy(SET CMP0135 NEW)
if(POLICY CMP0147)
  cmake_policy(SET CMP0147 OLD)
endif()

project(tritonpythonbackend LANGUAGES C CXX)

# Use C++17 standard as Triton's minimum required.
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_NVTX "Include nvtx markers collection in backend." OFF)

# FIXME: CI needs to enable the GPU flag. Python for window currently does not
# support GPU tensors. For simplicity, we will override this option here.
if(WIN32)
  set(TRITON_ENABLE_GPU OFF CACHE BOOL "GPU disabled" FORCE)
endif()

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent's composability isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

# We need to use ExternalProject because we want to use Boost 1.76 which is not
# available in Ubuntu 20.04
include(ExternalProject)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY "https://github.com/pybind/pybind11"
  # COMMIT ID for v2.12.0
  GIT_TAG "3e9dfa2866941655c56877882565e7577de6fc7b"
  GIT_SHALLOW ON
)

# RHEL base container has multiple version of Python installed. By default
# it seems like pybind will pickup v3.6, so we specifically assign it to
# search for 3.12 here.
set(RHEL_BUILD OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(STRINGS "/etc/os-release" DISTRO_ID_LIKE REGEX "ID_LIKE")
  if(${DISTRO_ID_LIKE} MATCHES "rhel|centos")
    set(RHEL_BUILD ON)
  endif(${DISTRO_ID_LIKE} MATCHES "rhel|centos")
endif()

# macOS Python detection - ensure we use the correct Python version
if(APPLE)
  # First try to find Python3 using the modern FindPython3 module
  find_package(Python3 COMPONENTS Interpreter Development QUIET)
  if(Python3_FOUND)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE FILEPATH "Python executable" FORCE)
    set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS} CACHE PATH "Python include directories" FORCE)
    set(PYTHON_LIBRARIES ${Python3_LIBRARIES} CACHE FILEPATH "Python libraries" FORCE)
    message(STATUS "Found Python3: ${Python3_EXECUTABLE} (found version \"${Python3_VERSION}\")")
    message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
  else()
    # Fallback to manual detection for Homebrew Python
    execute_process(
      COMMAND brew --prefix python3
      OUTPUT_VARIABLE PYTHON_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(PYTHON_PREFIX)
      set(PYTHON_EXECUTABLE "${PYTHON_PREFIX}/bin/python3" CACHE FILEPATH "Python executable" FORCE)
      message(STATUS "Found Homebrew Python3: ${PYTHON_EXECUTABLE}")
    endif()
  endif()
endif()

FetchContent_MakeAvailable(pybind11)

#
# DLPack
#
FetchContent_Declare(
  dlpack
  GIT_REPOSITORY "https://github.com/dmlc/dlpack"
  GIT_TAG "v0.8"
  GIT_SHALLOW ON
)
# Option must be set off so WIN32 build does not break
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(BUILD_MOCK OFF)
FetchContent_MakeAvailable(dlpack)

#
# Boost
#
ExternalProject_Add(
  boostorg
  URL https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.gz
  URL_HASH SHA256=c0685b68dd44cc46574cce86c4e17c0f611b15e195be9848dfd0769a0a207628
  PREFIX "boost-src"
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E copy_directory
                    <SOURCE_DIR>/boost/ ${CMAKE_BINARY_DIR}/boost
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
)
set(boostorg_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/boost/")

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Using CUDA ${CUDA_VERSION}")
  set(CUDA_NVCC_FLAGS -std=c++${TRITON_MIN_CXX_STANDARD})
elseif()
  message(WARNING "TRITON_ENABLE_GPU is OFF, GPU Tensor support will be disabled")
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_NVTX})
  add_definitions(-DTRITON_ENABLE_NVTX=1)
endif() # TRITON_ENABLE_NVTX

find_package(ZLIB REQUIRED)

if(NOT WIN32)
  find_package(Threads REQUIRED)
endif()

include_directories(${CMAKE_BINARY_DIR})
configure_file(src/libtriton_python.ldscript libtriton_python.ldscript COPYONLY)

set(
  COMMON_SRCS
  src/correlation_id.cc
  src/correlation_id.h
  src/infer_response.cc
  src/infer_response.h
  src/infer_request.cc
  src/infer_request.h
  src/infer_trace.cc
  src/infer_trace.h
  src/message_queue.h
  src/ipc_message.cc
  src/ipc_message.h
  src/pb_string.cc
  src/pb_string.h
  src/pb_map.cc
  src/pb_map.h
  src/scoped_defer.cc
  src/scoped_defer.h
  src/pb_error.cc
  src/pb_error.h
  src/pb_log.cc
  src/pb_log.h
  src/pb_memory.cc
  src/pb_memory.h
  src/pb_tensor.cc
  src/pb_tensor.h
  src/pb_utils.cc
  src/pb_utils.h
  src/shm_manager.cc
  src/shm_manager.h
  src/pb_exception.h
  src/pb_preferred_memory.h
  src/metric.h
  src/metric.cc
  src/metric_family.h
  src/metric_family.cc
  src/gpu_buffers.cc
  src/gpu_buffers.h
  src/model_loader.h
  src/model_loader.cc
)

set(
    PYTHON_BACKEND_SRCS
    src/python_be.cc
    src/python_be.h
    src/pb_env.cc
    src/pb_env.h
    src/pb_metric_reporter.cc
    src/pb_metric_reporter.h
    src/memory_manager.cc
    src/memory_manager.h
    src/request_executor.cc
    src/request_executor.h
    src/stub_launcher.h
    src/stub_launcher.cc
    src/infer_payload.h
    src/infer_payload.cc
)

list(APPEND
  PYTHON_BACKEND_SRCS
  ${COMMON_SRCS}
)

add_library(
  triton-python-backend SHARED
  ${PYTHON_BACKEND_SRCS}
)

set(
  PYTHON_BACKEND_STUB_SRCS
  src/pb_stub_utils.h
  src/pb_stub_utils.cc
  src/response_sender.cc
  src/response_sender.h
  src/pb_stub.h
  src/pb_stub.cc
  src/pb_stub_log.h
  src/pb_stub_log.cc
  src/pb_response_iterator.h
  src/pb_response_iterator.cc
  src/pb_cancel.cc
  src/pb_cancel.h
  src/pb_bls_cancel.cc
  src/pb_bls_cancel.h
)

list(APPEND
  PYTHON_BACKEND_STUB_SRCS
  ${COMMON_SRCS}
)

add_executable(
  triton-python-backend-stub
  ${PYTHON_BACKEND_STUB_SRCS}
)

add_dependencies(triton-python-backend boostorg)
add_dependencies(triton-python-backend-stub boostorg)

set_property(TARGET triton-python-backend-stub PROPERTY OUTPUT_NAME triton_python_backend_stub)

add_library(
  TritonPythonBackend::triton-python-backend ALIAS triton-python-backend
)

target_compile_features(triton-python-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-python-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Wno-enum-constexpr-conversion -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor>
)

target_compile_features(triton-python-backend-stub PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-python-backend-stub PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Wno-unused-private-field -Wno-pessimizing-move -Wno-deprecated-declarations -Wno-unused-but-set-variable -Wno-unused-lambda-capture -Werror -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor>
)
target_compile_definitions(triton-python-backend-stub PRIVATE TRITON_PB_STUB)

# RHEL assets are not released in a container environment nor do the current
# Python lib versions in the manylinux base container match those currently
# available for RHEL8 package managers. Therefore, we package the correct
# python libs in the backend folder and adjust the stub executable to look
# in its own folder at runtime.
if(RHEL_BUILD)
  set_target_properties(
    triton-python-backend-stub
    PROPERTIES
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH "$\{ORIGIN\}"
  )
endif(RHEL_BUILD)


# Platform-specific linking
if(WIN32)
  target_link_libraries(
    triton-python-backend
    PRIVATE
      dlpack
      triton-backend-utils          # from repo-backend
      -lrt                          # shared memory
      triton-core-serverstub        # from repo-core
      ZLIB::ZLIB
      -larchive
  )

  target_link_libraries(
    triton-python-backend-stub
    PRIVATE
    dlpack
    triton-backend-utils           # from repo-backend
    pybind11::embed
    -lrt                           # shared memory
    -larchive                      # libarchive
  )
elseif(APPLE)
  # Find libarchive on macOS (usually installed via Homebrew)
  # Since libarchive is keg-only, we need to help CMake find it
  set(LibArchive_ROOT "/opt/homebrew/opt/libarchive")
  find_package(LibArchive REQUIRED)
  
  target_link_libraries(
    triton-python-backend
    PRIVATE
      dlpack
      Threads::Threads
      triton-backend-utils          # from repo-backend
      ${CMAKE_DL_LIBS}              # dlopen and dlclose
      triton-core-serverstub        # from repo-core
      ZLIB::ZLIB
      ${LibArchive_LIBRARIES}
  )

  target_link_libraries(
    triton-python-backend-stub
    PRIVATE
    dlpack
    Threads::Threads
    triton-backend-utils           # from repo-backend
    ${CMAKE_DL_LIBS}               # dlopen and dlclose
    pybind11::embed
    ${LibArchive_LIBRARIES}
  )
  
  # Add include directories for libarchive
  target_include_directories(triton-python-backend PRIVATE ${LibArchive_INCLUDE_DIRS})
  target_include_directories(triton-python-backend-stub PRIVATE ${LibArchive_INCLUDE_DIRS})
else()
  target_link_libraries(
    triton-python-backend
    PRIVATE
      dlpack
      Threads::Threads
      triton-backend-utils          # from repo-backend
      ${CMAKE_DL_LIBS}              # dlopen and dlclose
      -lrt                          # shared memory
      triton-core-serverstub        # from repo-core
      ZLIB::ZLIB
      -larchive
  )

  target_link_libraries(
    triton-python-backend-stub
    PRIVATE
    dlpack
    Threads::Threads
    triton-backend-utils           # from repo-backend
    ${CMAKE_DL_LIBS}               # dlopen and dlclose
    pybind11::embed
    -lrt                           # shared memory
    -larchive                      # libarchive
  )
endif()

if(WIN32)
  set_target_properties(
    triton-python-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_python
  )
elseif(APPLE)
  set_target_properties(
    triton-python-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_python
  )
else()
  set_target_properties(
    triton-python-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_python
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_python.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_python.ldscript"
  )
endif()

add_subdirectory(./src/shm_monitor)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonPythonBackend)

install(
  TARGETS
    triton-python-backend
    triton-python-backend-stub
  EXPORT
    triton-python-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/python
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/python
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/python
)

install(
  EXPORT
    triton-python-backend-targets
  FILE
    TritonPythonBackendTargets.cmake
  NAMESPACE
    TritonPythonBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

install(
  FILES
    src/resources/triton_python_backend_utils.py
  DESTINATION
    ${CMAKE_INSTALL_PREFIX}/backends/python
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonPythonBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonPythonBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonPythonBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-python-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonPythonBackendTargets.cmake
  NAMESPACE TritonPythonBackend::
)

export(PACKAGE TritonPythonBackend)
