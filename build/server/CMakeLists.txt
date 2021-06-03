# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

cmake_minimum_required (VERSION 3.18)
project (server)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message("Using MSVC as compiler, default target on Windows 10. "
		  "If the target system is not Windows 10, please update _WIN32_WINNT "
		  "to corresponding value.")
  set(CMAKE_CXX_FLAGS "/W4 /D_WIN32_WINNT=0x0A00 /EHsc")
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
else()
  # Disable Triton kernel if on "aarch64" platform
  execute_process(COMMAND uname -p OUTPUT_VARIABLE ARCH)
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -Werror")
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
)

set(TRITON_COMMON_ENABLE_PROTOBUF ON)
if(TRITON_ENABLE_GRPC)
  set(TRITON_COMMON_ENABLE_GRPC ON)
endif() # TRITON_ENABLE_GRPC

FetchContent_MakeAvailable(repo-common repo-core repo-backend)


add_definitions(-DTRITON_VERSION="${TRITON_VERSION}")

if(${TRITON_ENABLE_ASAN})
  set(CMAKE_BUILD_TYPE Debug)
  add_definitions(-DTRITON_ENABLE_ASAN=1)

  set(_ASAN_FLAGS "-static-libstdc++ -static-libasan -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${_ASAN_FLAGS}")
  set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} ${_ASAN_FLAGS}")
endif() # TRITON_ENABLE_ASAN

if(${TRITON_ENABLE_NVTX})
  add_definitions(-DTRITON_ENABLE_NVTX=1)
endif() # TRITON_ENABLE_NVTX

if(${TRITON_ENABLE_TRACING})
  add_definitions(-DTRITON_ENABLE_TRACING=1)
endif() # TRITON_ENABLE_TRACING

if(${TRITON_ENABLE_LOGGING})
  add_definitions(-DTRITON_ENABLE_LOGGING=1)
endif() # TRITON_ENABLE_LOGGING
if(${TRITON_ENABLE_STATS})
  add_definitions(-DTRITON_ENABLE_STATS=1)
endif() # TRITON_ENABLE_STATS

if(${TRITON_ENABLE_GPU})
  add_definitions(-DTRITON_ENABLE_GPU=1)
  add_definitions(-DTRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY})
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_HTTP})
  add_definitions(-DTRITON_ENABLE_HTTP=1)
endif() # TRITON_ENABLE_HTTP

if(${TRITON_ENABLE_SAGEMAKER})
  add_definitions(-DTRITON_ENABLE_SAGEMAKER=1)
endif() # TRITON_ENABLE_SAGEMAKER

if(${TRITON_ENABLE_GRPC})
  add_definitions(-DTRITON_ENABLE_GRPC=1)
endif() # TRITON_ENABLE_GRPC

if(${TRITON_ENABLE_METRICS})
  add_definitions(-DTRITON_ENABLE_METRICS=1)
endif() # TRITON_ENABLE_METRICS

if(${TRITON_ENABLE_METRICS_GPU})
  add_definitions(-DTRITON_ENABLE_METRICS_GPU=1)
endif() # TRITON_ENABLE_METRICS_GPU

if(${TRITON_ENABLE_GCS})
  add_definitions(-DTRITON_ENABLE_GCS=1)
endif() # TRITON_ENABLE_GCS

if(${TRITON_ENABLE_AZURE_STORAGE})
  add_definitions(-DTRITON_ENABLE_AZURE_STORAGE=1)
endif() # TRITON_ENABLE_AZURE_STORAGE

if(${TRITON_ENABLE_S3})
  add_definitions(-DTRITON_ENABLE_S3=1)
endif() # TRITON_ENABLE_S3

if(${TRITON_ENABLE_TENSORFLOW})
  add_definitions(-DTRITON_ENABLE_TENSORFLOW=1)
endif() # TRITON_ENABLE_TENSORFLOW

if(${TRITON_ENABLE_PYTHON})
  add_definitions(-DTRITON_ENABLE_PYTHON=1)
endif() # TRITON_ENABLE_PYTHON

if(${TRITON_ENABLE_TENSORRT})
  add_definitions(-DTRITON_ENABLE_TENSORRT=1)
  if (NOT "${TRITON_TENSORRT_INCLUDE_PATHS}" STREQUAL "")
    include_directories("${TRITON_TENSORRT_INCLUDE_PATHS}")
  endif()
endif() # TRITON_ENABLE_TENSORRT

if(${TRITON_ENABLE_ONNXRUNTIME})
  add_definitions(-DTRITON_ENABLE_ONNXRUNTIME=1)
endif() # TRITON_ENABLE_ONNXRUNTIME

if(${TRITON_ENABLE_ONNXRUNTIME_TENSORRT})
  add_definitions(-DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=1)
endif() # TRITON_ENABLE_ONNXRUNTIME_TENSORRT

if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
  add_definitions(-DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=1)
endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

if(${TRITON_ENABLE_PYTORCH})
  add_definitions(-DTRITON_ENABLE_PYTORCH=1)
endif() # TRITON_ENABLE_PYTORCH

if(${TRITON_ENABLE_ENSEMBLE})
  add_definitions(-DTRITON_ENABLE_ENSEMBLE=1)
endif() # TRITON_ENABLE_ENSEMBLE

include_directories("${PROJECT_SOURCE_DIR}/../..")
include_directories("${PROJECT_BINARY_DIR}")

set(TRITON_EXTRA_LDFLAGS "")
FOREACH(p ${TRITON_EXTRA_LIB_PATHS})
  set(TRITON_EXTRA_LDFLAGS ${TRITON_EXTRA_LDFLAGS} "-L${p}")
ENDFOREACH(p)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDA REQUIRED)
  message(STATUS "Using CUDA ${CUDA_VERSION}")
  set(CUDA_NVCC_FLAGS -std=c++11)

  if(CUDA_VERSION VERSION_GREATER "10.1" OR CUDA_VERSION VERSION_EQUAL "10.1")
    add_definitions(-DTRITON_ENABLE_CUDA_GRAPH=1)
  else()
    message(WARNING "CUDA ${CUDA_VERSION} does not support CUDA graphs.")
  endif()
endif() # TRITON_ENABLE_GPU

#
# Boost
#
find_package(Boost REQUIRED)
message(STATUS "Using Boost ${Boost_VERSION}")
include_directories(${Boost_INCLUDE_DIRS})

#
# libevent
#
if(${TRITON_ENABLE_HTTP} OR ${TRITON_ENABLE_METRICS} OR ${TRITON_ENABLE_SAGEMAKER})
  find_package(Libevent CONFIG REQUIRED)
  message(STATUS "Using libevent ${Libevent_VERSION}")
  include_directories(${LIBEVENT_INCLUDE_DIRS})
endif()

#
# Protobuf
#
set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
include_directories(${Protobuf_INCLUDE_DIRS})

#
# GRPC
#
if(${TRITON_ENABLE_GRPC})
  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Using gRPC ${gRPC_VERSION}")
  include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

#
# Prometheus
#
if(${TRITON_ENABLE_METRICS})
  find_package(prometheus-cpp CONFIG REQUIRED)
  message(STATUS "Using prometheus-cpp ${prometheus-cpp_VERSION}")
endif() # TRITON_ENABLE_METRICS


add_subdirectory(../../src/core src/core)
add_subdirectory(../../src/backends/backend src/backends/backend)
add_subdirectory(../../src/backends/ensemble src/backends/ensemble)
add_subdirectory(../../src/backends/onnx src/backends/onnx)
add_subdirectory(../../src/backends/pytorch src/backends/pytorch)
add_subdirectory(../../src/backends/tensorflow src/backends/tensorflow)
add_subdirectory(../../src/backends/tensorrt src/backends/tensorrt)
add_subdirectory(../../src/servers src/servers)
