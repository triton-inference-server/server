# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

cmake_minimum_required (VERSION 3.5)
project (trtis)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(${TRTIS_ENABLE_ASAN})
  set(CMAKE_BUILD_TYPE Debug)
  add_definitions(-DTRTIS_ENABLE_ASAN=1)

  set(_ASAN_FLAGS "-static-libstdc++ -static-libasan -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${_ASAN_FLAGS}")
  set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} ${_ASAN_FLAGS}")
endif() # TRTIS_ENABLE_ASAN

if(${TRTIS_ENABLE_GPU})
  add_definitions(-DTRTIS_ENABLE_GPU=1)
  add_definitions(-DTRTIS_MIN_COMPUTE_CAPABILITY=${TRTIS_MIN_COMPUTE_CAPABILITY})
endif() # TRTIS_ENABLE_GPU

if(${TRTIS_ENABLE_HTTP})
  add_definitions(-DTRTIS_ENABLE_HTTP=1)
endif() # TRTIS_ENABLE_HTTP

if(${TRTIS_ENABLE_GRPC})
  add_definitions(-DTRTIS_ENABLE_GRPC=1)
endif() # TRTIS_ENABLE_GRPC

if(${TRTIS_ENABLE_METRICS})
  add_definitions(-DTRTIS_ENABLE_METRICS=1)
endif() # TRTIS_ENABLE_METRICS

if(${TRTIS_ENABLE_GCS})
  add_definitions(-DTRTIS_ENABLE_GCS=1)
endif() # TRTIS_ENABLE_GCS

if(${TRTIS_ENABLE_S3})
  add_definitions(-DTRTIS_ENABLE_S3=1)
endif() # TRTIS_ENABLE_S3

if(${TRTIS_ENABLE_TENSORFLOW})
  add_definitions(-DTRTIS_ENABLE_TENSORFLOW=1)
endif() # TRTIS_ENABLE_TENSORFLOW

if(${TRTIS_ENABLE_TENSORRT})
  add_definitions(-DTRTIS_ENABLE_TENSORRT=1)
endif() # TRTIS_ENABLE_TENSORRT

if(${TRTIS_ENABLE_CAFFE2})
  add_definitions(-DTRTIS_ENABLE_CAFFE2=1)
endif() # TRTIS_ENABLE_CAFFE2

if(${TRTIS_ENABLE_ONNXRUNTIME})
  add_definitions(-DTRTIS_ENABLE_ONNXRUNTIME=1)
endif() # TRTIS_ENABLE_ONNXRUNTIME

if(${TRTIS_ENABLE_PYTORCH})
  add_definitions(-DTRTIS_ENABLE_PYTORCH=1)
endif() # TRTIS_ENABLE_PYTORCH

if(${TRTIS_ENABLE_CUSTOM})
  add_definitions(-DTRTIS_ENABLE_CUSTOM=1)
endif() # TRTIS_ENABLE_CUSTOM

include_directories("${PROJECT_SOURCE_DIR}/../..")
include_directories("${PROJECT_BINARY_DIR}")

set(TRTIS_EXTRA_LDFLAGS "")
FOREACH(p ${TRTIS_EXTRA_LIB_PATHS})
  set(TRTIS_EXTRA_LDFLAGS ${TRTIS_EXTRA_LDFLAGS} "-L${p}")
ENDFOREACH(p)

#
# CUDA
#
if(${TRTIS_ENABLE_GPU})
  find_package(CUDA REQUIRED)
  message(STATUS "Using CUDA ${CUDA_VERSION}")
  set(CUDA_NVCC_FLAGS -std=c++11)
endif() # TRTIS_ENABLE_GPU

#
# libevent
#
find_package(Libevent CONFIG REQUIRED)
message(STATUS "Using libevent ${Libevent_VERSION}")
include_directories(${LIBEVENT_INCLUDE_DIRS})

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
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)

add_subdirectory(../../src/core src/core)
add_subdirectory(../../src/backends/caffe2 src/backends/caffe2)
add_subdirectory(../../src/backends/custom src/backends/custom)
add_subdirectory(../../src/backends/ensemble src/backends/ensemble)
add_subdirectory(../../src/backends/onnx src/backends/onnx)
add_subdirectory(../../src/backends/pytorch src/backends/pytorch)
add_subdirectory(../../src/backends/tensorflow src/backends/tensorflow)
add_subdirectory(../../src/backends/tensorrt src/backends/tensorrt)
add_subdirectory(../../src/servers src/servers)
