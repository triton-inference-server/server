# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#
# caffe2plan
#
if(${TRITON_ENABLE_TENSORRT})
add_executable(caffe2plan caffe2plan.cc)
target_include_directories(caffe2plan PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(
  caffe2plan
  PRIVATE -lnvinfer
  PRIVATE -lnvcaffe_parser
  PRIVATE ${CUDA_LIBRARIES}
)
install(
  TARGETS caffe2plan
  RUNTIME DESTINATION bin
)
endif() # TRITON_ENABLE_TENSORRT

#
# CudaMemoryManger
#
set(
  CUDA_MEMORY_MANAGER_SRCS
  ../core/cuda_memory_manager.cc
  ../core/status.cc
  ../core/cuda_utils.cc
)

set(
  CUDA_MEMORY_MANAGER_HDRS
  ../core/cuda_memory_manager.h
  ../core/status.h
  ../core/cuda_utils.h
)

#
# PinnedMemoryManger
#
set(
  PINNED_MEMORY_MANAGER_SRCS
  ../core/pinned_memory_manager.cc
  ../core/status.cc
  ../core/cuda_utils.cc
)

set(
  PINNED_MEMORY_MANAGER_HDRS
  ../core/pinned_memory_manager.h
  ../core/status.h
  ../core/cuda_utils.h
)

#
# Memory
#
set(
  MEMORY_SRCS
  ../core/memory.cc
)

set(
  MEMORY_HDRS
  ../core/memory.h
  ../core/constants.h
)

set(
  MEMORY_TEST_SRCS
  memory_test.cc
  ${MEMORY_SRCS}
  ${CUDA_MEMORY_MANAGER_SRCS}
  ${PINNED_MEMORY_MANAGER_SRCS}
)

set(
  MEMORY_TEST_HDRS
  ${MEMORY_HDRS}
  ${CUDA_MEMORY_MANAGER_HDRS}
  ${PINNED_MEMORY_MANAGER_HDRS}
)

find_package(GTest REQUIRED)
add_executable(
  memory_test
  ${MEMORY_TEST_SRCS}
  ${MEMORY_TEST_HDRS}
  $<TARGET_OBJECTS:proto-library>
)
set_target_properties(
  memory_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)
target_include_directories(
  memory_test
  PRIVATE ${GTEST_INCLUDE_DIR}
  PRIVATE ${CUDA_INCLUDE_DIRS}
  PRIVATE ${CNMEM_PATH}/include
)
target_link_libraries(
  memory_test
  PRIVATE triton-core-serverapi  # from repo-core
  PRIVATE ${GTEST_LIBRARY}
  PRIVATE ${GTEST_MAIN_LIBRARY}
  PRIVATE ${CUDA_LIBRARIES}
  PRIVATE -lpthread
  PRIVATE protobuf::libprotobuf
  PRIVATE -L${CNMEM_PATH}/lib
  PRIVATE -lcnmem
)
install(
  TARGETS memory_test
  RUNTIME DESTINATION bin
)

#
# RateLimiter
#
protobuf_generate_cpp(MODEL_CONFIG_PROTO_SRC MODEL_CONFIG_PROTO_HDR triton/core/model_config.proto)

set(
  RATE_LIMITER_SRCS
  ../core/rate_limiter.cc
  ../core/status.cc
)

set(
  RATE_LIMITER_HDRS
  ../core/rate_limiter.h
  ${MODEL_CONFIG_PROTO_HDR}
)

set(
  RATE_LIMITER_TEST_SRCS
  rate_limiter_test.cc
  ${RATE_LIMITER_SRCS}
)

set(
  RATE_LIMITER_TEST_HDRS
  ${RATE_LIMITER_HDRS}
)

add_executable(
  rate_limiter_test
  ${RATE_LIMITER_TEST_SRCS}
  ${RATE_LIMITER_TEST_HDRS}
  $<TARGET_OBJECTS:proto-library>
)
set_target_properties(
  rate_limiter_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)
target_include_directories(
  rate_limiter_test
  PRIVATE ${GTEST_INCLUDE_DIR}
)
if(${TRITON_ENABLE_GPU})
  target_include_directories(rate_limiter_test PRIVATE ${CUDA_INCLUDE_DIRS})
endif() # TRITON_ENABLE_GPU
target_link_libraries(
  rate_limiter_test
  PRIVATE triton-core-serverapi  # from repo-core
  PRIVATE ${GTEST_LIBRARY}
  PRIVATE ${GTEST_MAIN_LIBRARY}
  PRIVATE protobuf::libprotobuf
)
install(
  TARGETS rate_limiter_test
  RUNTIME DESTINATION bin
)

add_subdirectory(custom/sdk custom/sdk)
add_subdirectory(custom/addsub custom/addsub)
add_subdirectory(custom/identity custom/identity)
add_subdirectory(custom/image_preprocess custom/image_preprocess)
add_subdirectory(custom/param custom/param)
add_subdirectory(custom/sequence custom/sequence)
add_subdirectory(custom/dyna_sequence custom/dyna_sequence)
