# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.18)

project(tritonserver LANGUAGES C CXX)

include(CMakeDependentOption)

set(TRITON_VERSION "0.0.0" CACHE STRING "The version of the Triton shared library" )

option(TRITON_ENABLE_LOGGING "Include logging support in server" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in server" ON)
option(TRITON_ENABLE_TRACING "Include tracing support in server" OFF)
option(TRITON_ENABLE_NVTX "Include NVTX support in server" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in server" ON)
option(TRITON_ENABLE_MALI_GPU "Enable Arm Mali GPU support in server" OFF)
option(TRITON_IGPU_BUILD "Enable options for iGPU compilation in sever" OFF)
set(TRITON_MIN_COMPUTE_CAPABILITY "6.0" CACHE STRING
    "The minimum CUDA compute capability supported by Triton" )
set(TRITON_EXTRA_LIB_PATHS "" CACHE PATH "Extra library paths for Triton Server build")

# Ensemble
option(TRITON_ENABLE_ENSEMBLE "Include ensemble support in server" OFF)

# Endpoints
option(TRITON_ENABLE_HTTP "Include HTTP API in server" ON)
option(TRITON_ENABLE_GRPC "Include GRPC API in server" ON)
option(TRITON_ENABLE_SAGEMAKER "Include AWS SageMaker API in server" OFF)
option(TRITON_ENABLE_VERTEX_AI "Include Vertex AI API in server" OFF)

# Metrics
option(TRITON_ENABLE_METRICS "Include metrics support in server" ON)
option(TRITON_ENABLE_METRICS_GPU "Include GPU metrics support in server" ON)
option(TRITON_ENABLE_METRICS_CPU "Include CPU metrics support in server" ON)

# Cloud storage
option(TRITON_ENABLE_GCS "Include GCS Filesystem support in server" OFF)
option(TRITON_ENABLE_S3 "Include S3 Filesystem support in server" OFF)
option(TRITON_ENABLE_AZURE_STORAGE "Include Azure Storage Filesystem support in server" OFF)

# Need to know if TensorRT is available when building unit tests
option(TRITON_ENABLE_TENSORRT "Include TensorRT backend in server" OFF)

# ASAN
option(TRITON_ENABLE_ASAN "Build with address sanitizer" OFF)

# Repo tags
set(TRITON_THIRD_PARTY_REPO_TAG "main" CACHE STRING
    "Tag for triton-inference-server/third_party repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

# Third-party location
set(TRITON_THIRD_PARTY_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/third-party" CACHE STRING "Location of third-party build")
set(TRITON_THIRD_PARTY_SRC_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/third-party-src" CACHE STRING "Location of third-party source")

if(TRITON_ENABLE_METRICS AND NOT TRITON_ENABLE_STATS)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS=ON requires TRITON_ENABLE_STATS=ON")
endif()

if(TRITON_ENABLE_TRACING AND NOT TRITON_ENABLE_STATS)
  message(FATAL_ERROR "TRITON_ENABLE_TRACING=ON requires TRITON_ENABLE_STATS=ON")
endif()

if (TRITON_ENABLE_METRICS_CPU AND NOT TRITON_ENABLE_METRICS)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS_CPU=ON requires TRITON_ENABLE_METRICS=ON")
endif()

if (TRITON_ENABLE_METRICS_GPU AND NOT TRITON_ENABLE_METRICS)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS_GPU=ON requires TRITON_ENABLE_METRICS=ON")
endif()

if (TRITON_ENABLE_METRICS_GPU AND NOT TRITON_ENABLE_GPU)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS_GPU=ON requires TRITON_ENABLE_GPU=ON")
endif()

if(TRITON_ENABLE_ASAN AND TRITON_ENABLE_GPU)
  message(FATAL_ERROR "TRITON_ENABLE_ASAN=ON requires TRITON_ENABLE_GPU=OFF")
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
)
FetchContent_Declare(
  repo-third-party
  GIT_REPOSITORY https://github.com/triton-inference-server/third_party.git
  GIT_TAG ${TRITON_THIRD_PARTY_REPO_TAG}
)

# Some libs are installed to ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib64 instead
# of ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib on Centos
set (LIB_DIR "lib")
# /etc/os-release does not exist on Windows
if(EXISTS "/etc/os-release")
  file(STRINGS /etc/os-release DISTRO REGEX "^NAME=")
  string(REGEX REPLACE "NAME=\"(.*)\"" "\\1" DISTRO "${DISTRO}")
  message(STATUS "Distro Name: ${DISTRO}")
  if(DISTRO MATCHES "CentOS.*")
    set (LIB_DIR "lib64")
  endif()
endif()

set(TRITON_CORE_HEADERS_ONLY OFF)

FetchContent_MakeAvailable(repo-third-party repo-core)

#
# Triton server executable and examples
#

# Need to use ExternalProject for our builds so that we can get the
# correct dependencies between Triton executable and the
# ExternalProject dependencies (found in the third_party repo)
include(ExternalProject)

# If CMAKE_TOOLCHAIN_FILE is set, propagate that hint path to the external
# projects.
set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "")
if (CMAKE_TOOLCHAIN_FILE)
  set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "-DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}")
endif()

# If VCPKG_TARGET_TRIPLET is set, propagate that hint path to the external
# projects.
set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "")
if (VCPKG_TARGET_TRIPLET)
  set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "-DVCPKG_TARGET_TRIPLET:STRING=${VCPKG_TARGET_TRIPLET}")
endif()

# If OPENSSL_ROOT_DIR is set, propagate that hint path to the external
# projects with OpenSSL dependency.
set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
if (OPENSSL_ROOT_DIR)
  set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
endif()

# Location where protobuf-config.cmake will be installed varies by
# platform
if (WIN32)
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/cmake")
else()
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/${LIB_DIR}/cmake/protobuf")
endif()

# Triton with Opentelemetry is not supported on Windows
# FIXME: add location for Windows, when support is added
# JIRA DLIS-4786
if (WIN32)
  set(_FINDPACKAGE_OPENTELEMETRY_CONFIG_DIR "")
else()
  set(_FINDPACKAGE_OPENTELEMETRY_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/opentelemetry-cpp/${LIB_DIR}/cmake/opentelemetry-cpp")
endif()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRITON_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)
else()
  set(TRITON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

set(TRITON_DEPENDS triton-core protobuf googletest re2)
if(${TRITON_ENABLE_GCS})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} google-cloud-cpp)
endif() # TRITON_ENABLE_GCS
if(${TRITON_ENABLE_S3})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} aws-sdk-cpp)
endif() # TRITON_ENABLE_S3
if(${TRITON_ENABLE_HTTP} OR ${TRITON_ENABLE_METRICS} OR ${TRITON_ENABLE_SAGEMAKER} OR ${TRITON_ENABLE_VERTEX_AI})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} libevent libevhtp)
endif() # TRITON_ENABLE_HTTP || TRITON_ENABLE_METRICS || TRITON_ENABLE_SAGEMAKER || TRITON_ENABLE_VERTEX_AI
if(${TRITON_ENABLE_GRPC})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} grpc)
endif() # TRITON_ENABLE_GRPC
if(NOT WIN32 AND ${TRITON_ENABLE_TRACING})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} opentelemetry-cpp)
endif() # TRITON_ENABLE_TRACING

ExternalProject_Add(triton-server
  PREFIX triton-server
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/triton-server"
  CMAKE_CACHE_ARGS
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DGTEST_ROOT:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/googletest
    -DgRPC_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/lib/cmake/grpc
    -Dc-ares_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/${LIB_DIR}/cmake/c-ares
    -Dre2_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/re2/${LIB_DIR}/cmake/re2
    -Dabsl_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/${LIB_DIR}/cmake/absl
    -DCURL_DIR:STRING=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl/${LIB_DIR}/cmake/CURL
    -Dnlohmann_json_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/nlohmann_json/${LIB_DIR}/cmake/nlohmann_json
    -DLibevent_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevent/lib/cmake/libevent
    -Dlibevhtp_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevhtp/lib/cmake/libevhtp
    -Dstorage_client_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/google-cloud-cpp/${LIB_DIR}/cmake/storage_client
    -Dgoogle_cloud_cpp_common_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/google-cloud-cpp/${LIB_DIR}/cmake/google_cloud_cpp_common
    -DCrc32c_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/crc32c/${LIB_DIR}/cmake/Crc32c
    -DAWSSDK_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/cmake/AWSSDK
    -Daws-cpp-sdk-core_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/cmake/aws-cpp-sdk-core
    -Daws-cpp-sdk-s3_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/cmake/aws-cpp-sdk-s3
    -Daws-c-event-stream_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-event-stream/cmake
    -Daws-c-common_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-common/cmake
    -Daws-checksums_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-checksums/cmake
    -Dopentelemetry-cpp_DIR:PATH=${_FINDPACKAGE_OPENTELEMETRY_CONFIG_DIR}
    -DTRITON_IGPU_BUILD:BOOL=${TRITON_IGPU_BUILD}
    -DTRITON_THIRD_PARTY_REPO_TAG:STRING=${TRITON_THIRD_PARTY_REPO_TAG}
    -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
    -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
    -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG}
    -DTRITON_EXTRA_LIB_PATHS:PATH=${TRITON_EXTRA_LIB_PATHS}
    -DTRITON_ENABLE_ASAN:BOOL=${TRITON_ENABLE_ASAN}
    -DTRITON_ENABLE_NVTX:BOOL=${TRITON_ENABLE_NVTX}
    -DTRITON_ENABLE_TRACING:BOOL=${TRITON_ENABLE_TRACING}
    -DTRITON_ENABLE_LOGGING:BOOL=${TRITON_ENABLE_LOGGING}
    -DTRITON_ENABLE_STATS:BOOL=${TRITON_ENABLE_STATS}
    -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
    -DTRITON_ENABLE_MALI_GPU:BOOL=${TRITON_ENABLE_MALI_GPU}
    -DTRITON_ENABLE_HTTP:BOOL=${TRITON_ENABLE_HTTP}
    -DTRITON_ENABLE_SAGEMAKER:BOOL=${TRITON_ENABLE_SAGEMAKER}
    -DTRITON_ENABLE_VERTEX_AI:BOOL=${TRITON_ENABLE_VERTEX_AI}
    -DTRITON_ENABLE_GRPC:BOOL=${TRITON_ENABLE_GRPC}
    -DTRITON_MIN_COMPUTE_CAPABILITY:STRING=${TRITON_MIN_COMPUTE_CAPABILITY}
    -DTRITON_ENABLE_METRICS:BOOL=${TRITON_ENABLE_METRICS}
    -DTRITON_ENABLE_METRICS_GPU:BOOL=${TRITON_ENABLE_METRICS_GPU}
    -DTRITON_ENABLE_METRICS_CPU:BOOL=${TRITON_ENABLE_METRICS_CPU}
    -DTRITON_ENABLE_GCS:BOOL=${TRITON_ENABLE_GCS}
    -DTRITON_ENABLE_AZURE_STORAGE:BOOL=${TRITON_ENABLE_AZURE_STORAGE}
    -DTRITON_ENABLE_S3:BOOL=${TRITON_ENABLE_S3}
    -DTRITON_ENABLE_TENSORRT:BOOL=${TRITON_ENABLE_TENSORRT}
    -DTRITON_ENABLE_ENSEMBLE:BOOL=${TRITON_ENABLE_ENSEMBLE}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}
    -DTRITON_VERSION:STRING=${TRITON_VERSION}
  DEPENDS ${TRITON_DEPENDS}
)
