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
project (triton-inference-server)

include(CMakeDependentOption)
include(ExternalProject)
include(GNUInstallDirs)

# Backends
option(TRITON_ENABLE_TENSORRT "Include TensorRT backend in server" OFF)
option(TRITON_ENABLE_TENSORFLOW "Include TensorFlow backend in server" OFF)
option(TRITON_ENABLE_ONNXRUNTIME "Include ONNXRuntime backend in server" OFF)
option(TRITON_ENABLE_ONNXRUNTIME_TENSORRT
  "Enable TensorRT execution provider for ONNXRuntime backend in server" OFF)
option(TRITON_ENABLE_ONNXRUNTIME_OPENVINO
  "Enable OpenVINO execution provider for ONNXRuntime backend in server" OFF)
option(TRITON_ENABLE_PYTORCH "Include PyTorch backend in server" OFF)
option(TRITON_ENABLE_PYTHON "Include Python backend in server" OFF)
option(TRITON_ENABLE_ENSEMBLE "Include ensemble support in server" OFF)

# Endpoints
option(TRITON_ENABLE_HTTP "Include HTTP API in server" ON)
option(TRITON_ENABLE_GRPC "Include GRPC API in server" ON)
option(TRITON_ENABLE_SAGEMAKER "Include AWS SageMaker API in server" OFF)
option(TRITON_ENABLE_METRICS "Include metrics support in server" ON)
option(TRITON_ENABLE_METRICS_GPU "Include GPU metrics support in server" ON)

# Cloud storage
option(TRITON_ENABLE_GCS "Include GCS Filesystem support in server" OFF)
option(TRITON_ENABLE_S3 "Include S3 Filesystem support in server" OFF)
option(TRITON_ENABLE_AZURE_STORAGE "Include Azure Storage Filesystem support in server" OFF)

# Multiple paths may be specified by separating them with semicolon
set(TRITON_ONNXRUNTIME_INCLUDE_PATHS "" CACHE PATH "Paths to ONNXRuntime includes")
set(TRITON_PYTORCH_INCLUDE_PATHS "" CACHE PATH "Paths to PyTorch includes")
# Used for the case where TensorRT is not installed under default search path,
# i.e. Windows build, this flag should be able to remove once TensorRT backend
# is ported to use backend API
set(TRITON_TENSORRT_INCLUDE_PATHS "" CACHE PATH "Paths to TensorRT includes")
set(TRITON_EXTRA_LIB_PATHS "" CACHE PATH "Extra library paths for Triton Server build")

option(TRITON_ENABLE_LOGGING "Include logging support in server" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in server" ON)
option(TRITON_ENABLE_TRACING "Include tracing support in server" OFF)
option(TRITON_ENABLE_NVTX "Include NVTX support in server" OFF)
option(TRITON_ENABLE_ASAN "Build with address sanitizer" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in server" ON)
option(TRITON_ENABLE_MALI_GPU "Enable Arm Mali GPU support in server" OFF)
set(TRITON_MIN_COMPUTE_CAPABILITY "6.0" CACHE STRING
    "The minimum CUDA compute capability supported by Triton" )

# Repo tags
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_THIRD_PARTY_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/third_party repo")

# Version
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../TRITON_VERSION" TRITON_VERSION)

if(TRITON_ENABLE_METRICS AND NOT TRITON_ENABLE_STATS)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS=ON requires TRITON_ENABLE_STATS=ON")
endif()

if(TRITON_ENABLE_TRACING AND NOT TRITON_ENABLE_STATS)
  message(FATAL_ERROR "TRITON_ENABLE_TRACING=ON requires TRITON_ENABLE_STATS=ON")
endif()

if(TRITON_ENABLE_TENSORRT AND NOT TRITON_ENABLE_GPU)
  message(FATAL_ERROR "TRITON_ENABLE_TENSORRT=ON requires TRITON_ENABLE_GPU=ON")
endif()

if (TRITON_ENABLE_METRICS_GPU AND NOT TRITON_ENABLE_METRICS)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS_GPU=ON requires TRITON_ENABLE_METRICS=ON")
endif()

if (TRITON_ENABLE_METRICS_GPU AND NOT TRITON_ENABLE_GPU)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS_GPU=ON requires TRITON_ENABLE_GPU=ON")
endif()

if(TRITON_ENABLE_ONNXRUNTIME_TENSORRT AND NOT TRITON_ENABLE_ONNXRUNTIME)
  message(FATAL_ERROR "TRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON requires TRITON_ENABLE_ONNXRUNTIME=ON")
endif()
if(TRITON_ENABLE_ONNXRUNTIME_TENSORRT AND NOT TRITON_ENABLE_TENSORRT)
  message(FATAL_ERROR "TRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON requires TRITON_ENABLE_TENSORRT=ON")
endif()

if(TRITON_ENABLE_ONNXRUNTIME_OPENVINO AND NOT TRITON_ENABLE_ONNXRUNTIME)
  message(FATAL_ERROR "TRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON requires TRITON_ENABLE_ONNXRUNTIME=ON")
endif()

if(TRITON_ENABLE_ASAN AND TRITON_ENABLE_GPU)
  message(FATAL_ERROR "TRITON_ENABLE_ASAN=ON requires TRITON_ENABLE_GPU=OFF")
endif()

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

#
# Get the protobuf and grpc source used for the GRPC endpoint. We must
# use v1.25.0 because later GRPC has significant performance
# regressions (e.g. resnet50 bs128).
#
ExternalProject_Add(grpc-repo
  PREFIX grpc-repo
  GIT_REPOSITORY "https://github.com/grpc/grpc.git"
  GIT_TAG "v1.25.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)

#
# The GCS build requires a abseil newer than the one found in grpc
# v1.25.0 so grab a newer grpc just for abseil.
#
ExternalProject_Add(grpc-repo-new
  PREFIX grpc-repo-new
  GIT_REPOSITORY "https://github.com/grpc/grpc.git"
  GIT_TAG "v1.29.1"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo-new/src/grpc"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)

#
# Get and build nlohmann/json
#
ExternalProject_Add(nlohmann-json
  PREFIX nlohmann-json
  GIT_REPOSITORY "https://github.com/nlohmann/json.git"
  GIT_TAG "v3.9.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/json"
  CMAKE_CACHE_ARGS
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/nlohmann_json
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DBUILD_SHARED_LIBS:STRING=yes
    -DBUILD_TESTING:BOOL=OFF
    -H. -Bcmake-out
)

#
# Build protobuf project from grpc-repo
#
# Use -DCMAKE_BUILD_TYPE:STRING=RELEASE to workaround the protobuf build issue
# described in https://stackoverflow.com/a/60218066/2091555
ExternalProject_Add(protobuf
  PREFIX protobuf
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc/third_party/protobuf/cmake"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -Dprotobuf_BUILD_TESTS:BOOL=OFF
    -Dprotobuf_WITH_ZLIB:BOOL=OFF
    -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
    -DCMAKE_BUILD_TYPE:STRING=RELEASE
    -DBUILD_SHARED_LIBS:STRING=no
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/protobuf
  DEPENDS grpc-repo
)

#
# Build absl (abseil) project from grpc-repo-new (see comment above).
#
ExternalProject_Add(absl
  PREFIX absl
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo-new/src/grpc/third_party/abseil-cpp"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/absl
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DBUILD_TESTING:BOOL=OFF
  DEPENDS grpc-repo-new
)

#
# Build googletest project from grpc-repo
#
ExternalProject_Add(googletest
  PREFIX protobuf
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc/third_party/googletest"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/googletest
  DEPENDS grpc-repo
)

# Location where protobuf-config.cmake will be installed varies by
# platform
if (WIN32)
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/cmake")
else()
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/${CMAKE_INSTALL_LIBDIR}/cmake/protobuf")
endif()

#
# Build c-area project from grpc-repo
#
ExternalProject_Add(c-ares
  PREFIX c-ares
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc/third_party/cares/cares"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCARES_SHARED:BOOL=OFF
    -DCARES_STATIC:BOOL=ON
    -DCARES_STATIC_PIC:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares
  DEPENDS grpc-repo
)

#
# Build GRPC
#
ExternalProject_Add(grpc
  PREFIX grpc
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    -DgRPC_INSTALL:BOOL=ON
    -DgRPC_BUILD_TESTS:BOOL=OFF
    -DgRPC_PROTOBUF_PROVIDER:STRING=package
    -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
    -DgRPC_ZLIB_PROVIDER:STRING=package
    -DgRPC_CARES_PROVIDER:STRING=package
    -DgRPC_SSL_PROVIDER:STRING=package
    -Dc-ares_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/grpc
  DEPENDS grpc-repo c-ares protobuf
)

#
# Build libcurl
#
ExternalProject_Add(curl
  PREFIX curl
  GIT_REPOSITORY "https://github.com/curl/curl.git"
  GIT_TAG "curl-7_66_0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/curl/src/curl"
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DBUILD_CURL_EXE:BOOL=OFF
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DCURL_STATICLIB:BOOL=ON
    -DHTTP_ONLY:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/curl/install
)

#
# Build libevent
#
ExternalProject_Add(libevent
  PREFIX libevent
  GIT_REPOSITORY "https://github.com/libevent/libevent.git"
  GIT_TAG "release-2.1.8-stable"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/libevent/src/libevent"
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevent/install
)

#
# Get patched third party repo
#
ExternalProject_Add(third-party-repo
  PREFIX third-party-repo
  GIT_REPOSITORY "https://github.com/triton-inference-server/third_party.git"
  GIT_TAG ${TRITON_THIRD_PARTY_REPO_TAG}
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third-party-repo/src/third_party"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)

#
# Build patched libevhtp
#
if (WIN32)
# FIXME evhtp can't be built successfully in native Windows environment
# CMAKE_PREFIX_PATH is needed to find libevent
ExternalProject_Add(libevhtp
  PREFIX libevhtp
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third-party-repo/src/third_party/libevhtp/libevhtp"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DEVHTP_DISABLE_REGEX:BOOL=ON
    -DEVHTP_DISABLE_SSL:BOOL=ON
    -DEVHTP_TRITON_ENABLE_HTTP_CONTIGUOUS:BOOL=ON
    -DEVHTP_TRITON_ENABLE_TRACING:BOOL=${TRITON_ENABLE_TRACING}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevent/install
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevhtp/install
  DEPENDS libevent third-party-repo
)
else()
ExternalProject_Add(libevhtp
  PREFIX libevhtp
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third-party-repo/src/third_party/libevhtp/libevhtp"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DEVHTP_DISABLE_REGEX:BOOL=ON
    -DEVHTP_DISABLE_SSL:BOOL=ON
    -DEVHTP_TRITON_ENABLE_HTTP_CONTIGUOUS:BOOL=ON
    -DEVHTP_TRITON_ENABLE_TRACING:BOOL=${TRITON_ENABLE_TRACING}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DLibevent_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevent/install/lib/cmake/libevent
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevhtp/install
  DEPENDS libevent third-party-repo
)
endif()

#
# Build Prometheus C++ library
#
ExternalProject_Add(prometheus-cpp
  PREFIX prometheus-cpp
  URL "https://github.com/jupp0r/prometheus-cpp/archive/v0.7.0.tar.gz"
  URL_HASH SHA256=93907d937fa7eab9605cba786123d3eba4e87e3dca8ecec93ff9eae4eef8de5a
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/prometheus-cpp/src/prometheus-cpp"
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DENABLE_PUSH:BOOL=OFF
    -DENABLE_PULL:BOOL=OFF
    -DENABLE_TESTING:BOOL=OFF
    -DUSE_THIRDPARTY_LIBRARIES:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/prometheus-cpp/install
)

#
# Build crc32c
#
ExternalProject_Add(crc32c
  PREFIX crc32c
  GIT_REPOSITORY "https://github.com/google/crc32c.git"
  GIT_TAG "1.1.1"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/crc32c/src/crc32c"
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCRC32C_BUILD_TESTS:BOOL=OFF
    -DBUILD_SHARED_LIBS:STRING=no
    -DCRC32C_BUILD_BENCHMARKS:BOOL=OFF
    -DCRC32C_USE_GLOG:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/crc32c
)

# This is necessary because google-cloud-cpp
# does not pass protobuf_DIR and gRPC_DIR etc to its
# external projects but expects them in CMAKE_PREFIX_PATH.
#
# absl must preceed grpc to ensure that the absl from the newer grpc
# repo is used and not the absl from the older grpc.
set(GCS_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
   ${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
   ${CMAKE_CURRENT_BINARY_DIR}/absl/lib/cmake/absl
   ${CMAKE_CURRENT_BINARY_DIR}/crc32c/lib/cmake/Crc32c
   ${CMAKE_CURRENT_BINARY_DIR}/grpc/lib/cmake/grpc
   ${CMAKE_CURRENT_BINARY_DIR}/nlohmann_json/lib/cmake/nlohmann_json
   ${_FINDPACKAGE_PROTOBUF_CONFIG_DIR})

#
# Build google-cloud-cpp
#
ExternalProject_Add(google-cloud-cpp
  PREFIX google-cloud-cpp
  GIT_REPOSITORY "https://github.com/googleapis/google-cloud-cpp.git"
  GIT_TAG "v1.22.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/google-cloud-cpp/src/google-cloud-cpp"
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DGOOGLE_CLOUD_CPP_DEPENDENCY_PROVIDER:STRING=package
    -DBUILD_TESTING:BOOL=OFF
    -DCMAKE_PREFIX_PATH:PATH=${GCS_CMAKE_PREFIX_PATH}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/google-cloud-cpp/install
    -Dnlohmann_json_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/nlohmann_json/lib/cmake/nlohmann_json
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -DCrc32c_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/crc32c/lib/cmake/Crc32c
  DEPENDS grpc c-ares protobuf crc32c absl nlohmann-json
)

#
# Build azure-storage-cpplite
#
ExternalProject_Add(azure-storage-cpplite
  PREFIX azure-storage-cpplite
  GIT_REPOSITORY "https://github.com/Azure/azure-storage-cpplite.git"
  GIT_TAG "v0.3.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/azure-storage-cpplite/src/azure-storage-cpplite"
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCURL_INCLUDE_DIR:STRING=${CMAKE_CURRENT_BINARY_DIR}/curl/install/include
    -DCURL_LIBRARY:STRING=${CMAKE_CURRENT_BINARY_DIR}/curl/install/lib
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/azure-storage-cpplite/install
  DEPENDS curl
)

#
# Build CNMeM (CUDA memeory management library)
#
ExternalProject_Add(cnmem
  PREFIX cnmem
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third-party-repo/src/third_party/cnmem/cnmem"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/cnmem
    DEPENDS third-party-repo
)

#
# Build AWS sdk for S3 support
#
ExternalProject_Add(aws-sdk-cpp
  PREFIX aws-sdk-cpp
  GIT_REPOSITORY "https://github.com/aws/aws-sdk-cpp.git"
  GIT_TAG "1.7.129"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/src/aws-sdk-cpp"
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DBUILD_ONLY:STRING=s3
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DMINIMIZE_SIZE:BOOL=ON
    -DENABLE_TESTING:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install
)

#
# Build Triton Server test utilities
#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRITON_TEST_UTILS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/test-util/install)
else()
  set(TRITON_TEST_UTILS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

ExternalProject_Add(test-util
  PREFIX test-util
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test-util"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-util"
  BUILD_ALWAYS 1
  CMAKE_CACHE_ARGS
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -DGTEST_ROOT:PATH=${CMAKE_CURRENT_BINARY_DIR}/googletest
    -DLibevent_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevent/install/lib/cmake/libevent
    -DCNMEM_PATH:PATH=${CMAKE_CURRENT_BINARY_DIR}/cnmem
    -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
    -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
    -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG}
    -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
    -DTRITON_MIN_COMPUTE_CAPABILITY:STRING=${TRITON_MIN_COMPUTE_CAPABILITY}
    -DTRITON_ENABLE_TENSORRT:BOOL=${TRITON_ENABLE_TENSORRT}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_TEST_UTILS_INSTALL_PREFIX}
  DEPENDS protobuf googletest cnmem libevent
)

#
# Build Triton Server library and main executable
#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRITON_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/server/install)
else()
  set(TRITON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

set(TRITON_DEPENDS protobuf)
if(${TRITON_ENABLE_GCS})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} google-cloud-cpp)
endif() # TRITON_ENABLE_GCS
if(${TRITON_ENABLE_S3})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} aws-sdk-cpp)
endif() # TRITON_ENABLE_S3
if(${TRITON_ENABLE_AZURE_STORAGE})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} azure-storage-cpplite)
endif() # TRITON_ENABLE_AZURE_STORAGE
if(${TRITON_ENABLE_HTTP} OR ${TRITON_ENABLE_METRICS} OR ${TRITON_ENABLE_SAGEMAKER})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} libevent libevhtp)
endif() # TRITON_ENABLE_HTTP || TRITON_ENABLE_METRICS || TRITON_ENABLE_SAGEMAKER
if(${TRITON_ENABLE_GRPC})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} grpc)
endif() # TRITON_ENABLE_GRPC
if(${TRITON_ENABLE_METRICS})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} prometheus-cpp)
endif() # TRITON_ENABLE_METRICS
if(${TRITON_ENABLE_GPU})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} cnmem)
endif() # TRITON_ENABLE_GPU

ExternalProject_Add(server
  PREFIX server
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/server"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/server"
  BUILD_ALWAYS 1
  CMAKE_CACHE_ARGS
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DgRPC_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/grpc/lib/cmake/grpc
    -Dc-ares_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
    -Dabsl_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/absl/lib/cmake/absl
    -Dnlohmann_json_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/nlohmann_json/lib/cmake/nlohmann_json
    -DLibevent_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevent/install/lib/cmake/libevent
    -Dlibevhtp_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevhtp/install/lib/cmake/libevhtp
    -Dprometheus-cpp_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/prometheus-cpp/install/lib/cmake/prometheus-cpp
    -Dstorage_client_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/google-cloud-cpp/install/lib/cmake/storage_client
    -Dazure-storage-cpplite_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/azure-storage-cpplite/install
    -Dgoogle_cloud_cpp_common_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/google-cloud-cpp/install/lib/cmake/google_cloud_cpp_common
    -DCrc32c_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/crc32c/lib/cmake/Crc32c
    -DAWSSDK_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/cmake/AWSSDK
    -Daws-cpp-sdk-core_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/cmake/aws-cpp-sdk-core
    -Daws-cpp-sdk-s3_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/cmake/aws-cpp-sdk-s3
    -Daws-c-event-stream_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/aws-c-event-stream/cmake
    -Daws-c-common_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/aws-c-common/cmake
    -Daws-checksums_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/aws-checksums/cmake
    -DCNMEM_PATH:PATH=${CMAKE_CURRENT_BINARY_DIR}/cnmem
    -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
    -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
    -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG}
    -DTRITON_ONNXRUNTIME_INCLUDE_PATHS:PATH=${TRITON_ONNXRUNTIME_INCLUDE_PATHS}
    -DTRITON_PYTORCH_INCLUDE_PATHS:PATH=${TRITON_PYTORCH_INCLUDE_PATHS}
    -DTRITON_TENSORRT_INCLUDE_PATHS:PATH=${TRITON_TENSORRT_INCLUDE_PATHS}
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
    -DTRITON_ENABLE_GRPC:BOOL=${TRITON_ENABLE_GRPC}
    -DTRITON_MIN_COMPUTE_CAPABILITY:STRING=${TRITON_MIN_COMPUTE_CAPABILITY}
    -DTRITON_ENABLE_METRICS:BOOL=${TRITON_ENABLE_METRICS}
    -DTRITON_ENABLE_METRICS_GPU:BOOL=${TRITON_ENABLE_METRICS_GPU}
    -DTRITON_ENABLE_GCS:BOOL=${TRITON_ENABLE_GCS}
    -DTRITON_ENABLE_AZURE_STORAGE:BOOL=${TRITON_ENABLE_AZURE_STORAGE}
    -DTRITON_ENABLE_S3:BOOL=${TRITON_ENABLE_S3}
    -DTRITON_ENABLE_TENSORFLOW:BOOL=${TRITON_ENABLE_TENSORFLOW}
    -DTRITON_ENABLE_PYTHON:BOOL=${TRITON_ENABLE_PYTHON}
    -DTRITON_ENABLE_TENSORRT:BOOL=${TRITON_ENABLE_TENSORRT}
    -DTRITON_ENABLE_ONNXRUNTIME:BOOL=${TRITON_ENABLE_ONNXRUNTIME}
    -DTRITON_ENABLE_ONNXRUNTIME_TENSORRT:BOOL=${TRITON_ENABLE_ONNXRUNTIME_TENSORRT}
    -DTRITON_ENABLE_ONNXRUNTIME_OPENVINO:BOOL=${TRITON_ENABLE_ONNXRUNTIME_OPENVINO}
    -DTRITON_ENABLE_PYTORCH:BOOL=${TRITON_ENABLE_PYTORCH}
    -DTRITON_ENABLE_ENSEMBLE:BOOL=${TRITON_ENABLE_ENSEMBLE}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}
    -DTRITON_VERSION:STRING=${TRITON_VERSION}
  DEPENDS ${TRITON_DEPENDS}
)

unset(CMAKE_INSTALL_PREFIX CACHE)
