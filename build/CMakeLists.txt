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
project (tensorrt-inference-server)

include(CMakeDependentOption)
include(ExternalProject)

# Backends
option(TRTIS_ENABLE_TENSORRT "Include TensorRT backend in server" OFF)
option(TRTIS_ENABLE_TENSORFLOW "Include TensorFlow backend in server" OFF)
option(TRTIS_ENABLE_CAFFE2 "Include Caffe2 backend in server" OFF)
option(TRTIS_ENABLE_ONNXRUNTIME "Include ONNXRuntime backend in server" OFF)
option(TRTIS_ENABLE_PYTORCH "Include PyTorch backend in server" OFF)
option(TRTIS_ENABLE_CUSTOM "Include Custom backend in server" OFF)

# Endpoints
option(TRTIS_ENABLE_HTTP "Include HTTP API in server" ON)
option(TRTIS_ENABLE_GRPC "Include GRPC API in server" ON)
option(TRTIS_ENABLE_METRICS "Include metrics support in server" ON)

# Cloud storage
option(TRTIS_ENABLE_GCS "Include GCS Filesystem support in server" OFF)
option(TRTIS_ENABLE_S3 "Include S3 Filesystem support in server" OFF)

# Multiple paths may be specified by separating them with semicolon
set(TRTIS_ONNXRUNTIME_INCLUDE_PATHS "" CACHE PATH "Paths to ONNXRuntime includes")
set(TRTIS_PYTORCH_INCLUDE_PATHS "" CACHE PATH "Paths to PyTorch includes")
set(TRTIS_EXTRA_LIB_PATHS "" CACHE PATH "Extra library paths for TRTIS build")

option(TRTIS_ENABLE_TRACING "Include tracing support in server" OFF)
option(TRTIS_ENABLE_ASAN "Build with address sanitizer" OFF)
option(TRTIS_ENABLE_GPU "Enable GPU support in server" ON)
set(TRTIS_MIN_COMPUTE_CAPABILITY "6.0" CACHE STRING
    "The minimum CUDA compute capability supported by TRTIS" )

if(TRTIS_ENABLE_TENSORRT AND NOT TRTIS_ENABLE_GPU)
  message(FATAL_ERROR "TRTIS_ENABLE_TENSORRT=ON requires TRTIS_ENABLE_GPU=ON")
endif()

if(TRTIS_ENABLE_ASAN AND TRTIS_ENABLE_GPU)
  message(FATAL_ERROR "TRTIS_ENABLE_ASAN=ON requires TRTIS_ENABLE_GPU=OFF")
endif()

# If OPENSSL_ROOT_DIR is set, propagate that hint path to the external
# projects with OpenSSL dependency.
set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
if (OPENSSL_ROOT_DIR)
  set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
endif()

#
# Get the protobuf and grpc source
#
ExternalProject_Add(grpc-repo
  PREFIX grpc-repo
  GIT_REPOSITORY "https://github.com/grpc/grpc.git"
  GIT_TAG "v1.19.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)

#
# Build protobuf project from grpc-repo
#
ExternalProject_Add(protobuf
  PREFIX protobuf
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc/third_party/protobuf/cmake"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -Dprotobuf_BUILD_TESTS:BOOL=OFF
    -Dprotobuf_WITH_ZLIB:BOOL=OFF
    -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
    -DCMAKE_BUILD_TYPE:BOOL=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/protobuf
  DEPENDS grpc-repo
)

# Location where protobuf-config.cmake will be installed varies by
# platform
if (WIN32)
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/cmake")
else()
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf/lib/cmake/protobuf")
endif()

#
# Build c-area project from grpc-repo
#
ExternalProject_Add(c-ares
  PREFIX c-ares
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc/third_party/cares/cares"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    -DCARES_SHARED:BOOL=OFF
    -DCARES_STATIC:BOOL=ON
    -DCARES_STATIC_PIC:BOOL=ON
    -DCMAKE_BUILD_TYPE:BOOL=${CMAKE_BUILD_TYPE}
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
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -DgRPC_ZLIB_PROVIDER:STRING=package
    -DgRPC_CARES_PROVIDER:STRING=package
    -Dc-ares_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
    -DgRPC_SSL_PROVIDER:STRING=package
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    -DCMAKE_BUILD_TYPE:BOOL=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/grpc
  DEPENDS grpc-repo c-ares protobuf
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
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:BOOL=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevent/install
)

#
# Build libevhtp
#
# Need to patch due to https://github.com/criticalstack/libevhtp/issues/135
#
ExternalProject_Add(libevhtp
  PREFIX libevhtp
  URL "https://github.com/criticalstack/libevhtp/archive/1.2.18.zip"
  URL_HASH SHA256=3194dc6eb4e8d6aa1e7dd3dc60bfbe066f38f9a0b5881463f0e149badd82a7bb
  PATCH_COMMAND sed -i "s/LibEvent/Libevent/g" ${CMAKE_CURRENT_BINARY_DIR}/libevhtp/src/libevhtp/CMakeLists.txt
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/libevhtp/src/libevhtp"
  CMAKE_CACHE_ARGS
    -DEVHTP_DISABLE_SSL:BOOL=ON
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DLibevent_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevent/install/lib/cmake/libevent
    -DCMAKE_BUILD_TYPE:BOOL=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevhtp/install
  DEPENDS libevent
)

#
# Build Prometheus C++ library
#
ExternalProject_Add(prometheus-cpp
  PREFIX prometheus-cpp
  URL "https://github.com/jupp0r/prometheus-cpp/archive/v0.5.0.tar.gz"
  URL_HASH SHA256=76c60f9b1bba19bcfb29b06749ec7530db7151692eb12443d00d3511e16696e2
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/prometheus-cpp/src/prometheus-cpp"
  CMAKE_CACHE_ARGS
    -DENABLE_PUSH:BOOL=OFF
    -DENABLE_PULL:BOOL=OFF
    -DENABLE_TESTING:BOOL=OFF
    -DUSE_THIRDPARTY_LIBRARIES:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:BOOL=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/prometheus-cpp/install
)

#
# Build crc32c
#
ExternalProject_Add(crc32c
  PREFIX crc32c
  GIT_REPOSITORY "https://github.com/google/crc32c.git"
  GIT_TAG "1.1.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/crc32c/src/crc32c"
  CMAKE_CACHE_ARGS
    -DCRC32C_BUILD_TESTS:BOOL=OFF
    -DBUILD_SHARED_LIBS:STRING=no
    -DCRC32C_BUILD_BENCHMARKS:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/crc32c
)

# This is necessary because google-cloud-cpp
# does not pass protobuf_DIR and gRPC_DIR etc to its
# external projects but expects them in CMAKE_PREFIX_PATH
set(GCS_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
   ${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
   ${CMAKE_CURRENT_BINARY_DIR}/grpc/lib/cmake/grpc
   ${CMAKE_CURRENT_BINARY_DIR}/crc32c/lib/cmake/Crc32c
   ${_FINDPACKAGE_PROTOBUF_CONFIG_DIR})

#
# Build google-cloud-cpp
#
ExternalProject_Add(google-cloud-cpp
  PREFIX google-cloud-cpp
  GIT_REPOSITORY "https://github.com/googleapis/google-cloud-cpp.git"
  GIT_TAG "v0.10.0"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/google-cloud-cpp/src/google-cloud-cpp"
  CMAKE_CACHE_ARGS
    -DGOOGLE_CLOUD_CPP_DEPENDENCY_PROVIDER:STRING=package
    -DBUILD_TESTING:BOOL=OFF
    -DCMAKE_PREFIX_PATH:PATH=${GCS_CMAKE_PREFIX_PATH}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/google-cloud-cpp/install
  DEPENDS grpc protobuf crc32c
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
    -DBUILD_ONLY:STRING=s3
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DMINIMIZE_SIZE:BOOL=ON
    -DENABLE_TESTING:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install
)

#
# Build TRTIS test utilities
#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRTIS_TEST_UTILS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/trtis-test-utils/install)
else()
  set(TRTIS_TEST_UTILS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

ExternalProject_Add(trtis-test-utils
  PREFIX trtis-test-utils
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtis-test-utils"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/trtis-test-utils"
  BUILD_ALWAYS 1
  CMAKE_CACHE_ARGS
    -DTRTIS_ENABLE_GPU:BOOL=${TRTIS_ENABLE_GPU}
    -DTRTIS_MIN_COMPUTE_CAPABILITY:STRING=${TRTIS_MIN_COMPUTE_CAPABILITY}
    -DTRTIS_ENABLE_TENSORRT:BOOL=${TRTIS_ENABLE_TENSORRT}
    -DCMAKE_BUILD_TYPE:BOOL=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRTIS_TEST_UTILS_INSTALL_PREFIX}
)

#
# Build example custom backends
#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/trtis-custom-backends/install)
else()
  set(TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

ExternalProject_Add(trtis-custom-backends
  PREFIX trtis-custom-backends
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtis-custom-backends"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/trtis-custom-backends"
  BUILD_ALWAYS 1
  CMAKE_CACHE_ARGS
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -Dc-ares_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    -DgRPC_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/grpc/lib/cmake/grpc
    -DTRTIS_ENABLE_GPU:BOOL=${TRTIS_ENABLE_GPU}
    -DTRTIS_MIN_COMPUTE_CAPABILITY:STRING=${TRTIS_MIN_COMPUTE_CAPABILITY}
    -DCMAKE_BUILD_TYPE:BOOL=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX}
  DEPENDS protobuf grpc
)

#
# Build TRTIS client libraries and examples
#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRTIS_CLIENTS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/trtis-clients/install)
else()
  set(TRTIS_CLIENTS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

ExternalProject_Add(trtis-clients
  PREFIX trtis-clients
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtis-clients"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/trtis-clients"
  BUILD_ALWAYS 1
  CMAKE_CACHE_ARGS
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -Dc-ares_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    -DgRPC_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/grpc/lib/cmake/grpc
    -DTRTIS_ENABLE_METRICS:BOOL=OFF
    -DCMAKE_BUILD_TYPE:BOOL=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRTIS_CLIENTS_INSTALL_PREFIX}
  DEPENDS protobuf grpc
)

#
# Build TRTIS server library and main executable
#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRTIS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/trtis/install)
else()
  set(TRTIS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

set(TRTIS_DEPENDS protobuf)
if(${TRTIS_ENABLE_GCS})
  set(TRTIS_DEPENDS ${TRTIS_DEPENDS} google-cloud-cpp)
endif() # TRTIS_ENABLE_GCS
if(${TRTIS_ENABLE_S3})
  set(TRTIS_DEPENDS ${TRTIS_DEPENDS} aws-sdk-cpp)
endif() # TRTIS_ENABLE_S3
if(${TRTIS_ENABLE_HTTP} OR ${TRTIS_ENABLE_METRICS})
  set(TRTIS_DEPENDS ${TRTIS_DEPENDS} libevent libevhtp)
endif() # TRTIS_ENABLE_HTTP || TRTIS_ENABLE_METRICS
if(${TRTIS_ENABLE_GRPC})
  set(TRTIS_DEPENDS ${TRTIS_DEPENDS} grpc)
endif() # TRTIS_ENABLE_GRPC
if(${TRTIS_ENABLE_METRICS})
  set(TRTIS_DEPENDS ${TRTIS_DEPENDS} prometheus-cpp)
endif() # TRTIS_ENABLE_METRICS

ExternalProject_Add(trtis
  PREFIX trtis
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtis"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/trtis"
  BUILD_ALWAYS 1
  CMAKE_CACHE_ARGS
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -Dc-ares_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    -DgRPC_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/grpc/lib/cmake/grpc
    -DLibevent_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevent/install/lib/cmake/libevent
    -Dlibevhtp_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevhtp/install/lib/cmake/libevhtp
    -Dprometheus-cpp_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/prometheus-cpp/install/lib/cmake/prometheus-cpp
    -Dstorage_client_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/google-cloud-cpp/install/lib/cmake/storage_client
    -Dgoogle_cloud_cpp_common_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/google-cloud-cpp/install/lib/cmake/google_cloud_cpp_common
    -DCrc32c_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/crc32c/lib/cmake/Crc32c
    -DAWSSDK_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/cmake/AWSSDK
    -Daws-cpp-sdk-core_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/cmake/aws-cpp-sdk-core
    -Daws-cpp-sdk-s3_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/cmake/aws-cpp-sdk-s3
    -Daws-c-event-stream_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/aws-c-event-stream/cmake
    -Daws-c-common_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/aws-c-common/cmake
    -Daws-checksums_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/aws-sdk-cpp/install/lib/aws-checksums/cmake
    -DTRTIS_ONNXRUNTIME_INCLUDE_PATHS:PATH=${TRTIS_ONNXRUNTIME_INCLUDE_PATHS}
    -DTRTIS_PYTORCH_INCLUDE_PATHS:PATH=${TRTIS_PYTORCH_INCLUDE_PATHS}
    -DTRTIS_EXTRA_LIB_PATHS:PATH=${TRTIS_EXTRA_LIB_PATHS}
    -DTRTIS_ENABLE_ASAN:BOOL=${TRTIS_ENABLE_ASAN}
    -DTRTIS_ENABLE_TRACING:BOOL=${TRTIS_ENABLE_TRACING}
    -DTRTIS_ENABLE_GPU:BOOL=${TRTIS_ENABLE_GPU}
    -DTRTIS_ENABLE_HTTP:BOOL=${TRTIS_ENABLE_HTTP}
    -DTRTIS_ENABLE_GRPC:BOOL=${TRTIS_ENABLE_GRPC}
    -DTRTIS_MIN_COMPUTE_CAPABILITY:STRING=${TRTIS_MIN_COMPUTE_CAPABILITY}
    -DTRTIS_ENABLE_METRICS:BOOL=${TRTIS_ENABLE_METRICS}
    -DTRTIS_ENABLE_GCS:BOOL=${TRTIS_ENABLE_GCS}
    -DTRTIS_ENABLE_S3:BOOL=${TRTIS_ENABLE_S3}
    -DTRTIS_ENABLE_TENSORFLOW:BOOL=${TRTIS_ENABLE_TENSORFLOW}
    -DTRTIS_ENABLE_TENSORRT:BOOL=${TRTIS_ENABLE_TENSORRT}
    -DTRTIS_ENABLE_CAFFE2:BOOL=${TRTIS_ENABLE_CAFFE2}
    -DTRTIS_ENABLE_ONNXRUNTIME:BOOL=${TRTIS_ENABLE_ONNXRUNTIME}
    -DTRTIS_ENABLE_PYTORCH:BOOL=${TRTIS_ENABLE_PYTORCH}
    -DTRTIS_ENABLE_CUSTOM:BOOL=${TRTIS_ENABLE_CUSTOM}
    -DCMAKE_BUILD_TYPE:BOOL=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRTIS_INSTALL_PREFIX}
  DEPENDS ${TRTIS_DEPENDS}
)

unset(CMAKE_INSTALL_PREFIX CACHE)
