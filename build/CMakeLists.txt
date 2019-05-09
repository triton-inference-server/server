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

option(TRTIS_ENABLE_METRICS "Include metrics support in server" ON)
option(TRTIS_ENABLE_TENSORFLOW "Include TensorFlow backend in server" ON)
option(TRTIS_ENABLE_TENSORRT "Include TensorRT backend in server" ON)
option(TRTIS_ENABLE_CAFFE2 "Include Caffe2 backend in server" ON)
option(TRTIS_ENABLE_CUSTOM "Include Custom backend in server" ON)
option(TRTIS_NODEPENDS "Don't build required dependencies for TRTIS targets" OFF)

include(ExternalProject)

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
# Build zlib project from grpc-repo
#
ExternalProject_Add(zlib
  PREFIX zlib
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc-repo/src/grpc/third_party/zlib"
  DOWNLOAD_COMMAND ""
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/zlib
  DEPENDS grpc-repo
)

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
    -DZLIB_ROOT:STRING=${CMAKE_CURRENT_BINARY_DIR}/zlib
    -DgRPC_CARES_PROVIDER:STRING=package
    -Dc-ares_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
    -DgRPC_SSL_PROVIDER:STRING=package
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/grpc
  DEPENDS grpc-repo c-ares protobuf zlib
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
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/prometheus-cpp/install
)

#
# Build TRTIS test utilities
#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRTIS_TEST_UTILS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/trtis-test-utils/install)
else()
  set(TRTIS_TEST_UTILS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

if (${TRTIS_NODEPENDS})
  set(TRTIS_TEST_UTILS_DEPENDS "")
else()
  set(TRTIS_TEST_UTILS_DEPENDS "")
endif()

ExternalProject_Add(trtis-test-utils
  PREFIX trtis-test-utils
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtis-test-utils"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/trtis-test-utils"
  CMAKE_CACHE_ARGS
    -DTRTIS_ENABLE_TENSORRT:BOOL=${TRTIS_ENABLE_TENSORRT}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRTIS_TEST_UTILS_INSTALL_PREFIX}
  DEPENDS ${TRTIS_TEST_UTILS_DEPENDS}
)

#
# Build example custom backends
#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/trtis-custom-backends/install)
else()
  set(TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

if (${TRTIS_NODEPENDS})
  set(TRTIS_CUSTOM_BACKENDS_DEPENDS "")
else()
  set(TRTIS_CUSTOM_BACKENDS_DEPENDS "")
endif()

ExternalProject_Add(trtis-custom-backends
  PREFIX trtis-custom-backends
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtis-custom-backends"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/trtis-custom-backends"
  CMAKE_CACHE_ARGS
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -Dc-ares_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
    -DZLIB_ROOT:STRING=${CMAKE_CURRENT_BINARY_DIR}/zlib
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    -DgRPC_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/grpc/lib/cmake/grpc
    -DTRTIS_ENABLE_CUSTOM:BOOL=${TRTIS_ENABLE_CUSTOM}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRTIS_CUSTOM_BACKENDS_INSTALL_PREFIX}
  DEPENDS ${TRTIS_CUSTOM_BACKENDS_DEPENDS}
)

#
# Build TRTIS client libraries and examples
#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRTIS_CLIENTS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/trtis-clients/install)
else()
  set(TRTIS_CLIENTS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

if (${TRTIS_NODEPENDS})
  set(TRTIS_CLIENTS_DEPENDS "")
else()
  set(TRTIS_CLIENTS_DEPENDS protobuf grpc)
endif()

ExternalProject_Add(trtis-clients
  PREFIX trtis-clients
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtis-clients"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/trtis-clients"
  CMAKE_CACHE_ARGS
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -Dc-ares_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
    -DZLIB_ROOT:STRING=${CMAKE_CURRENT_BINARY_DIR}/zlib
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    -DgRPC_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/grpc/lib/cmake/grpc
    -DTRTIS_ENABLE_METRICS:BOOL=OFF
    -DCMAKE_INSTALL_PREFIX:PATH=${TRTIS_CLIENTS_INSTALL_PREFIX}
  DEPENDS ${TRTIS_CLIENTS_DEPENDS}
)

#
# Build TRTIS server library and main executable
#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRTIS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/trtis/install)
else()
  set(TRTIS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

if (${TRTIS_NODEPENDS})
  set(TRTIS_DEPENDS "")
else()
  set(TRTIS_DEPENDS protobuf grpc libevent libevhtp)
  if(${TRTIS_ENABLE_METRICS})
    set(TRTIS_DEPENDS ${TRTIS_DEPENDS} prometheus-cpp)
  endif() # TRTIS_ENABLE_METRICS
endif()

ExternalProject_Add(trtis
  PREFIX trtis
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtis"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/trtis"
  CMAKE_CACHE_ARGS
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    -Dc-ares_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/c-ares/lib/cmake/c-ares
    -DZLIB_ROOT:STRING=${CMAKE_CURRENT_BINARY_DIR}/zlib
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    -DgRPC_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/grpc/lib/cmake/grpc
    -DLibevent_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevent/install/lib/cmake/libevent
    -Dlibevhtp_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/libevhtp/install/lib/cmake/libevhtp
    -Dprometheus-cpp_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/prometheus-cpp/install/lib/cmake/prometheus-cpp
    -DTRTIS_ENABLE_METRICS:BOOL=${TRTIS_ENABLE_METRICS}
    -DTRTIS_ENABLE_TENSORFLOW:BOOL=${TRTIS_ENABLE_TENSORFLOW}
    -DTRTIS_ENABLE_TENSORRT:BOOL=${TRTIS_ENABLE_TENSORRT}
    -DTRTIS_ENABLE_CAFFE2:BOOL=${TRTIS_ENABLE_CAFFE2}
    -DTRTIS_ENABLE_CUSTOM:BOOL=${TRTIS_ENABLE_CUSTOM}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRTIS_INSTALL_PREFIX}
  DEPENDS ${TRTIS_DEPENDS}
)

unset(CMAKE_INSTALL_PREFIX CACHE)
unset(TRTIS_NODEPENDS CACHE)
