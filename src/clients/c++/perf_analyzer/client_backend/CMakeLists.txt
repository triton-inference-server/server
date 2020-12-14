# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

if(${TRITON_CURL_WITHOUT_CONFIG})
  find_package(CURL REQUIRED)
else()
  find_package(CURL CONFIG REQUIRED)
endif() # TRITON_CURL_WITHOUT_CONFIG
message(STATUS "Using curl ${CURL_VERSION}")
add_definitions(-DCURL_STATICLIB)

add_subdirectory(triton)
add_subdirectory(tensorflow_serving)
add_subdirectory(torchserve)

set(
  CLIENT_BACKEND_SRCS
  client_backend.cc
  ../../examples/shm_utils.cc
)

set(
  CLIENT_BACKEND_HDRS
  client_backend.h
  ../../examples/shm_utils.h
)

add_library(
  client-backend-library
  ${CLIENT_BACKEND_SRCS}
  ${CLIENT_BACKEND_HDRS}
  $<TARGET_OBJECTS:triton-client-backend-library>
  $<TARGET_OBJECTS:tfs-client-backend-library>
  $<TARGET_OBJECTS:ts-client-backend-library>
)

target_link_libraries(
  client-backend-library
  PUBLIC triton-common-json        # from repo-common
  PUBLIC $<TARGET_PROPERTY:triton-client-backend-library,LINK_LIBRARIES>
  PUBLIC $<TARGET_PROPERTY:tfs-client-backend-library,LINK_LIBRARIES>
  PUBLIC $<TARGET_PROPERTY:ts-client-backend-library,LINK_LIBRARIES>
)

target_include_directories(
    client-backend-library
    PRIVATE $<TARGET_PROPERTY:triton-client-backend-library,INCLUDE_DIRECTORIES>
    PRIVATE $<TARGET_PROPERTY:tfs-client-backend-library,INCLUDE_DIRECTORIES>
    PRIVATE $<TARGET_PROPERTY:ts-client-backend-library,INCLUDE_DIRECTORIES>
)
