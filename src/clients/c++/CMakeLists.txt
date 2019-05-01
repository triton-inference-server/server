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

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
include_directories(${Protobuf_INCLUDE_DIRS})

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)

#
# librequest.so
#
find_package(CURL REQUIRED)
add_library(
  request SHARED
  request.cc request_common.cc request_http.cc request_grpc.cc
  $<TARGET_OBJECTS:model-config-library>
  $<TARGET_OBJECTS:proto-library>
  $<TARGET_OBJECTS:grpc-library>
)
target_include_directories(request PRIVATE ${CURL_INCLUDE_DIRS})
target_link_libraries(request gRPC::grpc++ gRPC::grpc -lcurl)

#
# ensemble_image_client
#
add_executable(ensemble_image_client ensemble_image_client.cc)
target_link_libraries(ensemble_image_client request)

#
# image_client
#
find_package(OpenCV REQUIRED)
add_executable(image_client image_client.cc)
target_include_directories(image_client PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_client request ${OpenCV_LIBS})

#
# perf_client
#
add_executable(perf_client perf_client.cc)
target_link_libraries(perf_client request)

#
# simple_client
#
add_executable(simple_client simple_client.cc)
target_link_libraries(simple_client request)

#
# simple_sequence_client
#
add_executable(simple_sequence_client simple_sequence_client.cc)
target_link_libraries(simple_sequence_client request)

#
# simple_string_client
#
add_executable(simple_string_client simple_string_client.cc)
target_link_libraries(simple_string_client request)
