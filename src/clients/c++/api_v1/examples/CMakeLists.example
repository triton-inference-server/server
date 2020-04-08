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

set(CMAKE_CXX_STANDARD 11)

#
# Dependencies
#
find_package(TRTIS CONFIG REQUIRED)
message(STATUS "Using TRTIS client library ${TRTIS_VERSION}")

#
# Example applications
#
include_directories("${PROJECT_SOURCE_DIR}/../../include")

# simple_client using static library, librequest_static.a
add_executable(
  simple_client_static
  ${PROJECT_SOURCE_DIR}/../../src/simple_client.cc
)
target_link_libraries(
  simple_client_static
  PRIVATE -L${PROJECT_SOURCE_DIR}/../lib
  PRIVATE TRTIS::request_static
  PRIVATE gRPC::grpc++
  PRIVATE gRPC::grpc
  PRIVATE CURL::libcurl
  PUBLIC protobuf::libprotobuf
)
install(
  TARGETS simple_client_static
  RUNTIME DESTINATION bin
)

# simple_client using shared library, librequest.so
add_executable(
  simple_client_shared
  ${PROJECT_SOURCE_DIR}/../../src/simple_client.cc
)
target_link_libraries(
  simple_client_shared
  PRIVATE -L${PROJECT_SOURCE_DIR}/../lib
  PRIVATE TRTIS::request
  PUBLIC protobuf::libprotobuf
)
install(
  TARGETS simple_client_shared
  RUNTIME DESTINATION bin
)
