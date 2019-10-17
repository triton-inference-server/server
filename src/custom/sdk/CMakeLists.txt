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

#
# libcustombackend.a
#
configure_file(combine.sh combine.sh COPYONLY)

add_library(
  custombackendparts STATIC
  custom_instance.cc
  error_codes.cc
  custom_instance.h
  error_codes.h
  $<TARGET_OBJECTS:model-config-library>
  $<TARGET_OBJECTS:proto-library>
)

set(TRTIS_CUSTOMBACKEND "libcustombackend.a")
add_custom_command(
  OUTPUT libcustombackend.a
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Protobuf_LIBRARY} libcustombackendprotobuf.a
  COMMAND bash -x combine.sh
  DEPENDS custombackendparts combine.sh
)
add_custom_target(custom-backend-target ALL DEPENDS libcustombackend.a)

add_library(custombackend STATIC IMPORTED GLOBAL)
add_dependencies(custombackend custom-backend-target)
set_target_properties(custombackend
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libcustombackend.a
)

install(
  PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/libcustombackend.a
  DESTINATION lib
)
install(
  FILES
    custom_instance.cc
    custom_instance.h
    error_codes.cc
    error_codes.h
  DESTINATION include/src/custom/sdk
)

# Install the TRTIS and protobuf headers needed to build a custom
# backend
install(
  FILES
    ../../core/model_config.h
    ${CMAKE_CURRENT_BINARY_DIR}/../../core/model_config.pb.h
  DESTINATION include/src/core
)
install(
  FILES ../../backends/custom/custom.h
  DESTINATION include/src/backends/custom
)
install(
  DIRECTORY ${Protobuf_INCLUDE_DIRS}/google
  DESTINATION include
)

# Install protoc with same version as the protobuf headers. This makes
# it easy for custom backends that want to compile their own protobufs
# in a way that is compatible with the protobuf version in
# libcustombackend library.
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
install(
  PROGRAMS ${_PROTOBUF_PROTOC}
  DESTINATION bin
)

# Install the param custom backend source as an example
install(
  FILES ../param/param.cc
  DESTINATION src
)
