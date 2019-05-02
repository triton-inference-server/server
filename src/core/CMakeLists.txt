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
# Protobuf
#
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

include_directories(${Protobuf_INCLUDE_DIRS})

file(GLOB proto-srcs *.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${proto-srcs})
protobuf_generate_python(PROTO_PY ${proto-srcs})

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

add_library(
  proto-library EXCLUDE_FROM_ALL OBJECT
  ${PROTO_SRCS} ${PROTO_HDRS}
)

add_custom_target(proto-py-library DEPENDS ${PROTO_PY})

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/api.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/model_config.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/request_status.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/server_status.pb.h
  DESTINATION include
)

#
# GRPC
#
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)

get_filename_component(grpc_service_proto_abspath "grpc_service.proto" ABSOLUTE)
get_filename_component(grpc_service_proto_dir "${grpc_service_proto_abspath}" PATH)
set(GRPC_SRCS "grpc_service.grpc.pb.cc")
set(GRPC_HDRS "grpc_service.grpc.pb.h")
set(GRPC_PY "grpc_service.grpc.py")

add_custom_command(
  OUTPUT "${GRPC_SRCS}" "${GRPC_HDRS}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS
    --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
    --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
    -I "${grpc_service_proto_dir}"
    --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
    "grpc_service.proto"
  DEPENDS "grpc_service.proto"
)

find_program(PYTHON "python" REQUIRED)
add_custom_command(
  OUTPUT "${GRPC_PY}"
  COMMAND ${PYTHON}
  ARGS
    -m grpc_tools.protoc
    -I "${grpc_service_proto_dir}"
    --grpc_python_out "${CMAKE_CURRENT_BINARY_DIR}"
    "grpc_service.proto"
  DEPENDS "grpc_service.proto"
)

add_library(
  grpc-library EXCLUDE_FROM_ALL OBJECT
  ${GRPC_SRCS} ${GRPC_HDRS}
)

add_custom_target(grpc-py-library DEPENDS ${GRPC_PY})

#
# Model configuration utilities used by both clients and server.
#
add_library(
  model-config-library EXCLUDE_FROM_ALL OBJECT
  model_config.cc model_config.h
)
add_dependencies(model-config-library proto-library)

add_library(
  model-config-cuda-library EXCLUDE_FROM_ALL OBJECT
  model_config_cuda.cc model_config_cuda.h
)
target_include_directories(model-config-cuda-library PRIVATE ${CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_dependencies(model-config-library proto-library)
