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

cmake_minimum_required (VERSION 3.5)

#
# libpython_interpreter.so
#
configure_file(libtriton_python.ldscript libtriton_python.ldscript COPYONLY)

find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)


set(gRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
list(APPEND Protobuf_IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../../core)
list(APPEND Protobuf_IMPORT_DIRS ${CMAKE_CURRENT_BINARY_DIR}../../../../../../../../core)
message("Protobuf Import dif ${Protobuf_IMPORT_DIRS}")

# protobuf_generate_cpp(PROTO_CORE_SRCS PROTO_CORE_HDRS ../../../core/model_config.proto)
protobuf_generate_cpp(PROTO_HOST_SRCS PROTO_HOST_HDRS python_host.proto)

# protobuf_generate_python(PROTO_CORE_PY ../../../core/model_config.proto)
protobuf_generate_python(PROTO_HOST_PY python_host.proto)

set (protocol_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/python_host.grpc.pb.cc")
set (protocol_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/python_host.grpc.pb.h")
set (protocol_grpc_py "${CMAKE_CURRENT_BINARY_DIR}/python_host_pb2_grpc.py")
set (protocol_core_py "${CMAKE_CURRENT_BINARY_DIR}/model_config_pb2.py")

add_custom_command(
      OUTPUT "${protocol_grpc_srcs}" "${protocol_grpc_hdrs}"
      COMMAND protobuf::protoc
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}/../../../core"
        -I "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        --plugin=protoc-gen-grpc="${gRPC_CPP_PLUGIN_EXECUTABLE}"
        "python_host.proto"
        DEPENDS "${PROTO_CORE_SRCS}")

add_custom_command(
        OUTPUT "${protocol_grpc_py}"
        COMMAND python3
        ARGS -m grpc_tools.protoc
        --python_out "${CMAKE_CURRENT_BINARY_DIR}"
        --grpc_python_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}/../../../core"
        -I "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        "python_host.proto"
        DEPENDS "${PROTO_CORE_SRCS}")

add_custom_target(
        python_libs ALL 
        DEPENDS ${PROTO_CORE_PY} ${PROTO_HOST_PY} ${protocol_grpc_py})

add_library(
  triton_python SHARED
  python_interpreter.cc
  ${PROTO_CORE_SRCS}
  ${PROTO_CORE_HDRS}
  ${PROTO_HOST_SRCS}
  ${PROTO_HOST_HDRS}

  ${protocol_grpc_hdrs}
  ${protocol_grpc_srcs})

set_target_properties(
  triton_python
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_python.ldscript
)

set_target_properties(
  triton_python
  PROPERTIES LINK_FLAGS "-Wl,--version-script libpython_interpreter.ldscript"
)

target_include_directories(
        triton_python
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
  triton_python
  PRIVATE gRPC::grpc++
)

if(${TRTIS_ENABLE_GPU})
target_include_directories(
        triton_python
        PRIVATE ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(
  triton_python
  PUBLIC -L/usr/local/cuda/lib64/stubs
  PUBLIC -lnvidia-ml
  PRIVATE ${CUDA_LIBRARIES}
)
endif() # TRTIS_ENABLE_GPU

install(
  TARGETS triton_python
  LIBRARY DESTINATION backends
)

install(FILES ${protocol_core_py} ${PROTO_HOST_PY} ${protocol_grpc_py} resources/startup.py
        DESTINATION lib/python/runtime)
