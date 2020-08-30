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

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/python_host.proto DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
set(Protobuf_SRC_ROOT_FOLDER ${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/python_host.proto)

protobuf_generate_python(PROTO_PY ${CMAKE_CURRENT_BINARY_DIR}/python_host.proto)

set (protocol_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/python_host.grpc.pb.cc")
set (protocol_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/python_host.grpc.pb.h")
set (protocol_grpc_py "${CMAKE_CURRENT_BINARY_DIR}/python_host_pb2_grpc.py")

add_custom_command(
      OUTPUT "${protocol_grpc_srcs}" "${protocol_grpc_hdrs}"
      COMMAND protobuf::protoc
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        --plugin=protoc-gen-grpc="${gRPC_CPP_PLUGIN_EXECUTABLE}"
        -I "${CMAKE_CURRENT_BINARY_DIR}"
        "python_host.proto"
        DEPENDS "${PROTO_SRCS}")

add_custom_command(
        OUTPUT "${protocol_grpc_py}"
        COMMAND python3
        ARGS -m grpc_tools.protoc
        --python_out "${CMAKE_CURRENT_BINARY_DIR}"
        --grpc_python_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_BINARY_DIR}"
        "python_host.proto"
        DEPENDS "${PROTO_SRCS}")

add_custom_target(
        python_libs ALL 
        DEPENDS ${PROTO_PY} ${protocol_grpc_py})

include_directories(${CMAKE_CURRENT_BINARY_DIR})

# FIXME The relative paths created by the protobuf_generate_cpp and
# protobuf_generate_python are incorrect
string(REPLACE "../../../../" "../../../../../../../../../"  PROTOBUF_HEADERS ${PROTO_HDRS})
string(REPLACE "../../../../" "../../../../../../../../../"  PROTOBUF_SRCS ${PROTO_SRCS})

set_source_files_properties(${PROTOBUF_HEADERS} PROPERTIES GENERATED TRUE)
set_source_files_properties(${PROTOBUF_SRCS} PROPERTIES GENERATED TRUE)

add_library(
  triton_python SHARED
  python.cc
  ${PROTOBUF_HEADERS}
  ${PROTOBUF_SRCS}

  ${protocol_grpc_hdrs}
  ${protocol_grpc_srcs}
  $<TARGET_OBJECTS:backend-library>)

if(${TRITON_ENABLE_GPU})
  target_include_directories(triton_python PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(
    triton_python
    PUBLIC -L/usr/local/cuda/lib64/stubs
    PUBLIC -lnvidia-ml
    PRIVATE ${CUDA_LIBRARIES}
  )
endif() # TRITON_ENABLE_GPU

set_target_properties(
  triton_python
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_python.ldscript
)

set_target_properties(
  triton_python
  PROPERTIES LINK_FLAGS "-Wl,--version-script libtriton_python.ldscript"
)

target_include_directories(
        triton_python
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
  triton_python
  PRIVATE gRPC::grpc++
)

install(
  TARGETS triton_python
  LIBRARY DESTINATION backends/python
)

# FIXME The relative paths created by the protobuf_generate_cpp and
# protobuf_generate_python are incorrect
string(REPLACE "../../../../" "../../../../../../../../../"  PROTO_FIX_PY ${PROTO_PY})

install(FILES ${protocol_core_py} ${PROTO_FIX_PY} ${protocol_grpc_py} resources/startup.py
       DESTINATION lib/python/runtime)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/triton_python_backend_utils.py
       DESTINATION lib/python/runtime)
