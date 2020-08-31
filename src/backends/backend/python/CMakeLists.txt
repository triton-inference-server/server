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
# libtriton_python.so
#
configure_file(libtriton_python.ldscript libtriton_python.ldscript COPYONLY)

find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

get_filename_component(python_host_proto_abspath "python_host.proto" ABSOLUTE)
get_filename_component(python_host_proto_dir "${python_host_proto_abspath}" PATH)

set(GRPC_SRCS
    "${CMAKE_CURRENT_BINARY_DIR}/python_host.grpc.pb.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/python_host.pb.cc")

set(GRPC_HDRS
    "${CMAKE_CURRENT_BINARY_DIR}/python_host.grpc.pb.h"
    "${CMAKE_CURRENT_BINARY_DIR}/python_host.pb.h")

set(GRPC_PY
    "${CMAKE_CURRENT_BINARY_DIR}/python_host_pb2_grpc.py"
    "${CMAKE_CURRENT_BINARY_DIR}/python_host_pb2.py")

add_custom_command(
  OUTPUT ${GRPC_SRCS} ${GRPC_HDRS}
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS
    --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
    --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
    -I "${python_host_proto_dir}"
    --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
    "python_host.proto"
  DEPENDS "python_host.proto" proto-library
)

find_program(PYTHON "python" REQUIRED)
add_custom_command(
  OUTPUT ${GRPC_PY}
  COMMAND ${PYTHON}
  ARGS
    -m grpc_tools.protoc
    -I "${python_host_proto_dir}"
    --grpc_python_out "${CMAKE_CURRENT_BINARY_DIR}"
    --python_out "${CMAKE_CURRENT_BINARY_DIR}"
    "python_host.proto"
  DEPENDS "python_host.proto" proto-library
)

add_custom_target(python-grpc-py-library ALL
                  DEPENDS ${GRPC_PY})

add_library(
  python-grpc-library EXCLUDE_FROM_ALL OBJECT
  ${GRPC_SRCS} ${GRPC_HDRS}
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(
  triton_python SHARED
  python.cc

  $<TARGET_OBJECTS:backend-library>
  $<TARGET_OBJECTS:python-grpc-library>)

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

install(FILES
          ${GRPC_PY}
          resources/startup.py
        DESTINATION lib/python/runtime)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/triton_python_backend_utils.py
       DESTINATION lib/python/runtime)
