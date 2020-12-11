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

FetchContent_Declare(tensorflow-serving-repo
  PREFIX tensorflow-serving-rep
)
FetchContent_GetProperties(tensorflow-serving-repo)
if(NOT tensorflow-serving-repo_POPULATED)
  FetchContent_Populate(tensorflow-serving-repo
  GIT_REPOSITORY "https://github.com/tensorflow/serving.git"
  GIT_TAG "2.3.0"
 SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-serving-repo/src/tensorflow_serving"
)
endif()

FetchContent_Declare(tensorflow-repo
  PREFIX tensorflow-repo
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-repo/src/tensorflow"
)
FetchContent_GetProperties(tensorflow-repo)
if(NOT tensorflow-repo_POPULATED)
  FetchContent_Populate(tensorflow-repo
  GIT_REPOSITORY "https://github.com/tensorflow/tensorflow.git"
  GIT_TAG "v2.3.0"
 SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-repo/src/tensorflow"
)
endif()


set(TENSORFLOW_PATH ${CMAKE_CURRENT_BINARY_DIR}/tensorflow-repo/src/tensorflow)
set(TFSERVE_PATH ${CMAKE_CURRENT_BINARY_DIR}/tensorflow-serving-repo/src/tensorflow_serving)

# Copy the repos to a proto staging area.
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/protos)
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${TENSORFLOW_PATH}/tensorflow
                                                           ${CMAKE_BINARY_DIR}/protos/tensorflow)
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${TFSERVE_PATH}/tensorflow_serving
                                                           ${CMAKE_BINARY_DIR}/protos/tensorflow_serving)

# Protobuf compiler dependency.
include(CompileProto.cmake)

# Protobuf sources of the TensorFlow Serving to be compiled without a gRPC plugin.
file(GLOB_RECURSE TFSERVING_PROTOS ${CMAKE_BINARY_DIR}/protos/tensorflow_serving/*.proto)
file(GLOB TF_EXAMPLE_PROTOS ${CMAKE_BINARY_DIR}/protos/tensorflow/core/example/*.proto)
file(GLOB TF_FW_PROTOS ${CMAKE_BINARY_DIR}/protos/tensorflow/core/framework/*.proto)
file(GLOB TF_PROTOBUF_PROTOS ${CMAKE_BINARY_DIR}/protos/tensorflow/core/protobuf/*.proto)

# This is a dirty hack to prevent unneccesary leaking dependency
list(FILTER TF_PROTOBUF_PROTOS EXCLUDE REGEX "autotuning.proto$|conv_autotuning.proto$")

# Compiling CPP sources from proto files.
compile_proto(0 "${CMAKE_BINARY_DIR}/protos" "${CMAKE_CURRENT_BINARY_DIR}/compiled" PB_SOURCES PB_HEADERS
        ${TFSERVING_PROTOS}  ${TF_EXAMPLE_PROTOS} ${TF_FW_PROTOS} ${TF_PROTOBUF_PROTOS})

# Compiling CPP sources with gRPC plugin.
compile_proto(1 "${CMAKE_BINARY_DIR}/protos" "${CMAKE_CURRENT_BINARY_DIR}/compiled" PB_GRPC_SOURCES PB_GRPC_HEADERS
        ${CMAKE_BINARY_DIR}/protos/tensorflow_serving/apis/prediction_service.proto)

set(
    TFS_CLIENT_BACKEND_SRCS
    tfserve_client_backend.cc
    tfserve_infer_input.cc
    grpc_client.cc
    ${PB_SOURCES}
    ${PB_GRPC_SOURCES}
)

set(
    TFS_CLIENT_BACKEND_HDRS
    tfserve_client_backend.h
    tfserve_infer_input.h
    grpc_client.h
    ${PB_HEADERS} 
    ${PB_GRPC_HEADERS} 
)

add_library(
    tfs-client-backend-library  EXCLUDE_FROM_ALL OBJECT
    ${TFS_CLIENT_BACKEND_SRCS}
    ${TFS_CLIENT_BACKEND_HDRS}
)

target_include_directories(tfs-client-backend-library PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/compiled)

target_link_libraries(
  tfs-client-backend-library
  PUBLIC gRPC::grpc++
  PUBLIC gRPC::grpc
  PUBLIC protobuf::libprotobuf
  PUBLIC TRITON::grpcclient_static
)

if(${TRITON_ENABLE_GPU})
    target_include_directories(tfs-client-backend-library PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(tfs-client-backend-library PRIVATE ${CUDA_LIBRARIES})
endif() # TRITON_ENABLE_GPU
