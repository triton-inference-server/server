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
file(GLOB proto-srcs *.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${proto-srcs})
protobuf_generate_python(PROTO_PY ${proto-srcs})

set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

add_library(
  proto-library EXCLUDE_FROM_ALL OBJECT
  ${PROTO_SRCS} ${PROTO_HDRS}
)

add_custom_target(proto-py-library DEPENDS ${PROTO_PY})

#
# GRPC
#
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
  DEPENDS "grpc_service.proto" proto-library
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
  DEPENDS "grpc_service.proto" proto-library
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

if(${TRTIS_ENABLE_GPU})
  add_library(
    model-config-cuda-library EXCLUDE_FROM_ALL OBJECT
    model_config_cuda.cc model_config_cuda.h
  )
  target_include_directories(model-config-cuda-library PRIVATE ${CUDA_INCLUDE_DIRS})
  add_dependencies(model-config-cuda-library proto-library)
endif() # TRTIS_ENABLE_GPU

#
# Inference server core
#
if(${TRTIS_ENABLE_METRICS})
  find_package(prometheus-cpp CONFIG REQUIRED)
  message(STATUS "Using prometheus-cpp ${prometheus-cpp_VERSION}")
endif() # TRTIS_ENABLE_METRICS

if(${TRTIS_ENABLE_GCS})
  find_package(storage_client REQUIRED)
  message(STATUS "Using google-cloud-cpp ${storage_client_VERSION}")
  set_source_files_properties(filesystem.cc PROPERTIES COMPILE_FLAGS -Wno-missing-field-initializers)
endif() # TRTIS_ENABLE_GCS

if(${TRTIS_ENABLE_S3})
  find_package(AWSSDK REQUIRED COMPONENTS s3)
  message(STATUS "Using aws-sdk-cpp ${AWSSDK_VERSION}")
endif()

set(
  SERVER_SRCS
  autofill.cc
  backend.cc
  backend_context.cc
  dynamic_batch_scheduler.cc
  ensemble_scheduler.cc
  ensemble_utils.cc
  filesystem.cc
  label_provider.cc
  logging.cc
  metric_model_reporter.cc
  metrics.cc
  model_config_utils.cc
  model_repository_manager.cc
  tracing.cc
  provider.cc
  provider_utils.cc
  sequence_batch_scheduler.cc
  server.cc
  server_status.cc
  shared_memory_manager.cc
  status.cc
  trtserver.cc
)

set(
  SERVER_HDRS
  autofill.h
  backend.h
  backend_context.h
  constants.h
  dynamic_batch_scheduler.h
  ensemble_scheduler.h
  ensemble_utils.h
  filesystem.h
  label_provider.h
  logging.h
  metric_model_reporter.h
  metrics.h
  model_config_utils.h
  model_repository_manager.h
  tracing.h
  provider.h
  provider_utils.h
  scheduler.h
  sequence_batch_scheduler.h
  server.h
  server_status.h
  shared_memory_manager.h
  status.h
  trtserver.h
)

add_library(
  server-library EXCLUDE_FROM_ALL OBJECT
  ${SERVER_SRCS} ${SERVER_HDRS}
)
add_dependencies(server-library proto-library)

if(${TRTIS_ENABLE_GPU})
  target_include_directories(
    server-library
    PRIVATE ${CUDA_INCLUDE_DIRS}
  )
endif() # TRTIS_ENABLE_GPU

if(${TRTIS_ENABLE_METRICS})
  target_include_directories(
    server-library
    PRIVATE $<TARGET_PROPERTY:prometheus-cpp::core,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif() # TRTIS_ENABLE_METRICS

if(${TRTIS_ENABLE_GCS})
  target_include_directories(
    server-library
    PRIVATE $<TARGET_PROPERTY:storage_client,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif() # TRTIS_ENABLE_GCS

if(${TRTIS_ENABLE_S3})
  target_include_directories(
    server-library
    PRIVATE $<TARGET_PROPERTY:aws-cpp-sdk-s3,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif() # TRTIS_ENABLE_S3
