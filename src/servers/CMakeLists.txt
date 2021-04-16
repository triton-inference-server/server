# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

find_library(RE2_LIBRARY NAMES re2)

#
# grpc endpoint
#
set(GRPC_ENDPOINT_OBJECTS "")
set(GRPC_ENDPOINT_LIBRARIES "")

if(${TRITON_ENABLE_GRPC})
  list(APPEND
    GRPC_ENDPOINT_SRCS
    grpc_server.cc
    )

  list(APPEND
    GRPC_ENDPOINT_HDRS
    grpc_server.h
    )

  add_library(
    grpc-endpoint-library EXCLUDE_FROM_ALL OBJECT
    ${GRPC_ENDPOINT_SRCS} ${GRPC_ENDPOINT_HDRS}
  )

  if(${TRITON_ENABLE_GPU})
    target_include_directories(grpc-endpoint-library PRIVATE ${CUDA_INCLUDE_DIRS})
  endif() # TRITON_ENABLE_GPU

  add_dependencies(grpc-endpoint-library proto-library)
  if(${TRITON_ENABLE_GRPC})
    add_dependencies(grpc-endpoint-library grpc-library)
  endif() # TRITON_ENABLE_GRPC

  target_link_libraries(
    grpc-endpoint-library
    PUBLIC
      triton-core-serverapi   # from repo-core
      triton-common-json      # from repo-common
  )

  set(
    GRPC_ENDPOINT_OBJECTS
    $<TARGET_OBJECTS:grpc-endpoint-library>
    $<TARGET_OBJECTS:proto-library>
    )
  if(${TRITON_ENABLE_GRPC})
    list(APPEND
      GRPC_ENDPOINT_OBJECTS
      $<TARGET_OBJECTS:grpc-library>
      )
  endif() # TRITON_ENABLE_GRPC

  set(
    GRPC_ENDPOINT_LIBRARIES
    PRIVATE gRPC::grpc++
    PRIVATE gRPC::grpc
    PRIVATE protobuf::libprotobuf
    )
endif() # TRITON_ENABLE_GRPC

#
# http endpoint
#
set(HTTP_ENDPOINT_OBJECTS "")
set(HTTP_ENDPOINT_LIBRARIES "")

if(${TRITON_ENABLE_HTTP} OR ${TRITON_ENABLE_METRICS})
  find_package(libevhtp CONFIG REQUIRED)
  message(STATUS "Using libevhtp ${libevhtp_VERSION}")

  if(${TRITON_ENABLE_HTTP} OR ${TRITON_ENABLE_METRICS})
    list(APPEND
      HTTP_ENDPOINT_SRCS
      http_server.cc
      )
    list(APPEND
      HTTP_ENDPOINT_HDRS
      http_server.h
      )
  endif() # TRITON_ENABLE_HTTP || TRITON_ENABLE_METRICS

  add_library(
    http-endpoint-library EXCLUDE_FROM_ALL OBJECT
    ${HTTP_ENDPOINT_SRCS} ${HTTP_ENDPOINT_HDRS}
  )

  if(${TRITON_ENABLE_GPU})
    target_include_directories(http-endpoint-library PRIVATE ${CUDA_INCLUDE_DIRS})
  endif() # TRITON_ENABLE_GPU

  add_dependencies(http-endpoint-library proto-library)
  target_include_directories(
    http-endpoint-library
    PRIVATE $<TARGET_PROPERTY:libevhtp::evhtp,INTERFACE_INCLUDE_DIRECTORIES>
  )

  target_link_libraries(
    http-endpoint-library
    PUBLIC
      triton-core-serverapi   # from repo-core
      triton-common-json      # from repo-common
  )

  set(
    HTTP_ENDPOINT_OBJECTS
    $<TARGET_OBJECTS:http-endpoint-library>
    $<TARGET_OBJECTS:proto-library>
    $<TARGET_OBJECTS:model-config-library>
  )

  set(
    HTTP_ENDPOINT_LIBRARIES
    PRIVATE ${LIBEVENT_LIBRARIES}
    PRIVATE libevhtp::evhtp
    PRIVATE protobuf::libprotobuf
    PRIVATE ${RE2_LIBRARY}
    PRIVATE b64
  )
endif() # TRITON_ENABLE_HTTP || TRITON_ENABLE_METRICS

#
# tracing
#
set(TRACING_OBJECTS "")
set(TRACING_LIBRARIES "")

if(${TRITON_ENABLE_TRACING})
  message(STATUS "Using tracing ${TRITON_TRACE_INSTALL_PATH}")

  set(
    TRACING_SRCS
    tracer.cc
  )

  set(
    TRACING_HDRS
    tracer.h
  )

  add_library(
    tracing-library EXCLUDE_FROM_ALL OBJECT
    ${TRACING_SRCS} ${TRACING_HDRS}
  )
  if(${TRITON_ENABLE_GPU})
    target_include_directories(tracing-library PRIVATE ${CUDA_INCLUDE_DIRS})
  endif() # TRITON_ENABLE_GPU
  target_link_libraries(
    tracing-library
    PRIVATE triton-core-serverapi
  )

  set(
    TRACING_OBJECTS
    $<TARGET_OBJECTS:tracing-library>
  )
endif() # TRITON_ENABLE_TRACING

#
# tritonserver
#
set(
  TRITONSERVER_SRCS
  classification.cc
  common.cc
  main.cc
  shared_memory_manager.cc
  signal.cc
  ../core/logging.cc
)

set(
  TRITONSERVER_HDRS
  classification.h
  common.h
  shared_memory_manager.h
  signal.h
  ../core/logging.h
)

add_executable(
  main
  ${TRITONSERVER_SRCS}
  ${TRITONSERVER_HDRS}
  ${HTTP_ENDPOINT_OBJECTS}
  ${GRPC_ENDPOINT_OBJECTS}
  ${TRACING_OBJECTS}
  $<TARGET_OBJECTS:proto-library>
)
set_property(TARGET main PROPERTY OUTPUT_NAME tritonserver)
if(${TRITON_ENABLE_GPU})
  target_include_directories(main PRIVATE ${CUDA_INCLUDE_DIRS})
endif() # TRITON_ENABLE_GPU
set_target_properties(
  main
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}/../lib"
)
target_link_libraries(
  main
  ${HTTP_ENDPOINT_LIBRARIES}
  ${GRPC_ENDPOINT_LIBRARIES}
  ${TRACING_LIBRARIES}
  PRIVATE protobuf::libprotobuf
  PRIVATE triton-common-async-work-queue
  PRIVATE triton-common-error
  PRIVATE triton-core-serverapi
  PRIVATE tritonserver
)
if (NOT WIN32)
target_link_libraries(
  main
  PRIVATE rt
)
endif()
if(${TRITON_ENABLE_GPU})
target_link_libraries(
  main
  PRIVATE ${CUDA_LIBRARIES}
)
endif() # TRITON_ENABLE_GPU
install(
  TARGETS main
  RUNTIME DESTINATION bin
)

if (NOT WIN32)
#
# simple
#
set(
  SIMPLE_SRCS
  simple.cc
)

set(
  SIMPLE_HDRS
)

add_executable(
  simple
  ${SIMPLE_SRCS}
  ${SIMPLE_HDRS}
  $<TARGET_OBJECTS:proto-library>
)
set_target_properties(
  simple
  PROPERTIES
    COMPILE_FLAGS "-Wno-type-limits"
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)
target_link_libraries(
  simple
  PRIVATE triton-common-async-work-queue  # from repo-common
  PRIVATE triton-common-error             # from repo-common
  PRIVATE triton-core-serverapi           # from repo-core
  PRIVATE tritonserver
  PRIVATE protobuf::libprotobuf
  )

if(${TRITON_ENABLE_GPU})
target_include_directories(simple PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(
  simple
  PUBLIC ${CUDA_nvml_LIBRARY}
  PRIVATE ${CUDA_LIBRARIES}
)
endif() # TRITON_ENABLE_GPU
install(
  TARGETS simple
  RUNTIME DESTINATION bin
)

#
# multi_server
#
set(
  MULTISERVER_SRCS
  multi_server.cc
)

add_executable(
  multi_server
  ${MULTISERVER_SRCS}
  $<TARGET_OBJECTS:proto-library>
)
set_target_properties(
  multi_server
  PROPERTIES
    COMPILE_FLAGS "-Wno-type-limits"
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)
target_link_libraries(
  multi_server
  PRIVATE triton-common-async-work-queue # from repo-common
  PRIVATE triton-common-error             # from repo-common
  PRIVATE triton-core-serverapi          # from repo-core
  PRIVATE tritonserver
  PRIVATE protobuf::libprotobuf
  )

if(${TRITON_ENABLE_GPU})
target_include_directories(multi_server PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(
  multi_server
  PUBLIC -L/usr/local/cuda/lib64/stubs
  PUBLIC -lnvidia-ml
  PRIVATE ${CUDA_LIBRARIES}
)
endif() # TRITON_ENABLE_GPU
install(
  TARGETS multi_server
  RUNTIME DESTINATION bin
)

#
# memory_alloc
#
if(${TRITON_ENABLE_GPU})
set(
  MEMORY_ALLOC_SRCS
  memory_alloc.cc
)

set(
  MEMORY_ALLOC_HDRS
)

add_executable(
  memory_alloc
  ${MEMORY_ALLOC_SRCS}
  ${MEMORY_ALLOC_HDRS}
  $<TARGET_OBJECTS:proto-library>
)
target_include_directories(memory_alloc PRIVATE ${CUDA_INCLUDE_DIRS})
set_target_properties(
  memory_alloc
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)
target_link_libraries(
  memory_alloc
  PUBLIC ${CUDA_nvml_LIBRARY}
  PRIVATE ${CUDA_LIBRARIES}
  PRIVATE triton-core-serverapi   # from repo-core
  PRIVATE tritonserver
  PRIVATE protobuf::libprotobuf
)
install(
  TARGETS memory_alloc
  RUNTIME DESTINATION bin
)
endif() # TRITON_ENABLE_GPU
endif() # NOT WIN32

#
# libtritonserver.so
#
if(${TRITON_ENABLE_TENSORFLOW})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:tensorflow-backend-library>)
endif() # TRITON_ENABLE_TENSORFLOW
if(${TRITON_ENABLE_TENSORRT})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:tensorrt-backend-library>)
endif() # TRITON_ENABLE_TENSORRT
if(${TRITON_ENABLE_ONNXRUNTIME})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:onnxruntime-backend-library>)
endif() # TRITON_ENABLE_ONNXRUNTIME
if(${TRITON_ENABLE_PYTORCH})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:libtorch-backend-library>)
endif() # TRITON_ENABLE_PYTORCH
if(${TRITON_ENABLE_CUSTOM})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:custom-backend-library>)
endif() # TRITON_ENABLE_CUSTOM
if(${TRITON_ENABLE_ENSEMBLE})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:ensemble-backend-library>)
endif() # TRITON_ENABLE_ENSEMBLE

set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:triton-backend-library>)
set(REPOAGENT_OBJS ${REPOAGENT_OBJS} $<TARGET_OBJECTS:triton-repoagent-library>)

set(CUDA_OBJS "")
if(${TRITON_ENABLE_GPU})
  set(CUDA_OBJS $<TARGET_OBJECTS:model-config-cuda-library>)
endif() # TRITON_ENABLE_GPU

configure_file(libtritonserver.ldscript libtritonserver.ldscript COPYONLY)

add_library(
  tritonserver SHARED
  $<TARGET_OBJECTS:server-library>
  $<TARGET_OBJECTS:model-config-library>
  $<TARGET_OBJECTS:proto-library>
  ${CUDA_OBJS}
  ${BACKEND_OBJS}
  ${REPOAGENT_OBJS}
)
# FIXME need to check how Windows exposes symbols
if (WIN32)
set_target_properties(
  tritonserver
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}:$\{ORIGIN\}/pytorch"
)
else ()
set_target_properties(
  tritonserver
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}:$\{ORIGIN\}/pytorch"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtritonserver.ldscript
    LINK_FLAGS "-Wl,--version-script libtritonserver.ldscript"
)
target_link_libraries(
  tritonserver
  PRIVATE dl
)
endif()
target_link_libraries(
  tritonserver
  PRIVATE triton-core-backendapi
  PRIVATE triton-core-repoagentapi
  PRIVATE triton-core-serverapi
  PRIVATE triton-common-async-work-queue
  PRIVATE triton-common-error
  PRIVATE triton-common-json
  PRIVATE triton-common-table-printer
  PRIVATE ${TRITON_EXTRA_LDFLAGS}
  PRIVATE protobuf::libprotobuf
)

if(${TRITON_ENABLE_TENSORRT})
  find_library(NVINFER_LIBRARY NAMES nvinfer)
  find_library(NVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin)
  target_link_libraries(
    tritonserver
    PUBLIC ${NVINFER_LIBRARY}
    PUBLIC ${NVINFER_PLUGIN_LIBRARY}
  )
endif() # TRITON_ENABLE_TENSORRT

if(${TRITON_ENABLE_METRICS})
  find_package(prometheus-cpp CONFIG REQUIRED)
  message(STATUS "Using prometheus-cpp ${prometheus-cpp_VERSION}")
  target_link_libraries(
    tritonserver
    PRIVATE prometheus-cpp::core
  )
endif() # TRITON_ENABLE_METRICS

if(${TRITON_ENABLE_GCS})
  find_package(storage_client REQUIRED)
  message(STATUS "Using google-cloud-cpp ${storage_client_VERSION}")
  target_link_libraries(
    tritonserver
    PRIVATE storage_client
  )
endif() # TRITON_ENABLE_GCS

if(${TRITON_ENABLE_S3})
  find_package(AWSSDK REQUIRED COMPONENTS s3)
  message(STATUS "Using aws-cpp-sdk-s3 ${AWSSDK_VERSION}")
  target_link_libraries(
    tritonserver
    PRIVATE aws-cpp-sdk-s3
    PRIVATE ${RE2_LIBRARY}
  )
endif() # TRITON_ENABLE_S3

if(${TRITON_ENABLE_AZURE_STORAGE})
  target_link_libraries(
    tritonserver
    PRIVATE -L${azure-storage-cpplite_DIR}/lib
    PRIVATE -lazure-storage-lite
    PRIVATE -lre2
    PRIVATE -luuid
    PRIVATE -lcurl
  )
endif() # TRITON_ENABLE_AZURE_STORAGE

if(${TRITON_ENABLE_GPU})
  find_library(CNMEM_LIBRARY NAMES cnmem PATHS ${CNMEM_PATH}/lib)
  target_link_libraries(
    tritonserver
    PUBLIC ${CUDA_nvml_LIBRARY}
    PRIVATE ${CUDA_LIBRARIES}
    PRIVATE ${CNMEM_LIBRARY}
  )
  target_link_libraries(
    tritonserver
    PRIVATE kernel-library
  )
endif() # TRITON_ENABLE_GPU

install(
  TARGETS tritonserver
  LIBRARY DESTINATION lib
)
