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
# grpc endpoint
#
set(GRPC_ENDPOINT_OBJECTS "")
set(GRPC_ENDPOINT_LIBRARIES "")

if(${TRTIS_ENABLE_GRPC})
  set(
    GRPC_ENDPOINT_SRCS
    grpc_server.cc
  )

  set(
    GRPC_ENDPOINT_HDRS
    grpc_server.h
  )

  add_library(
    grpc-endpoint-library EXCLUDE_FROM_ALL OBJECT
    ${GRPC_ENDPOINT_SRCS} ${GRPC_ENDPOINT_HDRS}
  )
  if(${TRTIS_ENABLE_GPU})
    target_include_directories(grpc-endpoint-library PRIVATE ${CUDA_INCLUDE_DIRS})
  endif() # TRTIS_ENABLE_GPU
  add_dependencies(grpc-endpoint-library grpc-library proto-library)

  set(
    GRPC_ENDPOINT_OBJECTS
    $<TARGET_OBJECTS:grpc-endpoint-library>
    $<TARGET_OBJECTS:grpc-library>
    $<TARGET_OBJECTS:proto-library>
  )
  set(
    GRPC_ENDPOINT_LIBRARIES
    PRIVATE gRPC::grpc++
    PRIVATE gRPC::grpc
    PRIVATE protobuf::libprotobuf
  )
endif() # TRTIS_ENABLE_GRPC

#
# http endpoint
#
set(HTTP_ENDPOINT_OBJECTS "")
set(HTTP_ENDPOINT_LIBRARIES "")

if(${TRTIS_ENABLE_HTTP} OR ${TRTIS_ENABLE_METRICS})
  find_package(libevhtp CONFIG REQUIRED)
  message(STATUS "Using libevhtp ${libevhtp_VERSION}")

  if(${TRTIS_ENABLE_HTTP_V2})
    set(
      HTTP_ENDPOINT_SRCS
      http_v2_server.cc
    )
  else()
    set(
      HTTP_ENDPOINT_SRCS
      http_server.cc
    )
  endif() # TRTIS_ENABLE_HTTP_V2

  set(
    HTTP_ENDPOINT_HDRS
    http_server.h
  )

  add_library(
    http-endpoint-library EXCLUDE_FROM_ALL OBJECT
    ${HTTP_ENDPOINT_SRCS} ${HTTP_ENDPOINT_HDRS}
  )

  if(${TRTIS_ENABLE_GPU})
    target_include_directories(http-endpoint-library PRIVATE ${CUDA_INCLUDE_DIRS})
  endif() # TRTIS_ENABLE_GPU

  add_dependencies(http-endpoint-library proto-library)
  target_include_directories(
    http-endpoint-library
    PRIVATE $<TARGET_PROPERTY:libevhtp::evhtp,INTERFACE_INCLUDE_DIRECTORIES>
  )

  set(
    HTTP_ENDPOINT_OBJECTS
    $<TARGET_OBJECTS:http-endpoint-library>
    $<TARGET_OBJECTS:proto-library>
  )
  if(${TRTIS_ENABLE_HTTP_V2})
    set(
      HTTP_ENDPOINT_OBJECTS 
      ${HTTP_ENDPOINT_OBJECTS} 
      $<TARGET_OBJECTS:grpc-library>
    )
  endif() # TRTIS_ENABLE_HTTP_V2
  set(
    HTTP_ENDPOINT_LIBRARIES
    PRIVATE ${LIBEVENT_LIBRARIES}
    PRIVATE libevhtp::evhtp
    PRIVATE protobuf::libprotobuf
    PRIVATE -lre2
  )
  if(${TRTIS_ENABLE_HTTP_V2})
    set(
      HTTP_ENDPOINT_LIBRARIES 
      ${HTTP_ENDPOINT_LIBRARIES} 
      PRIVATE gRPC::grpc++
      PRIVATE gRPC::grpc
    )
  endif() # TRTIS_ENABLE_HTTP_V2
endif() # TRTIS_ENABLE_HTTP || TRTIS_ENABLE_METRICS

#
# tracing
#
set(TRACING_OBJECTS "")
set(TRACING_LIBRARIES "")

if(${TRTIS_ENABLE_TRACING})
  message(STATUS "Using tracing ${TRTIS_TRACE_INSTALL_PATH}")

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
  if(${TRTIS_ENABLE_GPU})
    target_include_directories(tracing-library PRIVATE ${CUDA_INCLUDE_DIRS})
  endif() # TRTIS_ENABLE_GPU
  set(
    TRACING_OBJECTS
    $<TARGET_OBJECTS:tracing-library>
  )
endif() # TRTIS_ENABLE_TRACING

#
# trtserver
#
set(
  TRTSERVER_SRCS
  main.cc
  common.cc
  shared_memory_block_manager.cc
  ../core/logging.cc
)

set(
  TRTSERVER_HDRS
  common.h
  shared_memory_block_manager.h
  ../core/logging.h
)

add_executable(
  main
  ${TRTSERVER_SRCS}
  ${TRTSERVER_HDRS}
  ${HTTP_ENDPOINT_OBJECTS}
  ${GRPC_ENDPOINT_OBJECTS}
  ${TRACING_OBJECTS}
  $<TARGET_OBJECTS:proto-library>
)
set_property(TARGET main PROPERTY OUTPUT_NAME trtserver)
if(${TRTIS_ENABLE_GPU})
  target_include_directories(main PRIVATE ${CUDA_INCLUDE_DIRS})
endif() # TRTIS_ENABLE_GPU
target_link_libraries(
  main
  ${HTTP_ENDPOINT_LIBRARIES}
  ${GRPC_ENDPOINT_LIBRARIES}
  ${TRACING_LIBRARIES}
  PRIVATE protobuf::libprotobuf
  PRIVATE trtserver
)
install(
  TARGETS main
  RUNTIME DESTINATION bin
)

#
# simple
#
set(
  SIMPLE_SRCS
  simple.cc common.cc ../core/logging.cc
)

set(
  SIMPLE_HDRS
  common.h ../core/logging.h
)

add_executable(
  simple
  ${SIMPLE_SRCS}
  ${SIMPLE_HDRS}
  $<TARGET_OBJECTS:proto-library>
  ${TRACING_OBJECTS}
)
target_link_libraries(
  simple
  PRIVATE trtserver
  PRIVATE protobuf::libprotobuf
  ${TRACING_LIBRARIES}
)
if(${TRTIS_ENABLE_GPU})
target_include_directories(simple PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(
  simple
  PUBLIC -L/usr/local/cuda/lib64/stubs
  PUBLIC -lnvidia-ml
  PRIVATE ${CUDA_LIBRARIES}
)
endif() # TRTIS_ENABLE_GPU
install(
  TARGETS simple
  RUNTIME DESTINATION bin
)

#
# memory_alloc (only build when TRTIS_ENABLE_GPU given the purpose)
#
if(${TRTIS_ENABLE_GPU})
set(
  MEMORY_ALLOC_SRCS
  memory_alloc.cc common.cc ../core/logging.cc
)

set(
  MEMORY_ALLOC_HDRS
  common.h ../core/logging.h
)

add_executable(
  memory_alloc
  ${MEMORY_ALLOC_SRCS}
  ${MEMORY_ALLOC_HDRS}
  $<TARGET_OBJECTS:proto-library>
  ${TRACING_OBJECTS}
)
target_link_libraries(
  memory_alloc
  PUBLIC -L/usr/local/cuda/lib64/stubs
  PUBLIC -lnvidia-ml
  PRIVATE ${CUDA_LIBRARIES}
  PRIVATE trtserver
  PRIVATE protobuf::libprotobuf
  ${TRACING_LIBRARIES}
)
target_include_directories(memory_alloc PRIVATE ${CUDA_INCLUDE_DIRS})
install(
  TARGETS memory_alloc
  RUNTIME DESTINATION bin
)
endif() # TRTIS_ENABLE_GPU

#
# libtrtserver.so
#
set(BACKEND_OBJS $<TARGET_OBJECTS:ensemble-backend-library>)
if(${TRTIS_ENABLE_TENSORFLOW})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:tensorflow-backend-library>)
endif() # TRTIS_ENABLE_TENSORFLOW
if(${TRTIS_ENABLE_TENSORRT})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:tensorrt-backend-library>)
endif() # TRTIS_ENABLE_TENSORRT
if(${TRTIS_ENABLE_CAFFE2})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:caffe2-backend-library>)
endif() # TRTIS_ENABLE_CAFFE2
if(${TRTIS_ENABLE_ONNXRUNTIME})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:onnxruntime-backend-library>)
endif() # TRTIS_ENABLE_ONNXRUNTIME
if(${TRTIS_ENABLE_PYTORCH})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:libtorch-backend-library>)
endif() # TRTIS_ENABLE_PYTORCH
if(${TRTIS_ENABLE_CUSTOM})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:custom-backend-library>)
endif() # TRTIS_ENABLE_CUSTOM

set(CUDA_OBJS "")
if(${TRTIS_ENABLE_GPU})
  set(CUDA_OBJS $<TARGET_OBJECTS:model-config-cuda-library>)
endif() # TRTIS_ENABLE_GPU

configure_file(libtrtserver.ldscript libtrtserver.ldscript COPYONLY)

add_library(
  trtserver SHARED
  $<TARGET_OBJECTS:server-library>
  $<TARGET_OBJECTS:model-config-library>
  $<TARGET_OBJECTS:proto-library>
  ${CUDA_OBJS}
  ${BACKEND_OBJS}
)
set_target_properties(
  trtserver
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtrtserver.ldscript
)
set_target_properties(
  trtserver
  PROPERTIES LINK_FLAGS "-Wl,--version-script libtrtserver.ldscript"
)
target_link_libraries(
  trtserver
  PRIVATE -ldl
  PRIVATE -lrt
  PRIVATE ${TRTIS_EXTRA_LDFLAGS}
  PRIVATE protobuf::libprotobuf
)

if(${TRTIS_ENABLE_TENSORRT})
  target_link_libraries(
    trtserver
    PUBLIC -lnvinfer
    PUBLIC -lnvinfer_plugin
    PUBLIC -lnvonnxparser_runtime
  )
endif() # TRTIS_ENABLE_TENSORRT

if(${TRTIS_ENABLE_TENSORFLOW})
  target_link_libraries(
    trtserver
    PUBLIC ${TRTIS_EXTRA_LDFLAGS}
    PUBLIC -ltensorflow_cc
  )
endif() # TRTIS_ENABLE_TENSORFLOW

if(${TRTIS_ENABLE_CAFFE2} OR ${TRTIS_ENABLE_PYTORCH})
  target_link_libraries(
    trtserver
    PUBLIC ${TRTIS_EXTRA_LDFLAGS}
    PUBLIC -ltorch
    PUBLIC -lcaffe2_detectron_ops_gpu
    PUBLIC -lcaffe2_nvrtc
    PUBLIC -lc10
    PUBLIC -lc10_cuda
    PUBLIC -lmkl_core
    PUBLIC -lmkl_gnu_thread
    PUBLIC -lmkl_avx2
    PUBLIC -lmkl_def
    PUBLIC -lmkl_intel_lp64
    PUBLIC -lmkl_rt
    PUBLIC -lmkl_vml_def
  )
endif() # TRTIS_ENABLE_CAFFE2

if(${TRTIS_ENABLE_ONNXRUNTIME})
  target_link_libraries(
    trtserver
    PUBLIC ${TRTIS_EXTRA_LDFLAGS}
    PUBLIC -lonnxruntime
  )
endif() # TRTIS_ENABLE_ONNXRUNTIME
if(${TRTIS_ENABLE_ONNXRUNTIME_OPENVINO})
  target_link_libraries(
    trtserver
    PUBLIC ${TRTIS_EXTRA_LDFLAGS}
    PUBLIC -ltbb
    PUBLIC -linference_engine
  )
endif() # TRTIS_ENABLE_ONNXRUNTIME_OPENVINO

if(${TRTIS_ENABLE_METRICS})
  find_package(prometheus-cpp CONFIG REQUIRED)
  message(STATUS "Using prometheus-cpp ${prometheus-cpp_VERSION}")
  target_link_libraries(
    trtserver
    PRIVATE prometheus-cpp::core
  )
endif() # TRTIS_ENABLE_METRICS

if(${TRTIS_ENABLE_GCS})
  find_package(storage_client REQUIRED)
  message(STATUS "Using google-cloud-cpp ${storage_client_VERSION}")
  target_link_libraries(
    trtserver
    PRIVATE storage_client
  )
endif() # TRTIS_ENABLE_GCS

if(${TRTIS_ENABLE_S3})
  find_package(AWSSDK REQUIRED COMPONENTS s3)
  message(STATUS "Using aws-cpp-sdk-s3 ${AWSSDK_VERSION}")
  target_link_libraries(
    trtserver
    PRIVATE aws-cpp-sdk-s3
  )

endif() # TRTIS_ENABLE_S3

if(${TRTIS_ENABLE_GPU})
  target_link_libraries(
    trtserver
    PUBLIC -L/usr/local/cuda/lib64/stubs
    PUBLIC -lnvidia-ml
    PRIVATE ${CUDA_LIBRARIES}
  )
endif() # TRTIS_ENABLE_GPU

install(
  TARGETS trtserver
  LIBRARY DESTINATION lib
)

install(
  FILES ../core/trtserver.h
  DESTINATION include
)
