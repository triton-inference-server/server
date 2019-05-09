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
# trtserver
#
find_package(libevhtp CONFIG REQUIRED)
message(STATUS "Using libevhtp ${libevhtp_VERSION}")

set(
  TRTSERVER_SRCS
  main.cc
  grpc_server.cc
  http_server.cc
)

set(
  TRTSERVER_HDRS
  grpc_server.h
  http_server.h
)

add_executable(
  main
  ${TRTSERVER_SRCS} ${TRTSERVER_HDRS}
  $<TARGET_OBJECTS:nvrpc-library>
)
set_property(TARGET main PROPERTY OUTPUT_NAME trtserver)
add_dependencies(main proto-library)
target_include_directories(
  main
  PRIVATE $<TARGET_PROPERTY:libevhtp::evhtp,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(
  main
  trtserver
  libevhtp::evhtp
  -lre2
)
install(
  TARGETS main
  RUNTIME DESTINATION bin
)

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
if(${TRTIS_ENABLE_CUSTOM})
  set(BACKEND_OBJS ${BACKEND_OBJS} $<TARGET_OBJECTS:custom-backend-library>)
endif() # TRTIS_ENABLE_CUSTOM

configure_file(libtrtserver.ldscript libtrtserver.ldscript)

add_library(
  trtserver SHARED
  $<TARGET_OBJECTS:server-library>
  $<TARGET_OBJECTS:model-config-library>
  $<TARGET_OBJECTS:model-config-cuda-library>
  $<TARGET_OBJECTS:request-library>
  $<TARGET_OBJECTS:proto-library>
  $<TARGET_OBJECTS:grpc-library>
  ${BACKEND_OBJS}
)
target_link_libraries(
  trtserver
  ${TRTIS_EXTRA_LDFLAGS}
  -L/usr/local/cuda/lib64/stubs
  -lnvidia-ml
  gRPC::grpc++
  gRPC::grpc
  ${LIBEVENT_LIBRARIES}
  ${CUDA_LIBRARIES}
  "-Wl,--version-script libtrtserver.ldscript"
)

if(${TRTIS_ENABLE_TENSORRT})
  target_link_libraries(
    trtserver
    -lnvinfer
    -lnvonnxparser_runtime
  )
endif() # TRTIS_ENABLE_TENSORRT

if(${TRTIS_ENABLE_TENSORFLOW})
  target_link_libraries(
    trtserver
    ${TRTIS_EXTRA_LDFLAGS}
    -ltensorflow_trtis
  )
endif() # TRTIS_ENABLE_TENSORFLOW

if(${TRTIS_ENABLE_CAFFE2})
  target_link_libraries(
    trtserver
    ${TRTIS_EXTRA_LDFLAGS}
    -lcaffe2
    -lcaffe2_detectron_ops_gpu
    -lcaffe2_gpu
    -lc10
    -lc10_cuda
    -lmkl_core
    -lmkl_gnu_thread
    -lmkl_avx2
    -lmkl_def
    -lmkl_intel_lp64
    -lmkl_rt
    -lmkl_vml_def
  )
endif() # TRTIS_ENABLE_CAFFE2

if(${TRTIS_ENABLE_METRICS})
  find_package(prometheus-cpp CONFIG REQUIRED)
  message(STATUS "Using prometheus-cpp ${prometheus-cpp_VERSION}")
  target_link_libraries(
    trtserver
    prometheus-cpp::core
  )
endif() # TRTIS_ENABLE_METRICS

install(
  TARGETS trtserver
  LIBRARY DESTINATION lib
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/../core/api.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/../core/model_config.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/../core/request_status.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/../core/server_status.pb.h
  DESTINATION include
)

#
# simple_inprocess
#
add_executable(simple_inprocess simple_inprocess.cc)
target_link_libraries(simple_inprocess trtserver)
install(
  TARGETS simple_inprocess
  RUNTIME DESTINATION bin
)
