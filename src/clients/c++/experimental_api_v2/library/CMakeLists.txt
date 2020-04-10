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

find_package(Threads REQUIRED)

if(${TRTIS_ENABLE_GRPC_V2})
  #
  # libgrpcclient.so and libgrpcclient_static.a
  #
  configure_file(libgrpcclient.ldscript libgrpcclient.ldscript COPYONLY)

  # libgrpcclient object build
  set(
    REQUEST_SRCS
    grpc_client.cc utils.cc
  )

  set(
    REQUEST_HDRS
    grpc_client.h utils.h
  )

  add_library(
    grpc-client-library EXCLUDE_FROM_ALL OBJECT
    ${REQUEST_SRCS} ${REQUEST_HDRS}
  )
  if(${TRTIS_ENABLE_GPU})
    target_include_directories(grpc-client-library PUBLIC ${CUDA_INCLUDE_DIRS})
  endif() # TRTIS_ENABLE_GPU
  add_dependencies(
    grpc-client-library
    model-config-library grpc-v2-library proto-library
  )

  # libgrpcclient.a
  add_library(
    grpcclient_static STATIC
    $<TARGET_OBJECTS:grpc-v2-library>
    $<TARGET_OBJECTS:model-config-library>
    $<TARGET_OBJECTS:proto-library>
    $<TARGET_OBJECTS:grpc-client-library>
  )
  add_library(
    TRTIS::grpcclient_static ALIAS grpcclient_static
  )
  if(${TRTIS_ENABLE_GPU})
    target_include_directories(grpcclient_static PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(grpcclient_static PRIVATE ${CUDA_LIBRARIES})
  endif() # TRTIS_ENABLE_GPU
  target_include_directories(
    grpcclient_static
    PUBLIC
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
  target_link_libraries(
    grpcclient_static
    PRIVATE gRPC::grpc++
    PRIVATE gRPC::grpc
    PUBLIC protobuf::libprotobuf
    PUBLIC Threads::Threads
  )

  # libgrpcclient.so
  add_library(
    grpcclient SHARED
    $<TARGET_OBJECTS:grpc-v2-library>
    $<TARGET_OBJECTS:model-config-library>
    $<TARGET_OBJECTS:proto-library>
    $<TARGET_OBJECTS:grpc-client-library>
  )
  add_library(
    TRTIS::grpcclient ALIAS grpcclient
  )

  if (NOT WIN32)
  set_target_properties(
    grpcclient
    PROPERTIES
      LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libgrpcclient.ldscript
      LINK_FLAGS "-Wl,--version-script=libgrpcclient.ldscript"
  )
  endif() # NOT WIN32

  if(${TRTIS_ENABLE_GPU})
    target_include_directories(grpcclient PUBLIC ${CUDA_INCLUDE_DIRS})
  endif() # TRTIS_ENABLE_GPU
  target_include_directories (grpcclient
    PUBLIC
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
  target_link_libraries(
    grpcclient
    PRIVATE gRPC::grpc++
    PRIVATE gRPC::grpc
    PUBLIC protobuf::libprotobuf
    PUBLIC Threads::Threads
  )

  install(
    TARGETS
      grpcclient
      grpcclient_static
    EXPORT trtis-export
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../core/model_config.pb.h
    DESTINATION include
  )

endif()  # TRTIS_ENABLE_GRPC_V2

if(${TRTIS_ENABLE_HTTP_V2})
  #
  # libhttpclient.so and libhttpclient_static.a
  #

  find_package(RapidJSON CONFIG REQUIRED)
  
  find_package(CURL CONFIG REQUIRED)
  message(STATUS "Using curl ${CURL_VERSION}")
  add_definitions(-DCURL_STATICLIB)

  configure_file(libhttpclient.ldscript libhttpclient.ldscript COPYONLY)

  # libhttpclient object build
  set(
    REQUEST_SRCS
    http_client.cc utils.cc
  )

  set(
    REQUEST_HDRS
    http_client.h utils.h
  )

  add_library(
    http-client-library EXCLUDE_FROM_ALL OBJECT
    ${REQUEST_SRCS} ${REQUEST_HDRS}
  )

  target_include_directories(http-client-library 
    PRIVATE $<TARGET_PROPERTY:CURL::libcurl,INTERFACE_INCLUDE_DIRECTORIES>
    PUBLIC ${RapidJSON_INCLUDE_DIRS})
  if(${TRTIS_ENABLE_GPU})
    target_include_directories(http-client-library PUBLIC ${CUDA_INCLUDE_DIRS})
  endif() # TRTIS_ENABLE_GPU

  # libhttpclient.a
  add_library(
    httpclient_static STATIC
    $<TARGET_OBJECTS:http-client-library>
  )
  add_library(
    TRTIS::httpclient_static ALIAS httpclient_static
  )
  if(${TRTIS_ENABLE_GPU})
    target_include_directories(httpclient_static PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(httpclient_static PRIVATE ${CUDA_LIBRARIES})
  endif() # TRTIS_ENABLE_GPU
  target_include_directories(
    httpclient_static
    PUBLIC
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
  target_link_libraries(
    httpclient_static
    PRIVATE CURL::libcurl
    PUBLIC Threads::Threads
  )

  # libhttpclient.so
  add_library(
    httpclient SHARED
    $<TARGET_OBJECTS:http-client-library>
  )
  add_library(
    TRTIS::httpclient ALIAS httpclient
  )

  if (NOT WIN32)
  set_target_properties(
    httpclient
    PROPERTIES
      LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libhttpclient.ldscript
      LINK_FLAGS "-Wl,--version-script=libhttpclient.ldscript"
  )
  endif() # NOT WIN32

  if(${TRTIS_ENABLE_GPU})
    target_include_directories(httpclient PUBLIC ${CUDA_INCLUDE_DIRS})
  endif() # TRTIS_ENABLE_GPU
  target_include_directories (httpclient
    PUBLIC
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
  target_link_libraries(
    httpclient
    PRIVATE CURL::libcurl
    PUBLIC Threads::Threads
  )

  install(
    TARGETS
      httpclient
      httpclient_static
    EXPORT trtis-export
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

endif()  # TRTIS_ENABLE_HTTP_V2

# cmake configuration
include (CMakePackageConfigHelpers)
set(_LIB_CMAKE_DIR lib/cmake/TRTIS)
install(
  EXPORT trtis-export
  FILE TRTISTargets.cmake
  NAMESPACE TRTIS::
  DESTINATION ${_LIB_CMAKE_DIR}
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/TRTISConfigVersion.cmake
  VERSION ${TRTIS_VERSION}
  COMPATIBILITY ExactVersion
)
configure_package_config_file (
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TRTISConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TRTISConfig.cmake
  INSTALL_DESTINATION ${_LIB_CMAKE_DIR}
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TRTISConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TRTISConfigVersion.cmake
  DESTINATION ${_LIB_CMAKE_DIR}
)
