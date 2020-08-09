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

function(EXTRACT_STATIC_LIBRARY LIB_NAME)
  # Extracts a single static library into ${LIB_NAME}-t
  string(REPLACE "/" ";" LIB_NAME_MODIFIED ${LIB_NAME})
  list(GET LIB_NAME_MODIFIED -1 CUSTOM_TARGET_NAME)
  add_custom_target(${CUSTOM_TARGET_NAME}-t ALL
                    COMMAND rm -rf ${CUSTOM_TARGET_NAME}-dir
                    COMMAND mkdir ${CUSTOM_TARGET_NAME}-dir
                    COMMAND cp ${LIB_NAME} ${CUSTOM_TARGET_NAME}-dir
                    COMMAND cd ${CUSTOM_TARGET_NAME}-dir && ar -x ${LIB_NAME}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

endfunction()

function(COMBINE_STATIC_LIBRARIES LIBRARY_LOCATIONS LIBRARY_NAME OBJ_FILES DEPS)
  # Extracts the object files into the correct location and archives them together again
  # so that there is only single static library required
  set(MODIFIED_LIB_ROUTES "")

  foreach(LIBRARY_LOC IN ITEMS ${LIBRARY_LOCATIONS})
    string(REPLACE "/" ";" LIB_NAME_MODIFIED ${LIBRARY_LOC})
    list(GET LIB_NAME_MODIFIED -1 CUSTOM_TARGET_NAME)
    list(APPEND MODIFIED_LIB_ROUTES "${CUSTOM_TARGET_NAME}-dir/*.o")
    EXTRACT_STATIC_LIBRARY(${LIBRARY_LOC})
  endforeach()
  message("loc value is ${MODIFIED_LIB_ROUTES}")

  string(REPLACE "/" ";" LIBRARY_TARGET_NAME ${LIBRARY_NAME})
  list(GET LIBRARY_TARGET_NAME -1 LIBRARY_TARGET_NAME)
  add_custom_target(${LIBRARY_TARGET_NAME}-combined ALL
      COMMAND ${CMAKE_AR} -qcs ${LIBRARY_NAME} ${MODIFIED_LIB_ROUTES} ${OBJ_FILES}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${DEPS}
      COMMAND_EXPAND_LISTS
  )
endfunction()

if(${TRITON_ENABLE_GRPC})
  #
  # libgrpcclient.so and libgrpcclient_static.a
  #
  configure_file(libgrpcclient.ldscript libgrpcclient.ldscript COPYONLY)

  # libgrpcclient object build
  set(
      REQUEST_SRCS
      grpc_client.cc common.cc
  )

  set(
      REQUEST_HDRS
      grpc_client.h common.h ipc.h
  )

  set(TRITON_CLIENT_HEADER_FLAT 0)
  configure_file(grpc_client.h.in grpc_client.h @ONLY)

  add_library(
      grpc-client-library EXCLUDE_FROM_ALL OBJECT
      ${REQUEST_SRCS} ${REQUEST_HDRS}
  )
  if(${TRITON_ENABLE_GPU})
    target_include_directories(grpc-client-library PUBLIC ${CUDA_INCLUDE_DIRS})
  endif() # TRITON_ENABLE_GPU
  add_dependencies(
      grpc-client-library
      model-config-library grpc-library proto-library
  )

  # libgrpcclient_static.a
  add_library(
      grpcclient_static STATIC
      $<TARGET_OBJECTS:grpc-library>
      $<TARGET_OBJECTS:model-config-library>
      $<TARGET_OBJECTS:proto-library>
      $<TARGET_OBJECTS:grpc-client-library>
      $<TARGET_OBJECTS:triton-json-library>
  )

  add_library(
      TRITON::grpcclient_static ALIAS grpcclient_static
  )

  if(${TRITON_ENABLE_GPU})
    target_include_directories(grpcclient_static PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(grpcclient_static PRIVATE ${CUDA_LIBRARIES})
  endif() # TRITON_ENABLE_GPU
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
      $<TARGET_OBJECTS:grpc-library>
      $<TARGET_OBJECTS:model-config-library>
      $<TARGET_OBJECTS:proto-library>
      $<TARGET_OBJECTS:grpc-client-library>
      $<TARGET_OBJECTS:triton-json-library>
  )
  add_library(
      TRITON::grpcclient ALIAS grpcclient
  )

  if (NOT WIN32)
     set_target_properties(
         grpcclient
         PROPERTIES
           LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libgrpcclient.ldscript
           LINK_FLAGS "-Wl,--version-script=libgrpcclient.ldscript"
     )
   endif() # NOT WIN32

  if(${TRITON_ENABLE_GPU})
    target_include_directories(grpcclient PUBLIC ${CUDA_INCLUDE_DIRS})
  endif() # TRITON_ENABLE_GPU
  target_include_directories(
      grpcclient
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
      EXPORT triton-export
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
  )

  set(GRPCCLIENT_STATIC_OBJS $<TARGET_OBJECTS:grpc-library> $<TARGET_OBJECTS:model-config-library>
      $<TARGET_OBJECTS:proto-library> $<TARGET_OBJECTS:triton-json-library> $<TARGET_OBJECTS:grpc-client-library>)
  set(DEPS grpc-client-library model-config-library grpc-library proto-library)

  if (NOT WIN32)
     set(GRPCCLIENT_STATIC_LIB_PATH ${CMAKE_CURRENT_BINARY_DIR}/libgrpcclient_static.a)
     set(STATIC_LIBRARIES_PATH
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../c-ares/lib/libcares.a
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../grpc/lib/libgpr.a
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../grpc/lib/libgrpc.a
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../grpc/lib/libgrpc++.a
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../grpc/lib/libaddress_sorting.a
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../protobuf/lib/libprotobuf.a
     )
     COMBINE_STATIC_LIBRARIES("${STATIC_LIBRARIES_PATH}" "${GRPCCLIENT_STATIC_LIB_PATH}" "${GRPCCLIENT_STATIC_OBJS}" "${DEPS}")

     install(
         FILES
         ${GRPCCLIENT_STATIC_LIB_PATH}
         DESTINATION lib
     )
  else()
     set(GRPCCLIENT_STATIC_LIB_PATH ${CMAKE_CURRENT_BINARY_DIR}/libgrpcclient_static.lib)
     set(STATIC_LIBRARIES_PATH
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../c-ares/lib/cares.lib
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../grpc/lib/gpr.lib
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../grpc/lib/grpc.lib
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../grpc/lib/grpc++.lib
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../grpc/lib/address_sorting.lib
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../protobuf/lib/libprotobuf.lib
     )
     COMBINE_STATIC_LIBRARIES("${STATIC_LIBRARIES_PATH}" "${GRPCCLIENT_STATIC_LIB_PATH}" "${GRPCCLIENT_STATIC_OBJS}" "${DEPS}")
     install(
         FILES
         ${GRPCCLIENT_STATIC_LIB_PATH}
         DESTINATION lib
     )
  endif() # NOT WIN32

  set(TRITON_CLIENT_HEADER_FLAT 1)
  configure_file(grpc_client.h.in grpc_client_flat.h @ONLY)

  install(
      FILES
      ${CMAKE_CURRENT_BINARY_DIR}/grpc_client_flat.h
      RENAME grpc_client.h
      DESTINATION include
  )

  install(
      FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/common.h
      ${CMAKE_CURRENT_SOURCE_DIR}/ipc.h
      DESTINATION include
  )

  install(
      FILES
      ${CMAKE_CURRENT_BINARY_DIR}/../../../core/model_config.pb.h
      ${CMAKE_CURRENT_BINARY_DIR}/../../../core/grpc_service.grpc.pb.h
      ${CMAKE_CURRENT_BINARY_DIR}/../../../core/grpc_service.pb.h
      DESTINATION include
  )
endif() # TRITON_ENABLE_GRPC

if(${TRITON_ENABLE_HTTP})
  #
  # libhttpclient.so and libhttpclient_static.a
  #
  find_package(CURL CONFIG REQUIRED)
  message(STATUS "Using curl ${CURL_VERSION}")
  add_definitions(-DCURL_STATICLIB)

  configure_file(libhttpclient.ldscript libhttpclient.ldscript COPYONLY)

  # libhttpclient object build
  set(
      REQUEST_SRCS
      http_client.cc common.cc cencode.c
  )

  set(
      REQUEST_HDRS
      http_client.h common.h ipc.h cencode.h
  )

  set(TRITON_CLIENT_HEADER_FLAT 0)
  configure_file(http_client.h.in http_client.h @ONLY)

  add_library(
      http-client-library EXCLUDE_FROM_ALL OBJECT
      ${REQUEST_SRCS} ${REQUEST_HDRS}
  )

  target_include_directories(
      http-client-library
      PRIVATE $<TARGET_PROPERTY:CURL::libcurl,INTERFACE_INCLUDE_DIRECTORIES>
  )
  if(${TRITON_ENABLE_GPU})
    target_include_directories(http-client-library PUBLIC ${CUDA_INCLUDE_DIRS})
  endif() # TRITON_ENABLE_GPU

  # libhttpclient_static.a
  add_library(
      httpclient_static STATIC
      $<TARGET_OBJECTS:http-client-library>
      $<TARGET_OBJECTS:triton-json-library>
  )
  add_library(
      TRITON::httpclient_static ALIAS httpclient_static
  )
  if(${TRITON_ENABLE_GPU})
    target_include_directories(httpclient_static PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(httpclient_static PRIVATE ${CUDA_LIBRARIES})
  endif() # TRITON_ENABLE_GPU
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
      $<TARGET_OBJECTS:triton-json-library>
  )
  add_library(
      TRITON::httpclient ALIAS httpclient
  )

  if (NOT WIN32)
     set_target_properties(
       httpclient
       PROPERTIES
         LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libhttpclient.ldscript
         LINK_FLAGS "-Wl,--version-script=libhttpclient.ldscript"
     )
  endif() # NOT WIN32

  if(${TRITON_ENABLE_GPU})
    target_include_directories(httpclient PUBLIC ${CUDA_INCLUDE_DIRS})
  endif() # TRITON_ENABLE_GPU
  target_include_directories (
      httpclient
      PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
  target_link_libraries(
      httpclient
      PRIVATE CURL::libcurl
      PUBLIC Threads::Threads
  )

  set(HTTPCLIENT_STATIC_OBJS $<TARGET_OBJECTS:http-client-library> $<TARGET_OBJECTS:triton-json-library>)
  set(DEPS http-client-library triton-json-library)

  set(TRITON_CLIENT_HEADER_FLAT 1)
  configure_file(http_client.h.in http_client_flat.h @ONLY)
  if (NOT WIN32)
     set(HTTPCLIENT_STATIC_LIB_PATH ${CMAKE_CURRENT_BINARY_DIR}/libhttpclient_static.a)
     set(STATIC_LIBRARIES_PATH
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../curl/install/lib/libcurl.a
     )
     COMBINE_STATIC_LIBRARIES("${STATIC_LIBRARIES_PATH}" "${HTTPCLIENT_STATIC_LIB_PATH}" "${HTTPCLIENT_STATIC_OBJS}" "${DEPS}")
     install(
         FILES
         ${HTTPCLIENT_STATIC_LIB_PATH}
         DESTINATION lib
     )
  else()
     set(HTTPCLIENT_STATIC_LIB_PATH ${CMAKE_CURRENT_BINARY_DIR}/libhttpclient_static.lib)
     set(STATIC_LIBRARIES_PATH
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../curl/install/lib/libcurl.lib
     )
     COMBINE_STATIC_LIBRARIES("${STATIC_LIBRARIES_PATH}" "${HTTPCLIENT_STATIC_LIB_PATH}" "${HTTPCLIENT_STATIC_OBJS}" "${DEPS}")
     install(
         FILES
         ${HTTPCLIENT_STATIC_LIB_PATH}
         DESTINATION lib
     )
  endif() # NOT WIN32

  install(
      FILES
      ${CMAKE_CURRENT_BINARY_DIR}/http_client_flat.h
      RENAME http_client.h
      DESTINATION include
  )

  install(
      FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/common.h
      DESTINATION include
  )

  install(
      TARGETS
        httpclient
      EXPORT triton-export
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
  )
endif() # TRITON_ENABLE_HTTP

# cmake configuration
include (CMakePackageConfigHelpers)
set(_LIB_CMAKE_DIR lib/cmake/TRITON)
install(
  EXPORT triton-export
  FILE TRITONTargets.cmake
  NAMESPACE TRITON::
  DESTINATION ${_LIB_CMAKE_DIR}
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/TRITONConfigVersion.cmake
  VERSION ${TRITON_VERSION}
  COMPATIBILITY ExactVersion
)
configure_package_config_file (
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TRITONConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TRITONConfig.cmake
  INSTALL_DESTINATION ${_LIB_CMAKE_DIR}
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TRITONConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TRITONConfigVersion.cmake
  DESTINATION ${_LIB_CMAKE_DIR}
)