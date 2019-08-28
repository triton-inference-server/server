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
# librequest.so and librequest_static.a
#
find_package(CURL REQUIRED)

configure_file(librequest.ldscript librequest.ldscript COPYONLY)

# Headers with include paths adjusted to reflect that all headers are
# in the same directory. These headers are used in the install.
set(TRTIS_CLIENT_HEADER_FLAT 1)
configure_file(request.h.in flat_include/request.h @ONLY)
configure_file(request_grpc.h.in flat_include/request_grpc.h @ONLY)
configure_file(request_http.h.in flat_include/request_http.h @ONLY)

# Headers with full include paths needed for build.
set(TRTIS_CLIENT_HEADER_FLAT 0)
configure_file(request.h.in request.h @ONLY)
configure_file(request_grpc.h.in request_grpc.h @ONLY)
configure_file(request_http.h.in request_http.h @ONLY)

set(
  REQUEST_SRCS
  request.cc request_common.cc request_http.cc request_grpc.cc
)

set(
  REQUEST_HDRS
  request.h request_common.h
)

add_library(
  request-library EXCLUDE_FROM_ALL OBJECT
  ${REQUEST_SRCS} ${REQUEST_HDRS}
)
add_dependencies(
  request-library
  model-config-library grpc-library proto-library
)

# librequest_static.a
add_library(
  request_static STATIC
  $<TARGET_OBJECTS:grpc-library>
  $<TARGET_OBJECTS:model-config-library>
  $<TARGET_OBJECTS:proto-library>
  $<TARGET_OBJECTS:request-library>
)
target_link_libraries(
  request_static
  PRIVATE gRPC::grpc++
  PRIVATE gRPC::grpc
  PUBLIC protobuf::libprotobuf
  PUBLIC ${CURL_LIBRARY}
)
if(NOT WIN32)
  target_link_libraries(
    request_static
    PUBLIC -lpthread
  )
endif()

# librequest.so
add_library(
  request SHARED
  $<TARGET_OBJECTS:grpc-library>
  $<TARGET_OBJECTS:model-config-library>
  $<TARGET_OBJECTS:proto-library>
  $<TARGET_OBJECTS:request-library>
)
target_include_directories(request PRIVATE ${CURL_INCLUDE_DIRS})
set_target_properties(
  request
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/librequest.ldscript
)
set_target_properties(
  request
  PROPERTIES LINK_FLAGS "-Wl,--version-script=librequest.ldscript"
)
target_link_libraries(
  request
  PRIVATE gRPC::grpc++
  PRIVATE gRPC::grpc
  PUBLIC protobuf::libprotobuf
  PUBLIC ${CURL_LIBRARY}
)
if(NOT WIN32)
  target_link_libraries(
    request
    PUBLIC -lpthread
  )
endif()

install(
  TARGETS request
  LIBRARY DESTINATION lib
)
install(
  TARGETS request_static
  ARCHIVE DESTINATION lib
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/flat_include/request.h
    ${CMAKE_CURRENT_BINARY_DIR}/flat_include/request_grpc.h
    ${CMAKE_CURRENT_BINARY_DIR}/flat_include/request_http.h
  DESTINATION include
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/../../core/api.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/../../core/model_config.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/../../core/request_status.pb.h
    ${CMAKE_CURRENT_BINARY_DIR}/../../core/server_status.pb.h
  DESTINATION include
)

if(WIN32)
  message("Currently, CMake will only build client library and skip example "
          "clients in Windows. Because the examples use functionalities that "
		      "are UNIX specific.")
else()
#
# ensemble_image_client
#
add_executable(ensemble_image_client ensemble_image_client.cc)
target_link_libraries(
  ensemble_image_client
  PRIVATE request_static
)
install(
  TARGETS ensemble_image_client
  RUNTIME DESTINATION bin
)

#
# image_client
#
find_package(OpenCV REQUIRED)
add_executable(image_client image_client.cc)
target_include_directories(image_client PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(
  image_client
  PRIVATE request_static
  PRIVATE ${OpenCV_LIBS}
)
install(
  TARGETS image_client
  RUNTIME DESTINATION bin
)

#
# perf_client
#
set(
  PERF_CLIENT_SRCS
  perf_client.cc
  context_factory.cc
  concurrency_manager.cc
  inference_profiler.cc
  perf_utils.cc
)

set(
  PERF_CLIENT_HDRS
  context_factory.h
  inference_profiler.h
  perf_utils.h
  load_manager.h
)

add_executable(perf_client
  ${PERF_CLIENT_SRCS} ${PERF_CLIENT_HDRS})
target_link_libraries(
  perf_client
  PRIVATE request_static
)
install(
  TARGETS perf_client
  RUNTIME DESTINATION bin
)

#
# simple_client
#
add_executable(simple_client simple_client.cc)
target_link_libraries(
  simple_client
  PRIVATE request_static
)
install(
  TARGETS simple_client
  RUNTIME DESTINATION bin
)

#
# simple_shm_client
#
add_executable(simple_shm_client simple_shm_client.cc)
target_link_libraries(
  simple_shm_client
  PRIVATE request_static
  rt
)
install(
  TARGETS simple_shm_client
  RUNTIME DESTINATION bin
)

#
# simple_callback_client
#
add_executable(simple_callback_client simple_callback_client.cc)
target_link_libraries(
  simple_callback_client
  PRIVATE request_static
)
install(
  TARGETS simple_callback_client
  RUNTIME DESTINATION bin
)

#
# simple_sequence_client
#
add_executable(simple_sequence_client simple_sequence_client.cc)
target_link_libraries(
  simple_sequence_client
  PRIVATE request_static
)
install(
  TARGETS simple_sequence_client
  RUNTIME DESTINATION bin
)

#
# simple_string_client
#
add_executable(simple_string_client simple_string_client.cc)
target_link_libraries(
  simple_string_client
  PRIVATE request_static
)
install(
  TARGETS simple_string_client
  RUNTIME DESTINATION bin
)
endif()
