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
# librequest.so
#
find_package(CURL REQUIRED)

add_library(
  request SHARED
  request.cc request.h
  request_common.cc request_common.h
  request_http.cc request_grpc.cc
  $<TARGET_OBJECTS:model-config-library>
  $<TARGET_OBJECTS:proto-library>
  $<TARGET_OBJECTS:grpc-library>
)

target_link_libraries(
  request
  PRIVATE gRPC::grpc++
  PRIVATE gRPC::grpc
  PUBLIC protobuf::libprotobuf
  PUBLIC ${CURL_LIBRARY}
)
# use different syntax for Windows
if(NOT WIN32)
  target_link_libraries(
    request
    PUBLIC -lpthread
  )
endif()
target_include_directories(request PRIVATE ${CURL_INCLUDE_DIRS})


install(
  TARGETS request
  LIBRARY DESTINATION lib
)
install(
  FILES request.h request_grpc.h request_http.h
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
  PRIVATE request
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
  PRIVATE request
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
  PRIVATE request
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
  PRIVATE request
)
install(
  TARGETS simple_client
  RUNTIME DESTINATION bin
)

#
# simple_callback_client
#
add_executable(simple_callback_client simple_callback_client.cc)
target_link_libraries(
  simple_callback_client
  PRIVATE request
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
  PRIVATE request
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
  PRIVATE request
)
install(
  TARGETS simple_string_client
  RUNTIME DESTINATION bin
)
endif()
