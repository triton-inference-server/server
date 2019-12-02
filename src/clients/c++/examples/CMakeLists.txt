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

# Source with include paths adjusted to reflect that all headers are
# in the same directory. These source versions are used in the client
# container install.
set(TRTIS_CLIENT_HEADER_FLAT 1)
configure_file(simple_client.cc.in flat_src/simple_client.cc @ONLY)

# Source with full include paths needed for build.
set(TRTIS_CLIENT_HEADER_FLAT 0)
configure_file(simple_client.cc.in simple_client.cc @ONLY)

if(WIN32)
  message("Currently, CMake will only build client library and skip example "
          "clients in Windows. Because the examples use functionalities that "
		      "are UNIX specific.")
else()
install(
  FILES
    ${CMAKE_SOURCE_DIR}/../../VERSION
  DESTINATION .
)
# Install cmake files that builds the samples within the client install.
install(
  FILES
    ${CMAKE_BINARY_DIR}/../CMakeLists.txt
  DESTINATION build
)
install(
  FILES
    CMakeLists.example
  DESTINATION build/trtis-clients
  RENAME CMakeLists.txt
)

#
# ensemble_image_client
#
add_executable(ensemble_image_client ensemble_image_client.cc)
target_link_libraries(
  ensemble_image_client
  PRIVATE TRTIS::request_static
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
  PRIVATE TRTIS::request_static
  PRIVATE ${OpenCV_LIBS}
)
install(
  TARGETS image_client
  RUNTIME DESTINATION bin
)

#
# simple_callback_client
#
add_executable(simple_callback_client simple_callback_client.cc)
target_link_libraries(
  simple_callback_client
  PRIVATE TRTIS::request_static
)
install(
  TARGETS simple_callback_client
  RUNTIME DESTINATION bin
)

#
# simple_client
#
add_executable(
  simple_client
  ${CMAKE_CURRENT_BINARY_DIR}/simple_client.cc
)
target_link_libraries(
  simple_client
  PRIVATE TRTIS::request_static
)
install(
  TARGETS simple_client
  RUNTIME DESTINATION bin
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/flat_src/simple_client.cc
  DESTINATION src
)

if(${TRTIS_ENABLE_HTTP_V2})
#
# simple_v2_client
#
add_executable(
  simple_v2_client
  simple_v2_client.cc
)
target_link_libraries(
  simple_v2_client
  PRIVATE TRTIS::request_static
)
install(
  TARGETS simple_v2_client
  RUNTIME DESTINATION bin
)
endif() # TRTIS_ENABLE_HTTP_V2

#
# simple_shm_client
#
set(
  SIMPLE_SHM_SRCS
  simple_shm_client.cc
  shm_utils.cc
)
set(
  SIMPLE_SHM_HDRS
  shm_utils.h
)
add_executable(simple_shm_client ${SIMPLE_SHM_SRCS} ${SIMPLE_SHM_HDRS})
target_link_libraries(
  simple_shm_client
  PRIVATE TRTIS::request_static
)
install(
  TARGETS simple_shm_client
  RUNTIME DESTINATION bin
)

#
# simple_shm_string_client
#
set(
  SIMPLE_SHM_STRING_SRCS
  simple_shm_string_client.cc
  shm_utils.cc
)
set(
  SIMPLE_SHM_STRING_HDRS
  shm_utils.h
)
add_executable(simple_shm_string_client ${SIMPLE_SHM_STRING_SRCS} ${SIMPLE_SHM_STRING_HDRS})
target_link_libraries(
  simple_shm_string_client
  PRIVATE TRTIS::request_static
)
install(
  TARGETS simple_shm_string_client
  RUNTIME DESTINATION bin
)

if(${TRTIS_ENABLE_GPU})
  #
  # simple_cuda_shm_client
  #
  set(
    SIMPLE_CUDA_SHM_SRCS
    simple_cuda_shm_client.cc
  )
  set(
    SIMPLE_CUDA_SHM_HDRS
  )
  add_executable(simple_cuda_shm_client ${SIMPLE_CUDA_SHM_SRCS} ${SIMPLE_CUDA_SHM_HDRS})
  target_include_directories(simple_cuda_shm_client PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(
    simple_cuda_shm_client
    PRIVATE TRTIS::request_static
    PRIVATE ${CUDA_LIBRARIES}
  )
  install(
    TARGETS simple_cuda_shm_client
    RUNTIME DESTINATION bin
  )

  #
  # simple_cuda_shm_string_client
  #
  set(
    SIMPLE_CUDA_SHM_STRING_SRCS
    simple_cuda_shm_client.cc
  )
  set(
    SIMPLE_CUDA_SHM_STRING_HDRS
  )
  add_executable(simple_cuda_shm_string_client ${SIMPLE_CUDA_SHM_STRING_SRCS} ${SIMPLE_CUDA_SHM_STRING_HDRS})
  target_include_directories(simple_cuda_shm_string_client PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(
    simple_cuda_shm_string_client
    PRIVATE TRTIS::request_static
    PRIVATE ${CUDA_LIBRARIES}
  )
  install(
    TARGETS simple_cuda_shm_string_client
    RUNTIME DESTINATION bin
  )
endif() # TRTIS_ENABLE_GPU

#
# simple_sequence_client
#
add_executable(simple_sequence_client simple_sequence_client.cc)
target_link_libraries(
  simple_sequence_client
  PRIVATE TRTIS::request_static
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
  PRIVATE TRTIS::request_static
)
install(
  TARGETS simple_string_client
  RUNTIME DESTINATION bin
)

#
# simple_perf_client
#
add_executable(simple_perf_client simple_perf_client.cc)
target_link_libraries(
  simple_perf_client
  PRIVATE TRTIS::request_static
)
install(
  TARGETS simple_perf_client
  RUNTIME DESTINATION bin
)

endif()
