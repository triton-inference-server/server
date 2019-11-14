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
# libcrequest.so
#
add_library(crequest SHARED crequest.cc)
if(${TRTIS_ENABLE_GPU})
  target_include_directories(crequest PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(
    crequest
    PRIVATE ${CUDA_LIBRARIES}
  )
endif() # TRTIS_ENABLE_GPU
target_link_libraries(
  crequest
  PUBLIC TRTIS::request
)

if(NOT WIN32)
#
# libcshm.so
#
add_library(cshm SHARED shared_memory/shared_memory.cc)
if(${TRTIS_ENABLE_GPU})
  target_include_directories(cshm PRIVATE ${CUDA_INCLUDE_DIRS})
endif() # TRTIS_ENABLE_GPU
target_link_libraries(cshm rt)

#
# libccudashm.so
#
if(${TRTIS_ENABLE_GPU})
  add_library(ccudashm SHARED cuda_shared_memory/cuda_shared_memory.cc)
  target_include_directories(ccudashm PUBLIC ${CUDA_INCLUDE_DIRS})
  target_link_libraries(ccudashm PRIVATE ${CUDA_LIBRARIES})
endif() # TRTIS_ENABLE_GPU
endif() # NOT WIN32

#
# Wheel file
#
set(wheel_stamp_file "stamp.whl")
configure_file(../../../VERSION VERSION COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)
if(NOT WIN32)
  configure_file(shared_memory/__init__.py shared_memory/__init__.py COPYONLY)
  if(${TRTIS_ENABLE_GPU})
    configure_file(cuda_shared_memory/__init__.py cuda_shared_memory/__init__.py COPYONLY)
  endif() # TRTIS_ENABLE_GPU
endif()
configure_file(setup.py setup.py COPYONLY)

add_custom_command(
  OUTPUT "${wheel_stamp_file}"
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/build_wheel.sh"
  ARGS "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/VERSION
    ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py
    crequest
    proto-py-library
    grpc-py-library
)

add_custom_target(
  client-wheel ALL
  DEPENDS
    "${wheel_stamp_file}"
)

file(
  GLOB generated-py
  ${CMAKE_CURRENT_BINARY_DIR}/../../core/*_pb2.py
  ${CMAKE_CURRENT_BINARY_DIR}/../../core/*_grpc.py
)

install(
  FILES ${generated-py}
  DESTINATION python
)

install(
  PROGRAMS
    image_client.py
    ensemble_image_client.py
    grpc_image_client.py
    simple_client.py
    simple_callback_client.py
    simple_string_client.py
    simple_sequence_client.py
    simple_shm_client.py
    simple_shm_string_client.py
    simple_cuda_shm_client.py
    simple_cuda_shm_string_client.py
  DESTINATION python
)

install(
  CODE "file(GLOB _Wheel \"${CMAKE_CURRENT_BINARY_DIR}/tensorrtserver*.whl\")"
  CODE "file(INSTALL \${_Wheel} DESTINATION \"${CMAKE_INSTALL_PREFIX}/python\")"
)
