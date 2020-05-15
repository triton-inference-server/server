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
    add_library(
        ccudashm SHARED
        cuda_shared_memory/cuda_shared_memory.cc
        ../../c++/library/cencode.c
        ../../c++/library/cencode.h
    )
    target_include_directories(ccudashm PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(
        ccudashm
        PRIVATE ${CUDA_LIBRARIES})
  endif() # TRTIS_ENABLE_GPU
endif() # NOT WIN32

configure_file(../../../../VERSION VERSION COPYONLY)
configure_file(utils.py utils.py COPYONLY)

if(NOT WIN32)
  configure_file(shared_memory/__init__.py shared_memory/__init__.py COPYONLY)
  if(${TRTIS_ENABLE_GPU})
    configure_file(cuda_shared_memory/__init__.py cuda_shared_memory/__init__.py COPYONLY)
  endif() # TRTIS_ENABLE_GPU
endif() # NOT WIN32

#
# HTTP wheel file
#
if(${TRTIS_ENABLE_HTTP})
  set(http_wheel_stamp_file "http_stamp.whl")
  configure_file(httpclient.py httpclient.py COPYONLY)
  configure_file(http_setup.py http_setup.py COPYONLY)

  add_custom_command(
    OUTPUT "${http_wheel_stamp_file}"
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/http_build_wheel.sh"
    ARGS "${CMAKE_CURRENT_BINARY_DIR}/http/"
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/VERSION
      ${CMAKE_CURRENT_BINARY_DIR}/httpclient.py
      ${CMAKE_CURRENT_BINARY_DIR}/utils.py
      ${CMAKE_CURRENT_BINARY_DIR}/http_setup.py
  )

  add_custom_target(
    client-wheel-http ALL
    DEPENDS
      "${http_wheel_stamp_file}"
  )

  install(
    CODE "file(GLOB _Wheel \"${CMAKE_CURRENT_BINARY_DIR}/http/tritonhttpclient-*.whl\")"
    CODE "file(INSTALL \${_Wheel} DESTINATION \"${CMAKE_INSTALL_PREFIX}/python\")"
  )
endif() # TRTIS_ENABLE_HTTP


#
# gRPC wheel file
#
if(${TRTIS_ENABLE_GRPC})
  set(grpc_wheel_stamp_file "grpc_stamp.whl")

  configure_file(grpcclient.py grpcclient.py COPYONLY)
  configure_file(grpc_setup.py grpc_setup.py COPYONLY)

  add_custom_command(
    OUTPUT "${grpc_wheel_stamp_file}"
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/grpc_build_wheel.sh"
    ARGS "${CMAKE_CURRENT_BINARY_DIR}/grpc/"
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/VERSION
      ${CMAKE_CURRENT_BINARY_DIR}/grpcclient.py
      ${CMAKE_CURRENT_BINARY_DIR}/utils.py
      ${CMAKE_CURRENT_BINARY_DIR}/grpc_setup.py
      proto-py-library
      grpc-py-library
  )

  add_custom_target(
    client-wheel-grpc ALL
    DEPENDS
      "${grpc_wheel_stamp_file}"
  )
  add_dependencies(
    client-wheel-grpc
    model-config-library grpc-library proto-library
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
    CODE "file(GLOB _Wheel \"${CMAKE_CURRENT_BINARY_DIR}/grpc/tritongrpcclient-*.whl\")"
    CODE "file(INSTALL \${_Wheel} DESTINATION \"${CMAKE_INSTALL_PREFIX}/python\")"
  )
endif() # TRTIS_ENABLE_GRPC

#
# triton client utils wheel file
#
if(NOT WIN32)
  set(utils_wheel_stamp_file "utils_stamp.whl")
  configure_file(utils_setup.py utils_setup.py COPYONLY)

  if(${TRTIS_ENABLE_GPU})
    add_custom_command(
        OUTPUT "${utils_wheel_stamp_file}"
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/utils_build_wheel.sh"
        ARGS "${CMAKE_CURRENT_BINARY_DIR}/utils/"
        DEPENDS
          ${CMAKE_CURRENT_BINARY_DIR}/VERSION
          ${CMAKE_CURRENT_BINARY_DIR}/utils_setup.py
          cshm
          ccudashm
    )
  else()
    add_custom_command(
        OUTPUT "${utils_wheel_stamp_file}"
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/utils_build_wheel.sh"
        ARGS "${CMAKE_CURRENT_BINARY_DIR}/utils/"
        DEPENDS
          ${CMAKE_CURRENT_BINARY_DIR}/VERSION
          ${CMAKE_CURRENT_BINARY_DIR}/utils_setup.py
          cshm
    )
  endif()

  add_custom_target(
      client-wheel-utils ALL
      DEPENDS
        "${utils_wheel_stamp_file}"
  )

  install(
      CODE "file(GLOB _Wheel \"${CMAKE_CURRENT_BINARY_DIR}/utils/tritonclientutils-*.whl\")"
      CODE "file(INSTALL \${_Wheel} DESTINATION \"${CMAKE_INSTALL_PREFIX}/python\")"
  )
endif() # NOT WIN32
