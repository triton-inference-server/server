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
  if(${TRITON_ENABLE_GPU})
    target_include_directories(cshm PRIVATE ${CUDA_INCLUDE_DIRS})
  endif() # TRITON_ENABLE_GPU
  target_link_libraries(cshm rt)

  #
  # libccudashm.so
  #
  if(${TRITON_ENABLE_GPU})
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
  endif() # TRITON_ENABLE_GPU
endif() # NOT WIN32

if(NOT WIN32)
  configure_file(shared_memory/__init__.py shared_memory/__init__.py COPYONLY)
  if(${TRITON_ENABLE_GPU})
    configure_file(cuda_shared_memory/__init__.py cuda_shared_memory/__init__.py COPYONLY)
  endif() # TRITON_ENABLE_GPU
endif() # NOT WIN32

configure_file(../../../../VERSION VERSION COPYONLY)
configure_file(LICENSE.txt LICENSE.txt COPYONLY)
configure_file(README.md README.md COPYONLY)
file(COPY requirements DESTINATION .)
configure_file(utils.py utils.py COPYONLY)
configure_file(setup.py setup.py COPYONLY)
configure_file(x86_linux_setup.py x86_linux_setup.py COPYONLY)

set(WHEEL_DEPENDS 
      ${CMAKE_CURRENT_BINARY_DIR}/VERSION
      ${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt
      ${CMAKE_CURRENT_BINARY_DIR}/README.md
      ${CMAKE_CURRENT_BINARY_DIR}/setup.py
      ${CMAKE_CURRENT_BINARY_DIR}/utils.py
)

if(${TRITON_ENABLE_GRPC})
  configure_file(grpcclient.py grpcclient.py COPYONLY)
  set(WHEEL_DEPENDS 
        ${CMAKE_CURRENT_BINARY_DIR}/grpcclient.py
        proto-py-library
        grpc-py-library
        ${WHEEL_DEPENDS}
  )
endif() # TRITON_ENABLE_GRPC

if(${TRITON_ENABLE_HTTP})
  configure_file(httpclient.py httpclient.py COPYONLY)
  set(WHEEL_DEPENDS 
        ${CMAKE_CURRENT_BINARY_DIR}/httpclient.py
        ${WHEEL_DEPENDS}
  )
endif() # TRITON_ENABLE_HTTP

#
# Generic Wheel file
#
set(wheel_stamp_file "stamp.whl")

add_custom_command(
  OUTPUT "${wheel_stamp_file}"
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/build_wheel.sh"
  ARGS "${CMAKE_CURRENT_BINARY_DIR}/generic/" "false"
  DEPENDS ${WHEEL_DEPENDS}
)

add_custom_target(
  generic-client-wheel ALL
  DEPENDS
    "${wheel_stamp_file}"
)

#
# x86-Linux specific Wheel file
#
if (NOT WIN32)
  # Can genereate x86-linux specific wheel file on
  # linux systems only.
  set(X86LINUX_WHEEL_DEPENDS
        cshm
        ${CMAKE_CURRENT_BINARY_DIR}/x86_linux_setup.py
        ${WHEEL_DEPENDS}
  )
  if(NOT ${TRITON_CLIENT_SKIP_EXAMPLES})
    set(X86LINUX_WHEEL_DEPENDS
        perf_client
        ${X86LINUX_WHEEL_DEPENDS}
    )
  endif() # TRITON_CLIENT_SKIP_EXAMPLES

  if(${TRITON_ENABLE_GPU})
    set(X86LINUX_WHEEL_DEPENDS
          ccudashm
          ${X86LINUX_WHEEL_DEPENDS}
    )
  endif() # TRITON_ENABLE_GPU

  set(x86_linux_wheel_stamp_file "x86_linux_stamp.whl")
  add_custom_command(
    OUTPUT "${x86_linux_wheel_stamp_file}"
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/build_wheel.sh"
    ARGS
      "${CMAKE_CURRENT_BINARY_DIR}/x86_linux/"
      "true"
      ${CMAKE_CURRENT_BINARY_DIR}/../../c++/perf_client/perf_client
    DEPENDS ${X86LINUX_WHEEL_DEPENDS}
  )

  add_custom_target(
    x86-linux-client-wheel ALL
    DEPENDS
      "${x86_linux_wheel_stamp_file}"
  )
endif() # NOT WIN32

if(${TRITON_ENABLE_GRPC})
  add_dependencies(
    generic-client-wheel
    model-config-library grpc-library proto-library
  )

  if (NOT WIN32)
    add_dependencies(
      x86-linux-client-wheel
      model-config-library grpc-library proto-library
  )
  endif() # NOT WIN32

  file(
    GLOB generated-py
    ${CMAKE_CURRENT_BINARY_DIR}/../../core/*_pb2.py
    ${CMAKE_CURRENT_BINARY_DIR}/../../core/*_grpc.py
  )

  install(
    FILES ${generated-py}
    DESTINATION python
  )
endif() # TRITON_ENABLE_GRPC

install(
  CODE "file(GLOB _Wheel \"${CMAKE_CURRENT_BINARY_DIR}/generic/triton*.whl\")"
  CODE "file(INSTALL \${_Wheel} DESTINATION \"${CMAKE_INSTALL_PREFIX}/python\")"
)

if (NOT WIN32)
install(
    CODE "file(GLOB _Wheel \"${CMAKE_CURRENT_BINARY_DIR}/x86_linux/triton*.whl\")"
    CODE "file(INSTALL \${_Wheel} DESTINATION \"${CMAKE_INSTALL_PREFIX}/python\")"
  )
endif() # NOT WIN32