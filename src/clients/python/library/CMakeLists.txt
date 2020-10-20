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

cmake_minimum_required (VERSION 3.18)

add_subdirectory(tritonclient)

# Needed for the backwards-compatibility
# Remove when moving completely to the
# new structure.
if(${TRITON_ENABLE_GRPC})
  file(COPY tritongrpcclient DESTINATION .)
endif() # TRITON_ENABLE_GRPC
if(${TRITON_ENABLE_HTTP})
  file(COPY tritonhttpclient DESTINATION .)
endif() # TRITON_ENABLE_HTTP
file(COPY tritonclientutils DESTINATION .)
if (NOT WIN32)
  file(COPY tritonshmutils DESTINATION .)
endif() # NOT WIN32
####################################

configure_file(../../../../VERSION VERSION COPYONLY)
configure_file(LICENSE.txt LICENSE.txt COPYONLY)
configure_file(setup.py setup.py COPYONLY)
file(COPY requirements DESTINATION .)

set(WHEEL_DEPENDS 
      ${CMAKE_CURRENT_BINARY_DIR}/VERSION
      ${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt
      ${CMAKE_CURRENT_BINARY_DIR}/setup.py
)

if(${TRITON_ENABLE_GRPC})
  set(WHEEL_DEPENDS 
        ${CMAKE_CURRENT_BINARY_DIR}/tritonclient/grpc
        proto-py-library
        grpc-py-library
        ${WHEEL_DEPENDS}
  )
endif() # TRITON_ENABLE_GRPC

if(${TRITON_ENABLE_HTTP})
  set(WHEEL_DEPENDS 
        ${CMAKE_CURRENT_BINARY_DIR}/tritonclient/http
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
# Linux specific Wheel file. Compatible with x86, x64 and aarch64
#
if (NOT WIN32)
  # Can genereate linux specific wheel file on
  # linux systems only.
  set(LINUX_WHEEL_DEPENDS
        cshm
        ${WHEEL_DEPENDS}
  )
  if(NOT ${TRITON_CLIENT_SKIP_EXAMPLES})
    set(X86LINUX_WHEEL_DEPENDS
        perf_analyzer
        ${X86LINUX_WHEEL_DEPENDS}
    )
  endif() # TRITON_CLIENT_SKIP_EXAMPLES

  if(${TRITON_ENABLE_GPU})
    set(LINUX_WHEEL_DEPENDS
          ccudashm
          ${LINUX_WHEEL_DEPENDS}
    )
  endif() # TRITON_ENABLE_GPU

  set(linux_wheel_stamp_file "linux_stamp.whl")
  add_custom_command(
    OUTPUT "${linux_wheel_stamp_file}"
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/build_wheel.sh"
    ARGS
      "${CMAKE_CURRENT_BINARY_DIR}/linux/"
      "true"
      ${CMAKE_CURRENT_BINARY_DIR}/../../c++/perf_analyzer/perf_analyzer
    DEPENDS ${X86LINUX_WHEEL_DEPENDS}
  )

  add_custom_target(
    linux-client-wheel ALL
    DEPENDS
      "${linux_wheel_stamp_file}"
  )
endif() # NOT WIN32

if(${TRITON_ENABLE_GRPC})
  add_dependencies(
    generic-client-wheel
    model-config-library grpc-library proto-library
  )

  if (NOT WIN32)
    add_dependencies(
      linux-client-wheel
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
    CODE "file(GLOB _Wheel \"${CMAKE_CURRENT_BINARY_DIR}/linux/triton*.whl\")"
    CODE "file(INSTALL \${_Wheel} DESTINATION \"${CMAKE_INSTALL_PREFIX}/python\")"
  )
endif() # NOT WIN32
