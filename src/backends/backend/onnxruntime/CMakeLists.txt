# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
# ONNXRuntime backend
#
configure_file(libtriton_onnxruntime.ldscript libtriton_onnxruntime.ldscript COPYONLY)

set(
  ONNXRUNTIME_SRCS
  loader.cc
  onnxruntime.cc
  onnx_utils.cc
)

set(
  ONNXRUNTIME_HDRS
  loader.h
  onnx_utils.h
)

add_library(
 triton_onnxruntime SHARED
  ${ONNXRUNTIME_SRCS} ${ONNXRUNTIME_HDRS}
  $<TARGET_OBJECTS:backend-library>
)

target_include_directories(triton_onnxruntime PRIVATE ${TRITON_ONNXRUNTIME_INCLUDE_PATHS})
if(${TRITON_ENABLE_GPU})
  target_include_directories(triton_onnxruntime PRIVATE ${CUDA_INCLUDE_DIRS})
endif() # TRITON_ENABLE_GPU

set_target_properties(
  triton_onnxruntime
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_onnxruntime.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_onnxruntime.ldscript"
)

target_link_libraries(
  triton_onnxruntime
  PUBLIC ${TRITON_EXTRA_LDFLAGS}
  PUBLIC -lonnxruntime
)
if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
  target_link_libraries(
    triton_onnxruntime
    PUBLIC ${TRITON_EXTRA_LDFLAGS}
    PUBLIC -ltbb
    PUBLIC -linference_engine
  )
endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

install(
  TARGETS
    triton_onnxruntime
  LIBRARY DESTINATION backends/onnxruntime
)
