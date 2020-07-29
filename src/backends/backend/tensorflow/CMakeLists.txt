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

#
# libtriton_tensorflow.so
#
configure_file(libtriton_tensorflow.ldscript libtriton_tensorflow.ldscript COPYONLY)

set(
  TENSORFLOW_SRCS
  model_instance.cc
  tensorflow.cc
  tf_utils.cc
)

set(
  TENSORFLOW_HDRS
  model_instance.h
  tf_utils.h
)

add_library(
  triton_tensorflow SHARED
  ${TENSORFLOW_SRCS} ${TENSORFLOW_HDRS}
  $<TARGET_OBJECTS:backend-utils-library>
)
if(${TRITON_ENABLE_GPU})
  target_include_directories(triton_tensorflow PRIVATE ${CUDA_INCLUDE_DIRS})
endif() # TRITON_ENABLE_GPU
set_target_properties(
  triton_tensorflow
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_tensorflow.ldscript
)
set_target_properties(
  triton_tensorflow
  PROPERTIES LINK_FLAGS "-Wl,--version-script libtriton_tensorflow.ldscript"
)

install(
  TARGETS
    triton_tensorflow
  LIBRARY DESTINATION backends
)
