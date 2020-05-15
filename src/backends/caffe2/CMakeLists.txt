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
# NetDef backend
#
# netdef_backend_c2.[cc,h] are also in the PyTorch container build so
# that the Caffe2 build includes their functionality (and also ensures
# that they are built with the same protobuf as Caffe2). So
# netdef_backend_c2.cc is not included in the inference server build
# and so does not appear below.
#
set(
  NETDEF_SRCS
  autofill.cc
  netdef_backend_factory.cc
  netdef_backend.cc
)

set(
  NETDEF_HDRS
  autofill.h
  netdef_backend_factory.h
  netdef_backend.h
  netdef_backend_c2.h
)

add_library(
  caffe2-backend-library EXCLUDE_FROM_ALL OBJECT
  ${NETDEF_SRCS} ${NETDEF_HDRS}
)
add_dependencies(caffe2-backend-library proto-library)

if(${TRITON_ENABLE_GPU})
  target_include_directories(caffe2-backend-library PRIVATE ${CUDA_INCLUDE_DIRS})
endif() # TRITON_ENABLE_GPU
if(${TRITON_ENABLE_METRICS})
  target_include_directories(
    caffe2-backend-library
    PRIVATE $<TARGET_PROPERTY:prometheus-cpp::core,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif() # TRITON_ENABLE_METRICS
