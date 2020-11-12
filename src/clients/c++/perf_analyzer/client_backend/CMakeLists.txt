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

set(
  CLIENT_BACKEND_SRCS
  client_backend.cc
  triton/triton_client_backend.cc
  ../../examples/shm_utils.cc
)

set(
  CLIENT_BACKEND_HDRS
  client_backend.h
  triton/triton_client_backend.h
  ../../examples/shm_utils.h
)

add_library(
  client-backend-library
  ${CLIENT_BACKEND_SRCS}
  ${CLIENT_BACKEND_HDRS}
)

target_link_libraries(
  client-backend-library
  PUBLIC triton-common-json        # from repo-common
  PRIVATE model-config-library
  PRIVATE TRITON::grpcclient_static
  PRIVATE TRITON::httpclient_static
)

if(${TRITON_ENABLE_GPU})
    target_include_directories(client-backend-library PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(client-backend-library PRIVATE ${CUDA_LIBRARIES})
endif() # TRITON_ENABLE_GPU

