# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_library(
  grpc-endpoint-library EXCLUDE_FROM_ALL
  grpc_server.cc
  grpc_server.h
  grpc_handler.h
  grpc_utils.cc
  grpc_utils.h
  infer_handler.cc
  infer_handler.h
  stream_infer_handler.h
  stream_infer_handler.cc
)

target_compile_features(grpc-endpoint-library PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(
    grpc-endpoint-library
    PRIVATE
      /W1 /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor
  )
else()
  target_compile_options(
    grpc-endpoint-library
    PRIVATE
      -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations -Werror
  )
endif()

set_target_properties(
  grpc-endpoint-library
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(
  grpc-endpoint-library
  PUBLIC
    proto-library                 # from repo-common
    triton-common-logging         # from repo-common
    triton-common-table-printer   # from repo-common
    triton-common-json            # from repo-common
    grpc-health-library           # from repo-common
    grpc-service-library          # from repo-common
    triton-core-serverapi         # from repo-core
    triton-core-serverstub        # from repo-core
    gRPC::grpc++
    gRPC::grpc
    protobuf::libprotobuf
)

target_include_directories(
  grpc-endpoint-library
  PRIVATE $<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>
)

# FIXME when Triton support of OpenTelemetry is available on Windows
# add ${OPENTELEMETRY_CPP_INCLUDE_DIRS} to above target_include_directories
# JIRA DLIS-4786
if (NOT WIN32 AND ${TRITON_ENABLE_TRACING})
  target_include_directories(
    grpc-endpoint-library
    PRIVATE ${OPENTELEMETRY_CPP_INCLUDE_DIRS}
  )
endif()

target_compile_definitions(
  grpc-endpoint-library
  PRIVATE TRITON_ENABLE_GRPC=1
)

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    grpc-endpoint-library
    PRIVATE TRITON_ENABLE_GPU=1
    PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
  )

  target_link_libraries(
    grpc-endpoint-library
    PUBLIC
      CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_METRICS})
  target_compile_definitions(
    grpc-endpoint-library
    PRIVATE TRITON_ENABLE_METRICS=1
  )
endif() # TRITON_ENABLE_METRICS

if(${TRITON_ENABLE_LOGGING})
  target_compile_definitions(
    grpc-endpoint-library
    PRIVATE TRITON_ENABLE_LOGGING=1
  )
endif() # TRITON_ENABLE_LOGGING

if(${TRITON_ENABLE_STATS})
  target_compile_definitions(
    grpc-endpoint-library
    PRIVATE TRITON_ENABLE_STATS=1
  )
endif() # TRITON_ENABLE_STATS

if(${TRITON_ENABLE_TRACING})
  target_compile_definitions(
    grpc-endpoint-library
    PRIVATE TRITON_ENABLE_TRACING=1
  )
endif() # TRITON_ENABLE_TRACING

if(${TRITON_ENABLE_NVTX})
  target_compile_definitions(
    grpc-endpoint-library
    PRIVATE TRITON_ENABLE_NVTX=1
  )
endif() # TRITON_ENABLE_NVTX
