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

if(WIN32)
  message("Currently, CMake will not build perf_client because it uses functionalities that "
		      "are UNIX specific.")
else()

set(
  PERF_CLIENT_SRCS
  perf_client.cc
  model_parser.cc
  perf_utils.cc
  load_manager.cc
  data_loader.cc
  concurrency_manager.cc
  request_rate_manager.cc
  custom_load_manager.cc
  triton_client_wrapper.cc
  inference_profiler.cc
  ../examples/shm_utils.cc
)

set(
  PERF_CLIENT_HDRS
  model_parser.h
  perf_utils.h
  load_manager.h
  data_loader.h
  concurrency_manager.h
  request_rate_manager.h
  custom_load_manager.h
  triton_client_wrapper.h
  inference_profiler.h
 ../examples/shm_utils.h
)

add_executable(
  perf_client
  ${PERF_CLIENT_SRCS}
  ${PERF_CLIENT_HDRS}
  $<TARGET_OBJECTS:json-utils-library>
)
target_link_libraries(
  perf_client
  PRIVATE TRITON::grpcclient_static
  PRIVATE TRITON::httpclient_static
  PRIVATE -lb64
)

# If gpu is enabled then compile with CUDA dependencies
if(${TRITON_ENABLE_GPU})
  target_include_directories(perf_client PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(perf_client PRIVATE ${CUDA_LIBRARIES})
endif()


install(
  TARGETS perf_client
  RUNTIME DESTINATION bin
)

endif()
