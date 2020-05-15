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
  message("Currently, CMake will only build client library and skip example "
          "clients in Windows. Because the examples use functionalities that "
		      "are UNIX specific.")
else()
  install(
    FILES
      ${CMAKE_SOURCE_DIR}/../../VERSION
    DESTINATION .
  )

  #
  # image_client
  #
  find_package(OpenCV REQUIRED)
  add_executable(image_client image_client.cc)
  target_include_directories(image_client PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(
    image_client
    PRIVATE TRTIS::grpcclient_static
    PRIVATE TRTIS::httpclient_static
    PRIVATE ${OpenCV_LIBS}
  )
  install(
    TARGETS image_client
    RUNTIME DESTINATION bin
  )

  #
  # ensemble_image_client
  #
  add_executable(ensemble_image_client ensemble_image_client.cc)
  target_link_libraries(
    ensemble_image_client
    PRIVATE TRTIS::grpcclient_static
    PRIVATE TRTIS::httpclient_static
  )
  install(
    TARGETS ensemble_image_client
    RUNTIME DESTINATION bin
  )

  #
  # simple_grpc_health_metadata
  #
  add_executable(simple_grpc_health_metadata simple_grpc_health_metadata.cc)
  target_link_libraries(
    simple_grpc_health_metadata
    PRIVATE TRTIS::grpcclient_static
  )
  install(
    TARGETS simple_grpc_health_metadata
    RUNTIME DESTINATION bin
  )

  #
  # simple_grpc_model_control
  #
  add_executable(simple_grpc_model_control simple_grpc_model_control.cc)
  target_link_libraries(
    simple_grpc_model_control
    PRIVATE TRTIS::grpcclient_static
  )
  install(
    TARGETS simple_grpc_model_control
    RUNTIME DESTINATION bin
  )

  #
  # simple_grpc_infer_client
  #
  add_executable(simple_grpc_infer_client simple_grpc_infer_client.cc)
  target_link_libraries(
    simple_grpc_infer_client
    PRIVATE TRTIS::grpcclient_static
  )
  install(
    TARGETS simple_grpc_infer_client
    RUNTIME DESTINATION bin
  )

  #
  # simple_grpc_string_infer_client
  #
  add_executable(simple_grpc_string_infer_client simple_grpc_string_infer_client.cc)
  target_link_libraries(
    simple_grpc_string_infer_client
    PRIVATE TRTIS::grpcclient_static
  )
  install(
    TARGETS simple_grpc_string_infer_client
    RUNTIME DESTINATION bin
  )

  #
  # simple_grpc_async_infer_client
  #
  add_executable(simple_grpc_async_infer_client simple_grpc_async_infer_client.cc)
  target_link_libraries(
    simple_grpc_async_infer_client
    PRIVATE TRTIS::grpcclient_static
  )
  install(
    TARGETS simple_grpc_async_infer_client
    RUNTIME DESTINATION bin
  )

  #
  # simple_grpc_sequence_stream_infer_client
  #
  add_executable(simple_grpc_sequence_stream_infer_client simple_grpc_sequence_stream_infer_client.cc)
  target_link_libraries(
    simple_grpc_sequence_stream_infer_client
    PRIVATE TRTIS::grpcclient_static
  )
  install(
    TARGETS simple_grpc_sequence_stream_infer_client
    RUNTIME DESTINATION bin
  )

  #
  # simple_grpc_sequence_sync_infer_client
  #
  add_executable(simple_grpc_sequence_sync_infer_client simple_grpc_sequence_sync_infer_client.cc)
  target_link_libraries(
    simple_grpc_sequence_sync_infer_client
    PRIVATE TRTIS::grpcclient_static
  )
  install(
    TARGETS simple_grpc_sequence_sync_infer_client
    RUNTIME DESTINATION bin
  )

  #
  # simple_grpc_shm_client
  #
  set(
    SIMPLE_GRPC_SHM_SRCS
    simple_grpc_shm_client.cc
    shm_utils.cc
  )
  set(
    SIMPLE_GRPC_SHM_HDRS
    shm_utils.h
  )
  add_executable(simple_grpc_shm_client ${SIMPLE_GRPC_SHM_SRCS} ${SIMPLE_GRPC_SHM_HDRS})
  target_link_libraries(
    simple_grpc_shm_client
    PRIVATE TRTIS::grpcclient_static
  )
  install(
    TARGETS simple_grpc_shm_client
    RUNTIME DESTINATION bin
  )

  if(${TRTIS_ENABLE_GPU})
    #
    # simple_grpc_cudashm_client
    #
    set(
      SIMPLE_GRPC_CUDA_SHM_SRCS
      simple_grpc_cudashm_client.cc
    )
    set(
      SIMPLE_GRPC_CUDA_SHM_HDRS
    )
    add_executable(simple_grpc_cudashm_client ${SIMPLE_GRPC_CUDA_SHM_SRCS} ${SIMPLE_GRPC_CUDA_SHM_HDRS})
    target_include_directories(simple_grpc_cudashm_client PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(
      simple_grpc_cudashm_client
      PRIVATE TRTIS::grpcclient_static
      PRIVATE ${CUDA_LIBRARIES}
    )
    install(
      TARGETS simple_grpc_cudashm_client
      RUNTIME DESTINATION bin
    )
  endif() # TRTIS_ENABLE_GPU

  #
  # simple_http_health_metadata
  #
  add_executable(simple_http_health_metadata simple_http_health_metadata.cc)
  target_link_libraries(
    simple_http_health_metadata
    PRIVATE TRTIS::httpclient_static
  )
  install(
    TARGETS simple_http_health_metadata
    RUNTIME DESTINATION bin
  )

  #
  # simple_http_model_control
  #
  add_executable(simple_http_model_control simple_http_model_control.cc)
  target_link_libraries(
    simple_http_model_control
    PRIVATE TRTIS::httpclient_static
  )
  install(
    TARGETS simple_http_model_control
    RUNTIME DESTINATION bin
  )

  #
  # simple_http_infer_client
  #
  add_executable(simple_http_infer_client simple_http_infer_client.cc)
  target_link_libraries(
    simple_http_infer_client
    PRIVATE TRTIS::httpclient_static
  )
  install(
    TARGETS simple_http_infer_client
    RUNTIME DESTINATION bin
  )

  #
  # simple_http_string_infer_client
  #
  add_executable(simple_http_string_infer_client simple_http_string_infer_client.cc)
  target_link_libraries(
    simple_http_string_infer_client
    PRIVATE TRTIS::httpclient_static
  )
  install(
    TARGETS simple_http_string_infer_client
    RUNTIME DESTINATION bin
  )

  #
  # simple_http_async_infer_client
  #
  add_executable(simple_http_async_infer_client simple_http_async_infer_client.cc)
  target_link_libraries(
    simple_http_async_infer_client
    PRIVATE TRTIS::httpclient_static
  )
  install(
    TARGETS simple_http_async_infer_client
    RUNTIME DESTINATION bin
  )

  #
  # simple_http_sequence_sync_infer_client
  #
  add_executable(simple_http_sequence_sync_infer_client simple_http_sequence_sync_infer_client.cc)
  target_link_libraries(
    simple_http_sequence_sync_infer_client
    PRIVATE TRTIS::httpclient_static
  )
  install(
    TARGETS simple_http_sequence_sync_infer_client
    RUNTIME DESTINATION bin
  )

  #
  # simple_http_shm_client
  #
  set(
    SIMPLE_HTTP_SHM_SRCS
    simple_http_shm_client.cc
    shm_utils.cc
  )
  set(
    SIMPLE_HTTP_SHM_HDRS
    shm_utils.h
  )
  add_executable(simple_http_shm_client ${SIMPLE_HTTP_SHM_SRCS} ${SIMPLE_HTTP_SHM_HDRS})
  target_link_libraries(
    simple_http_shm_client
    PRIVATE TRTIS::httpclient_static
  )
  install(
    TARGETS simple_http_shm_client
    RUNTIME DESTINATION bin
  )

  if(${TRTIS_ENABLE_GPU})
    #
    # simple_http_cudashm_client
    #
    set(
      SIMPLE_HTTP_CUDA_SHM_SRCS
      simple_http_cudashm_client.cc
    )
    set(
      SIMPLE_HTTP_CUDA_SHM_HDRS
    )
    add_executable(simple_http_cudashm_client ${SIMPLE_HTTP_CUDA_SHM_SRCS} ${SIMPLE_HTTP_CUDA_SHM_HDRS})
    target_include_directories(simple_http_cudashm_client PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(
      simple_http_cudashm_client
      PRIVATE TRTIS::httpclient_static
      PRIVATE ${CUDA_LIBRARIES}
    )
    install(
      TARGETS simple_http_cudashm_client
      RUNTIME DESTINATION bin
    )
  endif() # TRTIS_ENABLE_GPU
endif() # WIN32
