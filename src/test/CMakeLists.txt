# Copyright 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required (VERSION 3.31.8)

#
# Unit tests
#
find_package(GTest REQUIRED)

#
# Unit test for DataCompressor
#
if(${TRITON_ENABLE_HTTP} OR ${TRITON_ENABLE_METRICS} OR
    ${TRITON_ENABLE_SAGEMAKER} OR ${TRITON_ENABLE_VERTEX_AI})
  add_executable(
    data_compressor_test
    data_compressor_test.cc
    ../data_compressor.h
    ../common.h
  )

  set_target_properties(
    data_compressor_test
    PROPERTIES
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH ""
  )

  target_include_directories(
    data_compressor_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${GTEST_INCLUDE_DIRS}
      ${LIBEVENT_INCLUDE_DIRS}
  )

  target_link_libraries(
    data_compressor_test
    PRIVATE
      triton-core-serverapi   # from repo-core
      triton-core-serverstub  # from repo-core
      GTest::gtest
      GTest::gtest_main
      ${LIBEVENT_LIBRARIES}
      -lz
  )

  install(
    TARGETS data_compressor_test
    RUNTIME DESTINATION bin
  )
endif()

#
# Unit test for Backend + Common + Core tensor size APIs (GetElementCount,
# GetByteSize). Core template headers are used directly; the few core symbols
# not available via libtritonserver.so (hidden by linker script) are defined
# in the test source.
#
if(TARGET proto-library)
  add_executable(
    tensor_size_test
    tensor_size_test.cc
  )

  set_target_properties(
    tensor_size_test
    PROPERTIES
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH "$ORIGIN/../lib"
  )

  target_include_directories(
    tensor_size_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${GTEST_INCLUDE_DIRS}
      ${repo-core_SOURCE_DIR}/src
      $<TARGET_PROPERTY:proto-library,INTERFACE_INCLUDE_DIRECTORIES>
  )

  target_link_libraries(
    tensor_size_test
    PRIVATE
      triton-backend-utils
      triton-core-serverapi
      triton-core-serverstub
      triton-common-model-config
      proto-library
      protobuf::libprotobuf
      GTest::gtest
      GTest::gtest_main
  )

  install(
    TARGETS tensor_size_test
    RUNTIME DESTINATION bin
  )
endif()

add_subdirectory(repoagent/relocation_repoagent repoagent/relocation_repoagent)

add_subdirectory(distributed_addsub distributed_addsub)
add_subdirectory(dyna_sequence dyna_sequence)
add_subdirectory(iterative_sequence iterative_sequence)
add_subdirectory(query_backend query_backend)

if(${TRITON_ENABLE_GPU})
  add_subdirectory(sequence sequence)
  add_subdirectory(implicit_state implicit_state)
endif()
