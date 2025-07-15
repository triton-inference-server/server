# Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Unit test for Backend API
#
add_executable(
  backend_tensor_size_test
  backend_tensor_size_test.cc
)

set_target_properties(
  backend_tensor_size_test
  PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH ""
)

target_include_directories(
  backend_tensor_size_test
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(
  backend_tensor_size_test
  PRIVATE
    triton-backend-utils    # from repo-backend
    triton-core-serverapi   # from repo-core
    triton-core-serverstub  # from repo-core
    GTest::gtest
    GTest::gtest_main
)

install(
  TARGETS backend_tensor_size_test
  RUNTIME DESTINATION bin
)

add_subdirectory(repoagent/relocation_repoagent repoagent/relocation_repoagent)

add_subdirectory(distributed_addsub distributed_addsub)
add_subdirectory(dyna_sequence dyna_sequence)
add_subdirectory(iterative_sequence iterative_sequence)
add_subdirectory(query_backend query_backend)

if(${TRITON_ENABLE_GPU})
  add_subdirectory(sequence sequence)
  add_subdirectory(implicit_state implicit_state)
endif()

#
# Apple Silicon Backend Tests
#
if(APPLE AND TRITON_ENABLE_APPLE_OPTIMIZATIONS)
  # AMX Provider Test
  add_executable(
    amx_provider_test
    amx_provider_test_simple.cc
  )

  set_target_properties(
    amx_provider_test
    PROPERTIES
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH ""
  )

  target_include_directories(
    amx_provider_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${GTEST_INCLUDE_DIRS}
  )

  target_link_libraries(
    amx_provider_test
    PRIVATE
      triton-core-serverapi
      triton-core-serverstub
      triton-apple-optimizations
      GTest::gtest
      GTest::gtest_main
  )

  # ANE Provider Test
  # TODO: Fix compilation issues with CoreML headers
  # add_executable(
  #   ane_provider_test
  #   ane_provider_test.cc
  # )

  # set_target_properties(
  #   ane_provider_test
  #   PROPERTIES
  #     SKIP_BUILD_RPATH TRUE
  #     BUILD_WITH_INSTALL_RPATH TRUE
  #     INSTALL_RPATH_USE_LINK_PATH FALSE
  #     INSTALL_RPATH ""
  # )

  # target_include_directories(
  #   ane_provider_test
  #   PRIVATE
  #     ${CMAKE_CURRENT_SOURCE_DIR}/..
  #     ${GTEST_INCLUDE_DIRS}
  # )

  # target_link_libraries(
  #   ane_provider_test
  #   PRIVATE
  #     triton-core-serverapi
  #     triton-core-serverstub
  #     GTest::gtest
  #     GTest::gtest_main
  #     triton-apple-silicon
  #     "-framework CoreML"
  # )

  # AMX-Metal Interop Test
  # TODO: Fix compilation issues with Objective-C++ and Metal headers
  # add_executable(
  #   amx_metal_interop_test
  #   amx_metal_interop_test.cc
  # )

  # set_target_properties(
  #   amx_metal_interop_test
  #   PROPERTIES
  #     SKIP_BUILD_RPATH TRUE
  #     BUILD_WITH_INSTALL_RPATH TRUE
  #     INSTALL_RPATH_USE_LINK_PATH FALSE
  #     INSTALL_RPATH ""
  # )

  # target_include_directories(
  #   amx_metal_interop_test
  #   PRIVATE
  #     ${CMAKE_CURRENT_SOURCE_DIR}/..
  #     ${GTEST_INCLUDE_DIRS}
  # )

  # target_link_libraries(
  #   amx_metal_interop_test
  #   PRIVATE
  #     triton-core-serverapi
  #     triton-core-serverstub
  #     GTest::gtest
  #     GTest::gtest_main
  #     triton-apple-silicon
  #     triton-metal-backend
  #     "-framework Metal"
  # )

  # ANE Transformer Engine Test
  # TODO: Fix compilation issues with CoreML headers
  # add_executable(
  #   ane_transformer_engine_test
  #   ane_transformer_engine_test.cc
  # )

  # set_target_properties(
  #   ane_transformer_engine_test
  #   PROPERTIES
  #     SKIP_BUILD_RPATH TRUE
  #     BUILD_WITH_INSTALL_RPATH TRUE
  #     INSTALL_RPATH_USE_LINK_PATH FALSE
  #     INSTALL_RPATH ""
  # )

  # target_include_directories(
  #   ane_transformer_engine_test
  #   PRIVATE
  #     ${CMAKE_CURRENT_SOURCE_DIR}/..
  #     ${GTEST_INCLUDE_DIRS}
  # )

  # target_link_libraries(
  #   ane_transformer_engine_test
  #   PRIVATE
  #     triton-core-serverapi
  #     triton-core-serverstub
  #     GTest::gtest
  #     GTest::gtest_main
  #     triton-apple-silicon
  #     "-framework CoreML"
  # )

  # Apple Silicon Integration Test (simple version)
  add_executable(
    apple_silicon_integration_test
    apple_silicon_integration_test_simple.cc
  )

  set_target_properties(
    apple_silicon_integration_test
    PROPERTIES
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH ""
  )

  target_include_directories(
    apple_silicon_integration_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${GTEST_INCLUDE_DIRS}
  )

  target_link_libraries(
    apple_silicon_integration_test
    PRIVATE
      triton-core-serverapi
      triton-core-serverstub
      triton-apple-optimizations
      GTest::gtest
      GTest::gtest_main
  )

  # Winograd Convolution Test
  add_executable(
    winograd_conv3x3_test
    winograd_conv3x3_test.cc
  )

  set_target_properties(
    winograd_conv3x3_test
    PROPERTIES
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH ""
  )

  target_include_directories(
    winograd_conv3x3_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${GTEST_INCLUDE_DIRS}
  )

  target_link_libraries(
    winograd_conv3x3_test
    PRIVATE
      triton-core-serverapi
      triton-core-serverstub
      triton-apple-optimizations
      GTest::gtest
      GTest::gtest_main
  )

  # Metal Allocator Test
  if(TRITON_ENABLE_METAL)
    add_executable(
      metal_allocator_test
      metal_allocator_test.cc
    )

    set_target_properties(
      metal_allocator_test
      PROPERTIES
        SKIP_BUILD_RPATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH FALSE
        INSTALL_RPATH ""
    )

    target_include_directories(
      metal_allocator_test
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${GTEST_INCLUDE_DIRS}
    )

    target_link_libraries(
      metal_allocator_test
      PRIVATE
        triton-core-serverapi
        triton-core-serverstub
        triton-metal-allocator
        GTest::gtest
        GTest::gtest_main
    )
  endif()

  # Install test executables
  install(
    TARGETS
      amx_provider_test
      apple_silicon_integration_test
      winograd_conv3x3_test
    RUNTIME DESTINATION bin
  )

  if(TRITON_ENABLE_METAL)
    install(
      TARGETS
        metal_allocator_test
      RUNTIME DESTINATION bin
    )
  endif()
endif()
