# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.17)

project(tritonsequencebackend LANGUAGES C CXX)

#
# libtriton_sequence.so
# Shared library implementing the Triton Sequence Backend API
#
configure_file(src/libtriton_sequence.ldscript libtriton_sequence.ldscript COPYONLY)

add_library(
  triton-sequence-backend SHARED
  src/sequence.cc
)

add_library(
  TritonSequenceBackend::triton-sequence-backend ALIAS triton-sequence-backend
)

target_compile_features(triton-sequence-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-sequence-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

target_link_libraries(
  triton-sequence-backend
  PRIVATE
    triton-backend-utils    # from repo-backend
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
)

set_target_properties(
  triton-sequence-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_sequence
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_sequence.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_sequence.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonSequenceBackend)

install(
  TARGETS
    triton-sequence-backend
  EXPORT
    triton-sequence-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/sequence
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/sequence
)

install(
  EXPORT
    triton-sequence-backend-targets
  FILE
    TritonSequenceBackendTargets.cmake
  NAMESPACE
    TritonSequenceBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonSequenceBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonSequenceBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonSequenceBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-sequence-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonSequenceBackendTargets.cmake
  NAMESPACE TritonSequenceBackend::
)

export(PACKAGE TritonSequenceBackend)
