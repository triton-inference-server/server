# Copyright 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.31.8)

project(tritonimplicitsequencebackend LANGUAGES C CXX)

#
# libtriton_implicit_state.so
# Shared library implementing the Triton Implicit Sequence Backend API
#
configure_file(src/libtriton_implicit_state.ldscript libtriton_implicit_state.ldscript COPYONLY)

add_library(
  triton-implicit-state-backend SHARED
  src/implicit_state.cc
)

add_library(
  TritonImplicitStateBackend::triton-implicit-state-backend ALIAS triton-implicit-state-backend
)

target_compile_features(triton-implicit-state-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-implicit-state-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

target_link_libraries(
  triton-implicit-state-backend
  PRIVATE
    triton-backend-utils    # from repo-backend
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
)

set_target_properties(
  triton-implicit-state-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_implicit_state
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_implicit_state.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_implicit_state.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonImplicitStateBackend)

install(
  TARGETS
    triton-implicit-state-backend
  EXPORT
    triton-implicit-state-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/implicit_state
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/implicit_state
)

install(
  EXPORT
    triton-implicit-state-backend-targets
  FILE
    TritonImplicitStateBackendTargets.cmake
  NAMESPACE
    TritonImplicitStateBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonImplicitStateBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonImplicitStateBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonImplicitStateBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-implicit-state-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonImplicitStateBackendTargets.cmake
  NAMESPACE TritonImplicitStateBackend::
)

export(PACKAGE TritonImplicitStateBackend)
