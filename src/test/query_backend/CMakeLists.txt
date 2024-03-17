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

project(tritonquerybackend LANGUAGES C CXX)

#
# libtriton_query.so
# Shared library implementing the Triton Query Backend API
#
configure_file(src/libtriton_query.ldscript libtriton_query.ldscript COPYONLY)

add_library(
  triton-query-backend SHARED
  src/query.cc
)

add_library(
  TritonQueryBackend::triton-query-backend ALIAS triton-query-backend
)

target_compile_features(triton-query-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-query-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

target_link_libraries(
  triton-query-backend
  PRIVATE
    triton-backend-utils    # from repo-backend
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
)

set_target_properties(
  triton-query-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_query
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_query.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_query.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonQueryBackend)

install(
  TARGETS
    triton-query-backend
  EXPORT
    triton-query-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/query
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/query
)

install(
  EXPORT
    triton-query-backend-targets
  FILE
    TritonQueryBackendTargets.cmake
  NAMESPACE
    TritonQueryBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonQueryBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonQueryBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonQueryBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-query-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonQueryBackendTargets.cmake
  NAMESPACE TritonQueryBackend::
)

export(PACKAGE TritonQueryBackend)
