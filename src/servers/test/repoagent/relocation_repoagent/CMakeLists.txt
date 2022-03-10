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

project(tritonrelocationrepoagent LANGUAGES C CXX)

configure_file(
  src/libtritonrepoagent_relocation.ldscript
  libtritonrepoagent_relocation.ldscript COPYONLY)

add_library(
  triton-relocation-repoagent SHARED
  src/relocation.cc
)

add_library(
  TritonRelocationRepoAgent::triton-relocation-repoagent ALIAS triton-relocation-repoagent
)

target_compile_features(triton-relocation-repoagent PRIVATE cxx_std_11)
target_compile_options(
  triton-relocation-repoagent PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

target_link_libraries(
  triton-relocation-repoagent
  PRIVATE
    triton-core-serverapi     # from repo-core
    triton-core-repoagentapi  # from repo-core
    triton-core-serverstub    # from repo-core
)

set_target_properties(
  triton-relocation-repoagent PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME tritonrepoagent_relocation
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtritonrepoagent_relocation.ldscript
  LINK_FLAGS "-Wl,--version-script libtritonrepoagent_relocation.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonRelocationRepoAgent)

install(
  TARGETS
    triton-relocation-repoagent
  EXPORT
    triton-relocation-repoagent-targets
  LIBRARY DESTINATION lib
)

install(
  EXPORT
    triton-relocation-repoagent-targets
  FILE
    TritonRelocationRepoAgentTargets.cmake
  NAMESPACE
    TritonRelocationRepoAgent::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonRelocationRepoAgentConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonRelocationRepoAgentConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonRelocationRepoAgentConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-relocation-repoagent-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonRelocationRepoAgentTargets.cmake
  NAMESPACE TritonRelocationRepoAgent::
)

export(PACKAGE TritonRelocationRepoAgent)
