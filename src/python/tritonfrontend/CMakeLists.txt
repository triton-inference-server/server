# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(tritonfrontend-bindings LANGUAGES C CXX)

# ================= Ensures Package is Structured Properly ==================
# Top level module entry point and typed marker
file(COPY __init__.py DESTINATION .)
file(COPY py.typed DESTINATION .)
# Copy the '__init__.py' for the '_c' module
file(COPY _c/__init__.py DESTINATION ./_c/.)
file(COPY _c/__init__.pyi DESTINATION ./_c/.)
# file(COPY _c/triton_bindings.pyi DESTINATION ./_c/.)
# Find and copy _api modules
file(GLOB PYTHON_MODULE_FILES ./_api/*.py)
file(COPY ${PYTHON_MODULE_FILES} DESTINATION ./_api/.)
# ================================= END =====================================


# =================== Downloading and Installing pybind11 ===================
include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.1
    GIT_SHALLOW ON
)

FetchContent_MakeAvailable(pybind11)

# ================================= END =====================================


# add_library(
#   python-endpoint-wrapper
#     _c/tritonfrontend.h
#     ../../server_interface.h
# )

# target_link_libraries(
#   python-endpoint-wrapper
#   PUBLIC
#   python-binding-dependency
#   shared-memory-manager
# )

# set_target_properties(python-endpoint-wrapper PROPERTIES POSITION_INDEPENDENT_CODE ON)
# target_include_directories(python-endpoint-wrapper PRIVATE ${pybind11_INCLUDE_DIRS})
# target_link_libraries(python-endpoint-wrapper PRIVATE ${pybind11_LIBRARIES})

# target_compile_features(python-endpoint-wrapper PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})

# target_compile_definitions(
#     python-endpoint-wrapper
#     PRIVATE TRITON_ENABLE_GPU=1
#     PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
#   )



set(
  PYTHON_FRONTEND_BINDING_SRCS
  _c/tritonfrontend_pybind.cc
  _c/tritonfrontend.h
)

find_package(libevhtp CONFIG REQUIRED)
message(STATUS "Using libevhtp ${libevhtp_VERSION}")



pybind11_add_module(py-bindings MODULE ${PYTHON_FRONTEND_BINDING_SRCS})

target_include_directories(py-bindings PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(
    py-bindings
    PRIVATE
    ${LIBTRITONSERVER_LIBRARY}
    triton-core-serverstub
    triton-core-serverapi
    http-endpoint-library
    triton-common-async-work-queue  # from repo-common
    triton-common-error             # from repo-common
    triton-common-json
    triton-common-logging           # from repo-common
    CUDA::cudart
    grpc-endpoint-library
    python-binding-dependency
    # shared-memory-manager
    # python-endpoint-wrapper
)


# target_compile_features(py-bindings PRIVATE cxx_std_17)

set_property(TARGET py-bindings PROPERTY OUTPUT_NAME tritonfrontend_bindings)
# Add Triton library default path in 'rpath' for runtime library lookup
set_target_properties(
          py-bindings
          PROPERTIES
          BUILD_RPATH "$ORIGIN:/opt/tritonserver/lib"
)
