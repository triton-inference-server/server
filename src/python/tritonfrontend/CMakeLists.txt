# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.18)

# ================= Ensures Package is Structured Properly ==================
# Top level module entry point and typed marker
file(COPY __init__.py DESTINATION .)
file(COPY py.typed DESTINATION .)
# Copy the '__init__.py' for the '_c' module
file(COPY _c/__init__.py DESTINATION ./_c/.)
file(COPY _c/__init__.pyi DESTINATION ./_c/.)
file(COPY _c/tritonfrontend_bindings.pyi DESTINATION ./_c/.)
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

# ================== Collect the Dependencies ===============================
set(
  PYTHON_FRONTEND_BINDING_DEPS
  ../../shared_memory_manager.h
  ../../shared_memory_manager.cc
  ../../data_compressor.h
  ../../restricted_features.h
  ../../classification.cc
  ../../common.h
  ../../common.cc
)

set(PY_BINDING_DEPENDENCY_LIBS
      b64) # Dependency from common.h

# Conditional Linking Based on Flags
if(${TRITON_ENABLE_HTTP})
  list(APPEND PY_BINDING_DEPENDENCY_LIBS
      http-endpoint-library
    )
endif()

if(${TRITON_ENABLE_GRPC})
  list(APPEND PY_BINDING_DEPENDENCY_LIBS
      grpc-endpoint-library
  )
endif()

if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
  list(APPEND PY_BINDING_DEPENDENCY_LIBS
      CUDA::cudart
  )
endif()

if(${TRITON_ENABLE_TRACING})
  message("TRACING/STATS IS CURRENTLY NOT SUPPORTED.")
  list(
      APPEND PY_BINDING_DEPENDENCY_LIBS
      tracing-library
  )
endif()

# ===================== End of Collection ===================================

# ================== Create Python Frontend Bindings ========================
set(
  PYTHON_FRONTEND_BINDING_SRCS
  _c/tritonfrontend.h
  _c/tritonfrontend_pybind.cc
)

pybind11_add_module(
  py-bindings
  MODULE
  ${PYTHON_FRONTEND_BINDING_DEPS}
  ${PYTHON_FRONTEND_BINDING_SRCS}
)

target_link_libraries(
    py-bindings
    PRIVATE
    ${PY_BINDING_DEPENDENCY_LIBS}
)

if(${TRITON_ENABLE_HTTP})
  target_compile_definitions(
    py-bindings
    PRIVATE TRITON_ENABLE_HTTP=1
  )
endif()

if(${TRITON_ENABLE_GRPC})
  target_compile_definitions(
    py-bindings
    PRIVATE TRITON_ENABLE_GRPC=1
  )
endif()

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    py-bindings
    PRIVATE TRITON_ENABLE_GPU=1
    PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
  )
endif()

if(${TRITON_ENABLE_TRACING})
    target_compile_definitions(
      py-bindings
      PRIVATE TRITON_ENABLE_TRACING=1
    )
endif()

if(${TRITON_ENABLE_STATS})
  target_compile_definitions(
    py-bindings
    PRIVATE TRITON_ENABLE_STATS=1
  )
endif()

if(${TRITON_ENABLE_METRICS})
  target_compile_definitions(
    py-bindings
    PRIVATE TRITON_ENABLE_METRICS=1
  )
endif()

set_property(TARGET py-bindings PROPERTY OUTPUT_NAME tritonfrontend_bindings)

target_include_directories(
  py-bindings
  PRIVATE
  ${repo-core_SOURCE_DIR}/include
  ${repo-common_SOURCE_DIR}/include
)

set_target_properties(
    py-bindings
    PROPERTIES
      BUILD_RPATH "$ORIGIN:/opt/tritonserver/lib"
      POSITION_INDEPENDENT_CODE ON
)
# ===================== End of Python Bindings ==============================
