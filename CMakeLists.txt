# Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(tritonserver LANGUAGES C CXX)

include(CMakeDependentOption)

# Expose local cmake/ modules (FindDCGM, TritonCheckCxxAbi, etc.)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Validate that the C++ ABI (_GLIBCXX_USE_CXX11_ABI) matches the ABI used
# to compile Conan-provided binary packages.  Must run after the toolchain
# file (conan_toolchain.cmake) has been processed so CONAN_COMPILER_LIBCXX
# is populated.
include(TritonCheckCxxAbi)
triton_check_cxx_abi()

# Use C++17 standard as Triton's minimum required.
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

set(TRITON_VERSION "0.0.0" CACHE STRING "The version of the Triton shared library")

option(TRITON_ENABLE_LOGGING "Include logging support in server" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in server" ON)
option(TRITON_ENABLE_TRACING "Include tracing support in server" OFF)
option(TRITON_ENABLE_NVTX "Include NVTX support in server" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in server" ON)
option(TRITON_ENABLE_MALI_GPU "Enable Arm Mali GPU support in server" OFF)
option(TRITON_IGPU_BUILD "Enable options for iGPU compilation in server" OFF)
set(TRITON_MIN_COMPUTE_CAPABILITY "7.5" CACHE STRING
    "The minimum CUDA compute capability supported by Triton")
set(TRITON_EXTRA_LIB_PATHS "" CACHE PATH "Extra library paths for Triton Server build")

# Ensemble
option(TRITON_ENABLE_ENSEMBLE "Include ensemble support in server" OFF)

# Endpoints
option(TRITON_ENABLE_HTTP "Include HTTP API in server" ON)
option(TRITON_ENABLE_GRPC "Include GRPC API in server" ON)
option(TRITON_ENABLE_SAGEMAKER "Include AWS SageMaker API in server" OFF)
option(TRITON_ENABLE_VERTEX_AI "Include Vertex AI API in server" OFF)

# Metrics
option(TRITON_ENABLE_METRICS "Include metrics support in server" ON)
option(TRITON_ENABLE_METRICS_GPU "Include GPU metrics support in server" ON)
option(TRITON_ENABLE_METRICS_CPU "Include CPU metrics support in server" ON)

# Cloud storage
option(TRITON_ENABLE_GCS "Include GCS Filesystem support in server" OFF)
option(TRITON_ENABLE_S3 "Include S3 Filesystem support in server" OFF)
option(TRITON_ENABLE_AZURE_STORAGE "Include Azure Storage Filesystem support in server" OFF)

# Need to know if TensorRT is available when building unit tests
option(TRITON_ENABLE_TENSORRT "Include TensorRT backend in server" OFF)

# ASAN
option(TRITON_ENABLE_ASAN "Build with address sanitizer" OFF)

# Skip FetchContent/ExternalProject for third-party deps when Conan provides them
option(TRITON_SKIP_THIRD_PARTY_FETCH
  "Skip FetchContent/ExternalProject for third-party deps (provided by Conan)" ON)

# ---------------------------------------------------------------------------
# Backend implementations — each defaults OFF; toggle ON to include in build.
# Source dirs default to sibling checkout paths; override via -D if needed.
# ---------------------------------------------------------------------------
option(TRITON_ENABLE_PYTHON_BACKEND      "Build Python backend"          OFF)
option(TRITON_ENABLE_ONNXRUNTIME_BACKEND "Build ONNX Runtime backend"    OFF)
option(TRITON_ENABLE_PYTORCH_BACKEND     "Build PyTorch backend"         OFF)
option(TRITON_ENABLE_TENSORRT_BACKEND    "Build TensorRT backend"        OFF)
option(TRITON_ENABLE_OPENVINO_BACKEND    "Build OpenVINO backend"        OFF)
option(TRITON_ENABLE_DALI_BACKEND        "Build DALI backend"            OFF)
option(TRITON_ENABLE_FIL_BACKEND         "Build FIL backend"             OFF)
option(TRITON_ENABLE_VLLM_BACKEND        "Build vLLM backend"            OFF)
option(TRITON_ENABLE_TENSORRTLLM_BACKEND "Build TensorRT-LLM backend"    OFF)
option(TRITON_ENABLE_IDENTITY_BACKEND    "Build Identity (test) backend" OFF)
option(TRITON_ENABLE_REPEAT_BACKEND      "Build Repeat (test) backend"   OFF)
option(TRITON_ENABLE_SQUARE_BACKEND      "Build Square (test) backend"   OFF)

# Source dirs — prefer sibling checkouts; FetchContent is the fallback.
# Override per-backend with -DTRITON_<NAME>_BACKEND_SOURCE_DIR=<path>.
set(TRITON_PYTHON_BACKEND_SOURCE_DIR      "${CMAKE_CURRENT_SOURCE_DIR}/../python_backend"
    CACHE PATH "Local checkout of python_backend repo")
set(TRITON_ONNXRUNTIME_BACKEND_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime_backend"
    CACHE PATH "Local checkout of onnxruntime_backend repo")
set(TRITON_PYTORCH_BACKEND_SOURCE_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/../pytorch_backend"
    CACHE PATH "Local checkout of pytorch_backend repo")
set(TRITON_TENSORRT_BACKEND_SOURCE_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/../tensorrt_backend"
    CACHE PATH "Local checkout of tensorrt_backend repo")
set(TRITON_OPENVINO_BACKEND_SOURCE_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/../openvino_backend"
    CACHE PATH "Local checkout of openvino_backend repo")
set(TRITON_DALI_BACKEND_SOURCE_DIR        "${CMAKE_CURRENT_SOURCE_DIR}/../dali_backend"
    CACHE PATH "Local checkout of dali_backend repo")
set(TRITON_FIL_BACKEND_SOURCE_DIR         "${CMAKE_CURRENT_SOURCE_DIR}/../fil_backend"
    CACHE PATH "Local checkout of fil_backend repo")
set(TRITON_VLLM_BACKEND_SOURCE_DIR        "${CMAKE_CURRENT_SOURCE_DIR}/../vllm_backend"
    CACHE PATH "Local checkout of vllm_backend repo")
set(TRITON_TENSORRTLLM_BACKEND_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tensorrtllm_backend"
    CACHE PATH "Local checkout of tensorrtllm_backend repo")
set(TRITON_IDENTITY_BACKEND_SOURCE_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/../identity_backend"
    CACHE PATH "Local checkout of identity_backend repo")
set(TRITON_REPEAT_BACKEND_SOURCE_DIR      "${CMAKE_CURRENT_SOURCE_DIR}/../repeat_backend"
    CACHE PATH "Local checkout of repeat_backend repo")
set(TRITON_SQUARE_BACKEND_SOURCE_DIR      "${CMAKE_CURRENT_SOURCE_DIR}/../square_backend"
    CACHE PATH "Local checkout of square_backend repo")

# Git tags used when a backend is fetched (no local checkout found).
# Override per-backend with -DTRITON_<NAME>_BACKEND_REPO_TAG=<ref>.
set(TRITON_PYTHON_BACKEND_REPO_TAG      "main" CACHE STRING "Git ref for python_backend FetchContent")
set(TRITON_ONNXRUNTIME_BACKEND_REPO_TAG "main" CACHE STRING "Git ref for onnxruntime_backend FetchContent")
set(TRITON_PYTORCH_BACKEND_REPO_TAG     "main" CACHE STRING "Git ref for pytorch_backend FetchContent")
set(TRITON_TENSORRT_BACKEND_REPO_TAG    "main" CACHE STRING "Git ref for tensorrt_backend FetchContent")
set(TRITON_OPENVINO_BACKEND_REPO_TAG    "main" CACHE STRING "Git ref for openvino_backend FetchContent")
set(TRITON_DALI_BACKEND_REPO_TAG        "main" CACHE STRING "Git ref for dali_backend FetchContent")
set(TRITON_FIL_BACKEND_REPO_TAG         "main" CACHE STRING "Git ref for fil_backend FetchContent")
set(TRITON_VLLM_BACKEND_REPO_TAG        "main" CACHE STRING "Git ref for vllm_backend FetchContent")
set(TRITON_TENSORRTLLM_BACKEND_REPO_TAG "main" CACHE STRING "Git ref for tensorrtllm_backend FetchContent")
set(TRITON_IDENTITY_BACKEND_REPO_TAG    "main" CACHE STRING "Git ref for identity_backend FetchContent")
set(TRITON_REPEAT_BACKEND_REPO_TAG      "main" CACHE STRING "Git ref for repeat_backend FetchContent")
set(TRITON_SQUARE_BACKEND_REPO_TAG      "main" CACHE STRING "Git ref for square_backend FetchContent")

#
# Validation
#
if(TRITON_ENABLE_METRICS AND NOT TRITON_ENABLE_STATS)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS=ON requires TRITON_ENABLE_STATS=ON")
endif()

if(TRITON_ENABLE_TRACING AND NOT TRITON_ENABLE_STATS)
  message(FATAL_ERROR "TRITON_ENABLE_TRACING=ON requires TRITON_ENABLE_STATS=ON")
endif()

if(TRITON_ENABLE_METRICS_CPU AND NOT TRITON_ENABLE_METRICS)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS_CPU=ON requires TRITON_ENABLE_METRICS=ON")
endif()

if(TRITON_ENABLE_METRICS_GPU AND NOT TRITON_ENABLE_METRICS)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS_GPU=ON requires TRITON_ENABLE_METRICS=ON")
endif()

if(TRITON_ENABLE_METRICS_GPU AND NOT TRITON_ENABLE_GPU)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS_GPU=ON requires TRITON_ENABLE_GPU=ON")
endif()

if(TRITON_ENABLE_ASAN AND TRITON_ENABLE_GPU)
  message(FATAL_ERROR "TRITON_ENABLE_ASAN=ON requires TRITON_ENABLE_GPU=OFF")
endif()

#
# Internal Triton repos — prefer local checkouts, fall back to FetchContent.
# Override source dirs via -DTRITON_*_SOURCE_DIR=<path>.
# Override fetch tags via -DTRITON_*_REPO_TAG=<branch|tag|SHA>.
#
set(TRITON_COMMON_SOURCE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../common"
    CACHE PATH "Local checkout of triton-inference-server/common (empty = FetchContent)")
set(TRITON_CORE_SOURCE_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/../core"
    CACHE PATH "Local checkout of triton-inference-server/core (empty = FetchContent)")
set(TRITON_BACKEND_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../backend"
    CACHE PATH "Local checkout of triton-inference-server/backend (empty = FetchContent)")

set(TRITON_COMMON_REPO_TAG  "main" CACHE STRING "Git tag/branch for common when using FetchContent")
set(TRITON_CORE_REPO_TAG    "main" CACHE STRING "Git tag/branch for core when using FetchContent")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Git tag/branch for backend when using FetchContent")

include(FetchContent)

macro(triton_add_internal_repo _name _src_dir _repo_url _repo_tag _subdir)
  if(EXISTS "${${_src_dir}}/CMakeLists.txt")
    message(STATUS "triton/${_name}: using local checkout at ${${_src_dir}}")
    add_subdirectory(${${_src_dir}} ${_subdir})
  else()
    message(STATUS "triton/${_name}: local checkout not found — fetching ${${_repo_tag}} from ${_repo_url}")
    FetchContent_Declare(
      ${_name}
      GIT_REPOSITORY ${_repo_url}
      GIT_TAG        ${${_repo_tag}}
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(${_name})
  endif()
endmacro()

set(TRITON_CORE_HEADERS_ONLY OFF)

# Enable protobuf and gRPC in common before adding it as a subdirectory so
# that model_config.pb.h and proto targets are generated when core/src needs them.
set(TRITON_COMMON_ENABLE_PROTOBUF ON CACHE BOOL "Enable protobuf in common" FORCE)
if(TRITON_ENABLE_GRPC)
  set(TRITON_COMMON_ENABLE_GRPC ON CACHE BOOL "Enable gRPC in common" FORCE)
endif()
set(TRITON_COMMON_ENABLE_JSON ON CACHE BOOL "Enable JSON in common" FORCE)

triton_add_internal_repo(
  triton-common TRITON_COMMON_SOURCE_DIR
  "https://github.com/triton-inference-server/common.git"
  TRITON_COMMON_REPO_TAG triton-common)

triton_add_internal_repo(
  triton-core TRITON_CORE_SOURCE_DIR
  "https://github.com/triton-inference-server/core.git"
  TRITON_CORE_REPO_TAG triton-core-top)

triton_add_internal_repo(
  triton-backend TRITON_BACKEND_SOURCE_DIR
  "https://github.com/triton-inference-server/backend.git"
  TRITON_BACKEND_REPO_TAG triton-backend)

# Compatibility aliases for legacy FetchContent variable names still used by
# tests in server/src/test/ (repo-core_SOURCE_DIR, etc.).
set(repo-common_SOURCE_DIR "${TRITON_COMMON_SOURCE_DIR}"  CACHE INTERNAL "")
set(repo-core_SOURCE_DIR   "${TRITON_CORE_SOURCE_DIR}"    CACHE INTERNAL "")
set(repo-backend_SOURCE_DIR "${TRITON_BACKEND_SOURCE_DIR}" CACHE INTERNAL "")

#
# External C++ deps — resolved by Conan; conan_toolchain.cmake sets CMAKE_PREFIX_PATH
#
find_package(re2 REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(Boost REQUIRED)

if(TRITON_ENABLE_GRPC)
  find_package(gRPC CONFIG REQUIRED)
endif()

if(TRITON_ENABLE_HTTP OR TRITON_ENABLE_METRICS OR
   TRITON_ENABLE_SAGEMAKER OR TRITON_ENABLE_VERTEX_AI)
  find_package(Libevent CONFIG REQUIRED)
  find_package(libevhtp CONFIG REQUIRED)
endif()

if(TRITON_ENABLE_METRICS)
  find_package(prometheus-cpp CONFIG REQUIRED)
endif()

if(TRITON_ENABLE_TRACING)
  find_package(opentelemetry-cpp CONFIG REQUIRED)
  find_package(CURL CONFIG REQUIRED)
  find_package(nlohmann_json CONFIG REQUIRED)
endif()

if(TRITON_ENABLE_GCS)
  find_package(google_cloud_cpp_storage CONFIG REQUIRED)
  find_package(Crc32c CONFIG REQUIRED)
endif()

if(TRITON_ENABLE_S3)
  find_package(AWSSDK CONFIG REQUIRED COMPONENTS s3)
endif()

if(TRITON_ENABLE_AZURE_STORAGE)
  find_package(azure-storage-blobs-cpp CONFIG REQUIRED)
endif()

if(TRITON_ENABLE_GPU)
  find_package(cnmem CONFIG REQUIRED)
  find_package(DCGM CONFIG REQUIRED)
endif()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRITON_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)
else()
  set(TRITON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

#
# Triton server executable
#
add_subdirectory(src)

# ---------------------------------------------------------------------------
# Backend implementations
#
# Each backend is included via add_subdirectory.  When enabled but no local
# checkout is found, FetchContent downloads the repo automatically (requires
# network access at configure time).
#
# Build a single backend:
#   cmake --build --preset <preset> --target <name>-backend
#
# Build all enabled backends:
#   cmake --build --preset <preset> --target backends
#
# Install all enabled backends (produces deployable backend packages):
#   cmake --install build/<preset> --prefix /opt/tritonserver
#
# For standalone backend builds (without the full server), backends can link
# against the triton::tritonserver_stub interface target which provides the
# TRITONSERVER_* API headers without requiring libtritonserver.so at build time.
# ---------------------------------------------------------------------------

# Controls whether missing backend checkouts are fetched automatically.
option(TRITON_FETCH_MISSING_BACKENDS
  "Automatically fetch backends via FetchContent when local checkout is absent" ON)

# All backends pass these flags into their sub-build so they skip their own
# FetchContent fetches for common/core (already in scope from this build).
set(_backend_passthrough
  -DTRITON_SKIP_THIRD_PARTY_FETCH=ON
  -DTRITON_ENABLE_GPU=${TRITON_ENABLE_GPU}
)

# _triton_backend_label tracks which add_subdirectory labels were registered
# so we can build the 'backends' aggregate target afterwards.
set(_triton_backend_labels "")

macro(triton_add_backend _opt _dir _repo_name _repo_tag_var _label)
  if(${_opt})
    if(EXISTS "${${_dir}}/CMakeLists.txt")
      message(STATUS "triton/${_repo_name}: using local checkout at ${${_dir}}")
      add_subdirectory("${${_dir}}" "${_label}")
      list(APPEND _triton_backend_labels "${_label}")
    elseif(EXISTS "${${_dir}}")
      # Directory present but no CMakeLists.txt — Python-only backend.
      # No CMake compilation step; deployment is via pip install / wheel.
      message(STATUS "triton/${_repo_name}: Python-only backend at ${${_dir}} — no CMake build step")
    elseif(TRITON_FETCH_MISSING_BACKENDS)
      message(STATUS "triton/${_repo_name}: not found locally — fetching "
              "https://github.com/triton-inference-server/${_repo_name}.git "
              "@${${_repo_tag_var}}")
      FetchContent_Declare(
        ${_repo_name}
        GIT_REPOSITORY "https://github.com/triton-inference-server/${_repo_name}.git"
        GIT_TAG        "${${_repo_tag_var}}"
        GIT_SHALLOW    TRUE
        SOURCE_DIR     "${${_dir}}"
      )
      FetchContent_GetProperties(${_repo_name})
      if(NOT ${_repo_name}_POPULATED)
        FetchContent_Populate(${_repo_name})
      endif()
      if(EXISTS "${${_dir}}/CMakeLists.txt")
        add_subdirectory("${${_dir}}" "${_label}")
        list(APPEND _triton_backend_labels "${_label}")
      else()
        message(STATUS "triton/${_repo_name}: fetched — Python-only backend, no CMake build step")
      endif()
    else()
      message(WARNING
        "${_opt}=ON but '${${_dir}}' not found and "
        "TRITON_FETCH_MISSING_BACKENDS=OFF — skipping ${_repo_name}")
    endif()
  endif()
endmacro()

triton_add_backend(TRITON_ENABLE_PYTHON_BACKEND      TRITON_PYTHON_BACKEND_SOURCE_DIR
  python_backend      TRITON_PYTHON_BACKEND_REPO_TAG      python_backend)
triton_add_backend(TRITON_ENABLE_ONNXRUNTIME_BACKEND TRITON_ONNXRUNTIME_BACKEND_SOURCE_DIR
  onnxruntime_backend TRITON_ONNXRUNTIME_BACKEND_REPO_TAG onnxruntime_backend)
triton_add_backend(TRITON_ENABLE_PYTORCH_BACKEND     TRITON_PYTORCH_BACKEND_SOURCE_DIR
  pytorch_backend     TRITON_PYTORCH_BACKEND_REPO_TAG     pytorch_backend)
triton_add_backend(TRITON_ENABLE_TENSORRT_BACKEND    TRITON_TENSORRT_BACKEND_SOURCE_DIR
  tensorrt_backend    TRITON_TENSORRT_BACKEND_REPO_TAG    tensorrt_backend)
triton_add_backend(TRITON_ENABLE_OPENVINO_BACKEND    TRITON_OPENVINO_BACKEND_SOURCE_DIR
  openvino_backend    TRITON_OPENVINO_BACKEND_REPO_TAG    openvino_backend)
triton_add_backend(TRITON_ENABLE_DALI_BACKEND        TRITON_DALI_BACKEND_SOURCE_DIR
  dali_backend        TRITON_DALI_BACKEND_REPO_TAG        dali_backend)
triton_add_backend(TRITON_ENABLE_FIL_BACKEND         TRITON_FIL_BACKEND_SOURCE_DIR
  fil_backend         TRITON_FIL_BACKEND_REPO_TAG         fil_backend)
triton_add_backend(TRITON_ENABLE_VLLM_BACKEND        TRITON_VLLM_BACKEND_SOURCE_DIR
  vllm_backend        TRITON_VLLM_BACKEND_REPO_TAG        vllm_backend)
triton_add_backend(TRITON_ENABLE_TENSORRTLLM_BACKEND TRITON_TENSORRTLLM_BACKEND_SOURCE_DIR
  tensorrtllm_backend TRITON_TENSORRTLLM_BACKEND_REPO_TAG tensorrtllm_backend)
triton_add_backend(TRITON_ENABLE_IDENTITY_BACKEND    TRITON_IDENTITY_BACKEND_SOURCE_DIR
  identity_backend    TRITON_IDENTITY_BACKEND_REPO_TAG    identity_backend)
triton_add_backend(TRITON_ENABLE_REPEAT_BACKEND      TRITON_REPEAT_BACKEND_SOURCE_DIR
  repeat_backend      TRITON_REPEAT_BACKEND_REPO_TAG      repeat_backend)
triton_add_backend(TRITON_ENABLE_SQUARE_BACKEND      TRITON_SQUARE_BACKEND_SOURCE_DIR
  square_backend      TRITON_SQUARE_BACKEND_REPO_TAG      square_backend)

# 'backends' aggregate target — builds all enabled backends via cmake --target backends.
# Individual targets: cmake --build --preset <p> --target <name>-backend
# (each backend's CMakeLists.txt is expected to define install() rules that place
#  the backend .so under ${CMAKE_INSTALL_PREFIX}/backends/<name>/)
add_custom_target(backends
  COMMENT "Building all enabled backend implementations"
)
foreach(_bl ${_triton_backend_labels})
  # Create a per-backend target alias: python-backend, onnxruntime-backend, etc.
  # The label (e.g. "python_backend") maps to the add_subdirectory binary subdir;
  # cmake --build with that subdir name builds all targets from that backend.
  string(REPLACE "_" "-" _backend_target_name "${_bl}")
  add_custom_target("${_backend_target_name}"
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}/${_bl}"
                             --config $<CONFIG>
    COMMENT "Building ${_bl}"
    VERBATIM
  )
  add_dependencies(backends "${_backend_target_name}")
endforeach()

# ---------------------------------------------------------------------------
# Triton plugins: repository agents and cache providers
#
# These are server-side extension plugins that are shipped alongside the server
# but live in separate repositories.  Each can be built standalone from its own
# checkout, or included here via the same local-checkout / FetchContent pattern.
#
# Build a single plugin:
#   cmake --build --preset <preset> --target <name>
#
# Build all enabled plugins:
#   cmake --build --preset <preset> --target plugins
# ---------------------------------------------------------------------------

option(TRITON_ENABLE_CHECKSUM_REPO_AGENT "Build checksum repository agent" OFF)
option(TRITON_ENABLE_LOCAL_CACHE        "Build local response cache"       OFF)
option(TRITON_ENABLE_REDIS_CACHE        "Build Redis response cache"       OFF)

set(TRITON_CHECKSUM_REPO_AGENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../checksum_repository_agent"
    CACHE PATH "Local checkout of checksum_repository_agent repo")
set(TRITON_LOCAL_CACHE_SOURCE_DIR         "${CMAKE_CURRENT_SOURCE_DIR}/../local_cache"
    CACHE PATH "Local checkout of local_cache repo")
set(TRITON_REDIS_CACHE_SOURCE_DIR         "${CMAKE_CURRENT_SOURCE_DIR}/../redis_cache"
    CACHE PATH "Local checkout of redis_cache repo")

set(TRITON_CHECKSUM_REPO_AGENT_REPO_TAG "main" CACHE STRING "Git ref for checksum_repository_agent FetchContent")
set(TRITON_LOCAL_CACHE_REPO_TAG         "main" CACHE STRING "Git ref for local_cache FetchContent")
set(TRITON_REDIS_CACHE_REPO_TAG         "main" CACHE STRING "Git ref for redis_cache FetchContent")

set(_triton_plugin_labels "")

macro(triton_add_plugin _opt _dir _repo_name _repo_tag_var _label)
  if(${_opt})
    if(EXISTS "${${_dir}}/CMakeLists.txt")
      message(STATUS "triton/${_repo_name}: using local checkout at ${${_dir}}")
      add_subdirectory("${${_dir}}" "${_label}")
    elseif(TRITON_FETCH_MISSING_BACKENDS)
      message(STATUS "triton/${_repo_name}: not found locally — fetching "
              "https://github.com/triton-inference-server/${_repo_name}.git "
              "@${${_repo_tag_var}}")
      FetchContent_Declare(
        ${_repo_name}
        GIT_REPOSITORY "https://github.com/triton-inference-server/${_repo_name}.git"
        GIT_TAG        "${${_repo_tag_var}}"
        GIT_SHALLOW    TRUE
        SOURCE_DIR     "${${_dir}}"
      )
      FetchContent_MakeAvailable(${_repo_name})
    else()
      message(WARNING
        "${_opt}=ON but '${${_dir}}' not found and "
        "TRITON_FETCH_MISSING_BACKENDS=OFF — skipping ${_repo_name}")
    endif()
    list(APPEND _triton_plugin_labels "${_label}")
  endif()
endmacro()

triton_add_plugin(TRITON_ENABLE_CHECKSUM_REPO_AGENT TRITON_CHECKSUM_REPO_AGENT_SOURCE_DIR
  checksum_repository_agent TRITON_CHECKSUM_REPO_AGENT_REPO_TAG checksum_repository_agent)
triton_add_plugin(TRITON_ENABLE_LOCAL_CACHE TRITON_LOCAL_CACHE_SOURCE_DIR
  local_cache TRITON_LOCAL_CACHE_REPO_TAG local_cache)
triton_add_plugin(TRITON_ENABLE_REDIS_CACHE TRITON_REDIS_CACHE_SOURCE_DIR
  redis_cache TRITON_REDIS_CACHE_REPO_TAG redis_cache)

# 'plugins' aggregate target — builds all enabled repo agents and cache providers.
add_custom_target(plugins
  COMMENT "Building all enabled Triton plugin extensions"
)
foreach(_pl ${_triton_plugin_labels})
  string(REPLACE "_" "-" _plugin_target_name "${_pl}")
  add_custom_target("${_plugin_target_name}"
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}/${_pl}"
                             --config $<CONFIG>
    COMMENT "Building ${_pl}"
    VERBATIM
  )
  add_dependencies(plugins "${_plugin_target_name}")
endforeach()

# ---------------------------------------------------------------------------
# Client & perf_analyzer — optional; buildable without the server library.
# When TRITON_SERVER_LIBRARY is empty/not found, downstream CMakeLists.txt
# must link against the stub target (provided via core/include/):
#   target_link_libraries(<target> PRIVATE triton::tritonserver_stub)
# ---------------------------------------------------------------------------
option(TRITON_ENABLE_CLIENT       "Build Triton client libraries"  OFF)
option(TRITON_ENABLE_PERF_ANALYZER "Build perf_analyzer tool"      OFF)

set(TRITON_CLIENT_SOURCE_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/../client"
    CACHE PATH "Local checkout of client repo")
set(TRITON_PERF_ANALYZER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../perf_analyzer"
    CACHE PATH "Local checkout of perf_analyzer repo")

set(TRITON_CLIENT_REPO_TAG       "main" CACHE STRING "Git ref for client FetchContent fallback")
set(TRITON_PERF_ANALYZER_REPO_TAG "main" CACHE STRING "Git ref for perf_analyzer FetchContent fallback")

# Provide a header-only stub target so client/perf_analyzer can build
# without libtritonserver.so present (they communicate via gRPC/HTTP only).
if(TRITON_ENABLE_CLIENT OR TRITON_ENABLE_PERF_ANALYZER)
  if(NOT TARGET triton::tritonserver_stub)
    add_library(tritonserver_stub INTERFACE)
    add_library(triton::tritonserver_stub ALIAS tritonserver_stub)
    target_include_directories(tritonserver_stub INTERFACE
      ${TRITON_CORE_SOURCE_DIR}/include)
  endif()
endif()

if(TRITON_ENABLE_CLIENT)
  triton_add_internal_repo(
    triton-client TRITON_CLIENT_SOURCE_DIR
    "https://github.com/triton-inference-server/client.git"
    TRITON_CLIENT_REPO_TAG client)
endif()

if(TRITON_ENABLE_PERF_ANALYZER)
  if(EXISTS "${TRITON_PERF_ANALYZER_SOURCE_DIR}/CMakeLists.txt")
    message(STATUS "triton/perf_analyzer: using local checkout at ${TRITON_PERF_ANALYZER_SOURCE_DIR}")
    add_subdirectory(${TRITON_PERF_ANALYZER_SOURCE_DIR} perf_analyzer)
  else()
    message(STATUS "triton/perf_analyzer: local checkout not found — fetching ${TRITON_PERF_ANALYZER_REPO_TAG}")
    FetchContent_Declare(
      triton-perf-analyzer
      GIT_REPOSITORY "https://github.com/triton-inference-server/perf_analyzer.git"
      GIT_TAG        ${TRITON_PERF_ANALYZER_REPO_TAG}
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(triton-perf-analyzer)
  endif()
endif()

# ---------------------------------------------------------------------------
# Python wheel targets (Req 29)
# Built AFTER the main CMake compilation as an explicit additional step:
#   cmake --build --preset <preset> --target wheels
#
# Individual wheel targets:
#   tritonserver-wheel  — tritonserver Python SDK wheel (from core/python/)
#   client-wheel        — tritonclient wheels (from client/)
#   wheels              — convenience target that builds all wheels
# ---------------------------------------------------------------------------
find_package(Python3 QUIET COMPONENTS Interpreter)

if(Python3_FOUND)
  # Server Python SDK wheel (core/python/tritonserver package)
  set(_server_py_dir "${CMAKE_CURRENT_SOURCE_DIR}/../core/python")
  if(EXISTS "${_server_py_dir}/setup.py")
    add_custom_target(tritonserver-wheel
      COMMAND ${Python3_EXECUTABLE} -m pip wheel
        --no-deps
        --wheel-dir "${CMAKE_BINARY_DIR}/wheels"
        "${_server_py_dir}"
      COMMENT "Building tritonserver Python wheel"
      VERBATIM
    )
  endif()

  # Client wheels (tritonclient[grpc] + tritonclient[http])
  set(_client_py_dir "${CMAKE_CURRENT_SOURCE_DIR}/../client")
  if(EXISTS "${_client_py_dir}/setup.py" OR EXISTS "${_client_py_dir}/pyproject.toml")
    add_custom_target(client-wheel
      COMMAND ${Python3_EXECUTABLE} -m pip wheel
        --no-deps
        --wheel-dir "${CMAKE_BINARY_DIR}/wheels"
        "${_client_py_dir}"
      COMMENT "Building tritonclient Python wheels"
      VERBATIM
    )
  endif()

  # Convenience umbrella target
  add_custom_target(wheels
    COMMENT "Building all Python wheels into ${CMAKE_BINARY_DIR}/wheels/"
  )
  if(TARGET tritonserver-wheel)
    add_dependencies(wheels tritonserver-wheel)
  endif()
  if(TARGET client-wheel)
    add_dependencies(wheels client-wheel)
  endif()
else()
  message(STATUS "Python3 not found — 'wheels' target will not be available")
endif()
