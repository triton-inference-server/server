# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(tritononnxruntimebackend LANGUAGES C CXX)

#
# Options
#
# To build the ONNX Runtime backend you must either:
#
#   - Point to an already built ONNX Runtime using
#     TRITON_ONNXRUNTIME_INCLUDE_PATHS and
#     TRITON_ONNXRUNTIME_LIB_PATHS
#
#   or:
#
#   - Set TRITON_BUILD_ONNXRUNTIME_VERSION to the version of ONNX
#     Runtime that you want to be built for the backend.
#
#   - Set TRITON_BUILD_CONTAINER to the Triton container to use as a
#     base for the build. On linux you can instead set
#     TRITON_BUILD_CONTAINER_VERSION to the Triton version that you
#     want to target with the build and the corresponding container
#     from NGC will be used.
#
#   - Optionally set TRITON_BUILD_CUDA_VERSION and
#     TRITON_BUILD_CUDA_HOME. If not set these are automatically set
#     by using the standard cuda install location. For example on
#     windows these will be automatically set based on CUDA_PATH, for
#     example:
#
#         TRITON_BUILD_CUDA_VERSION=11.1
#         TRITON_BUILD_CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\v11.1"
#
#   - If you want TensorRT support set
#     TRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON and set TRITON_BUILD_TENSORRT_HOME.
#
#     Optionally set TRITON_ONNX_TENSORRT_REPO_TAG to specify a branch in https://github.com/onnx/onnx-tensorrt repo
#     example:
#         TRITON_ONNX_TENSORRT_REPO_TAG=master
#     This enables using a version of tensorrt which is not yet supported in ONNXRuntime release branch.
#     By default we pick the default branch which comes with the requested version of onnxruntime.
#
#     Optionally set TRT_VERSION to specify the version of TRT which is being used.
#     This along with TRITON_BUILD_ONNXRUNTIME_VERSION is used to pick the right onnx tensorrt parser version.
#     When TRITON_ONNX_TENSORRT_REPO_TAG is set TRT_VERSION is ignored.
#     When neither TRITON_ONNX_TENSORRT_REPO_TAG or TRT_VERSION are set
#     the default parser version which comes with ORT is picked.
#
#   - If you want OpenVINO support set
#     TRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON and set
#     TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION to the OpenVino
#     version that is compatible with the specified version of ONNX
#     Runtime.
#
#   - Optionally set TRITON_BUILD_TARGET_PLATFORM to either linux, windows or
#     igpu. If not set, the current platform will be used. If building on
#     Jetpack, always set to igpu to avoid misdetection.
#
#   - If you want to disable GPU usage, set TRITON_ENABLE_GPU=OFF.
#    This will make builds with CUDA and TensorRT flags to fail.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_ONNXRUNTIME_TENSORRT
  "Enable TensorRT execution provider for ONNXRuntime backend in server" OFF)
option(TRITON_ENABLE_ONNXRUNTIME_OPENVINO
  "Enable OpenVINO execution provider for ONNXRuntime backend in server" OFF)
set(TRITON_BUILD_CONTAINER "" CACHE STRING "Triton container to use a base for build")
set(TRITON_BUILD_CONTAINER_VERSION "" CACHE STRING "Triton container version to target")
set(TRITON_BUILD_ONNXRUNTIME_VERSION "" CACHE STRING "ONNXRuntime version to build")
set(TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION "" CACHE STRING "ONNXRuntime OpenVINO version to build")
set(TRITON_BUILD_TARGET_PLATFORM "" CACHE STRING "Target platform for ONNXRuntime build")
set(TRITON_BUILD_CUDA_VERSION "" CACHE STRING "Version of CUDA install")
set(TRITON_BUILD_CUDA_HOME "" CACHE PATH "Path to CUDA install")
set(TRITON_BUILD_CUDNN_HOME "" CACHE PATH "Path to CUDNN install")
set(TRITON_BUILD_TENSORRT_HOME "" CACHE PATH "Path to TensorRT install")
set(TRITON_ONNXRUNTIME_INCLUDE_PATHS "" CACHE PATH "Paths to ONNXRuntime includes")
set(TRITON_ONNX_TENSORRT_REPO_TAG "" CACHE STRING "Tag for onnx-tensorrt repo")
set(TRT_VERSION "" CACHE STRING "TRT version for this build.")
set(TRITON_ONNXRUNTIME_LIB_PATHS "" CACHE PATH "Paths to ONNXRuntime libraries")

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

# Use C++17 standard as Triton's minimum required.
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

if (WIN32)
  if(TRITON_ENABLE_ONNXRUNTIME_OPENVINO)
    message(FATAL_ERROR
      "TRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON not supported for Windows")
  endif()
endif() # WIN32

if (NOT TRITON_ENABLE_GPU)
  if (TRITON_ENABLE_ONNXRUNTIME_TENSORRT)
    message(FATAL_ERROR "TRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON requires TRITON_ENABLE_GPU=ON")
  endif() # TRITON_ENABLE_ONNXRUNTIME_TENSORRT
endif() # NOT TRITON_ENABLE_GPU

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TRITON_ONNXRUNTIME_DOCKER_BUILD OFF)
# Download onnxruntime
include(cmake/download_onnxruntime.cmake)
if(TRITON_ONNXRUNTIME_LIB_PATHS STREQUAL "")
  set(TRITON_ONNXRUNTIME_DOCKER_BUILD ON)
endif()

message(STATUS "Using Onnxruntime docker: ${TRITON_ONNXRUNTIME_DOCKER_BUILD}")

if(NOT TRITON_ONNXRUNTIME_DOCKER_BUILD)
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS ${TRITON_ONNXRUNTIME_LIB_PATHS} REQUIRED)
  if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
    find_library(OV_LIBRARY
      NAMES openvino
      PATHS ${TRITON_ONNXRUNTIME_LIB_PATHS})
  endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

else()

  if(NOT TRITON_BUILD_CONTAINER AND NOT TRITON_BUILD_CONTAINER_VERSION)
    message(FATAL_ERROR
      "TRITON_BUILD_ONNXRUNTIME_VERSION requires TRITON_BUILD_CONTAINER or TRITON_BUILD_CONTAINER_VERSION")
  endif()

  if(NOT TRITON_BUILD_CONTAINER)
    set(TRITON_BUILD_CONTAINER "nvcr.io/nvidia/tritonserver:${TRITON_BUILD_CONTAINER_VERSION}-py3-min")
  endif()

  set(TRITON_ONNXRUNTIME_DOCKER_IMAGE "tritonserver_onnxruntime")
  set(TRITON_ONNXRUNTIME_DOCKER_MEMORY "$<IF:$<BOOL:WIN32>,32g,8g>")
  set(TRITON_ONNXRUNTIME_INCLUDE_PATHS "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/include")
  set(TRITON_ONNXRUNTIME_LIB_PATHS "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib")
  if (WIN32)
    set(ONNXRUNTIME_LIBRARY "onnxruntime")
  else()
    set(ONNXRUNTIME_LIBRARY "libonnxruntime.so")
  endif() # WIN32
  if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
    set(OV_LIBRARY "libopenvino.so")
  endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO
endif()

#
# Dependencies
#
# FetchContent's composability isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
endif() # TRITON_ENABLE_GPU

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_onnxruntime.ldscript libtriton_onnxruntime.ldscript COPYONLY)

add_library(
  triton-onnxruntime-backend SHARED
  src/onnxruntime.cc
  src/onnxruntime_loader.cc
  src/onnxruntime_loader.h
  src/onnxruntime_utils.cc
  src/onnxruntime_utils.h
)

add_library(
  TritonOnnxRuntimeBackend::triton-onnxruntime-backend ALIAS triton-onnxruntime-backend
)

target_include_directories(
  triton-onnxruntime-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TRITON_ONNXRUNTIME_INCLUDE_PATHS}
)

target_compile_features(triton-onnxruntime-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-onnxruntime-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor>
)

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    triton-onnxruntime-backend
    PRIVATE TRITON_ENABLE_GPU=1
  )
endif() # TRITON_ENABLE_GPU
if(${TRITON_ENABLE_ONNXRUNTIME_TENSORRT})
  target_compile_definitions(
    triton-onnxruntime-backend
    PRIVATE TRITON_ENABLE_ONNXRUNTIME_TENSORRT=1
  )
endif() # TRITON_ENABLE_ONNXRUNTIME_TENSORRT
if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
  target_compile_definitions(
    triton-onnxruntime-backend
    PRIVATE TRITON_ENABLE_ONNXRUNTIME_OPENVINO=1
  )
endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

if (WIN32)
set_target_properties(
  triton-onnxruntime-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_onnxruntime
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
)
else ()
set_target_properties(
  triton-onnxruntime-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_onnxruntime
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_onnxruntime.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_onnxruntime.ldscript"
)
endif()

FOREACH(p ${TRITON_ONNXRUNTIME_LIB_PATHS})
  target_link_directories(
    triton-onnxruntime-backend
    PRIVATE ${p}
  )
ENDFOREACH(p)

target_link_libraries(
  triton-onnxruntime-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    ${TRITON_ONNXRUNTIME_LDFLAGS}
    ${ONNXRUNTIME_LIBRARY}
)

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-onnxruntime-backend
    PRIVATE
      CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
  target_link_libraries(
    triton-onnxruntime-backend
    PRIVATE
      ${OV_LIBRARY}
  )
endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

#
# Build the ONNX Runtime libraries using docker.
#
if(TRITON_ONNXRUNTIME_DOCKER_BUILD)
  set(_GEN_FLAGS "")
  if(NOT ${TRITON_BUILD_TARGET_PLATFORM} STREQUAL "")
    set(_GEN_FLAGS ${_GEN_FLAGS} "--target-platform=${TRITON_BUILD_TARGET_PLATFORM}")
  endif() # TRITON_BUILD_TARGET_PLATFORM
  if(NOT ${TRITON_BUILD_CUDA_VERSION} STREQUAL "")
    set(_GEN_FLAGS ${_GEN_FLAGS} "--cuda-version=${TRITON_BUILD_CUDA_VERSION}")
  endif() # TRITON_BUILD_CUDA_VERSION
  if(NOT ${TRITON_BUILD_CUDA_HOME} STREQUAL "")
    set(_GEN_FLAGS ${_GEN_FLAGS} "--cuda-home=${TRITON_BUILD_CUDA_HOME}")
  endif() # TRITON_BUILD_CUDA_HOME
  if(NOT ${TRITON_BUILD_CUDNN_HOME} STREQUAL "")
    set(_GEN_FLAGS ${_GEN_FLAGS} "--cudnn-home=${TRITON_BUILD_CUDNN_HOME}")
  endif() # TRITON_BUILD_CUDNN_HOME
  if(NOT ${TRITON_BUILD_TENSORRT_HOME} STREQUAL "")
    set(_GEN_FLAGS ${_GEN_FLAGS} "--tensorrt-home=${TRITON_BUILD_TENSORRT_HOME}")
  endif() # TRITON_BUILD_TENSORRT_HOME
  if(${TRITON_ENABLE_ONNXRUNTIME_TENSORRT})
    set(_GEN_FLAGS ${_GEN_FLAGS} "--ort-tensorrt")
  endif() # TRITON_ENABLE_ONNXRUNTIME_TENSORRT
  if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
    set(_GEN_FLAGS ${_GEN_FLAGS} "--ort-openvino=${TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION}")
  endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

  set(ENABLE_GPU_EXTRA_ARGS "")
  if(${TRITON_ENABLE_GPU})
    set(ENABLE_GPU_EXTRA_ARGS "--enable-gpu")
  endif() # TRITON_ENABLE_GPU

  if (WIN32)
    add_custom_command(
      OUTPUT
        onnxruntime/lib/${ONNXRUNTIME_LIBRARY}
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_ort_dockerfile.py --triton-container="${TRITON_BUILD_CONTAINER}" --ort-version="${TRITON_BUILD_ONNXRUNTIME_VERSION}" --trt-version="${TRT_VERSION}" --onnx-tensorrt-tag="${TRITON_ONNX_TENSORRT_REPO_TAG}" ${_GEN_FLAGS} --output=Dockerfile.ort ${ENABLE_GPU_EXTRA_ARGS}
      COMMAND docker build --memory ${TRITON_ONNXRUNTIME_DOCKER_MEMORY} --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE} --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE}_cache0 --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE}_cache1 -t ${TRITON_ONNXRUNTIME_DOCKER_IMAGE} -f ./Dockerfile.ort ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND powershell.exe -noprofile -c "docker rm onnxruntime_backend_ort > $null 2>&1; if ($LASTEXITCODE) { 'error ignored...' }; exit 0"
      COMMAND docker create --name onnxruntime_backend_ort ${TRITON_ONNXRUNTIME_DOCKER_IMAGE}
      COMMAND rmdir /s/q onnxruntime
      COMMAND docker cp onnxruntime_backend_ort:/opt/onnxruntime onnxruntime
      COMMAND docker rm onnxruntime_backend_ort
      COMMENT "Building ONNX Runtime"
    )
  else()
    add_custom_command(
      OUTPUT
        onnxruntime/lib/${ONNXRUNTIME_LIBRARY}
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_ort_dockerfile.py  --ort-build-config="${CMAKE_BUILD_TYPE}" --triton-container="${TRITON_BUILD_CONTAINER}" --ort-version="${TRITON_BUILD_ONNXRUNTIME_VERSION}" --trt-version="${TRT_VERSION}" --onnx-tensorrt-tag="${TRITON_ONNX_TENSORRT_REPO_TAG}" ${_GEN_FLAGS} --output=Dockerfile.ort ${ENABLE_GPU_EXTRA_ARGS}
      COMMAND docker build --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE} --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE}_cache0 --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE}_cache1 -t ${TRITON_ONNXRUNTIME_DOCKER_IMAGE} -f ./Dockerfile.ort ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND docker rm onnxruntime_backend_ort || echo 'error ignored...' || true
      COMMAND docker create --name onnxruntime_backend_ort ${TRITON_ONNXRUNTIME_DOCKER_IMAGE}
      COMMAND rm -fr onnxruntime
      COMMAND docker cp onnxruntime_backend_ort:/opt/onnxruntime onnxruntime
      COMMAND docker rm onnxruntime_backend_ort
      COMMENT "Building ONNX Runtime"
    )
  endif() # WIN32

  add_custom_target(ort_target DEPENDS onnxruntime/lib/${ONNXRUNTIME_LIBRARY})
  add_library(onnxruntime-library SHARED IMPORTED GLOBAL)
  add_dependencies(onnxruntime-library ort_target)
  add_dependencies(triton-onnxruntime-backend onnxruntime-library)

  if (WIN32)
    set_target_properties(
      onnxruntime-library
      PROPERTIES
        IMPORTED_LOCATION onnxruntime/bin/${ONNXRUNTIME_LIBRARY}
    )
  else()
    set_target_properties(
      onnxruntime-library
      PROPERTIES
        IMPORTED_LOCATION onnxruntime/lib/${ONNXRUNTIME_LIBRARY}
    )
  endif() # WIN32
endif() # TRITON_ONNXRUNTIME_DOCKER_BUILD

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonOnnxRuntimeBackend)

install(
  TARGETS
    triton-onnxruntime-backend
  EXPORT
    triton-onnxruntime-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
)

# For Jetson, we build the onnxruntime backend once and re-use
# that tar file. We copy over the libraries and other requirements
# prior to running this build and therefore these set of install
# commands are not needed.
if(TRITON_ONNXRUNTIME_DOCKER_BUILD OR DEFINED TRITON_ONNXRUNTIME_PACKAGE_URL)
  install(
    DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
    PATTERN *lib EXCLUDE
    PATTERN *bin EXCLUDE
    PATTERN *include EXCLUDE
    PATTERN *test EXCLUDE
  )

  install(
    DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/bin/
    USE_SOURCE_PERMISSIONS
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
  )

  if (NOT WIN32)
    install(
      DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib/
      USE_SOURCE_PERMISSIONS
      DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
    )

    install(
      DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/test
      USE_SOURCE_PERMISSIONS
      DESTINATION ${CMAKE_INSTALL_PREFIX}
    )
  endif() # NOT WIN32
endif() # TRITON_ONNXRUNTIME_DOCKER_BUILD

install(
  EXPORT
    triton-onnxruntime-backend-targets
  FILE
    TritonOnnxRuntimeBackendTargets.cmake
  NAMESPACE
    TritonOnnxRuntimeBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonOnnxRuntimeBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-onnxruntime-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendTargets.cmake
  NAMESPACE TritonOnnxRuntimeBackend::
)

export(PACKAGE TritonOnnxRuntimeBackend)
