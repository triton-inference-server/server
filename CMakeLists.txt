# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

cmake_minimum_required(VERSION 3.31.8)

# Must precede project(): TRITON_VERSION_NUMERIC is needed to satisfy VERSION below.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(TritonUtils)
include(TritonCompilerCache)

triton_server_read_version("${CMAKE_CURRENT_SOURCE_DIR}/TRITON_VERSION"
                           TRITON_VERSION TRITON_VERSION_NUMERIC)

project(tritonserver VERSION ${TRITON_VERSION_NUMERIC} LANGUAGES C CXX)

# C++ standard is defined once here and inherited by every subdirectory. CACHE is
# required for -DCMAKE_CXX_STANDARD=<n> to take effect: a plain set() would shadow
# the command-line value instead of deferring to it.
set(CMAKE_CXX_STANDARD 17 CACHE STRING
    "The minimum C++ standard which features are requested to build this target.")
set_property(CACHE CMAKE_CXX_STANDARD PROPERTY STRINGS 17 20 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

triton_server_cache_enable_compiler()

# Conan's CMakeDeps configs call check_build_type_defined() and abort when no
# build type is set, so single-config generators need a default here.
get_property(_triton_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _triton_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               Debug Release RelWithDebInfo MinSizeRel)
endif()

#
# Feature options
#
option(TRITON_ENABLE_LOGGING "Include logging support in server" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in server" ON)
option(TRITON_ENABLE_TRACING "Include tracing support in server" OFF)
option(TRITON_ENABLE_NVTX "Include NVTX support in server" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in server" ON)
option(TRITON_ENABLE_MALI_GPU "Enable Arm Mali GPU support in server" OFF)
option(TRITON_IGPU_BUILD "Enable options for iGPU compilation in server" OFF)
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

# Testing and diagnostics
option(TRITON_ENABLE_TENSORRT "Include TensorRT backend in server" OFF)
option(TRITON_ENABLE_TESTS "Build unit tests, which require GoogleTest" OFF)
option(TRITON_ENABLE_ASAN "Build with address sanitizer" OFF)

set(TRITON_MIN_COMPUTE_CAPABILITY "7.5" CACHE STRING
    "The minimum CUDA compute capability supported by Triton")

#
# Option consistency
#
# Same constraints the previous root CMakeLists enforced by hand, one call each.
#
triton_server_require_option(TRITON_ENABLE_METRICS TRITON_ENABLE_STATS)
triton_server_require_option(TRITON_ENABLE_TRACING TRITON_ENABLE_STATS)
triton_server_require_option(TRITON_ENABLE_METRICS_CPU TRITON_ENABLE_METRICS)
triton_server_require_option(TRITON_ENABLE_METRICS_GPU TRITON_ENABLE_METRICS)
triton_server_require_option(TRITON_ENABLE_METRICS_GPU TRITON_ENABLE_GPU)
triton_server_forbid_option(TRITON_ENABLE_ASAN TRITON_ENABLE_GPU)

#
# Dependencies
#
# Duplicated from src/CMakeLists.txt. Imported targets created here are inherited
# by add_subdirectory(src), so resolving them at the top level fails fast before
# descending. Guards must stay in sync with the copies in src/.
#
include(TritonDependencies)

triton_server_conan_add_remote()

# Dependency versions.
#
# abseil, protobuf, re2, c-ares and googletest are deliberately absent: gRPC
# vendors them and the Triton gRPC recipe builds those submodules, so pinning
# them here would express a version that nothing reads. See
# triton_server_grpc_provide_bundle().
set(TRITON_GRPC_VERSION "1.81.1" CACHE STRING "gRPC version")
set(TRITON_GRPC_USER_CHANNEL "@tritonserver/stable" CACHE STRING
    "Conan user/channel for the Triton gRPC build; empty uses ConanCenter")
set(TRITON_GRPC_REF "grpc/${TRITON_GRPC_VERSION}${TRITON_GRPC_USER_CHANNEL}")

set(TRITON_RAPIDJSON_VERSION "cci.20230929" CACHE STRING "RapidJSON version")
set(TRITON_LIBEVENT_VERSION "2.1.12" CACHE STRING "libevent version")
set(TRITON_LIBEVHTP_VERSION "1.2.18" CACHE STRING "libevhtp version")
set(TRITON_CURL_VERSION "8.18.0" CACHE STRING "libcurl version")
set(TRITON_NLOHMANN_JSON_VERSION "3.11.3" CACHE STRING "nlohmann_json version")
set(TRITON_OPENTELEMETRY_VERSION "1.9.1" CACHE STRING "opentelemetry-cpp version")
set(TRITON_PROMETHEUS_VERSION "1.2.4" CACHE STRING "prometheus-cpp version")
set(TRITON_CRC32C_VERSION "1.1.2" CACHE STRING "crc32c version")
set(TRITON_GOOGLE_CLOUD_VERSION "2.28.0" CACHE STRING "google-cloud-cpp version")
set(TRITON_AWS_SDK_VERSION "1.11.692" CACHE STRING "aws-sdk-cpp version")
set(TRITON_AZURE_SDK_VERSION "1.16.1" CACHE STRING "azure-sdk-for-cpp version")
set(TRITON_AZURE_IOT_VERSION "lts_03_2024_ref02" CACHE STRING "azure-iot-sdk-c version")
set(TRITON_CNMEM_VERSION "1.0.0" CACHE STRING "cnmem version")
set(TRITON_GTEST_VERSION "1.14.0" CACHE STRING "GoogleTest version")

# Threads is provided by the toolchain, not by Conan; common/CMakeLists.txt:53
# requires it.
find_package(Threads REQUIRED)

# common/CMakeLists.txt:48 requires RapidJSON.
triton_server_dependency_provider(NAME RapidJSON REF rapidjson/${TRITON_RAPIDJSON_VERSION} CONFIG)

if(TRITON_ENABLE_GRPC)
  # gRPC plus the abseil/protobuf/re2/c-ares it vendors, from one package.
  triton_server_grpc_provide_bundle()
else()
  # Without gRPC nothing bundles these, so resolve them on their own.
  triton_server_dependency_provider(NAME re2      REF re2/20230301)
  triton_server_dependency_provider(NAME Protobuf REF protobuf/6.33.5 CONFIG)
  triton_server_dependency_provider(NAME absl     REF abseil/20250512.1 CONFIG)
endif()

if(TRITON_ENABLE_GPU)
  # Tied to the host driver; no Conan reference exists.
  triton_server_dependency_provider(NAME CUDAToolkit ALLOW_SYSTEM)
  triton_server_dependency_provider(NAME cnmem REF cnmem/${TRITON_CNMEM_VERSION} CONFIG)
endif()

if(TRITON_ENABLE_METRICS)
  triton_server_dependency_provider(NAME prometheus-cpp REF prometheus-cpp/${TRITON_PROMETHEUS_VERSION} CONFIG)
endif()

if(TRITON_ENABLE_METRICS_GPU)
  # Ships with the datacenter GPU manager and is tied to the host driver, so
  # whatever is installed is the right version. Note the capitalisation: the
  # config it ships is DCGMConfig.cmake and find_package() is case-sensitive.
  # REQUIRED because no third_party fallback could build it.
  triton_server_dependency_provider(NAME DCGM CONFIG ALLOW_SYSTEM REQUIRED)
endif()

if(TRITON_ENABLE_HTTP
    OR TRITON_ENABLE_METRICS
    OR TRITON_ENABLE_SAGEMAKER
    OR TRITON_ENABLE_VERTEX_AI)
  triton_server_dependency_provider(NAME Libevent REF libevent/${TRITON_LIBEVENT_VERSION} CONFIG)
  triton_server_dependency_provider(NAME libevhtp REF libevhtp/${TRITON_LIBEVHTP_VERSION} CONFIG)

  # src/CMakeLists.txt:197,364 and src/test/CMakeLists.txt:60,70 consume libevent
  # through the module-style LIBEVENT_INCLUDE_DIRS / LIBEVENT_LIBRARIES that
  # FindLibEvent.cmake would define. Conan's config package exports the
  # libevent::libevent target instead, leaving those variables empty -- which
  # surfaces as "event2/buffer.h: No such file or directory" at compile time
  # rather than as a configure error. Bridge them once here instead of editing
  # four call sites.
  if(TARGET libevent::libevent AND NOT LIBEVENT_LIBRARIES)
    # The imported target carries no INTERFACE_INCLUDE_DIRECTORIES at this point;
    # CMakeDeps records the paths in its data file variables instead.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _triton_cfg)
    if(libevent_INCLUDE_DIRS_${_triton_cfg})
      set(LIBEVENT_INCLUDE_DIRS "${libevent_INCLUDE_DIRS_${_triton_cfg}}")
    elseif(libevent_PACKAGE_FOLDER_${_triton_cfg})
      set(LIBEVENT_INCLUDE_DIRS "${libevent_PACKAGE_FOLDER_${_triton_cfg}}/include")
    endif()
    set(LIBEVENT_LIBRARIES libevent::libevent)
    message(STATUS "libevent: bridged LIBEVENT_INCLUDE_DIRS=${LIBEVENT_INCLUDE_DIRS}")
  endif()
endif()

if(TRITON_ENABLE_TRACING)
  triton_server_dependency_provider(NAME CURL              REF libcurl/${TRITON_CURL_VERSION} CONFIG)
  triton_server_dependency_provider(NAME nlohmann_json     REF nlohmann_json/${TRITON_NLOHMANN_JSON_VERSION} CONFIG)
  triton_server_dependency_provider(NAME opentelemetry-cpp REF opentelemetry-cpp/${TRITON_OPENTELEMETRY_VERSION} CONFIG)
endif()

if(TRITON_ENABLE_GCS)
  triton_server_dependency_provider(NAME google_cloud_cpp_storage REF google-cloud-cpp/${TRITON_GOOGLE_CLOUD_VERSION} CONFIG)
  triton_server_dependency_provider(NAME Crc32c REF crc32c/${TRITON_CRC32C_VERSION} CONFIG)
endif()

if(TRITON_ENABLE_S3)
  triton_server_dependency_provider(NAME AWSSDK REF aws-sdk-cpp/${TRITON_AWS_SDK_VERSION} CONFIG)
endif()

if(TRITON_ENABLE_AZURE_STORAGE)
  triton_server_dependency_provider(NAME azure-storage-blobs-cpp REF azure-sdk-for-cpp/${TRITON_AZURE_SDK_VERSION} CONFIG)
  # No recipe on any remote; comes from conan/recipes/azure-iot-sdk-c.
  triton_server_dependency_provider(NAME azure_iot_sdks REF azure-iot-sdk-c/${TRITON_AZURE_IOT_VERSION}
                                    RECIPE azure-iot-sdk-c CONFIG)
endif()

# GoogleTest is not optional in practice: src/CMakeLists.txt:713 adds test/
# unconditionally, and src/test/CMakeLists.txt:32 calls find_package(GTest
# REQUIRED). TRITON_ENABLE_TESTS therefore only selects whether the tests are
# built, not whether GoogleTest must be resolvable.
triton_server_dependency_provider(NAME GTest REF gtest/${TRITON_GTEST_VERSION} CONFIG)

# src/test asks for GTest in module mode, and CMake's FindGTest looks for
# GTEST_LIBRARY / GTEST_INCLUDE_DIR / GTEST_MAIN_LIBRARY rather than the Conan
# config. Point GTEST_ROOT at the package so the module finds them -- the old
# root file did the same thing (CMakeLists.txt.bkp:207).
_triton_server_conan_package_folder("gtest/${TRITON_GTEST_VERSION}" TRITON_GTEST_ROOT)
if(TRITON_GTEST_ROOT)
  set(GTEST_ROOT "${TRITON_GTEST_ROOT}" CACHE PATH "GoogleTest root for FindGTest" FORCE)
endif()

triton_server_conan_publish_targets(TRITON_CONAN_PUBLISHED)
message(STATUS "Conan-published packages: ${TRITON_CONAN_PUBLISHED}")

triton_server_dependency_report()

#
# Triton repositories
#
# Must precede add_subdirectory(src): src/CMakeLists.txt declares and populates
# repo-common, repo-core and repo-backend, and needs these variables set first.
#
include(TritonRepos)

triton_server_fetchcontent_declare_repos()

add_subdirectory(src)
