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
# Only the options guarding the dependency lookups below are declared so far; the
# remaining TRITON_ENABLE_* options still need to be added.
option(TRITON_ENABLE_GPU "Enable GPU support in server" ON)
option(TRITON_ENABLE_HTTP "Include HTTP API in server" ON)
option(TRITON_ENABLE_GRPC "Include GRPC API in server" ON)
option(TRITON_ENABLE_METRICS "Include metrics support in server" ON)
option(TRITON_ENABLE_TRACING "Include tracing support in server" OFF)
option(TRITON_ENABLE_SAGEMAKER "Include AWS SageMaker API in server" OFF)
option(TRITON_ENABLE_VERTEX_AI "Include Vertex AI API in server" OFF)

#
# Dependencies
#
# Duplicated from src/CMakeLists.txt. Imported targets created here are inherited
# by add_subdirectory(src), so resolving them at the top level fails fast before
# descending. Guards must stay in sync with the copies in src/.
#
include(TritonDependencies)

triton_server_conan_add_remote()

# Versions track what gRPC v1.81.1 vendors in its third_party submodules, since
# that is what third_party/CMakeLists.txt builds against today:
#   protobuf v33.5 -> conan 6.33.5, abseil 20250512.1, googletest v1.17.0
set(TRITON_RE2_VERSION "20230301" CACHE STRING "re2 version resolved via Conan")
set(TRITON_PROTOBUF_VERSION "3.21.12" CACHE STRING "protobuf version resolved via Conan")
set(TRITON_LIBEVENT_VERSION "2.1.12" CACHE STRING "libevent version resolved via Conan")
set(TRITON_LIBEVHTP_VERSION "1.2.18" CACHE STRING "libevhtp version resolved via Conan")
set(TRITON_GRPC_VERSION "1.81.1" CACHE STRING "gRPC version resolved via Conan")

triton_server_conan_provide_package(NAME re2      REF re2/${TRITON_RE2_VERSION})
triton_server_conan_provide_package(NAME Protobuf REF protobuf/${TRITON_PROTOBUF_VERSION} CONFIG)

if(TRITON_ENABLE_GPU)
  find_package(CUDAToolkit REQUIRED)
endif()

if(TRITON_ENABLE_HTTP
    OR TRITON_ENABLE_METRICS
    OR TRITON_ENABLE_SAGEMAKER
    OR TRITON_ENABLE_VERTEX_AI)
  triton_server_conan_provide_package(NAME Libevent REF libevent/${TRITON_LIBEVENT_VERSION} CONFIG)
  triton_server_conan_provide_package(NAME libevhtp REF libevhtp/${TRITON_LIBEVHTP_VERSION} CONFIG)
endif()

if(TRITON_ENABLE_TRACING)
  find_package(absl CONFIG REQUIRED)
  find_package(CURL CONFIG REQUIRED)
  find_package(nlohmann_json CONFIG REQUIRED)
  find_package(opentelemetry-cpp CONFIG REQUIRED)
endif()

if(TRITON_ENABLE_GRPC)
  triton_server_conan_provide_package(NAME gRPC REF grpc/${TRITON_GRPC_VERSION} CONFIG)
endif()

triton_server_dependency_report()

add_subdirectory(src)
