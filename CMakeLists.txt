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
include(TritonConanGraph)

triton_server_conan_add_remote()

# Dependency versions.
#
# abseil, protobuf, re2 and c-ares are pinned to what gRPC v1.81.1 vendors,
# resolved from its third_party submodule SHAs rather than guessed. They were
# briefly absent from this list, while the gRPC recipe compiled them as
# submodules; they are ordinary packages again so that everything else -- most
# immediately opentelemetry-cpp, which needs protobuf -- shares one copy.
set(TRITON_ABSEIL_VERSION "20250512.1" CACHE STRING "abseil version")
set(TRITON_PROTOBUF_VERSION "6.33.5" CACHE STRING "protobuf version (upstream v33.5)")
set(TRITON_RE2_VERSION "20230301" CACHE STRING "re2 version")
set(TRITON_CARES_VERSION "1.34.6" CACHE STRING "c-ares version")
set(TRITON_GRPC_VERSION "1.81.1" CACHE STRING "gRPC version")
# Published without a user/channel, deliberately shadowing ConanCenter's
# grpc/1.81.1 -- which is the one that does not compile, since it pairs this gRPC
# with abseil/20240116.2.
#
# The shadowing is not merely cosmetic. Recipes that depend on gRPC pin it by
# plain reference (google-cloud-cpp requires grpc/1.54.3), and Conan treats
# grpc/<v>@user/channel as a different package entirely, so an override cannot
# redirect them. Only a plain reference can be reconciled with force=True.
set(TRITON_GRPC_USER_CHANNEL "" CACHE STRING
    "Conan user/channel suffix for the gRPC reference; empty is the Triton build")
set(TRITON_GRPC_REF "grpc/${TRITON_GRPC_VERSION}${TRITON_GRPC_USER_CHANNEL}")

set(TRITON_RAPIDJSON_VERSION "cci.20230929" CACHE STRING "RapidJSON version")
set(TRITON_BOOST_VERSION "1.85.0" CACHE STRING "Boost version")
set(TRITON_LIBEVENT_VERSION "2.1.12" CACHE STRING "libevent version")
set(TRITON_LIBEVHTP_VERSION "1.2.18" CACHE STRING "libevhtp version")
set(TRITON_CURL_VERSION "8.18.0" CACHE STRING "libcurl version")
set(TRITON_NLOHMANN_JSON_VERSION "3.11.3" CACHE STRING "nlohmann_json version")
# 1.22.0 rather than the v1.13.0 third_party pins, or the stale 1.9.1 that was
# carried over here: CMake 4 removed compatibility below policy 3.5, and
# v1.13.0 declares cmake_minimum_required(VERSION 3.1), so it cannot configure
# at all. 1.22.0 is the first release ConanCenter carries whose minimum (3.14)
# clears that bar. Building v1.13.0 ourselves would hit the same wall, since
# the constraint is upstream's.
set(TRITON_OPENTELEMETRY_VERSION "1.22.0" CACHE STRING "opentelemetry-cpp version")
set(TRITON_PROMETHEUS_VERSION "1.2.4" CACHE STRING "prometheus-cpp version")
set(TRITON_CRC32C_VERSION "1.1.2" CACHE STRING "crc32c version")
set(TRITON_GOOGLE_CLOUD_VERSION "2.28.0" CACHE STRING "google-cloud-cpp version")
set(TRITON_AWS_SDK_VERSION "1.11.692" CACHE STRING "aws-sdk-cpp version")
set(TRITON_AZURE_SDK_VERSION "1.16.1" CACHE STRING "azure-sdk-for-cpp version")
set(TRITON_AZURE_IOT_VERSION "lts_03_2024_ref02" CACHE STRING "azure-iot-sdk-c version")
set(TRITON_CNMEM_VERSION "1.0.0" CACHE STRING "cnmem version")
set(TRITON_GTEST_VERSION "1.14.0" CACHE STRING "GoogleTest version")

# Dependencies are declared first and resolved in a single Conan graph.
#
# One `conan install` per package produced one graph per package, and anything
# appearing in several of them was resolved independently. Two graphs then
# claimed the same CMake targets in one scope -- re2::re2 and libevhtp::evhtp
# both failed with "CONAN_LIB::... but the target was not found" once GCS was
# enabled and google-cloud-cpp brought its own gRPC and re2. Separate output
# folders stopped the files overwriting each other but could not stop the target
# collision. Only one graph can, because only then does Conan reconcile shared
# packages to a single version.
#
# See cmake/TritonConanGraph.cmake.

# gRPC and the packages it is built against. These used to be vendored inside
# the gRPC package (gRPC_*_PROVIDER=module), which hid them from Conan so no
# other consumer could share them -- opentelemetry-cpp pulling protobuf brought
# a second abseil and collided in CMake's target namespace. They are ordinary
# packages now, at the versions gRPC v1.81.1 vendors.
triton_server_conan_graph_require(REF ${TRITON_GRPC_REF} OVERRIDE)
triton_server_conan_graph_require(REF abseil/${TRITON_ABSEIL_VERSION} OVERRIDE)
triton_server_conan_graph_require(REF protobuf/${TRITON_PROTOBUF_VERSION} OVERRIDE)
triton_server_conan_graph_require(REF re2/${TRITON_RE2_VERSION})
triton_server_conan_graph_require(REF c-ares/${TRITON_CARES_VERSION})
triton_server_conan_graph_require(REF rapidjson/${TRITON_RAPIDJSON_VERSION})
triton_server_conan_graph_require(REF boost/${TRITON_BOOST_VERSION}
                                  OPTIONS "boost/*:header_only=True")
triton_server_conan_graph_require(REF gtest/${TRITON_GTEST_VERSION})

if(TRITON_ENABLE_GPU)
  triton_server_conan_graph_require(REF cnmem/${TRITON_CNMEM_VERSION})
endif()

if(TRITON_ENABLE_METRICS)
  triton_server_conan_graph_require(REF prometheus-cpp/${TRITON_PROMETHEUS_VERSION})
endif()

if(TRITON_ENABLE_HTTP OR TRITON_ENABLE_METRICS
    OR TRITON_ENABLE_SAGEMAKER OR TRITON_ENABLE_VERTEX_AI)
  triton_server_conan_graph_require(REF libevent/${TRITON_LIBEVENT_VERSION})
  triton_server_conan_graph_require(REF libevhtp/${TRITON_LIBEVHTP_VERSION})
endif()

if(TRITON_ENABLE_TRACING)
  triton_server_conan_graph_require(REF libcurl/${TRITON_CURL_VERSION})
  triton_server_conan_graph_require(REF nlohmann_json/${TRITON_NLOHMANN_JSON_VERSION})
  # Triton uses the OTLP *HTTP* exporter only (otlp_http_exporter_factory), so
  # with_otlp_grpc stays off -- enabling it would pull a second gRPC and, with
  # it, a second abseil, colliding with the bundle.
  triton_server_conan_graph_require(REF opentelemetry-cpp/${TRITON_OPENTELEMETRY_VERSION}
                                    OPTIONS "opentelemetry-cpp/*:with_otlp_grpc=False"
                                            "opentelemetry-cpp/*:with_otlp_http=True")
endif()

if(TRITON_ENABLE_GCS)
  triton_server_conan_graph_require(REF google-cloud-cpp/${TRITON_GOOGLE_CLOUD_VERSION})
  triton_server_conan_graph_require(REF crc32c/${TRITON_CRC32C_VERSION})
endif()

if(TRITON_ENABLE_S3)
  # Mirrors third_party/CMakeLists.txt:529 (-DBUILD_ONLY=s3 -DMINIMIZE_SIZE=ON).
  # Leaving text-to-speech on pulls audio playback and, through it, X11:
  # aws-sdk-cpp -> pulseaudio -> libalsa -> xorg/system -> 43 X11 -dev packages.
  triton_server_conan_graph_require(REF aws-sdk-cpp/${TRITON_AWS_SDK_VERSION}
                                    OPTIONS "aws-sdk-cpp/*:s3=True"
                                            "aws-sdk-cpp/*:text-to-speech=False"
                                            "aws-sdk-cpp/*:min_size=True")
endif()

if(TRITON_ENABLE_AZURE_STORAGE)
  triton_server_conan_graph_require(REF azure-sdk-for-cpp/${TRITON_AZURE_SDK_VERSION})
endif()

triton_server_conan_graph_resolve(TRITON_CONAN_DIR)

#
# Now bind the packages, in dependency order.
#
# Threads comes from the toolchain, not Conan; common/CMakeLists.txt:53 needs it.
find_package(Threads REQUIRED)

if(TRITON_ENABLE_GPU)
  # Tied to the host driver; no Conan reference exists.
  triton_server_dependency_provider(NAME CUDAToolkit ALLOW_SYSTEM)
endif()

if(TRITON_ENABLE_METRICS_GPU)
  # Ships with the datacenter GPU manager and is tied to the host driver, so
  # whatever is installed is the right version. Note the capitalisation: its
  # config is DCGMConfig.cmake and find_package() is case-sensitive. REQUIRED
  # because no third_party fallback could build it.
  triton_server_dependency_provider(NAME DCGM CONFIG ALLOW_SYSTEM REQUIRED)
endif()

find_package(ZLIB CONFIG REQUIRED)
find_package(OpenSSL CONFIG REQUIRED)

# Plain find_package now that nothing is vendored. The previous
# triton_server_grpc_provide_bundle() had to locate gRPC's package tree and
# adopt absl/utf8_range/Protobuf/re2/c-ares out of it, in a fixed order because
# gRPC's config referenced protobuf targets that had to already exist.
find_package(absl CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(re2 CONFIG REQUIRED)

if(TRITON_ENABLE_GRPC)
  find_package(c-ares CONFIG REQUIRED)
  # The recipe keeps cmake_find_mode=none so gRPC's own gRPCConfig.cmake is
  # used; that lives in the package tree, which has to be on CMAKE_PREFIX_PATH.
  _triton_server_conan_package_folder("${TRITON_GRPC_REF}" _triton_grpc_root)
  if(_triton_grpc_root)
    list(PREPEND CMAKE_PREFIX_PATH "${_triton_grpc_root}")
  endif()
  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Using gRPC ${gRPC_VERSION} with abseil ${absl_VERSION}, protobuf ${Protobuf_VERSION}")
endif()

find_package(RapidJSON CONFIG REQUIRED)
find_package(Boost REQUIRED)
find_package(GTest CONFIG REQUIRED)

if(TRITON_ENABLE_GPU)
  find_package(cnmem CONFIG REQUIRED)
  # core locates cnmem the old third_party way:
  #   find_library(CNMEM_LIBRARY NAMES cnmem PATHS ${CNMEM_PATH}/lib)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _triton_cfg)
  if(cnmem_PACKAGE_FOLDER_${_triton_cfg})
    set(CNMEM_PATH "${cnmem_PACKAGE_FOLDER_${_triton_cfg}}" CACHE PATH "cnmem root" FORCE)
  endif()
endif()

if(TRITON_ENABLE_METRICS)
  find_package(prometheus-cpp CONFIG REQUIRED)
endif()

if(TRITON_ENABLE_HTTP OR TRITON_ENABLE_METRICS
    OR TRITON_ENABLE_SAGEMAKER OR TRITON_ENABLE_VERTEX_AI)
  find_package(Libevent CONFIG REQUIRED)
  find_package(libevhtp CONFIG REQUIRED)

  # src/ and src/test/ consume libevent through the module-style
  # LIBEVENT_INCLUDE_DIRS / LIBEVENT_LIBRARIES in four places; Conan exports the
  # libevent::libevent target instead, and the empty variables surface only as
  # "event2/buffer.h: No such file or directory" at compile time.
  if(TARGET libevent::libevent AND NOT LIBEVENT_LIBRARIES)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _triton_cfg)
    if(libevent_INCLUDE_DIRS_${_triton_cfg})
      set(LIBEVENT_INCLUDE_DIRS "${libevent_INCLUDE_DIRS_${_triton_cfg}}")
    elseif(libevent_PACKAGE_FOLDER_${_triton_cfg})
      set(LIBEVENT_INCLUDE_DIRS "${libevent_PACKAGE_FOLDER_${_triton_cfg}}/include")
    endif()
    set(LIBEVENT_LIBRARIES libevent::libevent)
  endif()
endif()

if(TRITON_ENABLE_TRACING)
  find_package(CURL CONFIG REQUIRED)
  find_package(nlohmann_json CONFIG REQUIRED)
  find_package(opentelemetry-cpp CONFIG REQUIRED)
endif()

if(TRITON_ENABLE_GCS)
  # Triton's recipe preserves upstream's per-component layout, so this is the
  # storage client's own config rather than a consolidated one.
  find_package(google_cloud_cpp_storage REQUIRED)
  find_package(Crc32c CONFIG REQUIRED)
endif()

if(TRITON_ENABLE_S3)
  find_package(AWSSDK REQUIRED COMPONENTS s3)
endif()

if(TRITON_ENABLE_AZURE_STORAGE)
  # One package covers blobs and identity: it exports AzureSDKConfig.cmake, not
  # the per-repo azure-storage-blobs-cpp / azure-identity-cpp configs.
  find_package(AzureSDK CONFIG REQUIRED)
endif()

# src/test asks for GTest in module mode, which looks for GTEST_LIBRARY /
# GTEST_INCLUDE_DIR rather than the config; the old root file did the same
# (CMakeLists.txt.bkp:207).
if(GTest_DIR)
  set(GTEST_ROOT "${GTest_DIR}" CACHE PATH "GoogleTest root for FindGTest" FORCE)
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

# core defaults TRITON_CORE_HEADERS_ONLY to ON, which builds only the stub
# (core/CMakeLists.txt:34). The old root file set this explicitly
# (CMakeLists.txt.bkp:135); without it libtritonserver.so is a stub and core/src
# -- with its own Boost/Protobuf/AWS/Azure dependencies -- is never configured.
set(TRITON_CORE_HEADERS_ONLY OFF)

triton_server_fetchcontent_declare_repos()

add_subdirectory(src)
