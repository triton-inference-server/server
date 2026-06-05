# Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(tritonserverexe LANGUAGES C CXX)

include(GNUInstallDirs)

#
# Dependencies
#
# We must include the transitive closure of all repos so that we can
# override the tag. The backend repo is needed for the tests.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
)

if(TRITON_ENABLE_GRPC)
  set(TRITON_COMMON_ENABLE_PROTOBUF ON)
  set(TRITON_COMMON_ENABLE_GRPC ON)
endif() # TRITON_ENABLE_GRPC

FetchContent_MakeAvailable(repo-common repo-core repo-backend)

# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Using CUDA ${CUDA_VERSION}")
endif() # TRITON_ENABLE_GPU

# libevent
#
if(${TRITON_ENABLE_HTTP} OR ${TRITON_ENABLE_METRICS} OR
    ${TRITON_ENABLE_SAGEMAKER} OR ${TRITON_ENABLE_VERTEX_AI})
  find_package(Libevent CONFIG REQUIRED)
  message(STATUS "Using libevent ${Libevent_VERSION}")
endif()

# OpenTelemetry
#
if (NOT WIN32 AND ${TRITON_ENABLE_TRACING})
    find_package(absl CONFIG REQUIRED)
    find_package(CURL CONFIG REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)
    find_package(opentelemetry-cpp CONFIG REQUIRED)
    message(STATUS "Using opentelemetry-cpp ${opentelemetry-cpp_VERSION}")
endif()

# re2
#
find_package(re2 REQUIRED)

#
# tritonserver executable
#
add_executable(
  main
  classification.cc
  command_line_parser.cc
  common.cc
  main.cc
  shared_memory_manager.cc
  triton_signal.cc
  classification.h
  common.h
  shared_memory_manager.h
  triton_signal.h
)

# On windows a *.lib file can be generated for a exe. When creating
# tritonserver.exe if we try to create tritonserver.lib it will fail
# because there is already a trtionserver.lib for tritonserver.dll,
# this causes the build to fail. To avoid we keep the build name as
# main.exe and then for windows after installing we rename it to
# tritonserver.exe (below in the install steps).
if (NOT WIN32)
  set_property(TARGET main PROPERTY OUTPUT_NAME tritonserver)
endif()

target_compile_features(main PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
if(WIN32)
  message("Using MSVC as compiler, default target on Windows 10. "
    "If the target system is not Windows 10, please update _WIN32_WINNT "
    "to corresponding value.")
  target_compile_options(
    main
    PRIVATE
      /W1 /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor
  )
  target_compile_definitions(main
    PRIVATE
      NOMINMAX)

  # Dependency from common.h
  find_library(B64_LIBRARY NAMES b64)
  target_link_libraries(
    main
    PRIVATE
      ${B64_LIBRARY}
  )

else()

  target_compile_options(
    main
    PRIVATE
      -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations -Werror
  )

  # Dependency from common.h
  target_link_libraries(
    main
    PRIVATE
      b64
  )
  endif()

set(LIB_DIR "lib")
if(LINUX)
  file(STRINGS "/etc/os-release" DISTRO_ID_LIKE REGEX "ID_LIKE")
  if(${DISTRO_ID_LIKE} MATCHES "rhel|centos")
    set (LIB_DIR "lib64")
  endif(${DISTRO_ID_LIKE} MATCHES "rhel|centos")
endif(LINUX)
set(TRITON_CORE_HEADERS_ONLY OFF)

set_target_properties(
  main
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}/../${LIB_DIR}"
)

target_link_libraries(
  main
  PRIVATE
    triton-common-async-work-queue  # from repo-common
    triton-common-error             # from repo-common
    triton-common-logging           # from repo-common
    triton-core-serverapi           # from repo-core
    triton-core-serverstub          # from repo-core
)

if(${TRITON_ENABLE_ASAN})
  set(CMAKE_BUILD_TYPE Debug)
  target_compile_definitions(
    main
    PRIVATE TRITON_ENABLE_ASAN=1
  )
  set(_ASAN_FLAGS "-static-libstdc++ -static-libasan -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${_ASAN_FLAGS}")
  set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} ${_ASAN_FLAGS}")
endif() # TRITON_ENABLE_ASAN

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    main
    PRIVATE TRITON_ENABLE_GPU=1
    PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
  )

  target_link_libraries(
    main
    PRIVATE
      CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_HTTP} OR ${TRITON_ENABLE_METRICS} OR
    ${TRITON_ENABLE_SAGEMAKER} OR ${TRITON_ENABLE_VERTEX_AI})
  target_include_directories(
    main
    PRIVATE
      ${LIBEVENT_INCLUDE_DIRS}
  )
endif()


if(${TRITON_ENABLE_HTTP})
  target_compile_definitions(
    main
    PRIVATE TRITON_ENABLE_HTTP=1
  )
endif() # TRITON_ENABLE_HTTP

if(${TRITON_ENABLE_SAGEMAKER})
  target_compile_definitions(
    main
    PRIVATE TRITON_ENABLE_SAGEMAKER=1
  )
endif() # TRITON_ENABLE_SAGEMAKER

if(${TRITON_ENABLE_VERTEX_AI})
  target_compile_definitions(
    main
    PRIVATE TRITON_ENABLE_VERTEX_AI=1
  )
endif() # TRITON_ENABLE_VERTEX_AI

if(${TRITON_ENABLE_LOGGING})
  target_compile_definitions(
    main
    PRIVATE TRITON_ENABLE_LOGGING=1
  )
endif() # TRITON_ENABLE_LOGGING

if(${TRITON_ENABLE_METRICS})
  target_compile_definitions(
    main
    PRIVATE TRITON_ENABLE_METRICS=1
  )
endif() # TRITON_ENABLE_METRICS

if(${TRITON_ENABLE_STATS})
  target_compile_definitions(
    main
    PRIVATE TRITON_ENABLE_STATS=1
  )
endif() # TRITON_ENABLE_STATS

if(${TRITON_ENABLE_TRACING})
  target_compile_definitions(
    main
    PRIVATE TRITON_ENABLE_TRACING=1
  )
endif() # TRITON_ENABLE_TRACING

if(${TRITON_ENABLE_NVTX})
  target_compile_definitions(
    main
    PRIVATE TRITON_ENABLE_NVTX=1
  )
endif() # TRITON_ENABLE_NVTX

if (NOT WIN32)
  target_link_libraries(
    main
    PRIVATE
      rt
      dl
  )
endif() # NOT WIN32

if (NOT WIN32)
  install(
    TARGETS main
    RUNTIME DESTINATION bin
  )
else()
  # See explanation above as to why we need to rename main.exe to
  # tritonserver.exe as part of the install process on windows.
  install(
    PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/main.exe
    DESTINATION bin
    RENAME tritonserver.exe
  )
endif()

if(${TRITON_ENABLE_GRPC})
  #
  # GRPC
  #
  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Using gRPC ${gRPC_VERSION}")

  add_subdirectory(grpc)
  target_link_libraries(
      main
      PRIVATE
        grpc-endpoint-library
  )

  target_include_directories(
    main
    PRIVATE
      $<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>
  )

  target_compile_definitions(
    main
    PRIVATE TRITON_ENABLE_GRPC=1
  )
endif()

# http endpoint
#
if(${TRITON_ENABLE_HTTP}
      OR ${TRITON_ENABLE_METRICS}
      OR ${TRITON_ENABLE_SAGEMAKER}
      OR ${TRITON_ENABLE_VERTEX_AI})
  find_package(libevhtp CONFIG REQUIRED)
  message(STATUS "Using libevhtp ${libevhtp_VERSION}")

  list(APPEND
    HTTP_ENDPOINT_SRCS
    http_server.cc
    orca_http.cc
  )
  list(APPEND
    HTTP_ENDPOINT_HDRS
    http_server.h
    orca_http.h
  )

  # Add header / src files based on HTTP related endpoint requested
  if(${TRITON_ENABLE_SAGEMAKER})
    list(APPEND
      HTTP_ENDPOINT_SRCS
      sagemaker_server.cc
    )
    list(APPEND
      HTTP_ENDPOINT_HDRS
      sagemaker_server.h
    )
  endif() # TRITON_ENABLE_SAGEMAKER

  if(${TRITON_ENABLE_VERTEX_AI})
    list(APPEND
      HTTP_ENDPOINT_SRCS
      vertex_ai_server.cc
    )
    list(APPEND
      HTTP_ENDPOINT_HDRS
      vertex_ai_server.h
    )
  endif() # TRITON_ENABLE_VERTEX_AI

  add_library(
    http-endpoint-library EXCLUDE_FROM_ALL
    ${HTTP_ENDPOINT_SRCS} ${HTTP_ENDPOINT_HDRS}
  )

  target_compile_features(http-endpoint-library PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(
      http-endpoint-library
      PRIVATE
        /W1 /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor
    )
  else()
    target_compile_options(
      http-endpoint-library
      PRIVATE
        -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations -Wno-error=maybe-uninitialized -Werror
    )
  endif()

  set_target_properties(
    http-endpoint-library
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
  )

  target_link_libraries(
    http-endpoint-library
    PUBLIC
      triton-common-json      # from repo-common
      triton-common-logging   # from repo-common
      triton-core-serverapi   # from repo-core
      triton-core-serverstub  # from repo-core
      ${LIBEVENT_LIBRARIES}
      libevhtp::evhtp
      re2::re2
  )

  target_include_directories(
    http-endpoint-library
    PRIVATE $<TARGET_PROPERTY:libevhtp::evhtp,INTERFACE_INCLUDE_DIRECTORIES>
  )

  # FIXME when Triton support of Opentelemetry is available on Windows
  # add ${OPENTELEMETRY_CPP_INCLUDE_DIRS} to above target_include_directories
  # JIRA DLIS-4786
  if (NOT WIN32 AND ${TRITON_ENABLE_TRACING})
    target_link_libraries(
      http-endpoint-library
      PRIVATE tracing-library
    )
  endif()

  if(${TRITON_ENABLE_GPU})
    target_compile_definitions(
      http-endpoint-library
      PRIVATE TRITON_ENABLE_GPU=1
      PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
    )

    target_link_libraries(
      http-endpoint-library
      PUBLIC
        CUDA::cudart
    )
  endif() # TRITON_ENABLE_GPU

  if(${TRITON_ENABLE_HTTP})
    target_compile_definitions(
      http-endpoint-library
      PRIVATE TRITON_ENABLE_HTTP=1
    )
  endif() # TRITON_ENABLE_HTTP

  if(${TRITON_ENABLE_SAGEMAKER})
    target_compile_definitions(
      http-endpoint-library
      PRIVATE TRITON_ENABLE_SAGEMAKER=1
    )
  endif() # TRITON_ENABLE_SAGEMAKER

  if(${TRITON_ENABLE_VERTEX_AI})
    target_compile_definitions(
      http-endpoint-library
      PRIVATE TRITON_ENABLE_VERTEX_AI=1
    )
  endif() # TRITON_ENABLE_VERTEX_AI

  if(${TRITON_ENABLE_METRICS})
    target_compile_definitions(
      http-endpoint-library
      PRIVATE TRITON_ENABLE_METRICS=1
    )
  endif() # TRITON_ENABLE_METRICS

  if(${TRITON_ENABLE_LOGGING})
    target_compile_definitions(
      http-endpoint-library
      PRIVATE TRITON_ENABLE_LOGGING=1
    )
  endif() # TRITON_ENABLE_LOGGING

  if(${TRITON_ENABLE_STATS})
    target_compile_definitions(
      http-endpoint-library
      PRIVATE TRITON_ENABLE_STATS=1
    )
  endif() # TRITON_ENABLE_STATS

  if(${TRITON_ENABLE_TRACING})
    target_compile_definitions(
      http-endpoint-library
      PRIVATE TRITON_ENABLE_TRACING=1
    )
  endif() # TRITON_ENABLE_TRACING

  if(${TRITON_ENABLE_NVTX})
    target_compile_definitions(
      http-endpoint-library
      PRIVATE TRITON_ENABLE_NVTX=1
    )
  endif() # TRITON_ENABLE_NVTX

  if (WIN32)
    find_library(B64_LIBRARY NAMES b64)
    find_library(ZLIB_LIBRARY NAMES zlib)
    target_link_libraries(
      http-endpoint-library
      PUBLIC
        ${B64_LIBRARY}
        ${ZLIB_LIBRARY}
    )
  else()
    target_link_libraries(
      http-endpoint-library
      PUBLIC
        b64
        z
    )
  endif()

  target_link_libraries(
    main
    PRIVATE
      http-endpoint-library
  )
endif() # TRITON_ENABLE_HTTP || TRITON_ENABLE_METRICS ||
        # TRITON_ENABLE_SAGEMAKER || TRITON_ENABLE_VERTEX_AI

# tracing
#
if(${TRITON_ENABLE_TRACING})
  message(STATUS "Using tracing ${TRITON_TRACE_INSTALL_PATH}")

  add_library(
    tracing-library EXCLUDE_FROM_ALL
    tracer.cc tracer.h
  )

  target_compile_features(tracing-library PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
  # FIXME: remove, when Windows support is added for Opentelemetry
  if (NOT WIN32)
    target_include_directories(
      tracing-library
      PUBLIC ${OPENTELEMETRY_CPP_INCLUDE_DIRS}
    )

    target_link_libraries(
      tracing-library
      PUBLIC
      ${OPENTELEMETRY_CPP_LIBRARIES})
  endif()

  set_target_properties(
    tracing-library
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    )

  target_link_libraries(
    tracing-library
    PUBLIC
      triton-common-logging    # from repo-common
      triton-common-json      # from repo-common
      triton-core-serverapi    # from repo-core
      triton-core-serverstub   # from repo-core
  )

  target_compile_definitions(
    tracing-library
    PRIVATE TRITON_ENABLE_TRACING=1
  )

  if(${TRITON_ENABLE_GPU})
    target_compile_definitions(
      tracing-library
      PRIVATE TRITON_ENABLE_GPU=1
      PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
    )

    target_link_libraries(
      tracing-library
      PUBLIC
        CUDA::cudart
    )
  endif() # TRITON_ENABLE_GPU

  if(${TRITON_ENABLE_METRICS})
    target_compile_definitions(
      tracing-library
      PRIVATE TRITON_ENABLE_METRICS=1
    )
  endif() # TRITON_ENABLE_METRICS

  if(${TRITON_ENABLE_LOGGING})
    target_compile_definitions(
      tracing-library
      PRIVATE TRITON_ENABLE_LOGGING=1
    )
  endif() # TRITON_ENABLE_LOGGING

  if(${TRITON_ENABLE_STATS})
    target_compile_definitions(
      tracing-library
      PRIVATE TRITON_ENABLE_STATS=1
    )
  endif() # TRITON_ENABLE_STATS

  if(${TRITON_ENABLE_NVTX})
    target_compile_definitions(
      tracing-library
      PRIVATE TRITON_ENABLE_NVTX=1
    )
  endif() # TRITON_ENABLE_NVTX

  target_link_libraries(
    main
    PRIVATE
      tracing-library
  )
endif() # TRITON_ENABLE_TRACING

if (NOT WIN32)
  #
  # simple
  #
  add_executable(
    simple
    simple.cc
  )

  target_compile_features(simple PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message("Using MSVC as compiler, default target on Windows 10. "
            "If the target system is not Windows 10, please update _WIN32_WINNT "
            "to corresponding value.")
    target_compile_options(
      simple
      PRIVATE
        /W1 /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor
    )
  else()
    target_compile_options(
      simple
      PRIVATE
        -Wall -Wextra -Wno-type-limits -Wno-unused-parameter -Wno-deprecated-declarations -Werror
    )
  endif()

  set_target_properties(
    simple
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH ""
  )

  target_link_libraries(
    simple
    PRIVATE
      triton-common-async-work-queue  # from repo-common
      triton-common-error             # from repo-common
      triton-core-serverapi           # from repo-core
      triton-core-serverstub          # from repo-core
    )

  if(${TRITON_ENABLE_GPU})
    target_compile_definitions(
      simple
      PRIVATE TRITON_ENABLE_GPU=1
      PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
    )

    target_link_libraries(
      simple
      PRIVATE
        CUDA::cudart
    )
  endif() # TRITON_ENABLE_GPU

  install(
    TARGETS simple
    RUNTIME DESTINATION bin
  )

  #
  # multi_server example
  #
  add_executable(
    multi_server
    multi_server.cc
  )

  target_compile_features(multi_server PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message("Using MSVC as compiler, default target on Windows 10. "
            "If the target system is not Windows 10, please update _WIN32_WINNT "
            "to corresponding value.")
    target_compile_options(
      multi_server
      PRIVATE
        /W1 /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor
    )
  else()
    target_compile_options(
      multi_server
      PRIVATE
        -Wall -Wextra -Wno-type-limits -Wno-unused-parameter -Wno-deprecated-declarations -Werror
    )
  endif()

  set_target_properties(
    multi_server
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH ""
  )

  target_link_libraries(
    multi_server
    PRIVATE
      triton-common-async-work-queue  # from repo-common
      triton-common-error             # from repo-common
      triton-core-serverapi           # from repo-core
      triton-core-serverstub          # from repo-core
    )

  if(${TRITON_ENABLE_GPU})
    target_compile_definitions(
      multi_server
      PRIVATE TRITON_ENABLE_GPU=1
      PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
    )

    target_link_libraries(
      multi_server
      PRIVATE
        CUDA::cudart
    )
  endif() # TRITON_ENABLE_GPU

  install(
    TARGETS multi_server
    RUNTIME DESTINATION bin
  )

  if(${TRITON_ENABLE_GPU})
    #
    # memory_alloc example
    #
    add_executable(
      memory_alloc
      memory_alloc.cc
    )

    target_compile_features(memory_alloc PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      message("Using MSVC as compiler, default target on Windows 10. "
              "If the target system is not Windows 10, please update _WIN32_WINNT "
              "to corresponding value.")
      target_compile_options(
        memory_alloc
        PRIVATE
          /W1 /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor
      )
    else()
      target_compile_options(
        memory_alloc
        PRIVATE
          -Wall -Wextra -Wno-type-limits -Wno-unused-parameter -Wno-deprecated-declarations -Werror
      )
    endif()

    set_target_properties(
      memory_alloc
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        SKIP_BUILD_RPATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH FALSE
        INSTALL_RPATH ""
    )

    target_compile_definitions(
      memory_alloc
      PRIVATE TRITON_ENABLE_GPU=1
      PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
    )

    target_link_libraries(
      memory_alloc
      PRIVATE
        triton-common-async-work-queue  # from repo-common
        triton-common-error             # from repo-common
        triton-core-serverapi           # from repo-core
        triton-core-serverstub          # from repo-core
        CUDA::cudart
      )

    install(
      TARGETS memory_alloc
      RUNTIME DESTINATION bin
    )
  endif() # TRITON_ENABLE_GPU
endif() # NOT WIN32

# DLIS-7292: Extend tritonfrontend to build for Windows
if (NOT WIN32)
  # tritonfrontend python package
  add_subdirectory(python)
endif (NOT WIN32)

# Currently unit tests do not build for windows...
if ( NOT WIN32)
  add_subdirectory(test test)
endif() # NOT WIN32

