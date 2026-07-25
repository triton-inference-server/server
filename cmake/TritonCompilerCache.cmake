# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

include_guard(GLOBAL)

# Compiler cache integration.
#
# The cache backend is configured entirely through environment variables read by
# sccache/ccache at compile time, not by CMake. CMake cannot inject variables into
# the build environment for every generator, so this module validates the
# environment and reports what will actually happen rather than setting it.
#
# sccache remote backends (first match wins):
#   SCCACHE_WEBDAV_ENDPOINT           WebDAV -- what Artifactory generic repos speak
#     + SCCACHE_WEBDAV_TOKEN, or SCCACHE_WEBDAV_USERNAME and SCCACHE_WEBDAV_PASSWORD
#   SCCACHE_BUCKET                    S3   (+ SCCACHE_REGION or SCCACHE_ENDPOINT)
#   SCCACHE_REDIS / SCCACHE_REDIS_ENDPOINT   Redis
#   SCCACHE_GCS_BUCKET                GCS  (+ SCCACHE_GCS_KEY_PATH or _CREDENTIALS_URL)
#   SCCACHE_AZURE_CONNECTION_STRING   Azure (+ SCCACHE_AZURE_BLOB_CONTAINER)
#   SCCACHE_MEMCACHED_ENDPOINT        Memcached
#   SCCACHE_GHA_ENABLED               GitHub Actions cache
#
# ccache remote backend:
#   CCACHE_REMOTE_STORAGE             redis://... or http://...  (ccache >= 4.4)

option(TRITON_ENABLE_COMPILER_CACHE "Speed up rebuilds with sccache or ccache" ON)

option(TRITON_REQUIRE_COMPILER_CACHE_REMOTE
       "Fail configuration when no remote cache backend is configured" OFF)

set(TRITON_COMPILER_CACHE "sccache" CACHE STRING
    "Compiler cache to use: sccache, ccache, or a full path")
set_property(CACHE TRITON_COMPILER_CACHE PROPERTY STRINGS sccache ccache)

# _triton_server_redact(<value> <out>)
#
# Blank out embedded credentials before a URL reaches the log. Endpoints are
# routinely pasted as https://user:token@host.
function(_triton_server_redact value out)
  string(REGEX REPLACE "//[^/@]+@" "//<redacted>@" _v "${value}")
  set(${out} "${_v}" PARENT_SCOPE)
endfunction()

# triton_server_validate_compiler_cache_remote()
#
# Identify which remote backend the environment selects, verify its companion
# variables are present, and report the result. Sets
# TRITON_COMPILER_CACHE_REMOTE_BACKEND in the caller's scope ("none" when the
# cache is local-disk only).
function(triton_server_validate_compiler_cache_remote)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered")

  set(_backend "none")
  set(_detail "")
  set(_incomplete "")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 1/3: detecting backend from environment")

  if(TRITON_COMPILER_CACHE_KIND STREQUAL "ccache")
    if(DEFINED ENV{CCACHE_REMOTE_STORAGE})
      set(_backend "ccache-remote")
      _triton_server_redact("$ENV{CCACHE_REMOTE_STORAGE}" _detail)
    endif()
  else()
    # sccache: probe in the documented precedence order.
    if(DEFINED ENV{SCCACHE_WEBDAV_ENDPOINT})
      set(_backend "webdav")
      _triton_server_redact("$ENV{SCCACHE_WEBDAV_ENDPOINT}" _detail)
      if(NOT DEFINED ENV{SCCACHE_WEBDAV_TOKEN}
         AND NOT (DEFINED ENV{SCCACHE_WEBDAV_USERNAME} AND DEFINED ENV{SCCACHE_WEBDAV_PASSWORD}))
        set(_incomplete "SCCACHE_WEBDAV_TOKEN, or SCCACHE_WEBDAV_USERNAME and SCCACHE_WEBDAV_PASSWORD")
      endif()
    elseif(DEFINED ENV{SCCACHE_BUCKET})
      set(_backend "s3")
      set(_detail "$ENV{SCCACHE_BUCKET}")
      if(NOT DEFINED ENV{SCCACHE_REGION} AND NOT DEFINED ENV{SCCACHE_ENDPOINT})
        set(_incomplete "SCCACHE_REGION or SCCACHE_ENDPOINT")
      endif()
    elseif(DEFINED ENV{SCCACHE_REDIS} OR DEFINED ENV{SCCACHE_REDIS_ENDPOINT})
      set(_backend "redis")
      if(DEFINED ENV{SCCACHE_REDIS})
        _triton_server_redact("$ENV{SCCACHE_REDIS}" _detail)
      else()
        _triton_server_redact("$ENV{SCCACHE_REDIS_ENDPOINT}" _detail)
      endif()
    elseif(DEFINED ENV{SCCACHE_GCS_BUCKET})
      set(_backend "gcs")
      set(_detail "$ENV{SCCACHE_GCS_BUCKET}")
      if(NOT DEFINED ENV{SCCACHE_GCS_KEY_PATH}
         AND NOT DEFINED ENV{SCCACHE_GCS_CREDENTIALS_URL}
         AND NOT DEFINED ENV{SCCACHE_GCS_SERVICE_ACCOUNT})
        set(_incomplete "SCCACHE_GCS_KEY_PATH, SCCACHE_GCS_CREDENTIALS_URL or SCCACHE_GCS_SERVICE_ACCOUNT")
      endif()
    elseif(DEFINED ENV{SCCACHE_AZURE_CONNECTION_STRING})
      set(_backend "azure")
      set(_detail "<connection string set>")
      if(NOT DEFINED ENV{SCCACHE_AZURE_BLOB_CONTAINER})
        set(_incomplete "SCCACHE_AZURE_BLOB_CONTAINER")
      endif()
    elseif(DEFINED ENV{SCCACHE_MEMCACHED_ENDPOINT})
      set(_backend "memcached")
      _triton_server_redact("$ENV{SCCACHE_MEMCACHED_ENDPOINT}" _detail)
    elseif(DEFINED ENV{SCCACHE_GHA_ENABLED})
      set(_backend "gha")
      set(_detail "GitHub Actions cache")
    endif()
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/3: backend='${_backend}' target='${_detail}'")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 3/3: validating configuration")
  if(_incomplete)
    message(WARNING
      "Compiler cache backend '${_backend}' is selected but incomplete: missing ${_incomplete}. "
      "sccache will silently fall back to local disk.")
  endif()

  if(_backend STREQUAL "none")
    if(TRITON_REQUIRE_COMPILER_CACHE_REMOTE)
      message(FATAL_ERROR
        "TRITON_REQUIRE_COMPILER_CACHE_REMOTE=ON but no remote cache is configured. "
        "Set SCCACHE_WEBDAV_ENDPOINT (Artifactory), SCCACHE_BUCKET, SCCACHE_REDIS, "
        "or CCACHE_REMOTE_STORAGE.")
    endif()
    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: local disk cache only")
  else()
    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: remote backend '${_backend}' configured")
  endif()

  set(TRITON_COMPILER_CACHE_REMOTE_BACKEND "${_backend}" PARENT_SCOPE)
endfunction()

# triton_server_enable_compiler_cache()
#
# Route the C and C++ compilers through a compiler cache by setting
# CMAKE_<LANG>_COMPILER_LAUNCHER. Falls back to ccache when the preferred cache is
# absent, and is a no-op when neither is installed -- a missing cache should slow
# the build down, not break it.
function(triton_server_enable_compiler_cache)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered: requested='${TRITON_COMPILER_CACHE}'")

  if(NOT TRITON_ENABLE_COMPILER_CACHE)
    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: disabled by TRITON_ENABLE_COMPILER_CACHE=OFF")
    return()
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 1/4: locating '${TRITON_COMPILER_CACHE}'")
  find_program(TRITON_COMPILER_CACHE_EXECUTABLE NAMES "${TRITON_COMPILER_CACHE}")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/4: falling back to ccache if needed")
  if(NOT TRITON_COMPILER_CACHE_EXECUTABLE AND NOT TRITON_COMPILER_CACHE STREQUAL "ccache")
    find_program(TRITON_COMPILER_CACHE_EXECUTABLE NAMES ccache)
  endif()

  if(NOT TRITON_COMPILER_CACHE_EXECUTABLE)
    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: no compiler cache found, building without one")
    return()
  endif()

  # Which tool actually won, so the remote validation probes the right variables.
  get_filename_component(_exe_name "${TRITON_COMPILER_CACHE_EXECUTABLE}" NAME_WE)
  set(TRITON_COMPILER_CACHE_KIND "${_exe_name}" CACHE INTERNAL "Resolved compiler cache tool")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 3/4: validating remote backend")
  triton_server_validate_compiler_cache_remote()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 4/4: routing C and CXX through '${TRITON_COMPILER_CACHE_EXECUTABLE}'")
  set(CMAKE_C_COMPILER_LAUNCHER "${TRITON_COMPILER_CACHE_EXECUTABLE}" PARENT_SCOPE)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${TRITON_COMPILER_CACHE_EXECUTABLE}" PARENT_SCOPE)

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: ${_exe_name} active, remote='${TRITON_COMPILER_CACHE_REMOTE_BACKEND}'")
endfunction()
