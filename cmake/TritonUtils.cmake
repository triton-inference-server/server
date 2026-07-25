# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

include_guard(GLOBAL)

# Function naming convention: triton_<repository>_<verb>_<object>
#
# This is the `server` repository, hence the triton_server_ prefix. The verb is
# always imperative -- read, require, forbid -- so the names read as commands and
# stay consistent as more are added. Prefer `require` over `requires`.
#
# Every function announces entry and each of its steps. The log prefix comes from
# CMAKE_CURRENT_FUNCTION (CMake 3.17+) so renaming a function cannot leave a stale
# name behind in its output.

# triton_server_read_version(<path> <full_var> <numeric_var>)
#
# Read the Triton version from <path> (normally server/TRITON_VERSION) and set:
#   <full_var>    the verbatim contents, e.g. "2.72.0dev"
#   <numeric_var> the leading major.minor.patch, e.g. "2.72.0"
#
# project(VERSION ...) rejects non-numeric components, so the dev suffix must be
# stripped there while the full string stays available for anything reporting the
# real Triton version. Configuration fails loudly if the file is malformed --
# without that check a bad file yields an empty version and still configures.
function(triton_server_read_version path full_var numeric_var)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered: path='${path}'")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 1/4: verifying file exists")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Triton version file not found: ${path}")
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/4: reading first line")
  file(STRINGS "${path}" _version LIMIT_COUNT 1)

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 3/4: validating '${_version}'")
  if(NOT _version MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+)")
    message(FATAL_ERROR "Malformed version in ${path}: '${_version}'")
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 4/4: exporting '${full_var}' and '${numeric_var}'")
  set(${full_var} "${_version}" PARENT_SCOPE)
  set(${numeric_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: full='${_version}' numeric='${CMAKE_MATCH_1}'")
endfunction()

# triton_server_require_option(<option> <dependency>)
#
# Fail configuration when <option> is enabled without <dependency>.
function(triton_server_require_option option dependency)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered: ${option} requires ${dependency}")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 1/2: evaluating ${option}='${${option}}' ${dependency}='${${dependency}}'")
  if(${option} AND NOT ${dependency})
    message(FATAL_ERROR "${option}=ON requires ${dependency}=ON")
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/2: constraint satisfied")
endfunction()

# triton_server_forbid_option(<option> <other>)
#
# Fail configuration when two mutually exclusive options are both enabled.
function(triton_server_forbid_option option other)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered: ${option} forbids ${other}")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 1/2: evaluating ${option}='${${option}}' ${other}='${${other}}'")
  if(${option} AND ${other})
    message(FATAL_ERROR "${option}=ON requires ${other}=OFF")
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/2: constraint satisfied")
endfunction()
