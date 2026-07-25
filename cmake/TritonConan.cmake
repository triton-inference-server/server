# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

include_guard(GLOBAL)

# Conan integration. See TritonUtils.cmake for the naming and logging conventions.
#
# Environment variables consulted:
#   CONAN_REMOTE_URL    remote to register before resolving (optional)
#   CONAN_REMOTE_NAME   name for that remote (optional, default "triton")
#   ARTIFACTORY_USER    username for the remote (optional)
#   ARTIFACTORY_TOKEN   token/password for the remote (optional)
#
# Credentials are passed to `conan remote login` rather than embedded in the
# remote URL: Conan persists remotes.json in plaintext, so a URL of the form
# https://user:token@host would leave the token on disk.

set(TRITON_SERVER_CONAN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/conan" CACHE PATH
    "Directory where Conan writes generated CMake package files.")

set(TRITON_SERVER_CONAN_PROFILE "" CACHE STRING
    "Conan profile to resolve with. Empty uses Conan's default profile.")

# triton_server_conan_add_remote()
#
# Register and authenticate the Conan remote described by the environment. Does
# nothing when CONAN_REMOTE_URL is unset, so a developer with a working default
# Conan setup needs no extra configuration.
function(triton_server_conan_add_remote)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 1/4: reading environment")
  set(_url "$ENV{CONAN_REMOTE_URL}")
  set(_name "$ENV{CONAN_REMOTE_NAME}")
  if(NOT _name)
    set(_name "triton")
  endif()

  if(NOT _url)
    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/4: CONAN_REMOTE_URL unset, using existing remotes")
    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: no remote registered")
    return()
  endif()
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/4: remote '${_name}' requested")

  find_program(TRITON_SERVER_CONAN_EXECUTABLE NAMES conan REQUIRED)

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 3/4: registering remote '${_name}'")
  execute_process(
    COMMAND "${TRITON_SERVER_CONAN_EXECUTABLE}" remote add "${_name}" "${_url}" --force
    RESULT_VARIABLE _result
    OUTPUT_QUIET
    ERROR_VARIABLE _stderr)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "conan remote add '${_name}' failed (exit ${_result}): ${_stderr}")
  endif()

  # Credentials are optional: the remote may allow anonymous reads.
  set(_user "$ENV{ARTIFACTORY_USER}")
  set(_token "$ENV{ARTIFACTORY_TOKEN}")
  if(_user AND _token)
    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 4/4: authenticating as '${_user}'")
    execute_process(
      COMMAND "${TRITON_SERVER_CONAN_EXECUTABLE}" remote login "${_name}" "${_user}" -p "${_token}"
      RESULT_VARIABLE _result
      OUTPUT_QUIET
      ERROR_QUIET)
    if(NOT _result EQUAL 0)
      # Deliberately does not echo the command: it carries the token.
      message(FATAL_ERROR "conan remote login to '${_name}' failed (exit ${_result})")
    endif()
  else()
    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 4/4: no credentials in environment, staying anonymous")
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: remote '${_name}' ready")
endfunction()

# triton_server_conan_install_package(<ref> [<ref>...])
#
# Resolve one or more Conan references (e.g. re2/20230301) and make them findable
# by a subsequent find_package(). Runs `conan install` with the CMakeDeps
# generator and prepends the output directory to CMAKE_PREFIX_PATH in the
# caller's scope.
#
# Pass every reference in a single call. Separate calls resolve separate
# dependency graphs, so a conflict in a shared transitive dependency (abseil,
# typically) would go unnoticed until link time instead of failing here.
function(triton_server_conan_install_package)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered: refs='${ARGN}'")

  if(NOT ARGN)
    message(FATAL_ERROR "triton_server_conan_install_package requires at least one reference")
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 1/4: locating conan executable")
  find_program(TRITON_SERVER_CONAN_EXECUTABLE NAMES conan)
  if(NOT TRITON_SERVER_CONAN_EXECUTABLE)
    message(FATAL_ERROR "conan not found in PATH; install Conan 2 or set TRITON_SERVER_CONAN_EXECUTABLE")
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/4: assembling arguments")
  set(_args "")
  foreach(_ref IN LISTS ARGN)
    list(APPEND _args "--requires=${_ref}")
  endforeach()

  # Only forward a build type when CMake has one; otherwise defer to the profile.
  if(CMAKE_BUILD_TYPE)
    list(APPEND _args "-s" "build_type=${CMAKE_BUILD_TYPE}")
  endif()

  if(TRITON_SERVER_CONAN_PROFILE)
    list(APPEND _args "-pr:h" "${TRITON_SERVER_CONAN_PROFILE}"
                      "-pr:b" "${TRITON_SERVER_CONAN_PROFILE}")
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 3/4: running conan install into '${TRITON_SERVER_CONAN_OUTPUT_DIR}'")
  execute_process(
    COMMAND "${TRITON_SERVER_CONAN_EXECUTABLE}" install
            ${_args}
            --generator=CMakeDeps
            --output-folder=${TRITON_SERVER_CONAN_OUTPUT_DIR}
            --build=missing
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)

  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "conan install failed (exit ${_result}) for: ${ARGN}\n"
      "--- stdout ---\n${_stdout}\n"
      "--- stderr ---\n${_stderr}")
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 4/4: extending CMAKE_PREFIX_PATH")
  list(PREPEND CMAKE_PREFIX_PATH "${TRITON_SERVER_CONAN_OUTPUT_DIR}")
  list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: ${ARGN} available to find_package()")
endfunction()
