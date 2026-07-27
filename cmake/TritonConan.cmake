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

  # --index 0 is load-bearing. Conan searches remotes in list order and takes the
  # first hit, so with conancenter ahead of this one an empty cache resolves
  # grpc/1.81.1 to ConanCenter's recipe rather than Triton's. ConanCenter's pairs
  # gRPC with re2/[>=20251105] while Triton pins re2/20230301, and the graph dies
  # on "Version conflict ... originates from grpc/1.81.1". A warm cache hides
  # this completely -- the cached recipe wins and no remote is ever consulted --
  # so it only ever reproduces on a clean machine, which is to say on CI.
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 3/4: registering remote '${_name}' ahead of the others")
  execute_process(
    COMMAND "${TRITON_SERVER_CONAN_EXECUTABLE}" remote add "${_name}" "${_url}" --index 0 --force
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


# _triton_server_conan_output_dir(<ref> <out_var>)
#
# Give every reference its own generators folder.
#
# A single shared folder does not work: each triton_server_dependency_provider()
# call runs its own `conan install`, and CMakeDeps rewrites a config for every
# package in that graph -- including transitive ones. Enabling GCS, whose
# google-cloud-cpp depends on re2 and gRPC, regenerated re2-Target-release.cmake
# and gRPC-Target-release.cmake over the versions the gRPC bundle had already
# published, leaving targets pointing at libraries that resolve did not contain:
#
#   The link interface of target "re2::re2" contains: CONAN_LIB::re2_re2_RELEASE
#   but the target was not found.
#
# Per-reference folders keep each resolve self-consistent; CMAKE_PREFIX_PATH is
# extended with each one in turn, and find_package() caches <Pkg>_DIR on first
# success, so the earliest resolve of a shared package wins.
function(_triton_server_conan_output_dir ref out_var)
  string(REGEX REPLACE "[/@]" "_" _slug "${ref}")
  set(${out_var} "${TRITON_SERVER_CONAN_OUTPUT_DIR}/${_slug}" PARENT_SCOPE)
endfunction()

# triton_server_conan_publish_targets([<out_var>])
#
# Publish every package CMakeDeps generated into the CMake cache as <Pkg>_DIR.
#
# A Conan resolve produces config files for the whole transitive closure, not just
# the reference asked for: resolving gRPC also yields protobuf, abseil, re2 and
# c-ares at the versions gRPC was built against. Exporting each as a cache entry
# means every later find_package() -- including the ones inside common/, core/ and
# backend/, which run in their own directory scope -- binds to that same set
# rather than resolving independently and risking a second, mismatched copy.
#
# The cache entries are FORCEd because find_package() writes <Pkg>_DIR-NOTFOUND on
# a failed probe, and a plain cache set will not displace an existing entry.
# Optionally sets <out_var> to the list of published package names.
function(triton_server_conan_publish_targets)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered: dir='${TRITON_SERVER_CONAN_OUTPUT_DIR}'")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 1/3: scanning for generated configs")
  # Reference subfolders as well as the root, since each resolve writes its own.
  file(GLOB_RECURSE _configs
       "${TRITON_SERVER_CONAN_OUTPUT_DIR}/*-config.cmake"
       "${TRITON_SERVER_CONAN_OUTPUT_DIR}/*Config.cmake")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/3: publishing <Pkg>_DIR entries")
  set(_published "")
  foreach(_cfg IN LISTS _configs)
    get_filename_component(_file "${_cfg}" NAME)
    # Conan emits both <pkg>-config.cmake and <Pkg>Config.cmake spellings.
    string(REGEX REPLACE "(-config|Config)\\.cmake$" "" _pkg "${_file}")
    if(_pkg AND NOT _pkg IN_LIST _published)
      get_filename_component(_cfg_dir "${_cfg}" DIRECTORY)
      set(${_pkg}_DIR "${_cfg_dir}" CACHE PATH "Conan-provided ${_pkg}" FORCE)
      list(APPEND _published "${_pkg}")
    endif()
  endforeach()

  list(SORT _published)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 3/3: published ${_published}")

  if(ARGC GREATER 0)
    set(${ARGV0} "${_published}" PARENT_SCOPE)
  endif()
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done")
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
