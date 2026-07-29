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

# Conan resolves against two profiles: the host profile describes what the
# artifacts must run on, the build profile what compiles them. They are the same
# for a native build and differ for a cross build, where protoc and
# grpc_cpp_plugin must run on the builder while the libraries target something
# else. Setting both from one variable -- as this did originally -- silently
# makes cross-compilation impossible.
#
# TRITON_SERVER_CONAN_PROFILE remains the shorthand that sets both.
set(TRITON_SERVER_CONAN_PROFILE "" CACHE STRING
    "Conan profile for both contexts, as a path or a profile name. Empty selects the in-repo profile.")
set(TRITON_SERVER_CONAN_PROFILE_HOST "" CACHE STRING
    "Conan host profile: what the built artifacts run on. Overrides TRITON_SERVER_CONAN_PROFILE.")
set(TRITON_SERVER_CONAN_PROFILE_BUILD "" CACHE STRING
    "Conan build profile: what compiles them. Defaults to the host profile (native build).")

# Default to the in-repo profile rather than Conan's `default`.
#
# `default` exists only where someone has run `conan profile detect`. A fresh
# CONAN_HOME has none, and Conan then aborts with "The default build profile
# ... doesn't exist" before resolving anything -- which is every CI runner and
# every clean checkout. Worse, a detected profile describes whatever the host
# happens to be, while every package published to the triton remote was built
# with the settings below; disagreeing on compiler version alone changes every
# package ID and silently turns a download into a 30-minute source build.
#
# Pass a path to use a different profile, or the name `default` to opt back into
# Conan's own.
set(_triton_conan_repo_profile
    "${CMAKE_CURRENT_SOURCE_DIR}/conan/profiles/linux-gcc13-release")

if(NOT TRITON_SERVER_CONAN_PROFILE_HOST)
  if(TRITON_SERVER_CONAN_PROFILE)
    set(TRITON_SERVER_CONAN_PROFILE_HOST "${TRITON_SERVER_CONAN_PROFILE}")
  elseif(EXISTS "${_triton_conan_repo_profile}")
    set(TRITON_SERVER_CONAN_PROFILE_HOST "${_triton_conan_repo_profile}")
    message(STATUS "[TritonConan] Conan host profile defaulted to ${TRITON_SERVER_CONAN_PROFILE_HOST}")
  endif()
endif()

# Native build unless told otherwise. _triton_server_conan_check_profile() warns
# when this default looks like an unintended cross build.
if(NOT TRITON_SERVER_CONAN_PROFILE_BUILD)
  set(TRITON_SERVER_CONAN_PROFILE_BUILD "${TRITON_SERVER_CONAN_PROFILE_HOST}")
endif()

# _triton_server_conan_profile_args(<out_args>)
#
# The -pr:h / -pr:b pair for every conan invocation. Centralised so the two
# contexts cannot drift apart between call sites.
function(_triton_server_conan_profile_args out_args)
  set(_a "")
  if(TRITON_SERVER_CONAN_PROFILE_HOST)
    list(APPEND _a "-pr:h" "${TRITON_SERVER_CONAN_PROFILE_HOST}")
  endif()
  if(TRITON_SERVER_CONAN_PROFILE_BUILD)
    list(APPEND _a "-pr:b" "${TRITON_SERVER_CONAN_PROFILE_BUILD}")
  endif()
  set(${out_args} "${_a}" PARENT_SCOPE)
endfunction()

# _triton_server_conan_check_profile()
#
# Warn when the compiler CMake selected disagrees with the one the profile names.
# The mismatch is not an error -- Conan will happily rebuild everything from
# source under the new ID -- but it is never what anyone intends, and the only
# visible symptom is an inexplicably long build.
function(_triton_server_conan_check_profile)
  if(NOT TRITON_SERVER_CONAN_PROFILE_HOST OR NOT EXISTS "${TRITON_SERVER_CONAN_PROFILE_HOST}")
    return()
  endif()

  # A host profile targeting an architecture this machine is not, with the build
  # profile left defaulted to it, is a cross build with no build context -- Conan
  # would try to run the target's protoc on this machine.
  file(STRINGS "${TRITON_SERVER_CONAN_PROFILE_HOST}" _arch_line REGEX "^arch=")
  string(REPLACE "arch=" "" _profile_arch "${_arch_line}")
  set(_machine_arch "${CMAKE_HOST_SYSTEM_PROCESSOR}")
  if(_machine_arch STREQUAL "aarch64")
    set(_machine_arch "armv8")
  endif()
  if(_profile_arch AND NOT _profile_arch STREQUAL _machine_arch
     AND TRITON_SERVER_CONAN_PROFILE_HOST STREQUAL TRITON_SERVER_CONAN_PROFILE_BUILD)
    message(WARNING
      "Conan host profile targets arch '${_profile_arch}' but this machine is "
      "'${_machine_arch}', and no build profile was given. Conan will try to run "
      "target-architecture build tools here. Set TRITON_SERVER_CONAN_PROFILE_BUILD "
      "to a profile describing this machine.")
  endif()

  file(STRINGS "${TRITON_SERVER_CONAN_PROFILE_HOST}" _compiler_line REGEX "^compiler=")
  file(STRINGS "${TRITON_SERVER_CONAN_PROFILE_HOST}" _version_line REGEX "^compiler\\.version=")
  string(REPLACE "compiler=" "" _profile_compiler "${_compiler_line}")
  string(REPLACE "compiler.version=" "" _profile_version "${_version_line}")

  # CMake spells them GNU/Clang; Conan spells them gcc/clang.
  set(_actual_compiler "${CMAKE_CXX_COMPILER_ID}")
  if(_actual_compiler STREQUAL "GNU")
    set(_actual_compiler "gcc")
  else()
    string(TOLOWER "${_actual_compiler}" _actual_compiler)
  endif()

  # Conan profiles carry only the major for gcc, so compare on that.
  string(REGEX REPLACE "\\..*$" "" _actual_major "${CMAKE_CXX_COMPILER_VERSION}")

  if(_profile_compiler AND NOT _profile_compiler STREQUAL _actual_compiler)
    message(WARNING
      "Conan profile names compiler '${_profile_compiler}' but CMake selected "
      "'${_actual_compiler}'. No published binary will match, so every package "
      "will be rebuilt from source.")
  elseif(_profile_version AND NOT _profile_version STREQUAL _actual_major)
    message(WARNING
      "Conan profile pins ${_profile_compiler} ${_profile_version} but CMake selected "
      "${_actual_compiler} ${CMAKE_CXX_COMPILER_VERSION}. The compiler version is part "
      "of every Conan package ID, so nothing on the remote will match and every package "
      "will be rebuilt from source. Install ${_profile_compiler}-${_profile_version}, or "
      "rebuild and re-upload the recipes under the new compiler.")
  endif()
endfunction()

set(TRITON_SERVER_CONAN_RECIPE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conan/recipes"
    CACHE PATH "Directory holding local Conan recipes")

# Conan CLI verbosity, forwarded as -v<level>. Levels are Conan's own, least to
# most verbose. Anything above the default also streams output live and raises
# the build tools' own verbosity -- see _triton_server_conan_verbosity_args().
set(TRITON_SERVER_CONAN_VERBOSITY "status" CACHE STRING
    "Conan verbosity: quiet|error|warning|notice|status|verbose|debug|trace")
set_property(CACHE TRITON_SERVER_CONAN_VERBOSITY PROPERTY STRINGS
             quiet error warning notice status verbose debug trace)

# _triton_server_conan_verbosity_args(<out_args> <out_echo>)
#
# <out_args> receives the conan CLI flags for the configured verbosity.
# <out_echo> receives execute_process() keywords, empty unless output should be
# streamed.
#
# Two separate things have to be turned up to see a from-source build. -v<level>
# governs Conan's own logging only; the compiler command lines come from the
# build system Conan drives, which stays quiet until tools.build:verbosity and
# tools.compilation:verbosity say otherwise. -c:a applies them to the host and
# build contexts both, so a build requirement compiled from source is not silent
# either.
#
# Streaming matters as much as the level. execute_process() captures output, so
# without ECHO_*_VARIABLE a gRPC compile shows nothing at all for ~30 minutes and
# the captured text only ever appears if the command fails. The ECHO_ forms
# duplicate to the terminal while still capturing, so the failure message below
# keeps its contents.
function(_triton_server_conan_verbosity_args out_args out_echo)
  set(_args "-v${TRITON_SERVER_CONAN_VERBOSITY}")
  set(_echo "")

  if(TRITON_SERVER_CONAN_VERBOSITY MATCHES "^(verbose|debug|trace)$")
    list(APPEND _args "-c:a" "tools.build:verbosity=verbose"
                      "-c:a" "tools.compilation:verbosity=verbose")
    set(_echo ECHO_OUTPUT_VARIABLE ECHO_ERROR_VARIABLE COMMAND_ECHO STDOUT)
  endif()

  set(${out_args} "${_args}" PARENT_SCOPE)
  set(${out_echo} "${_echo}" PARENT_SCOPE)
endfunction()

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

# _triton_server_conan_export_recipe(<recipe_name> <out_ok>)
#
# Export a local recipe into the Conan cache so a subsequent install can resolve
# it. Lives here rather than in TritonDependencies because both resolution paths
# need it and TritonConanGraph must not depend on TritonDependencies.
#
# Exporting is what makes a local recipe authoritative. Conan resolves an
# unrevisioned reference from the cache without consulting any remote, so an
# exported grpc/1.81.1 wins over ConanCenter's regardless of remote order -- see
# TRITON_GRPC_USER_CHANNEL in CMakeLists.txt for why the reference is published
# without a user/channel and therefore has to shadow ConanCenter's outright.
function(_triton_server_conan_export_recipe recipe_name out_ok)
  # Probed in order:
  #   1. conan/recipes/<name>/          recipes owned by this repo
  #   2. ../<name>/                     the package's own checkout
  #
  # cnmem and libevhtp each carry a maintained conanfile.py at the root of their
  # own repository. Those are the authoritative recipes, so they are used in
  # place rather than copied here -- #8734 vendored its own copies and they had
  # already drifted from the originals (a different pinned commit for cnmem, and
  # 26 differing lines for libevhtp).
  set(_dir "")
  foreach(_candidate "${TRITON_SERVER_CONAN_RECIPE_DIR}/${recipe_name}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../${recipe_name}")
    get_filename_component(_abs "${_candidate}" ABSOLUTE)
    if(EXISTS "${_abs}/conanfile.py")
      set(_dir "${_abs}")
      break()
    endif()
  endforeach()

  if(NOT _dir)
    set(${out_ok} FALSE PARENT_SCOPE)
    return()
  endif()

  find_program(TRITON_SERVER_CONAN_EXECUTABLE NAMES conan REQUIRED)

  # Only -v here: `conan export` takes no profile arguments, so the -c:a confs
  # that _triton_server_conan_verbosity_args() adds would be rejected outright.
  set(_export_output OUTPUT_QUIET)
  if(TRITON_SERVER_CONAN_VERBOSITY MATCHES "^(verbose|debug|trace)$")
    set(_export_output "")
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] exporting recipe from ${_dir}")
  execute_process(
    COMMAND "${TRITON_SERVER_CONAN_EXECUTABLE}" export "${_dir}"
            "-v${TRITON_SERVER_CONAN_VERBOSITY}"
    RESULT_VARIABLE _result
    ${_export_output}
    ERROR_VARIABLE _stderr)

  if(_result EQUAL 0)
    set(${out_ok} TRUE PARENT_SCOPE)
  else()
    # Not fatal on its own: the caller may still resolve the reference from a
    # remote. Surfaced because a broken local recipe otherwise fails much later
    # as an unresolvable reference, which reads like a missing package.
    message(WARNING "conan export of '${_dir}' failed (exit ${_result}): ${_stderr}")
    set(${out_ok} FALSE PARENT_SCOPE)
  endif()
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

  _triton_server_conan_profile_args(_profile_args)
  list(APPEND _args ${_profile_args})

  _triton_server_conan_verbosity_args(_verbosity_args _verbosity_echo)
  list(APPEND _args ${_verbosity_args})

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 3/4: running conan install into '${TRITON_SERVER_CONAN_OUTPUT_DIR}'")
  execute_process(
    COMMAND "${TRITON_SERVER_CONAN_EXECUTABLE}" install
            ${_args}
            --generator=CMakeDeps
            --output-folder=${TRITON_SERVER_CONAN_OUTPUT_DIR}
            --build=missing
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    ${_verbosity_echo})

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
