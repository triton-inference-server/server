# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

include_guard(GLOBAL)

include(TritonConan)

# Dependency resolution cascade. For each package, in order:
#
#   1. find_package()               already installed (system, toolchain, sysroot)
#   2. Conan remote                 conan install --build=missing
#   3. Local recipe                 conan/recipes/<name>, exported then installed
#   4. Source build                 deferred to third_party/CMakeLists.txt
#
# Step 4 is where the gRPC coupling matters. third_party builds absl, protobuf,
# re2, googletest and c-ares *out of the gRPC checkout*
# (SOURCE_DIR .../grpc-repo/src/grpc/third_party/<pkg>), so they have no version
# independent of the gRPC tag. If gRPC itself falls back to a source build, those
# five come with it and must be consumed from that build rather than resolved
# separately -- otherwise two different copies end up on the link line.

option(TRITON_ENABLE_CONAN "Resolve dependencies with Conan" ON)

# Host-installed packages are not used by default. An unqualified find_package()
# silently accepts any version the machine happens to carry, which is how a system
# re2 11.0.0 was picked up while TRITON_RE2_VERSION pinned 20230301. Conan is the
# provider; opting in makes the resulting build host-dependent.
option(TRITON_ALLOW_SYSTEM_PACKAGES
       "Allow dependencies to resolve from the host instead of Conan" OFF)

# Packages exempt from the rule above. CUDAToolkit and DCGM ship with the CUDA
# and datacenter-GPU installations and are tied to the host driver, so taking
# them from the system is correct rather than a reproducibility hazard -- a
# Conan-supplied copy could disagree with the installed driver.
set(TRITON_SYSTEM_PACKAGE_ALLOWLIST "CUDAToolkit;dcgm;DCGM;Threads" CACHE STRING
    "Packages that may resolve from the host even when TRITON_ALLOW_SYSTEM_PACKAGES=OFF")

# third_party/ is being retired: its ExternalProject graph is exactly what Conan
# replaces, and it is the reason absl/protobuf/re2/googletest/c-ares have no
# version of their own. Step 4 remains only so a package with no recipe yet still
# has somewhere to go, and warns when it is used.
option(TRITON_ENABLE_THIRD_PARTY_FALLBACK
       "Permit falling back to the deprecated third_party source build" ON)

# Packages that a gRPC source build produces as a side effect. Kept in sync with
# third_party/CMakeLists.txt, where each is an ExternalProject whose SOURCE_DIR
# points into the gRPC checkout.
set(TRITON_SERVER_GRPC_BUNDLED_PACKAGES absl protobuf re2 googletest c-ares
    CACHE STRING "Packages provided by a gRPC source build")
mark_as_advanced(TRITON_SERVER_GRPC_BUNDLED_PACKAGES)

set(TRITON_SERVER_CONAN_RECIPE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/conan/recipes"
    CACHE PATH "Directory holding local Conan recipes")

# Accumulates packages that reached step 4, for the source-build stage to consume.
set_property(GLOBAL PROPERTY TRITON_SERVER_SOURCE_BUILD_PACKAGES "")

# _triton_server_conan_try_install(<ref> <out_ok>)
#
# Attempt a Conan resolve without aborting on failure, so the caller can continue
# down the cascade. triton_server_conan_install_package() is fatal by design; this
# is the tolerant variant used while probing.
function(_triton_server_conan_try_install ref out_ok out_mode)
  find_program(TRITON_SERVER_CONAN_EXECUTABLE NAMES conan)
  if(NOT TRITON_SERVER_CONAN_EXECUTABLE)
    set(${out_ok} FALSE PARENT_SCOPE)
    set(${out_mode} "" PARENT_SCOPE)
    return()
  endif()

  set(_args "--requires=${ref}")
  if(CMAKE_BUILD_TYPE)
    list(APPEND _args "-s" "build_type=${CMAKE_BUILD_TYPE}")
  endif()
  if(TRITON_SERVER_CONAN_PROFILE)
    list(APPEND _args "-pr:h" "${TRITON_SERVER_CONAN_PROFILE}"
                      "-pr:b" "${TRITON_SERVER_CONAN_PROFILE}")
  endif()

  execute_process(
    COMMAND "${TRITON_SERVER_CONAN_EXECUTABLE}" install
            ${_args}
            --generator=CMakeDeps
            --output-folder=${TRITON_SERVER_CONAN_OUTPUT_DIR}
            --build=missing
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)

  if(_result EQUAL 0)
    set(${out_ok} TRUE PARENT_SCOPE)
    # Conan says "Building from source" when no prebuilt binary matched the
    # profile and --build=missing had to compile it. Anything else came down
    # from a remote or was already in the local cache.
    if(_stdout MATCHES "Building from source|Build succeeded")
      set(${out_mode} "build" PARENT_SCOPE)
    else()
      set(${out_mode} "remote" PARENT_SCOPE)
    endif()
  else()
    set(${out_ok} FALSE PARENT_SCOPE)
    set(${out_mode} "" PARENT_SCOPE)
    set(_TRITON_CONAN_LAST_ERROR "${_stderr}" PARENT_SCOPE)
  endif()
endfunction()

# _triton_server_conan_export_recipe(<recipe_name> <out_ok>)
#
# Export a local recipe from conan/recipes/<recipe_name> into the Conan cache so
# a subsequent install can resolve it. Used for packages with no ConanCenter or
# Artifactory recipe (cnmem, dcgm, libevhtp).
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

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] exporting recipe from ${_dir}")
  execute_process(
    COMMAND "${TRITON_SERVER_CONAN_EXECUTABLE}" export "${_dir}"
    RESULT_VARIABLE _result
    OUTPUT_QUIET
    ERROR_QUIET)

  if(_result EQUAL 0)
    set(${out_ok} TRUE PARENT_SCOPE)
  else()
    set(${out_ok} FALSE PARENT_SCOPE)
  endif()
endfunction()

# _triton_server_locate(<name> <ref> <provider>)
#
# Set _tsp_where to a path a reader can act on. Conan's generated data files
# expose <pkg>_PACKAGE_FOLDER_<CONFIG>, pointing into the Conan cache; that is
# far more useful than <Pkg>_DIR, which is the shared generators folder and
# identical for every Conan package. Module-mode finds (CUDAToolkit, Threads)
# set neither, so the variables a Find module conventionally exports are tried
# in turn. The Conan package name comes from the reference because it often
# differs in case from the find_package() name (protobuf vs Protobuf).
macro(_triton_server_locate _loc_name _loc_ref _loc_provider)
  set(_tsp_where "")

  if(NOT "${_loc_ref}" STREQUAL "")
    string(REGEX REPLACE "/.*" "" _loc_pkg "${_loc_ref}")
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _loc_cfg)
    if(DEFINED ${_loc_pkg}_PACKAGE_FOLDER_${_loc_cfg})
      set(_tsp_where "${${_loc_pkg}_PACKAGE_FOLDER_${_loc_cfg}}")
    endif()
  endif()

  foreach(_loc_var "${_loc_name}_DIR" "${_loc_name}_ROOT" "${_loc_name}_TARGET_DIR"
                   "${_loc_name}_LIBRARY_DIR" "${_loc_name}_INCLUDE_DIRS"
                   "${_loc_name}_INCLUDE_DIR" "${_loc_name}_BIN_DIR")
    if(NOT _tsp_where AND ${_loc_var})
      list(GET ${_loc_var} 0 _tsp_where)
    endif()
  endforeach()

  if(NOT _tsp_where)
    if("${_loc_provider}" STREQUAL "source")
      set(_tsp_where "<pending source build>")
    else()
      set(_tsp_where "<location unknown>")
    endif()
  endif()
endmacro()

# triton_server_dependency_provider(NAME <pkg> [REF <ref>] [RECIPE <dir>]
#                                   [VERSION <v>] [CONFIG] [ALLOW_SYSTEM])
#
# Resolve one package, in order:
#
#   1. Conan    the pinned reference, from a remote or a local recipe
#   2. System   the host copy, when permitted
#   3. Source   deferred to third_party (deprecated)
#
# Conan is tried first so the pinned version wins. The host is a fallback for
# what Conan cannot supply, not a shortcut past the pins -- taking it silently is
# how a system re2 11.0.0 got used while the pin said 20230301, so an unexpected
# host hit warns. Packages in TRITON_SYSTEM_PACKAGE_ALLOWLIST (CUDAToolkit, DCGM)
# are tied to the installed driver and are expected from the host, so they do not.
#
# Declared as a macro so find_package()'s many result variables land in the
# caller's scope. CMAKE_CURRENT_FUNCTION does not work in a macro, so the log tag
# is a literal that gains a ":<provider>" suffix once the provider is known.
macro(triton_server_dependency_provider)
  cmake_parse_arguments(_tsp "CONFIG;QUIET;ALLOW_SYSTEM;REQUIRED" "NAME;REF;RECIPE;VERSION" "" ${ARGN})

  if(NOT _tsp_NAME)
    message(FATAL_ERROR "triton_server_dependency_provider requires NAME")
  endif()

  set(_tsp_tag "triton_server_dependency_provider")
  set(_tsp_provider "unresolved")
  set(_tsp_mode "")
  set(_tsp_ok FALSE)
  set(_tsp_cfg "")
  if(_tsp_CONFIG)
    set(_tsp_cfg CONFIG)
  endif()
  if(NOT _tsp_RECIPE)
    set(_tsp_RECIPE "${_tsp_NAME}")
  endif()
  set(${_tsp_NAME}_FOUND FALSE)

  message(STATUS "[${_tsp_tag}] entered: NAME='${_tsp_NAME}' REF='${_tsp_REF}'")

  # ---- step 1: Conan ---------------------------------------------------------
  if(TRITON_ENABLE_CONAN AND _tsp_REF)
    message(STATUS "[${_tsp_tag}] step 1/3: resolving ${_tsp_REF} via Conan")
    _triton_server_conan_try_install("${_tsp_REF}" _tsp_ok _tsp_mode)

    if(NOT _tsp_ok)
      message(STATUS "[${_tsp_tag}] step 1/3: no remote binary, trying a local recipe for ${_tsp_RECIPE}")
      _triton_server_conan_export_recipe("${_tsp_RECIPE}" _tsp_exported)
      if(_tsp_exported)
        _triton_server_conan_try_install("${_tsp_REF}" _tsp_ok _tsp_mode)
        if(_tsp_ok)
          set(_tsp_provider "conan-recipe")
        endif()
      endif()
    endif()

    if(_tsp_ok)
      if(_tsp_provider STREQUAL "unresolved")
        set(_tsp_provider "conan-${_tsp_mode}")
      endif()
      list(PREPEND CMAKE_PREFIX_PATH "${TRITON_SERVER_CONAN_OUTPUT_DIR}")
      list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
      find_package(${_tsp_NAME} QUIET ${_tsp_cfg})
    endif()
  elseif(NOT TRITON_ENABLE_CONAN)
    message(STATUS "[${_tsp_tag}] step 1/3: skipped, TRITON_ENABLE_CONAN=OFF")
  endif()

  # ---- step 2: system fallback -----------------------------------------------
  if(NOT ${_tsp_NAME}_FOUND)
    set(_tsp_system_ok FALSE)
    if(TRITON_ALLOW_SYSTEM_PACKAGES OR _tsp_ALLOW_SYSTEM)
      set(_tsp_system_ok TRUE)
    elseif(_tsp_NAME IN_LIST TRITON_SYSTEM_PACKAGE_ALLOWLIST)
      set(_tsp_system_ok TRUE)
    endif()

    if(_tsp_system_ok)
      message(STATUS "[${_tsp_tag}] step 2/3: falling back to a host copy of ${_tsp_NAME} ${_tsp_VERSION}")
      find_package(${_tsp_NAME} ${_tsp_VERSION} QUIET ${_tsp_cfg})
      if(${_tsp_NAME}_FOUND)
        set(_tsp_provider "system")
        if(NOT _tsp_ALLOW_SYSTEM AND NOT _tsp_NAME IN_LIST TRITON_SYSTEM_PACKAGE_ALLOWLIST)
          message(WARNING
            "${_tsp_NAME} came from the host rather than Conan, so its version is "
            "whatever this machine has installed. This build will not reproduce "
            "elsewhere.")
        endif()
      endif()
    else()
      message(STATUS "[${_tsp_tag}] step 2/3: host fallback not permitted for ${_tsp_NAME}")
    endif()
  endif()

  # ---- step 3: source build --------------------------------------------------
  if(NOT ${_tsp_NAME}_FOUND AND _tsp_REQUIRED)
    message(FATAL_ERROR
      "${_tsp_NAME} is required but was not found. It is not available through "
      "Conan and no copy is installed on this system. Install it, or turn off the "
      "feature that needs it.")
  endif()

  if(NOT ${_tsp_NAME}_FOUND)
    set(_tsp_provider "source")
    message(STATUS "[${_tsp_tag}] step 3/3: deferring ${_tsp_NAME} to a source build")
    triton_server_dependency_defer_source(${_tsp_NAME})
  endif()

  set(_tsp_tag "triton_server_dependency_provider:${_tsp_provider}")
  _triton_server_locate("${_tsp_NAME}" "${_tsp_REF}" "${_tsp_provider}")
  message(STATUS "[${_tsp_tag}] done: ${_tsp_NAME} at ${_tsp_where}")
endmacro()

# triton_server_dependency_defer_source(<pkg>)
#
# Record that <pkg> must come from third_party's source build. When gRPC is
# deferred, the five packages it vendors are recorded too: they are produced by
# that single build and must be taken from it rather than resolved separately.
function(triton_server_dependency_defer_source pkg)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered: pkg='${pkg}'")

  if(NOT TRITON_ENABLE_THIRD_PARTY_FALLBACK)
    message(FATAL_ERROR
      "${pkg} could not be resolved and the third_party fallback is disabled. "
      "Add a Conan reference for it, or a recipe under "
      "${TRITON_SERVER_CONAN_RECIPE_DIR}/${pkg}.")
  endif()

  message(DEPRECATION
    "${pkg} is falling back to the third_party source build. third_party/ is being "
    "retired in favour of Conan; add a reference or a local recipe for ${pkg}.")

  get_property(_pending GLOBAL PROPERTY TRITON_SERVER_SOURCE_BUILD_PACKAGES)

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 1/2: recording '${pkg}'")
  list(APPEND _pending "${pkg}")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/2: checking gRPC bundling")
  if(pkg STREQUAL "gRPC" OR pkg STREQUAL "grpc")
    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] gRPC source build also provides: ${TRITON_SERVER_GRPC_BUNDLED_PACKAGES}")
    list(APPEND _pending ${TRITON_SERVER_GRPC_BUNDLED_PACKAGES})
    set_property(GLOBAL PROPERTY TRITON_SERVER_GRPC_PROVIDES_BUNDLED TRUE)
  endif()

  list(REMOVE_DUPLICATES _pending)
  set_property(GLOBAL PROPERTY TRITON_SERVER_SOURCE_BUILD_PACKAGES "${_pending}")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: pending source builds = ${_pending}")
endfunction()


# _triton_server_conan_package_folder(<ref> <out_path>)
#
# Resolve a Conan reference to its package folder in the cache.
#
# Needed because the Triton gRPC recipe sets cmake_find_mode=none, so CMakeDeps
# generates no config and therefore no <pkg>_PACKAGE_FOLDER_<CONFIG> data file to
# read. That is deliberate: gRPC installs a complete gRPCConfig.cmake of its own,
# and a Conan-generated grpc-config.cmake would shadow it (find_package matches
# both spellings) while exposing none of gRPC's real targets.
function(_triton_server_conan_package_folder ref out_path)
  execute_process(
    COMMAND "${TRITON_SERVER_CONAN_EXECUTABLE}" list "${ref}:*"
    OUTPUT_VARIABLE _listing RESULT_VARIABLE _rc ERROR_QUIET)

  set(${out_path} "" PARENT_SCOPE)
  if(NOT _rc EQUAL 0)
    return()
  endif()

  # CMake's regex engine has no {n} quantifier, so a 40-char hex pattern cannot
  # be written directly. Collect every hex run and keep the one of package-id
  # length.
  string(REGEX MATCHALL "[0-9a-f]+" _tokens "${_listing}")
  set(_pid "")
  foreach(_token IN LISTS _tokens)
    string(LENGTH "${_token}" _len)
    if(_len EQUAL 40 AND NOT _pid)
      set(_pid "${_token}")
    endif()
  endforeach()
  if(NOT _pid)
    return()
  endif()

  execute_process(
    COMMAND "${TRITON_SERVER_CONAN_EXECUTABLE}" cache path "${ref}:${_pid}"
    OUTPUT_VARIABLE _folder RESULT_VARIABLE _rc ERROR_QUIET)
  if(NOT _rc EQUAL 0)
    return()
  endif()

  string(STRIP "${_folder}" _folder)
  set(${out_path} "${_folder}" PARENT_SCOPE)
endfunction()

# triton_server_grpc_provide_bundle()
#
# Resolve gRPC together with the packages it vendors, from one build.
#
# gRPC v1.81.1 vendors abseil 20250512 and protobuf 33.5 in its submodules, and
# does not compile against ConanCenter's pairing of abseil 20240116.2. The Triton
# recipe builds those submodules (gRPC_<dep>_PROVIDER=module), so one package
# carries a self-consistent set. They must therefore be taken from that package
# rather than resolved independently, or two copies of each land on the link line.
#
# Order matters: gRPCConfig.cmake references protobuf::libprotobuf and
# protobuf::libprotoc as imported targets, so protobuf -- and abseil and
# utf8_range beneath it -- must be found first, or gRPC sets gRPC_FOUND=FALSE.
macro(triton_server_grpc_provide_bundle)
  set(_tsg_tag "triton_server_grpc_provide_bundle")
  message(STATUS "[${_tsg_tag}] entered: ref='${TRITON_GRPC_REF}'")

  message(STATUS "[${_tsg_tag}] step 1/3: resolving ${TRITON_GRPC_REF} via Conan")
  _triton_server_conan_try_install("${TRITON_GRPC_REF}" _tsg_ok _tsg_mode)
  if(NOT _tsg_ok)
    _triton_server_conan_export_recipe("grpc" _tsg_exported)
    if(_tsg_exported)
      _triton_server_conan_try_install("${TRITON_GRPC_REF}" _tsg_ok _tsg_mode)
      set(_tsg_mode "recipe")
    endif()
  endif()

  if(NOT _tsg_ok)
    message(FATAL_ERROR
      "Could not resolve ${TRITON_GRPC_REF}. Build it with\n"
      "  conan create conan/recipes/grpc --user=tritonserver --channel=stable --build=missing")
  endif()

  message(STATUS "[${_tsg_tag}] step 2/3: locating the package tree")
  _triton_server_conan_package_folder("${TRITON_GRPC_REF}" _tsg_root)
  if(NOT _tsg_root)
    message(FATAL_ERROR "Resolved ${TRITON_GRPC_REF} but could not locate its package folder")
  endif()
  list(PREPEND CMAKE_PREFIX_PATH "${_tsg_root}")
  list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
  set(TRITON_SERVER_GRPC_PACKAGE_ROOT "${_tsg_root}" CACHE PATH
      "Package tree supplying gRPC and its vendored dependencies" FORCE)

  message(STATUS "[${_tsg_tag}] step 3/3: adopting the vendored dependencies")
  # Dependency order: each is referenced by the next.
  foreach(_tsg_pkg absl utf8_range Protobuf re2 c-ares)
    find_package(${_tsg_pkg} CONFIG REQUIRED)
    message(STATUS "[triton_server_dependency_provider:grpc-bundled] done: ${_tsg_pkg} "
                   "${${_tsg_pkg}_VERSION} inherited from ${TRITON_GRPC_REF}")
  endforeach()

  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "[triton_server_dependency_provider:conan-${_tsg_mode}] done: gRPC "
                 "${gRPC_VERSION} at ${_tsg_root}")
endmacro()

# triton_server_dependency_report()
#
# Summarise how each dependency was resolved. Printed once after the cascade so a
# CI log shows the whole picture without reading every step line.
function(triton_server_dependency_report)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered")

  get_property(_pending GLOBAL PROPERTY TRITON_SERVER_SOURCE_BUILD_PACKAGES)
  get_property(_grpc_bundled GLOBAL PROPERTY TRITON_SERVER_GRPC_PROVIDES_BUNDLED)

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] conan enabled       : ${TRITON_ENABLE_CONAN}")
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] pending source build: ${_pending}")
  if(_grpc_bundled)
    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] gRPC source build supplies its vendored dependencies; "
                   "do not resolve them independently")
  endif()

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done")
endfunction()
