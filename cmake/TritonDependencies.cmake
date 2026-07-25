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
function(_triton_server_conan_try_install ref out_ok)
  find_program(TRITON_SERVER_CONAN_EXECUTABLE NAMES conan)
  if(NOT TRITON_SERVER_CONAN_EXECUTABLE)
    set(${out_ok} FALSE PARENT_SCOPE)
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
    OUTPUT_QUIET
    ERROR_VARIABLE _stderr)

  if(_result EQUAL 0)
    set(${out_ok} TRUE PARENT_SCOPE)
  else()
    set(${out_ok} FALSE PARENT_SCOPE)
    set(_TRITON_CONAN_LAST_ERROR "${_stderr}" PARENT_SCOPE)
  endif()
endfunction()

# _triton_server_conan_export_recipe(<recipe_name> <out_ok>)
#
# Export a local recipe from conan/recipes/<recipe_name> into the Conan cache so
# a subsequent install can resolve it. Used for packages with no ConanCenter or
# Artifactory recipe (cnmem, dcgm, libevhtp).
function(_triton_server_conan_export_recipe recipe_name out_ok)
  set(_dir "${TRITON_SERVER_CONAN_RECIPE_DIR}/${recipe_name}")
  if(NOT EXISTS "${_dir}/conanfile.py")
    set(${out_ok} FALSE PARENT_SCOPE)
    return()
  endif()

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

# triton_server_provide_package(NAME <pkg> [REF <ref>] [RECIPE <dir>] [CONFIG])
#
# Run the cascade for one package. Declared as a macro rather than a function so
# that find_package()'s many result variables (<pkg>_FOUND, _VERSION, _DIR, ...)
# land in the caller's scope without being enumerated and re-exported.
#
# Because it is a macro, CMAKE_CURRENT_FUNCTION is unavailable and the log tag is
# written literally.
macro(triton_server_provide_package)
  cmake_parse_arguments(_tsp "CONFIG;QUIET" "NAME;REF;RECIPE" "" ${ARGN})

  if(NOT _tsp_NAME)
    message(FATAL_ERROR "triton_server_provide_package requires NAME")
  endif()

  set(_tsp_tag "triton_server_provide_package")
  set(_tsp_cfg "")
  if(_tsp_CONFIG)
    set(_tsp_cfg CONFIG)
  endif()
  if(NOT _tsp_RECIPE)
    set(_tsp_RECIPE "${_tsp_NAME}")
  endif()

  message(STATUS "[${_tsp_tag}] entered: NAME='${_tsp_NAME}' REF='${_tsp_REF}'")

  # ---- step 1: already available? -------------------------------------------
  message(STATUS "[${_tsp_tag}] step 1/4: probing for an existing ${_tsp_NAME}")
  find_package(${_tsp_NAME} QUIET ${_tsp_cfg})

  if(${_tsp_NAME}_FOUND)
    message(STATUS "[${_tsp_tag}] done: ${_tsp_NAME} already available (source=preinstalled)")
  else()

    # ---- step 2: Conan remote ------------------------------------------------
    if(TRITON_ENABLE_CONAN AND _tsp_REF)
      message(STATUS "[${_tsp_tag}] step 2/4: resolving ${_tsp_REF} via Conan")
      _triton_server_conan_try_install("${_tsp_REF}" _tsp_ok)

      # ---- step 3: local recipe ---------------------------------------------
      if(NOT _tsp_ok)
        message(STATUS "[${_tsp_tag}] step 3/4: no remote recipe, trying ${TRITON_SERVER_CONAN_RECIPE_DIR}/${_tsp_RECIPE}")
        _triton_server_conan_export_recipe("${_tsp_RECIPE}" _tsp_exported)
        if(_tsp_exported)
          _triton_server_conan_try_install("${_tsp_REF}" _tsp_ok)
        endif()
      endif()

      if(_tsp_ok)
        list(PREPEND CMAKE_PREFIX_PATH "${TRITON_SERVER_CONAN_OUTPUT_DIR}")
        list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
        find_package(${_tsp_NAME} QUIET ${_tsp_cfg})
      endif()
    elseif(NOT TRITON_ENABLE_CONAN)
      message(STATUS "[${_tsp_tag}] step 2/4: skipped, TRITON_ENABLE_CONAN=OFF")
    endif()

    # ---- step 4: source build ------------------------------------------------
    if(${_tsp_NAME}_FOUND)
      message(STATUS "[${_tsp_tag}] done: ${_tsp_NAME} resolved (source=conan)")
    else()
      message(STATUS "[${_tsp_tag}] step 4/4: deferring ${_tsp_NAME} to a source build")
      triton_server_defer_source_build(${_tsp_NAME})
    endif()
  endif()
endmacro()

# triton_server_defer_source_build(<pkg>)
#
# Record that <pkg> must come from third_party's source build. When gRPC is
# deferred, the five packages it vendors are recorded too: they are produced by
# that single build and must be taken from it rather than resolved separately.
function(triton_server_defer_source_build pkg)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered: pkg='${pkg}'")

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

# triton_server_report_dependencies()
#
# Summarise how each dependency was resolved. Printed once after the cascade so a
# CI log shows the whole picture without reading every step line.
function(triton_server_report_dependencies)
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
