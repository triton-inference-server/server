# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

include_guard(GLOBAL)

include(FetchContent)

# Triton's own repositories -- common, core and backend -- are not third-party
# packages and have no Conan recipes. src/CMakeLists.txt pulls them with
# FetchContent as ${TRITON_REPO_ORGANIZATION}/<name>.git, so without these
# variables the URL degrades to "/common.git" and the clone fails.
#
# These declarations are made from the top level, before add_subdirectory(src).
# FetchContent honours the *first* declaration of a content name and silently
# ignores later ones, so declaring here overrides src/CMakeLists.txt without
# editing it -- which is what allows a per-repo organisation or tag.
#
# Two modes:
#   TRITON_DEVELOPMENT_LOCAL=ON   build the checkout next to server/, no network
#   otherwise                     clone <organisation>/<repo>.git at <tag>
#
# The option can also be switched on by exporting TRITON_DEVELOPMENT_LOCAL.

if(DEFINED ENV{TRITON_DEVELOPMENT_LOCAL} AND NOT DEFINED TRITON_DEVELOPMENT_LOCAL)
  set(_triton_dev_local_default ON)
else()
  set(_triton_dev_local_default OFF)
endif()

option(TRITON_DEVELOPMENT_LOCAL
       "Build sibling checkouts of common/core/backend instead of cloning them"
       ${_triton_dev_local_default})

set(TRITON_REPO_ORGANIZATION "https://github.com/mc-nv"
    CACHE STRING "Default git organisation to pull Triton repositories from")

# Per-repo overrides. Empty means "use TRITON_REPO_ORGANIZATION".
set(TRITON_COMMON_REPO_ORGANIZATION "" CACHE STRING "Organisation override for the common repo")
set(TRITON_CORE_REPO_ORGANIZATION "" CACHE STRING "Organisation override for the core repo")
set(TRITON_BACKEND_REPO_ORGANIZATION "" CACHE STRING "Organisation override for the backend repo")

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for the common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for the core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for the backend repo")

# Explicit local paths. Empty means "search next to server/".
set(TRITON_COMMON_REPO_LOCAL_DIR "" CACHE PATH "Local checkout of the common repo")
set(TRITON_CORE_REPO_LOCAL_DIR "" CACHE PATH "Local checkout of the core repo")
set(TRITON_BACKEND_REPO_LOCAL_DIR "" CACHE PATH "Local checkout of the backend repo")

# _triton_server_repo_find_local(<repo_name> <out_dir>)
#
# Resolve the local checkout for <repo_name>, preferring an explicit
# TRITON_<REPO>_REPO_LOCAL_DIR. Otherwise probe <source>/<repo> first and then
# <source>/../<repo>, which is where the repositories sit when every Triton repo
# is checked out side by side. Sets <out_dir> to "" when nothing is found.
function(_triton_server_repo_find_local repo_name out_dir)
  string(TOUPPER "${repo_name}" _uc)

  if(TRITON_${_uc}_REPO_LOCAL_DIR)
    set(${out_dir} "${TRITON_${_uc}_REPO_LOCAL_DIR}" PARENT_SCOPE)
    return()
  endif()

  foreach(_candidate "${CMAKE_CURRENT_SOURCE_DIR}/${repo_name}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../${repo_name}")
    get_filename_component(_abs "${_candidate}" ABSOLUTE)
    if(EXISTS "${_abs}/CMakeLists.txt")
      set(${out_dir} "${_abs}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set(${out_dir} "" PARENT_SCOPE)
endfunction()

# triton_server_fetchcontent_declare_repo(<content_name> <repo_name>)
#
# Declare one Triton repository, either from a local checkout or from git.
# Declaring it here means src/CMakeLists.txt's own declaration is ignored.
function(triton_server_fetchcontent_declare_repo content_name repo_name)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered: content='${content_name}' repo='${repo_name}'")

  string(TOUPPER "${repo_name}" _uc)

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 1/3: selecting source mode")
  if(TRITON_DEVELOPMENT_LOCAL)
    _triton_server_repo_find_local("${repo_name}" _local)
    if(NOT _local)
      message(FATAL_ERROR
        "TRITON_DEVELOPMENT_LOCAL=ON but no checkout of '${repo_name}' was found next to "
        "${CMAKE_CURRENT_SOURCE_DIR}. Set TRITON_${_uc}_REPO_LOCAL_DIR to its path, or "
        "switch TRITON_DEVELOPMENT_LOCAL off to clone it instead.")
    endif()

    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/3: local checkout at ${_local}")
    FetchContent_Declare(${content_name} SOURCE_DIR "${_local}")

    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 3/3: declared from local source")
    message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: ${content_name} <- ${_local}")
    return()
  endif()

  # Per-repo organisation wins over the shared default.
  set(_org "${TRITON_${_uc}_REPO_ORGANIZATION}")
  if(NOT _org)
    set(_org "${TRITON_REPO_ORGANIZATION}")
  endif()
  set(_tag "${TRITON_${_uc}_REPO_TAG}")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 2/3: cloning from ${_org}/${repo_name}.git @ ${_tag}")
  FetchContent_Declare(
    ${content_name}
    GIT_REPOSITORY "${_org}/${repo_name}.git"
    GIT_TAG "${_tag}")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 3/3: declared from git")
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done: ${content_name} <- ${_org}/${repo_name}.git @ ${_tag}")
endfunction()

# triton_server_fetchcontent_declare_repos()
#
# Declare every Triton repository that src/CMakeLists.txt consumes.
function(triton_server_fetchcontent_declare_repos)
  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] entered: TRITON_DEVELOPMENT_LOCAL=${TRITON_DEVELOPMENT_LOCAL}")

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] step 1/1: declaring common, core and backend")
  triton_server_fetchcontent_declare_repo(repo-common common)
  triton_server_fetchcontent_declare_repo(repo-core core)
  triton_server_fetchcontent_declare_repo(repo-backend backend)

  message(STATUS "[${CMAKE_CURRENT_FUNCTION}] done")
endfunction()
