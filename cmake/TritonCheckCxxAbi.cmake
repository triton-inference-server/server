# TritonCheckCxxAbi.cmake
#
# Detects the _GLIBCXX_USE_CXX11_ABI value that the current build will use and
# validates that the Conan-provided binary packages were compiled with the same
# ABI setting.  A mismatch causes silent link failures or runtime crashes when
# mixing old ABI (=0) and new ABI (=1) translation units.
#
# Usage (call once, early in the root CMakeLists.txt after find_package calls):
#   include(cmake/TritonCheckCxxAbi.cmake)
#   triton_check_cxx_abi()
#
# The macro sets TRITON_GLIBCXX_USE_CXX11_ABI (cache variable) to 0 or 1.

macro(triton_check_cxx_abi)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        return()
    endif()

    # ---------------------------------------------------------------
    # 1. Probe the ABI the compiler will use for this build.
    # ---------------------------------------------------------------
    include(CheckCXXSourceCompiles)
    set(_abi_probe_src "
#include <string>
#if _GLIBCXX_USE_CXX11_ABI
int use_new_abi = 1;
#else
int use_new_abi = 0;
#endif
int main() { return use_new_abi; }
")
    # Ask the compiler directly via try_compile output rather than
    # running the binary, so cross-compilation works too.
    file(WRITE "${CMAKE_BINARY_DIR}/_abi_probe.cpp" "${_abi_probe_src}")
    try_compile(
        _abi_compile_ok
        "${CMAKE_BINARY_DIR}/_abi_probe_build"
        SOURCES "${CMAKE_BINARY_DIR}/_abi_probe.cpp"
        CXX_STANDARD 17
        OUTPUT_VARIABLE _abi_compile_output
    )

    if(NOT _abi_compile_ok)
        message(WARNING "TritonCheckCxxAbi: could not compile ABI probe. Skipping ABI check.")
        return()
    endif()

    # Extract the _GLIBCXX_USE_CXX11_ABI value by compiling a file that
    # emits a preprocessor-expanded value we can grep.
    file(WRITE "${CMAKE_BINARY_DIR}/_abi_value.cpp"
        "#include <string>\nint abi_value = _GLIBCXX_USE_CXX11_ABI;\n")
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -E -dM
                "${CMAKE_BINARY_DIR}/_abi_value.cpp"
        OUTPUT_VARIABLE _abi_macros
        ERROR_QUIET
    )

    if(_abi_macros MATCHES "#define _GLIBCXX_USE_CXX11_ABI ([01])")
        set(_detected_abi "${CMAKE_MATCH_1}")
    else()
        # Default: GCC >= 5 defaults to 1 (new ABI).
        set(_detected_abi 1)
    endif()

    set(TRITON_GLIBCXX_USE_CXX11_ABI "${_detected_abi}"
        CACHE STRING "Detected _GLIBCXX_USE_CXX11_ABI value (0=old, 1=new)" FORCE)
    message(STATUS "TritonCheckCxxAbi: _GLIBCXX_USE_CXX11_ABI = ${TRITON_GLIBCXX_USE_CXX11_ABI}")

    # ---------------------------------------------------------------
    # 2. Validate Conan-provided binary packages that embed ABI info.
    # ---------------------------------------------------------------
    # Conan 2 records the compiler.libcxx setting in the package ID.
    # If the Conan toolchain was generated we can cross-check via the
    # conanbuildinfo vars; otherwise we skip silently.
    if(DEFINED CONAN_COMPILER_LIBCXX)
        if("${CONAN_COMPILER_LIBCXX}" STREQUAL "libstdc++" AND
           "${TRITON_GLIBCXX_USE_CXX11_ABI}" STREQUAL "1")
            message(FATAL_ERROR
                "ABI MISMATCH: Build is using new C++11 ABI "
                "(_GLIBCXX_USE_CXX11_ABI=1) but Conan packages were built "
                "with libstdc++ (old ABI). Re-run 'conan install' with "
                "compiler.libcxx=libstdc++11 in the host profile.")
        endif()
        if("${CONAN_COMPILER_LIBCXX}" STREQUAL "libstdc++11" AND
           "${TRITON_GLIBCXX_USE_CXX11_ABI}" STREQUAL "0")
            message(FATAL_ERROR
                "ABI MISMATCH: Build is using old ABI "
                "(_GLIBCXX_USE_CXX11_ABI=0) but Conan packages were built "
                "with libstdc++11 (new ABI). Re-run 'conan install' with "
                "compiler.libcxx=libstdc++ in the host profile, or remove "
                "-D_GLIBCXX_USE_CXX11_ABI=0 from your compile flags.")
        endif()
    endif()

    # ---------------------------------------------------------------
    # 3. Warn if a user has manually set a conflicting compile flag.
    # ---------------------------------------------------------------
    if(DEFINED CMAKE_CXX_FLAGS AND
       CMAKE_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI=([01])")
        set(_flag_abi "${CMAKE_MATCH_1}")
        if(NOT "${_flag_abi}" STREQUAL "${_detected_abi}")
            message(WARNING
                "TritonCheckCxxAbi: CMAKE_CXX_FLAGS sets "
                "_GLIBCXX_USE_CXX11_ABI=${_flag_abi} but the compiler "
                "default is ${_detected_abi}. This will likely cause "
                "linker errors with Conan-provided binary packages.")
        endif()
    endif()
endmacro()
