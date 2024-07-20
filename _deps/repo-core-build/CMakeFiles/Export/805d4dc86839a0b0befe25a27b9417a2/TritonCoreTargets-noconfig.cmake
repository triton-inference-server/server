#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TritonCore::triton-core-serverstub" for configuration ""
set_property(TARGET TritonCore::triton-core-serverstub APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(TritonCore::triton-core-serverstub PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/stubs/libtritonserver.so"
  IMPORTED_SONAME_NOCONFIG "libtritonserver.so"
  )

list(APPEND _cmake_import_check_targets TritonCore::triton-core-serverstub )
list(APPEND _cmake_import_check_files_for_TritonCore::triton-core-serverstub "${_IMPORT_PREFIX}/lib/stubs/libtritonserver.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
