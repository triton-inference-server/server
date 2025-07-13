# Protobuf Header Conflict Fix Summary

## Problem
The build was failing with protobuf errors:
- `error: unknown type name 'PROTOBUF_NAMESPACE_OPEN'`
- `error: unknown type name 'PROTOBUF_NAMESPACE_CLOSE'`

## Root Cause
The system protobuf headers (version 29.3) from Homebrew at `/opt/homebrew/include` were being included before the vendored protobuf headers (version 3.21.12) at `/Volumes/Untitled/coder/server/build/third-party/protobuf/include`.

This happened because:
1. Boost is installed via Homebrew and its include directory `/opt/homebrew/include` was added to the include paths
2. The CMakeLists.txt was adding `${Boost_INCLUDE_DIRS}` before `${Protobuf_INCLUDE_DIRS}`
3. libevent cmake files were also adding `/opt/homebrew/include` to their interface include directories

## Solution Applied

### 1. Fixed CMakeLists.txt Include Order
Modified `/Volumes/Untitled/coder/server/build/_deps/repo-core-src/src/CMakeLists.txt` to:
- Separate the include directories
- Add Protobuf includes first as SYSTEM includes
- Add Boost includes after as SYSTEM includes

### 2. Fixed libevent CMake Files
Created and ran `fix_libevent_includes.sh` to remove `/opt/homebrew/include` from libevent's interface include directories.

### 3. Fixed Compilation Flags
Edited `/Volumes/Untitled/coder/server/build/_deps/repo-core-build/triton-core/CMakeFiles/triton-core.dir/flags.make` to ensure protobuf includes come first in the compilation command.

## Result
The protobuf header errors are now resolved. The build successfully compiles the files that were previously failing with protobuf namespace errors.

## Permanent Fix Recommendation
To prevent this issue in future builds:
1. Always ensure vendored library headers are included before system headers
2. Consider using CMake's `CMAKE_PREFIX_PATH` more carefully to avoid picking up system libraries
3. Add a CMake module to clean up problematic include paths from dependencies