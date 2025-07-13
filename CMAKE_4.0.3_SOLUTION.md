# CMake 4.0.3 Compatibility Solution for Triton Build

## Problem Summary

The build is failing due to CMake 4.0.3 having stricter compatibility requirements. Specifically:
- CMake 4.0.3 requires that `cmake_minimum_required` specifies a version >= 3.5
- Many dependencies and older CMakeLists.txt files specify versions < 3.5
- This causes CMake to fail with policy errors

## Solution Overview

We've created a comprehensive solution with multiple scripts and patches:

### 1. **cmake_compatibility_patch.sh**
- Automatically finds and updates all CMakeLists.txt files
- Updates `cmake_minimum_required` to version 3.10 (compatible with CMake 4.x)
- Adds necessary policy settings for compatibility
- Creates a unified patch file for reference

### 2. **fix_cmake_build.sh**
- Main orchestration script
- Detects CMake 4.x and applies all necessary fixes
- Creates a patched version of build_macos.sh
- Sets up environment variables for compatibility
- Provides pre-build hooks for dependency fixes

### 3. **cmake_fetchcontent_fix.cmake**
- CMake module that provides compatibility fixes
- Handles FetchContent and ExternalProject issues
- Provides fallbacks for missing functionality

### 4. **libevent_cmake_fix.patch**
- Specific patch for libevent dependency
- Already exists in the repository

## Usage Instructions

### Quick Fix (Recommended)

1. Run the comprehensive fix script:
   ```bash
   ./fix_cmake_build.sh
   ```

2. When prompted, choose to start the build immediately, or run manually:
   ```bash
   ./build_macos_patched.sh
   ```

### Manual Steps

If you prefer to apply fixes manually:

1. Apply compatibility patches to all CMakeLists.txt files:
   ```bash
   ./cmake_compatibility_patch.sh
   ```

2. Set environment variables:
   ```bash
   export CMAKE_POLICY_DEFAULT_CMP0054=NEW
   export CMAKE_POLICY_DEFAULT_CMP0057=NEW
   export CMAKE_POLICY_DEFAULT_CMP0074=NEW
   export CMAKE_POLICY_DEFAULT_CMP0091=NEW
   ```

3. Run the build:
   ```bash
   ./build_macos.sh
   ```

### If Build Still Fails

If you encounter issues during the build with downloaded dependencies:

1. Run the pre-build fix:
   ```bash
   ./prebuild_cmake_fix.sh
   ```

2. This will patch any CMakeLists.txt files in:
   - `build/_deps/` (downloaded dependencies)
   - `third-party/` (third-party sources)

3. Retry the build:
   ```bash
   cd build && cmake --build . --parallel
   ```

## What the Fixes Do

### Version Updates
- Changes `cmake_minimum_required(VERSION 3.1)` → `cmake_minimum_required(VERSION 3.10)`
- Version 3.10 is fully compatible with CMake 4.x while maintaining backward compatibility

### Policy Settings
Adds the following policies for modern CMake behavior:
- `CMP0054`: Only interpret `if()` arguments as variables or keywords when unquoted
- `CMP0057`: Support new `if()` IN_LIST operator
- `CMP0074`: find_package uses `<PackageName>_ROOT` variables
- `CMP0091`: MSVC runtime library flags are selected by an abstraction

### Build Script Modifications
- Injects compatibility settings into the build environment
- Applies patches before CMake configuration
- Handles dependencies that get downloaded during build

## Verification

After applying the fixes, you can verify:

1. Check that patches were applied:
   ```bash
   grep "cmake_minimum_required" CMakeLists.txt
   # Should show VERSION 3.10 or higher
   ```

2. Check the patch file:
   ```bash
   cat cmake_4.0.3_compatibility.patch
   ```

3. Run a test configuration:
   ```bash
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

## Troubleshooting

### Issue: "Policy CMP0XXX is not set"
- Run `./fix_cmake_build.sh` to ensure all policies are set
- Check that environment variables are exported

### Issue: "CMake 3.X or lower is required"
- Run `./prebuild_cmake_fix.sh` to patch downloaded dependencies
- Check `build/_deps` for unpatched CMakeLists.txt files

### Issue: Build fails on specific dependency
- Note the dependency name and path
- Manually patch its CMakeLists.txt:
  ```bash
  sed -i.bak 's/VERSION 3\.[0-4]/VERSION 3.10/' path/to/CMakeLists.txt
  ```

## Alternative Solutions

If the automated fixes don't work:

1. **Use CMake 3.x**: Downgrade to CMake 3.27 or 3.28
   ```bash
   brew uninstall cmake
   brew install cmake@3.28
   ```

2. **Docker Build**: Use the official Triton Docker build environment

3. **Manual Patching**: Edit each problematic CMakeLists.txt file individually

## Summary

The CMake 4.0.3 compatibility issue is now resolved through:
- Automated patching scripts
- Environment configuration
- Pre-build hooks for dependencies
- Comprehensive error handling

Run `./fix_cmake_build.sh` to apply all fixes and start building Triton with CMake 4.0.3.