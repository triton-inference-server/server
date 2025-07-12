# Python Backend macOS Implementation

This document describes the complete implementation of Python backend support for macOS in Triton Inference Server.

## Overview

The Python backend allows running Python models in Triton. This implementation adds full macOS support by addressing platform-specific differences in:

1. Python detection and linking
2. Dynamic library loading (DYLD_LIBRARY_PATH vs LD_LIBRARY_PATH)
3. Shared memory handling
4. Python framework paths

## Key Changes Made

### 1. CMakeLists.txt - Python Detection on macOS

Added macOS-specific Python detection that handles both system Python and Homebrew installations:

```cmake
# macOS Python detection - ensure we use the correct Python version
if(APPLE)
  # First try to find Python3 using the modern FindPython3 module
  find_package(Python3 COMPONENTS Interpreter Development QUIET)
  if(Python3_FOUND)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE FILEPATH "Python executable" FORCE)
    set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS} CACHE PATH "Python include directories" FORCE)
    set(PYTHON_LIBRARIES ${Python3_LIBRARIES} CACHE FILEPATH "Python libraries" FORCE)
  else()
    # Fallback to manual detection for Homebrew Python
    execute_process(
      COMMAND brew --prefix python3
      OUTPUT_VARIABLE PYTHON_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(PYTHON_PREFIX)
      set(PYTHON_EXECUTABLE "${PYTHON_PREFIX}/bin/python3" CACHE FILEPATH "Python executable" FORCE)
    endif()
  endif()
endif()
```

### 2. stub_launcher.cc - macOS Library Path Handling

Updated the stub launcher to use DYLD_LIBRARY_PATH on macOS instead of LD_LIBRARY_PATH:

```cpp
#ifdef __APPLE__
    ss << "source " << path_to_activate_
       << " && exec env DYLD_LIBRARY_PATH=" << path_to_libpython_
       << ":$DYLD_LIBRARY_PATH " << python_backend_stub << " " << model_path_
       // ... rest of arguments
#else
    ss << "source " << path_to_activate_
       << " && exec env LD_LIBRARY_PATH=" << path_to_libpython_
       << ":$LD_LIBRARY_PATH " << python_backend_stub << " " << model_path_
       // ... rest of arguments
#endif
```

### 3. Python Framework Path Detection

Added support for Python framework installations (common on macOS):

```cpp
// On macOS, also check for Python framework paths
#ifdef __APPLE__
  // Check if this is a framework-based Python (e.g., from Homebrew)
  std::string framework_path = python_execution_env + "/Frameworks/Python.framework/Versions/Current/lib";
  if (FileExists(framework_path)) {
    path_to_libpython_ = framework_path;
  }
#endif
```

### 4. pb_stub.cc - Dynamic Library Path Environment

Updated the stub process to handle DYLD_LIBRARY_PATH correctly:

```cpp
#ifdef __APPLE__
      const char* lib_path_var = "DYLD_LIBRARY_PATH";
#else
      const char* lib_path_var = "LD_LIBRARY_PATH";
#endif
```

### 5. shm_monitor/CMakeLists.txt - Remove -lrt on macOS

Fixed shared memory monitor build by not linking with -lrt on macOS (it doesn't exist):

```cmake
if(NOT APPLE)
  target_link_libraries(
    triton-shm-monitor
    PRIVATE
      -lrt # shared memory
  )
endif()
```

## Building on macOS

Use the provided build script:

```bash
./build_macos.sh
```

Or manually:

```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX:PATH=$(pwd)/install \
    -DTRITON_ENABLE_GPU=OFF \
    -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
make install
```

## Testing

A simple test model is provided in `test_models/simple_python/`:

- `config.pbtxt`: Model configuration
- `1/model.py`: Python model that adds 1 to input

To test:
1. Build the Python backend
2. Copy the test model to your Triton model repository
3. Start Triton with the Python backend
4. Send inference requests to the `simple_python` model

## Platform-Specific Considerations

### macOS System Integrity Protection (SIP)

macOS SIP can interfere with DYLD_LIBRARY_PATH. If you encounter issues:
- Use Python virtual environments
- Install Python via Homebrew
- Avoid modifying system Python

### Shared Memory

The implementation uses Boost.Interprocess which handles platform differences automatically. No macOS-specific changes were needed for shared memory.

### Process Management

Process spawning, signals, and wait operations work the same on macOS as on Linux, so no changes were needed.

## Future Improvements

1. Add support for Apple Silicon (M1/M2) specific optimizations
2. Test with conda environments on macOS
3. Add macOS-specific CI/CD pipeline
4. Performance profiling on macOS

## Troubleshooting

### Common Issues

1. **Python not found**: Ensure Python 3 is installed via Homebrew or system package manager
2. **Library loading errors**: Check DYLD_LIBRARY_PATH is set correctly
3. **Build failures**: Install required dependencies: `brew install cmake boost libarchive`

### Debug Tips

- Use `TRITONSERVER_LOG_VERBOSE=1` for detailed logging
- Check `otool -L` to verify library dependencies
- Use `DYLD_PRINT_LIBRARIES=1` to debug library loading

## Conclusion

This implementation provides full Python backend support for macOS, handling all platform-specific differences while maintaining compatibility with the existing Linux implementation.