# macOS Dynamic Library Loading Changes

## Summary of Changes Made

### 1. Created Platform Library Utility Header
- Created `/Volumes/Untitled/coder/server/src/platform_library.h` with platform-specific library handling utilities
- Provides functions for:
  - Getting correct library extension (.so for Linux, .dylib for macOS, .dll for Windows)
  - Getting correct library prefix (lib for Unix-like, empty for Windows)
  - Converting base library names to platform-specific format
  - Platform-specific dlopen flags (RTLD_LOCAL for macOS, RTLD_GLOBAL for Linux)
  - Platform-specific library path handling

### 2. Updated Command Line Parser
- Modified `/Volumes/Untitled/coder/server/src/command_line_parser.cc` to:
  - Include the new platform_library.h header
  - Use platform-specific library name for cache library documentation (libtritoncache.so → libtritoncache.dylib on macOS)

### 3. Updated CMakeLists.txt Files
- Updated all test backend CMakeLists.txt files to conditionally apply linker scripts:
  - `/Volumes/Untitled/coder/server/src/test/sequence/CMakeLists.txt`
  - `/Volumes/Untitled/coder/server/src/test/query_backend/CMakeLists.txt`
  - `/Volumes/Untitled/coder/server/src/test/dyna_sequence/CMakeLists.txt`
  - `/Volumes/Untitled/coder/server/src/test/implicit_state/CMakeLists.txt`
  - `/Volumes/Untitled/coder/server/src/test/iterative_sequence/CMakeLists.txt`
  - `/Volumes/Untitled/coder/server/src/test/distributed_addsub/CMakeLists.txt`
  - `/Volumes/Untitled/coder/server/src/test/repoagent/relocation_repoagent/CMakeLists.txt`
- Linker scripts (--version-script) are now only applied on non-Apple platforms
- Python backend CMakeLists.txt already had macOS support

## Key Platform Differences Handled

### Library Extensions
- Linux: `.so` (shared object)
- macOS: `.dylib` (dynamic library)
- Windows: `.dll` (dynamic link library)

### Linker Scripts
- Linux: Supports GNU ld version scripts for symbol visibility
- macOS: Does not support version scripts; uses different mechanisms for symbol visibility
- Windows: Uses .def files or __declspec for symbol export

### dlopen Flags
- Linux: Typically uses `RTLD_NOW | RTLD_GLOBAL` for plugin systems
- macOS: Prefers `RTLD_NOW | RTLD_LOCAL` to avoid symbol conflicts
- Windows: Uses LoadLibrary API instead of dlopen

## Usage Example

```cpp
#include "platform_library.h"

// Get platform-specific library name
std::string cache_lib = triton::core::GetPlatformLibraryName("tritoncache");
// Returns: "libtritoncache.so" on Linux
//          "libtritoncache.dylib" on macOS
//          "tritoncache.dll" on Windows

// Get dlopen flags
int flags = triton::core::GetPlatformDlopenFlags();
// Returns appropriate flags for the platform

// Build library path
std::string lib_path = triton::core::GetPlatformLibraryPath("/opt/tritonserver/backends", cache_lib);
```

## Testing Recommendations

1. Test that all backends load correctly on macOS
2. Verify that shared libraries are generated with .dylib extension
3. Check that symbol visibility is correct without version scripts
4. Test plugin loading with appropriate dlopen flags
5. Verify that library paths are resolved correctly

## Additional Considerations

1. **Runtime Library Paths**: macOS uses different mechanisms for runtime library paths:
   - `@rpath`: Runtime search path
   - `@loader_path`: Directory containing the loading library
   - `@executable_path`: Directory containing the main executable
   - These may need to be handled in the future if runtime path issues arise

2. **Symbol Visibility**: Without version scripts on macOS, consider using:
   - Compiler visibility attributes: `__attribute__((visibility("default")))`
   - Exported symbols list files with `-exported_symbols_list`

3. **Backend Loading**: The actual backend loading code appears to be in the Triton core repository, which may need similar updates.