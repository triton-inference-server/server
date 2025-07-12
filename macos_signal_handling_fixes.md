# macOS Signal Handling Fixes for Triton Server

## Overview
This document describes the signal handling compatibility issues on macOS and the implemented fixes.

## Key Issues Addressed

### 1. SIGPIPE Handling
**Problem**: On macOS, broken pipe errors (SIGPIPE) can terminate the process when writing to closed sockets. This is critical for HTTP/gRPC servers.

**Linux vs macOS Differences**:
- Linux: Can use `MSG_NOSIGNAL` flag with `send()` to prevent SIGPIPE
- macOS: Does not support `MSG_NOSIGNAL`, must use `SO_NOSIGPIPE` socket option

**Solution**: 
- Ignore SIGPIPE globally using `sigaction()` with `SIG_IGN`
- Created `macos_socket_utils.h` for socket-specific handling if needed

### 2. Signal Handler Registration
**Problem**: Using deprecated `signal()` function which has portability issues.

**Solution**: 
- Replaced `signal()` with `sigaction()` for more reliable and portable signal handling
- Added proper signal mask initialization
- Used appropriate flags (`SA_RESTART` for graceful shutdown signals, `SA_RESETHAND` for error signals)

### 3. Stack Trace on Crashes
**Problem**: Boost.Stacktrace may not work correctly on macOS due to different debugging symbol formats.

**Solution**: 
- Added try-catch block around stack trace generation on macOS
- Provides graceful fallback if stack trace fails

## Implementation Details

### Modified Files

1. **src/triton_signal.cc**
   - Added platform-specific includes for macOS
   - Replaced `signal()` with `sigaction()` for all signal handlers
   - Added SIGPIPE handling with proper macOS guards
   - Enhanced error signal handler with macOS-specific error handling

2. **src/macos_socket_utils.h** (new file)
   - Utility functions for macOS socket configuration
   - `ConfigureMacOSSocket()`: Sets SO_NOSIGPIPE on sockets
   - `SafeSend()`: Wrapper for send() that handles platform differences

### Signal Handlers Implemented

1. **Graceful Shutdown Signals** (SIGINT, SIGTERM)
   - Uses `SA_RESTART` flag to restart interrupted system calls
   - Allows server to cleanly shutdown

2. **Error Signals** (SIGSEGV, SIGABRT)
   - Uses `SA_RESETHAND` flag to reset handler after execution
   - Attempts to print stack trace before core dump
   - Handles macOS stack trace failures gracefully

3. **SIGPIPE** (macOS only)
   - Set to `SIG_IGN` to prevent process termination
   - Essential for network servers

## Testing Recommendations

1. **SIGPIPE Testing**:
   ```bash
   # Start server
   ./tritonserver --model-repository=/path/to/models
   
   # In another terminal, test broken pipe
   # Send request and close connection abruptly
   curl -X POST http://localhost:8000/v2/models/test/infer --max-time 0.1
   ```

2. **Signal Handler Testing**:
   ```bash
   # Test graceful shutdown
   kill -SIGTERM <pid>
   kill -SIGINT <pid>  # Ctrl+C
   
   # Verify server shuts down cleanly
   ```

3. **Stack Trace Testing**:
   ```bash
   # Force segmentation fault (requires debug build)
   # Server should print stack trace before crashing
   ```

## Platform Detection

The code uses standard macOS detection:
```cpp
#ifdef __APPLE__
  // macOS-specific code
#endif
```

## Future Considerations

1. **HTTP/gRPC Libraries**: If evhtp or gRPC libraries create their own sockets, they may need to be configured with SO_NOSIGPIPE. The provided utilities in `macos_socket_utils.h` can be used for this purpose.

2. **Performance**: Ignoring SIGPIPE globally is the most robust solution. The alternative of setting SO_NOSIGPIPE on each socket has minimal performance impact but requires more code changes.

3. **Debugging**: On macOS, core dumps may need to be enabled explicitly:
   ```bash
   ulimit -c unlimited
   ```

## Compatibility

- Fully backward compatible with Linux
- No changes to external APIs
- No performance impact on Linux builds
- Minimal performance impact on macOS (signal handler setup is one-time cost)