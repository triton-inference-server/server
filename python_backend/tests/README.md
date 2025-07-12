# Triton Python Backend Test Suite for macOS

This comprehensive test suite validates macOS compatibility for the Triton Python Backend, covering all aspects from core functionality to platform-specific features.

## Overview

The test suite is designed to ensure full compatibility of Triton Python Backend on macOS, including both Intel (x86_64) and Apple Silicon (arm64) architectures.

## Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── core/               # Core functionality tests
│   ├── python/             # Python backend specific tests
│   ├── onnx/              # ONNX Runtime tests
│   └── pytorch/           # PyTorch backend tests
├── integration/            # Integration tests
├── platform/              # Platform-specific tests
│   └── macos/            # macOS-specific tests
├── performance/           # Performance benchmarks
├── data/                  # Test data
├── fixtures/              # Test fixtures and models
├── scripts/               # Test scripts
├── ci/                    # CI/CD configurations
└── results/               # Test results and reports
```

## Test Categories

### 1. Core Functionality Tests (`unit/core/`)
- **test_server_lifecycle.cpp**: Server startup/shutdown, signal handling
- **test_shared_memory.cpp**: Shared memory operations, IPC
- **test_dynamic_loading.cpp**: Dynamic library loading, DYLD paths
- **test_thread_safety.cpp**: Thread safety, mutexes, atomic operations
- **test_memory_management.cpp**: Memory allocation, leak detection

### 2. Python Backend Tests (`unit/python/`)
- **test_python_backend.cpp**: Python interpreter integration
- **test_model_loading.cpp**: Model loading and initialization
- **test_python_inference.cpp**: Inference execution

### 3. Platform-Specific Tests (`platform/macos/`)
- **test_macos_compatibility.cpp**: macOS-specific features
- **test_macos_performance.cpp**: Performance on macOS
- **test_architecture_specific.cpp**: x86_64 vs arm64 tests

### 4. Integration Tests (`integration/`)
- **test_end_to_end.cpp**: Full server workflow
- **test_multi_backend.cpp**: Multiple backend integration
- **test_model_repository.cpp**: Model repository operations

## Running Tests

### Quick Start
```bash
# Run all tests
./tests/scripts/run_tests.sh

# Run specific category
./tests/scripts/run_tests.sh --category unit
./tests/scripts/run_tests.sh --category integration
./tests/scripts/run_tests.sh --category macos

# Run with filter
./tests/scripts/run_tests.sh --filter "memory"
```

### Manual Testing
```bash
# Build tests
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(sysctl -n hw.ncpu)

# Run with CTest
ctest -V

# Run specific test
./tests/unit/core/test_shared_memory
```

### Generate Test Data
```bash
python tests/scripts/generate_test_data.py --output-dir tests/data
```

### Generate Test Report
```bash
python tests/scripts/generate_report.py tests/results/
```

## CI/CD Integration

The test suite includes GitHub Actions workflows for:
- Automated testing on both Intel and Apple Silicon Macs
- Multiple Python version testing (3.8-3.11)
- Performance regression testing
- Cross-platform validation

## Test Models

Example test models are provided in `fixtures/test_models.py`:
- **SimpleAddModel**: Basic arithmetic operations
- **MatrixMultiplyModel**: Matrix operations
- **ErrorTestModel**: Error handling
- **MemoryIntensiveModel**: Memory management
- **ConcurrencyTestModel**: Concurrent execution
- **PlatformSpecificModel**: Platform-specific features
- **BatchProcessingModel**: Batch processing

## Platform-Specific Considerations

### macOS Features Tested
- Grand Central Dispatch (GCD)
- Mach time APIs
- DYLD library paths
- System Integrity Protection (SIP)
- File system case sensitivity
- pthread macOS extensions
- Quality of Service (QoS) classes

### Architecture Differences
- x86_64 vs arm64 instruction sets
- Memory alignment requirements
- Performance characteristics
- Library compatibility

## Performance Benchmarks

Performance tests measure:
- Inference latency
- Throughput (requests/second)
- Memory usage
- CPU utilization
- Multi-threading scalability

## Debugging Tests

### Enable Verbose Output
```bash
./tests/scripts/run_tests.sh --verbose
```

### Run with Debugger
```bash
lldb ./tests/unit/core/test_shared_memory
```

### Check Test Logs
```bash
cat tests/results/test_output.log
```

## Contributing

When adding new tests:
1. Place unit tests in appropriate subdirectory
2. Use descriptive test names
3. Include both positive and negative test cases
4. Add platform-specific tests when needed
5. Update CMakeLists.txt
6. Document any special requirements

## Test Requirements

- CMake 3.17+
- C++17 compiler
- Python 3.8+
- Google Test framework
- NumPy (for Python tests)
- Boost (optional)

## Known Issues

- Some tests may require disabling SIP on macOS
- Python multiprocessing has limitations due to macOS fork() safety
- File system case sensitivity varies by macOS configuration

## Support

For issues or questions about the test suite:
1. Check test logs in `results/`
2. Review platform-specific test output
3. Verify all dependencies are installed
4. Check macOS security settings