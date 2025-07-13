# Script Quick Reference Guide

## Essential Scripts (in root directory)
- `./QUICK_START.sh` - Quick start guide and setup
- `./build.py` - Main build script for Triton server
- `./compose.py` - Docker compose utilities

## Build Scripts
Located in `scripts/build/`:
- `scripts/build/build_macos.sh` - Build Triton on macOS
- `scripts/build/check_macos_env.sh` - Verify macOS environment

## Apple Silicon Scripts
Located in `scripts/apple-silicon/`:
- `scripts/apple-silicon/run_apple_silicon_tests.sh` - Run full test suite
- `scripts/apple-silicon/benchmark_apple_silicon.py` - Performance benchmarks
- `scripts/apple-silicon/setup_qwen3_apple_silicon.py` - Setup Qwen3 model

## Testing Scripts
Located in `scripts/testing/`:
- `scripts/testing/validate_environment.sh` - Validate environment setup
- `scripts/testing/test_macos_build.sh` - Test macOS build

## Backend-Specific Scripts
(These remain with their backends)
- `backends/coreml/build_macos.sh` - Build CoreML backend
- `backends/pytorch/build_macos.sh` - Build PyTorch backend
- `backends/metal_mps/build_macos.sh` - Build Metal MPS backend

## Quick Commands

### To build Triton:
```bash
./build.py
```

### To run Apple Silicon tests:
```bash
./scripts/apple-silicon/run_apple_silicon_tests.sh
```

### To validate environment:
```bash
./scripts/testing/validate_environment.sh
```