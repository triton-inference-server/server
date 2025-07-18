# Scripts Directory Structure

This directory contains organized scripts for the NVIDIA Triton Inference Server project.

## Directory Structure

### build/
Build and compilation scripts for the server:
- `build_macos.sh` - Main macOS build script
- `build_macos_patched.sh` - Patched version for compatibility
- `check_macos_env.sh` - Environment verification script
- `cmake_compatibility_patch.sh` - CMake compatibility fixes
- `cmake_wrapper.sh` - CMake wrapper for build process
- `fix_cmake_build.sh` - CMake build fixes
- `fix_libevent_includes.sh` - Libevent include path fixes
- `fix_triton_build_final.sh` - Final Triton build fixes
- `prebuild_cmake_fix.sh` - Pre-build CMake fixes

### apple-silicon/
Apple Silicon optimization and testing scripts:
- `benchmark_apple_silicon.py` - Performance benchmarking
- `benchmark_transformer.py` - Transformer model benchmarks
- `convert_bert_to_coreml.py` - BERT to CoreML conversion
- `demo_apple_silicon.py` - Apple Silicon demo
- `generate_performance_charts.py` - Performance visualization
- `monitor_apple_silicon.sh` - Hardware monitoring during execution
- `monitor_hardware.py` - Python hardware monitoring
- `qwen3_advanced_optimization.py` - Qwen3 model optimizations
- `run_apple_silicon_tests.sh` - Test runner for Apple Silicon
- `run_qwen3_full.py` - Full Qwen3 execution
- `run_transformer_demo.py` - Transformer demo runner
- `setup_qwen3_apple_silicon.py` - Qwen3 setup for Apple Silicon
- `setup_transformer_demo.sh` - Transformer demo setup
- `test_apple_silicon_optimizations.sh` - Optimization tests
- `test_transformer.py` - Transformer model tests

### testing/
General testing scripts:
- `test_macos_build.sh` - macOS build testing
- `test_models_directly.py` - Direct model testing
- `validate_environment.sh` - Environment validation

### utilities/
Utility and helper scripts:
- `execute_safe_reorganization.sh` - Safe reorganization execution
- `execute_unified_solution.sh` - Unified solution runner
- `llm_inference_pipeline.py` - LLM inference pipeline
- `phase1_agent_deployment.py` - Phase 1 deployment script
- `pipeline_client_example.py` - Pipeline client examples
- `rollback_reorganization.sh` - Reorganization rollback
- `verify_reorganization.sh` - Reorganization verification

### deployment/
(Currently empty - for future deployment scripts)

## Important Notes

- The main `build.py` and `compose.py` remain in the root directory as they are essential entry points
- `QUICK_START.sh` also remains in the root for easy access
- Backend-specific scripts remain with their respective backends in the `backends/` directory
- All scripts maintain their original permissions (executable flags preserved)