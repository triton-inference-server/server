#!/bin/bash
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Script to run Apple Silicon backend tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "Apple Silicon Backend Test Runner"
echo "=================================="

# Check if we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}Error: This script must be run on macOS${NC}"
    exit 1
fi

# Check if we're on Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: Not running on Apple Silicon (arm64)${NC}"
    echo "Some tests may be skipped or fail"
fi

# Default build directory
BUILD_DIR="${BUILD_DIR:-build}"

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory '$BUILD_DIR' not found${NC}"
    echo "Please build the project first or set BUILD_DIR environment variable"
    exit 1
fi

# Function to run a test
run_test() {
    local test_name=$1
    local test_path="$BUILD_DIR/bin/$test_name"
    
    if [ ! -f "$test_path" ]; then
        echo -e "${YELLOW}Test $test_name not found, skipping${NC}"
        return 1
    fi
    
    echo -e "\n${GREEN}Running $test_name...${NC}"
    
    # Run the test with timeout
    if timeout 300 "$test_path" --gtest_color=yes; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        return 1
    fi
}

# Function to run benchmarks
run_benchmark() {
    local test_name=$1
    local test_path="$BUILD_DIR/bin/$test_name"
    
    if [ ! -f "$test_path" ]; then
        echo -e "${YELLOW}Benchmark $test_name not found, skipping${NC}"
        return 1
    fi
    
    echo -e "\n${GREEN}Running benchmark $test_name...${NC}"
    
    # Run benchmark tests (they are disabled by default)
    "$test_path" --gtest_color=yes --gtest_filter="*DISABLED_Benchmark*" --gtest_also_run_disabled_tests || true
}

# Test categories
declare -a UNIT_TESTS=(
    "amx_provider_test"
    "ane_provider_test"
    "amx_metal_interop_test"
    "ane_transformer_engine_test"
)

declare -a INTEGRATION_TESTS=(
    "apple_silicon_integration_test"
)

declare -a EXISTING_TESTS=(
    "metal_allocator_test"
    "metal_unified_memory_benchmark"
    "model_router_test"
    "profile_guided_optimizer_test"
    "winograd_conv3x3_test"
    "ane_performance_profiler_test"
)

# Parse command line arguments
RUN_UNIT=true
RUN_INTEGRATION=true
RUN_BENCHMARKS=false
RUN_EXISTING=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit-only)
            RUN_INTEGRATION=false
            RUN_EXISTING=false
            shift
            ;;
        --integration-only)
            RUN_UNIT=false
            RUN_EXISTING=false
            shift
            ;;
        --benchmarks)
            RUN_BENCHMARKS=true
            shift
            ;;
        --all)
            RUN_EXISTING=true
            RUN_BENCHMARKS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --unit-only        Run only unit tests"
            echo "  --integration-only Run only integration tests"
            echo "  --benchmarks       Also run benchmark tests"
            echo "  --all              Run all tests including existing ones"
            echo "  --verbose          Enable verbose output"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set test environment
export GTEST_COLOR=1
if [ "$VERBOSE" = true ]; then
    export GTEST_PRINT_TIME=1
fi

# Track test results
PASSED=0
FAILED=0
SKIPPED=0

# Run unit tests
if [ "$RUN_UNIT" = true ]; then
    echo -e "\n${YELLOW}=== Running Unit Tests ===${NC}"
    for test in "${UNIT_TESTS[@]}"; do
        if run_test "$test"; then
            ((PASSED++))
        else
            ((FAILED++))
        fi
    done
fi

# Run integration tests
if [ "$RUN_INTEGRATION" = true ]; then
    echo -e "\n${YELLOW}=== Running Integration Tests ===${NC}"
    for test in "${INTEGRATION_TESTS[@]}"; do
        if run_test "$test"; then
            ((PASSED++))
        else
            ((FAILED++))
        fi
    done
fi

# Run existing tests if requested
if [ "$RUN_EXISTING" = true ]; then
    echo -e "\n${YELLOW}=== Running Existing Apple Silicon Tests ===${NC}"
    for test in "${EXISTING_TESTS[@]}"; do
        if run_test "$test"; then
            ((PASSED++))
        else
            ((FAILED++))
        fi
    done
fi

# Run benchmarks if requested
if [ "$RUN_BENCHMARKS" = true ]; then
    echo -e "\n${YELLOW}=== Running Benchmarks ===${NC}"
    for test in "${UNIT_TESTS[@]}" "${INTEGRATION_TESTS[@]}"; do
        run_benchmark "$test"
    done
fi

# Summary
echo -e "\n=================================="
echo "Test Summary"
echo "=================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED${NC}"
echo "=================================="

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi