#!/bin/bash
# Test runner script for Triton Python Backend on macOS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${TEST_ROOT}/../build"
RESULTS_DIR="${TEST_ROOT}/results"

# Platform detection
PLATFORM="unknown"
ARCH="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
    ARCH=$(uname -m)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
    ARCH=$(uname -m)
fi

echo -e "${GREEN}Triton Python Backend Test Suite${NC}"
echo "Platform: $PLATFORM"
echo "Architecture: $ARCH"
echo "Build Directory: $BUILD_DIR"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to run a test category
run_test_category() {
    local category=$1
    local filter=$2
    
    echo -e "${YELLOW}Running $category tests...${NC}"
    
    if [[ "$PLATFORM" == "macos" ]]; then
        # Set macOS specific environment
        export DYLD_LIBRARY_PATH="${BUILD_DIR}/lib:${DYLD_LIBRARY_PATH}"
        export DYLD_FALLBACK_LIBRARY_PATH="/usr/local/lib:/usr/lib"
    fi
    
    # Run tests with timeout
    timeout_cmd=""
    if command -v gtimeout &> /dev/null; then
        timeout_cmd="gtimeout 300"
    elif command -v timeout &> /dev/null; then
        timeout_cmd="timeout 300"
    fi
    
    if [[ -n "$filter" ]]; then
        $timeout_cmd ctest -V -R "$filter" --output-junit "${RESULTS_DIR}/${category}_results.xml" || true
    else
        $timeout_cmd ctest -V --output-junit "${RESULTS_DIR}/${category}_results.xml" || true
    fi
}

# Parse command line arguments
TEST_FILTER=""
TEST_CATEGORY="all"
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --category)
            TEST_CATEGORY="$2"
            shift 2
            ;;
        --filter)
            TEST_FILTER="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --category <category>  Run specific test category (unit, integration, platform, performance)"
            echo "  --filter <pattern>     Filter tests by pattern"
            echo "  --verbose             Enable verbose output"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Change to build directory
cd "$BUILD_DIR"

# Build tests if needed
if [[ ! -f "${BUILD_DIR}/tests/Makefile" ]]; then
    echo -e "${YELLOW}Building tests...${NC}"
    cmake .. -DBUILD_TESTS=ON
    make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
fi

# Run tests based on category
case $TEST_CATEGORY in
    unit)
        run_test_category "unit" "$TEST_FILTER"
        ;;
    integration)
        run_test_category "integration" "$TEST_FILTER"
        ;;
    platform)
        run_test_category "platform" "$TEST_FILTER"
        ;;
    performance)
        run_test_category "performance" "$TEST_FILTER"
        ;;
    macos)
        run_test_category "macOS specific" "macos"
        ;;
    all)
        run_test_category "unit" "unit"
        run_test_category "integration" "integration"
        run_test_category "platform" "platform"
        if [[ "$PLATFORM" == "macos" ]]; then
            run_test_category "macOS specific" "macos"
        fi
        ;;
    *)
        echo -e "${RED}Unknown test category: $TEST_CATEGORY${NC}"
        exit 1
        ;;
esac

# Generate test report
echo -e "${YELLOW}Generating test report...${NC}"
python3 "${SCRIPT_DIR}/generate_report.py" "$RESULTS_DIR"

echo -e "${GREEN}Test run complete!${NC}"
echo "Results saved to: $RESULTS_DIR"