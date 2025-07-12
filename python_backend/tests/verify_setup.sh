#!/bin/bash
# Verify test suite setup

echo "Verifying Triton Python Backend Test Suite Setup..."
echo "================================================="

# Check directory structure
echo -n "Checking directory structure... "
required_dirs=(
    "unit/core"
    "unit/python"
    "unit/onnx"
    "unit/pytorch"
    "integration"
    "platform/macos"
    "performance"
    "data"
    "fixtures"
    "scripts"
    "ci"
    "results"
)

all_good=true
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "\n  Missing directory: $dir"
        all_good=false
    fi
done

if $all_good; then
    echo "OK"
else
    echo "Some directories are missing!"
fi

# Check key files
echo -n "Checking key files... "
required_files=(
    "CMakeLists.txt"
    "scripts/run_tests.sh"
    "scripts/generate_test_data.py"
    "scripts/generate_report.py"
    "README.md"
)

all_good=true
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "\n  Missing file: $file"
        all_good=false
    fi
done

if $all_good; then
    echo "OK"
else
    echo "Some files are missing!"
fi

# Check executability
echo -n "Checking script permissions... "
if [ -x "scripts/run_tests.sh" ] && [ -x "scripts/generate_test_data.py" ]; then
    echo "OK"
else
    echo "Scripts need executable permissions"
fi

# Count test files
echo ""
echo "Test file inventory:"
echo "  Unit tests (C++): $(find unit -name "*.cpp" 2>/dev/null | wc -l)"
echo "  Integration tests: $(find integration -name "*.cpp" 2>/dev/null | wc -l)"
echo "  Platform tests: $(find platform -name "*.cpp" 2>/dev/null | wc -l)"
echo "  Test scripts: $(find scripts -name "*.py" -o -name "*.sh" 2>/dev/null | wc -l)"

# Check dependencies
echo ""
echo "Checking dependencies:"
echo -n "  CMake: "
if command -v cmake &> /dev/null; then
    cmake --version | head -1
else
    echo "NOT FOUND"
fi

echo -n "  Python: "
if command -v python3 &> /dev/null; then
    python3 --version
else
    echo "NOT FOUND"
fi

echo -n "  NumPy: "
if python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null; then
    :
else
    echo "NOT FOUND"
fi

echo ""
echo "Setup verification complete!"
echo ""
echo "To run tests:"
echo "  ./scripts/run_tests.sh"
echo ""
echo "To generate test data:"
echo "  python3 scripts/generate_test_data.py"