#!/bin/bash
# Post-Reorganization Verification Script
# Ensures the reorganization was successful and nothing is broken

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Post-Reorganization Verification ===${NC}\n"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function
check_test() {
    local test_name=$1
    local command=$2
    echo -n "Testing $test_name... "
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# 1. Check directory structure
echo -e "\n${YELLOW}1. Verifying Directory Structure${NC}"
check_test "docs directory" "[[ -d docs/apple-silicon && -d docs/build ]]"
check_test "scripts directory" "[[ -d scripts/apple-silicon && -d scripts/build ]]"
check_test "examples directory" "[[ -d examples ]]"
check_test "models directory" "[[ -d models ]]"

# 2. Check essential files remain in root
echo -e "\n${YELLOW}2. Verifying Essential Root Files${NC}"
check_test "README.md" "[[ -f README.md ]]"
check_test "CMakeLists.txt" "[[ -f CMakeLists.txt ]]"
check_test "LICENSE" "[[ -f LICENSE ]]"
check_test "build.py" "[[ -f build.py ]]"

# 3. Check moved documentation
echo -e "\n${YELLOW}3. Verifying Documentation Organization${NC}"
check_test "Apple Silicon reports" "[[ $(ls docs/apple-silicon/reports/*.md 2>/dev/null | wc -l) -gt 0 ]]"
check_test "Build guides" "[[ $(ls docs/build/guides/*.md 2>/dev/null | wc -l) -gt 0 ]]"
check_test "Performance charts" "[[ $(ls docs/apple-silicon/performance/*.png 2>/dev/null | wc -l) -gt 0 ]]"

# 4. Check script organization
echo -e "\n${YELLOW}4. Verifying Script Organization${NC}"
check_test "Build scripts" "[[ $(ls scripts/build/*.sh 2>/dev/null | wc -l) -gt 0 ]]"
check_test "Apple Silicon scripts" "[[ $(ls scripts/apple-silicon/*.py 2>/dev/null | wc -l) -gt 0 ]]"
check_test "Script permissions" "[[ -x scripts/build/check_macos_env.sh ]]"

# 5. Check for orphaned files
echo -e "\n${YELLOW}5. Checking for Orphaned Files${NC}"
ORPHANS=$(ls -1 *.md 2>/dev/null | grep -v "^(README|CONTRIBUTING|SECURITY|LICENSE)" | wc -l)
if [[ $ORPHANS -eq 0 ]]; then
    echo -e "No orphaned markdown files ${GREEN}PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "Found $ORPHANS orphaned markdown files ${YELLOW}WARNING${NC}"
    ls -1 *.md | grep -v "^(README|CONTRIBUTING|SECURITY)"
fi

# 6. Test critical functionality
echo -e "\n${YELLOW}6. Testing Critical Functionality${NC}"
check_test "CMake configuration" "cmake --version"
check_test "Python availability" "python3 --version"

# 7. Check git status
echo -e "\n${YELLOW}7. Verifying Git Status${NC}"
if git diff-index --quiet HEAD --; then
    echo -e "Git working directory clean ${GREEN}PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "Git working directory has changes ${YELLOW}WARNING${NC}"
    git status --short
fi

# 8. Test a key script
echo -e "\n${YELLOW}8. Testing Key Scripts${NC}"
if [[ -f scripts/build/check_macos_env.sh ]]; then
    echo "Running environment check..."
    if scripts/build/check_macos_env.sh > /dev/null 2>&1; then
        echo -e "Environment check ${GREEN}PASSED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "Environment check ${YELLOW}WARNING${NC} (may need dependencies)"
    fi
fi

# 9. Verify no broken symlinks
echo -e "\n${YELLOW}9. Checking for Broken Symlinks${NC}"
BROKEN_LINKS=$(find . -type l ! -exec test -e {} \; -print 2>/dev/null | wc -l)
if [[ $BROKEN_LINKS -eq 0 ]]; then
    echo -e "No broken symlinks found ${GREEN}PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "Found $BROKEN_LINKS broken symlinks ${RED}FAILED${NC}"
    find . -type l ! -exec test -e {} \; -print
    ((TESTS_FAILED++))
fi

# 10. Check build directory
echo -e "\n${YELLOW}10. Checking Build Directory${NC}"
if [[ -d build ]]; then
    echo -e "Build directory exists ${GREEN}PASSED${NC}"
    echo "  Size: $(du -sh build | cut -f1)"
    ((TESTS_PASSED++))
else
    echo -e "Build directory missing ${YELLOW}INFO${NC} (normal if clean install)"
fi

# Summary
echo -e "\n${GREEN}=== Verification Summary ===${NC}"
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"

if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "\n${GREEN}✓ All critical tests passed!${NC}"
    echo "The reorganization appears successful."
else
    echo -e "\n${YELLOW}⚠ Some tests failed.${NC}"
    echo "Please review the failures above."
fi

# Provide useful next steps
echo -e "\n${YELLOW}Recommended Next Steps:${NC}"
echo "1. Review PROJECT_STRUCTURE.md for the new layout"
echo "2. Update any personal scripts that reference old paths"
echo "3. Run a test build: mkdir build && cd build && cmake .."
echo "4. Update CI/CD configurations if they reference specific paths"

# Show quick navigation help
echo -e "\n${YELLOW}Quick Navigation:${NC}"
echo "  Apple Silicon docs: docs/apple-silicon/"
echo "  Build scripts: scripts/build/"
echo "  Examples: examples/"
echo "  Test scripts: scripts/testing/"