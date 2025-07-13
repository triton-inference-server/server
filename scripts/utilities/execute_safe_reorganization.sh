#!/bin/bash
# Safe Project Reorganization Script
# This script implements a cautious, step-by-step reorganization with checkpoints

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

checkpoint() {
    local phase=$1
    local message=$2
    echo -e "\n${GREEN}=== CHECKPOINT $phase ====${NC}"
    echo "$message"
    echo -n "Continue? (y/n/rollback): "
    read response
    case $response in
        y|Y) return 0 ;;
        rollback|r|R) 
            log_warning "Rolling back to previous checkpoint..."
            git reset --hard HEAD~1
            exit 1
            ;;
        *) 
            log_info "Stopping at checkpoint $phase"
            exit 0
            ;;
    esac
}

# Phase 0: Pre-flight Checks
echo -e "\n${GREEN}=== PHASE 0: Pre-flight Checks ===${NC}"

# Check git status
if ! git diff-index --quiet HEAD --; then
    log_error "Working directory not clean! Please commit or stash changes first."
    exit 1
fi

# Check for active processes
if ps aux | grep -E "(triton|server)" | grep -v grep | grep -v $0; then
    log_warning "Found potentially active Triton processes"
    echo -n "Continue anyway? (y/n): "
    read response
    [[ $response != "y" && $response != "Y" ]] && exit 1
fi

# Create safety tag
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_TAG="pre-reorganization-$TIMESTAMP"
git tag -a "$BACKUP_TAG" -m "Backup before project reorganization"
log_info "Created backup tag: $BACKUP_TAG"

# Show current structure summary
log_info "Current root directory files:"
ls -1 *.md 2>/dev/null | wc -l | xargs echo "  Markdown files:"
ls -1 *.sh 2>/dev/null | wc -l | xargs echo "  Shell scripts:"
ls -1 *.py 2>/dev/null | wc -l | xargs echo "  Python scripts:"

checkpoint 0 "Pre-flight checks complete. Ready to start reorganization?"

# Phase 1: Documentation Organization
echo -e "\n${GREEN}=== PHASE 1: Documentation Organization ===${NC}"

# Create directory structure
log_info "Creating documentation directories..."
mkdir -p docs/apple-silicon/{guides,reports,performance}
mkdir -p docs/build/{macos,guides}
mkdir -p docs/development
mkdir -p docs/benchmarks

# Move Apple Silicon documentation
log_info "Moving Apple Silicon documentation..."
MOVED_COUNT=0

# Reports
for file in APPLE_SILICON_*_REPORT.md ANE_*.md AMX_*.md METAL_*.md PHASE*_*.md QWEN3_*.md WINOGRAD_*.md; do
    if [[ -f "$file" ]]; then
        mv "$file" docs/apple-silicon/reports/
        ((MOVED_COUNT++))
    fi
done

# Guides
for file in APPLE_SILICON_*_GUIDE.md APPLE_SILICON_*_STRATEGY.md PRODUCTION_LLM_OPTIMIZATION_GUIDE.md; do
    if [[ -f "$file" ]]; then
        mv "$file" docs/apple-silicon/guides/
        ((MOVED_COUNT++))
    fi
done

# Performance docs
for file in apple_silicon_*.pdf apple_silicon_*.png qwen3_*.png; do
    if [[ -f "$file" ]]; then
        mv "$file" docs/apple-silicon/performance/
        ((MOVED_COUNT++))
    fi
done

log_info "Moved $MOVED_COUNT Apple Silicon files"

# Move Build Documentation
log_info "Moving build documentation..."
MOVED_COUNT=0

for file in MACOS_*.md; do
    if [[ -f "$file" ]]; then
        mv "$file" docs/build/macos/
        ((MOVED_COUNT++))
    fi
done

for file in CMAKE_*.md BUILD_*.md PROTOBUF_*.md; do
    if [[ -f "$file" ]]; then
        mv "$file" docs/build/guides/
        ((MOVED_COUNT++))
    fi
done

[[ -f DEPENDENCIES_MACOS.md ]] && mv DEPENDENCIES_MACOS.md docs/build/macos/ && ((MOVED_COUNT++))
[[ -f TODO_IMPLEMENTATION_SUMMARY.md ]] && mv TODO_IMPLEMENTATION_SUMMARY.md docs/development/ && ((MOVED_COUNT++))

log_info "Moved $MOVED_COUNT build documentation files"

# Commit Phase 1
git add -A
git commit -m "Reorganize documentation into structured directories

- Move Apple Silicon docs to docs/apple-silicon/
- Move build documentation to docs/build/
- Separate reports, guides, and performance docs"

checkpoint 1 "Documentation reorganization complete. Continue with scripts?"

# Phase 2: Scripts and Tools Organization
echo -e "\n${GREEN}=== PHASE 2: Scripts Organization ===${NC}"

# Create script directories
log_info "Creating script directories..."
mkdir -p scripts/{apple-silicon,build,testing,benchmarks,utilities}

# Move scripts with existence checks
log_info "Moving scripts..."
MOVED_COUNT=0

# Apple Silicon scripts
for file in setup_qwen3_apple_silicon.py run_qwen3_full.py qwen3_advanced_optimization.py \
            test_apple_silicon_optimizations.sh run_apple_silicon_tests.sh demo_apple_silicon.py; do
    if [[ -f "$file" ]]; then
        mv "$file" scripts/apple-silicon/
        ((MOVED_COUNT++))
    fi
done

# Build scripts
for file in build_macos*.sh fix_*.sh execute_unified_solution.sh prebuild_cmake_fix.sh \
            check_macos_env.sh validate_environment.sh cmake_*.sh QUICK_START.sh; do
    if [[ -f "$file" ]]; then
        mv "$file" scripts/build/
        ((MOVED_COUNT++))
    fi
done

# Testing scripts
for file in test_*.py test_*.sh run_transformer_demo.py; do
    if [[ -f "$file" ]] && [[ ! "$file" == "test_data" ]]; then
        mv "$file" scripts/testing/
        ((MOVED_COUNT++))
    fi
done

# Benchmark scripts
for file in benchmark_*.py generate_performance_charts.py monitor_*.py monitor_*.sh; do
    if [[ -f "$file" ]]; then
        mv "$file" scripts/benchmarks/
        ((MOVED_COUNT++))
    fi
done

# Utilities
for file in convert_bert_to_coreml.py llm_inference_pipeline.py pipeline_client_example.py; do
    if [[ -f "$file" ]]; then
        mv "$file" scripts/utilities/
        ((MOVED_COUNT++))
    fi
done

log_info "Moved $MOVED_COUNT script files"

# Update execute permissions
find scripts/ -name "*.sh" -exec chmod +x {} \;

# Commit Phase 2
git add -A
git commit -m "Reorganize scripts into functional directories

- Apple Silicon scripts in scripts/apple-silicon/
- Build scripts in scripts/build/
- Testing scripts in scripts/testing/
- Benchmark scripts in scripts/benchmarks/"

checkpoint 2 "Script reorganization complete. Continue with remaining files?"

# Phase 3: Clean up and Final Organization
echo -e "\n${GREEN}=== PHASE 3: Final Cleanup ===${NC}"

# Move any example files
if ls *example*.py *demo*.py 2>/dev/null; then
    log_info "Moving example files..."
    find . -maxdepth 1 -name "*example*.py" -exec mv {} examples/ \; 2>/dev/null || true
    find . -maxdepth 1 -name "*demo*.py" -exec mv {} examples/ \; 2>/dev/null || true
fi

# Clean temporary files
log_info "Cleaning temporary files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Organize logs
mkdir -p logs
for file in *.log; do
    if [[ -f "$file" ]]; then
        mv "$file" logs/
    fi
done

# Move test executables
for file in test_*; do
    if [[ -f "$file" ]] && [[ ! "$file" =~ \.(py|sh|md)$ ]]; then
        mv "$file" logs/
    fi
done

# Move JSON report files
mkdir -p docs/benchmarks
for file in *_report.json benchmark_results.json; do
    if [[ -f "$file" ]]; then
        mv "$file" docs/benchmarks/
    fi
done

# Create project structure documentation
cat > PROJECT_STRUCTURE.md << 'EOF'
# Project Structure

## Directory Organization

```
triton-inference-server/
├── src/                    # Core source code
├── backends/              # Backend implementations
├── docs/                  # All documentation
│   ├── apple-silicon/     # Apple Silicon specific
│   │   ├── guides/       # Implementation guides
│   │   ├── reports/      # Performance reports
│   │   └── performance/  # Benchmarks & charts
│   ├── build/            # Build documentation
│   │   ├── macos/       # macOS specific
│   │   └── guides/      # General build guides
│   ├── development/      # Development docs
│   └── benchmarks/       # Benchmark results
├── scripts/              # All scripts
│   ├── apple-silicon/    # Apple Silicon scripts
│   ├── build/           # Build & setup scripts
│   ├── testing/         # Test scripts
│   ├── benchmarks/      # Benchmark scripts
│   └── utilities/       # Utility scripts
├── examples/            # Example code
├── models/              # Model files
├── qa/                  # Quality assurance
├── python/              # Python bindings
├── build/               # Build output (git ignored)
└── logs/                # Log files (git ignored)
```

## Key Files in Root
- README.md - Main project documentation
- CMakeLists.txt - Main build configuration
- LICENSE - Project license
- CONTRIBUTING.md - Contribution guidelines
- SECURITY.md - Security policy
EOF

# Final commit
git add -A
git commit -m "Complete project reorganization

- Clean temporary files
- Organize logs and reports
- Add PROJECT_STRUCTURE.md
- Root directory now contains only essential files"

# Create completion tag
COMPLETE_TAG="reorganization-complete-$TIMESTAMP"
git tag -a "$COMPLETE_TAG" -m "Project reorganization completed successfully"

# Final summary
echo -e "\n${GREEN}=== REORGANIZATION COMPLETE ===${NC}"
log_info "Successfully reorganized project structure"
log_info "Backup tag: $BACKUP_TAG"
log_info "Completion tag: $COMPLETE_TAG"

echo -e "\n${GREEN}Remaining files in root:${NC}"
ls -1 | grep -E "\.(md|txt|yml|yaml|toml)$" | grep -v "^(docs|scripts|examples|models|qa|python|src|backends|build|logs)/"

echo -e "\n${GREEN}Quick verification:${NC}"
echo "  Documentation files: $(find docs -name "*.md" | wc -l)"
echo "  Script files: $(find scripts -name "*.sh" -o -name "*.py" | wc -l)"
echo "  Root directory files: $(ls -1 | wc -l)"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Review PROJECT_STRUCTURE.md"
echo "2. Test a build: scripts/build/check_macos_env.sh"
echo "3. Update any CI/CD configurations if needed"
echo "4. Update team documentation"

echo -e "\n${GREEN}To rollback if needed:${NC}"
echo "  git reset --hard $BACKUP_TAG"