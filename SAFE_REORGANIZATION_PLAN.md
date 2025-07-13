# Safe Project Reorganization Execution Plan

## Phase 0: Pre-flight Checks and Backup
**Risk Level: LOW**
**Estimated Time: 5 minutes**

### Safety Checks
```bash
# 1. Verify clean git status
git status
# Expected: "nothing to commit, working tree clean"

# 2. Check for active processes
ps aux | grep -E "(triton|server|python)" | grep -v grep
# Expected: No active Triton processes

# 3. Check for open files
lsof +D /Volumes/Untitled/coder/server 2>/dev/null | grep -v "cwd"
# Expected: Only shell sessions

# 4. Create safety tag
git tag -a "pre-reorganization-$(date +%Y%m%d-%H%M%S)" -m "Backup before project reorganization"

# 5. Verify build directory can be safely moved
du -sh build/
ls -la build/ | head -10
```

### Rollback Command
```bash
# If needed at any point:
git reset --hard pre-reorganization-[timestamp]
```

## Phase 1: Documentation Organization
**Risk Level: LOW**
**Estimated Time: 10 minutes**

### Create Documentation Structure
```bash
# Create directory structure
mkdir -p docs/apple-silicon/{guides,reports,performance}
mkdir -p docs/build/{macos,guides}
mkdir -p docs/development
mkdir -p docs/benchmarks

# Verify directories created
ls -la docs/
```

### Move Apple Silicon Documentation
```bash
# Move reports
mv APPLE_SILICON_*_REPORT.md docs/apple-silicon/reports/
mv ANE_*.md docs/apple-silicon/reports/
mv AMX_*.md docs/apple-silicon/reports/
mv METAL_*.md docs/apple-silicon/reports/
mv PHASE*_*.md docs/apple-silicon/reports/
mv QWEN3_*.md docs/apple-silicon/reports/
mv WINOGRAD_*.md docs/apple-silicon/reports/

# Move guides
mv APPLE_SILICON_*_GUIDE.md docs/apple-silicon/guides/
mv APPLE_SILICON_*_STRATEGY.md docs/apple-silicon/guides/
mv PRODUCTION_LLM_OPTIMIZATION_GUIDE.md docs/apple-silicon/guides/

# Move performance docs
mv apple_silicon_*.pdf docs/apple-silicon/performance/
mv apple_silicon_*.png docs/apple-silicon/performance/
mv qwen3_*.png docs/apple-silicon/performance/

# Verify moves
ls docs/apple-silicon/reports/ | wc -l
ls docs/apple-silicon/guides/ | wc -l
```

### Move Build Documentation
```bash
# Move macOS build docs
mv MACOS_*.md docs/build/macos/
mv CMAKE_*.md docs/build/guides/
mv BUILD_*.md docs/build/guides/
mv DEPENDENCIES_MACOS.md docs/build/macos/
mv PROTOBUF_*.md docs/build/guides/
mv TODO_IMPLEMENTATION_SUMMARY.md docs/development/

# Verify moves
ls docs/build/macos/
ls docs/build/guides/
```

### Checkpoint 1
```bash
# Commit documentation moves
git add -A
git commit -m "Reorganize documentation into structured directories

- Move Apple Silicon docs to docs/apple-silicon/
- Move build documentation to docs/build/
- Separate reports, guides, and performance docs"

# Verify nothing broken
ls *.md | grep -E "(APPLE|MACOS|BUILD|CMAKE)"
# Should show only README.md, CONTRIBUTING.md, SECURITY.md
```

## Phase 2: Scripts and Tools Organization
**Risk Level: LOW-MEDIUM**
**Estimated Time: 15 minutes**

### Create Script Directories
```bash
mkdir -p scripts/{apple-silicon,build,testing,benchmarks,utilities}
```

### Move Apple Silicon Scripts
```bash
# Move Apple Silicon specific scripts
mv setup_qwen3_apple_silicon.py scripts/apple-silicon/
mv run_qwen3_full.py scripts/apple-silicon/
mv qwen3_advanced_optimization.py scripts/apple-silicon/
mv test_apple_silicon_optimizations.sh scripts/apple-silicon/
mv run_apple_silicon_tests.sh scripts/apple-silicon/
mv demo_apple_silicon.py scripts/apple-silicon/

# Move build scripts
mv build_macos*.sh scripts/build/
mv fix_*.sh scripts/build/
mv execute_unified_solution.sh scripts/build/
mv prebuild_cmake_fix.sh scripts/build/
mv check_macos_env.sh scripts/build/
mv validate_environment.sh scripts/build/
mv cmake_*.sh scripts/build/
mv QUICK_START.sh scripts/build/

# Move testing scripts
mv test_*.py scripts/testing/
mv test_*.sh scripts/testing/
mv run_transformer_demo.py scripts/testing/

# Move benchmark scripts
mv benchmark_*.py scripts/benchmarks/
mv generate_performance_charts.py scripts/benchmarks/
mv monitor_*.py scripts/benchmarks/
mv monitor_*.sh scripts/benchmarks/

# Move utilities
mv convert_bert_to_coreml.py scripts/utilities/
mv llm_inference_pipeline.py scripts/utilities/
mv pipeline_client_example.py scripts/utilities/
```

### Update Execute Permissions
```bash
# Ensure all shell scripts remain executable
find scripts/ -name "*.sh" -exec chmod +x {} \;
```

### Checkpoint 2
```bash
# Commit script reorganization
git add -A
git commit -m "Reorganize scripts into functional directories

- Apple Silicon scripts in scripts/apple-silicon/
- Build scripts in scripts/build/
- Testing scripts in scripts/testing/
- Benchmark scripts in scripts/benchmarks/"

# Test a critical script still works
scripts/build/check_macos_env.sh
```

## Phase 3: Examples Organization
**Risk Level: MEDIUM**
**Estimated Time: 10 minutes**

### Reorganize Examples
```bash
# Check current examples structure
ls -la examples/

# Move example scripts if not already organized
find . -maxdepth 1 -name "*example*.py" -exec mv {} examples/ \;
find . -maxdepth 1 -name "*demo*.py" -exec mv {} examples/ \;

# Ensure example data is organized
mkdir -p examples/data
mv test_data/ examples/data/ 2>/dev/null || true
```

### Checkpoint 3
```bash
git add -A
git commit -m "Consolidate example scripts in examples directory"
```

## Phase 4: Model Organization
**Risk Level: MEDIUM**
**Estimated Time: 10 minutes**

### Verify Model Structure
```bash
# Check current model organization
ls -la models/

# Models should already be organized, just verify
find models/ -type f -name "*.pt" -o -name "*.pbtxt" | head -20
```

## Phase 5: Temporary Files Cleanup
**Risk Level: LOW**
**Estimated Time: 5 minutes**

### Clean Temporary Files
```bash
# Remove Python cache files (safe as they regenerate)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Clean any log files in root
mkdir -p logs
mv *.log logs/ 2>/dev/null || true

# Clean test outputs
mv test_* logs/ 2>/dev/null || true
```

### Checkpoint 5
```bash
git add -A
git commit -m "Clean temporary files and organize logs"
```

## Phase 6: Update Critical Files
**Risk Level: HIGH**
**Estimated Time: 20 minutes**

### Update README.md
```bash
# Backup README first
cp README.md README.md.backup

# Update paths in README
# This needs manual review - create a script to help
cat > scripts/utilities/update_readme_paths.py << 'EOF'
import re

with open('README.md', 'r') as f:
    content = f.read()

# Update documentation paths
content = re.sub(r'\[([^\]]+)\]\(([A-Z_]+\.md)\)', 
                lambda m: f'[{m.group(1)}](docs/{m.group(2)})', content)

# Update script paths
content = re.sub(r'```bash\n([^`]+\.sh)', 
                lambda m: f'```bash\nscripts/build/{m.group(1)}', content)

with open('README.md.updated', 'w') as f:
    f.write(content)

print("Review README.md.updated before replacing")
EOF

python scripts/utilities/update_readme_paths.py
# Manually review and update
```

### Update CI/CD Configurations
```bash
# Check for CI files that need path updates
find . -name "*.yml" -o -name "*.yaml" | xargs grep -l "\.sh\|\.py" | grep -E "(github|gitlab|ci)"
```

### Final Checkpoint
```bash
# Final commit
git add -A
git commit -m "Update README and configurations for new structure"

# Create completion tag
git tag -a "reorganization-complete-$(date +%Y%m%d-%H%M%S)" -m "Project reorganization completed"
```

## Phase 7: Verification
**Risk Level: LOW**
**Estimated Time: 15 minutes**

### Verify Build Still Works
```bash
# Test basic build
cd /Volumes/Untitled/coder/server
scripts/build/check_macos_env.sh

# Test cmake configuration
mkdir -p build_test
cd build_test
cmake .. -DCMAKE_BUILD_TYPE=Release
cd ..
rm -rf build_test
```

### Verify Documentation Links
```bash
# Check for broken internal links
find docs/ -name "*.md" -exec grep -l "\[.*\](" {} \; | head -10
```

### Create Summary Report
```bash
cat > REORGANIZATION_COMPLETE.md << 'EOF'
# Project Reorganization Summary

## Completed: $(date)

### New Structure:
- `/docs/` - All documentation organized by topic
  - `/apple-silicon/` - Apple Silicon specific docs
  - `/build/` - Build system documentation
  - `/development/` - Development guides
- `/scripts/` - All scripts organized by function
  - `/apple-silicon/` - Apple Silicon specific scripts
  - `/build/` - Build and setup scripts
  - `/testing/` - Test scripts
  - `/benchmarks/` - Benchmark scripts
  - `/utilities/` - Utility scripts
- `/examples/` - All example code and demos
- `/models/` - Model files and configurations
- `/logs/` - Log files and test outputs

### Benefits:
1. Clear separation of concerns
2. Easy navigation for new developers
3. Reduced root directory clutter
4. Logical grouping of related files
5. Maintained git history for all files

### Next Steps:
1. Update CI/CD pipelines if needed
2. Update developer documentation
3. Notify team of new structure
EOF
```

## Rollback Plan

If anything goes wrong at any phase:

```bash
# Option 1: Reset to last checkpoint
git reset --hard HEAD~1

# Option 2: Reset to pre-reorganization state
git reset --hard pre-reorganization-[timestamp]

# Option 3: Cherry-pick specific good commits
git log --oneline
git cherry-pick [commit-hash]
```

## Execution Commands Summary

To execute the entire plan safely:

```bash
# Run each phase sequentially, checking after each
bash -c 'set -e; source SAFE_REORGANIZATION_PLAN.md'
```

Or execute phase by phase manually, verifying each checkpoint before proceeding.