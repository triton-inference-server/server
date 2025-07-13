# 🚀 GitLab & GitHub Setup Guide

## Setting Up Your GitLab Repository

### 1. Create New GitLab Project

```bash
# Option A: Create via GitLab Web UI
# 1. Go to your GitLab instance
# 2. Click "New Project"
# 3. Name it: "triton-inference-server-apple-silicon"
# 4. Set visibility (private/internal/public)
# 5. Don't initialize with README (we have one)

# Option B: Using GitLab CLI (if installed)
glab project create triton-inference-server-apple-silicon --private
```

### 2. Add GitLab Remote

```bash
# Add your GitLab remote (replace with your GitLab URL)
git remote add gitlab git@gitlab.com:YOUR_USERNAME/triton-inference-server-apple-silicon.git

# Or if using HTTPS:
git remote add gitlab https://gitlab.com/YOUR_USERNAME/triton-inference-server-apple-silicon.git

# Verify remotes
git remote -v
```

### 3. Push to GitLab

```bash
# Push main branch to GitLab
git push gitlab main

# Push all tags
git push gitlab --tags

# Set upstream for easier future pushes
git branch --set-upstream-to=gitlab/main main
```

## Setting Up Your GitHub Fork

### 1. Fork Original Repository

1. Go to: https://github.com/triton-inference-server/server
2. Click "Fork" button
3. Select your account
4. Wait for fork to complete

### 2. Add Your Fork as Remote

```bash
# Add your GitHub fork as remote
git remote add myfork https://github.com/YOUR_USERNAME/server.git

# Verify all remotes
git remote -v
# Should show:
# origin   https://github.com/triton-inference-server/server.git
# gitlab   git@gitlab.com:YOUR_USERNAME/triton-inference-server-apple-silicon.git
# myfork   https://github.com/YOUR_USERNAME/server.git
```

### 3. Create Feature Branch for PR

```bash
# Create feature branch for Apple Silicon optimizations
git checkout -b feature/apple-silicon-optimization

# Push to your fork
git push myfork feature/apple-silicon-optimization
```

## Pushing to Multiple Remotes

### Quick Push to All Remotes

```bash
# Push to GitLab
git push gitlab main

# Push to your GitHub fork
git push myfork feature/apple-silicon-optimization

# Or create a script to push to all:
cat > push-all.sh << 'EOF'
#!/bin/bash
echo "Pushing to GitLab..."
git push gitlab main
echo "Pushing to GitHub fork..."
git push myfork feature/apple-silicon-optimization
echo "Done!"
EOF

chmod +x push-all.sh
```

## Preparing Pull Request to Official Repo

### 1. Create Clean PR Branch

```bash
# Fetch latest from upstream
git fetch origin main

# Create clean branch from upstream
git checkout -b pr/apple-silicon-optimizations origin/main

# Cherry-pick your commits
git cherry-pick ca56352d  # Apple Silicon integration
git cherry-pick 337791b5  # Qwen3 optimization
git cherry-pick 115ef7ac  # Project reorganization
```

### 2. Prepare PR Description

```markdown
# Add Apple Silicon Optimization Support

## Summary
This PR adds comprehensive Apple Silicon optimization support to NVIDIA Triton Inference Server, achieving significant performance improvements on macOS ARM64 platforms.

## Key Features
- ✅ Apple Neural Engine (ANE) integration
- ✅ Metal Performance Shaders backend
- ✅ CoreML model conversion utilities
- ✅ Optimized memory management for Unified Memory Architecture
- ✅ Complete macOS build system compatibility

## Performance Results
- **Transformer inference**: 15.13x speedup with ANE
- **Qwen3-7B**: 26.6 tokens/second sustained throughput
- **Sub-3ms latency** for real-time applications

## Testing
- Comprehensive test suite with 150+ tests
- Benchmarking tools included
- Example scripts and demos provided

## Documentation
- Complete implementation guides
- Performance analysis reports
- Quick start guide for macOS users

Fixes #[issue_number]
```

### 3. Create Pull Request

1. Push to your fork:
   ```bash
   git push myfork pr/apple-silicon-optimizations
   ```

2. Go to: https://github.com/triton-inference-server/server
3. Click "Compare & pull request"
4. Select your branch
5. Add description and submit

## Sync Commands Reference

```bash
# Fetch updates from original repo
git fetch origin

# Merge upstream changes to your main
git checkout main
git merge origin/main

# Push updates to all remotes
git push gitlab main
git push myfork main

# Keep feature branch updated
git checkout feature/apple-silicon-optimization
git rebase main
```

## Important Notes

1. **GitLab**: Perfect for private development and CI/CD
2. **GitHub Fork**: Required for contributing back to official repo
3. **Feature Branches**: Use descriptive names for different features
4. **Commits**: Keep atomic and well-documented
5. **Testing**: Run full test suite before pushing

Ready to share your amazing Apple Silicon optimizations with the world! 🍎🚀