# macOS Build Quick Reference

## 🚀 Quick Commands

### First Time Setup
```bash
# Run complete build with all dependencies
./build_macos.sh
```

### Common Build Commands
```bash
# Debug build
./build_macos.sh --build-type=Debug

# Clean rebuild
./build_macos.sh --clean

# Verbose output
./build_macos.sh --verbose

# Custom install location
./build_macos.sh --install-prefix=$HOME/triton

# Fast rebuild with ccache
./build_macos.sh --ccache

# Build with tests
./build_macos.sh --run-tests
```

### Feature Control
```bash
# Enable cloud storage
./build_macos.sh --enable-s3 --enable-gcs

# Minimal build
./build_macos.sh --disable-metrics --disable-ensemble

# Full-featured build
./build_macos.sh --enable-ensemble --enable-tracing --enable-s3
```

## 📋 Prerequisites Checklist

- [ ] macOS 11.0 (Big Sur) or later
- [ ] Xcode Command Line Tools (`xcode-select --install`)
- [ ] Homebrew (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- [ ] 10GB free disk space
- [ ] 8GB RAM (16GB recommended)

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Xcode Command Line Tools not found" | Run `xcode-select --install` |
| "cmake: command not found" | Run `brew install cmake` |
| "Permission denied" during install | Use `--install-prefix=$HOME/triton` |
| Build fails on M1/M2/M3 | Ensure Homebrew is in PATH: `eval "$(/opt/homebrew/bin/brew shellenv)"` |
| "Package not found" errors | Don't use `--skip-deps` flag |

## 🏗️ Build Options Summary

| Option | Description | Default |
|--------|-------------|---------|
| `--build-type=<Debug\|Release>` | Build configuration | Release |
| `--install-prefix=<path>` | Install location | /usr/local |
| `--clean` | Clean before build | No |
| `--verbose` | Verbose output | No |
| `--parallel=<n>` | Build jobs | Auto |
| `--ccache` | Use ccache | No |
| `--skip-deps` | Skip dependencies | No |
| `--run-tests` | Run tests | No |

## ✅ Post-Build Verification

```bash
# Check installation
which tritonserver
tritonserver --help

# Run server (requires model repository)
tritonserver --model-repository=/path/to/models

# Check version
tritonserver --version
```

## 📁 Important Paths

- **Build Script**: `./build_macos.sh`
- **Build Directory**: `./build/`
- **CMake Module**: `./cmake/MacOS.cmake`
- **Default Install**: `/usr/local/bin/tritonserver`
- **Logs**: `./build/CMakeFiles/CMakeOutput.log`

## 🔗 Related Documentation

- [Detailed Build Guide](./MACOS_BUILD_GUIDE.md)
- [Dependencies List](./DEPENDENCIES_MACOS.md)
- [macOS Porting Changes](./MACOS_BUILD_CHANGES.md)

## 💡 Tips

1. **First build is slow** - Dependencies download and compilation takes time
2. **Use ccache** - Speeds up rebuilds significantly
3. **Debug builds** - Easier to troubleshoot but slower
4. **Parallel builds** - Default uses all cores; reduce if system becomes unresponsive
5. **Apple Silicon** - Homebrew installs to `/opt/homebrew` instead of `/usr/local`