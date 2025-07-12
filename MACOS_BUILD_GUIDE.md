# Triton Inference Server macOS Build Guide

This guide provides detailed instructions for building Triton Inference Server on macOS using the automated build script.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Build Script Usage](#build-script-usage)
- [Build Options](#build-options)
- [Common Build Scenarios](#common-build-scenarios)
- [Troubleshooting](#troubleshooting)
- [Post-Build Steps](#post-build-steps)
- [Developer Notes](#developer-notes)

## Prerequisites

### System Requirements
- **macOS Version**: 11.0 (Big Sur) or later
- **Architecture**: Intel x86_64 or Apple Silicon (M1/M2/M3)
- **Disk Space**: At least 10GB free space
- **Memory**: 8GB RAM minimum, 16GB recommended

### Required Software
The build script will automatically check for and install these dependencies:
- Xcode Command Line Tools
- Homebrew package manager
- CMake 3.18 or later
- Various development libraries

## Quick Start

For a basic build with default settings:

```bash
# Clone the repository (if not already done)
git clone https://github.com/triton-inference-server/server.git
cd server

# Run the build script
./build_macos.sh
```

This will:
1. Check for required dependencies
2. Install missing dependencies via Homebrew
3. Configure CMake for a Release build
4. Build Triton with CPU-only support
5. Install to `/usr/local`

## Build Script Usage

### Basic Syntax
```bash
./build_macos.sh [options]
```

### Display Help
```bash
./build_macos.sh --help
```

## Build Options

### Build Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--build-type=<Debug\|Release>` | Set build type | Release |
| `--install-prefix=<path>` | Installation directory | /usr/local |
| `--clean` | Remove build directory before building | No |
| `--verbose` | Enable verbose build output | No |
| `--parallel=<n>` | Number of parallel build jobs | Auto-detect |
| `--ccache` | Use ccache for faster rebuilds | No |

### Feature Flags

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-http` / `--disable-http` | HTTP/REST endpoint support | Enabled |
| `--enable-grpc` / `--disable-grpc` | gRPC endpoint support | Enabled |
| `--enable-metrics` / `--disable-metrics` | Metrics collection | Enabled |
| `--enable-logging` / `--disable-logging` | Logging support | Enabled |
| `--enable-stats` / `--disable-stats` | Statistics collection | Enabled |
| `--enable-tracing` / `--disable-tracing` | OpenTelemetry tracing | Disabled |
| `--enable-ensemble` / `--disable-ensemble` | Ensemble model support | Disabled |
| `--enable-s3` / `--disable-s3` | AWS S3 model repository | Disabled |
| `--enable-gcs` / `--disable-gcs` | Google Cloud Storage | Disabled |
| `--enable-azure` / `--disable-azure` | Azure Storage | Disabled |

### Other Options

| Option | Description |
|--------|-------------|
| `--skip-deps` | Skip dependency installation |
| `--run-tests` | Run tests after building |

## Common Build Scenarios

### Debug Build
For development with debug symbols:
```bash
./build_macos.sh --build-type=Debug --verbose
```

### Clean Rebuild
Force a complete rebuild:
```bash
./build_macos.sh --clean
```

### Custom Installation
Install to a custom location:
```bash
./build_macos.sh --install-prefix=/opt/triton
```

### Cloud Storage Support
Enable cloud storage backends:
```bash
./build_macos.sh --enable-s3 --enable-gcs --enable-azure
```

### Minimal Build
Disable optional features for a smaller build:
```bash
./build_macos.sh --disable-metrics --disable-stats --disable-ensemble
```

### Fast Rebuilds with ccache
Speed up incremental builds:
```bash
# First install ccache
brew install ccache

# Use ccache in builds
./build_macos.sh --ccache
```

### Parallel Build Control
Limit parallel jobs (useful for systems with limited memory):
```bash
./build_macos.sh --parallel=4
```

## Troubleshooting

### Common Issues

#### 1. Xcode Command Line Tools Not Found
**Error**: "Xcode Command Line Tools not found"

**Solution**:
```bash
xcode-select --install
```

#### 2. Homebrew Not Found
**Error**: "Homebrew not found"

**Solution**: The script will offer to install Homebrew automatically. Alternatively:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 3. CMake Version Too Old
**Error**: "CMake version X.Y.Z is too old"

**Solution**:
```bash
brew upgrade cmake
```

#### 4. Permission Denied During Installation
**Error**: "Permission denied" when installing to system directories

**Solution**: The script will automatically use `sudo` when needed, or use a different prefix:
```bash
./build_macos.sh --install-prefix=$HOME/triton
```

#### 5. Build Fails with Missing Dependencies
**Error**: "Could not find package X"

**Solution**: Make sure not to skip dependencies:
```bash
./build_macos.sh  # Don't use --skip-deps
```

#### 6. Apple Silicon Specific Issues
On M1/M2/M3 Macs, ensure Homebrew is in your PATH:
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### Build Artifacts

Build artifacts are located in:
- **Build Directory**: `./build/`
- **Executables**: `./build/src/tritonserver`
- **Libraries**: `./build/src/*.dylib`

### Cleaning Up

To remove build artifacts:
```bash
rm -rf build/
```

To remove all installed dependencies (use with caution):
```bash
brew list | grep -E "(protobuf|grpc|libevent|rapidjson|boost|re2)" | xargs brew uninstall
```

## Post-Build Steps

### 1. Verify Installation
```bash
# Check if tritonserver is in PATH
which tritonserver

# Or run directly
/usr/local/bin/tritonserver --help
```

### 2. Set Up Model Repository
```bash
# Create a model repository
mkdir -p ~/models

# Add your models to the repository
# See Triton documentation for model layout
```

### 3. Run Triton Server
```bash
tritonserver --model-repository=$HOME/models
```

### 4. Test with Client
```bash
# Install Python client
pip install tritonclient[all]

# Test with a simple client script
python3 -c "import tritonclient.http; print('Client installed successfully')"
```

## Developer Notes

### Build System Details

The build script:
1. **Detects System Configuration**
   - macOS version checking
   - Architecture detection (Intel vs Apple Silicon)
   - Automatic Homebrew path configuration

2. **Manages Dependencies**
   - Checks for Xcode Command Line Tools
   - Installs required packages via Homebrew
   - Configures compiler and linker paths

3. **Configures CMake**
   - Sets appropriate macOS flags
   - Disables unsupported features (CUDA, TensorRT)
   - Configures for CPU-only operation

4. **Handles Platform Differences**
   - Uses `clang/clang++` compilers
   - Sets up proper RPATH for dynamic libraries
   - Configures for `.dylib` instead of `.so`

### Environment Variables

The script sets these environment variables during build:
```bash
CC=clang
CXX=clang++
LDFLAGS="-L/opt/homebrew/lib"  # Or /usr/local/lib on Intel
CPPFLAGS="-I/opt/homebrew/include"  # Or /usr/local/include on Intel
```

### CMake Cache

To inspect CMake configuration:
```bash
# After running the build script
cd build
ccmake .  # Interactive CMake cache editor
```

### Adding Custom CMake Options

To pass additional CMake options, modify the script or run CMake directly:
```bash
cd build
cmake -DCUSTOM_OPTION=value ..
make -j$(sysctl -n hw.ncpu)
```

### Integration with IDEs

#### Xcode
Generate Xcode project:
```bash
cd build
cmake -G Xcode ..
open tritonserver.xcodeproj
```

#### CLion / VSCode
The script generates `compile_commands.json` for IDE integration:
```bash
# After building, link for IDE
ln -s build/compile_commands.json .
```

### Contributing

When contributing changes:
1. Test on both Intel and Apple Silicon if possible
2. Ensure the build script remains compatible with both architectures
3. Update this documentation for any new options or requirements
4. Follow the existing script style and conventions

## Additional Resources

- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [macOS Porting Notes](./MACOS_BUILD_CHANGES.md)
- [Dependency Mapping](./DEPENDENCIES_MACOS.md)
- [Apple Silicon Adaptation Strategy](./APPLE_SILICON_ADAPTATION_STRATEGY.md)

## Support

For issues specific to the macOS build:
1. Check the troubleshooting section above
2. Review build logs in the `build` directory
3. Open an issue with:
   - macOS version
   - Architecture (Intel/Apple Silicon)
   - Complete error message
   - Output of `./build_macos.sh --verbose`