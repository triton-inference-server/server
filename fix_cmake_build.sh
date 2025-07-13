#!/bin/bash
# Comprehensive CMake 4.0.3 Build Fix Script
# This script handles all CMake compatibility issues for building Triton on macOS with CMake 4.0.3

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Function to check CMake version
check_cmake_version() {
    print_info "Checking CMake version..."
    
    if ! command -v cmake &> /dev/null; then
        print_error "CMake not found!"
        exit 1
    fi
    
    CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")
    print_info "CMake version: $CMAKE_VERSION"
    
    # Check if it's version 4.x
    CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
    if [[ $CMAKE_MAJOR -eq 4 ]]; then
        print_warning "CMake 4.x detected. Applying compatibility fixes..."
        return 0
    else
        print_info "CMake version is not 4.x. No special handling needed."
        return 1
    fi
}

# Function to apply libevent patch
apply_libevent_patch() {
    print_info "Checking for libevent patch..."
    
    if [[ -f "${PROJECT_ROOT}/libevent_cmake_fix.patch" ]]; then
        print_info "Found libevent patch. This will be applied during build."
    else
        print_warning "libevent patch not found. Creating one..."
        cat > "${PROJECT_ROOT}/libevent_cmake_fix.patch" << 'EOF'
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -19,7 +19,7 @@
 #       start libevent.sln
 #
 
-cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
+cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
 
 if (POLICY CMP0054)
     cmake_policy(SET CMP0054 NEW)
EOF
    fi
}

# Function to create CMake wrapper
create_cmake_wrapper() {
    print_info "Creating CMake wrapper for dependency builds..."
    
    local wrapper_file="${PROJECT_ROOT}/cmake_wrapper.sh"
    cat > "$wrapper_file" << 'EOF'
#!/bin/bash
# CMake wrapper to handle version compatibility

# Get the real cmake path
REAL_CMAKE=$(which cmake)

# Check if we're building a dependency that needs patching
if [[ "$*" =~ "libevent" ]] || [[ "$PWD" =~ "libevent" ]]; then
    # For libevent, we need to patch CMakeLists.txt first
    if [[ -f "CMakeLists.txt" ]] && grep -q "cmake_minimum_required.*VERSION.*3\.[0-4]" CMakeLists.txt; then
        echo "Patching libevent CMakeLists.txt for CMake 4.0.3 compatibility..."
        sed -i.bak 's/cmake_minimum_required.*VERSION.*3\.[0-4]/cmake_minimum_required(VERSION 3.10/' CMakeLists.txt
    fi
fi

# Execute the real cmake
exec "$REAL_CMAKE" "$@"
EOF
    
    chmod +x "$wrapper_file"
    print_success "Created CMake wrapper: $wrapper_file"
}

# Function to set up build environment
setup_build_environment() {
    print_info "Setting up build environment for CMake 4.0.3..."
    
    # Export environment variables for the build
    export CMAKE_POLICY_DEFAULT_CMP0054=NEW
    export CMAKE_POLICY_DEFAULT_CMP0057=NEW
    export CMAKE_POLICY_DEFAULT_CMP0074=NEW
    export CMAKE_POLICY_DEFAULT_CMP0091=NEW
    
    # Use our wrapper if needed
    if [[ -f "${PROJECT_ROOT}/cmake_wrapper.sh" ]]; then
        export CMAKE_COMMAND="${PROJECT_ROOT}/cmake_wrapper.sh"
    fi
    
    print_success "Build environment configured"
}

# Function to patch build script
patch_build_script() {
    print_info "Patching build_macos.sh for CMake 4.0.3 compatibility..."
    
    local build_script="${PROJECT_ROOT}/build_macos.sh"
    local patched_script="${PROJECT_ROOT}/build_macos_patched.sh"
    
    if [[ ! -f "$build_script" ]]; then
        print_error "build_macos.sh not found!"
        return 1
    fi
    
    # Copy the original script
    cp "$build_script" "$patched_script"
    
    # Add our compatibility fixes after the environment setup
    sed -i '' '/setup_environment() {/,/^}$/ {
        /print_success "Environment configured"/i\
    \
    # CMake 4.0.3 compatibility\
    if cmake --version | grep -q "cmake version 4"; then\
        print_info "Applying CMake 4.0.3 compatibility settings..."\
        export CMAKE_POLICY_DEFAULT_CMP0054=NEW\
        export CMAKE_POLICY_DEFAULT_CMP0057=NEW\
        export CMAKE_POLICY_DEFAULT_CMP0074=NEW\
        export CMAKE_POLICY_DEFAULT_CMP0091=NEW\
    fi
    }' "$patched_script"
    
    # Add patch application in configure_cmake function
    sed -i '' '/configure_cmake() {/,/^}$/ {
        /local CMAKE_ARGS=/i\
    \
    # Apply compatibility patches if needed\
    if [[ -f "${PROJECT_ROOT}/cmake_compatibility_patch.sh" ]]; then\
        print_info "Applying CMake compatibility patches..."\
        "${PROJECT_ROOT}/cmake_compatibility_patch.sh"\
    fi
    }' "$patched_script"
    
    chmod +x "$patched_script"
    print_success "Created patched build script: $patched_script"
}

# Function to create pre-build hook
create_prebuild_hook() {
    print_info "Creating pre-build hook..."
    
    local hook_file="${PROJECT_ROOT}/prebuild_cmake_fix.sh"
    cat > "$hook_file" << 'EOF'
#!/bin/bash
# Pre-build hook to fix CMake issues

# Fix any downloaded dependencies
if [[ -d "build/_deps" ]]; then
    echo "Fixing CMake files in downloaded dependencies..."
    find build/_deps -name "CMakeLists.txt" -type f | while read -r file; do
        if grep -q "cmake_minimum_required.*VERSION.*3\.[0-4]" "$file"; then
            echo "Patching: $file"
            sed -i.bak 's/cmake_minimum_required.*VERSION.*3\.[0-4]/cmake_minimum_required(VERSION 3.10/' "$file"
        fi
    done
fi

# Fix any third-party dependencies
if [[ -d "third-party" ]]; then
    echo "Fixing CMake files in third-party..."
    find third-party -name "CMakeLists.txt" -type f | while read -r file; do
        if grep -q "cmake_minimum_required.*VERSION.*3\.[0-4]" "$file"; then
            echo "Patching: $file"
            sed -i.bak 's/cmake_minimum_required.*VERSION.*3\.[0-4]/cmake_minimum_required(VERSION 3.10/' "$file"
        fi
    done
fi
EOF
    
    chmod +x "$hook_file"
    print_success "Created pre-build hook: $hook_file"
}

# Main execution
main() {
    print_info "CMake 4.0.3 Build Fix Script"
    echo "============================"
    echo ""
    
    # Check if we need to apply fixes
    if ! check_cmake_version; then
        print_info "CMake 4.x not detected. Exiting."
        exit 0
    fi
    
    # Apply various fixes
    print_info "Applying compatibility fixes..."
    
    # 1. Run the compatibility patch script
    if [[ -f "${PROJECT_ROOT}/cmake_compatibility_patch.sh" ]]; then
        "${PROJECT_ROOT}/cmake_compatibility_patch.sh"
    else
        print_warning "cmake_compatibility_patch.sh not found"
    fi
    
    # 2. Apply libevent patch
    apply_libevent_patch
    
    # 3. Create CMake wrapper
    create_cmake_wrapper
    
    # 4. Patch build script
    patch_build_script
    
    # 5. Create pre-build hook
    create_prebuild_hook
    
    # 6. Set up environment
    setup_build_environment
    
    echo ""
    print_success "All CMake 4.0.3 compatibility fixes applied!"
    echo ""
    echo "Next steps:"
    echo "1. Run the patched build script:"
    echo "   ./build_macos_patched.sh"
    echo ""
    echo "2. Or run the original build script with fixes:"
    echo "   ./prebuild_cmake_fix.sh && ./build_macos.sh"
    echo ""
    echo "3. If you encounter issues during build, run:"
    echo "   ./prebuild_cmake_fix.sh"
    echo "   Then retry the build"
    echo ""
    
    # Offer to run the build immediately
    read -p "Would you like to start the build now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Starting build with patched script..."
        "${PROJECT_ROOT}/build_macos_patched.sh"
    fi
}

# Run main function
main "$@"