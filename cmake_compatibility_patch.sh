#!/bin/bash
# CMake 4.0.3 Compatibility Patch Script
# This script patches CMakeLists.txt files to be compatible with CMake 4.0.3

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

# Function to update cmake_minimum_required version
update_cmake_minimum_version() {
    local file="$1"
    local backup="${file}.bak"
    
    # Create backup
    cp "$file" "$backup"
    
    # Check current minimum version
    local current_version=$(grep -E "cmake_minimum_required.*VERSION.*[0-9]" "$file" | grep -oE "[0-9]+\.[0-9]+(\.[0-9]+)?" | head -1)
    
    if [[ -z "$current_version" ]]; then
        print_warning "No cmake_minimum_required found in $file"
        return
    fi
    
    # Parse version components
    local major=$(echo "$current_version" | cut -d. -f1)
    local minor=$(echo "$current_version" | cut -d. -f2)
    
    # Check if version is less than 3.5
    if [[ $major -lt 3 ]] || ([[ $major -eq 3 ]] && [[ $minor -lt 5 ]]); then
        print_info "Updating cmake_minimum_required from $current_version to 3.10 in $file"
        
        # Update the version to 3.10 (compatible with CMake 4.x)
        sed -i '' -E "s/cmake_minimum_required\s*\(\s*VERSION\s+[0-9]+\.[0-9]+(\.[0-9]+)?\s*/cmake_minimum_required(VERSION 3.10 /" "$file"
        
        # Add policy settings for compatibility
        if ! grep -q "cmake_policy" "$file"; then
            # Find the line after cmake_minimum_required
            local line_num=$(grep -n "cmake_minimum_required" "$file" | head -1 | cut -d: -f1)
            if [[ -n "$line_num" ]]; then
                # Insert policy settings after cmake_minimum_required
                sed -i '' "${line_num}a\\
\\
# CMake compatibility policies\\
if(POLICY CMP0054)\\
    cmake_policy(SET CMP0054 NEW)\\
endif()\\
if(POLICY CMP0057)\\
    cmake_policy(SET CMP0057 NEW)\\
endif()\\
if(POLICY CMP0074)\\
    cmake_policy(SET CMP0074 NEW)\\
endif()\\
if(POLICY CMP0091)\\
    cmake_policy(SET CMP0091 NEW)\\
endif()\\
" "$file"
            fi
        fi
        
        print_success "Updated $file"
    else
        print_info "Version $current_version in $file is already compatible"
        rm "$backup"
    fi
}

# Function to add modern CMake compatibility
add_cmake_compatibility() {
    local file="$1"
    
    # Check if we need to add CMAKE_CXX_STANDARD
    if ! grep -q "CMAKE_CXX_STANDARD" "$file" && grep -q "project\s*(" "$file"; then
        print_info "Adding C++ standard settings to $file"
        
        # Find project() line
        local line_num=$(grep -n "project\s*(" "$file" | head -1 | cut -d: -f1)
        if [[ -n "$line_num" ]]; then
            sed -i '' "${line_num}a\\
\\
# Set C++ standard\\
set(CMAKE_CXX_STANDARD 17)\\
set(CMAKE_CXX_STANDARD_REQUIRED ON)\\
set(CMAKE_CXX_EXTENSIONS OFF)\\
" "$file"
        fi
    fi
}

# Main execution
main() {
    print_info "Starting CMake 4.0.3 compatibility patch process"
    echo ""
    
    # Find all CMakeLists.txt files
    print_info "Finding all CMakeLists.txt files..."
    local cmake_files=()
    while IFS= read -r -d '' file; do
        cmake_files+=("$file")
    done < <(find "$PROJECT_ROOT" -name "CMakeLists.txt" -type f -print0)
    
    print_info "Found ${#cmake_files[@]} CMakeLists.txt files"
    echo ""
    
    # Process each file
    local updated_count=0
    for file in "${cmake_files[@]}"; do
        # Skip third-party and build directories
        if [[ "$file" =~ "third-party" ]] || [[ "$file" =~ "build/" ]] || [[ "$file" =~ "_deps/" ]]; then
            continue
        fi
        
        print_info "Processing: ${file#$PROJECT_ROOT/}"
        
        # Update cmake_minimum_required
        update_cmake_minimum_version "$file"
        
        # Add compatibility settings
        add_cmake_compatibility "$file"
        
        if [[ -f "${file}.bak" ]]; then
            ((updated_count++))
        fi
    done
    
    echo ""
    print_success "Patch process completed!"
    print_info "Updated $updated_count files"
    
    # Create a patch file for future reference
    if [[ $updated_count -gt 0 ]]; then
        print_info "Creating unified patch file..."
        local patch_file="${PROJECT_ROOT}/cmake_4.0.3_compatibility.patch"
        > "$patch_file"
        
        for file in "${cmake_files[@]}"; do
            if [[ -f "${file}.bak" ]]; then
                echo "--- ${file#$PROJECT_ROOT/}" >> "$patch_file"
                echo "+++ ${file#$PROJECT_ROOT/}" >> "$patch_file"
                diff -u "${file}.bak" "$file" >> "$patch_file" || true
                echo "" >> "$patch_file"
            fi
        done
        
        print_success "Created patch file: $patch_file"
    fi
    
    # Cleanup backup files
    print_info "Cleaning up backup files..."
    find "$PROJECT_ROOT" -name "*.bak" -type f -delete
    
    echo ""
    print_success "CMake 4.0.3 compatibility patches applied successfully!"
    echo ""
    echo "You can now run the build with:"
    echo "  ./build_macos.sh"
}

# Run main function
main "$@"