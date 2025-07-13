#!/bin/bash

# Fix libevent cmake files that incorrectly include /opt/homebrew/include
# This causes protobuf header conflicts on macOS

echo "Fixing libevent cmake files to remove /opt/homebrew/include..."

# Find all LibeventTargets*.cmake files and remove /opt/homebrew/include from INTERFACE_INCLUDE_DIRECTORIES
find build -name "LibeventTargets*.cmake" -type f | while read -r file; do
    echo "Fixing: $file"
    # Create backup
    cp "$file" "$file.bak"
    
    # Remove ;/opt/homebrew/include from INTERFACE_INCLUDE_DIRECTORIES lines
    sed -i '' 's|;/opt/homebrew/include"$|"|g' "$file"
    
    # Also handle cases where it might be the only include directory
    sed -i '' 's|INTERFACE_INCLUDE_DIRECTORIES "/opt/homebrew/include"|INTERFACE_INCLUDE_DIRECTORIES ""|g' "$file"
done

echo "Done! Libevent cmake files have been fixed."
echo "You can now run 'make' again to continue the build."