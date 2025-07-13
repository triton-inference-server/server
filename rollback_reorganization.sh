#!/bin/bash
# Emergency Rollback Script for Project Reorganization

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${RED}=== REORGANIZATION ROLLBACK ===${NC}\n"

# Check for backup tags
echo -e "${YELLOW}Available backup points:${NC}"
git tag -l "pre-reorganization-*" | sort -r | head -10

if [[ $(git tag -l "pre-reorganization-*" | wc -l) -eq 0 ]]; then
    echo -e "${RED}No reorganization backup tags found!${NC}"
    echo "Looking for recent commits..."
    git log --oneline -10
    exit 1
fi

# Get the most recent backup tag
LATEST_BACKUP=$(git tag -l "pre-reorganization-*" | sort -r | head -1)
echo -e "\nMost recent backup: ${GREEN}$LATEST_BACKUP${NC}"

# Show current status
echo -e "\n${YELLOW}Current status:${NC}"
git status --short

# Confirm rollback
echo -e "\n${RED}WARNING: This will discard all changes since the reorganization!${NC}"
echo -n "Rollback to $LATEST_BACKUP? (yes/no): "
read response

if [[ "$response" != "yes" ]]; then
    echo "Rollback cancelled."
    exit 0
fi

# Perform rollback
echo -e "\n${YELLOW}Rolling back...${NC}"

# Stash any uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Stashing uncommitted changes..."
    git stash push -m "Before reorganization rollback"
fi

# Reset to backup point
git reset --hard "$LATEST_BACKUP"

echo -e "\n${GREEN}Rollback complete!${NC}"
echo "Repository restored to state before reorganization."

# Show restored structure
echo -e "\n${YELLOW}Restored structure:${NC}"
ls -la | head -20

# Cleanup tags
echo -n -e "\n${YELLOW}Remove reorganization tags? (y/n): ${NC}"
read response
if [[ "$response" == "y" || "$response" == "Y" ]]; then
    git tag -l "pre-reorganization-*" | xargs -n1 git tag -d
    git tag -l "reorganization-complete-*" | xargs -n1 git tag -d
    echo "Reorganization tags removed."
fi

echo -e "\n${GREEN}Done!${NC} Project structure has been restored."