#!/bin/bash
# Setup Git Remotes for Triton Apple Silicon Project

echo "🚀 Setting up Git remotes for your Apple Silicon optimizations"
echo "============================================================"

# Function to prompt for input
prompt_for_input() {
    local prompt_text="$1"
    local var_name="$2"
    echo -n "$prompt_text"
    read -r "$var_name"
}

# Get user information
echo -e "\n📝 Please provide your repository information:\n"

prompt_for_input "Your GitLab username: " GITLAB_USERNAME
prompt_for_input "Your GitLab instance URL (e.g., gitlab.com or your company GitLab): " GITLAB_INSTANCE
prompt_for_input "Your GitHub username: " GITHUB_USERNAME
prompt_for_input "Use SSH (recommended) or HTTPS? [ssh/https]: " PROTOCOL

# Construct remote URLs
if [[ "$PROTOCOL" == "ssh" ]]; then
    GITLAB_URL="git@${GITLAB_INSTANCE}:${GITLAB_USERNAME}/triton-inference-server-apple-silicon.git"
    GITHUB_FORK_URL="git@github.com:${GITHUB_USERNAME}/server.git"
else
    GITLAB_URL="https://${GITLAB_INSTANCE}/${GITLAB_USERNAME}/triton-inference-server-apple-silicon.git"
    GITHUB_FORK_URL="https://github.com/${GITHUB_USERNAME}/server.git"
fi

echo -e "\n🔧 Configuring remotes...\n"

# Add GitLab remote
echo "Adding GitLab remote..."
git remote add gitlab "$GITLAB_URL" 2>/dev/null || {
    echo "GitLab remote already exists. Updating URL..."
    git remote set-url gitlab "$GITLAB_URL"
}

# Add GitHub fork remote
echo "Adding GitHub fork remote..."
git remote add myfork "$GITHUB_FORK_URL" 2>/dev/null || {
    echo "GitHub fork remote already exists. Updating URL..."
    git remote set-url myfork "$GITHUB_FORK_URL"
}

# Display configured remotes
echo -e "\n✅ Configured remotes:"
git remote -v

# Create feature branch
echo -e "\n🌿 Creating feature branch for Apple Silicon optimizations..."
git checkout -b feature/apple-silicon-optimization 2>/dev/null || {
    echo "Branch already exists. Switching to it..."
    git checkout feature/apple-silicon-optimization
}

echo -e "\n📋 Next steps:\n"
echo "1. Create GitLab project:"
echo "   - Go to https://${GITLAB_INSTANCE}"
echo "   - Create new project: 'triton-inference-server-apple-silicon'"
echo "   - Set visibility as desired"
echo ""
echo "2. Fork GitHub repository:"
echo "   - Go to https://github.com/triton-inference-server/server"
echo "   - Click 'Fork' button"
echo "   - Select your account"
echo ""
echo "3. Push to your repositories:"
echo "   git push gitlab main                    # Push to GitLab"
echo "   git push myfork feature/apple-silicon-optimization  # Push to GitHub fork"
echo ""
echo "4. Create Pull Request:"
echo "   - Go to your GitHub fork"
echo "   - Click 'Compare & pull request'"
echo "   - Submit PR to official repository"

# Create push-all convenience script
cat > push-all.sh << 'EOF'
#!/bin/bash
# Push to all remotes

echo "🚀 Pushing to all remotes..."
echo "=========================="

# Push to GitLab
echo -e "\n📤 Pushing to GitLab..."
git push gitlab main --tags

# Push to GitHub fork  
echo -e "\n📤 Pushing to GitHub fork..."
git push myfork feature/apple-silicon-optimization

# Push to origin (optional, only if you have write access)
# echo -e "\n📤 Pushing to origin..."
# git push origin main

echo -e "\n✅ All pushes complete!"
EOF

chmod +x push-all.sh

echo -e "\n✅ Setup complete! Created 'push-all.sh' for convenience."