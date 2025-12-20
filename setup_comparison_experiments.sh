#!/bin/bash
# Setup script for comparison experiments on HPC
# This script should be run on HPC after syncing

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hpc_config.sh"

REPO_PATH="/scratch/hc4569/repos/PhysVideoGenerator"

echo "=========================================="
echo "Setting up comparison experiments"
echo "=========================================="
echo "Repository path: ${REPO_PATH}"
echo "=========================================="
echo ""

cd "$REPO_PATH" || exit 1

# 1. Check repository status
echo "=========================================="
echo "1. Checking repository status"
echo "=========================================="
git status --short
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Working directory is not clean. Please commit or stash changes first."
    exit 1
fi
echo "✅ Repository is clean"
echo ""

# 2. Fetch latest changes
echo "=========================================="
echo "2. Fetching latest changes"
echo "=========================================="
git fetch origin
echo "✅ Latest changes fetched"
echo ""

# 3. Switch to ablation-experiment branch
echo "=========================================="
echo "3. Switching to ablation-experiment branch"
echo "=========================================="
git checkout ablation-experiment 2>/dev/null || {
    echo "Branch ablation-experiment not found locally, checking out from remote..."
    git checkout -b ablation-experiment origin/ablation-experiment || {
        echo "❌ Failed to checkout ablation-experiment branch"
        exit 1
    }
}
git pull origin ablation-experiment
echo "✅ Switched to ablation-experiment branch"
echo "Current branch: $(git branch --show-current)"
echo "Latest commit: $(git log -1 --oneline)"
echo ""

# 4. Display repository structure
echo "=========================================="
echo "4. Repository structure"
echo "=========================================="
echo "Main directories:"
ls -d */ 2>/dev/null | head -20
echo ""
echo "Python files:"
find . -name "*.py" -type f | head -20
echo ""

# 5. Create comparison-experiments branch
echo "=========================================="
echo "5. Creating comparison-experiments branch"
echo "=========================================="
if git show-ref --verify --quiet refs/heads/comparison-experiments; then
    echo "Branch comparison-experiments already exists, switching to it..."
    git checkout comparison-experiments
    git merge ablation-experiment --no-edit
else
    git checkout -b comparison-experiments
fi
echo "✅ Created/switched to comparison-experiments branch"
echo ""

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "Repository: ${REPO_PATH}"
echo "Current branch: $(git branch --show-current)"
echo ""
echo "Next steps:"
echo "1. Review the repository structure"
echo "2. Add comparison model evaluation scripts"
echo "3. Commit changes"
echo ""


