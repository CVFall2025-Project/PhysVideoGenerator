#!/bin/bash
# Commit comparison experiments changes on HPC
# This script should be run on HPC after syncing files

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/hpc_config.sh"

REPO_PATH="/scratch/hc4569/repos/PhysVideoGenerator"

echo "=========================================="
echo "Committing comparison experiments changes"
echo "=========================================="
echo "Repository path: ${REPO_PATH}"
echo "=========================================="
echo ""

cd "$REPO_PATH" || exit 1

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "comparison-experiments" ]; then
    echo "⚠️  Current branch is '$CURRENT_BRANCH', not 'comparison-experiments'"
    echo "Switching to comparison-experiments branch..."
    git checkout comparison-experiments 2>/dev/null || {
        echo "Branch comparison-experiments does not exist. Creating it from ablation-experiment..."
        git checkout ablation-experiment
        git pull origin ablation-experiment
        git checkout -b comparison-experiments
    }
fi

echo "Current branch: $(git branch --show-current)"
echo ""

# Check repository status
echo "=========================================="
echo "Repository status"
echo "=========================================="
git status --short
echo ""

# Check if there are changes to commit
if [ -z "$(git status --porcelain)" ]; then
    echo "⚠️  No changes to commit. Repository is clean."
    echo "Make sure you have synced the files to HPC first."
    exit 0
fi

# Show what will be committed
echo "=========================================="
echo "Files to be committed"
echo "=========================================="
git status --short
echo ""

# Stage all changes
echo "Staging all changes..."
git add .

# Show staged changes
echo ""
echo "=========================================="
echo "Staged changes"
echo "=========================================="
git status --short
echo ""

# Commit with message
COMMIT_MESSAGE="Add evaluation scripts for comparison experiments (OpenSora, VideoCrafter2, HunyuanVideo)

- Add evaluation_utils.py with common metrics (PSNR, SSIM, FVD, CLIP score)
- Add opensora_eval.py for OpenSora model evaluation (using official API)
- Add videocrafter2_eval.py for VideoCrafter2 model evaluation (using official API patterns)
- Add hunyuanvideo_eval.py for HunyuanVideo model evaluation (using official API patterns)
- Add run_comparison.py for unified comparison runner
- Add requirements_comparison.txt with dependencies
- Add setup_comparison_experiments.sh for HPC setup
- Add commit_comparison_experiments.sh for HPC commit
- Add evaluation/README.md and API_UPDATE_NOTES.md with usage documentation
- Update all scripts to use official APIs instead of placeholders"

echo "=========================================="
echo "Committing changes"
echo "=========================================="
echo "Commit message:"
echo "$COMMIT_MESSAGE"
echo ""

git commit -m "$COMMIT_MESSAGE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Changes committed successfully!"
    echo ""
    echo "Latest commit:"
    git log -1 --oneline
    echo ""
    
    # Ask if user wants to push
    echo "=========================================="
    echo "Push to remote?"
    echo "=========================================="
    read -p "Push to remote? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Pushing to remote..."
        git push -u origin comparison-experiments
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Changes pushed to remote successfully!"
        else
            echo ""
            echo "⚠️  Push failed. You may need to:"
            echo "   1. Set up SSH keys on HPC"
            echo "   2. Or manually run: git push -u origin comparison-experiments"
        fi
    else
        echo ""
        echo "Skipping push. To push later, run:"
        echo "  git push -u origin comparison-experiments"
    fi
else
    echo ""
    echo "❌ Commit failed. Please check the error messages above."
    exit 1
fi

