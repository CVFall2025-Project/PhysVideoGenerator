#!/bin/bash
# Script to run on HPC for committing and pushing comparison experiments
# Usage: After connecting to HPC, run: bash commit_and_push_on_hpc.sh

REPO_PATH="/scratch/hc4569/repos/PhysVideoGenerator"

echo "=========================================="
echo "Commit and Push Comparison Experiments"
echo "=========================================="
echo "Repository path: ${REPO_PATH}"
echo "=========================================="
echo ""

cd "$REPO_PATH" || {
    echo "❌ Error: Cannot access repository at ${REPO_PATH}"
    echo "Please check the path and ensure you have access."
    exit 1
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

# Fetch latest changes
echo "Fetching latest changes from remote..."
git fetch origin
echo ""

# Switch to comparison-experiments branch or create it
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: ${CURRENT_BRANCH}"

if [ "$CURRENT_BRANCH" != "comparison-experiments" ]; then
    echo "Switching to comparison-experiments branch..."
    if git show-ref --verify --quiet refs/heads/comparison-experiments; then
        git checkout comparison-experiments
        git pull origin comparison-experiments 2>/dev/null || true
    else
        # Create branch from ablation-experiment if it exists, otherwise from main
        if git show-ref --verify --quiet refs/heads/ablation-experiment; then
            git checkout ablation-experiment
            git pull origin ablation-experiment 2>/dev/null || true
            git checkout -b comparison-experiments
        else
            git checkout main
            git pull origin main 2>/dev/null || true
            git checkout -b comparison-experiments
        fi
    fi
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
    echo ""
    echo "If you need to add files, make sure they are in the repository directory."
    echo "Files should be in:"
    echo "  - evaluation/"
    echo "  - evaluation_utils.py"
    echo "  - requirements_comparison.txt"
    echo "  - setup_comparison_experiments.sh"
    echo "  - commit_comparison_experiments.sh"
    echo "  - COMPARISON_EXPERIMENTS_SETUP.md"
    echo ""
    read -p "Do you want to check for files to add? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Add evaluation files
echo "=========================================="
echo "Adding files to staging area"
echo "=========================================="

# Add files if they exist
FILES_ADDED=0

if [ -d "evaluation" ]; then
    git add evaluation/
    echo "✅ Added evaluation/ directory"
    FILES_ADDED=1
fi

if [ -f "evaluation_utils.py" ]; then
    git add evaluation_utils.py
    echo "✅ Added evaluation_utils.py"
    FILES_ADDED=1
fi

if [ -f "requirements_comparison.txt" ]; then
    git add requirements_comparison.txt
    echo "✅ Added requirements_comparison.txt"
    FILES_ADDED=1
fi

if [ -f "setup_comparison_experiments.sh" ]; then
    git add setup_comparison_experiments.sh
    echo "✅ Added setup_comparison_experiments.sh"
    FILES_ADDED=1
fi

if [ -f "commit_comparison_experiments.sh" ]; then
    git add commit_comparison_experiments.sh
    echo "✅ Added commit_comparison_experiments.sh"
    FILES_ADDED=1
fi

if [ -f "COMPARISON_EXPERIMENTS_SETUP.md" ]; then
    git add COMPARISON_EXPERIMENTS_SETUP.md
    echo "✅ Added COMPARISON_EXPERIMENTS_SETUP.md"
    FILES_ADDED=1
fi

if [ -f "COMMIT_INSTRUCTIONS.md" ]; then
    git add COMMIT_INSTRUCTIONS.md
    echo "✅ Added COMMIT_INSTRUCTIONS.md"
    FILES_ADDED=1
fi

if [ $FILES_ADDED -eq 0 ]; then
    echo "⚠️  No files found to add. Please ensure files are synced to HPC first."
    echo ""
    echo "To sync files from local machine:"
    echo "  cd /path/to/local/Project"
    echo "  ./sync_to_hpc.sh"
    exit 1
fi

echo ""

# Show staged changes
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
    
    # Push to remote
    echo "=========================================="
    echo "Pushing to remote"
    echo "=========================================="
    echo "Pushing comparison-experiments branch to origin..."
    
    git push -u origin comparison-experiments
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Changes pushed to remote successfully!"
        echo ""
        echo "You can view the changes at:"
        echo "  https://github.com/CVFall2025-Project/PhysVideoGenerator/tree/comparison-experiments"
    else
        echo ""
        echo "❌ Push failed. Possible reasons:"
        echo "  1. SSH keys not configured on HPC"
        echo "  2. No write access to repository"
        echo "  3. Network issues"
        echo ""
        echo "To push manually later:"
        echo "  git push -u origin comparison-experiments"
        echo ""
        echo "Or use HTTPS with token:"
        echo "  git remote set-url origin https://YOUR_TOKEN@github.com/CVFall2025-Project/PhysVideoGenerator.git"
        echo "  git push -u origin comparison-experiments"
    fi
else
    echo ""
    echo "❌ Commit failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="


