# Comparison Experiments Setup - Summary

This document summarizes the comparison experiments setup for PhysVideoGenerator.

## Files Created

### Evaluation Scripts
1. **evaluation_utils.py** - Common utilities for video evaluation metrics
   - PSNR, SSIM, FVD, CLIP score computation
   - Test prompts loading
   - Results saving utilities

2. **evaluation/opensora_eval.py** - OpenSora model evaluation script
3. **evaluation/videocrafter2_eval.py** - VideoCrafter2 model evaluation script
4. **evaluation/hunyuanvideo_eval.py** - HunyuanVideo model evaluation script
5. **evaluation/run_comparison.py** - Unified comparison runner
6. **evaluation/__init__.py** - Module initialization
7. **evaluation/README.md** - Usage documentation

### Setup Scripts
1. **setup_comparison_experiments.sh** - HPC setup script
   - Connects to repository
   - Switches to ablation-experiment branch
   - Creates comparison-experiments branch

2. **commit_comparison_experiments.sh** - HPC commit script
   - Commits all changes with proper message
   - Can be run on HPC after syncing files

### Configuration
1. **requirements_comparison.txt** - Additional dependencies for comparison models

## Usage Instructions

### Step 1: Sync Files to HPC
```bash
cd Project
./sync_to_hpc.sh
```

### Step 2: Connect to HPC and Setup
```bash
./connect_hpc.sh burst
# On HPC:
cd /scratch/hc4569/repos/PhysVideoGenerator
bash /scratch/hc4569/setup_comparison_experiments.sh
```

### Step 3: Sync Evaluation Scripts
The evaluation scripts should be synced to HPC. They will be in:
- `/scratch/hc4569/repos/PhysVideoGenerator/evaluation/`
- `/scratch/hc4569/repos/PhysVideoGenerator/evaluation_utils.py`

### Step 4: Commit Changes
On HPC:
```bash
cd /scratch/hc4569/repos/PhysVideoGenerator
bash /scratch/hc4569/commit_comparison_experiments.sh
```

Or manually:
```bash
cd /scratch/hc4569/repos/PhysVideoGenerator
git checkout comparison-experiments
git add evaluation/ evaluation_utils.py requirements_comparison.txt setup_comparison_experiments.sh commit_comparison_experiments.sh
git commit -m "Add evaluation scripts for comparison experiments (OpenSora, VideoCrafter2, HunyuanVideo)"
git push -u origin comparison-experiments
```

## Next Steps

1. Install model-specific dependencies on HPC
2. Download model checkpoints
3. Prepare test prompts
4. Run evaluations using the scripts
5. Compare results across models

## Notes

- All evaluation scripts use placeholder model loading functions
- Adjust model loading based on actual model APIs from official repositories
- Scripts are designed to be modular and easy to customize
- Metrics computation is standardized for fair comparison


