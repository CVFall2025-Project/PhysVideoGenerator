#!/bin/bash

# --- SLURM CONFIGURATIOON ---
#SBATCH --requeue
#SBATCH --job-name=prepare_video_dataset
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --account=csci_ga_2271_001-2025fa
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --partition=c12m85-a100-1

# Optional: Set GPU type (uncomment if needed)
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "======================================"

# Change to project root
cd /scratch/sk12590/PhysVideoGenerator # Update this to your actual project path

# Load environment modules (adjust based on your cluster)
source /scratch/sk12590/miniconda3/etc/profile.d/conda.sh
conda activate cv_project

# Run the pipeline with configurable options
python -u src/01_prepare_video_dataset.py \
    --parts 1 \
    --limit 100 \
    --no-clean \
    --no-vae \
    --no-vjepa \
    --no-text \

echo "======================================"
echo "Pipeline execution completed"
echo "======================================"

echo "Training finished. Starting data transfer..."
# Define where your results are currently sitting (Local Scratch)
LOCAL_RESULTS="/scratch/$USER/PhysVideoGenerator/data"

# Define where you want them to go (login node / network scratch)
TARGET="sk12590@greene.hpc.nyu.edu:/scratch/sk12590/PhysVideoGenerator/data/"

# Ensure results directory exists
if [ ! -d "$LOCAL_RESULTS" ]; then
  echo "No results found at $LOCAL_RESULTS - skipping transfer."
else
  ARCHIVE="results_${SLURM_JOB_ID}.tar.gz"
  echo "Creating archive $ARCHIVE from $LOCAL_RESULTS (contents)..."
  # -C change to directory, then archive contents (avoids storing absolute paths)
  tar -C "$LOCAL_RESULTS" -czf "$ARCHIVE" . || { echo "tar failed"; exit 1; }

  # Check archive size
  echo "Archive created: $(du -h "$ARCHIVE" | cut -f1)"

  # Send it home. Use BatchMode to avoid password prompt (fail fast if no keys)
  echo "Transferring $ARCHIVE -> $TARGET"
  scp -o BatchMode=yes "$ARCHIVE" "$TARGET"
  if [ $? -ne 0 ]; then
    echo "scp failed. Ensure SSH keys are configured and compute node can reach target."
    # Optionally keep the archive for manual transfer and exit with non-zero
    exit 1
  fi

  echo "Transfer succeeded. Removing local archive."
  rm -f "$ARCHIVE"
  echo "Data transfer completed."
fi