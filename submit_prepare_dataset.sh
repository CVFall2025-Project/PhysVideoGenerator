#!/bin/bash

# --- SLURM CONFIGURATIOON ---
#SBATCH --requeue
#SBATCH --job-name=prepare_video_dataset
#SBATCH --output=logs/prepare_video_dataset_%j.out
#SBATCH --error=logs/prepare_video_dataset_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --partition=gpu

# Optional: Set GPU type (uncomment if needed)
# #SBATCH --gres=gpu:v100:1

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
source /scratch/sk12590/miniconda3/bin/activate cv_project

# Run the pipeline with configurable options
python -u src/01_prepare_video_dataset.py \
    --root . \
    --parts 1 \
    --limit 100 \
    --no-clean \
    --no-vae \
    --no-vjepa \
    --no-text \

echo "======================================"
echo "Pipeline execution completed"
echo "======================================"
