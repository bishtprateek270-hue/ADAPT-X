#!/bin/bash
#SBATCH --job-name=adaptx_dgx_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_dgx_%j.log
#SBATCH --error=logs/slurm_dgx_%j.err

# ==============================================================================
# Slurm Job Script for NVIDIA DGX A100/H100 SuperPOD Training
# ==============================================================================

mkdir -p logs
mkdir -p checkpoints

echo "Job Started on Host: $(hostname)"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"

# Run Vectorized DGX Trainer (128 parallel environments, 2,000,000 steps)
python3 train_dgx.py --env drone --num-envs 128 --timesteps 2000000 --device 0

echo "Job Finished at: $(date)"
