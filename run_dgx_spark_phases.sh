#!/bin/bash
# ==============================================================================
# ADAPT-X: NVIDIA DGX SPARK PHASED RL LAUNCHER
# System: NVIDIA DGX Spark (NVIDIA Grace Blackwell GB10 / CUDA 13.0)
# ==============================================================================

set -e

ROBOT="${1:-quadruped}"
PHASE="${2:-all}"
MODE="${3:---fast}"

echo "======================================================================"
echo "          ADAPT-X: NVIDIA DGX SPARK PHASED RL TRAINING"
echo "======================================================================"
echo " Machine Hostname  : $(hostname)"
echo " Kernel Arch       : $(uname -m)"
echo " Target Robot      : ${ROBOT}"
echo " Training Phase    : ${PHASE}"
echo " Execution Mode    : ${MODE}"
echo "----------------------------------------------------------------------"

# 1. GPU Verification
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
echo "----------------------------------------------------------------------"

# 2. Check for Isaac Lab
if command -v isaaclab.sh &> /dev/null; then
    echo "[NVIDIA Isaac Lab Detected!] Launching Isaac Lab Phased Pipeline..."
    ./isaaclab.sh -p isaac_lab/train_isaac_lab_phases.py --robot "${ROBOT}" --phase "${PHASE}" --num_envs 4096
else
    echo "[Running Native DGX Spark GPU Tensor Trainer on GB10 GPU...]"
    /home/piet/ADAPT-X/.venv/bin/python train_dgx_phases.py \
        --robot "${ROBOT}" \
        --phase "${PHASE}" \
        --num-envs 32 \
        --device 0 \
        ${MODE}
fi

echo "======================================================================"
echo " [SUCCESS] DGX Spark Phased Training Run Completed!"
echo " Checkpoints stored in : checkpoints/dgx_spark/"
echo " Exported models in    : deployments/"
echo "======================================================================"
