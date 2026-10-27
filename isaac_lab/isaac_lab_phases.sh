#!/bin/bash
# ==============================================================================
# ADAPT-X: NVIDIA Isaac Lab / Isaac Sim Phased Training Launcher
# ==============================================================================

set -e

ROBOT="${1:-quadruped}"
PHASE="${2:-all}"
NUM_ENVS="${3:-4096}"

echo "======================================================================"
echo "          ADAPT-X: NVIDIA ISAAC LAB & ISAAC SIM LAUNCHER"
echo "======================================================================"
echo " Target Robot : ${ROBOT}"
echo " Phase        : ${PHASE}"
echo " Actors       : ${NUM_ENVS}"
echo "----------------------------------------------------------------------"

if command -v isaaclab.sh &> /dev/null; then
    ./isaaclab.sh -p isaac_lab/train_isaac_lab_phases.py --robot "${ROBOT}" --phase "${PHASE}" --num_envs "${NUM_ENVS}"
else
    echo "[Notice] isaaclab.sh not found in PATH."
    echo "Falling back to DGX Spark GPU Tensor Trainer..."
    python3 train_dgx_phases.py --robot "${ROBOT}" --phase "${PHASE}" --num-envs 32
fi
