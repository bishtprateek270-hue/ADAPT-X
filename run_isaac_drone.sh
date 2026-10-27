#!/bin/bash
# ==============================================================================
# ADAPT-X: NVIDIA ISAAC LAB & DGX SPARK DRONE RL TRAINING LAUNCHER
# Automated phased RL training for Quadcopter UAV with mid-air rotor recovery
# ==============================================================================

set -e

PHASE="${1:-all}"
SCALE="${2:-1.0}"
NUM_ENVS="${3:-32}"

echo "======================================================================"
echo "    ADAPT-X: DRONE REINFORCEMENT LEARNING TRAINING (ISAAC / DGX)"
echo "======================================================================"
echo " Target Robot      : Autonomous Quadcopter Drone (UAV)"
echo " Training Phase    : ${PHASE}"
echo " Timestep Scale    : ${SCALE}x"
echo " Parallel Workers  : ${NUM_ENVS}"
echo "----------------------------------------------------------------------"

# GPU Check
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
echo "----------------------------------------------------------------------"

# Check for Isaac Lab Omniverse Kit
if command -v isaaclab.sh &> /dev/null; then
    echo "[NVIDIA Isaac Lab Detected!] Launching 4,096-Drone Omniverse GPU Simulation..."
    ./isaaclab.sh -p isaac_lab/train_isaac_drone.py --phase "${PHASE}" --num_envs 4096
else
    echo "[Running Native DGX Spark GPU Tensor Engine on GB10 GPU...]"
    /home/piet/ADAPT-X/.venv/bin/python train_heavy_dgx.py \
        --robot drone \
        --phase "${PHASE}" \
        --scale "${SCALE}" \
        --num-envs "${NUM_ENVS}" \
        --device 0
fi

echo "======================================================================"
echo " [SUCCESS] Drone Training Session Complete!"
echo " Checkpoints stored in : checkpoints/dgx_spark_heavy/drone_*"
echo " Exported ONNX model   : deployments/adaptx_heavy_drone.onnx"
echo "======================================================================"
