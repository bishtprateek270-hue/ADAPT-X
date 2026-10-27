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

# Check for Isaac Lab Omniverse Kit or Isaac Sim Docker Image
ISAAC_CONTAINER="nvcr.io/nvidia/isaac-sim:6.2.0"
CONTAINER_EXISTS=$(docker images -q "${ISAAC_CONTAINER}" 2>/dev/null || docker images -q "nvcr.io/nvidia/isaac-sim:6.0.1" 2>/dev/null || docker images -q "nvcr.io/nvidia/isaac-sim:5.1.0" 2>/dev/null || true)
if [ -z "${CONTAINER_EXISTS}" ]; then
    ISAAC_CONTAINER="nvcr.io/nvidia/isaac-sim:6.2.0"
fi

if [ -n "${CONTAINER_EXISTS}" ]; then
    echo "[NVIDIA Isaac Sim Container Detected: ${ISAAC_CONTAINER}]"
    echo "Launching 4,096-Drone Omniverse GPU PhysX 5 Simulation on GB10 GPU..."
    docker run --rm \
        --gpus all \
        --network host \
        -e "ACCEPT_EULA=Y" \
        -e "PRIVACY_CONSENT=Y" \
        -v /home/piet/ADAPT-X:/workspace/ADAPT-X \
        -v /home/piet/IsaacLab:/workspace/IsaacLab \
        -w /workspace/ADAPT-X \
        --entrypoint bash \
        "${ISAAC_CONTAINER}" \
        -c "PYTHONPATH=/workspace/ADAPT-X:/workspace/IsaacLab/source/extensions/omni.isaac.lab /isaac-sim/python.sh isaac_lab/train_isaac_drone.py --phase '${PHASE}' --num_envs 4096"
elif command -v isaaclab.sh &> /dev/null || [ -f "/home/piet/IsaacLab/isaaclab.sh" ]; then
    ISAACLAB_BIN=$(command -v isaaclab.sh || echo "/home/piet/IsaacLab/isaaclab.sh")
    echo "[NVIDIA Isaac Lab Detected at ${ISAACLAB_BIN}!] Launching 4,096-Drone Omniverse GPU Simulation..."
    "${ISAACLAB_BIN}" -p isaac_lab/train_isaac_drone.py --phase "${PHASE}" --num_envs 4096
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
