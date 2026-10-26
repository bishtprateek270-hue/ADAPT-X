#!/bin/bash
# ==============================================================================
# ADAPT-X: NVIDIA DGX Training Launcher
# Automatically detects available DGX GPUs and launches high-speed parallel RL
# ==============================================================================

echo "======================================================================"
echo "          ADAPT-X: NVIDIA DGX HIGH-PERFORMANCE TRAINING"
echo "======================================================================"

# 1. Print GPU Information
if command -v nvidia-smi &> /dev/null
then
    echo "[NVIDIA System Info]"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "----------------------------------------------------------------------"
fi

# 2. Setup Python Virtual Environment if needed
if [ ! -d "venv_dgx" ]; then
    echo "[Notice] Creating DGX Python virtual environment..."
    python3 -m venv venv_dgx
    source venv_dgx/bin/activate
    pip install --upgrade pip
    pip install torch gymnasium mujoco onnx onnxruntime pyyaml numpy
else
    source venv_dgx/bin/activate
fi

# 3. Check for NVIDIA Isaac Lab
if command -v isaaclab.sh &> /dev/null; then
    echo "[NVIDIA Isaac Lab Detected!] Launching 4,096-Actor GPU Simulation..."
    ./isaaclab.sh -p isaac_lab/train_isaac_lab.py --num_envs 4096 --robot drone
else
    echo "[Launching Vectorized Multi-Worker DGX Trainer (128 Envs, 1,000,000 Steps)...]"
    python3 train_dgx.py --env drone --num-envs 128 --timesteps 1000000 --device 0
fi

echo "======================================================================"
echo " [SUCCESS] DGX Training Session Complete!"
echo " Copy 'checkpoints/dgx_drone_best_policy.pt' to your laptop to visualize."
echo "======================================================================"
