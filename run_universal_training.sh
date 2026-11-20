#!/bin/bash
# ==============================================================================
# ADAPT-X: UNIVERSAL SCALED DRONE TRAINING ON NVIDIA DGX SPARK (~25.6M PARAMS)
# Multi-Head Temporal Transformer (RMA) + 1024-wide Residual Actor-Critic
# Trained across every drone dimension & mass class existing in the world (100g to 20kg)
# Automatic Online System Identification / Parameter "Fetching" via Temporal Attention
# ==============================================================================

set -e

PHASE="${1:-all}"
PROFILE="${2:-all}"
NUM_ENVS="${3:-32}"
FAST_FLAG=""

if [ "$4" == "--fast" ] || [ "$1" == "--fast" ]; then
    FAST_FLAG="--fast"
    if [ "$1" == "--fast" ]; then PHASE="all"; fi
fi

echo "=============================================================================="
echo "    ADAPT-X: UNIVERSAL SCALED DRONE RL TRAINING (~25.6M PARAMETERS)"
echo "=============================================================================="
echo " Platform Hostname : $(hostname)"
echo " Neural Model      : Multi-Head Temporal Transformer + 1024 ResNet (~25.6M Params)"
echo " Morphology Class  : ${PROFILE^^} (Covers 100g to 20kg, 6cm to 75cm arm lengths)"
echo " Target Phase      : ${PHASE}"
echo " Parallel Workers  : ${NUM_ENVS} Vectorized Simulation Workers"
echo " Fast Test Mode    : $([ -n "$FAST_FLAG" ] && echo "ENABLED (5% timesteps)" || echo "FULL RUN (100% timesteps)")"
echo "------------------------------------------------------------------------------"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
echo "------------------------------------------------------------------------------"

/home/piet/ADAPT-X/.venv/bin/python train_universal_drone.py \
    --phase "${PHASE}" \
    --profile "${PROFILE}" \
    --num-envs "${NUM_ENVS}" \
    --device 0 \
    ${FAST_FLAG}

echo "=============================================================================="
echo " [SUCCESS] Universal Scaled Drone Training Session Complete!"
echo " Checkpoints stored in : checkpoints/dgx_spark_universal/"
echo " Exported models in    : deployments/adaptx_universal_drone.onnx & .torchscript.pt"
echo "=============================================================================="
