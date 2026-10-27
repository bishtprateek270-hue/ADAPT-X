#!/bin/bash
# ==============================================================================
# ADAPT-X: HEAVY DEEP RL NEURAL TRAINING ON NVIDIA DGX SPARK
# Multi-Head Temporal Transformer (RMA) + 512-wide Deep ResNet Actor-Critic (3.3M Params)
# BFloat16 Tensor Core Acceleration on NVIDIA GB10 Blackwell GPU
# ==============================================================================

set -e

PHASE="${1:-all}"
SCALE="${2:-1.0}"
NUM_ENVS="${3:-32}"

echo "======================================================================"
echo "      ADAPT-X: HEAVY DEEP RL TRAINING ON NVIDIA DGX SPARK"
echo "======================================================================"
echo " Platform Hostname : $(hostname)"
echo " Neural Model      : Multi-Head Temporal Transformer + ResNet (3.3M Params)"
echo " Target Phase      : ${PHASE}"
echo " Timestep Scale    : ${SCALE}x"
echo " Parallel Workers  : ${NUM_ENVS}"
echo "----------------------------------------------------------------------"

# 1. Print GPU Information
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
echo "----------------------------------------------------------------------"

# 2. Execute Heavy DGX Spark Training
/home/piet/ADAPT-X/.venv/bin/python train_heavy_dgx.py \
    --phase "${PHASE}" \
    --scale "${SCALE}" \
    --num-envs "${NUM_ENVS}" \
    --device 0

echo "======================================================================"
echo " [SUCCESS] Heavy Deep RL Training Session Complete!"
echo " Checkpoints stored in : checkpoints/dgx_spark_heavy/"
echo " Exported models in    : deployments/adaptx_heavy_dgx_spark.*"
echo "======================================================================"
