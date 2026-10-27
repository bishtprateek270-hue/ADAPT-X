#!/bin/bash
# ==============================================================================
# ADAPT-X: PHOTOREALISTIC NVIDIA ISAAC SIM 6.2.0 OMNIVERSE GUI LAUNCHER
# Launches NVIDIA Omniverse with RTX real-time ray tracing & photorealistic USD
# ==============================================================================

set -e

echo "======================================================================"
echo "    ADAPT-X: NVIDIA ISAAC SIM 6.2.0 PHOTOREALISTIC OMNIVERSE GUI"
echo "======================================================================"
echo " Engine      : NVIDIA Omniverse Kit / Isaac Sim 6.2.0"
echo " Hardware    : NVIDIA DGX Spark (GB10 Blackwell GPU)"
echo " Renderer    : RTX Real-Time Ray Tracing & PhysX 5 Tensor Dynamics"
echo " Stream URL  : http://localhost:8211/streaming/client/"
echo "----------------------------------------------------------------------"
echo " Open your browser (Firefox / Chrome) to: http://localhost:8211/streaming/client/"
echo " to interact with the photorealistic 3D Omniverse drone world!"
echo "======================================================================"

docker run --rm -it \
    --gpus all \
    --network host \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -v /home/piet/ADAPT-X:/workspace/ADAPT-X \
    -v /home/piet/IsaacLab:/workspace/IsaacLab \
    nvcr.io/nvidia/isaac-sim:6.2.0 \
    /isaac-sim/isaac-sim.streaming.sh
