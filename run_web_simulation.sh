#!/bin/bash
# ==============================================================================
# ADAPT-X: 3D PHOTOREALISTIC WEB SIMULATION LAUNCHER
# Zero-install web visualizer: View and interact with the drone in ANY browser!
# ==============================================================================

PORT=${1:-8080}
HOST_IP=$(hostname -I | awk '{print $1}')

echo "======================================================================"
echo "    ADAPT-X: 3D PHOTOREALISTIC DRONE WEB VISUALIZER"
echo "======================================================================"
echo " Server Port  : $PORT"
echo " Local URL    : http://localhost:$PORT"
echo " Network URL  : http://${HOST_IP}:$PORT"
echo " Compatible   : Chrome, Safari, Firefox, Edge (Laptop, Phone, Tablet)"
echo " Controls     : Real-time Camera Orbit, FPV Gimbal, Motor Failure Injection"
echo "======================================================================"
echo " Open your browser now to: http://${HOST_IP}:$PORT"
echo "======================================================================"

export MUJOCO_GL=egl
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 0.5

exec /home/piet/ADAPT-X/.venv/bin/python /home/piet/ADAPT-X/web_drone_viewer.py --port "$PORT" --host 0.0.0.0
