#!/usr/bin/env bash
# ==============================================================================
# ADAPT-X: 1-Click Unified Full-Stack Launcher (Frontend + Backend)
# Launches 3D MuJoCo Physics Backend & Real-Time Web Telemetry Cockpit
# ==============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HTTP_PORT="${1:-8080}"
WS_PORT="${2:-8081}"
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$HOST_IP" ] && HOST_IP="127.0.0.1"

# ANSI Colors
CYAN="\033[1;36m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
BOLD="\033[1m"
NC="\033[0m"

echo -e "${CYAN}"
cat << 'EOF'
    █████╗ ██████╗  █████╗ ██████╗ ████████╗   ██╗  ██╗
   ██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝   ╚██╗██╔╝
   ███████║██║  ██║███████║██████╔╝   ██║  █████╗╚███╔╝ 
   ██╔══██║██║  ██║██╔══██║██╔═══╝    ██║  ╚════╝██╔██╗ 
   ██║  ██║██████╔╝██║  ██║██║        ██║       ██╔╝ ██╗
   ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝        ╚═╝       ╚═╝  ╚═╝
  SIM-TO-REAL REINFORCEMENT LEARNING FOR DYNAMIC DAMAGE ADAPTATION
EOF
echo -e "${NC}"

echo -e "${BOLD}======================================================================${NC}"
echo -e " ${GREEN}●${NC} Starting ADAPT-X 3D Drone Flight Simulation & Telemetry System"
echo -e "${BOLD}======================================================================${NC}"
echo -e " Hardware Engine   : ${CYAN}NVIDIA EGL Hardware Accelerated Rendering${NC}"
echo -e " Physics Backend   : ${CYAN}MuJoCo 3.17 (50 Hz Simulation & Geofence Containment)${NC}"
echo -e " Neural Policy     : ${CYAN}ADAPT-X Heavy Drone ONNX (DGX Spark Trained)${NC}"
echo -e " Web Frontend      : ${CYAN}http://localhost:${HTTP_PORT}${NC}"
echo -e " Network Access    : ${CYAN}http://${HOST_IP}:${HTTP_PORT}${NC}"
echo -e " WebSocket Stream  : ${CYAN}ws://localhost:${WS_PORT}${NC}"
echo -e "${BOLD}======================================================================${NC}"

# Locate Virtual Environment Python
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    echo -e "${YELLOW}[!] Project venv not found at $VENV_PYTHON. Checking python3...${NC}"
    VENV_PYTHON="$(which python3)"
fi

# Ensure previous instances on these ports are cleanly terminated
echo -e "${YELLOW}[→] Checking and freeing ports ${HTTP_PORT} and ${WS_PORT}...${NC}"
fuser -k "${HTTP_PORT}/tcp" 2>/dev/null || true
fuser -k "${WS_PORT}/tcp" 2>/dev/null || true
pkill -f "web_drone_viewer.py --port ${HTTP_PORT}" 2>/dev/null || true
sleep 0.4

# Set environment
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

# Start the unified backend & frontend server in background
echo -e "${GREEN}[→] Launching Simulation Backend & Web Server...${NC}"
"$VENV_PYTHON" "$SCRIPT_DIR/web_drone_viewer.py" --port "$HTTP_PORT" --ws-port "$WS_PORT" --host 0.0.0.0 &
SERVER_PID=$!

# Trap signals for graceful cleanup
CLEANED_UP=0
cleanup() {
    if [ $CLEANED_UP -eq 1 ]; then
        return
    fi
    CLEANED_UP=1
    echo ""
    echo -e "${YELLOW}[!] Shutting down ADAPT-X Server (PID: ${SERVER_PID})...${NC}"
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    fuser -k "${HTTP_PORT}/tcp" 2>/dev/null || true
    fuser -k "${WS_PORT}/tcp" 2>/dev/null || true
    echo -e "${GREEN}[✓] Shutdown clean. Ports ${HTTP_PORT} and ${WS_PORT} released.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Health check: Wait until web server responds
echo -e "${CYAN}[→] Awaiting server readiness on port ${HTTP_PORT}...${NC}"
READY=0
for i in {1..35}; do
    if curl -s -f "http://127.0.0.1:${HTTP_PORT}/" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 0.2
done

if [ $READY -eq 1 ]; then
    echo -e "${GREEN}[✓] ADAPT-X Full-Stack Server is LIVE and READY!${NC}"
    
    # Notify desktop environment
    if which notify-send >/dev/null 2>&1; then
        ICON_PATH="$SCRIPT_DIR/assets/drone_logo.png"
        [ ! -f "$ICON_PATH" ] && ICON_PATH="adaptx-drone"
        notify-send -i "$ICON_PATH" \
            "ADAPT-X Drone Simulation" \
            "System is LIVE! Opening Cockpit at http://localhost:${HTTP_PORT}" 2>/dev/null || true
    fi

    # Automatically open default browser
    echo -e "${GREEN}[→] Opening 3D Cockpit in default browser...${NC}"
    if which xdg-open >/dev/null 2>&1; then
        xdg-open "http://localhost:${HTTP_PORT}" 2>/dev/null &
    elif which google-chrome >/dev/null 2>&1; then
        google-chrome "http://localhost:${HTTP_PORT}" 2>/dev/null &
    elif which firefox >/dev/null 2>&1; then
        firefox "http://localhost:${HTTP_PORT}" 2>/dev/null &
    fi
else
    echo -e "${RED}[✗] Server failed to respond within timeout. Check logs above.${NC}"
fi

echo -e "\n${BOLD}${GREEN}✔ ADAPT-X is running in live stream mode.${NC}"
echo -e "${CYAN}Press Ctrl+C in this terminal at any time to stop the simulation.${NC}\n"

# Keep script running attached to the server process
wait "$SERVER_PID"
