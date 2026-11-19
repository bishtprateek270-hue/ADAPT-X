#!/usr/bin/env bash
# ==============================================================================
# ADAPT-X: 1-Click Unified Full-Stack Launcher (Frontend + Backend)
# Launches 3D MuJoCo Physics Backend & Real-Time Web Telemetry Cockpit
# ==============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/logs"
LOG_FILE="$SCRIPT_DIR/logs/launcher.log"

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

# Locate Virtual Environment Python
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    echo -e "${YELLOW}[!] Project venv not found at $VENV_PYTHON. Checking python3...${NC}"
    VENV_PYTHON="$(which python3)"
fi

# Ensure all photorealistic 3D city textures are verified/generated
if [ ! -f "$SCRIPT_DIR/models/textures/fountain_water.png" ] || [ ! -f "$SCRIPT_DIR/models/textures/prop_blur.png" ]; then
    echo -e "${CYAN}[→] Generating high-resolution photorealistic city textures...${NC}"
    "$VENV_PYTHON" "$SCRIPT_DIR/generate_city_textures.py" 2>/dev/null || true
fi

# Clean up any previously running instance of web_drone_viewer
echo -e "${YELLOW}[→] Terminating any stale ADAPT-X viewer instances...${NC}"
pkill -f "web_drone_viewer.py" 2>/dev/null || true
sleep 0.5

# Parse arguments intelligently (ports and resolution)
PREF_HTTP=8080
PREF_WS=8081
RES="720p"
SET_HTTP=""

for arg in "$@"; do
    case "$arg" in
        1080p|1080|fhd|FHD) RES="1080p" ;;
        720p|720|hd|HD) RES="720p" ;;
        450p|450|sd|SD) RES="450p" ;;
        *)
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                if [ -z "$SET_HTTP" ]; then
                    PREF_HTTP="$arg"
                    SET_HTTP=1
                else
                    PREF_WS="$arg"
                fi
            fi
            ;;
    esac
done

HTTP_PORT=$("$VENV_PYTHON" -c "
import socket
pref = int('$PREF_HTTP')
candidates = [pref] + list(range(8082, 8100)) + list(range(8880, 8900))
for p in candidates:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('0.0.0.0', p))
        s.close()
        print(p)
        break
    except OSError:
        pass
")

WS_PORT=$("$VENV_PYTHON" -c "
import socket
pref = int('$PREF_WS')
avoid = int('$HTTP_PORT')
candidates = [pref] + list(range(8081, 8100)) + list(range(8880, 8900))
for p in candidates:
    if p == avoid:
        continue
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('0.0.0.0', p))
        s.close()
        print(p)
        break
    except OSError:
        pass
")

HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$HOST_IP" ] && HOST_IP="127.0.0.1"

if [ "$HTTP_PORT" != "$PREF_HTTP" ]; then
    echo -e "${YELLOW}[i] Port $PREF_HTTP is occupied (e.g. by Docker). Automatically selected free port: ${GREEN}${HTTP_PORT}${NC}"
fi

echo -e " 3D Graphics Core  : ${CYAN}Photorealistic Urban City (Textured High-Rises, 3D Vehicles, Plaza Fountain)${NC}"
echo -e " Rotor Dynamics    : ${CYAN}High-RPM Aerodynamic Motion Blur & Adaptive Throttle Inversion${NC}"
echo -e " Mission Planner   : ${GREEN}Interactive 2D City Map (Click-to-Fly, Custom Patrol Routes & Zero-Drift Hover)${NC}"
echo -e " Fail-Safe Core    : ${CYAN}Dynamic Allocation Rebalancing & DGX Neural RMA (Trajectory Hold with Dead Motors)${NC}"
echo -e " Anti-Collision    : ${GREEN}100% Guaranteed Urban Geofence Containment Field (Zero Building Strikes)${NC}"
echo -e " Stream Engine     : ${CYAN}WebSocket 50 FPS Persistent Binary Stream + Instant HTTP Auto-Fallback${NC}"
echo -e " Stream Resolution : ${GREEN}${RES}${NC} (HD Quality)"
echo -e " Web Frontend      : ${GREEN}http://localhost:${HTTP_PORT}${NC}"
echo -e " Network Access    : ${CYAN}http://${HOST_IP}:${HTTP_PORT}${NC}"
echo -e " WebSocket Stream  : ${CYAN}ws://localhost:${WS_PORT}${NC}"
echo -e "${BOLD}======================================================================${NC}"

# Set environment
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

# Start the unified backend & frontend server in background
echo -e "${GREEN}[→] Launching Simulation Backend & Web Server...${NC}"
"$VENV_PYTHON" "$SCRIPT_DIR/web_drone_viewer.py" --port "$HTTP_PORT" --ws-port "$WS_PORT" --res "$RES" --host 0.0.0.0 2>&1 | tee "$LOG_FILE" &
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
    echo -e "${GREEN}[✓] Shutdown clean. Ports released.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Health check: Wait until web server responds
echo -e "${CYAN}[→] Awaiting server readiness on port ${HTTP_PORT}...${NC}"
READY=0
for i in {1..40}; do
    # Check if process is still running
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo -e "${RED}[✗] Server process exited unexpectedly! See error log above.${NC}"
        break
    fi
    if curl -s -f "http://127.0.0.1:${HTTP_PORT}/telemetry" >/dev/null 2>&1 || curl -s -f "http://127.0.0.1:${HTTP_PORT}/" >/dev/null 2>&1; then
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
            "Cockpit is LIVE at http://localhost:${HTTP_PORT}" 2>/dev/null || true
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

    echo -e "\n${BOLD}${GREEN}✔ ADAPT-X is running in live stream mode.${NC}"
    echo -e "${CYAN}Press Ctrl+C in this terminal at any time to stop the simulation.${NC}\n"

    # Keep script running attached to the server process
    wait "$SERVER_PID"
else
    echo -e "${RED}[✗] Failed to initialize ADAPT-X simulation server.${NC}"
    echo -e "Review logs in: ${LOG_FILE}"
    echo ""
    read -n 1 -s -r -p "Press any key to close..."
    echo ""
    exit 1
fi
