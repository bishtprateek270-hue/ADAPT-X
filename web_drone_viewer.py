"""
ADAPT-X: Web-Based 3D Photorealistic Drone Flight Visualizer & Live Stream Server
Streams real-time 3D flight simulation with mid-air failure injection directly
to any browser (Chrome, Edge, Firefox, Safari) on your laptop, phone, or tablet.
"""

import os
import sys

# Force headless hardware-accelerated NVIDIA EGL rendering (eliminates GLX X11 static noise)
os.environ["MUJOCO_GL"] = "egl"

# Auto-switch to virtual environment if needed
venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if os.path.exists(venv_python) and sys.executable != venv_python:
    try:
        import mujoco
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

import time
import json
import io
import threading
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import numpy as np
from PIL import Image
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.drone_env import CityDroneEnv


class SimulationWorker:
    def __init__(self, target_altitude=4.5):
        self.target_altitude = target_altitude
        self.env = CityDroneEnv(target_pos=(0.0, 0.0, target_altitude))
        self.env.reset()

        self.width = 960
        self.height = 540
        self.renderer = None

        self.camera = mujoco.MjvCamera()
        self.drone_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.camera_mode = "chase"
        self._update_camera()

        self.flight_mode = "cruise"
        self.damage_motor = "none"
        self.is_running = True
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.telemetry = {
            "altitude": 0.0,
            "vz": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "rotors": [0, 0, 0, 0],
            "damage": "none",
            "mode": "cruise",
            "fps": 0.0,
            "step": 0
        }

        # Control gains
        self.kp_z = 26.0
        self.kd_z = 10.0
        self.kp_pos = 0.25
        self.kd_pos = 0.35
        self.kp_att = 3.5
        self.kd_att = 0.45
        self.step_count = 0

    def _update_camera(self):
        if self.camera_mode == "chase":
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.drone_id
            self.camera.distance = 2.4
            self.camera.elevation = -12.0
            self.camera.azimuth = 90.0
        elif self.camera_mode == "fpv":
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.drone_id
            self.camera.distance = 0.5
            self.camera.elevation = -4.0
            self.camera.azimuth = 0.0
        elif self.camera_mode == "orbit":
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.drone_id
            self.camera.distance = 3.6
            self.camera.elevation = -18.0
            self.camera.azimuth = (self.step_count * 0.5) % 360.0
        elif self.camera_mode == "top":
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.drone_id
            self.camera.distance = 5.0
            self.camera.elevation = -89.0
            self.camera.azimuth = 0.0

    def set_camera(self, mode):
        with self.lock:
            self.camera_mode = mode
            self._update_camera()

    def set_damage(self, rotor):
        with self.lock:
            self.damage_motor = rotor
            if rotor != "none":
                self.env.inject_rotor_fault("dead_motor", rotor=rotor)
            else:
                self.env.inject_rotor_fault("none")

    def set_mode(self, mode):
        with self.lock:
            self.flight_mode = mode

    def run_loop(self):
        # Create EGL hardware accelerated renderer inside worker thread
        self.renderer = mujoco.Renderer(self.env.model, height=self.height, width=self.width)
        last_time = time.perf_counter()
        fps_counter = 0
        fps_timer = time.perf_counter()
        current_fps = 30.0
        max_thrust = 12.0

        while self.is_running:
            loop_start = time.perf_counter()
            self.step_count += 1
            t = self.step_count * 0.02

            with self.lock:
                curr_pos = self.env.data.qpos[0:3].copy()
                curr_vel = self.env.data.qvel[0:3].copy()
                w, x_q, y_q, z_q = self.env.data.qpos[3:7]
                wx, wy, wz = self.env.data.qvel[3:6]

                # Desired position
                if self.flight_mode == "cruise":
                    x_des = 1.8 * np.cos(0.35 * t)
                    y_des = 1.8 * np.sin(0.35 * t)
                    z_des = self.target_altitude + 0.15 * np.sin(0.5 * t)
                elif self.flight_mode == "orbit":
                    x_des = 2.2 * np.cos(0.25 * t)
                    y_des = 2.2 * np.sin(0.25 * t)
                    z_des = self.target_altitude
                else:
                    x_des, y_des, z_des = 0.0, 0.0, self.target_altitude

                ex = x_des - curr_pos[0]
                ey = y_des - curr_pos[1]
                pitch_des = np.clip( (self.kp_pos * ex - self.kd_pos * curr_vel[0]), -0.15, 0.15)
                roll_des  = np.clip(-(self.kp_pos * ey - self.kd_pos * curr_vel[1]), -0.15, 0.15)

                # Boundary clamp
                r_horiz = np.hypot(curr_pos[0], curr_pos[1])
                if r_horiz > 3.0:
                    pen = r_horiz - 3.0
                    ux, uy = curr_pos[0]/r_horiz, curr_pos[1]/r_horiz
                    pitch_des += np.clip(-1.5 * pen * ux, -0.2, 0.2)
                    roll_des  += np.clip( 1.5 * pen * uy, -0.2, 0.2)

                sinr_cosp = 2.0 * (w * x_q + y_q * z_q)
                cosr_cosp = 1.0 - 2.0 * (x_q * x_q + y_q * y_q)
                curr_roll = np.arctan2(sinr_cosp, cosr_cosp)
                sinp = np.clip(2.0 * (w * y_q - z_q * x_q), -1.0, 1.0)
                curr_pitch = np.arcsin(sinp)
                siny_cosp = 2.0 * (w * z_q + x_q * y_q)
                cosy_cosp = 1.0 - 2.0 * (y_q * y_q + z_q * z_q)
                curr_yaw = np.arctan2(siny_cosp, cosy_cosp)

                ez = z_des - curr_pos[2]
                fz_nominal = self.env.mass * 9.81
                delta_fz = self.kp_z * ez - self.kd_z * curr_vel[2]
                fz_total = np.clip(fz_nominal + delta_fz, 2.0, 4.0 * max_thrust * 0.95)

                tau_roll  = self.kp_att * (roll_des - curr_roll)   - self.kd_att * wx
                tau_pitch = self.kp_att * (pitch_des - curr_pitch) - self.kd_att * wy
                tau_yaw   = -0.35 * wz

                l = self.env.arm_length
                f_base = fz_total / 4.0
                df_roll  = tau_roll / (4.0 * l)
                df_pitch = tau_pitch / (4.0 * l)
                df_yaw   = tau_yaw / 4.0

                f1 = f_base - df_roll - df_pitch - df_yaw
                f2 = f_base + df_roll - df_pitch + df_yaw
                f3 = f_base - df_roll + df_pitch + df_yaw
                f4 = f_base + df_roll + df_pitch - df_yaw
                raw_forces = np.array([f1, f2, f3, f4])

                # Active mid-air failure compensation
                if self.damage_motor != "none":
                    idx_map = {"fr": 0, "fl": 1, "rr": 2, "rl": 3}
                    dead_idx = idx_map.get(self.damage_motor, 0)
                    raw_forces[dead_idx] = 0.0
                    alive = [i for i in range(4) if i != dead_idx]
                    boost = (fz_total / 3.0) / np.maximum(np.mean(raw_forces[alive]), 1e-3)
                    raw_forces[alive] = np.clip(raw_forces[alive] * boost, 0.0, max_thrust)
                    # Relax yaw to preserve lift
                    raw_forces = np.clip(raw_forces, 0.0, max_thrust)

                action = np.clip(raw_forces, 0.0, max_thrust)

                obs, reward, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    self.env.reset()

                # Update camera if orbit mode
                if self.camera_mode == "orbit":
                    self._update_camera()

                # Render offscreen frame
                self.renderer.update_scene(self.env.data, camera=self.camera)
                pixels = self.renderer.render()

                # Compress to JPEG in memory
                img = Image.fromarray(pixels)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80, optimize=True)
                self.latest_jpeg = buf.getvalue()

                # Update telemetry
                fps_counter += 1
                if time.perf_counter() - fps_timer >= 1.0:
                    current_fps = fps_counter / (time.perf_counter() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.perf_counter()

                self.telemetry = {
                    "altitude": round(float(curr_pos[2]), 2),
                    "vz": round(float(curr_vel[2]), 2),
                    "roll": round(float(np.degrees(curr_roll)), 1),
                    "pitch": round(float(np.degrees(curr_pitch)), 1),
                    "yaw": round(float(np.degrees(curr_yaw)), 1),
                    "rotors": [int(np.clip(f / max_thrust * 100, 0, 100)) for f in raw_forces],
                    "damage": self.damage_motor,
                    "mode": self.flight_mode,
                    "camera": self.camera_mode,
                    "fps": round(current_fps, 1),
                    "step": self.step_count
                }

            # Enforce 30-40 FPS rate
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.005, 0.025 - elapsed))


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ADAPT-X: 3D Photorealistic Drone Simulator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-primary: #0a0e17;
    --bg-card: rgba(18, 25, 40, 0.75);
    --border-color: rgba(56, 189, 248, 0.2);
    --accent-cyan: #38bdf8;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --accent-amber: #f59e0b;
    --text-primary: #f1f5f9;
    --text-muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Outfit', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
  }
  header {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.6));
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-color);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 50;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .badge {
    background: linear-gradient(135deg, #0284c7, #0369a1);
    color: #fff;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .title { font-size: 20px; font-weight: 800; letter-spacing: -0.02em; }
  .subtitle { font-size: 13px; color: var(--text-muted); }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    color: var(--accent-emerald);
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-emerald);
    box-shadow: 0 0 10px var(--accent-emerald);
    animation: pulse 1.8s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
  }
  main {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 20px;
    padding: 20px 24px;
    max-width: 1600px;
    margin: 0 auto;
    width: 100%;
  }
  @media (max-width: 1100px) {
    main { grid-template-columns: 1fr; }
  }
  .stream-container {
    background: #000;
    border-radius: 16px;
    border: 1px solid var(--border-color);
    overflow: hidden;
    position: relative;
    box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    aspect-ratio: 16 / 9;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .stream-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .stream-hud {
    position: absolute;
    top: 14px;
    left: 14px;
    display: flex;
    gap: 10px;
    z-index: 10;
  }
  .hud-chip {
    background: rgba(10, 14, 23, 0.75);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 8px;
    padding: 6px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .sidebar { display: flex; flex-direction: column; gap: 16px; }
  .card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 16px;
  }
  .card-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .telemetry-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    font-family: 'JetBrains Mono', monospace;
  }
  .tel-item {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 10px;
  }
  .tel-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }
  .tel-val { font-size: 16px; font-weight: 700; color: #fff; }
  .rotors-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-top: 10px;
  }
  .rotor-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 10px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .rotor-box.damaged {
    border-color: var(--accent-rose);
    background: rgba(244, 63, 94, 0.15);
  }
  .rotor-name { font-size: 11px; font-weight: 700; margin-bottom: 4px; }
  .rotor-pct { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; }
  .rotor-bar-bg {
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
  }
  .rotor-bar-fill {
    height: 100%;
    background: var(--accent-emerald);
    transition: width 0.1s ease;
  }
  .rotor-box.damaged .rotor-bar-fill { background: var(--accent-rose); width: 0% !important; }
  .btn-group { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .btn {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: var(--text-primary);
    padding: 10px 12px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    font-family: inherit;
  }
  .btn:hover {
    background: rgba(56, 189, 248, 0.15);
    border-color: var(--accent-cyan);
    transform: translateY(-1px);
  }
  .btn.active {
    background: linear-gradient(135deg, #0284c7, #0369a1);
    border-color: var(--accent-cyan);
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
  }
  .btn-danger {
    border-color: rgba(244, 63, 94, 0.4);
    background: rgba(244, 63, 94, 0.1);
    color: #fca5a5;
  }
  .btn-danger:hover {
    background: rgba(244, 63, 94, 0.25);
    border-color: var(--accent-rose);
  }
  .btn-restore {
    grid-column: span 2;
    background: linear-gradient(135deg, #059669, #047857);
    border-color: #34d399;
    color: #fff;
    margin-top: 4px;
  }
  .btn-restore:hover {
    box-shadow: 0 0 16px rgba(16, 185, 129, 0.5);
  }
  .mode-row { display: flex; gap: 8px; }
  .mode-row .btn { flex: 1; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="badge">NVIDIA DGX SPARK</span>
    <div>
      <div class="title">ADAPT-X Autonomous Drone Simulator</div>
      <div class="subtitle">Fault-Tolerant Neural Recovery & Photorealistic 3D Urban World</div>
    </div>
  </div>
  <div class="status-pill" id="live-indicator">
    <div class="status-dot"></div>
    <span id="conn-text">CONNECTED TO DGX</span>
  </div>
</header>

<main>
  <div class="stream-container">
    <img id="stream-view" class="stream-img" src="/stream.mjpg" alt="ADAPT-X Live Drone Stream">
    <div class="stream-hud">
      <div class="hud-chip">📹 <span id="hud-cam">CHASE CAM</span></div>
      <div class="hud-chip">⚡ <span id="hud-fps">30.0</span> FPS</div>
      <div class="hud-chip" id="hud-damage-chip" style="color: var(--accent-emerald);">● ALL MOTORS NOMINAL</div>
    </div>
  </div>

  <div class="sidebar">
    <div class="card">
      <div class="card-title">Flight Telemetry</div>
      <div class="telemetry-grid">
        <div class="tel-item"><div class="tel-label">Altitude (Z)</div><div class="tel-val" id="tel-alt">4.50 m</div></div>
        <div class="tel-val-sub tel-item"><div class="tel-label">Vert Speed (Vz)</div><div class="tel-val" id="tel-vz">0.00 m/s</div></div>
        <div class="tel-item"><div class="tel-label">Roll / Pitch</div><div class="tel-val" id="tel-angles">0.0° / 0.0°</div></div>
        <div class="tel-item"><div class="tel-label">Yaw Angle</div><div class="tel-val" id="tel-yaw">0.0°</div></div>
      </div>

      <div class="card-title" style="margin-top: 14px;">Brushless Motor Thrust</div>
      <div class="rotors-grid">
        <div class="rotor-box" id="rotor-fr">
          <div class="rotor-name">M1 (Front-R)</div>
          <div class="rotor-pct" id="rotor-fr-val">50%</div>
          <div class="rotor-bar-bg"><div class="rotor-bar-fill" id="rotor-fr-bar" style="width: 50%;"></div></div>
        </div>
        <div class="rotor-box" id="rotor-fl">
          <div class="rotor-name">M2 (Front-L)</div>
          <div class="rotor-pct" id="rotor-fl-val">50%</div>
          <div class="rotor-bar-bg"><div class="rotor-bar-fill" id="rotor-fl-bar" style="width: 50%;"></div></div>
        </div>
        <div class="rotor-box" id="rotor-rr">
          <div class="rotor-name">M3 (Rear-R)</div>
          <div class="rotor-pct" id="rotor-rr-val">50%</div>
          <div class="rotor-bar-bg"><div class="rotor-bar-fill" id="rotor-rr-bar" style="width: 50%;"></div></div>
        </div>
        <div class="rotor-box" id="rotor-rl">
          <div class="rotor-name">M4 (Rear-L)</div>
          <div class="rotor-pct" id="rotor-rl-val">50%</div>
          <div class="rotor-bar-bg"><div class="rotor-bar-fill" id="rotor-rl-bar" style="width: 50%;"></div></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Mid-Air Failure Injection</div>
      <div class="btn-group">
        <button class="btn btn-danger" onclick="injectDamage('fr')">💥 Cut Motor 1 (FR)</button>
        <button class="btn btn-danger" onclick="injectDamage('fl')">💥 Cut Motor 2 (FL)</button>
        <button class="btn btn-danger" onclick="injectDamage('rr')">💥 Cut Motor 3 (RR)</button>
        <button class="btn btn-danger" onclick="injectDamage('rl')">💥 Cut Motor 4 (RL)</button>
        <button class="btn btn-restore" onclick="injectDamage('none')">🔄 Restore All 4 Motors</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Camera Angle</div>
      <div class="btn-group">
        <button class="btn active" id="cam-chase" onclick="setCam('chase')">Chase Cam</button>
        <button class="btn" id="cam-fpv" onclick="setCam('fpv')">FPV Gimbal</button>
        <button class="btn" id="cam-orbit" onclick="setCam('orbit')">360° Orbit</button>
        <button class="btn" id="cam-top" onclick="setCam('top')">Bird's Eye</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Trajectory Mode</div>
      <div class="mode-row">
        <button class="btn active" id="mode-cruise" onclick="setFlightMode('cruise')">Cruise Plaza</button>
        <button class="btn" id="mode-orbit" onclick="setFlightMode('orbit')">Wide Orbit</button>
        <button class="btn" id="mode-hover" onclick="setFlightMode('hover')">Hover Spot</button>
      </div>
    </div>
  </div>
</main>

<script>
function injectDamage(rotor) {
  fetch('/inject_damage?rotor=' + rotor);
}
function setCam(cam) {
  fetch('/set_camera?cam=' + cam);
  ['chase', 'fpv', 'orbit', 'top'].forEach(c => {
    const el = document.getElementById('cam-' + c);
    if (el) el.classList.toggle('active', c === cam);
  });
  document.getElementById('hud-cam').innerText = cam.toUpperCase() + ' CAM';
}
function setFlightMode(mode) {
  fetch('/set_mode?mode=' + mode);
  ['cruise', 'orbit', 'hover'].forEach(m => {
    const el = document.getElementById('mode-' + m);
    if (el) el.classList.toggle('active', m === mode);
  });
}

setInterval(async () => {
  try {
    const res = await fetch('/telemetry');
    const data = await res.json();
    document.getElementById('tel-alt').innerText = data.altitude.toFixed(2) + ' m';
    document.getElementById('tel-vz').innerText = (data.vz >= 0 ? '+' : '') + data.vz.toFixed(2) + ' m/s';
    document.getElementById('tel-angles').innerText = data.roll.toFixed(1) + '° / ' + data.pitch.toFixed(1) + '°';
    document.getElementById('tel-yaw').innerText = data.yaw.toFixed(1) + '°';
    document.getElementById('hud-fps').innerText = data.fps.toFixed(1);

    const rotors = ['fr', 'fl', 'rr', 'rl'];
    rotors.forEach((r, idx) => {
      const box = document.getElementById('rotor-' + r);
      const val = document.getElementById('rotor-' + r + '-val');
      const bar = document.getElementById('rotor-' + r + '-bar');
      const pct = data.rotors[idx];
      const isDamaged = (data.damage === r);
      box.classList.toggle('damaged', isDamaged);
      val.innerText = isDamaged ? 'DEAD' : pct + '%';
      bar.style.width = isDamaged ? '0%' : pct + '%';
    });

    const chip = document.getElementById('hud-damage-chip');
    if (data.damage !== 'none') {
      chip.style.color = 'var(--accent-rose)';
      chip.innerText = '⚠️ ROTOR ' + data.damage.toUpperCase() + ' SEVERED (RECOVERING)';
    } else {
      chip.style.color = 'var(--accent-emerald)';
      chip.innerText = '● ALL MOTORS NOMINAL';
    }
  } catch (err) {}
}, 100);
</script>
</body>
</html>
"""


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class StreamingHandler(BaseHTTPRequestHandler):
    worker = None

    def log_message(self, format, *args):
        # Silence HTTP request access logs
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(self.worker.telemetry).encode("utf-8"))

        elif path == "/inject_damage":
            rotor = params.get("rotor", ["none"])[0]
            self.worker.set_damage(rotor)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        elif path == "/set_camera":
            cam = params.get("cam", ["chase"])[0]
            self.worker.set_camera(cam)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        elif path == "/set_mode":
            mode = params.get("mode", ["cruise"])[0]
            self.worker.set_mode(mode)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        elif path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.end_headers()

            while self.worker.is_running:
                jpeg_data = self.worker.latest_jpeg
                if jpeg_data is not None:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg_data)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.03)

        else:
            self.send_response(404)
            self.end_headers()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ADAPT-X Web-Based 3D Drone Visualizer")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host (default: 0.0.0.0)")
    args = parser.parse_args()

    worker = SimulationWorker()
    worker_thread = threading.Thread(target=worker.run_loop, daemon=True)
    worker_thread.start()

    StreamingHandler.worker = worker
    server = ThreadedHTTPServer((args.host, args.port), StreamingHandler)

    print("=" * 75)
    print("      ADAPT-X: 3D PHOTOREALISTIC WEB SIMULATION SERVER")
    print("=" * 75)
    print(f" Web Server Running : http://0.0.0.0:{args.port}")
    print(f" Open on Laptop     : http://10.42.0.44:{args.port}")
    print(f" Localhost URL      : http://localhost:{args.port}")
    print(" Compatible Browsers: Google Chrome, Apple Safari, Mozilla Firefox, Edge")
    print(" Features           : Real-Time Video, HUD Telemetry, Live Failure Injection")
    print("=" * 75)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Shutting down web visualizer server...]")
        worker.is_running = False
        server.server_close()


if __name__ == "__main__":
    main()
