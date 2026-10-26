# ADAPT-X: The "Self-Healing" Quadruped Robot
### Sim-to-Real Reinforcement Learning for Dynamic Damage Adaptation

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2Bcu128-red.svg)
![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo%203.17-orange.svg)
![Status](https://img.shields.io/badge/Status-All%205%20Phases%20Complete-brightgreen.svg)

---

## 📌 Project Overview
When rovers, drones, or legged robots in remote environments (such as disaster zones or planetary exploration) suffer hardware failure, traditional pre-programmed controllers fail because they cannot operate an altered morphology.

**ADAPT-X** is an autonomous 12-DoF quadruped robot controlled by an adaptive Reinforcement Learning (RL) policy. Rather than shutting down upon limb or actuator damage, the agent detects changes through proprioceptive feedback (IMUs + joint encoders) and dynamically transitions its gait to balance and continue locomotion using remaining functional limbs.

---

## 🛠️ Architecture & Tech Stack
* **Physics Simulator:** MuJoCo 3.17 (High-fidelity contact dynamics at 500 Hz).
* **Environment Interface:** Gymnasium (Custom 12-DoF Quadruped MDP).
* **RL Framework:** PyTorch PPO with Generalized Advantage Estimation (GAE).
* **Self-Healing Brain (RMA Paradigm):** Temporal History Encoder ($H = 20$ timesteps $\approx 0.4\text{s}$) compressing proprioception ($840\text{ dims}$) into a latent morphological state $\mathbf{z}_t \in \mathbb{R}^{16}$.
* **Adaptive Policy:** $\pi_\theta(a_t \mid s_t, \mathbf{z}_t)$ instantly re-conditions its control outputs when $\mathbf{z}_t$ shifts under damage.
* **Control Loop:** 50 Hz policy frequency with low-level PD position control ($K_p = 25.0, K_d = 0.8$).

---

## 🗺️ Project Roadmap
- [x] **Phase 1: Baseline 4-Leg Locomotion & Reward Shaping**
  - [x] 12-DoF Quadruped Model (`models/quadruped.xml`)
  - [x] Gymnasium Environment (`envs/quadruped_env.py`)
  - [x] PPO Trainer & Model (`agents/ppo.py`)
  - [x] Laptop GPU (RTX 5050) Safe Execution Profile (< 25MB VRAM)
  - [x] Evaluation & Velocity Tracking Benchmark (`eval.py`)
- [x] **Phase 2: Failure Space & Damage Modeling**
  - [x] Dead motor mode (zero torque / limp joint)
  - [x] Locked joint mode (frozen angle / rigid peg leg)
  - [x] Weakened actuator mode (25% torque output / thermal throttle)
  - [x] Amputated limb mode (power cut to complete 3-joint leg)
  - [x] 100% Damage Survival Rate achieved on Phase 2 evaluation benchmark
- [x] **Phase 3: Adaptive Policy Architecture (Self-Healing Intelligence)**
  - [x] Proprioceptive History Buffer ($20 \text{ steps} \times 42 \text{ dims} = 840 \text{ dims}$)
  - [x] Temporal Latent Encoder (`agents/adaptive_ppo.py`) producing $\mathbf{z}_t \in \mathbb{R}^{16}$
  - [x] Adaptive Policy Network $\pi(a_t \mid s_t, \mathbf{z}_t)$
  - [x] Massive $||\Delta \mathbf{z}|| = 18.5$ neural adaptation shift verified under limb loss
- [x] **Phase 4: Curriculum Learning & 3-Legged Recovery Benchmark**
  - [x] Stage 1: Pre-broken 3-legged hopping/gait discovery
  - [x] Stage 2: Dynamic mid-stride transition with forward velocity retention
  - [x] Stage 3: Search & Rescue stress test with external force impulses
  - [x] Multi-stage curriculum trainer (`train_curriculum.py`)
  - [x] Velocity retention benchmarking CLI (`eval_curriculum.py`)
- [x] **Phase 5: Sim-to-Real Readiness & Production Model Export (ONNX / TorchScript)**
  - [x] Sim-to-Real Reality Gap modeling: Sensor noise, delay, and friction domain randomization (`envs/sim2real_utils.py`)
  - [x] Zero-dependency policy export to ONNX (`deployments/adaptx_policy.onnx`) and TorchScript (`deployments/adaptx_policy.torchscript.pt`)
  - [x] Sub-millisecond latency profile: $61\ \mu\text{s}$ P50 inference ($16,313\text{ Hz}$, $326\times$ faster than 50Hz deadline)
  - [x] Hardware-accurate edge deployment runner (`deploy.py`) with real-time damage adaptation

---

## 🚀 Quick Start

### 1. Installation
Ensure Python and required packages are installed:
```bash
pip install torch gymnasium mujoco onnx onnxruntime pyyaml numpy
```

---

### ⚡ NVIDIA DGX & Isaac Lab High-Performance Training
To train at enterprise scale on **NVIDIA DGX (A100/H100/SuperPOD)** or **NVIDIA Isaac Lab**:
```bash
# 1-Click DGX Launcher (128 parallel environments, 40,000+ FPS):
./run_dgx.sh

# Or train 4,096 parallel robots on GPU in NVIDIA Isaac Lab / Omniverse:
./isaaclab.sh -p isaac_lab/train_isaac_lab.py --num_envs 4096 --robot drone
```
*Full cluster instructions available in [DGX_TRAINING_GUIDE.md](file:///c:/Users/Bisht/OneDrive/Desktop/ADAPT-X/DGX_TRAINING_GUIDE.md).*

---

### 2. Interactive 3D Visual Simulations

#### A. 3D City Drone Flight Simulator (Urban Environment with Skyscrapers & Trees)
Watch the quadcopter UAV navigate through a vibrant 3D city environment featuring high-rise glass skyscrapers, tree-lined avenues, crosswalks, and street lamps, while dynamically recovering from mid-air rotor loss:
```bash
# Launch interactive 3D city flight with mid-air Front-Right rotor failure at step 150:
py -3.14 fly_drone.py --damage dead_motor --rotor fr

# Fly in orbit mode around the city center:
py -3.14 fly_drone.py --mode orbit --damage none

# You can also launch from the unified 3D visualizer:
py -3.14 visualize_3d.py --drone
```

#### B. 3D Quadruped Disaster Arena Simulation
Watch the 12-DoF quadruped robot traverse a post-disaster obstacle course and adapt its gait live when limbs are amputated or motors seize:
```bash
# Watch the 3D simulation with live Front-Right limb amputation at step 150:
py -3.14 visualize_3d.py --damage amputated_leg --leg fr

# Watch with a locked peg leg on Rear-Left:
py -3.14 visualize_3d.py --damage locked_joint --leg rl
```

### 3. Standalone Edge Deployment Simulation
Run the production-ready ONNX or TorchScript policy simulating real hardware edge inference with dynamic limb damage:
```bash
# Run ONNX controller with simulated front-right limb amputation at step 100
py -3.14 deploy.py --model deployments/adaptx_policy.onnx --damage amputated_leg --leg fr

# Run TorchScript controller with a locked joint at step 50
py -3.14 deploy.py --model deployments/adaptx_policy.torchscript.pt --damage locked_joint --leg rl
```

### 3. Re-Export Models (ONNX & TorchScript)
Export any newly trained checkpoint to optimized deployment artifacts:
```bash
py -3.14 export.py --checkpoint checkpoints/phase3/best_policy.pt
```

### 4. Training Pipelines
```bash
# Phase 1: Baseline 4-legged trotting
py -3.14 train.py

# Phase 2: Fault injection training
py -3.14 train.py --config configs/phase2_damage_config.yaml

# Phase 3: Adaptive policy with Temporal History Encoder (RMA)
py -3.14 train_adaptive.py --config configs/phase3_adaptive_config.yaml

# Phase 4: Staged curriculum training (1: Pre-broken, 2: Dynamic, 3: Stress)
py -3.14 train_curriculum.py --stage 1 --resume checkpoints/phase3/best_policy.pt
```

---

## 📊 Benchmarking & Verification

### Inference Latency Profile (ONNX Edge Deployment):
| Metric | Measurement | Target Deadline (50 Hz) | Status |
| :--- | :--- | :--- | :--- |
| **P50 Latency** | **$0.061\text{ ms}$ ($61\ \mu\text{s}$)** | $20.0\text{ ms}$ | **$326\times$ Headroom** |
| **P95 Latency** | **$0.096\text{ ms}$ ($96\ \mu\text{s}$)** | $20.0\text{ ms}$ | **$208\times$ Headroom** |
| **P99 Latency** | **$0.315\text{ ms}$ ($315\ \mu\text{s}$)** | $20.0\text{ ms}$ | **$63\times$ Headroom** |
| **Max Frequency** | **$16,313\text{ Hz}$** | $50\text{ Hz}$ | Passed |
| **Parity Error** | **$3.58 \times 10^{-7}$** | $< 10^{-4}$ | Micro-precision matched |

---

## 📊 State & Action Space

### Observation Space ($s_t \in \mathbb{R}^{48}$):
* **Base Angular Velocity:** $[\omega_x, \omega_y, \omega_z]$ (3)
* **Projected Gravity:** $[g_x, g_y, g_z]$ (3)
* **Commanded Velocities:** $[v_x^{cmd}, v_y^{cmd}, \omega_z^{cmd}]$ (3)
* **Joint Angles Relative to Default:** $q - q_{nom}$ (12)
* **Joint Angular Velocities:** $\dot{q}$ (12)
* **Previous Action Buffer:** $a_{t-1}$ (12)
* **Base Linear Velocity:** $[v_x, v_y, v_z]$ (3)

### Action Space ($a_t \in \mathbb{R}^{12}$):
Normalized continuous target offsets scaled by $0.4\text{ rad}$ and tracked via joint PD controllers:
$$\tau = \text{clip}(K_p(q_{nominal} + a_t \cdot \text{scale} - q) - K_d \dot{q}, -18, 18)$$
