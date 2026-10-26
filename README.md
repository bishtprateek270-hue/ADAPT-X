# ADAPT-X: The "Self-Healing" Quadruped Robot
### Sim-to-Real Reinforcement Learning for Dynamic Damage Adaptation

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2Bcu128-red.svg)
![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo%203.17-orange.svg)
![Status](https://img.shields.io/badge/Status-Phase%204%20Curriculum%20Active-green.svg)

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
- [ ] **Phase 5: Sim-to-Real Readiness & Model Export (ONNX)**

---

## 🚀 Quick Start

### 1. Installation
Ensure Python and required packages are installed:
```bash
pip install torch gymnasium mujoco pyyaml numpy
```

### 2. Training Phase 4: Staged Curriculum Locomotion
Warm-start from the Phase 3 adaptive model and discover true 3-legged marching gaits:
```bash
# Stage 1: Pre-broken 3-leg locomotion discovery
py -3.14 train_curriculum.py --stage 1 --resume checkpoints/phase3/best_policy.pt

# Stage 2: Dynamic mid-stride failure transition with forward momentum
py -3.14 train_curriculum.py --stage 2 --resume checkpoints/phase4/stage_1/best_policy.pt

# Stage 3: Extreme Search & Rescue stress testing with push impulses
py -3.14 train_curriculum.py --stage 3 --resume checkpoints/phase4/stage_2/best_policy.pt
```

### 3. Evaluating 3-Legged Velocity Retention
Benchmark how fast the quadruped marches using only 3 functional limbs:
```bash
# Evaluate Stage 1 (Pre-broken 3-leg gait)
py -3.14 eval_curriculum.py --checkpoint checkpoints/phase4/stage_1/best_policy.pt --stage 1 --damage amputated_leg --leg fr

# Evaluate Stage 2 (Dynamic transition + Velocity retention)
py -3.14 eval_curriculum.py --checkpoint checkpoints/phase4/stage_2/best_policy.pt --stage 2 --damage amputated_leg --leg fr
```
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
