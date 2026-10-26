# ADAPT-X: The "Self-Healing" Quadruped Robot
### Sim-to-Real Reinforcement Learning for Dynamic Damage Adaptation

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2Bcu128-red.svg)
![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo%203.17-orange.svg)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-green.svg)

---

## 📌 Project Overview
When rovers, drones, or legged robots in remote environments (such as disaster zones or planetary exploration) suffer hardware failure, traditional pre-programmed controllers fail because they cannot operate an altered morphology.

**ADAPT-X** is an autonomous 12-DoF quadruped robot controlled by an adaptive Reinforcement Learning (RL) policy. Rather than shutting down upon limb or actuator damage, the agent detects changes through proprioceptive feedback (IMUs + joint encoders) and dynamically transitions its gait to balance and continue locomotion using remaining functional limbs.

---

## 🛠️ Architecture & Tech Stack
* **Physics Simulator:** MuJoCo 3.17 (High-fidelity contact dynamics at 500 Hz).
* **Environment Interface:** Gymnasium (Custom 12-DoF Quadruped MDP).
* **RL Framework:** PyTorch PPO (Proximal Policy Optimization) with Generalized Advantage Estimation (GAE).
* **Policy Architecture:** Compact Actor-Critic MLP (`[48 -> 256 -> 128 -> 12]`) designed for microsecond inference and hardware safety.
* **Control Loop:** 50 Hz policy frequency with low-level PD position control ($K_p = 25.0, K_d = 0.8$).

---

## 🗺️ Project Roadmap
- [x] **Phase 1: Baseline 4-Leg Locomotion & Reward Shaping**
  - [x] 12-DoF Quadruped Model (`models/quadruped.xml`)
  - [x] Gymnasium Environment (`envs/quadruped_env.py`)
  - [x] PPO Trainer & Model (`agents/ppo.py`)
  - [x] Laptop GPU (RTX 5050) Safe Execution Profile (< 25MB VRAM)
  - [x] Evaluation & Velocity Tracking Benchmark (`eval.py`)
- [ ] **Phase 2: Failure Space & Damage Injection**
  - [ ] Dead motor (zero torque / free swing)
  - [ ] Locked joint (rigid peg mode)
  - [ ] Dynamic limb fracture injection mid-episode
- [ ] **Phase 3: Adaptive Policy Architecture (Self-Healing Intelligence)**
  - [ ] Temporal History Proprioceptive Encoder (RMA / Teacher-Student)
  - [ ] Latent Space morphological state estimation
- [ ] **Phase 4: Curriculum Learning & 3-Legged Recovery Benchmark**
- [ ] **Phase 5: Sim-to-Real Readiness & Model Export (ONNX)**

---

## 🚀 Quick Start

### 1. Installation
Ensure Python and required packages are installed:
```bash
pip install torch gymnasium mujoco pyyaml numpy
```

### 2. Verify Setup (Dry-Run)
```bash
py -3.14 train.py --dry-run
```

### 3. Training Baseline Locomotion (Phase 1)
```bash
# Full training (500k steps, ~15 minutes on RTX 5050)
py -3.14 train.py

# Or custom steps
py -3.14 train.py --steps 100000
```

### 4. Evaluate Trained Policy
```bash
py -3.14 eval.py --checkpoint checkpoints/best_policy.pt --episodes 5
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
