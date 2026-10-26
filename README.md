# ADAPT-X: The "Self-Healing" Quadruped Robot
### Sim-to-Real Reinforcement Learning for Dynamic Damage Adaptation

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2Bcu128-red.svg)
![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo%203.17-orange.svg)
![Status](https://img.shields.io/badge/Status-Phase%202%20Active-green.svg)

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
- [x] **Phase 2: Failure Space & Damage Modeling**
  - [x] Dead motor mode (zero torque / limp joint)
  - [x] Locked joint mode (frozen angle / rigid peg leg)
  - [x] Weakened actuator mode (25% torque output / thermal throttle)
  - [x] Amputated limb mode (power cut to complete 3-joint leg)
  - [x] Randomized mid-episode fault injector (`trigger_step_range: [150, 400]`)
  - [x] Deterministic failure benchmarking CLI (`eval.py --damage ...`)
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

### 2. Training Phase 2: Fault-Tolerant Locomotion
Train the policy under randomized dynamic mid-episode hardware failures:
```bash
# Fine-tune with Phase 2 damage injection (from healthy Phase 1 checkpoint)
py -3.14 train.py --config configs/phase2_damage_config.yaml --resume checkpoints/best_policy.pt

# Or train Phase 2 from scratch (500k steps, ~15 mins on RTX 5050)
py -3.14 train.py --config configs/phase2_damage_config.yaml
```

### 3. Evaluating Failure Resilience
Benchmark the robot against specific hardware damage:
```bash
# Test dead motor on Front-Right leg at step 200
py -3.14 eval.py --checkpoint checkpoints/phase2/best_policy.pt --damage dead_motor --leg fr

# Test locked joint (frozen peg leg) on Rear-Left leg
py -3.14 eval.py --checkpoint checkpoints/phase2/best_policy.pt --damage locked_joint --leg rl

# Test full limb power loss
py -3.14 eval.py --checkpoint checkpoints/phase2/best_policy.pt --damage amputated_leg --leg fl
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
