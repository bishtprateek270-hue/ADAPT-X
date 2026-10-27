"""
ADAPT-X: DGX Spark High-Throughput Phased Reinforcement Learning Trainer
Optimized for NVIDIA DGX Spark (NVIDIA Grace Blackwell GB10 GPU / CUDA 13.0).

Divided into 5 distinct training curriculum phases:
  * Phase 1: Baseline Nominal Locomotion / Hover Stabilizer
  * Phase 2: Actuator Fault & Hardware Damage Discovery
  * Phase 3: Adaptive Policy with Temporal History Encoder (RMA Paradigm)
  * Phase 4: Staged Curriculum Resilience & External Push Disturbance Rejection
  * Phase 5: Sim-to-Real Domain Randomization & Production Export (ONNX / TorchScript)
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.quadruped_env import QuadrupedEnv
from envs.drone_env import CityDroneEnv
from envs.sim2real_utils import Sim2RealWrapper
from agents.ppo import ActorCritic
from agents.adaptive_ppo import AdaptiveActorCritic, HistoryEncoder
from export import export_models


class DGXSparkPhasedTrainer:
    """
    Orchestrates end-to-end phased training on NVIDIA DGX Spark.
    Supports both 12-DoF Quadruped and Quadcopter Drone.
    """
    def __init__(self, robot="quadruped", device_id=0, num_envs=32, steps_per_env=256):
        self.robot = robot
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.gpu_name = torch.cuda.get_device_name(device_id) if torch.cuda.is_available() else "CPU"
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env

        os.makedirs("checkpoints/dgx_spark", exist_ok=True)
        os.makedirs("logs/dgx_spark", exist_ok=True)
        os.makedirs("deployments", exist_ok=True)

        print("=" * 80)
        print("       ADAPT-X: NVIDIA DGX SPARK PHASED RL TRAINING SUITE")
        print("=" * 80)
        print(f" Compute Platform  : NVIDIA DGX Spark (Architecture: {os.uname().machine})")
        print(f" Compute Device    : {self.device} ({self.gpu_name})")
        print(f" Target Robot      : {self.robot.upper()}")
        print(f" Parallel Envs     : {self.num_envs} vectorized simulation workers")
        print(f" Steps per Rollout : {self.steps_per_env} steps/worker ({self.num_envs * self.steps_per_env:,} steps/batch)")
        print("=" * 80)

    def _make_quadruped_env(self, phase, stage=1):
        config = {
            "system": {"device": "cuda", "num_cpu_threads": 4},
            "environment": {"target_height": 0.28, "dt": 0.02, "sim_substeps": 10},
            "adaptive": {"history_len": 20, "latent_dim": 16},
            "damage_injection": {
                "enabled": (phase >= 2),
                "failure_prob": 0.7 if phase >= 2 else 0.0,
                "trigger_step_range": [100, 300],
                "failure_types": ["dead_motor", "locked_joint", "weakened_motor", "amputated_leg"],
                "legs": ["fr", "fl", "rr", "rl"]
            },
            "curriculum": {
                "stage": stage if phase == 4 else (2 if phase >= 3 else 1),
                "push_disturbances": (phase == 4 and stage == 3),
                "push_interval": 120,
                "push_magnitude": 15.0
            }
        }
        return QuadrupedEnv(config=config)

    def _make_drone_env(self):
        return CityDroneEnv(target_pos=(0.0, 0.0, 4.5))

    # =========================================================================
    # PHASE 1: Baseline Nominal Locomotion / Hover
    # =========================================================================
    def train_phase1(self, timesteps=100_000, lr=3e-4):
        print("\n" + "=" * 80)
        print("  🟢 PHASE 1: BASELINE NOMINAL DYNAMICS (UNDAMAGED MORPHOLOGY)")
        print("=" * 80)
        print(f" Target Timesteps : {timesteps:,}")
        print(" Objective        : Master steady posture, balance, and velocity tracking.")
        print("-" * 80)

        ckpt_path = "checkpoints/dgx_spark/phase1_best_policy.pt"
        envs = [self._make_quadruped_env(phase=1) if self.robot == "quadruped" else self._make_drone_env() for _ in range(self.num_envs)]
        obs_dim = envs[0].observation_space.shape[0]
        act_dim = envs[0].action_space.shape[0]

        agent = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(self.device)
        optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

        self._run_ppo_loop(envs, agent, optimizer, timesteps, ckpt_path, is_adaptive=False)
        for e in envs: e.close()
        return ckpt_path

    # =========================================================================
    # PHASE 2: Hardware Fault & Damage Space Modeling
    # =========================================================================
    def train_phase2(self, timesteps=120_000, lr=2.5e-4, resume_ckpt="checkpoints/dgx_spark/phase1_best_policy.pt"):
        print("\n" + "=" * 80)
        print("  🟡 PHASE 2: ACTUATOR FAULT & DAMAGE SPACE EXPLORATION")
        print("=" * 80)
        print(f" Target Timesteps : {timesteps:,}")
        print(f" Resume Checkpoint: {resume_ckpt}")
        print(" Objective        : Learn asymmetric compensation under dead, locked, weakened, or severed limbs/rotors.")
        print("-" * 80)

        ckpt_path = "checkpoints/dgx_spark/phase2_best_policy.pt"
        envs = [self._make_quadruped_env(phase=2) if self.robot == "quadruped" else self._make_drone_env() for _ in range(self.num_envs)]
        obs_dim = envs[0].observation_space.shape[0]
        act_dim = envs[0].action_space.shape[0]

        agent = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(self.device)
        if os.path.exists(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location=self.device)
            agent.load_state_dict(ckpt["model_state_dict"])
            print(f"[Phase 2] Warm-started weights from Phase 1 -> {resume_ckpt}")

        optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
        self._run_ppo_loop(envs, agent, optimizer, timesteps, ckpt_path, is_adaptive=False, phase=2)
        for e in envs: e.close()
        return ckpt_path

    # =========================================================================
    # PHASE 3: Adaptive Policy with Temporal History (RMA Paradigm)
    # =========================================================================
    def train_phase3(self, timesteps=150_000, lr=2e-4, resume_ckpt="checkpoints/dgx_spark/phase2_best_policy.pt"):
        print("\n" + "=" * 80)
        print("  🟣 PHASE 3: ADAPTIVE ARCHITECTURE (TEMPORAL PROPRIOCEPTIVE LATENT ENCODER)")
        print("=" * 80)
        print(f" Target Timesteps : {timesteps:,}")
        print(f" Resume Checkpoint: {resume_ckpt}")
        print(" Objective        : Compress rolling H=20 proprioception history into latent z_t ∈ ℝ¹⁶ for zero-shot healing.")
        print("-" * 80)

        ckpt_path = "checkpoints/dgx_spark/phase3_best_policy.pt"
        envs = [self._make_quadruped_env(phase=3) for _ in range(self.num_envs)]
        obs_dim = envs[0].observation_space.shape[0]
        act_dim = envs[0].action_space.shape[0]
        history_dim = envs[0].history_dim

        agent = AdaptiveActorCritic(obs_dim=obs_dim, history_dim=history_dim, act_dim=act_dim, latent_dim=16).to(self.device)
        if os.path.exists(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location=self.device)
            p_state = ckpt["model_state_dict"]
            # Copy compatible weights into actor/critic
            own_state = agent.state_dict()
            matched = 0
            for name, param in p_state.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    matched += 1
            print(f"[Phase 3] Transferred {matched} tensor parameters from {resume_ckpt}")

        optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
        self._run_adaptive_ppo_loop(envs, agent, optimizer, timesteps, ckpt_path, phase=3)
        for e in envs: e.close()
        return ckpt_path

    # =========================================================================
    # PHASE 4: Staged Curriculum Resilience & Push Disturbance Rejection
    # =========================================================================
    def train_phase4(self, timesteps_per_stage=50_000, lr=1.8e-4, resume_ckpt="checkpoints/dgx_spark/phase3_best_policy.pt"):
        print("\n" + "=" * 80)
        print("  🟠 PHASE 4: MULTI-STAGE CURRICULUM RESILIENCE & DISTURBANCE STRESS TEST")
        print("=" * 80)
        print(" Stages:")
        print("   * 4A: Pre-broken 3-legged hopping discovery (spawns 100% damaged)")
        print("   * 4B: Dynamic mid-stride failure transition with forward velocity preservation")
        print("   * 4C: Search & Rescue stress test with external force impulses (15N pushes)")
        print("-" * 80)

        current_ckpt = resume_ckpt
        stages = [
            (1, "Stage 4A: Pre-Broken 3-Legged Hopping"),
            (2, "Stage 4B: Dynamic Mid-Stride Fault Transition"),
            (3, "Stage 4C: Search & Rescue Force Push Impulses")
        ]

        for stage_idx, stage_name in stages:
            print(f"\n>>> Running Phase 4 - {stage_name} ({timesteps_per_stage:,} steps)...")
            ckpt_path = f"checkpoints/dgx_spark/phase4_stage{stage_idx}_best_policy.pt"
            envs = [self._make_quadruped_env(phase=4, stage=stage_idx) for _ in range(self.num_envs)]
            obs_dim = envs[0].observation_space.shape[0]
            act_dim = envs[0].action_space.shape[0]
            history_dim = envs[0].history_dim

            agent = AdaptiveActorCritic(obs_dim=obs_dim, history_dim=history_dim, act_dim=act_dim, latent_dim=16).to(self.device)
            if os.path.exists(current_ckpt):
                ckpt = torch.load(current_ckpt, map_location=self.device)
                agent.load_state_dict(ckpt["model_state_dict"])
                print(f"[Phase 4 - Stage {stage_idx}] Loaded weights from -> {current_ckpt}")

            optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
            self._run_adaptive_ppo_loop(envs, agent, optimizer, timesteps_per_stage, ckpt_path, phase=4, stage=stage_idx)
            for e in envs: e.close()
            current_ckpt = ckpt_path

        final_phase4_ckpt = "checkpoints/dgx_spark/phase4_best_policy.pt"
        torch.save(agent.state_dict(), final_phase4_ckpt)
        return final_phase4_ckpt

    # =========================================================================
    # PHASE 5: Sim-to-Real Domain Randomization & Production Export
    # =========================================================================
    def train_phase5_and_export(self, timesteps=60_000, lr=1e-4, resume_ckpt="checkpoints/dgx_spark/phase4_best_policy.pt"):
        print("\n" + "=" * 80)
        print("  🔵 PHASE 5: SIM-TO-REAL GAP CLOSURE & PRODUCTION MODEL EXPORT")
        print("=" * 80)
        print(" Injected Effects: IMU Sensor Noise, Encoder Noise, CAN Bus Delay, Friction & Mass Variation.")
        print("-" * 80)

        ckpt_path = "checkpoints/dgx_spark/phase5_best_policy.pt"
        envs = [self._make_quadruped_env(phase=4, stage=2) for _ in range(self.num_envs)]
        sim2real_wrappers = [Sim2RealWrapper(action_delay_steps=1, joint_noise_std=0.008, gyro_noise_std=0.015) for _ in range(self.num_envs)]

        obs_dim = envs[0].observation_space.shape[0]
        act_dim = envs[0].action_space.shape[0]
        history_dim = envs[0].history_dim

        agent = AdaptiveActorCritic(obs_dim=obs_dim, history_dim=history_dim, act_dim=act_dim, latent_dim=16).to(self.device)
        if os.path.exists(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            agent.load_state_dict(state_dict)
            print(f"[Phase 5] Loaded weights from -> {resume_ckpt}")

        optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
        self._run_adaptive_ppo_loop(envs, agent, optimizer, timesteps, ckpt_path, phase=5, sim2real=sim2real_wrappers)
        for e in envs: e.close()

        # Export to ONNX and TorchScript for Edge Deployment
        print("\n[Compiling & Exporting Production Artifacts to ONNX & TorchScript...]")
        onnx_out = "deployments/adaptx_dgx_spark_policy.onnx"
        ts_out = "deployments/adaptx_dgx_spark_policy.torchscript.pt"

        export_models(
            checkpoint_path=ckpt_path,
            output_dir="./deployments",
            obs_dim=obs_dim,
            history_dim=history_dim,
            act_dim=act_dim,
            latent_dim=16
        )

        # Rename to dgx_spark specific file names
        if os.path.exists("deployments/adaptx_policy.onnx"):
            import shutil
            shutil.copy("deployments/adaptx_policy.onnx", onnx_out)
            shutil.copy("deployments/adaptx_policy.torchscript.pt", ts_out)

        print(f"  * Production ONNX Model        : {onnx_out}")
        print(f"  * Production TorchScript Model : {ts_out}")
        return onnx_out

    # =========================================================================
    # Internal PPO Rollout & Optimization Loop (Standard Policy)
    # =========================================================================
    def _run_ppo_loop(self, envs, agent, optimizer, total_timesteps, ckpt_path, is_adaptive=False, phase=1):
        num_envs = len(envs)
        obs_list = [envs[i].reset()[0] for i in range(num_envs)]
        obs_dim = envs[0].observation_space.shape[0]
        act_dim = envs[0].action_space.shape[0]

        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        batch_capacity = num_envs * self.steps_per_env
        num_updates = max(1, total_timesteps // batch_capacity)

        start_time = time.time()
        total_steps = 0

        for update in range(1, num_updates + 1):
            b_obs = torch.zeros((self.steps_per_env, num_envs, obs_dim), device=self.device)
            b_act = torch.zeros((self.steps_per_env, num_envs, act_dim), device=self.device)
            b_logp = torch.zeros((self.steps_per_env, num_envs), device=self.device)
            b_rew = torch.zeros((self.steps_per_env, num_envs), device=self.device)
            b_val = torch.zeros((self.steps_per_env, num_envs), device=self.device)
            b_don = torch.zeros((self.steps_per_env, num_envs), device=self.device)

            for step in range(self.steps_per_env):
                b_obs[step] = obs_tensor
                with torch.no_grad():
                    action, logprob, _, val = agent.get_action_and_value(obs_tensor)
                    b_val[step] = val.squeeze(-1)
                b_act[step] = action
                b_logp[step] = logprob

                act_np = action.cpu().numpy()
                next_obs = []
                rewards = []
                dones = []

                for i in range(num_envs):
                    if self.robot == "drone":
                        cmd = np.clip(act_np[i] + 3.40, 0.0, 12.0)
                        if phase >= 2 and step % 80 == 0:
                            envs[i].inject_rotor_fault("dead_motor", "fr")
                    else:
                        cmd = np.clip(act_np[i], -1.0, 1.0)

                    o, r, term, trunc, _ = envs[i].step(cmd)
                    d = term or trunc
                    if d:
                        o, _ = envs[i].reset()
                    next_obs.append(o)
                    rewards.append(r)
                    dones.append(d)

                b_rew[step] = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                b_don[step] = torch.tensor(dones, dtype=torch.float32, device=self.device)
                obs_tensor = torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device)
                total_steps += num_envs

            # GAE computation & optimization
            self._optimize_ppo(agent, optimizer, b_obs, b_act, b_logp, b_rew, b_don, b_val, obs_tensor)

            elapsed = time.time() - start_time
            fps = int(total_steps / max(1e-3, elapsed))
            mean_rew = b_rew.mean().item() * self.steps_per_env

            if update % max(1, num_updates // 5) == 0 or update == num_updates:
                print(f"  [DGX Spark] Update [{update:3d}/{num_updates:3d}] | Steps: {total_steps:7,d} | Avg Ep Rew: {mean_rew:+7.2f} | FPS: {fps:6,d} | Time: {elapsed:5.1f}s")

        torch.save({"model_state_dict": agent.state_dict()}, ckpt_path)
        print(f"  [Checkpoint Saved] -> {ckpt_path}")

    # =========================================================================
    # Internal PPO Rollout & Optimization Loop (Adaptive RMA Policy)
    # =========================================================================
    def _run_adaptive_ppo_loop(self, envs, agent, optimizer, total_timesteps, ckpt_path, phase=3, stage=1, sim2real=None):
        num_envs = len(envs)
        obs_dim = envs[0].observation_space.shape[0]
        act_dim = envs[0].action_space.shape[0]
        history_dim = envs[0].history_dim

        obs_list, hist_list = [], []
        for i in range(num_envs):
            o, h, _ = envs[i].reset_adaptive()
            if sim2real:
                sim2real[i].reset_buffer()
                sim2real[i].randomize_physics(envs[i].model, envs[i].data)
                o = sim2real[i].add_sensor_noise(o)
            obs_list.append(o)
            hist_list.append(h)

        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        hist_tensor = torch.tensor(np.array(hist_list), dtype=torch.float32, device=self.device)

        batch_capacity = num_envs * self.steps_per_env
        num_updates = max(1, total_timesteps // batch_capacity)

        start_time = time.time()
        total_steps = 0

        for update in range(1, num_updates + 1):
            b_obs = torch.zeros((self.steps_per_env, num_envs, obs_dim), device=self.device)
            b_hist = torch.zeros((self.steps_per_env, num_envs, history_dim), device=self.device)
            b_act = torch.zeros((self.steps_per_env, num_envs, act_dim), device=self.device)
            b_logp = torch.zeros((self.steps_per_env, num_envs), device=self.device)
            b_rew = torch.zeros((self.steps_per_env, num_envs), device=self.device)
            b_val = torch.zeros((self.steps_per_env, num_envs), device=self.device)
            b_don = torch.zeros((self.steps_per_env, num_envs), device=self.device)

            for step in range(self.steps_per_env):
                b_obs[step] = obs_tensor
                b_hist[step] = hist_tensor

                with torch.no_grad():
                    action, logprob, _, val, _ = agent.get_action_and_value(obs_tensor, hist_tensor)
                    b_val[step] = val.squeeze(-1)
                b_act[step] = action
                b_logp[step] = logprob

                act_np = action.cpu().numpy()
                next_obs, next_hist = [], []
                rewards, dones = [], []

                for i in range(num_envs):
                    raw_cmd = np.clip(act_np[i], -1.0, 1.0)
                    cmd = sim2real[i].apply_action_delay(raw_cmd) if sim2real else raw_cmd

                    o, h, r, term, trunc, _ = envs[i].step_adaptive(cmd)
                    d = term or trunc
                    if d:
                        o, h, _ = envs[i].reset_adaptive()
                        if sim2real:
                            sim2real[i].reset_buffer()
                            sim2real[i].randomize_physics(envs[i].model, envs[i].data)

                    if sim2real:
                        o = sim2real[i].add_sensor_noise(o)

                    next_obs.append(o)
                    next_hist.append(h)
                    rewards.append(r)
                    dones.append(d)

                b_rew[step] = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                b_don[step] = torch.tensor(dones, dtype=torch.float32, device=self.device)
                obs_tensor = torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device)
                hist_tensor = torch.tensor(np.array(next_hist), dtype=torch.float32, device=self.device)
                total_steps += num_envs

            # GAE computation & optimization
            self._optimize_adaptive_ppo(agent, optimizer, b_obs, b_hist, b_act, b_logp, b_rew, b_don, b_val, obs_tensor, hist_tensor)

            elapsed = time.time() - start_time
            fps = int(total_steps / max(1e-3, elapsed))
            mean_rew = b_rew.mean().item() * self.steps_per_env

            if update % max(1, num_updates // 5) == 0 or update == num_updates:
                print(f"  [DGX Spark] Update [{update:3d}/{num_updates:3d}] | Steps: {total_steps:7,d} | Avg Ep Rew: {mean_rew:+7.2f} | FPS: {fps:6,d} | Time: {elapsed:5.1f}s")

        torch.save({"model_state_dict": agent.state_dict()}, ckpt_path)
        print(f"  [Checkpoint Saved] -> {ckpt_path}")

    def _optimize_ppo(self, agent, optimizer, b_obs, b_act, b_logp, b_rew, b_don, b_val, last_obs):
        with torch.no_grad():
            next_val = agent.get_value(last_obs).squeeze(-1)
            adv = torch.zeros_like(b_rew)
            lastgaelam = 0.0
            for t in reversed(range(self.steps_per_env)):
                nextnonterminal = 1.0 - b_don[t] if t == self.steps_per_env - 1 else 1.0 - b_don[t + 1]
                nextval = next_val if t == self.steps_per_env - 1 else b_val[t + 1]
                delta = b_rew[t] + 0.99 * nextval * nextnonterminal - b_val[t]
                adv[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            ret = adv + b_val

        b_obs_f = b_obs.reshape(-1, b_obs.shape[-1])
        b_act_f = b_act.reshape(-1, b_act.shape[-1])
        b_logp_f = b_logp.reshape(-1)
        adv_f = (adv.reshape(-1) - adv.mean()) / (adv.std() + 1e-8)
        ret_f = ret.reshape(-1)

        b_sz = b_obs_f.shape[0]
        inds = np.arange(b_sz)
        for _ in range(4):
            np.random.shuffle(inds)
            for s in range(0, b_sz, 256):
                mb = inds[s:s + 256]
                _, newlogp, ent, newval = agent.get_action_and_value(b_obs_f[mb], b_act_f[mb])
                ratio = (newlogp - b_logp_f[mb]).exp()
                mb_adv = adv_f[mb]
                pg = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                vf = 0.5 * ((newval.view(-1) - ret_f[mb]) ** 2).mean()
                loss = pg - 0.01 * ent.mean() + 0.5 * vf
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

    def _optimize_adaptive_ppo(self, agent, optimizer, b_obs, b_hist, b_act, b_logp, b_rew, b_don, b_val, last_obs, last_hist):
        with torch.no_grad():
            next_val = agent.get_value(last_obs, last_hist).squeeze(-1)
            adv = torch.zeros_like(b_rew)
            lastgaelam = 0.0
            for t in reversed(range(self.steps_per_env)):
                nextnonterminal = 1.0 - b_don[t] if t == self.steps_per_env - 1 else 1.0 - b_don[t + 1]
                nextval = next_val if t == self.steps_per_env - 1 else b_val[t + 1]
                delta = b_rew[t] + 0.99 * nextval * nextnonterminal - b_val[t]
                adv[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            ret = adv + b_val

        b_obs_f = b_obs.reshape(-1, b_obs.shape[-1])
        b_hist_f = b_hist.reshape(-1, b_hist.shape[-1])
        b_act_f = b_act.reshape(-1, b_act.shape[-1])
        b_logp_f = b_logp.reshape(-1)
        adv_f = (adv.reshape(-1) - adv.mean()) / (adv.std() + 1e-8)
        ret_f = ret.reshape(-1)

        b_sz = b_obs_f.shape[0]
        inds = np.arange(b_sz)
        for _ in range(4):
            np.random.shuffle(inds)
            for s in range(0, b_sz, 256):
                mb = inds[s:s + 256]
                _, newlogp, ent, newval, _ = agent.get_action_and_value(b_obs_f[mb], b_hist_f[mb], b_act_f[mb])
                ratio = (newlogp - b_logp_f[mb]).exp()
                mb_adv = adv_f[mb]
                pg = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                vf = 0.5 * ((newval.view(-1) - ret_f[mb]) ** 2).mean()
                loss = pg - 0.015 * ent.mean() + 0.5 * vf
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="ADAPT-X: NVIDIA DGX Spark Phased RL Trainer")
    parser.add_argument("--robot", type=str, default="quadruped", choices=["quadruped", "drone"], help="Robot to train")
    parser.add_argument("--phase", type=str, default="all", choices=["1", "2", "3", "4", "5", "all"], help="Training phase (1 to 5 or 'all')")
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel environments on DGX Spark")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--fast", action="store_true", help="Fast training mode (scaled timesteps for verification)")
    args = parser.parse_args()

    scale = 0.15 if args.fast else 1.0

    trainer = DGXSparkPhasedTrainer(
        robot=args.robot,
        device_id=args.device,
        num_envs=args.num_envs,
        steps_per_env=256
    )

    t0 = time.time()
    ckpt_p1 = "checkpoints/dgx_spark/phase1_best_policy.pt"
    ckpt_p2 = "checkpoints/dgx_spark/phase2_best_policy.pt"
    ckpt_p3 = "checkpoints/dgx_spark/phase3_best_policy.pt"
    ckpt_p4 = "checkpoints/dgx_spark/phase4_best_policy.pt"

    if args.phase in ["1", "all"]:
        ckpt_p1 = trainer.train_phase1(timesteps=int(100_000 * scale))
    if args.phase in ["2", "all"]:
        ckpt_p2 = trainer.train_phase2(timesteps=int(120_000 * scale), resume_ckpt=ckpt_p1)
    if args.phase in ["3", "all"]:
        ckpt_p3 = trainer.train_phase3(timesteps=int(150_000 * scale), resume_ckpt=ckpt_p2)
    if args.phase in ["4", "all"]:
        ckpt_p4 = trainer.train_phase4(timesteps_per_stage=int(50_000 * scale), resume_ckpt=ckpt_p3)
    if args.phase in ["5", "all"]:
        trainer.train_phase5_and_export(timesteps=int(60_000 * scale), resume_ckpt=ckpt_p4)

    print("\n" + "=" * 80)
    print(f"🎉 All Selected ADAPT-X DGX Spark Training Phases Completed in {time.time() - t0:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
