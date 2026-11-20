"""
ADAPT-X: Universal Scaled Drone Phased RL Training Suite (~25.6M Parameters)
Optimized for NVIDIA DGX Spark / Grace Blackwell GB10 GPU with BFloat16 Tensor Cores.

Trained across the full spectrum of world quadrotor morphologies:
  - Class 1 (Nano / FPV Micro): 100g - 450g, 6cm - 13cm arms, high KV, agile
  - Class 2 (Compact / Consumer / Inspection): 450g - 2.5kg, 14cm - 28cm arms (Mavic/Skydio style)
  - Class 3 (Commercial / Industrial / Enterprise): 2.5kg - 7.5kg, 28cm - 48cm arms (Matrice style)
  - Class 4 (Heavy Industrial / Tactical / BARC Lifter): 7.5kg - 20kg+, 48cm - 75cm arms

Online Parameter 'Fetching' & Adaptation:
  - Deep Multi-Head Temporal Transformer (H=20 timesteps, d_model=512, 8 heads, 4 layers)
  - Auxiliary Online System Identification Head predicting:
      [mass, arm_length, I_xx, I_yy, I_zz, hover_thrust, max_thrust] in R^7
  - Wide 1024-wide 4-Block Residual Actor-Critic Trunks
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

from envs.drone_env import CityDroneEnv
from envs.sim2real_utils import Sim2RealWrapper
from agents.heavy_adaptive_ppo import (
    UniversalAdaptiveActorCritic,
    UniversalDeployablePolicy
)


class UniversalDroneTrainer:
    """
    High-Throughput Universal Drone Reinforcement Learning Trainer.
    Trains across arbitrary drone dimensions, weights, and motor dynamics.
    """
    def __init__(self, device_id=0, num_envs=32, steps_per_env=256, use_amp=True, morph_profile="all"):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.gpu_name = torch.cuda.get_device_name(device_id) if torch.cuda.is_available() else "CPU"
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.morph_profile = morph_profile
        self.use_amp = use_amp and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        os.makedirs("checkpoints/dgx_spark_universal", exist_ok=True)
        os.makedirs("logs/dgx_spark_universal", exist_ok=True)
        os.makedirs("deployments", exist_ok=True)

        print("=" * 90)
        print("    ADAPT-X: UNIVERSAL SCALED DRONE REINFORCEMENT LEARNING TRAINER")
        print("=" * 90)
        print(f" Compute Platform  : NVIDIA DGX Spark ({os.uname().machine})")
        print(f" Acceleration Core : {self.device} ({self.gpu_name})")
        print(f" Precision Engine  : {'CUDA AMP (' + str(self.amp_dtype).split('.')[-1] + ')' if self.use_amp else 'FP32'}")
        print(f" Target Drone Class: FULL WORLD SPECTRUM (100g to 20kg, 6cm to 75cm arm radius)")
        print(f" Parallel Workers  : {self.num_envs} vectorized simulation workers")
        print(f" Batch Capacity    : {self.num_envs * self.steps_per_env:,} steps per policy update")
        print("=" * 90)

    def _make_env(self, phase, env_idx=0):
        env = CityDroneEnv(target_pos=(0.0, 0.0, 4.5), max_radius=28.0, max_altitude=16.0)
        env.morphology_randomization = True
        env.action_mode = "normalized"

        # Stagger profiles across workers if 'all' is selected
        profiles = ["nano_fpv", "compact_inspection", "commercial_enterprise", "heavy_industrial_barc"]
        if self.morph_profile == "all":
            profile = profiles[env_idx % len(profiles)]
        else:
            profile = self.morph_profile

        env.sample_universal_morphology(profile=profile)
        return env

    def _init_universal_model(self, env):
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        single_proprio_dim = env.single_proprio_dim
        history_len = env.history_len

        model = UniversalAdaptiveActorCritic(
            obs_dim=obs_dim,
            single_proprio_dim=single_proprio_dim,
            history_len=history_len,
            act_dim=act_dim,
            latent_dim=64,
            d_model=512,
            nhead=8,
            num_trans_layers=4,
            hidden_dim=1024,
            num_res_blocks=4,
            morph_dim=7
        ).to(self.device)

        total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model Initialized] Universal Actor-Critic Architecture: {total_p:,} parameters ({total_p/1e6:.2f} Million)")
        return model

    # =========================================================================
    # PHASE 1: Universal Baseline Nominal Dynamics (100g to 20kg Drones)
    # =========================================================================
    def train_phase1(self, timesteps=100_000, lr=3e-4):
        print("\n" + "=" * 90)
        print("  🟢 PHASE 1: UNIVERSAL BASELINE NOMINAL DYNAMICS (ALL WEIGHT & DIM CLASSES)")
        print("=" * 90)
        print(f" Target Timesteps : {timesteps:,}")
        print(" Objective        : Learn normalized attitude hold and position tracking across all drone sizes.")
        print("-" * 90)

        ckpt_path = "checkpoints/dgx_spark_universal/drone_phase1_best_policy.pt"
        envs = [self._make_env(phase=1, env_idx=i) for i in range(self.num_envs)]
        model = self._init_universal_model(envs[0])
        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)

        self._run_training_loop(envs, model, optimizer, timesteps, ckpt_path, phase=1)
        for e in envs: e.close()
        return ckpt_path

    # =========================================================================
    # PHASE 2: Full-Spectrum Actuator Fault & Mid-Air Rotor Failure
    # =========================================================================
    def train_phase2(self, timesteps=120_000, lr=2.5e-4, resume_ckpt="checkpoints/dgx_spark_universal/drone_phase1_best_policy.pt"):
        print("\n" + "=" * 90)
        print("  🟡 PHASE 2: MID-AIR HARDWARE DAMAGE & ROTOR LOSS EXPLORATION")
        print("=" * 90)
        print(f" Target Timesteps : {timesteps:,}")
        print(f" Resume Model     : {resume_ckpt}")
        print(" Fault Modes      : Sudden mid-air loss of 1 rotor (0% thrust) & damaged blade (50% thrust).")
        print("-" * 90)

        ckpt_path = "checkpoints/dgx_spark_universal/drone_phase2_best_policy.pt"
        envs = [self._make_env(phase=2, env_idx=i) for i in range(self.num_envs)]
        model = self._init_universal_model(envs[0])

        if os.path.exists(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Phase 2] Successfully warm-started weights from -> {resume_ckpt}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)
        self._run_training_loop(envs, model, optimizer, timesteps, ckpt_path, phase=2)
        for e in envs: e.close()
        return ckpt_path

    # =========================================================================
    # PHASE 3: Online Parameter 'Fetching' & Rapid Motor Adaptation
    # =========================================================================
    def train_phase3(self, timesteps=150_000, lr=2e-4, resume_ckpt="checkpoints/dgx_spark_universal/drone_phase2_best_policy.pt"):
        print("\n" + "=" * 90)
        print("  🟣 PHASE 3: ONLINE PARAMETER 'FETCHING' & RAPID MOTOR ADAPTATION (RMA)")
        print("=" * 90)
        print(f" Target Timesteps : {timesteps:,}")
        print(f" Resume Model     : {resume_ckpt}")
        print(" Objective        : Dual-loss training (PPO + SysID). Automatically decodes [mass, arm, I_xx, I_yy, I_zz]")
        print("                    from high-frequency IMU history in real time.")
        print("-" * 90)

        ckpt_path = "checkpoints/dgx_spark_universal/drone_phase3_best_policy.pt"
        envs = [self._make_env(phase=3, env_idx=i) for i in range(self.num_envs)]
        model = self._init_universal_model(envs[0])

        if os.path.exists(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Phase 3] Transferred weights from -> {resume_ckpt}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)
        self._run_training_loop(envs, model, optimizer, timesteps, ckpt_path, phase=3, train_sysid=True)
        for e in envs: e.close()
        return ckpt_path

    # =========================================================================
    # PHASE 4: Multi-Curriculum Turbulence & Heavy Push Disturbances
    # =========================================================================
    def train_phase4(self, timesteps_per_stage=50_000, lr=1.8e-4, resume_ckpt="checkpoints/dgx_spark_universal/drone_phase3_best_policy.pt"):
        print("\n" + "=" * 90)
        print("  🟠 PHASE 4: STAGED CURRICULUM RESILIENCE & SEVERE WIND DISTURBANCE")
        print("=" * 90)

        current_ckpt = resume_ckpt
        stages = [
            (1, "Stage 4A: Pre-Damaged 3-Rotor Spawn Recovery"),
            (2, "Stage 4B: Dynamic Mid-Air Fault Transition with Obstacle Avoidance"),
            (3, "Stage 4C: Extreme Wind Gust Impulses (15N - 30N Pushes)")
        ]

        for stage_idx, stage_name in stages:
            print(f"\n>>> Running Phase 4 - {stage_name} ({timesteps_per_stage:,} steps)...")
            ckpt_path = f"checkpoints/dgx_spark_universal/drone_phase4_stage{stage_idx}_best_policy.pt"
            envs = [self._make_env(phase=4, env_idx=i) for i in range(self.num_envs)]
            model = self._init_universal_model(envs[0])

            if os.path.exists(current_ckpt):
                ckpt = torch.load(current_ckpt, map_location=self.device)
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"[Stage {stage_idx}] Loaded weights from -> {current_ckpt}")

            optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)
            self._run_training_loop(envs, model, optimizer, timesteps_per_stage, ckpt_path, phase=4, stage=stage_idx, train_sysid=True)
            for e in envs: e.close()
            current_ckpt = ckpt_path

        final_phase4_ckpt = "checkpoints/dgx_spark_universal/drone_phase4_best_policy.pt"
        torch.save({"model_state_dict": model.state_dict()}, final_phase4_ckpt)
        return final_phase4_ckpt

    # =========================================================================
    # PHASE 5: Sim-to-Real Domain Gap Closure & Production Model Export
    # =========================================================================
    def train_phase5_and_export(self, timesteps=60_000, lr=1e-4, resume_ckpt="checkpoints/dgx_spark_universal/drone_phase4_best_policy.pt"):
        print("\n" + "=" * 90)
        print("  🔵 PHASE 5: SIM-TO-REAL GAP CLOSURE & PRODUCTION MODEL EXPORT")
        print("=" * 90)
        print(" Injected Effects: IMU Sensor Noise, ESC Delay Jitter, CoM Offsets & Aerodynamic Drag Randomization.")
        print("-" * 90)

        ckpt_path = "checkpoints/dgx_spark_universal/drone_phase5_best_policy.pt"
        envs = [self._make_env(phase=5, env_idx=i) for i in range(self.num_envs)]
        sim2real_wrappers = [Sim2RealWrapper(action_delay_steps=1, joint_noise_std=0.01, gyro_noise_std=0.02) for _ in range(self.num_envs)]

        model = self._init_universal_model(envs[0])
        if os.path.exists(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Phase 5] Loaded weights from -> {resume_ckpt}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)
        self._run_training_loop(envs, model, optimizer, timesteps, ckpt_path, phase=5, sim2real=sim2real_wrappers, train_sysid=True)
        for e in envs: e.close()

        # Production compilation & export
        print("\n[Exporting Universal Production Artifacts to ONNX & TorchScript...]")
        onnx_out = "deployments/adaptx_universal_drone.onnx"
        ts_out = "deployments/adaptx_universal_drone.torchscript.pt"
        self._export_universal_model(model, envs[0], onnx_out, ts_out)
        return onnx_out

    # =========================================================================
    # Universal Rollout & PPO + SysID Optimization Loop
    # =========================================================================
    def _run_training_loop(self, envs, model, optimizer, total_timesteps, ckpt_path, phase=1, stage=1, sim2real=None, train_sysid=False):
        num_envs = len(envs)
        obs_dim = envs[0].observation_space.shape[0]
        act_dim = envs[0].action_space.shape[0]
        history_dim = envs[0].history_dim
        morph_dim = 7

        obs_list, hist_list, morph_list = [], [], []
        for i in range(num_envs):
            o, h, _ = envs[i].reset_adaptive()
            if sim2real:
                sim2real[i].reset_buffer()
                o = sim2real[i].add_sensor_noise(o)
            obs_list.append(o)
            hist_list.append(h)
            morph_list.append(envs[i].get_true_morphology())

        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        hist_tensor = torch.tensor(np.array(hist_list), dtype=torch.float32, device=self.device)
        morph_tensor = torch.tensor(np.array(morph_list), dtype=torch.float32, device=self.device)

        batch_capacity = num_envs * self.steps_per_env
        num_updates = max(1, total_timesteps // batch_capacity)

        start_time = time.time()
        total_steps = 0

        for update in range(1, num_updates + 1):
            b_obs = torch.zeros((self.steps_per_env, num_envs, obs_dim), device=self.device)
            b_hist = torch.zeros((self.steps_per_env, num_envs, history_dim), device=self.device)
            b_morph = torch.zeros((self.steps_per_env, num_envs, morph_dim), device=self.device)
            b_act = torch.zeros((self.steps_per_env, num_envs, act_dim), device=self.device)
            b_logp = torch.zeros((self.steps_per_env, num_envs), device=self.device)
            b_rew = torch.zeros((self.steps_per_env, num_envs), device=self.device)
            b_val = torch.zeros((self.steps_per_env, num_envs), device=self.device)
            b_don = torch.zeros((self.steps_per_env, num_envs), device=self.device)

            latent_magnitudes = []
            sysid_errors = []

            for step in range(self.steps_per_env):
                b_obs[step] = obs_tensor
                b_hist[step] = hist_tensor
                b_morph[step] = morph_tensor

                with torch.no_grad():
                    if self.use_amp:
                        with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                            action, logprob, _, val, z, pred_morph = model.get_action_and_value(obs_tensor, hist_tensor)
                    else:
                        action, logprob, _, val, z, pred_morph = model.get_action_and_value(obs_tensor, hist_tensor)

                    b_val[step] = val.squeeze(-1)
                    latent_magnitudes.append(torch.norm(z, dim=-1).mean().item())
                    # Track mass inference relative error
                    err_mass = torch.abs(pred_morph[:, 0] - morph_tensor[:, 0]).mean().item()
                    sysid_errors.append(err_mass)

                b_act[step] = action
                b_logp[step] = logprob

                act_np = action.float().cpu().numpy()
                next_obs, next_hist, next_morph = [], [], []
                rewards, dones = [], []

                for i in range(num_envs):
                    raw_cmd = np.clip(act_np[i], -1.0, 1.0)
                    cmd = sim2real[i].apply_action_delay(raw_cmd) if sim2real else raw_cmd

                    # Phase-specific fault injection
                    if phase == 2 and np.random.rand() < 0.05 and envs[i].fault_type == "none":
                        victim = np.random.choice(["fr", "fl", "rr", "rl"])
                        fault = "dead_motor" if np.random.rand() < 0.8 else "damaged_blade"
                        envs[i].inject_rotor_fault(fault, victim)

                    if phase == 4:
                        if stage == 1 and envs[i].current_step == 0:
                            # Pre-damaged 3-rotor spawn
                            envs[i].inject_rotor_fault("dead_motor", "fr")
                        elif stage == 2 and np.random.rand() < 0.04 and envs[i].fault_type == "none":
                            victim = np.random.choice(["fr", "fl", "rr", "rl"])
                            envs[i].inject_rotor_fault("dead_motor", victim)
                        elif stage == 3 and step % 40 == 0:
                            envs[i].apply_wind_gust(18.0)

                    o, h, r, term, trunc, info = envs[i].step_adaptive(cmd)
                    d = term or trunc

                    if d:
                        o, h, _ = envs[i].reset_adaptive()
                        if sim2real:
                            sim2real[i].reset_buffer()

                    if sim2real:
                        o = sim2real[i].add_sensor_noise(o)

                    next_obs.append(o)
                    next_hist.append(h)
                    next_morph.append(envs[i].get_true_morphology())
                    rewards.append(r)
                    dones.append(d)

                b_rew[step] = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                b_don[step] = torch.tensor(dones, dtype=torch.float32, device=self.device)
                obs_tensor = torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device)
                hist_tensor = torch.tensor(np.array(next_hist), dtype=torch.float32, device=self.device)
                morph_tensor = torch.tensor(np.array(next_morph), dtype=torch.float32, device=self.device)
                total_steps += num_envs

            # GAE computation & optimization with AMP
            loss_info = self._optimize_universal_ppo(
                model, optimizer, b_obs, b_hist, b_morph, b_act, b_logp, b_rew, b_don, b_val,
                obs_tensor, hist_tensor, train_sysid=train_sysid
            )

            elapsed = time.time() - start_time
            fps = int(total_steps / max(1e-3, elapsed))
            mean_rew = b_rew.mean().item() * self.steps_per_env
            mean_z = np.mean(latent_magnitudes) if latent_magnitudes else 0.0
            mean_sysid_err = np.mean(sysid_errors) if sysid_errors else 0.0

            if update % max(1, num_updates // 5) == 0 or update == num_updates:
                vram_used = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
                print(f"  [Universal DGX] Update [{update:3d}/{num_updates:3d}] | Steps: {total_steps:7,d} | Ep Rew: {mean_rew:+7.2f} | Loss: {loss_info['loss']:.4f} | Mass Err: {mean_sysid_err:4.2f}kg | ||z||: {mean_z:.2f} | FPS: {fps:6,d} | VRAM: {vram_used:.1f}MB | Time: {elapsed:5.1f}s")

        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        print(f"  [Universal Checkpoint Saved] -> {ckpt_path}")

    def _optimize_universal_ppo(self, model, optimizer, b_obs, b_hist, b_morph, b_act, b_logp, b_rew, b_don, b_val, last_obs, last_hist, train_sysid=False):
        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    next_val = model.get_value(last_obs, last_hist).squeeze(-1)
            else:
                next_val = model.get_value(last_obs, last_hist).squeeze(-1)

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
        b_morph_f = b_morph.reshape(-1, b_morph.shape[-1])
        b_act_f = b_act.reshape(-1, b_act.shape[-1])
        b_logp_f = b_logp.reshape(-1)
        adv_f = (adv.reshape(-1) - adv.mean()) / (adv.std() + 1e-8)
        ret_f = ret.reshape(-1)

        b_sz = b_obs_f.shape[0]
        inds = np.arange(b_sz)
        total_loss = 0.0

        for _ in range(4):
            np.random.shuffle(inds)
            for s in range(0, b_sz, 256):
                mb = inds[s:s + 256]

                if self.use_amp:
                    with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                        _, newlogp, ent, newval, _, pred_morph = model.get_action_and_value(b_obs_f[mb], b_hist_f[mb], b_act_f[mb])
                        ratio = (newlogp - b_logp_f[mb]).exp()
                        mb_adv = adv_f[mb]
                        pg = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                        vf = 0.5 * ((newval.view(-1) - ret_f[mb]) ** 2).mean()

                        loss = pg - 0.015 * ent.mean() + 0.5 * vf
                        if train_sysid:
                            # Online physical morphology estimation loss
                            sysid_loss = nn.functional.mse_loss(pred_morph, b_morph_f[mb])
                            loss = loss + 0.10 * sysid_loss
                else:
                    _, newlogp, ent, newval, _, pred_morph = model.get_action_and_value(b_obs_f[mb], b_hist_f[mb], b_act_f[mb])
                    ratio = (newlogp - b_logp_f[mb]).exp()
                    mb_adv = adv_f[mb]
                    pg = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                    vf = 0.5 * ((newval.view(-1) - ret_f[mb]) ** 2).mean()

                    loss = pg - 0.015 * ent.mean() + 0.5 * vf
                    if train_sysid:
                        sysid_loss = nn.functional.mse_loss(pred_morph, b_morph_f[mb])
                        loss = loss + 0.10 * sysid_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()

        return {"loss": total_loss / max(1, (b_sz // 256) * 4)}

    def _export_universal_model(self, full_model, env, onnx_path, ts_path):
        device = torch.device("cpu")
        full_model = full_model.to(device)
        full_model.eval()

        deployable = UniversalDeployablePolicy(
            history_encoder=full_model.history_encoder,
            morphology_estimator=full_model.morphology_estimator,
            actor_stem=full_model.actor_stem,
            actor_blocks=full_model.actor_blocks,
            actor_head=full_model.actor_head
        ).to(device)
        deployable.eval()

        dummy_obs = torch.randn(1, env.observation_space.shape[0], dtype=torch.float32, device=device)
        dummy_hist = torch.randn(1, env.history_dim, dtype=torch.float32, device=device)

        # 1. Export TorchScript
        try:
            traced = torch.jit.trace(deployable, (dummy_obs, dummy_hist), check_trace=False, strict=False)
            traced.save(ts_path)
            print(f"  * TorchScript Exported -> {ts_path} ({os.path.getsize(ts_path)/1024:.1f} KB)")
        except Exception as e:
            print(f"  [TorchScript Export Warning] {e}")

        # 2. Export ONNX
        try:
            torch.onnx.export(
                deployable,
                (dummy_obs, dummy_hist),
                onnx_path,
                input_names=["obs", "history"],
                output_names=["action", "pred_morphology"],
                dynamic_axes={
                    "obs": {0: "batch_size"},
                    "history": {0: "batch_size"},
                    "action": {0: "batch_size"},
                    "pred_morphology": {0: "batch_size"}
                },
                opset_version=14,
                dynamo=False
            )
            print(f"  * ONNX Exported        -> {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")
        except Exception as e:
            print(f"  [ONNX Export Warning] {e}")


def main():
    parser = argparse.ArgumentParser(description="ADAPT-X: Universal Scaled Drone RL Trainer (~25.6M Parameters)")
    parser.add_argument("--phase", type=str, default="all", choices=["1", "2", "3", "4", "5", "all"], help="Training phase (1 to 5 or 'all')")
    parser.add_argument("--num-envs", type=int, default=32, help="Vectorized simulation workers on DGX Spark")
    parser.add_argument("--profile", type=str, default="all", choices=["all", "nano_fpv", "compact_inspection", "commercial_enterprise", "heavy_industrial_barc"], help="Drone morphology spectrum profile")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode (scaled timesteps for verification)")
    args = parser.parse_args()

    scale = 0.05 if args.fast else 1.0

    trainer = UniversalDroneTrainer(
        device_id=args.device,
        num_envs=args.num_envs,
        steps_per_env=256,
        morph_profile=args.profile
    )

    t0 = time.time()
    ckpt_p1 = "checkpoints/dgx_spark_universal/drone_phase1_best_policy.pt"
    ckpt_p2 = "checkpoints/dgx_spark_universal/drone_phase2_best_policy.pt"
    ckpt_p3 = "checkpoints/dgx_spark_universal/drone_phase3_best_policy.pt"
    ckpt_p4 = "checkpoints/dgx_spark_universal/drone_phase4_best_policy.pt"

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

    print("\n" + "=" * 90)
    print(f"🎉 Universal Scaled Drone Training Pipeline Completed in {time.time() - t0:.1f}s!")
    print("=" * 90)


if __name__ == "__main__":
    main()
