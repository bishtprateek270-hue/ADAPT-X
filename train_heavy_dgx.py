"""
ADAPT-X: Heavy Deep RL Training Suite for NVIDIA DGX Spark
Leverages NVIDIA Grace Blackwell GB10 GPU with BFloat16 Tensor Cores,
Multi-Head Temporal Transformer (RMA), and Deep ResNet Actor-Critic (3.3M parameters).

Phases:
  * Phase 1: Baseline Nominal Locomotion (Undamaged morphology, 4-leg coordination)
  * Phase 2: Actuator Fault & Hardware Damage Discovery (Dead motor, locked peg-leg, amputated limb)
  * Phase 3: Temporal Transformer Self-Healing Adaptation (H=20 timesteps -> z_t in R^32)
  * Phase 4: Staged Curriculum Resilience (4A: Pre-broken, 4B: Dynamic mid-stride, 4C: 15N force push)
  * Phase 5: Sim-to-Real Gap Closure & Production ONNX / TorchScript Export
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
from agents.heavy_adaptive_ppo import HeavyAdaptiveActorCritic, HeavyDeployablePolicy


class HeavyDGXSparkTrainer:
    """
    Heavy Deep Reinforcement Learning Trainer tailored for NVIDIA DGX Spark.
    Utilizes PyTorch CUDA BFloat16 Mixed Precision on Blackwell Tensor Cores.
    """
    def __init__(self, robot="drone", device_id=0, num_envs=32, steps_per_env=256, use_amp=True):
        self.robot = robot
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.gpu_name = torch.cuda.get_device_name(device_id) if torch.cuda.is_available() else "CPU"
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.use_amp = use_amp and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        os.makedirs("checkpoints/dgx_spark_heavy", exist_ok=True)
        os.makedirs("logs/dgx_spark_heavy", exist_ok=True)
        os.makedirs("deployments", exist_ok=True)

        print("=" * 85)
        print("    ADAPT-X: HEAVY DEEP RL NEURAL TRAINING SUITE (NVIDIA DGX SPARK)")
        print("=" * 85)
        print(f" Platform Arch     : NVIDIA DGX Spark ({os.uname().machine})")
        print(f" Compute Engine    : {self.device} ({self.gpu_name})")
        print(f" Precision Mode    : {'CUDA AMP (' + str(self.amp_dtype).split('.')[-1] + ')' if self.use_amp else 'FP32'}")
        print(f" Parallel Envs     : {self.num_envs} vectorized simulation workers")
        print(f" Batch Capacity    : {self.num_envs * self.steps_per_env:,} steps per policy update")
        print("=" * 85)

    def _make_env(self, phase, stage=1):
        if self.robot == "drone":
            return CityDroneEnv(target_pos=(0.0, 0.0, 4.5))
        config = {
            "system": {"device": "cuda", "num_cpu_threads": 4},
            "environment": {"target_height": 0.28, "dt": 0.02, "sim_substeps": 10},
            "adaptive": {"history_len": 20, "latent_dim": 32},
            "damage_injection": {
                "enabled": (phase >= 2),
                "failure_prob": 0.75 if phase >= 2 else 0.0,
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

    def _init_heavy_model(self, env):
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        single_proprio_dim = env.single_proprio_dim
        history_len = env.history_len

        model = HeavyAdaptiveActorCritic(
            obs_dim=obs_dim,
            single_proprio_dim=single_proprio_dim,
            history_len=history_len,
            act_dim=act_dim,
            latent_dim=32,
            hidden_dim=512
        ).to(self.device)
        return model

    # =========================================================================
    # Phase 1: Baseline Nominal Locomotion
    # =========================================================================
    def train_phase1(self, timesteps=100_000, lr=3e-4):
        robot_title = "QUADCOPTER DRONE" if self.robot == "drone" else "4-LEGGED QUADRUPED"
        print("\n" + "=" * 85)
        print(f"  🟢 PHASE 1: BASELINE NOMINAL DYNAMICS ({robot_title})")
        print("=" * 85)
        print(f" Target Timesteps : {timesteps:,}")
        print(" Neural Trunk     : 4-Head Temporal Transformer + 512-wide Deep ResNet")
        print("-" * 85)

        prefix = f"{self.robot}_" if self.robot != "quadruped" else ""
        ckpt_path = f"checkpoints/dgx_spark_heavy/{prefix}phase1_best_policy.pt"
        envs = [self._make_env(phase=1) for _ in range(self.num_envs)]
        model = self._init_heavy_model(envs[0])
        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)

        self._run_training_loop(envs, model, optimizer, timesteps, ckpt_path, phase=1)
        for e in envs: e.close()
        return ckpt_path

    # =========================================================================
    # Phase 2: Hardware Damage Space Exploration
    # =========================================================================
    def train_phase2(self, timesteps=120_000, lr=2.5e-4, resume_ckpt="checkpoints/dgx_spark_heavy/phase1_best_policy.pt"):
        print("\n" + "=" * 85)
        print("  🟡 PHASE 2: ACTUATOR FAULT & DAMAGE SPACE EXPLORATION")
        print("=" * 85)
        print(f" Target Timesteps : {timesteps:,}")
        print(f" Resume Model     : {resume_ckpt}")
        print(" Fault Modes      : Dead motor, locked joint, weakened actuator, limb amputation")
        print("-" * 85)

        prefix = f"{self.robot}_" if self.robot != "quadruped" else ""
        ckpt_path = f"checkpoints/dgx_spark_heavy/{prefix}phase2_best_policy.pt"
        envs = [self._make_env(phase=2) for _ in range(self.num_envs)]
        model = self._init_heavy_model(envs[0])

        if os.path.exists(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Phase 2] Warm-started weights from -> {resume_ckpt}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)
        self._run_training_loop(envs, model, optimizer, timesteps, ckpt_path, phase=2)
        for e in envs: e.close()
        return ckpt_path

    # =========================================================================
    # Phase 3: Temporal Transformer RMA Self-Healing Adaptation
    # =========================================================================
    def train_phase3(self, timesteps=150_000, lr=2e-4, resume_ckpt="checkpoints/dgx_spark_heavy/phase2_best_policy.pt"):
        print("\n" + "=" * 85)
        print("  🟣 PHASE 3: TEMPORAL TRANSFORMER LATENT ADAPTATION (RMA PARADIGM)")
        print("=" * 85)
        print(f" Target Timesteps : {timesteps:,}")
        print(f" Resume Model     : {resume_ckpt}")
        print(" Attention Focus  : Multi-Head Self-Attention over H=20 rolling proprioception steps")
        print(" Latent Space     : Compresses 840 proprioception dimensions into z_t in R^32")
        print("-" * 85)

        prefix = f"{self.robot}_" if self.robot != "quadruped" else ""
        ckpt_path = f"checkpoints/dgx_spark_heavy/{prefix}phase3_best_policy.pt"
        envs = [self._make_env(phase=3) for _ in range(self.num_envs)]
        model = self._init_heavy_model(envs[0])

        if os.path.exists(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Phase 3] Warm-started weights from -> {resume_ckpt}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)
        self._run_training_loop(envs, model, optimizer, timesteps, ckpt_path, phase=3)
        for e in envs: e.close()
        return ckpt_path

    # =========================================================================
    # Phase 4: Staged Curriculum Resilience & External Push Impulses
    # =========================================================================
    def train_phase4(self, timesteps_per_stage=50_000, lr=1.8e-4, resume_ckpt=None):
        prefix = f"{self.robot}_" if self.robot != "quadruped" else ""
        if resume_ckpt is None:
            resume_ckpt = f"checkpoints/dgx_spark_heavy/{prefix}phase3_best_policy.pt"

        print("\n" + "=" * 85)
        print("  🟠 PHASE 4: MULTI-STAGE CURRICULUM RESILIENCE & DISTURBANCE STRESS TEST")
        print("=" * 85)
        print(" Stages:")
        print("   * 4A: Pre-broken / Rotor Loss recovery discovery")
        print("   * 4B: Dynamic mid-stride / in-flight failure transition")
        print("   * 4C: Search & Rescue stress test with external force impulses")
        print("-" * 85)

        current_ckpt = resume_ckpt
        stages = [
            (1, "Stage 4A: Damage Recovery Discovery"),
            (2, "Stage 4B: Dynamic Mid-Flight/Stride Fault Transition"),
            (3, "Stage 4C: External Force Impulses & Wind Gusts")
        ]

        for stage_idx, stage_name in stages:
            print(f"\n>>> Running Phase 4 - {stage_name} ({timesteps_per_stage:,} steps)...")
            ckpt_path = f"checkpoints/dgx_spark_heavy/{prefix}phase4_stage{stage_idx}_best_policy.pt"
            envs = [self._make_env(phase=4, stage=stage_idx) for _ in range(self.num_envs)]
            model = self._init_heavy_model(envs[0])

            if os.path.exists(current_ckpt):
                ckpt = torch.load(current_ckpt, map_location=self.device)
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"[Phase 4 - Stage {stage_idx}] Loaded weights from -> {current_ckpt}")

            optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)
            self._run_training_loop(envs, model, optimizer, timesteps_per_stage, ckpt_path, phase=4, stage=stage_idx)
            for e in envs: e.close()
            current_ckpt = ckpt_path

        final_phase4_ckpt = f"checkpoints/dgx_spark_heavy/{prefix}phase4_best_policy.pt"
        torch.save({"model_state_dict": model.state_dict()}, final_phase4_ckpt)
        return final_phase4_ckpt

    # =========================================================================
    # Phase 5: Sim-to-Real Gap Closure & Production ONNX Export
    # =========================================================================
    def train_phase5_and_export(self, timesteps=60_000, lr=1e-4, resume_ckpt=None):
        prefix = f"{self.robot}_" if self.robot != "quadruped" else ""
        if resume_ckpt is None:
            resume_ckpt = f"checkpoints/dgx_spark_heavy/{prefix}phase4_best_policy.pt"

        print("\n" + "=" * 85)
        print("  🔵 PHASE 5: SIM-TO-REAL GAP CLOSURE & HEAVY PRODUCTION EXPORT")
        print("=" * 85)
        print(" Domain Randomization: IMU Gyro Noise, Joint Encoder Noise, CAN Bus Delay, Torso Mass & Friction.")
        print("-" * 85)

        ckpt_path = f"checkpoints/dgx_spark_heavy/{prefix}phase5_best_policy.pt"
        envs = [self._make_env(phase=4, stage=2) for _ in range(self.num_envs)]
        sim2real = [Sim2RealWrapper(action_delay_steps=1, joint_noise_std=0.008, gyro_noise_std=0.015) for _ in range(self.num_envs)]
        model = self._init_heavy_model(envs[0])

        if os.path.exists(resume_ckpt):
            ckpt = torch.load(resume_ckpt, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Phase 5] Loaded weights from -> {resume_ckpt}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)
        self._run_training_loop(envs, model, optimizer, timesteps, ckpt_path, phase=5, sim2real=sim2real)
        for e in envs: e.close()

        # Production ONNX & TorchScript Compilation
        print("\n[Compiling & Exporting Production Heavy Model to ONNX & TorchScript...]")
        onnx_out = f"deployments/adaptx_heavy_{self.robot}.onnx"
        ts_out = f"deployments/adaptx_heavy_{self.robot}.torchscript.pt"

        self._export_heavy_model(model, envs[0], onnx_out, ts_out)
        return onnx_out

    # =========================================================================
    # Core Vectorized PPO Training Loop with BFloat16 Mixed Precision
    # =========================================================================
    def _run_training_loop(self, envs, model, optimizer, total_timesteps, ckpt_path, phase=1, stage=1, sim2real=None):
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

            latent_magnitudes = []
            vx_list = []

            for step in range(self.steps_per_env):
                b_obs[step] = obs_tensor
                b_hist[step] = hist_tensor

                with torch.no_grad():
                    if self.use_amp:
                        with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                            action, logprob, _, val, z = model.get_action_and_value(obs_tensor, hist_tensor)
                    else:
                        action, logprob, _, val, z = model.get_action_and_value(obs_tensor, hist_tensor)

                    b_val[step] = val.squeeze(-1)
                    latent_magnitudes.append(torch.norm(z, dim=-1).mean().item())

                b_act[step] = action
                b_logp[step] = logprob

                act_np = action.float().cpu().numpy()
                next_obs, next_hist = [], []
                rewards, dones = [], []

                for i in range(num_envs):
                    if self.robot == "drone":
                        cmd = np.clip(act_np[i] + 3.40, 0.0, 12.0)
                        if phase >= 2 and step % 80 == 0:
                            envs[i].inject_rotor_fault("dead_motor", "fr")
                        if phase == 4 and step % 60 == 0:
                            envs[i].apply_wind_gust(12.0)
                    else:
                        raw_cmd = np.clip(act_np[i], -1.0, 1.0)
                        cmd = sim2real[i].apply_action_delay(raw_cmd) if sim2real else raw_cmd

                    o, h, r, term, trunc, info = envs[i].step_adaptive(cmd)
                    d = term or trunc
                    vx_list.append(info.get("vx", info.get("altitude", 0.0)))

                    if d:
                        o, h, _ = envs[i].reset_adaptive()
                        if sim2real:
                            sim2real[i].reset_buffer()
                            if hasattr(envs[i], "model"):
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

            # GAE computation & optimization with AMP
            loss_info = self._optimize_heavy_ppo(model, optimizer, b_obs, b_hist, b_act, b_logp, b_rew, b_don, b_val, obs_tensor, hist_tensor)

            elapsed = time.time() - start_time
            fps = int(total_steps / max(1e-3, elapsed))
            mean_rew = b_rew.mean().item() * self.steps_per_env
            mean_z = np.mean(latent_magnitudes) if latent_magnitudes else 0.0
            mean_vx = np.mean(vx_list) if vx_list else 0.0

            if update % max(1, num_updates // 5) == 0 or update == num_updates:
                vram_used = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
                print(f"  [Heavy DGX] Update [{update:3d}/{num_updates:3d}] | Steps: {total_steps:7,d} | Ep Rew: {mean_rew:+7.2f} | Loss: {loss_info['loss']:.4f} | ||z||: {mean_z:.2f} | Vx: {mean_vx:+.2f}m/s | FPS: {fps:6,d} | VRAM: {vram_used:.1f}MB | Time: {elapsed:5.1f}s")

        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        print(f"  [Heavy Model Checkpoint Saved] -> {ckpt_path}")

    def _optimize_heavy_ppo(self, model, optimizer, b_obs, b_hist, b_act, b_logp, b_rew, b_don, b_val, last_obs, last_hist):
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
                        _, newlogp, ent, newval, _ = model.get_action_and_value(b_obs_f[mb], b_hist_f[mb], b_act_f[mb])
                        ratio = (newlogp - b_logp_f[mb]).exp()
                        mb_adv = adv_f[mb]
                        pg = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                        vf = 0.5 * ((newval.view(-1) - ret_f[mb]) ** 2).mean()
                        loss = pg - 0.015 * ent.mean() + 0.5 * vf
                else:
                    _, newlogp, ent, newval, _ = model.get_action_and_value(b_obs_f[mb], b_hist_f[mb], b_act_f[mb])
                    ratio = (newlogp - b_logp_f[mb]).exp()
                    mb_adv = adv_f[mb]
                    pg = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                    vf = 0.5 * ((newval.view(-1) - ret_f[mb]) ** 2).mean()
                    loss = pg - 0.015 * ent.mean() + 0.5 * vf

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()

        return {"loss": total_loss / max(1, (b_sz // 256) * 4)}

    def _export_heavy_model(self, full_model, env, onnx_path, ts_path):
        device = torch.device("cpu")
        full_model = full_model.to(device)
        full_model.eval()

        deployable = HeavyDeployablePolicy(
            history_encoder=full_model.history_encoder,
            actor_stem=full_model.actor_stem,
            actor_res1=full_model.actor_res1,
            actor_res2=full_model.actor_res2,
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
            print(f"  [TorchScript Warning] Tracing failed: {e}")

        # 2. Export ONNX
        try:
            torch.onnx.export(
                deployable,
                (dummy_obs, dummy_hist),
                onnx_path,
                export_params=True,
                opset_version=14,
                dynamo=False,
                do_constant_folding=True,
                input_names=["observation", "history_buffer"],
                output_names=["target_joint_actions"],
                dynamic_axes={
                    "observation": {0: "batch_size"},
                    "history_buffer": {0: "batch_size"},
                    "target_joint_actions": {0: "batch_size"}
                }
            )
            print(f"  * ONNX Exported        -> {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")

            # Benchmark ONNX Runtime latency
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            feeds = {
                "observation": dummy_obs.numpy(),
                "history_buffer": dummy_hist.numpy()
            }
            # Warmup
            for _ in range(50): session.run(None, feeds)
            times = []
            for _ in range(500):
                t0 = time.perf_counter()
                session.run(None, feeds)
                times.append((time.perf_counter() - t0) * 1000)

            p50 = np.percentile(times, 50)
            p95 = np.percentile(times, 95)
            p99 = np.percentile(times, 99)
            print("=" * 70)
            print("      HEAVY ONNX EDGE INFERENCE LATENCY BENCHMARK (500 runs)")
            print("=" * 70)
            print(f"  P50 Inference Latency : {p50:.3f} ms ({p50*1000:.1f} microseconds)")
            print(f"  P95 Inference Latency : {p95:.3f} ms")
            print(f"  P99 Inference Latency : {p99:.3f} ms")
            print(f"  Max Control Loop Rate : {1000.0/p50:.0f} Hz (50 Hz real-time required)")
            print(f"  Headroom Margin       : {20.0/p50:.1f}x faster than real-time deadline")
            print("=" * 70)
        except Exception as e:
            print(f"  [ONNX Warning] Export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="ADAPT-X Heavy Deep RL Trainer for NVIDIA DGX Spark")
    parser.add_argument("--robot", type=str, default="drone", choices=["drone", "quadruped"], help="Target robot (drone or quadruped)")
    parser.add_argument("--phase", type=str, default="all", choices=["1", "2", "3", "4", "5", "all"], help="Training phase")
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel environments on DGX Spark")
    parser.add_argument("--device", type=int, default=0, help="Target CUDA device ID")
    parser.add_argument("--scale", type=float, default=1.0, help="Timestep multiplier (1.0 = full production, 0.2 = fast)")
    args = parser.parse_args()

    trainer = HeavyDGXSparkTrainer(
        robot=args.robot,
        device_id=args.device,
        num_envs=args.num_envs,
        steps_per_env=256,
        use_amp=True
    )

    t0 = time.time()
    prefix = f"{args.robot}_" if args.robot != "quadruped" else ""
    ckpt_p1 = f"checkpoints/dgx_spark_heavy/{prefix}phase1_best_policy.pt"
    ckpt_p2 = f"checkpoints/dgx_spark_heavy/{prefix}phase2_best_policy.pt"
    ckpt_p3 = f"checkpoints/dgx_spark_heavy/{prefix}phase3_best_policy.pt"
    ckpt_p4 = f"checkpoints/dgx_spark_heavy/{prefix}phase4_best_policy.pt"

    if args.phase in ["1", "all"]:
        ckpt_p1 = trainer.train_phase1(timesteps=int(100_000 * args.scale))
    if args.phase in ["2", "all"]:
        ckpt_p2 = trainer.train_phase2(timesteps=int(120_000 * args.scale), resume_ckpt=ckpt_p1)
    if args.phase in ["3", "all"]:
        ckpt_p3 = trainer.train_phase3(timesteps=int(150_000 * args.scale), resume_ckpt=ckpt_p2)
    if args.phase in ["4", "all"]:
        ckpt_p4 = trainer.train_phase4(timesteps_per_stage=int(50_000 * args.scale), resume_ckpt=ckpt_p3)
    if args.phase in ["5", "all"]:
        trainer.train_phase5_and_export(timesteps=int(60_000 * args.scale), resume_ckpt=ckpt_p4)

    print("\n" + "=" * 85)
    print(f"🎉 Heavy Deep RL Training for {args.robot.upper()} Completed in {time.time() - t0:.1f}s on NVIDIA DGX Spark!")
    print("=" * 85)


if __name__ == "__main__":
    main()
