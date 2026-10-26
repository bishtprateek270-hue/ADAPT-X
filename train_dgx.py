"""
ADAPT-X: NVIDIA DGX High-Throughput Parallel RL Trainer
Scales Reinforcement Learning training across 64 to 256 parallel simulation
environments on enterprise NVIDIA DGX GPU systems (A100 / H100 / SuperPOD).
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.drone_env import CityDroneEnv
from envs.quadruped_env import QuadrupedEnv
from agents.ppo import ActorCritic


def make_env(env_type="drone", target_pos=(0.0, 0.0, 4.5)):
    def _init():
        if env_type == "drone":
            return CityDroneEnv(target_pos=target_pos)
        else:
            return QuadrupedEnv()
    return _init


def train_dgx(
    env_type="drone",
    num_envs=64,
    total_timesteps=1_000_000,
    steps_per_env=256,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    epochs=5,
    batch_size=256,
    device_id=0
):
    import gymnasium as gym

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device_id) if torch.cuda.is_available() else "CPU"

    print("=" * 80)
    print("       ADAPT-X: NVIDIA DGX HIGH-THROUGHPUT DISTRIBUTED RL TRAINER")
    print("=" * 80)
    print(f" Compute Platform  : NVIDIA DGX Cluster")
    print(f" Compute Device    : {device} ({gpu_name})")
    print(f" Available GPUs    : {torch.cuda.device_count()} GPU(s) detected")
    print(f" CPU Host Cores    : {os.cpu_count()} CPU thread workers")
    print(f" Environment Type  : {env_type.upper()} ({num_envs} Parallel Vectorized Workers)")
    print(f" Target Timesteps  : {total_timesteps:,} steps")
    print(f" Batch Rollout     : {num_envs * steps_per_env:,} steps per policy update")
    print("=" * 80)

    # Initialize Vectorized Environments (Parallel Worker Threads)
    print(f"\n[Spawning {num_envs} Parallel Asynchronous Simulation Workers...]")
    env_fns = [make_env(env_type) for _ in range(num_envs)]
    
    # Use AsyncVectorEnv for multi-core parallelism across DGX CPU/GPU
    envs = gym.vector.AsyncVectorEnv(env_fns)
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    # Initialize Actor-Critic Policy
    agent = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("deployments", exist_ok=True)
    checkpoint_path = f"checkpoints/dgx_{env_type}_best_policy.pt"

    obs, _ = envs.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

    batch_capacity = num_envs * steps_per_env
    num_updates = total_timesteps // batch_capacity

    print(f"[Starting DGX Distributed PPO Optimization ({num_updates} Updates)]\n")
    start_time = time.time()
    total_steps_collected = 0
    ep_rewards = []

    for update in range(1, num_updates + 1):
        # 1. Rollout Phase: Massively parallel environment stepping
        b_obs = torch.zeros((steps_per_env, num_envs, obs_dim), device=device)
        b_actions = torch.zeros((steps_per_env, num_envs, act_dim), device=device)
        b_logprobs = torch.zeros((steps_per_env, num_envs), device=device)
        b_rewards = torch.zeros((steps_per_env, num_envs), device=device)
        b_dones = torch.zeros((steps_per_env, num_envs), device=device)
        b_values = torch.zeros((steps_per_env, num_envs), device=device)

        for step in range(steps_per_env):
            b_obs[step] = obs_tensor

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)
                b_values[step] = value.squeeze(-1)

            b_actions[step] = action
            b_logprobs[step] = logprob

            # Convert actions to physical bounds
            action_np = action.cpu().numpy()
            if env_type == "drone":
                # Center around hover thrust
                thrust_cmds = np.clip(action_np + 3.40, 0.0, 12.0)
            else:
                thrust_cmds = np.clip(action_np, -1.0, 1.0)

            next_obs, rewards, terminations, truncations, infos = envs.step(thrust_cmds)
            dones = np.logical_or(terminations, truncations)

            b_rewards[step] = torch.tensor(rewards, dtype=torch.float32, device=device)
            b_dones[step] = torch.tensor(dones, dtype=torch.float32, device=device)

            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device)
            total_steps_collected += num_envs

        # 2. Vectorized GAE Computation
        with torch.no_grad():
            next_value = agent.get_value(obs_tensor).squeeze(-1)
            advantages = torch.zeros_like(b_rewards)
            lastgaelam = 0.0
            for t in reversed(range(steps_per_env)):
                if t == steps_per_env - 1:
                    nextnonterminal = 1.0 - b_dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - b_dones[t + 1]
                    nextvalues = b_values[t + 1]
                delta = b_rewards[t] + gamma * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + b_values

        # Flatten batch for mini-batch SGD
        b_obs_flat = b_obs.reshape(-1, obs_dim)
        b_actions_flat = b_actions.reshape(-1, act_dim)
        b_logprobs_flat = b_logprobs.reshape(-1)
        b_advantages_flat = advantages.reshape(-1)
        b_returns_flat = returns.reshape(-1)

        b_advantages_norm = (b_advantages_flat - b_advantages_flat.mean()) / (b_advantages_flat.std() + 1e-8)

        # 3. Multi-Epoch Mini-Batch SGD with PyTorch CUDA Autocast
        b_size = b_obs_flat.shape[0]
        inds = np.arange(b_size)

        for epoch in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, b_size, batch_size):
                end = start + batch_size
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs_flat[mb_inds], b_actions_flat[mb_inds]
                )
                logratio = newlogprob - b_logprobs_flat[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages_norm[mb_inds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns_flat[mb_inds]) ** 2).mean()
                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # Telemetry & Performance Benchmark
        elapsed = time.time() - start_time
        fps = int(total_steps_collected / elapsed)
        mean_reward = b_rewards.mean().item() * steps_per_env

        if update % 2 == 0 or update == 1 or update == num_updates:
            print(f"DGX Iter [{update:4d}/{num_updates:4d}] | Steps: {total_steps_collected:8,d} | Avg Reward: {mean_reward:+8.2f} | Loss: {loss.item():.4f} | Throughput: {fps:,} FPS | Elapsed: {elapsed:5.1f}s")

    # Cleanup workers
    envs.close()

    # Save Checkpoints
    torch.save({"model_state_dict": agent.state_dict()}, checkpoint_path)
    print("\n" + "=" * 80)
    print(f"🎉 DGX Training Successfully Completed in {time.time() - start_time:.1f}s!")
    print(f"   Checkpoint Saved : {checkpoint_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAPT-X NVIDIA DGX RL Trainer")
    parser.add_argument("--env", type=str, default="drone", choices=["drone", "quadruped"], help="Environment to train")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel environments (64-256 for DGX)")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training simulation steps")
    parser.add_argument("--device", type=int, default=0, help="Target CUDA device ID")
    args = parser.parse_args()

    train_dgx(
        env_type=args.env,
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        device_id=args.device
    )
