"""
ADAPT-X: Drone Reinforcement Learning Training Pipeline (PPO)
Trains a Neural Network policy in the MuJoCo 3D City simulation
from zero knowledge to stable hover and mid-air failure recovery.
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
from agents.ppo import ActorCritic


def train_drone(total_timesteps=100_000, steps_per_env=512, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, epochs=4, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("       ADAPT-X: DRONE REINFORCEMENT LEARNING TRAINER (PPO)")
    print("=" * 70)
    print(f" Compute Device   : {device} (RTX 5050 / CPU)")
    print(f" Target Timesteps : {total_timesteps:,} steps")
    print(f" Rollout Buffer   : {steps_per_env} steps per update")
    print(f" Architecture     : Actor-Critic MLP [17 dims -> (256, 128) -> 4 dims]")
    print("=" * 70)

    # Initialize simulation environment
    env = CityDroneEnv(target_pos=(0.0, 0.0, 4.5))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Neural network policy
    agent = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/drone_policy.pt"

    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    num_updates = total_timesteps // steps_per_env
    ep_rewards = []
    curr_ep_reward = 0.0

    print("\n[Starting RL Training Loop: The Policy Learns by Trial-and-Error...]\n")
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # 1. Rollout Phase: Interact with simulation and collect experiences
        b_obs = torch.zeros((steps_per_env, obs_dim), device=device)
        b_actions = torch.zeros((steps_per_env, act_dim), device=device)
        b_logprobs = torch.zeros(steps_per_env, device=device)
        b_rewards = torch.zeros(steps_per_env, device=device)
        b_dones = torch.zeros(steps_per_env, device=device)
        b_values = torch.zeros(steps_per_env, device=device)

        for step in range(steps_per_env):
            b_obs[step] = obs_tensor.squeeze(0)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)
                b_values[step] = value.flatten()

            b_actions[step] = action.squeeze(0)
            b_logprobs[step] = logprob.squeeze(0)

            # Map neural network continuous action to motor thrust [0 to 12 Newtons]
            action_np = action.cpu().numpy().squeeze(0)
            thrust_cmd = np.clip(action_np + env.hover_thrust_per_rotor, 0.0, 12.0)

            # Random damage injection during training (domain randomization)
            if np.random.rand() < 0.05 and env.fault_type == "none":
                victim_rotor = np.random.choice(["fr", "fl", "rr", "rl"])
                env.inject_rotor_fault("dead_motor", rotor=victim_rotor)

            next_obs, reward, terminated, truncated, info = env.step(thrust_cmd)
            curr_ep_reward += reward

            b_rewards[step] = float(reward)
            done = terminated or truncated
            b_dones[step] = float(done)

            if done:
                ep_rewards.append(curr_ep_reward)
                curr_ep_reward = 0.0
                obs, _ = env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

        # 2. Advantage Estimation (Generalized Advantage Estimation - GAE)
        with torch.no_grad():
            next_value = agent.get_value(obs_tensor).reshape(1, -1)
            advantages = torch.zeros_like(b_rewards)
            lastgaelam = 0.0
            for t in reversed(range(steps_per_env)):
                if t == steps_per_env - 1:
                    nextnonterminal = 1.0 - float(done)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - b_dones[t + 1]
                    nextvalues = b_values[t + 1]
                delta = b_rewards[t] + gamma * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + b_values

        # 3. Policy Optimization (PPO Clipped Surrogate Loss)
        b_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        inds = np.arange(steps_per_env)
        for epoch in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, steps_per_env, batch_size):
                end = start + batch_size
                mb_inds = inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Policy Loss
                mb_adv = b_advantages[mb_inds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - 0.01 * entropy_loss + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # Telemetry Logging
        if update % 5 == 0 or update == 1:
            fps = int((update * steps_per_env) / (time.time() - start_time))
            avg_rew = np.mean(ep_rewards[-10:]) if len(ep_rewards) > 0 else 0.0
            print(f"Iter [{update:3d}/{num_updates:3d}] | Total Steps: {update * steps_per_env:6d} | Avg Ep Reward: {avg_rew:+7.2f} | Policy Loss: {pg_loss.item():.4f} | FPS: {fps}")

    # Save trained policy weights
    torch.save({"model_state_dict": agent.state_dict()}, checkpoint_path)
    print("=" * 70)
    print(f"[SUCCESS] Training Complete! Saved trained RL policy to: {checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Drone RL Policy in Simulation")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Total training steps")
    args = parser.parse_args()

    train_drone(total_timesteps=args.timesteps)
