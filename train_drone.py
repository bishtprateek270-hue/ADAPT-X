"""
ADAPT-X: Autonomous Drone Reinforcement Learning Pipeline (PPO + Obstacle Avoidance)
Trains an Adaptive Neural Policy (RMA) in the MuJoCo 3D City Simulation
to navigate waypoint trajectories through city canyons without striking buildings
or street obstacles, including under mid-air rotor damage conditions.
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
from agents.adaptive_ppo import AdaptiveActorCritic


class DeployableDronePolicy(nn.Module):
    """Zero-dependency production deployment module for edge ONNX inference."""
    def __init__(self, history_encoder, actor_mean):
        super().__init__()
        self.history_encoder = history_encoder
        self.actor_mean = actor_mean

    def forward(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        z = self.history_encoder(history)
        x = torch.cat([obs, z], dim=-1)
        return self.actor_mean(x)


def export_drone_onnx(model, output_path="deployments/adaptx_heavy_drone.onnx"):
    """Exports trained drone policy to optimized ONNX format for web simulation."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device("cpu")
    model_cpu = model.to(device)
    model_cpu.eval()

    deployable = DeployableDronePolicy(
        history_encoder=model_cpu.history_encoder,
        actor_mean=model_cpu.actor_mean
    ).to(device)
    deployable.eval()

    dummy_obs = torch.randn(1, 17, dtype=torch.float32, device=device)
    dummy_hist = torch.randn(1, 340, dtype=torch.float32, device=device)

    torch.onnx.export(
        deployable,
        (dummy_obs, dummy_hist),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        input_names=["observation", "history_buffer"],
        output_names=["target_joint_actions"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "history_buffer": {0: "batch_size"},
            "target_joint_actions": {0: "batch_size"}
        }
    )
    print(f"[Export] Saved Collision-Free ONNX Model -> {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")


def train_drone(total_timesteps=40_000, steps_per_env=512, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, epochs=4, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 80)
    print("   ADAPT-X: DRONE COLLISION-FREE WAYPOINT RL TRAINER (DGX SPARK / PPO)")
    print("=" * 80)
    print(f" Compute Engine   : {device} ({gpu_name})")
    print(f" Target Steps     : {total_timesteps:,}")
    print(f" Rollout Capacity : {steps_per_env} steps per policy update")
    print(f" Architecture     : Adaptive RMA Actor-Critic (17 obs + 340 hist -> 4 thrusts)")
    print(f" Objective        : Urban Trajectory Tracking with Zero Building/Obstacle Collisions")
    print("=" * 80)

    # Initialize environment with obstacle avoidance and dynamic avenue waypoints
    env = CityDroneEnv(target_pos=(0.0, 0.0, 5.2))
    env.randomize_target = True  # Enable dynamic city avenue waypoint sequences

    obs_dim = env.observation_space.shape[0]          # 17
    hist_dim = env.history_len * env.single_proprio_dim  # 20 * 17 = 340
    act_dim = env.action_space.shape[0]               # 4

    # Neural network policy
    agent = AdaptiveActorCritic(obs_dim=obs_dim, history_dim=hist_dim, act_dim=act_dim, latent_dim=16).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/drone_policy.pt"

    obs, hist, _ = env.reset_adaptive()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    hist_tensor = torch.tensor(hist, dtype=torch.float32, device=device).unsqueeze(0)

    num_updates = total_timesteps // steps_per_env
    ep_rewards = []
    curr_ep_reward = 0.0
    total_collisions = 0
    total_waypoints_reached = 0

    print("\n[Initiating Urban Canyon Collision-Avoidance Reinforcement Learning...]\n")
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # 1. Rollout Phase
        b_obs = torch.zeros((steps_per_env, obs_dim), device=device)
        b_hist = torch.zeros((steps_per_env, hist_dim), device=device)
        b_actions = torch.zeros((steps_per_env, act_dim), device=device)
        b_logprobs = torch.zeros(steps_per_env, device=device)
        b_rewards = torch.zeros(steps_per_env, device=device)
        b_dones = torch.zeros(steps_per_env, device=device)
        b_values = torch.zeros(steps_per_env, device=device)

        batch_collisions = 0
        batch_waypoints = 0

        for step in range(steps_per_env):
            b_obs[step] = obs_tensor.squeeze(0)
            b_hist[step] = hist_tensor.squeeze(0)

            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(obs_tensor, hist_tensor)
                b_values[step] = value.flatten()

            b_actions[step] = action.squeeze(0)
            b_logprobs[step] = logprob.squeeze(0)

            # Map neural network continuous action to motor thrust [0 to 12 Newtons]
            action_np = action.cpu().numpy().squeeze(0)
            thrust_cmd = np.clip(action_np + env.hover_thrust_per_rotor, 0.0, 12.0)

            # Mid-air rotor damage injection curriculum (5% probability)
            if np.random.rand() < 0.05 and env.fault_type == "none":
                victim_rotor = np.random.choice(["fr", "fl", "rr", "rl"])
                env.inject_rotor_fault("dead_motor", rotor=victim_rotor)

            next_obs, next_hist, reward, terminated, truncated, info = env.step_adaptive(thrust_cmd)
            curr_ep_reward += reward

            if info.get("obstacle_collision", False):
                batch_collisions += 1
                total_collisions += 1

            if info.get("dist_target", 99.0) < 0.8:
                batch_waypoints += 1
                total_waypoints_reached += 1

            b_rewards[step] = float(reward)
            done = terminated or truncated
            b_dones[step] = float(done)

            if done:
                ep_rewards.append(curr_ep_reward)
                curr_ep_reward = 0.0
                obs, hist, _ = env.reset_adaptive()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                hist_tensor = torch.tensor(hist, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                hist_tensor = torch.tensor(next_hist, dtype=torch.float32, device=device).unsqueeze(0)

        # 2. Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(obs_tensor, hist_tensor).reshape(1, -1)
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

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_hist[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - 0.01 * entropy_loss + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # Telemetry Logging
        if update % 5 == 0 or update == 1 or update == num_updates:
            fps = int((update * steps_per_env) / (time.time() - start_time))
            avg_rew = float(np.mean(ep_rewards[-15:])) if len(ep_rewards) > 0 else 0.0
            print(f"Iter [{update:3d}/{num_updates:3d}] | Steps: {update * steps_per_env:6d} | Avg Ep Reward: {avg_rew:+7.1f} | Collisions: {batch_collisions:2d} | Waypoints Hit: {batch_waypoints:2d} | FPS: {fps}")

    # 4. Save Trained Checkpoint
    torch.save({
        "model_state_dict": agent.state_dict(),
        "total_timesteps": total_timesteps,
        "total_collisions": total_collisions,
        "total_waypoints": total_waypoints_reached
    }, checkpoint_path)
    print("=" * 80)
    print(f"[SUCCESS] Training Finished! Saved policy weights to: {checkpoint_path}")

    # 5. Export Production ONNX Model
    export_drone_onnx(agent, output_path="deployments/adaptx_heavy_drone.onnx")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Drone RL Policy with Obstacle Avoidance")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Total training steps")
    args = parser.parse_args()

    train_drone(total_timesteps=args.timesteps)
