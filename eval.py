import os
import sys
import time
import argparse
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.quadruped_env import QuadrupedEnv
from agents.ppo import ActorCritic

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_VIEWER_AVAILABLE = True
except Exception:
    MUJOCO_VIEWER_AVAILABLE = False


def evaluate(checkpoint_path, config_path, num_episodes=5, render=False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env = QuadrupedEnv(config=config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, act_dim).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Loaded Checkpoint] {checkpoint_path}")
    else:
        print(f"[Warning] Checkpoint {checkpoint_path} not found. Running with random initial weights.")

    model.eval()

    print("\n" + "=" * 60)
    print(f"       ADAPT-X EVALUATION BENCHMARK ({num_episodes} Episodes)")
    print("=" * 60)

    ep_rewards = []
    ep_lengths = []
    vel_errors = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        recorded_vx = []

        # Optional viewer setup
        viewer = None
        if render and MUJOCO_VIEWER_AVAILABLE and env.model is not None:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # Use deterministic mean action for evaluation
                action = model.actor_mean(obs_tensor).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            recorded_vx.append(info.get("vx", 0.0))

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(env.dt)

        if viewer is not None:
            viewer.close()

        mean_vx = np.mean(recorded_vx) if len(recorded_vx) > 0 else 0.0
        cmd_vx = env.commands[0]
        vel_err = abs(mean_vx - cmd_vx)

        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        vel_errors.append(vel_err)

        print(
            f"Ep [{ep}/{num_episodes}] | "
            f"Reward: {total_reward:>7.2f} | "
            f"Steps: {steps:>4d} | "
            f"Cmd Vx: {cmd_vx:.2f} m/s | "
            f"Actual Vx: {mean_vx:.2f} m/s | "
            f"Vel Err: {vel_err:.2f} m/s"
        )

    print("-" * 60)
    print(f"Mean Reward        : {np.mean(ep_rewards):.2f} +/- {np.std(ep_rewards):.2f}")
    print(f"Mean Survival Time : {np.mean(ep_lengths) * env.dt:.2f}s ({np.mean(ep_lengths):.0f} steps)")
    print(f"Mean Tracking Error: {np.mean(vel_errors):.3f} m/s")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_policy.pt")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.config, args.episodes, args.render)
