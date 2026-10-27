import os
import sys

# Auto-switch to project virtual environment if running in unconfigured interpreter
venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if os.path.exists(venv_python) and sys.executable != venv_python:
    try:
        import torch
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

import time
import argparse
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.quadruped_env import QuadrupedEnv
from agents.adaptive_ppo import AdaptiveActorCritic

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_VIEWER_AVAILABLE = True
except Exception:
    MUJOCO_VIEWER_AVAILABLE = False


def evaluate_adaptive(checkpoint_path, config_path, num_episodes=5, render=False, damage="none", leg="fr", damage_step=200):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "damage_injection" in config:
        config["damage_injection"]["enabled"] = False

    env = QuadrupedEnv(config=config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    history_dim = env.history_dim
    latent_dim = int(config.get("adaptive", {}).get("latent_dim", 16))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveActorCritic(
        obs_dim=obs_dim,
        history_dim=history_dim,
        act_dim=act_dim,
        latent_dim=latent_dim
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Loaded Checkpoint] {checkpoint_path}")
    else:
        print(f"[Warning] Checkpoint {checkpoint_path} not found. Running with random initial weights.")

    model.eval()

    print("\n" + "=" * 70)
    print(f"    ADAPT-X PHASE 3: ADAPTIVE EVALUATION BENCHMARK ({num_episodes} Episodes)")
    if damage != "none":
        print(f"  [DAMAGE INJECTED] Mode: {damage.upper()} on Leg: {leg.upper()} at Step {damage_step}")
    else:
        print("  [DAMAGE STATUS] None (Healthy Baseline)")
    print("=" * 70)

    ep_rewards = []
    ep_lengths = []
    vel_errors = []
    post_damage_survivals = []
    latent_shifts = []

    for ep in range(1, num_episodes + 1):
        obs, hist, _ = env.reset_adaptive()
        done = False
        total_reward = 0.0
        steps = 0
        recorded_vx = []

        baseline_z = None
        post_damage_z = None

        viewer = None
        if render and MUJOCO_VIEWER_AVAILABLE and env.model is not None:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        while not done:
            if damage != "none" and steps == damage_step:
                env.inject_fault(damage, leg=leg)
                if not render:
                    print(f"  >>> Step {steps}: INJECTED HARDWARE DAMAGE [{damage.upper()}] on {leg.upper()}!")

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            hist_t = torch.tensor(hist, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                z = model.get_latent(hist_t)
                x = torch.cat([obs_t, z], dim=-1)
                action = model.actor_mean(x).squeeze(0).cpu().numpy()

            # Record latent representations around damage
            if steps == damage_step - 5:
                baseline_z = z.cpu().numpy().copy()
            if steps == damage_step + 30:
                post_damage_z = z.cpu().numpy().copy()

            obs, hist, reward, terminated, truncated, info = env.step_adaptive(action)
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
        survived = steps > damage_step if damage != "none" else True

        # Calculate latent adaptation magnitude
        z_shift = 0.0
        if baseline_z is not None and post_damage_z is not None:
            z_shift = float(np.linalg.norm(post_damage_z - baseline_z))
            latent_shifts.append(z_shift)

        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        vel_errors.append(vel_err)
        post_damage_survivals.append(survived)

        print(
            f"Ep [{ep}/{num_episodes}] | "
            f"Reward: {total_reward:>7.2f} | "
            f"Steps: {steps:>4d} | "
            f"Cmd Vx: {cmd_vx:.2f} m/s | "
            f"Actual Vx: {mean_vx:.2f} m/s | "
            f"Latent Shift ||dz||: {z_shift:.3f} | "
            f"Survived: {'YES' if survived else 'NO (FELL)'}"
        )

    print("-" * 70)
    print(f"Mean Reward         : {np.mean(ep_rewards):.2f} +/- {np.std(ep_rewards):.2f}")
    print(f"Mean Survival Time  : {np.mean(ep_lengths) * env.dt:.2f}s ({np.mean(ep_lengths):.0f} steps)")
    print(f"Mean Tracking Error : {np.mean(vel_errors):.3f} m/s")
    if len(latent_shifts) > 0:
        print(f"Mean Latent Adaptation: {np.mean(latent_shifts):.3f} (Neural state shift)")
    if damage != "none":
        recovery_rate = np.mean(post_damage_survivals) * 100.0
        print(f"Damage Survival Rate: {recovery_rate:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ADAPT-X Phase 3 Adaptive Policy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase3/best_policy.pt")
    parser.add_argument("--config", type=str, default="configs/phase3_adaptive_config.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--damage", type=str, default="none", choices=["none", "dead_motor", "locked_joint", "weakened_motor", "amputated_leg"])
    parser.add_argument("--leg", type=str, default="fr", choices=["fr", "fl", "rr", "rl"])
    parser.add_argument("--damage-step", type=int, default=200)
    args = parser.parse_args()

    evaluate_adaptive(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.episodes,
        render=args.render,
        damage=args.damage,
        leg=args.leg,
        damage_step=args.damage_step
    )
