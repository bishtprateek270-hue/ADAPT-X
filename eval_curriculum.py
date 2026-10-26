import os
import sys
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


def evaluate_curriculum(checkpoint_path, config_path, stage=2, num_episodes=5, render=False, damage="amputated_leg", leg="fr", damage_step=200):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["curriculum"]["stage"] = stage
    if stage == 3:
        config["curriculum"]["push_disturbances"] = True
    else:
        config["curriculum"]["push_disturbances"] = False

    if "damage_injection" in config:
        config["damage_injection"]["enabled"] = (stage != 1)

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
        print(f"[Warning] Checkpoint {checkpoint_path} not found. Running with initial weights.")

    model.eval()

    stage_names = {
        1: "STAGE 1: PRE-BROKEN 3-LEG LOCOMOTION",
        2: "STAGE 2: DYNAMIC TRANSITION & VELOCITY RETENTION",
        3: "STAGE 3: SEARCH & RESCUE STRESS TEST (DISTURBANCE PUSHES)"
    }

    print("\n" + "=" * 72)
    print(f"      ADAPT-X PHASE 4 CURRICULUM BENCHMARK ({stage_names[stage]})")
    print(f"  Target Damage: {damage.upper()} on {leg.upper()} | Push Disturbances: {env.push_disturbances}")
    print("=" * 72)

    ep_rewards = []
    ep_lengths = []
    actual_speeds = []
    speed_retentions = []
    survivals = []

    for ep in range(1, num_episodes + 1):
        obs, hist, _ = env.reset_adaptive()
        
        # If Stage 1, pre-damage target leg immediately
        if stage == 1:
            env.inject_fault(damage, leg=leg)

        done = False
        total_reward = 0.0
        steps = 0
        recorded_vx = []

        viewer = None
        if render and MUJOCO_VIEWER_AVAILABLE and env.model is not None:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        while not done:
            # Stage 2/3 dynamic transition
            if stage in [2, 3] and steps == damage_step:
                env.inject_fault(damage, leg=leg)
                if not render:
                    print(f"  >>> Step {steps}: TRIGGERED FAILURE [{damage.upper()}] on {leg.upper()} leg!")

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            hist_t = torch.tensor(hist, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                z = model.get_latent(hist_t)
                x = torch.cat([obs_t, z], dim=-1)
                action = model.actor_mean(x).squeeze(0).cpu().numpy()

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

        # Compute speed after damage
        if stage == 1:
            post_vx = np.mean(recorded_vx)
        else:
            post_vx = np.mean(recorded_vx[damage_step:]) if len(recorded_vx) > damage_step else 0.0

        cmd_vx = env.commands[0]
        retention = max(0.0, (post_vx / max(cmd_vx, 1e-4))) * 100.0
        survived = steps >= 900

        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        actual_speeds.append(post_vx)
        speed_retentions.append(retention)
        survivals.append(survived)

        print(
            f"Ep [{ep}/{num_episodes}] | "
            f"Reward: {total_reward:>7.2f} | "
            f"Steps: {steps:>4d} | "
            f"Cmd: {cmd_vx:.2f} m/s | "
            f"3-Leg Vx: {post_vx:>.2f} m/s | "
            f"Speed Retained: {retention:>5.1f}% | "
            f"Survived: {'YES' if survived else 'NO'}"
        )

    print("-" * 72)
    print(f"Mean Reward          : {np.mean(ep_rewards):.2f} +/- {np.std(ep_rewards):.2f}")
    print(f"Mean Survival Time   : {np.mean(ep_lengths) * env.dt:.2f}s ({np.mean(ep_lengths):.0f} steps)")
    print(f"Mean 3-Leg Velocity  : {np.mean(actual_speeds):.2f} m/s")
    print(f"Mean Velocity Retained: {np.mean(speed_retentions):.1f}% of commanded speed")
    print(f"Overall Survival Rate: {np.mean(survivals) * 100.0:.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ADAPT-X Phase 4 Curriculum")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase4/best_policy.pt")
    parser.add_argument("--config", type=str, default="configs/phase4_curriculum_config.yaml")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--damage", type=str, default="amputated_leg")
    parser.add_argument("--leg", type=str, default="fr")
    parser.add_argument("--damage-step", type=int, default=200)
    args = parser.parse_args()

    evaluate_curriculum(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        stage=args.stage,
        num_episodes=args.episodes,
        render=args.render,
        damage=args.damage,
        leg=args.leg,
        damage_step=args.damage_step
    )
