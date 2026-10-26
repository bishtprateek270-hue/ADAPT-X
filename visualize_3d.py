import os
import sys
import time
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.quadruped_env import QuadrupedEnv
from agents.adaptive_ppo import AdaptiveActorCritic

try:
    import mujoco
    import mujoco.viewer
    VIEWER_AVAILABLE = True
except Exception as e:
    VIEWER_AVAILABLE = False


def launch_3d_disaster_simulation(checkpoint_path="checkpoints/phase3/best_policy.pt", damage="amputated_leg", leg="fr", damage_step=150):
    if not VIEWER_AVAILABLE:
        print("[Error] MuJoCo graphical viewer is not available. Please verify graphics drivers.")
        return

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    arena_path = os.path.join(curr_dir, "models", "disaster_arena.xml")

    # Initialize environment with 3D Disaster Arena model
    env = QuadrupedEnv(model_path=arena_path)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    history_dim = env.history_dim

    device = torch.device("cpu")
    model = AdaptiveActorCritic(
        obs_dim=obs_dim,
        history_dim=history_dim,
        act_dim=act_dim,
        latent_dim=16
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Loaded Checkpoint] {checkpoint_path}")
    else:
        print(f"[Notice] Checkpoint {checkpoint_path} not found. Running with initial weights.")

    model.eval()

    print("=" * 70)
    print("      ADAPT-X: 3D DISASTER ARENA INTERACTIVE SIMULATION")
    print("=" * 70)
    print("Controls in 3D Window:")
    print("  * Left Click + Drag  : Rotate 3D Camera")
    print("  * Right Click + Drag : Zoom in/out")
    print("  * Double Click Link  : Select body / follow robot")
    print(f"Scheduled Failure    : [{damage.upper()} on {leg.upper()} leg] at step {damage_step} (~3.0s)")
    print("=" * 70)

    obs, hist, _ = env.reset_adaptive()

    # Launch MuJoCo 3D Interactive Desktop Window
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        while viewer.is_running():
            step_start = time.perf_counter()
            step += 1

            # Inject failure mid-run
            if damage != "none" and step == damage_step:
                env.inject_fault(damage, leg=leg)
                print(f"\n  >>> STEP {step}: LIVE DAMAGE INJECTED [{damage.upper()} on {leg.upper()}]!")
                print("  >>> Watch the robot re-balance and adapt its 3-legged gait in the 3D window!\n")

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            hist_t = torch.tensor(hist, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                z = model.get_latent(hist_t)
                x = torch.cat([obs_t, z], dim=-1)
                action = model.actor_mean(x).squeeze(0).numpy()

            obs, hist, reward, terminated, truncated, info = env.step_adaptive(action)

            # Sync 3D physics state to visual window
            viewer.sync()

            # Maintain natural real-time 50 Hz visual rate
            elapsed = time.perf_counter() - step_start
            time.sleep(max(0.0, env.dt - elapsed))

            if terminated or truncated:
                print(f"[Resetting Episode after {step} steps...]")
                obs, hist, _ = env.reset_adaptive()
                step = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch ADAPT-X 3D Visual Simulation Window")
    parser.add_argument("--drone", action="store_true", help="Launch 3D City Drone Flight Simulation instead of Quadruped")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase3/best_policy.pt")
    parser.add_argument("--damage", type=str, default="amputated_leg")
    parser.add_argument("--leg", type=str, default="fr", choices=["fr", "fl", "rr", "rl"])
    parser.add_argument("--damage-step", type=int, default=150)
    args = parser.parse_args()

    if args.drone:
        from fly_drone import run_drone_simulation
        # Map leg to rotor if specified
        rotor = args.leg if args.leg in ["fr", "fl", "rr", "rl"] else "fr"
        drone_damage = "dead_motor" if args.damage != "none" else "none"
        run_drone_simulation(mode="cruise", damage=drone_damage, rotor=rotor, damage_step=args.damage_step)
    else:
        launch_3d_disaster_simulation(
            checkpoint_path=args.checkpoint,
            damage=args.damage,
            leg=args.leg,
            damage_step=args.damage_step
        )

