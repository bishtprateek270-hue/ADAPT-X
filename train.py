import os
import sys
import time
import argparse
import yaml
import torch
import numpy as np

# Ensure root dir is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.quadruped_env import QuadrupedEnv
from agents.ppo import PPOTrainer


def make_env_fn(config):
    def _init(cfg=None):
        return QuadrupedEnv(config=cfg or config)
    return _init


def main():
    parser = argparse.ArgumentParser(description="ADAPT-X: Train Phase 1 Quadruped Locomotion")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to config YAML")
    parser.add_argument("--steps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel environments")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 iteration to verify pipeline without full training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.steps is not None:
        config["ppo"]["total_timesteps"] = args.steps
    if args.num_envs is not None:
        config["ppo"]["num_envs"] = args.num_envs
    if args.dry_run:
        config["ppo"]["total_timesteps"] = config["ppo"]["num_envs"] * config["ppo"]["rollout_steps"] * 2

    total_timesteps = int(config["ppo"]["total_timesteps"])
    num_envs = int(config["ppo"]["num_envs"])
    rollout_steps = int(config["ppo"]["rollout_steps"])
    batch_size = num_envs * rollout_steps
    num_iterations = total_timesteps // batch_size

    print("=" * 70)
    print("      ADAPT-X: SELF-HEALING QUADRUPED (PHASE 1 LOCOMOTION)")
    print("=" * 70)
    print(f"Total Target Timesteps : {total_timesteps:,}")
    print(f"Parallel Environments  : {num_envs} (Laptop GPU Safe Mode)")
    print(f"Rollout Steps per Env  : {rollout_steps}")
    print(f"Batch Size per Iter    : {batch_size:,} transitions")
    print(f"Total PPO Iterations   : {num_iterations}")
    print("=" * 70)

    # Instantiate Trainer
    env_fn = make_env_fn(config)
    trainer = PPOTrainer(env_fn, config)

    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)

    # Initialize environment states
    reset_obs = []
    for env in trainer.envs:
        o, _ = env.reset()
        reset_obs.append(o)
    
    next_obs = torch.tensor(np.array(reset_obs), dtype=torch.float32, device=trainer.device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=trainer.device)

    start_time = time.time()
    best_mean_reward = -float("inf")

    print("\n[Starting Training Loop... Press Ctrl+C to stop and save checkpoint]\n")
    try:
        for iteration in range(1, num_iterations + 1):
            iter_start = time.time()

            # 1. Collect rollouts
            next_obs, next_done = trainer.collect_rollouts(next_obs, next_done)

            # 2. Compute GAE
            advantages, returns = trainer.compute_gae(next_obs, next_done)

            # 3. Policy & Value Updates
            loss_dict = trainer.update_policy(advantages, returns)

            # 4. Metrics & Logging
            iter_time = time.time() - iter_start
            fps = int(batch_size / max(iter_time, 1e-4))
            elapsed = time.time() - start_time
            mean_step_reward = trainer.rew_buf.mean().item()
            approx_ep_reward = mean_step_reward * 50.0  # Approx 1-second window reward

            # Track VRAM usage safely
            vram_mb = 0.0
            if trainer.device.type == "cuda":
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)

            print(
                f"Iter [{iteration:04d}/{num_iterations:04d}] | "
                f"Steps: {iteration * batch_size:>7,d} | "
                f"FPS: {fps:>5d} | "
                f"Mean Rew: {mean_step_reward:>6.3f} | "
                f"PG Loss: {loss_dict['pg_loss']:>6.3f} | "
                f"VRAM: {vram_mb:>5.1f}MB | "
                f"Time: {elapsed:>5.1f}s"
            )

            # Save best checkpoint
            if mean_step_reward > best_mean_reward:
                best_mean_reward = mean_step_reward
                trainer.save_checkpoint(os.path.join(trainer.checkpoint_dir, "best_policy.pt"))

            # Save periodic checkpoint
            if iteration % 10 == 0:
                trainer.save_checkpoint(os.path.join(trainer.checkpoint_dir, f"checkpoint_step_{iteration*batch_size}.pt"))

            if args.dry_run and iteration >= 2:
                print("\n[Dry-Run Complete] Pipeline verified successfully with 0 errors!")
                break

    except KeyboardInterrupt:
        print("\n[Interrupted by User] Saving emergency checkpoint...")
        trainer.save_checkpoint(os.path.join(trainer.checkpoint_dir, "interrupted_policy.pt"))

    # Final save
    trainer.save_checkpoint(os.path.join(trainer.checkpoint_dir, "latest_policy.pt"))
    print(f"\n[Training Finished] Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
