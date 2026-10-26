import os
import sys
import time
import argparse
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.quadruped_env import QuadrupedEnv
from agents.adaptive_ppo import AdaptivePPOTrainer


def make_env_fn(config):
    def _init(cfg=None):
        return QuadrupedEnv(config=cfg or config)
    return _init


def main():
    parser = argparse.ArgumentParser(description="ADAPT-X: Train Phase 3 Adaptive Architecture")
    parser.add_argument("--config", type=str, default="configs/phase3_adaptive_config.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

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

    history_len = config.get("adaptive", {}).get("history_len", 20)
    latent_dim = config.get("adaptive", {}).get("latent_dim", 16)

    print("=" * 72)
    print("      ADAPT-X: PHASE 3 ADAPTIVE ARCHITECTURE (SELF-HEALING BRAIN)")
    print("=" * 72)
    print(f"Total Target Timesteps : {total_timesteps:,}")
    print(f"Parallel Environments  : {num_envs} (Laptop GPU Safe Mode)")
    print(f"Rollout Steps per Env  : {rollout_steps}")
    print(f"Batch Size per Iter    : {batch_size:,} transitions")
    print(f"Temporal History Buffer: {history_len} steps (0.40s memory window)")
    print(f"Latent Morph. Space z_t: {latent_dim} dimensions")
    print(f"Total PPO Iterations   : {num_iterations}")
    print("=" * 72)

    env_fn = make_env_fn(config)
    trainer = AdaptivePPOTrainer(env_fn, config)

    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)

    # Initialize environment adaptive states
    reset_obs = []
    reset_hist = []
    for env in trainer.envs:
        o, h, _ = env.reset_adaptive()
        reset_obs.append(o)
        reset_hist.append(h)
    
    next_obs = torch.tensor(np.array(reset_obs), dtype=torch.float32, device=trainer.device)
    next_hist = torch.tensor(np.array(reset_hist), dtype=torch.float32, device=trainer.device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=trainer.device)

    start_time = time.time()
    best_mean_reward = -float("inf")

    print("\n[Starting Phase 3 Adaptive Training Loop... Press Ctrl+C to stop]\n")
    try:
        for iteration in range(1, num_iterations + 1):
            iter_start = time.time()

            # 1. Rollout with history encoder
            next_obs, next_hist, next_done = trainer.collect_rollouts(next_obs, next_hist, next_done)

            # 2. Compute GAE
            advantages, returns = trainer.compute_gae(next_obs, next_hist, next_done)

            # 3. Policy update
            loss_dict = trainer.update_policy(advantages, returns)

            # 4. Metrics & Logging
            iter_time = time.time() - iter_start
            fps = int(batch_size / max(iter_time, 1e-4))
            elapsed = time.time() - start_time
            mean_step_reward = trainer.rew_buf.mean().item()

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

            if mean_step_reward > best_mean_reward:
                best_mean_reward = mean_step_reward
                trainer.save_checkpoint(os.path.join(trainer.checkpoint_dir, "best_policy.pt"))

            if iteration % 10 == 0:
                trainer.save_checkpoint(os.path.join(trainer.checkpoint_dir, f"checkpoint_step_{iteration*batch_size}.pt"))

            if args.dry_run and iteration >= 2:
                print("\n[Dry-Run Complete] Phase 3 Adaptive pipeline verified successfully with 0 errors!")
                break

    except KeyboardInterrupt:
        print("\n[Interrupted by User] Saving emergency checkpoint...")
        trainer.save_checkpoint(os.path.join(trainer.checkpoint_dir, "interrupted_policy.pt"))

    trainer.save_checkpoint(os.path.join(trainer.checkpoint_dir, "latest_policy.pt"))
    print(f"\n[Training Finished] Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
