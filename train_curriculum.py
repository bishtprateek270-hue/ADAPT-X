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
    parser = argparse.ArgumentParser(description="ADAPT-X: Train Phase 4 Curriculum Locomotion")
    parser.add_argument("--config", type=str, default="configs/phase4_curriculum_config.yaml")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2, 3], help="Curriculum stage (1: Pre-broken, 2: Dynamic, 3: Stress)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=str, default="checkpoints/phase3/best_policy.pt", help="Warm-start from Phase 3 adaptive model")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.stage is not None:
        config["curriculum"]["stage"] = args.stage
        if args.stage == 3:
            config["curriculum"]["push_disturbances"] = True
    if args.steps is not None:
        config["ppo"]["total_timesteps"] = args.steps
    if args.num_envs is not None:
        config["ppo"]["num_envs"] = args.num_envs
    if args.dry_run:
        config["ppo"]["total_timesteps"] = config["ppo"]["num_envs"] * config["ppo"]["rollout_steps"] * 2

    stage = config.get("curriculum", {}).get("stage", 1)
    stage_desc = {
        1: "Stage 1: Pre-Broken 3-Legged Gait Discovery (Spawns 100% Damaged)",
        2: "Stage 2: Dynamic Mid-Stride Fault Transition with Forward Momentum",
        3: "Stage 3: Search & Rescue Stress Test (Dynamic Damage + Push Disturbance Impulses)"
    }[stage]

    total_timesteps = int(config["ppo"]["total_timesteps"])
    num_envs = int(config["ppo"]["num_envs"])
    rollout_steps = int(config["ppo"]["rollout_steps"])
    batch_size = num_envs * rollout_steps
    num_iterations = total_timesteps // batch_size

    print("=" * 75)
    print(f"      ADAPT-X: PHASE 4 CURRICULUM TRAINING ({stage_desc.split(':')[0]})")
    print("=" * 75)
    print(f"Curriculum Focus       : {stage_desc}")
    print(f"Total Target Timesteps : {total_timesteps:,}")
    print(f"Parallel Environments  : {num_envs} (Laptop GPU Safe Mode)")
    print(f"Batch Size per Iter    : {batch_size:,} transitions")
    print(f"Push Disturbances      : {config.get('curriculum', {}).get('push_disturbances', False)}")
    print(f"Total PPO Iterations   : {num_iterations}")
    print("=" * 75)

    env_fn = make_env_fn(config)
    trainer = AdaptivePPOTrainer(env_fn, config)

    # Warm-start from Phase 3 checkpoint if available
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
        print(f"[Warm-Start] Successfully loaded Phase 3 checkpoint from: {args.resume}")
    else:
        print("[Notice] Training from initialized weights.")

    # Reset environments
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

    stage_checkpoint_dir = os.path.join(trainer.checkpoint_dir, f"stage_{stage}")
    os.makedirs(stage_checkpoint_dir, exist_ok=True)

    print("\n[Starting Phase 4 Curriculum Training... Press Ctrl+C to save and exit]\n")
    try:
        for iteration in range(1, num_iterations + 1):
            iter_start = time.time()

            next_obs, next_hist, next_done = trainer.collect_rollouts(next_obs, next_hist, next_done)
            advantages, returns = trainer.compute_gae(next_obs, next_hist, next_done)
            loss_dict = trainer.update_policy(advantages, returns)

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
                trainer.save_checkpoint(os.path.join(stage_checkpoint_dir, "best_policy.pt"))
                trainer.save_checkpoint(os.path.join(trainer.checkpoint_dir, "best_policy.pt"))

            if iteration % 10 == 0:
                trainer.save_checkpoint(os.path.join(stage_checkpoint_dir, f"step_{iteration*batch_size}.pt"))

            if args.dry_run and iteration >= 2:
                print("\n[Dry-Run Complete] Phase 4 Curriculum verified successfully with 0 errors!")
                break

    except KeyboardInterrupt:
        print("\n[Interrupted by User] Saving emergency checkpoint...")
        trainer.save_checkpoint(os.path.join(stage_checkpoint_dir, "interrupted_policy.pt"))

    trainer.save_checkpoint(os.path.join(stage_checkpoint_dir, "latest_policy.pt"))
    trainer.save_checkpoint(os.path.join(trainer.checkpoint_dir, "latest_policy.pt"))
    print(f"\n[Training Finished] Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
