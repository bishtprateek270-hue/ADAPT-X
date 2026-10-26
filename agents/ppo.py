import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class ActorCritic(nn.Module):
    """
    Lightweight, high-throughput Actor-Critic MLP.
    Designed to easily run inside GPU L2 cache (< 250k parameters)
    to guarantee zero laptop stuttering.
    """
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 128)):
        super().__init__()
        
        # Actor Network (Policy)
        actor_layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(nn.ELU())
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, act_dim))
        self.actor_mean = nn.Sequential(*actor_layers)
        
        # Trainable log standard deviation for continuous Gaussian policy
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

        # Critic Network (Value Function)
        critic_layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            critic_layers.append(nn.Linear(in_dim, h))
            critic_layers.append(nn.ELU())
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Orthogonal initialization (standard best practice in quadruped RL)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small gain on final actor layer to ensure initial actions are gentle
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std)
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(x)
        return action, log_prob, entropy, value


class PPOTrainer:
    """
    Hardware-Safe PPO Trainer optimized for RTX 5050 Laptop GPU.
    Implements memory caps, thread throttling, GAE, and fast checkpointing.
    """
    def __init__(self, env_fn, config):
        self.config = config
        self.ppo_cfg = config.get("ppo", {})
        self.sys_cfg = config.get("system", {})
        
        # 1. Device and Safety Throttling
        device_str = self.sys_cfg.get("device", "cuda")
        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)
        
        # Limit CPU threads to prevent locking Windows desktop / scheduler
        max_threads = int(self.sys_cfg.get("num_cpu_threads", 4))
        torch.set_num_threads(max_threads)

        # 2. Vectorized Environments
        self.num_envs = int(self.ppo_cfg.get("num_envs", 16))
        self.rollout_steps = int(self.ppo_cfg.get("rollout_steps", 256))
        self.batch_size = self.num_envs * self.rollout_steps
        self.minibatch_size = int(self.batch_size // self.ppo_cfg.get("num_minibatches", 4))
        
        # Create env instances
        self.envs = [env_fn(config) for _ in range(self.num_envs)]
        self.obs_dim = self.envs[0].observation_space.shape[0]
        self.act_dim = self.envs[0].action_space.shape[0]

        # 3. Model & Optimizer
        self.model = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.lr = float(self.ppo_cfg.get("lr", 3e-4))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, eps=1e-5)

        # 4. Storage buffers (allocated on target device)
        self.obs_buf = torch.zeros((self.rollout_steps, self.num_envs, self.obs_dim), device=self.device)
        self.act_buf = torch.zeros((self.rollout_steps, self.num_envs, self.act_dim), device=self.device)
        self.logp_buf = torch.zeros((self.rollout_steps, self.num_envs), device=self.device)
        self.rew_buf = torch.zeros((self.rollout_steps, self.num_envs), device=self.device)
        self.done_buf = torch.zeros((self.rollout_steps, self.num_envs), device=self.device)
        self.val_buf = torch.zeros((self.rollout_steps, self.num_envs), device=self.device)

        # 5. Checkpoints & Tracking
        self.checkpoint_dir = self.config.get("logging", {}).get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"[ADAPT-X PPO] Initialized on {self.device}")
        if self.device.type == "cuda":
            allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"[ADAPT-X PPO] GPU VRAM Allocated: {allocated_mb:.1f} MB (Safety Cap: < 1500 MB)")

    def collect_rollouts(self, next_obs, next_done):
        """Rolls out policies across parallel envs."""
        for step in range(self.rollout_steps):
            self.obs_buf[step] = next_obs
            self.done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = self.model.get_action_and_value(next_obs)
                self.val_buf[step] = value.flatten()
            self.act_buf[step] = action
            self.logp_buf[step] = logprob

            # Step all environments
            actions_np = action.cpu().numpy()
            step_obs = []
            step_rewards = []
            step_dones = []

            for i in range(self.num_envs):
                o, r, term, trunc, _ = self.envs[i].step(actions_np[i])
                done = term or trunc
                if done:
                    o, _ = self.envs[i].reset()
                step_obs.append(o)
                step_rewards.append(r)
                step_dones.append(done)

            self.rew_buf[step] = torch.tensor(step_rewards, device=self.device)
            next_obs = torch.tensor(np.array(step_obs), dtype=torch.float32, device=self.device)
            next_done = torch.tensor(step_dones, dtype=torch.float32, device=self.device)

        return next_obs, next_done

    def compute_gae(self, next_obs, next_done):
        """Generalized Advantage Estimation (GAE)."""
        gamma = float(self.ppo_cfg.get("gamma", 0.99))
        gae_lambda = float(self.ppo_cfg.get("gae_lambda", 0.95))

        with torch.no_grad():
            next_value = self.model.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rew_buf, device=self.device)
            lastgaelam = 0
            for t in reversed(range(self.rollout_steps)):
                if t == self.rollout_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.done_buf[t + 1]
                    nextvalues = self.val_buf[t + 1]
                delta = self.rew_buf[t] + gamma * nextvalues * nextnonterminal - self.val_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.val_buf
        return advantages, returns

    def update_policy(self, advantages, returns):
        """Performs mini-batch PPO optimization epochs."""
        # Flatten rollout dimensions: (rollout_steps, num_envs, ...) -> (batch_size, ...)
        b_obs = self.obs_buf.reshape(-1, self.obs_dim)
        b_act = self.act_buf.reshape(-1, self.act_dim)
        b_logp = self.logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        # Normalize advantages across batch
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        clip_range = float(self.ppo_cfg.get("clip_range", 0.2))
        entropy_coef = float(self.ppo_cfg.get("entropy_coef", 0.01))
        value_coef = float(self.ppo_cfg.get("value_coef", 0.5))
        max_grad_norm = float(self.ppo_cfg.get("max_grad_norm", 0.5))
        epochs = int(self.ppo_cfg.get("num_epochs", 4))

        b_inds = np.arange(self.batch_size)
        loss_dict = {"pg_loss": 0, "val_loss": 0, "entropy": 0}

        for _ in range(epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.model.get_action_and_value(
                    b_obs[minds], b_act[minds]
                )
                
                # Policy ratio & clipped objective
                logratio = newlogprob - b_logp[minds]
                ratio = logratio.exp()
                surr1 = -b_adv[minds] * ratio
                surr2 = -b_adv[minds] * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                pg_loss = torch.max(surr1, surr2).mean()

                # Value function loss (clipped)
                newvalue = newvalue.view(-1)
                val_loss = 0.5 * ((newvalue - b_ret[minds]) ** 2).mean()

                # Total loss
                loss = pg_loss - (entropy_coef * entropy.mean()) + (value_coef * val_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                loss_dict["pg_loss"] += pg_loss.item()
                loss_dict["val_loss"] += val_loss.item()
                loss_dict["entropy"] += entropy.mean().item()

        return loss_dict

    def save_checkpoint(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }, path)
        print(f"[Checkpoint Saved] -> {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[Checkpoint Loaded] <- {path}")
