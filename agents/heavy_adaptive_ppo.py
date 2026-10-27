"""
ADAPT-X: Heavy Deep RL Neural Architecture for NVIDIA DGX Enterprise Systems
Optimized for NVIDIA DGX Spark (Grace Blackwell GB10 / CUDA 13.0).

Features:
  1. Temporal Transformer History Encoder:
     - Sequence modeling over H=20 rolling proprioception timesteps.
     - Multi-Head Self-Attention (4 heads, d_model=256) + LayerNorm + GELU.
     - Positional Embeddings capturing phase and foot-impact dynamics.
     - Compresses sequence to latent morphological vector z_t in R^32.
  2. Deep Residual Actor & Critic Backbones:
     - 512-wide residual MLP blocks with skip connections and LayerNorm.
     - Separate deep critic for high-fidelity value surface approximation.
  3. Native BF16 / FP16 Tensor Core Acceleration.
"""

import os
import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal proprioceptive sequence."""
    def __init__(self, d_model=256, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TemporalTransformerEncoder(nn.Module):
    """
    Multi-Head Temporal Transformer for Rapid Motor Adaptation (RMA).
    Takes a rolling sequence of proprioceptive observations (B, H=20, D=42)
    and uses self-attention across timesteps to infer morphology, damage,
    and ground compliance into a latent state z_t in R^32.
    """
    def __init__(self, single_dim=42, seq_len=20, d_model=256, nhead=4, num_layers=2, latent_dim=32):
        super().__init__()
        self.seq_len = seq_len
        self.single_dim = single_dim
        self.latent_dim = latent_dim

        # Input projection: raw proprioceptive vector -> transformer model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(single_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection head: pooled features -> latent morphology vector z_t
        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p, gain=1.0)
            elif p.dim() == 1:
                nn.init.zeros_(p)

    def forward(self, history):
        """
        history: either (B, seq_len * single_dim) flat buffer,
                 or (B, seq_len, single_dim) 3D sequence tensor.
        """
        if history.dim() == 2:
            b = history.size(0)
            history = history.view(b, self.seq_len, self.single_dim)

        x = self.input_proj(history)  # (B, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)       # (B, seq_len, d_model)

        # Global temporal attention pooling (mean over timesteps + last step feature)
        pooled = 0.5 * (x.mean(dim=1) + x[:, -1, :])
        z = self.out_head(pooled)     # (B, latent_dim)
        return z


class ResBlock(nn.Module):
    """Heavy Residual MLP Block with LayerNorm and GELU."""
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.block(x)


class HeavyAdaptiveActorCritic(nn.Module):
    """
    Heavy Deep RL Policy for DGX Spark / Blackwell Enterprise Systems.
    Pairs a Temporal Transformer history encoder with deep 512-wide
    Residual Actor and Critic trunks.
    """
    def __init__(
        self,
        obs_dim=48,
        single_proprio_dim=42,
        history_len=20,
        act_dim=12,
        latent_dim=32,
        hidden_dim=512
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.history_len = history_len
        self.history_dim = history_len * single_proprio_dim

        # 1. Temporal Transformer History Encoder
        self.history_encoder = TemporalTransformerEncoder(
            single_dim=single_proprio_dim,
            seq_len=history_len,
            d_model=256,
            nhead=4,
            num_layers=2,
            latent_dim=latent_dim
        )

        in_dim = obs_dim + latent_dim

        # 2. Deep Residual Actor Backbone
        self.actor_stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU()
        )
        self.actor_res1 = ResBlock(hidden_dim)
        self.actor_res2 = ResBlock(hidden_dim)
        self.actor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, act_dim)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

        # 3. Deep Residual Critic Backbone
        self.critic_stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU()
        )
        self.critic_res1 = ResBlock(hidden_dim)
        self.critic_res2 = ResBlock(hidden_dim)
        self.critic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self._init_actor_critic_weights()

    def _init_actor_critic_weights(self):
        for m in [self.actor_stem, self.actor_res1, self.actor_res2,
                  self.critic_stem, self.critic_res1, self.critic_res2]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def get_latent(self, history):
        return self.history_encoder(history)

    def get_value(self, obs, history):
        z = self.history_encoder(history)
        x = torch.cat([obs, z], dim=-1)
        feat = self.critic_stem(x)
        feat = self.critic_res1(feat)
        feat = self.critic_res2(feat)
        return self.critic_head(feat)

    def get_action_and_value(self, obs, history, action=None):
        z = self.history_encoder(history)
        x = torch.cat([obs, z], dim=-1)

        # Actor pass
        act_feat = self.actor_stem(x)
        act_feat = self.actor_res1(act_feat)
        act_feat = self.actor_res2(act_feat)
        mean = self.actor_head(act_feat)

        std = torch.exp(self.actor_log_std)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)

        # Critic pass
        crit_feat = self.critic_stem(x)
        crit_feat = self.critic_res1(crit_feat)
        crit_feat = self.critic_res2(crit_feat)
        value = self.critic_head(crit_feat)

        return action, log_prob, entropy, value, z


class HeavyDeployablePolicy(nn.Module):
    """
    Zero-overhead inference wrapper for Heavy Transformer-Actor policy.
    Compiles cleanly to ONNX and TorchScript.
    """
    def __init__(self, history_encoder, actor_stem, actor_res1, actor_res2, actor_head):
        super().__init__()
        self.history_encoder = history_encoder
        self.actor_stem = actor_stem
        self.actor_res1 = actor_res1
        self.actor_res2 = actor_res2
        self.actor_head = actor_head

    def forward(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        z = self.history_encoder(history)
        x = torch.cat([obs, z], dim=-1)
        feat = self.actor_stem(x)
        feat = self.actor_res1(feat)
        feat = self.actor_res2(feat)
        return self.actor_head(feat)
