# -*- coding: utf-8 -*-
"""Probabilistic Transformer target predictor for the graduation-thesis gate.

The public interface intentionally matches ``GRUGaussianPredictor`` so the
belief manager and evaluation code can compare both models without changing
the closed-loop semantics:

``forward(seq) -> (mu, var, latent)`` and
``rollout_lost(seq, steps) -> (mus, vars)``.

Inputs and outputs use the same normalized units as the GRU predictor:
``[x/L, y/L, vx/8, vy/8, visibility]`` -> future ``[x/L, y/L]``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal_encoding(
    length: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a deterministic ``(1, length, width)`` positional encoding."""
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    even_width = (width + 1) // 2
    scale = torch.exp(
        torch.arange(even_width, device=device, dtype=dtype)
        * (-math.log(10_000.0) / max(width, 1))
        * 2.0
    )
    angles = position * scale.unsqueeze(0)
    encoding = torch.zeros(length, width, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(angles[:, :encoding[:, 0::2].shape[1]])
    encoding[:, 1::2] = torch.cos(angles[:, :encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class TransformerGaussianPredictor(nn.Module):
    """Causal temporal encoder with a diagonal-Gaussian multi-step head."""

    def __init__(
        self,
        in_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        horizon: int = 5,
        ff_dim: int | None = None,
        dropout: float = 0.1,
        causal: bool = True,
        rollout_eta: float = 0.05,
        rollout_q0: float = 0.04,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError(
                f"d_model={d_model} must be divisible by nhead={nhead}")
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.horizon = int(horizon)
        self.ff_dim = int(ff_dim or 4 * d_model)
        self.dropout = float(dropout)
        self.causal = bool(causal)
        self.rollout_eta = float(rollout_eta)
        self.rollout_q0 = float(rollout_q0)

        self.input_proj = nn.Linear(self.in_dim, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, self.num_layers)
        self.final_norm = nn.LayerNorm(self.d_model)
        self.head_mu = nn.Linear(self.d_model, 2 * self.horizon)
        self.head_logvar = nn.Linear(self.d_model, 2 * self.horizon)

    def model_kwargs(self) -> dict:
        """Serializable constructor arguments for versioned checkpoints."""
        return {
            "in_dim": self.in_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "horizon": self.horizon,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "causal": self.causal,
            "rollout_eta": self.rollout_eta,
            "rollout_q0": self.rollout_q0,
        }

    def forward(self, seq, h0=None):
        """``seq: (B,T,5)`` -> ``mu,var: (B,K,2)`` and final latent."""
        del h0  # Kept only for interface parity with the GRU predictor.
        x = self.input_proj(seq) * math.sqrt(self.d_model)
        x = x + _sinusoidal_encoding(
            x.shape[1], self.d_model, device=x.device, dtype=x.dtype)
        mask = None
        if self.causal:
            mask = torch.triu(
                torch.ones(
                    x.shape[1], x.shape[1], device=x.device,
                    dtype=torch.bool),
                diagonal=1,
            )
        encoded = self.encoder(x, mask=mask)
        latent = self.final_norm(encoded[:, -1])
        mu = self.head_mu(latent).view(-1, self.horizon, 2)
        logvar = self.head_logvar(latent).view(
            -1, self.horizon, 2).clamp(-8, 6)
        return mu, logvar.exp(), latent

    @staticmethod
    def nll_loss(mu, var, target):
        return 0.5 * (
            ((target - mu) ** 2) / var + var.log()).sum(-1).mean()

    @torch.no_grad()
    def rollout_lost(self, seq, steps, eta=None, q0=None):
        """Autoregressive lost-observation rollout, matching the GRU contract."""
        eta = self.rollout_eta if eta is None else float(eta)
        q0 = self.rollout_q0 if q0 is None else float(q0)
        mus, vars_ = [], []
        cur = seq.clone()
        var_acc = torch.zeros(
            cur.shape[0], 2, device=cur.device, dtype=cur.dtype)
        for t in range(steps):
            mu, var, _ = self.forward(cur)
            step_mu = mu[:, 0]
            var_acc = var_acc + var[:, 0] + q0 * (1 + eta * (t + 1))
            mus.append(step_mu)
            vars_.append(var_acc.clone())
            pseudo = torch.cat(
                [
                    step_mu,
                    torch.zeros_like(step_mu),
                    -torch.ones(
                        step_mu.shape[0], 1,
                        device=step_mu.device, dtype=step_mu.dtype),
                ],
                dim=-1,
            )
            cur = torch.cat([cur[:, 1:], pseudo.unsqueeze(1)], dim=1)
        if not mus:
            shape = (cur.shape[0], 0, 2)
            return cur.new_empty(shape), cur.new_empty(shape)
        return torch.stack(mus, 1), torch.stack(vars_, 1)

    @staticmethod
    def confidence(
        t_lost: float,
        sigma_trace: float,
        l1: float = 0.02,
        l2: float = 0.05,
    ) -> float:
        return float(math.exp(-l1 * t_lost - l2 * sigma_trace))

    @staticmethod
    def picp(mu, var, target, z: float = 1.96) -> float:
        sigma = var.sqrt()
        inside = ((target - mu).abs() <= z * sigma).all(-1).float()
        return float(inside.mean())
