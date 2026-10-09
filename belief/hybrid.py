# -*- coding: utf-8 -*-
"""G1门后的IMM先验 + GRU高斯残差信念模块。

IMM始终负责物理状态与协方差传播。GRU读取最近一段观测/IMM历史，只在失观步
预测位置残差及其对角协方差；可见步直接采用IMM量测更新，避免学习器覆盖真实量测。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from belief.gru_predictor import GRUGaussianPredictor
from belief.imm_filter import IMMFilter


HYBRID_IN_DIM = 15
VEL_NORM = 8.0
LOST_NORM = 100.0


class IMMResidualGRU(nn.Module):
    """由IMM先验与观测历史预测归一化位置残差和残差方差。"""

    def __init__(self, in_dim: int = HYBRID_IN_DIM, hidden: int = 64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.gru = nn.GRU(self.in_dim, self.hidden, batch_first=True)
        self.head_mu = nn.Linear(self.hidden, 2)
        self.head_logvar = nn.Linear(self.hidden, 2)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(seq)
        last = out[:, -1]
        delta = self.head_mu(last)
        var = self.head_logvar(last).clamp(-10.0, 4.0).exp()
        return delta, var

    @staticmethod
    def nll_loss(delta: torch.Tensor, var: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        return 0.5 * (((target - delta) ** 2) / var + var.log()).sum(-1).mean()


def hybrid_feature(observed_pos: np.ndarray, observed_vel: np.ndarray,
                   visible: bool, imm_mu: np.ndarray, imm_cov: np.ndarray,
                   model_probs: np.ndarray, lost_steps: int, arena: float,
                   vel_norm: float = VEL_NORM,
                   lost_norm: float = LOST_NORM) -> np.ndarray:
    """构造15维单步特征；失观时代理观测取IMM先验而不偷看真值。"""
    imm_mu = np.asarray(imm_mu, dtype=np.float32)
    imm_cov = np.asarray(imm_cov, dtype=np.float32)
    probs = np.asarray(model_probs, dtype=np.float32).reshape(-1)
    if probs.size != 3:
        raise ValueError(f"hybrid固定使用CV+双CT三模型IMM，收到{probs.size}个模型概率")
    if visible:
        proxy_pos = np.asarray(observed_pos, dtype=np.float32)
        proxy_vel = np.asarray(observed_vel, dtype=np.float32)
    else:
        proxy_pos = imm_mu[:2]
        proxy_vel = imm_mu[2:4]
    pos_var_norm = np.maximum(np.diag(imm_cov)[:2] / (float(arena) ** 2), 0.0)
    feature = np.concatenate([
        proxy_pos / float(arena),
        proxy_vel / float(vel_norm),
        np.array([1.0 if visible else -1.0], dtype=np.float32),
        imm_mu[:2] / float(arena),
        imm_mu[2:4] / float(vel_norm),
        np.log1p(pos_var_norm),
        probs,
        np.array([min(max(int(lost_steps), 0), int(lost_norm)) /
                  float(lost_norm)], dtype=np.float32),
    ]).astype(np.float32)
    if feature.shape != (HYBRID_IN_DIM,) or not np.isfinite(feature).all():
        raise FloatingPointError(f"invalid hybrid feature: shape={feature.shape}")
    return feature


def load_hybrid_checkpoint(path: str) -> tuple[IMMResidualGRU, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"hybrid checkpoint格式错误: {path}")
    model = IMMResidualGRU(
        in_dim=int(checkpoint.get("in_dim", HYBRID_IN_DIM)),
        hidden=int(checkpoint.get("hidden", 64)),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {key: value for key, value in checkpoint.items() if key != "model"}
    return model, metadata


@dataclass
class HybridEstimate:
    mean: np.ndarray
    covariance: np.ndarray
    imm_mean: np.ndarray
    imm_covariance: np.ndarray
    residual_mean: np.ndarray
    residual_variance: np.ndarray


class HybridBeliefManager:
    """与train_belief中的其他manager同构的在环hybrid状态机。"""

    def __init__(self, residual_model: IMMResidualGRU, variant: str,
                 arena: float, seq_len: int = 20,
                 residual_temperature: float = 1.0,
                 filter_kwargs: dict | None = None):
        if variant not in ("point", "belief"):
            raise ValueError(variant)
        self.model = residual_model
        self.variant = variant
        self.arena = float(arena)
        self.seq_len = int(seq_len)
        self.residual_temperature = float(residual_temperature)
        self.filter_kwargs = dict(filter_kwargs or {})
        self.filt = IMMFilter(**self.filter_kwargs)
        self.window: torch.Tensor | None = None
        self.last_estimate: HybridEstimate | None = None

    def _push(self, feature: np.ndarray) -> None:
        row = torch.as_tensor(feature, dtype=torch.float32).view(1, 1, -1)
        if self.window is None:
            self.window = row.repeat(1, self.seq_len, 1)
        else:
            self.window = torch.cat([self.window[:, 1:], row], dim=1)

    def reset(self, env) -> np.ndarray:
        self.filt.reset(env.last_seen_pos.copy(), env.t_vel.copy())
        mu, cov = self.filt.estimate()
        feature = hybrid_feature(env.last_seen_pos, env.t_vel, True, mu, cov,
                                 self.filt.mu_m, 0, self.arena)
        self.window = None
        self._push(feature)
        self.last_estimate = None
        return self._inject_estimate(env, mu, cov, visible=True)

    @torch.no_grad()
    def _inject_estimate(self, env, imm_mu: np.ndarray, imm_cov: np.ndarray,
                         visible: bool) -> np.ndarray:
        residual_mean = np.zeros(2, dtype=np.float32)
        residual_var = np.zeros(2, dtype=np.float32)
        mean = np.asarray(imm_mu[:2], dtype=float).copy()
        covariance = np.asarray(imm_cov[:2, :2], dtype=float).copy()
        if not visible:
            delta, var = self.model(self.window)
            residual_mean = delta[0].cpu().numpy()
            residual_var = (var[0].cpu().numpy() * self.residual_temperature)
            mean = mean + residual_mean * self.arena
            covariance = covariance + np.diag(
                residual_var * (self.arena ** 2))
        mean = np.clip(mean, 0.0, self.arena)
        covariance = 0.5 * (covariance + covariance.T)
        if self.variant == "point":
            sigma_feat = np.zeros(3)
            conf = 1.0
        else:
            cov_norm = covariance / (self.arena ** 2)
            sigma_feat = np.array([
                max(cov_norm[0, 0], 0.0),
                max(cov_norm[1, 1], 0.0),
                0.0,
            ])
            conf = GRUGaussianPredictor.confidence(
                int(env.t_lost), float(sigma_feat[:2].sum()))
        self.last_estimate = HybridEstimate(
            mean=mean.copy(), covariance=covariance.copy(),
            imm_mean=np.asarray(imm_mu[:2], dtype=float).copy(),
            imm_covariance=np.asarray(imm_cov[:2, :2], dtype=float).copy(),
            residual_mean=residual_mean.copy(),
            residual_variance=residual_var.copy(),
        )
        env.set_belief(mean, sigma_feat, conf)
        return env._get_obs()

    def inject(self, env) -> np.ndarray:
        visible = bool(env.visible_each.any())
        measured_pos = getattr(
            env, "target_measurement_pos", env.t_pos)
        if visible:
            mu, cov, probs, _ = self.filt.update(
                measured_pos.copy(), env.cfg.dt)
        else:
            mu, cov, probs = self.filt.predict(env.cfg.dt)
        noise_std = float(getattr(
            env.cfg, "target_position_noise_std_m", 0.0))
        feature_position = (
            env.t_pos
            if noise_std == 0.0
            else (measured_pos if visible else env.last_seen_pos)
        )
        feature = hybrid_feature(
            feature_position, env.t_vel, visible, mu, cov, probs,
                                 int(env.t_lost), self.arena)
        self._push(feature)
        return self._inject_estimate(env, mu, cov, visible)
