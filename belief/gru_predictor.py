# -*- coding: utf-8 -*-
"""GRU 概率目标预测器(第3章核心, 对应开题 5.2 节, 式5-8 ~ 5-12).

- 输入: 目标观测序列 [rel_or_abs_pos(2), vel(2), visible_flag(1)] (失观步用上一步预测均值构成伪观测);
- 输出: 未来 K 步位置的高斯 (mu, diag Sigma), NLL 损失训练(式5-10);
- 失观递推: predict_rollout() 自回归 + 协方差随失观时长膨胀(式5-11);
- 置信度: confidence() = exp(-l1*t_lost - l2*tr(Sigma)) (式5-12);
- 校准评估: picp() 区间覆盖率(式5-PICP).

依赖 torch —— 在本机 rl_env 运行; 数据由 scripts/collect_trajectories.py 生成.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GRUGaussianPredictor(nn.Module):
    def __init__(
        self,
        in_dim=5,
        hidden=64,
        horizon=5,
        rollout_eta=0.05,
        rollout_q0=0.04,
    ):
        super().__init__()
        self.horizon = horizon
        self.rollout_eta = float(rollout_eta)
        self.rollout_q0 = float(rollout_q0)
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head_mu = nn.Linear(hidden, 2 * horizon)
        self.head_logvar = nn.Linear(hidden, 2 * horizon)

    def forward(self, seq, h0=None):
        """seq: (B, T, in_dim) -> mu, var: (B, K, 2); h: 末隐状态."""
        out, h = self.gru(seq, h0)                      # 式(5-8)
        last = out[:, -1]
        mu = self.head_mu(last).view(-1, self.horizon, 2)          # 式(5-9)
        logvar = self.head_logvar(last).view(-1, self.horizon, 2).clamp(-8, 6)
        return mu, logvar.exp(), h

    @staticmethod
    def nll_loss(mu, var, target):
        """高斯负对数似然(对角协方差, 式5-10). target: (B, K, 2)."""
        return 0.5 * (((target - mu) ** 2) / var + var.log()).sum(-1).mean()

    @torch.no_grad()
    def rollout_lost(self, seq, steps, eta=None, q0=None):
        """失观期间自回归递推: 每步把预测均值当伪观测续接, 协方差按
        Sigma_{t+1} = Sigma_t + Q*(1+eta*t_lost) 膨胀(式5-11, 对角近似)."""
        eta = self.rollout_eta if eta is None else float(eta)
        q0 = self.rollout_q0 if q0 is None else float(q0)
        mus, vars_ = [], []
        cur = seq.clone()
        var_acc = torch.zeros(
            cur.shape[0], 2, device=cur.device, dtype=cur.dtype)
        for t in range(steps):
            mu, var, _ = self.forward(cur)
            step_mu = mu[:, 0]                      # 取一步预测
            var_acc = var_acc + var[:, 0] + q0 * (1 + eta * (t + 1))
            mus.append(step_mu)
            vars_.append(var_acc.clone())
            pseudo = torch.cat([step_mu, torch.zeros_like(step_mu),
                                -torch.ones(step_mu.shape[0], 1)], dim=-1)
            cur = torch.cat([cur[:, 1:], pseudo.unsqueeze(1)], dim=1)
        return torch.stack(mus, 1), torch.stack(vars_, 1)

    @staticmethod
    def confidence(t_lost: float, sigma_trace: float,
                   l1: float = 0.02, l2: float = 0.05) -> float:
        """标量置信度 c_t (式5-12)."""
        return float(math.exp(-l1 * t_lost - l2 * sigma_trace))

    @staticmethod
    def picp(mu, var, target, z: float = 1.96) -> float:
        """区间覆盖率: 目标落入 mu ± z*sigma (逐维) 的比例, 校准指标."""
        sigma = var.sqrt()
        inside = ((target - mu).abs() <= z * sigma).all(-1).float()
        return float(inside.mean())
