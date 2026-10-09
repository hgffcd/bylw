# -*- coding: utf-8 -*-
"""匀速外推器(论文A S-A2, 纯 numpy) —— 最弱估计基线(比 last 强一档).

最后可见位置 + 有限差分速度做线性外推; 无量测更新增益, 可见步直接吸附观测,
失观步位置匀速外推、协方差按固定增长率 q_growth [m^2/s] 线性膨胀.
状态 [x, y, vx, vy] 绝对米制; 接口与 imm_filter.IMMFilter 同构:
    reset(z_pos=None, vel=None)
    update(z_pos, dt) -> (mu, Sigma, model_probs, likelihood)   # model_probs 恒 [1.0]
    predict(dt)       -> (mu, Sigma, model_probs)
likelihood 按外推先验 N(z; x+v*dt, P_pos+R) 计算(仅供接口兼容/诊断).
"""
from __future__ import annotations

import numpy as np

from belief.imm_filter import _LIK_FLOOR, _gauss2_pdf


class CVExtrapolator:
    def __init__(self, q_growth: float = 4.0, r: float = 0.25,
                 p0_pos: float = 0.25, p0_vel: float = 4.0, alpha: float = 1.0):
        """q_growth: 失观期位置方差固定增长率 [m^2/s]; alpha: 速度差分平滑系数(1=纯差分)."""
        self.q_growth = float(q_growth)
        self.r = float(r)
        self.p0_pos, self.p0_vel = float(p0_pos), float(p0_vel)
        self.alpha = float(alpha)
        self.reset()

    def _base_P(self) -> np.ndarray:
        return np.diag([self.p0_pos, self.p0_pos, self.p0_vel, self.p0_vel])

    def reset(self, z_pos=None, vel=None):
        pos = np.zeros(2) if z_pos is None else np.asarray(z_pos, float)
        v = np.zeros(2) if vel is None else np.asarray(vel, float)
        self.x = np.concatenate([pos, v])
        self.P = self._base_P()

    def update(self, z_pos, dt: float):
        """可见步: 位置吸附观测, 速度=有限差分(可平滑), 协方差回落基准值."""
        z = np.asarray(z_pos, float)
        pred_pos = self.x[:2] + self.x[2:] * dt
        innov = z - pred_pos
        S = self.P[:2, :2] + self.r * np.eye(2)
        lik = max(_gauss2_pdf(innov, S), _LIK_FLOOR)
        v_fd = (z - self.x[:2]) / max(dt, 1e-9)
        self.x[2:] = (1.0 - self.alpha) * self.x[2:] + self.alpha * v_fd
        self.x[:2] = z
        self.P = self._base_P()
        return self.x.copy(), self.P.copy(), np.ones(1), float(lik)

    def predict(self, dt: float):
        """失观步: 位置匀速外推, 位置方差按固定增长率线性膨胀."""
        self.x[:2] = self.x[:2] + self.x[2:] * dt
        self.P[0, 0] += self.q_growth * dt
        self.P[1, 1] += self.q_growth * dt
        return self.x.copy(), self.P.copy(), np.ones(1)
