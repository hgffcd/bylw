# -*- coding: utf-8 -*-
"""单模型 CV 卡尔曼滤波器(论文A S-A2, 纯 numpy) —— IMM 的消融基线.

状态 [x, y, vx, vy] 绝对米制; 观测 z = 目标绝对位置 [m].
接口与 imm_filter.IMMFilter 同构:
    reset(z_pos=None, vel=None)
    update(z_pos, dt) -> (mu, Sigma, model_probs, likelihood)   # model_probs 恒 [1.0]
    predict(dt)       -> (mu, Sigma, model_probs)               # 失观步纯传播
"""
from __future__ import annotations

import numpy as np

from belief.imm_filter import _H, _LIK_FLOOR, _gauss2_pdf, cv_ct_transition, white_accel_Q


class KFSingleCV:
    def __init__(self, q: float = 2.0, r: float = 0.25,
                 p0_pos: float = 1.0, p0_vel: float = 25.0):
        self.q = float(q)
        self.R = float(r) * np.eye(2)
        self.p0_pos, self.p0_vel = float(p0_pos), float(p0_vel)
        self.reset()

    def reset(self, z_pos=None, vel=None):
        pos = np.zeros(2) if z_pos is None else np.asarray(z_pos, float)
        v = np.zeros(2) if vel is None else np.asarray(vel, float)
        self.x = np.concatenate([pos, v])
        self.P = np.diag([self.p0_pos, self.p0_pos, self.p0_vel, self.p0_vel])

    def update(self, z_pos, dt: float):
        """可见步: CV 预测 + 位置量测更新."""
        z = np.asarray(z_pos, float)
        F = cv_ct_transition(0.0, dt)
        xp = F @ self.x
        Pp = F @ self.P @ F.T + white_accel_Q(self.q, dt)
        innov = z - _H @ xp
        S = _H @ Pp @ _H.T + self.R
        K = Pp @ _H.T @ np.linalg.inv(S)
        self.x = xp + K @ innov
        P = (np.eye(4) - K @ _H) @ Pp
        self.P = 0.5 * (P + P.T)
        lik = max(_gauss2_pdf(innov, S), _LIK_FLOOR)
        return self.x.copy(), self.P.copy(), np.ones(1), float(lik)

    def predict(self, dt: float):
        """失观步纯传播: 协方差自然增长."""
        F = cv_ct_transition(0.0, dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + white_accel_Q(self.q, dt)
        return self.x.copy(), self.P.copy(), np.ones(1)
