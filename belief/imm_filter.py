# -*- coding: utf-8 -*-
"""IMM 交互多模型滤波器(论文A S-A1, 纯 numpy 无 torch 依赖).

- 模型集: CV + CT(已知转弯率)族, 由 turn_rates 配置(转弯率 0 即 CV, 默认 (0, ±0.3) rad/s);
- 状态: [x, y, vx, vy], 绝对米制坐标(与 env.t_pos 同口径); CT 转弯率为模型常数, 不入状态;
- 观测: z = 目标绝对位置 [m](可见步为 env 真值), R = r*I2;
- 过程噪声: 白噪声加速度模型 Q(q, dt)(离散化), CV/CT 共用同一强度 q(可分开配);
- 接口(与 kf_single.KFSingleCV / cv_extrap.CVExtrapolator 同构):
    reset(z_pos=None, vel=None)                      初始化各模型状态与均匀模式概率
    update(z_pos, dt) -> (mu, Sigma, model_probs, likelihood)   可见步完整 IMM 周期
    predict(dt)       -> (mu, Sigma, model_probs)               失观步纯传播(无混合无更新)
  mu 为组合后状态均值 (4,), Sigma 为组合协方差 (4,4)(含模型间均值散布项),
  model_probs 与 turn_rates 对位, likelihood = sum_j c_j * L_j(总观测似然).
- 失观期 predict: 各模型独立传播(P_j += Q 自然增长), 模式概率仅按马尔可夫链
  时间更新 mu <- Pi^T mu(不做状态混合), 保证 tr(Sigma) 单调增且实现简单.

注入 set_belief 的归一化换算(米制方差 / arena^2)由 train_belief.FilterBeliefManager
负责, 本文件保持纯米制口径.
"""
from __future__ import annotations

import math

import numpy as np

_H = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])
_LIK_FLOOR = 1e-300


def cv_ct_transition(omega: float, dt: float) -> np.ndarray:
    """CV/CT 状态转移矩阵 F(omega, dt); omega≈0 时退化为 CV."""
    if abs(omega) < 1e-9:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        return F
    wd = omega * dt
    s, c = math.sin(wd), math.cos(wd)
    return np.array([
        [1.0, 0.0, s / omega, -(1.0 - c) / omega],
        [0.0, 1.0, (1.0 - c) / omega, s / omega],
        [0.0, 0.0, c, -s],
        [0.0, 0.0, s, c],
    ])


def white_accel_Q(q: float, dt: float) -> np.ndarray:
    """白噪声加速度过程噪声(强度 q [m^2/s^3])的离散化."""
    d3, d2 = dt ** 3 / 3.0, dt ** 2 / 2.0
    return q * np.array([
        [d3, 0.0, d2, 0.0],
        [0.0, d3, 0.0, d2],
        [d2, 0.0, dt, 0.0],
        [0.0, d2, 0.0, dt],
    ])


def _gauss2_pdf(innov: np.ndarray, S: np.ndarray) -> float:
    """2 维高斯 N(innov; 0, S) 密度(带下限防下溢)."""
    det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
    det = max(det, 1e-12)
    Sinv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
    m2 = float(innov @ Sinv @ innov)
    return max(math.exp(-0.5 * m2) / (2.0 * math.pi * math.sqrt(det)), _LIK_FLOOR)


class IMMFilter:
    """CV+CT 交互多模型滤波器(转弯率集合可配)."""

    def __init__(self, turn_rates=(0.0, -0.3, 0.3), q: float = 2.0,
                 q_ct: float | None = None, r: float = 0.25,
                 p_stay: float = 0.95, p0_pos: float = 1.0, p0_vel: float = 25.0):
        self.omegas = tuple(float(w) for w in turn_rates)
        self.M = len(self.omegas)
        assert self.M >= 2, "IMM 至少需要 2 个模型(CV+CT)"
        self.q = float(q)
        self.q_ct = float(q_ct) if q_ct is not None else float(q)
        self.R = float(r) * np.eye(2)
        self.p0_pos, self.p0_vel = float(p0_pos), float(p0_vel)
        # 马尔可夫模式转移矩阵: 对角 p_stay, 其余均分
        off = (1.0 - float(p_stay)) / (self.M - 1)
        self.Pi = np.full((self.M, self.M), off)
        np.fill_diagonal(self.Pi, float(p_stay))
        self.reset()

    # ---------------- 接口 ----------------
    def reset(self, z_pos=None, vel=None):
        pos = np.zeros(2) if z_pos is None else np.asarray(z_pos, float)
        v = np.zeros(2) if vel is None else np.asarray(vel, float)
        x0 = np.concatenate([pos, v])
        self.x = np.tile(x0, (self.M, 1))                       # (M, 4)
        P0 = np.diag([self.p0_pos, self.p0_pos, self.p0_vel, self.p0_vel])
        self.P = np.tile(P0, (self.M, 1, 1))                    # (M, 4, 4)
        self.mu_m = np.full(self.M, 1.0 / self.M)               # 模式概率

    def update(self, z_pos, dt: float):
        """可见步: 混合 -> 各模型预测+KF更新 -> 模式概率更新 -> 组合输出."""
        z = np.asarray(z_pos, float)
        # 1) 交互混合
        c = self.Pi.T @ self.mu_m                               # (M,) 先验模式概率
        c = np.maximum(c, 1e-12)
        w = (self.Pi * self.mu_m[:, None]) / c[None, :]         # w[i,j] = P(i|j)
        x_mix = np.einsum("ij,ik->jk", w, self.x)               # (M, 4)
        P_mix = np.empty_like(self.P)
        for j in range(self.M):
            Pj = np.zeros((4, 4))
            for i in range(self.M):
                d = self.x[i] - x_mix[j]
                Pj += w[i, j] * (self.P[i] + np.outer(d, d))
            P_mix[j] = Pj
        # 2) 各模型预测 + 量测更新
        lik = np.empty(self.M)
        for j, om in enumerate(self.omegas):
            F = cv_ct_transition(om, dt)
            Q = white_accel_Q(self.q if abs(om) < 1e-9 else self.q_ct, dt)
            xp = F @ x_mix[j]
            Pp = F @ P_mix[j] @ F.T + Q
            innov = z - _H @ xp
            S = _H @ Pp @ _H.T + self.R
            K = Pp @ _H.T @ np.linalg.inv(S)
            self.x[j] = xp + K @ innov
            Pj = (np.eye(4) - K @ _H) @ Pp
            self.P[j] = 0.5 * (Pj + Pj.T)
            lik[j] = _gauss2_pdf(innov, S)
        # 3) 模式概率更新
        post = c * lik
        likelihood = float(post.sum())
        self.mu_m = post / max(likelihood, _LIK_FLOOR)
        # 4) 组合
        mu, Sigma = self._combine()
        return mu, Sigma, self.mu_m.copy(), likelihood

    def predict(self, dt: float):
        """失观步纯传播: 各模型独立递推(协方差自然增长), 模式概率仅时间更新."""
        for j, om in enumerate(self.omegas):
            F = cv_ct_transition(om, dt)
            Q = white_accel_Q(self.q if abs(om) < 1e-9 else self.q_ct, dt)
            self.x[j] = F @ self.x[j]
            self.P[j] = F @ self.P[j] @ F.T + Q
        self.mu_m = self.Pi.T @ self.mu_m
        self.mu_m /= max(self.mu_m.sum(), 1e-12)
        mu, Sigma = self._combine()
        return mu, Sigma, self.mu_m.copy()

    def estimate(self):
        """返回当前组合估计，不推进时间；供复合信念模块初始化使用。"""
        mu, Sigma = self._combine()
        return mu, Sigma

    # ---------------- 内部 ----------------
    def _combine(self):
        mu = self.mu_m @ self.x                                 # (4,)
        Sigma = np.zeros((4, 4))
        for j in range(self.M):
            d = self.x[j] - mu
            Sigma += self.mu_m[j] * (self.P[j] + np.outer(d, d))
        return mu, Sigma
