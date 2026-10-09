# -*- coding: utf-8 -*-
"""不确定度自适应 CBF 安全层(第5章核心, 对应开题 5.3.4 节, 式5-23 ~ 5-25).

实现说明:
- 采用离散时间控制障碍函数条件 h(x_{k+1}) >= (1-alpha) h(x_k);
- 对 unicycle 以一步欧拉近似把条件线性化为对动作 (a, w) 的约束;
- 二次规划: min ||u - u_nom||^2, 带盒约束与松弛变量(大惩罚);
- 求解: 约束数少, 用投影梯度 + 主动集近似(纯 numpy, 无外部 QP 依赖);
  正式实验建议切换 osqp/qpsolvers(接口已留, solve(..., backend="osqp"));
- 自适应安全距离 d_s = d0 + k1*sqrt(lambda_max(Sigma)) + k2*v_rel*tau (式5-25).

基线训练可先不启用安全层(--safety off), 第5章实验再开启.
"""
from __future__ import annotations

import numpy as np


class CBFSafetyLayer:
    def __init__(self, d0_agent=2.5, d0_obs=1.0, alpha=0.3,
                 k1=1.0, k2=0.3, tau=0.5, slack_penalty=1e3,
                 a_max=4.0, w_max=1.5, dt=0.1):
        self.d0_agent, self.d0_obs = d0_agent, d0_obs
        self.alpha, self.k1, self.k2, self.tau = alpha, k1, k2, tau
        self.slack_penalty = slack_penalty
        self.a_max, self.w_max, self.dt = a_max, w_max, dt
        self.correction_count = 0
        self.call_count = 0
        self.nominal_unsafe_count = 0
        self.output_unsafe_count = 0
        self.total_correction_norm = 0.0
        self.max_correction_norm = 0.0
        self.minimum_solver_residual = np.inf
        self.last_diagnostics = {}

    # ---------- 自适应裕度 (式5-25) ----------
    def adaptive_margin(self, d0, sigma_max=0.0, v_rel=0.0):
        return d0 + self.k1 * np.sqrt(max(sigma_max, 0.0)) + self.k2 * max(v_rel, 0.0) * self.tau

    # ---------- 单机安全投影 ----------
    def project(self, u_nom, pos, psi, v, neighbors, obstacles,
                sigma_max=0.0):
        """u_nom: (2,) 归一化动作 [a_hat, w_hat];
        neighbors: list[(pos(2,), vel(2,))]; obstacles: list[(pos, radius, vel)].
        返回安全动作(归一化)与是否发生修正."""
        self.call_count += 1
        dt = self.dt
        cons = []  # 每条约束: (grad_u(2,), b) 表示 grad_u @ u + b >= 0

        def add_pair(rel_p, rel_v, d_s):
            """h = ||rel_p||^2 - d_s^2; 离散条件 h_next >= (1-alpha) h.
            rel_p 为对方指向本机的向量的相反数? 统一取 rel_p = p_other - p_self."""
            h = float(rel_p @ rel_p) - d_s ** 2
            # p_self 一步近似: p += [v cos psi, v sin psi]*dt + 0.5*[a cos, a sin]*dt^2 (忽略w对位置的一步影响)
            e = np.array([np.cos(psi), np.sin(psi)])
            # h_next ~ h + 2 rel_p @ (rel_v*dt - e*v*dt - e*a*dt^2*0.5*a_max)
            #   相对速度 rel_v = v_other - v_self(向量), 本机速度 v*e
            base = h + 2.0 * float(rel_p @ (rel_v * dt - e * v * dt))
            ga = -float(rel_p @ e) * dt * dt * self.a_max   # 对 a_hat 的梯度
            gw_vec = np.array([-np.sin(psi), np.cos(psi)]) * v * dt * dt * self.w_max
            gw = -float(rel_p @ gw_vec)
            rhs = (1 - self.alpha) * h
            cons.append((np.array([ga, gw]), base - rhs))

        for (p_o, v_o) in neighbors:
            rel_p = p_o - pos
            v_rel = float(np.linalg.norm(v_o - v * np.array([np.cos(psi), np.sin(psi)])))
            d_s = self.adaptive_margin(self.d0_agent, sigma_max, v_rel)
            add_pair(rel_p, v_o - v * np.array([np.cos(psi), np.sin(psi)]), d_s)
        for (p_o, r_o, v_o) in obstacles:
            rel_p = p_o - pos
            v_rel = float(np.linalg.norm(v_o - v * np.array([np.cos(psi), np.sin(psi)])))
            d_s = self.adaptive_margin(self.d0_obs + r_o, sigma_max, v_rel)
            add_pair(rel_p, v_o - v * np.array([np.cos(psi), np.sin(psi)]), d_s)

        if not cons:
            u = np.clip(u_nom, -1, 1)
            self.last_diagnostics = {
                "corrected": False,
                "nominal_min_residual": np.inf,
                "output_min_residual": np.inf,
            }
            return u, False
        u = np.clip(np.asarray(u_nom, float), -1, 1).copy()
        nominal_min_residual = min(float(g @ u + b) for g, b in cons)
        violated = nominal_min_residual < 0
        if not violated:
            self.minimum_solver_residual = min(
                self.minimum_solver_residual, nominal_min_residual)
            self.last_diagnostics = {
                "corrected": False,
                "nominal_min_residual": nominal_min_residual,
                "output_min_residual": nominal_min_residual,
            }
            return u, False
        self.nominal_unsafe_count += 1
        # 投影梯度求近似最小干预解
        u0 = u.copy()
        for _ in range(60):
            grad = 2 * (u - u0)
            for g, b in cons:
                margin = g @ u + b
                if margin < 0:
                    grad += 2 * self.slack_penalty * margin * g
            u = np.clip(u - 0.05 * grad, -1, 1)
        self.correction_count += 1
        correction_norm = float(np.linalg.norm(u - u0))
        self.total_correction_norm += correction_norm
        self.max_correction_norm = max(
            self.max_correction_norm, correction_norm)
        output_min_residual = min(float(g @ u + b) for g, b in cons)
        self.minimum_solver_residual = min(
            self.minimum_solver_residual, output_min_residual)
        if output_min_residual < -1.0e-9:
            self.output_unsafe_count += 1
        self.last_diagnostics = {
            "corrected": True,
            "correction_norm": correction_norm,
            "nominal_min_residual": nominal_min_residual,
            "output_min_residual": output_min_residual,
        }
        return u, True

    @property
    def correction_rate(self):
        return self.correction_count / max(self.call_count, 1)

    def summary(self):
        corrections = max(self.correction_count, 1)
        return {
            "safety_version": "legacy",
            "calls": self.call_count,
            "corrections": self.correction_count,
            "correction_rate": self.correction_rate,
            "nominal_unsafe": self.nominal_unsafe_count,
            "nominal_unsafe_rate": (
                self.nominal_unsafe_count / max(self.call_count, 1)
            ),
            "qp_attempts": self.nominal_unsafe_count,
            "qp_failures": self.output_unsafe_count,
            "fallbacks": 0,
            "fallback_rate": 0.0,
            "verified_safe": self.call_count - self.output_unsafe_count,
            "output_unsafe": self.output_unsafe_count,
            "mean_correction_norm": (
                self.total_correction_norm / corrections
            ),
            "max_correction_norm": self.max_correction_norm,
            "minimum_solver_residual": (
                self.minimum_solver_residual
                if np.isfinite(self.minimum_solver_residual)
                else float("nan")
            ),
            "minimum_predicted_clearance": float("nan"),
        }
