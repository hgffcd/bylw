# -*- coding: utf-8 -*-
"""Versioned predictive safety layer for moving-obstacle experiments.

This module intentionally does not modify ``safety.cbf_qp.CBFSafetyLayer``.
The legacy implementation remains the default reproduction path for E1--E5,
J1, and J2a.  ``DynamicCBFSafetyLayer`` is enabled only when a configuration
or evaluation command explicitly selects ``dynamic_v2``.

The layer uses the same semi-implicit unicycle update as the environment,
predicts moving circular obstacles (including boundary reflection), solves a
joint linearized CBF-QP for all pursuers, verifies the candidate with the
nonlinear rollout, and uses a deterministic emergency search when the QP is
infeasible or the nonlinear verification fails.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import osqp
    from scipy import sparse
except ImportError:  # pragma: no cover - exercised by the explicit fallback
    osqp = None
    sparse = None


@dataclass(frozen=True)
class DynamicSafetyConfig:
    horizon_s: float = 1.5
    safety_buffer: float = 0.5
    activation_distance: float = 8.0
    finite_difference: float = 1.0e-3
    qp_eps_abs: float = 1.0e-6
    qp_eps_rel: float = 1.0e-6
    qp_max_iter: int = 4000
    scp_iterations: int = 3
    verification_tolerance: float = 1.0e-6
    emergency_ranking: str = "horizon"


class DynamicCBFSafetyLayer:
    """Fixed-parameter predictive CBF layer for a complete environment state."""

    version = "dynamic_v2"
    revision = "2.2"

    def __init__(
        self,
        *,
        dt: float,
        a_max: float,
        w_max: float,
        v_max: float,
        pursuer_radius: float,
        horizon_s: float = 1.5,
        safety_buffer: float = 0.5,
        activation_distance: float = 8.0,
        finite_difference: float = 1.0e-3,
        qp_eps_abs: float = 1.0e-6,
        qp_eps_rel: float = 1.0e-6,
        qp_max_iter: int = 4000,
        scp_iterations: int = 3,
        verification_tolerance: float = 1.0e-6,
        emergency_ranking: str = "horizon",
    ):
        self.dt = float(dt)
        self.a_max = float(a_max)
        self.w_max = float(w_max)
        self.v_max = float(v_max)
        self.pursuer_radius = float(pursuer_radius)
        self.cfg = DynamicSafetyConfig(
            horizon_s=float(horizon_s),
            safety_buffer=float(safety_buffer),
            activation_distance=float(activation_distance),
            finite_difference=float(finite_difference),
            qp_eps_abs=float(qp_eps_abs),
            qp_eps_rel=float(qp_eps_rel),
            qp_max_iter=int(qp_max_iter),
            scp_iterations=int(scp_iterations),
            verification_tolerance=float(verification_tolerance),
            emergency_ranking=str(emergency_ranking),
        )
        if self.cfg.horizon_s < self.dt:
            raise ValueError("horizon_s must be at least one environment step")
        if self.cfg.safety_buffer < 0:
            raise ValueError("safety_buffer must be non-negative")
        if self.cfg.emergency_ranking not in {"horizon", "safe_prefix"}:
            raise ValueError(
                "emergency_ranking must be 'horizon' or 'safe_prefix'")
        self.revision = (
            "2.2" if self.cfg.emergency_ranking == "horizon" else "2.5"
        )
        self.prediction_steps = max(
            1, int(math.ceil(self.cfg.horizon_s / self.dt)))
        self._fallback_cache: dict[int, np.ndarray] = {}
        self.reset_statistics()

    @classmethod
    def from_env_config(
        cls, env_cfg: Any, safety_params: dict[str, Any] | None = None,
    ) -> "DynamicCBFSafetyLayer":
        return cls(
            dt=env_cfg.dt,
            a_max=env_cfg.a_max,
            w_max=env_cfg.w_max,
            v_max=env_cfg.v_max,
            pursuer_radius=env_cfg.pursuer_radius,
            **dict(safety_params or {}),
        )

    def reset_statistics(self) -> None:
        self.call_count = 0
        self.correction_count = 0
        self.nominal_unsafe_count = 0
        self.qp_attempt_count = 0
        self.qp_failure_count = 0
        self.fallback_count = 0
        self.verified_safe_count = 0
        self.output_unsafe_count = 0
        self.output_horizon_physical_unsafe_count = 0
        self.output_next_step_physical_unsafe_count = 0
        self.total_correction_norm = 0.0
        self.max_correction_norm = 0.0
        self.minimum_solver_residual = math.inf
        self.minimum_rejected_solver_residual = math.inf
        self.minimum_predicted_clearance = math.inf
        self.minimum_horizon_physical_clearance = math.inf
        self.minimum_next_step_physical_clearance = math.inf
        self.last_diagnostics: dict[str, Any] = {}

    def summary(self) -> dict[str, float | int | str]:
        calls = max(self.call_count, 1)
        corrections = max(self.correction_count, 1)
        return {
            "safety_version": self.version,
            "safety_revision": self.revision,
            "calls": self.call_count,
            "corrections": self.correction_count,
            "correction_rate": self.correction_count / calls,
            "nominal_unsafe": self.nominal_unsafe_count,
            "nominal_unsafe_rate": self.nominal_unsafe_count / calls,
            "qp_attempts": self.qp_attempt_count,
            "qp_failures": self.qp_failure_count,
            "fallbacks": self.fallback_count,
            "fallback_rate": self.fallback_count / calls,
            "verified_safe": self.verified_safe_count,
            "output_unsafe": self.output_unsafe_count,
            "output_horizon_physical_unsafe": (
                self.output_horizon_physical_unsafe_count
            ),
            "output_next_step_physical_unsafe": (
                self.output_next_step_physical_unsafe_count
            ),
            "mean_correction_norm": self.total_correction_norm / corrections,
            "max_correction_norm": self.max_correction_norm,
            "minimum_solver_residual": (
                self.minimum_solver_residual
                if math.isfinite(self.minimum_solver_residual)
                else float("nan")
            ),
            "minimum_rejected_solver_residual": (
                self.minimum_rejected_solver_residual
                if math.isfinite(self.minimum_rejected_solver_residual)
                else float("nan")
            ),
            "minimum_predicted_clearance": (
                self.minimum_predicted_clearance
                if math.isfinite(self.minimum_predicted_clearance)
                else float("nan")
            ),
            "minimum_horizon_physical_clearance": (
                self.minimum_horizon_physical_clearance
                if math.isfinite(self.minimum_horizon_physical_clearance)
                else float("nan")
            ),
            "minimum_next_step_physical_clearance": (
                self.minimum_next_step_physical_clearance
                if math.isfinite(self.minimum_next_step_physical_clearance)
                else float("nan")
            ),
        }

    @staticmethod
    def _wrap_angle(angle: np.ndarray) -> np.ndarray:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _predict_agents(
        self, env: Any, actions: np.ndarray,
    ) -> np.ndarray:
        actions = np.asarray(actions, dtype=float).reshape(env.n, 2)
        pos = np.asarray(env.p_pos, dtype=float).copy()
        psi = np.asarray(env.p_psi, dtype=float).copy()
        speed = np.asarray(env.p_v, dtype=float).copy()
        path = np.empty((self.prediction_steps, env.n, 2), dtype=float)
        acceleration = actions[:, 0] * self.a_max
        turn_rate = actions[:, 1] * self.w_max
        for k in range(self.prediction_steps):
            speed = np.clip(
                speed + acceleration * self.dt, 0.0, self.v_max)
            psi = self._wrap_angle(psi + turn_rate * self.dt)
            pos = pos + np.stack(
                [speed * np.cos(psi), speed * np.sin(psi)], axis=-1,
            ) * self.dt
            pos = np.clip(pos, 0.0, env.cfg.arena_size)
            path[k] = pos
        return path

    def _predict_obstacles(self, env: Any) -> np.ndarray:
        count = len(env.obs_pos)
        path = np.empty((self.prediction_steps, count, 2), dtype=float)
        if count == 0:
            return path
        pos = np.asarray(env.obs_pos, dtype=float).copy()
        velocity = np.asarray(env.obs_vel, dtype=float).copy()
        radius = np.asarray(env.obs_r, dtype=float)
        arena = float(env.cfg.arena_size)
        for k in range(self.prediction_steps):
            pos = pos + velocity * self.dt
            for j in range(count):
                for axis in range(2):
                    if (pos[j, axis] < radius[j] or
                            pos[j, axis] > arena - radius[j]):
                        velocity[j, axis] *= -1.0
                        pos[j, axis] = np.clip(
                            pos[j, axis], radius[j], arena - radius[j])
            path[k] = pos
        return path

    def _constraint_values(
        self,
        env: Any,
        actions: np.ndarray,
        obstacle_path: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return squared CBF values and signed geometric clearances."""
        agent_path = self._predict_agents(env, actions)
        values: list[float] = []
        clearances: list[float] = []
        d_agent = 2.0 * self.pursuer_radius + self.cfg.safety_buffer
        for k in range(self.prediction_steps):
            for i in range(env.n):
                for j in range(i + 1, env.n):
                    distance = float(
                        np.linalg.norm(agent_path[k, i] - agent_path[k, j]))
                    values.append(distance * distance - d_agent * d_agent)
                    clearances.append(distance - d_agent)
            for i in range(env.n):
                for j, radius in enumerate(env.obs_r):
                    d_obstacle = (
                        self.pursuer_radius
                        + float(radius)
                        + self.cfg.safety_buffer
                    )
                    distance = float(np.linalg.norm(
                        agent_path[k, i] - obstacle_path[k, j]))
                    values.append(
                        distance * distance - d_obstacle * d_obstacle)
                    clearances.append(distance - d_obstacle)
        return np.asarray(values), np.asarray(clearances)

    def _linearized_constraints(
        self,
        env: Any,
        center: np.ndarray,
        obstacle_path: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = np.asarray(center, dtype=float).reshape(-1)
        shape = (env.n, 2)
        h0, clearance = self._constraint_values(
            env, center.reshape(shape), obstacle_path)
        active = clearance <= self.cfg.activation_distance
        if not np.any(active):
            return np.empty((0, center.size)), np.empty(0), clearance
        h_active = h0[active]
        gradient = np.empty((h_active.size, center.size), dtype=float)
        epsilon = self.cfg.finite_difference
        for d in range(center.size):
            plus = center.copy()
            minus = center.copy()
            plus[d] = min(1.0, plus[d] + epsilon)
            minus[d] = max(-1.0, minus[d] - epsilon)
            denominator = plus[d] - minus[d]
            if denominator <= 0:
                gradient[:, d] = 0.0
                continue
            hp, _ = self._constraint_values(
                env, plus.reshape(shape), obstacle_path)
            hm, _ = self._constraint_values(
                env, minus.reshape(shape), obstacle_path)
            gradient[:, d] = (hp[active] - hm[active]) / denominator
        constant = h_active - gradient @ center
        return gradient, constant, clearance

    def _solve_qp(
        self,
        nominal: np.ndarray,
        gradient: np.ndarray,
        constant: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        self.qp_attempt_count += 1
        if osqp is None or sparse is None:
            self.qp_failure_count += 1
            return None, float("-inf"), "osqp_unavailable"
        dimension = nominal.size
        identity = sparse.eye(dimension, format="csc")
        matrix = sparse.vstack(
            [sparse.csc_matrix(gradient), identity], format="csc")
        lower = np.concatenate([-constant, -np.ones(dimension)])
        upper = np.concatenate([
            np.full(len(constant), np.inf), np.ones(dimension)])
        solver = osqp.OSQP()
        solver.setup(
            P=2.0 * identity,
            q=-2.0 * nominal,
            A=matrix,
            l=lower,
            u=upper,
            verbose=False,
            polishing=True,
            eps_abs=self.cfg.qp_eps_abs,
            eps_rel=self.cfg.qp_eps_rel,
            max_iter=self.cfg.qp_max_iter,
            adaptive_rho=False,
        )
        result = solver.solve(raise_error=False)
        status = str(result.info.status).lower()
        if result.x is None or "solved" not in status:
            self.qp_failure_count += 1
            return None, float("-inf"), status
        action = np.clip(np.asarray(result.x, dtype=float), -1.0, 1.0)
        residual = float(np.min(gradient @ action + constant))
        # Keep the runtime acceptance rule identical to the preregistered
        # solver-residual gate.  A looser internal tolerance can otherwise
        # mark a QP as accepted even though the formal audit must reject it.
        if residual < -self.cfg.qp_eps_abs:
            self.minimum_rejected_solver_residual = min(
                self.minimum_rejected_solver_residual, residual)
            self.qp_failure_count += 1
            return None, residual, "residual_failure"
        self.minimum_solver_residual = min(
            self.minimum_solver_residual, residual)
        return action, residual, status

    def _fallback_candidates(self, n_agents: int) -> np.ndarray:
        cached = self._fallback_cache.get(n_agents)
        if cached is not None:
            return cached
        single = [
            np.array([acceleration, steering], dtype=float)
            for acceleration in (-1.0, -0.5, 0.0, 0.5, 1.0)
            for steering in (-1.0, 0.0, 1.0)
        ]
        candidates = np.asarray(list(itertools.product(single, repeat=n_agents)))
        self._fallback_cache[n_agents] = candidates
        return candidates

    def _predict_agents_batch(
        self, env: Any, actions: np.ndarray,
    ) -> np.ndarray:
        """Vectorized constant-control rollouts for emergency candidates."""
        actions = np.asarray(actions, dtype=float)
        count = actions.shape[0]
        pos = np.broadcast_to(
            np.asarray(env.p_pos, dtype=float), (count, env.n, 2),
        ).copy()
        psi = np.broadcast_to(
            np.asarray(env.p_psi, dtype=float), (count, env.n),
        ).copy()
        speed = np.broadcast_to(
            np.asarray(env.p_v, dtype=float), (count, env.n),
        ).copy()
        path = np.empty(
            (self.prediction_steps, count, env.n, 2), dtype=float)
        acceleration = actions[:, :, 0] * self.a_max
        turn_rate = actions[:, :, 1] * self.w_max
        for k in range(self.prediction_steps):
            speed = np.clip(
                speed + acceleration * self.dt, 0.0, self.v_max)
            psi = self._wrap_angle(psi + turn_rate * self.dt)
            pos = pos + np.stack(
                [speed * np.cos(psi), speed * np.sin(psi)], axis=-1,
            ) * self.dt
            pos = np.clip(pos, 0.0, env.cfg.arena_size)
            path[k] = pos
        return path

    def _batch_clearance_metrics(
        self,
        env: Any,
        candidates: np.ndarray,
        obstacle_path: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return clearance metrics and safe physical prefix length."""
        path = self._predict_agents_batch(env, candidates)
        count = candidates.shape[0]
        horizon_buffer = np.full(count, np.inf, dtype=float)
        next_buffer = np.full(count, np.inf, dtype=float)
        next_physical = np.full(count, np.inf, dtype=float)
        physical_by_step = np.full(
            (self.prediction_steps, count), np.inf, dtype=float)
        d_agent_physical = 2.0 * self.pursuer_radius
        d_agent_buffer = d_agent_physical + self.cfg.safety_buffer
        for i in range(env.n):
            for j in range(i + 1, env.n):
                distance = np.linalg.norm(
                    path[:, :, i] - path[:, :, j], axis=-1)
                horizon_buffer = np.minimum(
                    horizon_buffer,
                    np.min(distance - d_agent_buffer, axis=0),
                )
                next_buffer = np.minimum(
                    next_buffer, distance[0] - d_agent_buffer)
                next_physical = np.minimum(
                    next_physical, distance[0] - d_agent_physical)
                physical_by_step = np.minimum(
                    physical_by_step, distance - d_agent_physical)
        for i in range(env.n):
            for j, radius in enumerate(env.obs_r):
                d_obstacle_physical = self.pursuer_radius + float(radius)
                d_obstacle_buffer = (
                    d_obstacle_physical + self.cfg.safety_buffer)
                distance = np.linalg.norm(
                    path[:, :, i] - obstacle_path[:, None, j], axis=-1)
                horizon_buffer = np.minimum(
                    horizon_buffer,
                    np.min(distance - d_obstacle_buffer, axis=0),
                )
                next_buffer = np.minimum(
                    next_buffer, distance[0] - d_obstacle_buffer)
                next_physical = np.minimum(
                    next_physical, distance[0] - d_obstacle_physical)
                physical_by_step = np.minimum(
                    physical_by_step, distance - d_obstacle_physical)
        unsafe = physical_by_step < -self.cfg.verification_tolerance
        any_unsafe = np.any(unsafe, axis=0)
        first_unsafe = np.argmax(unsafe, axis=0)
        safe_prefix_steps = np.where(
            any_unsafe, first_unsafe, self.prediction_steps)
        return (
            horizon_buffer,
            next_buffer,
            next_physical,
            safe_prefix_steps,
        )

    def _emergency_action(
        self,
        env: Any,
        nominal: np.ndarray,
        obstacle_path: np.ndarray,
        extra_candidate: np.ndarray | None,
    ) -> tuple[np.ndarray, float, str]:
        candidates = self._fallback_candidates(env.n)
        if extra_candidate is not None:
            candidates = np.concatenate([
                np.asarray(extra_candidate, dtype=float).reshape(1, env.n, 2),
                candidates,
            ], axis=0)
        (
            clearances,
            next_buffer,
            next_physical,
            safe_prefix_steps,
        ) = self._batch_clearance_metrics(env, candidates, obstacle_path)
        deviations = np.sum(
            (candidates - nominal[None, :, :]) ** 2, axis=(1, 2))
        if self.cfg.emergency_ranking == "horizon":
            best_clearance = float(np.max(clearances))
            tied = np.flatnonzero(
                clearances >= best_clearance - 1.0e-12)
            best_index = int(tied[np.argmin(deviations[tied])])
            return (
                candidates[best_index].copy(),
                best_clearance,
                "horizon",
            )
        tolerance = self.cfg.verification_tolerance
        pool = np.flatnonzero(next_buffer >= -tolerance)
        tier = "next_buffer_safe"
        if pool.size == 0:
            pool = np.flatnonzero(next_physical >= -tolerance)
            tier = "next_physical_safe"
        if pool.size == 0:
            # If no sampled command can avoid a next-step physical violation,
            # first maximize the clearance of the command that will actually
            # be executed.  Full-horizon viability is only a secondary goal.
            best_next = float(np.max(next_physical))
            pool = np.flatnonzero(
                next_physical >= best_next - 1.0e-12)
            tier = "maximize_next_physical"
        best_prefix = int(np.max(safe_prefix_steps[pool]))
        pool = pool[safe_prefix_steps[pool] == best_prefix]
        best_clearance = float(np.max(clearances[pool]))
        tied = pool[
            clearances[pool] >= best_clearance - 1.0e-12
        ]
        best_index = int(tied[np.argmin(deviations[tied])])
        return candidates[best_index].copy(), best_clearance, tier

    def _physical_clearance_metrics(
        self,
        env: Any,
        actions: np.ndarray,
        obstacle_path: np.ndarray,
    ) -> tuple[float, float]:
        """Return horizon and executed-next-step physical clearances.

        ``output_min_clearance`` includes the design buffer and tests a
        constant action across the complete predictive horizon.  In a
        receding-horizon controller only the first action is executed, so a
        later buffered-horizon violation is a viability diagnostic rather
        than an unsafe command.  These physical metrics keep that distinction
        explicit without discarding the stricter buffered diagnostic.
        """
        path = self._predict_agents(env, actions)
        horizon_min = math.inf
        next_step_min = math.inf
        d_agent = 2.0 * self.pursuer_radius
        for i in range(env.n):
            for j in range(i + 1, env.n):
                distance = np.linalg.norm(
                    path[:, i] - path[:, j], axis=-1) - d_agent
                horizon_min = min(horizon_min, float(np.min(distance)))
                next_step_min = min(next_step_min, float(distance[0]))
        for i in range(env.n):
            for j, radius in enumerate(env.obs_r):
                d_obstacle = self.pursuer_radius + float(radius)
                distance = np.linalg.norm(
                    path[:, i] - obstacle_path[:, j], axis=-1
                ) - d_obstacle
                horizon_min = min(horizon_min, float(np.min(distance)))
                next_step_min = min(next_step_min, float(distance[0]))
        return horizon_min, next_step_min

    def _record_projection(
        self,
        env: Any,
        obstacle_path: np.ndarray,
        actor_nominal: np.ndarray,
        output: np.ndarray,
        output_min: float,
        diagnostics: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        output = np.clip(np.asarray(output, dtype=float), -1.0, 1.0)
        correction_norm = float(np.linalg.norm(output - actor_nominal))
        corrected = correction_norm > 1.0e-10
        if corrected:
            self.correction_count += 1
            self.total_correction_norm += correction_norm
            self.max_correction_norm = max(
                self.max_correction_norm, correction_norm)
        if output_min >= -self.cfg.verification_tolerance:
            self.verified_safe_count += 1
        else:
            self.output_unsafe_count += 1
        horizon_physical, next_step_physical = (
            self._physical_clearance_metrics(env, output, obstacle_path)
        )
        if horizon_physical < -self.cfg.verification_tolerance:
            self.output_horizon_physical_unsafe_count += 1
        if next_step_physical < -self.cfg.verification_tolerance:
            self.output_next_step_physical_unsafe_count += 1
        self.minimum_predicted_clearance = min(
            self.minimum_predicted_clearance, output_min)
        self.minimum_horizon_physical_clearance = min(
            self.minimum_horizon_physical_clearance, horizon_physical)
        self.minimum_next_step_physical_clearance = min(
            self.minimum_next_step_physical_clearance, next_step_physical)
        diagnostics.update({
            "corrected": corrected,
            "correction_norm": correction_norm,
            "output_min_clearance": output_min,
            "output_horizon_physical_clearance": horizon_physical,
            "output_next_step_physical_clearance": next_step_physical,
        })
        self.last_diagnostics = diagnostics
        return output, diagnostics

    def project_joint(
        self,
        env: Any,
        nominal_actions: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Project all pursuer actions and return actions plus diagnostics."""
        self.call_count += 1
        nominal = np.clip(
            np.asarray(nominal_actions, dtype=float).reshape(env.n, 2),
            -1.0,
            1.0,
        )
        actor_nominal = nominal
        obstacle_path = self._predict_obstacles(env)
        _, nominal_clearances = self._constraint_values(
            env, nominal, obstacle_path)
        nominal_min = (
            float(np.min(nominal_clearances))
            if nominal_clearances.size
            else math.inf
        )
        diagnostics: dict[str, Any] = {
            "safety_version": self.version,
            "corrected": False,
            "nominal_min_clearance": nominal_min,
            "output_min_clearance": nominal_min,
            "solver_status": "not_needed",
            "solver_residual": float("nan"),
            "fallback": False,
        }
        if nominal_min >= -self.cfg.verification_tolerance:
            return self._record_projection(
                env, obstacle_path, actor_nominal, nominal, nominal_min,
                diagnostics)

        self.nominal_unsafe_count += 1
        nominal_flat = nominal.reshape(-1)
        center = nominal_flat.copy()
        candidate: np.ndarray | None = None
        residual = float("-inf")
        status = "not_solved"
        output_min = nominal_min
        for _ in range(self.cfg.scp_iterations):
            gradient, constant, _ = self._linearized_constraints(
                env, center, obstacle_path)
            if gradient.size == 0:
                break
            solved, residual, status = self._solve_qp(
                nominal_flat, gradient, constant)
            if solved is None:
                break
            candidate = solved.reshape(env.n, 2)
            _, output_clearances = self._constraint_values(
                env, candidate, obstacle_path)
            output_min = (
                float(np.min(output_clearances))
                if output_clearances.size
                else math.inf
            )
            if output_min >= -self.cfg.verification_tolerance:
                break
            center = solved

        use_fallback = (
            candidate is None
            or output_min < -self.cfg.verification_tolerance
        )
        if use_fallback:
            self.fallback_count += 1
            candidate, output_min, emergency_tier = self._emergency_action(
                env, nominal, obstacle_path, candidate)
            status = f"{status}|emergency"
        else:
            emergency_tier = "not_used"
        output = np.clip(candidate, -1.0, 1.0)
        diagnostics.update({
            "solver_status": status,
            "solver_residual": residual,
            "fallback": use_fallback,
            "emergency_tier": emergency_tier,
        })
        return self._record_projection(
            env, obstacle_path, actor_nominal, output, output_min, diagnostics)
