"""Frozen 2-D method bridge for P5 Isaac evaluation.

The bridge preserves the evaluated order

``actor -> tracker/scout + handoff -> dynamic_v2 -> simulator action``

and translates only the coordinate origin: Isaac environments use centered
local coordinates while the frozen 2-D method uses ``[0, arena_size]``.
It contains no Isaac imports, so the method composition can be tested on CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import safety.cbf_dynamic_v2 as dynamic_safety_module
from safety.cbf_dynamic_v2 import DynamicCBFSafetyLayer
from utils.reacquisition import ReacquisitionController

from .checkpoint import inspect_actor_checkpoint
from .config import P5TaskSpec


@dataclass(frozen=True)
class MethodBatchState:
    pursuer_xy_centered_m: np.ndarray
    pursuer_yaw_rad: np.ndarray
    pursuer_speed_mps: np.ndarray
    pursuer_velocity_xy_mps: np.ndarray
    last_seen_target_xy_legacy_m: np.ndarray
    visible_each: np.ndarray
    lost_steps: np.ndarray
    obstacle_xy_centered_m: np.ndarray
    obstacle_radius_m: np.ndarray
    obstacle_velocity_mps: np.ndarray

    def validate(self, num_envs: int, n_pursuers: int) -> None:
        expected = {
            "pursuer_xy_centered_m": (num_envs, n_pursuers, 2),
            "pursuer_yaw_rad": (num_envs, n_pursuers),
            "pursuer_speed_mps": (num_envs, n_pursuers),
            "pursuer_velocity_xy_mps": (num_envs, n_pursuers, 2),
            "last_seen_target_xy_legacy_m": (num_envs, 2),
            "visible_each": (num_envs, n_pursuers),
            "lost_steps": (num_envs,),
        }
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got "
                    f"{np.asarray(getattr(self, name)).shape}"
                )
        obstacle_shape = np.asarray(self.obstacle_xy_centered_m).shape
        if len(obstacle_shape) != 3 or obstacle_shape[0] != num_envs or obstacle_shape[2] != 2:
            raise ValueError("obstacle_xy_centered_m must have shape [env, obstacle, 2]")
        if np.asarray(self.obstacle_radius_m).shape != obstacle_shape[:2]:
            raise ValueError("obstacle_radius_m shape mismatch")
        if np.asarray(self.obstacle_velocity_mps).shape != obstacle_shape:
            raise ValueError("obstacle_velocity_mps shape mismatch")
        numeric = (
            self.pursuer_xy_centered_m,
            self.pursuer_yaw_rad,
            self.pursuer_speed_mps,
            self.pursuer_velocity_xy_mps,
            self.last_seen_target_xy_legacy_m,
            self.obstacle_xy_centered_m,
            self.obstacle_radius_m,
            self.obstacle_velocity_mps,
        )
        if not all(np.isfinite(np.asarray(value)).all() for value in numeric):
            raise ValueError("method state contains NaN or Inf")


@dataclass
class MethodActionResult:
    actor_actions: np.ndarray
    guided_actions: np.ndarray
    executed_actions: np.ndarray
    reacquisition_diagnostics: list[dict[str, Any]]
    safety_diagnostics: list[dict[str, Any]]
    actor_time_ms: float
    reacquisition_time_ms: float
    safety_time_ms: float


def load_frozen_actor(
    checkpoint: str | Path,
    *,
    device: str | torch.device,
    expected_sha256: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Strictly load the frozen 39->128->2 MLP actor."""
    report = inspect_actor_checkpoint(checkpoint, 39, 2)
    if expected_sha256 is not None and report.sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint SHA256 {report.sha256} != expected {expected_sha256}"
        )
    from algos.networks import build_gaussian_actor

    actor = build_gaussian_actor(
        report.input_dim,
        report.action_dim,
        hidden=report.hidden_dim,
        n_agent=3,
        actor_encoder="mlp",
    )
    path = Path(checkpoint).resolve()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    actor.load_state_dict(payload["actor"], strict=True)
    actor.to(device)
    actor.eval()
    return actor, report.as_dict()


class FrozenMethodBridge:
    """Compose a frozen actor with the exact CPU method-side controllers."""

    def __init__(
        self,
        actor: torch.nn.Module,
        task_spec: P5TaskSpec,
        num_envs: int,
        *,
        reacquisition_params: dict[str, Any] | None,
        safety_params: dict[str, Any] | None = None,
        physical_safety: bool = False,
    ) -> None:
        task_spec.validate()
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.actor = actor
        self.task_spec = task_spec
        self.num_envs = int(num_envs)
        self.reacquisition_params = (
            dict(reacquisition_params) if reacquisition_params is not None else None
        )
        self.reacquisition = [
            ReacquisitionController(self.reacquisition_params)
            if self.reacquisition_params is not None else None
            for _ in range(self.num_envs)
        ]
        safety_values = dict(safety_params or {})
        safety_type = (
            IsaacPhysicalSafetyLayer
            if physical_safety
            else IsaacOSQPCompatibleSafetyLayer
        )
        self.safety = [
            safety_type(
                dt=task_spec.timing.high_level_dt,
                a_max=task_spec.a_max_mps2,
                w_max=task_spec.yaw_rate_max_rps,
                v_max=task_spec.v_max_mps,
                pursuer_radius=task_spec.pursuer_radius_m,
                **safety_values,
            )
            for _ in range(self.num_envs)
        ]

    def reset_indices(self, indices: np.ndarray | list[int]) -> None:
        for index in np.asarray(indices, dtype=int).reshape(-1):
            if not 0 <= int(index) < self.num_envs:
                raise IndexError(index)
            controller = self.reacquisition[int(index)]
            if controller is not None:
                controller.reset()
            self.safety[int(index)].reset_statistics()

    def _legacy_view(self, state: MethodBatchState, index: int) -> Any:
        offset = 0.5 * self.task_spec.arena_size_m
        cfg = SimpleNamespace(
            obs_mode=self.task_spec.obs_mode,
            arena_size=self.task_spec.arena_size_m,
            dt=self.task_spec.timing.high_level_dt,
            a_max=self.task_spec.a_max_mps2,
            w_max=self.task_spec.yaw_rate_max_rps,
            v_max=self.task_spec.v_max_mps,
            pursuer_radius=self.task_spec.pursuer_radius_m,
            lost_cap_steps=self.task_spec.lost_cap_steps,
        )
        return SimpleNamespace(
            n=self.task_spec.n_pursuers,
            cfg=cfg,
            belief=None,
            p_pos=np.asarray(state.pursuer_xy_centered_m[index], dtype=float) + offset,
            p_psi=np.asarray(state.pursuer_yaw_rad[index], dtype=float),
            p_v=np.asarray(state.pursuer_speed_mps[index], dtype=float),
            p_velocity_xy=np.asarray(
                state.pursuer_velocity_xy_mps[index], dtype=float
            ),
            last_seen_pos=np.asarray(
                state.last_seen_target_xy_legacy_m[index], dtype=float
            ),
            visible_each=np.asarray(state.visible_each[index], dtype=bool),
            t_lost=int(state.lost_steps[index]),
            obs_pos=np.asarray(state.obstacle_xy_centered_m[index], dtype=float) + offset,
            obs_r=np.asarray(state.obstacle_radius_m[index], dtype=float),
            obs_vel=np.asarray(state.obstacle_velocity_mps[index], dtype=float),
        )


    def act(
        self,
        observations: torch.Tensor,
        state: MethodBatchState,
    ) -> MethodActionResult:
        if observations.shape != (
            self.num_envs,
            self.task_spec.n_pursuers,
            self.task_spec.observation_dim,
        ):
            raise ValueError(
                "observations must have shape "
                f"[{self.num_envs}, {self.task_spec.n_pursuers}, "
                f"{self.task_spec.observation_dim}]"
            )
        state.validate(self.num_envs, self.task_spec.n_pursuers)
        actor_device = next(self.actor.parameters()).device
        actor_start = time.perf_counter()
        with torch.no_grad():
            flat = observations.to(actor_device).reshape(-1, observations.shape[-1])
            mean, _ = self.actor(flat)
            actor_actions = torch.tanh(mean).reshape(
                self.num_envs, self.task_spec.n_pursuers, 2
            ).cpu().numpy()
        actor_time_ms = (time.perf_counter() - actor_start) * 1000.0

        guided = actor_actions.copy()
        reacquisition_diagnostics: list[dict[str, Any]] = []
        reacquisition_start = time.perf_counter()
        views = [self._legacy_view(state, index) for index in range(self.num_envs)]
        for index, (view, controller) in enumerate(zip(views, self.reacquisition)):
            if controller is None:
                reacquisition_diagnostics.append(
                    {"active": False, "phase": "disabled", "scout_index": -1}
                )
                continue
            guided[index], diagnostics = controller.apply(view, guided[index])
            reacquisition_diagnostics.append(diagnostics)
        reacquisition_time_ms = (time.perf_counter() - reacquisition_start) * 1000.0

        executed = guided.copy()
        safety_diagnostics: list[dict[str, Any]] = []
        safety_start = time.perf_counter()
        for index, (view, layer) in enumerate(zip(views, self.safety)):
            executed[index], diagnostics = layer.project_joint(view, guided[index])
            safety_diagnostics.append(diagnostics)
        safety_time_ms = (time.perf_counter() - safety_start) * 1000.0
        if not all(np.isfinite(values).all() for values in (actor_actions, guided, executed)):
            raise RuntimeError("method bridge emitted NaN or Inf")
        return MethodActionResult(
            actor_actions=np.clip(actor_actions, -1.0, 1.0),
            guided_actions=np.clip(guided, -1.0, 1.0),
            executed_actions=np.clip(executed, -1.0, 1.0),
            reacquisition_diagnostics=reacquisition_diagnostics,
            safety_diagnostics=safety_diagnostics,
            actor_time_ms=actor_time_ms,
            reacquisition_time_ms=reacquisition_time_ms,
            safety_time_ms=safety_time_ms,
        )


class IsaacOSQPCompatibleSafetyLayer(DynamicCBFSafetyLayer):
    """P5-only OSQP 0.6/1.x API adapter for the frozen safety equations."""

    def _solve_qp(
        self,
        nominal: np.ndarray,
        gradient: np.ndarray,
        constant: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        self.qp_attempt_count += 1
        osqp = dynamic_safety_module.osqp
        sparse = dynamic_safety_module.sparse
        if osqp is None or sparse is None:
            self.qp_failure_count += 1
            return None, float("-inf"), "osqp_unavailable"
        dimension = nominal.size
        identity = sparse.eye(dimension, format="csc")
        matrix = sparse.vstack(
            [sparse.csc_matrix(gradient), identity], format="csc"
        )
        lower = np.concatenate([-constant, -np.ones(dimension)])
        upper = np.concatenate(
            [np.full(len(constant), np.inf), np.ones(dimension)]
        )
        setup_args = {
            "P": 2.0 * identity,
            "q": -2.0 * nominal,
            "A": matrix,
            "l": lower,
            "u": upper,
            "verbose": False,
            "eps_abs": self.cfg.qp_eps_abs,
            "eps_rel": self.cfg.qp_eps_rel,
            "max_iter": self.cfg.qp_max_iter,
            "adaptive_rho": False,
        }
        solver = osqp.OSQP()
        try:
            solver.setup(**setup_args, polishing=True)
        except TypeError as exc:
            if "polishing" not in str(exc):
                raise
            solver = osqp.OSQP()
            solver.setup(**setup_args, polish=True)
        try:
            result = solver.solve(raise_error=False)
        except TypeError:
            result = solver.solve()
        status = str(result.info.status).lower()
        if result.x is None or "solved" not in status:
            self.qp_failure_count += 1
            return None, float("-inf"), status
        action = np.clip(np.asarray(result.x, dtype=float), -1.0, 1.0)
        residual = float(np.min(gradient @ action + constant))
        if residual < -self.cfg.qp_eps_abs:
            self.minimum_rejected_solver_residual = min(
                self.minimum_rejected_solver_residual, residual
            )
            self.qp_failure_count += 1
            return None, residual, "residual_failure"
        self.minimum_solver_residual = min(self.minimum_solver_residual, residual)
        return action, residual, status


class IsaacPhysicalSafetyLayer(IsaacOSQPCompatibleSafetyLayer):
    """Simulator-transfer predictor for the existing dynamic_v2 projection.

    The 2-D layer assumes velocity is always aligned with yaw and that actor
    acceleration is realized within one 10 Hz step.  A quadrotor retains a
    lateral velocity component and tracks the velocity reference through the
    100 Hz low-level loop.  This subclass changes only that path predictor;
    The Isaac-only transfer guard also adds team-shared target keep-out and a
    braking-time viability horizon.  The frozen 2-D ``dynamic_v2`` source and
    checkpoint configuration remain unchanged.
    """

    version = "dynamic_v2_isaac_physical_target_guard"
    revision = "p5.2"
    physics_dt = 0.01
    velocity_kp = 2.0
    max_planar_accel = 4.0
    target_contact_distance_m = 0.3
    target_guard_lost_steps = 20
    minimum_physical_horizon_s = 2.5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        requested_horizon = float(kwargs.get("horizon_s", 1.5))
        kwargs["horizon_s"] = max(
            requested_horizon, self.minimum_physical_horizon_s
        )
        super().__init__(*args, **kwargs)
        self.requested_horizon_s = requested_horizon
        self.revision = "p5.5-single-rollout-viability-2p5s"

    def reset_statistics(self) -> None:
        super().reset_statistics()
        self._previous_target_position: np.ndarray | None = None
        self._previous_target_visible = False
        self._target_belief_position: np.ndarray | None = None
        self._target_belief_velocity = np.zeros(2, dtype=float)
        self._target_guard_active = False

    def _update_target_guard_belief(self, env: Any) -> None:
        position = np.asarray(env.last_seen_pos, dtype=float).reshape(2)
        team_visible = bool(np.asarray(env.visible_each, dtype=bool).any())
        lost_steps = int(env.t_lost)
        if team_visible:
            if self._previous_target_position is not None and self._previous_target_visible:
                velocity = (position - self._previous_target_position) / self.dt
                self._target_belief_velocity = self._clip_vector_norm(
                    velocity[None], self.v_max
                )[0]
            else:
                self._target_belief_velocity = np.zeros(2, dtype=float)
            self._previous_target_position = position.copy()
            self._target_belief_position = position.copy()
            self._target_guard_active = True
        elif (
            self._previous_target_position is not None
            and lost_steps <= self.target_guard_lost_steps
        ):
            self._target_belief_position = (
                self._previous_target_position
                + self._target_belief_velocity * lost_steps * self.dt
            )
            self._target_guard_active = True
        else:
            self._target_belief_position = None
            self._target_guard_active = False
        self._previous_target_visible = team_visible

    def _predict_target(self) -> np.ndarray:
        if not self._target_guard_active or self._target_belief_position is None:
            return np.empty((self.prediction_steps, 2), dtype=float)
        steps = np.arange(1, self.prediction_steps + 1, dtype=float)[:, None]
        return (
            self._target_belief_position[None]
            + steps * self.dt * self._target_belief_velocity[None]
        )

    def _target_clearance_from_path(
        self, path: np.ndarray, *, buffered: bool
    ) -> np.ndarray:
        if not self._target_guard_active:
            return np.full((self.prediction_steps, path.shape[1]), np.inf)
        target_path = self._predict_target()
        margin = self.target_contact_distance_m
        if buffered:
            margin += self.cfg.safety_buffer
        distance = np.linalg.norm(
            path - target_path[:, None, None, :], axis=-1
        )
        return distance.min(axis=-1) - margin

    def _target_clearance_by_step(
        self, env: Any, actions: np.ndarray, *, buffered: bool
    ) -> np.ndarray:
        actions = np.asarray(actions, dtype=float)
        if actions.ndim == 2:
            actions = actions[None]
        path = self._predict_agents_batch(env, actions)
        return self._target_clearance_from_path(path, buffered=buffered)

    @property
    def boundary_buffer(self) -> float:
        """Worst-case straight-line braking distance at the speed limit."""
        return self.v_max * self.v_max / (2.0 * self.max_planar_accel)

    @staticmethod
    def _clip_vector_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
        norm = np.linalg.norm(vector, axis=-1, keepdims=True)
        scale = np.minimum(1.0, maximum / np.maximum(norm, 1.0e-12))
        return vector * scale

    def _predict_agents_batch(
        self, env: Any, actions: np.ndarray,
    ) -> np.ndarray:
        actions = np.asarray(actions, dtype=float)
        if actions.ndim == 2:
            actions = actions[None]
        count = actions.shape[0]
        pos = np.broadcast_to(
            np.asarray(env.p_pos, dtype=float), (count, env.n, 2)
        ).copy()
        yaw = np.broadcast_to(
            np.asarray(env.p_psi, dtype=float), (count, env.n)
        ).copy()
        velocity = np.broadcast_to(
            np.asarray(env.p_velocity_xy, dtype=float), (count, env.n, 2)
        ).copy()
        path = np.empty((self.prediction_steps, count, env.n, 2), dtype=float)
        acceleration = actions[..., 0] * self.a_max
        turn_rate = actions[..., 1] * self.w_max
        substeps = max(1, int(round(self.dt / self.physics_dt)))
        substep_dt = self.dt / substeps
        for high_step in range(self.prediction_steps):
            speed = np.linalg.norm(velocity, axis=-1)
            desired_speed = np.clip(
                speed + acceleration * self.dt, 0.0, self.v_max
            )
            desired_yaw = self._wrap_angle(yaw + turn_rate * self.dt)
            desired_velocity = np.stack(
                (
                    desired_speed * np.cos(desired_yaw),
                    desired_speed * np.sin(desired_yaw),
                ),
                axis=-1,
            )
            for _ in range(substeps):
                planar_acceleration = self._clip_vector_norm(
                    self.velocity_kp * (desired_velocity - velocity),
                    self.max_planar_accel,
                )
                velocity = velocity + planar_acceleration * substep_dt
                pos = pos + velocity * substep_dt
            yaw = desired_yaw
            path[high_step] = pos
        return path

    def _predict_agents(self, env: Any, actions: np.ndarray) -> np.ndarray:
        return self._predict_agents_batch(env, np.asarray(actions)[None])[:, 0]

    @staticmethod
    def _boundary_clearances(
        path: np.ndarray, arena_size: float, margin: float
    ) -> np.ndarray:
        """Return [step, candidate] minimum four-wall clearance."""
        lower = path - margin
        upper = arena_size - margin - path
        return np.minimum(lower, upper).min(axis=(-1, -2))

    def _constraint_values(
        self, env: Any, actions: np.ndarray, obstacle_path: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        values, clearances = super()._constraint_values(
            env, actions, obstacle_path
        )
        path = self._predict_agents(env, actions)
        margin = self.pursuer_radius + self.boundary_buffer
        wall = np.concatenate(
            (path - margin, env.cfg.arena_size - margin - path), axis=-1
        ).reshape(-1)
        if not self._target_guard_active:
            return np.concatenate((values, wall)), np.concatenate((clearances, wall))
        target_path = self._predict_target()
        target_distance = np.linalg.norm(
            path - target_path[:, None, :], axis=-1
        ).reshape(-1)
        target_margin = self.target_contact_distance_m + self.cfg.safety_buffer
        target_values = target_distance * target_distance - target_margin * target_margin
        target_clearance = target_distance - target_margin
        return (
            np.concatenate((values, wall, target_values)),
            np.concatenate((clearances, wall, target_clearance)),
        )

    def _batch_clearance_metrics(
        self, env: Any, candidates: np.ndarray, obstacle_path: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # A fallback has 15**3=3375 joint candidates.  Reuse one physical
        # rollout for every constraint family: the earlier transfer audit
        # recomputed this identical 2.5 s trajectory five times per call.
        path = self._predict_agents_batch(env, candidates)
        count = candidates.shape[0]
        horizon = np.full(count, np.inf, dtype=float)
        next_buffer = np.full(count, np.inf, dtype=float)
        next_physical = np.full(count, np.inf, dtype=float)
        physical_by_step = np.full(
            (self.prediction_steps, count), np.inf, dtype=float
        )
        agent_obstacle_buffer = np.full(
            (self.prediction_steps, count), np.inf, dtype=float
        )
        agent_physical_margin = 2.0 * self.pursuer_radius
        agent_buffer_margin = agent_physical_margin + self.cfg.safety_buffer
        for left in range(env.n):
            for right in range(left + 1, env.n):
                distance = np.linalg.norm(
                    path[:, :, left] - path[:, :, right], axis=-1
                )
                physical = distance - agent_physical_margin
                buffered = distance - agent_buffer_margin
                horizon = np.minimum(horizon, buffered.min(axis=0))
                next_buffer = np.minimum(next_buffer, buffered[0])
                next_physical = np.minimum(next_physical, physical[0])
                physical_by_step = np.minimum(physical_by_step, physical)
                agent_obstacle_buffer = np.minimum(
                    agent_obstacle_buffer, buffered
                )
        for agent in range(env.n):
            for obstacle, radius in enumerate(env.obs_r):
                physical_margin = self.pursuer_radius + float(radius)
                buffer_margin = physical_margin + self.cfg.safety_buffer
                distance = np.linalg.norm(
                    path[:, :, agent] - obstacle_path[:, None, obstacle], axis=-1
                )
                physical = distance - physical_margin
                buffered = distance - buffer_margin
                horizon = np.minimum(horizon, buffered.min(axis=0))
                next_buffer = np.minimum(next_buffer, buffered[0])
                next_physical = np.minimum(next_physical, physical[0])
                physical_by_step = np.minimum(physical_by_step, physical)
                agent_obstacle_buffer = np.minimum(
                    agent_obstacle_buffer, buffered
                )
        physical_unsafe = physical_by_step < -self.cfg.verification_tolerance
        physical_any_unsafe = np.any(physical_unsafe, axis=0)
        physical_first_unsafe = np.argmax(physical_unsafe, axis=0)
        safe_prefix = np.where(
            physical_any_unsafe, physical_first_unsafe, self.prediction_steps
        )
        wall_buffer = self._boundary_clearances(
            path,
            float(env.cfg.arena_size),
            self.pursuer_radius + self.boundary_buffer,
        )
        wall_physical = self._boundary_clearances(
            path, float(env.cfg.arena_size), self.pursuer_radius
        )
        horizon = np.minimum(horizon, wall_buffer.min(axis=0))
        next_buffer = np.minimum(next_buffer, wall_buffer[0])
        next_physical = np.minimum(next_physical, wall_physical[0])
        # The wall buffer is a speed-limit braking distance, not an arbitrary
        # comfort margin.  Rank emergency candidates by retaining that
        # recoverable set; waiting for a physical wall crossing is too late.
        unsafe = wall_buffer < -self.cfg.verification_tolerance
        any_unsafe = np.any(unsafe, axis=0)
        first_unsafe = np.argmax(unsafe, axis=0)
        wall_safe_prefix = np.where(
            any_unsafe, first_unsafe, self.prediction_steps
        )
        target_buffer = self._target_clearance_from_path(path, buffered=True)
        target_physical = self._target_clearance_from_path(path, buffered=False)
        horizon = np.minimum(horizon, target_buffer.min(axis=0))
        next_buffer = np.minimum(next_buffer, target_buffer[0])
        next_physical = np.minimum(next_physical, target_physical[0])
        # The target buffer absorbs velocity-estimation/model error.  Treat it
        # as the viability prefix during emergency ranking instead of merely
        # postponing a physical 0.3 m contact one receding-horizon step at a time.
        target_unsafe = target_buffer < -self.cfg.verification_tolerance
        target_any_unsafe = np.any(target_unsafe, axis=0)
        target_first_unsafe = np.argmax(target_unsafe, axis=0)
        target_safe_prefix = np.where(
            target_any_unsafe, target_first_unsafe, self.prediction_steps
        )
        agent_obstacle_unsafe = (
            agent_obstacle_buffer < -self.cfg.verification_tolerance
        )
        agent_obstacle_any_unsafe = np.any(agent_obstacle_unsafe, axis=0)
        agent_obstacle_first_unsafe = np.argmax(agent_obstacle_unsafe, axis=0)
        agent_obstacle_safe_prefix = np.where(
            agent_obstacle_any_unsafe,
            agent_obstacle_first_unsafe,
            self.prediction_steps,
        )
        return (
            horizon,
            next_buffer,
            next_physical,
            np.minimum.reduce(
                (
                    safe_prefix,
                    wall_safe_prefix,
                    target_safe_prefix,
                    agent_obstacle_safe_prefix,
                )
            ),
        )

    def _physical_clearance_metrics(
        self, env: Any, actions: np.ndarray, obstacle_path: np.ndarray
    ) -> tuple[float, float]:
        horizon, next_step = super()._physical_clearance_metrics(
            env, actions, obstacle_path
        )
        path = self._predict_agents(env, actions)
        wall = self._boundary_clearances(
            path[:, None], float(env.cfg.arena_size), self.pursuer_radius
        )[:, 0]
        target = self._target_clearance_by_step(
            env, np.asarray(actions)[None], buffered=False
        )[:, 0]
        return (
            min(horizon, float(wall.min()), float(target.min())),
            min(next_step, float(wall[0]), float(target[0])),
        )

    def _record_projection(
        self,
        env: Any,
        obstacle_path: np.ndarray,
        actor_nominal: np.ndarray,
        output: np.ndarray,
        output_min: float,
        diagnostics: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        output, diagnostics = super()._record_projection(
            env,
            obstacle_path,
            actor_nominal,
            output,
            output_min,
            diagnostics,
        )
        collision_horizon, collision_next = (
            DynamicCBFSafetyLayer._physical_clearance_metrics(
                self, env, output, obstacle_path
            )
        )
        path = self._predict_agents(env, output)
        wall = self._boundary_clearances(
            path[:, None], float(env.cfg.arena_size), self.pursuer_radius
        )[:, 0]
        target = self._target_clearance_by_step(
            env, np.asarray(output)[None], buffered=False
        )[:, 0]
        diagnostics.update(
            {
                "output_horizon_collision_clearance": collision_horizon,
                "output_next_step_collision_clearance": collision_next,
                "output_horizon_boundary_clearance": float(wall.min()),
                "output_next_step_boundary_clearance": float(wall[0]),
                "output_horizon_target_clearance": float(target.min()),
                "output_next_step_target_clearance": float(target[0]),
            }
        )
        self.last_diagnostics = diagnostics
        return output, diagnostics

    def project_joint(
        self, env: Any, nominal_actions: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._update_target_guard_belief(env)
        nominal = np.clip(
            np.asarray(nominal_actions, dtype=float).reshape(env.n, 2),
            -1.0,
            1.0,
        )
        nominal_target = self._target_clearance_by_step(
            env, nominal[None], buffered=True
        )[:, 0]
        output, diagnostics = super().project_joint(env, nominal)
        diagnostics.update(
            {
                "target_guard_active": self._target_guard_active,
                "target_guard_lost_steps_limit": self.target_guard_lost_steps,
                "target_guard_physical_distance_m": self.target_contact_distance_m,
                "requested_horizon_s": self.requested_horizon_s,
                "effective_physical_horizon_s": self.prediction_steps * self.dt,
                "target_viability_uses_design_buffer": True,
                "boundary_viability_uses_braking_buffer": True,
                "agent_obstacle_viability_uses_design_buffer": True,
                "target_belief_position": (
                    self._target_belief_position.tolist()
                    if self._target_belief_position is not None
                    else None
                ),
                "target_belief_velocity_mps": self._target_belief_velocity.tolist(),
                "nominal_horizon_target_buffer_clearance": float(
                    nominal_target.min()
                ),
                "nominal_next_step_target_buffer_clearance": float(
                    nominal_target[0]
                ),
            }
        )
        self.last_diagnostics = diagnostics
        return output, diagnostics
