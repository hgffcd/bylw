# -*- coding: utf-8 -*-
"""Belief/last-seen guided tracker-scout controller for J9.

The controller is a deterministic method-side action composition layer.  It
uses only team kinematics and the same target reference exposed to the policy
(IMM belief mean or last-seen position); it never reads the hidden true target
state during loss.  Existing E1--J8 behavior is unchanged unless the controller
is explicitly enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class ReacquisitionConfig:
    enabled: bool = True
    start_lost_steps: int = 1
    slot_radius_m: float = 8.0
    scout_blend: float = 0.85
    tracker_blend: float = 0.55
    blend_growth: float = 0.10
    scout_speed_ratio: float = 1.0
    tracker_speed_ratio: float = 0.65
    heading_tau_s: float = 0.55
    speed_tau_s: float = 1.0
    slot_arrival_m: float = 1.5
    # J11 opt-in handoff: once any pursuer reacquires the target, keep the
    # deterministic slot controller active briefly and contract the search
    # ring into the strict capture radius.  A zero hold preserves J9/J10
    # behavior exactly.
    post_reacquisition_hold_steps: int = 0
    capture_slot_radius_m: float = 4.0
    contraction_steps: int = 30
    visible_tracker_blend: float = 0.65
    # J15 diagnostic baseline: keep geometric slot guidance active while the
    # target is visible.  False preserves every frozen J9--J14 behavior.
    always_on_visible: bool = False

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown reacquisition parameters: {unknown}")
        cfg = cls(**values)
        if cfg.start_lost_steps < 1:
            raise ValueError("start_lost_steps must be >= 1")
        if cfg.post_reacquisition_hold_steps < 0:
            raise ValueError("post_reacquisition_hold_steps must be >= 0")
        if cfg.contraction_steps < 1:
            raise ValueError("contraction_steps must be >= 1")
        if cfg.slot_radius_m <= 0 or cfg.heading_tau_s <= 0 or cfg.speed_tau_s <= 0:
            raise ValueError("slot radius and controller time constants must be positive")
        if not 0 < cfg.capture_slot_radius_m <= cfg.slot_radius_m:
            raise ValueError(
                "capture_slot_radius_m must be in (0, slot_radius_m]")
        for name in ("scout_blend", "tracker_blend", "visible_tracker_blend"):
            value = getattr(cfg, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        return cfg


class ReacquisitionController:
    """Assign one scout and preserve two supporting encirclement slots on loss."""

    def __init__(self, params=None):
        self.cfg = ReacquisitionConfig.from_mapping(params)
        self.reset()

    def reset(self):
        self.active = False
        self.scout_index = None
        self.slot_angles = None
        self.calls = 0
        self.active_steps = 0
        self.visible_hold_steps = 0

    @staticmethod
    def target_reference(env):
        if env.cfg.obs_mode == "belief" and env.belief is not None:
            center = np.asarray(env.belief["mu"], dtype=float)
            source = "belief"
        else:
            center = np.asarray(env.last_seen_pos, dtype=float)
            source = "last_seen"
        return np.clip(center, 0.0, env.cfg.arena_size), source

    @staticmethod
    def _equal_slot_assignment(positions, center):
        bearings = np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0])
        order = np.argsort(bearings)
        offsets = 2.0 * math.pi * np.arange(len(order), dtype=float) / len(order)
        phase_terms = np.exp(1j * (bearings[order] - offsets))
        phase = float(np.angle(phase_terms.mean()))
        assigned = np.empty(len(order), dtype=float)
        assigned[order] = _wrap(phase + offsets)
        return assigned

    def _activate(self, env, center):
        distances = np.linalg.norm(env.p_pos - center, axis=1)
        self.scout_index = int(np.argmin(distances))
        self.slot_angles = self._equal_slot_assignment(env.p_pos, center)
        self.active = True

    def _guidance_action(self, env, agent_index, goal, face_center, speed_ratio):
        delta = goal - env.p_pos[agent_index]
        distance = float(np.linalg.norm(delta))
        if face_center:
            heading_vector = goal - env.p_pos[agent_index]
        else:
            heading_vector = delta
        if np.linalg.norm(heading_vector) < 1e-8:
            heading_vector = np.array(
                [math.cos(env.p_psi[agent_index]), math.sin(env.p_psi[agent_index])])
        desired_heading = math.atan2(heading_vector[1], heading_vector[0])
        heading_error = _wrap(desired_heading - float(env.p_psi[agent_index]))
        turn = np.clip(
            heading_error / (env.cfg.w_max * self.cfg.heading_tau_s), -1.0, 1.0)
        desired_speed = float(speed_ratio) * env.cfg.v_max
        if distance < self.cfg.slot_arrival_m:
            desired_speed = 0.0
        accel = np.clip(
            (desired_speed - float(env.p_v[agent_index]))
            / (env.cfg.a_max * self.cfg.speed_tau_s),
            -1.0,
            1.0,
        )
        return np.array([accel, turn], dtype=float)

    def _slot_guidance(self, env, actions, center, radius, blend):
        """Guide all pursuers to persistent equal-angle slots around center."""
        composed = actions.copy()
        for index in range(env.n):
            angle = float(self.slot_angles[index])
            goal = center + radius * np.array([
                math.cos(angle), math.sin(angle)])
            distance_to_slot = float(np.linalg.norm(goal - env.p_pos[index]))
            if distance_to_slot < self.cfg.slot_arrival_m:
                guide = self._guidance_action(
                    env, index, center, True, 0.0)
            else:
                guide = self._guidance_action(
                    env, index, goal, False, self.cfg.tracker_speed_ratio)
            composed[index] = (1.0 - blend) * actions[index] + blend * guide
        return np.clip(composed, -1.0, 1.0)

    def apply(self, env, policy_actions):
        actions = np.asarray(policy_actions, dtype=float).reshape(env.n, 2)
        self.calls += 1
        if not self.cfg.enabled:
            return actions.copy(), {"active": False, "scout_index": -1}
        if bool(env.visible_each.any()):
            if self.cfg.always_on_visible:
                center, source = self.target_reference(env)
                if not self.active:
                    self._activate(env, center)
                self.visible_hold_steps += 1
                self.active_steps += 1
                radius = self.cfg.capture_slot_radius_m
                composed = self._slot_guidance(
                    env, actions, center, radius,
                    self.cfg.visible_tracker_blend)
                return composed, {
                    "active": True,
                    "phase": "visible_geometric",
                    "scout_index": int(self.scout_index),
                    "reference_source": source,
                    "mean_blend": float(self.cfg.visible_tracker_blend),
                    "slot_radius_m": float(radius),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                }
            if self.active and self.cfg.post_reacquisition_hold_steps > 0:
                center, source = self.target_reference(env)
                self.visible_hold_steps += 1
                self.active_steps += 1
                fraction = min(
                    self.visible_hold_steps / self.cfg.contraction_steps, 1.0)
                radius = (
                    (1.0 - fraction) * self.cfg.slot_radius_m
                    + fraction * self.cfg.capture_slot_radius_m
                )
                composed = self._slot_guidance(
                    env, actions, center, radius,
                    self.cfg.visible_tracker_blend)
                diagnostics = {
                    "active": True,
                    "phase": "visible_contraction",
                    "scout_index": int(self.scout_index),
                    "reference_source": source,
                    "mean_blend": float(self.cfg.visible_tracker_blend),
                    "slot_radius_m": float(radius),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                }
                if (
                    self.visible_hold_steps
                    >= self.cfg.post_reacquisition_hold_steps
                ):
                    self.reset()
                    self.calls = 1
                return composed, diagnostics
            if self.active:
                self.reset()
                self.calls = 1
            return actions.copy(), {"active": False, "scout_index": -1}
        if int(env.t_lost) < self.cfg.start_lost_steps:
            return actions.copy(), {"active": False, "scout_index": -1}

        center, source = self.target_reference(env)
        if not self.active:
            self._activate(env, center)
        self.visible_hold_steps = 0
        self.active_steps += 1
        loss_fraction = min(float(env.t_lost) / max(env.cfg.lost_cap_steps, 1), 1.0)
        composed = actions.copy()
        blends = np.zeros(env.n, dtype=float)
        for index in range(env.n):
            if index == self.scout_index:
                goal = center
                guide = self._guidance_action(
                    env, index, goal, False, self.cfg.scout_speed_ratio)
                blend = min(1.0, self.cfg.scout_blend + self.cfg.blend_growth * loss_fraction)
            else:
                angle = float(self.slot_angles[index])
                goal = center + self.cfg.slot_radius_m * np.array(
                    [math.cos(angle), math.sin(angle)])
                distance_to_slot = float(np.linalg.norm(goal - env.p_pos[index]))
                if distance_to_slot < self.cfg.slot_arrival_m:
                    # Once on station, stop and face the belief/last-seen center.
                    guide = self._guidance_action(
                        env, index, center, True, 0.0)
                else:
                    guide = self._guidance_action(
                        env, index, goal, False, self.cfg.tracker_speed_ratio)
                blend = min(1.0, self.cfg.tracker_blend + self.cfg.blend_growth * loss_fraction)
            composed[index] = (1.0 - blend) * actions[index] + blend * guide
            blends[index] = blend
        return np.clip(composed, -1.0, 1.0), {
            "active": True,
            "phase": "lost_search",
            "scout_index": int(self.scout_index),
            "reference_source": source,
            "mean_blend": float(blends.mean()),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
        }


def build_reacquisition_controller(params=None):
    values = dict(params or {})
    if not values or not bool(values.get("enabled", True)):
        return None
    return ReacquisitionController(values)
