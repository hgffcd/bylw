"""Auditable P5 step-metric contract and episode accumulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


P5_STEP_METRIC_FIELDS = (
    "episode_seed",
    "step",
    "sim_time_s",
    "team_visible",
    "visibility_status",
    "lost_steps",
    "recapture_count",
    "high_level_action_norm",
    "safety_called",
    "safety_corrected",
    "safety_infeasible",
    "safety_fallback",
    "safety_correction_norm",
    "min_agent_clearance_m",
    "min_static_obstacle_clearance_m",
    "min_dynamic_obstacle_clearance_m",
    "height_rmse_m",
    "speed_tracking_rmse_mps",
    "yaw_rate_tracking_rmse_rps",
    "thrust_saturation_fraction",
    "moment_saturation_fraction",
    "policy_time_ms",
    "safety_time_ms",
    "controller_time_ms",
    "physics_time_ms",
    "success_6m",
    "success_5m",
    "contact_0p3m",
    "collision_agent",
    "collision_static",
    "collision_dynamic",
    "timeout",
)


def validate_step_metric(record: dict) -> None:
    missing = [field for field in P5_STEP_METRIC_FIELDS if field not in record]
    extra = sorted(set(record) - set(P5_STEP_METRIC_FIELDS))
    if missing or extra:
        raise ValueError(f"metric schema mismatch: missing={missing}, extra={extra}")
    for name, value in record.items():
        if isinstance(value, bool) or isinstance(value, str):
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"metric {name!r} must be finite, got {value!r}")


@dataclass
class EpisodeMetricAccumulator:
    episode_seed: int
    steps: int = 0
    recapture_count: int = 0
    minimum_agent_clearance_m: float = math.inf
    minimum_static_clearance_m: float = math.inf
    minimum_dynamic_clearance_m: float = math.inf
    safety_calls: int = 0
    safety_corrections: int = 0
    safety_infeasible: int = 0
    safety_fallbacks: int = 0
    terminal: dict = field(default_factory=dict)

    def update(self, record: dict) -> None:
        validate_step_metric(record)
        if int(record["episode_seed"]) != self.episode_seed:
            raise ValueError("episode seed changed inside an accumulator")
        self.steps += 1
        self.recapture_count = max(self.recapture_count, int(record["recapture_count"]))
        self.minimum_agent_clearance_m = min(
            self.minimum_agent_clearance_m, float(record["min_agent_clearance_m"])
        )
        self.minimum_static_clearance_m = min(
            self.minimum_static_clearance_m,
            float(record["min_static_obstacle_clearance_m"]),
        )
        self.minimum_dynamic_clearance_m = min(
            self.minimum_dynamic_clearance_m,
            float(record["min_dynamic_obstacle_clearance_m"]),
        )
        self.safety_calls += int(bool(record["safety_called"]))
        self.safety_corrections += int(bool(record["safety_corrected"]))
        self.safety_infeasible += int(bool(record["safety_infeasible"]))
        self.safety_fallbacks += int(bool(record["safety_fallback"]))
        if any(
            bool(record[key])
            for key in (
                "success_6m",
                "collision_agent",
                "collision_static",
                "collision_dynamic",
                "timeout",
            )
        ):
            self.terminal = {
                key: record[key]
                for key in (
                    "success_6m",
                    "success_5m",
                    "contact_0p3m",
                    "collision_agent",
                    "collision_static",
                    "collision_dynamic",
                    "timeout",
                )
            }

    def summary(self) -> dict:
        return {
            "episode_seed": self.episode_seed,
            "steps": self.steps,
            "recapture_count": self.recapture_count,
            "min_agent_clearance_m": self.minimum_agent_clearance_m,
            "min_static_obstacle_clearance_m": self.minimum_static_clearance_m,
            "min_dynamic_obstacle_clearance_m": self.minimum_dynamic_clearance_m,
            "safety_calls": self.safety_calls,
            "safety_corrections": self.safety_corrections,
            "safety_infeasible": self.safety_infeasible,
            "safety_fallbacks": self.safety_fallbacks,
            **self.terminal,
        }
