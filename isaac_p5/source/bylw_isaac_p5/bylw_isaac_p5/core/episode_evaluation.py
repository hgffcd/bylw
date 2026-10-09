"""CPU-only episode accounting for P5-1 matched Isaac evaluation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


UINT32_MODULUS = 2**32
EPISODE_SEED_STRIDE = 1_000_003


def episode_quotas(total_episodes: int, num_envs: int) -> list[int]:
    """Deterministically allocate exactly ``total_episodes`` across env ids."""
    if total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    quotient, remainder = divmod(int(total_episodes), int(num_envs))
    return [quotient + int(index < remainder) for index in range(num_envs)]


def expected_episode_seed(base_seed: int, env_id: int, episode_ordinal: int) -> int:
    if env_id < 0 or episode_ordinal < 0:
        raise ValueError("env_id and episode_ordinal must be non-negative")
    return int(
        (int(base_seed) + int(env_id) + int(episode_ordinal) * EPISODE_SEED_STRIDE)
        % UINT32_MODULUS
    )


def _new_accumulator() -> dict[str, Any]:
    return {
        "episode_seed": None,
        "episode_steps": 0,
        "visible_steps": 0,
        "lost_steps_total": 0,
        "max_consecutive_loss_steps": 0,
        "recapture_count": 0,
        "lost_search_steps": 0,
        "visible_contraction_steps": 0,
        "safety_corrected_steps": 0,
        "safety_qp_attempt_steps": 0,
        "safety_fallback_steps": 0,
        "safety_correction_norm_sum": 0.0,
        "safety_correction_norm_max": 0.0,
        "min_predicted_physical_clearance_m": math.inf,
        "min_predicted_target_clearance_m": math.inf,
        "min_agent_clearance_m": math.inf,
        "min_static_clearance_m": math.inf,
        "min_pursuer_boundary_clearance_m": math.inf,
        "max_speed_tracking_error_mps": 0.0,
        "max_velocity_yaw_slip_rad": 0.0,
        "thrust_saturation_steps": 0,
        "moment_saturation_steps": 0,
        "success_6m_raw": False,
        "success_5m_raw": False,
        "contact_0p3m": False,
        "collision_agent": False,
        "collision_static": False,
        "boundary_invalid_pursuer": False,
        "boundary_invalid_evader": False,
        "height_invalid_pursuer": False,
        "height_invalid_evader": False,
        "timeout": False,
    }


def classify_episode(row: dict[str, Any], long_loss_steps: int = 50) -> str:
    """Assign one mutually exclusive outcome without hiding unsafe captures."""
    ordered_failures = (
        ("collision_agent", "collision_agent"),
        ("collision_static", "collision_static"),
        ("contact_0p3m", "contact_target"),
        ("boundary_invalid_pursuer", "boundary_pursuer"),
        ("boundary_invalid_evader", "boundary_evader"),
        ("height_invalid_pursuer", "height_pursuer"),
        ("height_invalid_evader", "height_evader"),
    )
    for field_name, label in ordered_failures:
        if bool(row[field_name]):
            return label
    if bool(row["success_6m_raw"]):
        return "success_6m"
    saturated = (
        int(row["thrust_saturation_steps"]) > 0
        or int(row["moment_saturation_steps"]) > 0
    )
    if bool(row["timeout"]) and saturated:
        return "timeout_control_saturation"
    if (
        bool(row["timeout"])
        and int(row["max_consecutive_loss_steps"]) >= int(long_loss_steps)
    ):
        return "timeout_long_loss"
    if bool(row["timeout"]):
        return "timeout_strategy"
    return "unclassified_terminal"


@dataclass
class EpisodeLedger:
    """Accumulate synchronized vector-environment telemetry into exact quotas."""

    method: str
    base_seed: int
    num_envs: int
    total_episodes: int
    long_loss_steps: int = 50
    quotas: list[int] = field(init=False)
    completed_per_env: list[int] = field(init=False)
    rows: list[dict[str, Any]] = field(init=False, default_factory=list)
    _active: list[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        if self.method not in {"full", "plain"}:
            raise ValueError("method must be full or plain")
        if self.long_loss_steps <= 0:
            raise ValueError("long_loss_steps must be positive")
        self.quotas = episode_quotas(self.total_episodes, self.num_envs)
        self.completed_per_env = [0 for _ in range(self.num_envs)]
        self._active = [_new_accumulator() for _ in range(self.num_envs)]

    @property
    def complete(self) -> bool:
        return self.completed_per_env == self.quotas

    @property
    def completed(self) -> int:
        return len(self.rows)

    def needs_env(self, env_id: int) -> bool:
        return self.completed_per_env[env_id] < self.quotas[env_id]

    def current_episode_steps(self, env_id: int) -> int:
        if not 0 <= env_id < self.num_envs:
            raise IndexError(env_id)
        return int(self._active[env_id]["episode_steps"])

    def update(self, env_id: int, step: dict[str, Any]) -> dict[str, Any] | None:
        if not 0 <= env_id < self.num_envs:
            raise IndexError(env_id)
        if not self.needs_env(env_id):
            return None
        accumulator = self._active[env_id]
        episode_seed = int(step["episode_seed"])
        if accumulator["episode_seed"] is None:
            accumulator["episode_seed"] = episode_seed
        elif accumulator["episode_seed"] != episode_seed:
            raise RuntimeError("episode seed changed before a terminal step")

        accumulator["episode_steps"] += 1
        visible = bool(step["team_visible"])
        accumulator["visible_steps"] += int(visible)
        accumulator["lost_steps_total"] += int(not visible)
        accumulator["max_consecutive_loss_steps"] = max(
            accumulator["max_consecutive_loss_steps"], int(step["lost_steps"])
        )
        accumulator["recapture_count"] = max(
            accumulator["recapture_count"], int(step["recapture_count"])
        )
        phase = str(step["reacquisition_phase"])
        accumulator["lost_search_steps"] += int(phase == "lost_search")
        accumulator["visible_contraction_steps"] += int(
            phase == "visible_contraction"
        )
        accumulator["safety_corrected_steps"] += int(step["safety_corrected"])
        accumulator["safety_qp_attempt_steps"] += int(step["safety_qp_attempted"])
        accumulator["safety_fallback_steps"] += int(step["safety_fallback"])
        correction = float(step["safety_correction_norm"])
        accumulator["safety_correction_norm_sum"] += correction
        accumulator["safety_correction_norm_max"] = max(
            accumulator["safety_correction_norm_max"], correction
        )
        accumulator["min_predicted_physical_clearance_m"] = min(
            accumulator["min_predicted_physical_clearance_m"],
            float(step["predicted_physical_clearance_m"]),
        )
        accumulator["min_predicted_target_clearance_m"] = min(
            accumulator["min_predicted_target_clearance_m"],
            float(step.get("predicted_target_clearance_m", math.inf)),
        )
        for field_name in (
            "min_agent_clearance_m",
            "min_static_clearance_m",
            "min_pursuer_boundary_clearance_m",
        ):
            accumulator[field_name] = min(
                accumulator[field_name], float(step[field_name])
            )
        for field_name in (
            "max_speed_tracking_error_mps",
            "max_velocity_yaw_slip_rad",
        ):
            accumulator[field_name] = max(
                accumulator[field_name], float(step[field_name])
            )
        accumulator["thrust_saturation_steps"] += int(step["thrust_saturated"])
        accumulator["moment_saturation_steps"] += int(step["moment_saturated"])
        for field_name in (
            "success_6m_raw",
            "success_5m_raw",
            "contact_0p3m",
            "collision_agent",
            "collision_static",
            "boundary_invalid_pursuer",
            "boundary_invalid_evader",
            "height_invalid_pursuer",
            "height_invalid_evader",
            "timeout",
        ):
            accumulator[field_name] = bool(accumulator[field_name] or step[field_name])

        terminal = any(
            bool(step[field_name])
            for field_name in (
                "success_6m_raw",
                "contact_0p3m",
                "collision_agent",
                "collision_static",
                "boundary_invalid_pursuer",
                "boundary_invalid_evader",
                "height_invalid_pursuer",
                "height_invalid_evader",
                "timeout",
            )
        )
        if not terminal:
            return None

        episode_ordinal = self.completed_per_env[env_id]
        expected_seed = expected_episode_seed(self.base_seed, env_id, episode_ordinal)
        if int(accumulator["episode_seed"]) != expected_seed:
            raise RuntimeError(
                f"episode seed mismatch for env {env_id} episode {episode_ordinal}: "
                f"{accumulator['episode_seed']} != {expected_seed}"
            )
        row = dict(accumulator)
        row.update(
            {
                "method": self.method,
                "env_id": env_id,
                "episode_ordinal": episode_ordinal,
                "pair_key": f"{env_id}:{episode_ordinal}:{expected_seed}",
                "mean_safety_correction_norm": (
                    accumulator["safety_correction_norm_sum"]
                    / accumulator["episode_steps"]
                ),
            }
        )
        row["outcome"] = classify_episode(row, self.long_loss_steps)
        row["success_6m"] = int(row["outcome"] == "success_6m")
        row["success_5m"] = int(
            row["success_6m"] and bool(row["success_5m_raw"])
        )
        row["long_loss_5s"] = int(
            int(row["max_consecutive_loss_steps"]) >= self.long_loss_steps
        )
        row["short_loss_2s"] = int(
            int(row["max_consecutive_loss_steps"]) >= 20
        )
        del row["safety_correction_norm_sum"]
        self.rows.append(row)
        self.completed_per_env[env_id] += 1
        self._active[env_id] = _new_accumulator()
        return row


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [math.nan, math.nan]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def summarize_episodes(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    episode_rows = list(rows)
    if not episode_rows:
        raise ValueError("cannot summarize zero episodes")
    count = len(episode_rows)
    successes_6m = sum(int(row["success_6m"]) for row in episode_rows)
    successes_5m = sum(int(row["success_5m"]) for row in episode_rows)
    collision_count = sum(
        int(bool(row["collision_agent"]) or bool(row["collision_static"]))
        for row in episode_rows
    )
    outcome_counts = Counter(str(row["outcome"]) for row in episode_rows)
    return {
        "episodes": count,
        "success_6m": successes_6m / count,
        "success_6m_wilson95": wilson_interval(successes_6m, count),
        "success_5m": successes_5m / count,
        "success_5m_wilson95": wilson_interval(successes_5m, count),
        "collision": collision_count / count,
        "timeout": sum(int(bool(row["timeout"])) for row in episode_rows) / count,
        "long_loss_5s": sum(int(row["long_loss_5s"]) for row in episode_rows) / count,
        "short_loss_2s": sum(int(row["short_loss_2s"]) for row in episode_rows) / count,
        "mean_episode_steps": float(
            np.mean([int(row["episode_steps"]) for row in episode_rows])
        ),
        "maximum_speed_tracking_error_mps": max(
            float(row["max_speed_tracking_error_mps"]) for row in episode_rows
        ),
        "minimum_predicted_target_clearance_m": min(
            float(row.get("min_predicted_target_clearance_m", math.inf))
            for row in episode_rows
        ),
        "thrust_saturation_steps": sum(
            int(row["thrust_saturation_steps"]) for row in episode_rows
        ),
        "moment_saturation_steps": sum(
            int(row["moment_saturation_steps"]) for row in episode_rows
        ),
        "outcome_counts": dict(sorted(outcome_counts.items())),
    }


def matched_rows(
    full_rows: Iterable[dict[str, Any]], plain_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    full_list = list(full_rows)
    plain_list = list(plain_rows)
    full = {str(row["pair_key"]): row for row in full_list}
    plain = {str(row["pair_key"]): row for row in plain_list}
    if len(full) != len(full_list) or len(plain) != len(plain_list):
        raise ValueError("duplicate pair_key within a method")
    if set(full) != set(plain):
        missing_plain = sorted(set(full) - set(plain))
        missing_full = sorted(set(plain) - set(full))
        raise ValueError(
            f"matched seed sets differ: missing_plain={missing_plain[:3]}, "
            f"missing_full={missing_full[:3]}"
        )
    result = []
    for key in sorted(full, key=lambda value: tuple(int(part) for part in value.split(":"))):
        full_row = full[key]
        plain_row = plain[key]
        full_success = int(full_row["success_6m"])
        plain_success = int(plain_row["success_6m"])
        result.append(
            {
                "pair_key": key,
                "env_id": int(full_row["env_id"]),
                "episode_ordinal": int(full_row["episode_ordinal"]),
                "episode_seed": int(full_row["episode_seed"]),
                "full_success_6m": full_success,
                "plain_success_6m": plain_success,
                "success_difference": full_success - plain_success,
                "transition": (
                    "both_success" if full_success and plain_success
                    else "plain_to_full" if full_success
                    else "full_to_plain" if plain_success
                    else "both_failure"
                ),
                "full_outcome": str(full_row["outcome"]),
                "plain_outcome": str(plain_row["outcome"]),
            }
        )
    return result


def paired_bootstrap_interval(
    pairs: Iterable[dict[str, Any]], *, seed: int = 20260824, draws: int = 20_000
) -> list[float]:
    paired = list(pairs)
    if not paired:
        raise ValueError("cannot bootstrap zero pairs")
    if draws <= 0:
        raise ValueError("draws must be positive")
    differences = np.asarray(
        [float(row["success_difference"]) for row in paired], dtype=float
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(differences), size=(draws, len(differences)))
    estimates = differences[indices].mean(axis=1)
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]
