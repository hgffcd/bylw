"""Frozen P5-0A task and timing contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimingSpec:
    """Separate physics, low-level-control and high-level-policy rates."""

    physics_hz: int = 100
    low_level_control_hz: int = 100
    high_level_policy_hz: int = 10

    def validate(self) -> None:
        rates = (
            self.physics_hz,
            self.low_level_control_hz,
            self.high_level_policy_hz,
        )
        if any(rate <= 0 for rate in rates):
            raise ValueError("all timing rates must be positive")
        if self.physics_hz % self.low_level_control_hz:
            raise ValueError("physics_hz must be divisible by low_level_control_hz")
        if self.physics_hz % self.high_level_policy_hz:
            raise ValueError("physics_hz must be divisible by high_level_policy_hz")
        if self.low_level_control_hz != self.physics_hz:
            raise ValueError(
                "P5 v1 requires the low-level controller at every physics substep"
            )

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def high_level_dt(self) -> float:
        return 1.0 / self.high_level_policy_hz

    @property
    def decimation(self) -> int:
        self.validate()
        return self.physics_hz // self.high_level_policy_hz


@dataclass(frozen=True)
class P5TaskSpec:
    """Method invariants transferred from the frozen 2-D task."""

    timing: TimingSpec = field(default_factory=TimingSpec)
    n_pursuers: int = 3
    arena_size_m: float = 100.0
    fixed_altitude_m: float = 3.0
    n_static_obstacles: int = 8
    n_dynamic_obstacles: int = 0
    sense_range_m: float = 25.0
    fov_half_rad: float = math.pi / 3.0
    visual_occlusion: bool = True
    capture_radius_main_m: float = 6.0
    capture_radius_strict_m: float = 5.0
    min_capture_agents: int = 3
    max_capture_gap_rad: float = math.pi
    contact_radius_m: float = 0.3
    pursuer_radius_m: float = 0.5
    v_max_mps: float = 8.0
    a_max_mps2: float = 4.0
    yaw_rate_max_rps: float = 1.5
    k_nearest_obstacles: int = 4
    lost_cap_steps: int = 100
    obs_mode: str = "last"
    team_target_sharing: bool = False
    # The original P5 contract is frozen to three pursuers.  P5-C opts in to
    # a separate variable-scale protocol without changing the default path.
    allow_variable_scale: bool = False

    def validate(self) -> None:
        self.timing.validate()
        if self.n_pursuers < 3:
            raise ValueError("P5 requires at least three pursuers")
        if not self.allow_variable_scale and self.n_pursuers != 3:
            raise ValueError("P5 v1 is frozen to exactly three pursuers")
        if self.n_dynamic_obstacles != 0:
            raise ValueError("P5-0A/P5-0B must start without dynamic obstacles")
        if not 3 <= self.min_capture_agents <= self.n_pursuers:
            raise ValueError("min_capture_agents must be in [3, n_pursuers]")
        if not self.allow_variable_scale and self.min_capture_agents != self.n_pursuers:
            raise ValueError("P5 v1 uses the frozen 3-of-3 encirclement criterion")
        if not 0 < self.capture_radius_strict_m < self.capture_radius_main_m:
            raise ValueError("strict capture radius must be below the main radius")
        if self.contact_radius_m >= self.capture_radius_strict_m:
            raise ValueError("contact is telemetry, not the encirclement criterion")
        if not 0 < self.fov_half_rad <= math.pi:
            raise ValueError("fov_half_rad must be in (0, pi]")
        if self.obs_mode != "last":
            raise ValueError("P5 v1 checkpoint contract is frozen to Last observations")
        if self.lost_cap_steps <= 0 or self.k_nearest_obstacles <= 0:
            raise ValueError("lost_cap_steps and k_nearest_obstacles must be positive")

    @property
    def observation_dim(self) -> int:
        return 5 + 6 + 4 * (self.n_pursuers - 1) + 5 * self.k_nearest_obstacles

    @property
    def centralized_state_dim(self) -> int:
        return 5 * self.n_pursuers + 4 + 1 + 3 * self.k_nearest_obstacles

    @classmethod
    def from_mapping(cls, values: dict) -> "P5TaskSpec":
        task_values = dict(values)
        timing_values = dict(task_values.pop("timing", {}))
        spec = cls(timing=TimingSpec(**timing_values), **task_values)
        spec.validate()
        return spec
