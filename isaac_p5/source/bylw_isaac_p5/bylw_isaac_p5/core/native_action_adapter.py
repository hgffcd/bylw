"""Isaac-native high-level velocity command adapter.

Unlike the legacy two-action adapter, this adapter does not reproduce the
2-D unicycle dynamics.  The learned policy commands world-frame planar
velocity and yaw rate while the accepted 100 Hz low-level controller keeps
the vehicle at the configured altitude and emits collective thrust/moments.
"""

from __future__ import annotations

import torch

from .action_adapter import HighLevelCommand
from .config import P5TaskSpec


class WorldVelocityYawAdapter:
    """Map normalized ``[vx, vy, yaw_rate]`` actions to flight references."""

    action_dim = 3

    def __init__(self, spec: P5TaskSpec, maximum_speed_mps: float | None = None):
        spec.validate()
        self.spec = spec
        self.maximum_speed_mps = float(
            spec.v_max_mps if maximum_speed_mps is None else maximum_speed_mps
        )
        if not 0.0 < self.maximum_speed_mps <= spec.v_max_mps:
            raise ValueError("maximum_speed_mps must be in (0, spec.v_max_mps]")

    def step(self, normalized_action: torch.Tensor) -> HighLevelCommand:
        if normalized_action.shape[-1] != self.action_dim:
            raise ValueError(
                "native action must end with [vx_world, vy_world, yaw_rate]"
            )
        action = normalized_action.clamp(-1.0, 1.0)
        desired_velocity = action[..., :2] * self.maximum_speed_mps
        speed = desired_velocity.norm(dim=-1, keepdim=True)
        scale = (self.maximum_speed_mps / speed.clamp_min(1.0e-12)).clamp(max=1.0)
        desired_velocity = desired_velocity * scale
        desired_speed = desired_velocity.norm(dim=-1)
        yaw_rate = action[..., 2] * self.spec.yaw_rate_max_rps
        altitude = torch.full_like(yaw_rate, self.spec.fixed_altitude_m)
        return HighLevelCommand(
            desired_velocity_xy_w=desired_velocity,
            desired_altitude_m=altitude,
            desired_yaw_rate_rps=yaw_rate,
            desired_speed_mps=desired_speed,
            normalized_action=action,
        )
