"""Frozen 2-D actor action to fixed-height flight-reference adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import P5TaskSpec


@dataclass
class HighLevelCommand:
    desired_velocity_xy_w: torch.Tensor
    desired_altitude_m: torch.Tensor
    desired_yaw_rate_rps: torch.Tensor
    desired_speed_mps: torch.Tensor
    normalized_action: torch.Tensor


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class PlanarCommandAdapter:
    """Replicate the 2-D unicycle action scaling without changing the actor."""

    def __init__(self, spec: P5TaskSpec):
        spec.validate()
        self.spec = spec
        self._speed_command: torch.Tensor | None = None

    def reset(self, planar_speed_mps: torch.Tensor) -> None:
        self._speed_command = planar_speed_mps.detach().clone().clamp(
            0.0, self.spec.v_max_mps
        )

    def reset_indices(self, indices: torch.Tensor) -> None:
        if self._speed_command is not None:
            self._speed_command[indices] = 0.0

    def step(
        self,
        normalized_action: torch.Tensor,
        current_yaw_rad: torch.Tensor,
        current_speed_mps: torch.Tensor | None = None,
    ) -> HighLevelCommand:
        if normalized_action.shape[-1] != 2:
            raise ValueError("frozen actor action must end with [acceleration, yaw_rate]")
        if normalized_action.shape[:-1] != current_yaw_rad.shape:
            raise ValueError("action batch dimensions must match current_yaw_rad")
        if current_speed_mps is not None:
            if current_speed_mps.shape != current_yaw_rad.shape:
                raise ValueError("current_speed_mps must match current_yaw_rad")
            # The frozen actor observes physical speed, and dynamic_v2 also
            # predicts from physical speed.  Rebase the command integrator on
            # that same state so [acceleration, yaw_rate] keeps its 2-D
            # semantics despite low-level tracking lag.
            self.reset(current_speed_mps)
        elif self._speed_command is None or self._speed_command.shape != current_yaw_rad.shape:
            self._speed_command = torch.zeros_like(current_yaw_rad)

        action = normalized_action.clamp(-1.0, 1.0)
        acceleration = action[..., 0] * self.spec.a_max_mps2
        yaw_rate = action[..., 1] * self.spec.yaw_rate_max_rps
        self._speed_command = (
            self._speed_command + acceleration * self.spec.timing.high_level_dt
        ).clamp(0.0, self.spec.v_max_mps)

        # The 2-D environment updates yaw before position within the same step.
        yaw_for_velocity = wrap_angle(
            current_yaw_rad + yaw_rate * self.spec.timing.high_level_dt
        )
        desired_velocity = torch.stack(
            (
                self._speed_command * torch.cos(yaw_for_velocity),
                self._speed_command * torch.sin(yaw_for_velocity),
            ),
            dim=-1,
        )
        altitude = torch.full_like(current_yaw_rad, self.spec.fixed_altitude_m)
        return HighLevelCommand(
            desired_velocity_xy_w=desired_velocity,
            desired_altitude_m=altitude,
            desired_yaw_rate_rps=yaw_rate,
            desired_speed_mps=self._speed_command.clone(),
            normalized_action=action,
        )
