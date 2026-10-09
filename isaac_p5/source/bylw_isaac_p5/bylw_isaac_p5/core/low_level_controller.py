"""Small fixed-height velocity/rate controller for P5 simulator smoke tests.

This controller emits body-frame collective thrust and moments. It is a
deterministic adapter to validate the frozen high-level policy, not a claimed
new learning contribution. Gains must be accepted by P5-0B before evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .action_adapter import HighLevelCommand, wrap_angle


@dataclass(frozen=True)
class LowLevelControllerSpec:
    gravity_mps2: float = 9.81
    velocity_kp: float = 2.0
    height_kp: float = 5.0
    vertical_velocity_kd: float = 3.0
    attitude_kp: float = 5.0
    rate_moment_gain: float = 0.002
    max_planar_accel_mps2: float = 4.0
    max_vertical_accel_mps2: float = 5.0
    max_tilt_rad: float = 0.55
    max_body_rate_rps: float = 6.0
    max_moment_nm: float = 0.01
    thrust_to_weight_max: float = 1.9


@dataclass
class LowLevelCommand:
    thrust_b: torch.Tensor
    moment_b: torch.Tensor
    desired_body_rate_rps: torch.Tensor
    desired_roll_pitch_rad: torch.Tensor
    thrust_saturated: torch.Tensor
    moment_saturated: torch.Tensor


def quaternion_to_euler_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert Isaac's scalar-first quaternion convention to XYZ Euler angles."""
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must use (..., 4) scalar-first layout")
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack((roll, pitch, yaw), dim=-1)


class FixedHeightRateController:
    def __init__(self, spec: LowLevelControllerSpec | None = None):
        self.spec = spec or LowLevelControllerSpec()

    @staticmethod
    def _clip_vector_norm(vector: torch.Tensor, maximum: float) -> torch.Tensor:
        scale = (maximum / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
        return vector * scale

    def compute(
        self,
        command: HighLevelCommand,
        position_w: torch.Tensor,
        quaternion_wxyz: torch.Tensor,
        linear_velocity_w: torch.Tensor,
        angular_velocity_b: torch.Tensor,
        mass_kg: torch.Tensor | float,
    ) -> LowLevelCommand:
        expected = (*command.desired_velocity_xy_w.shape[:-1], 3)
        for name, tensor in (
            ("position_w", position_w),
            ("linear_velocity_w", linear_velocity_w),
            ("angular_velocity_b", angular_velocity_b),
        ):
            if tensor.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if quaternion_wxyz.shape != (*expected[:-1], 4):
            raise ValueError("quaternion batch shape must match the command")

        cfg = self.spec
        euler = quaternion_to_euler_wxyz(quaternion_wxyz)
        roll, pitch, yaw = euler.unbind(dim=-1)
        velocity_error = command.desired_velocity_xy_w - linear_velocity_w[..., :2]
        planar_accel = self._clip_vector_norm(
            cfg.velocity_kp * velocity_error, cfg.max_planar_accel_mps2
        )
        height_error = command.desired_altitude_m - position_w[..., 2]
        vertical_accel = (
            cfg.height_kp * height_error
            - cfg.vertical_velocity_kd * linear_velocity_w[..., 2]
        ).clamp(-cfg.max_vertical_accel_mps2, cfg.max_vertical_accel_mps2)

        forward_accel = planar_accel[..., 0] * torch.cos(yaw) + planar_accel[..., 1] * torch.sin(yaw)
        left_accel = -planar_accel[..., 0] * torch.sin(yaw) + planar_accel[..., 1] * torch.cos(yaw)
        support_accel = (cfg.gravity_mps2 + vertical_accel).clamp_min(0.25 * cfg.gravity_mps2)
        pitch_des = torch.atan2(forward_accel, support_accel).clamp(-cfg.max_tilt_rad, cfg.max_tilt_rad)
        roll_des = torch.atan2(-left_accel, support_accel).clamp(-cfg.max_tilt_rad, cfg.max_tilt_rad)

        desired_rate = torch.stack(
            (
                cfg.attitude_kp * wrap_angle(roll_des - roll),
                cfg.attitude_kp * wrap_angle(pitch_des - pitch),
                command.desired_yaw_rate_rps,
            ),
            dim=-1,
        ).clamp(-cfg.max_body_rate_rps, cfg.max_body_rate_rps)
        moment_raw = cfg.rate_moment_gain * (desired_rate - angular_velocity_b)
        moment = moment_raw.clamp(-cfg.max_moment_nm, cfg.max_moment_nm)

        mass = torch.as_tensor(mass_kg, dtype=position_w.dtype, device=position_w.device)
        while mass.ndim < position_w.ndim - 1:
            mass = mass.unsqueeze(-1)
        tilt_cosine = (torch.cos(roll) * torch.cos(pitch)).clamp_min(0.25)
        thrust_raw = mass * support_accel / tilt_cosine
        maximum_thrust = mass * cfg.gravity_mps2 * cfg.thrust_to_weight_max
        thrust = thrust_raw.clamp(min=0.0)
        thrust = torch.minimum(thrust, maximum_thrust)
        thrust_b = torch.zeros_like(position_w)
        thrust_b[..., 2] = thrust
        return LowLevelCommand(
            thrust_b=thrust_b,
            moment_b=moment,
            desired_body_rate_rps=desired_rate,
            desired_roll_pitch_rad=torch.stack((roll_des, pitch_des), dim=-1),
            thrust_saturated=(thrust_raw < 0.0) | (thrust_raw > maximum_thrust),
            moment_saturated=(moment_raw.abs() > cfg.max_moment_nm).any(dim=-1),
        )
