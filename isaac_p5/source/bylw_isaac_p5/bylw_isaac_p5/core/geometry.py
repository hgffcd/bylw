"""Batched fixed-height visibility and encirclement geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .action_adapter import wrap_angle


TEAM_STATUS = {
    0: "visible",
    1: "pure_range",
    2: "pure_fov",
    3: "pure_occlusion",
    4: "mixed",
}


@dataclass
class VisibilityResult:
    visible_each: torch.Tensor
    range_blocked_each: torch.Tensor
    fov_blocked_each: torch.Tensor
    occluded_each: torch.Tensor
    reason_bits_each: torch.Tensor
    team_visible: torch.Tensor
    team_status_code: torch.Tensor


@dataclass
class CaptureResult:
    success: torch.Tensor
    inside_count: torch.Tensor
    max_gap_rad: torch.Tensor
    min_distance_m: torch.Tensor


@dataclass
class StrongRingResult:
    satisfied: torch.Tensor
    inside_count: torch.Tensor
    max_gap_rad: torch.Tensor
    mean_gap_error_rad: torch.Tensor
    min_distance_m: torch.Tensor
    max_distance_m: torch.Tensor


def visibility_masks(
    pursuer_position_w: torch.Tensor,
    pursuer_yaw_rad: torch.Tensor,
    target_position_w: torch.Tensor,
    obstacle_xy_w: torch.Tensor,
    obstacle_radius_m: torch.Tensor,
    obstacle_height_m: torch.Tensor,
    sense_range_m: float,
    fov_half_rad: float,
    visual_occlusion: bool = True,
) -> VisibilityResult:
    """Return independent distance/FOV/cylinder-occlusion causes.

    Shapes are `[batch, pursuer, 3]`, `[batch, pursuer]`, `[batch, 3]`,
    `[batch, obstacle, 2]`, and `[batch, obstacle]` for the last two obstacle
    parameters. Range remains horizontal to preserve the frozen 2-D task.
    """
    if pursuer_position_w.ndim != 3 or pursuer_position_w.shape[-1] != 3:
        raise ValueError("pursuer_position_w must have shape [batch, pursuer, 3]")
    batch, agents, _ = pursuer_position_w.shape
    if pursuer_yaw_rad.shape != (batch, agents):
        raise ValueError("pursuer_yaw_rad shape mismatch")
    if target_position_w.shape != (batch, 3):
        raise ValueError("target_position_w shape mismatch")
    if obstacle_xy_w.ndim != 3 or obstacle_xy_w.shape[0] != batch or obstacle_xy_w.shape[-1] != 2:
        raise ValueError("obstacle_xy_w must have shape [batch, obstacle, 2]")
    if obstacle_radius_m.shape != obstacle_xy_w.shape[:2] or obstacle_height_m.shape != obstacle_xy_w.shape[:2]:
        raise ValueError("obstacle radius/height shapes must match obstacle_xy_w")

    relative_xy = target_position_w[:, None, :2] - pursuer_position_w[..., :2]
    distance = relative_xy.norm(dim=-1)
    range_blocked = distance > sense_range_m
    bearing = torch.atan2(relative_xy[..., 1], relative_xy[..., 0])
    fov_blocked = wrap_angle(bearing - pursuer_yaw_rad).abs() > fov_half_rad

    obstacle_count = obstacle_xy_w.shape[1]
    if not visual_occlusion or obstacle_count == 0:
        occluded = torch.zeros_like(range_blocked)
    else:
        start_xy = pursuer_position_w[..., :2].unsqueeze(2)
        line_xy = relative_xy.unsqueeze(2)
        center_offset = obstacle_xy_w[:, None, :, :] - start_xy
        denominator = (line_xy * line_xy).sum(dim=-1).clamp_min(1e-12)
        projection = (center_offset * line_xy).sum(dim=-1) / denominator
        projection = projection.clamp(0.0, 1.0)
        nearest = start_xy + projection.unsqueeze(-1) * line_xy
        radial_hit = (
            nearest - obstacle_xy_w[:, None, :, :]
        ).norm(dim=-1) <= obstacle_radius_m[:, None, :]
        line_z = (
            pursuer_position_w[..., 2].unsqueeze(2)
            + projection
            * (target_position_w[:, None, 2] - pursuer_position_w[..., 2]).unsqueeze(2)
        )
        height_hit = (line_z >= 0.0) & (line_z <= obstacle_height_m[:, None, :])
        occluded = (radial_hit & height_hit).any(dim=-1)

    reason_bits = (
        range_blocked.to(torch.int64)
        + 2 * fov_blocked.to(torch.int64)
        + 4 * occluded.to(torch.int64)
    )
    visible_each = reason_bits == 0
    team_visible = visible_each.any(dim=1)
    team_code = torch.full((batch,), 4, dtype=torch.int64, device=pursuer_position_w.device)
    team_code[team_visible] = 0
    for bits, code in ((1, 1), (2, 2), (4, 3)):
        team_code[(reason_bits == bits).all(dim=1)] = code
    return VisibilityResult(
        visible_each=visible_each,
        range_blocked_each=range_blocked,
        fov_blocked_each=fov_blocked,
        occluded_each=occluded,
        reason_bits_each=reason_bits,
        team_visible=team_visible,
        team_status_code=team_code,
    )


def encirclement_capture(
    pursuer_xy_w: torch.Tensor,
    target_xy_w: torch.Tensor,
    capture_radius_m: float,
    min_capture_agents: int,
    max_gap_rad: float,
) -> CaptureResult:
    """Apply the frozen distance-plus-angular-gap encirclement criterion."""
    if pursuer_xy_w.ndim != 3 or pursuer_xy_w.shape[-1] != 2:
        raise ValueError("pursuer_xy_w must have shape [batch, pursuer, 2]")
    if target_xy_w.shape != (pursuer_xy_w.shape[0], 2):
        raise ValueError("target_xy_w shape mismatch")
    if not 1 <= min_capture_agents <= pursuer_xy_w.shape[1]:
        raise ValueError("invalid min_capture_agents")
    relative = pursuer_xy_w - target_xy_w[:, None, :]
    distance = relative.norm(dim=-1)
    inside = distance <= capture_radius_m
    inside_count = inside.sum(dim=-1)
    max_gaps = []
    for batch_index in range(pursuer_xy_w.shape[0]):
        angles = torch.atan2(relative[batch_index, :, 1], relative[batch_index, :, 0])[
            inside[batch_index]
        ]
        if angles.numel() <= 1:
            max_gaps.append(torch.full((), 2.0 * math.pi, device=distance.device, dtype=distance.dtype))
            continue
        angles = torch.sort(angles).values
        gaps = angles[1:] - angles[:-1]
        wrap_gap = 2.0 * math.pi - (angles[-1] - angles[0])
        max_gaps.append(torch.cat((gaps, wrap_gap.unsqueeze(0))).max())
    max_gap = torch.stack(max_gaps)
    success = (inside_count >= min_capture_agents) & (max_gap <= max_gap_rad)
    return CaptureResult(
        success=success,
        inside_count=inside_count,
        max_gap_rad=max_gap,
        min_distance_m=distance.min(dim=-1).values,
    )


def strong_ring_geometry(
    pursuer_xy_w: torch.Tensor,
    target_xy_w: torch.Tensor,
    minimum_radius_m: float,
    maximum_radius_m: float,
    maximum_gap_rad: float,
    maximum_mean_gap_error_rad: float,
) -> StrongRingResult:
    """Return a strict radial-band and near-uniform angular enclosure test."""
    if pursuer_xy_w.ndim != 3 or pursuer_xy_w.shape[-1] != 2:
        raise ValueError("pursuer_xy_w must have shape [batch, pursuer, 2]")
    if target_xy_w.shape != (pursuer_xy_w.shape[0], 2):
        raise ValueError("target_xy_w shape mismatch")
    if pursuer_xy_w.shape[1] < 3:
        raise ValueError("strong ring requires at least three pursuers")
    if not 0.0 < minimum_radius_m < maximum_radius_m:
        raise ValueError("strong ring radii must satisfy 0 < minimum < maximum")
    if maximum_gap_rad <= 0.0 or maximum_mean_gap_error_rad < 0.0:
        raise ValueError("strong ring angular thresholds are invalid")
    relative = pursuer_xy_w - target_xy_w[:, None, :]
    distance = relative.norm(dim=-1)
    radial_inside = (distance >= minimum_radius_m) & (
        distance <= maximum_radius_m
    )
    angles = torch.sort(
        torch.remainder(
            torch.atan2(relative[..., 1], relative[..., 0]), 2.0 * math.pi
        ),
        dim=1,
    ).values
    gaps = torch.cat(
        (
            angles[:, 1:] - angles[:, :-1],
            2.0 * math.pi - angles[:, -1:] + angles[:, :1],
        ),
        dim=1,
    )
    desired_gap = 2.0 * math.pi / pursuer_xy_w.shape[1]
    max_gap = gaps.max(dim=1).values
    mean_gap_error = (gaps - desired_gap).abs().mean(dim=1)
    satisfied = (
        radial_inside.all(dim=1)
        & (max_gap <= maximum_gap_rad)
        & (mean_gap_error <= maximum_mean_gap_error_rad)
    )
    return StrongRingResult(
        satisfied=satisfied,
        inside_count=radial_inside.sum(dim=1),
        max_gap_rad=max_gap,
        mean_gap_error_rad=mean_gap_error,
        min_distance_m=distance.min(dim=1).values,
        max_distance_m=distance.max(dim=1).values,
    )
