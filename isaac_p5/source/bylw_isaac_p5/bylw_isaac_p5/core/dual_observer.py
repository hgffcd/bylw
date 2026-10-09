"""C98 measured observer-pair reference; no target-truth access during loss."""
from dataclasses import dataclass
import math
import torch


@dataclass
class DualObserverReference:
    heading_rad: torch.Tensor
    blocker_index: torch.Tensor
    valid: torch.Tensor
    pair_angle_rad: torch.Tensor
    cost: torch.Tensor


def segment_clearance(start, end, obstacle_xy, obstacle_radius):
    """Minimum planar segment-to-obstacle surface clearance, [batch, candidate]."""
    if obstacle_xy.shape[1] == 0:
        return torch.full(start.shape[:2], torch.inf, device=start.device, dtype=start.dtype)
    line = end - start
    offset = obstacle_xy[:, None] - start[:, :, None]
    t = ((offset * line[:, :, None]).sum(-1)
         / line.square().sum(-1)[:, :, None].clamp_min(1e-8)).clamp(0, 1)
    closest = start[:, :, None] + t[..., None] * line[:, :, None]
    return ((closest - obstacle_xy[:, None]).norm(dim=-1)
            - obstacle_radius[:, None]).min(dim=-1).values


def dual_observer_reference(pursuer_xy, pursuer_velocity_xy, target_estimate_xy,
                            measured_target_velocity_xy, visible_each, pursuer_yaw,
                            reference_radius, previous_blocker, obstacle_xy,
                            obstacle_radius, *, maximum_speed_mps=4.0,
                            fov_half_rad=math.pi / 3, hysteresis_m=1.0,
                            keep_valid_pair=False):
    """Select two currently visible observers and an affordable opposite role.

    The opposite of their angular bisector anchors the third role. Pair choice
    considers angular coverage, current FoV margin, predicted LOS, a straight
    path cost and a bounded-CV interception deficit. These are ranking costs,
    NOT a dynamics-feasibility or continuous-visibility guarantee. Invalid
    pairs fall back to the caller's original one-observer/search controller.
    """
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[1:] != (3, 2):
        raise ValueError("dual observer reference requires exactly three XY agents")
    batch = pursuer_xy.shape[0]
    if (pursuer_velocity_xy.shape != pursuer_xy.shape
            or target_estimate_xy.shape != (batch, 2)
            or measured_target_velocity_xy.shape != (batch, 2)
            or visible_each.shape != (batch, 3) or visible_each.dtype != torch.bool
            or pursuer_yaw.shape != (batch, 3)
            or reference_radius.shape != (batch,)
            or previous_blocker.shape != (batch,)):
        raise ValueError("dual observer input shape or visibility dtype mismatch")
    if maximum_speed_mps <= 0 or fov_half_rad <= 0 or hysteresis_m < 0:
        raise ValueError("invalid dual observer control parameters")
    pair = torch.tensor([[1, 2], [2, 0], [0, 1]], device=pursuer_xy.device)
    relative = pursuer_xy - target_estimate_xy[:, None]
    radius = relative.norm(dim=-1)
    unit = relative / radius[..., None].clamp_min(1e-6)
    observer_unit = unit[:, pair]
    opposite = -observer_unit.sum(dim=2)
    opposite_norm = opposite.norm(dim=-1)
    heading = torch.atan2(opposite[..., 1], opposite[..., 0])
    direction = opposite / opposite_norm[..., None].clamp_min(1e-6)
    pair_angle = torch.acos((observer_unit[:, :, 0] * observer_unit[:, :, 1]).sum(-1).clamp(-1, 1))
    valid = visible_each[:, pair].all(dim=-1) & (opposite_norm > .25)
    finite = (torch.isfinite(pursuer_xy).all(dim=(1, 2))
              & torch.isfinite(pursuer_velocity_xy).all(dim=(1, 2))
              & torch.isfinite(target_estimate_xy).all(dim=1)
              & torch.isfinite(measured_target_velocity_xy).all(dim=1)
              & torch.isfinite(pursuer_yaw).all(dim=1)
              & torch.isfinite(reference_radius) & (radius.min(dim=1).values > 1e-6)
              & (reference_radius > 0))
    valid &= finite[:, None]
    goals = target_estimate_xy[:, None] + reference_radius[:, None, None] * direction
    travel = (goals - pursuer_xy).norm(dim=-1)
    horizon = torch.linspace(.25, 8., 33, device=pursuer_xy.device, dtype=pursuer_xy.dtype)
    displacement = (goals[:, :, None] + horizon[None, None, :, None]
                    * measured_target_velocity_xy[:, None, None] - pursuer_xy[:, :, None])
    reach_deficit = (displacement.norm(dim=-1) - maximum_speed_mps * horizon).min(-1).values.clamp_min(0)
    angle_violation = ((math.pi / 3 - pair_angle).clamp_min(0)
                       + (pair_angle - 5 * math.pi / 6).clamp_min(0))
    angle_cost = reference_radius[:, None] * (
        angle_violation + .25 * (pair_angle - 2 * math.pi / 3).abs())
    bearing = torch.atan2(-relative[..., 1], -relative[..., 0])
    yaw_error = torch.atan2(torch.sin(bearing - pursuer_yaw), torch.cos(bearing - pursuer_yaw))
    fov_margin = (1 - yaw_error.abs() / fov_half_rad).clamp(0, 1)
    future_start = pursuer_xy + .5 * pursuer_velocity_xy
    future_target = (target_estimate_xy + .5 * measured_target_velocity_xy)[:, None].expand_as(pursuer_xy)
    los = segment_clearance(future_start, future_target, obstacle_xy, obstacle_radius)
    los_cost = 4 * (.5 - los[:, pair]).clamp_min(0).sum(dim=-1)
    path = segment_clearance(pursuer_xy, goals, obstacle_xy, obstacle_radius)
    cost = (travel + 2 * reach_deficit + angle_cost + los_cost
            + 8 * (1. - path).clamp_min(0) - .5 * fov_margin[:, pair].sum(dim=-1))
    cost = torch.where(valid, torch.nan_to_num(cost, nan=torch.inf), torch.inf)
    best = cost.argmin(dim=1)
    previous_safe = previous_blocker.clamp(0, 2)
    previous_cost = cost.gather(1, previous_safe[:, None]).squeeze(1)
    best_cost = cost.gather(1, best[:, None]).squeeze(1)
    keep = ((previous_blocker >= 0) & (previous_blocker < 3)
            & torch.isfinite(previous_cost)
            & ((best_cost + hysteresis_m >= previous_cost) | bool(keep_valid_pair)))
    selected = torch.where(keep, previous_safe, best)
    selected_valid = valid.gather(1, selected[:, None]).squeeze(1)
    selected_heading = heading.gather(1, selected[:, None]).squeeze(1)
    return DualObserverReference(
        torch.where(selected_valid, selected_heading, torch.zeros_like(selected_heading)),
        torch.where(selected_valid, selected, torch.full_like(selected, -1)),
        selected_valid,
        torch.where(selected_valid, pair_angle.gather(1, selected[:, None]).squeeze(1), 0.),
        cost.gather(1, selected[:, None]).squeeze(1),
    )
