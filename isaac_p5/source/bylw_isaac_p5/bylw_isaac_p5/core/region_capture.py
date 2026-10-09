"""C97 region enclosure protocol, separate from historical stable-ring scoring.

The region is a task goal; the inner radius is the existing target safety
buffer, not a formation radius. All functions can be audited without Isaac.
"""
from __future__ import annotations

import math
import torch


def target_buffer_failure(pursuer_xy, target_xy, minimum_distance_m=1.5):
    """Actual sampled center-distance violation; distinct from contact.

    Equality is allowed, matching region_capture. Called after physics and
    before rewards/reset; this does not certify the intervening 100 Hz path.
    """
    distance = (pursuer_xy - target_xy[:, None, :]).norm(dim=-1)
    return (distance < minimum_distance_m).any(dim=-1)

from .geometry import CaptureResult, encirclement_capture
from .native_training import circular_gap_statistics

REGION_PROTOCOL = "region_v1_r5p5_gap150_first_hit"
REGION_REWARD = "region_v1_terminal_potential"
REGION_OBSERVATION_DIM = 64
REGION_STATE_DIM = 62


def configure_region_capture(cfg, *, align_objective: bool) -> None:
    """Apply the explicit protocol after C91 configuration; preserve safety."""
    cfg.region_capture_enabled = True
    cfg.is_finite_horizon = True
    cfg.state_space = REGION_STATE_DIM
    cfg.region_capture_radius_m = 5.5
    cfg.region_capture_maximum_gap_rad = math.radians(150)
    cfg.region_capture_minimum_target_distance_m = (
        cfg.predictive_target_minimum_center_distance_m
    )
    cfg.training_capture_radius_m = cfg.region_capture_radius_m
    cfg.region_objective_enabled = bool(align_objective)
    if not align_objective:
        return
    # Keep legacy strong-ring thresholds untouched for separately named
    # diagnostics. Only execution constraints use the physical inner buffer.
    cfg.region_control_minimum_radius_m = (
        cfg.region_capture_minimum_target_distance_m
    )
    cfg.predictive_ring_activation_requires_inside_inner_boundary = False
    cfg.predictive_ring_minimum_radius_after_strong_entry = False
    cfg.predictive_ring_common_acceleration_feedforward_after_strong_entry = False
    cfg.predictive_ring_activation_requires_strong_ring_entry = False
    cfg.strong_ring_confirmation_stabilization_enabled = False
    cfg.strong_ring_confirmation_maximum_phase_gain_enabled = False
    cfg.moving_target_slot_phase_angular_velocity_damping_confirmation_only = False


def remaining_task_time(episode_steps, maximum_episode_length):
    if maximum_episode_length <= 1:
        raise ValueError("finite task horizon must contain at least two steps")
    # Match the existing timeout at max_episode_length - 1 exactly.
    return (1.0 - episode_steps.float() / (maximum_episode_length - 1)).clamp(0, 1)


def region_critic_context(remaining_time, reference_radius, radius_scale,
                          unlocked, alignment_steps, required_alignment_steps):
    if radius_scale <= 0 or required_alignment_steps <= 0:
        raise ValueError("invalid region controller context scales")
    return torch.stack((remaining_time, reference_radius / radius_scale,
                        unlocked.float(),
                        (alignment_steps.float() / required_alignment_steps).clamp(0, 1).mean(dim=1)), dim=-1)


def expand_region_critic_state(critic_state, new_state_dim=REGION_STATE_DIM):
    """Zero-pad the critic first layer; preserve its pre-training value map."""
    copied = {key: value.detach().clone() for key, value in critic_state.items()}
    weight = copied["v.0.weight"]
    if weight.shape[1] > new_state_dim:
        raise ValueError("critic state may not shrink")
    extended = weight.new_zeros(weight.shape[0], new_state_dim)
    extended[:, :weight.shape[1]] = weight
    copied["v.0.weight"] = extended
    return copied


def region_capture(
    pursuer_xy, target_xy, visible_each, unlocked, unsafe, *,
    radius_m=5.5, maximum_gap_rad=math.radians(150),
    minimum_target_distance_m=1.5,
) -> CaptureResult:
    if not 0 < minimum_target_distance_m < radius_m:
        raise ValueError("physical safety radius must be positive and below capture radius")
    if not 0 < maximum_gap_rad < math.pi:
        raise ValueError("region enclosure requires a maximum gap strictly below pi")
    result = encirclement_capture(
        pursuer_xy, target_xy, radius_m, pursuer_xy.shape[1], maximum_gap_rad
    )
    finite = torch.isfinite(pursuer_xy).all(dim=(1, 2)) & torch.isfinite(target_xy).all(dim=1)
    result.success = (
        result.success & visible_each.all(dim=1) & unlocked & ~unsafe & finite
        & (result.min_distance_m >= minimum_target_distance_m)
    )
    return result


def region_potential(pursuer_xy, target_xy, *, radius_m=5.5,
                     maximum_gap_rad=math.radians(150)):
    """Bounded potential for the worst radial and angular task violations.

Zero everywhere inside the legal outer region with sufficient angle coverage;
no penalty for safe radii below 4 m or for non-uniform but valid angles.
The physical safety constraints are scored separately.
"""
    if radius_m <= 0 or not 0 < maximum_gap_rad < math.pi:
        raise ValueError("invalid region potential thresholds")
    radius = (pursuer_xy - target_xy[:, None]).norm(dim=-1).max(dim=1).values
    gap, _ = circular_gap_statistics(pursuer_xy, target_xy)
    radial_violation = ((radius - radius_m).clamp_min(0) / radius_m).clamp_max(4)
    angular_violation = (gap - maximum_gap_rad).clamp_min(0) / (2 * math.pi - maximum_gap_rad)
    return -(radial_violation + angular_violation)


def region_task_reward(previous_potential, potential, success, unsafe, timeout,
                       *, safety_cost, height_cost, action_cost, gamma=0.99):
    """Finite 30 s task: every terminal transition has absorbing potential 0.

Discounted shaping sums to -Phi(initial), independent of path or duration.
Safety failures take precedence over simultaneous geometric success.
"""
    if not 0 < gamma <= 1:
        raise ValueError("invalid shaping discount")
    safe_success = success & ~unsafe
    terminal = safe_success | unsafe | timeout
    next_phi = torch.where(terminal, torch.zeros_like(potential), potential)
    shaping = gamma * next_phi - previous_potential
    reward = (
        shaping - 0.01 + 5.0 * safe_success.float() - 5.0 * unsafe.float()
        - safety_cost - height_cost - action_cost
    )
    return reward, shaping
