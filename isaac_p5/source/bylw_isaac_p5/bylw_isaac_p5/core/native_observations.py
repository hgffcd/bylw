"""Isaac-native observations for dynamics-aware MAPPO training."""

from __future__ import annotations

import torch

from .config import P5TaskSpec


NATIVE_OBSERVATION_DIM = 53
NATIVE_ROLE_PHASE_FEATURE_DIM = 10
NATIVE_ROLE_PHASE_OBSERVATION_DIM = (
    NATIVE_OBSERVATION_DIM + NATIVE_ROLE_PHASE_FEATURE_DIM
)
NATIVE_STATE_DIM = 58


def build_role_phase_features(
    spec: P5TaskSpec,
    pursuer_xy: torch.Tensor,
    assigned_slot_xy: torch.Tensor,
    slot_assignment: torch.Tensor,
    slot_goal_valid: torch.Tensor,
    phase_index: torch.Tensor,
    reference_radius_m: torch.Tensor,
    team_ready_fraction: torch.Tensor,
    *,
    reference_radius_scale_m: float,
) -> torch.Tensor:
    """Build the C27 decentralized role/goal/phase observation suffix.

    Slot goals must already be expressed using the environment's observable
    target estimate (current shared measurement or Last during loss).  This
    helper deliberately has no target-truth input.
    """
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("pursuer_xy must have shape [batch, agent, 2]")
    batch, agents, _ = pursuer_xy.shape
    if assigned_slot_xy.shape != (batch, agents, 2):
        raise ValueError("assigned_slot_xy shape mismatch")
    if slot_assignment.shape != (batch, agents):
        raise ValueError("slot_assignment shape mismatch")
    for name, value in (
        ("slot_goal_valid", slot_goal_valid),
        ("phase_index", phase_index),
        ("reference_radius_m", reference_radius_m),
        ("team_ready_fraction", team_ready_fraction),
    ):
        if value.shape != (batch,):
            raise ValueError(f"{name} shape mismatch")
    if reference_radius_scale_m <= 0.0:
        raise ValueError("reference_radius_scale_m must be positive")
    if ((team_ready_fraction < 0.0) | (team_ready_fraction > 1.0)).any():
        raise ValueError("team_ready_fraction must be in [0, 1]")

    valid_assignment = (
        slot_goal_valid[:, None]
        & (slot_assignment >= 0)
        & (slot_assignment < agents)
    )
    relative_goal = (assigned_slot_xy - pursuer_xy) / spec.arena_size_m
    relative_goal = torch.where(
        valid_assignment[..., None], relative_goal, torch.zeros_like(relative_goal)
    )
    role_one_hot = torch.zeros(
        batch, agents, 3, dtype=pursuer_xy.dtype, device=pursuer_xy.device
    )
    safe_role = slot_assignment.clamp(0, 2)
    role_one_hot.scatter_(2, safe_role[..., None], 1.0)
    role_one_hot = role_one_hot * valid_assignment[..., None]

    valid_phase = (phase_index >= 0) & (phase_index < 3)
    phase_one_hot = torch.zeros(
        batch, 3, dtype=pursuer_xy.dtype, device=pursuer_xy.device
    )
    phase_one_hot.scatter_(1, phase_index.clamp(0, 2)[:, None], 1.0)
    phase_one_hot = phase_one_hot * valid_phase[:, None]
    phase_one_hot = phase_one_hot[:, None, :].expand(-1, agents, -1)
    normalized_radius = (
        reference_radius_m / reference_radius_scale_m
    ).clamp(0.0, 1.5)
    shared_tail = torch.stack(
        (normalized_radius, team_ready_fraction), dim=-1
    )[:, None, :].expand(-1, agents, -1)
    result = torch.cat(
        (relative_goal, role_one_hot, phase_one_hot, shared_tail), dim=-1
    )
    if result.shape != (batch, agents, NATIVE_ROLE_PHASE_FEATURE_DIM):
        raise RuntimeError("role/phase feature dimension drift")
    return result


def _normalize_position_delta(delta: torch.Tensor, spec: P5TaskSpec) -> torch.Tensor:
    result = delta.clone()
    result[..., :2] = result[..., :2] / spec.arena_size_m
    result[..., 2] = result[..., 2] / max(spec.fixed_altitude_m, 1.0)
    return result


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )


def build_native_observations(
    spec: P5TaskSpec,
    pursuer_position: torch.Tensor,
    pursuer_quaternion: torch.Tensor,
    pursuer_velocity: torch.Tensor,
    pursuer_angular_velocity: torch.Tensor,
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    visible_each: torch.Tensor,
    last_seen_target_xy: torch.Tensor,
    lost_steps: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radius: torch.Tensor,
    obstacle_velocity_xy: torch.Tensor,
    role_phase_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build decentralized observations containing real 3-D response state.

    The target truth is exposed only when the agent (or, when enabled, the
    team) has a valid measurement.  During loss, the XY component comes from
    ``last_seen_target_xy`` and the target velocity is zeroed.
    """

    batch, agents, xyz = pursuer_position.shape
    if agents != spec.n_pursuers or xyz != 3:
        raise ValueError("pursuer_position must have shape [batch, 3, 3]")
    if pursuer_quaternion.shape != (batch, agents, 4):
        raise ValueError("pursuer_quaternion shape mismatch")
    if pursuer_velocity.shape != (batch, agents, 3):
        raise ValueError("pursuer_velocity shape mismatch")
    if pursuer_angular_velocity.shape != (batch, agents, 3):
        raise ValueError("pursuer_angular_velocity shape mismatch")
    if target_position.shape != (batch, 3) or target_velocity.shape != (batch, 3):
        raise ValueError("target state shape mismatch")
    if visible_each.shape != (batch, agents):
        raise ValueError("visible_each shape mismatch")

    team_visible = visible_each.any(dim=1)
    lost_fraction = (
        lost_steps.to(pursuer_position.dtype).clamp(max=spec.lost_cap_steps)
        / spec.lost_cap_steps
    )
    outputs = []
    for agent_index in range(agents):
        own_position = pursuer_position[:, agent_index]
        self_block = torch.cat(
            (
                _normalize_position_delta(own_position, spec),
                pursuer_velocity[:, agent_index] / spec.v_max_mps,
                pursuer_quaternion[:, agent_index],
                pursuer_angular_velocity[:, agent_index] / 6.0,
            ),
            dim=-1,
        )

        measurement_available = visible_each[:, agent_index]
        if spec.team_target_sharing:
            measurement_available = measurement_available | team_visible
        remembered_target = torch.cat(
            (
                last_seen_target_xy,
                torch.full(
                    (batch, 1),
                    spec.fixed_altitude_m,
                    dtype=target_position.dtype,
                    device=target_position.device,
                ),
            ),
            dim=-1,
        )
        perceived_target = torch.where(
            measurement_available[:, None], target_position, remembered_target
        )
        perceived_velocity = torch.where(
            measurement_available[:, None],
            target_velocity,
            torch.zeros_like(target_velocity),
        )
        target_block = torch.cat(
            (
                _normalize_position_delta(perceived_target - own_position, spec),
                perceived_velocity / spec.v_max_mps,
                torch.where(
                    measurement_available,
                    torch.ones_like(lost_fraction),
                    -torch.ones_like(lost_fraction),
                )[:, None],
                lost_fraction[:, None],
            ),
            dim=-1,
        )

        teammate_blocks = []
        teammate_bearings = []
        own_yaw = _yaw_from_wxyz(pursuer_quaternion[:, agent_index])
        for teammate_index in range(agents):
            if teammate_index == agent_index:
                continue
            relative_position = pursuer_position[:, teammate_index] - own_position
            teammate_blocks.append(
                torch.cat(
                    (
                        _normalize_position_delta(relative_position, spec),
                        (
                            pursuer_velocity[:, teammate_index]
                            - pursuer_velocity[:, agent_index]
                        )
                        / spec.v_max_mps,
                    ),
                    dim=-1,
                )
            )
            teammate_bearings.append(
                torch.remainder(
                    torch.atan2(relative_position[:, 1], relative_position[:, 0])
                    - own_yaw,
                    2.0 * torch.pi,
                )
            )
        teammate_tensor = torch.stack(teammate_blocks, dim=1)
        bearing_tensor = torch.stack(teammate_bearings, dim=1)
        teammate_order = torch.argsort(bearing_tensor, dim=1)
        teammate_tensor = torch.gather(
            teammate_tensor,
            1,
            teammate_order[..., None].expand(-1, -1, teammate_tensor.shape[-1]),
        )

        obstacle_block = torch.zeros(
            (batch, 5 * spec.k_nearest_obstacles),
            dtype=pursuer_position.dtype,
            device=pursuer_position.device,
        )
        obstacle_count = obstacle_xy.shape[1]
        if obstacle_count:
            surface_distance = (
                obstacle_xy - own_position[:, None, :2]
            ).norm(dim=-1) - obstacle_radius
            nearest_count = min(spec.k_nearest_obstacles, obstacle_count)
            nearest_index = torch.topk(
                surface_distance, nearest_count, dim=1, largest=False
            ).indices
            gather_xy = nearest_index[..., None].expand(-1, -1, 2)
            nearest_xy = torch.gather(obstacle_xy, 1, gather_xy)
            nearest_velocity = torch.gather(obstacle_velocity_xy, 1, gather_xy)
            nearest_radius = torch.gather(obstacle_radius, 1, nearest_index)
            features = torch.cat(
                (
                    (nearest_xy - own_position[:, None, :2]) / spec.arena_size_m,
                    nearest_radius[..., None] / spec.arena_size_m,
                    nearest_velocity / spec.v_max_mps,
                ),
                dim=-1,
            )
            obstacle_block[:, : 5 * nearest_count] = features.reshape(batch, -1)

        outputs.append(
            torch.cat(
                (
                    self_block,
                    target_block,
                    teammate_tensor.reshape(batch, -1),
                    obstacle_block,
                ),
                dim=-1,
            )
        )
    observations = torch.stack(outputs, dim=1)
    expected_dim = NATIVE_OBSERVATION_DIM
    if role_phase_features is not None:
        expected_shape = (batch, agents, NATIVE_ROLE_PHASE_FEATURE_DIM)
        if role_phase_features.shape != expected_shape:
            raise ValueError(
                f"role_phase_features must have shape {expected_shape}"
            )
        observations = torch.cat((observations, role_phase_features), dim=-1)
        expected_dim = NATIVE_ROLE_PHASE_OBSERVATION_DIM
    if observations.shape[-1] != expected_dim:
        raise RuntimeError(
            f"native observation dimension {observations.shape[-1]} != {expected_dim}"
        )
    return observations


def build_native_centralized_state(
    spec: P5TaskSpec,
    pursuer_position: torch.Tensor,
    pursuer_quaternion: torch.Tensor,
    pursuer_velocity: torch.Tensor,
    pursuer_angular_velocity: torch.Tensor,
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    lost_steps: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radius: torch.Tensor,
) -> torch.Tensor:
    """Build the centralized training-only state with physical response terms."""

    batch, agents, _ = pursuer_position.shape
    pursuer_part = torch.cat(
        (
            _normalize_position_delta(pursuer_position, spec),
            pursuer_velocity / spec.v_max_mps,
            pursuer_quaternion,
            pursuer_angular_velocity / 6.0,
        ),
        dim=-1,
    ).reshape(batch, 13 * agents)
    target_part = torch.cat(
        (
            _normalize_position_delta(target_position, spec),
            target_velocity / spec.v_max_mps,
        ),
        dim=-1,
    )
    lost_part = (
        lost_steps.to(pursuer_position.dtype).clamp(max=spec.lost_cap_steps)
        / spec.lost_cap_steps
    )[:, None]
    obstacle_part = torch.zeros(
        (batch, 3 * spec.k_nearest_obstacles),
        dtype=pursuer_position.dtype,
        device=pursuer_position.device,
    )
    if obstacle_xy.shape[1]:
        distance = (obstacle_xy - target_position[:, None, :2]).norm(dim=-1)
        nearest_count = min(spec.k_nearest_obstacles, obstacle_xy.shape[1])
        nearest_index = torch.topk(distance, nearest_count, dim=1, largest=False).indices
        nearest_xy = torch.gather(
            obstacle_xy, 1, nearest_index[..., None].expand(-1, -1, 2)
        )
        nearest_radius = torch.gather(obstacle_radius, 1, nearest_index)
        features = torch.cat(
            (nearest_xy / spec.arena_size_m, nearest_radius[..., None] / spec.arena_size_m),
            dim=-1,
        )
        obstacle_part[:, : 3 * nearest_count] = features.reshape(batch, -1)
    state = torch.cat((pursuer_part, target_part, lost_part, obstacle_part), dim=-1)
    if state.shape[-1] != NATIVE_STATE_DIM:
        raise RuntimeError(f"native state dimension {state.shape[-1]} != {NATIVE_STATE_DIM}")
    return state
