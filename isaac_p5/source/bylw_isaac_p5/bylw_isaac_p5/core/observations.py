"""Torch implementation of the frozen 2-D observation/state layouts."""

from __future__ import annotations

import torch

from .config import P5TaskSpec


def build_legacy_observations(
    spec: P5TaskSpec,
    pursuer_xy: torch.Tensor,
    pursuer_yaw: torch.Tensor,
    pursuer_speed: torch.Tensor,
    target_xy: torch.Tensor,
    target_velocity_xy: torch.Tensor,
    visible_each: torch.Tensor,
    last_seen_target_xy: torch.Tensor,
    lost_steps: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radius: torch.Tensor,
    obstacle_velocity_xy: torch.Tensor,
) -> torch.Tensor:
    """Build `[batch, pursuer, 39]` observations for frozen J12/J13 actors."""
    spec.validate()
    batch, agents, _ = pursuer_xy.shape
    if agents != spec.n_pursuers:
        raise ValueError("pursuer count differs from the frozen P5 contract")
    if pursuer_yaw.shape != (batch, agents) or pursuer_speed.shape != (batch, agents):
        raise ValueError("pursuer yaw/speed shape mismatch")
    if target_xy.shape != (batch, 2) or target_velocity_xy.shape != (batch, 2):
        raise ValueError("target state shape mismatch")
    if visible_each.shape != (batch, agents):
        raise ValueError("visible_each shape mismatch")
    if obstacle_xy.shape[:2] != obstacle_radius.shape or obstacle_velocity_xy.shape != obstacle_xy.shape:
        raise ValueError("obstacle tensor shape mismatch")

    arena = spec.arena_size_m
    velocity_xy = torch.stack(
        (
            pursuer_speed * torch.cos(pursuer_yaw),
            pursuer_speed * torch.sin(pursuer_yaw),
        ),
        dim=-1,
    )
    team_visible = visible_each.any(dim=1)
    lost_fraction = lost_steps.to(pursuer_xy.dtype).clamp(max=spec.lost_cap_steps) / spec.lost_cap_steps
    outputs = []
    for agent_index in range(agents):
        self_block = torch.stack(
            (
                pursuer_xy[:, agent_index, 0] / arena,
                pursuer_xy[:, agent_index, 1] / arena,
                torch.cos(pursuer_yaw[:, agent_index]),
                torch.sin(pursuer_yaw[:, agent_index]),
                pursuer_speed[:, agent_index] / spec.v_max_mps,
            ),
            dim=-1,
        )

        measurement_available = visible_each[:, agent_index]
        if spec.team_target_sharing:
            measurement_available = measurement_available | team_visible
        current_relative = (target_xy - pursuer_xy[:, agent_index]) / arena
        last_relative = (last_seen_target_xy - pursuer_xy[:, agent_index]) / arena
        target_relative = torch.where(
            measurement_available[:, None], current_relative, last_relative
        )
        target_velocity = torch.where(
            measurement_available[:, None],
            target_velocity_xy / spec.v_max_mps,
            torch.zeros_like(target_velocity_xy),
        )
        visibility_flag = torch.where(
            measurement_available,
            torch.ones_like(lost_fraction),
            -torch.ones_like(lost_fraction),
        )
        target_block = torch.cat(
            (
                target_relative,
                target_velocity,
                visibility_flag[:, None],
                lost_fraction[:, None],
            ),
            dim=-1,
        )

        teammate_blocks = []
        for teammate_index in range(agents):
            if teammate_index == agent_index:
                continue
            teammate_blocks.append(
                torch.cat(
                    (
                        (pursuer_xy[:, teammate_index] - pursuer_xy[:, agent_index]) / arena,
                        (velocity_xy[:, teammate_index] - velocity_xy[:, agent_index]) / spec.v_max_mps,
                    ),
                    dim=-1,
                )
            )

        obstacle_block = torch.zeros(
            (batch, 5 * spec.k_nearest_obstacles),
            dtype=pursuer_xy.dtype,
            device=pursuer_xy.device,
        )
        obstacle_count = obstacle_xy.shape[1]
        if obstacle_count:
            surface_distance = (
                obstacle_xy - pursuer_xy[:, agent_index, None, :]
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
                    (nearest_xy - pursuer_xy[:, agent_index, None, :]) / arena,
                    nearest_radius[..., None] / arena,
                    nearest_velocity / spec.v_max_mps,
                ),
                dim=-1,
            )
            obstacle_block[:, : 5 * nearest_count] = features.reshape(batch, -1)

        outputs.append(
            torch.cat((self_block, target_block, *teammate_blocks, obstacle_block), dim=-1)
        )
    observations = torch.stack(outputs, dim=1)
    if observations.shape[-1] != spec.observation_dim:
        raise RuntimeError(
            f"observation contract produced {observations.shape[-1]}, expected {spec.observation_dim}"
        )
    return observations


def build_legacy_centralized_state(
    spec: P5TaskSpec,
    pursuer_xy: torch.Tensor,
    pursuer_yaw: torch.Tensor,
    pursuer_speed: torch.Tensor,
    target_xy: torch.Tensor,
    target_velocity_xy: torch.Tensor,
    lost_steps: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radius: torch.Tensor,
) -> torch.Tensor:
    """Build the frozen 32-dimensional centralized critic state."""
    spec.validate()
    batch, agents, _ = pursuer_xy.shape
    pursuer_parts = torch.stack(
        (
            pursuer_xy[..., 0] / spec.arena_size_m,
            pursuer_xy[..., 1] / spec.arena_size_m,
            torch.cos(pursuer_yaw),
            torch.sin(pursuer_yaw),
            pursuer_speed / spec.v_max_mps,
        ),
        dim=-1,
    ).reshape(batch, 5 * agents)
    target_part = torch.cat(
        (target_xy / spec.arena_size_m, target_velocity_xy / spec.v_max_mps), dim=-1
    )
    lost_part = (
        lost_steps.to(pursuer_xy.dtype).clamp(max=spec.lost_cap_steps) / spec.lost_cap_steps
    )[:, None]
    obstacle_part = torch.zeros(
        (batch, 3 * spec.k_nearest_obstacles),
        dtype=pursuer_xy.dtype,
        device=pursuer_xy.device,
    )
    if obstacle_xy.shape[1]:
        distance = (obstacle_xy - target_xy[:, None, :]).norm(dim=-1)
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
    state = torch.cat((pursuer_parts, target_part, lost_part, obstacle_part), dim=-1)
    if state.shape[-1] != spec.centralized_state_dim:
        raise RuntimeError("centralized state dimension drift")
    return state
