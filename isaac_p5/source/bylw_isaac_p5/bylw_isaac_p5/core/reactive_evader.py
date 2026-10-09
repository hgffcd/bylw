"""Deterministic batched CV/CT/STOP evader used by the P5 Isaac smoke task.

This is a Torch translation of ``envs.evaders.ModeSwitchEvader``.  Episode
randomness is derived from each environment's seed instead of a shared RNG so
the trajectory of an episode does not depend on the number of parallel worlds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .action_adapter import wrap_angle


MODE_CV = 0
MODE_CT = 1
MODE_STOP = 2


@dataclass(frozen=True)
class ReactiveEvaderSpec:
    dt_s: float = 0.1
    maximum_speed_mps: float = 4.8
    minimum_episode_speed_mps: float = 0.0
    arena_size_m: float = 100.0
    dwell_min_s: float = 2.0
    dwell_max_s: float = 4.0
    turn_rate_rps: float = 0.3
    stop_tau_s: float = 0.3
    obstacle_range_m: float = 8.0
    obstacle_gain: float = 0.8
    wall_margin_m: float = 8.0
    pursuer_repulsion_gain: float = 1.0
    pursuer_reaction_range_m: float = 30.0
    allow_stop_mode: bool = True
    gap_escape_weight: float = 0.0
    maintain_episode_speed: bool = False

    def validate(self) -> None:
        positive = (
            self.dt_s,
            self.maximum_speed_mps,
            self.arena_size_m,
            self.dwell_min_s,
            self.dwell_max_s,
            self.stop_tau_s,
            self.obstacle_range_m,
            self.wall_margin_m,
            self.pursuer_reaction_range_m,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("reactive evader scale parameters must be positive")
        if self.dwell_max_s < self.dwell_min_s:
            raise ValueError("dwell_max_s must be at least dwell_min_s")
        if not 0.0 <= self.minimum_episode_speed_mps <= self.maximum_speed_mps:
            raise ValueError("minimum episode speed must be in [0, maximum]")
        if self.obstacle_gain < 0 or self.pursuer_repulsion_gain < 0:
            raise ValueError("repulsion gains must be non-negative")
        if not 0.0 <= self.gap_escape_weight <= 1.0:
            raise ValueError("gap escape weight must be in [0, 1]")


class BatchedReactiveEvader:
    """Parallelism-independent batched reactive mode-switch controller."""

    def __init__(self, spec: ReactiveEvaderSpec):
        spec.validate()
        self.spec = spec
        self._seed: torch.Tensor | None = None
        self.mode: torch.Tensor | None = None
        self.heading: torch.Tensor | None = None
        self.speed: torch.Tensor | None = None
        self.ct_direction: torch.Tensor | None = None
        self.dwell_left: torch.Tensor | None = None
        self.switch_count: torch.Tensor | None = None
        self.episode_speed_limit: torch.Tensor | None = None

    @staticmethod
    def _hash(seed: torch.Tensor, counter: torch.Tensor, salt: int) -> torch.Tensor:
        # All intermediates remain well inside int64 for uint32 episode seeds.
        return (
            seed.to(torch.int64) * 1_103_515_245
            + (counter.to(torch.int64) + salt) * 12_345
            + salt * 2_654_435_761
        ).remainder(2_147_483_647)

    def _sample_mode(self, seed: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
        mode_count = 3 if self.spec.allow_stop_mode else 2
        return self._hash(seed, counter, 11).remainder(mode_count)

    def _sample_episode_speed(self, seed: torch.Tensor) -> torch.Tensor:
        minimum = self.spec.minimum_episode_speed_mps
        maximum = self.spec.maximum_speed_mps
        if minimum <= 0.0 or minimum == maximum:
            return torch.full(
                seed.shape,
                maximum,
                dtype=torch.float32,
                device=seed.device,
            )
        unit = self._hash(seed, torch.zeros_like(seed), 53).float() / 2_147_483_646.0
        return minimum + (maximum - minimum) * unit

    def _sample_dwell(self, seed: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
        low = max(1, round(self.spec.dwell_min_s / self.spec.dt_s))
        high = max(low, round(self.spec.dwell_max_s / self.spec.dt_s))
        return low + self._hash(seed, counter, 23).remainder(high - low + 1)

    def _sample_direction(self, seed: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
        bit = self._hash(seed, counter, 37).remainder(2)
        return torch.where(bit == 0, -torch.ones_like(bit), torch.ones_like(bit)).float()

    def reset(self, episode_seeds: torch.Tensor, initial_heading_rad: torch.Tensor) -> None:
        if episode_seeds.ndim != 1 or initial_heading_rad.shape != episode_seeds.shape:
            raise ValueError("episode_seeds and initial_heading_rad must have shape [batch]")
        self._seed = episode_seeds.to(device=initial_heading_rad.device, dtype=torch.int64).clone()
        self.switch_count = torch.zeros_like(self._seed)
        self.mode = self._sample_mode(self._seed, self.switch_count)
        self.heading = wrap_angle(initial_heading_rad.detach().clone())
        self.episode_speed_limit = self._sample_episode_speed(self._seed)
        self.speed = self.episode_speed_limit.clone()
        self.ct_direction = self._sample_direction(self._seed, self.switch_count)
        self.dwell_left = self._sample_dwell(self._seed, self.switch_count)

    def reset_indices(
        self,
        indices: torch.Tensor,
        episode_seeds: torch.Tensor,
        initial_heading_rad: torch.Tensor,
    ) -> None:
        if self._seed is None:
            raise RuntimeError("reset the full batch before resetting indices")
        indices = indices.to(device=self._seed.device, dtype=torch.long)
        seed = episode_seeds.to(device=self._seed.device, dtype=torch.int64)
        counter = torch.zeros_like(seed)
        self._seed[indices] = seed
        self.switch_count[indices] = counter
        self.mode[indices] = self._sample_mode(seed, counter)
        self.heading[indices] = wrap_angle(initial_heading_rad.to(self.heading.device))
        self.episode_speed_limit[indices] = self._sample_episode_speed(seed)
        self.speed[indices] = self.episode_speed_limit[indices]
        self.ct_direction[indices] = self._sample_direction(seed, counter)
        self.dwell_left[indices] = self._sample_dwell(seed, counter)

    def _switch_due_modes(self) -> None:
        due = self.dwell_left <= 0
        if not bool(due.any()):
            return
        self.switch_count[due] += 1
        current = self.mode[due]
        if self.spec.allow_stop_mode:
            # Match the zero-diagonal 0.5/0.5 transition matrix: pick one of
            # the two modes other than the current mode.
            choice = self._hash(
                self._seed[due], self.switch_count[due], 41
            ).remainder(2)
            candidates_a = (current + 1).remainder(3)
            candidates_b = (current + 2).remainder(3)
            self.mode[due] = torch.where(choice == 0, candidates_a, candidates_b)
        else:
            self.mode[due] = 1 - current
        self.ct_direction[due] = self._sample_direction(
            self._seed[due], self.switch_count[due]
        )
        moving = self.mode[due] != MODE_STOP
        updated_speed = self.speed[due]
        updated_speed[moving] = self.episode_speed_limit[due][moving]
        self.speed[due] = updated_speed
        self.dwell_left[due] = self._sample_dwell(
            self._seed[due], self.switch_count[due]
        )

    def step(
        self,
        target_xy_m: torch.Tensor,
        pursuer_xy_m: torch.Tensor,
        obstacle_xy_m: torch.Tensor,
        obstacle_radius_m: torch.Tensor,
    ) -> torch.Tensor:
        """Return desired target velocity in world XY, shape ``[batch, 2]``."""
        if self._seed is None:
            raise RuntimeError("reset must be called before step")
        if self.episode_speed_limit is None:
            raise RuntimeError("episode speed limit is not initialized")
        batch = self._seed.shape[0]
        if target_xy_m.shape != (batch, 2) or pursuer_xy_m.shape[:1] != (batch,):
            raise ValueError("target/pursuer batch shape mismatch")
        if obstacle_xy_m.shape[:1] != (batch,) or obstacle_xy_m.shape[-1] != 2:
            raise ValueError("obstacle_xy_m must have shape [batch, obstacle, 2]")
        if obstacle_radius_m.shape != obstacle_xy_m.shape[:2]:
            raise ValueError("obstacle radius shape mismatch")

        self._switch_due_modes()
        self.dwell_left -= 1
        is_turn = self.mode == MODE_CT
        self.heading = wrap_angle(
            self.heading
            + is_turn.float()
            * self.ct_direction
            * self.spec.turn_rate_rps
            * self.spec.dt_s
        )
        stopped_speed = self.speed * math.exp(-self.spec.dt_s / self.spec.stop_tau_s)
        self.speed = torch.where(
            self.mode == MODE_STOP,
            torch.where(stopped_speed < 1e-3, torch.zeros_like(stopped_speed), stopped_speed),
            self.episode_speed_limit,
        )
        mode_direction = torch.stack(
            (torch.cos(self.heading), torch.sin(self.heading)), dim=-1
        )
        if self.spec.gap_escape_weight > 0.0:
            relative = pursuer_xy_m - target_xy_m[:, None, :]
            angles = torch.sort(
                torch.remainder(
                    torch.atan2(relative[..., 1], relative[..., 0]),
                    2.0 * math.pi,
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
            largest_index = gaps.argmax(dim=1)
            gap_start = angles.gather(1, largest_index[:, None]).squeeze(1)
            gap_heading = torch.remainder(
                gap_start + 0.5 * gaps.max(dim=1).values,
                2.0 * math.pi,
            )
            gap_direction = torch.stack(
                (torch.cos(gap_heading), torch.sin(gap_heading)), dim=-1
            )
            blended_direction = (
                (1.0 - self.spec.gap_escape_weight) * mode_direction
                + self.spec.gap_escape_weight * gap_direction
            )
            mode_direction = blended_direction / blended_direction.norm(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-12)
        intent = self.speed[:, None] * mode_direction

        force = torch.zeros_like(intent)
        if obstacle_xy_m.shape[1]:
            delta = target_xy_m[:, None, :] - obstacle_xy_m
            center_distance = delta.norm(dim=-1).clamp_min(1e-6)
            surface_distance = (center_distance - obstacle_radius_m).clamp_min(1e-6)
            active = surface_distance < self.spec.obstacle_range_m
            obstacle_force = (
                delta / center_distance[..., None]
                * (self.spec.obstacle_gain / surface_distance)[..., None]
                * active[..., None]
            )
            force += obstacle_force.sum(dim=1)

        half = 0.5 * self.spec.arena_size_m
        lower = -half + self.spec.wall_margin_m
        upper = half - self.spec.wall_margin_m
        force += torch.where(
            target_xy_m < lower,
            (lower - target_xy_m) / self.spec.wall_margin_m,
            torch.zeros_like(target_xy_m),
        )
        force -= torch.where(
            target_xy_m > upper,
            (target_xy_m - upper) / self.spec.wall_margin_m,
            torch.zeros_like(target_xy_m),
        )

        if self.spec.pursuer_repulsion_gain > 0:
            delta = target_xy_m[:, None, :] - pursuer_xy_m
            distance = delta.norm(dim=-1).clamp_min(1e-6)
            active = distance < self.spec.pursuer_reaction_range_m
            pursuer_force = (
                delta / distance[..., None]
                / (distance - 1.0).clamp_min(0.5)[..., None]
                * active[..., None]
            )
            force += self.spec.pursuer_repulsion_gain * pursuer_force.sum(dim=1)

        speed_limit = self.episode_speed_limit[:, None]
        command = intent + speed_limit * force
        # Pursuer repulsion can otherwise overpower the soft wall potential.
        # Once inside the wall margin, impose a deterministic inward component
        # while retaining tangential motion and the global speed limit.
        wall_depth_lower = ((lower - target_xy_m) / self.spec.wall_margin_m).clamp(0.0, 1.0)
        wall_depth_upper = ((target_xy_m - upper) / self.spec.wall_margin_m).clamp(0.0, 1.0)
        minimum_inward = 0.25 * speed_limit
        lower_inward = torch.maximum(
            minimum_inward.expand_as(command),
            speed_limit * wall_depth_lower,
        )
        upper_inward = torch.maximum(
            minimum_inward.expand_as(command),
            speed_limit * wall_depth_upper,
        )
        command = torch.where(
            target_xy_m < lower,
            torch.maximum(command, lower_inward),
            command,
        )
        command = torch.where(
            target_xy_m > upper,
            torch.minimum(command, -upper_inward),
            command,
        )
        command_norm = command.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.spec.maintain_episode_speed:
            return command * speed_limit / command_norm
        scale = (speed_limit / command_norm).clamp(max=1.0)
        return command * scale
