"""Guarded DirectMARLEnv wrapper for P5-0B/P5-0C.

P5-0A validates this file syntactically but never imports or instantiates it.
The runtime environment variable is an explicit second gate in addition to
the external GPU resource check.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch
from ....core.visibility_history import advance_visibility_history

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from ....core.action_adapter import HighLevelCommand, PlanarCommandAdapter, wrap_angle
from ....core.config import P5TaskSpec, TimingSpec
from ....core.geometry import encirclement_capture, visibility_masks
from ....core.low_level_controller import (
    FixedHeightRateController,
    LowLevelControllerSpec,
    quaternion_to_euler_wxyz,
)
from ....core.observations import build_legacy_centralized_state, build_legacy_observations
from ....core.reactive_evader import BatchedReactiveEvader, ReactiveEvaderSpec
from ....core.reset_sampler import (
    sample_near_ring_curriculum_reset,
    sample_irregular_obstacle_layout,
    sample_tracked_ring_reset,
)
from .bylw_pursuit_env_cfg import BylwPursuitEnvCfg


class BylwPursuitEnv(DirectMARLEnv):
    cfg: BylwPursuitEnvCfg

    def __init__(self, cfg: BylwPursuitEnvCfg, render_mode: str | None = None, **kwargs):
        self._forced_episode_seed_plan: torch.Tensor | None = None
        self._forced_episode_seed_cursor = 0
        if os.environ.get(cfg.runtime_gate_env_var) != "1":
            raise RuntimeError(
                "P5 Isaac runtime is locked. Complete the GPU resource gate and P5-0B approval, "
                f"then set {cfg.runtime_gate_env_var}=1 for the smoke command only."
            )
        if not (
            len(cfg.obstacle_specs)
            == len(cfg.obstacle_shape_kinds)
            == len(cfg.obstacle_box_sizes_xy_m)
        ):
            raise ValueError(
                "obstacle specs, shapes, and box sizes must have equal lengths"
            )
        for index, ((_, _, envelope_radius), kind, box_size) in enumerate(
            zip(
                cfg.obstacle_specs,
                cfg.obstacle_shape_kinds,
                cfg.obstacle_box_sizes_xy_m,
            )
        ):
            if kind not in {"cylinder", "box"}:
                raise ValueError(f"unsupported obstacle shape {kind!r} at {index}")
            if kind == "box":
                size_x, size_y = box_size
                if size_x <= 0.0 or size_y <= 0.0:
                    raise ValueError(f"box obstacle {index} must have positive XY size")
                circumscribed_radius = 0.5 * math.hypot(size_x, size_y)
                if envelope_radius + 1.0e-6 < circumscribed_radius:
                    raise ValueError(
                        f"box obstacle {index} safety envelope is smaller than its "
                        "circumscribed circle"
                    )
        dynamic_indices = tuple(int(index) for index in cfg.dynamic_obstacle_indices)
        if len(set(dynamic_indices)) != len(dynamic_indices):
            raise ValueError("dynamic obstacle indices must be unique")
        if any(index < 0 or index >= len(cfg.obstacle_specs) for index in dynamic_indices):
            raise ValueError("dynamic obstacle index is out of range")
        if dynamic_indices and (
            cfg.dynamic_obstacle_speed_mps <= 0.0
            or cfg.dynamic_obstacle_reversal_interval_s <= 0.0
        ):
            raise ValueError("dynamic obstacle motion parameters must be positive")
        self._task_spec = P5TaskSpec(
            timing=TimingSpec(
                physics_hz=round(1.0 / cfg.sim.dt),
                low_level_control_hz=round(1.0 / cfg.sim.dt),
                high_level_policy_hz=round(1.0 / (cfg.sim.dt * cfg.decimation)),
            ),
            n_static_obstacles=len(cfg.obstacle_specs),
            fixed_altitude_m=cfg.fixed_altitude_m,
            arena_size_m=cfg.arena_size_m,
            sense_range_m=cfg.sense_range_m,
            fov_half_rad=cfg.fov_half_rad,
            capture_radius_main_m=cfg.capture_radius_main_m,
            capture_radius_strict_m=cfg.capture_radius_strict_m,
            max_capture_gap_rad=cfg.max_capture_gap_rad,
            contact_radius_m=cfg.contact_radius_m,
            pursuer_radius_m=cfg.pursuer_radius_m,
            n_pursuers=cfg.n_pursuers,
            min_capture_agents=cfg.min_capture_agents,
            allow_variable_scale=cfg.allow_variable_scale,
        )
        self._task_spec.validate()
        expected_agents = [f"pursuer_{index}" for index in range(cfg.n_pursuers)]
        if list(cfg.possible_agents) != expected_agents:
            raise ValueError("possible_agents must match n_pursuers")
        if cfg.reset_layout not in {"legacy_fixed_cluster", "tracked_ring"}:
            raise ValueError(
                "reset_layout must be legacy_fixed_cluster or tracked_ring"
            )
        if cfg.pursuer_max_planar_accel_mps2 <= 0.0:
            raise ValueError("pursuer_max_planar_accel_mps2 must be positive")
        if cfg.evader_max_planar_accel_mps2 <= 0.0:
            raise ValueError("evader_max_planar_accel_mps2 must be positive")
        super().__init__(cfg, render_mode, **kwargs)

        self._pursuer_body_ids = [robot.find_bodies("body")[0] for robot in self._pursuers]
        self._evader_body_id = self._evader.find_bodies("body")[0]
        self._pursuer_mass = torch.stack(
            [robot.root_physx_view.get_masses().sum(dim=1) for robot in self._pursuers],
            dim=1,
        )
        self._evader_mass = self._evader.root_physx_view.get_masses().sum(dim=1)
        self._adapter = PlanarCommandAdapter(self._task_spec)
        self._controller = FixedHeightRateController(
            LowLevelControllerSpec(
                max_planar_accel_mps2=cfg.pursuer_max_planar_accel_mps2
            )
        )
        self._evader_controller = FixedHeightRateController(
            LowLevelControllerSpec(
                max_planar_accel_mps2=cfg.evader_max_planar_accel_mps2
            )
        )
        self._evader_model = BatchedReactiveEvader(
            ReactiveEvaderSpec(
                dt_s=self._task_spec.timing.high_level_dt,
                maximum_speed_mps=cfg.evader_speed_mps,
                minimum_episode_speed_mps=(
                    cfg.evader_minimum_episode_speed_mps
                ),
                arena_size_m=cfg.arena_size_m,
                dwell_min_s=cfg.evader_dwell_min_s,
                dwell_max_s=cfg.evader_dwell_max_s,
                turn_rate_rps=cfg.evader_turn_rate_rps,
                stop_tau_s=cfg.evader_stop_tau_s,
                pursuer_repulsion_gain=cfg.evader_pursuer_repulsion_gain,
                pursuer_reaction_range_m=cfg.evader_pursuer_reaction_range_m,
                allow_stop_mode=cfg.evader_allow_stop_mode,
                gap_escape_weight=cfg.evader_gap_escape_weight,
                maintain_episode_speed=cfg.evader_maintain_episode_speed,
            )
        )
        self._high_level_command = None
        self._evader_high_level_command = None
        self._last_low_level = None
        self._last_evader_low_level = None
        self._episode_ids = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._episode_seeds = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._curriculum_reset_attempts = 0
        self._curriculum_reset_applied = 0
        self._reset_initial_radius_m = torch.zeros(
            self.num_envs, cfg.n_pursuers, device=self.device
        )
        self._last_seen_target_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._last_seen_target_velocity_xy = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self._lost_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_visibility_tick = torch.full_like(self._lost_steps, -1)
        self._previous_team_visible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_refresh_counter = -1
        self._recapture_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._obstacle_xy_local = torch.tensor(
            [[x, y] for x, y, _ in self.cfg.obstacle_specs],
            dtype=torch.float32,
            device=self.device,
        )[None].expand(self.num_envs, -1, -1).clone()
        self._obstacle_radius = torch.tensor(
            [radius for _, _, radius in self.cfg.obstacle_specs],
            dtype=torch.float32,
            device=self.device,
        )[None].expand(self.num_envs, -1).clone()
        self._obstacle_height = torch.full_like(
            self._obstacle_radius, self.cfg.obstacle_height_m
        )
        self._obstacle_velocity = torch.zeros_like(self._obstacle_xy_local)
        self._dynamic_obstacle_axis = torch.zeros(
            self.num_envs, 2, dtype=torch.float32, device=self.device
        )
        self._episode_obstacle_start_xy = self._obstacle_xy_local.clone()
        self._episode_target_start_xy = torch.zeros(
            self.num_envs, 2, dtype=torch.float32, device=self.device
        )

    def set_forced_episode_seed_plan(self, episode_seeds: Sequence[int]) -> None:
        """Force upcoming resets to consume an explicit auditable seed plan."""
        seeds = torch.as_tensor(
            list(episode_seeds), dtype=torch.long, device=self.device
        )
        if seeds.ndim != 1 or seeds.numel() == 0:
            raise ValueError("forced episode seed plan must be a nonempty vector")
        if seeds.unique().numel() != seeds.numel():
            raise ValueError("forced episode seed plan must contain unique seeds")
        if ((seeds < 0) | (seeds >= 2**32)).any():
            raise ValueError("forced episode seeds must be uint32-compatible")
        self._forced_episode_seed_plan = seeds
        self._forced_episode_seed_cursor = 0

    def clear_forced_episode_seed_plan(self) -> None:
        self._forced_episode_seed_plan = None
        self._forced_episode_seed_cursor = 0

    def _setup_scene(self) -> None:
        cfgs = tuple(
            getattr(self.cfg, f"pursuer_{index}_cfg")
            for index in range(self.cfg.n_pursuers)
        )
        self._pursuers = [Articulation(robot_cfg) for robot_cfg in cfgs]
        self._evader = Articulation(self.cfg.evader_cfg)
        self._obstacles: list[RigidObject] = []
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        for index, (x, y, radius) in enumerate(self.cfg.obstacle_specs):
            common_spawn_kwargs = dict(
                rigid_props=(
                    sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    )
                    if self.cfg.randomize_obstacles_on_reset
                    else None
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.25, 0.30, 0.35)
                ),
            )
            obstacle_kind = self.cfg.obstacle_shape_kinds[index]
            if obstacle_kind == "cylinder":
                obstacle_cfg = sim_utils.CylinderCfg(
                    radius=radius,
                    height=self.cfg.obstacle_height_m,
                    **common_spawn_kwargs,
                )
            else:
                size_x, size_y = self.cfg.obstacle_box_sizes_xy_m[index]
                obstacle_cfg = sim_utils.CuboidCfg(
                    size=(size_x, size_y, self.cfg.obstacle_height_m),
                    **common_spawn_kwargs,
                )
            if self.cfg.randomize_obstacles_on_reset:
                rigid_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Obstacle_{index}",
                    spawn=obstacle_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(x, y, 0.5 * self.cfg.obstacle_height_m)
                    ),
                )
                self._obstacles.append(RigidObject(rigid_cfg))
            else:
                obstacle_cfg.func(
                    f"/World/envs/env_0/Obstacle_{index}",
                    obstacle_cfg,
                    translation=(x, y, 0.5 * self.cfg.obstacle_height_m),
                )
        self.scene.clone_environments(copy_from_source=False)
        for index, robot in enumerate(self._pursuers):
            self.scene.articulations[f"pursuer_{index}"] = robot
        self.scene.articulations["evader"] = self._evader
        for index, obstacle in enumerate(self._obstacles):
            self.scene.rigid_objects[f"obstacle_{index}"] = obstacle
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)
        )
        light_cfg.func("/World/Light", light_cfg)

    def _stack_pursuer_state(self):
        position = torch.stack([robot.data.root_pos_w for robot in self._pursuers], dim=1)
        quaternion = torch.stack([robot.data.root_quat_w for robot in self._pursuers], dim=1)
        linear_velocity = torch.stack(
            [robot.data.root_lin_vel_w for robot in self._pursuers], dim=1
        )
        angular_velocity = torch.stack(
            [robot.data.root_ang_vel_b for robot in self._pursuers], dim=1
        )
        return position, quaternion, linear_velocity, angular_velocity

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self._advance_dynamic_obstacles()
        normalized = torch.stack(
            [actions[agent] for agent in self.cfg.possible_agents], dim=1
        ).clamp(-1.0, 1.0)
        _, quaternion, linear_velocity, _ = self._stack_pursuer_state()
        yaw = quaternion_to_euler_wxyz(quaternion)[..., 2]
        self._high_level_command = self._adapter.step(
            normalized,
            yaw,
            current_speed_mps=linear_velocity[..., :2].norm(dim=-1),
        )
        evader_position = self._evader.data.root_pos_w - self.scene.env_origins
        evader_quaternion = self._evader.data.root_quat_w
        evader_yaw = quaternion_to_euler_wxyz(evader_quaternion)[:, 2]
        if self.cfg.evader_enabled:
            desired_velocity = self._evader_model.step(
                evader_position[:, :2],
                self._pursuer_position_for_control(),
                self._obstacle_xy_local,
                self._obstacle_radius,
            )
        else:
            desired_velocity = torch.zeros(self.num_envs, 2, device=self.device)
        desired_heading = torch.atan2(desired_velocity[:, 1], desired_velocity[:, 0])
        desired_yaw_rate = (
            wrap_angle(desired_heading - evader_yaw) / 0.5
        ).clamp(-self._task_spec.yaw_rate_max_rps, self._task_spec.yaw_rate_max_rps)
        desired_yaw_rate = torch.where(
            desired_velocity.norm(dim=-1) > 1e-3,
            desired_yaw_rate,
            torch.zeros_like(desired_yaw_rate),
        )
        self._evader_high_level_command = HighLevelCommand(
            desired_velocity_xy_w=desired_velocity,
            desired_altitude_m=torch.full_like(evader_yaw, self.cfg.fixed_altitude_m),
            desired_yaw_rate_rps=desired_yaw_rate,
            desired_speed_mps=desired_velocity.norm(dim=-1),
            normalized_action=torch.zeros(self.num_envs, 2, device=self.device),
        )

    def _pursuer_position_for_control(self) -> torch.Tensor:
        position = torch.stack([robot.data.root_pos_w for robot in self._pursuers], dim=1)
        return (position - self.scene.env_origins[:, None, :])[..., :2]

    def _apply_action(self) -> None:
        if self._high_level_command is None:
            return
        position, quaternion, linear_velocity, angular_velocity = self._stack_pursuer_state()
        low_level = self._controller.compute(
            self._high_level_command,
            position,
            quaternion,
            linear_velocity,
            angular_velocity,
            self._pursuer_mass,
        )
        self._last_low_level = low_level
        for index, robot in enumerate(self._pursuers):
            robot.set_external_force_and_torque(
                low_level.thrust_b[:, index].unsqueeze(1),
                low_level.moment_b[:, index].unsqueeze(1),
                body_ids=self._pursuer_body_ids[index],
            )
        evader_low_level = self._evader_controller.compute(
            self._evader_high_level_command,
            self._evader.data.root_pos_w - self.scene.env_origins,
            self._evader.data.root_quat_w,
            self._evader.data.root_lin_vel_w,
            self._evader.data.root_ang_vel_b,
            self._evader_mass,
        )
        self._last_evader_low_level = evader_low_level
        self._evader.set_external_force_and_torque(
            evader_low_level.thrust_b.unsqueeze(1),
            evader_low_level.moment_b.unsqueeze(1),
            body_ids=self._evader_body_id,
        )

    def _refresh_task_state(self, force: bool = False) -> None:
        if not force and self._last_refresh_counter == self.common_step_counter:
            return
        self._last_refresh_counter = self.common_step_counter
        self._sync_obstacle_state_from_sim()
        position, quaternion, linear_velocity, angular_velocity = self._stack_pursuer_state()
        origins = self.scene.env_origins[:, None, :]
        self._pursuer_position_local = position - origins
        self._pursuer_quaternion = quaternion
        self._pursuer_velocity = linear_velocity
        self._pursuer_angular_velocity = angular_velocity
        self._pursuer_yaw = quaternion_to_euler_wxyz(quaternion)[..., 2]
        self._pursuer_speed = linear_velocity[..., :2].norm(dim=-1)
        self._target_position_local = self._evader.data.root_pos_w - self.scene.env_origins
        self._target_velocity = self._evader.data.root_lin_vel_w
        self._visibility = visibility_masks(
            self._pursuer_position_local,
            self._pursuer_yaw,
            self._target_position_local,
            self._obstacle_xy_local,
            self._obstacle_radius,
            self._obstacle_height,
            self.cfg.sense_range_m,
            self.cfg.fov_half_rad,
            visual_occlusion=True,
        )
        if self.cfg.frame_accurate_loss_counters:
            (self._lost_steps, self._recapture_count, self._previous_team_visible,
             self._last_visibility_tick) = advance_visibility_history(
                self._lost_steps, self._recapture_count, self._previous_team_visible,
                self._visibility.team_visible, self._last_visibility_tick, self.common_step_counter)
        else:
            just_recaptured = self._visibility.team_visible & ~self._previous_team_visible
            self._recapture_count += just_recaptured.to(torch.long)
            self._lost_steps = torch.where(
                self._visibility.team_visible,
                torch.zeros_like(self._lost_steps),
                self._lost_steps + 1,
            )
        offset = 0.5 * self.cfg.arena_size_m
        current_target_xy = self._target_position_local[:, :2] + offset
        self._last_seen_target_xy = torch.where(
            self._visibility.team_visible[:, None],
            current_target_xy,
            self._last_seen_target_xy,
        )
        self._last_seen_target_velocity_xy = torch.where(
            self._visibility.team_visible[:, None],
            self._target_velocity[:, :2],
            self._last_seen_target_velocity_xy,
        )
        self._previous_team_visible = self._visibility.team_visible
        pursuer_xy = self._pursuer_position_local[..., :2]
        target_xy = self._target_position_local[:, :2]
        self._capture_main = encirclement_capture(
            pursuer_xy,
            target_xy,
            self.cfg.capture_radius_main_m,
            self.cfg.min_capture_agents,
            self.cfg.max_capture_gap_rad,
        )
        self._capture_strict = encirclement_capture(
            pursuer_xy,
            target_xy,
            self.cfg.capture_radius_strict_m,
            self.cfg.min_capture_agents,
            self.cfg.max_capture_gap_rad,
        )
        self._agent_collision = self._agent_collision_mask(pursuer_xy)
        (
            self._static_obstacle_collision,
            self._dynamic_obstacle_collision,
        ) = self._obstacle_collision_components(pursuer_xy)
        self._obstacle_collision = (
            self._static_obstacle_collision | self._dynamic_obstacle_collision
        )
        distance_to_target = (pursuer_xy - target_xy[:, None, :]).norm(dim=-1)
        self._contact = (distance_to_target <= self.cfg.contact_radius_m).any(dim=1)
        self._min_agent_clearance = self._minimum_agent_clearance(pursuer_xy)
        self._min_static_clearance = self._minimum_static_clearance(pursuer_xy)
        half = 0.5 * self.cfg.arena_size_m
        self._boundary_invalid_pursuer = (
            self._pursuer_position_local[..., :2].abs() > half
        ).any(dim=(1, 2))
        self._boundary_invalid_evader = (
            self._target_position_local[..., :2].abs() > half
        ).any(dim=1)
        self._boundary_invalid = (
            self._boundary_invalid_pursuer | self._boundary_invalid_evader
        )
        self._min_pursuer_boundary_clearance = (
            half - self.cfg.pursuer_radius_m
            - self._pursuer_position_local[..., :2].abs()
        ).flatten(1).min(dim=1).values
        self._height_invalid_pursuer = (
            (self._pursuer_position_local[..., 2] - self.cfg.fixed_altitude_m).abs()
            > self.cfg.invalid_height_tolerance_m
        ).any(dim=1)
        self._height_invalid_evader = (
            (self._target_position_local[..., 2] - self.cfg.fixed_altitude_m).abs()
            > self.cfg.invalid_height_tolerance_m
        )
        self._height_invalid = (
            self._height_invalid_pursuer | self._height_invalid_evader
        )

    def _minimum_agent_clearance(self, pursuer_xy: torch.Tensor) -> torch.Tensor:
        clearances = []
        for left in range(len(self._pursuers)):
            for right in range(left + 1, len(self._pursuers)):
                clearances.append(
                    (pursuer_xy[:, left] - pursuer_xy[:, right]).norm(dim=-1)
                    - 2.0 * self.cfg.pursuer_radius_m
                )
        return torch.stack(clearances, dim=1).min(dim=1).values

    def _minimum_static_clearance(self, pursuer_xy: torch.Tensor) -> torch.Tensor:
        clearance = (
            pursuer_xy[:, :, None, :] - self._obstacle_xy_local[:, None, :, :]
        ).norm(dim=-1) - self._obstacle_radius[:, None, :] - self.cfg.pursuer_radius_m
        return clearance.flatten(1).min(dim=1).values

    def _agent_collision_mask(self, pursuer_xy: torch.Tensor) -> torch.Tensor:
        collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for left in range(len(self._pursuers)):
            for right in range(left + 1, len(self._pursuers)):
                collision |= (
                    pursuer_xy[:, left] - pursuer_xy[:, right]
                ).norm(dim=-1) < 2.0 * self.cfg.pursuer_radius_m
        return collision

    def _obstacle_collision_components(
        self, pursuer_xy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clearance = (
            pursuer_xy[:, :, None, :] - self._obstacle_xy_local[:, None, :, :]
        ).norm(dim=-1) - self._obstacle_radius[:, None, :] - self.cfg.pursuer_radius_m
        collision_each = (clearance < 0.0).any(dim=1)
        dynamic_indices = tuple(int(index) for index in self.cfg.dynamic_obstacle_indices)
        dynamic_mask = torch.zeros(
            len(self.cfg.obstacle_specs), dtype=torch.bool, device=self.device
        )
        if dynamic_indices:
            dynamic_mask[list(dynamic_indices)] = True
        static_collision = (
            collision_each[:, ~dynamic_mask].any(dim=1)
            if (~dynamic_mask).any()
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        dynamic_collision = (
            collision_each[:, dynamic_mask].any(dim=1)
            if dynamic_mask.any()
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        return static_collision, dynamic_collision

    def _obstacle_collision_mask(self, pursuer_xy: torch.Tensor) -> torch.Tensor:
        static_collision, dynamic_collision = self._obstacle_collision_components(
            pursuer_xy
        )
        return static_collision | dynamic_collision

    def _sync_obstacle_state_from_sim(self) -> None:
        if len(self._obstacles) != len(self.cfg.obstacle_specs):
            return
        for index, obstacle in enumerate(self._obstacles):
            self._obstacle_xy_local[:, index] = (
                obstacle.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
            )
            self._obstacle_velocity[:, index] = obstacle.data.root_lin_vel_w[:, :2]

    def _advance_dynamic_obstacles(self) -> None:
        """Write a deterministic bounded triangle-wave kinematic trajectory."""
        dynamic_indices = tuple(int(index) for index in self.cfg.dynamic_obstacle_indices)
        if not dynamic_indices:
            return
        elapsed_s = (
            self.episode_length_buf.to(torch.float32)
            * self._task_spec.timing.high_level_dt
        )
        interval_s = self.cfg.dynamic_obstacle_reversal_interval_s
        phase_s = torch.remainder(elapsed_s, 4.0 * interval_s)
        sign = torch.where(
            (phase_s < interval_s) | (phase_s >= 3.0 * interval_s),
            torch.ones(self.num_envs, device=self.device),
            -torch.ones(self.num_envs, device=self.device),
        )
        amplitude_m = self.cfg.dynamic_obstacle_speed_mps * interval_s
        displacement_m = torch.where(
            phase_s < interval_s,
            self.cfg.dynamic_obstacle_speed_mps * phase_s,
            torch.where(
                phase_s < 3.0 * interval_s,
                amplitude_m
                - self.cfg.dynamic_obstacle_speed_mps * (phase_s - interval_s),
                -amplitude_m
                + self.cfg.dynamic_obstacle_speed_mps
                * (phase_s - 3.0 * interval_s),
            ),
        )
        velocity_xy = (
            sign[:, None]
            * self.cfg.dynamic_obstacle_speed_mps
            * self._dynamic_obstacle_axis
        )
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        for index in dynamic_indices:
            # Both dynamic bodies share the per-environment axis and bounded
            # displacement.  Their mutual spacing is therefore unchanged.
            position_xy = (
                self._episode_obstacle_start_xy[:, index]
                + displacement_m[:, None] * self._dynamic_obstacle_axis
            )
            self._obstacle_xy_local[:, index] = position_xy
            self._obstacle_velocity[:, index] = velocity_xy
            root_pose = torch.zeros(
                self.num_envs, 7, dtype=torch.float32, device=self.device
            )
            root_pose[:, :2] = self.scene.env_origins[:, :2] + position_xy
            root_pose[:, 2] = (
                self.scene.env_origins[:, 2]
                + 0.5 * self.cfg.obstacle_height_m
            )
            root_pose[:, 3] = 1.0
            root_velocity = torch.zeros(
                self.num_envs, 6, dtype=torch.float32, device=self.device
            )
            root_velocity[:, :2] = velocity_xy
            self._obstacles[index].write_root_pose_to_sim(root_pose, env_ids)
            self._obstacles[index].write_root_velocity_to_sim(
                root_velocity, env_ids
            )
        self._last_refresh_counter = -1

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_task_state()
        offset = 0.5 * self.cfg.arena_size_m
        observations = build_legacy_observations(
            self._task_spec,
            self._pursuer_position_local[..., :2] + offset,
            self._pursuer_yaw,
            self._pursuer_speed,
            self._target_position_local[:, :2] + offset,
            self._target_velocity[:, :2],
            self._visibility.visible_each,
            self._last_seen_target_xy,
            self._lost_steps,
            self._obstacle_xy_local + offset,
            self._obstacle_radius,
            self._obstacle_velocity,
        )
        return {
            agent: observations[:, index]
            for index, agent in enumerate(self.cfg.possible_agents)
        }

    def _get_states(self) -> torch.Tensor:
        self._refresh_task_state()
        offset = 0.5 * self.cfg.arena_size_m
        return build_legacy_centralized_state(
            self._task_spec,
            self._pursuer_position_local[..., :2] + offset,
            self._pursuer_yaw,
            self._pursuer_speed,
            self._target_position_local[:, :2] + offset,
            self._target_velocity[:, :2],
            self._lost_steps,
            self._obstacle_xy_local + offset,
            self._obstacle_radius,
        )

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        self._refresh_task_state()
        distance = (
            self._pursuer_position_local[..., :2]
            - self._target_position_local[:, None, :2]
        ).norm(dim=-1)
        reward = -0.1 * (distance.mean(dim=1) - self.cfg.capture_radius_main_m).abs()
        reward = reward + 100.0 * self._capture_main.success.float()
        reward = reward - 20.0 * (self._agent_collision | self._obstacle_collision).float()
        return {agent: reward for agent in self.cfg.possible_agents}

    def _get_dones(self):
        self._refresh_task_state()
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        terminated_common = (
            self._capture_main.success
            | self._contact
            | self._agent_collision
            | self._obstacle_collision
            | self._boundary_invalid
            | self._height_invalid
        )
        if self._high_level_command is None:
            commanded_speed = torch.zeros_like(self._pursuer_speed)
        else:
            commanded_speed = self._high_level_command.desired_speed_mps
        velocity_heading = torch.atan2(
            self._pursuer_velocity[..., 1], self._pursuer_velocity[..., 0]
        )
        velocity_yaw_slip = torch.atan2(
            torch.sin(velocity_heading - self._pursuer_yaw),
            torch.cos(velocity_heading - self._pursuer_yaw),
        ).abs()
        velocity_yaw_slip = torch.where(
            self._pursuer_speed > 0.2,
            velocity_yaw_slip,
            torch.zeros_like(velocity_yaw_slip),
        )
        if self._last_low_level is None:
            thrust_saturated = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            moment_saturated = thrust_saturated.clone()
        else:
            thrust_saturated = self._last_low_level.thrust_saturated.any(dim=1)
            moment_saturated = self._last_low_level.moment_saturated.any(dim=1)
        telemetry = {
            "episode_seed": self._episode_seeds.clone(),
            "success_6m": self._capture_main.success.clone(),
            "success_5m": self._capture_strict.success.clone(),
            "contact_0p3m": self._contact.clone(),
            "collision_agent": self._agent_collision.clone(),
            "collision_static": self._static_obstacle_collision.clone(),
            "collision_dynamic": self._dynamic_obstacle_collision.clone(),
            "boundary_invalid": self._boundary_invalid.clone(),
            "height_invalid": self._height_invalid.clone(),
            "boundary_invalid_pursuer": self._boundary_invalid_pursuer.clone(),
            "boundary_invalid_evader": self._boundary_invalid_evader.clone(),
            "height_invalid_pursuer": self._height_invalid_pursuer.clone(),
            "height_invalid_evader": self._height_invalid_evader.clone(),
            "timeout": timeout.clone(),
            "team_visible": self._visibility.team_visible.clone(),
            "visibility_status_code": self._visibility.team_status_code.clone(),
            "lost_steps": self._lost_steps.clone(),
            "recapture_count": self._recapture_count.clone(),
            "min_agent_clearance_m": self._min_agent_clearance.clone(),
            "min_static_clearance_m": self._min_static_clearance.clone(),
            "min_pursuer_boundary_clearance_m": (
                self._min_pursuer_boundary_clearance.clone()
            ),
            "speed_tracking_error_max_mps": (
                commanded_speed - self._pursuer_speed
            ).abs().max(dim=1).values.clone(),
            "velocity_yaw_slip_max_rad": velocity_yaw_slip.max(dim=1).values.clone(),
            "thrust_saturated": thrust_saturated.clone(),
            "moment_saturated": moment_saturated.clone(),
            "target_position_local_m": self._target_position_local.clone(),
            "target_velocity_mps": self._target_velocity.clone(),
            "last_seen_target_velocity_xy_mps": (
                self._last_seen_target_velocity_xy.clone()
            ),
            "evader_episode_speed_limit_mps": (
                self._evader_model.episode_speed_limit.clone()
                if self._evader_model.episode_speed_limit is not None
                else torch.zeros(self.num_envs, device=self.device)
            ),
            "target_episode_displacement_m": (
                self._target_position_local[:, :2]
                - self._episode_target_start_xy
            ).norm(dim=1),
            "obstacle_xy_local_m": self._obstacle_xy_local.clone(),
        }
        dynamic_indices = tuple(int(index) for index in self.cfg.dynamic_obstacle_indices)
        if dynamic_indices:
            dynamic_index = torch.as_tensor(
                dynamic_indices, dtype=torch.long, device=self.device
            )
            telemetry["dynamic_obstacle_max_displacement_m"] = (
                self._obstacle_xy_local[:, dynamic_index]
                - self._episode_obstacle_start_xy[:, dynamic_index]
            ).norm(dim=-1).max(dim=1).values
            telemetry["dynamic_obstacle_max_speed_mps"] = (
                self._obstacle_velocity[:, dynamic_index]
                .norm(dim=-1)
                .max(dim=1)
                .values
            )
        else:
            telemetry["dynamic_obstacle_max_displacement_m"] = torch.zeros(
                self.num_envs, device=self.device
            )
            telemetry["dynamic_obstacle_max_speed_mps"] = torch.zeros(
                self.num_envs, device=self.device
            )
        if self.cfg.dense_physical_telemetry:
            if self._last_low_level is None:
                thrust_b = torch.zeros_like(self._pursuer_position_local)
                moment_b = torch.zeros_like(self._pursuer_position_local)
                desired_body_rate = torch.zeros_like(self._pursuer_position_local)
                desired_roll_pitch = torch.zeros(
                    self.num_envs, len(self._pursuers), 2, device=self.device
                )
            else:
                thrust_b = self._last_low_level.thrust_b
                moment_b = self._last_low_level.moment_b
                desired_body_rate = self._last_low_level.desired_body_rate_rps
                desired_roll_pitch = self._last_low_level.desired_roll_pitch_rad
            if self._high_level_command is None:
                commanded_velocity = torch.zeros_like(
                    self._pursuer_position_local[..., :2]
                )
                commanded_yaw_rate = torch.zeros_like(self._pursuer_yaw)
            else:
                commanded_velocity = self._high_level_command.desired_velocity_xy_w
                commanded_yaw_rate = self._high_level_command.desired_yaw_rate_rps
            telemetry.update(
                {
                    "pursuer_position_local_m": self._pursuer_position_local.clone(),
                    "pursuer_velocity_mps": self._pursuer_velocity.clone(),
                    "pursuer_angular_velocity_rps": (
                        self._pursuer_angular_velocity.clone()
                    ),
                    "pursuer_yaw_rad": self._pursuer_yaw.clone(),
                    "target_position_local_m": self._target_position_local.clone(),
                    "target_velocity_mps": self._target_velocity.clone(),
                    "commanded_velocity_xy_mps": commanded_velocity.clone(),
                    "commanded_yaw_rate_rps": commanded_yaw_rate.clone(),
                    "applied_thrust_b_n": thrust_b.clone(),
                    "applied_moment_b_nm": moment_b.clone(),
                    "desired_body_rate_rps": desired_body_rate.clone(),
                    "desired_roll_pitch_rad": desired_roll_pitch.clone(),
                }
            )
        for agent in self.cfg.possible_agents:
            self.extras[agent]["p5"] = telemetry
        terminated = {agent: terminated_common for agent in self.cfg.possible_agents}
        time_outs = {agent: timeout for agent in self.cfg.possible_agents}
        return terminated, time_outs

    @staticmethod
    def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
        quaternion = torch.zeros(yaw.shape[0], 4, device=yaw.device)
        quaternion[:, 0] = torch.cos(0.5 * yaw)
        quaternion[:, 3] = torch.sin(0.5 * yaw)
        return quaternion

    def _write_vehicle_reset(
        self,
        robot: Articulation,
        env_ids: torch.Tensor,
        local_position: tuple[float, float, float] | torch.Tensor,
        yaw_rad: float | torch.Tensor,
    ) -> None:
        root_state = robot.data.default_root_state[env_ids].clone()
        position = torch.as_tensor(
            local_position, dtype=torch.float32, device=self.device
        )
        if position.ndim == 1:
            position = position.expand(len(env_ids), -1)
        if position.shape != (len(env_ids), 3):
            raise ValueError("reset position must have shape [env_ids, 3]")
        root_state[:, :3] = self.scene.env_origins[env_ids] + position
        yaw = torch.as_tensor(yaw_rad, dtype=torch.float32, device=self.device)
        if yaw.ndim == 0:
            yaw = yaw.expand(len(env_ids))
        if yaw.shape != (len(env_ids),):
            raise ValueError("reset yaw must be scalar or match env_ids")
        root_state[:, 3:7] = self._yaw_quaternion(yaw)
        root_state[:, 7:] = 0.0
        robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos[env_ids],
            robot.data.default_joint_vel[env_ids],
            None,
            env_ids,
        )

    def _randomize_obstacles_for_reset(self, env_ids: torch.Tensor) -> None:
        """Move authored kinematic obstacles to seeded episode layouts."""
        if not self.cfg.randomize_obstacles_on_reset:
            return
        if len(self._obstacles) != len(self.cfg.obstacle_specs):
            raise RuntimeError("random obstacle assets were not initialized")
        layouts = np.stack(
            [
                sample_irregular_obstacle_layout(
                    int(seed),
                    self._obstacle_radius[env_id].detach().cpu().numpy(),
                    arena_size_m=self.cfg.arena_size_m,
                    sampling_half_extent_m=(
                        self.cfg.obstacle_sampling_half_extent_m
                    ),
                    minimum_surface_gap_m=(
                        self.cfg.obstacle_minimum_surface_gap_m
                    ),
                    central_keepout_m=self.cfg.obstacle_central_keepout_m,
                )
                for env_id, seed in zip(
                    env_ids.detach().cpu().tolist(),
                    self._episode_seeds[env_ids].detach().cpu().tolist(),
                )
            ]
        )
        sampled_xy = torch.as_tensor(
            layouts, dtype=torch.float32, device=self.device
        )
        self._obstacle_xy_local[env_ids] = sampled_xy
        self._episode_obstacle_start_xy[env_ids] = sampled_xy
        self._obstacle_velocity[env_ids] = 0.0
        motion_angles = torch.as_tensor(
            [
                np.random.default_rng(int(seed) ^ 0x5A17).uniform(-math.pi, math.pi)
                for seed in self._episode_seeds[env_ids].detach().cpu().tolist()
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._dynamic_obstacle_axis[env_ids, 0] = torch.cos(motion_angles)
        self._dynamic_obstacle_axis[env_ids, 1] = torch.sin(motion_angles)
        for index in self.cfg.dynamic_obstacle_indices:
            self._obstacle_velocity[env_ids, int(index)] = (
                self.cfg.dynamic_obstacle_speed_mps
                * self._dynamic_obstacle_axis[env_ids]
            )
        altitude = 0.5 * self.cfg.obstacle_height_m
        for index, obstacle in enumerate(self._obstacles):
            root_state = obstacle.data.default_root_state[env_ids].clone()
            root_state[:, :2] = (
                self.scene.env_origins[env_ids, :2] + sampled_xy[:, index]
            )
            root_state[:, 2] = self.scene.env_origins[env_ids, 2] + altitude
            root_state[:, 3] = 1.0
            root_state[:, 4:7] = 0.0
            root_state[:, 7:] = 0.0
            root_state[:, 7:9] = self._obstacle_velocity[env_ids, index]
            obstacle.write_root_pose_to_sim(root_state[:, :7], env_ids)
            obstacle.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._pursuers[0]._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        runtime_initialized = hasattr(self, "_lost_steps")
        if runtime_initialized:
            self._episode_ids[env_ids] += 1
            self._episode_seeds[env_ids] = (
                int(self.cfg.seed)
                + env_ids
                + self._episode_ids[env_ids] * 1_000_003
            ).remainder(2**32)
            if self._forced_episode_seed_plan is not None:
                remaining = (
                    self._forced_episode_seed_plan.numel()
                    - self._forced_episode_seed_cursor
                )
                forced_count = min(len(env_ids), max(0, remaining))
                if forced_count:
                    start = self._forced_episode_seed_cursor
                    stop = start + forced_count
                    self._episode_seeds[env_ids[:forced_count]] = (
                        self._forced_episode_seed_plan[start:stop]
                    )
                    self._forced_episode_seed_cursor = stop

            self._randomize_obstacles_for_reset(env_ids)

        if self.cfg.reset_layout == "tracked_ring" and runtime_initialized:
            reset_geometries = [
                sample_tracked_ring_reset(
                    int(seed),
                    self._obstacle_xy_local[env_id].detach().cpu().numpy(),
                    self._obstacle_radius[env_id].detach().cpu().numpy(),
                    arena_size_m=self.cfg.arena_size_m,
                    sense_range_m=(
                        self.cfg.sense_range_m
                        if self.cfg.tracked_reset_reference_range_m is None
                        else self.cfg.tracked_reset_reference_range_m
                    ),
                    capture_radius_m=self.cfg.capture_radius_main_m,
                    pursuer_radius_m=self.cfg.pursuer_radius_m,
                    safe_agent_distance_m=(
                        self.cfg.tracked_safe_agent_distance_m
                    ),
                    target_center_xy_m=self.cfg.tracked_target_center_xy_m,
                    target_spread_m=self.cfg.tracked_target_spread_m,
                    n_pursuers=self.cfg.n_pursuers,
                )
                for env_id, seed in zip(
                    env_ids.detach().cpu().tolist(),
                    self._episode_seeds[env_ids].detach().cpu().tolist(),
                )
            ]
            if self.cfg.near_ring_reset_curriculum_enabled:
                reset_geometries = [
                    sample_near_ring_curriculum_reset(
                        int(seed), geometry,
                        self._obstacle_xy_local[env_id].detach().cpu().numpy(),
                        self._obstacle_radius[env_id].detach().cpu().numpy(),
                        arena_size_m=self.cfg.arena_size_m,
                        pursuer_radius_m=self.cfg.pursuer_radius_m,
                        safe_agent_distance_m=self.cfg.tracked_safe_agent_distance_m,
                    )
                    for env_id, seed, geometry in zip(
                        env_ids.detach().cpu().tolist(),
                        self._episode_seeds[env_ids].detach().cpu().tolist(),
                        reset_geometries,
                    )
                ]
                self._curriculum_reset_attempts += len(reset_geometries)
                self._curriculum_reset_applied += sum(
                    geometry.near_ring_curriculum_applied for geometry in reset_geometries
                )
            target_xy = torch.as_tensor(
                np.stack([geometry.target_xy_m for geometry in reset_geometries]),
                dtype=torch.float32,
                device=self.device,
            )
            pursuer_xy = torch.as_tensor(
                np.stack([geometry.pursuer_xy_m for geometry in reset_geometries]),
                dtype=torch.float32,
                device=self.device,
            )
            pursuer_yaw = torch.as_tensor(
                np.stack([geometry.pursuer_yaw_rad for geometry in reset_geometries]),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            if self.cfg.n_pursuers != 3:
                raise ValueError(
                    "variable-scale P5-C runs require reset_layout='tracked_ring'"
                )
            target_xy = torch.tensor(
                [15.0, 0.0], dtype=torch.float32, device=self.device
            ).expand(len(env_ids), -1)
            pursuer_xy = torch.tensor(
                [[-5.0, 0.0], [-8.0, -8.0], [-8.0, 8.0]],
                dtype=torch.float32,
                device=self.device,
            ).expand(len(env_ids), -1, -1)
            pursuer_yaw = torch.atan2(
                target_xy[:, None, 1] - pursuer_xy[..., 1],
                target_xy[:, None, 0] - pursuer_xy[..., 0],
            )

        if runtime_initialized:
            self._reset_initial_radius_m[env_ids] = (
                pursuer_xy - target_xy[:, None, :]
            ).norm(dim=-1)
        altitude = torch.full(
            (len(env_ids), 1), self.cfg.fixed_altitude_m, device=self.device
        )
        for index, robot in enumerate(self._pursuers):
            self._write_vehicle_reset(
                robot,
                env_ids,
                torch.cat((pursuer_xy[:, index], altitude), dim=1),
                pursuer_yaw[:, index],
            )
        self._write_vehicle_reset(
            self._evader,
            env_ids,
            torch.cat((target_xy, altitude), dim=1),
            math.pi,
        )
        if runtime_initialized:
            self._episode_target_start_xy[env_ids] = target_xy
            self._lost_steps[env_ids] = 0
            self._last_visibility_tick[env_ids] = self.common_step_counter
            self._recapture_count[env_ids] = 0
            self._previous_team_visible[env_ids] = True
            self._last_seen_target_xy[env_ids] = (
                target_xy + 0.5 * self.cfg.arena_size_m
            )
            self._last_seen_target_velocity_xy[env_ids] = 0.0
            self._adapter.reset_indices(env_ids)
            initial_heading = torch.full((len(env_ids),), math.pi, device=self.device)
            if self._evader_model._seed is None:
                # DirectMARLEnv's public reset initializes all environments.
                all_heading = torch.full((self.num_envs,), math.pi, device=self.device)
                self._evader_model.reset(self._episode_seeds, all_heading)
            else:
                self._evader_model.reset_indices(
                    env_ids, self._episode_seeds[env_ids], initial_heading
                )
            self._high_level_command = None
            self._evader_high_level_command = None
            self._last_refresh_counter = -1
