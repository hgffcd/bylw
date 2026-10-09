"""Isaac-native training branch; legacy P5 environment remains unchanged."""

from __future__ import annotations

import math
import torch

from ....core.action_adapter import HighLevelCommand, wrap_angle
from ....core.geometry import CaptureResult, encirclement_capture, strong_ring_geometry
from ....core.low_level_controller import quaternion_to_euler_wxyz
from ....core.dual_observer import dual_observer_reference
from ....core.region_lookahead import joint_region_lookahead
from ....core.target_path_guard import joint_target_path_guard, observed_target_acceleration
from ....core.region_capture import (
    region_capture, region_potential, region_task_reward, target_buffer_failure,
    remaining_task_time, region_critic_context,
)
from ....core.native_action_adapter import WorldVelocityYawAdapter
from ....core.native_observations import (
    build_native_centralized_state,
    build_native_observations,
    build_role_phase_features,
)
from ....core.native_training import (
    advance_asynchronous_orientation_gate,
    advance_orientation_gate,
    bounded_last_seen_cv_target_state,
    bounded_constant_velocity_intercept,
    circular_gap_error,
    circular_gap_statistics,
    loss_reacquisition_guidance,
    moving_target_slot_guidance,
    native_team_reward,
    predictive_projection_reward,
    project_dynamics_consistent_angular_gap_velocity,
    project_dynamics_consistent_target_annulus_velocity,
    project_preentry_braking_reserve_velocity,
    slot_goal_progress_reward,
    strong_ring_violation_score,
    project_predictive_arena_boundary_velocity,
    project_predictive_pairwise_velocity,
    project_predictive_static_obstacle_velocity,
    project_predictive_target_annulus_velocity,
    project_predictive_target_separation_velocity,
    project_staged_contraction_velocity,
    release_intercept_horizon_during_contraction,
    reachable_role_slot_prediction_horizon,
    resolve_static_obstacle_tangent_velocity,
    target_facing_yaw_guidance,
)
from ....core.learning_diagnostics import measured_slot_guidance
from .bylw_pursuit_env import BylwPursuitEnv
from .bylw_pursuit_train_env_cfg import BylwPursuitTrainEnvCfg


class BylwPursuitTrainEnv(BylwPursuitEnv):
    """Train a new MAPPO policy against the accepted quadrotor dynamics."""

    cfg: BylwPursuitTrainEnvCfg

    def __init__(self, cfg: BylwPursuitTrainEnvCfg, render_mode: str | None = None, **kwargs):
        self._orientation_runtime_ready = False
        super().__init__(cfg, render_mode, **kwargs)
        self._native_adapter = WorldVelocityYawAdapter(
            self._task_spec, maximum_speed_mps=self.cfg.training_command_speed_mps
        )
        self._last_training_action = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 3, device=self.device
        )
        self._orientation_staging_reached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._orientation_alignment_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._orientation_contraction_unlocked = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._orientation_reference_radius_m = torch.full(
            (self.num_envs,),
            self.cfg.orientation_staging_radius_m,
            device=self.device,
        )
        self._reference_contraction_tracking_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reference_contraction_maximum_radius_error_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self._reference_contraction_maximum_relative_radial_speed_mps = (
            torch.zeros(self.num_envs, device=self.device)
        )
        self._orientation_yaw_error_rad = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), device=self.device
        )
        self._orientation_aligned_now = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._orientation_phase_ready = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        slot_shape = (self.num_envs, len(self.cfg.possible_agents))
        self._orientation_staging_reached_each = torch.zeros(
            slot_shape, dtype=torch.bool, device=self.device
        )
        self._orientation_alignment_steps_each = torch.zeros(
            slot_shape, dtype=torch.long, device=self.device
        )
        self._previous_asynchronous_orientation_progress = torch.zeros(
            self.num_envs, device=self.device
        )
        self._slot_assignment = torch.full(
            slot_shape, -1, dtype=torch.long, device=self.device
        )
        self._slot_anchor_heading_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self._slot_anchor_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._slot_guidance_active = torch.zeros_like(self._slot_anchor_valid)
        self._slot_preentry_radial_priority_active = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._slot_assignment_switched = torch.zeros_like(self._slot_anchor_valid)
        self._slot_handoff_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._slot_active_steps = torch.zeros_like(self._slot_handoff_count)
        self._dual_observer_active = torch.zeros_like(self._slot_anchor_valid)
        self._dual_observer_active_steps = torch.zeros_like(self._slot_handoff_count)
        self._dual_observer_role_switches = torch.zeros_like(self._slot_handoff_count)
        self._dual_observer_pair_angle_rad = torch.zeros(self.num_envs, device=self.device)
        self._path_guard_telemetry = {}
        self._path_guard_previous_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._path_guard_previous_target_velocity = torch.zeros(self.num_envs, 2, device=self.device)
        self._path_guard_corrected_steps = torch.zeros_like(self._slot_handoff_count)
        self._joint_lookahead_active = torch.zeros_like(self._slot_anchor_valid)
        self._joint_lookahead_corrected = torch.zeros_like(self._slot_anchor_valid)
        self._joint_lookahead_active_steps = torch.zeros_like(self._slot_handoff_count)
        self._joint_lookahead_corrected_steps = torch.zeros_like(self._slot_handoff_count)
        self._joint_lookahead_correction_mps = torch.zeros(self.num_envs, device=self.device)
        self._joint_lookahead_nominal_violation = torch.zeros(self.num_envs, device=self.device)
        self._joint_lookahead_selected_violation = torch.zeros(self.num_envs, device=self.device)
        self._slot_mean_error_m = torch.zeros(self.num_envs, device=self.device)
        self._slot_max_error_m = torch.zeros(self.num_envs, device=self.device)
        self._slot_goal_xy = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), 2, device=self.device
        )
        self._loss_reacquisition_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._loss_reacquisition_guidance_active = torch.zeros_like(
            self._loss_reacquisition_active
        )
        self._loss_reacquisition_activated = torch.zeros_like(
            self._loss_reacquisition_active
        )
        self._loss_reacquisition_just_reacquired = torch.zeros_like(
            self._loss_reacquisition_active
        )
        self._loss_reacquisition_scout_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._loss_reacquisition_slot_angle_rad = torch.zeros(
            self.num_envs, len(self.cfg.possible_agents), device=self.device
        )
        self._loss_reacquisition_mean_goal_distance_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self._loss_reacquisition_flank_waypoint_xy = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self._loss_reacquisition_flank_waypoint_valid = torch.zeros_like(
            self._loss_reacquisition_active
        )
        self._loss_reacquisition_flank_guidance_active = torch.zeros_like(
            self._loss_reacquisition_active
        )
        self._loss_reacquisition_flank_fallback_active = torch.zeros_like(
            self._loss_reacquisition_active
        )
        self._loss_reacquisition_flank_blocker_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._loss_reacquisition_flank_events = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._loss_reacquisition_flank_active_steps = torch.zeros_like(
            self._loss_reacquisition_flank_events
        )
        self._loss_reacquisition_flank_fallback_steps = torch.zeros_like(
            self._loss_reacquisition_flank_events
        )
        self._loss_reacquisition_flank_minimum_path_clearance_m = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._loss_reacquisition_flank_minimum_los_clearance_m = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._loss_reacquisition_sector_waypoint_xy = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self._loss_reacquisition_sector_waypoint_valid = torch.zeros_like(
            self._loss_reacquisition_active
        )
        self._loss_reacquisition_sector_guidance_active = torch.zeros_like(
            self._loss_reacquisition_active
        )
        self._loss_reacquisition_sector_fallback_active = torch.zeros_like(
            self._loss_reacquisition_active
        )
        self._loss_reacquisition_sector_phase_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._loss_reacquisition_sector_half_angle_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self._loss_reacquisition_sector_radius_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self._loss_reacquisition_sector_valid_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._loss_reacquisition_sector_fallback_steps = torch.zeros_like(
            self._loss_reacquisition_sector_valid_steps
        )
        self._loss_reacquisition_sector_no_path_candidate_steps = torch.zeros_like(
            self._loss_reacquisition_sector_valid_steps
        )
        self._loss_reacquisition_sector_no_boundary_candidate_steps = (
            torch.zeros_like(self._loss_reacquisition_sector_valid_steps)
        )
        self._loss_reacquisition_sector_minimum_path_clearance_m = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._loss_reacquisition_events = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._loss_reacquisition_completed = torch.zeros_like(
            self._loss_reacquisition_events
        )
        self._loss_reacquisition_active_steps = torch.zeros_like(
            self._loss_reacquisition_events
        )
        self._loss_reacquisition_segment_steps = torch.zeros_like(
            self._loss_reacquisition_events
        )
        self._loss_reacquisition_completed_duration_sum_steps = torch.zeros_like(
            self._loss_reacquisition_events
        )
        self._loss_reacquisition_max_segment_steps = torch.zeros_like(
            self._loss_reacquisition_events
        )
        self._loss_reacquisition_first_completed = torch.zeros_like(
            self._loss_reacquisition_active
        )
        self._loss_reacquisition_first_delay_steps = torch.full_like(
            self._loss_reacquisition_events, -1
        )
        self._intercept_horizon_s = torch.zeros(self.num_envs, device=self.device)
        self._intercept_reachable_fraction = torch.zeros(
            self.num_envs, device=self.device
        )
        self._intercept_mean_gap_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self._intercept_blocker_agent_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._intercept_blocker_reachable = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_angular_error_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self._strong_ring_hold_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._strong_ring_now = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._strong_ring_success = torch.zeros_like(self._strong_ring_now)
        self._strong_ring_confirmation_stabilization_active = torch.zeros_like(
            self._strong_ring_now
        )
        self._strong_ring_mean_gap_error_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self._strong_ring_minimum_distance_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self._strong_ring_maximum_distance_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self._previous_strong_ring_violation_score = torch.zeros(
            self.num_envs, device=self.device
        )
        self._strong_ring_last_counter = -1
        self._hybrid_rl_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._learning_diagnostics = {}
        self._target_buffer_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hybrid_traditional_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._hybrid_rl_steps = torch.zeros_like(self._hybrid_traditional_steps)
        self._hybrid_policy_residual_scale = torch.zeros(
            self.num_envs, device=self.device
        )
        # Episode accumulators used to attribute a learned Cartesian residual
        # to the analytic radial and adjacent-gap phase controllers.  They are
        # diagnostics only and never enter observations, rewards or commands.
        self._slot_policy_residual_active_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._slot_policy_residual_norm_sum_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._slot_policy_residual_radial_support_sum_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._slot_policy_residual_radial_opposition_sum_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._slot_policy_residual_phase_support_sum_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._slot_policy_residual_phase_opposition_sum_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        # Read-only C35 diagnostics for the equal-speed command budget.  These
        # sums never enter observations, rewards, termination, or actions.
        self._slot_speed_limit_active_agent_fraction_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._slot_speed_limit_mean_scale_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._slot_speed_limit_forward_loss_sum_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._slot_requested_tangential_speed_sum_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._slot_executed_tangential_speed_sum_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._target_projection_active = torch.zeros_like(self._slot_anchor_valid)
        self._target_projection_corrected = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._target_projection_correction_norm_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._target_projection_constraint_corrections = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._target_projection_maximum_separation_fallback_constraints = (
            torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        )
        self._target_projection_min_predicted_center_distance_m = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._target_projection_minimum_maximum_attainable_center_distance_m = (
            torch.full((self.num_envs,), torch.inf, device=self.device)
        )
        self._target_projection_unavoidable = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._target_projection_unavoidable_constraints = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._target_projection_infeasible = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._ring_projection_active = torch.zeros_like(self._slot_anchor_valid)
        self._ring_projection_activation_latched = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._ring_projection_corrected = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._ring_projection_correction_norm_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._ring_projection_corrected_constraints = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._ring_projection_min_predicted_radius_m = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._ring_projection_max_predicted_radius_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self._ring_projection_infeasible = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._angular_gap_projection_active = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._angular_gap_projection_corrected = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._angular_gap_projection_correction_norm_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._angular_gap_projection_max_predicted_gap_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self._angular_gap_projection_max_predicted_mean_gap_error_rad = (
            torch.zeros(self.num_envs, device=self.device)
        )
        self._angular_gap_projection_infeasible = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._preentry_braking_active = torch.zeros_like(self._slot_anchor_valid)
        self._preentry_braking_corrected = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._preentry_braking_correction_norm_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._preentry_braking_limited_agents = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._preentry_braking_minimum_stopping_margin_m = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._preentry_braking_infeasible = torch.zeros_like(
            self._slot_anchor_valid
        )
        self._previous_ring_target_velocity_xy = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self._ring_target_acceleration_xy = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self._safety_projection_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._safety_projection_corrected = torch.zeros_like(
            self._safety_projection_active
        )
        self._safety_projection_correction_norm_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._safety_projection_pair_corrections = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._safety_projection_min_predicted_center_distance_m = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._safety_projection_infeasible = torch.zeros_like(
            self._safety_projection_active
        )
        self._obstacle_projection_active = torch.zeros_like(
            self._safety_projection_active
        )
        self._obstacle_projection_corrected = torch.zeros_like(
            self._safety_projection_active
        )
        self._obstacle_projection_correction_norm_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._obstacle_projection_constraint_corrections = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._obstacle_projection_min_predicted_clearance_m = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._obstacle_projection_infeasible = torch.zeros_like(
            self._safety_projection_active
        )
        self._boundary_projection_active = torch.zeros_like(
            self._safety_projection_active
        )
        self._boundary_projection_corrected = torch.zeros_like(
            self._safety_projection_active
        )
        self._boundary_projection_correction_norm_mps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._boundary_projection_constraint_corrections = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._boundary_projection_min_predicted_clearance_m = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._boundary_projection_infeasible = torch.zeros_like(
            self._safety_projection_active
        )
        tangent_shape = (self.num_envs, len(self.cfg.possible_agents))
        self._tangent_obstacle_index = torch.full(
            tangent_shape, -1, dtype=torch.long, device=self.device
        )
        self._tangent_side = torch.zeros(
            tangent_shape, dtype=torch.long, device=self.device
        )
        self._tangent_blocked_steps = torch.zeros_like(self._tangent_side)
        self._tangent_hold_steps_remaining = torch.zeros_like(self._tangent_side)
        self._tangent_active = torch.zeros(
            tangent_shape, dtype=torch.bool, device=self.device
        )
        self._tangent_activated = torch.zeros_like(self._tangent_active)
        self._tangent_min_candidate_clearance_m = torch.full(
            tangent_shape, torch.inf, device=self.device
        )
        self._orientation_runtime_ready = True

    def _estimated_target_xy_for_guidance(self) -> torch.Tensor:
        remembered_local = (
            self._last_seen_target_xy - 0.5 * self.cfg.arena_size_m
        )
        return torch.where(
            self._visibility.team_visible[:, None],
            self._target_position_local[:, :2],
            remembered_local,
        )

    def _apply_loss_reacquisition_guidance(
        self, command: HighLevelCommand, pursuer_xy: torch.Tensor
    ) -> HighLevelCommand:
        result = loss_reacquisition_guidance(
            command.desired_velocity_xy_w,
            pursuer_xy,
            self._last_seen_target_xy - 0.5 * self.cfg.arena_size_m,
            self._visibility.team_visible,
            self._lost_steps,
            self._loss_reacquisition_active,
            self._loss_reacquisition_scout_index,
            self._loss_reacquisition_slot_angle_rad,
            enabled=self.cfg.loss_reacquisition_guidance_enabled,
            start_lost_steps=self.cfg.loss_reacquisition_start_lost_steps,
            slot_radius_m=self.cfg.loss_reacquisition_slot_radius_m,
            scout_blend=self.cfg.loss_reacquisition_scout_blend,
            tracker_blend=self.cfg.loss_reacquisition_tracker_blend,
            blend_growth=self.cfg.loss_reacquisition_blend_growth,
            scout_speed_ratio=self.cfg.loss_reacquisition_scout_speed_ratio,
            tracker_speed_ratio=self.cfg.loss_reacquisition_tracker_speed_ratio,
            scout_minimum_selection_distance_m=(
                self.cfg.loss_reacquisition_scout_minimum_selection_distance_m
            ),
            slot_arrival_m=self.cfg.loss_reacquisition_slot_arrival_m,
            maximum_speed_mps=self.cfg.training_command_speed_mps,
            lost_cap_steps=self._task_spec.lost_cap_steps,
            use_occlusion_flank=(
                self.cfg.loss_reacquisition_occlusion_flank_enabled
            ),
            obstacle_xy=self._obstacle_xy_local,
            obstacle_radius_m=self._obstacle_radius,
            previous_flank_waypoint_xy=(
                self._loss_reacquisition_flank_waypoint_xy
            ),
            previous_flank_valid=(
                self._loss_reacquisition_flank_waypoint_valid
            ),
            previous_flank_blocker_index=(
                self._loss_reacquisition_flank_blocker_index
            ),
            pursuer_radius_m=self.cfg.pursuer_radius_m,
            flank_minimum_path_clearance_m=(
                self.cfg.predictive_static_minimum_clearance_m
            ),
            flank_waypoint_margin_m=(
                self.cfg.loss_reacquisition_flank_waypoint_margin_m
            ),
            flank_los_margin_m=(
                self.cfg.loss_reacquisition_flank_los_margin_m
            ),
            flank_blocker_margin_m=(
                self.cfg.loss_reacquisition_flank_blocker_margin_m
            ),
            arena_half_extent_m=0.5 * self.cfg.arena_size_m,
            flank_boundary_clearance_m=(
                self.cfg.loss_reacquisition_flank_boundary_clearance_m
            ),
            flank_angular_samples=(
                self.cfg.loss_reacquisition_flank_angular_samples
            ),
            use_reachable_sector=(
                self.cfg.loss_reacquisition_reachable_sector_enabled
            ),
            last_seen_target_velocity_xy=(
                self._last_seen_target_velocity_xy
            ),
            sector_policy_dt_s=self._task_spec.timing.high_level_dt,
            sector_target_speed_bound_mps=(
                self.cfg.loss_reacquisition_sector_target_speed_bound_mps
            ),
            sector_velocity_heading_threshold_mps=(
                self.cfg.loss_reacquisition_sector_velocity_heading_threshold_mps
            ),
            sector_lookahead_s=(
                self.cfg.loss_reacquisition_sector_lookahead_s
            ),
            sector_minimum_radius_m=(
                self.cfg.loss_reacquisition_sector_minimum_radius_m
            ),
            sector_maximum_radius_m=(
                self.cfg.loss_reacquisition_sector_maximum_radius_m
            ),
            sector_initial_half_angle_rad=(
                self.cfg.loss_reacquisition_sector_initial_half_angle_rad
            ),
            sector_half_angle_growth_rps=(
                self.cfg.loss_reacquisition_sector_half_angle_growth_rps
            ),
            sector_phase_hold_steps=(
                self.cfg.loss_reacquisition_sector_phase_hold_steps
            ),
            sector_boundary_clearance_m=(
                self.cfg.loss_reacquisition_sector_boundary_clearance_m
            ),
            sector_detour_margin_m=(
                self.cfg.loss_reacquisition_sector_detour_margin_m
            ),
            sector_detour_maximum_extra_m=(
                self.cfg.loss_reacquisition_sector_detour_maximum_extra_m
            ),
            sector_detour_radial_samples=(
                self.cfg.loss_reacquisition_sector_detour_radial_samples
            ),
            sector_detour_angular_samples=(
                self.cfg.loss_reacquisition_sector_detour_angular_samples
            ),
        )
        completed_duration = self._loss_reacquisition_segment_steps
        first_completed_now = (
            result.just_reacquired & ~self._loss_reacquisition_first_completed
        )
        self._loss_reacquisition_events += result.activated.long()
        self._loss_reacquisition_completed += result.just_reacquired.long()
        self._loss_reacquisition_active_steps += result.active.long()
        self._loss_reacquisition_completed_duration_sum_steps += torch.where(
            result.just_reacquired,
            completed_duration,
            torch.zeros_like(completed_duration),
        )
        self._loss_reacquisition_max_segment_steps = torch.maximum(
            self._loss_reacquisition_max_segment_steps,
            torch.where(
                result.just_reacquired,
                completed_duration,
                torch.zeros_like(completed_duration),
            ),
        )
        self._loss_reacquisition_first_delay_steps = torch.where(
            first_completed_now,
            completed_duration,
            self._loss_reacquisition_first_delay_steps,
        )
        self._loss_reacquisition_first_completed |= result.just_reacquired
        self._loss_reacquisition_segment_steps = torch.where(
            result.active,
            torch.where(
                result.activated,
                torch.ones_like(self._loss_reacquisition_segment_steps),
                self._loss_reacquisition_segment_steps + 1,
            ),
            torch.zeros_like(self._loss_reacquisition_segment_steps),
        )
        self._loss_reacquisition_max_segment_steps = torch.maximum(
            self._loss_reacquisition_max_segment_steps,
            self._loss_reacquisition_segment_steps,
        )
        self._loss_reacquisition_active = result.active
        self._loss_reacquisition_guidance_active = result.guidance_active
        self._loss_reacquisition_activated = result.activated
        self._loss_reacquisition_just_reacquired = result.just_reacquired
        self._loss_reacquisition_scout_index = result.scout_index
        self._loss_reacquisition_slot_angle_rad = result.slot_angle_rad
        self._loss_reacquisition_mean_goal_distance_m = (
            result.distance_to_goal_m.mean(dim=1)
        )
        self._loss_reacquisition_flank_waypoint_xy = result.flank_waypoint_xy
        self._loss_reacquisition_flank_waypoint_valid = (
            result.flank_waypoint_valid
        )
        self._loss_reacquisition_flank_guidance_active = (
            result.flank_guidance_active
        )
        self._loss_reacquisition_flank_fallback_active = (
            result.flank_fallback_active
        )
        self._loss_reacquisition_flank_blocker_index = (
            result.flank_blocker_index
        )
        flank_selected = result.activated & result.flank_waypoint_valid
        self._loss_reacquisition_flank_events += flank_selected.long()
        self._loss_reacquisition_flank_active_steps += (
            result.flank_guidance_active.long()
        )
        self._loss_reacquisition_flank_fallback_steps += (
            result.flank_fallback_active.long()
        )
        selected_path_clearance = torch.where(
            result.flank_selected_path_clearance_m >= 0.0,
            result.flank_selected_path_clearance_m,
            torch.full_like(
                result.flank_selected_path_clearance_m, torch.inf
            ),
        )
        selected_los_clearance = torch.where(
            result.flank_selected_los_clearance_m >= 0.0,
            result.flank_selected_los_clearance_m,
            torch.full_like(
                result.flank_selected_los_clearance_m, torch.inf
            ),
        )
        self._loss_reacquisition_flank_minimum_path_clearance_m = torch.minimum(
            self._loss_reacquisition_flank_minimum_path_clearance_m,
            selected_path_clearance,
        )
        self._loss_reacquisition_flank_minimum_los_clearance_m = torch.minimum(
            self._loss_reacquisition_flank_minimum_los_clearance_m,
            selected_los_clearance,
        )
        self._loss_reacquisition_sector_waypoint_xy = result.sector_waypoint_xy
        self._loss_reacquisition_sector_waypoint_valid = (
            result.sector_waypoint_valid
        )
        self._loss_reacquisition_sector_guidance_active = (
            result.sector_guidance_active
        )
        self._loss_reacquisition_sector_fallback_active = (
            result.sector_fallback_active
        )
        self._loss_reacquisition_sector_phase_index = result.sector_phase_index
        self._loss_reacquisition_sector_half_angle_rad = (
            result.sector_half_angle_rad
        )
        self._loss_reacquisition_sector_radius_m = result.sector_radius_m
        self._loss_reacquisition_sector_valid_steps += (
            result.sector_guidance_active.long()
        )
        self._loss_reacquisition_sector_fallback_steps += (
            result.sector_fallback_active.long()
        )
        self._loss_reacquisition_sector_no_path_candidate_steps += (
            (
                result.sector_fallback_active
                & ~result.sector_any_path_candidate
            ).long()
        )
        self._loss_reacquisition_sector_no_boundary_candidate_steps += (
            (
                result.sector_fallback_active
                & ~result.sector_any_boundary_candidate
            ).long()
        )
        sector_selected_path_clearance = torch.where(
            result.sector_selected_path_clearance_m >= 0.0,
            result.sector_selected_path_clearance_m,
            torch.full_like(
                result.sector_selected_path_clearance_m, torch.inf
            ),
        )
        self._loss_reacquisition_sector_minimum_path_clearance_m = torch.minimum(
            self._loss_reacquisition_sector_minimum_path_clearance_m,
            sector_selected_path_clearance,
        )
        loss_velocity = result.velocity_xy
        if self.cfg.loss_reacquisition_tracker_hold_enabled:
            agent_index = torch.arange(
                loss_velocity.shape[1], device=self.device
            )[None, :]
            tracker_hold = result.guidance_active[:, None] & (
                agent_index != result.scout_index[:, None]
            )
            loss_velocity = torch.where(
                tracker_hold[..., None],
                torch.zeros_like(loss_velocity),
                loss_velocity,
            )
        return HighLevelCommand(
            desired_velocity_xy_w=loss_velocity,
            desired_altitude_m=command.desired_altitude_m,
            desired_yaw_rate_rps=command.desired_yaw_rate_rps,
            desired_speed_mps=loss_velocity.norm(dim=-1),
            normalized_action=command.normalized_action,
        )

    def _apply_moving_target_slot_guidance(
        self, command: HighLevelCommand, pursuer_xy: torch.Tensor
    ) -> HighLevelCommand:
        estimated_target_xy = self._estimated_target_xy_for_guidance()
        distance = (pursuer_xy - estimated_target_xy[:, None, :]).norm(dim=-1)
        active = (
            (
                distance.max(dim=1).values
                <= self.cfg.moving_target_slot_activation_radius_m
            )
            | self._orientation_staging_reached
        ) & self._visibility.team_visible
        target_velocity = torch.where(
            self._visibility.team_visible[:, None],
            self._target_velocity[:, :2],
            torch.zeros_like(self._target_velocity[:, :2]),
        )
        prediction_horizon_s: float | torch.Tensor = (
            self.cfg.moving_target_slot_prediction_horizon_s
        )
        blocker_agent_index = None
        blocker_role_lock_weight_m = 0.0
        if self.cfg.reachable_intercept_guidance_enabled:
            intercept = bounded_constant_velocity_intercept(
                pursuer_xy,
                estimated_target_xy,
                target_velocity,
                pursuer_speed_mps=self.cfg.training_command_speed_mps,
                minimum_horizon_s=(
                    self.cfg.reachable_intercept_minimum_horizon_s
                ),
                maximum_horizon_s=(
                    self.cfg.reachable_intercept_maximum_horizon_s
                ),
                horizon_samples=self.cfg.reachable_intercept_horizon_samples,
            )
            target_unit = target_velocity / target_velocity.norm(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            forward_score = (
                (pursuer_xy - estimated_target_xy[:, None, :])
                * target_unit[:, None, :]
            ).sum(dim=-1)
            measured_blocker_agent_index = forward_score.argmax(dim=1)
            previous_blocker_valid = self._intercept_blocker_agent_index >= 0
            heading_available = target_velocity.norm(dim=-1) >= (
                self.cfg.moving_target_slot_minimum_target_speed_mps
            )
            blocker_agent_index = torch.where(
                previous_blocker_valid,
                self._intercept_blocker_agent_index,
                torch.where(
                    heading_available,
                    measured_blocker_agent_index,
                    torch.full_like(measured_blocker_agent_index, -1),
                ),
            )
            safe_blocker_agent_index = blocker_agent_index.clamp_min(0)
            blocker_time = torch.gather(
                intercept.time_s, 1, safe_blocker_agent_index[:, None]
            ).squeeze(1)
            blocker_reachable = torch.gather(
                intercept.reachable, 1, safe_blocker_agent_index[:, None]
            ).squeeze(1) & (blocker_agent_index >= 0)
            applied_blocker_time = blocker_time.clamp_max(
                self.cfg.reachable_intercept_slot_horizon_cap_s
            )
            applied_slot_horizon_s = torch.where(
                blocker_agent_index >= 0,
                applied_blocker_time,
                torch.full_like(
                    applied_blocker_time,
                    self.cfg.moving_target_slot_prediction_horizon_s,
                ),
            )
            prediction_horizon_s = applied_slot_horizon_s
            if self.cfg.reachable_intercept_blocker_only_slot_lead:
                blocker_only_prediction_horizon = torch.zeros(
                    self.num_envs,
                    len(self.cfg.possible_agents),
                    dtype=pursuer_xy.dtype,
                    device=self.device,
                )
                blocker_only_prediction_horizon[:, 0] = applied_slot_horizon_s
                prediction_horizon_s = blocker_only_prediction_horizon
            if self.cfg.reachable_intercept_reachable_role_slot_lead:
                prediction_horizon_s = reachable_role_slot_prediction_horizon(
                    intercept.time_s,
                    intercept.reachable,
                    self._slot_assignment,
                    blocker_agent_index,
                    maximum_horizon_s=(
                        self.cfg.reachable_intercept_slot_horizon_cap_s
                    ),
                )
            if self.cfg.reachable_intercept_release_during_contraction:
                prediction_horizon_s = release_intercept_horizon_during_contraction(
                    prediction_horizon_s,
                    self._orientation_reference_radius_m,
                    fade_start_radius_m=(
                        self.cfg.reachable_intercept_lead_fade_start_radius_m
                    ),
                    zero_lead_radius_m=(
                        self.cfg.reachable_intercept_lead_zero_radius_m
                    ),
                )
            confirmation_control_enabled = (
                self.cfg.strong_ring_confirmation_stabilization_enabled
                or self.cfg.strong_ring_confirmation_maximum_phase_gain_enabled
                or self.cfg.moving_target_slot_phase_angular_velocity_damping_confirmation_only
            )
            self._strong_ring_confirmation_stabilization_active = (
                torch.full_like(
                    self._strong_ring_now,
                    confirmation_control_enabled,
                )
                & (self._strong_ring_hold_steps > 0)
                & ~self._strong_ring_success
            )
            if self.cfg.strong_ring_confirmation_stabilization_enabled:
                prediction_horizon_s = torch.as_tensor(
                    prediction_horizon_s,
                    dtype=pursuer_xy.dtype,
                    device=self.device,
                )
                if prediction_horizon_s.ndim == 0:
                    prediction_horizon_s = prediction_horizon_s.expand(
                        self.num_envs, len(self.cfg.possible_agents)
                    )
                elif prediction_horizon_s.ndim == 1:
                    prediction_horizon_s = prediction_horizon_s[:, None].expand(
                        -1, len(self.cfg.possible_agents)
                    )
                prediction_horizon_s = torch.where(
                    self._strong_ring_confirmation_stabilization_active[:, None],
                    torch.zeros_like(prediction_horizon_s),
                    prediction_horizon_s,
                )
            blocker_role_lock_weight_m = (
                self.cfg.reachable_intercept_blocker_role_lock_weight_m
            )
            reported_horizon_s = (
                prediction_horizon_s.max(dim=1).values
                if isinstance(prediction_horizon_s, torch.Tensor)
                and prediction_horizon_s.ndim == 2
                else prediction_horizon_s
            )
            self._intercept_horizon_s = torch.where(
                active, reported_horizon_s, torch.zeros_like(blocker_time)
            )
            self._intercept_reachable_fraction = torch.where(
                active,
                intercept.reachable.float().mean(dim=1),
                torch.zeros_like(blocker_time),
            )
            self._intercept_mean_gap_m = torch.where(
                active,
                intercept.reachability_gap_m.clamp_min(0.0).mean(dim=1),
                torch.zeros_like(blocker_time),
            )
            self._intercept_blocker_agent_index = torch.where(
                active & (blocker_agent_index >= 0),
                blocker_agent_index,
                torch.full_like(blocker_agent_index, -1),
            )
            self._intercept_blocker_reachable = active & blocker_reachable
        else:
            self._strong_ring_confirmation_stabilization_active.zero_()
            self._intercept_horizon_s.zero_()
            self._intercept_reachable_fraction.zero_()
            self._intercept_mean_gap_m.zero_()
            self._intercept_blocker_agent_index.fill_(-1)
            self._intercept_blocker_reachable.zero_()
        observer_reference = None
        if self.cfg.dual_observer_formation_enabled:
            previous_blocker = torch.where(
                (self._slot_assignment == 0).any(dim=1),
                (self._slot_assignment == 0).long().argmax(dim=1),
                torch.full_like(self._intercept_blocker_agent_index, -1))
            observer_reference = dual_observer_reference(
                pursuer_xy, self._pursuer_velocity[..., :2], estimated_target_xy,
                target_velocity, self._visibility.visible_each, self._pursuer_yaw,
                self._orientation_reference_radius_m, previous_blocker,
                self._obstacle_xy_local, self._obstacle_radius,
                maximum_speed_mps=self.cfg.training_command_speed_mps,
                fov_half_rad=self._task_spec.fov_half_rad,
                hysteresis_m=self.cfg.dual_observer_role_hysteresis_m,
                keep_valid_pair=self.cfg.dual_observer_role_lock_enabled)
            observer_reference.valid &= active
            blocker_agent_index = torch.where(
                observer_reference.valid, observer_reference.blocker_index,
                blocker_agent_index if blocker_agent_index is not None else previous_blocker)
            self._dual_observer_active = observer_reference.valid
            self._dual_observer_active_steps += observer_reference.valid.long()
            self._dual_observer_role_switches += (
                observer_reference.valid & (previous_blocker >= 0)
                & (observer_reference.blocker_index != previous_blocker)).long()
            self._dual_observer_pair_angle_rad = observer_reference.pair_angle_rad
        policy_residual_scale = (
            self._hybrid_policy_residual_scale
            if self.cfg.hybrid_phase_supervisor_enabled
            else torch.full_like(
                self._hybrid_policy_residual_scale,
                self.cfg.moving_target_slot_policy_residual_scale,
            )
        )
        if (
            self.cfg.strong_ring_confirmation_stabilization_enabled
            and self.cfg.strong_ring_confirmation_suppress_policy_residual
        ):
            policy_residual_scale = torch.where(
                self._strong_ring_confirmation_stabilization_active,
                torch.zeros_like(policy_residual_scale),
                policy_residual_scale,
            )
        target_feedforward_scale = self.cfg.moving_target_slot_feedforward_scale
        if self.cfg.moving_target_slot_full_feedforward_above_strong_ring:
            target_feedforward_scale = torch.where(
                self._orientation_reference_radius_m
                > self.cfg.strong_ring_maximum_radius_m,
                torch.ones_like(self._orientation_reference_radius_m),
                torch.full_like(
                    self._orientation_reference_radius_m,
                    self.cfg.moving_target_slot_feedforward_scale,
                ),
            )
        radial_correction_priority_active = None
        if self.cfg.moving_target_slot_preentry_radial_priority_enabled:
            maximum_gap, mean_gap_error = circular_gap_statistics(
                pursuer_xy, estimated_target_xy
            )
            radial_correction_priority_active = (
                active
                & (self._strong_ring_hold_steps == 0)
                & (
                    self._orientation_reference_radius_m
                    <= self.cfg.strong_ring_maximum_radius_m + 1.0e-6
                )
                & (
                    distance.max(dim=1).values
                    <= self.cfg.strong_ring_maximum_radius_m
                    + self.cfg.predictive_ring_activation_margin_m
                )
                & (maximum_gap <= self.cfg.strong_ring_maximum_gap_rad)
                & (
                    mean_gap_error
                    <= self.cfg.strong_ring_maximum_mean_gap_error_rad
                )
            )
            self._slot_preentry_radial_priority_active = (
                radial_correction_priority_active
            )
        else:
            self._slot_preentry_radial_priority_active.zero_()
        gain_residual_action = command.normalized_action[..., :2]
        if self.cfg.strong_ring_confirmation_maximum_phase_gain_enabled:
            confirmation_phase_action = torch.where(
                self._strong_ring_confirmation_stabilization_active[:, None],
                torch.ones_like(gain_residual_action[..., 1]),
                gain_residual_action[..., 1],
            )
            gain_residual_action = torch.stack(
                (gain_residual_action[..., 0], confirmation_phase_action),
                dim=-1,
            )
        guidance = measured_slot_guidance if self.cfg.learning_diagnostics_enabled else moving_target_slot_guidance
        result = guidance(
            command.desired_velocity_xy_w,
            pursuer_xy,
            estimated_target_xy,
            target_velocity,
            self._orientation_reference_radius_m,
            active,
            self._slot_assignment,
            self._slot_anchor_heading_rad,
            self._slot_anchor_valid,
            pursuer_velocity_xy=self._pursuer_velocity[..., :2],
            position_gain=self.cfg.moving_target_slot_position_gain,
            policy_residual_scale=policy_residual_scale,
            target_feedforward_scale=target_feedforward_scale,
            maximum_speed_mps=self.cfg.training_command_speed_mps,
            minimum_target_speed_mps=(
                self.cfg.moving_target_slot_minimum_target_speed_mps
            ),
            maximum_anchor_turn_rate_rps=(
                self.cfg.moving_target_slot_maximum_anchor_turn_rate_rps
            ),
            policy_dt_s=self._task_spec.timing.high_level_dt,
            assignment_hysteresis_m=(
                self.cfg.moving_target_slot_assignment_hysteresis_m
            ),
            target_prediction_horizon_s=prediction_horizon_s,
            blocker_agent_index=blocker_agent_index,
            blocker_role_lock_weight_m=blocker_role_lock_weight_m,
            prepare_assignment_when_inactive=(
                self.cfg.moving_target_slot_prepare_assignment_for_observation
            ),
            obstacle_xy=(
                self._obstacle_xy_local
                if self.cfg.moving_target_slot_obstacle_aware_assignment
                else None
            ),
            obstacle_radius_m=(
                self._obstacle_radius
                if self.cfg.moving_target_slot_obstacle_aware_assignment
                else None
            ),
            pursuer_radius_m=self.cfg.pursuer_radius_m,
            assignment_path_clearance_m=(
                self.cfg.moving_target_slot_assignment_path_clearance_m
            ),
            assignment_obstacle_penalty=(
                self.cfg.moving_target_slot_assignment_obstacle_penalty
            ),
            cyclic_order_preserving_assignment=(
                self.cfg.moving_target_slot_cyclic_order_preserving_assignment
            ),
            lock_valid_assignment=(
                self.cfg.moving_target_slot_lock_valid_assignment
            ),
            phase_separation_control_enabled=(
                self.cfg.moving_target_slot_phase_separation_control_enabled
            ),
            phase_gain=self.cfg.moving_target_slot_phase_gain,
            phase_angular_velocity_damping_gain_s=(
                self.cfg.moving_target_slot_phase_angular_velocity_damping_gain_s
            ),
            phase_angular_velocity_damping_active=(
                self._strong_ring_confirmation_stabilization_active
                if self.cfg.moving_target_slot_phase_angular_velocity_damping_confirmation_only
                else None
            ),
            radial_gain=self.cfg.moving_target_slot_radial_gain,
            radial_velocity_damping_gain=(
                self.cfg.moving_target_slot_radial_velocity_damping_gain
            ),
            maximum_radial_speed_mps=(
                self.cfg.moving_target_slot_maximum_radial_speed_mps
                if self.cfg.moving_target_slot_maximum_radial_speed_mps > 0.0
                else None
            ),
            phase_scheduled_radial_speed_budget=(
                self.cfg.moving_target_slot_phase_scheduled_radial_speed_budget
            ),
            maximum_tangential_speed_mps=(
                self.cfg.moving_target_slot_maximum_tangential_speed_mps
            ),
            phase_activation_margin_m=(
                self.cfg.moving_target_slot_phase_activation_margin_m
            ),
            feedforward_priority_projection_enabled=(
                self.cfg.moving_target_slot_feedforward_priority_projection_enabled
            ),
            radial_correction_priority_active=(
                radial_correction_priority_active
            ),
            far_slot_intercept_control_enabled=(
                self.cfg.moving_target_slot_far_intercept_control_enabled
            ),
            far_slot_intercept_minimum_horizon_s=(
                self.cfg.reachable_intercept_minimum_horizon_s
            ),
            far_slot_intercept_maximum_horizon_s=(
                self.cfg.reachable_intercept_maximum_horizon_s
            ),
            far_slot_intercept_horizon_samples=(
                self.cfg.reachable_intercept_horizon_samples
            ),
            blocker_phase_anchor_control_enabled=(
                self.cfg.moving_target_slot_blocker_phase_anchor_control_enabled
            ),
            blocker_phase_anchor_gain=(
                self.cfg.moving_target_slot_blocker_phase_anchor_gain
            ),
            blocker_phase_anchor_maximum_speed_mps=(
                self.cfg.moving_target_slot_blocker_phase_anchor_maximum_speed_mps
            ),
            gain_residual_action=gain_residual_action,
            gain_residual_control_enabled=(
                self.cfg.moving_target_slot_gain_residual_control_enabled
            ),
            radial_gain_residual_scale=(
                self.cfg.moving_target_slot_radial_gain_residual_scale
            ),
            phase_gain_residual_scale=(
                self.cfg.moving_target_slot_phase_gain_residual_scale
            ),
            radial_velocity_residual_action=(
                command.normalized_action[..., 2]
            ),
            radial_velocity_residual_control_enabled=(
                self.cfg.moving_target_slot_radial_velocity_residual_control_enabled
            ),
            radial_velocity_residual_scale_mps=(
                self.cfg.moving_target_slot_radial_velocity_residual_scale_mps
            ),
            tangential_velocity_residual_action=(
                command.normalized_action[..., 1]
            ),
            tangential_velocity_residual_control_enabled=(
                self.cfg.moving_target_slot_tangential_velocity_residual_control_enabled
            ),
            tangential_velocity_residual_scale_mps=(
                self.cfg.moving_target_slot_tangential_velocity_residual_scale_mps
            ),
            region_minimum_radius_m=(
                self.cfg.region_control_minimum_radius_m
                if self.cfg.region_objective_enabled else None
            ),
            region_maximum_gap_rad=(
                self.cfg.region_capture_maximum_gap_rad
                - (math.pi / 12.0 if self.cfg.region_gap_reserve_enabled else 0.0)
                if self.cfg.region_objective_enabled else None
            ),
            observer_reference_heading_rad=(
                observer_reference.heading_rad if observer_reference is not None else None),
            observer_reference_valid=(
                observer_reference.valid if observer_reference is not None else None),
            observer_phase_tolerance_rad=self.cfg.dual_observer_phase_tolerance_rad,
        )
        if self.cfg.learning_diagnostics_enabled:
            self._learning_diagnostics.update(result.learning_diagnostics)
        self._slot_assignment = result.assignment
        self._slot_anchor_heading_rad = result.anchor_heading_rad
        self._slot_anchor_valid = result.anchor_valid
        self._slot_guidance_active = result.active
        self._slot_assignment_switched = result.assignment_switched
        self._slot_handoff_count += result.assignment_switched.long()
        self._slot_active_steps += result.active.long()
        self._slot_mean_error_m = result.mean_slot_error_m
        self._slot_goal_xy = result.assigned_slot_xy
        self._slot_max_error_m = torch.where(
            result.active,
            (result.assigned_slot_xy - pursuer_xy).norm(dim=-1).max(dim=1).values,
            torch.zeros_like(result.mean_slot_error_m),
        )
        residual_diagnostic_active = result.active & (policy_residual_scale > 0.0)
        self._slot_policy_residual_active_steps += (
            residual_diagnostic_active.long()
        )
        self._slot_policy_residual_norm_sum_mps += (
            result.policy_residual_norm_mps
        )
        self._slot_policy_residual_radial_support_sum_mps += (
            result.policy_residual_radial_support_mps
        )
        self._slot_policy_residual_radial_opposition_sum_mps += (
            result.policy_residual_radial_opposition_mps
        )
        self._slot_policy_residual_phase_support_sum_mps += (
            result.policy_residual_phase_support_mps
        )
        self._slot_policy_residual_phase_opposition_sum_mps += (
            result.policy_residual_phase_opposition_mps
        )
        self._slot_speed_limit_active_agent_fraction_sum += (
            result.speed_limit_active_agent_fraction
        )
        self._slot_speed_limit_mean_scale_sum += result.speed_limit_mean_scale
        self._slot_speed_limit_forward_loss_sum_mps += (
            result.speed_limit_forward_loss_mps
        )
        self._slot_requested_tangential_speed_sum_mps += (
            result.requested_tangential_speed_mps
        )
        self._slot_executed_tangential_speed_sum_mps += (
            result.executed_tangential_speed_mps
        )
        return HighLevelCommand(
            desired_velocity_xy_w=result.velocity_xy,
            desired_altitude_m=command.desired_altitude_m,
            desired_yaw_rate_rps=command.desired_yaw_rate_rps,
            desired_speed_mps=result.velocity_xy.norm(dim=-1),
            normalized_action=command.normalized_action,
        )

    def _apply_orientation_guidance(
        self, command: HighLevelCommand, pursuer_xy: torch.Tensor
    ) -> HighLevelCommand:
        estimated_target_xy = self._estimated_target_xy_for_guidance()
        target_velocity_xy = torch.where(
            self._visibility.team_visible[:, None],
            self._target_velocity[:, :2],
            torch.zeros_like(self._target_velocity[:, :2]),
        )
        yaw_gain_multiplier = None
        if (
            self.cfg.moving_target_slot_gain_residual_control_enabled
            and not self.cfg.moving_target_slot_radial_velocity_residual_control_enabled
        ):
            yaw_gain_multiplier = 1.0 + (
                (self._hybrid_policy_residual_scale > 0.0).to(pursuer_xy.dtype)[
                    :, None
                ]
                * self.cfg.orientation_yaw_gain_residual_scale
                * command.normalized_action[..., 2].clamp(-1.0, 1.0)
            )
        yaw_rate, yaw_error = target_facing_yaw_guidance(
            pursuer_xy,
            self._pursuer_yaw,
            estimated_target_xy,
            command.desired_yaw_rate_rps,
            proportional_gain=self.cfg.orientation_yaw_gain,
            policy_residual_scale=(
                torch.zeros_like(self._hybrid_policy_residual_scale)
                if self.cfg.moving_target_slot_gain_residual_control_enabled
                else self._hybrid_policy_residual_scale
                if self.cfg.hybrid_phase_supervisor_enabled
                else self.cfg.orientation_policy_residual_scale
            ),
            maximum_yaw_rate_rps=self._task_spec.yaw_rate_max_rps,
            gain_multiplier=yaw_gain_multiplier,
        )
        if self.cfg.loss_reacquisition_reachable_sector_enabled:
            sector_yaw_rate, sector_yaw_error = target_facing_yaw_guidance(
                pursuer_xy,
                self._pursuer_yaw,
                self._loss_reacquisition_sector_waypoint_xy,
                command.desired_yaw_rate_rps,
                proportional_gain=self.cfg.orientation_yaw_gain,
                policy_residual_scale=self.cfg.orientation_policy_residual_scale,
                maximum_yaw_rate_rps=self._task_spec.yaw_rate_max_rps,
            )
            sector_scout_mask = (
                torch.arange(
                    len(self.cfg.possible_agents), device=self.device
                )[None, :]
                == self._loss_reacquisition_scout_index[:, None]
            ) & self._loss_reacquisition_sector_guidance_active[:, None]
            yaw_rate = torch.where(sector_scout_mask, sector_yaw_rate, yaw_rate)
            yaw_error = torch.where(sector_scout_mask, sector_yaw_error, yaw_error)
        staging_hold = (
            self._orientation_staging_reached_each
            if self.cfg.asynchronous_orientation_gate_enabled
            else self._orientation_staging_reached
        )
        if self.cfg.orientation_target_relative_staging_enabled:
            staging_hold = (
                pursuer_xy - estimated_target_xy[:, None, :]
            ).norm(dim=-1) <= self.cfg.orientation_staging_radius_m
        desired_velocity = project_staged_contraction_velocity(
            command.desired_velocity_xy_w,
            pursuer_xy,
            estimated_target_xy,
            staging_hold,
            self._current_contraction_active(),
            maximum_inward_speed_mps=(
                self.cfg.orientation_maximum_inward_speed_mps
            ),
            target_velocity_xy=(
                target_velocity_xy
                if self.cfg.orientation_target_relative_staging_enabled
                else None
            ),
        )
        self._orientation_yaw_error_rad = yaw_error
        return HighLevelCommand(
            desired_velocity_xy_w=desired_velocity,
            desired_altitude_m=command.desired_altitude_m,
            desired_yaw_rate_rps=yaw_rate,
            desired_speed_mps=desired_velocity.norm(dim=-1),
            normalized_action=command.normalized_action,
        )

    def _apply_predictive_ring_projection(
        self, command: HighLevelCommand, pursuer_xy: torch.Tensor
    ) -> HighLevelCommand:
        control_minimum_radius = (
            self.cfg.region_control_minimum_radius_m
            if self.cfg.region_objective_enabled
            else self.cfg.strong_ring_minimum_radius_m
        )
        control_mean_gap_limit = (
            torch.pi if self.cfg.region_objective_enabled
            else self.cfg.strong_ring_maximum_mean_gap_error_rad
        )
        current_radius = (
            pursuer_xy - self._target_position_local[:, None, :2]
        ).norm(dim=-1)
        phase_ready_for_projection = (
            self._orientation_contraction_unlocked
            & self._orientation_phase_ready
            & self._visibility.team_visible
        )
        active = (
            phase_ready_for_projection
            & (
                self._orientation_reference_radius_m
                <= self.cfg.strong_ring_maximum_radius_m + 1.0e-6
            )
        )
        if self.cfg.predictive_ring_activation_margin_m > 0.0:
            measured_entry = (
                (
                    current_radius.min(dim=1).values
                    >= control_minimum_radius
                    - self.cfg.predictive_ring_activation_margin_m
                )
                & (
                    current_radius.max(dim=1).values
                    <= self.cfg.strong_ring_maximum_radius_m
                    + self.cfg.predictive_ring_activation_margin_m
                )
            )
            if self.cfg.predictive_ring_activation_requires_inside_inner_boundary:
                measured_entry = measured_entry & (
                    current_radius.min(dim=1).values
                    >= control_minimum_radius
                )
            if self.cfg.predictive_ring_activation_requires_strong_ring_entry:
                measured_entry = measured_entry & (self._strong_ring_hold_steps > 0)
            if (
                self.cfg.predictive_ring_activation_requires_maximum_gap_eligibility
            ):
                maximum_gap, _ = circular_gap_statistics(
                    pursuer_xy,
                    self._target_position_local[:, :2],
                )
                measured_entry = measured_entry & (
                    maximum_gap
                    <= self.cfg.strong_ring_maximum_gap_rad
                    + self.cfg.predictive_ring_activation_maximum_gap_margin_rad
                )
            if self.cfg.predictive_ring_activation_latched_after_measured_entry:
                self._ring_projection_activation_latched |= active & measured_entry
                active = active & self._ring_projection_activation_latched
            else:
                active = active & measured_entry
        target_acceleration_xy = (
            self._ring_target_acceleration_xy
            if self.cfg.predictive_ring_target_acceleration_enabled
            else None
        )
        minimum_radius_active = (
            self._strong_ring_hold_steps > 0
            if self.cfg.predictive_ring_minimum_radius_after_strong_entry
            else None
        )
        ring_input_velocity = command.desired_velocity_xy_w
        preentry_braking = None
        if self.cfg.predictive_ring_preentry_braking_enabled:
            preentry_braking = project_preentry_braking_reserve_velocity(
                ring_input_velocity,
                self._pursuer_velocity[..., :2],
                pursuer_xy,
                self._target_position_local[:, :2],
                self._target_velocity[:, :2],
                phase_ready_for_projection,
                reference_radius_m=0.5
                * (
                    control_minimum_radius
                    + self.cfg.strong_ring_maximum_radius_m
                ),
                activation_radius_m=(
                    self.cfg.predictive_ring_preentry_braking_activation_radius_m
                ),
                maximum_speed_mps=self.cfg.training_command_speed_mps,
                maximum_planar_acceleration_mps2=(
                    self.cfg.pursuer_max_planar_accel_mps2
                ),
            )
            ring_input_velocity = preentry_braking.velocity_xy
            self._preentry_braking_active = preentry_braking.active
            self._preentry_braking_corrected = preentry_braking.corrected
            self._preentry_braking_correction_norm_mps = (
                preentry_braking.correction_norm_mps
            )
            self._preentry_braking_limited_agents = (
                preentry_braking.limited_agents
            )
            self._preentry_braking_minimum_stopping_margin_m = (
                preentry_braking.minimum_stopping_margin_m
            )
            self._preentry_braking_infeasible = preentry_braking.infeasible
        if self.cfg.predictive_ring_dynamics_consistent_enabled:
            projection = project_dynamics_consistent_target_annulus_velocity(
                ring_input_velocity,
                self._pursuer_velocity[..., :2],
                pursuer_xy,
                self._target_position_local[:, :2],
                self._target_velocity[:, :2],
                active,
                target_acceleration_xy=target_acceleration_xy,
                target_maximum_speed_mps=self.cfg.evader_speed_mps,
                minimum_radius_m=control_minimum_radius,
                maximum_radius_m=self.cfg.strong_ring_maximum_radius_m,
                prediction_horizon_s=(
                    self.cfg.predictive_ring_prediction_horizon_s
                ),
                maximum_speed_mps=self.cfg.training_command_speed_mps,
                low_level_dt_s=self.cfg.sim.dt,
                velocity_kp=self.cfg.predictive_ring_low_level_velocity_kp,
                maximum_planar_acceleration_mps2=(
                    self.cfg.pursuer_max_planar_accel_mps2
                ),
                radial_candidate_count=(
                    self.cfg.predictive_ring_radial_candidate_count
                ),
                minimum_radius_active=minimum_radius_active,
                common_acceleration_feedforward_active=(
                    self._strong_ring_hold_steps > 0
                    if self.cfg.predictive_ring_common_acceleration_feedforward_after_strong_entry
                    else None
                ),
                preentry_velocity_sync_active=(
                    active & (self._strong_ring_hold_steps == 0)
                    if self.cfg.predictive_ring_preentry_velocity_sync_enabled
                    else None
                ),
            )
        else:
            projection = project_predictive_target_annulus_velocity(
                ring_input_velocity,
                self._pursuer_velocity[..., :2],
                pursuer_xy,
                self._target_position_local[:, :2],
                self._target_velocity[:, :2],
                active,
                target_acceleration_xy=target_acceleration_xy,
                target_maximum_speed_mps=self.cfg.evader_speed_mps,
                acceleration_prediction_substeps=(
                    self.cfg.predictive_ring_target_acceleration_substeps
                ),
                minimum_radius_m=control_minimum_radius,
                maximum_radius_m=self.cfg.strong_ring_maximum_radius_m,
                prediction_horizon_s=(
                    self.cfg.predictive_ring_prediction_horizon_s
                ),
                command_response_ratio=(
                    self.cfg.predictive_ring_command_response_ratio
                ),
                maximum_speed_mps=self.cfg.training_command_speed_mps,
                projection_iterations=(
                    self.cfg.predictive_ring_projection_iterations
                ),
                response_trajectory_substeps=(
                    self.cfg.predictive_ring_response_trajectory_substeps
                ),
                enforce_minimum_radius=(
                    self.cfg.predictive_ring_enforce_minimum_radius
                ),
                minimum_radius_active=minimum_radius_active,
            )
        final_velocity = projection.velocity_xy
        final_corrected = projection.corrected | (
            preentry_braking.corrected
            if preentry_braking is not None
            else torch.zeros_like(projection.corrected)
        )
        final_correction_norm = (
            projection.velocity_xy - command.desired_velocity_xy_w
        ).flatten(start_dim=1).norm(dim=1)
        final_corrected_constraints = projection.corrected_constraints + (
            preentry_braking.limited_agents
            if preentry_braking is not None
            else torch.zeros_like(projection.corrected_constraints)
        )
        final_minimum_radius = projection.minimum_predicted_radius_m
        final_maximum_radius = projection.maximum_predicted_radius_m
        final_infeasible = projection.infeasible
        if self.cfg.predictive_ring_angular_gap_projection_enabled:
            angular_active = projection.active
            if self.cfg.predictive_ring_angular_gap_preentry_enabled:
                angular_active = angular_active | (
                    phase_ready_for_projection
                    & (
                        current_radius.max(dim=1).values
                        <= self.cfg.predictive_ring_angular_gap_preentry_activation_radius_m
                    )
                )
            angular_projection = project_dynamics_consistent_angular_gap_velocity(
                projection.velocity_xy,
                self._pursuer_velocity[..., :2],
                pursuer_xy,
                self._target_position_local[:, :2],
                self._target_velocity[:, :2],
                angular_active,
                target_acceleration_xy=target_acceleration_xy,
                target_maximum_speed_mps=self.cfg.evader_speed_mps,
                minimum_radius_m=control_minimum_radius,
                maximum_radius_m=self.cfg.strong_ring_maximum_radius_m,
                maximum_gap_rad=self.cfg.strong_ring_maximum_gap_rad,
                maximum_mean_gap_error_rad=(
                    control_mean_gap_limit
                ),
                prediction_horizon_s=(
                    self.cfg.predictive_ring_prediction_horizon_s
                ),
                maximum_speed_mps=self.cfg.training_command_speed_mps,
                low_level_dt_s=self.cfg.sim.dt,
                velocity_kp=self.cfg.predictive_ring_low_level_velocity_kp,
                maximum_planar_acceleration_mps2=(
                    self.cfg.pursuer_max_planar_accel_mps2
                ),
                correction_candidate_count=(
                    self.cfg.predictive_ring_angular_gap_candidate_count
                ),
                minimum_radius_active=minimum_radius_active,
                radial_constraints_active=projection.active,
            )
            final_velocity = angular_projection.velocity_xy
            final_corrected = final_corrected | angular_projection.corrected
            final_correction_norm = (
                final_velocity - command.desired_velocity_xy_w
            ).flatten(start_dim=1).norm(dim=1)
            final_corrected_constraints = (
                projection.corrected_constraints
                + angular_projection.corrected_constraints
            )
            final_minimum_radius = angular_projection.minimum_predicted_radius_m
            final_maximum_radius = angular_projection.maximum_predicted_radius_m
            final_infeasible = angular_projection.infeasible
            if preentry_braking is not None:
                final_infeasible = (
                    final_infeasible
                    | projection.infeasible
                    | preentry_braking.infeasible
                )
            self._angular_gap_projection_active = angular_projection.active
            self._angular_gap_projection_corrected = angular_projection.corrected
            self._angular_gap_projection_correction_norm_mps = (
                angular_projection.correction_norm_mps
            )
            self._angular_gap_projection_max_predicted_gap_rad = (
                angular_projection.maximum_predicted_gap_rad
            )
            self._angular_gap_projection_max_predicted_mean_gap_error_rad = (
                angular_projection.maximum_predicted_mean_gap_error_rad
            )
            self._angular_gap_projection_infeasible = angular_projection.infeasible
        self._ring_projection_active = projection.active
        self._ring_projection_corrected = final_corrected
        self._ring_projection_correction_norm_mps = final_correction_norm
        self._ring_projection_corrected_constraints = final_corrected_constraints
        self._ring_projection_min_predicted_radius_m = final_minimum_radius
        self._ring_projection_max_predicted_radius_m = final_maximum_radius
        self._ring_projection_infeasible = final_infeasible
        return HighLevelCommand(
            desired_velocity_xy_w=final_velocity,
            desired_altitude_m=command.desired_altitude_m,
            desired_yaw_rate_rps=command.desired_yaw_rate_rps,
            desired_speed_mps=final_velocity.norm(dim=-1),
            normalized_action=command.normalized_action,
        )

    def _apply_target_path_guard(self, command, pursuer_xy):
        target = self._estimated_target_xy_for_guidance()
        target_velocity = torch.where(
            self._visibility.team_visible[:, None], self._target_velocity[:, :2],
            torch.zeros_like(self._target_velocity[:, :2]))
        target_acceleration = None
        acceleration_valid = torch.zeros_like(self._visibility.team_visible)
        if self.cfg.target_path_guard_acceleration_enabled:
            target_acceleration, acceleration_valid = observed_target_acceleration(
                target_velocity, self._path_guard_previous_target_velocity,
                self._visibility.team_visible, self._path_guard_previous_visible,
                dt=self.cfg.sim.dt*self.cfg.decimation,
                maximum_acceleration=self.cfg.evader_max_planar_accel_mps2)
        result = joint_target_path_guard(
            command.desired_velocity_xy_w, self._pursuer_velocity[..., :2], pursuer_xy,
            target, target_velocity, self._visibility.team_visible,
            self._obstacle_xy_local, self._obstacle_radius,
            maximum_speed=self.cfg.training_command_speed_mps,
            horizon=(1.0 if self.cfg.target_path_guard_extended_enabled
                     else self.cfg.predictive_ring_prediction_horizon_s), dt=self.cfg.sim.dt,
            velocity_kp=self.cfg.predictive_ring_low_level_velocity_kp,
            acceleration=self.cfg.pursuer_max_planar_accel_mps2,
            target_separation=self.cfg.predictive_target_minimum_center_distance_m,
            agent_separation=self.cfg.predictive_agent_minimum_center_distance_m,
            pursuer_radius=self.cfg.pursuer_radius_m,
            obstacle_clearance=self.cfg.predictive_static_minimum_clearance_m,
            arena_half_extent=.5*self.cfg.arena_size_m,
            boundary_clearance=self.cfg.predictive_boundary_minimum_clearance_m,
            target_acceleration=target_acceleration,
            target_prediction_speed=self.cfg.evader_speed_mps,
            margin_tiebreak=self.cfg.target_path_guard_margin_enabled,
            prediction_reserve_m=(.5 if self.cfg.target_path_guard_reserve_enabled else 0.))
        self._path_guard_corrected_steps += result.corrected.long()
        self._path_guard_telemetry = {
            "target_path_guard_active": result.active,
            "target_path_guard_corrected": result.corrected,
            "target_path_guard_corrected_steps": self._path_guard_corrected_steps,
            "target_path_guard_candidate_index": result.selected_index,
            "target_path_guard_nominal_minimum_m": result.nominal_minimum_m,
            "target_path_guard_selected_minimum_m": result.selected_minimum_m,
            "target_path_guard_selected_infeasible": result.selected_infeasible,
            "target_path_guard_correction_mps": result.maximum_agent_correction_mps,
            "target_path_guard_pre_pursuer_xy_m": pursuer_xy.clone(),
            "target_path_guard_pre_pursuer_velocity_mps": self._pursuer_velocity[..., :2].clone(),
            "target_path_guard_pre_target_xy_m": target.clone(),
            "target_path_guard_pre_target_velocity_mps": target_velocity.clone(),
            "target_path_guard_pre_command_mps": command.desired_velocity_xy_w.clone(),
        }
        if self.cfg.target_path_guard_acceleration_enabled:
            self._path_guard_telemetry.update(
                target_path_guard_pre_target_acceleration_mps2=target_acceleration.clone(),
                target_path_guard_acceleration_valid=acceleration_valid.clone())
            self._path_guard_previous_visible.copy_(self._visibility.team_visible)
            self._path_guard_previous_target_velocity.copy_(target_velocity)
        return HighLevelCommand(
            desired_velocity_xy_w=result.velocity_xy,
            desired_altitude_m=command.desired_altitude_m,
            desired_yaw_rate_rps=command.desired_yaw_rate_rps,
            desired_speed_mps=result.velocity_xy.norm(dim=-1),
            normalized_action=command.normalized_action)

    def _apply_region_joint_lookahead(
        self, command: HighLevelCommand, pursuer_xy: torch.Tensor
    ) -> HighLevelCommand:
        active = self._visibility.visible_each.all(dim=1) & self._orientation_contraction_unlocked
        measured_velocity = torch.where(
            self._visibility.team_visible[:, None], self._target_velocity[:, :2],
            torch.zeros_like(self._target_velocity[:, :2]))
        result = joint_region_lookahead(
            command.desired_velocity_xy_w, self._pursuer_velocity[..., :2], pursuer_xy,
            self._estimated_target_xy_for_guidance(), measured_velocity, active,
            self._obstacle_xy_local, self._obstacle_radius,
            maximum_speed=self.cfg.training_command_speed_mps,
            correction_bound=self.cfg.moving_target_slot_radial_velocity_residual_scale_mps,
            horizon=self.cfg.predictive_ring_prediction_horizon_s,
            low_level_dt=self.cfg.sim.dt,
            velocity_kp=self.cfg.predictive_ring_low_level_velocity_kp,
            acceleration=self.cfg.pursuer_max_planar_accel_mps2,
            sample_dt=self._task_spec.timing.high_level_dt,
            capture_radius=self.cfg.region_capture_radius_m,
            maximum_gap=self.cfg.region_capture_maximum_gap_rad,
            target_separation=self.cfg.predictive_target_minimum_center_distance_m,
            agent_separation=self.cfg.predictive_agent_minimum_center_distance_m,
            pursuer_radius=self.cfg.pursuer_radius_m,
            obstacle_clearance=self.cfg.predictive_static_minimum_clearance_m,
            arena_half_extent=.5*self.cfg.arena_size_m,
            boundary_clearance=self.cfg.predictive_boundary_minimum_clearance_m)
        self._joint_lookahead_active = result.active
        self._joint_lookahead_corrected = result.corrected
        self._joint_lookahead_active_steps += result.active.long()
        self._joint_lookahead_corrected_steps += result.corrected.long()
        self._joint_lookahead_correction_mps = result.maximum_agent_correction_mps
        self._joint_lookahead_nominal_violation = result.nominal_violation
        self._joint_lookahead_selected_violation = result.selected_violation
        return HighLevelCommand(
            desired_velocity_xy_w=result.velocity_xy,
            desired_altitude_m=command.desired_altitude_m,
            desired_yaw_rate_rps=command.desired_yaw_rate_rps,
            desired_speed_mps=result.velocity_xy.norm(dim=-1),
            normalized_action=command.normalized_action)

    def _apply_predictive_agent_projection(
        self, command: HighLevelCommand, pursuer_xy: torch.Tensor
    ) -> HighLevelCommand:
        projection_active = (
            torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            if self.cfg.predictive_agent_always_active
            else self._orientation_contraction_unlocked
        )
        projection = project_predictive_pairwise_velocity(
            command.desired_velocity_xy_w,
            self._pursuer_velocity[..., :2],
            pursuer_xy,
            projection_active,
            minimum_center_distance_m=(
                self.cfg.predictive_agent_minimum_center_distance_m
            ),
            prediction_horizon_s=(
                self.cfg.predictive_agent_prediction_horizon_s
            ),
            command_response_ratio=(
                self.cfg.predictive_agent_command_response_ratio
            ),
            maximum_speed_mps=self.cfg.training_command_speed_mps,
            projection_iterations=(
                self.cfg.predictive_agent_projection_iterations
            ),
            restore_unsafe_buffer=(
                self.cfg.predictive_agent_restore_unsafe_buffer
            ),
        )
        self._safety_projection_active = projection.active
        self._safety_projection_corrected = projection.corrected
        self._safety_projection_correction_norm_mps = (
            projection.correction_norm_mps
        )
        self._safety_projection_pair_corrections = (
            projection.corrected_pair_constraints
        )
        self._safety_projection_min_predicted_center_distance_m = (
            projection.minimum_predicted_center_distance_m
        )
        self._safety_projection_infeasible = projection.infeasible
        return HighLevelCommand(
            desired_velocity_xy_w=projection.velocity_xy,
            desired_altitude_m=command.desired_altitude_m,
            desired_yaw_rate_rps=command.desired_yaw_rate_rps,
            desired_speed_mps=projection.velocity_xy.norm(dim=-1),
            normalized_action=command.normalized_action,
        )

    def _apply_predictive_static_projection(
        self, command: HighLevelCommand, pursuer_xy: torch.Tensor
    ) -> HighLevelCommand:
        projection = project_predictive_static_obstacle_velocity(
            command.desired_velocity_xy_w,
            self._pursuer_velocity[..., :2],
            pursuer_xy,
            self._obstacle_xy_local,
            self._obstacle_radius,
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
            obstacle_velocity_xy=(
                self._obstacle_velocity
                if self.cfg.predictive_static_use_obstacle_velocity
                else torch.zeros_like(self._obstacle_velocity)
            ),
            pursuer_radius_m=self.cfg.pursuer_radius_m,
            minimum_clearance_m=self.cfg.predictive_static_minimum_clearance_m,
            prediction_horizon_s=(
                self.cfg.predictive_static_prediction_horizon_s
            ),
            command_response_ratio=(
                self.cfg.predictive_static_command_response_ratio
            ),
            maximum_speed_mps=self.cfg.training_command_speed_mps,
            projection_iterations=(
                self.cfg.predictive_static_projection_iterations
            ),
        )
        self._obstacle_projection_active = projection.active
        self._obstacle_projection_corrected = projection.corrected
        self._obstacle_projection_correction_norm_mps = (
            projection.correction_norm_mps
        )
        self._obstacle_projection_constraint_corrections = (
            projection.corrected_constraints
        )
        self._obstacle_projection_min_predicted_clearance_m = (
            projection.minimum_predicted_clearance_m
        )
        self._obstacle_projection_infeasible = projection.infeasible
        return HighLevelCommand(
            desired_velocity_xy_w=projection.velocity_xy,
            desired_altitude_m=command.desired_altitude_m,
            desired_yaw_rate_rps=command.desired_yaw_rate_rps,
            desired_speed_mps=projection.velocity_xy.norm(dim=-1),
            normalized_action=command.normalized_action,
        )

    def _apply_predictive_boundary_projection(
        self, command: HighLevelCommand, pursuer_xy: torch.Tensor
    ) -> HighLevelCommand:
        projection = project_predictive_arena_boundary_velocity(
            command.desired_velocity_xy_w,
            self._pursuer_velocity[..., :2],
            pursuer_xy,
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
            arena_half_extent_m=0.5 * self.cfg.arena_size_m,
            minimum_clearance_m=(
                self.cfg.predictive_boundary_minimum_clearance_m
            ),
            prediction_horizon_s=(
                self.cfg.predictive_boundary_prediction_horizon_s
            ),
            command_response_ratio=(
                self.cfg.predictive_boundary_command_response_ratio
            ),
            maximum_speed_mps=self.cfg.training_command_speed_mps,
        )
        # A control step performs several joint projection sweeps.  Keep the
        # complete step-level audit instead of overwriting an early repair with
        # the final (already feasible) sweep.
        self._boundary_projection_active |= projection.active
        self._boundary_projection_corrected |= projection.corrected
        self._boundary_projection_correction_norm_mps += (
            projection.correction_norm_mps
        )
        self._boundary_projection_constraint_corrections += (
            projection.corrected_constraints
        )
        self._boundary_projection_min_predicted_clearance_m = torch.minimum(
            self._boundary_projection_min_predicted_clearance_m,
            projection.minimum_predicted_clearance_m,
        )
        self._boundary_projection_infeasible |= projection.infeasible
        return HighLevelCommand(
            desired_velocity_xy_w=projection.velocity_xy,
            desired_altitude_m=command.desired_altitude_m,
            desired_yaw_rate_rps=command.desired_yaw_rate_rps,
            desired_speed_mps=projection.velocity_xy.norm(dim=-1),
            normalized_action=command.normalized_action,
        )

    def _apply_predictive_target_projection(
        self, command: HighLevelCommand, pursuer_xy: torch.Tensor
    ) -> HighLevelCommand:
        projection_target_xy = self._target_position_local[:, :2]
        projection_target_velocity_xy = self._target_velocity[:, :2]
        projection_active = self._visibility.team_visible
        projection_minimum_center_distance_m = (
            self.cfg.predictive_target_minimum_center_distance_m
        )
        if self.cfg.predictive_target_last_seen_cv_during_loss_enabled:
            (
                projection_target_xy,
                projection_target_velocity_xy,
                projection_active,
            ) = bounded_last_seen_cv_target_state(
                self._target_position_local[:, :2],
                self._target_velocity[:, :2],
                self._last_seen_target_xy - 0.5 * self.cfg.arena_size_m,
                self._last_seen_target_velocity_xy,
                self._visibility.team_visible,
                self._lost_steps,
                policy_dt_s=self.cfg.sim.dt * self.cfg.decimation,
                maximum_loss_time_s=(
                    self.cfg.predictive_target_last_seen_cv_maximum_loss_time_s
                ),
            )
            if self.cfg.predictive_target_last_seen_reachable_tube_enabled:
                loss_time_s = (
                    self._lost_steps.to(projection_target_xy.dtype)
                    * self.cfg.sim.dt
                    * self.cfg.decimation
                ).clamp(
                    max=self.cfg.predictive_target_last_seen_cv_maximum_loss_time_s
                )
                projection_minimum_center_distance_m = (
                    torch.full_like(
                        loss_time_s,
                        self.cfg.predictive_target_minimum_center_distance_m,
                    )
                    + 0.5
                    * self.cfg.evader_max_planar_accel_mps2
                    * loss_time_s.square()
                )
        projection = project_predictive_target_separation_velocity(
            command.desired_velocity_xy_w,
            self._pursuer_velocity[..., :2],
            pursuer_xy,
            projection_target_xy,
            projection_target_velocity_xy,
            projection_active,
            minimum_center_distance_m=projection_minimum_center_distance_m,
            prediction_horizon_s=(
                self.cfg.predictive_target_prediction_horizon_s
            ),
            command_response_ratio=(
                self.cfg.predictive_target_command_response_ratio
            ),
            maximum_speed_mps=self.cfg.training_command_speed_mps,
            projection_iterations=(
                self.cfg.predictive_target_projection_iterations
            ),
            restore_unsafe_buffer=(
                self.cfg.predictive_target_restore_unsafe_buffer
            ),
        )
        self._target_projection_active = projection.active
        self._target_projection_corrected = projection.corrected
        self._target_projection_correction_norm_mps = (
            projection.correction_norm_mps
        )
        self._target_projection_constraint_corrections = (
            projection.corrected_constraints
        )
        self._target_projection_maximum_separation_fallback_constraints = (
            projection.maximum_separation_fallback_constraints
        )
        self._target_projection_min_predicted_center_distance_m = (
            projection.minimum_predicted_center_distance_m
        )
        self._target_projection_minimum_maximum_attainable_center_distance_m = (
            projection.minimum_maximum_attainable_center_distance_m
        )
        self._target_projection_unavoidable = projection.unavoidable
        self._target_projection_unavoidable_constraints = (
            projection.unavoidable_constraints
        )
        self._target_projection_infeasible = projection.infeasible
        return HighLevelCommand(
            desired_velocity_xy_w=projection.velocity_xy,
            desired_altitude_m=command.desired_altitude_m,
            desired_yaw_rate_rps=command.desired_yaw_rate_rps,
            desired_speed_mps=projection.velocity_xy.norm(dim=-1),
            normalized_action=command.normalized_action,
        )

    def _apply_static_tangent_bypass(
        self, command: HighLevelCommand, pursuer_xy: torch.Tensor
    ) -> HighLevelCommand:
        bypass = resolve_static_obstacle_tangent_velocity(
            command.desired_velocity_xy_w,
            self._pursuer_velocity[..., :2],
            pursuer_xy,
            self._estimated_target_xy_for_guidance(),
            self._obstacle_xy_local,
            self._obstacle_radius,
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
            self._tangent_obstacle_index,
            self._tangent_side,
            self._tangent_blocked_steps,
            self._tangent_hold_steps_remaining,
            pursuer_radius_m=self.cfg.pursuer_radius_m,
            minimum_clearance_m=self.cfg.predictive_static_minimum_clearance_m,
            prediction_horizon_s=self.cfg.predictive_static_prediction_horizon_s,
            command_response_ratio=(
                self.cfg.predictive_static_command_response_ratio
            ),
            maximum_speed_mps=self.cfg.training_command_speed_mps,
            activation_steps=self.cfg.static_tangent_activation_steps,
            hold_steps=self.cfg.static_tangent_hold_steps,
            tangent_speed_mps=self.cfg.static_tangent_speed_mps,
            outward_speed_mps=self.cfg.static_tangent_outward_speed_mps,
            release_distance_m=self.cfg.static_tangent_release_distance_m,
        )
        self._tangent_obstacle_index = bypass.obstacle_index
        self._tangent_side = bypass.side
        self._tangent_blocked_steps = bypass.blocked_steps
        self._tangent_hold_steps_remaining = bypass.hold_steps_remaining
        self._tangent_active = bypass.active
        self._tangent_activated = bypass.activated
        self._tangent_min_candidate_clearance_m = (
            bypass.minimum_candidate_clearance_m
        )
        return HighLevelCommand(
            desired_velocity_xy_w=bypass.velocity_xy,
            desired_altitude_m=command.desired_altitude_m,
            desired_yaw_rate_rps=command.desired_yaw_rate_rps,
            desired_speed_mps=bypass.velocity_xy.norm(dim=-1),
            normalized_action=command.normalized_action,
        )

    def _current_contraction_active(self) -> torch.Tensor:
        active = self._orientation_contraction_unlocked
        if self.cfg.orientation_contraction_pause_without_team_ready or (
            self.cfg.hybrid_phase_supervisor_enabled
            and self.cfg.hybrid_require_current_team_ready
        ):
            active = active & self._orientation_aligned_now
        return active

    def _advance_orientation_phase(self) -> None:
        pursuer_xy = self._pursuer_position_local[..., :2]
        target_xy = self._target_position_local[:, :2]
        previously_unlocked = self._orientation_contraction_unlocked.clone()
        distance = (pursuer_xy - target_xy[:, None, :]).norm(dim=-1)
        _, yaw_error = target_facing_yaw_guidance(
            pursuer_xy,
            self._pursuer_yaw,
            self._estimated_target_xy_for_guidance(),
            torch.zeros_like(self._pursuer_yaw),
            proportional_gain=self.cfg.orientation_yaw_gain,
            policy_residual_scale=0.0,
            maximum_yaw_rate_rps=self._task_spec.yaw_rate_max_rps,
        )
        maximum_gap, mean_gap_error = circular_gap_statistics(
            pursuer_xy, target_xy
        )
        if self.cfg.orientation_phase_gate_enabled:
            self._orientation_phase_ready = (
                maximum_gap <= self.cfg.orientation_phase_maximum_gap_rad
            ) & (
                mean_gap_error
                <= self.cfg.orientation_phase_maximum_mean_gap_error_rad
            )
        else:
            self._orientation_phase_ready.fill_(True)
        gated_yaw_error = torch.where(
            self._orientation_phase_ready[:, None],
            yaw_error,
            torch.full_like(
                yaw_error, self.cfg.orientation_yaw_tolerance_rad + 1.0
            ),
        )
        if self.cfg.asynchronous_orientation_gate_enabled:
            (
                self._orientation_staging_reached_each,
                self._orientation_alignment_steps_each,
                self._orientation_staging_reached,
                self._orientation_contraction_unlocked,
                self._orientation_aligned_now,
            ) = advance_asynchronous_orientation_gate(
                distance,
                self._visibility.visible_each,
                gated_yaw_error,
                self._orientation_staging_reached_each,
                self._orientation_alignment_steps_each,
                self._orientation_contraction_unlocked,
                staging_radius_m=self.cfg.orientation_staging_radius_m,
                yaw_tolerance_rad=self.cfg.orientation_yaw_tolerance_rad,
                required_alignment_steps=(
                    self.cfg.orientation_required_alignment_steps
                ),
                require_current_team_ready_to_unlock=(
                    self.cfg.asynchronous_require_current_team_ready
                ),
            )
            self._orientation_alignment_steps = (
                self._orientation_alignment_steps_each.min(dim=1).values
            )
        else:
            (
                self._orientation_staging_reached,
                self._orientation_alignment_steps,
                self._orientation_contraction_unlocked,
                self._orientation_aligned_now,
            ) = advance_orientation_gate(
                distance,
                self._visibility.visible_each,
                gated_yaw_error,
                self._orientation_staging_reached,
                self._orientation_alignment_steps,
                self._orientation_contraction_unlocked,
                staging_radius_m=self.cfg.orientation_staging_radius_m,
                yaw_tolerance_rad=self.cfg.orientation_yaw_tolerance_rad,
                required_alignment_steps=(
                    self.cfg.orientation_required_alignment_steps
                ),
            )
        if self.cfg.orientation_reference_anchor_to_mean_radius_on_unlock:
            newly_unlocked = (
                self._orientation_contraction_unlocked & ~previously_unlocked
            )
            measured_reference_radius = distance.mean(dim=1).clamp(
                min=self.cfg.desired_training_radius_m,
                max=self.cfg.orientation_staging_radius_m,
            )
            self._orientation_reference_radius_m = torch.where(
                newly_unlocked,
                measured_reference_radius,
                self._orientation_reference_radius_m,
            )
        contraction_step = (
            self.cfg.orientation_reference_contraction_speed_mps
            * self._task_spec.timing.high_level_dt
        )
        contracted_reference = (
            self._orientation_reference_radius_m - contraction_step
        ).clamp_min(self.cfg.desired_training_radius_m)
        reference_contraction_active = self._current_contraction_active()
        if self.cfg.orientation_reference_contraction_latched_after_unlock:
            reference_contraction_active = (
                self._orientation_contraction_unlocked
            )
        relative_position = pursuer_xy - target_xy[:, None, :]
        radial_unit = relative_position / distance[..., None].clamp_min(1.0e-6)
        relative_radial_speed = (
            (
                self._pursuer_velocity[..., :2]
                - self._target_velocity[:, None, :2]
            )
            * radial_unit
        ).sum(dim=-1)
        self._reference_contraction_maximum_radius_error_m = (
            (distance - self._orientation_reference_radius_m[:, None])
            .abs()
            .max(dim=1)
            .values
        )
        self._reference_contraction_maximum_relative_radial_speed_mps = (
            relative_radial_speed.abs().max(dim=1).values
        )
        half_ring_band_m = 0.5 * (
            self.cfg.strong_ring_maximum_radius_m
            - self.cfg.strong_ring_minimum_radius_m
        )
        confirmation_duration_s = (
            self.cfg.strong_ring_required_hold_steps
            * self._task_spec.timing.high_level_dt
        )
        maximum_handoff_radial_speed_mps = min(
            self.cfg.orientation_reference_contraction_speed_mps,
            half_ring_band_m / confirmation_duration_s,
        )
        self._reference_contraction_tracking_ready = (
            self._visibility.team_visible
            & self._orientation_phase_ready
            & (
                self._reference_contraction_maximum_radius_error_m
                <= half_ring_band_m
            )
            & (
                self._reference_contraction_maximum_relative_radial_speed_mps
                <= maximum_handoff_radial_speed_mps
            )
        )
        if self.cfg.orientation_reference_contraction_tracking_gate_enabled:
            reference_contraction_active = (
                reference_contraction_active
                & self._reference_contraction_tracking_ready
            )
        if self.cfg.orientation_contraction_pause_without_team_ready:
            # A short loss after handoff pauses contraction at its current
            # radius.  Expanding straight back to the staging ring would erase
            # completed handoff progress and induce an avoidable radial chase.
            self._orientation_reference_radius_m = torch.where(
                reference_contraction_active,
                contracted_reference,
                torch.where(
                    self._orientation_contraction_unlocked,
                    self._orientation_reference_radius_m,
                    torch.full_like(
                        self._orientation_reference_radius_m,
                        self.cfg.orientation_staging_radius_m,
                    ),
                ),
            )
        else:
            self._orientation_reference_radius_m = torch.where(
                reference_contraction_active,
                contracted_reference,
                torch.full_like(
                    self._orientation_reference_radius_m,
                    self.cfg.orientation_staging_radius_m,
                ),
            )
        self._orientation_yaw_error_rad = yaw_error

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self._advance_dynamic_obstacles()
        pursuer_xy = self._pursuer_position_for_control()
        target_xy = (
            self._evader.data.root_pos_w - self.scene.env_origins
        )[:, :2]
        if self.cfg.region_objective_enabled:
            self._previous_region_potential = region_potential(
                pursuer_xy, target_xy,
                radius_m=self.cfg.region_capture_radius_m,
                maximum_gap_rad=self.cfg.region_capture_maximum_gap_rad,
            )
        self._previous_team_distance_m = (
            pursuer_xy - target_xy[:, None, :]
        ).norm(dim=-1).mean(dim=1)
        self._previous_angular_error_rad = circular_gap_error(
            pursuer_xy, target_xy
        )
        self._previous_strong_ring_violation_score = strong_ring_violation_score(
            pursuer_xy,
            target_xy,
            self.cfg.strong_ring_minimum_radius_m,
            self.cfg.strong_ring_maximum_radius_m,
            self.cfg.strong_ring_maximum_gap_rad,
            self.cfg.strong_ring_maximum_mean_gap_error_rad,
        )
        normalized = torch.stack(
            [actions[agent] for agent in self.cfg.possible_agents], dim=1
        ).clamp(-1.0, 1.0)
        self._last_training_action = normalized
        if self.cfg.learning_diagnostics_enabled:
            self._learning_diagnostics = {'learning_diag_action':normalized.detach().clone()}
        command = self._native_adapter.step(normalized)
        self._boundary_projection_active.zero_()
        self._boundary_projection_corrected.zero_()
        self._boundary_projection_correction_norm_mps.zero_()
        self._boundary_projection_constraint_corrections.zero_()
        self._boundary_projection_min_predicted_clearance_m.fill_(torch.inf)
        self._boundary_projection_infeasible.zero_()
        self._ring_projection_active.zero_()
        self._ring_projection_corrected.zero_()
        self._ring_projection_correction_norm_mps.zero_()
        self._ring_projection_corrected_constraints.zero_()
        self._ring_projection_min_predicted_radius_m.fill_(torch.inf)
        self._ring_projection_max_predicted_radius_m.zero_()
        self._ring_projection_infeasible.zero_()
        self._angular_gap_projection_active.zero_()
        self._angular_gap_projection_corrected.zero_()
        self._angular_gap_projection_correction_norm_mps.zero_()
        self._angular_gap_projection_max_predicted_gap_rad.zero_()
        self._angular_gap_projection_max_predicted_mean_gap_error_rad.zero_()
        self._angular_gap_projection_infeasible.zero_()
        self._preentry_braking_active.zero_()
        self._preentry_braking_corrected.zero_()
        self._preentry_braking_correction_norm_mps.zero_()
        self._preentry_braking_limited_agents.zero_()
        self._preentry_braking_minimum_stopping_margin_m.fill_(torch.inf)
        self._preentry_braking_infeasible.zero_()
        if self.cfg.orientation_gate_enabled:
            self._refresh_task_state()
            policy_dt_s = self.cfg.sim.dt * self.cfg.decimation
            raw_target_acceleration = (
                self._target_velocity[:, :2]
                - self._previous_ring_target_velocity_xy
            ) / policy_dt_s
            acceleration_norm = raw_target_acceleration.norm(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            self._ring_target_acceleration_xy = raw_target_acceleration * (
                self.cfg.evader_max_planar_accel_mps2 / acceleration_norm
            ).clamp_max(1.0)
            self._previous_ring_target_velocity_xy.copy_(
                self._target_velocity[:, :2]
            )
            hybrid_rl_ready = self._current_contraction_active()
            if self.cfg.hybrid_rl_activation_on_staging:
                hybrid_rl_ready = self._orientation_staging_reached
            self._hybrid_rl_active = (
                hybrid_rl_ready & self._visibility.team_visible
                if self.cfg.hybrid_phase_supervisor_enabled
                else torch.zeros_like(self._orientation_contraction_unlocked)
            )
            self._hybrid_policy_residual_scale = torch.where(
                self._hybrid_rl_active,
                torch.full_like(
                    self._hybrid_policy_residual_scale,
                    self.cfg.hybrid_encirclement_policy_residual_scale,
                ),
                torch.zeros_like(self._hybrid_policy_residual_scale),
            )
            if self.cfg.hybrid_phase_supervisor_enabled:
                self._hybrid_traditional_steps += (~self._hybrid_rl_active).long()
                self._hybrid_rl_steps += self._hybrid_rl_active.long()
            if self.cfg.moving_target_slot_guidance_enabled:
                command = self._apply_moving_target_slot_guidance(
                    command, pursuer_xy
                )
            command = self._apply_loss_reacquisition_guidance(
                command, pursuer_xy
            )
            command = self._apply_orientation_guidance(command, pursuer_xy)
            if self.cfg.static_tangent_bypass_enabled:
                command = self._apply_static_tangent_bypass(command, pursuer_xy)
            if self.cfg.predictive_ring_stabilization_enabled:
                command = self._apply_predictive_ring_projection(
                    command, pursuer_xy
                )
            if self.cfg.region_joint_lookahead_enabled:
                command = self._apply_region_joint_lookahead(command, pursuer_xy)
            if self.cfg.learning_diagnostics_enabled:
                self._learning_diagnostics['learning_diag_pre_safety_velocity'] = command.desired_velocity_xy_w.detach().clone()
            for _ in range(self.cfg.joint_safety_projection_sweeps):
                if self.cfg.predictive_agent_projection_enabled:
                    command = self._apply_predictive_agent_projection(command, pursuer_xy)
                if (
                    self.cfg.predictive_target_projection_enabled
                    and not self.cfg.predictive_target_projection_last
                ):
                    command = self._apply_predictive_target_projection(
                        command, pursuer_xy
                    )
                if self.cfg.predictive_static_projection_enabled:
                    command = self._apply_predictive_static_projection(
                        command, pursuer_xy
                    )
                if self.cfg.predictive_boundary_projection_enabled:
                    command = self._apply_predictive_boundary_projection(
                        command, pursuer_xy
                    )
                if (
                    self.cfg.predictive_target_projection_enabled
                    and self.cfg.predictive_target_projection_last
                ):
                    command = self._apply_predictive_target_projection(
                        command, pursuer_xy
                    )
            if self.cfg.target_path_guard_enabled:
                command = self._apply_target_path_guard(command, pursuer_xy)
        self._high_level_command = command
        if self.cfg.learning_diagnostics_enabled:
            self._learning_diagnostics['learning_diag_final_velocity'] = command.desired_velocity_xy_w.detach().clone()

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
            desired_velocity.norm(dim=-1) > 1.0e-3,
            desired_yaw_rate,
            torch.zeros_like(desired_yaw_rate),
        )
        self._evader_high_level_command = HighLevelCommand(
            desired_velocity_xy_w=desired_velocity,
            desired_altitude_m=torch.full_like(evader_yaw, self.cfg.fixed_altitude_m),
            desired_yaw_rate_rps=desired_yaw_rate,
            desired_speed_mps=desired_velocity.norm(dim=-1),
            normalized_action=torch.zeros(self.num_envs, 3, device=self.device),
        )

    def _compute_training_capture(self):
        capture = encirclement_capture(
            self._pursuer_position_local[..., :2],
            self._target_position_local[:, :2],
            self.cfg.training_capture_radius_m,
            3,
            self.cfg.max_capture_gap_rad,
        )
        if self.cfg.strong_ring_termination_enabled:
            ring = strong_ring_geometry(
                self._pursuer_position_local[..., :2],
                self._target_position_local[:, :2],
                self.cfg.strong_ring_minimum_radius_m,
                self.cfg.strong_ring_maximum_radius_m,
                self.cfg.strong_ring_maximum_gap_rad,
                self.cfg.strong_ring_maximum_mean_gap_error_rad,
            )
            visible = (
                self._visibility.visible_each.all(dim=1)
                if self.cfg.strong_ring_require_all_visible
                else torch.ones_like(ring.satisfied)
            )
            self._strong_ring_now = (
                ring.satisfied
                & visible
                & self._orientation_contraction_unlocked
            )
            if self._strong_ring_last_counter != self.common_step_counter:
                self._strong_ring_hold_steps = torch.where(
                    self._strong_ring_now,
                    self._strong_ring_hold_steps + 1,
                    torch.zeros_like(self._strong_ring_hold_steps),
                )
                self._strong_ring_last_counter = self.common_step_counter
            self._strong_ring_success = (
                self._strong_ring_hold_steps
                >= self.cfg.strong_ring_required_hold_steps
            )
            self._strong_ring_mean_gap_error_rad = ring.mean_gap_error_rad
            self._strong_ring_minimum_distance_m = ring.min_distance_m
            self._strong_ring_maximum_distance_m = ring.max_distance_m
            if self.cfg.region_capture_enabled:
                return region_capture(
                    self._pursuer_position_local[..., :2],
                    self._target_position_local[:, :2],
                    self._visibility.visible_each,
                    self._orientation_contraction_unlocked,
                    self._contact | self._agent_collision | self._obstacle_collision
                    | self._boundary_invalid | self._height_invalid,
                    radius_m=self.cfg.region_capture_radius_m,
                    maximum_gap_rad=self.cfg.region_capture_maximum_gap_rad,
                    minimum_target_distance_m=self.cfg.region_capture_minimum_target_distance_m,
                )
            return CaptureResult(
                success=self._strong_ring_success,
                inside_count=ring.inside_count,
                max_gap_rad=ring.max_gap_rad,
                min_distance_m=ring.min_distance_m,
            )
        if not self.cfg.orientation_gate_enabled:
            return capture
        return CaptureResult(
            success=capture.success & self._orientation_contraction_unlocked,
            inside_count=capture.inside_count,
            max_gap_rad=capture.max_gap_rad,
            min_distance_m=capture.min_distance_m,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_task_state()
        role_phase_features = None
        if self.cfg.role_phase_observation_enabled:
            estimated_target_xy = self._estimated_target_xy_for_guidance()
            target_bearing = torch.atan2(
                estimated_target_xy[:, None, 1]
                - self._pursuer_position_local[..., 1],
                estimated_target_xy[:, None, 0]
                - self._pursuer_position_local[..., 0],
            )
            yaw_error = wrap_angle(target_bearing - self._pursuer_yaw)
            current_ready_each = (
                (
                    self._pursuer_position_local[..., :2]
                    - estimated_target_xy[:, None, :]
                ).norm(dim=-1)
                <= self.cfg.orientation_staging_radius_m
            ) & self._visibility.team_visible[:, None] & (
                yaw_error.abs() <= self.cfg.orientation_yaw_tolerance_rad
            )
            phase_index = torch.where(
                self._orientation_contraction_unlocked,
                torch.full_like(self._lost_steps, 2),
                torch.where(
                    self._orientation_staging_reached,
                    torch.ones_like(self._lost_steps),
                    torch.zeros_like(self._lost_steps),
                ),
            )
            role_phase_features = build_role_phase_features(
                self._task_spec,
                self._pursuer_position_local[..., :2],
                self._slot_goal_xy,
                self._slot_assignment,
                self._slot_anchor_valid,
                phase_index,
                self._orientation_reference_radius_m,
                current_ready_each.float().mean(dim=1),
                reference_radius_scale_m=(
                    self.cfg.orientation_staging_radius_m
                ),
            )
        observations = build_native_observations(
            self._task_spec,
            self._pursuer_position_local,
            self._pursuer_quaternion,
            self._pursuer_velocity,
            self._pursuer_angular_velocity,
            self._target_position_local,
            self._target_velocity,
            self._visibility.visible_each,
            self._last_seen_target_xy - 0.5 * self.cfg.arena_size_m,
            self._lost_steps,
            self._obstacle_xy_local,
            self._obstacle_radius,
            self._obstacle_velocity,
            role_phase_features,
        )
        if self.cfg.region_capture_enabled:
            remaining = remaining_task_time(self.episode_length_buf, self.max_episode_length)
            observations = torch.cat((observations, remaining[:, None, None].expand(-1, 3, 1)), dim=-1)
        return {
            agent: observations[:, index]
            for index, agent in enumerate(self.cfg.possible_agents)
        }

    def _get_states(self) -> torch.Tensor:
        self._refresh_task_state()
        state = build_native_centralized_state(
            self._task_spec,
            self._pursuer_position_local,
            self._pursuer_quaternion,
            self._pursuer_velocity,
            self._pursuer_angular_velocity,
            self._target_position_local,
            self._target_velocity,
            self._lost_steps,
            self._obstacle_xy_local,
            self._obstacle_radius,
        )
        if self.cfg.region_capture_enabled:
            context = region_critic_context(
                remaining_task_time(self.episode_length_buf, self.max_episode_length),
                self._orientation_reference_radius_m, self.cfg.orientation_staging_radius_m,
                self._orientation_contraction_unlocked, self._orientation_alignment_steps_each,
                self.cfg.orientation_required_alignment_steps,
            )
            state = torch.cat((state, context), dim=-1)
        return state

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        self._refresh_task_state()
        if self.cfg.orientation_gate_enabled and not self.cfg.region_capture_enabled:
            self._advance_orientation_phase()
        self._capture_training = self._compute_training_capture()
        action = getattr(
            self,
            "_last_training_action",
            torch.zeros(self.num_envs, 3, 3, device=self.device),
        )
        desired_radius = (
            self._orientation_reference_radius_m
            if self.cfg.orientation_gate_enabled
            else self.cfg.desired_training_radius_m
        )
        terms = native_team_reward(
            self._pursuer_position_local,
            self._target_position_local,
            self._pursuer_velocity,
            action,
            self._capture_training.success,
            self._contact,
            self._agent_collision,
            self._obstacle_collision,
            self._boundary_invalid,
            self._height_invalid,
            self._visibility.team_visible,
            self._min_static_clearance,
            self._min_pursuer_boundary_clearance,
            previous_team_distance_m=getattr(
                self, "_previous_team_distance_m", None
            ),
            previous_angular_error_rad=getattr(
                self, "_previous_angular_error_rad", None
            ),
            desired_radius_m=desired_radius,
            near_switch_m=self.cfg.training_near_switch_m,
            command_speed_mps=self.cfg.training_command_speed_mps,
            policy_dt_s=self._task_spec.timing.high_level_dt,
            fixed_altitude_m=self.cfg.fixed_altitude_m,
            safe_agent_center_distance_m=self.cfg.tracked_safe_agent_distance_m,
            team_visibility_bonus=(
                0.0 if self.cfg.orientation_gate_enabled else 0.01
            ),
            angular_error_weight=(
                self.cfg.moving_target_angular_error_reward_weight
                if self.cfg.moving_target_slot_guidance_enabled
                else 0.35
            ),
            angular_progress_weight=(
                self.cfg.moving_target_angular_progress_reward_weight
                if self.cfg.moving_target_slot_guidance_enabled
                else 0.0
            ),
        )
        orientation_reward = torch.zeros_like(terms.reward)
        asynchronous_orientation_reward = torch.zeros_like(terms.reward)
        slot_reward = torch.zeros_like(terms.reward)
        slot_progress_reward = torch.zeros_like(terms.reward)
        slot_progress_normalized = torch.zeros_like(terms.reward)
        slot_post_action_error_m = torch.zeros_like(terms.reward)
        strong_ring_step_reward = (
            self.cfg.strong_ring_step_reward_weight
            * self._strong_ring_now.float()
        )
        strong_ring_violation_score_now = strong_ring_violation_score(
            self._pursuer_position_local[..., :2],
            self._target_position_local[:, :2],
            self.cfg.strong_ring_minimum_radius_m,
            self.cfg.strong_ring_maximum_radius_m,
            self.cfg.strong_ring_maximum_gap_rad,
            self.cfg.strong_ring_maximum_mean_gap_error_rad,
        )
        strong_ring_violation_progress_normalized = (
            self._previous_strong_ring_violation_score
            - strong_ring_violation_score_now
        ).clamp(-1.0, 1.0)
        strong_ring_violation_progress_reward = (
            self.cfg.strong_ring_violation_progress_reward_weight
            * strong_ring_violation_progress_normalized
        )
        terms.reward = (
            terms.reward
            + strong_ring_step_reward
            + strong_ring_violation_progress_reward
        )
        projection_reward = torch.zeros_like(terms.reward)
        boundary_projection_reward = torch.zeros_like(terms.reward)
        if self.cfg.orientation_gate_enabled:
            visible_fraction = self._visibility.visible_each.float().mean(dim=1)
            normalized_yaw_error = (
                self._orientation_yaw_error_rad.abs().mean(dim=1) / torch.pi
            )
            orientation_reward = (
                -0.10 * (1.0 - visible_fraction)
                -0.10 * (~self._visibility.visible_each.all(dim=1)).float()
                -0.10 * normalized_yaw_error
                -1.00 * terms.agent_proximity_penalty
            )
            terms.reward = terms.reward + orientation_reward
        if self.cfg.asynchronous_orientation_gate_enabled:
            if self.cfg.asynchronous_current_readiness_reward_enabled:
                distance_to_target = (
                    self._pursuer_position_local[..., :2]
                    - self._target_position_local[:, None, :2]
                ).norm(dim=-1)
                current_ready_each = (
                    (distance_to_target <= self.cfg.orientation_staging_radius_m)
                    & self._visibility.visible_each
                    & (
                        self._orientation_yaw_error_rad.abs()
                        <= self.cfg.orientation_yaw_tolerance_rad
                    )
                )
                staging_fraction = current_ready_each.float().mean(dim=1)
            else:
                staging_fraction = (
                    self._orientation_staging_reached_each.float().mean(dim=1)
                )
            alignment_fraction = (
                self._orientation_alignment_steps_each.float()
                / float(self.cfg.orientation_required_alignment_steps)
            ).clamp(max=1.0).mean(dim=1)
            asynchronous_progress = 0.5 * (
                staging_fraction + alignment_fraction
            )
            asynchronous_orientation_reward = (
                self.cfg.asynchronous_orientation_progress_reward_weight
                * (
                    asynchronous_progress
                    - self._previous_asynchronous_orientation_progress
                )
            )
            self._previous_asynchronous_orientation_progress = (
                asynchronous_progress.detach()
            )
            terms.reward = terms.reward + asynchronous_orientation_reward
        if self.cfg.moving_target_slot_guidance_enabled:
            slot_reward = (
                -self.cfg.moving_target_slot_error_reward_weight
                * self._slot_guidance_active.float()
                * (
                    (
                        self._slot_max_error_m
                        if self.cfg.moving_target_slot_error_use_maximum_error
                        else self._slot_mean_error_m
                    )
                    / self._orientation_reference_radius_m.clamp_min(1.0)
                ).clamp(max=2.0)
            )
            terms.reward = terms.reward + slot_reward
            slot_progress = slot_goal_progress_reward(
                (
                    self._slot_max_error_m
                    if self.cfg.moving_target_slot_progress_use_maximum_error
                    else self._slot_mean_error_m
                ),
                self._pursuer_position_local[..., :2],
                self._slot_goal_xy,
                self._slot_guidance_active,
                command_speed_mps=self.cfg.training_command_speed_mps,
                policy_dt_s=self._task_spec.timing.high_level_dt,
                reward_weight=(
                    self.cfg.moving_target_slot_progress_reward_weight
                ),
                use_maximum_error=(
                    self.cfg.moving_target_slot_progress_use_maximum_error
                ),
            )
            slot_progress_reward = slot_progress.reward
            slot_progress_normalized = slot_progress.normalized_progress
            slot_post_action_error_m = slot_progress.post_action_mean_error_m
            terms.reward = terms.reward + slot_progress_reward
        if self.cfg.predictive_agent_projection_enabled:
            projection_reward = predictive_projection_reward(
                self._safety_projection_correction_norm_mps,
                self._safety_projection_infeasible,
                command_speed_mps=self.cfg.training_command_speed_mps,
                correction_weight=(
                    self.cfg.predictive_agent_correction_reward_weight
                ),
                infeasible_penalty=(
                    self.cfg.predictive_agent_infeasible_reward_penalty
                ),
            )
            terms.reward = terms.reward + projection_reward
        if self.cfg.predictive_boundary_projection_enabled:
            boundary_projection_reward = predictive_projection_reward(
                self._boundary_projection_correction_norm_mps,
                self._boundary_projection_infeasible,
                command_speed_mps=self.cfg.training_command_speed_mps,
                correction_weight=(
                    self.cfg.predictive_boundary_correction_reward_weight
                ),
                infeasible_penalty=(
                    self.cfg.predictive_boundary_infeasible_reward_penalty
                ),
            )
            terms.reward = terms.reward + boundary_projection_reward
        region_shaping = torch.zeros_like(terms.reward)
        if self.cfg.region_objective_enabled:
            unsafe = (self._contact | self._agent_collision | self._obstacle_collision
                      | self._boundary_invalid | self._height_invalid)
            if self.cfg.target_buffer_failure_enabled:
                unsafe = unsafe | self._target_buffer_failure
            potential = region_potential(
                self._pursuer_position_local[..., :2], self._target_position_local[:, :2],
                radius_m=self.cfg.region_capture_radius_m,
                maximum_gap_rad=self.cfg.region_capture_maximum_gap_rad,
            )
            # Replace the complete legacy task reward, including slot/strong-
            # ring progress; retain physical safety, height and action costs.
            terms.reward, region_shaping = region_task_reward(
                self._previous_region_potential, potential,
                self._capture_training.success, unsafe,
                self.episode_length_buf >= self.max_episode_length - 1,
                safety_cost=(2.0 * terms.agent_proximity_penalty
                             + 0.75 * terms.obstacle_proximity_penalty
                             + 0.50 * terms.boundary_proximity_penalty),
                height_cost=0.10 * terms.height_error_m + 0.02 * terms.vertical_speed_mps,
                action_cost=0.01 * terms.action_energy,
            )
            terms.reward = terms.reward + projection_reward + boundary_projection_reward
        training_telemetry = {
            **{key: value.detach().clone() for key, value in self._path_guard_telemetry.items()},
            "joint_lookahead_active": self._joint_lookahead_active.detach().clone(),
            "joint_lookahead_corrected": self._joint_lookahead_corrected.detach().clone(),
            "joint_lookahead_active_steps": self._joint_lookahead_active_steps.detach().clone(),
            "joint_lookahead_corrected_steps": self._joint_lookahead_corrected_steps.detach().clone(),
            "joint_lookahead_correction_mps": self._joint_lookahead_correction_mps.detach().clone(),
            "joint_lookahead_nominal_violation": self._joint_lookahead_nominal_violation.detach().clone(),
            "joint_lookahead_selected_violation": self._joint_lookahead_selected_violation.detach().clone(),
            "region_shaping_reward": region_shaping.detach().clone(),
            "dual_observer_active": self._dual_observer_active.detach().clone(),
            "dual_observer_active_steps": self._dual_observer_active_steps.detach().clone(),
            "dual_observer_role_switches": self._dual_observer_role_switches.detach().clone(),
            "dual_observer_pair_angle_rad": self._dual_observer_pair_angle_rad.detach().clone(),
            "region_capture_enabled": torch.full_like(
                self._capture_training.success, self.cfg.region_capture_enabled
            ),
            "success_training": self._capture_training.success.detach().clone(),
            "strong_ring_enabled": torch.full_like(
                self._capture_training.success,
                self.cfg.strong_ring_termination_enabled,
            ),
            "strong_ring_now": self._strong_ring_now.detach().clone(),
            "strong_ring_success": self._strong_ring_success.detach().clone(),
            "strong_ring_hold_steps": self._strong_ring_hold_steps.detach().clone(),
            "strong_ring_confirmation_stabilization_active": (
                self._strong_ring_confirmation_stabilization_active.detach().clone()
            ),
            "slot_preentry_radial_priority_active": (
                self._slot_preentry_radial_priority_active.detach().clone()
            ),
            "ring_projection_active": self._ring_projection_active.detach().clone(),
            "ring_projection_activation_latched": (
                self._ring_projection_activation_latched.detach().clone()
            ),
            "ring_projection_corrected": (
                self._ring_projection_corrected.detach().clone()
            ),
            "ring_projection_correction_norm_mps": (
                self._ring_projection_correction_norm_mps.detach().clone()
            ),
            "ring_projection_corrected_constraints": (
                self._ring_projection_corrected_constraints.detach().clone()
            ),
            "ring_projection_min_predicted_radius_m": (
                self._ring_projection_min_predicted_radius_m.detach().clone()
            ),
            "ring_projection_max_predicted_radius_m": (
                self._ring_projection_max_predicted_radius_m.detach().clone()
            ),
            "ring_projection_infeasible": (
                self._ring_projection_infeasible.detach().clone()
            ),
            "angular_gap_projection_active": (
                self._angular_gap_projection_active.detach().clone()
            ),
            "angular_gap_projection_corrected": (
                self._angular_gap_projection_corrected.detach().clone()
            ),
            "angular_gap_projection_correction_norm_mps": (
                self._angular_gap_projection_correction_norm_mps.detach().clone()
            ),
            "angular_gap_projection_max_predicted_gap_rad": (
                self._angular_gap_projection_max_predicted_gap_rad.detach().clone()
            ),
            "angular_gap_projection_max_predicted_mean_gap_error_rad": (
                self._angular_gap_projection_max_predicted_mean_gap_error_rad
                .detach()
                .clone()
            ),
            "angular_gap_projection_infeasible": (
                self._angular_gap_projection_infeasible.detach().clone()
            ),
            "preentry_braking_active": (
                self._preentry_braking_active.detach().clone()
            ),
            "preentry_braking_corrected": (
                self._preentry_braking_corrected.detach().clone()
            ),
            "preentry_braking_correction_norm_mps": (
                self._preentry_braking_correction_norm_mps.detach().clone()
            ),
            "preentry_braking_limited_agents": (
                self._preentry_braking_limited_agents.detach().clone()
            ),
            "preentry_braking_minimum_stopping_margin_m": (
                self._preentry_braking_minimum_stopping_margin_m.detach().clone()
            ),
            "preentry_braking_infeasible": (
                self._preentry_braking_infeasible.detach().clone()
            ),
            "ring_target_acceleration_mps2": (
                self._ring_target_acceleration_xy.norm(dim=-1).detach().clone()
            ),
            "strong_ring_mean_gap_error_rad": (
                self._strong_ring_mean_gap_error_rad.detach().clone()
            ),
            "strong_ring_minimum_distance_m": (
                self._strong_ring_minimum_distance_m.detach().clone()
            ),
            "strong_ring_maximum_distance_m": (
                self._strong_ring_maximum_distance_m.detach().clone()
            ),
            "training_capture_radius_m": torch.full_like(
                terms.team_distance_m, self.cfg.training_capture_radius_m
            ),
            "team_distance_m": terms.team_distance_m.detach().clone(),
            "distance_progress_normalized": (
                terms.distance_progress_normalized.detach().clone()
            ),
            "near_gate": terms.near_gate.detach().clone(),
            "radial_error_m": terms.radial_error_m.detach().clone(),
            "radial_error_normalized": (
                terms.radial_error_normalized.detach().clone()
            ),
            "angular_error_rad": terms.angular_error_rad.detach().clone(),
            "angular_error_normalized": (
                terms.angular_error_normalized.detach().clone()
            ),
            "angular_progress_normalized": (
                terms.angular_progress_normalized.detach().clone()
            ),
            "height_error_m": terms.height_error_m.detach().clone(),
            "vertical_speed_mps": terms.vertical_speed_mps.detach().clone(),
            "agent_proximity_penalty": terms.agent_proximity_penalty.detach().clone(),
            "obstacle_proximity_penalty": (
                terms.obstacle_proximity_penalty.detach().clone()
            ),
            "boundary_proximity_penalty": (
                terms.boundary_proximity_penalty.detach().clone()
            ),
            "action_energy": terms.action_energy.detach().clone(),
            "orientation_reward": orientation_reward.detach().clone(),
            "asynchronous_orientation_reward": (
                asynchronous_orientation_reward.detach().clone()
            ),
            "slot_guidance_reward": slot_reward.detach().clone(),
            "slot_progress_reward": slot_progress_reward.detach().clone(),
            "slot_progress_normalized": slot_progress_normalized.detach().clone(),
            "strong_ring_step_reward": strong_ring_step_reward.detach().clone(),
            "strong_ring_violation_score": (
                strong_ring_violation_score_now.detach().clone()
            ),
            "strong_ring_violation_progress_normalized": (
                strong_ring_violation_progress_normalized.detach().clone()
            ),
            "strong_ring_violation_progress_reward": (
                strong_ring_violation_progress_reward.detach().clone()
            ),
            "slot_post_action_error_m": slot_post_action_error_m.detach().clone(),
            "safety_projection_reward": projection_reward.detach().clone(),
            "boundary_projection_reward": (
                boundary_projection_reward.detach().clone()
            ),
            "orientation_staging_reached": (
                self._orientation_staging_reached.detach().clone()
            ),
            "asynchronous_orientation_gate_enabled": torch.full_like(
                self._capture_training.success,
                self.cfg.asynchronous_orientation_gate_enabled,
            ),
            "orientation_staging_fraction": (
                self._orientation_staging_reached_each.float()
                .mean(dim=1)
                .detach()
                .clone()
            ),
            "orientation_alignment_fraction": (
                (
                    self._orientation_alignment_steps_each.float()
                    / float(self.cfg.orientation_required_alignment_steps)
                )
                .clamp(max=1.0)
                .mean(dim=1)
                .detach()
                .clone()
            ),
            "orientation_alignment_steps": (
                self._orientation_alignment_steps.detach().clone()
            ),
            "orientation_contraction_unlocked": (
                self._orientation_contraction_unlocked.detach().clone()
            ),
            "orientation_aligned_now": self._orientation_aligned_now.detach().clone(),
            "orientation_phase_ready": (
                self._orientation_phase_ready.detach().clone()
            ),
            "orientation_all_visible": (
                self._visibility.visible_each.all(dim=1).detach().clone()
            ),
            "orientation_yaw_error_max_rad": (
                self._orientation_yaw_error_rad.abs().max(dim=1).values.detach().clone()
            ),
            "orientation_reference_radius_m": (
                self._orientation_reference_radius_m.detach().clone()
            ),
            "reference_contraction_tracking_ready": (
                self._reference_contraction_tracking_ready.detach().clone()
            ),
            "reference_contraction_maximum_radius_error_m": (
                self._reference_contraction_maximum_radius_error_m.detach().clone()
            ),
            "reference_contraction_maximum_relative_radial_speed_mps": (
                self._reference_contraction_maximum_relative_radial_speed_mps
                .detach()
                .clone()
            ),
            "slot_guidance_active": self._slot_guidance_active.detach().clone(),
            "slot_assignment_switched": (
                self._slot_assignment_switched.detach().clone()
            ),
            "slot_handoff_count": self._slot_handoff_count.detach().clone(),
            "slot_active_steps": self._slot_active_steps.detach().clone(),
            "slot_mean_error_m": self._slot_mean_error_m.detach().clone(),
            "slot_max_error_m": self._slot_max_error_m.detach().clone(),
            "hybrid_phase_supervisor_enabled": torch.full_like(
                self._capture_training.success,
                self.cfg.hybrid_phase_supervisor_enabled,
            ),
            "hybrid_rl_active": self._hybrid_rl_active.detach().clone(),
            "hybrid_traditional_steps": (
                self._hybrid_traditional_steps.detach().clone()
            ),
            "hybrid_rl_steps": self._hybrid_rl_steps.detach().clone(),
            "hybrid_policy_residual_scale": (
                self._hybrid_policy_residual_scale.detach().clone()
            ),
            "slot_policy_residual_active_steps": (
                self._slot_policy_residual_active_steps.detach().clone()
            ),
            "slot_policy_residual_norm_sum_mps": (
                self._slot_policy_residual_norm_sum_mps.detach().clone()
            ),
            "slot_policy_residual_radial_support_sum_mps": (
                self._slot_policy_residual_radial_support_sum_mps.detach().clone()
            ),
            "slot_policy_residual_radial_opposition_sum_mps": (
                self._slot_policy_residual_radial_opposition_sum_mps.detach().clone()
            ),
            "slot_policy_residual_phase_support_sum_mps": (
                self._slot_policy_residual_phase_support_sum_mps.detach().clone()
            ),
            "slot_policy_residual_phase_opposition_sum_mps": (
                self._slot_policy_residual_phase_opposition_sum_mps.detach().clone()
            ),
            "slot_speed_limit_active_agent_fraction_sum": (
                self._slot_speed_limit_active_agent_fraction_sum.detach().clone()
            ),
            "slot_speed_limit_mean_scale_sum": (
                self._slot_speed_limit_mean_scale_sum.detach().clone()
            ),
            "slot_speed_limit_forward_loss_sum_mps": (
                self._slot_speed_limit_forward_loss_sum_mps.detach().clone()
            ),
            "slot_requested_tangential_speed_sum_mps": (
                self._slot_requested_tangential_speed_sum_mps.detach().clone()
            ),
            "slot_executed_tangential_speed_sum_mps": (
                self._slot_executed_tangential_speed_sum_mps.detach().clone()
            ),
            "slot_anchor_heading_rad": (
                self._slot_anchor_heading_rad.detach().clone()
            ),
            "slot_assignment": self._slot_assignment.detach().clone(),
            "loss_reacquisition_guidance_enabled": torch.full_like(
                self._capture_training.success,
                self.cfg.loss_reacquisition_guidance_enabled,
            ),
            "loss_reacquisition_active": (
                self._loss_reacquisition_active.detach().clone()
            ),
            "loss_reacquisition_guidance_active": (
                self._loss_reacquisition_guidance_active.detach().clone()
            ),
            "loss_reacquisition_activated": (
                self._loss_reacquisition_activated.detach().clone()
            ),
            "loss_reacquisition_just_reacquired": (
                self._loss_reacquisition_just_reacquired.detach().clone()
            ),
            "loss_reacquisition_scout_index": (
                self._loss_reacquisition_scout_index.detach().clone()
            ),
            "loss_reacquisition_mean_goal_distance_m": (
                self._loss_reacquisition_mean_goal_distance_m.detach().clone()
            ),
            "loss_reacquisition_events": (
                self._loss_reacquisition_events.detach().clone()
            ),
            "loss_reacquisition_completed": (
                self._loss_reacquisition_completed.detach().clone()
            ),
            "loss_reacquisition_active_steps": (
                self._loss_reacquisition_active_steps.detach().clone()
            ),
            "loss_reacquisition_completed_duration_sum_steps": (
                self._loss_reacquisition_completed_duration_sum_steps.detach().clone()
            ),
            "loss_reacquisition_max_segment_steps": (
                self._loss_reacquisition_max_segment_steps.detach().clone()
            ),
            "loss_reacquisition_first_completed": (
                self._loss_reacquisition_first_completed.detach().clone()
            ),
            "loss_reacquisition_first_delay_steps": (
                self._loss_reacquisition_first_delay_steps.detach().clone()
            ),
            "loss_reacquisition_flank_enabled": torch.full_like(
                self._capture_training.success,
                self.cfg.loss_reacquisition_occlusion_flank_enabled,
            ),
            "loss_reacquisition_flank_waypoint_valid": (
                self._loss_reacquisition_flank_waypoint_valid.detach().clone()
            ),
            "loss_reacquisition_flank_guidance_active": (
                self._loss_reacquisition_flank_guidance_active.detach().clone()
            ),
            "loss_reacquisition_flank_fallback_active": (
                self._loss_reacquisition_flank_fallback_active.detach().clone()
            ),
            "loss_reacquisition_flank_blocker_index": (
                self._loss_reacquisition_flank_blocker_index.detach().clone()
            ),
            "loss_reacquisition_flank_events": (
                self._loss_reacquisition_flank_events.detach().clone()
            ),
            "loss_reacquisition_flank_active_steps": (
                self._loss_reacquisition_flank_active_steps.detach().clone()
            ),
            "loss_reacquisition_flank_fallback_steps": (
                self._loss_reacquisition_flank_fallback_steps.detach().clone()
            ),
            "loss_reacquisition_flank_minimum_path_clearance_m": (
                self._loss_reacquisition_flank_minimum_path_clearance_m
                .detach()
                .clone()
            ),
            "loss_reacquisition_flank_minimum_los_clearance_m": (
                self._loss_reacquisition_flank_minimum_los_clearance_m
                .detach()
                .clone()
            ),
            "loss_reacquisition_sector_enabled": torch.full_like(
                self._capture_training.success,
                self.cfg.loss_reacquisition_reachable_sector_enabled,
            ),
            "loss_reacquisition_sector_waypoint_valid": (
                self._loss_reacquisition_sector_waypoint_valid.detach().clone()
            ),
            "loss_reacquisition_sector_guidance_active": (
                self._loss_reacquisition_sector_guidance_active.detach().clone()
            ),
            "loss_reacquisition_sector_fallback_active": (
                self._loss_reacquisition_sector_fallback_active.detach().clone()
            ),
            "loss_reacquisition_sector_phase_index": (
                self._loss_reacquisition_sector_phase_index.detach().clone()
            ),
            "loss_reacquisition_sector_half_angle_rad": (
                self._loss_reacquisition_sector_half_angle_rad.detach().clone()
            ),
            "loss_reacquisition_sector_radius_m": (
                self._loss_reacquisition_sector_radius_m.detach().clone()
            ),
            "loss_reacquisition_sector_valid_steps": (
                self._loss_reacquisition_sector_valid_steps.detach().clone()
            ),
            "loss_reacquisition_sector_fallback_steps": (
                self._loss_reacquisition_sector_fallback_steps.detach().clone()
            ),
            "loss_reacquisition_sector_no_path_candidate_steps": (
                self._loss_reacquisition_sector_no_path_candidate_steps
                .detach()
                .clone()
            ),
            "loss_reacquisition_sector_no_boundary_candidate_steps": (
                self._loss_reacquisition_sector_no_boundary_candidate_steps
                .detach()
                .clone()
            ),
            "loss_reacquisition_sector_minimum_path_clearance_m": (
                self._loss_reacquisition_sector_minimum_path_clearance_m
                .detach()
                .clone()
            ),
            "reachable_intercept_guidance_enabled": torch.full_like(
                self._capture_training.success,
                self.cfg.reachable_intercept_guidance_enabled,
            ),
            "intercept_horizon_s": self._intercept_horizon_s.detach().clone(),
            "intercept_reachable_fraction": (
                self._intercept_reachable_fraction.detach().clone()
            ),
            "intercept_mean_gap_m": self._intercept_mean_gap_m.detach().clone(),
            "intercept_blocker_agent_index": (
                self._intercept_blocker_agent_index.detach().clone()
            ),
            "intercept_blocker_reachable": (
                self._intercept_blocker_reachable.detach().clone()
            ),
            "boundary_projection_active": (
                self._boundary_projection_active.detach().clone()
            ),
            "boundary_projection_corrected": (
                self._boundary_projection_corrected.detach().clone()
            ),
            "boundary_projection_correction_norm_mps": (
                self._boundary_projection_correction_norm_mps.detach().clone()
            ),
            "boundary_projection_constraint_corrections": (
                self._boundary_projection_constraint_corrections.detach().clone()
            ),
            "boundary_projection_min_predicted_clearance_m": (
                self._boundary_projection_min_predicted_clearance_m.detach().clone()
            ),
            "boundary_projection_infeasible": (
                self._boundary_projection_infeasible.detach().clone()
            ),
        }
        for agent in self.cfg.possible_agents:
            self.extras[agent]["p5_training"] = training_telemetry
        return {agent: terms.reward for agent in self.cfg.possible_agents}

    def _get_dones(self):
        terminated, time_outs = super()._get_dones()
        # Base dones refreshes the actual post-physics state. Preserve this
        # event through reward calculation and publish a clone before reset.
        self._target_buffer_failure = target_buffer_failure(
            self._pursuer_position_local[..., :2], self._target_position_local[:, :2],
            self.cfg.region_capture_minimum_target_distance_m,
        ) if self.cfg.target_buffer_failure_enabled else torch.zeros_like(self._contact)
        # Isaac calls dones before rewards. For first-hit region capture,
        # update the phase once here so the payout and termination agree.
        if self.cfg.region_capture_enabled and self.cfg.orientation_gate_enabled:
            self._advance_orientation_phase()
        self._capture_training = self._compute_training_capture()
        if self.cfg.orientation_gate_enabled:
            safety_terminal = (
                self._contact
                | self._agent_collision
                | self._obstacle_collision
                | self._boundary_invalid
                | self._height_invalid
            )
        for agent in self.cfg.possible_agents:
            if self.cfg.orientation_gate_enabled:
                terminated[agent] = safety_terminal | self._capture_training.success
            else:
                terminated[agent] = terminated[agent] | self._capture_training.success
            terminated[agent] = terminated[agent] | self._target_buffer_failure
            if self.cfg.target_buffer_failure_enabled:
                self.extras[agent]["p5"]["target_buffer_failure"] = self._target_buffer_failure.clone()
            self.extras[agent]["p5"]["success_training"] = (
                self._capture_training.success.clone()
            )
            self.extras[agent]["p5"]["training_capture_radius_m"] = torch.full_like(
                self._capture_training.min_distance_m,
                self.cfg.training_capture_radius_m,
            )
            self.extras[agent]["p5"].update(
                {
                    "pursuer_target_distance_m": (
                        self._pursuer_position_local[..., :2]
                        - self._target_position_local[:, None, :2]
                    ).norm(dim=-1),
                    "pursuer_position_xy_m": (
                        self._pursuer_position_local[..., :2].clone()
                    ),
                    "pursuer_velocity_xy_mps": (
                        self._pursuer_velocity[..., :2].clone()
                    ),
                    "commanded_velocity_xy_mps": (
                        self._high_level_command.desired_velocity_xy_w.clone()
                    ),
                    "target_position_xy_m": (
                        self._target_position_local[:, :2].clone()
                    ),
                    "target_velocity_xy_mps": (
                        self._target_velocity[:, :2].clone()
                    ),
                    "target_acceleration_xy_mps2": (
                        self._ring_target_acceleration_xy.clone()
                    ),
                    "training_capture_inside_count": (
                        self._capture_training.inside_count.clone()
                    ),
                    "training_capture_max_gap_rad": (
                        self._capture_training.max_gap_rad.clone()
                    ),
                    "strong_ring_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.strong_ring_termination_enabled,
                    ),
                    "strong_ring_now": self._strong_ring_now.clone(),
                    "strong_ring_success": self._strong_ring_success.clone(),
                    "strong_ring_hold_steps": self._strong_ring_hold_steps.clone(),
                    "strong_ring_confirmation_stabilization_active": (
                        self._strong_ring_confirmation_stabilization_active.clone()
                    ),
                    "slot_preentry_radial_priority_active": (
                        self._slot_preentry_radial_priority_active.clone()
                    ),
                    "ring_projection_active": self._ring_projection_active.clone(),
                    "ring_projection_activation_latched": (
                        self._ring_projection_activation_latched.clone()
                    ),
                    "ring_projection_corrected": (
                        self._ring_projection_corrected.clone()
                    ),
                    "ring_projection_correction_norm_mps": (
                        self._ring_projection_correction_norm_mps.clone()
                    ),
                    "ring_projection_corrected_constraints": (
                        self._ring_projection_corrected_constraints.clone()
                    ),
                    "ring_projection_min_predicted_radius_m": (
                        self._ring_projection_min_predicted_radius_m.clone()
                    ),
                    "ring_projection_max_predicted_radius_m": (
                        self._ring_projection_max_predicted_radius_m.clone()
                    ),
                    "ring_projection_infeasible": (
                        self._ring_projection_infeasible.clone()
                    ),
                    "angular_gap_projection_active": (
                        self._angular_gap_projection_active.clone()
                    ),
                    "angular_gap_projection_corrected": (
                        self._angular_gap_projection_corrected.clone()
                    ),
                    "angular_gap_projection_correction_norm_mps": (
                        self._angular_gap_projection_correction_norm_mps.clone()
                    ),
                    "angular_gap_projection_max_predicted_gap_rad": (
                        self._angular_gap_projection_max_predicted_gap_rad.clone()
                    ),
                    "angular_gap_projection_max_predicted_mean_gap_error_rad": (
                        self._angular_gap_projection_max_predicted_mean_gap_error_rad
                        .clone()
                    ),
                    "angular_gap_projection_infeasible": (
                        self._angular_gap_projection_infeasible.clone()
                    ),
                    "preentry_braking_active": (
                        self._preentry_braking_active.clone()
                    ),
                    "preentry_braking_corrected": (
                        self._preentry_braking_corrected.clone()
                    ),
                    "preentry_braking_correction_norm_mps": (
                        self._preentry_braking_correction_norm_mps.clone()
                    ),
                    "preentry_braking_limited_agents": (
                        self._preentry_braking_limited_agents.clone()
                    ),
                    "preentry_braking_minimum_stopping_margin_m": (
                        self._preentry_braking_minimum_stopping_margin_m.clone()
                    ),
                    "preentry_braking_infeasible": (
                        self._preentry_braking_infeasible.clone()
                    ),
                    "ring_target_acceleration_mps2": (
                        self._ring_target_acceleration_xy.norm(dim=-1).clone()
                    ),
                    "strong_ring_mean_gap_error_rad": (
                        self._strong_ring_mean_gap_error_rad.clone()
                    ),
                    "strong_ring_minimum_distance_m": (
                        self._strong_ring_minimum_distance_m.clone()
                    ),
                    "strong_ring_maximum_distance_m": (
                        self._strong_ring_maximum_distance_m.clone()
                    ),
                    "orientation_gate_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.orientation_gate_enabled,
                    ),
                    "orientation_staging_reached": (
                        self._orientation_staging_reached.clone()
                    ),
                    "asynchronous_orientation_gate_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.asynchronous_orientation_gate_enabled,
                    ),
                    "orientation_staging_fraction": (
                        self._orientation_staging_reached_each.float()
                        .mean(dim=1)
                        .clone()
                    ),
                    "orientation_alignment_fraction": (
                        (
                            self._orientation_alignment_steps_each.float()
                            / float(self.cfg.orientation_required_alignment_steps)
                        )
                        .clamp(max=1.0)
                        .mean(dim=1)
                        .clone()
                    ),
                    "orientation_contraction_unlocked": (
                        self._orientation_contraction_unlocked.clone()
                    ),
                    "orientation_phase_ready": (
                        self._orientation_phase_ready.clone()
                    ),
                    "orientation_all_visible": (
                        self._visibility.visible_each.all(dim=1).clone()
                    ),
                    "orientation_alignment_steps": (
                        self._orientation_alignment_steps.clone()
                    ),
                    "orientation_yaw_error_max_rad": (
                        self._orientation_yaw_error_rad.abs().max(dim=1).values.clone()
                    ),
                    "orientation_reference_radius_m": (
                        self._orientation_reference_radius_m.clone()
                    ),
                    "reference_contraction_tracking_ready": (
                        self._reference_contraction_tracking_ready.clone()
                    ),
                    "reference_contraction_maximum_radius_error_m": (
                        self._reference_contraction_maximum_radius_error_m.clone()
                    ),
                    "reference_contraction_maximum_relative_radial_speed_mps": (
                        self._reference_contraction_maximum_relative_radial_speed_mps.clone()
                    ),
                    "slot_guidance_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.moving_target_slot_guidance_enabled,
                    ),
                    "slot_guidance_active": self._slot_guidance_active.clone(),
                    "slot_assignment_switched": (
                        self._slot_assignment_switched.clone()
                    ),
                    "slot_handoff_count": self._slot_handoff_count.clone(),
                    "slot_active_steps": self._slot_active_steps.clone(),
                    "slot_mean_error_m": self._slot_mean_error_m.clone(),
                    "slot_max_error_m": self._slot_max_error_m.clone(),
                    "hybrid_phase_supervisor_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.hybrid_phase_supervisor_enabled,
                    ),
                    "hybrid_rl_active": self._hybrid_rl_active.clone(),
                    "hybrid_traditional_steps": (
                        self._hybrid_traditional_steps.clone()
                    ),
                    "hybrid_rl_steps": self._hybrid_rl_steps.clone(),
                    "hybrid_policy_residual_scale": (
                        self._hybrid_policy_residual_scale.clone()
                    ),
                    "slot_policy_residual_active_steps": (
                        self._slot_policy_residual_active_steps.clone()
                    ),
                    "slot_policy_residual_norm_sum_mps": (
                        self._slot_policy_residual_norm_sum_mps.clone()
                    ),
                    "slot_policy_residual_radial_support_sum_mps": (
                        self._slot_policy_residual_radial_support_sum_mps.clone()
                    ),
                    "slot_policy_residual_radial_opposition_sum_mps": (
                        self._slot_policy_residual_radial_opposition_sum_mps.clone()
                    ),
                    "slot_policy_residual_phase_support_sum_mps": (
                        self._slot_policy_residual_phase_support_sum_mps.clone()
                    ),
                    "slot_policy_residual_phase_opposition_sum_mps": (
                        self._slot_policy_residual_phase_opposition_sum_mps.clone()
                    ),
                    "slot_speed_limit_active_agent_fraction_sum": (
                        self._slot_speed_limit_active_agent_fraction_sum.clone()
                    ),
                    "slot_speed_limit_mean_scale_sum": (
                        self._slot_speed_limit_mean_scale_sum.clone()
                    ),
                    "slot_speed_limit_forward_loss_sum_mps": (
                        self._slot_speed_limit_forward_loss_sum_mps.clone()
                    ),
                    "slot_requested_tangential_speed_sum_mps": (
                        self._slot_requested_tangential_speed_sum_mps.clone()
                    ),
                    "slot_executed_tangential_speed_sum_mps": (
                        self._slot_executed_tangential_speed_sum_mps.clone()
                    ),
                    "slot_anchor_heading_rad": (
                        self._slot_anchor_heading_rad.clone()
                    ),
                    "slot_assignment": self._slot_assignment.clone(),
                    "loss_reacquisition_guidance_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.loss_reacquisition_guidance_enabled,
                    ),
                    "loss_reacquisition_active": (
                        self._loss_reacquisition_active.clone()
                    ),
                    "loss_reacquisition_guidance_active": (
                        self._loss_reacquisition_guidance_active.clone()
                    ),
                    "loss_reacquisition_activated": (
                        self._loss_reacquisition_activated.clone()
                    ),
                    "loss_reacquisition_just_reacquired": (
                        self._loss_reacquisition_just_reacquired.clone()
                    ),
                    "loss_reacquisition_scout_index": (
                        self._loss_reacquisition_scout_index.clone()
                    ),
                    "loss_reacquisition_mean_goal_distance_m": (
                        self._loss_reacquisition_mean_goal_distance_m.clone()
                    ),
                    "loss_reacquisition_events": (
                        self._loss_reacquisition_events.clone()
                    ),
                    "loss_reacquisition_completed": (
                        self._loss_reacquisition_completed.clone()
                    ),
                    "loss_reacquisition_active_steps": (
                        self._loss_reacquisition_active_steps.clone()
                    ),
                    "loss_reacquisition_completed_duration_sum_steps": (
                        self._loss_reacquisition_completed_duration_sum_steps.clone()
                    ),
                    "loss_reacquisition_max_segment_steps": (
                        self._loss_reacquisition_max_segment_steps.clone()
                    ),
                    "loss_reacquisition_first_completed": (
                        self._loss_reacquisition_first_completed.clone()
                    ),
                    "loss_reacquisition_first_delay_steps": (
                        self._loss_reacquisition_first_delay_steps.clone()
                    ),
                    "loss_reacquisition_flank_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.loss_reacquisition_occlusion_flank_enabled,
                    ),
                    "loss_reacquisition_flank_waypoint_valid": (
                        self._loss_reacquisition_flank_waypoint_valid.clone()
                    ),
                    "loss_reacquisition_flank_guidance_active": (
                        self._loss_reacquisition_flank_guidance_active.clone()
                    ),
                    "loss_reacquisition_flank_fallback_active": (
                        self._loss_reacquisition_flank_fallback_active.clone()
                    ),
                    "loss_reacquisition_flank_blocker_index": (
                        self._loss_reacquisition_flank_blocker_index.clone()
                    ),
                    "loss_reacquisition_flank_events": (
                        self._loss_reacquisition_flank_events.clone()
                    ),
                    "loss_reacquisition_flank_active_steps": (
                        self._loss_reacquisition_flank_active_steps.clone()
                    ),
                    "loss_reacquisition_flank_fallback_steps": (
                        self._loss_reacquisition_flank_fallback_steps.clone()
                    ),
                    "loss_reacquisition_flank_minimum_path_clearance_m": (
                        self._loss_reacquisition_flank_minimum_path_clearance_m
                        .clone()
                    ),
                    "loss_reacquisition_flank_minimum_los_clearance_m": (
                        self._loss_reacquisition_flank_minimum_los_clearance_m
                        .clone()
                    ),
                    "loss_reacquisition_sector_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.loss_reacquisition_reachable_sector_enabled,
                    ),
                    "loss_reacquisition_sector_waypoint_valid": (
                        self._loss_reacquisition_sector_waypoint_valid.clone()
                    ),
                    "loss_reacquisition_sector_guidance_active": (
                        self._loss_reacquisition_sector_guidance_active.clone()
                    ),
                    "loss_reacquisition_sector_fallback_active": (
                        self._loss_reacquisition_sector_fallback_active.clone()
                    ),
                    "loss_reacquisition_sector_phase_index": (
                        self._loss_reacquisition_sector_phase_index.clone()
                    ),
                    "loss_reacquisition_sector_half_angle_rad": (
                        self._loss_reacquisition_sector_half_angle_rad.clone()
                    ),
                    "loss_reacquisition_sector_radius_m": (
                        self._loss_reacquisition_sector_radius_m.clone()
                    ),
                    "loss_reacquisition_sector_valid_steps": (
                        self._loss_reacquisition_sector_valid_steps.clone()
                    ),
                    "loss_reacquisition_sector_fallback_steps": (
                        self._loss_reacquisition_sector_fallback_steps.clone()
                    ),
                    "loss_reacquisition_sector_no_path_candidate_steps": (
                        self._loss_reacquisition_sector_no_path_candidate_steps
                        .clone()
                    ),
                    "loss_reacquisition_sector_no_boundary_candidate_steps": (
                        self._loss_reacquisition_sector_no_boundary_candidate_steps
                        .clone()
                    ),
                    "loss_reacquisition_sector_minimum_path_clearance_m": (
                        self._loss_reacquisition_sector_minimum_path_clearance_m
                        .clone()
                    ),
                    "reachable_intercept_guidance_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.reachable_intercept_guidance_enabled,
                    ),
                    "intercept_horizon_s": self._intercept_horizon_s.clone(),
                    "intercept_reachable_fraction": (
                        self._intercept_reachable_fraction.clone()
                    ),
                    "intercept_mean_gap_m": self._intercept_mean_gap_m.clone(),
                    "intercept_blocker_agent_index": (
                        self._intercept_blocker_agent_index.clone()
                    ),
                    "intercept_blocker_reachable": (
                        self._intercept_blocker_reachable.clone()
                    ),
                    "target_projection_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.predictive_target_projection_enabled,
                    ),
                    "target_projection_active": self._target_projection_active.clone(),
                    "target_projection_corrected": (
                        self._target_projection_corrected.clone()
                    ),
                    "target_projection_correction_norm_mps": (
                        self._target_projection_correction_norm_mps.clone()
                    ),
                    "target_projection_constraint_corrections": (
                        self._target_projection_constraint_corrections.clone()
                    ),
                    "target_projection_maximum_separation_fallback_constraints": (
                        self._target_projection_maximum_separation_fallback_constraints.clone()
                    ),
                    "target_projection_min_predicted_center_distance_m": (
                        self._target_projection_min_predicted_center_distance_m.clone()
                    ),
                    "target_projection_minimum_maximum_attainable_center_distance_m": (
                        self._target_projection_minimum_maximum_attainable_center_distance_m.clone()
                    ),
                    "target_projection_unavoidable": (
                        self._target_projection_unavoidable.clone()
                    ),
                    "target_projection_unavoidable_constraints": (
                        self._target_projection_unavoidable_constraints.clone()
                    ),
                    "target_projection_infeasible": (
                        self._target_projection_infeasible.clone()
                    ),
                    "safety_projection_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.predictive_agent_projection_enabled,
                    ),
                    "safety_projection_restorative_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.predictive_agent_restore_unsafe_buffer,
                    ),
                    "safety_projection_active": self._safety_projection_active.clone(),
                    "safety_projection_corrected": (
                        self._safety_projection_corrected.clone()
                    ),
                    "safety_projection_correction_norm_mps": (
                        self._safety_projection_correction_norm_mps.clone()
                    ),
                    "safety_projection_pair_corrections": (
                        self._safety_projection_pair_corrections.clone()
                    ),
                    "safety_projection_min_predicted_center_distance_m": (
                        self._safety_projection_min_predicted_center_distance_m.clone()
                    ),
                    "safety_projection_infeasible": (
                        self._safety_projection_infeasible.clone()
                    ),
                    "boundary_projection_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.predictive_boundary_projection_enabled,
                    ),
                    "boundary_projection_active": (
                        self._boundary_projection_active.clone()
                    ),
                    "boundary_projection_corrected": (
                        self._boundary_projection_corrected.clone()
                    ),
                    "boundary_projection_correction_norm_mps": (
                        self._boundary_projection_correction_norm_mps.clone()
                    ),
                    "boundary_projection_constraint_corrections": (
                        self._boundary_projection_constraint_corrections.clone()
                    ),
                    "boundary_projection_min_predicted_clearance_m": (
                        self._boundary_projection_min_predicted_clearance_m.clone()
                    ),
                    "boundary_projection_infeasible": (
                        self._boundary_projection_infeasible.clone()
                    ),
                    "obstacle_projection_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.predictive_static_projection_enabled,
                    ),
                    "obstacle_projection_active": (
                        self._obstacle_projection_active.clone()
                    ),
                    "obstacle_projection_corrected": (
                        self._obstacle_projection_corrected.clone()
                    ),
                    "obstacle_projection_correction_norm_mps": (
                        self._obstacle_projection_correction_norm_mps.clone()
                    ),
                    "obstacle_projection_constraint_corrections": (
                        self._obstacle_projection_constraint_corrections.clone()
                    ),
                    "obstacle_projection_min_predicted_clearance_m": (
                        self._obstacle_projection_min_predicted_clearance_m.clone()
                    ),
                    "obstacle_projection_infeasible": (
                        self._obstacle_projection_infeasible.clone()
                    ),
                    "tangent_bypass_enabled": torch.full_like(
                        self._capture_training.success,
                        self.cfg.static_tangent_bypass_enabled,
                    ),
                    "tangent_bypass_active_agent_count": (
                        self._tangent_active.sum(dim=1).clone()
                    ),
                    "tangent_bypass_activation_count": (
                        self._tangent_activated.sum(dim=1).clone()
                    ),
                    "tangent_bypass_min_candidate_clearance_m": (
                        self._tangent_min_candidate_clearance_m.min(dim=1).values.clone()
                    ),
                }
            )
        return terminated, time_outs

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not self._orientation_runtime_ready:
            return
        if env_ids is None:
            env_ids = self._pursuers[0]._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._orientation_staging_reached[env_ids] = False
        self._orientation_alignment_steps[env_ids] = 0
        self._orientation_staging_reached_each[env_ids] = False
        self._orientation_alignment_steps_each[env_ids] = 0
        self._previous_asynchronous_orientation_progress[env_ids] = 0.0
        self._orientation_contraction_unlocked[env_ids] = False
        self._orientation_reference_radius_m[env_ids] = (
            self.cfg.orientation_staging_radius_m
        )
        self._reference_contraction_tracking_ready[env_ids] = False
        self._reference_contraction_maximum_radius_error_m[env_ids] = 0.0
        self._reference_contraction_maximum_relative_radial_speed_mps[
            env_ids
        ] = 0.0
        self._orientation_yaw_error_rad[env_ids] = 0.0
        self._orientation_aligned_now[env_ids] = False
        self._orientation_phase_ready[env_ids] = False
        self._slot_assignment[env_ids] = -1
        self._slot_anchor_heading_rad[env_ids] = 0.0
        self._slot_anchor_valid[env_ids] = False
        self._slot_guidance_active[env_ids] = False
        self._slot_assignment_switched[env_ids] = False
        self._slot_handoff_count[env_ids] = 0
        self._slot_active_steps[env_ids] = 0
        self._slot_mean_error_m[env_ids] = 0.0
        self._slot_max_error_m[env_ids] = 0.0
        self._slot_goal_xy[env_ids] = 0.0
        self._dual_observer_active[env_ids] = False
        self._dual_observer_active_steps[env_ids] = 0
        self._dual_observer_role_switches[env_ids] = 0
        self._dual_observer_pair_angle_rad[env_ids] = 0.0
        self._path_guard_previous_visible[env_ids] = False
        self._path_guard_previous_target_velocity[env_ids] = 0.
        self._path_guard_corrected_steps[env_ids] = 0
        for value in self._path_guard_telemetry.values():
            value[env_ids] = 0
        self._joint_lookahead_active[env_ids] = False
        self._joint_lookahead_corrected[env_ids] = False
        self._joint_lookahead_active_steps[env_ids] = 0
        self._joint_lookahead_corrected_steps[env_ids] = 0
        self._joint_lookahead_correction_mps[env_ids] = 0.
        self._joint_lookahead_nominal_violation[env_ids] = 0.
        self._joint_lookahead_selected_violation[env_ids] = 0.
        self._slot_preentry_radial_priority_active[env_ids] = False
        self._loss_reacquisition_active[env_ids] = False
        self._loss_reacquisition_guidance_active[env_ids] = False
        self._loss_reacquisition_activated[env_ids] = False
        self._loss_reacquisition_just_reacquired[env_ids] = False
        self._loss_reacquisition_scout_index[env_ids] = -1
        self._loss_reacquisition_slot_angle_rad[env_ids] = 0.0
        self._loss_reacquisition_mean_goal_distance_m[env_ids] = 0.0
        self._loss_reacquisition_flank_waypoint_xy[env_ids] = 0.0
        self._loss_reacquisition_flank_waypoint_valid[env_ids] = False
        self._loss_reacquisition_flank_guidance_active[env_ids] = False
        self._loss_reacquisition_flank_fallback_active[env_ids] = False
        self._loss_reacquisition_flank_blocker_index[env_ids] = -1
        self._loss_reacquisition_flank_events[env_ids] = 0
        self._loss_reacquisition_flank_active_steps[env_ids] = 0
        self._loss_reacquisition_flank_fallback_steps[env_ids] = 0
        self._loss_reacquisition_flank_minimum_path_clearance_m[env_ids] = (
            torch.inf
        )
        self._loss_reacquisition_flank_minimum_los_clearance_m[env_ids] = (
            torch.inf
        )
        self._loss_reacquisition_sector_waypoint_xy[env_ids] = 0.0
        self._loss_reacquisition_sector_waypoint_valid[env_ids] = False
        self._loss_reacquisition_sector_guidance_active[env_ids] = False
        self._loss_reacquisition_sector_fallback_active[env_ids] = False
        self._loss_reacquisition_sector_phase_index[env_ids] = -1
        self._loss_reacquisition_sector_half_angle_rad[env_ids] = 0.0
        self._loss_reacquisition_sector_radius_m[env_ids] = 0.0
        self._loss_reacquisition_sector_valid_steps[env_ids] = 0
        self._loss_reacquisition_sector_fallback_steps[env_ids] = 0
        self._loss_reacquisition_sector_no_path_candidate_steps[env_ids] = 0
        self._loss_reacquisition_sector_no_boundary_candidate_steps[env_ids] = 0
        self._loss_reacquisition_sector_minimum_path_clearance_m[env_ids] = (
            torch.inf
        )
        self._loss_reacquisition_events[env_ids] = 0
        self._loss_reacquisition_completed[env_ids] = 0
        self._loss_reacquisition_active_steps[env_ids] = 0
        self._loss_reacquisition_segment_steps[env_ids] = 0
        self._loss_reacquisition_completed_duration_sum_steps[env_ids] = 0
        self._loss_reacquisition_max_segment_steps[env_ids] = 0
        self._loss_reacquisition_first_completed[env_ids] = False
        self._loss_reacquisition_first_delay_steps[env_ids] = -1
        self._intercept_horizon_s[env_ids] = 0.0
        self._intercept_reachable_fraction[env_ids] = 0.0
        self._intercept_mean_gap_m[env_ids] = 0.0
        self._intercept_blocker_agent_index[env_ids] = -1
        self._intercept_blocker_reachable[env_ids] = False
        self._previous_angular_error_rad[env_ids] = 0.0
        self._strong_ring_hold_steps[env_ids] = 0
        self._strong_ring_now[env_ids] = False
        self._strong_ring_success[env_ids] = False
        self._strong_ring_confirmation_stabilization_active[env_ids] = False
        self._strong_ring_mean_gap_error_rad[env_ids] = 0.0
        self._strong_ring_minimum_distance_m[env_ids] = 0.0
        self._strong_ring_maximum_distance_m[env_ids] = 0.0
        self._previous_strong_ring_violation_score[env_ids] = 0.0
        self._hybrid_rl_active[env_ids] = False
        self._hybrid_traditional_steps[env_ids] = 0
        self._hybrid_rl_steps[env_ids] = 0
        self._hybrid_policy_residual_scale[env_ids] = 0.0
        self._slot_policy_residual_active_steps[env_ids] = 0
        self._slot_policy_residual_norm_sum_mps[env_ids] = 0.0
        self._slot_policy_residual_radial_support_sum_mps[env_ids] = 0.0
        self._slot_policy_residual_radial_opposition_sum_mps[env_ids] = 0.0
        self._slot_policy_residual_phase_support_sum_mps[env_ids] = 0.0
        self._slot_policy_residual_phase_opposition_sum_mps[env_ids] = 0.0
        self._slot_speed_limit_active_agent_fraction_sum[env_ids] = 0.0
        self._slot_speed_limit_mean_scale_sum[env_ids] = 0.0
        self._slot_speed_limit_forward_loss_sum_mps[env_ids] = 0.0
        self._slot_requested_tangential_speed_sum_mps[env_ids] = 0.0
        self._slot_executed_tangential_speed_sum_mps[env_ids] = 0.0
        self._target_projection_active[env_ids] = False
        self._target_projection_corrected[env_ids] = False
        self._target_projection_correction_norm_mps[env_ids] = 0.0
        self._target_projection_constraint_corrections[env_ids] = 0
        self._target_projection_maximum_separation_fallback_constraints[
            env_ids
        ] = 0
        self._target_projection_min_predicted_center_distance_m[env_ids] = torch.inf
        self._target_projection_minimum_maximum_attainable_center_distance_m[
            env_ids
        ] = torch.inf
        self._target_projection_unavoidable[env_ids] = False
        self._target_projection_unavoidable_constraints[env_ids] = 0
        self._target_projection_infeasible[env_ids] = False
        self._ring_projection_active[env_ids] = False
        self._ring_projection_activation_latched[env_ids] = False
        self._ring_projection_corrected[env_ids] = False
        self._ring_projection_correction_norm_mps[env_ids] = 0.0
        self._ring_projection_corrected_constraints[env_ids] = 0
        self._ring_projection_min_predicted_radius_m[env_ids] = torch.inf
        self._ring_projection_max_predicted_radius_m[env_ids] = 0.0
        self._ring_projection_infeasible[env_ids] = False
        self._angular_gap_projection_active[env_ids] = False
        self._angular_gap_projection_corrected[env_ids] = False
        self._angular_gap_projection_correction_norm_mps[env_ids] = 0.0
        self._angular_gap_projection_max_predicted_gap_rad[env_ids] = 0.0
        self._angular_gap_projection_max_predicted_mean_gap_error_rad[
            env_ids
        ] = 0.0
        self._angular_gap_projection_infeasible[env_ids] = False
        self._preentry_braking_active[env_ids] = False
        self._preentry_braking_corrected[env_ids] = False
        self._preentry_braking_correction_norm_mps[env_ids] = 0.0
        self._preentry_braking_limited_agents[env_ids] = 0
        self._preentry_braking_minimum_stopping_margin_m[env_ids] = torch.inf
        self._preentry_braking_infeasible[env_ids] = False
        self._previous_ring_target_velocity_xy[env_ids] = 0.0
        self._ring_target_acceleration_xy[env_ids] = 0.0
        self._safety_projection_active[env_ids] = False
        self._safety_projection_corrected[env_ids] = False
        self._safety_projection_correction_norm_mps[env_ids] = 0.0
        self._safety_projection_pair_corrections[env_ids] = 0
        self._safety_projection_min_predicted_center_distance_m[env_ids] = torch.inf
        self._safety_projection_infeasible[env_ids] = False
        self._obstacle_projection_active[env_ids] = False
        self._obstacle_projection_corrected[env_ids] = False
        self._obstacle_projection_correction_norm_mps[env_ids] = 0.0
        self._obstacle_projection_constraint_corrections[env_ids] = 0
        self._obstacle_projection_min_predicted_clearance_m[env_ids] = torch.inf
        self._obstacle_projection_infeasible[env_ids] = False
        self._boundary_projection_active[env_ids] = False
        self._boundary_projection_corrected[env_ids] = False
        self._boundary_projection_correction_norm_mps[env_ids] = 0.0
        self._boundary_projection_constraint_corrections[env_ids] = 0
        self._boundary_projection_min_predicted_clearance_m[env_ids] = torch.inf
        self._boundary_projection_infeasible[env_ids] = False
        self._tangent_obstacle_index[env_ids] = -1
        self._tangent_side[env_ids] = 0
        self._tangent_blocked_steps[env_ids] = 0
        self._tangent_hold_steps_remaining[env_ids] = 0
        self._tangent_active[env_ids] = False
        self._tangent_activated[env_ids] = False
        self._tangent_min_candidate_clearance_m[env_ids] = torch.inf
