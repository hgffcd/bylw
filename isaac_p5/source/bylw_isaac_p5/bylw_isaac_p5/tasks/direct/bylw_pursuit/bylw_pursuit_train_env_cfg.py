"""Configuration for Isaac-native fixed-height MAPPO curriculum training."""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from ....core.native_observations import NATIVE_OBSERVATION_DIM, NATIVE_STATE_DIM
from .bylw_pursuit_env_cfg import BylwPursuitEnvCfg


@configclass
class BylwPursuitTrainEnvCfg(BylwPursuitEnvCfg):
    possible_agents = ["pursuer_0", "pursuer_1", "pursuer_2"]
    action_spaces = {agent: 3 for agent in possible_agents}
    observation_spaces = {
        agent: NATIVE_OBSERVATION_DIM for agent in possible_agents
    }
    state_space = NATIVE_STATE_DIM
    episode_length_s = 20.0
    reset_layout = "tracked_ring"
    tracked_target_center_xy_m = (0.0, 0.0)
    tracked_target_spread_m = 5.0
    sense_range_m = 100.0
    tracked_reset_reference_range_m = 100.0
    fov_half_rad = math.pi
    evader_enabled = False
    dense_physical_telemetry = False
    # The training radius follows A0 12 m -> A1 9 m -> A2/B 6 m.  The base
    # task's capture_radius_main_m/strict_m remain frozen to 6 m/5 m so every
    # evaluation continues to report the thesis criteria.
    training_capture_radius_m = 6.0
    desired_training_radius_m = 4.5
    training_near_switch_m = 9.0
    training_command_speed_mps = 4.0
    # Optional finite-FOV staging gate used by the C0 orientation curriculum.
    orientation_gate_enabled = False
    orientation_staging_radius_m = 9.5
    orientation_required_alignment_steps = 5
    orientation_yaw_tolerance_rad = math.pi / 6.0
    orientation_yaw_gain = 1.5
    orientation_policy_residual_scale = 0.25
    orientation_maximum_inward_speed_mps = 0.5
    orientation_reference_contraction_speed_mps = 0.5
    orientation_reference_contraction_tracking_gate_enabled = False
    orientation_reference_anchor_to_mean_radius_on_unlock = False
    # C9 adds target-velocity-anchored angular slots.  Assignment hysteresis
    # avoids crossings while still allowing a handoff after a target turn.
    moving_target_slot_guidance_enabled = False
    moving_target_slot_activation_radius_m = 14.0
    moving_target_slot_position_gain = 0.70
    moving_target_slot_policy_residual_scale = 0.35
    moving_target_slot_feedforward_scale = 1.0
    moving_target_slot_full_feedforward_above_strong_ring = False
    # Zero preserves C9--C18 exactly. C19 first audits a bounded CV lead only.
    moving_target_slot_prediction_horizon_s = 0.0
    moving_target_slot_minimum_target_speed_mps = 0.15
    moving_target_slot_maximum_anchor_turn_rate_rps = 0.80
    moving_target_slot_assignment_hysteresis_m = 1.0
    # Optional obstacle-aware assignment adds a path-clearance cost to each
    # agent/slot permutation.  Earlier curricula keep pure Euclidean matching.
    moving_target_slot_obstacle_aware_assignment = False
    moving_target_slot_assignment_path_clearance_m = 1.0
    moving_target_slot_assignment_obstacle_penalty = 30.0
    # C31 removes assignment crossings by preserving the current cyclic order,
    # locks that topology for the episode, and controls radial/phase errors as
    # separate continuous components. Defaults preserve C30 and earlier stages.
    moving_target_slot_cyclic_order_preserving_assignment = False
    moving_target_slot_lock_valid_assignment = False
    moving_target_slot_phase_separation_control_enabled = False
    moving_target_slot_phase_gain = 0.8
    moving_target_slot_phase_angular_velocity_damping_gain_s = 0.0
    # C84 confines phase-rate damping to the measured strong-ring confirmation
    # window.  Earlier stages retain their original global/no-damping paths.
    moving_target_slot_phase_angular_velocity_damping_confirmation_only = False
    moving_target_slot_radial_gain = 1.0
    # Optional physical radial-velocity damping. Zero exactly preserves all
    # earlier proportional-only slot controllers.
    moving_target_slot_radial_velocity_damping_gain = 0.0
    # Zero keeps the legacy unbounded radial component before total-speed
    # projection. C36 audits a 1.0 m/s component budget.
    moving_target_slot_maximum_radial_speed_mps = 0.0
    moving_target_slot_phase_scheduled_radial_speed_budget = False
    moving_target_slot_maximum_tangential_speed_mps = 3.0
    moving_target_slot_phase_activation_margin_m = 2.0
    # Keep target-motion feedforward before scaling radial/phase corrections
    # into the remaining 4 m/s command ball.  Defaults off for exact replay.
    moving_target_slot_feedforward_priority_projection_enabled = False
    # C93 development arm: once the exact angular entry constraints are met,
    # spend the remaining equal-speed command budget on radial annulus entry
    # before tangential refinement. Disabled for every historical replay.
    moving_target_slot_preentry_radial_priority_enabled = False
    moving_target_slot_far_intercept_control_enabled = False
    # Optional weak absolute-phase anchor for the designated forward blocker.
    # Defaults off so C0--C33 replay unchanged.
    moving_target_slot_blocker_phase_anchor_control_enabled = False
    moving_target_slot_blocker_phase_anchor_gain = 0.3
    moving_target_slot_blocker_phase_anchor_maximum_speed_mps = 0.8
    # C98: measured observer pair anchors a soft triangle; disabled in C97.
    dual_observer_formation_enabled = False
    dual_observer_role_lock_enabled = False
    # C100 starts angular correction 15 degrees before the success boundary.
    region_gap_reserve_enabled = False
    region_joint_lookahead_enabled = False
    dual_observer_role_hysteresis_m = 1.0
    dual_observer_phase_tolerance_rad = math.pi / 12.0
    # C32 constrains learning to positive multiplicative gains on the analytic
    # radial and phase commands.  The actor cannot reverse either direction.
    moving_target_slot_gain_residual_control_enabled = False
    moving_target_slot_radial_gain_residual_scale = 0.25
    moving_target_slot_phase_gain_residual_scale = 0.25
    moving_target_slot_radial_velocity_residual_control_enabled = False
    moving_target_slot_radial_velocity_residual_scale_mps = 0.75
    moving_target_slot_tangential_velocity_residual_control_enabled = False
    moving_target_slot_tangential_velocity_residual_scale_mps = 1.0
    orientation_yaw_gain_residual_scale = 0.25
    moving_target_slot_error_reward_weight = 0.15
    # C26 optionally adds a potential-difference reward toward the assigned
    # moving role slot. Zero preserves every earlier checkpoint/task exactly.
    moving_target_slot_progress_reward_weight = 0.0
    moving_target_slot_progress_use_maximum_error = False
    moving_target_slot_error_use_maximum_error = False
    # C27 exposes the existing analytic role goal and phase to the actor.
    # Defaults remain false so all 53-D checkpoints replay exactly.
    role_phase_observation_enabled = False
    moving_target_slot_prepare_assignment_for_observation = False
    # C30 hierarchical route: the analytic supervisor owns search, approach,
    # target-facing and staging.  MAPPO is executed and updated only while the
    # full team is currently ready for encirclement.
    hybrid_phase_supervisor_enabled = False
    hybrid_encirclement_policy_residual_scale = 0.20
    hybrid_require_current_team_ready = True
    hybrid_rl_activation_on_staging = False
    orientation_contraction_pause_without_team_ready = False
    # Optional curriculum repair: after the full team has legally unlocked
    # once, keep shrinking the reference ring even if current readiness is
    # briefly lost.  This does not bypass visibility gates for RL or safety.
    orientation_reference_contraction_latched_after_unlock = False
    orientation_phase_gate_enabled = False
    orientation_phase_maximum_gap_rad = 5.0 * math.pi / 6.0
    orientation_phase_maximum_mean_gap_error_rad = math.pi / 9.0
    moving_target_angular_error_reward_weight = 0.50
    moving_target_angular_progress_reward_weight = 0.35
    # P5-L0 ports the frozen 2-D Last-guided one-scout/two-tracker recovery
    # structure.  The default remains off so every prior checkpoint/task keeps
    # its exact execution path.  Event telemetry is collected in either arm.
    loss_reacquisition_guidance_enabled = False
    loss_reacquisition_start_lost_steps = 1
    loss_reacquisition_slot_radius_m = 8.0
    loss_reacquisition_scout_blend = 0.85
    loss_reacquisition_tracker_blend = 0.55
    loss_reacquisition_blend_growth = 0.10
    loss_reacquisition_scout_speed_ratio = 1.0
    loss_reacquisition_tracker_speed_ratio = 0.65
    loss_reacquisition_scout_minimum_selection_distance_m = 0.0
    loss_reacquisition_tracker_hold_enabled = False
    loss_reacquisition_slot_arrival_m = 1.5
    # P5-L2 keeps the Last belief but sends only the locked scout to a
    # visibility-restoring side waypoint around the first mapped blocker.
    loss_reacquisition_occlusion_flank_enabled = False
    loss_reacquisition_flank_waypoint_margin_m = 0.5
    loss_reacquisition_flank_los_margin_m = 0.25
    loss_reacquisition_flank_blocker_margin_m = 0.25
    loss_reacquisition_flank_boundary_clearance_m = 1.0
    loss_reacquisition_flank_angular_samples = 32
    # P5-L3 searches a physically bounded, expanding sector built only from
    # the last team-visible target state.  It remains off for exact replay of
    # every earlier experiment.
    loss_reacquisition_reachable_sector_enabled = False
    loss_reacquisition_sector_target_speed_bound_mps = 4.0
    loss_reacquisition_sector_velocity_heading_threshold_mps = 0.25
    loss_reacquisition_sector_lookahead_s = 0.4
    loss_reacquisition_sector_minimum_radius_m = 1.5
    loss_reacquisition_sector_maximum_radius_m = 12.0
    loss_reacquisition_sector_initial_half_angle_rad = math.pi / 6.0
    loss_reacquisition_sector_half_angle_growth_rps = math.pi / 3.0
    loss_reacquisition_sector_phase_hold_steps = 4
    loss_reacquisition_sector_boundary_clearance_m = 1.0
    loss_reacquisition_sector_detour_margin_m = 0.5
    loss_reacquisition_sector_detour_maximum_extra_m = 4.5
    loss_reacquisition_sector_detour_radial_samples = 5
    loss_reacquisition_sector_detour_angular_samples = 32
    # C21 replaces a fixed CV lead with a bounded equal-speed reachability
    # oracle and locks the most forward pursuer to the leading blocker slot.
    reachable_intercept_guidance_enabled = False
    reachable_intercept_minimum_horizon_s = 0.25
    reachable_intercept_maximum_horizon_s = 8.0
    reachable_intercept_horizon_samples = 33
    reachable_intercept_slot_horizon_cap_s = 1.0
    reachable_intercept_blocker_only_slot_lead = False
    reachable_intercept_reachable_role_slot_lead = False
    reachable_intercept_release_during_contraction = False
    reachable_intercept_lead_fade_start_radius_m = 9.0
    reachable_intercept_lead_zero_radius_m = 5.5
    reachable_intercept_blocker_role_lock_weight_m = 20.0
    # Per-agent latching lets tracker/scout handoff finish asynchronously while
    # preserving the all-agent requirement before contraction.
    asynchronous_orientation_gate_enabled = False
    asynchronous_orientation_progress_reward_weight = 0.20
    # C24 keeps staging in the moving target's relative frame, permits a
    # latched pursuer that has fallen outside the ring to re-approach, and
    # requires all pursuers to be currently ready before team contraction.
    orientation_target_relative_staging_enabled = False
    asynchronous_require_current_team_ready = False
    asynchronous_current_readiness_reward_enabled = False
    strong_ring_termination_enabled = False
    # C97 is opt-in and keeps all historical strong-ring diagnostics intact.
    region_capture_enabled = False
    region_objective_enabled = False
    region_capture_radius_m = 5.5
    region_capture_maximum_gap_rad = 5.0 * math.pi / 6.0
    region_capture_minimum_target_distance_m = 1.5
    target_buffer_failure_enabled = False
    region_control_minimum_radius_m = 1.5
    strong_ring_minimum_radius_m = 4.0
    strong_ring_maximum_radius_m = 5.5
    strong_ring_maximum_gap_rad = 5.0 * math.pi / 6.0
    strong_ring_maximum_mean_gap_error_rad = math.pi / 9.0
    strong_ring_required_hold_steps = 10
    strong_ring_require_all_visible = True
    # Optional dense credit for a legal confirmation step.  Zero preserves
    # every earlier stage; it never alters the geometry or required hold.
    strong_ring_step_reward_weight = 0.0
    # Optional dense credit for reducing normalized violations of the exact
    # radial-band and angular strong-ring constraints. Zero preserves history.
    strong_ring_violation_progress_reward_weight = 0.0
    strong_ring_confirmation_stabilization_enabled = False
    strong_ring_confirmation_suppress_policy_residual = False
    strong_ring_confirmation_maximum_phase_gain_enabled = False
    predictive_ring_stabilization_enabled = False
    predictive_ring_prediction_horizon_s = 0.5
    predictive_ring_command_response_ratio = 0.5
    predictive_ring_projection_iterations = 4
    # One preserves the historical terminal-only projection exactly.  Values
    # above one constrain intermediate low-level response positions as well.
    predictive_ring_response_trajectory_substeps = 1
    predictive_ring_enforce_minimum_radius = True
    predictive_ring_minimum_radius_after_strong_entry = False
    predictive_ring_dynamics_consistent_enabled = False
    predictive_ring_low_level_velocity_kp = 2.0
    predictive_ring_radial_candidate_count = 33
    predictive_ring_common_acceleration_feedforward_after_strong_entry = False
    predictive_ring_preentry_velocity_sync_enabled = False
    predictive_ring_activation_margin_m = 0.0
    predictive_ring_activation_latched_after_measured_entry = False
    predictive_ring_activation_requires_strong_ring_entry = False
    predictive_ring_activation_requires_maximum_gap_eligibility = False
    predictive_ring_activation_maximum_gap_margin_rad = 0.0
    predictive_ring_activation_requires_inside_inner_boundary = False
    predictive_ring_target_acceleration_enabled = False
    predictive_ring_target_acceleration_substeps = 5
    # C86 adds a joint, dynamics-consistent line projection over the exact
    # strong-ring angular limits.  It is disabled for every historical stage.
    predictive_ring_angular_gap_projection_enabled = False
    predictive_ring_angular_gap_candidate_count = 17
    # C87 applies an analytic stopping-distance guard to early arrivals and
    # permits angular correction in the 9 m approach zone before annulus entry.
    predictive_ring_preentry_braking_enabled = False
    predictive_ring_preentry_braking_activation_radius_m = 9.0
    predictive_ring_angular_gap_preentry_enabled = False
    predictive_ring_angular_gap_preentry_activation_radius_m = 9.0
    target_path_guard_enabled = False
    target_path_guard_acceleration_enabled = False
    target_path_guard_extended_enabled = False
    target_path_guard_margin_enabled = False
    target_path_guard_reserve_enabled = False
    learning_diagnostics_enabled = False
    predictive_target_projection_enabled = False
    predictive_target_minimum_center_distance_m = 1.0
    predictive_target_prediction_horizon_s = 0.75
    predictive_target_command_response_ratio = 0.5
    predictive_target_projection_iterations = 2
    predictive_target_restore_unsafe_buffer = False
    predictive_target_projection_last = False
    # C89 keeps separation active for a bounded short occlusion using only
    # the last observed constant-velocity target state.
    predictive_target_last_seen_cv_during_loss_enabled = False
    predictive_target_last_seen_cv_maximum_loss_time_s = 2.0
    predictive_target_last_seen_reachable_tube_enabled = False
    # C1 adds a zero-shot predictive joint-action safety execution layer.
    predictive_agent_projection_enabled = False
    predictive_agent_minimum_center_distance_m = 2.5
    predictive_agent_prediction_horizon_s = 0.75
    predictive_agent_command_response_ratio = 0.5
    predictive_agent_projection_iterations = 4
    # P5-S1 optionally makes an already-violated safety buffer restorative.
    # False preserves the accepted execution path exactly.
    predictive_agent_restore_unsafe_buffer = False
    # C7 repairs moving-target approach collisions by applying the pairwise
    # projection before, as well as after, orientation/contraction unlock.
    predictive_agent_always_active = False
    predictive_agent_correction_reward_weight = 0.0
    predictive_agent_infeasible_reward_penalty = 0.0
    # C4 applies an always-active obstacle projection before the staging ring.
    predictive_static_projection_enabled = False
    # Evaluation ablation only: retain moving obstacle physics/observations while
    # treating obstacle velocity as zero inside the predictive safety projection.
    predictive_static_use_obstacle_velocity = True
    predictive_static_minimum_clearance_m = 1.0
    predictive_static_prediction_horizon_s = 1.0
    predictive_static_command_response_ratio = 0.5
    predictive_static_projection_iterations = 4
    # C23 closes the remaining arena-wall gap in the predictive execution
    # layer; previous stages retain reward-only boundary handling.
    predictive_boundary_projection_enabled = False
    predictive_boundary_minimum_clearance_m = 1.0
    predictive_boundary_prediction_horizon_s = 1.5
    predictive_boundary_command_response_ratio = 0.5
    predictive_boundary_correction_reward_weight = 0.02
    predictive_boundary_infeasible_reward_penalty = 0.10
    # C5/G0 breaks repeated radial-projection deadlocks with a stable tangent.
    static_tangent_bypass_enabled = False
    static_tangent_activation_steps = 5
    static_tangent_hold_steps = 15
    static_tangent_speed_mps = 2.5
    static_tangent_outward_speed_mps = 0.35
    static_tangent_release_distance_m = 3.0
    joint_safety_projection_sweeps = 1
    # Stage A keeps physical obstacles outside the tracked-start working area.
    obstacle_specs = (
        (-43.0, -43.0, 2.0),
        (-43.0, 43.0, 2.0),
        (43.0, -43.0, 2.0),
        (43.0, 43.0, 2.0),
        (-43.0, -33.0, 2.0),
        (-43.0, 33.0, 2.0),
        (43.0, -33.0, 2.0),
        (43.0, 33.0, 2.0),
    )
