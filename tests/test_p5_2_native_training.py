import ast
import inspect
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = ROOT / "isaac_p5" / "source" / "bylw_isaac_p5"
sys.path.insert(0, str(PACKAGE_SOURCE))

from bylw_isaac_p5.core.config import P5TaskSpec  # noqa: E402
from bylw_isaac_p5.core.native_action_adapter import (  # noqa: E402
    WorldVelocityYawAdapter,
)
from bylw_isaac_p5.core.native_observations import (  # noqa: E402
    NATIVE_OBSERVATION_DIM,
    NATIVE_ROLE_PHASE_FEATURE_DIM,
    NATIVE_ROLE_PHASE_OBSERVATION_DIM,
    NATIVE_STATE_DIM,
    build_native_centralized_state,
    build_native_observations,
    build_role_phase_features,
)
from bylw_isaac_p5.core.native_training import (  # noqa: E402
    advance_asynchronous_orientation_gate,
    advance_orientation_gate,
    bounded_last_seen_cv_target_state,
    bounded_per_agent_constant_velocity_intercept,
    bounded_constant_velocity_intercept,
    circular_gap_statistics,
    compose_feedforward_priority_velocity,
    compose_feedforward_radial_priority_velocity,
    loss_reacquisition_guidance,
    moving_target_slot_guidance,
    native_curriculum_stage,
    native_team_reward,
    occlusion_flank_waypoint,
    predictive_projection_reward,
    project_dynamics_consistent_angular_gap_velocity,
    project_dynamics_consistent_target_annulus_velocity,
    project_preentry_braking_reserve_velocity,
    project_predictive_arena_boundary_velocity,
    project_predictive_pairwise_velocity,
    project_predictive_static_obstacle_velocity,
    project_predictive_target_annulus_velocity,
    project_predictive_target_separation_velocity,
    project_staged_contraction_velocity,
    project_staged_contraction_velocity,
    release_intercept_horizon_during_contraction,
    reachable_sector_waypoint,
    reachable_role_slot_prediction_horizon,
    resolve_static_obstacle_tangent_velocity,
    slot_goal_progress_reward,
    smooth_clearance_barrier,
    strong_ring_violation_score,
    target_facing_yaw_guidance,
)


def test_c28_two_mps_contraction_is_compatible_with_three_mps_feedforward():
    desired = torch.tensor([[[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]]])
    pursuer = torch.tensor(
        [[[0.0, -9.5], [0.0, -9.5], [0.0, -9.5]]]
    )
    projected = project_staged_contraction_velocity(
        desired,
        pursuer,
        torch.zeros(1, 2),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        maximum_inward_speed_mps=2.0,
        target_velocity_xy=torch.tensor([[3.0, 0.0]]),
    )
    assert torch.allclose(projected, torch.tensor([[[3.0, 2.0]]]).expand_as(projected))
    assert projected.norm(dim=-1).max().item() < 4.0


def test_c29_inherits_c27_but_separates_40s_training_from_30s_evaluation():
    c27 = native_curriculum_stage("c27")
    c29 = native_curriculum_stage("c29")
    assert c29.training_capture_radius_m == c27.training_capture_radius_m
    assert c29.desired_radius_m == c27.desired_radius_m
    assert c29.near_switch_m == c27.near_switch_m
    assert c29.static_obstacles == c27.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if args_cli.stage in ("c29", "c30"):' in trainer_source
    assert "30.0 if args_cli.evaluation_only else 40.0" in trainer_source
    assert 'if stage in ("c27", "c29", "c30"):' in trainer_source


def test_c30_freezes_route_a_contract_and_enables_hybrid_route_b():
    c27 = native_curriculum_stage("c27")
    c30 = native_curriculum_stage("c30")
    assert c30.training_capture_radius_m == c27.training_capture_radius_m
    assert c30.desired_radius_m == c27.desired_radius_m
    assert c30.near_switch_m == c27.near_switch_m
    assert c30.static_obstacles == c27.static_obstacles
    assert c30.target_success_rate == c27.target_success_rate
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if args_cli.stage in ("c29", "c30"):' in trainer_source
    assert 'if stage == "c30":' in trainer_source
    assert "cfg.hybrid_phase_supervisor_enabled = True" in trainer_source
    assert "cfg.moving_target_slot_activation_radius_m = 1.0e6" in trainer_source
    assert "cfg.orientation_contraction_pause_without_team_ready = True" in (
        trainer_source
    )
    assert "cfg.predictive_target_restore_unsafe_buffer = True" in trainer_source
    assert "cfg.predictive_target_projection_last = True" in trainer_source
    assert "cfg.predictive_target_projection_iterations = 16" in trainer_source
    assert 'p5["hybrid_rl_active"].float()' in trainer_source


def test_c40_is_static_equal_capability_and_keeps_c30_hybrid_contract():
    c30 = native_curriculum_stage("c30")
    c40 = native_curriculum_stage("c40")
    assert c40.training_capture_radius_m == c30.training_capture_radius_m
    assert c40.desired_radius_m == c30.desired_radius_m
    assert c40.near_switch_m == c30.near_switch_m
    assert c40.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c40":' in trainer_source
    assert 'configure_stage(cfg, "c30")' in trainer_source
    assert "cfg.dynamic_obstacle_indices = ()" in trainer_source
    assert "cfg.evader_speed_mps = 4.0" in trainer_source
    assert "cfg.evader_minimum_episode_speed_mps = 4.0" in trainer_source
    assert "cfg.evader_allow_stop_mode = False" in trainer_source
    assert "cfg.evader_maintain_episode_speed = True" in trainer_source
    assert "cfg.pursuer_max_planar_accel_mps2 = 4.0" in trainer_source
    assert "cfg.evader_max_planar_accel_mps2 = 4.0" in trainer_source
    assert "C40/C41 require every physical obstacle to be static" in trainer_source
    assert "C40 freezes both training and evaluation to the 30 s horizon" in (
        trainer_source
    )
    assert '"--evaluation_orientation_staging_radius_m"' in trainer_source
    assert "choices=(12.0, 15.0)" in trainer_source
    assert "cfg.orientation_staging_radius_m = (" in trainer_source
    assert '"--evaluation_lock_slot_assignment"' in trainer_source
    assert "cfg.moving_target_slot_lock_valid_assignment = True" in trainer_source
    assert '"--evaluation_phase_separation_control"' in trainer_source
    assert (
        "cfg.moving_target_slot_phase_separation_control_enabled = True"
        in trainer_source
    )
    assert "cfg.moving_target_slot_phase_activation_margin_m = 6.0" in (
        trainer_source
    )
    assert '"--evaluation_latched_reference_contraction"' in trainer_source
    assert (
        "cfg.orientation_reference_contraction_latched_after_unlock = True"
        in trainer_source
    )
    train_env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "reference_contraction_active = (" in train_env_source
    assert "self._orientation_contraction_unlocked" in train_env_source
    assert '"--evaluation_tracked_reset_reference_range_m"' in trainer_source
    assert "choices=(12.0, 15.0, 20.0, 25.0)" in trainer_source
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_env.py"
    ).read_text(encoding="utf-8")
    assert "LowLevelControllerSpec(" in env_source
    assert "max_planar_accel_mps2=cfg.pursuer_max_planar_accel_mps2" in (
        env_source
    )
    assert "max_planar_accel_mps2=cfg.evader_max_planar_accel_mps2" in env_source


def test_c41_freezes_the_viable_c40_near_ring_training_entry():
    c40 = native_curriculum_stage("c40")
    c41 = native_curriculum_stage("c41")
    assert c41.training_capture_radius_m == c40.training_capture_radius_m
    assert c41.desired_radius_m == c40.desired_radius_m
    assert c41.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c41":' in trainer_source
    assert "cfg.tracked_reset_reference_range_m = 12.0" in trainer_source
    assert "cfg.orientation_staging_radius_m = 12.0" in trainer_source
    assert "cfg.moving_target_slot_phase_separation_control_enabled = True" in (
        trainer_source
    )
    assert "cfg.orientation_reference_contraction_latched_after_unlock = True" in (
        trainer_source
    )
    assert 'if args_cli.stage in ("c40", "c41", "c42", "c43", "c44", "c45", "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55", "c56", "c57", "c58", "c60", "c61", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79", "c80", "c81", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91"):' in trainer_source
    assert 'best_strict_success = initial_eval["success_rate_5m"]' in (
        trainer_source
    )
    assert (
        'best_strong_ring_hold = initial_eval["maximum_strong_ring_hold_steps"]'
        in trainer_source
    )


def test_feedforward_priority_projection_keeps_translation_and_fits_correction():
    feedforward = torch.tensor([[[3.0, 0.0]]])
    correction = torch.tensor([[[0.0, 4.0]]])
    velocity, scale = compose_feedforward_priority_velocity(
        feedforward, correction, 4.0
    )
    assert torch.allclose(velocity[..., 0], torch.tensor([[3.0]]), atol=1.0e-6)
    assert torch.allclose(velocity.norm(dim=-1), torch.tensor([[4.0]]), atol=1.0e-6)
    assert torch.allclose(scale, torch.tensor([[math.sqrt(7.0) / 4.0]]), atol=1.0e-6)
    free_velocity, free_scale = compose_feedforward_priority_velocity(
        torch.tensor([[[2.0, 0.0]]]), torch.tensor([[[1.0, 0.0]]]), 4.0
    )
    assert torch.equal(free_velocity, torch.tensor([[[3.0, 0.0]]]))
    assert torch.equal(free_scale, torch.ones(1, 1))


def test_c93_preentry_priority_spends_budget_on_radius_before_phase():
    feedforward = torch.tensor([[[3.0, 0.0]], [[3.0, 0.0]]])
    radial = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    phase = torch.tensor([[[0.0, 2.0]], [[0.0, 2.0]]])
    active = torch.tensor([[True], [False]])
    velocity, radial_scale, phase_scale = (
        compose_feedforward_radial_priority_velocity(
            feedforward, radial, phase, active, 4.0
        )
    )
    assert torch.allclose(velocity[0, 0], torch.tensor([4.0, 0.0]))
    assert radial_scale[0, 0].item() == 1.0
    assert phase_scale[0, 0].item() == 0.0
    historical, historical_scale = compose_feedforward_priority_velocity(
        feedforward[1:], radial[1:] + phase[1:], 4.0
    )
    assert torch.allclose(velocity[1:], historical)
    assert torch.allclose(radial_scale[1:], historical_scale)
    assert torch.allclose(phase_scale[1:], historical_scale)


def test_c93_flag_is_opt_in_and_audited_without_new_stage_semantics():
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    cfg_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct/"
        "bylw_pursuit/bylw_pursuit_train_env_cfg.py"
    ).read_text(encoding="utf-8")
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct/"
        "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "moving_target_slot_preentry_radial_priority_enabled = False" in cfg_source
    assert '"--preentry_radial_priority"' in trainer_source
    assert '"preentry_radial_priority": args_cli.preentry_radial_priority' in trainer_source
    assert "radial_correction_priority_active" in env_source
    assert "self._strong_ring_hold_steps == 0" in env_source


def test_c94_extended_horizon_is_explicit_and_does_not_unfreeze_default_c91():
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert '"--c94_extended_episode_horizon"' in trainer_source
    assert 'args_cli.stage != "c91"' in trainer_source
    assert "math.isclose(args_cli.episode_length_s, 45.0)" in trainer_source
    assert "and not args_cli.c94_extended_episode_horizon" in trainer_source
    assert '"c94_extended_episode_horizon": (' in trainer_source


def test_slot_guidance_accepts_per_environment_target_feedforward_scale():
    pursuer = torch.tensor(
        [[[4.5, 0.0], [-2.25, 3.8971143], [-2.25, -3.8971143]]]
    ).repeat(2, 1, 1)
    result = moving_target_slot_guidance(
        torch.zeros_like(pursuer),
        pursuer,
        torch.zeros(2, 2),
        torch.tensor([[4.0, 0.0], [4.0, 0.0]]),
        torch.tensor([4.5, 4.5]),
        torch.ones(2, dtype=torch.bool),
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.0,
        target_feedforward_scale=torch.tensor([0.75, 1.0]),
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.15,
        maximum_anchor_turn_rate_rps=0.8,
        policy_dt_s=0.1,
        assignment_hysteresis_m=1.0,
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
        phase_separation_control_enabled=True,
        feedforward_priority_projection_enabled=True,
    )
    assert torch.allclose(result.velocity_xy[0, :, 0], torch.full((3,), 3.0))
    assert torch.allclose(result.velocity_xy[1, :, 0], torch.full((3,), 4.0))


def test_c42_adds_only_feedforward_priority_to_c41():
    c41 = native_curriculum_stage("c41")
    c42 = native_curriculum_stage("c42")
    assert c42.training_capture_radius_m == c41.training_capture_radius_m
    assert c42.desired_radius_m == c41.desired_radius_m
    assert c42.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c42":' in trainer_source
    assert 'configure_stage(cfg, "c41")' in trainer_source
    assert (
        "cfg.moving_target_slot_feedforward_priority_projection_enabled = True"
        in trainer_source
    )


def test_c43_centres_the_blocker_slot_on_the_live_target():
    c41 = native_curriculum_stage("c41")
    c43 = native_curriculum_stage("c43")
    assert c43.training_capture_radius_m == c41.training_capture_radius_m
    assert c43.desired_radius_m == c41.desired_radius_m
    assert c43.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c43":' in trainer_source
    assert 'configure_stage(cfg, "c41")' in trainer_source
    assert "cfg.reachable_intercept_blocker_only_slot_lead = False" in (
        trainer_source
    )


def test_intercept_lead_fades_only_inside_the_near_ring():
    horizon = torch.tensor([[1.0, 0.0], [0.8, 0.0], [1.0, 0.0]])
    reference = torch.tensor([12.0, 7.25, 5.5])
    released = release_intercept_horizon_during_contraction(
        horizon,
        reference,
        fade_start_radius_m=9.0,
        zero_lead_radius_m=5.5,
    )
    assert torch.allclose(released[0], horizon[0])
    assert torch.allclose(released[1], torch.tensor([0.4, 0.0]))
    assert torch.equal(released[2], torch.zeros(2))


def test_c44_fades_c41_blocker_lead_during_reference_contraction():
    c41 = native_curriculum_stage("c41")
    c44 = native_curriculum_stage("c44")
    assert c44.training_capture_radius_m == c41.training_capture_radius_m
    assert c44.desired_radius_m == c41.desired_radius_m
    assert c44.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c44":' in trainer_source
    assert "cfg.reachable_intercept_release_during_contraction = True" in (
        trainer_source
    )


def test_c45_stabilizes_the_strong_ring_confirmation_handoff():
    c41 = native_curriculum_stage("c41")
    c45 = native_curriculum_stage("c45")
    assert c45.training_capture_radius_m == c41.training_capture_radius_m
    assert c45.desired_radius_m == c41.desired_radius_m
    assert c45.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c45":' in trainer_source
    assert "cfg.strong_ring_confirmation_stabilization_enabled = True" in (
        trainer_source
    )
    assert "cfg.strong_ring_confirmation_suppress_policy_residual = True" in (
        trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "self._strong_ring_hold_steps > 0" in env_source
    assert "torch.zeros_like(policy_residual_scale)" in env_source


def test_c46_retains_policy_residual_during_confirmation_handoff():
    c41 = native_curriculum_stage("c41")
    c46 = native_curriculum_stage("c46")
    assert c46.training_capture_radius_m == c41.training_capture_radius_m
    assert c46.desired_radius_m == c41.desired_radius_m
    assert c46.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c46":' in trainer_source
    assert "cfg.strong_ring_confirmation_suppress_policy_residual = False" in (
        trainer_source
    )


def test_predictive_annulus_projection_brakes_outer_and_inner_drift():
    pursuer = torch.tensor([[[5.4, 0.0], [4.1, 0.0], [0.0, 4.5]]])
    desired = torch.tensor([[[3.0, 0.0], [-3.0, 0.0], [1.0, 0.0]]])
    result = project_predictive_target_annulus_velocity(
        desired,
        torch.zeros_like(desired),
        pursuer,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, dtype=torch.bool),
        minimum_radius_m=4.0,
        maximum_radius_m=5.5,
        prediction_horizon_s=0.5,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        projection_iterations=4,
    )
    assert result.corrected.item()
    assert not result.infeasible.item()
    assert result.minimum_predicted_radius_m.item() >= 4.0 - 1.0e-5
    assert result.maximum_predicted_radius_m.item() <= 5.5 + 1.0e-5
    assert torch.allclose(result.velocity_xy[0, 0], torch.tensor([0.4, 0.0]))
    assert torch.allclose(result.velocity_xy[0, 1], torch.tensor([-0.4, 0.0]))


def test_multinode_annulus_audit_detects_terminal_only_transient_escape():
    pursuer = torch.tensor([[[5.0, 0.0]]])
    current = torch.tensor([[[4.0, 0.0]]])
    desired = torch.tensor([[[-4.0, 0.0]]])
    common = dict(
        minimum_radius_m=4.0,
        maximum_radius_m=5.5,
        prediction_horizon_s=0.5,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        projection_iterations=4,
    )
    legacy = project_predictive_target_annulus_velocity(
        desired,
        current,
        pursuer,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, dtype=torch.bool),
        **common,
    )
    multinode = project_predictive_target_annulus_velocity(
        desired,
        current,
        pursuer,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, dtype=torch.bool),
        response_trajectory_substeps=5,
        **common,
    )
    assert not legacy.infeasible.item()
    assert legacy.maximum_predicted_radius_m.item() <= 5.5
    assert multinode.infeasible.item()
    assert multinode.maximum_predicted_radius_m.item() > 5.5


def test_multinode_outer_only_brake_does_not_compete_with_inner_recovery():
    pursuer = torch.tensor([[[3.9, 0.0], [5.4, 0.0]]])
    current = torch.zeros_like(pursuer)
    desired = torch.tensor([[[-2.0, 0.0], [3.0, 0.0]]])
    result = project_predictive_target_annulus_velocity(
        desired,
        current,
        pursuer,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, dtype=torch.bool),
        minimum_radius_m=4.0,
        maximum_radius_m=5.5,
        prediction_horizon_s=0.5,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        projection_iterations=4,
        response_trajectory_substeps=5,
        enforce_minimum_radius=False,
    )
    assert torch.equal(result.velocity_xy[0, 0], desired[0, 0])
    assert result.velocity_xy[0, 1, 0] < desired[0, 1, 0]


def test_multinode_inner_boundary_can_activate_after_legal_entry():
    pursuer = torch.tensor([[[3.9, 0.0]], [[3.9, 0.0]]])
    desired = torch.tensor([[[-2.0, 0.0]], [[-2.0, 0.0]]])
    result = project_predictive_target_annulus_velocity(
        desired,
        torch.zeros_like(pursuer),
        pursuer,
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.ones(2, dtype=torch.bool),
        minimum_radius_m=4.0,
        maximum_radius_m=5.5,
        prediction_horizon_s=0.5,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        projection_iterations=4,
        response_trajectory_substeps=5,
        enforce_minimum_radius=False,
        minimum_radius_active=torch.tensor([False, True]),
    )
    assert torch.equal(result.velocity_xy[0], desired[0])
    assert result.velocity_xy[1, 0, 0] > desired[1, 0, 0]


def test_dynamics_consistent_projection_uses_acceleration_limited_rollout():
    result = project_dynamics_consistent_target_annulus_velocity(
        torch.zeros(1, 1, 2),
        torch.tensor([[[2.0, 0.0]]]),
        torch.tensor([[[5.0, 0.0]]]),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, dtype=torch.bool),
        minimum_radius_m=4.0,
        maximum_radius_m=5.5,
        prediction_horizon_s=0.5,
        maximum_speed_mps=4.0,
        low_level_dt_s=0.01,
        velocity_kp=2.0,
        maximum_planar_acceleration_mps2=4.0,
        radial_candidate_count=33,
        minimum_radius_active=torch.zeros(1, dtype=torch.bool),
    )
    assert result.corrected.item()
    assert result.velocity_xy[0, 0, 0] < 0.0
    assert result.maximum_predicted_radius_m.item() <= 5.5 + 1.0e-4


def test_dynamics_consistent_projection_can_follow_target_velocity_change():
    result = project_dynamics_consistent_target_annulus_velocity(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.tensor([[[5.0, 0.0]]]),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, dtype=torch.bool),
        target_acceleration_xy=torch.tensor([[2.0, 0.0]]),
        target_maximum_speed_mps=4.0,
        minimum_radius_m=1.0,
        maximum_radius_m=100.0,
        prediction_horizon_s=0.5,
        maximum_speed_mps=4.0,
        low_level_dt_s=0.01,
        velocity_kp=2.0,
        maximum_planar_acceleration_mps2=4.0,
        radial_candidate_count=33,
        common_acceleration_feedforward_active=torch.ones(
            1, dtype=torch.bool
        ),
    )
    assert torch.allclose(
        result.velocity_xy, torch.tensor([[[1.0, 0.0]]]), atol=1.0e-5
    )


def test_preentry_sync_brakes_unavoidable_initial_annulus_violation():
    kwargs = {
        "desired_velocity_xy": torch.zeros(1, 1, 2),
        "current_velocity_xy": torch.tensor([[[-2.0, 0.0]]]),
        "pursuer_xy": torch.tensor([[[5.8, 0.0]]]),
        "target_xy": torch.zeros(1, 2),
        "target_velocity_xy": torch.zeros(1, 2),
        "active": torch.ones(1, dtype=torch.bool),
        "minimum_radius_m": 4.0,
        "maximum_radius_m": 5.5,
        "prediction_horizon_s": 0.5,
        "maximum_speed_mps": 4.0,
        "low_level_dt_s": 0.01,
        "velocity_kp": 2.0,
        "maximum_planar_acceleration_mps2": 4.0,
        "radial_candidate_count": 33,
        "minimum_radius_active": torch.zeros(1, dtype=torch.bool),
    }
    legacy = project_dynamics_consistent_target_annulus_velocity(**kwargs)
    synchronized = project_dynamics_consistent_target_annulus_velocity(
        **kwargs,
        preentry_velocity_sync_active=torch.ones(1, dtype=torch.bool),
    )
    assert legacy.infeasible.item() and synchronized.infeasible.item()
    assert legacy.correction_norm_mps.item() == 0.0
    assert synchronized.velocity_xy[0, 0, 0].item() > 0.0


def test_c47_adds_only_predictive_ring_stabilization_to_c41():
    c41 = native_curriculum_stage("c41")
    c47 = native_curriculum_stage("c47")
    assert c47.training_capture_radius_m == c41.training_capture_radius_m
    assert c47.desired_radius_m == c41.desired_radius_m
    assert c47.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c47":' in trainer_source
    assert "cfg.predictive_ring_stabilization_enabled = True" in trainer_source
    assert "cfg.predictive_ring_prediction_horizon_s = 0.5" in trainer_source


def test_c48_gates_c47_on_measured_team_radius():
    c47 = native_curriculum_stage("c47")
    c48 = native_curriculum_stage("c48")
    assert c48.training_capture_radius_m == c47.training_capture_radius_m
    assert c48.desired_radius_m == c47.desired_radius_m
    assert c48.static_obstacles == c47.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c48":' in trainer_source
    assert 'configure_stage(cfg, "c47")' in trainer_source
    assert "cfg.predictive_ring_activation_margin_m = 0.5" in trainer_source
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "current_radius.min(dim=1).values" in env_source
    assert "current_radius.max(dim=1).values" in env_source


def test_c49_latches_c48_after_measured_ring_entry():
    c48 = native_curriculum_stage("c48")
    c49 = native_curriculum_stage("c49")
    assert c49.training_capture_radius_m == c48.training_capture_radius_m
    assert c49.desired_radius_m == c48.desired_radius_m
    assert c49.static_obstacles == c48.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c49":' in trainer_source
    assert 'configure_stage(cfg, "c48")' in trainer_source
    assert (
        "cfg.predictive_ring_activation_latched_after_measured_entry = True"
        in trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "self._ring_projection_activation_latched |= active & measured_entry" in (
        env_source
    )
    assert "self._ring_projection_activation_latched[env_ids] = False" in env_source


def test_c50_delays_latched_projection_until_strong_ring_entry():
    c49 = native_curriculum_stage("c49")
    c50 = native_curriculum_stage("c50")
    assert c50.training_capture_radius_m == c49.training_capture_radius_m
    assert c50.desired_radius_m == c49.desired_radius_m
    assert c50.static_obstacles == c49.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c50":' in trainer_source
    assert 'configure_stage(cfg, "c49")' in trainer_source
    assert (
        "cfg.predictive_ring_activation_requires_strong_ring_entry = True"
        in trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "measured_entry & (self._strong_ring_hold_steps > 0)" in env_source


def test_c51_requires_frozen_maximum_gap_before_latched_projection():
    c49 = native_curriculum_stage("c49")
    c51 = native_curriculum_stage("c51")
    assert c51.training_capture_radius_m == c49.training_capture_radius_m
    assert c51.desired_radius_m == c49.desired_radius_m
    assert c51.static_obstacles == c49.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c51":' in trainer_source
    assert (
        "cfg.predictive_ring_activation_requires_maximum_gap_eligibility = True"
        in trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "circular_gap_statistics(\n                    pursuer_xy," in env_source
    assert "maximum_gap\n                    <= self.cfg.strong_ring_maximum_gap_rad" in (
        env_source
    )


def test_c52_uses_one_step_gap_reachability_after_strict_inner_entry():
    c51 = native_curriculum_stage("c51")
    c52 = native_curriculum_stage("c52")
    assert c52.training_capture_radius_m == c51.training_capture_radius_m
    assert c52.desired_radius_m == c51.desired_radius_m
    assert c52.static_obstacles == c51.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c52":' in trainer_source
    assert (
        "cfg.predictive_ring_activation_requires_inside_inner_boundary = True"
        in trainer_source
    )
    assert "2.0\n            * cfg.training_command_speed_mps" in trainer_source
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "current_radius.min(dim=1).values" in env_source
    assert "predictive_ring_activation_maximum_gap_margin_rad" in env_source


def test_c53_is_only_a_near_ring_reset_curriculum_for_c52():
    c52 = native_curriculum_stage("c52")
    c53 = native_curriculum_stage("c53")
    assert c53.training_capture_radius_m == c52.training_capture_radius_m
    assert c53.desired_radius_m == c52.desired_radius_m
    assert c53.static_obstacles == c52.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c53":' in trainer_source
    assert 'configure_stage(cfg, "c52")' in trainer_source
    assert "cfg.tracked_reset_reference_range_m = 8.0" in trainer_source
    assert "cfg.orientation_staging_radius_m = 8.0" in trainer_source


def test_c54_adds_only_a_dynamics_derived_alignment_margin_to_c53():
    c53 = native_curriculum_stage("c53")
    c54 = native_curriculum_stage("c54")
    assert c54.training_capture_radius_m == c53.training_capture_radius_m
    assert c54.desired_radius_m == c53.desired_radius_m
    assert c54.static_obstacles == c53.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c54":' in trainer_source
    assert 'configure_stage(cfg, "c53")' in trainer_source
    assert "cfg.orientation_staging_radius_m = 10.0" in trainer_source
    assert "8 + 2 = 10 m" in trainer_source


def test_acceleration_aware_annulus_predicts_bounded_target_displacement():
    pursuer = torch.tensor([[[-5.4, 0.0]]])
    desired = torch.tensor([[[2.0, 0.0]]])
    common = dict(
        minimum_radius_m=4.0,
        maximum_radius_m=5.5,
        prediction_horizon_s=0.5,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        projection_iterations=4,
    )
    cv = project_predictive_target_annulus_velocity(
        desired,
        desired,
        pursuer,
        torch.zeros(1, 2),
        torch.tensor([[2.0, 0.0]]),
        torch.ones(1, dtype=torch.bool),
        **common,
    )
    ca = project_predictive_target_annulus_velocity(
        desired,
        desired,
        pursuer,
        torch.zeros(1, 2),
        torch.tensor([[2.0, 0.0]]),
        torch.ones(1, dtype=torch.bool),
        target_acceleration_xy=torch.tensor([[4.0, 0.0]]),
        target_maximum_speed_mps=4.0,
        acceleration_prediction_substeps=5,
        **common,
    )
    assert not cv.corrected.item()
    assert ca.corrected.item()
    assert ca.velocity_xy[0, 0, 0] > cv.velocity_xy[0, 0, 0]


def test_c55_adds_only_target_acceleration_prediction_to_c54():
    c54 = native_curriculum_stage("c54")
    c55 = native_curriculum_stage("c55")
    assert c55.training_capture_radius_m == c54.training_capture_radius_m
    assert c55.desired_radius_m == c54.desired_radius_m
    assert c55.static_obstacles == c54.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c55":' in trainer_source
    assert 'configure_stage(cfg, "c54")' in trainer_source
    assert "cfg.predictive_ring_target_acceleration_enabled = True" in trainer_source


def test_c56_derives_activation_reachability_from_command_response():
    c55 = native_curriculum_stage("c55")
    c56 = native_curriculum_stage("c56")
    assert c56.training_capture_radius_m == c55.training_capture_radius_m
    assert c56.desired_radius_m == c55.desired_radius_m
    assert c56.static_obstacles == c55.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c56":' in trainer_source
    assert 'configure_stage(cfg, "c55")' in trainer_source
    assert "policy_dt_s / cfg.predictive_ring_command_response_ratio" in trainer_source


def test_c57_randomizes_training_speed_without_lowering_physical_capability():
    c55 = native_curriculum_stage("c55")
    c57 = native_curriculum_stage("c57")
    assert c57.training_capture_radius_m == c55.training_capture_radius_m
    assert c57.desired_radius_m == c55.desired_radius_m
    assert c57.static_obstacles == c55.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c57":' in trainer_source
    assert 'configure_stage(cfg, "c55")' in trainer_source
    assert "cfg.evader_minimum_episode_speed_mps = 2.0" in trainer_source
    assert "cfg.evader_speed_mps = 4.0" in trainer_source


def test_c58_broadens_only_the_first_speed_curriculum_rung():
    c57 = native_curriculum_stage("c57")
    c58 = native_curriculum_stage("c58")
    assert c58.training_capture_radius_m == c57.training_capture_radius_m
    assert c58.desired_radius_m == c57.desired_radius_m
    assert c58.static_obstacles == c57.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c58":' in trainer_source
    assert 'configure_stage(cfg, "c55")' in trainer_source
    assert "cfg.evader_minimum_episode_speed_mps = 1.0" in trainer_source


def test_c60_derives_radial_activation_margin_from_command_response():
    c56 = native_curriculum_stage("c56")
    c60 = native_curriculum_stage("c60")
    assert c60.training_capture_radius_m == c56.training_capture_radius_m
    assert c60.desired_radius_m == c56.desired_radius_m
    assert c60.static_obstacles == c56.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c60":' in trainer_source
    assert 'configure_stage(cfg, "c56")' in trainer_source
    assert "policy_dt_s / cfg.predictive_ring_command_response_ratio" in (
        trainer_source
    )
    assert "cfg.training_command_speed_mps * effective_response_s" in (
        trainer_source
    )


def test_c61_preserves_full_target_common_mode_before_corrections():
    c56 = native_curriculum_stage("c56")
    c61 = native_curriculum_stage("c61")
    assert c61.training_capture_radius_m == c56.training_capture_radius_m
    assert c61.desired_radius_m == c56.desired_radius_m
    assert c61.static_obstacles == c56.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c61":' in trainer_source
    assert 'configure_stage(cfg, "c56")' in trainer_source
    assert "cfg.moving_target_slot_feedforward_scale = 1.0" in trainer_source
    assert (
        "cfg.moving_target_slot_feedforward_priority_projection_enabled = True"
        in trainer_source
    )


def test_c62_restarts_from_neutral_direction_preserving_gain_residuals():
    c58 = native_curriculum_stage("c58")
    c62 = native_curriculum_stage("c62")
    assert c62.training_capture_radius_m == c58.training_capture_radius_m
    assert c62.desired_radius_m == c58.desired_radius_m
    assert c62.static_obstacles == c58.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c62":' in trainer_source
    assert 'configure_stage(cfg, "c58")' in trainer_source
    assert "cfg.moving_target_slot_gain_residual_control_enabled = True" in (
        trainer_source
    )
    assert 'args_cli.stage in ("c32", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79")' in trainer_source
    assert "gain-residual training requires a neutral actor output reset" in (
        trainer_source
    )


def test_c63_only_adds_dense_credit_for_a_legal_strong_ring_step():
    c62 = native_curriculum_stage("c62")
    c63 = native_curriculum_stage("c63")
    assert c63.training_capture_radius_m == c62.training_capture_radius_m
    assert c63.desired_radius_m == c62.desired_radius_m
    assert c63.static_obstacles == c62.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c63":' in trainer_source
    assert 'configure_stage(cfg, "c62")' in trainer_source
    assert "cfg.strong_ring_step_reward_weight = 0.5" in trainer_source
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "self._strong_ring_now.float()" in env_source
    assert "+ strong_ring_step_reward" in env_source


def test_c64_is_an_explicit_fixed_speed_gain_semantics_audit():
    c63 = native_curriculum_stage("c63")
    c64 = native_curriculum_stage("c64")
    assert c64.training_capture_radius_m == c63.training_capture_radius_m
    assert c64.desired_radius_m == c63.desired_radius_m
    assert c64.static_obstacles == c63.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c64":' in trainer_source
    assert 'configure_stage(cfg, "c63")' in trainer_source
    assert "cfg.evader_minimum_episode_speed_mps = 4.0" in trainer_source
    assert 'if stage in ("c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78")' in trainer_source


def test_c65_only_adds_multinode_response_projection_to_fixed4_c64():
    c64 = native_curriculum_stage("c64")
    c65 = native_curriculum_stage("c65")
    assert c65.training_capture_radius_m == c64.training_capture_radius_m
    assert c65.desired_radius_m == c64.desired_radius_m
    assert c65.static_obstacles == c64.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c65":' in trainer_source
    assert 'configure_stage(cfg, "c64")' in trainer_source
    assert "cfg.predictive_ring_response_trajectory_substeps = 5" in (
        trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "self.cfg.predictive_ring_response_trajectory_substeps" in env_source


def test_c66_only_removes_inner_competition_from_c65_execution_projection():
    c65 = native_curriculum_stage("c65")
    c66 = native_curriculum_stage("c66")
    assert c66.training_capture_radius_m == c65.training_capture_radius_m
    assert c66.desired_radius_m == c65.desired_radius_m
    assert c66.static_obstacles == c65.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c66":' in trainer_source
    assert 'configure_stage(cfg, "c65")' in trainer_source
    assert "cfg.predictive_ring_enforce_minimum_radius = False" in (
        trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "self.cfg.predictive_ring_enforce_minimum_radius" in env_source


def test_c67_reenables_inner_projection_only_after_a_legal_ring_step():
    c66 = native_curriculum_stage("c66")
    c67 = native_curriculum_stage("c67")
    assert c67.training_capture_radius_m == c66.training_capture_radius_m
    assert c67.desired_radius_m == c66.desired_radius_m
    assert c67.static_obstacles == c66.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c67":' in trainer_source
    assert 'configure_stage(cfg, "c66")' in trainer_source
    assert "cfg.predictive_ring_minimum_radius_after_strong_entry = True" in (
        trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "self._strong_ring_hold_steps > 0" in env_source


def test_c68_replaces_response_blend_with_low_level_forward_model():
    c67 = native_curriculum_stage("c67")
    c68 = native_curriculum_stage("c68")
    assert c68.training_capture_radius_m == c67.training_capture_radius_m
    assert c68.desired_radius_m == c67.desired_radius_m
    assert c68.static_obstacles == c67.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c68":' in trainer_source
    assert 'configure_stage(cfg, "c67")' in trainer_source
    assert "cfg.predictive_ring_dynamics_consistent_enabled = True" in (
        trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "project_dynamics_consistent_target_annulus_velocity(" in env_source
    assert "low_level_dt_s=self.cfg.sim.dt" in env_source


def test_c69_adds_common_target_acceleration_only_after_legal_entry():
    c68 = native_curriculum_stage("c68")
    c69 = native_curriculum_stage("c69")
    assert c69.training_capture_radius_m == c68.training_capture_radius_m
    assert c69.desired_radius_m == c68.desired_radius_m
    assert c69.static_obstacles == c68.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c69":' in trainer_source
    assert 'configure_stage(cfg, "c68")' in trainer_source
    assert (
        "cfg.predictive_ring_common_acceleration_feedforward_after_strong_entry = True"
        in trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "common_acceleration_feedforward_active=(" in env_source


def test_c70_adds_preentry_velocity_synchronization_to_c69():
    c69 = native_curriculum_stage("c69")
    c70 = native_curriculum_stage("c70")
    assert c70.training_capture_radius_m == c69.training_capture_radius_m
    assert c70.desired_radius_m == c69.desired_radius_m
    assert c70.static_obstacles == c69.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c70":' in trainer_source
    assert 'configure_stage(cfg, "c69")' in trainer_source
    assert "cfg.predictive_ring_preentry_velocity_sync_enabled = True" in (
        trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "preentry_velocity_sync_active=(" in env_source


def test_c71_gates_reference_contraction_on_measured_tracking_readiness():
    c69 = native_curriculum_stage("c69")
    c71 = native_curriculum_stage("c71")
    assert c71.training_capture_radius_m == c69.training_capture_radius_m
    assert c71.desired_radius_m == c69.desired_radius_m
    assert c71.static_obstacles == c69.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c71":' in trainer_source
    assert 'configure_stage(cfg, "c69")' in trainer_source
    assert "cfg.orientation_reference_contraction_tracking_gate_enabled = True" in (
        trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "half_ring_band_m = 0.5 * (" in env_source
    assert "half_ring_band_m / confirmation_duration_s" in env_source
    assert "reference_contraction_tracking_ready" in env_source


def test_c72_anchors_reference_to_measured_mean_only_on_first_unlock():
    c71 = native_curriculum_stage("c71")
    c72 = native_curriculum_stage("c72")
    assert c72.training_capture_radius_m == c71.training_capture_radius_m
    assert c72.desired_radius_m == c71.desired_radius_m
    assert c72.static_obstacles == c71.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c72":' in trainer_source
    assert 'configure_stage(cfg, "c71")' in trainer_source
    assert "cfg.orientation_reference_anchor_to_mean_radius_on_unlock = True" in (
        trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "newly_unlocked = (" in env_source
    assert "measured_reference_radius = distance.mean(dim=1).clamp(" in env_source


def test_c73_uses_full_target_feedforward_only_outside_the_strong_ring():
    c72 = native_curriculum_stage("c72")
    c73 = native_curriculum_stage("c73")
    assert c73.training_capture_radius_m == c72.training_capture_radius_m
    assert c73.desired_radius_m == c72.desired_radius_m
    assert c73.static_obstacles == c72.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c73":' in trainer_source
    assert 'configure_stage(cfg, "c72")' in trainer_source
    assert "cfg.moving_target_slot_full_feedforward_above_strong_ring = True" in (
        trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "self.cfg.moving_target_slot_full_feedforward_above_strong_ring" in (
        env_source
    )
    assert "> self.cfg.strong_ring_maximum_radius_m" in env_source


def test_c74_explicitly_prioritizes_common_target_feedforward():
    c73 = native_curriculum_stage("c73")
    c74 = native_curriculum_stage("c74")
    assert c74.training_capture_radius_m == c73.training_capture_radius_m
    assert c74.desired_radius_m == c73.desired_radius_m
    assert c74.static_obstacles == c73.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c74":' in trainer_source
    assert 'configure_stage(cfg, "c73")' in trainer_source
    assert (
        "cfg.moving_target_slot_feedforward_priority_projection_enabled = True"
        in trainer_source
    )
    assert (
        '"moving_target_slot_feedforward_priority_projection_enabled": ('
        in trainer_source
    )


def test_c75_adds_physical_radial_velocity_damping_only():
    c74 = native_curriculum_stage("c74")
    c75 = native_curriculum_stage("c75")
    assert c75.training_capture_radius_m == c74.training_capture_radius_m
    assert c75.desired_radius_m == c74.desired_radius_m
    assert c75.static_obstacles == c74.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c75":' in trainer_source
    assert 'configure_stage(cfg, "c74")' in trainer_source
    assert "math.sqrt(2.0 * cfg.moving_target_slot_radial_gain) - 1.0" in (
        trainer_source
    )
    assert '"moving_target_slot_radial_velocity_damping_gain": (' in (
        trainer_source
    )


def test_c76_only_removes_the_deadlocked_reference_tracking_gate():
    c75 = native_curriculum_stage("c75")
    c76 = native_curriculum_stage("c76")
    assert c76.training_capture_radius_m == c75.training_capture_radius_m
    assert c76.desired_radius_m == c75.desired_radius_m
    assert c76.static_obstacles == c75.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c76":' in trainer_source
    assert 'configure_stage(cfg, "c75")' in trainer_source
    assert "cfg.orientation_reference_contraction_tracking_gate_enabled = False" in (
        trainer_source
    )
    assert "if gate_passed\n                and (" in trainer_source


def test_c77_uses_low_level_response_time_for_ring_takeover_margin():
    c76 = native_curriculum_stage("c76")
    c77 = native_curriculum_stage("c77")
    assert c77.training_capture_radius_m == c76.training_capture_radius_m
    assert c77.desired_radius_m == c76.desired_radius_m
    assert c77.static_obstacles == c76.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c77":' in trainer_source
    assert 'configure_stage(cfg, "c76")' in trainer_source
    assert "policy_dt_s / cfg.predictive_ring_command_response_ratio" in (
        trainer_source
    )
    assert "cfg.predictive_ring_activation_maximum_gap_margin_rad = (" in (
        trainer_source
    )
    assert '"predictive_ring_activation_maximum_gap_margin_rad": (' in (
        trainer_source
    )


def test_c78_uses_existing_preentry_velocity_synchronization():
    c77 = native_curriculum_stage("c77")
    c78 = native_curriculum_stage("c78")
    assert c78.training_capture_radius_m == c77.training_capture_radius_m
    assert c78.desired_radius_m == c77.desired_radius_m
    assert c78.static_obstacles == c77.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c78":' in trainer_source
    assert 'configure_stage(cfg, "c77")' in trainer_source
    assert "cfg.predictive_ring_preentry_velocity_sync_enabled = True" in (
        trainer_source
    )


def test_c79_adds_only_exact_strong_ring_violation_progress_credit():
    c78 = native_curriculum_stage("c78")
    c79 = native_curriculum_stage("c79")
    assert c79.training_capture_radius_m == c78.training_capture_radius_m
    assert c79.desired_radius_m == c78.desired_radius_m
    assert c79.near_switch_m == c78.near_switch_m
    assert c79.static_obstacles == c78.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c79":' in trainer_source
    assert 'configure_stage(cfg, "c78")' in trainer_source
    assert "cfg.strong_ring_violation_progress_reward_weight = 0.5" in (
        trainer_source
    )
    assert '"p5_2_native_reward_v6_strong_ring_violation_progress"' in (
        trainer_source
    )

    cfg_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "strong_ring_violation_progress_reward_weight = 0.0" in cfg_source

    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "strong_ring_violation_score(" in env_source
    assert "self._previous_strong_ring_violation_score" in env_source
    assert ").clamp(-1.0, 1.0)" in env_source
    assert "+ strong_ring_violation_progress_reward" in env_source


def test_strong_ring_violation_score_matches_the_frozen_geometry():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    legal = 4.75 * torch.stack((angles.cos(), angles.sin()), dim=-1)[None]
    target = torch.zeros(1, 2)
    common = {
        "minimum_radius_m": 4.0,
        "maximum_radius_m": 5.5,
        "maximum_gap_rad": 5.0 * math.pi / 6.0,
        "maximum_mean_gap_error_rad": math.pi / 9.0,
    }
    legal_score = strong_ring_violation_score(legal, target, **common)
    assert torch.allclose(legal_score, torch.zeros_like(legal_score), atol=1.0e-6)

    outside = legal.clone()
    outside[:, 0] *= 6.0 / 4.75
    inside = legal.clone()
    inside[:, 1] *= 3.0 / 4.75
    bad_angles = torch.tensor([0.0, math.pi / 18.0, math.pi])
    angular = 4.75 * torch.stack(
        (bad_angles.cos(), bad_angles.sin()), dim=-1
    )[None]
    outside_score = strong_ring_violation_score(outside, target, **common)
    inside_score = strong_ring_violation_score(inside, target, **common)
    angular_score = strong_ring_violation_score(angular, target, **common)
    assert outside_score.item() > 0.0
    assert inside_score.item() > 0.0
    assert angular_score.item() > 0.0

    improvement = (outside_score - legal_score).clamp(-1.0, 1.0)
    regression = (legal_score - outside_score).clamp(-1.0, 1.0)
    unchanged = (outside_score - outside_score).clamp(-1.0, 1.0)
    assert improvement.item() > 0.0
    assert regression.item() < 0.0
    assert unchanged.item() == 0.0


def test_c80_repurposes_only_action_three_as_signed_radial_velocity():
    c79 = native_curriculum_stage("c79")
    c80 = native_curriculum_stage("c80")
    assert c80.training_capture_radius_m == c79.training_capture_radius_m
    assert c80.desired_radius_m == c79.desired_radius_m
    assert c80.near_switch_m == c79.near_switch_m
    assert c80.static_obstacles == c79.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c80":' in trainer_source
    assert 'configure_stage(cfg, "c79")' in trainer_source
    assert "cfg.moving_target_slot_radial_velocity_residual_control_enabled = True" in (
        trainer_source
    )
    assert '"radial_velocity_residual"' in trainer_source
    assert "cfg.strong_ring_required_hold_steps * policy_dt_s" in trainer_source
    score_block = trainer_source[
        trainer_source.index("            score = (") :
        trainer_source.index("            best_score = (")
    ]
    assert score_block.index('evaluation["success_rate_training"]') < (
        score_block.index('evaluation["maximum_strong_ring_hold_steps"]')
    )
    assert score_block.index('evaluation["maximum_strong_ring_hold_steps"]') < (
        score_block.index('evaluation["success_rate_6m"]')
    )

    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "radial_velocity_residual_action=(" in env_source
    assert "and not self.cfg.moving_target_slot_radial_velocity_residual_control_enabled" in (
        env_source
    )


def test_signed_radial_velocity_residual_has_direction_scale_and_phase_gate():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    radial_unit = torch.stack((angles.cos(), angles.sin()), dim=-1)[None]
    pursuer_xy = 4.75 * radial_unit
    common = dict(
        desired_velocity_xy=torch.zeros_like(pursuer_xy),
        pursuer_xy=pursuer_xy,
        estimated_target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.zeros(1, 2),
        reference_radius_m=torch.tensor([4.75]),
        active=torch.ones(1, dtype=torch.bool),
        previous_assignment=torch.tensor([[0, 1, 2]]),
        previous_anchor_heading_rad=torch.zeros(1),
        previous_anchor_valid=torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.0,
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
        phase_separation_control_enabled=True,
        phase_gain=0.8,
        radial_gain=1.0,
        maximum_tangential_speed_mps=3.0,
        gain_residual_action=torch.zeros_like(pursuer_xy),
        gain_residual_control_enabled=True,
        radial_velocity_residual_control_enabled=True,
        radial_velocity_residual_scale_mps=0.75,
    )
    neutral = moving_target_slot_guidance(
        **common,
        policy_residual_scale=0.2,
        radial_velocity_residual_action=torch.zeros(1, 3),
    )
    outward = moving_target_slot_guidance(
        **common,
        policy_residual_scale=0.2,
        radial_velocity_residual_action=torch.ones(1, 3),
    )
    inward = moving_target_slot_guidance(
        **common,
        policy_residual_scale=0.2,
        radial_velocity_residual_action=-torch.ones(1, 3),
    )
    gated = moving_target_slot_guidance(
        **common,
        policy_residual_scale=0.0,
        radial_velocity_residual_action=torch.ones(1, 3),
    )
    neutral_radial = (neutral.velocity_xy * radial_unit).sum(dim=-1)
    outward_radial = (outward.velocity_xy * radial_unit).sum(dim=-1)
    inward_radial = (inward.velocity_xy * radial_unit).sum(dim=-1)
    gated_radial = (gated.velocity_xy * radial_unit).sum(dim=-1)
    assert torch.allclose(neutral_radial, torch.zeros_like(neutral_radial), atol=1e-6)
    assert torch.allclose(outward_radial, torch.full_like(outward_radial, 0.75))
    assert torch.allclose(inward_radial, torch.full_like(inward_radial, -0.75))
    assert torch.allclose(gated_radial, torch.zeros_like(gated_radial), atol=1e-6)


def test_c81_repurposes_only_action_two_as_signed_tangential_velocity():
    c80 = native_curriculum_stage("c80")
    c81 = native_curriculum_stage("c81")
    assert c81.training_capture_radius_m == c80.training_capture_radius_m
    assert c81.desired_radius_m == c80.desired_radius_m
    assert c81.near_switch_m == c80.near_switch_m
    assert c81.static_obstacles == c80.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c81":' in trainer_source
    assert 'configure_stage(cfg, "c80")' in trainer_source
    assert (
        "cfg.moving_target_slot_tangential_velocity_residual_control_enabled = True"
        in trainer_source
    )
    assert '"tangential_velocity_residual"' in trainer_source
    assert "cfg.strong_ring_maximum_mean_gap_error_rad" in trainer_source
    assert "cfg.strong_ring_required_hold_steps * policy_dt_s" in trainer_source

    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "tangential_velocity_residual_action=(" in env_source
    assert "command.normalized_action[..., 1]" in env_source


def test_signed_tangential_velocity_residual_has_direction_scale_and_phase_gate():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    radial_unit = torch.stack((angles.cos(), angles.sin()), dim=-1)[None]
    tangent_unit = torch.stack((-angles.sin(), angles.cos()), dim=-1)[None]
    pursuer_xy = 4.75 * radial_unit
    scale_mps = 4.75 * (math.pi / 9.0) / 1.0
    common = dict(
        desired_velocity_xy=torch.zeros_like(pursuer_xy),
        pursuer_xy=pursuer_xy,
        estimated_target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.zeros(1, 2),
        reference_radius_m=torch.tensor([4.75]),
        active=torch.ones(1, dtype=torch.bool),
        previous_assignment=torch.tensor([[0, 1, 2]]),
        previous_anchor_heading_rad=torch.zeros(1),
        previous_anchor_valid=torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.0,
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
        phase_separation_control_enabled=True,
        phase_gain=0.8,
        radial_gain=1.0,
        maximum_tangential_speed_mps=3.0,
        gain_residual_action=torch.zeros_like(pursuer_xy),
        gain_residual_control_enabled=True,
        radial_velocity_residual_action=torch.zeros(1, 3),
        radial_velocity_residual_control_enabled=True,
        radial_velocity_residual_scale_mps=0.75,
        tangential_velocity_residual_control_enabled=True,
        tangential_velocity_residual_scale_mps=scale_mps,
    )
    neutral = moving_target_slot_guidance(
        **common,
        policy_residual_scale=0.2,
        tangential_velocity_residual_action=torch.zeros(1, 3),
    )
    positive = moving_target_slot_guidance(
        **common,
        policy_residual_scale=0.2,
        tangential_velocity_residual_action=torch.ones(1, 3),
    )
    negative = moving_target_slot_guidance(
        **common,
        policy_residual_scale=0.2,
        tangential_velocity_residual_action=-torch.ones(1, 3),
    )
    gated = moving_target_slot_guidance(
        **common,
        policy_residual_scale=0.0,
        tangential_velocity_residual_action=torch.ones(1, 3),
    )
    positive_delta = ((positive.velocity_xy - neutral.velocity_xy) * tangent_unit).sum(dim=-1)
    negative_delta = ((negative.velocity_xy - neutral.velocity_xy) * tangent_unit).sum(dim=-1)
    gated_delta = ((gated.velocity_xy - neutral.velocity_xy) * tangent_unit).sum(dim=-1)
    assert torch.allclose(positive_delta, torch.full_like(positive_delta, scale_mps))
    assert torch.allclose(negative_delta, torch.full_like(negative_delta, -scale_mps))
    assert torch.allclose(gated_delta, torch.zeros_like(gated_delta), atol=1e-6)


def test_c82_inherits_c80_and_locks_only_confirmation_phase_gain():
    c80 = native_curriculum_stage("c80")
    c82 = native_curriculum_stage("c82")
    assert c82.training_capture_radius_m == c80.training_capture_radius_m
    assert c82.desired_radius_m == c80.desired_radius_m
    assert c82.near_switch_m == c80.near_switch_m
    assert c82.static_obstacles == c80.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c82":' in trainer_source
    assert 'configure_stage(cfg, "c80")' in trainer_source
    assert (
        "cfg.strong_ring_confirmation_maximum_phase_gain_enabled = True"
        in trainer_source
    )
    assert 'if args_cli.stage in ("c80", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91")' in trainer_source

    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "confirmation_control_enabled = (" in env_source
    assert "self._strong_ring_hold_steps > 0" in env_source
    assert "strong_ring_confirmation_maximum_phase_gain_enabled" in env_source
    assert "torch.ones_like(gain_residual_action[..., 1])" in env_source
    assert "gain_residual_action=gain_residual_action" in env_source


def test_c83_inherits_c80_and_derives_three_agent_critical_phase_damping():
    c80 = native_curriculum_stage("c80")
    c83 = native_curriculum_stage("c83")
    assert c83.training_capture_radius_m == c80.training_capture_radius_m
    assert c83.desired_radius_m == c80.desired_radius_m
    assert c83.near_switch_m == c80.near_switch_m
    assert c83.static_obstacles == c80.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c83":' in trainer_source
    assert 'configure_stage(cfg, "c80")' in trainer_source
    assert "graph_laplacian_eigenvalue = float(agent_count)" in trainer_source
    assert "cfg.moving_target_slot_phase_angular_velocity_damping_gain_s" in (
        trainer_source
    )
    assert "expected_action_semantics=action_semantics_contract(args_cli.stage)" in (
        trainer_source
    )
    assert "checkpoint action semantics differ" in trainer_source


def test_c84_confines_critical_phase_damping_to_confirmation_window():
    c80 = native_curriculum_stage("c80")
    c84 = native_curriculum_stage("c84")
    assert c84.training_capture_radius_m == c80.training_capture_radius_m
    assert c84.desired_radius_m == c80.desired_radius_m
    assert c84.near_switch_m == c80.near_switch_m
    assert c84.static_obstacles == c80.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c84":' in trainer_source
    assert 'configure_stage(cfg, "c83")' in trainer_source
    assert (
        "cfg.moving_target_slot_phase_angular_velocity_damping_confirmation_only = True"
        in trainer_source
    )
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "phase_angular_velocity_damping_active=(" in env_source
    assert "self._strong_ring_confirmation_stabilization_active" in env_source


def test_c85_derives_discrete_confirmation_phase_rate_cancellation():
    c84 = native_curriculum_stage("c84")
    c85 = native_curriculum_stage("c85")
    assert c85.training_capture_radius_m == c84.training_capture_radius_m
    assert c85.desired_radius_m == c84.desired_radius_m
    assert c85.near_switch_m == c84.near_switch_m
    assert c85.static_obstacles == c84.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c85":' in trainer_source
    assert 'configure_stage(cfg, "c84")' in trainer_source
    assert "low_level_response_fraction = 1.0 - (" in trainer_source
    assert ") ** cfg.decimation" in trainer_source
    assert "(low_level_response_fraction * graph_laplacian_eigenvalue)" in (
        trainer_source
    )


def test_c86_inherits_c80_and_enables_only_predictive_angular_gap_projection():
    c80 = native_curriculum_stage("c80")
    c86 = native_curriculum_stage("c86")
    assert c86.training_capture_radius_m == c80.training_capture_radius_m
    assert c86.desired_radius_m == c80.desired_radius_m
    assert c86.near_switch_m == c80.near_switch_m
    assert c86.static_obstacles == c80.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c86":' in trainer_source
    assert 'configure_stage(cfg, "c80")' in trainer_source
    assert "cfg.predictive_ring_angular_gap_projection_enabled = True" in (
        trainer_source
    )


def _c86_projection_common(
    pursuer_xy: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    desired_velocity_xy: torch.Tensor,
    active: torch.Tensor,
):
    return project_dynamics_consistent_angular_gap_velocity(
        desired_velocity_xy,
        current_velocity_xy,
        pursuer_xy,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        active,
        minimum_radius_m=4.0,
        maximum_radius_m=5.5,
        maximum_gap_rad=5.0 * math.pi / 6.0,
        maximum_mean_gap_error_rad=math.pi / 9.0,
        prediction_horizon_s=0.5,
        maximum_speed_mps=4.0,
        low_level_dt_s=0.01,
        velocity_kp=2.0,
        maximum_planar_acceleration_mps2=4.0,
        correction_candidate_count=17,
        minimum_radius_active=torch.ones(1, dtype=torch.bool),
    )


def test_predictive_angular_gap_projection_leaves_legal_stationary_ring_unchanged():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    pursuer_xy = 4.75 * torch.stack((angles.cos(), angles.sin()), dim=-1)[None]
    zero_velocity = torch.zeros_like(pursuer_xy)
    result = _c86_projection_common(
        pursuer_xy,
        zero_velocity,
        zero_velocity,
        torch.ones(1, dtype=torch.bool),
    )
    assert result.active.item()
    assert not result.corrected.item()
    assert not result.infeasible.item()
    assert torch.allclose(result.velocity_xy, zero_velocity, atol=1.0e-7)
    assert result.maximum_predicted_gap_rad.item() < 5.0 * math.pi / 6.0
    assert result.maximum_predicted_mean_gap_error_rad.item() < math.pi / 9.0


def test_predictive_angular_gap_projection_reduces_predicted_gap_drift():
    angles = torch.deg2rad(torch.tensor([0.0, 100.0, 240.0]))
    radial_unit = torch.stack((angles.cos(), angles.sin()), dim=-1)[None]
    tangent_unit = torch.stack((-angles.sin(), angles.cos()), dim=-1)[None]
    pursuer_xy = 4.75 * radial_unit
    current_velocity = torch.zeros_like(pursuer_xy)
    current_velocity[:, 0] = 4.0 * tangent_unit[:, 0]
    desired_velocity = current_velocity.clone()
    nominal = _c86_projection_common(
        pursuer_xy,
        current_velocity,
        desired_velocity,
        torch.zeros(1, dtype=torch.bool),
    )
    projected = _c86_projection_common(
        pursuer_xy,
        current_velocity,
        desired_velocity,
        torch.ones(1, dtype=torch.bool),
    )
    assert projected.corrected.item()
    assert projected.correction_norm_mps.item() > 0.0
    assert (
        projected.maximum_predicted_gap_rad.item()
        < nominal.maximum_predicted_gap_rad.item()
    )
    assert (
        projected.maximum_predicted_mean_gap_error_rad.item()
        < nominal.maximum_predicted_mean_gap_error_rad.item()
    )


def test_c87_inherits_c86_and_enables_preentry_feasibility_reserve():
    c86 = native_curriculum_stage("c86")
    c87 = native_curriculum_stage("c87")
    assert c87.training_capture_radius_m == c86.training_capture_radius_m
    assert c87.desired_radius_m == c86.desired_radius_m
    assert c87.near_switch_m == c86.near_switch_m
    assert c87.static_obstacles == c86.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c87":' in trainer_source
    assert 'configure_stage(cfg, "c86")' in trainer_source
    assert "cfg.predictive_ring_preentry_braking_enabled = True" in trainer_source
    assert "cfg.predictive_ring_angular_gap_preentry_enabled = True" in (
        trainer_source
    )
    assert "cfg.training_near_switch_m" in trainer_source


def test_c88_keeps_only_the_preentry_braking_ablation():
    c86 = native_curriculum_stage("c86")
    c88 = native_curriculum_stage("c88")
    assert c88.training_capture_radius_m == c86.training_capture_radius_m
    assert c88.desired_radius_m == c86.desired_radius_m
    assert c88.near_switch_m == c86.near_switch_m
    assert c88.static_obstacles == c86.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    c88_branch = trainer_source.split('if stage == "c88":', 1)[1].split(
        'if stage == "c87":', 1
    )[0]
    assert 'configure_stage(cfg, "c86")' in c88_branch
    assert "cfg.predictive_ring_preentry_braking_enabled = True" in c88_branch
    assert "predictive_ring_angular_gap_preentry_enabled" not in c88_branch


def test_c89_adds_bounded_last_seen_cv_target_safety_to_c88():
    c88 = native_curriculum_stage("c88")
    c89 = native_curriculum_stage("c89")
    assert c89.training_capture_radius_m == c88.training_capture_radius_m
    assert c89.desired_radius_m == c88.desired_radius_m
    assert c89.near_switch_m == c88.near_switch_m
    assert c89.static_obstacles == c88.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    c89_branch = trainer_source.split('if stage == "c89":', 1)[1].split(
        'if stage == "c88":', 1
    )[0]
    assert 'configure_stage(cfg, "c88")' in c89_branch
    assert "cfg.predictive_target_last_seen_cv_during_loss_enabled = True" in (
        c89_branch
    )
    assert "maximum_loss_time_s = 2.0" in c89_branch


def test_bounded_last_seen_cv_target_state_never_uses_hidden_measurement():
    measured_xy = torch.tensor([[99.0, 99.0], [2.0, 3.0], [77.0, 77.0]])
    measured_velocity = torch.tensor([[9.0, 9.0], [0.5, 0.0], [8.0, 8.0]])
    last_xy = torch.tensor([[1.0, 2.0], [8.0, 8.0], [4.0, 5.0]])
    last_velocity = torch.tensor([[2.0, -1.0], [7.0, 7.0], [1.0, 0.0]])
    xy, velocity, active = bounded_last_seen_cv_target_state(
        measured_xy,
        measured_velocity,
        last_xy,
        last_velocity,
        torch.tensor([False, True, False]),
        torch.tensor([3, 0, 25]),
        policy_dt_s=0.1,
        maximum_loss_time_s=2.0,
    )
    assert torch.allclose(xy[0], torch.tensor([1.6, 1.7]))
    assert torch.equal(velocity[0], last_velocity[0])
    assert active[0]
    assert torch.equal(xy[1], measured_xy[1])
    assert torch.equal(velocity[1], measured_velocity[1])
    assert active[1]
    assert torch.equal(xy[2], torch.tensor([6.5, 5.0]))
    assert not active[2]


def test_c90_expands_c89_cv_state_into_acceleration_reachable_tube():
    c89 = native_curriculum_stage("c89")
    c90 = native_curriculum_stage("c90")
    assert c90.training_capture_radius_m == c89.training_capture_radius_m
    assert c90.desired_radius_m == c89.desired_radius_m
    assert c90.near_switch_m == c89.near_switch_m
    assert c90.static_obstacles == c89.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    c90_branch = trainer_source.split('if stage == "c90":', 1)[1].split(
        'if stage == "c89":', 1
    )[0]
    assert 'configure_stage(cfg, "c89")' in c90_branch
    assert "cfg.predictive_target_last_seen_reachable_tube_enabled = True" in (
        c90_branch
    )


def test_target_projection_accepts_per_environment_reachable_tube_radius():
    pursuer_xy = torch.tensor([[[4.0, 0.0]], [[4.0, 0.0]]])
    desired_velocity = torch.tensor([[[-4.0, 0.0]], [[-4.0, 0.0]]])
    result = project_predictive_target_separation_velocity(
        desired_velocity,
        torch.zeros_like(desired_velocity),
        pursuer_xy,
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.ones(2, dtype=torch.bool),
        minimum_center_distance_m=torch.tensor([3.0, 6.0]),
        prediction_horizon_s=0.75,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        projection_iterations=2,
        restore_unsafe_buffer=True,
    )
    assert result.corrected.tolist() == [True, True]
    assert result.velocity_xy[1, 0, 0] > result.velocity_xy[0, 0, 0]
    assert result.unavoidable.tolist() == [False, True]


def test_c91_selects_a_safe_scout_and_holds_trackers_during_loss():
    c88 = native_curriculum_stage("c88")
    c91 = native_curriculum_stage("c91")
    assert c91.training_capture_radius_m == c88.training_capture_radius_m
    assert c91.desired_radius_m == c88.desired_radius_m
    assert c91.near_switch_m == c88.near_switch_m
    assert c91.static_obstacles == c88.static_obstacles

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    c91_branch = trainer_source.split('if stage == "c91":', 1)[1].split(
        'if stage == "c90":', 1
    )[0]
    assert 'configure_stage(cfg, "c88")' in c91_branch
    assert "cfg.training_near_switch_m" in c91_branch
    assert "cfg.loss_reacquisition_tracker_hold_enabled = True" in c91_branch


def test_preentry_braking_reserve_limits_only_late_inward_command():
    pursuer_xy = torch.tensor([[[5.0, 0.0], [0.0, 8.0], [-10.0, 0.0]]])
    radial_unit = pursuer_xy / pursuer_xy.norm(dim=-1, keepdim=True)
    desired_velocity = -4.0 * radial_unit
    current_velocity = -2.0 * radial_unit
    result = project_preentry_braking_reserve_velocity(
        desired_velocity,
        current_velocity,
        pursuer_xy,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, dtype=torch.bool),
        reference_radius_m=4.75,
        activation_radius_m=9.0,
        maximum_speed_mps=4.0,
        maximum_planar_acceleration_mps2=4.0,
    )
    achieved_radial = (result.velocity_xy * radial_unit).sum(dim=-1)
    assert result.active.item()
    assert result.corrected.item()
    assert result.limited_agents.item() == 1
    assert math.isclose(
        achieved_radial[0, 0].item(), -math.sqrt(2.0), rel_tol=1.0e-5
    )
    assert math.isclose(achieved_radial[0, 1].item(), -4.0, abs_tol=1.0e-6)
    assert math.isclose(achieved_radial[0, 2].item(), -4.0, abs_tol=1.0e-6)
    assert result.minimum_stopping_margin_m.item() < 0.0


def test_preentry_braking_reserve_leaves_feasible_command_unchanged():
    pursuer_xy = torch.tensor([[[6.0, 0.0], [0.0, 6.0], [-6.0, 0.0]]])
    radial_unit = pursuer_xy / pursuer_xy.norm(dim=-1, keepdim=True)
    desired_velocity = -radial_unit
    result = project_preentry_braking_reserve_velocity(
        desired_velocity,
        torch.zeros_like(desired_velocity),
        pursuer_xy,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, dtype=torch.bool),
        reference_radius_m=4.75,
        activation_radius_m=9.0,
        maximum_speed_mps=4.0,
        maximum_planar_acceleration_mps2=4.0,
    )
    assert result.active.item()
    assert not result.corrected.item()
    assert not result.infeasible.item()
    assert torch.equal(result.velocity_xy, desired_velocity)


def test_preentry_angular_projection_ignores_outer_annulus_until_entry():
    angles = torch.deg2rad(torch.tensor([0.0, 100.0, 240.0]))
    radial_unit = torch.stack((angles.cos(), angles.sin()), dim=-1)[None]
    tangent_unit = torch.stack((-angles.sin(), angles.cos()), dim=-1)[None]
    pursuer_xy = 8.0 * radial_unit
    current_velocity = torch.zeros_like(pursuer_xy)
    current_velocity[:, 0] = 4.0 * tangent_unit[:, 0]
    result = project_dynamics_consistent_angular_gap_velocity(
        current_velocity,
        current_velocity,
        pursuer_xy,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, dtype=torch.bool),
        minimum_radius_m=4.0,
        maximum_radius_m=5.5,
        maximum_gap_rad=5.0 * math.pi / 6.0,
        maximum_mean_gap_error_rad=math.pi / 9.0,
        prediction_horizon_s=0.5,
        maximum_speed_mps=4.0,
        low_level_dt_s=0.01,
        velocity_kp=2.0,
        maximum_planar_acceleration_mps2=4.0,
        correction_candidate_count=17,
        minimum_radius_active=torch.zeros(1, dtype=torch.bool),
        radial_constraints_active=torch.zeros(1, dtype=torch.bool),
    )
    assert result.corrected.item()
    assert not result.infeasible.item()
    assert result.maximum_predicted_radius_m.item() > 5.5


def test_phase_angular_velocity_damping_opposes_gap_rate_laplacian():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    radial_unit = torch.stack((angles.cos(), angles.sin()), dim=-1)[None]
    tangent_unit = torch.stack((-angles.sin(), angles.cos()), dim=-1)[None]
    pursuer_xy = 4.75 * radial_unit
    pursuer_velocity_xy = torch.zeros_like(pursuer_xy)
    pursuer_velocity_xy[:, 0] = tangent_unit[:, 0]
    damping_gain_s = 2.0 * (math.sqrt(2.4) - 1.0) / 3.0
    common = dict(
        desired_velocity_xy=torch.zeros_like(pursuer_xy),
        pursuer_xy=pursuer_xy,
        estimated_target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.zeros(1, 2),
        reference_radius_m=torch.tensor([4.75]),
        active=torch.ones(1, dtype=torch.bool),
        previous_assignment=torch.tensor([[0, 1, 2]]),
        previous_anchor_heading_rad=torch.zeros(1),
        previous_anchor_valid=torch.ones(1, dtype=torch.bool),
        pursuer_velocity_xy=pursuer_velocity_xy,
        position_gain=1.0,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.0,
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
        phase_separation_control_enabled=True,
        phase_gain=0.8,
        radial_gain=1.0,
        maximum_tangential_speed_mps=3.0,
        policy_residual_scale=0.0,
    )
    undamped = moving_target_slot_guidance(
        **common, phase_angular_velocity_damping_gain_s=0.0
    )
    damped = moving_target_slot_guidance(
        **common, phase_angular_velocity_damping_gain_s=damping_gain_s
    )
    tangential_delta = (
        (damped.velocity_xy - undamped.velocity_xy) * tangent_unit
    ).sum(dim=-1)
    expected = torch.tensor(
        [[-damping_gain_s, 0.5 * damping_gain_s, 0.5 * damping_gain_s]]
    )
    assert torch.allclose(tangential_delta, expected, atol=1e-6)
    confirmation_inactive = moving_target_slot_guidance(
        **common,
        phase_angular_velocity_damping_gain_s=damping_gain_s,
        phase_angular_velocity_damping_active=torch.tensor([False]),
    )
    confirmation_active = moving_target_slot_guidance(
        **common,
        phase_angular_velocity_damping_gain_s=damping_gain_s,
        phase_angular_velocity_damping_active=torch.tensor([True]),
    )
    assert torch.allclose(confirmation_inactive.velocity_xy, undamped.velocity_xy)
    assert torch.allclose(confirmation_active.velocity_xy, damped.velocity_xy)


def test_training_evaluations_can_reuse_an_explicit_seed_plan():
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert "--evaluation_seed_plan is only valid with --evaluation_only" not in (
        trainer_source
    )
    assert trainer_source.count("episode_seed_plan=evaluation_seed_plan") >= 2
    reset_guard = (
        PACKAGE_SOURCE / "bylw_isaac_p5/core/evaluation_reset.py"
    ).read_text(encoding="utf-8")
    assert "@standard_reset_evaluation" in trainer_source
    assert "env.clear_forced_episode_seed_plan()" in reset_guard
    assert "env.reset(seed=training_seed)" in reset_guard


def test_evaluation_trace_is_explicit_and_evaluation_only():
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert '"--evaluation_trace"' in trainer_source
    assert "--evaluation_trace requires --evaluation_only" in trainer_source
    assert 'output_dir / "evaluation_trace.csv"' in trainer_source


def test_c31_inherits_c30_and_enables_phase_locked_topology():
    c30 = native_curriculum_stage("c30")
    c31 = native_curriculum_stage("c31")
    assert c31.training_capture_radius_m == c30.training_capture_radius_m
    assert c31.desired_radius_m == c30.desired_radius_m
    assert c31.near_switch_m == c30.near_switch_m
    assert c31.static_obstacles == c30.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c31":' in trainer_source
    assert 'configure_stage(cfg, "c30")' in trainer_source
    assert "cfg.moving_target_slot_cyclic_order_preserving_assignment = True" in (
        trainer_source
    )
    assert "cfg.moving_target_slot_lock_valid_assignment = True" in trainer_source
    assert "cfg.moving_target_slot_phase_separation_control_enabled = True" in (
        trainer_source
    )
    assert "cfg.orientation_phase_gate_enabled = True" in trainer_source
    assert "cfg.hybrid_rl_activation_on_staging = True" in trainer_source
    assert "traceback.print_exc()" in trainer_source
    env_source = (
        ROOT
        / "isaac_p5/source/bylw_isaac_p5/bylw_isaac_p5/tasks/direct"
        / "bylw_pursuit/bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "self._orientation_contraction_unlocked," in env_source
    assert "self._orientation_reference_radius_m," in env_source
    assert "self._hybrid_policy_residual_scale" in env_source
    assert '"hybrid_traditional_steps": (' in env_source
    assert '"hybrid_rl_steps": self._hybrid_rl_steps.clone()' in env_source


def test_c32_preserves_c31_contract_and_uses_bounded_gain_residuals():
    c31 = native_curriculum_stage("c31")
    c32 = native_curriculum_stage("c32")
    assert c32.training_capture_radius_m == c31.training_capture_radius_m
    assert c32.desired_radius_m == c31.desired_radius_m
    assert c32.near_switch_m == c31.near_switch_m
    assert c32.static_obstacles == c31.static_obstacles
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c32":' in trainer_source
    assert 'configure_stage(cfg, "c31")' in trainer_source
    assert "cfg.moving_target_slot_gain_residual_control_enabled = True" in (
        trainer_source
    )
    assert "cfg.moving_target_slot_radial_gain_residual_scale = 0.25" in (
        trainer_source
    )
    assert "cfg.moving_target_slot_phase_gain_residual_scale = 0.25" in (
        trainer_source
    )
    assert "cfg.orientation_yaw_gain_residual_scale = 0.25" in trainer_source
    assert "gain-residual training requires a neutral actor output reset" in (
        trainer_source
    )
from bylw_isaac_p5.core.reset_sampler import (  # noqa: E402
    sample_irregular_obstacle_layout,
)
from algos.mappo import MAPPO, expand_mlp_actor_observation_state  # noqa: E402


def _state_tensors(batch=2):
    pursuer_position = torch.tensor(
        [[[10.0, 0.0, 3.0], [-5.0, 8.0, 3.0], [-5.0, -8.0, 3.0]]]
    ).expand(batch, -1, -1).clone()
    quaternion = torch.zeros(batch, 3, 4)
    quaternion[..., 0] = 1.0
    velocity = torch.zeros(batch, 3, 3)
    angular_velocity = torch.zeros(batch, 3, 3)
    target_position = torch.tensor([[0.0, 0.0, 3.0]]).expand(batch, -1).clone()
    target_velocity = torch.zeros(batch, 3)
    obstacle_xy = torch.tensor(
        [[[-40.0, -40.0], [-40.0, 40.0], [40.0, -40.0], [40.0, 40.0]]]
    ).expand(batch, -1, -1).clone()
    obstacle_radius = torch.full((batch, 4), 2.0)
    return (
        pursuer_position,
        quaternion,
        velocity,
        angular_velocity,
        target_position,
        target_velocity,
        obstacle_xy,
        obstacle_radius,
    )


def _loss_guidance_call(
    *,
    desired_velocity_xy=None,
    pursuer_xy=None,
    team_visible=None,
    lost_steps=None,
    previous_active=None,
    previous_scout_index=None,
    previous_slot_angle_rad=None,
    enabled=True,
    scout_minimum_selection_distance_m=0.0,
):
    pursuer_xy = (
        torch.tensor([[[2.0, 0.0], [10.0, 0.0], [0.0, 10.0]]])
        if pursuer_xy is None
        else pursuer_xy
    )
    batch, agents, _ = pursuer_xy.shape
    return loss_reacquisition_guidance(
        torch.zeros_like(pursuer_xy)
        if desired_velocity_xy is None
        else desired_velocity_xy,
        pursuer_xy,
        torch.zeros(batch, 2),
        torch.zeros(batch, dtype=torch.bool)
        if team_visible is None
        else team_visible,
        torch.ones(batch, dtype=torch.long) if lost_steps is None else lost_steps,
        torch.zeros(batch, dtype=torch.bool)
        if previous_active is None
        else previous_active,
        torch.full((batch,), -1, dtype=torch.long)
        if previous_scout_index is None
        else previous_scout_index,
        torch.zeros(batch, agents)
        if previous_slot_angle_rad is None
        else previous_slot_angle_rad,
        enabled=enabled,
        start_lost_steps=1,
        slot_radius_m=8.0,
        scout_blend=0.85,
        tracker_blend=0.55,
        blend_growth=0.10,
        scout_speed_ratio=1.0,
        tracker_speed_ratio=0.65,
        slot_arrival_m=1.5,
        maximum_speed_mps=4.0,
        lost_cap_steps=50,
        scout_minimum_selection_distance_m=(
            scout_minimum_selection_distance_m
        ),
    )


def test_loss_guidance_selects_nearest_scout_outside_safety_radius():
    pursuer_xy = torch.tensor([[[2.0, 0.0], [10.0, 0.0], [12.0, 0.0]]])
    default = _loss_guidance_call(pursuer_xy=pursuer_xy)
    safe = _loss_guidance_call(
        pursuer_xy=pursuer_xy,
        scout_minimum_selection_distance_m=9.0,
    )
    assert default.scout_index.item() == 0
    assert safe.scout_index.item() == 1


def test_loss_reacquisition_selects_closest_scout_and_equal_tracker_slots():
    result = _loss_guidance_call()
    assert result.active.item()
    assert result.guidance_active.item()
    assert result.activated.item()
    assert result.scout_index.item() == 0
    assert torch.allclose(result.goal_xy[0, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(
        result.goal_xy[0, 1:].norm(dim=-1),
        torch.full((2,), 8.0),
        atol=1e-5,
    )
    assert result.velocity_xy[0, 0, 0].item() < 0.0
    assert result.velocity_xy.norm(dim=-1).max().item() <= 4.0 + 1e-6


def test_loss_reacquisition_locks_scout_until_visibility_returns():
    first = _loss_guidance_call()
    moved = torch.tensor([[[20.0, 0.0], [0.5, 0.0], [0.0, 10.0]]])
    second = _loss_guidance_call(
        pursuer_xy=moved,
        lost_steps=torch.tensor([2]),
        previous_active=first.active,
        previous_scout_index=first.scout_index,
        previous_slot_angle_rad=first.slot_angle_rad,
    )
    assert second.active.item()
    assert not second.activated.item()
    assert second.scout_index.item() == 0
    assert torch.allclose(second.slot_angle_rad, first.slot_angle_rad)


def test_loss_reacquisition_disabled_preserves_actions_but_tracks_event():
    nominal = torch.tensor(
        [[[0.25, -0.5], [1.0, 0.0], [-0.75, 0.25]]], dtype=torch.float32
    )
    result = _loss_guidance_call(
        desired_velocity_xy=nominal,
        enabled=False,
    )
    assert result.active.item()
    assert result.activated.item()
    assert not result.guidance_active.item()
    assert torch.equal(result.velocity_xy, nominal)


def test_loss_reacquisition_marks_reobservation_and_releases_action():
    nominal = torch.tensor(
        [[[0.25, -0.5], [1.0, 0.0], [-0.75, 0.25]]], dtype=torch.float32
    )
    first = _loss_guidance_call()
    visible = _loss_guidance_call(
        desired_velocity_xy=nominal,
        team_visible=torch.tensor([True]),
        lost_steps=torch.tensor([0]),
        previous_active=first.active,
        previous_scout_index=first.scout_index,
        previous_slot_angle_rad=first.slot_angle_rad,
    )
    assert not visible.active.item()
    assert visible.just_reacquired.item()
    assert not visible.guidance_active.item()
    assert torch.equal(visible.velocity_xy, nominal)


def test_occlusion_flank_selects_first_blocker_and_restores_clear_line():
    pursuer = torch.tensor([[[-10.0, 0.0], [-20.0, 15.0], [-20.0, -15.0]]])
    result = occlusion_flank_waypoint(
        pursuer,
        torch.tensor([[10.0, 0.0]]),
        torch.tensor([[[0.0, 0.0], [4.0, 12.0]]]),
        torch.tensor([[2.0, 2.0]]),
        torch.tensor([0]),
        pursuer_radius_m=0.35,
        minimum_path_clearance_m=1.5,
        waypoint_margin_m=0.5,
        los_margin_m=0.25,
        blocker_margin_m=0.25,
        arena_half_extent_m=50.0,
        boundary_clearance_m=1.0,
        angular_samples=32,
    )
    assert result.valid.item()
    assert result.blocker_index.item() == 0
    assert abs(result.waypoint_xy[0, 1].item()) > 2.0
    assert result.path_clearance_m.item() >= -1.0e-5
    assert result.los_clearance_m.item() >= -1.0e-5
    assert result.waypoint_xy.abs().max().item() <= 49.0


def test_occlusion_flank_rejects_candidates_outside_boundary_and_falls_back():
    result = occlusion_flank_waypoint(
        torch.tensor([[[-10.0, 0.0], [-20.0, 15.0], [-20.0, -15.0]]]),
        torch.tensor([[10.0, 0.0]]),
        torch.tensor([[[0.0, 0.0]]]),
        torch.tensor([[2.0]]),
        torch.tensor([0]),
        pursuer_radius_m=0.35,
        minimum_path_clearance_m=1.5,
        waypoint_margin_m=0.5,
        los_margin_m=0.25,
        blocker_margin_m=0.25,
        arena_half_extent_m=4.0,
        boundary_clearance_m=1.0,
        angular_samples=32,
    )
    assert not result.valid.item()
    assert torch.equal(result.waypoint_xy, torch.tensor([[10.0, 0.0]]))
    assert result.path_clearance_m.item() == -1.0


def test_occlusion_flank_has_no_hidden_target_truth_interface():
    parameters = inspect.signature(occlusion_flank_waypoint).parameters
    assert "last_seen_target_xy" in parameters
    assert "target_position_xy" not in parameters
    assert "target_velocity_xy" not in parameters


def test_flank_loss_guidance_locks_waypoint_and_preserves_trackers():
    pursuer = torch.tensor([[[-10.0, 0.0], [-20.0, 15.0], [-20.0, -15.0]]])
    nominal = torch.tensor([[[0.2, 0.0], [0.3, 0.2], [-0.4, 0.1]]])
    common = dict(
        enabled=True,
        start_lost_steps=1,
        slot_radius_m=8.0,
        scout_blend=0.85,
        tracker_blend=0.0,
        blend_growth=0.0,
        scout_speed_ratio=1.0,
        tracker_speed_ratio=0.65,
        slot_arrival_m=1.5,
        maximum_speed_mps=4.0,
        lost_cap_steps=50,
        use_occlusion_flank=True,
        obstacle_radius_m=torch.tensor([[2.0, 2.0]]),
        pursuer_radius_m=0.35,
        flank_minimum_path_clearance_m=1.5,
        flank_waypoint_margin_m=0.5,
        flank_los_margin_m=0.25,
        flank_blocker_margin_m=0.25,
        arena_half_extent_m=50.0,
        flank_boundary_clearance_m=1.0,
        flank_angular_samples=32,
    )
    first = loss_reacquisition_guidance(
        nominal,
        pursuer,
        torch.tensor([[10.0, 0.0]]),
        torch.tensor([False]),
        torch.tensor([1]),
        torch.tensor([False]),
        torch.tensor([-1]),
        torch.zeros(1, 3),
        obstacle_xy=torch.tensor([[[0.0, 0.0], [20.0, 20.0]]]),
        **common,
    )
    assert first.flank_guidance_active.item()
    assert first.flank_waypoint_valid.item()
    assert first.scout_index.item() == 0
    assert torch.equal(first.velocity_xy[0, 1:], nominal[0, 1:])
    assert not torch.equal(first.velocity_xy[0, 0], nominal[0, 0])
    assert first.velocity_xy.norm(dim=-1).max().item() <= 4.0 + 1.0e-6

    second = loss_reacquisition_guidance(
        nominal,
        pursuer,
        torch.tensor([[10.0, 0.0]]),
        torch.tensor([False]),
        torch.tensor([2]),
        first.active,
        first.scout_index,
        first.slot_angle_rad,
        obstacle_xy=torch.tensor([[[2.0, 0.0], [20.0, 20.0]]]),
        previous_flank_waypoint_xy=first.flank_waypoint_xy,
        previous_flank_valid=first.flank_waypoint_valid,
        previous_flank_blocker_index=first.flank_blocker_index,
        **common,
    )
    assert not second.activated.item()
    assert torch.equal(second.flank_waypoint_xy, first.flank_waypoint_xy)
    assert torch.equal(second.flank_blocker_index, first.flank_blocker_index)
    assert torch.equal(second.velocity_xy[0, 1:], nominal[0, 1:])


def test_flank_loss_guidance_uses_last_fallback_when_no_blocker_exists():
    pursuer = torch.tensor([[[-10.0, 0.0], [-20.0, 15.0], [-20.0, -15.0]]])
    result = loss_reacquisition_guidance(
        torch.zeros_like(pursuer),
        pursuer,
        torch.tensor([[10.0, 0.0]]),
        torch.tensor([False]),
        torch.tensor([1]),
        torch.tensor([False]),
        torch.tensor([-1]),
        torch.zeros(1, 3),
        enabled=True,
        start_lost_steps=1,
        slot_radius_m=8.0,
        scout_blend=0.85,
        tracker_blend=0.0,
        blend_growth=0.0,
        scout_speed_ratio=1.0,
        tracker_speed_ratio=0.65,
        slot_arrival_m=1.5,
        maximum_speed_mps=4.0,
        lost_cap_steps=50,
        use_occlusion_flank=True,
        obstacle_xy=torch.tensor([[[0.0, 20.0]]]),
        obstacle_radius_m=torch.tensor([[2.0]]),
    )
    assert not result.flank_waypoint_valid.item()
    assert result.flank_fallback_active.item()
    assert torch.equal(result.goal_xy[0, 0], torch.tensor([10.0, 0.0]))


def test_disabled_flank_guidance_is_bitwise_action_equivalent():
    pursuer = torch.tensor([[[-10.0, 0.0], [-20.0, 15.0], [-20.0, -15.0]]])
    nominal = torch.tensor([[[0.2, 0.0], [0.3, 0.2], [-0.4, 0.1]]])
    result = loss_reacquisition_guidance(
        nominal,
        pursuer,
        torch.tensor([[10.0, 0.0]]),
        torch.tensor([False]),
        torch.tensor([1]),
        torch.tensor([False]),
        torch.tensor([-1]),
        torch.zeros(1, 3),
        enabled=False,
        start_lost_steps=1,
        slot_radius_m=8.0,
        scout_blend=0.85,
        tracker_blend=0.0,
        blend_growth=0.0,
        scout_speed_ratio=1.0,
        tracker_speed_ratio=0.65,
        slot_arrival_m=1.5,
        maximum_speed_mps=4.0,
        lost_cap_steps=50,
        use_occlusion_flank=True,
        obstacle_xy=torch.tensor([[[0.0, 0.0]]]),
        obstacle_radius_m=torch.tensor([[2.0]]),
    )
    assert result.active.item()
    assert result.flank_waypoint_valid.item()
    assert not result.flank_guidance_active.item()
    assert torch.equal(result.velocity_xy, nominal)


def _sector_waypoint_call(*, lost_steps=1, velocity=None, obstacles=None, radii=None):
    return reachable_sector_waypoint(
        torch.tensor([[[-6.0, 0.0], [-10.0, 8.0], [-10.0, -8.0]]]),
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[4.0, 0.0]]) if velocity is None else velocity,
        torch.empty(1, 0, 2) if obstacles is None else obstacles,
        torch.empty(1, 0) if radii is None else radii,
        torch.tensor([0]),
        torch.tensor([lost_steps]),
        policy_dt_s=0.1,
        target_speed_bound_mps=4.0,
        velocity_heading_threshold_mps=0.25,
        lookahead_s=0.4,
        minimum_radius_m=1.5,
        maximum_radius_m=12.0,
        initial_half_angle_rad=math.pi / 6.0,
        half_angle_growth_rps=math.pi / 3.0,
        phase_hold_steps=4,
        pursuer_radius_m=0.5,
        minimum_path_clearance_m=1.5,
        arena_half_extent_m=50.0,
        boundary_clearance_m=1.0,
        detour_margin_m=0.5,
        detour_maximum_extra_m=4.5,
        detour_radial_samples=5,
        detour_angular_samples=32,
    )


def test_reachable_sector_expands_symmetrically_and_holds_phases():
    centre = _sector_waypoint_call(lost_steps=1)
    centre_last_hold = _sector_waypoint_call(lost_steps=4)
    positive_half = _sector_waypoint_call(lost_steps=5)
    negative_half = _sector_waypoint_call(lost_steps=9)
    assert centre.valid.item()
    assert centre.phase_index.item() == 0
    assert centre_last_hold.phase_index.item() == 0
    assert positive_half.phase_index.item() == 1
    assert negative_half.phase_index.item() == 2
    assert positive_half.waypoint_xy[0, 1].item() > 0.0
    assert negative_half.waypoint_xy[0, 1].item() < 0.0
    assert positive_half.radius_m.item() > centre.radius_m.item()
    assert positive_half.half_angle_rad.item() > centre.half_angle_rad.item()


def test_reachable_sector_uses_scout_last_direction_when_velocity_missing():
    result = _sector_waypoint_call(
        lost_steps=1, velocity=torch.zeros(1, 2)
    )
    assert result.valid.item()
    assert result.waypoint_xy[0, 0].item() > 0.0
    assert abs(result.waypoint_xy[0, 1].item()) < 1.0e-6


def test_reachable_sector_selects_safe_alternate_and_reports_clearance():
    result = _sector_waypoint_call(
        lost_steps=21,
        obstacles=torch.tensor([[[1.0, 0.0]]]),
        radii=torch.tensor([[0.5]]),
    )
    assert result.valid.item()
    assert result.route_complete.item()
    assert result.path_clearance_m.item() >= -1.0e-5


def test_reachable_sector_uses_two_leg_detour_when_all_direct_phases_blocked():
    result = _sector_waypoint_call(
        lost_steps=1,
        obstacles=torch.tensor([[[-2.0, 0.0]]]),
        radii=torch.tensor([[0.5]]),
    )
    assert result.valid.item()
    assert result.any_path_candidate.item()
    assert abs(result.waypoint_xy[0, 1].item()) > 1.0
    assert result.path_clearance_m.item() >= -1.0e-5


def test_reachable_sector_uses_safe_frontier_when_no_complete_route_exists():
    result = _sector_waypoint_call(
        lost_steps=1,
        obstacles=torch.tensor(
            [[[3.1723671, 0.6106052], [1.9079409, 5.8218908]]]
        ),
        radii=torch.tensor([[2.0675466, 0.6525826]]),
    )
    assert result.valid.item()
    assert not result.route_complete.item()
    assert result.any_path_candidate.item()
    assert result.path_clearance_m.item() >= -1.0e-5


def test_reachable_sector_allows_only_restorative_departure_from_soft_buffer():
    common = dict(
        pursuer_xy=torch.tensor([[[-3.0, 0.0], [-10.0, 8.0], [-10.0, -8.0]]]),
        last_seen_target_xy=torch.tensor([[-6.0, 0.0]]),
        last_seen_target_velocity_xy=torch.tensor([[-4.0, 0.0]]),
        obstacle_xy=torch.tensor([[[0.0, 0.0]]]),
        obstacle_radius_m=torch.tensor([[1.0]]),
        scout_index=torch.tensor([0]),
        lost_steps=torch.tensor([1]),
        policy_dt_s=0.1,
        target_speed_bound_mps=4.0,
        velocity_heading_threshold_mps=0.25,
        lookahead_s=0.4,
        minimum_radius_m=1.5,
        maximum_radius_m=12.0,
        initial_half_angle_rad=math.pi / 6.0,
        half_angle_growth_rps=math.pi / 3.0,
        phase_hold_steps=4,
        pursuer_radius_m=0.5,
        minimum_path_clearance_m=1.5,
        arena_half_extent_m=50.0,
        boundary_clearance_m=1.0,
        detour_margin_m=0.5,
        detour_maximum_extra_m=4.5,
        detour_radial_samples=5,
        detour_angular_samples=32,
    )
    result = reachable_sector_waypoint(**common)
    assert result.valid.item()
    assert result.waypoint_xy[0, 0].item() < -3.0
    assert result.path_clearance_m.item() > 0.0


def test_reachable_sector_falls_back_when_boundary_rejects_all_candidates():
    result = reachable_sector_waypoint(
        torch.tensor([[[-0.5, 0.0], [-0.5, 0.5], [-0.5, -0.5]]]),
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[4.0, 0.0]]),
        torch.empty(1, 0, 2),
        torch.empty(1, 0),
        torch.tensor([0]),
        torch.tensor([1]),
        policy_dt_s=0.1,
        target_speed_bound_mps=4.0,
        velocity_heading_threshold_mps=0.25,
        lookahead_s=0.4,
        minimum_radius_m=1.5,
        maximum_radius_m=12.0,
        initial_half_angle_rad=math.pi / 6.0,
        half_angle_growth_rps=math.pi / 3.0,
        phase_hold_steps=4,
        pursuer_radius_m=0.5,
        minimum_path_clearance_m=1.5,
        arena_half_extent_m=1.0,
        boundary_clearance_m=0.9,
        detour_margin_m=0.5,
        detour_maximum_extra_m=4.5,
        detour_radial_samples=5,
        detour_angular_samples=32,
    )
    assert not result.valid.item()
    assert result.phase_index.item() == -1
    assert torch.equal(result.waypoint_xy, torch.zeros(1, 2))


def test_reachable_sector_has_no_hidden_target_truth_interface():
    parameters = inspect.signature(reachable_sector_waypoint).parameters
    assert "last_seen_target_xy" in parameters
    assert "last_seen_target_velocity_xy" in parameters
    assert "target_position_xy" not in parameters
    assert "current_target_velocity_xy" not in parameters


def test_sector_loss_guidance_changes_only_locked_scout():
    pursuer = torch.tensor([[[-6.0, 0.0], [-10.0, 8.0], [-10.0, -8.0]]])
    nominal = torch.tensor([[[0.2, 0.0], [0.3, 0.2], [-0.4, 0.1]]])
    result = loss_reacquisition_guidance(
        nominal,
        pursuer,
        torch.zeros(1, 2),
        torch.tensor([False]),
        torch.tensor([1]),
        torch.tensor([False]),
        torch.tensor([-1]),
        torch.zeros(1, 3),
        enabled=True,
        start_lost_steps=1,
        slot_radius_m=8.0,
        scout_blend=0.85,
        tracker_blend=0.0,
        blend_growth=0.0,
        scout_speed_ratio=1.0,
        tracker_speed_ratio=0.65,
        slot_arrival_m=1.5,
        maximum_speed_mps=4.0,
        lost_cap_steps=50,
        use_reachable_sector=True,
        last_seen_target_velocity_xy=torch.tensor([[4.0, 0.0]]),
        obstacle_xy=torch.empty(1, 0, 2),
        obstacle_radius_m=torch.empty(1, 0),
    )
    assert result.sector_guidance_active.item()
    assert result.scout_index.item() == 0
    assert torch.equal(result.velocity_xy[0, 1:], nominal[0, 1:])
    assert not torch.equal(result.velocity_xy[0, 0], nominal[0, 0])
    assert result.velocity_xy.norm(dim=-1).max().item() <= 4.0 + 1.0e-6


def test_p5_l2_runtime_contract_is_evaluation_only_and_audited():
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    environment_source = (
        PACKAGE_SOURCE
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    base_environment_source = (
        PACKAGE_SOURCE
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_env.py"
    ).read_text(encoding="utf-8")
    assert 'choices=("off", "last", "flank", "sector")' in trainer_source
    assert 'args_cli.evaluation_loss_reacquisition == "flank"' in trainer_source
    assert 'args_cli.evaluation_loss_reacquisition == "sector"' in trainer_source
    assert "cfg.loss_reacquisition_tracker_blend = 0.0" in trainer_source
    assert "cfg.loss_reacquisition_blend_growth = 0.0" in trainer_source
    assert '"loss_reacquisition_flank_event_count"' in trainer_source
    assert "use_occlusion_flank=(" in environment_source
    assert "self._loss_reacquisition_flank_active_steps" in environment_source
    assert "use_reachable_sector=(" in environment_source
    assert "self._loss_reacquisition_sector_valid_steps" in environment_source
    assert "sector_scout_mask" in environment_source
    assert "self._last_seen_target_velocity_xy = torch.where(" in base_environment_source
    assert "self._visibility.team_visible[:, None]" in base_environment_source
    assert '"--evaluation_pairwise_recovery"' in trainer_source
    assert 'choices=("legacy", "restorative")' in trainer_source
    assert "cfg.predictive_agent_restore_unsafe_buffer = (" in trainer_source
    assert '"safety_projection_restorative_enabled"' in trainer_source


def test_native_velocity_adapter_clips_vector_norm():
    spec = P5TaskSpec()
    command = WorldVelocityYawAdapter(spec, maximum_speed_mps=4.0).step(
        torch.tensor([[[1.0, 1.0, -0.5], [0.5, 0.0, 1.0], [0.0, 0.0, 0.0]]])
    )
    assert tuple(command.desired_velocity_xy_w.shape) == (1, 3, 2)
    assert command.desired_speed_mps.max().item() <= 4.0 + 1.0e-6
    assert command.desired_yaw_rate_rps[0, 0].item() == -0.5 * spec.yaw_rate_max_rps


def test_native_observation_and_state_dimensions_are_physics_aware():
    spec = P5TaskSpec(team_target_sharing=True)
    tensors = _state_tensors()
    visible = torch.tensor([[True, False, False], [True, True, True]])
    last_seen = torch.zeros(2, 2)
    lost = torch.tensor([3, 0])
    obstacle_velocity = torch.zeros(2, 4, 2)
    observations = build_native_observations(
        spec, *tensors[:6], visible, last_seen, lost, tensors[6], tensors[7], obstacle_velocity
    )
    state = build_native_centralized_state(
        spec, *tensors[:6], lost, tensors[6], tensors[7]
    )
    assert observations.shape == (2, 3, NATIVE_OBSERVATION_DIM)
    assert state.shape == (2, NATIVE_STATE_DIM)
    assert torch.isfinite(observations).all()
    assert torch.isfinite(state).all()


def test_c27_role_phase_suffix_is_normalized_decentralized_and_appended():
    spec = P5TaskSpec(team_target_sharing=True)
    tensors = _state_tensors()
    pursuer_xy = tensors[0][..., :2]
    assigned = pursuer_xy + torch.tensor([[[10.0, 0.0], [0.0, 5.0], [-5.0, 0.0]]])
    role_phase = build_role_phase_features(
        spec,
        pursuer_xy,
        assigned,
        torch.tensor([[0, 1, 2], [2, 0, 1]]),
        torch.tensor([True, False]),
        torch.tensor([0, 2]),
        torch.tensor([9.5, 4.5]),
        torch.tensor([1.0 / 3.0, 1.0]),
        reference_radius_scale_m=9.5,
    )
    assert role_phase.shape == (2, 3, NATIVE_ROLE_PHASE_FEATURE_DIM)
    assert torch.allclose(role_phase[0, 0, :2], torch.tensor([0.1, 0.0]))
    assert torch.equal(role_phase[0, :, 2:5], torch.eye(3))
    assert torch.equal(role_phase[0, 0, 5:8], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.count_nonzero(role_phase[1, :, :5]) == 0

    visible = torch.ones(2, 3, dtype=torch.bool)
    lost = torch.zeros(2, dtype=torch.long)
    obstacle_velocity = torch.zeros(2, 4, 2)
    base = build_native_observations(
        spec, *tensors[:6], visible, torch.zeros(2, 2), lost,
        tensors[6], tensors[7], obstacle_velocity,
    )
    expanded = build_native_observations(
        spec, *tensors[:6], visible, torch.zeros(2, 2), lost,
        tensors[6], tensors[7], obstacle_velocity, role_phase,
    )
    assert expanded.shape == (2, 3, NATIVE_ROLE_PHASE_OBSERVATION_DIM)
    assert torch.equal(expanded[..., :NATIVE_OBSERVATION_DIM], base)
    assert torch.equal(expanded[..., NATIVE_OBSERVATION_DIM:], role_phase)


def test_c27_zero_padded_actor_is_action_equivalent_and_new_columns_learnable():
    torch.manual_seed(2701)
    parent = MAPPO(53, 58, 3, 3, hidden=32, device="cpu")
    expanded = MAPPO(63, 58, 3, 3, hidden=32, device="cpu")
    migrated = expand_mlp_actor_observation_state(
        parent.actor.state_dict(), 63
    )
    expanded.actor.load_state_dict(migrated)
    old_observation = torch.randn(7, 3, 53)
    added_features = torch.randn(7, 3, 10)
    old_mean, old_log_std = parent.actor(old_observation)
    new_mean, new_log_std = expanded.actor(
        torch.cat((old_observation, added_features), dim=-1)
    )
    assert torch.max(torch.abs(old_mean - new_mean)).item() <= 1.0e-6
    assert torch.equal(old_log_std, new_log_std)
    assert torch.count_nonzero(expanded.actor.body[0].weight[:, 53:]) == 0

    expanded.actor.zero_grad(set_to_none=True)
    new_mean.square().mean().backward()
    new_column_gradient = expanded.actor.body[0].weight.grad[:, 53:]
    assert torch.isfinite(new_column_gradient).all()
    assert torch.count_nonzero(new_column_gradient) > 0


def test_native_teammate_blocks_are_sweep_angle_ordered():
    spec = P5TaskSpec(team_target_sharing=True)
    tensors = list(_state_tensors(batch=1))
    visible = torch.ones(1, 3, dtype=torch.bool)
    last_seen = torch.zeros(1, 2)
    lost = torch.zeros(1, dtype=torch.long)
    obstacle_velocity = torch.zeros(1, 4, 2)
    original = build_native_observations(
        spec,
        *tensors[:6],
        visible,
        last_seen,
        lost,
        tensors[6],
        tensors[7],
        obstacle_velocity,
    )
    tensors[0][:, [1, 2]] = tensors[0][:, [2, 1]]
    tensors[2][:, [1, 2]] = tensors[2][:, [2, 1]]
    swapped = build_native_observations(
        spec,
        *tensors[:6],
        visible,
        last_seen,
        lost,
        tensors[6],
        tensors[7],
        obstacle_velocity,
    )
    assert torch.allclose(original[:, 0], swapped[:, 0])


def test_native_reward_prefers_a_balanced_capture_ring():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    balanced = 4.5 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)[None]
    poor = torch.tensor([[[15.0, 0.0], [14.0, 1.0], [14.0, -1.0]]])
    target = torch.zeros(1, 2)
    actions = torch.zeros(1, 3, 3)
    false = torch.zeros(1, dtype=torch.bool)
    balanced_terms = native_team_reward(
        torch.cat((balanced, torch.full((1, 3, 1), 3.0)), dim=-1),
        torch.cat((target, torch.full((1, 1), 3.0)), dim=-1),
        torch.zeros(1, 3, 3), actions, false, false, false, false, false, false,
        torch.ones(1, dtype=torch.bool), torch.full((1,), 10.0), torch.full((1,), 10.0),
    )
    poor_terms = native_team_reward(
        torch.cat((poor, torch.full((1, 3, 1), 3.0)), dim=-1),
        torch.cat((target, torch.full((1, 1), 3.0)), dim=-1),
        torch.zeros(1, 3, 3), actions, false, false, false, false, false, false,
        torch.ones(1, dtype=torch.bool), torch.full((1,), 10.0), torch.full((1,), 10.0),
    )
    assert balanced_terms.reward.item() > poor_terms.reward.item()
    assert balanced_terms.angular_error_rad.item() < 1.0e-5


def test_native_reward_penalizes_height_and_agent_proximity():
    target = torch.tensor([[0.0, 0.0, 3.0]])
    safe = torch.tensor(
        [[[4.5, 0.0, 3.0], [-2.25, 3.897, 3.0], [-2.25, -3.897, 3.0]]]
    )
    unsafe = safe.clone()
    unsafe[0, 1, :2] = unsafe[0, 0, :2] + torch.tensor([1.2, 0.0])
    unsafe[..., 2] = 4.0
    false = torch.zeros(1, dtype=torch.bool)
    common = dict(
        target_position=target,
        pursuer_velocity=torch.zeros(1, 3, 3),
        normalized_action=torch.zeros(1, 3, 3),
        success=false,
        contact=false,
        agent_collision=false,
        obstacle_collision=false,
        boundary_invalid=false,
        height_invalid=false,
        team_visible=torch.ones(1, dtype=torch.bool),
        minimum_static_clearance_m=torch.full((1,), 10.0),
        minimum_boundary_clearance_m=torch.full((1,), 10.0),
    )
    safe_terms = native_team_reward(pursuer_position=safe, **common)
    unsafe_terms = native_team_reward(pursuer_position=unsafe, **common)
    assert safe_terms.height_error_m.item() == 0.0
    assert unsafe_terms.height_error_m.item() == 1.0
    assert unsafe_terms.agent_proximity_penalty.item() > 0.0
    assert safe_terms.reward.item() > unsafe_terms.reward.item()


def test_native_reward_prefers_approach_progress_over_stagnation_and_retreat():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    unit_ring = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)[None]
    target = torch.tensor([[0.0, 0.0, 3.0]])
    false = torch.zeros(1, dtype=torch.bool)
    common = dict(
        target_position=target,
        pursuer_velocity=torch.zeros(1, 3, 3),
        normalized_action=torch.zeros(1, 3, 3),
        success=false,
        contact=false,
        agent_collision=false,
        obstacle_collision=false,
        boundary_invalid=false,
        height_invalid=false,
        team_visible=torch.ones(1, dtype=torch.bool),
        minimum_static_clearance_m=torch.full((1,), 10.0),
        minimum_boundary_clearance_m=torch.full((1,), 10.0),
        previous_team_distance_m=torch.tensor([15.0]),
        desired_radius_m=9.0,
        near_switch_m=15.0,
    )
    def reward_at(radius: float):
        xy = radius * unit_ring
        position = torch.cat((xy, torch.full((1, 3, 1), 3.0)), dim=-1)
        return native_team_reward(pursuer_position=position, **common)

    approaching = reward_at(14.6)
    stagnant = reward_at(15.0)
    retreating = reward_at(15.4)
    assert approaching.distance_progress_normalized.item() > 0.0
    assert retreating.distance_progress_normalized.item() < 0.0
    assert approaching.reward.item() > stagnant.reward.item() > retreating.reward.item()


def test_native_reward_uses_signed_angular_progress_for_moving_slots():
    position = torch.tensor(
        [[[6.0, 0.0, 3.0], [5.0, 1.0, 3.0], [5.0, -1.0, 3.0]]]
    )
    target = torch.tensor([[0.0, 0.0, 3.0]])
    false = torch.zeros(1, dtype=torch.bool)
    current_error = native_team_reward(
        pursuer_position=position,
        target_position=target,
        pursuer_velocity=torch.zeros(1, 3, 3),
        normalized_action=torch.zeros(1, 3, 3),
        success=false,
        contact=false,
        agent_collision=false,
        obstacle_collision=false,
        boundary_invalid=false,
        height_invalid=false,
        team_visible=torch.ones(1, dtype=torch.bool),
        minimum_static_clearance_m=torch.full((1,), 10.0),
        minimum_boundary_clearance_m=torch.full((1,), 10.0),
        angular_error_weight=0.0,
    ).angular_error_rad
    common = dict(
        pursuer_position=position,
        target_position=target,
        pursuer_velocity=torch.zeros(1, 3, 3),
        normalized_action=torch.zeros(1, 3, 3),
        success=false,
        contact=false,
        agent_collision=false,
        obstacle_collision=false,
        boundary_invalid=false,
        height_invalid=false,
        team_visible=torch.ones(1, dtype=torch.bool),
        minimum_static_clearance_m=torch.full((1,), 10.0),
        minimum_boundary_clearance_m=torch.full((1,), 10.0),
        previous_team_distance_m=position[..., :2].norm(dim=-1).mean(dim=1),
        angular_error_weight=0.0,
        angular_progress_weight=1.0,
    )
    improving = native_team_reward(
        previous_angular_error_rad=current_error + 0.5, **common
    )
    worsening = native_team_reward(
        previous_angular_error_rad=current_error - 0.5, **common
    )
    assert improving.angular_progress_normalized.item() > 0.0
    assert worsening.angular_progress_normalized.item() < 0.0
    assert improving.reward.item() > worsening.reward.item()


def test_native_reward_terminal_bonus_and_smooth_barrier_are_well_scaled():
    barrier = smooth_clearance_barrier(torch.tensor([10.0, 1.5, 0.0]), 1.5, 0.25)
    assert torch.isfinite(barrier).all()
    assert barrier[0].item() < 1.0e-6
    assert 0.0 < barrier[1].item() < barrier[2].item()
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    xy = 4.5 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)[None]
    position = torch.cat((xy, torch.full((1, 3, 1), 3.0)), dim=-1)
    target = torch.tensor([[0.0, 0.0, 3.0]])
    false = torch.zeros(1, dtype=torch.bool)
    common = dict(
        pursuer_position=position,
        target_position=target,
        pursuer_velocity=torch.zeros(1, 3, 3),
        normalized_action=torch.zeros(1, 3, 3),
        contact=false,
        agent_collision=false,
        obstacle_collision=false,
        boundary_invalid=false,
        height_invalid=false,
        team_visible=torch.ones(1, dtype=torch.bool),
        minimum_static_clearance_m=torch.full((1,), 10.0),
        minimum_boundary_clearance_m=torch.full((1,), 10.0),
        previous_team_distance_m=torch.tensor([4.5]),
    )
    ordinary = native_team_reward(success=false, **common)
    terminal = native_team_reward(success=torch.ones(1, dtype=torch.bool), **common)
    assert math.isclose(
        terminal.reward.item() - ordinary.reward.item(), 5.0, rel_tol=0.0, abs_tol=1e-5
    )


def test_native_curriculum_freezes_final_evaluation_while_shrinking_training_radius():
    stages = [native_curriculum_stage(name) for name in ("a0", "a1", "a2", "b")]
    assert [stage.training_capture_radius_m for stage in stages] == [12.0, 9.0, 6.0, 6.0]
    assert [stage.desired_radius_m for stage in stages] == [9.0, 6.5, 4.5, 4.5]
    assert not stages[0].static_obstacles
    assert stages[-1].static_obstacles


def test_c3_keeps_final_radius_and_enables_static_obstacle_curriculum():
    stage = native_curriculum_stage("c3")
    assert stage.training_capture_radius_m == 6.0
    assert stage.desired_radius_m == 4.5
    assert stage.static_obstacles
    assert stage.target_success_rate == 0.50


def test_c4_keeps_static_obstacles_for_joint_safety_projection():
    stage = native_curriculum_stage("c4")
    assert stage.training_capture_radius_m == 6.0
    assert stage.static_obstacles
    assert stage.target_success_rate == 0.50


def test_c6_is_moving_target_random_map_curriculum_stage():
    stage = native_curriculum_stage("c6")
    assert stage.training_capture_radius_m == 6.0
    assert stage.static_obstacles
    assert stage.target_success_rate == 0.30


def test_c7_keeps_c6_task_difficulty_for_all_phase_safety_repair():
    c6 = native_curriculum_stage("c6")
    c7 = native_curriculum_stage("c7")
    assert c7.training_capture_radius_m == c6.training_capture_radius_m
    assert c7.desired_radius_m == c6.desired_radius_m
    assert c7.static_obstacles
    assert c7.target_success_rate == c6.target_success_rate


def test_c8_is_the_slow_evader_curriculum_rollback_stage():
    c8 = native_curriculum_stage("c8")
    assert c8.training_capture_radius_m == 6.0
    assert c8.static_obstacles
    assert c8.target_success_rate == 0.20


def test_c9_keeps_c8_task_and_adds_slot_handoff_method_stage():
    c8 = native_curriculum_stage("c8")
    c9 = native_curriculum_stage("c9")
    assert c9.training_capture_radius_m == c8.training_capture_radius_m
    assert c9.desired_radius_m == c8.desired_radius_m
    assert c9.static_obstacles
    assert c9.target_success_rate == c8.target_success_rate


def test_c10_c12_preserve_c9_task_for_finite_visibility_sweep():
    c9 = native_curriculum_stage("c9")
    expected_thresholds = {"c10": 0.70, "c11": 0.60, "c12": 0.50}
    for name, threshold in expected_thresholds.items():
        stage = native_curriculum_stage(name)
        assert stage.training_capture_radius_m == c9.training_capture_radius_m
        assert stage.desired_radius_m == c9.desired_radius_m
        assert stage.near_switch_m == c9.near_switch_m
        assert stage.static_obstacles == c9.static_obstacles
        assert stage.target_success_rate == threshold


def test_finite_visibility_stages_freeze_tracked_reset_reference_range():
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    train_cfg_source = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_train_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "tracked_reset_reference_range_m = 100.0" in train_cfg_source
    assert '"c10": 50.0' in trainer_source
    assert '"c11": 30.0' in trainer_source
    assert '"c12": 25.0' in trainer_source
    assert '"sensor_ablation_changes_reset_geometry": False' in trainer_source


def test_strict_5m_evaluation_has_an_independent_terminal_override():
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert '"--evaluation_capture_radius_m"' in trainer_source
    assert "cfg.training_capture_radius_m = args_cli.evaluation_capture_radius_m" in trainer_source
    assert '"strict_5m_requires_separate_replay": True' in trainer_source


def test_evaluation_sensor_override_does_not_change_reset_geometry():
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert '"--evaluation_sense_range_m"' in trainer_source
    assert "cfg.sense_range_m = args_cli.evaluation_sense_range_m" in trainer_source
    assert '"evaluation_sense_range_override_m":' in trainer_source
    assert '"sensor_ablation_changes_reset_geometry": False' in trainer_source


def test_c13_dense_map_keeps_checkpoint_contract_and_has_twelve_obstacles():
    c9 = native_curriculum_stage("c9")
    c13 = native_curriculum_stage("c13")
    assert c13.training_capture_radius_m == c9.training_capture_radius_m
    assert c13.desired_radius_m == c9.desired_radius_m
    assert c13.static_obstacles
    assert c13.target_success_rate == 0.70

    cfg_path = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_env_cfg.py"
    )
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"))
    dense_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "DENSE_STATIC_OBSTACLE_SPECS"
            for target in node.targets
        )
    )
    assert isinstance(dense_assignment.value, ast.BinOp)
    assert isinstance(dense_assignment.value.right, ast.Tuple)
    assert len(dense_assignment.value.right.elts) == 4

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage == "c13"' in trainer_source
    assert "cfg.obstacle_specs = DENSE_STATIC_OBSTACLE_SPECS" in trainer_source
    assert '"physical_obstacle_count": len(cfg.obstacle_specs)' in trainer_source


def test_c14_mixed_shapes_use_exact_physics_and_conservative_envelopes():
    c13 = native_curriculum_stage("c13")
    c14 = native_curriculum_stage("c14")
    assert c14.training_capture_radius_m == c13.training_capture_radius_m
    assert c14.desired_radius_m == c13.desired_radius_m
    assert c14.target_success_rate == 0.70

    cfg_path = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_env_cfg.py"
    )
    cfg_source = cfg_path.read_text(encoding="utf-8")
    tree = ast.parse(cfg_source)

    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id
        in {
            "MIXED_STATIC_OBSTACLE_SPECS",
            "MIXED_STATIC_OBSTACLE_BOX_SIZES_XY_M",
        }
    }
    specs = assignments["MIXED_STATIC_OBSTACLE_SPECS"]
    box_sizes = assignments["MIXED_STATIC_OBSTACLE_BOX_SIZES_XY_M"]
    assert len(specs) == len(box_sizes) == 12
    assert [row[2] for row in specs[:6]] == [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    for (_, _, envelope_radius), (size_x, size_y) in zip(specs[6:], box_sizes[6:]):
        assert envelope_radius >= 0.5 * math.hypot(size_x, size_y) - 1.0e-6
    assert '("cylinder",) * 6 + ("box",) * 6' in cfg_source

    env_source = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_env.py"
    ).read_text(encoding="utf-8")
    assert "sim_utils.CuboidCfg(" in env_source
    assert "circumscribed_radius = 0.5 * math.hypot" in env_source

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage in ("c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")' in trainer_source
    assert "cfg.obstacle_specs = MIXED_STATIC_OBSTACLE_SPECS" in trainer_source
    assert "cfg.obstacle_shape_kinds = MIXED_STATIC_OBSTACLE_SHAPES" in trainer_source
    assert '"physical_obstacle_kinds": list(cfg.obstacle_shape_kinds)' in trainer_source
    assert "conservative_circumscribed_circle_" in trainer_source


def test_c16_dynamic_pilot_keeps_c14_contract_and_uses_relative_velocity():
    c14 = native_curriculum_stage("c14")
    c16 = native_curriculum_stage("c16")
    assert c16.training_capture_radius_m == c14.training_capture_radius_m
    assert c16.desired_radius_m == c14.desired_radius_m
    assert c16.target_success_rate == 0.70

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage in ("c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")' in trainer_source
    assert 'if stage in ("c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")' in trainer_source
    assert "cfg.dynamic_obstacle_indices = (10, 11)" in trainer_source
    train_env_source = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert "if self.cfg.predictive_static_use_obstacle_velocity" in train_env_source
    assert "else torch.zeros_like(self._obstacle_velocity)" in train_env_source
    assert '"collision_dynamic": int(collided_dynamic)' in trainer_source
    assert '"collision_dynamic_rate": collision_dynamic / completed' in trainer_source

    env_source = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_env.py"
    ).read_text(encoding="utf-8")
    assert "def _advance_dynamic_obstacles" in env_source
    assert "write_root_pose_to_sim(root_pose, env_ids)" in env_source
    assert '"collision_dynamic": self._dynamic_obstacle_collision.clone()' in env_source
    assert 'telemetry["dynamic_obstacle_max_displacement_m"]' in env_source


def test_c17_difficulty_overrides_are_evaluation_only_and_audited():
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert '"--evaluation_evader_speed_mps"' in trainer_source
    assert 'choices=(1.0, 1.5, 2.0, 4.0)' in trainer_source
    assert '"--evaluation_dynamic_obstacle_speed_mps"' in trainer_source
    assert 'choices=(0.6, 1.0)' in trainer_source
    assert (
        '"--evaluation_evader_speed_mps is only valid with --evaluation_only"'
        in trainer_source
    )
    assert (
        '"--evaluation_dynamic_obstacle_speed_mps is only valid with '
        '--evaluation_only"'
    ) in trainer_source
    assert "cfg.evader_speed_mps = args_cli.evaluation_evader_speed_mps" in (
        trainer_source
    )
    assert '"difficulty_overrides": {' in trainer_source


def test_dynamic_obstacle_velocity_ablation_is_evaluation_only_and_audited():
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert '"--evaluation_ignore_dynamic_obstacle_velocity"' in trainer_source
    assert (
        '"--evaluation_ignore_dynamic_obstacle_velocity is only valid with "'
        in trainer_source
    )
    assert "cfg.predictive_static_use_obstacle_velocity = False" in trainer_source
    assert '"predictive_static_use_obstacle_velocity": (' in trainer_source
    assert '"ignore_dynamic_obstacle_velocity": (' in trainer_source

    cfg_source = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_train_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "predictive_static_use_obstacle_velocity = True" in cfg_source


def test_c17_training_stage_changes_only_evader_speed_from_c16():
    c16 = native_curriculum_stage("c16")
    c17 = native_curriculum_stage("c17")
    assert c17.training_capture_radius_m == c16.training_capture_radius_m
    assert c17.desired_radius_m == c16.desired_radius_m
    assert c17.near_switch_m == c16.near_switch_m
    assert c17.static_obstacles == c16.static_obstacles
    assert c17.target_success_rate == c16.target_success_rate

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage in ("c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")' in trainer_source
    assert 'if stage in ("c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")' in trainer_source
    assert 'if stage == "c17"' in trainer_source


def test_c19_training_stage_changes_only_slot_prediction_from_c16():
    c16 = native_curriculum_stage("c16")
    c19 = native_curriculum_stage("c19")
    assert c19.training_capture_radius_m == c16.training_capture_radius_m
    assert c19.desired_radius_m == c16.desired_radius_m
    assert c19.near_switch_m == c16.near_switch_m
    assert c19.static_obstacles == c16.static_obstacles
    assert c19.target_success_rate == c16.target_success_rate

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage in ("c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")' in trainer_source
    assert 'if stage in ("c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")' in trainer_source
    assert 'if stage == "c19"' in trainer_source
    assert "cfg.moving_target_slot_prediction_horizon_s = 0.5" in trainer_source


def test_c20_freezes_equal_speed_evader_and_strong_ring_contract():
    c20 = native_curriculum_stage("c20")
    assert c20.training_capture_radius_m == 6.0
    assert c20.desired_radius_m == 4.5
    assert c20.static_obstacles
    assert c20.target_success_rate == 0.10

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    for contract in (
        'if stage == "c20"',
        "cfg.evader_speed_mps = (",
        "cfg.evader_minimum_episode_speed_mps = 4.0",
        "cfg.evader_allow_stop_mode = False",
        "cfg.evader_gap_escape_weight = 0.80",
        "cfg.evader_maintain_episode_speed = True",
        "cfg.moving_target_slot_activation_radius_m = 30.0",
        "cfg.moving_target_slot_prediction_horizon_s = 1.5",
        "cfg.strong_ring_termination_enabled = True",
        "cfg.predictive_agent_minimum_center_distance_m = 3.5",
        "cfg.predictive_agent_prediction_horizon_s = 1.5",
        "cfg.predictive_agent_projection_iterations = 8",
        "cfg.predictive_target_minimum_center_distance_m = 1.5",
        "cfg.predictive_static_minimum_clearance_m = 1.5",
        "cfg.joint_safety_projection_sweeps = 4",
        '"evader_minimum_episode_speed_mps": (',
        '"strong_ring_required_hold_steps":',
        '"evader_episode_speed_limit_mps": (',
        '"terminal_pursuer_target_distances_m": json.dumps(',
        '"maximum_strong_ring_hold_steps": strong_ring_max_hold_steps_global',
            "C20--C30 equal-speed stages only permit the 4.0 m/s diagnostic",
        'else "HOLD_REPAIR_METHOD"',
    ):
        assert contract in trainer_source


def test_c21_repairs_reachability_without_relaxing_c20_contract():
    c20 = native_curriculum_stage("c20")
    c21 = native_curriculum_stage("c21")
    assert c21.training_capture_radius_m == c20.training_capture_radius_m
    assert c21.desired_radius_m == c20.desired_radius_m
    assert c21.near_switch_m == c20.near_switch_m
    assert c21.static_obstacles == c20.static_obstacles
    assert c21.target_success_rate == c20.target_success_rate

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    env_source = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    for contract in (
        'if stage in ("c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")',
        "cfg.moving_target_slot_activation_radius_m = 100.0",
        "cfg.reachable_intercept_guidance_enabled = True",
        "cfg.reachable_intercept_maximum_horizon_s = 8.0",
        "cfg.reachable_intercept_slot_horizon_cap_s = 1.0",
        "cfg.reachable_intercept_blocker_role_lock_weight_m = 50.0",
        "cfg.asynchronous_orientation_gate_enabled = True",
        "cfg.strong_ring_termination_enabled = True",
        "cfg.predictive_agent_minimum_center_distance_m = 3.5",
        "cfg.joint_safety_projection_sweeps = 4",
        '"terminal_intercept_reachable_fraction": (',
        '"terminal_orientation_staging_fraction": (',
    ):
        assert contract in trainer_source


def test_c22_applies_intercept_lead_only_to_blocker_slot():
    c21 = native_curriculum_stage("c21")
    c22 = native_curriculum_stage("c22")
    assert c22 == type(c22)("c22", 6.0, 4.5, 9.0, True, 0.10)
    assert c22.training_capture_radius_m == c21.training_capture_radius_m
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    env_source = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert 'cfg.reachable_intercept_blocker_only_slot_lead = stage in ("c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")' in trainer_source
    assert "blocker_only_prediction_horizon[:, 0]" in env_source


def test_reachable_role_intercept_maps_only_reachable_agents_to_locked_slots():
    horizon = reachable_role_slot_prediction_horizon(
        torch.tensor([[0.5, 0.75, 1.5], [0.25, 0.75, 1.25]]),
        torch.tensor([[True, False, True], [False, True, True]]),
        torch.tensor([[0, 2, 1], [-1, -1, -1]]),
        torch.tensor([0, 1]),
        maximum_horizon_s=1.0,
    )
    assert torch.allclose(horizon[0], torch.tensor([0.5, 1.0, 0.0]))
    assert torch.allclose(horizon[1], torch.tensor([0.75, 0.0, 0.0]))


def test_per_agent_slot_intercept_preserves_agent_goal_correspondence():
    result = bounded_per_agent_constant_velocity_intercept(
        torch.tensor([[[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]]]),
        torch.tensor([[[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]]]),
        torch.tensor([[4.0, 0.0]]),
        pursuer_speed_mps=4.0,
        minimum_horizon_s=0.25,
        maximum_horizon_s=8.0,
        horizon_samples=33,
    )
    assert result.time_s.shape == (1, 3)
    assert result.point_xy.shape == (1, 3, 2)
    assert result.reachable.shape == (1, 3)
    assert result.reachability_gap_m.shape == (1, 3)
    assert result.point_xy[0, 0, 1].item() == 0.0
    assert result.point_xy[0, 1, 1].item() == 1.0
    assert result.point_xy[0, 2, 1].item() == -1.0


def test_c23_adds_predictive_boundary_projection_only():
    c22 = native_curriculum_stage("c22")
    c23 = native_curriculum_stage("c23")
    assert c23.training_capture_radius_m == c22.training_capture_radius_m
    assert c23.desired_radius_m == c22.desired_radius_m
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    env_source = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    assert 'cfg.predictive_boundary_projection_enabled = stage in ("c23", "c24", "c25", "c26", "c27", "c29", "c30")' in trainer_source
    assert '"predictive_boundary_projection_enabled": (' in trainer_source
    assert "self._boundary_projection_corrected |= projection.corrected" in env_source
    assert "self._boundary_projection_constraint_corrections += (" in env_source
    assert "self._boundary_projection_corrected.zero_()" in env_source
    for telemetry_field in (
        '"boundary_projection_call_steps": (',
        '"boundary_projection_corrected_steps": (',
        '"boundary_projection_constraint_corrections": (',
        '"boundary_projection_infeasible_steps": (',
        '"boundary_projection_min_predicted_clearance_m": (',
        '"boundary_projection_corrected_step_rate": (',
    ):
        assert telemetry_field in trainer_source


def test_c24_repairs_staging_in_the_moving_target_frame():
    c23 = native_curriculum_stage("c23")
    c24 = native_curriculum_stage("c24")
    assert c24.training_capture_radius_m == c23.training_capture_radius_m
    assert c24.desired_radius_m == c23.desired_radius_m
    assert c24.target_success_rate == c23.target_success_rate
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    env_source = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    for contract in (
        'cfg.orientation_target_relative_staging_enabled = stage in ("c24", "c25", "c26", "c27", "c29", "c30")',
        'cfg.asynchronous_require_current_team_ready = stage in ("c24", "c25", "c26", "c27", "c29", "c30")',
        'cfg.asynchronous_current_readiness_reward_enabled = stage in ("c24", "c25", "c26", "c27", "c29", "c30")',
        'cfg.predictive_boundary_projection_enabled = stage in ("c23", "c24", "c25", "c26", "c27", "c29", "c30")',
    ):
        assert contract in trainer_source


def test_c25_is_the_strict_equal_speed_thirty_second_curriculum_entry():
    c24 = native_curriculum_stage("c24")
    c25 = native_curriculum_stage("c25")
    assert c25.training_capture_radius_m == c24.training_capture_radius_m
    assert c25.desired_radius_m == c24.desired_radius_m
    assert c25.near_switch_m == c24.near_switch_m
    assert c25.static_obstacles == c24.static_obstacles
    assert c25.target_success_rate == c24.target_success_rate

    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    env_source = (
        ROOT
        / "isaac_p5"
        / "source"
        / "bylw_isaac_p5"
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
        / "bylw_pursuit_train_env.py"
    ).read_text(encoding="utf-8")
    for contract in (
        'if args_cli.stage in ("c25", "c26", "c27"):',
        "(30.0, 40.0) if args_cli.evaluation_only else (30.0,)",
        '"C25/C26/C27 training freezes 30 s; evaluation-only permits "',
        'if stage in ("c25", "c26", "c27", "c29", "c30")',
        'cfg.orientation_target_relative_staging_enabled = stage in ("c24", "c25", "c26", "c27", "c29", "c30")',
        'cfg.asynchronous_require_current_team_ready = stage in ("c24", "c25", "c26", "c27", "c29", "c30")',
        'cfg.asynchronous_current_readiness_reward_enabled = stage in ("c24", "c25", "c26", "c27", "c29", "c30")',
    ):
        assert contract in trainer_source
    assert "if self.cfg.orientation_target_relative_staging_enabled" in env_source
    assert "if self.cfg.asynchronous_current_readiness_reward_enabled" in env_source
    assert "self.cfg.asynchronous_require_current_team_ready" in env_source


def test_c26_adds_only_assigned_slot_progress_reward_to_c25_contract():
    c25 = native_curriculum_stage("c25")
    c26 = native_curriculum_stage("c26")
    assert c26 == c25.__class__("c26", 6.0, 4.5, 9.0, True, 0.10)
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert 'if stage in ("c26", "c27", "c29", "c30"):' in trainer_source
    assert "cfg.moving_target_slot_progress_reward_weight = 0.50" in trainer_source


def test_c27_inherits_c26_contract_and_enables_only_role_phase_protocol():
    c26 = native_curriculum_stage("c26")
    c27 = native_curriculum_stage("c27")
    assert c27 == c26.__class__("c27", 6.0, 4.5, 9.0, True, 0.10)
    trainer_source = (ROOT / "scripts/train_p5_2_native_mappo.py").read_text()
    assert 'if stage in ("c27", "c29", "c30"):' in trainer_source
    assert "cfg.role_phase_observation_enabled = True" in trainer_source
    assert (
        "cfg.moving_target_slot_prepare_assignment_for_observation = True"
        in trainer_source
    )
    assert "cfg.predictive_agent_restore_unsafe_buffer = True" in trainer_source
    assert '"p5_2_native_63_role_goal_phase_v1"' in trainer_source


def test_slot_goal_progress_reward_tracks_executed_motion_and_masks_inactive():
    assigned = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        ]
    )
    post = torch.tensor(
        [
            [[0.4, 0.0], [0.4, 0.0], [0.4, 0.0]],
            [[0.4, 0.0], [0.4, 0.0], [0.4, 0.0]],
        ]
    )
    result = slot_goal_progress_reward(
        torch.tensor([1.0, 1.0]),
        post,
        assigned,
        torch.tensor([True, False]),
        command_speed_mps=4.0,
        policy_dt_s=0.1,
        reward_weight=0.5,
    )
    assert torch.allclose(result.normalized_progress, torch.tensor([1.0, 0.0]))
    assert torch.allclose(result.reward, torch.tensor([0.5, 0.0]))
    assert torch.allclose(result.post_action_mean_error_m, torch.tensor([0.6, 0.0]))

    away = slot_goal_progress_reward(
        torch.tensor([0.2]),
        torch.tensor([[[-0.2, 0.0], [-0.2, 0.0], [-0.2, 0.0]]]),
        assigned[:1],
        torch.tensor([True]),
        command_speed_mps=4.0,
        policy_dt_s=0.1,
        reward_weight=0.5,
    )
    assert away.normalized_progress.item() == -1.0
    assert away.reward.item() == -0.5


def test_c27_slot_progress_uses_slowest_role_error_instead_of_team_mean():
    assigned = torch.zeros(1, 3, 2)
    post = torch.tensor([[[0.1, 0.0], [0.1, 0.0], [0.8, 0.0]]])
    result = slot_goal_progress_reward(
        torch.tensor([1.0]),
        post,
        assigned,
        torch.tensor([True]),
        command_speed_mps=4.0,
        policy_dt_s=0.1,
        reward_weight=0.5,
        use_maximum_error=True,
    )
    assert torch.allclose(result.normalized_progress, torch.tensor([0.5]))
    assert torch.allclose(result.reward, torch.tensor([0.25]))
    assert torch.allclose(result.post_action_mean_error_m, torch.tensor([0.8]))


def test_slot_role_lock_allows_unknown_blocker_until_heading_is_available():
    pursuer = torch.tensor([[[-4.0, 0.0], [0.0, 4.0], [0.0, -4.0]]])
    result = moving_target_slot_guidance(
        torch.zeros_like(pursuer),
        pursuer,
        torch.zeros(1, 2),
        torch.tensor([[4.0, 0.0]]),
        torch.tensor([4.5]),
        torch.tensor([True]),
        torch.full((1, 3), -1, dtype=torch.long),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.2,
        target_feedforward_scale=0.75,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.5,
        blocker_agent_index=torch.tensor([-1]),
        blocker_role_lock_weight_m=50.0,
    )
    assert sorted(result.assignment[0].tolist()) == [0, 1, 2]


def test_slot_guidance_accepts_blocker_only_prediction_horizon():
    pursuer = torch.tensor(
        [[[8.5, 0.0], [-2.25, 3.8971143], [-2.25, -3.8971143]]]
    )
    result = moving_target_slot_guidance(
        torch.zeros_like(pursuer),
        pursuer,
        torch.zeros(1, 2),
        torch.tensor([[4.0, 0.0]]),
        torch.tensor([4.5]),
        torch.tensor([True]),
        torch.tensor([[0, 1, 2]]),
        torch.zeros(1),
        torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.0,
        target_feedforward_scale=0.0,
        maximum_speed_mps=100.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.5,
        target_prediction_horizon_s=torch.tensor([[1.0, 0.0, 0.0]]),
        blocker_agent_index=torch.tensor([0]),
        blocker_role_lock_weight_m=50.0,
    )
    assert torch.allclose(result.velocity_xy, torch.zeros_like(pursuer), atol=1e-5)


def test_irregular_obstacle_layout_is_seeded_separated_and_not_two_rows():
    radii = torch.tensor([3.0, 3.0, 4.0, 4.0, 3.5, 3.5, 3.0, 3.0]).numpy()
    first = sample_irregular_obstacle_layout(123, radii)
    replay = sample_irregular_obstacle_layout(123, radii)
    different = sample_irregular_obstacle_layout(124, radii)
    assert first.shape == (8, 2)
    assert torch.equal(torch.from_numpy(first), torch.from_numpy(replay))
    assert not torch.equal(torch.from_numpy(first), torch.from_numpy(different))
    for left in range(len(radii)):
        assert float(torch.from_numpy(first[left]).norm()) > radii[left] + 5.0
        for right in range(left + 1, len(radii)):
            distance = float(torch.from_numpy(first[left] - first[right]).norm())
            assert distance > radii[left] + radii[right] + 2.0
    # A regular two-row map has only two unique y values; seeded layouts must
    # have broad, non-grid variation in both coordinates.
    assert len(set(first[:, 0].round(4))) >= 6
    assert len(set(first[:, 1].round(4))) >= 6


def test_target_facing_yaw_guidance_blends_policy_residual_and_clips():
    pursuer_xy = torch.tensor([[[0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]]])
    yaw = torch.tensor([[0.0, math.pi, 0.0]])
    target = torch.zeros(1, 2)
    policy_rate = torch.tensor([[0.4, -0.4, 0.4]])
    yaw_rate, yaw_error = target_facing_yaw_guidance(
        pursuer_xy,
        yaw,
        target,
        policy_rate,
        proportional_gain=1.5,
        policy_residual_scale=0.25,
        maximum_yaw_rate_rps=1.5,
    )
    assert math.isclose(yaw_error[0, 0].item(), math.pi / 2.0, abs_tol=1e-6)
    assert math.isclose(yaw_error[0, 1].item(), 0.0, abs_tol=1e-6)
    assert math.isclose(yaw_error[0, 2].item(), 0.0, abs_tol=1e-6)
    assert math.isclose(yaw_rate[0, 0].item(), 1.5, abs_tol=1e-6)
    assert math.isclose(yaw_rate[0, 1].item(), -0.1, abs_tol=1e-6)
    assert math.isclose(yaw_rate[0, 2].item(), 0.1, abs_tol=1e-6)


def test_positive_yaw_gain_multiplier_changes_magnitude_but_not_direction():
    pursuer_xy = torch.tensor([[[0.0, -1.0], [0.0, -1.0]]])
    yaw = torch.zeros(1, 2)
    target = torch.zeros(1, 2)
    yaw_rate, yaw_error = target_facing_yaw_guidance(
        pursuer_xy,
        yaw,
        target,
        torch.zeros_like(yaw),
        proportional_gain=0.5,
        policy_residual_scale=0.0,
        maximum_yaw_rate_rps=2.0,
        gain_multiplier=torch.tensor([[0.75, 1.25]]),
    )
    assert (yaw_rate > 0.0).all()
    assert yaw_rate[0, 1].item() > yaw_rate[0, 0].item()
    assert torch.allclose(yaw_error, torch.full_like(yaw, math.pi / 2.0))


def test_moving_target_slot_guidance_assigns_balanced_slots_and_feedforward():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    slots = 9.5 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    pursuer_xy = slots[torch.tensor([2, 0, 1])][None]
    result = moving_target_slot_guidance(
        torch.zeros(1, 3, 2),
        pursuer_xy,
        torch.zeros(1, 2),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([9.5]),
        torch.ones(1, dtype=torch.bool),
        torch.full((1, 3), -1, dtype=torch.long),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
        position_gain=0.7,
        policy_residual_scale=0.35,
        target_feedforward_scale=1.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.15,
        maximum_anchor_turn_rate_rps=0.8,
        policy_dt_s=0.1,
        assignment_hysteresis_m=1.0,
    )
    assert result.active.item()
    assert not result.assignment_switched.item()
    assert result.assignment.tolist() == [[2, 0, 1]]
    assert result.mean_slot_error_m.item() < 1.0e-5
    assert torch.allclose(
        result.velocity_xy,
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]),
        atol=1.0e-5,
    )


def test_cyclic_slot_assignment_preserves_order_and_locks_valid_topology():
    angles = torch.tensor([4.0 * math.pi / 3.0, 0.0, 2.0 * math.pi / 3.0])
    pursuer_xy = 9.5 * torch.stack(
        (torch.cos(angles), torch.sin(angles)), dim=-1
    )[None]
    common = dict(
        desired_velocity_xy=torch.zeros(1, 3, 2),
        estimated_target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.zeros(1, 2),
        reference_radius_m=torch.tensor([9.5]),
        active=torch.ones(1, dtype=torch.bool),
        previous_anchor_heading_rad=torch.zeros(1),
        previous_anchor_valid=torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.0,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.0,
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
    )
    seeded = moving_target_slot_guidance(
        pursuer_xy=pursuer_xy,
        previous_assignment=torch.full((1, 3), -1, dtype=torch.long),
        **common,
    )
    assert seeded.assignment.tolist() == [[2, 0, 1]]
    crossed_positions = pursuer_xy[:, torch.tensor([1, 2, 0]), :]
    locked = moving_target_slot_guidance(
        pursuer_xy=crossed_positions,
        previous_assignment=seeded.assignment,
        **common,
    )
    assert torch.equal(locked.assignment, seeded.assignment)
    assert not locked.assignment_switched.item()


def test_phase_separation_controller_balances_neighbour_gaps_without_radial_motion():
    angles = torch.tensor([0.0, 0.2, 4.0])
    pursuer_xy = 9.5 * torch.stack(
        (torch.cos(angles), torch.sin(angles)), dim=-1
    )[None]
    result = moving_target_slot_guidance(
        torch.zeros(1, 3, 2),
        pursuer_xy,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.tensor([9.5]),
        torch.ones(1, dtype=torch.bool),
        torch.tensor([[0, 1, 2]]),
        torch.zeros(1),
        torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.0,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.0,
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
        phase_separation_control_enabled=True,
        phase_gain=0.8,
        radial_gain=1.0,
        maximum_tangential_speed_mps=3.0,
    )
    radial_unit = pursuer_xy / pursuer_xy.norm(dim=-1, keepdim=True)
    tangent_unit = torch.stack(
        (-radial_unit[..., 1], radial_unit[..., 0]), dim=-1
    )
    radial_velocity = (result.velocity_xy * radial_unit).sum(dim=-1)
    tangent_velocity = (result.velocity_xy * tangent_unit).sum(dim=-1)
    assert torch.allclose(radial_velocity, torch.zeros_like(radial_velocity), atol=1e-5)
    assert tangent_velocity[0, 0].item() < -2.9
    assert tangent_velocity[0, 1].item() > 2.9
    assert tangent_velocity[0, 2].item() < -2.9
    assert result.velocity_xy.norm(dim=-1).max().item() <= 3.0001
    assert result.policy_residual_norm_mps.item() == 0.0


def test_radial_component_budget_prevents_large_error_from_dominating_command():
    angles = torch.tensor(
        [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
    )
    pursuer_xy = 20.0 * torch.stack(
        (torch.cos(angles), torch.sin(angles)), dim=-1
    )[None]
    result = moving_target_slot_guidance(
        torch.zeros_like(pursuer_xy),
        pursuer_xy,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.tensor([9.5]),
        torch.ones(1, dtype=torch.bool),
        torch.tensor([[0, 1, 2]]),
        torch.zeros(1),
        torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.0,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.0,
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
        phase_separation_control_enabled=True,
        phase_gain=0.8,
        radial_gain=1.0,
        maximum_radial_speed_mps=1.0,
        maximum_tangential_speed_mps=3.0,
    )
    radial_unit = pursuer_xy / pursuer_xy.norm(dim=-1, keepdim=True)
    radial_speed = (result.velocity_xy * radial_unit).sum(dim=-1)
    assert torch.allclose(radial_speed, -torch.ones_like(radial_speed), atol=1.0e-5)


def test_phase_scheduled_radial_budget_preserves_far_approach_authority():
    angles = torch.tensor(
        [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
    )
    pursuer_xy = 20.0 * torch.stack(
        (torch.cos(angles), torch.sin(angles)), dim=-1
    )[None]
    result = moving_target_slot_guidance(
        torch.zeros_like(pursuer_xy),
        pursuer_xy,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.tensor([9.5]),
        torch.ones(1, dtype=torch.bool),
        torch.tensor([[0, 1, 2]]),
        torch.zeros(1),
        torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.0,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.0,
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
        phase_separation_control_enabled=True,
        phase_gain=0.8,
        radial_gain=1.0,
        maximum_radial_speed_mps=1.0,
        phase_scheduled_radial_speed_budget=True,
        maximum_tangential_speed_mps=3.0,
        phase_activation_margin_m=6.0,
    )
    radial_unit = pursuer_xy / pursuer_xy.norm(dim=-1, keepdim=True)
    radial_speed = (result.velocity_xy * radial_unit).sum(dim=-1)
    assert torch.allclose(
        radial_speed, -4.0 * torch.ones_like(radial_speed), atol=1.0e-5
    )


def test_phase_residual_attribution_separates_support_from_opposition():
    angles = torch.tensor([0.0, 0.2, 4.0])
    pursuer_xy = 9.5 * torch.stack(
        (torch.cos(angles), torch.sin(angles)), dim=-1
    )[None]
    radial_unit = pursuer_xy / pursuer_xy.norm(dim=-1, keepdim=True)
    tangent_unit = torch.stack(
        (-radial_unit[..., 1], radial_unit[..., 0]), dim=-1
    )
    common = dict(
        pursuer_xy=pursuer_xy,
        estimated_target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.zeros(1, 2),
        reference_radius_m=torch.tensor([9.5]),
        active=torch.ones(1, dtype=torch.bool),
        previous_assignment=torch.tensor([[0, 1, 2]]),
        previous_anchor_heading_rad=torch.zeros(1),
        previous_anchor_valid=torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.5,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.0,
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
        phase_separation_control_enabled=True,
        phase_gain=0.8,
        radial_gain=1.0,
        maximum_tangential_speed_mps=3.0,
    )
    analytic = moving_target_slot_guidance(
        desired_velocity_xy=torch.zeros_like(pursuer_xy), **common
    )
    phase_direction = torch.sign(
        (analytic.velocity_xy * tangent_unit).sum(dim=-1)
    )
    supporting_action = phase_direction[..., None] * tangent_unit
    supporting = moving_target_slot_guidance(
        desired_velocity_xy=supporting_action, **common
    )
    opposing = moving_target_slot_guidance(
        desired_velocity_xy=-supporting_action, **common
    )
    assert supporting.policy_residual_phase_support_mps.item() > 0.49
    assert supporting.policy_residual_phase_opposition_mps.item() < 1.0e-6
    assert opposing.policy_residual_phase_support_mps.item() < 1.0e-6
    assert opposing.policy_residual_phase_opposition_mps.item() > 0.49


def test_radial_velocity_damping_brakes_a_fast_reference_ring_crossing():
    angles = torch.tensor(
        [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
    )
    radial_unit = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)[None]
    pursuer_xy = 5.0 * radial_unit
    inward_velocity = -2.0 * radial_unit
    common = dict(
        desired_velocity_xy=torch.zeros_like(pursuer_xy),
        pursuer_xy=pursuer_xy,
        estimated_target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.zeros(1, 2),
        reference_radius_m=torch.tensor([5.0]),
        active=torch.ones(1, dtype=torch.bool),
        previous_assignment=torch.tensor([[0, 1, 2]]),
        previous_anchor_heading_rad=torch.zeros(1),
        previous_anchor_valid=torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.0,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.0,
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
        phase_separation_control_enabled=True,
        phase_gain=0.8,
        radial_gain=1.0,
        maximum_tangential_speed_mps=3.0,
    )
    proportional_only = moving_target_slot_guidance(**common)
    damped = moving_target_slot_guidance(
        **common,
        pursuer_velocity_xy=inward_velocity,
        radial_velocity_damping_gain=math.sqrt(2.0) - 1.0,
    )
    proportional_radial = (
        proportional_only.velocity_xy * radial_unit
    ).sum(dim=-1)
    damped_radial = (damped.velocity_xy * radial_unit).sum(dim=-1)
    assert torch.allclose(proportional_radial, torch.zeros_like(proportional_radial))
    assert (damped_radial > 0.8).all()


def test_gain_residual_cannot_reverse_analytic_radial_or_phase_direction():
    balanced_angles = torch.tensor(
        [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
    )
    radial_positions = 10.0 * torch.stack(
        (torch.cos(balanced_angles), torch.sin(balanced_angles)), dim=-1
    )[None]
    common = dict(
        estimated_target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.zeros(1, 2),
        reference_radius_m=torch.tensor([9.5]),
        active=torch.ones(1, dtype=torch.bool),
        previous_assignment=torch.tensor([[0, 1, 2]]),
        previous_anchor_heading_rad=torch.zeros(1),
        previous_anchor_valid=torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.2,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.0,
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
        phase_separation_control_enabled=True,
        phase_gain=0.8,
        radial_gain=1.0,
        maximum_tangential_speed_mps=3.0,
        gain_residual_control_enabled=True,
        radial_gain_residual_scale=0.25,
        phase_gain_residual_scale=0.25,
    )
    radial_slow = moving_target_slot_guidance(
        desired_velocity_xy=torch.full_like(radial_positions, 4.0),
        pursuer_xy=radial_positions,
        gain_residual_action=-torch.ones_like(radial_positions),
        **common,
    )
    radial_fast = moving_target_slot_guidance(
        desired_velocity_xy=torch.full_like(radial_positions, -4.0),
        pursuer_xy=radial_positions,
        gain_residual_action=torch.ones_like(radial_positions),
        **common,
    )
    radial_unit = radial_positions / radial_positions.norm(
        dim=-1, keepdim=True
    )
    slow_radial_speed = (radial_slow.velocity_xy * radial_unit).sum(dim=-1)
    fast_radial_speed = (radial_fast.velocity_xy * radial_unit).sum(dim=-1)
    assert (slow_radial_speed < 0.0).all()
    assert (fast_radial_speed < slow_radial_speed).all()

    clustered_angles = torch.tensor([0.0, 0.2, 4.0])
    phase_positions = 9.5 * torch.stack(
        (torch.cos(clustered_angles), torch.sin(clustered_angles)), dim=-1
    )[None]
    neutral = moving_target_slot_guidance(
        desired_velocity_xy=torch.zeros_like(phase_positions),
        pursuer_xy=phase_positions,
        gain_residual_action=torch.zeros_like(phase_positions),
        **common,
    )
    minimum_gain = moving_target_slot_guidance(
        desired_velocity_xy=torch.full_like(phase_positions, 4.0),
        pursuer_xy=phase_positions,
        gain_residual_action=-torch.ones_like(phase_positions),
        **common,
    )
    phase_radial_unit = phase_positions / phase_positions.norm(
        dim=-1, keepdim=True
    )
    phase_tangent_unit = torch.stack(
        (-phase_radial_unit[..., 1], phase_radial_unit[..., 0]), dim=-1
    )
    neutral_tangent = (neutral.velocity_xy * phase_tangent_unit).sum(dim=-1)
    minimum_tangent = (
        minimum_gain.velocity_xy * phase_tangent_unit
    ).sum(dim=-1)
    assert torch.equal(neutral_tangent.sign(), minimum_tangent.sign())
    assert (minimum_tangent.abs() <= neutral_tangent.abs() + 1.0e-6).all()


def test_weak_blocker_phase_anchor_removes_collective_rotation_null_space():
    balanced_angles = torch.tensor(
        [math.pi / 2.0, math.pi / 2.0 + 2.0 * math.pi / 3.0,
         math.pi / 2.0 + 4.0 * math.pi / 3.0]
    )
    pursuer_xy = 9.5 * torch.stack(
        (torch.cos(balanced_angles), torch.sin(balanced_angles)), dim=-1
    )[None]
    common = dict(
        desired_velocity_xy=torch.zeros_like(pursuer_xy),
        pursuer_xy=pursuer_xy,
        estimated_target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.tensor([[4.0, 0.0]]),
        reference_radius_m=torch.tensor([9.5]),
        active=torch.ones(1, dtype=torch.bool),
        previous_assignment=torch.tensor([[0, 1, 2]]),
        previous_anchor_heading_rad=torch.zeros(1),
        previous_anchor_valid=torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.0,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.0,
        blocker_agent_index=torch.tensor([0]),
        cyclic_order_preserving_assignment=True,
        lock_valid_assignment=True,
        phase_separation_control_enabled=True,
        phase_gain=0.8,
        radial_gain=1.0,
        maximum_tangential_speed_mps=3.0,
    )
    unanchored = moving_target_slot_guidance(**common)
    anchored = moving_target_slot_guidance(
        **common,
        blocker_phase_anchor_control_enabled=True,
        blocker_phase_anchor_gain=0.3,
        blocker_phase_anchor_maximum_speed_mps=0.8,
    )
    radial_unit = pursuer_xy / pursuer_xy.norm(dim=-1, keepdim=True)
    tangent_unit = torch.stack(
        (-radial_unit[..., 1], radial_unit[..., 0]), dim=-1
    )
    unanchored_tangent = (unanchored.velocity_xy * tangent_unit).sum(dim=-1)
    anchored_tangent = (anchored.velocity_xy * tangent_unit).sum(dim=-1)
    assert torch.allclose(
        unanchored_tangent, torch.zeros_like(unanchored_tangent), atol=1.0e-5
    )
    assert -0.8001 <= anchored_tangent[0, 0].item() < -0.79
    assert anchored_tangent[0, 1].abs().item() < 1.0e-5
    assert anchored_tangent[0, 2].abs().item() < 1.0e-5

    budgeted = moving_target_slot_guidance(
        **{**common, "target_feedforward_scale": 1.0},
        blocker_phase_anchor_control_enabled=True,
        blocker_phase_anchor_gain=0.3,
        blocker_phase_anchor_maximum_speed_mps=0.8,
    )
    assert budgeted.speed_limit_active_agent_fraction.item() > 0.32
    assert budgeted.speed_limit_mean_scale.item() < 1.0
    assert budgeted.speed_limit_forward_loss_mps.item() > 0.0
    assert (
        budgeted.executed_tangential_speed_mps.item()
        < budgeted.requested_tangential_speed_mps.item()
    )


def test_circular_gap_statistics_separates_balanced_and_clustered_formations():
    balanced_angles = torch.tensor(
        [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
    )
    clustered_angles = torch.tensor([0.0, 0.2, 0.4])
    positions = torch.stack(
        (
            torch.stack(
                (torch.cos(balanced_angles), torch.sin(balanced_angles)), dim=-1
            ),
            torch.stack(
                (torch.cos(clustered_angles), torch.sin(clustered_angles)), dim=-1
            ),
        )
    )
    maximum_gap, mean_error = circular_gap_statistics(
        positions, torch.zeros(2, 2)
    )
    assert math.isclose(maximum_gap[0].item(), 2.0 * math.pi / 3.0, abs_tol=1e-5)
    assert mean_error[0].item() < 1e-5
    assert maximum_gap[1].item() > 5.0
    assert mean_error[1].item() > 1.0


def test_hybrid_slot_guidance_accepts_per_environment_policy_residual_mask():
    pursuer_xy = torch.tensor(
        [
            [[9.5, 0.0], [-4.75, 8.227], [-4.75, -8.227]],
            [[9.5, 0.0], [-4.75, 8.227], [-4.75, -8.227]],
        ]
    )
    policy = torch.full((2, 3, 2), 2.0)
    result = moving_target_slot_guidance(
        policy,
        pursuer_xy,
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.full((2,), 9.5),
        torch.ones(2, dtype=torch.bool),
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=torch.tensor([0.0, 0.5]),
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.5,
    )
    assert torch.allclose(result.velocity_xy[0], torch.zeros(3, 2), atol=3e-4)
    assert torch.allclose(result.velocity_xy[1], torch.ones(3, 2), atol=3e-4)


def test_obstacle_aware_slot_assignment_avoids_a_blocked_shortest_path():
    common = dict(
        desired_velocity_xy=torch.zeros(1, 3, 2),
        pursuer_xy=torch.tensor(
            [[[8.0, 0.0], [-2.25, 3.8971143], [-2.25, -3.8971143]]]
        ),
        estimated_target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.zeros(1, 2),
        reference_radius_m=torch.tensor([4.5]),
        active=torch.ones(1, dtype=torch.bool),
        previous_assignment=torch.full((1, 3), -1, dtype=torch.long),
        previous_anchor_heading_rad=torch.zeros(1),
        previous_anchor_valid=torch.ones(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.0,
        target_feedforward_scale=0.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.2,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=0.5,
    )
    euclidean = moving_target_slot_guidance(**common)
    obstacle_aware = moving_target_slot_guidance(
        **common,
        obstacle_xy=torch.tensor([[[6.0, 0.0]]]),
        obstacle_radius_m=torch.tensor([[0.5]]),
        pursuer_radius_m=0.5,
        assignment_path_clearance_m=0.5,
        assignment_obstacle_penalty=30.0,
    )
    assert euclidean.assignment.tolist() == [[0, 1, 2]]
    assert obstacle_aware.assignment.tolist() == [[1, 0, 2]]


def test_moving_target_slot_guidance_rate_limits_anchor_and_keeps_assignment():
    pursuer_xy = torch.tensor(
        [[[9.5, 0.0], [-4.75, 8.227], [-4.75, -8.227]]]
    )
    result = moving_target_slot_guidance(
        torch.zeros(1, 3, 2),
        pursuer_xy,
        torch.zeros(1, 2),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([9.5]),
        torch.ones(1, dtype=torch.bool),
        torch.tensor([[0, 1, 2]]),
        torch.zeros(1),
        torch.ones(1, dtype=torch.bool),
        position_gain=0.7,
        policy_residual_scale=0.35,
        target_feedforward_scale=1.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.15,
        maximum_anchor_turn_rate_rps=0.8,
        policy_dt_s=0.1,
        assignment_hysteresis_m=1.0,
    )
    assert math.isclose(result.anchor_heading_rad.item(), 0.08, abs_tol=1.0e-6)
    assert not result.assignment_switched.item()
    assert result.assignment.tolist() == [[0, 1, 2]]


def test_c27_prepares_observable_slot_assignment_during_team_loss_without_action_change():
    nominal = torch.tensor(
        [[[0.2, -0.1], [0.3, 0.4], [-0.5, 0.1]]]
    )
    result = moving_target_slot_guidance(
        nominal,
        torch.tensor([[[8.0, 0.0], [-4.0, 7.0], [-4.0, -7.0]]]),
        torch.tensor([[1.0, 0.0]]),
        torch.zeros(1, 2),
        torch.tensor([9.5]),
        torch.zeros(1, dtype=torch.bool),
        torch.full((1, 3), -1, dtype=torch.long),
        torch.zeros(1),
        torch.ones(1, dtype=torch.bool),
        position_gain=0.7,
        policy_residual_scale=0.35,
        target_feedforward_scale=1.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.15,
        maximum_anchor_turn_rate_rps=0.8,
        policy_dt_s=0.1,
        assignment_hysteresis_m=1.0,
        prepare_assignment_when_inactive=True,
    )
    assert not result.active.item()
    assert torch.equal(result.velocity_xy, nominal)
    assert (result.assignment >= 0).all()
    assert torch.isfinite(result.assigned_slot_xy).all()
    assert torch.count_nonzero(result.assigned_slot_xy) > 0


def test_moving_target_slot_prediction_leads_only_the_slot_centre():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    pursuer_xy = 4.5 * torch.stack(
        (torch.cos(angles), torch.sin(angles)), dim=-1
    )[None]
    common = dict(
        desired_velocity_xy=torch.zeros(1, 3, 2),
        pursuer_xy=pursuer_xy,
        estimated_target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.tensor([[1.0, 0.0]]),
        reference_radius_m=torch.tensor([4.5]),
        active=torch.ones(1, dtype=torch.bool),
        previous_assignment=torch.tensor([[0, 1, 2]]),
        previous_anchor_heading_rad=torch.zeros(1),
        previous_anchor_valid=torch.ones(1, dtype=torch.bool),
        position_gain=0.7,
        policy_residual_scale=0.35,
        target_feedforward_scale=1.0,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.15,
        maximum_anchor_turn_rate_rps=0.8,
        policy_dt_s=0.1,
        assignment_hysteresis_m=1.0,
    )
    baseline = moving_target_slot_guidance(**common)
    predicted = moving_target_slot_guidance(
        target_prediction_horizon_s=0.5, **common
    )
    expected_extra_x = torch.full((1, 3), 0.35)
    assert torch.allclose(
        predicted.velocity_xy[..., 0] - baseline.velocity_xy[..., 0],
        expected_extra_x,
        atol=1.0e-5,
    )
    # Capture geometry telemetry remains centred on the current target.
    assert torch.allclose(predicted.angular_error_rad, baseline.angular_error_rad)


def test_bounded_intercept_marks_forward_blocker_reachable_and_trailer_unreachable():
    pursuer_xy = torch.tensor([[[6.0, 0.0], [-6.0, 0.0], [0.0, 8.0]]])
    result = bounded_constant_velocity_intercept(
        pursuer_xy,
        torch.zeros(1, 2),
        torch.tensor([[4.4, 0.0]]),
        pursuer_speed_mps=4.0,
        minimum_horizon_s=0.25,
        maximum_horizon_s=8.0,
        horizon_samples=125,
    )
    assert result.reachable.tolist()[0][0]
    assert not result.reachable.tolist()[0][1]
    assert 0.5 < result.time_s[0, 0].item() < 1.0
    assert result.reachability_gap_m[0, 0].item() <= 0.0
    assert result.reachability_gap_m[0, 1].item() > 0.0
    assert result.point_xy.shape == pursuer_xy.shape


def test_slot_role_lock_assigns_forward_slot_to_selected_blocker():
    result = moving_target_slot_guidance(
        desired_velocity_xy=torch.zeros(1, 3, 2),
        pursuer_xy=torch.tensor([[[6.0, 0.0], [-3.0, 5.0], [-3.0, -5.0]]]),
        estimated_target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.tensor([[4.0, 0.0]]),
        reference_radius_m=torch.tensor([9.5]),
        active=torch.ones(1, dtype=torch.bool),
        previous_assignment=torch.full((1, 3), -1, dtype=torch.long),
        previous_anchor_heading_rad=torch.zeros(1),
        previous_anchor_valid=torch.zeros(1, dtype=torch.bool),
        position_gain=1.0,
        policy_residual_scale=0.2,
        target_feedforward_scale=0.25,
        maximum_speed_mps=4.0,
        minimum_target_speed_mps=0.15,
        maximum_anchor_turn_rate_rps=1.2,
        policy_dt_s=0.1,
        assignment_hysteresis_m=1.0,
        target_prediction_horizon_s=torch.tensor([0.75]),
        blocker_agent_index=torch.tensor([0]),
        blocker_role_lock_weight_m=50.0,
    )
    assert result.assignment[0, 0].item() == 0


def test_asynchronous_orientation_gate_latches_agents_at_different_times():
    staged = torch.zeros(1, 3, dtype=torch.bool)
    aligned = torch.zeros(1, 3, dtype=torch.long)
    unlocked = torch.zeros(1, dtype=torch.bool)
    visible = torch.ones(1, 3, dtype=torch.bool)
    yaw = torch.zeros(1, 3)
    for active_agent in range(3):
        distance = torch.full((1, 3), 20.0)
        distance[0, active_agent] = 8.0
        for _ in range(2):
            staged, aligned, team_staged, unlocked, _ = (
                advance_asynchronous_orientation_gate(
                    distance,
                    visible,
                    yaw,
                    staged,
                    aligned,
                    unlocked,
                    staging_radius_m=9.5,
                    yaw_tolerance_rad=math.pi / 6.0,
                    required_alignment_steps=2,
                )
            )
    assert team_staged.item()
    assert staged.all()
    assert (aligned == 2).all()
    assert unlocked.item()


def test_asynchronous_gate_requires_current_team_readiness_when_requested():
    staged = torch.ones(1, 3, dtype=torch.bool)
    aligned = torch.full((1, 3), 2, dtype=torch.long)
    unlocked = torch.zeros(1, dtype=torch.bool)
    visible = torch.ones(1, 3, dtype=torch.bool)
    yaw = torch.zeros(1, 3)
    result = advance_asynchronous_orientation_gate(
        torch.tensor([[8.0, 8.0, 12.0]]),
        visible,
        yaw,
        staged,
        aligned,
        unlocked,
        staging_radius_m=9.5,
        yaw_tolerance_rad=math.pi / 6.0,
        required_alignment_steps=2,
        require_current_team_ready_to_unlock=True,
    )
    assert not result[3].item()
    result = advance_asynchronous_orientation_gate(
        torch.tensor([[8.0, 8.0, 8.0]]),
        visible,
        yaw,
        result[0],
        result[1],
        result[3],
        staging_radius_m=9.5,
        yaw_tolerance_rad=math.pi / 6.0,
        required_alignment_steps=2,
        require_current_team_ready_to_unlock=True,
    )
    assert result[3].item()


def test_slot_prediction_evaluation_override_is_audited_and_evaluation_only():
    trainer_source = (
        ROOT / "scripts" / "train_p5_2_native_mappo.py"
    ).read_text(encoding="utf-8")
    assert '"--evaluation_slot_prediction_horizon_s"' in trainer_source
    assert (
        '"--evaluation_slot_prediction_horizon_s is only valid with "'
        in trainer_source
    )
    assert "cfg.moving_target_slot_prediction_horizon_s = (" in trainer_source
    assert '"moving_target_slot_prediction_horizon_s": (' in trainer_source
    assert '"slot_prediction_horizon_s": (' in trainer_source


def test_c19_video_replays_current_task_and_uses_unscaled_closeup_camera():
    recorder_source = (
        ROOT / "scripts" / "record_p5_2_native_success.py"
    ).read_text(encoding="utf-8")
    assert 'choices=("c5", "c8", "c19")' in recorder_source
    assert "MIXED_STATIC_OBSTACLE_SPECS" in recorder_source
    assert "cfg.dynamic_obstacle_indices = (10, 11)" in recorder_source
    assert "cfg.moving_target_slot_prediction_horizon_s = 0.5" in recorder_source
    assert "cfg.moving_target_slot_guidance_enabled = True" in recorder_source
    assert 'env.scene.sensors["closeup_camera"]' in recorder_source
    assert "REAL CRAZYFLIE CLOSE-UP" in recorder_source
    assert "unscaled" in recorder_source


def test_staged_contraction_projection_holds_then_limits_inward_speed():
    pursuer_xy = torch.tensor(
        [
            [[9.0, 0.0], [-4.5, 7.794], [-4.5, -7.794]],
            [[9.0, 0.0], [-4.5, 7.794], [-4.5, -7.794]],
            [[9.0, 0.0], [-4.5, 7.794], [-4.5, -7.794]],
        ]
    )
    inward = -pursuer_xy / pursuer_xy.norm(dim=-1, keepdim=True)
    desired = 2.0 * inward
    projected = project_staged_contraction_velocity(
        desired,
        pursuer_xy,
        torch.zeros(3, 2),
        torch.tensor([False, True, True]),
        torch.tensor([False, False, True]),
        maximum_inward_speed_mps=0.5,
    )
    projected_inward = (projected * inward).sum(dim=-1)
    assert torch.allclose(projected_inward[0], torch.full((3,), 2.0))
    assert torch.allclose(projected_inward[1], torch.zeros(3), atol=1e-6)
    assert torch.allclose(projected_inward[2], torch.full((3,), 0.5), atol=1e-6)


def test_staged_contraction_projection_can_hold_agents_asynchronously():
    pursuer_xy = torch.tensor([[[9.0, 0.0], [0.0, 9.0], [-9.0, 0.0]]])
    inward = -pursuer_xy / pursuer_xy.norm(dim=-1, keepdim=True)
    projected = project_staged_contraction_velocity(
        2.0 * inward,
        pursuer_xy,
        torch.zeros(1, 2),
        torch.tensor([[True, False, True]]),
        torch.tensor([False]),
        maximum_inward_speed_mps=0.5,
    )
    projected_inward = (projected * inward).sum(dim=-1)
    assert torch.allclose(
        projected_inward, torch.tensor([[0.0, 2.0, 0.0]]), atol=1e-6
    )


def test_target_relative_staging_preserves_translation_and_allows_reapproach():
    pursuer_xy = torch.tensor([[[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]]])
    desired = torch.tensor([[[-2.0, 0.0], [-3.0, 0.0], [-3.0, 0.0]]])
    projected = project_staged_contraction_velocity(
        desired,
        pursuer_xy,
        torch.zeros(1, 2),
        torch.tensor([[True, True, False]]),
        torch.tensor([False]),
        maximum_inward_speed_mps=0.5,
        target_velocity_xy=torch.tensor([[-2.0, 0.0]]),
    )
    # Target translation is retained for an agent holding the ring.
    assert torch.allclose(projected[0, 0], torch.tensor([-2.0, 0.0]))
    # Extra relative contraction is removed while currently inside staging.
    assert torch.allclose(projected[0, 1], torch.tensor([-2.0, 0.0]))
    # A latched agent currently outside staging is allowed to re-approach.
    assert torch.allclose(projected[0, 2], torch.tensor([-3.0, 0.0]))


def test_predictive_pairwise_projection_repairs_only_active_closing_commands():
    positions = torch.tensor(
        [
            [[-1.5, 0.0], [1.5, 0.0], [0.0, 8.0]],
            [[-1.5, 0.0], [1.5, 0.0], [0.0, 8.0]],
        ]
    )
    commands = torch.tensor(
        [
            [[2.0, 0.0], [-2.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [-2.0, 0.0], [0.0, 0.0]],
        ]
    )
    result = project_predictive_pairwise_velocity(
        commands,
        torch.zeros_like(commands),
        positions,
        torch.tensor([True, False]),
        minimum_center_distance_m=2.5,
        prediction_horizon_s=0.75,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
    )
    assert result.corrected.tolist() == [True, False]
    assert result.corrected_pair_constraints[0].item() >= 1
    assert not result.infeasible.any()
    assert torch.allclose(result.velocity_xy[1], commands[1])
    relative = result.velocity_xy[0, 0] - result.velocity_xy[0, 1]
    assert relative[0].item() <= (3.0 - 2.5) / 0.75 / 0.5 + 1e-5


def test_restorative_pairwise_projection_pushes_out_of_violated_buffer():
    positions = torch.tensor([[[-1.0, 0.0], [1.0, 0.0], [0.0, 8.0]]])
    command = torch.zeros_like(positions)
    common = dict(
        desired_velocity_xy=command,
        current_velocity_xy=torch.zeros_like(command),
        pursuer_xy=positions,
        active=torch.tensor([True]),
        minimum_center_distance_m=2.5,
        prediction_horizon_s=0.75,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
    )
    legacy = project_predictive_pairwise_velocity(**common)
    restorative = project_predictive_pairwise_velocity(
        **common, restore_unsafe_buffer=True
    )
    assert torch.equal(legacy.velocity_xy, command)
    assert restorative.corrected.item()
    assert restorative.velocity_xy[0, 0, 0].item() < 0.0
    assert restorative.velocity_xy[0, 1, 0].item() > 0.0
    realized = 0.5 * restorative.velocity_xy
    predicted = positions + 0.75 * realized
    pair_distance = (predicted[0, 0] - predicted[0, 1]).norm()
    assert pair_distance.item() >= 2.5 - 1.0e-5
    assert restorative.velocity_xy.norm(dim=-1).max().item() <= 4.0 + 1.0e-6


def test_predictive_boundary_projection_repairs_outward_command():
    position = torch.tensor([[[48.0, 0.0], [0.0, -48.0], [0.0, 0.0]]])
    command = torch.tensor([[[4.0, 0.0], [0.0, -4.0], [1.0, 1.0]]])
    result = project_predictive_arena_boundary_velocity(
        command,
        torch.zeros_like(command),
        position,
        torch.tensor([True]),
        arena_half_extent_m=50.0,
        minimum_clearance_m=1.0,
        prediction_horizon_s=1.5,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
    )
    realized = 0.5 * result.velocity_xy
    predicted = position + 1.5 * realized
    assert result.corrected.item()
    assert result.corrected_constraints.item() == 2
    assert not result.infeasible.item()
    assert predicted.abs().max().item() <= 49.0 + 1e-5
    assert torch.allclose(result.velocity_xy[0, 2], command[0, 2])


def test_predictive_pairwise_projection_preserves_safe_joint_command():
    positions = torch.tensor([[[-5.0, 0.0], [5.0, 0.0], [0.0, 8.0]]])
    commands = torch.tensor([[[-0.2, 0.0], [0.2, 0.0], [0.0, 0.1]]])
    result = project_predictive_pairwise_velocity(
        commands,
        torch.zeros_like(commands),
        positions,
        torch.ones(1, dtype=torch.bool),
        minimum_center_distance_m=2.5,
        prediction_horizon_s=0.75,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
    )
    assert not result.corrected.item()
    assert torch.allclose(result.velocity_xy, commands)
    restorative = project_predictive_pairwise_velocity(
        commands,
        torch.zeros_like(commands),
        positions,
        torch.ones(1, dtype=torch.bool),
        minimum_center_distance_m=2.5,
        prediction_horizon_s=0.75,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        restore_unsafe_buffer=True,
    )
    assert torch.equal(restorative.velocity_xy, result.velocity_xy)


def test_predictive_projection_reward_penalizes_correction_and_infeasibility():
    reward = predictive_projection_reward(
        torch.tensor([0.0, 2.0, 2.0]),
        torch.tensor([False, False, True]),
        command_speed_mps=4.0,
        correction_weight=0.05,
        infeasible_penalty=0.10,
    )
    assert torch.allclose(reward, torch.tensor([0.0, -0.025, -0.125]))


def test_predictive_target_projection_preserves_one_meter_contact_buffer():
    pursuer_xy = torch.tensor([[[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]])
    result = project_predictive_target_separation_velocity(
        desired_velocity_xy=torch.full((1, 3, 2), -4.0).mul(
            torch.tensor([1.0, 0.0])
        ),
        current_velocity_xy=torch.zeros(1, 3, 2),
        pursuer_xy=pursuer_xy,
        target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.zeros(1, 2),
        active=torch.ones(1, dtype=torch.bool),
        minimum_center_distance_m=1.0,
        prediction_horizon_s=0.75,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        projection_iterations=2,
    )
    assert result.corrected.item()
    assert result.corrected_constraints.item() > 0
    assert result.minimum_predicted_center_distance_m.item() >= 1.0 - 1.0e-5
    assert not result.infeasible.item()


def test_predictive_target_projection_restores_an_already_unsafe_buffer():
    common = dict(
        desired_velocity_xy=torch.zeros(1, 3, 2),
        current_velocity_xy=torch.zeros(1, 3, 2),
        pursuer_xy=torch.tensor([[[0.5, 0.0], [2.0, 0.0], [3.0, 0.0]]]),
        target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.zeros(1, 2),
        active=torch.ones(1, dtype=torch.bool),
        minimum_center_distance_m=1.5,
        prediction_horizon_s=1.0,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        projection_iterations=2,
    )
    legacy = project_predictive_target_separation_velocity(**common)
    restorative = project_predictive_target_separation_velocity(
        **common, restore_unsafe_buffer=True
    )
    assert legacy.velocity_xy[0, 0, 0].item() == 0.0
    assert legacy.infeasible.item()
    assert restorative.velocity_xy[0, 0, 0].item() > 0.0
    assert restorative.minimum_predicted_center_distance_m.item() >= 1.5 - 1e-5
    assert not restorative.infeasible.item()


def test_predictive_target_projection_uses_exact_future_separation_fallback():
    result = project_predictive_target_separation_velocity(
        desired_velocity_xy=torch.tensor([[[-1.9280977, -0.0064201]]]),
        current_velocity_xy=torch.tensor([[[0.0543056, 0.3773112]]]),
        pursuer_xy=torch.tensor([[[1.1473770, -2.9585657]]]),
        target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.tensor([[1.0120687, -3.8148303]]),
        active=torch.ones(1, dtype=torch.bool),
        minimum_center_distance_m=1.5,
        prediction_horizon_s=1.5,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        projection_iterations=1,
        restore_unsafe_buffer=True,
    )
    assert result.maximum_separation_fallback_constraints.item() == 1
    assert result.minimum_predicted_center_distance_m.item() >= 1.5 - 1.0e-5
    assert result.minimum_maximum_attainable_center_distance_m.item() >= 1.5
    assert not result.unavoidable.item()
    assert not result.infeasible.item()


def test_predictive_target_projection_labels_true_unavoidable_best_effort():
    result = project_predictive_target_separation_velocity(
        desired_velocity_xy=torch.zeros(1, 1, 2),
        current_velocity_xy=torch.zeros(1, 1, 2),
        pursuer_xy=torch.zeros(1, 1, 2),
        target_xy=torch.zeros(1, 2),
        target_velocity_xy=torch.zeros(1, 2),
        active=torch.ones(1, dtype=torch.bool),
        minimum_center_distance_m=2.0,
        prediction_horizon_s=1.0,
        command_response_ratio=0.1,
        maximum_speed_mps=1.0,
        projection_iterations=1,
        restore_unsafe_buffer=True,
    )
    assert result.maximum_separation_fallback_constraints.item() == 1
    assert result.unavoidable.item()
    assert result.unavoidable_constraints.item() == 1
    assert torch.allclose(result.velocity_xy, torch.tensor([[[1.0, 0.0]]]))
    assert torch.allclose(
        result.minimum_maximum_attainable_center_distance_m,
        torch.tensor([0.1]),
    )
    assert not result.infeasible.item()


def test_predictive_static_projection_stops_unsafe_radial_approach():
    result = project_predictive_static_obstacle_velocity(
        desired_velocity_xy=torch.tensor([[[2.0, 0.0]]]),
        current_velocity_xy=torch.zeros(1, 1, 2),
        pursuer_xy=torch.tensor([[[-3.0, 0.0]]]),
        obstacle_xy=torch.zeros(1, 1, 2),
        obstacle_radius_m=torch.ones(1, 1),
        active=torch.ones(1, dtype=torch.bool),
        pursuer_radius_m=0.5,
        minimum_clearance_m=1.0,
        prediction_horizon_s=1.0,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        projection_iterations=4,
    )
    assert result.corrected.item()
    assert result.corrected_constraints.item() >= 1
    assert result.velocity_xy[0, 0, 0].item() <= 1.0 + 1.0e-6
    assert result.minimum_predicted_clearance_m.item() >= 1.0 - 1.0e-6
    assert not result.infeasible.item()


def test_predictive_obstacle_projection_accounts_for_approaching_obstacle():
    common = dict(
        desired_velocity_xy=torch.zeros(1, 1, 2),
        current_velocity_xy=torch.zeros(1, 1, 2),
        pursuer_xy=torch.tensor([[[-3.0, 0.0]]]),
        obstacle_xy=torch.zeros(1, 1, 2),
        obstacle_radius_m=torch.ones(1, 1),
        active=torch.ones(1, dtype=torch.bool),
        pursuer_radius_m=0.5,
        minimum_clearance_m=1.0,
        prediction_horizon_s=1.0,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        projection_iterations=4,
    )
    static = project_predictive_static_obstacle_velocity(**common)
    approaching = project_predictive_static_obstacle_velocity(
        obstacle_velocity_xy=torch.tensor([[[-1.0, 0.0]]]),
        **common,
    )
    assert not static.corrected.item()
    assert approaching.corrected.item()
    assert approaching.velocity_xy[0, 0, 0].item() < 0.0
    assert approaching.minimum_predicted_clearance_m.item() >= 1.0 - 1.0e-6
    assert not approaching.infeasible.item()


def test_c5_keeps_c4_capture_contract_for_tangential_bypass():
    stage = native_curriculum_stage("c5")
    assert stage.static_obstacles
    assert stage.training_capture_radius_m == 6.0
    assert stage.desired_radius_m == 4.5


def test_static_tangent_bypass_activates_after_persistent_same_blocker():
    desired = torch.tensor([[[2.0, 0.0]]])
    current = torch.zeros_like(desired)
    pursuer = torch.tensor([[[-3.0, 0.0]]])
    target = torch.tensor([[10.0, 0.0]])
    obstacle = torch.zeros(1, 1, 2)
    radius = torch.ones(1, 1)
    obstacle_index = torch.full((1, 1), -1, dtype=torch.long)
    side = torch.zeros(1, 1, dtype=torch.long)
    blocked_steps = torch.zeros_like(side)
    hold_steps = torch.zeros_like(side)
    for step in range(5):
        result = resolve_static_obstacle_tangent_velocity(
            desired,
            current,
            pursuer,
            target,
            obstacle,
            radius,
            torch.ones(1, dtype=torch.bool),
            obstacle_index,
            side,
            blocked_steps,
            hold_steps,
            pursuer_radius_m=0.5,
            minimum_clearance_m=1.0,
            prediction_horizon_s=1.0,
            command_response_ratio=0.5,
            maximum_speed_mps=4.0,
            activation_steps=5,
            hold_steps=15,
            tangent_speed_mps=2.5,
            outward_speed_mps=0.35,
            release_distance_m=3.0,
        )
        obstacle_index = result.obstacle_index
        side = result.side
        blocked_steps = result.blocked_steps
        hold_steps = result.hold_steps_remaining
        assert result.active.item() is (step == 4)
    assert result.activated.item()
    assert result.side.item() in (-1, 1)
    assert abs(result.velocity_xy[0, 0, 1].item()) > 2.0
    assert result.minimum_candidate_clearance_m.item() >= 1.0


def test_static_tangent_bypass_holds_direction_after_radial_block_clears():
    result = resolve_static_obstacle_tangent_velocity(
        desired_velocity_xy=torch.tensor([[[0.0, 2.0]]]),
        current_velocity_xy=torch.zeros(1, 1, 2),
        pursuer_xy=torch.tensor([[[-3.0, 0.0]]]),
        estimated_target_xy=torch.tensor([[10.0, 0.0]]),
        obstacle_xy=torch.zeros(1, 1, 2),
        obstacle_radius_m=torch.ones(1, 1),
        active=torch.ones(1, dtype=torch.bool),
        previous_obstacle_index=torch.zeros(1, 1, dtype=torch.long),
        previous_side=torch.ones(1, 1, dtype=torch.long),
        previous_blocked_steps=torch.full((1, 1), 5, dtype=torch.long),
        previous_hold_steps_remaining=torch.full((1, 1), 10, dtype=torch.long),
        pursuer_radius_m=0.5,
        minimum_clearance_m=1.0,
        prediction_horizon_s=1.0,
        command_response_ratio=0.5,
        maximum_speed_mps=4.0,
        activation_steps=5,
        hold_steps=15,
        tangent_speed_mps=2.5,
        outward_speed_mps=0.35,
        release_distance_m=3.0,
    )
    assert result.active.item()
    assert not result.activated.item()
    assert result.side.item() == 1
    assert result.hold_steps_remaining.item() == 9
    assert result.velocity_xy[0, 0, 1].item() < -2.0


def test_orientation_gate_requires_consecutive_three_view_alignment():
    distance = torch.full((1, 3), 9.0)
    visible = torch.ones(1, 3, dtype=torch.bool)
    yaw_error = torch.zeros(1, 3)
    reached = torch.zeros(1, dtype=torch.bool)
    steps = torch.zeros(1, dtype=torch.long)
    unlocked = torch.zeros(1, dtype=torch.bool)
    for expected_steps in range(1, 6):
        reached, steps, unlocked, aligned = advance_orientation_gate(
            distance,
            visible,
            yaw_error,
            reached,
            steps,
            unlocked,
            staging_radius_m=9.5,
            yaw_tolerance_rad=math.pi / 6.0,
            required_alignment_steps=5,
        )
        assert reached.item()
        assert aligned.item()
        assert steps.item() == expected_steps
        assert unlocked.item() is (expected_steps >= 5)

    visible[:, 2] = False
    _, held_steps, held_unlocked, aligned = advance_orientation_gate(
        distance,
        visible,
        yaw_error,
        reached,
        steps,
        unlocked,
        staging_radius_m=9.5,
        yaw_tolerance_rad=math.pi / 6.0,
        required_alignment_steps=5,
    )
    assert not aligned.item()
    assert held_steps.item() == 5
    assert held_unlocked.item()


def test_native_reward_accepts_per_environment_reference_radius():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    unit_ring = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    radius = torch.tensor([9.0, 4.5])
    xy = radius[:, None, None] * unit_ring[None]
    position = torch.cat((xy, torch.full((2, 3, 1), 3.0)), dim=-1)
    false = torch.zeros(2, dtype=torch.bool)
    terms = native_team_reward(
        pursuer_position=position,
        target_position=torch.tensor([[0.0, 0.0, 3.0]]).expand(2, -1),
        pursuer_velocity=torch.zeros(2, 3, 3),
        normalized_action=torch.zeros(2, 3, 3),
        success=false,
        contact=false,
        agent_collision=false,
        obstacle_collision=false,
        boundary_invalid=false,
        height_invalid=false,
        team_visible=torch.ones(2, dtype=torch.bool),
        minimum_static_clearance_m=torch.full((2,), 10.0),
        minimum_boundary_clearance_m=torch.full((2,), 10.0),
        desired_radius_m=radius,
    )
    assert torch.allclose(terms.radial_error_m, torch.zeros(2), atol=1e-5)


def test_native_reward_visibility_bonus_can_be_disabled_for_gated_finetuning():
    angles = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    xy = 4.5 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)[None]
    position = torch.cat((xy, torch.full((1, 3, 1), 3.0)), dim=-1)
    false = torch.zeros(1, dtype=torch.bool)
    common = dict(
        pursuer_position=position,
        target_position=torch.tensor([[0.0, 0.0, 3.0]]),
        pursuer_velocity=torch.zeros(1, 3, 3),
        normalized_action=torch.zeros(1, 3, 3),
        success=false,
        contact=false,
        agent_collision=false,
        obstacle_collision=false,
        boundary_invalid=false,
        height_invalid=false,
        team_visible=torch.ones(1, dtype=torch.bool),
        minimum_static_clearance_m=torch.full((1,), 10.0),
        minimum_boundary_clearance_m=torch.full((1,), 10.0),
    )
    no_bonus = native_team_reward(team_visibility_bonus=0.0, **common)
    default_bonus = native_team_reward(team_visibility_bonus=0.01, **common)
    assert math.isclose(
        default_bonus.reward.item() - no_bonus.reward.item(),
        0.01,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
