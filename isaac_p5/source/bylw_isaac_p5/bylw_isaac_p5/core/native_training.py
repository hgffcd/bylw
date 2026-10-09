"""Pure Torch reward and curriculum terms for Isaac-native MAPPO training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class NativeCurriculumStage:
    name: str
    training_capture_radius_m: float
    desired_radius_m: float
    near_switch_m: float
    static_obstacles: bool
    target_success_rate: float


NATIVE_CURRICULUM = {
    "a0": NativeCurriculumStage("a0", 12.0, 9.0, 15.0, False, 0.60),
    "a1": NativeCurriculumStage("a1", 9.0, 6.5, 12.0, False, 0.60),
    "a2": NativeCurriculumStage("a2", 6.0, 4.5, 9.0, False, 0.70),
    "b": NativeCurriculumStage("b", 6.0, 4.5, 9.0, True, 0.50),
    "c0": NativeCurriculumStage("c0", 6.0, 4.5, 9.0, False, 0.70),
    "c1": NativeCurriculumStage("c1", 6.0, 4.5, 9.0, False, 0.70),
    "c2": NativeCurriculumStage("c2", 6.0, 4.5, 9.0, False, 0.70),
    # C3 keeps the frozen C1 perception/orientation/safety semantics and moves
    # the deterministic static obstacles into the tracked-start workspace.
    "c3": NativeCurriculumStage("c3", 6.0, 4.5, 9.0, True, 0.50),
    # C4 additionally projects commands away from static obstacle cylinders.
    "c4": NativeCurriculumStage("c4", 6.0, 4.5, 9.0, True, 0.50),
    # C5 adds a stateful tangential escape resolver before the C4 projections.
    "c5": NativeCurriculumStage("c5", 6.0, 4.5, 9.0, True, 0.50),
    # C6/G1 adds episode-randomized physical maps and a moving reactive target.
    "c6": NativeCurriculumStage("c6", 6.0, 4.5, 9.0, True, 0.30),
    # C7/G1R additionally activates pairwise predictive safety for all phases.
    "c7": NativeCurriculumStage("c7", 6.0, 4.5, 9.0, True, 0.30),
    # C8/G1A keeps C7 safety/randomization and rolls evader speed back to 1 m/s.
    "c8": NativeCurriculumStage("c8", 6.0, 4.5, 9.0, True, 0.20),
    # C9/G1B adds moving-target angular slots and hysteretic slot handoff.
    "c9": NativeCurriculumStage("c9", 6.0, 4.5, 9.0, True, 0.20),
    # C10--C12 freeze the C9 task/method and audit finite sensing ranges.  The
    # progressively lower pass thresholds are predeclared diagnostics, not
    # changes to reward, dynamics, map sampling, or capture semantics.
    "c10": NativeCurriculumStage("c10", 6.0, 4.5, 9.0, True, 0.70),
    "c11": NativeCurriculumStage("c11", 6.0, 4.5, 9.0, True, 0.60),
    "c12": NativeCurriculumStage("c12", 6.0, 4.5, 9.0, True, 0.50),
    # C13/D1 freezes C9 sensing/method semantics and raises physical obstacle
    # count from eight to twelve for a dense-map transfer gate.
    "c13": NativeCurriculumStage("c13", 6.0, 4.5, 9.0, True, 0.70),
    # C14 keeps the C13 count/arena/method contract but varies cylinder radii
    # and replaces half of the bodies with square/rectangular columns.
    "c14": NativeCurriculumStage("c14", 6.0, 4.5, 9.0, True, 0.70),
    # C15 is reserved for a possible arena-size ablation.  C16 adds two
    # bounded kinematic obstacles to the frozen C14 mixed-shape scene.
    "c16": NativeCurriculumStage("c16", 6.0, 4.5, 9.0, True, 0.70),
    # C17 keeps C16 geometry/safety and raises only the reactive evader speed
    # from 1.0 to 1.5 m/s for a bounded curriculum pilot.
    "c17": NativeCurriculumStage("c17", 6.0, 4.5, 9.0, True, 0.70),
    # C19 returns to the C16 difficulty and adds only a 0.5 s constant-velocity
    # lead to the moving slot centre after the frozen zero-shot C19 gate.
    "c19": NativeCurriculumStage("c19", 6.0, 4.5, 9.0, True, 0.70),
    # C20 is the strict equal-speed pursuit pilot.  Its evader and strong-ring
    # details live in configure_stage so the curriculum table remains a compact
    # record of the common radial task contract.
    "c20": NativeCurriculumStage("c20", 6.0, 4.5, 9.0, True, 0.10),
    # C21 repairs C20 reachability with bounded intercept roles and an
    # asynchronous staging/orientation handoff; strict speed and safety remain.
    "c21": NativeCurriculumStage("c21", 6.0, 4.5, 9.0, True, 0.10),
    # C22 applies C21 intercept lead only to the forward blocker slot.
    "c22": NativeCurriculumStage("c22", 6.0, 4.5, 9.0, True, 0.10),
    # C23 closes the predictive safety layer with arena-boundary constraints.
    "c23": NativeCurriculumStage("c23", 6.0, 4.5, 9.0, True, 0.10),
    # C24 keeps C23 but holds staging in the moving target's relative frame.
    "c24": NativeCurriculumStage("c24", 6.0, 4.5, 9.0, True, 0.10),
    # C25 is the NavRL-inspired learnable curriculum entry: equal speed and a
    # 30 s horizon, while retaining C24's strict strong-ring and safety rules.
    "c25": NativeCurriculumStage("c25", 6.0, 4.5, 9.0, True, 0.10),
    # C26 changes only the learning signal: progress is measured toward each
    # pursuer's assigned moving role slot instead of only toward the evader.
    "c26": NativeCurriculumStage("c26", 6.0, 4.5, 9.0, True, 0.10),
    # C27 keeps the C26 task and exposes role goal/phase features to the actor.
    "c27": NativeCurriculumStage("c27", 6.0, 4.5, 9.0, True, 0.10),
    # C29 keeps C27 method/task semantics but trains with a 40 s horizon;
    # acceptance is still evaluated on the frozen 30 s equal-speed task.
    "c29": NativeCurriculumStage("c29", 6.0, 4.5, 9.0, True, 0.10),
    # C30 is the hierarchical route: traditional search/approach/alignment,
    # followed by a masked MAPPO residual only during ready encirclement.
    "c30": NativeCurriculumStage("c30", 6.0, 4.5, 9.0, True, 0.10),
    # C31 preserves C30's task and safety contract, but locks cyclic formation
    # topology and requires angular separation before radial contraction.
    "c31": NativeCurriculumStage("c31", 6.0, 4.5, 9.0, True, 0.10),
    "c32": NativeCurriculumStage("c32", 6.0, 4.5, 9.0, True, 0.10),
    # C40 reopens only the static-obstacle equal-capability problem.  The
    # evader remains non-stopping at the pursuer speed/acceleration limits;
    # dynamic obstacles are removed as a causal isolation, not by weakening
    # the pursuit dynamics or strong-ring criterion.
    "c40": NativeCurriculumStage("c40", 6.0, 4.5, 9.0, True, 0.10),
    # C41 freezes the first viable C40 near-ring controller as a short-training
    # curriculum: early legal unlock, adjacent-gap control, and latched
    # reference contraction.  The equal-capability/static task is unchanged.
    "c41": NativeCurriculumStage("c41", 6.0, 4.5, 9.0, True, 0.10),
    # C42 changes only how the equal-speed command budget is composed: target
    # feedforward is retained before radial/phase corrections are scaled.
    "c42": NativeCurriculumStage("c42", 6.0, 4.5, 9.0, True, 0.10),
    # C43 returns to C41's command composition and removes the blocker slot's
    # intercept-centre lead after the near-ring curriculum has made the target
    # directly reachable.  The strict ring remains centred on the live target.
    "c43": NativeCurriculumStage("c43", 6.0, 4.5, 9.0, True, 0.10),
    # C44 retains the useful far-field blocker lead and fades it to zero only
    # while the latched reference circle contracts into the strict ring.
    "c44": NativeCurriculumStage("c44", 6.0, 4.5, 9.0, True, 0.10),
    # C45 adds a confirmation handoff only after first entering the strict
    # ring: current-target slots replace intercept lead and old RL residuals.
    "c45": NativeCurriculumStage("c45", 6.0, 4.5, 9.0, True, 0.10),
    # C46 isolates C45's two actions: it removes confirmation-time intercept
    # lead but retains the accepted policy's useful radial residual.
    "c46": NativeCurriculumStage("c46", 6.0, 4.5, 9.0, True, 0.10),
    # C47 leaves C41 nominal guidance unchanged and adds a predictive target-
    # relative annulus projection before the physical safety projection stack.
    "c47": NativeCurriculumStage("c47", 6.0, 4.5, 9.0, True, 0.10),
    # C48 gates C47 on the measured team radii entering a one-step guard band;
    # the contracted reference alone is not evidence of physical arrival.
    "c48": NativeCurriculumStage("c48", 6.0, 4.5, 9.0, True, 0.10),
    # C49 latches that measured entry so protection is not withdrawn precisely
    # when one vehicle drifts through the inner or outer guard boundary.
    "c49": NativeCurriculumStage("c49", 6.0, 4.5, 9.0, True, 0.10),
    # C50 delays the same latched projection until a physically valid strong-
    # ring step has occurred; it is a confirmation layer, not an approach law.
    "c50": NativeCurriculumStage("c50", 6.0, 4.5, 9.0, True, 0.10),
    # C51 advances entry to the first measured guard-band step whose maximum
    # angular gap already satisfies the frozen strong-ring topology bound.
    "c51": NativeCurriculumStage("c51", 6.0, 4.5, 9.0, True, 0.10),
    # C52 uses a one-policy-step angular reachability margin, but only after
    # every vehicle is outside the strict inner ring boundary.
    "c52": NativeCurriculumStage("c52", 6.0, 4.5, 9.0, True, 0.10),
    # C53 is a training-only near-ring curriculum for C52.  Capability,
    # obstacle, perception, safety, and success contracts are unchanged.
    "c53": NativeCurriculumStage("c53", 6.0, 4.5, 9.0, True, 0.10),
    # C54 keeps C53's 8 m reset, but gives the five-step yaw-alignment gate
    # enough range for an equal-speed evader to travel 2 m before unlock.
    "c54": NativeCurriculumStage("c54", 6.0, 4.5, 9.0, True, 0.10),
    # C55 augments only C52's confirmation projection with a bounded finite-
    # difference target-acceleration prediction; the task remains unchanged.
    "c55": NativeCurriculumStage("c55", 6.0, 4.5, 9.0, True, 0.10),
    # C56 accounts for the frozen 0.5 command response when deriving the
    # angular-reachability activation window for the same projection.
    "c56": NativeCurriculumStage("c56", 6.0, 4.5, 9.0, True, 0.10),
    # C57 is the learnable 2--4 m/s per-episode speed curriculum; the evader's
    # physical maximum speed and acceleration remain equal to the pursuers'.
    "c57": NativeCurriculumStage("c57", 6.0, 4.5, 9.0, True, 0.10),
    # C58 broadens only the first learning rung to 1--4 m/s so complete ring
    # positives coexist with near-equal-speed transfer pressure.
    "c58": NativeCurriculumStage("c58", 6.0, 4.5, 9.0, True, 0.10),
    # C60 keeps the fixed equal-capability C56 task and makes the predictive
    # ring's radial activation margin consistent with command response time.
    "c60": NativeCurriculumStage("c60", 6.0, 4.5, 9.0, True, 0.10),
    # C61 preserves the full target common-mode velocity before fitting
    # formation corrections into the unchanged pursuer velocity ball.
    "c61": NativeCurriculumStage("c61", 6.0, 4.5, 9.0, True, 0.10),
    # C62 returns to C58's speed curriculum but learns only bounded positive
    # gain residuals around the accepted analytic controller.
    "c62": NativeCurriculumStage("c62", 6.0, 4.5, 9.0, True, 0.10),
    # C63 keeps C62's execution contract and adds a dense reward for every
    # legal strong-ring confirmation step; the ten-step success rule is fixed.
    "c63": NativeCurriculumStage("c63", 6.0, 4.5, 9.0, True, 0.10),
    # C64 is the action-semantics-consistent fixed-4 m/s audit for C63.
    "c64": NativeCurriculumStage("c64", 6.0, 4.5, 9.0, True, 0.10),
    # C65 keeps the C64 fixed-capability contract and audits the complete
    # low-level response trajectory instead of only its 0.5 s endpoint.
    "c65": NativeCurriculumStage("c65", 6.0, 4.5, 9.0, True, 0.10),
    # C66 removes the harmful inner-boundary competition from C65's execution
    # projection while retaining the strict inner boundary in success scoring.
    "c66": NativeCurriculumStage("c66", 6.0, 4.5, 9.0, True, 0.10),
    # C67 restores the inner projection only after the first fully legal ring
    # step, avoiding phase-budget competition during approach.
    "c67": NativeCurriculumStage("c67", 6.0, 4.5, 9.0, True, 0.10),
    # C68 replaces the approximate command-response blend with the actual
    # acceleration-limited low-level velocity-loop forward model.
    "c68": NativeCurriculumStage("c68", 6.0, 4.5, 9.0, True, 0.10),
    "c69": NativeCurriculumStage("c69", 6.0, 4.5, 9.0, True, 0.10),
    "c70": NativeCurriculumStage("c70", 6.0, 4.5, 9.0, True, 0.10),
    "c71": NativeCurriculumStage("c71", 6.0, 4.5, 9.0, True, 0.10),
    "c72": NativeCurriculumStage("c72", 6.0, 4.5, 9.0, True, 0.10),
    "c73": NativeCurriculumStage("c73", 6.0, 4.5, 9.0, True, 0.10),
    "c74": NativeCurriculumStage("c74", 6.0, 4.5, 9.0, True, 0.10),
    "c75": NativeCurriculumStage("c75", 6.0, 4.5, 9.0, True, 0.10),
    "c76": NativeCurriculumStage("c76", 6.0, 4.5, 9.0, True, 0.10),
    "c77": NativeCurriculumStage("c77", 6.0, 4.5, 9.0, True, 0.10),
    "c78": NativeCurriculumStage("c78", 6.0, 4.5, 9.0, True, 0.10),
    "c79": NativeCurriculumStage("c79", 6.0, 4.5, 9.0, True, 0.10),
    "c80": NativeCurriculumStage("c80", 6.0, 4.5, 9.0, True, 0.10),
    "c81": NativeCurriculumStage("c81", 6.0, 4.5, 9.0, True, 0.10),
    "c82": NativeCurriculumStage("c82", 6.0, 4.5, 9.0, True, 0.10),
    "c83": NativeCurriculumStage("c83", 6.0, 4.5, 9.0, True, 0.10),
    "c84": NativeCurriculumStage("c84", 6.0, 4.5, 9.0, True, 0.10),
    "c85": NativeCurriculumStage("c85", 6.0, 4.5, 9.0, True, 0.10),
    # C86 predicts the complete target-relative angular-gap trajectory and
    # projects only commands that would otherwise leave the frozen ring set.
    "c86": NativeCurriculumStage("c86", 6.0, 4.5, 9.0, True, 0.10),
    # C87 reserves braking and phase authority before ring entry while
    # preserving the C86 task, policy and physical contracts.
    "c87": NativeCurriculumStage("c87", 6.0, 4.5, 9.0, True, 0.10),
    # C88 is the preregistered single-variable ablation: retain only C87's
    # analytic preentry braking reserve and leave C86 angular timing intact.
    "c88": NativeCurriculumStage("c88", 6.0, 4.5, 9.0, True, 0.10),
    # C89 keeps target separation active through a bounded short loss using
    # only the last observed position and velocity.
    "c89": NativeCurriculumStage("c89", 6.0, 4.5, 9.0, True, 0.10),
    # C90 expands the hidden target into the acceleration-bounded reachable
    # tube around C89's CV centre.
    "c90": NativeCurriculumStage("c90", 6.0, 4.5, 9.0, True, 0.10),
    # C91 keeps near-target vehicles as stationary trackers during loss and
    # selects the nearest scout outside the existing 9 m approach boundary.
    "c91": NativeCurriculumStage("c91", 6.0, 4.5, 9.0, True, 0.10),
}


def native_curriculum_stage(name: str) -> NativeCurriculumStage:
    try:
        return NATIVE_CURRICULUM[name]
    except KeyError as exc:
        raise ValueError(f"unknown native curriculum stage {name!r}") from exc


@dataclass
class NativeRewardTerms:
    reward: torch.Tensor
    team_distance_m: torch.Tensor
    distance_progress_normalized: torch.Tensor
    near_gate: torch.Tensor
    radial_error_m: torch.Tensor
    radial_error_normalized: torch.Tensor
    angular_error_rad: torch.Tensor
    angular_error_normalized: torch.Tensor
    angular_progress_normalized: torch.Tensor
    height_error_m: torch.Tensor
    vertical_speed_mps: torch.Tensor
    agent_proximity_penalty: torch.Tensor
    obstacle_proximity_penalty: torch.Tensor
    boundary_proximity_penalty: torch.Tensor
    action_energy: torch.Tensor


@dataclass
class PredictiveProjectionResult:
    velocity_xy: torch.Tensor
    active: torch.Tensor
    corrected: torch.Tensor
    correction_norm_mps: torch.Tensor
    corrected_pair_constraints: torch.Tensor
    minimum_predicted_center_distance_m: torch.Tensor
    infeasible: torch.Tensor


@dataclass
class PredictiveObstacleProjectionResult:
    velocity_xy: torch.Tensor
    active: torch.Tensor
    corrected: torch.Tensor
    correction_norm_mps: torch.Tensor
    corrected_constraints: torch.Tensor
    minimum_predicted_clearance_m: torch.Tensor
    infeasible: torch.Tensor


@dataclass
class PredictiveTargetProjectionResult:
    velocity_xy: torch.Tensor
    active: torch.Tensor
    corrected: torch.Tensor
    correction_norm_mps: torch.Tensor
    corrected_constraints: torch.Tensor
    maximum_separation_fallback_constraints: torch.Tensor
    minimum_predicted_center_distance_m: torch.Tensor
    minimum_maximum_attainable_center_distance_m: torch.Tensor
    unavoidable: torch.Tensor
    unavoidable_constraints: torch.Tensor
    infeasible: torch.Tensor


@dataclass
class PredictiveAnnulusProjectionResult:
    velocity_xy: torch.Tensor
    active: torch.Tensor
    corrected: torch.Tensor
    correction_norm_mps: torch.Tensor
    corrected_constraints: torch.Tensor
    minimum_predicted_radius_m: torch.Tensor
    maximum_predicted_radius_m: torch.Tensor
    infeasible: torch.Tensor


@dataclass
class PredictiveAngularGapProjectionResult:
    velocity_xy: torch.Tensor
    active: torch.Tensor
    corrected: torch.Tensor
    correction_norm_mps: torch.Tensor
    corrected_constraints: torch.Tensor
    minimum_predicted_radius_m: torch.Tensor
    maximum_predicted_radius_m: torch.Tensor
    maximum_predicted_gap_rad: torch.Tensor
    maximum_predicted_mean_gap_error_rad: torch.Tensor
    infeasible: torch.Tensor


@dataclass
class PredictivePreentryBrakingResult:
    velocity_xy: torch.Tensor
    active: torch.Tensor
    corrected: torch.Tensor
    correction_norm_mps: torch.Tensor
    limited_agents: torch.Tensor
    minimum_stopping_margin_m: torch.Tensor
    infeasible: torch.Tensor


@dataclass
class PredictiveBoundaryProjectionResult:
    velocity_xy: torch.Tensor
    active: torch.Tensor
    corrected: torch.Tensor
    correction_norm_mps: torch.Tensor
    corrected_constraints: torch.Tensor
    minimum_predicted_clearance_m: torch.Tensor
    infeasible: torch.Tensor


@dataclass
class StaticTangentBypassResult:
    velocity_xy: torch.Tensor
    active: torch.Tensor
    activated: torch.Tensor
    obstacle_index: torch.Tensor
    side: torch.Tensor
    blocked_steps: torch.Tensor
    hold_steps_remaining: torch.Tensor
    minimum_candidate_clearance_m: torch.Tensor


@dataclass
class MovingTargetSlotGuidanceResult:
    velocity_xy: torch.Tensor
    assignment: torch.Tensor
    assignment_switched: torch.Tensor
    anchor_heading_rad: torch.Tensor
    anchor_valid: torch.Tensor
    active: torch.Tensor
    mean_slot_error_m: torch.Tensor
    assigned_slot_xy: torch.Tensor
    angular_error_rad: torch.Tensor
    policy_residual_norm_mps: torch.Tensor
    policy_residual_radial_support_mps: torch.Tensor
    policy_residual_radial_opposition_mps: torch.Tensor
    policy_residual_phase_support_mps: torch.Tensor
    policy_residual_phase_opposition_mps: torch.Tensor
    speed_limit_active_agent_fraction: torch.Tensor
    speed_limit_mean_scale: torch.Tensor
    speed_limit_forward_loss_mps: torch.Tensor
    requested_tangential_speed_mps: torch.Tensor
    executed_tangential_speed_mps: torch.Tensor


@dataclass
class LossReacquisitionGuidanceResult:
    velocity_xy: torch.Tensor
    active: torch.Tensor
    guidance_active: torch.Tensor
    activated: torch.Tensor
    just_reacquired: torch.Tensor
    scout_index: torch.Tensor
    slot_angle_rad: torch.Tensor
    goal_xy: torch.Tensor
    distance_to_goal_m: torch.Tensor
    flank_waypoint_xy: torch.Tensor
    flank_waypoint_valid: torch.Tensor
    flank_guidance_active: torch.Tensor
    flank_fallback_active: torch.Tensor
    flank_blocker_index: torch.Tensor
    flank_selected_path_clearance_m: torch.Tensor
    flank_selected_los_clearance_m: torch.Tensor
    sector_waypoint_xy: torch.Tensor
    sector_waypoint_valid: torch.Tensor
    sector_guidance_active: torch.Tensor
    sector_fallback_active: torch.Tensor
    sector_phase_index: torch.Tensor
    sector_half_angle_rad: torch.Tensor
    sector_radius_m: torch.Tensor
    sector_selected_path_clearance_m: torch.Tensor
    sector_any_path_candidate: torch.Tensor
    sector_any_boundary_candidate: torch.Tensor


@dataclass
class OcclusionFlankWaypointResult:
    waypoint_xy: torch.Tensor
    valid: torch.Tensor
    blocker_index: torch.Tensor
    path_clearance_m: torch.Tensor
    los_clearance_m: torch.Tensor
    route_cost_m: torch.Tensor


@dataclass
class ReachableSectorWaypointResult:
    waypoint_xy: torch.Tensor
    valid: torch.Tensor
    phase_index: torch.Tensor
    half_angle_rad: torch.Tensor
    radius_m: torch.Tensor
    path_clearance_m: torch.Tensor
    route_cost_m: torch.Tensor
    route_complete: torch.Tensor
    any_path_candidate: torch.Tensor
    any_boundary_candidate: torch.Tensor


@dataclass
class SlotGoalProgressRewardResult:
    reward: torch.Tensor
    normalized_progress: torch.Tensor
    post_action_mean_error_m: torch.Tensor


@dataclass
class BoundedInterceptResult:
    time_s: torch.Tensor
    point_xy: torch.Tensor
    reachable: torch.Tensor
    reachability_gap_m: torch.Tensor


def circular_gap_error(pursuer_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    _, mean_gap_error = circular_gap_statistics(pursuer_xy, target_xy)
    return mean_gap_error


def circular_gap_statistics(
    pursuer_xy: torch.Tensor, target_xy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return maximum circular gap and mean error from uniform separation."""
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("pursuer_xy must have shape [batch, agent, 2]")
    if target_xy.shape != (pursuer_xy.shape[0], 2):
        raise ValueError("target_xy shape mismatch")
    if pursuer_xy.shape[1] < 2:
        raise ValueError("circular gap statistics require at least two pursuers")
    relative = pursuer_xy - target_xy[:, None, :]
    angles = torch.sort(
        torch.remainder(torch.atan2(relative[..., 1], relative[..., 0]), 2.0 * math.pi),
        dim=1,
    ).values
    gaps = torch.cat(
        (angles[:, 1:] - angles[:, :-1], 2.0 * math.pi - angles[:, -1:] + angles[:, :1]),
        dim=1,
    )
    desired_gap = 2.0 * math.pi / pursuer_xy.shape[1]
    return gaps.max(dim=1).values, (gaps - desired_gap).abs().mean(dim=1)


def strong_ring_violation_score(
    pursuer_xy: torch.Tensor,
    target_xy: torch.Tensor,
    minimum_radius_m: float,
    maximum_radius_m: float,
    maximum_gap_rad: float,
    maximum_mean_gap_error_rad: float,
) -> torch.Tensor:
    """Return a dimensionless distance to the frozen strong-ring constraints.

    Zero means that every radial and angular geometry constraint is satisfied.
    Each positive component is normalized by a threshold-derived geometric
    scale, so the reward does not depend on an empirically scanned coefficient.
    """
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("pursuer_xy must have shape [batch, agent, 2]")
    if target_xy.shape != (pursuer_xy.shape[0], 2):
        raise ValueError("target_xy shape mismatch")
    if pursuer_xy.shape[1] < 3:
        raise ValueError("strong ring requires at least three pursuers")
    if not 0.0 < minimum_radius_m < maximum_radius_m:
        raise ValueError("strong ring radii must satisfy 0 < minimum < maximum")
    if maximum_mean_gap_error_rad <= 0.0:
        raise ValueError("maximum_mean_gap_error_rad must be positive")

    desired_gap = 2.0 * math.pi / pursuer_xy.shape[1]
    maximum_gap_slack = maximum_gap_rad - desired_gap
    if maximum_gap_slack <= 0.0:
        raise ValueError("maximum_gap_rad must exceed the uniform angular gap")

    radius = (pursuer_xy - target_xy[:, None, :]).norm(dim=-1)
    radial_violation = (
        torch.relu(minimum_radius_m - radius)
        + torch.relu(radius - maximum_radius_m)
    ).mean(dim=1)
    radial_scale = 0.5 * (maximum_radius_m - minimum_radius_m)
    maximum_gap, mean_gap_error = circular_gap_statistics(pursuer_xy, target_xy)
    return (
        radial_violation / radial_scale
        + torch.relu(maximum_gap - maximum_gap_rad) / maximum_gap_slack
        + torch.relu(mean_gap_error - maximum_mean_gap_error_rad)
        / maximum_mean_gap_error_rad
    )


def _point_to_segment_distance(
    segment_start_xy: torch.Tensor,
    segment_end_xy: torch.Tensor,
    point_xy: torch.Tensor,
) -> torch.Tensor:
    direction = segment_end_xy - segment_start_xy
    offset = point_xy - segment_start_xy
    denominator = (direction * direction).sum(dim=-1).clamp_min(1.0e-9)
    projection = ((offset * direction).sum(dim=-1) / denominator).clamp(0.0, 1.0)
    nearest = segment_start_xy + projection[..., None] * direction
    return (point_xy - nearest).norm(dim=-1)


def occlusion_flank_waypoint(
    pursuer_xy: torch.Tensor,
    last_seen_target_xy: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radius_m: torch.Tensor,
    scout_index: torch.Tensor,
    *,
    pursuer_radius_m: float,
    minimum_path_clearance_m: float,
    waypoint_margin_m: float,
    los_margin_m: float,
    blocker_margin_m: float,
    arena_half_extent_m: float,
    boundary_clearance_m: float,
    angular_samples: int,
) -> OcclusionFlankWaypointResult:
    """Select a safe side waypoint around the first Last-line blocker.

    The helper receives only the remembered target point, current pursuer
    positions, and known obstacle geometry.  It deliberately has no current
    target position or velocity input.  Candidate routes must keep the scout's
    physical path clear of inflated obstacles and restore an unobstructed line
    from the waypoint to the remembered target point.
    """
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("pursuer_xy must have shape [batch, agent, 2]")
    batch, agents, _ = pursuer_xy.shape
    if last_seen_target_xy.shape != (batch, 2):
        raise ValueError("last_seen_target_xy shape mismatch")
    if obstacle_xy.ndim != 3 or obstacle_xy.shape[0] != batch or obstacle_xy.shape[-1] != 2:
        raise ValueError("obstacle_xy must have shape [batch, obstacle, 2]")
    if obstacle_radius_m.shape != obstacle_xy.shape[:2]:
        raise ValueError("obstacle radius shape mismatch")
    if scout_index.shape != (batch,):
        raise ValueError("scout_index shape mismatch")
    if angular_samples < 8:
        raise ValueError("angular_samples must be at least eight")
    for name, value in (
        ("pursuer_radius_m", pursuer_radius_m),
        ("minimum_path_clearance_m", minimum_path_clearance_m),
        ("waypoint_margin_m", waypoint_margin_m),
        ("los_margin_m", los_margin_m),
        ("blocker_margin_m", blocker_margin_m),
        ("arena_half_extent_m", arena_half_extent_m),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if boundary_clearance_m < 0.0 or boundary_clearance_m >= arena_half_extent_m:
        raise ValueError("invalid boundary clearance")

    safe_scout_index = scout_index.clamp(0, agents - 1)
    scout_xy = torch.gather(
        pursuer_xy,
        1,
        safe_scout_index[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    obstacle_count = obstacle_xy.shape[1]
    if obstacle_count == 0:
        return OcclusionFlankWaypointResult(
            waypoint_xy=last_seen_target_xy.clone(),
            valid=torch.zeros(batch, dtype=torch.bool, device=pursuer_xy.device),
            blocker_index=torch.full(
                (batch,), -1, dtype=torch.long, device=pursuer_xy.device
            ),
            path_clearance_m=torch.full(
                (batch,), -1.0, dtype=pursuer_xy.dtype, device=pursuer_xy.device
            ),
            los_clearance_m=torch.full(
                (batch,), -1.0, dtype=pursuer_xy.dtype, device=pursuer_xy.device
            ),
            route_cost_m=torch.full(
                (batch,), torch.inf, dtype=pursuer_xy.dtype, device=pursuer_xy.device
            ),
        )

    line = last_seen_target_xy - scout_xy
    line_length_sq = (line * line).sum(dim=-1).clamp_min(1.0e-9)
    obstacle_offset = obstacle_xy - scout_xy[:, None, :]
    line_projection = (
        (obstacle_offset * line[:, None, :]).sum(dim=-1)
        / line_length_sq[:, None]
    )
    nearest = scout_xy[:, None, :] + line_projection[..., None] * line[:, None, :]
    line_distance = (obstacle_xy - nearest).norm(dim=-1)
    blocks_last_line = (
        (line_projection > 0.02)
        & (line_projection < 0.98)
        & (line_distance <= obstacle_radius_m + blocker_margin_m)
    )
    blocker_score = torch.where(
        blocks_last_line,
        line_projection,
        torch.full_like(line_projection, torch.inf),
    )
    blocker_index = blocker_score.argmin(dim=1)
    blocker_found = blocks_last_line.any(dim=1)
    safe_blocker_index = blocker_index.clamp(0, obstacle_count - 1)
    blocker_center = torch.gather(
        obstacle_xy,
        1,
        safe_blocker_index[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    blocker_radius = torch.gather(
        obstacle_radius_m, 1, safe_blocker_index[:, None]
    ).squeeze(1)

    angles = (
        2.0
        * math.pi
        * torch.arange(
            angular_samples, dtype=pursuer_xy.dtype, device=pursuer_xy.device
        )
        / float(angular_samples)
    )
    unit = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    waypoint_radius = (
        blocker_radius
        + pursuer_radius_m
        + minimum_path_clearance_m
        + waypoint_margin_m
    )
    candidates = blocker_center[:, None, :] + waypoint_radius[:, None, None] * unit[None]

    candidate_start = scout_xy[:, None, None, :].expand(
        -1, angular_samples, obstacle_count, -1
    )
    candidate_end = candidates[:, :, None, :].expand(
        -1, -1, obstacle_count, -1
    )
    obstacle_points = obstacle_xy[:, None, :, :].expand(
        -1, angular_samples, -1, -1
    )
    path_distance = _point_to_segment_distance(
        candidate_start, candidate_end, obstacle_points
    )
    inflated_path_radius = (
        obstacle_radius_m + pursuer_radius_m + minimum_path_clearance_m
    )
    path_clearance = path_distance - inflated_path_radius[:, None, :]
    minimum_candidate_path_clearance = path_clearance.min(dim=2).values
    path_clear = (path_clearance >= -1.0e-5).all(dim=2)

    los_start = candidates[:, :, None, :].expand(-1, -1, obstacle_count, -1)
    los_end = last_seen_target_xy[:, None, None, :].expand(
        -1, angular_samples, obstacle_count, -1
    )
    los_distance = _point_to_segment_distance(los_start, los_end, obstacle_points)
    los_clearance = los_distance - (obstacle_radius_m + los_margin_m)[:, None, :]
    minimum_candidate_los_clearance = los_clearance.min(dim=2).values
    los_clear = (los_clearance >= -1.0e-5).all(dim=2)

    boundary_limit = arena_half_extent_m - boundary_clearance_m
    within_boundary = candidates.abs().amax(dim=2) <= boundary_limit
    valid_candidate = path_clear & los_clear & within_boundary & blocker_found[:, None]
    route_cost = (candidates - scout_xy[:, None, :]).norm(dim=-1) + 0.25 * (
        candidates - last_seen_target_xy[:, None, :]
    ).norm(dim=-1)
    scored_cost = torch.where(
        valid_candidate, route_cost, torch.full_like(route_cost, torch.inf)
    )
    selected_index = scored_cost.argmin(dim=1)
    valid = valid_candidate.any(dim=1)
    selected_waypoint = torch.gather(
        candidates,
        1,
        selected_index[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    selected_path_clearance = torch.gather(
        minimum_candidate_path_clearance, 1, selected_index[:, None]
    ).squeeze(1)
    selected_los_clearance = torch.gather(
        minimum_candidate_los_clearance, 1, selected_index[:, None]
    ).squeeze(1)
    selected_cost = torch.gather(scored_cost, 1, selected_index[:, None]).squeeze(1)
    blocker_index = torch.where(
        blocker_found,
        blocker_index,
        torch.full_like(blocker_index, -1),
    )
    return OcclusionFlankWaypointResult(
        waypoint_xy=torch.where(
            valid[:, None], selected_waypoint, last_seen_target_xy
        ),
        valid=valid,
        blocker_index=blocker_index,
        path_clearance_m=torch.where(
            valid,
            selected_path_clearance,
            torch.full_like(selected_path_clearance, -1.0),
        ),
        los_clearance_m=torch.where(
            valid,
            selected_los_clearance,
            torch.full_like(selected_los_clearance, -1.0),
        ),
        route_cost_m=selected_cost,
    )


def reachable_sector_waypoint(
    pursuer_xy: torch.Tensor,
    last_seen_target_xy: torch.Tensor,
    last_seen_target_velocity_xy: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radius_m: torch.Tensor,
    scout_index: torch.Tensor,
    lost_steps: torch.Tensor,
    *,
    policy_dt_s: float,
    target_speed_bound_mps: float,
    velocity_heading_threshold_mps: float,
    lookahead_s: float,
    minimum_radius_m: float,
    maximum_radius_m: float,
    initial_half_angle_rad: float,
    half_angle_growth_rps: float,
    phase_hold_steps: int,
    pursuer_radius_m: float,
    minimum_path_clearance_m: float,
    arena_half_extent_m: float,
    boundary_clearance_m: float,
    detour_margin_m: float,
    detour_maximum_extra_m: float,
    detour_radial_samples: int,
    detour_angular_samples: int,
) -> ReachableSectorWaypointResult:
    """Select a safe waypoint in an expanding last-state reachable sector.

    Only the last team-visible target state is accepted.  The five fixed scan
    phases are centre, positive half-offset, negative half-offset, positive
    edge, and negative edge.  A phase is held for several policy steps to
    avoid an unrealizable step-to-step zig-zag.  When the desired candidate is
    blocked, the nearest phase in the fixed cyclic order is used.  A complete
    two-leg route is preferred.  If another obstacle blocks the unexecuted
    second leg, a collision-checked first-leg frontier waypoint is returned
    and the route is replanned at the next policy step.
    """
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("pursuer_xy must have shape [batch, agent, 2]")
    batch, agents, _ = pursuer_xy.shape
    if last_seen_target_xy.shape != (batch, 2):
        raise ValueError("last_seen_target_xy shape mismatch")
    if last_seen_target_velocity_xy.shape != (batch, 2):
        raise ValueError("last_seen_target_velocity_xy shape mismatch")
    if obstacle_xy.ndim != 3 or obstacle_xy.shape[0] != batch or obstacle_xy.shape[-1] != 2:
        raise ValueError("obstacle_xy must have shape [batch, obstacle, 2]")
    if obstacle_radius_m.shape != obstacle_xy.shape[:2]:
        raise ValueError("obstacle radius shape mismatch")
    if scout_index.shape != (batch,) or lost_steps.shape != (batch,):
        raise ValueError("scout_index and lost_steps must have shape [batch]")
    if policy_dt_s <= 0.0 or target_speed_bound_mps <= 0.0:
        raise ValueError("sector timing and speed bound must be positive")
    if velocity_heading_threshold_mps < 0.0 or lookahead_s < 0.0:
        raise ValueError("sector heading threshold and lookahead must be nonnegative")
    if minimum_radius_m <= 0.0 or maximum_radius_m < minimum_radius_m:
        raise ValueError("invalid sector radius bounds")
    if not 0.0 <= initial_half_angle_rad <= math.pi:
        raise ValueError("initial sector half angle must be in [0, pi]")
    if half_angle_growth_rps < 0.0 or phase_hold_steps < 1:
        raise ValueError("invalid sector growth or phase hold")
    if pursuer_radius_m < 0.0 or minimum_path_clearance_m < 0.0:
        raise ValueError("sector path radii must be nonnegative")
    if arena_half_extent_m <= 0.0:
        raise ValueError("arena half extent must be positive")
    if boundary_clearance_m < 0.0 or boundary_clearance_m >= arena_half_extent_m:
        raise ValueError("invalid boundary clearance")
    if (
        detour_margin_m < 0.0
        or detour_maximum_extra_m < detour_margin_m
        or detour_radial_samples < 1
        or detour_angular_samples < 8
    ):
        raise ValueError("invalid sector detour sampling")

    safe_scout_index = scout_index.clamp(0, agents - 1)
    scout_xy = torch.gather(
        pursuer_xy,
        1,
        safe_scout_index[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    measured_speed = last_seen_target_velocity_xy.norm(dim=-1)
    measured_heading = last_seen_target_velocity_xy / measured_speed[:, None].clamp_min(
        1.0e-9
    )
    fallback_delta = last_seen_target_xy - scout_xy
    fallback_heading = fallback_delta / fallback_delta.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-9)
    default_heading = torch.zeros_like(fallback_heading)
    default_heading[:, 0] = 1.0
    fallback_heading = torch.where(
        (fallback_delta.norm(dim=-1) >= 1.0e-6)[:, None],
        fallback_heading,
        default_heading,
    )
    heading = torch.where(
        (measured_speed >= velocity_heading_threshold_mps)[:, None],
        measured_heading,
        fallback_heading,
    )
    base_angle = torch.atan2(heading[:, 1], heading[:, 0])

    lost_time_s = lost_steps.to(pursuer_xy.dtype).clamp_min(1) * policy_dt_s
    half_angle = (
        initial_half_angle_rad + half_angle_growth_rps * lost_time_s
    ).clamp(max=math.pi)
    radius = (target_speed_bound_mps * (lost_time_s + lookahead_s)).clamp(
        min=minimum_radius_m,
        max=maximum_radius_m,
    )
    phase_offsets = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0],
        dtype=pursuer_xy.dtype,
        device=pursuer_xy.device,
    )
    phase_count = int(phase_offsets.numel())
    phase_index = torch.div(
        lost_steps.clamp_min(1) - 1,
        phase_hold_steps,
        rounding_mode="floor",
    ).remainder(phase_count)
    candidate_angles = base_angle[:, None] + half_angle[:, None] * phase_offsets[None]
    candidates = last_seen_target_xy[:, None, :] + radius[:, None, None] * torch.stack(
        (torch.cos(candidate_angles), torch.sin(candidate_angles)), dim=-1
    )

    boundary_limit = arena_half_extent_m - boundary_clearance_m
    obstacle_count = obstacle_xy.shape[1]
    if obstacle_count:
        candidate_start = scout_xy[:, None, None, :].expand(
            -1, phase_count, obstacle_count, -1
        )
        candidate_end = candidates[:, :, None, :].expand(
            -1, -1, obstacle_count, -1
        )
        obstacle_points = obstacle_xy[:, None, :, :].expand(
            -1, phase_count, -1, -1
        )
        path_distance = _point_to_segment_distance(
            candidate_start, candidate_end, obstacle_points
        )
        inflated_radius = (
            obstacle_radius_m + pursuer_radius_m + minimum_path_clearance_m
        )
        safety_path_clearance = path_distance - inflated_radius[:, None, :]
        physical_path_clearance = path_distance - (
            obstacle_radius_m + pursuer_radius_m
        )[:, None, :]
        start_safety_clearance = (
            (scout_xy[:, None, :] - obstacle_xy).norm(dim=-1)
            - inflated_radius
        )
        # A scout may begin inside the soft 1.5 m prediction buffer while
        # remaining physically collision-free.  Rejecting every route in that
        # state creates a deadlock.  Permit only restorative paths: physical
        # clearance must stay nonnegative and an already-negative soft
        # clearance may not become more negative anywhere on the segment.
        restorative_floor = torch.minimum(
            start_safety_clearance,
            torch.zeros_like(start_safety_clearance),
        )
        direct_obstacle_clear = (
            (physical_path_clearance >= -1.0e-5)
            & (
                safety_path_clearance
                >= restorative_floor[:, None, :] - 1.0e-5
            )
        )
        direct_path_clear = direct_obstacle_clear.all(dim=2)
        direct_minimum_clearance = physical_path_clearance.min(
            dim=2
        ).values

        line = candidates - scout_xy[:, None, :]
        line_length_sq = (line * line).sum(dim=-1).clamp_min(1.0e-9)
        obstacle_offset = obstacle_xy[:, None, :, :] - scout_xy[:, None, None, :]
        line_projection = (
            (obstacle_offset * line[:, :, None, :]).sum(dim=-1)
            / line_length_sq[:, :, None]
        )
        blocking_obstacle = ~direct_obstacle_clear
        blocker_score = torch.where(
            blocking_obstacle,
            line_projection.clamp(0.0, 1.0),
            torch.full_like(line_projection, torch.inf),
        )
        blocker_index = blocker_score.argmin(dim=2)
        blocker_found = blocking_obstacle.any(dim=2)
        blocker_center = torch.gather(
            obstacle_xy[:, None, :, :].expand(-1, phase_count, -1, -1),
            2,
            blocker_index[:, :, None, None].expand(-1, -1, 1, 2),
        ).squeeze(2)
        blocker_radius = torch.gather(
            obstacle_radius_m[:, None, :].expand(-1, phase_count, -1),
            2,
            blocker_index[:, :, None],
        ).squeeze(2)

        detour_angles = (
            2.0
            * math.pi
            * torch.arange(
                detour_angular_samples,
                dtype=pursuer_xy.dtype,
                device=pursuer_xy.device,
            )
            / float(detour_angular_samples)
        )
        detour_unit = torch.stack(
            (torch.cos(detour_angles), torch.sin(detour_angles)), dim=-1
        )
        detour_extra = torch.linspace(
            detour_margin_m,
            detour_maximum_extra_m,
            detour_radial_samples,
            dtype=pursuer_xy.dtype,
            device=pursuer_xy.device,
        )
        detour_radius = (
            blocker_radius[:, :, None]
            + pursuer_radius_m
            + minimum_path_clearance_m
            + detour_extra[None, None, :]
        )
        detour_candidates = (
            blocker_center[:, :, None, None, :]
            + detour_radius[:, :, :, None, None]
            * detour_unit[None, None, None, :, :]
        ).reshape(batch, phase_count, -1, 2)
        detour_sample_count = detour_radial_samples * detour_angular_samples
        detour_within_boundary = (
            detour_candidates.abs().amax(dim=3) <= boundary_limit
        )

        expanded_obstacles = obstacle_xy[:, None, None, :, :].expand(
            -1, phase_count, detour_sample_count, -1, -1
        )
        first_leg_distance = _point_to_segment_distance(
            scout_xy[:, None, None, None, :].expand_as(expanded_obstacles),
            detour_candidates[:, :, :, None, :].expand_as(expanded_obstacles),
            expanded_obstacles,
        )
        first_leg_safety = first_leg_distance - inflated_radius[:, None, None, :]
        first_leg_physical = first_leg_distance - (
            obstacle_radius_m + pursuer_radius_m
        )[:, None, None, :]
        first_leg_clear = (
            (first_leg_physical >= -1.0e-5)
            & (
                first_leg_safety
                >= restorative_floor[:, None, None, :] - 1.0e-5
            )
        ).all(dim=3)

        second_leg_distance = _point_to_segment_distance(
            detour_candidates[:, :, :, None, :].expand_as(expanded_obstacles),
            candidates[:, :, None, None, :].expand_as(expanded_obstacles),
            expanded_obstacles,
        )
        second_leg_safety = second_leg_distance - inflated_radius[:, None, None, :]
        second_leg_physical = second_leg_distance - (
            obstacle_radius_m + pursuer_radius_m
        )[:, None, None, :]
        detour_start_safety = (
            (
                detour_candidates[:, :, :, None, :]
                - expanded_obstacles
            ).norm(dim=-1)
            - inflated_radius[:, None, None, :]
        )
        second_leg_floor = torch.minimum(
            detour_start_safety, torch.zeros_like(detour_start_safety)
        )
        second_leg_clear = (
            (second_leg_physical >= -1.0e-5)
            & (second_leg_safety >= second_leg_floor - 1.0e-5)
        ).all(dim=3)
        detour_complete_valid = (
            blocker_found[:, :, None]
            & detour_within_boundary
            & first_leg_clear
            & second_leg_clear
        )
        # Receding-horizon frontier fallback: the controller executes only the
        # first leg.  Requiring the still-unexecuted second leg to be globally
        # clear makes dense multi-obstacle layouts report no route even when a
        # safe move around the first blocker exists.  Preserve all physical
        # and restorative soft-clearance checks on the executed segment.
        detour_frontier_valid = (
            blocker_found[:, :, None]
            & detour_within_boundary
            & first_leg_clear
        )
        detour_route_cost = (
            (detour_candidates - scout_xy[:, None, None, :]).norm(dim=-1)
            + (candidates[:, :, None, :] - detour_candidates).norm(dim=-1)
        )
        complete_detour_available = detour_complete_valid.any(dim=2)
        selectable_detour = torch.where(
            complete_detour_available[:, :, None],
            detour_complete_valid,
            detour_frontier_valid,
        )
        scored_detour_cost = torch.where(
            selectable_detour,
            detour_route_cost,
            torch.full_like(detour_route_cost, torch.inf),
        )
        selected_detour_index = scored_detour_cost.argmin(dim=2)
        detour_available = selectable_detour.any(dim=2)
        selected_detour = torch.gather(
            detour_candidates,
            2,
            selected_detour_index[:, :, None, None].expand(-1, -1, 1, 2),
        ).squeeze(2)
        selected_detour_complete = torch.gather(
            detour_complete_valid,
            2,
            selected_detour_index[:, :, None],
        ).squeeze(2)
        selected_detour_cost = torch.gather(
            scored_detour_cost, 2, selected_detour_index[:, :, None]
        ).squeeze(2)
        first_leg_minimum_clearance = first_leg_physical.min(dim=3).values
        complete_detour_minimum_clearance = torch.minimum(
            first_leg_minimum_clearance,
            second_leg_physical.min(dim=3).values,
        )
        detour_minimum_clearance = torch.where(
            detour_complete_valid,
            complete_detour_minimum_clearance,
            first_leg_minimum_clearance,
        )
        selected_detour_clearance = torch.gather(
            detour_minimum_clearance,
            2,
            selected_detour_index[:, :, None],
        ).squeeze(2)

        path_clear = direct_path_clear | detour_available
        route_waypoints = torch.where(
            direct_path_clear[:, :, None], candidates, selected_detour
        )
        route_complete_by_phase = direct_path_clear | selected_detour_complete
        minimum_candidate_path_clearance = torch.where(
            direct_path_clear,
            direct_minimum_clearance,
            selected_detour_clearance,
        )
        route_cost = torch.where(
            direct_path_clear,
            (candidates - scout_xy[:, None, :]).norm(dim=-1),
            selected_detour_cost,
        )
    else:
        minimum_candidate_path_clearance = torch.full(
            (batch, phase_count),
            torch.inf,
            dtype=pursuer_xy.dtype,
            device=pursuer_xy.device,
        )
        path_clear = torch.ones(
            batch, phase_count, dtype=torch.bool, device=pursuer_xy.device
        )
        route_waypoints = candidates
        route_cost = (candidates - scout_xy[:, None, :]).norm(dim=-1)
        route_complete_by_phase = torch.ones(
            batch, phase_count, dtype=torch.bool, device=pursuer_xy.device
        )

    within_boundary = candidates.abs().amax(dim=2) <= boundary_limit
    valid_candidate = path_clear & within_boundary
    complete_candidate = route_complete_by_phase & within_boundary
    any_complete_candidate = complete_candidate.any(dim=1)
    valid_candidate = torch.where(
        any_complete_candidate[:, None], complete_candidate, valid_candidate
    )
    candidate_index = torch.arange(phase_count, device=pursuer_xy.device)[None, :]
    raw_phase_distance = (candidate_index - phase_index[:, None]).abs()
    cyclic_phase_distance = torch.minimum(
        raw_phase_distance, phase_count - raw_phase_distance
    ).to(pursuer_xy.dtype)
    scored_cost = cyclic_phase_distance * 1.0e3 + route_cost
    scored_cost = torch.where(
        valid_candidate, scored_cost, torch.full_like(scored_cost, torch.inf)
    )
    selected_index = scored_cost.argmin(dim=1)
    valid = valid_candidate.any(dim=1)
    selected_waypoint = torch.gather(
        route_waypoints, 1, selected_index[:, None, None].expand(-1, 1, 2)
    ).squeeze(1)
    selected_clearance = torch.gather(
        minimum_candidate_path_clearance, 1, selected_index[:, None]
    ).squeeze(1)
    selected_cost = torch.gather(scored_cost, 1, selected_index[:, None]).squeeze(1)
    selected_route_complete = torch.gather(
        route_complete_by_phase, 1, selected_index[:, None]
    ).squeeze(1)
    selected_phase = torch.where(
        valid,
        selected_index,
        torch.full_like(selected_index, -1),
    )
    return ReachableSectorWaypointResult(
        waypoint_xy=torch.where(valid[:, None], selected_waypoint, last_seen_target_xy),
        valid=valid,
        phase_index=selected_phase,
        half_angle_rad=half_angle,
        radius_m=radius,
        path_clearance_m=torch.where(
            valid, selected_clearance, torch.full_like(selected_clearance, -1.0)
        ),
        route_cost_m=selected_cost,
        route_complete=valid & selected_route_complete,
        any_path_candidate=path_clear.any(dim=1),
        any_boundary_candidate=within_boundary.any(dim=1),
    )


def smooth_clearance_barrier(
    clearance_m: torch.Tensor,
    safe_clearance_m: float,
    smoothing_m: float,
) -> torch.Tensor:
    """Normalized soft hinge that is about one at zero clearance."""
    if safe_clearance_m <= 0.0 or smoothing_m <= 0.0:
        raise ValueError("clearance and smoothing scales must be positive")
    argument = (safe_clearance_m - clearance_m) / smoothing_m
    return smoothing_m * functional.softplus(argument) / safe_clearance_m


def target_facing_yaw_guidance(
    pursuer_xy: torch.Tensor,
    pursuer_yaw_rad: torch.Tensor,
    estimated_target_xy: torch.Tensor,
    policy_yaw_rate_rps: torch.Tensor,
    *,
    proportional_gain: float,
    policy_residual_scale: float | torch.Tensor,
    maximum_yaw_rate_rps: float,
    gain_multiplier: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend target-facing proportional guidance with a learned yaw residual."""
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("pursuer_xy must have shape [batch, agent, 2]")
    if pursuer_yaw_rad.shape != pursuer_xy.shape[:2]:
        raise ValueError("pursuer_yaw_rad shape mismatch")
    if estimated_target_xy.shape != (pursuer_xy.shape[0], 2):
        raise ValueError("estimated_target_xy shape mismatch")
    if policy_yaw_rate_rps.shape != pursuer_yaw_rad.shape:
        raise ValueError("policy_yaw_rate_rps shape mismatch")
    if proportional_gain <= 0.0 or maximum_yaw_rate_rps <= 0.0:
        raise ValueError("yaw gain and rate limit must be positive")
    batch = pursuer_xy.shape[0]
    residual_scale = torch.as_tensor(
        policy_residual_scale,
        dtype=pursuer_xy.dtype,
        device=pursuer_xy.device,
    )
    if residual_scale.ndim == 0:
        residual_scale = residual_scale.expand(batch)
    elif residual_scale.shape != (batch,):
        raise ValueError("policy_residual_scale must be scalar or shape [batch]")
    if ((residual_scale < 0.0) | (residual_scale > 1.0)).any():
        raise ValueError("policy_residual_scale must be in [0, 1]")
    if gain_multiplier is None:
        gain_multiplier = torch.ones_like(pursuer_yaw_rad)
    elif gain_multiplier.shape != pursuer_yaw_rad.shape:
        raise ValueError("yaw gain multiplier shape mismatch")
    if (gain_multiplier <= 0.0).any():
        raise ValueError("yaw gain multiplier must stay positive")
    relative = estimated_target_xy[:, None, :] - pursuer_xy
    target_bearing = torch.atan2(relative[..., 1], relative[..., 0])
    yaw_error = torch.atan2(
        torch.sin(target_bearing - pursuer_yaw_rad),
        torch.cos(target_bearing - pursuer_yaw_rad),
    )
    yaw_rate = (
        proportional_gain * gain_multiplier * yaw_error
        + residual_scale[:, None] * policy_yaw_rate_rps
    ).clamp(-maximum_yaw_rate_rps, maximum_yaw_rate_rps)
    return yaw_rate, yaw_error


def bounded_constant_velocity_intercept(
    pursuer_xy: torch.Tensor,
    target_xy: torch.Tensor,
    target_velocity_xy: torch.Tensor,
    *,
    pursuer_speed_mps: float,
    minimum_horizon_s: float,
    maximum_horizon_s: float,
    horizon_samples: int,
) -> BoundedInterceptResult:
    """Find the earliest reachable CV intercept, or the least-miss fallback.

    A bounded grid is deliberate: it remains stable for equal-speed and
    faster-target cases where the usual quadratic root is singular or absent.
    The fallback exposes a positive reachability gap instead of pretending that
    an impossible constant-velocity intercept exists.
    """
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("pursuer_xy must have shape [batch, agent, 2]")
    batch = pursuer_xy.shape[0]
    if target_xy.shape != (batch, 2):
        raise ValueError("target_xy shape mismatch")
    if target_velocity_xy.shape != (batch, 2):
        raise ValueError("target_velocity_xy shape mismatch")
    if pursuer_speed_mps <= 0.0:
        raise ValueError("pursuer_speed_mps must be positive")
    if minimum_horizon_s < 0.0 or maximum_horizon_s <= minimum_horizon_s:
        raise ValueError("invalid intercept horizon bounds")
    if horizon_samples < 2:
        raise ValueError("horizon_samples must be at least two")

    horizon = torch.linspace(
        minimum_horizon_s,
        maximum_horizon_s,
        horizon_samples,
        dtype=pursuer_xy.dtype,
        device=pursuer_xy.device,
    )
    future_target = (
        target_xy[:, None, :]
        + horizon[None, :, None] * target_velocity_xy[:, None, :]
    )
    distance = (
        future_target[:, None, :, :] - pursuer_xy[:, :, None, :]
    ).norm(dim=-1)
    reachability_gap = distance - pursuer_speed_mps * horizon[None, None, :]
    reachable_candidates = reachability_gap <= 0.0
    any_reachable = reachable_candidates.any(dim=-1)
    earliest_reachable_index = reachable_candidates.to(torch.int64).argmax(dim=-1)
    least_miss_index = reachability_gap.argmin(dim=-1)
    selected_index = torch.where(
        any_reachable, earliest_reachable_index, least_miss_index
    )
    selected_time = horizon[selected_index]
    selected_point = torch.gather(
        future_target[:, None, :, :].expand(-1, pursuer_xy.shape[1], -1, -1),
        2,
        selected_index[..., None, None].expand(-1, -1, 1, 2),
    ).squeeze(2)
    selected_gap = torch.gather(
        reachability_gap, 2, selected_index[..., None]
    ).squeeze(2)
    return BoundedInterceptResult(
        time_s=selected_time,
        point_xy=selected_point,
        reachable=any_reachable,
        reachability_gap_m=selected_gap,
    )


def reachable_role_slot_prediction_horizon(
    intercept_time_s: torch.Tensor,
    intercept_reachable: torch.Tensor,
    agent_to_slot_assignment: torch.Tensor,
    blocker_agent_index: torch.Tensor,
    *,
    maximum_horizon_s: float,
) -> torch.Tensor:
    """Map only each reachable agent's intercept lead to its locked role slot.

    Before the agent-to-slot permutation becomes valid, use a conservative
    fallback that leads only slot zero and only when the selected blocker is
    itself reachable.  This avoids repeating the failed common-future-centre
    design that dragged unreachable scouts along the blocker's trajectory.
    """
    if intercept_time_s.ndim != 2:
        raise ValueError("intercept times must have shape [batch, agent]")
    if intercept_reachable.shape != intercept_time_s.shape:
        raise ValueError("intercept reachability shape mismatch")
    if agent_to_slot_assignment.shape != intercept_time_s.shape:
        raise ValueError("agent-to-slot assignment shape mismatch")
    batch, agent_count = intercept_time_s.shape
    if blocker_agent_index.shape != (batch,):
        raise ValueError("blocker agent index shape mismatch")
    if maximum_horizon_s <= 0.0:
        raise ValueError("maximum role intercept horizon must be positive")
    if ((blocker_agent_index < -1) | (blocker_agent_index >= agent_count)).any():
        raise ValueError("blocker agent index contains invalid agent")

    expected_slots = torch.arange(
        agent_count, device=intercept_time_s.device
    )[None, :]
    assignment_valid = (
        (agent_to_slot_assignment >= 0).all(dim=1)
        & (agent_to_slot_assignment < agent_count).all(dim=1)
        & (
            torch.sort(agent_to_slot_assignment, dim=1).values
            == expected_slots
        ).all(dim=1)
    )
    reachable_agent_horizon = torch.where(
        intercept_reachable,
        intercept_time_s.clamp(0.0, maximum_horizon_s),
        torch.zeros_like(intercept_time_s),
    )
    safe_assignment = agent_to_slot_assignment.clamp(0, agent_count - 1)
    role_horizon = torch.zeros_like(intercept_time_s)
    role_horizon.scatter_(1, safe_assignment, reachable_agent_horizon)

    safe_blocker = blocker_agent_index.clamp_min(0)
    blocker_reachable = torch.gather(
        intercept_reachable, 1, safe_blocker[:, None]
    ).squeeze(1) & (blocker_agent_index >= 0)
    blocker_horizon = torch.gather(
        reachable_agent_horizon, 1, safe_blocker[:, None]
    ).squeeze(1)
    initial_horizon = torch.zeros_like(intercept_time_s)
    initial_horizon[:, 0] = torch.where(
        blocker_reachable,
        blocker_horizon,
        torch.zeros_like(blocker_horizon),
    )
    return torch.where(assignment_valid[:, None], role_horizon, initial_horizon)


def release_intercept_horizon_during_contraction(
    horizon_s: torch.Tensor,
    reference_radius_m: torch.Tensor,
    *,
    fade_start_radius_m: float,
    zero_lead_radius_m: float,
) -> torch.Tensor:
    """Fade far-field intercept lead as the reference enters the live ring."""
    if fade_start_radius_m <= zero_lead_radius_m:
        raise ValueError("intercept lead fade start must exceed zero-lead radius")
    if reference_radius_m.ndim != 1:
        raise ValueError("reference radius must have shape [batch]")
    if horizon_s.shape[0] != reference_radius_m.shape[0]:
        raise ValueError("intercept horizon batch dimension mismatch")
    scale = (
        (reference_radius_m - zero_lead_radius_m)
        / (fade_start_radius_m - zero_lead_radius_m)
    ).clamp(0.0, 1.0)
    while scale.ndim < horizon_s.ndim:
        scale = scale.unsqueeze(-1)
    return horizon_s * scale


def bounded_per_agent_constant_velocity_intercept(
    pursuer_xy: torch.Tensor,
    moving_goal_xy: torch.Tensor,
    goal_velocity_xy: torch.Tensor,
    *,
    pursuer_speed_mps: float,
    minimum_horizon_s: float,
    maximum_horizon_s: float,
    horizon_samples: int,
) -> BoundedInterceptResult:
    """Apply the bounded CV intercept oracle to one moving goal per agent."""
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("pursuer_xy must have shape [batch, agent, 2]")
    if moving_goal_xy.shape != pursuer_xy.shape:
        raise ValueError("moving goals must match pursuer XY shape")
    batch, agent_count, _ = pursuer_xy.shape
    if goal_velocity_xy.shape != (batch, 2):
        raise ValueError("goal velocity shape mismatch")
    flat_result = bounded_constant_velocity_intercept(
        pursuer_xy.reshape(batch * agent_count, 1, 2),
        moving_goal_xy.reshape(batch * agent_count, 2),
        goal_velocity_xy[:, None, :]
        .expand(-1, agent_count, -1)
        .reshape(batch * agent_count, 2),
        pursuer_speed_mps=pursuer_speed_mps,
        minimum_horizon_s=minimum_horizon_s,
        maximum_horizon_s=maximum_horizon_s,
        horizon_samples=horizon_samples,
    )
    return BoundedInterceptResult(
        time_s=flat_result.time_s.reshape(batch, agent_count),
        point_xy=flat_result.point_xy.reshape(batch, agent_count, 2),
        reachable=flat_result.reachable.reshape(batch, agent_count),
        reachability_gap_m=flat_result.reachability_gap_m.reshape(
            batch, agent_count
        ),
    )


def loss_reacquisition_guidance(
    desired_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    last_seen_target_xy: torch.Tensor,
    team_visible: torch.Tensor,
    lost_steps: torch.Tensor,
    previous_active: torch.Tensor,
    previous_scout_index: torch.Tensor,
    previous_slot_angle_rad: torch.Tensor,
    *,
    enabled: bool,
    start_lost_steps: int,
    slot_radius_m: float,
    scout_blend: float,
    tracker_blend: float,
    blend_growth: float,
    scout_speed_ratio: float,
    tracker_speed_ratio: float,
    slot_arrival_m: float,
    maximum_speed_mps: float,
    lost_cap_steps: int,
    scout_minimum_selection_distance_m: float = 0.0,
    use_occlusion_flank: bool = False,
    obstacle_xy: torch.Tensor | None = None,
    obstacle_radius_m: torch.Tensor | None = None,
    previous_flank_waypoint_xy: torch.Tensor | None = None,
    previous_flank_valid: torch.Tensor | None = None,
    previous_flank_blocker_index: torch.Tensor | None = None,
    pursuer_radius_m: float = 0.35,
    flank_minimum_path_clearance_m: float = 1.5,
    flank_waypoint_margin_m: float = 0.5,
    flank_los_margin_m: float = 0.25,
    flank_blocker_margin_m: float = 0.25,
    arena_half_extent_m: float = 50.0,
    flank_boundary_clearance_m: float = 1.0,
    flank_angular_samples: int = 32,
    use_reachable_sector: bool = False,
    last_seen_target_velocity_xy: torch.Tensor | None = None,
    sector_policy_dt_s: float = 0.1,
    sector_target_speed_bound_mps: float = 4.0,
    sector_velocity_heading_threshold_mps: float = 0.25,
    sector_lookahead_s: float = 0.4,
    sector_minimum_radius_m: float = 1.5,
    sector_maximum_radius_m: float = 12.0,
    sector_initial_half_angle_rad: float = math.pi / 6.0,
    sector_half_angle_growth_rps: float = math.pi / 3.0,
    sector_phase_hold_steps: int = 4,
    sector_boundary_clearance_m: float = 1.0,
    sector_detour_margin_m: float = 0.5,
    sector_detour_maximum_extra_m: float = 4.5,
    sector_detour_radial_samples: int = 5,
    sector_detour_angular_samples: int = 32,
) -> LossReacquisitionGuidanceResult:
    """Compose Last-guided one-scout/two-tracker search velocities.

    The function deliberately receives no hidden target state.  A role is
    selected only when a continuous team-loss segment starts and remains fixed
    until visibility returns.  With ``enabled=False`` the same event state is
    advanced for matched diagnostics while the desired velocity is bitwise
    preserved.
    """
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("loss guidance tensors must have shape [batch, agent, 2]")
    batch, agent_count, _ = pursuer_xy.shape
    if agent_count < 3:
        raise ValueError("loss reacquisition requires at least three pursuers")
    if last_seen_target_xy.shape != (batch, 2):
        raise ValueError("last_seen_target_xy shape mismatch")
    for name, value in (
        ("team_visible", team_visible),
        ("lost_steps", lost_steps),
        ("previous_active", previous_active),
        ("previous_scout_index", previous_scout_index),
    ):
        if value.shape != (batch,):
            raise ValueError(f"{name} shape mismatch")
    if previous_slot_angle_rad.shape != (batch, agent_count):
        raise ValueError("previous_slot_angle_rad shape mismatch")
    if start_lost_steps < 1 or lost_cap_steps < 1:
        raise ValueError("loss step thresholds must be positive")
    if slot_radius_m <= 0.0 or slot_arrival_m <= 0.0 or maximum_speed_mps <= 0.0:
        raise ValueError("loss guidance distance and speed scales must be positive")
    for name, value in (
        ("scout_blend", scout_blend),
        ("tracker_blend", tracker_blend),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if blend_growth < 0.0:
        raise ValueError("blend_growth must be nonnegative")
    if not 0.0 <= scout_speed_ratio <= 1.0:
        raise ValueError("scout_speed_ratio must be in [0, 1]")
    if not 0.0 <= tracker_speed_ratio <= 1.0:
        raise ValueError("tracker_speed_ratio must be in [0, 1]")
    if scout_minimum_selection_distance_m < 0.0:
        raise ValueError("scout minimum selection distance must be nonnegative")
    if use_occlusion_flank and use_reachable_sector:
        raise ValueError("flank and reachable-sector search are mutually exclusive")

    active = (~team_visible) & (lost_steps >= start_lost_steps)
    activated = active & ~previous_active
    just_reacquired = team_visible & previous_active

    distance_to_center = (
        pursuer_xy - last_seen_target_xy[:, None, :]
    ).norm(dim=-1)
    scout_eligible = distance_to_center >= scout_minimum_selection_distance_m
    nearest_eligible_scout = torch.where(
        scout_eligible,
        distance_to_center,
        torch.full_like(distance_to_center, torch.inf),
    ).argmin(dim=1)
    selected_scout = torch.where(
        scout_eligible.any(dim=1),
        nearest_eligible_scout,
        distance_to_center.argmax(dim=1),
    )
    scout_index = torch.where(
        activated,
        selected_scout,
        previous_scout_index,
    )

    if previous_flank_waypoint_xy is None:
        previous_flank_waypoint_xy = last_seen_target_xy
    if previous_flank_valid is None:
        previous_flank_valid = torch.zeros(
            batch, dtype=torch.bool, device=pursuer_xy.device
        )
    if previous_flank_blocker_index is None:
        previous_flank_blocker_index = torch.full(
            (batch,), -1, dtype=torch.long, device=pursuer_xy.device
        )
    if previous_flank_waypoint_xy.shape != (batch, 2):
        raise ValueError("previous flank waypoint shape mismatch")
    if previous_flank_valid.shape != (batch,):
        raise ValueError("previous flank validity shape mismatch")
    if previous_flank_blocker_index.shape != (batch,):
        raise ValueError("previous flank blocker shape mismatch")
    if use_occlusion_flank:
        if obstacle_xy is None or obstacle_radius_m is None:
            raise ValueError("occlusion flank requires obstacle geometry")
        planned_flank = occlusion_flank_waypoint(
            pursuer_xy,
            last_seen_target_xy,
            obstacle_xy,
            obstacle_radius_m,
            scout_index,
            pursuer_radius_m=pursuer_radius_m,
            minimum_path_clearance_m=flank_minimum_path_clearance_m,
            waypoint_margin_m=flank_waypoint_margin_m,
            los_margin_m=flank_los_margin_m,
            blocker_margin_m=flank_blocker_margin_m,
            arena_half_extent_m=arena_half_extent_m,
            boundary_clearance_m=flank_boundary_clearance_m,
            angular_samples=flank_angular_samples,
        )
    else:
        planned_flank = OcclusionFlankWaypointResult(
            waypoint_xy=last_seen_target_xy,
            valid=torch.zeros(batch, dtype=torch.bool, device=pursuer_xy.device),
            blocker_index=torch.full(
                (batch,), -1, dtype=torch.long, device=pursuer_xy.device
            ),
            path_clearance_m=torch.full(
                (batch,), -1.0, dtype=pursuer_xy.dtype, device=pursuer_xy.device
            ),
            los_clearance_m=torch.full(
                (batch,), -1.0, dtype=pursuer_xy.dtype, device=pursuer_xy.device
            ),
            route_cost_m=torch.full(
                (batch,), torch.inf, dtype=pursuer_xy.dtype, device=pursuer_xy.device
            ),
        )
    if use_reachable_sector:
        if obstacle_xy is None or obstacle_radius_m is None:
            raise ValueError("reachable sector requires obstacle geometry")
        if last_seen_target_velocity_xy is None:
            raise ValueError("reachable sector requires last visible target velocity")
        planned_sector = reachable_sector_waypoint(
            pursuer_xy,
            last_seen_target_xy,
            last_seen_target_velocity_xy,
            obstacle_xy,
            obstacle_radius_m,
            scout_index,
            lost_steps,
            policy_dt_s=sector_policy_dt_s,
            target_speed_bound_mps=sector_target_speed_bound_mps,
            velocity_heading_threshold_mps=(
                sector_velocity_heading_threshold_mps
            ),
            lookahead_s=sector_lookahead_s,
            minimum_radius_m=sector_minimum_radius_m,
            maximum_radius_m=sector_maximum_radius_m,
            initial_half_angle_rad=sector_initial_half_angle_rad,
            half_angle_growth_rps=sector_half_angle_growth_rps,
            phase_hold_steps=sector_phase_hold_steps,
            pursuer_radius_m=pursuer_radius_m,
            minimum_path_clearance_m=flank_minimum_path_clearance_m,
            arena_half_extent_m=arena_half_extent_m,
            boundary_clearance_m=sector_boundary_clearance_m,
            detour_margin_m=sector_detour_margin_m,
            detour_maximum_extra_m=sector_detour_maximum_extra_m,
            detour_radial_samples=sector_detour_radial_samples,
            detour_angular_samples=sector_detour_angular_samples,
        )
    else:
        planned_sector = ReachableSectorWaypointResult(
            waypoint_xy=last_seen_target_xy,
            valid=torch.zeros(batch, dtype=torch.bool, device=pursuer_xy.device),
            phase_index=torch.full(
                (batch,), -1, dtype=torch.long, device=pursuer_xy.device
            ),
            half_angle_rad=torch.zeros(
                batch, dtype=pursuer_xy.dtype, device=pursuer_xy.device
            ),
            radius_m=torch.zeros(
                batch, dtype=pursuer_xy.dtype, device=pursuer_xy.device
            ),
            path_clearance_m=torch.full(
                (batch,), -1.0, dtype=pursuer_xy.dtype, device=pursuer_xy.device
            ),
            route_cost_m=torch.full(
                (batch,), torch.inf, dtype=pursuer_xy.dtype, device=pursuer_xy.device
            ),
            route_complete=torch.zeros(
                batch, dtype=torch.bool, device=pursuer_xy.device
            ),
            any_path_candidate=torch.zeros(
                batch, dtype=torch.bool, device=pursuer_xy.device
            ),
            any_boundary_candidate=torch.zeros(
                batch, dtype=torch.bool, device=pursuer_xy.device
            ),
        )
    flank_waypoint_valid = torch.where(
        activated, planned_flank.valid, previous_flank_valid
    )
    flank_waypoint_xy = torch.where(
        activated[:, None],
        planned_flank.waypoint_xy,
        previous_flank_waypoint_xy,
    )
    flank_blocker_index = torch.where(
        activated,
        planned_flank.blocker_index,
        previous_flank_blocker_index,
    )

    bearings = torch.atan2(
        pursuer_xy[..., 1] - last_seen_target_xy[:, None, 1],
        pursuer_xy[..., 0] - last_seen_target_xy[:, None, 0],
    )
    order = torch.argsort(bearings, dim=1)
    sorted_bearings = torch.gather(bearings, 1, order)
    offsets = (
        2.0
        * math.pi
        * torch.arange(agent_count, dtype=pursuer_xy.dtype, device=pursuer_xy.device)
        / agent_count
    )
    phase_terms = sorted_bearings - offsets[None, :]
    phase = torch.atan2(
        torch.sin(phase_terms).mean(dim=1),
        torch.cos(phase_terms).mean(dim=1),
    )
    sorted_slot_angles = phase[:, None] + offsets[None, :]
    candidate_slot_angles = torch.empty_like(sorted_slot_angles)
    candidate_slot_angles.scatter_(1, order, sorted_slot_angles)
    slot_angle_rad = torch.where(
        activated[:, None],
        candidate_slot_angles,
        previous_slot_angle_rad,
    )

    slot_unit = torch.stack(
        (torch.cos(slot_angle_rad), torch.sin(slot_angle_rad)), dim=-1
    )
    tracker_goal = last_seen_target_xy[:, None, :] + slot_radius_m * slot_unit
    safe_scout_index = scout_index.clamp(0, agent_count - 1)
    scout_mask = (
        torch.arange(agent_count, device=pursuer_xy.device)[None, :]
        == safe_scout_index[:, None]
    )
    flank_guidance_active = (
        active & bool(enabled) & bool(use_occlusion_flank) & flank_waypoint_valid
    )
    sector_guidance_active = (
        active & bool(enabled) & bool(use_reachable_sector) & planned_sector.valid
    )
    scout_goal_xy = torch.where(
        flank_guidance_active[:, None], flank_waypoint_xy, last_seen_target_xy
    )
    scout_goal_xy = torch.where(
        sector_guidance_active[:, None],
        planned_sector.waypoint_xy,
        scout_goal_xy,
    )
    goal_xy = torch.where(
        scout_mask[..., None],
        scout_goal_xy[:, None, :],
        tracker_goal,
    )
    delta = goal_xy - pursuer_xy
    distance_to_goal = delta.norm(dim=-1)
    direction = delta / distance_to_goal[..., None].clamp_min(1.0e-9)
    speed_ratio = torch.where(
        scout_mask,
        torch.full_like(distance_to_goal, scout_speed_ratio),
        torch.full_like(distance_to_goal, tracker_speed_ratio),
    )
    guide_speed = maximum_speed_mps * speed_ratio
    guide_speed = torch.where(
        distance_to_goal < slot_arrival_m,
        torch.zeros_like(guide_speed),
        guide_speed,
    )
    guide_velocity = direction * guide_speed[..., None]
    loss_fraction = (
        lost_steps.to(pursuer_xy.dtype).clamp(max=lost_cap_steps)
        / float(lost_cap_steps)
    )
    blend = torch.where(
        scout_mask,
        torch.full_like(distance_to_goal, scout_blend),
        torch.full_like(distance_to_goal, tracker_blend),
    )
    blend = (blend + blend_growth * loss_fraction[:, None]).clamp(max=1.0)
    composed = (1.0 - blend[..., None]) * desired_velocity_xy + (
        blend[..., None] * guide_velocity
    )
    composed_norm = composed.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    composed = composed * (maximum_speed_mps / composed_norm).clamp(max=1.0)
    guidance_active = active & bool(enabled)
    flank_fallback_active = (
        guidance_active & bool(use_occlusion_flank) & ~flank_waypoint_valid
    )
    sector_fallback_active = (
        guidance_active & bool(use_reachable_sector) & ~planned_sector.valid
    )
    output = torch.where(
        guidance_active[:, None, None], composed, desired_velocity_xy
    )
    return LossReacquisitionGuidanceResult(
        velocity_xy=output,
        active=active,
        guidance_active=guidance_active,
        activated=activated,
        just_reacquired=just_reacquired,
        scout_index=scout_index,
        slot_angle_rad=slot_angle_rad,
        goal_xy=goal_xy,
        distance_to_goal_m=distance_to_goal,
        flank_waypoint_xy=flank_waypoint_xy,
        flank_waypoint_valid=flank_waypoint_valid,
        flank_guidance_active=flank_guidance_active,
        flank_fallback_active=flank_fallback_active,
        flank_blocker_index=flank_blocker_index,
        flank_selected_path_clearance_m=torch.where(
            activated & planned_flank.valid,
            planned_flank.path_clearance_m,
            torch.full_like(planned_flank.path_clearance_m, -1.0),
        ),
        flank_selected_los_clearance_m=torch.where(
            activated & planned_flank.valid,
            planned_flank.los_clearance_m,
            torch.full_like(planned_flank.los_clearance_m, -1.0),
        ),
        sector_waypoint_xy=planned_sector.waypoint_xy,
        sector_waypoint_valid=planned_sector.valid,
        sector_guidance_active=sector_guidance_active,
        sector_fallback_active=sector_fallback_active,
        sector_phase_index=planned_sector.phase_index,
        sector_half_angle_rad=planned_sector.half_angle_rad,
        sector_radius_m=planned_sector.radius_m,
        sector_selected_path_clearance_m=torch.where(
            active & planned_sector.valid,
            planned_sector.path_clearance_m,
            torch.full_like(planned_sector.path_clearance_m, -1.0),
        ),
        sector_any_path_candidate=planned_sector.any_path_candidate,
        sector_any_boundary_candidate=planned_sector.any_boundary_candidate,
    )


def compose_feedforward_priority_velocity(
    feedforward_velocity_xy: torch.Tensor,
    correction_velocity_xy: torch.Tensor,
    maximum_speed_mps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a correction into the speed ball without uniformly shrinking feedforward.

    Returns the composed velocity and the scalar correction scale.  When the
    unconstrained sum is too fast, the largest alpha in [0, 1] satisfying
    ``||feedforward + alpha * correction|| <= maximum_speed_mps`` is obtained
    from the positive quadratic root.  This retains common target translation
    before spending the remaining authority on ring contraction/separation.
    """
    if feedforward_velocity_xy.shape != correction_velocity_xy.shape:
        raise ValueError("feedforward and correction velocities must match")
    if feedforward_velocity_xy.shape[-1] != 2:
        raise ValueError("priority velocity composition requires XY vectors")
    if maximum_speed_mps <= 0.0:
        raise ValueError("maximum speed must be positive")
    feedforward_norm = feedforward_velocity_xy.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-9)
    retained_feedforward = feedforward_velocity_xy * (
        maximum_speed_mps / feedforward_norm
    ).clamp(max=1.0)
    unlimited = retained_feedforward + correction_velocity_xy
    within_limit = unlimited.norm(dim=-1) <= maximum_speed_mps
    quadratic_a = correction_velocity_xy.square().sum(dim=-1)
    quadratic_b = 2.0 * (
        retained_feedforward * correction_velocity_xy
    ).sum(dim=-1)
    quadratic_c = (
        retained_feedforward.square().sum(dim=-1)
        - maximum_speed_mps * maximum_speed_mps
    )
    discriminant = (
        quadratic_b.square() - 4.0 * quadratic_a * quadratic_c
    ).clamp_min(0.0)
    positive_root = (
        -quadratic_b + torch.sqrt(discriminant)
    ) / (2.0 * quadratic_a.clamp_min(1.0e-9))
    correction_scale = positive_root.clamp(0.0, 1.0)
    correction_scale = torch.where(
        within_limit | (quadratic_a <= 1.0e-9),
        torch.ones_like(correction_scale),
        correction_scale,
    )
    velocity = retained_feedforward + correction_scale[..., None] * (
        correction_velocity_xy
    )
    return velocity, correction_scale


def compose_feedforward_radial_priority_velocity(
    feedforward_velocity_xy: torch.Tensor,
    radial_correction_velocity_xy: torch.Tensor,
    secondary_correction_velocity_xy: torch.Tensor,
    radial_priority_active: torch.Tensor,
    maximum_speed_mps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Spend the equal-speed correction budget on radius before phase.

    The default branch is exactly the historical joint correction projection.
    For selected pre-entry states, target feedforward is retained first, the
    radial correction receives the remaining speed-ball authority second, and
    tangential/secondary correction receives only the final remainder.  This
    changes command allocation, not the physical speed limit.
    """
    if not (
        feedforward_velocity_xy.shape
        == radial_correction_velocity_xy.shape
        == secondary_correction_velocity_xy.shape
    ):
        raise ValueError("feedforward and entry corrections must match")
    if radial_priority_active.shape != feedforward_velocity_xy.shape[:-1]:
        raise ValueError("radial_priority_active shape mismatch")
    if radial_priority_active.dtype != torch.bool:
        raise ValueError("radial_priority_active must be boolean")
    historical_velocity, historical_scale = compose_feedforward_priority_velocity(
        feedforward_velocity_xy,
        radial_correction_velocity_xy + secondary_correction_velocity_xy,
        maximum_speed_mps,
    )
    radial_first_velocity, radial_scale = compose_feedforward_priority_velocity(
        feedforward_velocity_xy,
        radial_correction_velocity_xy,
        maximum_speed_mps,
    )
    prioritized_velocity, secondary_scale = compose_feedforward_priority_velocity(
        radial_first_velocity,
        secondary_correction_velocity_xy,
        maximum_speed_mps,
    )
    active = radial_priority_active[..., None]
    return (
        torch.where(active, prioritized_velocity, historical_velocity),
        torch.where(radial_priority_active, radial_scale, historical_scale),
        torch.where(radial_priority_active, secondary_scale, historical_scale),
    )


def moving_target_slot_guidance(
    desired_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    estimated_target_xy: torch.Tensor,
    target_velocity_xy: torch.Tensor,
    reference_radius_m: torch.Tensor,
    active: torch.Tensor,
    previous_assignment: torch.Tensor,
    previous_anchor_heading_rad: torch.Tensor,
    previous_anchor_valid: torch.Tensor,
    *,
    pursuer_velocity_xy: torch.Tensor | None = None,
    position_gain: float,
    policy_residual_scale: float | torch.Tensor,
    target_feedforward_scale: float | torch.Tensor,
    maximum_speed_mps: float,
    minimum_target_speed_mps: float,
    maximum_anchor_turn_rate_rps: float,
    policy_dt_s: float,
    assignment_hysteresis_m: float,
    target_prediction_horizon_s: float | torch.Tensor = 0.0,
    blocker_agent_index: torch.Tensor | None = None,
    blocker_role_lock_weight_m: float = 0.0,
    prepare_assignment_when_inactive: bool = False,
    obstacle_xy: torch.Tensor | None = None,
    obstacle_radius_m: torch.Tensor | None = None,
    pursuer_radius_m: float = 0.5,
    assignment_path_clearance_m: float = 0.0,
    assignment_obstacle_penalty: float = 0.0,
    cyclic_order_preserving_assignment: bool = False,
    lock_valid_assignment: bool = False,
    phase_separation_control_enabled: bool = False,
    phase_gain: float = 0.8,
    phase_angular_velocity_damping_gain_s: float = 0.0,
    phase_angular_velocity_damping_active: torch.Tensor | None = None,
    radial_gain: float = 1.0,
    radial_velocity_damping_gain: float = 0.0,
    maximum_radial_speed_mps: float | None = None,
    phase_scheduled_radial_speed_budget: bool = False,
    maximum_tangential_speed_mps: float = 3.0,
    phase_activation_margin_m: float = 2.0,
    feedforward_priority_projection_enabled: bool = False,
    radial_correction_priority_active: torch.Tensor | None = None,
    far_slot_intercept_control_enabled: bool = False,
    far_slot_intercept_minimum_horizon_s: float = 0.25,
    far_slot_intercept_maximum_horizon_s: float = 8.0,
    far_slot_intercept_horizon_samples: int = 33,
    blocker_phase_anchor_control_enabled: bool = False,
    blocker_phase_anchor_gain: float = 0.3,
    blocker_phase_anchor_maximum_speed_mps: float = 0.8,
    gain_residual_action: torch.Tensor | None = None,
    gain_residual_control_enabled: bool = False,
    radial_gain_residual_scale: float = 0.25,
    phase_gain_residual_scale: float = 0.25,
    radial_velocity_residual_action: torch.Tensor | None = None,
    radial_velocity_residual_control_enabled: bool = False,
    radial_velocity_residual_scale_mps: float = 0.75,
    tangential_velocity_residual_action: torch.Tensor | None = None,
    tangential_velocity_residual_control_enabled: bool = False,
    tangential_velocity_residual_scale_mps: float = 1.0,
    region_minimum_radius_m: float | None = None,
    region_maximum_gap_rad: float | None = None,
    observer_reference_heading_rad: torch.Tensor | None = None,
    observer_reference_valid: torch.Tensor | None = None,
    observer_phase_tolerance_rad: float = math.pi / 12,
    diagnostic_sink: dict | None = None,
) -> MovingTargetSlotGuidanceResult:
    """Guide agents to stable target-centred angular slots.

    Slots rotate with a rate-limited target-velocity heading.  A minimum-cost
    assignment prevents unnecessary path crossings, while hysteresis preserves
    the previous assignment unless a new handoff materially shortens the joint
    path.  The learned action remains as a residual and all resulting commands
    still pass through the downstream predictive safety projections.
    """
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("slot guidance tensors must have shape [batch, agent, 2]")
    batch, agent_count, _ = pursuer_xy.shape
    if estimated_target_xy.shape != (batch, 2):
        raise ValueError("estimated_target_xy shape mismatch")
    if target_velocity_xy.shape != (batch, 2):
        raise ValueError("target_velocity_xy shape mismatch")
    if pursuer_velocity_xy is not None and pursuer_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("pursuer_velocity_xy shape mismatch")
    if reference_radius_m.shape != (batch,):
        raise ValueError("reference_radius_m shape mismatch")
    if active.shape != (batch,):
        raise ValueError("active shape mismatch")
    if previous_assignment.shape != (batch, agent_count):
        raise ValueError("previous_assignment shape mismatch")
    if previous_anchor_heading_rad.shape != (batch,):
        raise ValueError("previous_anchor_heading_rad shape mismatch")
    if previous_anchor_valid.shape != (batch,):
        raise ValueError("previous_anchor_valid shape mismatch")
    if (reference_radius_m <= 0.0).any():
        raise ValueError("reference radius must be positive")
    if position_gain <= 0.0 or maximum_speed_mps <= 0.0:
        raise ValueError("slot position gain and speed limit must be positive")
    residual_scale = torch.as_tensor(
        policy_residual_scale,
        dtype=pursuer_xy.dtype,
        device=pursuer_xy.device,
    )
    if residual_scale.ndim == 0:
        residual_scale = residual_scale.expand(batch)
    elif residual_scale.shape != (batch,):
        raise ValueError("policy_residual_scale must be scalar or shape [batch]")
    if ((residual_scale < 0.0) | (residual_scale > 1.0)).any():
        raise ValueError("policy_residual_scale must be in [0, 1]")
    feedforward_scale = torch.as_tensor(
        target_feedforward_scale,
        dtype=pursuer_xy.dtype,
        device=pursuer_xy.device,
    )
    if feedforward_scale.ndim == 0:
        feedforward_scale = feedforward_scale.expand(batch)
    elif feedforward_scale.shape != (batch,):
        raise ValueError("target feedforward scale must be scalar or shape [batch]")
    if (feedforward_scale < 0.0).any() or minimum_target_speed_mps < 0.0:
        raise ValueError("target feedforward parameters must be nonnegative")
    if maximum_anchor_turn_rate_rps <= 0.0 or policy_dt_s <= 0.0:
        raise ValueError("anchor turn rate and policy dt must be positive")
    if assignment_hysteresis_m < 0.0:
        raise ValueError("assignment hysteresis must be nonnegative")
    prediction_horizon = torch.as_tensor(
        target_prediction_horizon_s,
        dtype=pursuer_xy.dtype,
        device=pursuer_xy.device,
    )
    if prediction_horizon.ndim == 0:
        prediction_horizon = prediction_horizon.expand(batch, agent_count)
    elif prediction_horizon.shape == (batch,):
        prediction_horizon = prediction_horizon[:, None].expand(-1, agent_count)
    elif prediction_horizon.shape != (batch, agent_count):
        raise ValueError("target prediction horizon shape mismatch")
    if (prediction_horizon < 0.0).any():
        raise ValueError("target prediction horizon must be nonnegative")
    if blocker_role_lock_weight_m < 0.0:
        raise ValueError("blocker role lock weight must be nonnegative")
    if pursuer_radius_m <= 0.0 or assignment_path_clearance_m < 0.0:
        raise ValueError("assignment clearance scales are invalid")
    if assignment_obstacle_penalty < 0.0:
        raise ValueError("assignment obstacle penalty must be nonnegative")
    if phase_separation_control_enabled and (
        phase_gain <= 0.0
        or phase_angular_velocity_damping_gain_s < 0.0
        or radial_gain <= 0.0
        or radial_velocity_damping_gain < 0.0
        or maximum_tangential_speed_mps <= 0.0
        or phase_activation_margin_m < 0.0
    ):
        raise ValueError("phase/radial control gains and speed must be positive")
    if radial_velocity_damping_gain > 0.0 and pursuer_velocity_xy is None:
        raise ValueError("radial velocity damping requires pursuer velocity")
    if phase_angular_velocity_damping_gain_s > 0.0 and pursuer_velocity_xy is None:
        raise ValueError("phase angular-velocity damping requires pursuer velocity")
    if phase_angular_velocity_damping_active is not None:
        if phase_angular_velocity_damping_active.shape != (batch,):
            raise ValueError(
                "phase_angular_velocity_damping_active must have shape [batch]"
            )
        if phase_angular_velocity_damping_active.dtype != torch.bool:
            raise ValueError("phase_angular_velocity_damping_active must be boolean")
    if far_slot_intercept_control_enabled:
        if not phase_separation_control_enabled:
            raise ValueError("far slot intercept requires phase separation control")
        if (
            far_slot_intercept_minimum_horizon_s < 0.0
            or far_slot_intercept_maximum_horizon_s
            <= far_slot_intercept_minimum_horizon_s
            or far_slot_intercept_horizon_samples < 2
        ):
            raise ValueError("invalid far slot intercept horizon")
    if feedforward_priority_projection_enabled and not phase_separation_control_enabled:
        raise ValueError("feedforward-priority projection requires phase control")
    if radial_correction_priority_active is not None:
        if radial_correction_priority_active.shape != (batch,):
            raise ValueError(
                "radial_correction_priority_active must have shape [batch]"
            )
        if radial_correction_priority_active.dtype != torch.bool:
            raise ValueError("radial_correction_priority_active must be boolean")
        if not feedforward_priority_projection_enabled:
            raise ValueError("radial priority requires feedforward priority")
    if maximum_radial_speed_mps is not None and maximum_radial_speed_mps <= 0.0:
        raise ValueError("maximum radial speed must be positive when provided")
    if phase_scheduled_radial_speed_budget:
        if maximum_radial_speed_mps is None:
            raise ValueError("phase-scheduled radial budget requires a radial cap")
        if maximum_radial_speed_mps > maximum_speed_mps:
            raise ValueError("scheduled radial cap cannot exceed total speed")
    if blocker_phase_anchor_control_enabled:
        if not phase_separation_control_enabled:
            raise ValueError("blocker phase anchor requires phase separation control")
        if blocker_agent_index is None:
            raise ValueError("blocker phase anchor requires blocker_agent_index")
        if (
            blocker_phase_anchor_gain <= 0.0
            or blocker_phase_anchor_maximum_speed_mps <= 0.0
        ):
            raise ValueError("blocker phase anchor gain and speed must be positive")
    if gain_residual_control_enabled:
        if not phase_separation_control_enabled:
            raise ValueError("gain residual control requires phase separation control")
        if gain_residual_action is None or gain_residual_action.shape != pursuer_xy.shape:
            raise ValueError("gain_residual_action must match pursuer XY shape")
        if not 0.0 <= radial_gain_residual_scale < 1.0:
            raise ValueError("radial gain residual scale must be in [0, 1)")
        if not 0.0 <= phase_gain_residual_scale < 1.0:
            raise ValueError("phase gain residual scale must be in [0, 1)")
    if radial_velocity_residual_control_enabled:
        if not gain_residual_control_enabled:
            raise ValueError("radial velocity residual requires gain residual control")
        if (
            radial_velocity_residual_action is None
            or radial_velocity_residual_action.shape != pursuer_xy.shape[:2]
        ):
            raise ValueError(
                "radial_velocity_residual_action must have shape [batch, agent]"
            )
        if radial_velocity_residual_scale_mps <= 0.0:
            raise ValueError("radial velocity residual scale must be positive")
    if tangential_velocity_residual_control_enabled:
        if not gain_residual_control_enabled:
            raise ValueError("tangential velocity residual requires gain residual control")
        if (
            tangential_velocity_residual_action is None
            or tangential_velocity_residual_action.shape != pursuer_xy.shape[:2]
        ):
            raise ValueError(
                "tangential_velocity_residual_action must have shape [batch, agent]"
            )
        if tangential_velocity_residual_scale_mps <= 0.0:
            raise ValueError("tangential velocity residual scale must be positive")
    if (obstacle_xy is None) != (obstacle_radius_m is None):
        raise ValueError("obstacle positions and radii must be provided together")
    if obstacle_xy is not None:
        if obstacle_xy.ndim != 3 or obstacle_xy.shape[0] != batch:
            raise ValueError("obstacle_xy shape mismatch")
        if obstacle_xy.shape[-1] != 2:
            raise ValueError("obstacle_xy must end in XY")
        if obstacle_radius_m.shape != obstacle_xy.shape[:2]:
            raise ValueError("obstacle_radius_m shape mismatch")
        if (obstacle_radius_m <= 0.0).any():
            raise ValueError("obstacle radii must be positive")
    if blocker_agent_index is not None:
        if blocker_agent_index.shape != (batch,):
            raise ValueError("blocker_agent_index shape mismatch")
        if ((blocker_agent_index < -1) | (blocker_agent_index >= agent_count)).any():
            raise ValueError("blocker_agent_index contains invalid agent")

    target_speed = target_velocity_xy.norm(dim=-1)
    measured_heading = torch.atan2(
        target_velocity_xy[:, 1], target_velocity_xy[:, 0]
    )
    heading_available = target_speed >= minimum_target_speed_mps
    if observer_reference_heading_rad is not None:
        if (observer_reference_heading_rad.shape != (batch,)
                or observer_reference_valid is None
                or observer_reference_valid.shape != (batch,)
                or observer_reference_valid.dtype != torch.bool
                or blocker_agent_index is None or agent_count != 3
                or not 0 <= observer_phase_tolerance_rad < math.pi / 6
                or not phase_separation_control_enabled):
            raise ValueError("invalid dual-observer formation reference")
        measured_heading = torch.where(
            observer_reference_valid, observer_reference_heading_rad, measured_heading)
        heading_available = heading_available | observer_reference_valid
    seeded_heading = torch.where(
        previous_anchor_valid,
        previous_anchor_heading_rad,
        measured_heading,
    )
    heading_delta = torch.atan2(
        torch.sin(measured_heading - seeded_heading),
        torch.cos(measured_heading - seeded_heading),
    ).clamp(
        -maximum_anchor_turn_rate_rps * policy_dt_s,
        maximum_anchor_turn_rate_rps * policy_dt_s,
    )
    updated_heading = torch.where(
        heading_available,
        seeded_heading + heading_delta,
        torch.where(
            previous_anchor_valid,
            previous_anchor_heading_rad,
            torch.zeros_like(previous_anchor_heading_rad),
        ),
    )
    updated_heading = torch.atan2(torch.sin(updated_heading), torch.cos(updated_heading))
    updated_anchor_valid = previous_anchor_valid | heading_available

    slot_offsets = (
        2.0
        * math.pi
        * torch.arange(agent_count, device=pursuer_xy.device, dtype=pursuer_xy.dtype)
        / agent_count
    )
    slot_angles = updated_heading[:, None] + slot_offsets[None, :]
    slot_unit = torch.stack((torch.cos(slot_angles), torch.sin(slot_angles)), dim=-1)
    predicted_target_xy = (
        estimated_target_xy[:, None, :]
        + prediction_horizon[:, :, None] * target_velocity_xy[:, None, :]
    )
    slot_xy = (
        predicted_target_xy
        + reference_radius_m[:, None, None] * slot_unit
    )

    if cyclic_order_preserving_assignment:
        relative_angle = torch.remainder(
            torch.atan2(
                pursuer_xy[..., 1] - estimated_target_xy[:, None, 1],
                pursuer_xy[..., 0] - estimated_target_xy[:, None, 0],
            ),
            2.0 * math.pi,
        )
        agent_order = torch.argsort(relative_angle, dim=1)
        shifts = torch.arange(agent_count, device=pursuer_xy.device)
        ordered_slots = (
            torch.arange(agent_count, device=pursuer_xy.device)[None, :]
            + shifts[:, None]
        ) % agent_count
        assignment_candidates = torch.empty(
            batch,
            agent_count,
            agent_count,
            dtype=torch.long,
            device=pursuer_xy.device,
        )
        assignment_candidates.scatter_(
            2,
            agent_order[:, None, :].expand(-1, agent_count, -1),
            ordered_slots[None, :, :].expand(batch, -1, -1),
        )
    else:
        assignment_candidates = torch.tensor(
            list(permutations(range(agent_count))),
            dtype=torch.long,
            device=pursuer_xy.device,
        )[None, :, :].expand(batch, -1, -1)
    candidate_count = assignment_candidates.shape[1]
    candidate_slot_xy = torch.gather(
        slot_xy[:, None, :, :].expand(-1, candidate_count, -1, -1),
        2,
        assignment_candidates[..., None].expand(-1, -1, -1, 2),
    )
    candidate_cost = (
        candidate_slot_xy - pursuer_xy[:, None, :, :]
    ).norm(dim=-1).sum(dim=-1)
    use_obstacle_cost = (
        obstacle_xy is not None and assignment_obstacle_penalty > 0.0
    )
    if use_obstacle_cost:
        candidate_segment = candidate_slot_xy - pursuer_xy[:, None, :, :]
        obstacle_from_start = (
            obstacle_xy[:, None, None, :, :]
            - pursuer_xy[:, None, :, None, :]
        )
        candidate_projection = (
            (obstacle_from_start * candidate_segment[:, :, :, None, :])
            .sum(dim=-1)
            / candidate_segment.pow(2).sum(dim=-1)[:, :, :, None].clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)
        candidate_closest = (
            pursuer_xy[:, None, :, None, :]
            + candidate_projection[..., None]
            * candidate_segment[:, :, :, None, :]
        )
        candidate_center_distance = (
            candidate_closest - obstacle_xy[:, None, None, :, :]
        ).norm(dim=-1)
        required_center_distance = (
            obstacle_radius_m[:, None, None, :]
            + pursuer_radius_m
            + assignment_path_clearance_m
        )
        candidate_path_violation = (
            required_center_distance - candidate_center_distance
        ).clamp_min(0.0)
        candidate_cost = candidate_cost + assignment_obstacle_penalty * (
            candidate_path_violation.sum(dim=(-1, -2))
        )
    if blocker_agent_index is not None and blocker_role_lock_weight_m > 0.0:
        blocker_valid = blocker_agent_index >= 0
        safe_blocker_agent_index = blocker_agent_index.clamp_min(0)
        blocker_slot = torch.gather(
            assignment_candidates,
            2,
            safe_blocker_agent_index[:, None, None].expand(
                -1, candidate_count, 1
            ),
        ).squeeze(2)
        candidate_cost = candidate_cost + blocker_role_lock_weight_m * (
            blocker_valid[:, None] & (blocker_slot != 0)
        ).to(candidate_cost.dtype)
    best_candidate_index = candidate_cost.argmin(dim=1)
    if observer_reference_heading_rad is not None:
        designated_slot = torch.gather(
            assignment_candidates, 2,
            blocker_agent_index.clamp(0, agent_count - 1)[:, None, None].expand(-1, candidate_count, 1)
        ).squeeze(2)
        candidate_cost = torch.where(
            observer_reference_valid[:, None] & (designated_slot != 0),
            torch.inf, candidate_cost)
        best_candidate_index = candidate_cost.argmin(dim=1)
    best_assignment = torch.gather(
        assignment_candidates,
        1,
        best_candidate_index[:, None, None].expand(-1, 1, agent_count),
    ).squeeze(1)
    best_cost = candidate_cost.gather(1, best_candidate_index[:, None]).squeeze(1)

    expected_slots = torch.arange(agent_count, device=pursuer_xy.device)[None, :]
    previous_assignment_valid = (
        (previous_assignment >= 0).all(dim=1)
        & (previous_assignment < agent_count).all(dim=1)
        & (
            torch.sort(previous_assignment, dim=1).values
            == expected_slots
        ).all(dim=1)
    )
    safe_previous_assignment = previous_assignment.clamp(0, agent_count - 1)
    previous_slot_xy = torch.gather(
        slot_xy,
        1,
        safe_previous_assignment[..., None].expand(-1, -1, 2),
    )
    previous_cost = (previous_slot_xy - pursuer_xy).norm(dim=-1).sum(dim=1)
    if use_obstacle_cost:
        previous_segment = previous_slot_xy - pursuer_xy
        previous_obstacle_from_start = (
            obstacle_xy[:, None, :, :] - pursuer_xy[:, :, None, :]
        )
        previous_projection = (
            (previous_obstacle_from_start * previous_segment[:, :, None, :])
            .sum(dim=-1)
            / previous_segment.pow(2).sum(dim=-1)[:, :, None].clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)
        previous_closest = (
            pursuer_xy[:, :, None, :]
            + previous_projection[..., None] * previous_segment[:, :, None, :]
        )
        previous_center_distance = (
            previous_closest - obstacle_xy[:, None, :, :]
        ).norm(dim=-1)
        required_center_distance = (
            obstacle_radius_m[:, None, :]
            + pursuer_radius_m
            + assignment_path_clearance_m
        )
        previous_path_violation = (
            required_center_distance - previous_center_distance
        ).clamp_min(0.0)
        previous_cost = previous_cost + assignment_obstacle_penalty * (
            previous_path_violation.sum(dim=(-1, -2))
        )
    if blocker_agent_index is not None and blocker_role_lock_weight_m > 0.0:
        blocker_valid = blocker_agent_index >= 0
        safe_blocker_agent_index = blocker_agent_index.clamp_min(0)
        previous_blocker_slot = torch.gather(
            safe_previous_assignment, 1, safe_blocker_agent_index[:, None]
        ).squeeze(1)
        previous_cost = previous_cost + blocker_role_lock_weight_m * (
            blocker_valid & (previous_blocker_slot != 0)
        ).to(previous_cost.dtype)
    assignment_gate = active | bool(prepare_assignment_when_inactive)
    if lock_valid_assignment:
        should_switch = assignment_gate & ~previous_assignment_valid
    else:
        should_switch = assignment_gate & (
            ~previous_assignment_valid
            | (best_cost + assignment_hysteresis_m < previous_cost)
        )
    assignment = torch.where(
        should_switch[:, None], best_assignment, previous_assignment
    )
    if observer_reference_heading_rad is not None:
        role_change = (assignment_gate & observer_reference_valid
                       & (assignment != best_assignment).any(dim=1))
        should_switch = should_switch | role_change
        assignment = torch.where(role_change[:, None], best_assignment, assignment)
    safe_assignment = assignment.clamp(0, agent_count - 1)
    assigned_slot_xy = torch.gather(
        slot_xy,
        1,
        safe_assignment[..., None].expand(-1, -1, 2),
    )
    slot_error = assigned_slot_xy - pursuer_xy
    mean_slot_error = slot_error.norm(dim=-1).mean(dim=1)
    zero_diagnostic = torch.zeros(batch, dtype=pursuer_xy.dtype, device=pursuer_xy.device)
    policy_residual_norm_mps = zero_diagnostic
    policy_residual_radial_support_mps = zero_diagnostic
    policy_residual_radial_opposition_mps = zero_diagnostic
    policy_residual_phase_support_mps = zero_diagnostic
    policy_residual_phase_opposition_mps = zero_diagnostic
    analytic_tangential_speed = None
    if phase_separation_control_enabled:
        assigned_target_xy = torch.gather(
            predicted_target_xy,
            1,
            safe_assignment[..., None].expand(-1, -1, 2),
        )
        relative_to_assigned_target = pursuer_xy - assigned_target_xy
        current_radius = relative_to_assigned_target.norm(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        radial_unit = relative_to_assigned_target / current_radius
        # Equalize adjacent angular gaps around the current target rather than
        # chase an absolute phase tied to the evader's rapidly turning heading.
        # This leaves the collective ring phase free while still converging to
        # 2*pi/N separation and avoids wasting equal-speed authority on global
        # formation rotation.
        relative_to_current_target = (
            pursuer_xy - estimated_target_xy[:, None, :]
        )
        current_angle = torch.remainder(
            torch.atan2(
                relative_to_current_target[..., 1],
                relative_to_current_target[..., 0],
            ),
            2.0 * math.pi,
        )
        angular_order = torch.argsort(current_angle, dim=1)
        sorted_angle = torch.gather(current_angle, 1, angular_order)
        forward_gap = torch.cat(
            (
                sorted_angle[:, 1:] - sorted_angle[:, :-1],
                2.0 * math.pi
                - sorted_angle[:, -1:]
                + sorted_angle[:, :1],
            ),
            dim=1,
        )
        backward_gap = torch.roll(forward_gap, shifts=1, dims=1)
        phase_error = forward_gap - backward_gap
        if region_maximum_gap_rad is not None:
            if not 0 < region_maximum_gap_rad < math.pi:
                raise ValueError("invalid region maximum angular gap")
            phase_error = (
                (forward_gap - region_maximum_gap_rad).clamp_min(0)
                - (backward_gap - region_maximum_gap_rad).clamp_min(0)
            )
        ordered_tangential_speed = (
            0.5
            * phase_gain
            * reference_radius_m[:, None]
            * phase_error
        )
        if observer_reference_heading_rad is not None:
            assigned_angle = torch.gather(slot_angles, 1, safe_assignment)
            slot_phase_error = torch.atan2(
                torch.sin(assigned_angle - current_angle),
                torch.cos(assigned_angle - current_angle))
            soft_error = slot_phase_error.sign() * (
                slot_phase_error.abs() - observer_phase_tolerance_rad).clamp_min(0)
            observer_speed = phase_gain * reference_radius_m[:, None] * soft_error
            ordered_tangential_speed = torch.where(
                observer_reference_valid[:, None],
                torch.gather(observer_speed, 1, angular_order), ordered_tangential_speed)
        if phase_angular_velocity_damping_gain_s > 0.0:
            relative_velocity_to_current_target = (
                pursuer_velocity_xy - target_velocity_xy[:, None, :]
            )
            radius_squared = (
                relative_to_current_target * relative_to_current_target
            ).sum(dim=-1).clamp_min(1.0e-6)
            current_angular_rate = (
                relative_to_current_target[..., 0]
                * relative_velocity_to_current_target[..., 1]
                - relative_to_current_target[..., 1]
                * relative_velocity_to_current_target[..., 0]
            ) / radius_squared
            ordered_angular_rate = torch.gather(
                current_angular_rate, 1, angular_order
            )
            forward_gap_rate = torch.cat(
                (
                    ordered_angular_rate[:, 1:]
                    - ordered_angular_rate[:, :-1],
                    ordered_angular_rate[:, :1]
                    - ordered_angular_rate[:, -1:],
                ),
                dim=1,
            )
            backward_gap_rate = torch.roll(forward_gap_rate, shifts=1, dims=1)
            ordered_tangential_speed = ordered_tangential_speed + (
                0.5
                * phase_angular_velocity_damping_gain_s
                * reference_radius_m[:, None]
                * (forward_gap_rate - backward_gap_rate)
                * (
                    torch.ones(
                        batch,
                        1,
                        dtype=pursuer_xy.dtype,
                        device=pursuer_xy.device,
                    )
                    if phase_angular_velocity_damping_active is None
                    else phase_angular_velocity_damping_active[:, None].to(
                        pursuer_xy.dtype
                    )
                )
            )
        ordered_tangential_speed = ordered_tangential_speed.clamp(
            -maximum_tangential_speed_mps,
            maximum_tangential_speed_mps,
        )
        tangential_speed = torch.empty_like(ordered_tangential_speed)
        tangential_speed.scatter_(
            1, angular_order, ordered_tangential_speed
        )
        if blocker_phase_anchor_control_enabled:
            # Adjacent-gap feedback has a rotational null space: a balanced
            # ring may freely rotate away from the evader's forward direction.
            # Apply only a weak, bounded phase correction to the designated
            # reachable-intercept blocker.  The neighbouring gap controller
            # remains responsible for redistributing the other agents.
            blocker_valid = (blocker_agent_index >= 0) & updated_anchor_valid
            safe_blocker_agent_index = blocker_agent_index.clamp_min(0)
            blocker_angle = torch.gather(
                current_angle, 1, safe_blocker_agent_index[:, None]
            ).squeeze(1)
            blocker_phase_error = torch.atan2(
                torch.sin(updated_heading - blocker_angle),
                torch.cos(updated_heading - blocker_angle),
            )
            blocker_anchor_speed = (
                blocker_phase_anchor_gain
                * reference_radius_m
                * blocker_phase_error
            ).clamp(
                -blocker_phase_anchor_maximum_speed_mps,
                blocker_phase_anchor_maximum_speed_mps,
            )
            blocker_anchor_speed = torch.where(
                blocker_valid,
                blocker_anchor_speed,
                torch.zeros_like(blocker_anchor_speed),
            )
            tangential_speed.scatter_add_(
                1,
                safe_blocker_agent_index[:, None],
                blocker_anchor_speed[:, None],
            )
            tangential_speed = tangential_speed.clamp(
                -maximum_tangential_speed_mps,
                maximum_tangential_speed_mps,
            )
        current_target_radius = relative_to_current_target.norm(dim=-1)
        phase_blend = (
            (
                reference_radius_m[:, None]
                + phase_activation_margin_m
                - current_target_radius
            )
            / max(phase_activation_margin_m, 1.0e-6)
        ).clamp(0.0, 1.0)
        tangential_speed = phase_blend * tangential_speed
        tangent_unit = torch.stack(
            (-torch.sin(current_angle), torch.cos(current_angle)), dim=-1
        )
        if gain_residual_control_enabled:
            residual_gate = (residual_scale > 0.0).to(pursuer_xy.dtype)[:, None]
            bounded_gain_action = gain_residual_action.clamp(-1.0, 1.0)
            radial_gain_multiplier = 1.0 + (
                residual_gate
                * radial_gain_residual_scale
                * bounded_gain_action[..., 0]
            )
            phase_gain_multiplier = (
                torch.ones_like(tangential_speed)
                if tangential_velocity_residual_control_enabled
                else 1.0
                + residual_gate
                * phase_gain_residual_scale
                * bounded_gain_action[..., 1]
            )
            tangential_speed = tangential_speed * phase_gain_multiplier
        else:
            radial_gain_multiplier = torch.ones_like(tangential_speed)
        analytic_tangential_speed = tangential_speed
        radial_error = reference_radius_m[:, None] - current_radius.squeeze(-1)
        if region_minimum_radius_m is not None:
            if region_minimum_radius_m <= 0:
                raise ValueError("region physical minimum radius must be positive")
            # Safe agents inside the reference circle need not fly back out
            # to an exact slot radius. Physical separation remains downstream.
            radial_error = torch.minimum(radial_error, torch.zeros_like(radial_error))
            radial_error = torch.where(
                current_radius.squeeze(-1) < region_minimum_radius_m,
                region_minimum_radius_m - current_radius.squeeze(-1),
                radial_error,
            )
        radial_speed = radial_gain * radial_gain_multiplier * radial_error
        if radial_velocity_damping_gain > 0.0:
            relative_radial_velocity = (
                (
                    pursuer_velocity_xy
                    - target_velocity_xy[:, None, :]
                )
                * radial_unit
            ).sum(dim=-1)
            radial_speed = radial_speed - (
                radial_velocity_damping_gain * relative_radial_velocity
            )
        if maximum_radial_speed_mps is not None:
            if phase_scheduled_radial_speed_budget:
                effective_radial_speed_limit = (
                    maximum_radial_speed_mps
                    + (maximum_speed_mps - maximum_radial_speed_mps)
                    * (1.0 - phase_blend)
                )
            else:
                effective_radial_speed_limit = torch.full_like(
                    radial_speed, maximum_radial_speed_mps
                )
            radial_speed = torch.maximum(
                torch.minimum(radial_speed, effective_radial_speed_limit),
                -effective_radial_speed_limit,
            )
        if (
            radial_velocity_residual_control_enabled
            or tangential_velocity_residual_control_enabled
        ):
            residual_gate = (residual_scale > 0.0).to(pursuer_xy.dtype)[:, None]
            applied_policy_residual = torch.zeros_like(desired_velocity_xy)
            if radial_velocity_residual_control_enabled:
                radial_velocity_residual_speed = (
                    residual_gate
                    * radial_velocity_residual_scale_mps
                    * radial_velocity_residual_action.clamp(-1.0, 1.0)
                )
                applied_policy_residual = applied_policy_residual + (
                    radial_velocity_residual_speed[..., None] * radial_unit
                )
            if tangential_velocity_residual_control_enabled:
                tangential_velocity_residual_speed = (
                    residual_gate
                    * tangential_velocity_residual_scale_mps
                    * tangential_velocity_residual_action.clamp(-1.0, 1.0)
                )
                applied_policy_residual = applied_policy_residual + (
                    tangential_velocity_residual_speed[..., None] * tangent_unit
                )
        else:
            applied_policy_residual = (
                torch.zeros_like(desired_velocity_xy)
                if gain_residual_control_enabled
                else residual_scale[:, None, None] * desired_velocity_xy
            )
        residual_radial_speed = (
            applied_policy_residual * radial_unit
        ).sum(dim=-1)
        residual_tangential_speed = (
            applied_policy_residual * tangent_unit
        ).sum(dim=-1)
        radial_direction = torch.where(
            radial_speed.abs() > 1.0e-6,
            radial_speed.sign(),
            torch.zeros_like(radial_speed),
        )
        phase_direction = torch.where(
            tangential_speed.abs() > 1.0e-6,
            tangential_speed.sign(),
            torch.zeros_like(tangential_speed),
        )
        residual_radial_along_control = residual_radial_speed * radial_direction
        residual_phase_along_control = (
            residual_tangential_speed * phase_direction
        )
        diagnostic_active = active & (residual_scale > 0.0)
        policy_residual_norm_mps = torch.where(
            diagnostic_active,
            applied_policy_residual.norm(dim=-1).mean(dim=1),
            zero_diagnostic,
        )
        policy_residual_radial_support_mps = torch.where(
            diagnostic_active,
            residual_radial_along_control.clamp_min(0.0).mean(dim=1),
            zero_diagnostic,
        )
        policy_residual_radial_opposition_mps = torch.where(
            diagnostic_active,
            (-residual_radial_along_control).clamp_min(0.0).mean(dim=1),
            zero_diagnostic,
        )
        policy_residual_phase_support_mps = torch.where(
            diagnostic_active,
            residual_phase_along_control.clamp_min(0.0).mean(dim=1),
            zero_diagnostic,
        )
        policy_residual_phase_opposition_mps = torch.where(
            diagnostic_active,
            (-residual_phase_along_control).clamp_min(0.0).mean(dim=1),
            zero_diagnostic,
        )
        feedforward_velocity = (
            feedforward_scale[:, None, None] * target_velocity_xy[:, None, :]
        ).expand(-1, agent_count, -1)
        correction_velocity = (
            radial_speed[..., None] * radial_unit
            + tangential_speed[..., None] * tangent_unit
            + applied_policy_residual
        )
        unlimited_guided_velocity = (
            feedforward_velocity + correction_velocity
        )
        priority_correction_scale = torch.ones_like(radial_speed)
        if feedforward_priority_projection_enabled:
            if radial_correction_priority_active is None:
                guided_velocity, priority_correction_scale = (
                    compose_feedforward_priority_velocity(
                        feedforward_velocity,
                        correction_velocity,
                        maximum_speed_mps,
                    )
                )
            else:
                radial_correction = (
                    (correction_velocity * radial_unit).sum(dim=-1, keepdim=True)
                    * radial_unit
                )
                secondary_correction = correction_velocity - radial_correction
                (
                    guided_velocity,
                    _,
                    priority_correction_scale,
                ) = compose_feedforward_radial_priority_velocity(
                    feedforward_velocity,
                    radial_correction,
                    secondary_correction,
                    radial_correction_priority_active[:, None].expand(
                        -1, agent_count
                    ),
                    maximum_speed_mps,
                )
        else:
            guided_velocity = unlimited_guided_velocity
        if far_slot_intercept_control_enabled:
            current_slot_xy = (
                estimated_target_xy[:, None, :]
                + reference_radius_m[:, None, None] * slot_unit
            )
            assigned_current_slot_xy = torch.gather(
                current_slot_xy,
                1,
                safe_assignment[..., None].expand(-1, -1, 2),
            )
            slot_intercept = bounded_per_agent_constant_velocity_intercept(
                pursuer_xy,
                assigned_current_slot_xy,
                target_velocity_xy,
                pursuer_speed_mps=maximum_speed_mps,
                minimum_horizon_s=far_slot_intercept_minimum_horizon_s,
                maximum_horizon_s=far_slot_intercept_maximum_horizon_s,
                horizon_samples=far_slot_intercept_horizon_samples,
            )
            intercept_delta = slot_intercept.point_xy - pursuer_xy
            intercept_distance = intercept_delta.norm(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            intercept_velocity = (
                maximum_speed_mps * intercept_delta / intercept_distance
            )
            far_route_active = (
                active[:, None]
                & previous_assignment_valid[:, None]
                & (phase_blend <= 1.0e-6)
            )
            guided_velocity = torch.where(
                far_route_active[..., None],
                intercept_velocity,
                guided_velocity,
            )
    else:
        unlimited_guided_velocity = (
            feedforward_scale[:, None, None] * target_velocity_xy[:, None, :]
            + position_gain * slot_error
            + residual_scale[:, None, None] * desired_velocity_xy
        )
        guided_velocity = unlimited_guided_velocity
    guided_speed = unlimited_guided_velocity.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    if phase_separation_control_enabled and feedforward_priority_projection_enabled:
        speed_scale = priority_correction_scale[..., None]
    else:
        speed_scale = (maximum_speed_mps / guided_speed).clamp_max(1.0)
        guided_velocity = unlimited_guided_velocity * speed_scale
    speed_limit_active_agent_fraction = zero_diagnostic
    speed_limit_mean_scale = zero_diagnostic
    speed_limit_forward_loss_mps = zero_diagnostic
    requested_tangential_speed_mps = zero_diagnostic
    executed_tangential_speed_mps = zero_diagnostic
    if phase_separation_control_enabled:
        target_heading_unit = target_velocity_xy / target_speed[:, None].clamp_min(
            1.0e-6
        )
        target_heading_valid = target_speed >= minimum_target_speed_mps
        forward_loss = (
            (unlimited_guided_velocity - guided_velocity)
            * target_heading_unit[:, None, :]
        ).sum(dim=-1).clamp_min(0.0)
        budget_diagnostic_active = active & target_heading_valid
        speed_limit_active_agent_fraction = torch.where(
            budget_diagnostic_active,
            (guided_speed.squeeze(-1) > maximum_speed_mps + 1.0e-6)
            .to(pursuer_xy.dtype)
            .mean(dim=1),
            zero_diagnostic,
        )
        speed_limit_mean_scale = torch.where(
            budget_diagnostic_active,
            speed_scale.squeeze(-1).mean(dim=1),
            zero_diagnostic,
        )
        speed_limit_forward_loss_mps = torch.where(
            budget_diagnostic_active,
            forward_loss.mean(dim=1),
            zero_diagnostic,
        )
        requested_tangential_speed_mps = torch.where(
            budget_diagnostic_active,
            analytic_tangential_speed.abs().mean(dim=1),
            zero_diagnostic,
        )
        executed_tangential_speed_mps = torch.where(
            budget_diagnostic_active,
            (analytic_tangential_speed * speed_scale.squeeze(-1))
            .abs()
            .mean(dim=1),
            zero_diagnostic,
        )
    velocity = torch.where(active[:, None, None], guided_velocity, desired_velocity_xy)
    if diagnostic_sink is not None:
        diagnostic_sink.update(unlimited_velocity_xy=unlimited_guided_velocity.detach().clone(),
                               limited_velocity_xy=velocity.detach().clone())
    return MovingTargetSlotGuidanceResult(
        velocity_xy=velocity,
        assignment=assignment,
        assignment_switched=should_switch & previous_assignment_valid,
        anchor_heading_rad=updated_heading,
        anchor_valid=updated_anchor_valid,
        active=active,
        mean_slot_error_m=torch.where(
            active, mean_slot_error, torch.zeros_like(mean_slot_error)
        ),
        assigned_slot_xy=torch.where(
            assignment_gate[:, None, None],
            assigned_slot_xy,
            torch.zeros_like(assigned_slot_xy),
        ),
        angular_error_rad=circular_gap_error(pursuer_xy, estimated_target_xy),
        policy_residual_norm_mps=policy_residual_norm_mps,
        policy_residual_radial_support_mps=(
            policy_residual_radial_support_mps
        ),
        policy_residual_radial_opposition_mps=(
            policy_residual_radial_opposition_mps
        ),
        policy_residual_phase_support_mps=policy_residual_phase_support_mps,
        policy_residual_phase_opposition_mps=(
            policy_residual_phase_opposition_mps
        ),
        speed_limit_active_agent_fraction=speed_limit_active_agent_fraction,
        speed_limit_mean_scale=speed_limit_mean_scale,
        speed_limit_forward_loss_mps=speed_limit_forward_loss_mps,
        requested_tangential_speed_mps=requested_tangential_speed_mps,
        executed_tangential_speed_mps=executed_tangential_speed_mps,
    )


def slot_goal_progress_reward(
    pre_action_error_m: torch.Tensor,
    post_action_pursuer_xy: torch.Tensor,
    assigned_slot_xy: torch.Tensor,
    active: torch.Tensor,
    *,
    command_speed_mps: float,
    policy_dt_s: float,
    reward_weight: float,
    use_maximum_error: bool = False,
) -> SlotGoalProgressRewardResult:
    """Reward progress toward each pursuer's assigned moving role goal.

    The assigned slot is frozen for one high-level control step.  Comparing
    the pre-action error with the post-action position therefore measures the
    effect of the executed action, analogous to NavRL's goal-directed velocity
    progress while preserving the multi-agent slot and safety semantics.
    """
    if post_action_pursuer_xy.shape != assigned_slot_xy.shape:
        raise ValueError("post-action positions and assigned slots must match")
    if post_action_pursuer_xy.ndim != 3 or post_action_pursuer_xy.shape[-1] != 2:
        raise ValueError("slot progress tensors must have shape [batch, agent, 2]")
    batch = post_action_pursuer_xy.shape[0]
    if pre_action_error_m.shape != (batch,) or active.shape != (batch,):
        raise ValueError("slot progress batch shape mismatch")
    progress_scale = command_speed_mps * policy_dt_s
    if progress_scale <= 0.0:
        raise ValueError("command_speed_mps and policy_dt_s must be positive")
    if reward_weight < 0.0:
        raise ValueError("slot progress reward weight must be nonnegative")
    post_action_error_each = (
        assigned_slot_xy - post_action_pursuer_xy
    ).norm(dim=-1)
    post_action_error = (
        post_action_error_each.max(dim=1).values
        if use_maximum_error
        else post_action_error_each.mean(dim=1)
    )
    normalized_progress = (
        (pre_action_error_m - post_action_error) / progress_scale
    ).clamp(-1.0, 1.0)
    normalized_progress = torch.where(
        active, normalized_progress, torch.zeros_like(normalized_progress)
    )
    return SlotGoalProgressRewardResult(
        reward=reward_weight * normalized_progress,
        normalized_progress=normalized_progress,
        post_action_mean_error_m=torch.where(
            active, post_action_error, torch.zeros_like(post_action_error)
        ),
    )


def project_staged_contraction_velocity(
    desired_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    estimated_target_xy: torch.Tensor,
    staging_reached: torch.Tensor,
    contraction_unlocked: torch.Tensor,
    *,
    maximum_inward_speed_mps: float,
    target_velocity_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hold the staging ring until aligned, then limit relative contraction.

    When a moving-target velocity is supplied, the constraint acts on pursuer
    velocity relative to that target.  Target translation therefore remains
    available to maintain the staging radius instead of being mistaken for
    inward contraction in the world frame.
    """
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if estimated_target_xy.shape != (pursuer_xy.shape[0], 2):
        raise ValueError("estimated_target_xy shape mismatch")
    batch, agent_count = pursuer_xy.shape[:2]
    if staging_reached.shape == (batch,):
        staging_reached_each = staging_reached[:, None].expand(-1, agent_count)
    elif staging_reached.shape == (batch, agent_count):
        staging_reached_each = staging_reached
    else:
        raise ValueError("staging_reached shape mismatch")
    if contraction_unlocked.shape != (batch,):
        raise ValueError("contraction_unlocked shape mismatch")
    if maximum_inward_speed_mps <= 0.0:
        raise ValueError("maximum_inward_speed_mps must be positive")
    if target_velocity_xy is None:
        target_velocity_xy = torch.zeros(
            batch, 2, dtype=pursuer_xy.dtype, device=pursuer_xy.device
        )
    if target_velocity_xy.shape != (batch, 2):
        raise ValueError("target_velocity_xy shape mismatch")
    inward = estimated_target_xy[:, None, :] - pursuer_xy
    inward_unit = inward / inward.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    relative_velocity = desired_velocity_xy - target_velocity_xy[:, None, :]
    inward_speed = (relative_velocity * inward_unit).sum(dim=-1)
    locked = staging_reached_each & ~contraction_unlocked[:, None]
    allowed_inward = torch.where(
        contraction_unlocked[:, None],
        torch.full_like(inward_speed, maximum_inward_speed_mps),
        torch.where(
            locked,
            torch.zeros_like(inward_speed),
            torch.full_like(inward_speed, torch.inf),
        ),
    )
    excess = (inward_speed - allowed_inward).clamp_min(0.0)
    return desired_velocity_xy - excess[..., None] * inward_unit


def project_predictive_pairwise_velocity(
    desired_velocity_xy: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    active: torch.Tensor,
    *,
    minimum_center_distance_m: float | torch.Tensor,
    prediction_horizon_s: float,
    command_response_ratio: float,
    maximum_speed_mps: float,
    projection_iterations: int = 4,
    restore_unsafe_buffer: bool = False,
) -> PredictiveProjectionResult:
    """Minimally repair pairwise closing commands over a finite horizon.

    The layer projects the nominal joint command onto reciprocal pairwise
    half-spaces.  It models the velocity realized during the prediction window
    as a blend of measured and commanded velocity, then shares each correction
    equally between the two vehicles.  Repeated pair sweeps handle coupled
    constraints while preserving the MAPPO command whenever it is already safe.
    """
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if current_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("current velocity and pursuer positions must match")
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("pairwise projection tensors must have shape [batch, agent, 2]")
    if active.shape != (pursuer_xy.shape[0],):
        raise ValueError("active shape mismatch")
    if minimum_center_distance_m <= 0.0 or prediction_horizon_s <= 0.0:
        raise ValueError("safety distance and prediction horizon must be positive")
    if not 0.0 < command_response_ratio <= 1.0:
        raise ValueError("command_response_ratio must be in (0, 1]")
    if maximum_speed_mps <= 0.0 or projection_iterations <= 0:
        raise ValueError("speed and projection iterations must be positive")

    projected = desired_velocity_xy.clone()
    corrected_pair_constraints = torch.zeros(
        pursuer_xy.shape[0], dtype=torch.long, device=pursuer_xy.device
    )
    pairs = [
        (left, right)
        for left in range(pursuer_xy.shape[1])
        for right in range(left + 1, pursuer_xy.shape[1])
    ]
    for _ in range(projection_iterations):
        for left, right in pairs:
            realized = (
                command_response_ratio * projected
                + (1.0 - command_response_ratio) * current_velocity_xy
            )
            relative_position = pursuer_xy[:, left] - pursuer_xy[:, right]
            distance = relative_position.norm(dim=-1).clamp_min(1.0e-6)
            outward_unit = relative_position / distance[:, None]
            relative_velocity = realized[:, left] - realized[:, right]
            closing_speed = -(relative_velocity * outward_unit).sum(dim=-1)
            clearance_recovery_rate = (
                distance - minimum_center_distance_m
            ) / prediction_horizon_s
            allowed_closing_speed = (
                clearance_recovery_rate
                if restore_unsafe_buffer
                else clearance_recovery_rate.clamp_min(0.0)
            )
            violation = closing_speed - allowed_closing_speed
            repair = active & (violation > 1.0e-6)
            # A +/- correction changes relative command speed by twice its
            # magnitude and realized speed by command_response_ratio.
            magnitude = (
                0.5 * violation.clamp_min(0.0) / command_response_ratio
            )
            correction = magnitude[:, None] * outward_unit * repair[:, None]
            projected[:, left] = projected[:, left] + correction
            projected[:, right] = projected[:, right] - correction
            corrected_pair_constraints += repair.long()
        speed = projected.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        projected = projected * (maximum_speed_mps / speed).clamp_max(1.0)

    realized = (
        command_response_ratio * projected
        + (1.0 - command_response_ratio) * current_velocity_xy
    )
    predicted_distances = []
    final_violations = []
    for left, right in pairs:
        relative_position = pursuer_xy[:, left] - pursuer_xy[:, right]
        distance = relative_position.norm(dim=-1).clamp_min(1.0e-6)
        outward_unit = relative_position / distance[:, None]
        relative_velocity = realized[:, left] - realized[:, right]
        predicted_relative = (
            relative_position + prediction_horizon_s * relative_velocity
        )
        predicted_distances.append(predicted_relative.norm(dim=-1))
        closing_speed = -(relative_velocity * outward_unit).sum(dim=-1)
        clearance_recovery_rate = (
            distance - minimum_center_distance_m
        ) / prediction_horizon_s
        allowed_closing_speed = (
            clearance_recovery_rate
            if restore_unsafe_buffer
            else clearance_recovery_rate.clamp_min(0.0)
        )
        final_violations.append(closing_speed - allowed_closing_speed)
    minimum_predicted_distance = torch.stack(predicted_distances, dim=1).min(dim=1).values
    maximum_residual_violation = torch.stack(final_violations, dim=1).max(dim=1).values
    correction_norm = (projected - desired_velocity_xy).flatten(start_dim=1).norm(dim=1)
    return PredictiveProjectionResult(
        velocity_xy=projected,
        active=active.clone(),
        corrected=active & (correction_norm > 1.0e-6),
        correction_norm_mps=correction_norm,
        corrected_pair_constraints=corrected_pair_constraints,
        minimum_predicted_center_distance_m=minimum_predicted_distance,
        infeasible=active & (maximum_residual_violation > 1.0e-4),
    )


def project_predictive_arena_boundary_velocity(
    desired_velocity_xy: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    active: torch.Tensor,
    *,
    arena_half_extent_m: float,
    minimum_clearance_m: float,
    prediction_horizon_s: float,
    command_response_ratio: float,
    maximum_speed_mps: float,
) -> PredictiveBoundaryProjectionResult:
    """Project commands into finite-horizon axis-aligned arena half-spaces."""
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if current_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("current velocity and pursuer positions must match")
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("boundary projection tensors must be [batch, agent, 2]")
    if active.shape != (pursuer_xy.shape[0],):
        raise ValueError("active shape mismatch")
    if arena_half_extent_m <= 0.0 or minimum_clearance_m <= 0.0:
        raise ValueError("arena extent and boundary clearance must be positive")
    if minimum_clearance_m >= arena_half_extent_m:
        raise ValueError("boundary clearance must be smaller than arena extent")
    if prediction_horizon_s <= 0.0 or maximum_speed_mps <= 0.0:
        raise ValueError("prediction horizon and speed limit must be positive")
    if not 0.0 < command_response_ratio <= 1.0:
        raise ValueError("command_response_ratio must be in (0, 1]")

    safe_extent = arena_half_extent_m - minimum_clearance_m
    realized = (
        command_response_ratio * desired_velocity_xy
        + (1.0 - command_response_ratio) * current_velocity_xy
    )
    predicted = pursuer_xy + prediction_horizon_s * realized
    repaired_predicted = predicted.clamp(-safe_extent, safe_extent)
    violating = predicted.abs() > safe_extent
    realized_repair = (
        (repaired_predicted - predicted) / prediction_horizon_s
    )
    command_repair = realized_repair / command_response_ratio
    projected = torch.where(
        (active[:, None, None] & violating),
        desired_velocity_xy + command_repair,
        desired_velocity_xy,
    )
    speed = projected.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    projected = projected * (maximum_speed_mps / speed).clamp_max(1.0)

    realized_final = (
        command_response_ratio * projected
        + (1.0 - command_response_ratio) * current_velocity_xy
    )
    predicted_final = pursuer_xy + prediction_horizon_s * realized_final
    minimum_predicted_clearance = (
        arena_half_extent_m - predicted_final.abs()
    ).flatten(start_dim=1).min(dim=1).values
    residual_safe_violation = (
        predicted_final.abs() - safe_extent
    ).flatten(start_dim=1).max(dim=1).values
    correction_norm = (
        projected - desired_velocity_xy
    ).flatten(start_dim=1).norm(dim=1)
    return PredictiveBoundaryProjectionResult(
        velocity_xy=projected,
        active=active.clone(),
        corrected=active & (correction_norm > 1.0e-6),
        correction_norm_mps=correction_norm,
        corrected_constraints=(
            active[:, None, None] & violating
        ).sum(dim=(1, 2)),
        minimum_predicted_clearance_m=minimum_predicted_clearance,
        infeasible=active & (residual_safe_violation > 1.0e-4),
    )


def predictive_projection_reward(
    correction_norm_mps: torch.Tensor,
    infeasible: torch.Tensor,
    *,
    command_speed_mps: float,
    correction_weight: float,
    infeasible_penalty: float,
) -> torch.Tensor:
    """Penalize reliance on the safety layer without rewarding longer survival."""
    if correction_norm_mps.shape != infeasible.shape:
        raise ValueError("projection correction and infeasible shapes must match")
    if command_speed_mps <= 0.0:
        raise ValueError("command_speed_mps must be positive")
    if correction_weight < 0.0 or infeasible_penalty < 0.0:
        raise ValueError("projection reward weights must be nonnegative")
    normalized_correction = correction_norm_mps / command_speed_mps
    return (
        -correction_weight * normalized_correction
        -infeasible_penalty * infeasible.float()
    )


def project_predictive_target_annulus_velocity(
    desired_velocity_xy: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    target_xy: torch.Tensor,
    target_velocity_xy: torch.Tensor,
    active: torch.Tensor,
    *,
    target_acceleration_xy: torch.Tensor | None = None,
    target_maximum_speed_mps: float | None = None,
    acceleration_prediction_substeps: int = 5,
    minimum_radius_m: float,
    maximum_radius_m: float,
    prediction_horizon_s: float,
    command_response_ratio: float,
    maximum_speed_mps: float,
    projection_iterations: int = 4,
    response_trajectory_substeps: int = 1,
    enforce_minimum_radius: bool = True,
    minimum_radius_active: torch.Tensor | None = None,
) -> PredictiveAnnulusProjectionResult:
    """Project predicted target-relative positions into a capture annulus."""
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if current_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("current velocity and pursuer positions must match")
    batch = pursuer_xy.shape[0]
    if target_xy.shape != (batch, 2) or target_velocity_xy.shape != (batch, 2):
        raise ValueError("target state shape mismatch")
    if target_acceleration_xy is not None:
        if target_acceleration_xy.shape != (batch, 2):
            raise ValueError("target acceleration shape mismatch")
        if target_maximum_speed_mps is None or target_maximum_speed_mps <= 0.0:
            raise ValueError("positive target maximum speed is required with acceleration")
        if acceleration_prediction_substeps <= 0:
            raise ValueError("acceleration prediction substeps must be positive")
    if active.shape != (batch,):
        raise ValueError("active shape mismatch")
    if minimum_radius_active is None:
        minimum_radius_active = torch.full_like(active, enforce_minimum_radius)
    elif minimum_radius_active.shape != (batch,):
        raise ValueError("minimum radius active shape mismatch")
    if minimum_radius_m <= 0.0 or maximum_radius_m <= minimum_radius_m:
        raise ValueError("annulus radii are invalid")
    if prediction_horizon_s <= 0.0 or maximum_speed_mps <= 0.0:
        raise ValueError("prediction horizon and speed must be positive")
    if not 0.0 < command_response_ratio <= 1.0:
        raise ValueError("command_response_ratio must be in (0, 1]")
    if projection_iterations <= 0 or response_trajectory_substeps <= 0:
        raise ValueError("projection iterations and response substeps must be positive")

    projected = desired_velocity_xy.clone()
    corrected_constraints = torch.zeros(
        batch, dtype=torch.long, device=pursuer_xy.device
    )
    relative_position = pursuer_xy - target_xy[:, None, :]
    current_direction = relative_position / relative_position.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    if response_trajectory_substeps > 1:
        # The legacy projection constrains only the terminal point.  For a
        # multi-step response audit, derive the per-step first-order response
        # whose compounded gain equals the frozen terminal response ratio and
        # constrain every intermediate position in the same horizon.
        substeps = response_trajectory_substeps
        response_dt_s = prediction_horizon_s / float(substeps)
        response_alpha = 1.0 - (1.0 - command_response_ratio) ** (
            1.0 / float(substeps)
        )
        response_weights = torch.tensor(
            [
                1.0 - (1.0 - response_alpha) ** step
                for step in range(substeps + 1)
            ],
            dtype=pursuer_xy.dtype,
            device=pursuer_xy.device,
        )
        command_displacement_coefficients = torch.cumsum(
            0.5
            * response_dt_s
            * (response_weights[:-1] + response_weights[1:]),
            dim=0,
        )

        target_displacement_steps = []
        target_displacement = torch.zeros_like(target_velocity_xy)
        predicted_target_velocity = target_velocity_xy.clone()
        for _ in range(substeps):
            next_target_velocity = predicted_target_velocity
            if target_acceleration_xy is not None:
                next_target_velocity = (
                    predicted_target_velocity
                    + response_dt_s * target_acceleration_xy
                )
                target_speed = next_target_velocity.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1.0e-6)
                next_target_velocity = next_target_velocity * (
                    target_maximum_speed_mps / target_speed
                ).clamp_max(1.0)
            target_displacement = target_displacement + (
                0.5
                * response_dt_s
                * (predicted_target_velocity + next_target_velocity)
            )
            target_displacement_steps.append(target_displacement)
            predicted_target_velocity = next_target_velocity
        target_displacement_trajectory = torch.stack(
            target_displacement_steps, dim=1
        )

        def predicted_relative_trajectory(command: torch.Tensor) -> torch.Tensor:
            displacement = torch.zeros_like(command)
            trajectory = []
            previous_velocity = current_velocity_xy
            for step in range(1, substeps + 1):
                weight = response_weights[step]
                next_velocity = (
                    (1.0 - weight) * current_velocity_xy
                    + weight * command
                )
                displacement = displacement + (
                    0.5
                    * response_dt_s
                    * (previous_velocity + next_velocity)
                )
                trajectory.append(
                    relative_position
                    + displacement
                    - target_displacement_trajectory[:, step - 1, None, :]
                )
                previous_velocity = next_velocity
            return torch.stack(trajectory, dim=1)

        for _ in range(projection_iterations):
            for step in range(substeps):
                predicted_relative = predicted_relative_trajectory(projected)[
                    :, step
                ]
                predicted_radius = predicted_relative.norm(dim=-1)
                predicted_direction = predicted_relative / predicted_radius[
                    ..., None
                ].clamp_min(1.0e-6)
                predicted_direction = torch.where(
                    (predicted_radius > 1.0e-6)[..., None],
                    predicted_direction,
                    current_direction,
                )
                clamped_radius = predicted_radius.clamp_max(maximum_radius_m)
                clamped_radius = torch.where(
                    minimum_radius_active[:, None],
                    clamped_radius.clamp_min(minimum_radius_m),
                    clamped_radius,
                )
                violation = active[:, None] & (
                    (predicted_radius > maximum_radius_m + 1.0e-6)
                    | (
                        minimum_radius_active[:, None]
                        & (predicted_radius < minimum_radius_m - 1.0e-6)
                    )
                )
                desired_predicted_relative = (
                    clamped_radius[..., None] * predicted_direction
                )
                command_correction = (
                    desired_predicted_relative - predicted_relative
                ) / command_displacement_coefficients[step].clamp_min(1.0e-6)
                projected = projected + command_correction * violation[..., None]
                corrected_constraints += violation.sum(dim=1)
                speed = projected.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
                projected = projected * (
                    maximum_speed_mps / speed
                ).clamp_max(1.0)

        predicted_relative = predicted_relative_trajectory(projected)
        predicted_radius = predicted_relative.norm(dim=-1)
        residual = active[:, None, None] & (
            (predicted_radius > maximum_radius_m + 1.0e-4)
            | (
                minimum_radius_active[:, None, None]
                & (predicted_radius < minimum_radius_m - 1.0e-4)
            )
        )
        correction_norm = (projected - desired_velocity_xy).flatten(
            start_dim=1
        ).norm(dim=1)
        return PredictiveAnnulusProjectionResult(
            velocity_xy=projected,
            active=active.clone(),
            corrected=active & (correction_norm > 1.0e-6),
            correction_norm_mps=correction_norm,
            corrected_constraints=corrected_constraints,
            minimum_predicted_radius_m=predicted_radius.amin(dim=(1, 2)),
            maximum_predicted_radius_m=predicted_radius.amax(dim=(1, 2)),
            infeasible=active & residual.flatten(start_dim=1).any(dim=1),
        )

    target_displacement = prediction_horizon_s * target_velocity_xy
    if target_acceleration_xy is not None:
        target_displacement = torch.zeros_like(target_velocity_xy)
        predicted_target_velocity = target_velocity_xy.clone()
        acceleration_dt_s = (
            prediction_horizon_s / acceleration_prediction_substeps
        )
        for _ in range(acceleration_prediction_substeps):
            next_target_velocity = (
                predicted_target_velocity
                + acceleration_dt_s * target_acceleration_xy
            )
            target_speed = next_target_velocity.norm(dim=-1, keepdim=True).clamp_min(
                1.0e-6
            )
            next_target_velocity = next_target_velocity * (
                target_maximum_speed_mps / target_speed
            ).clamp_max(1.0)
            target_displacement += (
                0.5
                * acceleration_dt_s
                * (predicted_target_velocity + next_target_velocity)
            )
            predicted_target_velocity = next_target_velocity
    for _ in range(projection_iterations):
        realized = (
            command_response_ratio * projected
            + (1.0 - command_response_ratio) * current_velocity_xy
        )
        predicted_relative = (
            relative_position
            + prediction_horizon_s * realized
            - target_displacement[:, None, :]
        )
        predicted_radius = predicted_relative.norm(dim=-1)
        predicted_direction = predicted_relative / predicted_radius[
            ..., None
        ].clamp_min(1.0e-6)
        predicted_direction = torch.where(
            (predicted_radius > 1.0e-6)[..., None],
            predicted_direction,
            current_direction,
        )
        clamped_radius = predicted_radius.clamp_max(maximum_radius_m)
        clamped_radius = torch.where(
            minimum_radius_active[:, None],
            clamped_radius.clamp_min(minimum_radius_m),
            clamped_radius,
        )
        violation = active[:, None] & (
            (predicted_radius > maximum_radius_m + 1.0e-6)
            | (
                minimum_radius_active[:, None]
                & (predicted_radius < minimum_radius_m - 1.0e-6)
            )
        )
        desired_predicted_relative = (
            clamped_radius[..., None] * predicted_direction
        )
        command_correction = (
            desired_predicted_relative - predicted_relative
        ) / (prediction_horizon_s * command_response_ratio)
        projected = projected + command_correction * violation[..., None]
        corrected_constraints += violation.sum(dim=1)
        speed = projected.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        projected = projected * (maximum_speed_mps / speed).clamp_max(1.0)

    realized = (
        command_response_ratio * projected
        + (1.0 - command_response_ratio) * current_velocity_xy
    )
    predicted_radius = (
        relative_position
        + prediction_horizon_s * realized
        - target_displacement[:, None, :]
    ).norm(dim=-1)
    residual = active[:, None] & (
        (predicted_radius > maximum_radius_m + 1.0e-4)
        | (
            minimum_radius_active[:, None]
            & (predicted_radius < minimum_radius_m - 1.0e-4)
        )
    )
    correction_norm = (projected - desired_velocity_xy).flatten(
        start_dim=1
    ).norm(dim=1)
    return PredictiveAnnulusProjectionResult(
        velocity_xy=projected,
        active=active.clone(),
        corrected=active & (correction_norm > 1.0e-6),
        correction_norm_mps=correction_norm,
        corrected_constraints=corrected_constraints,
        minimum_predicted_radius_m=predicted_radius.min(dim=1).values,
        maximum_predicted_radius_m=predicted_radius.max(dim=1).values,
        infeasible=active & residual.any(dim=1),
    )


def project_dynamics_consistent_target_annulus_velocity(
    desired_velocity_xy: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    target_xy: torch.Tensor,
    target_velocity_xy: torch.Tensor,
    active: torch.Tensor,
    *,
    target_acceleration_xy: torch.Tensor | None = None,
    target_maximum_speed_mps: float | None = None,
    minimum_radius_m: float,
    maximum_radius_m: float,
    prediction_horizon_s: float,
    maximum_speed_mps: float,
    low_level_dt_s: float,
    velocity_kp: float,
    maximum_planar_acceleration_mps2: float,
    radial_candidate_count: int = 33,
    minimum_radius_active: torch.Tensor | None = None,
    common_acceleration_feedforward_active: torch.Tensor | None = None,
    preentry_velocity_sync_active: torch.Tensor | None = None,
) -> PredictiveAnnulusProjectionResult:
    """Select a radial command correction using the physical velocity loop.

    Each constant high-level command candidate is rolled forward with the same
    proportional velocity error, vector acceleration cap, and physics step as
    the Crazyflie execution controller.  Feasibility is lexicographically
    preferred over command change; among feasible candidates the smallest
    radial change is selected independently for every pursuer.
    """
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if current_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("current velocity and pursuer positions must match")
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("annulus projection tensors must be [batch, agent, 2]")
    batch, agent_count, _ = pursuer_xy.shape
    if target_xy.shape != (batch, 2) or target_velocity_xy.shape != (batch, 2):
        raise ValueError("target state shape mismatch")
    if active.shape != (batch,):
        raise ValueError("active shape mismatch")
    if minimum_radius_active is None:
        minimum_radius_active = torch.ones_like(active)
    elif minimum_radius_active.shape != (batch,):
        raise ValueError("minimum radius active shape mismatch")
    if common_acceleration_feedforward_active is None:
        common_acceleration_feedforward_active = torch.zeros_like(active)
    elif common_acceleration_feedforward_active.shape != (batch,):
        raise ValueError("common acceleration feedforward shape mismatch")
    if common_acceleration_feedforward_active.any() and (
        target_acceleration_xy is None
    ):
        raise ValueError(
            "target acceleration is required for common feedforward"
        )
    if preentry_velocity_sync_active is None:
        preentry_velocity_sync_active = torch.zeros_like(active)
    elif preentry_velocity_sync_active.shape != (batch,):
        raise ValueError("preentry velocity sync shape mismatch")
    if target_acceleration_xy is not None:
        if target_acceleration_xy.shape != (batch, 2):
            raise ValueError("target acceleration shape mismatch")
        if target_maximum_speed_mps is None or target_maximum_speed_mps <= 0.0:
            raise ValueError("positive target maximum speed is required")
    if minimum_radius_m <= 0.0 or maximum_radius_m <= minimum_radius_m:
        raise ValueError("annulus radii are invalid")
    if prediction_horizon_s <= 0.0 or maximum_speed_mps <= 0.0:
        raise ValueError("prediction horizon and speed must be positive")
    if low_level_dt_s <= 0.0 or velocity_kp <= 0.0:
        raise ValueError("low-level step and velocity gain must be positive")
    if maximum_planar_acceleration_mps2 <= 0.0:
        raise ValueError("maximum planar acceleration must be positive")
    if radial_candidate_count < 3 or radial_candidate_count % 2 == 0:
        raise ValueError("radial candidate count must be odd and at least three")
    prediction_steps = round(prediction_horizon_s / low_level_dt_s)
    if prediction_steps <= 0 or not math.isclose(
        prediction_steps * low_level_dt_s,
        prediction_horizon_s,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise ValueError("prediction horizon must be divisible by low-level step")

    target_position = target_xy.clone()
    predicted_target_velocity = target_velocity_xy.clone()
    target_positions = []
    for _ in range(prediction_steps):
        if target_acceleration_xy is not None:
            predicted_target_velocity = (
                predicted_target_velocity
                + low_level_dt_s * target_acceleration_xy
            )
            target_speed = predicted_target_velocity.norm(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            predicted_target_velocity = predicted_target_velocity * (
                target_maximum_speed_mps / target_speed
            ).clamp_max(1.0)
        target_position = (
            target_position + low_level_dt_s * predicted_target_velocity
        )
        target_positions.append(target_position)
    target_trajectory = torch.stack(target_positions, dim=1)

    relative_position = pursuer_xy - target_xy[:, None, :]
    radial_unit = relative_position / relative_position.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    radial_delta = torch.linspace(
        -2.0 * maximum_speed_mps,
        2.0 * maximum_speed_mps,
        radial_candidate_count,
        dtype=pursuer_xy.dtype,
        device=pursuer_xy.device,
    )
    nominal_index = radial_candidate_count // 2
    common_velocity_change = (
        predicted_target_velocity - target_velocity_xy
    )
    common_velocity_change = torch.where(
        common_acceleration_feedforward_active[:, None],
        common_velocity_change,
        torch.zeros_like(common_velocity_change),
    )
    feedforward_velocity = (
        desired_velocity_xy + common_velocity_change[:, None, :]
    )
    candidate_velocity = (
        feedforward_velocity[:, :, None, :]
        + radial_delta[None, None, :, None] * radial_unit[:, :, None, :]
    )
    candidate_speed = candidate_velocity.norm(dim=-1, keepdim=True).clamp_min(
        1.0e-6
    )
    candidate_velocity = candidate_velocity * (
        maximum_speed_mps / candidate_speed
    ).clamp_max(1.0)

    predicted_position = pursuer_xy[:, :, None, :].expand(
        -1, -1, radial_candidate_count, -1
    ).clone()
    predicted_velocity = current_velocity_xy[:, :, None, :].expand_as(
        predicted_position
    ).clone()
    radius_trajectory = []
    for step in range(prediction_steps):
        planar_acceleration = velocity_kp * (
            candidate_velocity - predicted_velocity
        )
        acceleration_norm = planar_acceleration.norm(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        planar_acceleration = planar_acceleration * (
            maximum_planar_acceleration_mps2 / acceleration_norm
        ).clamp_max(1.0)
        predicted_velocity = (
            predicted_velocity + low_level_dt_s * planar_acceleration
        )
        predicted_position = (
            predicted_position + low_level_dt_s * predicted_velocity
        )
        predicted_relative = (
            predicted_position
            - target_trajectory[:, step, None, None, :]
        )
        radius_trajectory.append(predicted_relative.norm(dim=-1))
    predicted_radius = torch.stack(radius_trajectory, dim=-1)

    outer_violation = (
        predicted_radius - maximum_radius_m
    ).clamp_min(0.0)
    inner_violation = (
        minimum_radius_m - predicted_radius
    ).clamp_min(0.0) * minimum_radius_active[:, None, None, None]
    maximum_violation = torch.maximum(
        outer_violation, inner_violation
    ).amax(dim=-1)
    feasible = maximum_violation <= 1.0e-4
    change_cost = radial_delta.abs()[None, None, :].expand(
        batch, agent_count, -1
    )
    selection_cost = torch.where(
        feasible,
        change_cost,
        1.0e6 + 1.0e3 * maximum_violation + change_cost,
    )
    terminal_relative_position = (
        predicted_position
        - target_trajectory[:, -1, None, None, :]
    )
    terminal_radial_unit = terminal_relative_position / (
        terminal_relative_position.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    )
    terminal_relative_velocity = (
        predicted_velocity
        - predicted_target_velocity[:, None, None, :]
    )
    terminal_radial_speed = (
        terminal_relative_velocity * terminal_radial_unit
    ).sum(dim=-1).abs()
    midpoint_radius_m = 0.5 * (minimum_radius_m + maximum_radius_m)
    synchronization_cost = (
        (predicted_radius[..., -1] - midpoint_radius_m).abs()
        + terminal_radial_speed / velocity_kp
        + 1.0e-3 * change_cost
    )
    selection_cost = torch.where(
        preentry_velocity_sync_active[:, None, None],
        synchronization_cost,
        selection_cost,
    )
    selected_index = selection_cost.argmin(dim=2)
    selected_index = torch.where(
        active[:, None],
        selected_index,
        torch.full_like(selected_index, nominal_index),
    )
    gather_velocity_index = selected_index[..., None, None].expand(
        -1, -1, 1, 2
    )
    selected_velocity = torch.gather(
        candidate_velocity, 2, gather_velocity_index
    ).squeeze(2)
    gather_radius_index = selected_index[..., None, None].expand(
        -1, -1, 1, prediction_steps
    )
    selected_radius = torch.gather(
        predicted_radius, 2, gather_radius_index
    ).squeeze(2)
    selected_violation = torch.gather(
        maximum_violation, 2, selected_index[..., None]
    ).squeeze(2)
    nominal_violation = maximum_violation[:, :, nominal_index]
    correction_norm = (
        selected_velocity - desired_velocity_xy
    ).flatten(start_dim=1).norm(dim=1)
    return PredictiveAnnulusProjectionResult(
        velocity_xy=selected_velocity,
        active=active.clone(),
        corrected=active & (correction_norm > 1.0e-6),
        correction_norm_mps=correction_norm,
        corrected_constraints=(
            active[:, None] & (nominal_violation > 1.0e-6)
        ).sum(dim=1),
        minimum_predicted_radius_m=selected_radius.amin(dim=(1, 2)),
        maximum_predicted_radius_m=selected_radius.amax(dim=(1, 2)),
        infeasible=active & (selected_violation > 1.0e-4).any(dim=1),
    )


def bounded_last_seen_cv_target_state(
    measured_target_xy: torch.Tensor,
    measured_target_velocity_xy: torch.Tensor,
    last_seen_target_xy: torch.Tensor,
    last_seen_target_velocity_xy: torch.Tensor,
    team_visible: torch.Tensor,
    lost_steps: torch.Tensor,
    *,
    policy_dt_s: float,
    maximum_loss_time_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select measured or bounded last-seen CV state for short-loss safety.

    Hidden ground-truth state is never selected.  Once the bounded loss window
    expires the returned active mask is false, preventing an increasingly
    stale estimate from masquerading as a physical safety guarantee.
    """
    if measured_target_xy.ndim != 2 or measured_target_xy.shape[-1] != 2:
        raise ValueError("target states must be [batch, 2]")
    expected = measured_target_xy.shape
    for name, value in (
        ("measured target velocity", measured_target_velocity_xy),
        ("last seen target", last_seen_target_xy),
        ("last seen target velocity", last_seen_target_velocity_xy),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} shape mismatch")
    batch = measured_target_xy.shape[0]
    if team_visible.shape != (batch,) or lost_steps.shape != (batch,):
        raise ValueError("visibility/loss shape mismatch")
    if policy_dt_s <= 0.0 or maximum_loss_time_s <= 0.0:
        raise ValueError("CV loss timing must be positive")
    maximum_loss_steps = max(1, round(maximum_loss_time_s / policy_dt_s))
    short_loss_active = (
        (~team_visible)
        & (lost_steps > 0)
        & (lost_steps <= maximum_loss_steps)
    )
    prediction_time_s = lost_steps.to(measured_target_xy.dtype) * policy_dt_s
    predicted_target_xy = (
        last_seen_target_xy
        + prediction_time_s[:, None] * last_seen_target_velocity_xy
    )
    selected_xy = torch.where(
        team_visible[:, None], measured_target_xy, predicted_target_xy
    )
    selected_velocity = torch.where(
        team_visible[:, None],
        measured_target_velocity_xy,
        last_seen_target_velocity_xy,
    )
    return selected_xy, selected_velocity, team_visible | short_loss_active


def project_preentry_braking_reserve_velocity(
    desired_velocity_xy: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    target_xy: torch.Tensor,
    target_velocity_xy: torch.Tensor,
    active: torch.Tensor,
    *,
    reference_radius_m: float,
    activation_radius_m: float,
    maximum_speed_mps: float,
    maximum_planar_acceleration_mps2: float,
) -> PredictivePreentryBrakingResult:
    """Reserve enough radial distance to cancel inward relative speed.

    For every pursuer inside the approach zone, the commanded inward
    target-relative radial speed is bounded by ``sqrt(2*a*d)``, where ``d``
    is the remaining distance to the middle of the accepted ring.  This is
    the exact constant-acceleration stopping-distance condition and therefore
    introduces no tuned success threshold.  Tangential command components are
    preserved unless the final physical speed ball makes that impossible.
    """
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if current_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("current velocity and pursuer positions must match")
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("preentry braking tensors must be [batch, agent, 2]")
    batch = pursuer_xy.shape[0]
    if target_xy.shape != (batch, 2) or target_velocity_xy.shape != (batch, 2):
        raise ValueError("target state shape mismatch")
    if active.shape != (batch,):
        raise ValueError("active shape mismatch")
    if not 0.0 < reference_radius_m < activation_radius_m:
        raise ValueError("preentry braking radii are invalid")
    if maximum_speed_mps <= 0.0 or maximum_planar_acceleration_mps2 <= 0.0:
        raise ValueError("preentry braking physical limits must be positive")

    relative = pursuer_xy - target_xy[:, None, :]
    radius = relative.norm(dim=-1).clamp_min(1.0e-6)
    radial_unit = relative / radius[..., None]
    agent_active = active[:, None] & (radius <= activation_radius_m)
    remaining_distance = torch.relu(radius - reference_radius_m)
    minimum_relative_radial_command = -torch.sqrt(
        2.0 * maximum_planar_acceleration_mps2 * remaining_distance
    )
    desired_relative_radial_speed = (
        (desired_velocity_xy - target_velocity_xy[:, None, :]) * radial_unit
    ).sum(dim=-1)
    required_outward_correction = torch.relu(
        minimum_relative_radial_command - desired_relative_radial_speed
    )
    required_outward_correction = torch.where(
        agent_active, required_outward_correction, torch.zeros_like(required_outward_correction)
    )
    corrected_velocity = (
        desired_velocity_xy
        + required_outward_correction[..., None] * radial_unit
    )
    corrected_speed = corrected_velocity.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    corrected_velocity = corrected_velocity * (
        maximum_speed_mps / corrected_speed
    ).clamp_max(1.0)
    corrected_velocity = torch.where(
        agent_active[..., None], corrected_velocity, desired_velocity_xy
    )

    current_relative_radial_speed = (
        (current_velocity_xy - target_velocity_xy[:, None, :]) * radial_unit
    ).sum(dim=-1)
    current_stopping_distance = (
        torch.relu(-current_relative_radial_speed).square()
        / (2.0 * maximum_planar_acceleration_mps2)
    )
    stopping_margin = remaining_distance - current_stopping_distance
    stopping_margin = torch.where(
        agent_active, stopping_margin, torch.full_like(stopping_margin, torch.inf)
    )
    achieved_relative_radial_speed = (
        (corrected_velocity - target_velocity_xy[:, None, :]) * radial_unit
    ).sum(dim=-1)
    agent_infeasible = agent_active & (
        achieved_relative_radial_speed < minimum_relative_radial_command - 1.0e-4
    )
    correction_norm = (
        corrected_velocity - desired_velocity_xy
    ).flatten(start_dim=1).norm(dim=1)
    return PredictivePreentryBrakingResult(
        velocity_xy=corrected_velocity,
        active=agent_active.any(dim=1),
        corrected=agent_active.any(dim=1) & (correction_norm > 1.0e-6),
        correction_norm_mps=correction_norm,
        limited_agents=(agent_active & (required_outward_correction > 1.0e-6)).sum(dim=1),
        minimum_stopping_margin_m=stopping_margin.amin(dim=1),
        infeasible=agent_infeasible.any(dim=1),
    )


def project_dynamics_consistent_angular_gap_velocity(
    desired_velocity_xy: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    target_xy: torch.Tensor,
    target_velocity_xy: torch.Tensor,
    active: torch.Tensor,
    *,
    target_acceleration_xy: torch.Tensor | None = None,
    target_maximum_speed_mps: float | None = None,
    minimum_radius_m: float,
    maximum_radius_m: float,
    maximum_gap_rad: float,
    maximum_mean_gap_error_rad: float,
    prediction_horizon_s: float,
    maximum_speed_mps: float,
    low_level_dt_s: float,
    velocity_kp: float,
    maximum_planar_acceleration_mps2: float,
    correction_candidate_count: int = 17,
    minimum_radius_active: torch.Tensor | None = None,
    radial_constraints_active: torch.Tensor | None = None,
) -> PredictiveAngularGapProjectionResult:
    """Project a joint command away from predicted strong-ring violations.

    The nominal command is first rolled through the same acceleration-limited
    velocity loop used by the quadrotors.  Its terminal angular order defines
    the nearest uniformly spaced ring without fixing a global phase.  A
    bounded line projection then selects the smallest joint tangential change
    whose complete predicted trajectory satisfies the frozen radial and
    angular limits.  If no candidate is feasible, the normalized exact
    constraint violation determines the restorative candidate.
    """
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if current_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("current velocity and pursuer positions must match")
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("angular-gap projection tensors must be [batch, agent, 2]")
    batch, agent_count, _ = pursuer_xy.shape
    if agent_count < 3:
        raise ValueError("angular-gap projection requires at least three pursuers")
    if target_xy.shape != (batch, 2) or target_velocity_xy.shape != (batch, 2):
        raise ValueError("target state shape mismatch")
    if active.shape != (batch,):
        raise ValueError("active shape mismatch")
    if minimum_radius_active is None:
        minimum_radius_active = torch.ones_like(active)
    elif minimum_radius_active.shape != (batch,):
        raise ValueError("minimum radius active shape mismatch")
    if radial_constraints_active is None:
        radial_constraints_active = torch.ones_like(active)
    elif radial_constraints_active.shape != (batch,):
        raise ValueError("radial constraints active shape mismatch")
    if target_acceleration_xy is not None:
        if target_acceleration_xy.shape != (batch, 2):
            raise ValueError("target acceleration shape mismatch")
        if target_maximum_speed_mps is None or target_maximum_speed_mps <= 0.0:
            raise ValueError("positive target maximum speed is required")
    if not 0.0 < minimum_radius_m < maximum_radius_m:
        raise ValueError("angular-gap projection radii are invalid")
    desired_gap = 2.0 * math.pi / agent_count
    maximum_gap_slack = maximum_gap_rad - desired_gap
    if maximum_gap_slack <= 0.0 or maximum_mean_gap_error_rad <= 0.0:
        raise ValueError("angular-gap projection thresholds are invalid")
    if prediction_horizon_s <= 0.0 or maximum_speed_mps <= 0.0:
        raise ValueError("prediction horizon and speed must be positive")
    if low_level_dt_s <= 0.0 or velocity_kp <= 0.0:
        raise ValueError("low-level step and velocity gain must be positive")
    if maximum_planar_acceleration_mps2 <= 0.0:
        raise ValueError("maximum planar acceleration must be positive")
    if correction_candidate_count < 3:
        raise ValueError("at least three angular-gap candidates are required")
    prediction_steps = round(prediction_horizon_s / low_level_dt_s)
    if prediction_steps <= 0 or not math.isclose(
        prediction_steps * low_level_dt_s,
        prediction_horizon_s,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise ValueError("prediction horizon must be divisible by low-level step")

    target_position = target_xy.clone()
    predicted_target_velocity = target_velocity_xy.clone()
    target_positions = []
    for _ in range(prediction_steps):
        if target_acceleration_xy is not None:
            predicted_target_velocity = (
                predicted_target_velocity
                + low_level_dt_s * target_acceleration_xy
            )
            target_speed = predicted_target_velocity.norm(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            predicted_target_velocity = predicted_target_velocity * (
                target_maximum_speed_mps / target_speed
            ).clamp_max(1.0)
        target_position = target_position + low_level_dt_s * predicted_target_velocity
        target_positions.append(target_position)
    target_trajectory = torch.stack(target_positions, dim=1)

    nominal_position = pursuer_xy.clone()
    nominal_velocity = current_velocity_xy.clone()
    for step in range(prediction_steps):
        nominal_acceleration = velocity_kp * (
            desired_velocity_xy - nominal_velocity
        )
        nominal_acceleration_norm = nominal_acceleration.norm(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        nominal_acceleration = nominal_acceleration * (
            maximum_planar_acceleration_mps2 / nominal_acceleration_norm
        ).clamp_max(1.0)
        nominal_velocity = nominal_velocity + low_level_dt_s * nominal_acceleration
        nominal_position = nominal_position + low_level_dt_s * nominal_velocity

    current_relative = pursuer_xy - target_xy[:, None, :]
    current_angle = torch.remainder(
        torch.atan2(current_relative[..., 1], current_relative[..., 0]),
        2.0 * math.pi,
    )
    angular_order = torch.argsort(current_angle, dim=1)
    current_angle_ordered = torch.gather(current_angle, 1, angular_order)
    nominal_terminal_relative = nominal_position - target_trajectory[:, -1, None, :]
    nominal_terminal_angle = torch.remainder(
        torch.atan2(
            nominal_terminal_relative[..., 1],
            nominal_terminal_relative[..., 0],
        ),
        2.0 * math.pi,
    )
    nominal_terminal_angle_ordered = torch.gather(
        nominal_terminal_angle, 1, angular_order
    )
    ordered_terminal_angle = current_angle_ordered + torch.atan2(
        torch.sin(nominal_terminal_angle_ordered - current_angle_ordered),
        torch.cos(nominal_terminal_angle_ordered - current_angle_ordered),
    )
    slot_offset = torch.arange(
        agent_count, dtype=pursuer_xy.dtype, device=pursuer_xy.device
    )[None, :] * desired_gap
    free_phase_anchor = (ordered_terminal_angle - slot_offset).mean(dim=1)
    desired_terminal_angle = free_phase_anchor[:, None] + slot_offset
    terminal_angle_correction = desired_terminal_angle - ordered_terminal_angle

    command_position_response_s = low_level_dt_s * sum(
        1.0 - (1.0 - velocity_kp * low_level_dt_s) ** step
        for step in range(1, prediction_steps + 1)
    )
    if command_position_response_s <= 0.0:
        raise ValueError("velocity loop has no positive command response")
    terminal_radius_ordered = torch.gather(
        nominal_terminal_relative.norm(dim=-1), 1, angular_order
    )
    ordered_tangential_correction_mps = (
        terminal_radius_ordered
        * terminal_angle_correction
        / command_position_response_s
    )
    tangential_correction_mps = torch.zeros_like(
        ordered_tangential_correction_mps
    ).scatter(1, angular_order, ordered_tangential_correction_mps)
    current_radius = current_relative.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    radial_unit = current_relative / current_radius
    tangent_unit = torch.stack((-radial_unit[..., 1], radial_unit[..., 0]), dim=-1)
    full_correction = tangential_correction_mps[..., None] * tangent_unit

    candidate_scale = torch.linspace(
        0.0,
        1.0,
        correction_candidate_count,
        dtype=pursuer_xy.dtype,
        device=pursuer_xy.device,
    )
    candidate_velocity = (
        desired_velocity_xy[:, None, :, :]
        + candidate_scale[None, :, None, None] * full_correction[:, None, :, :]
    )
    candidate_speed = candidate_velocity.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    candidate_velocity = candidate_velocity * (
        maximum_speed_mps / candidate_speed
    ).clamp_max(1.0)

    predicted_position = pursuer_xy[:, None, :, :].expand(
        -1, correction_candidate_count, -1, -1
    ).clone()
    predicted_velocity = current_velocity_xy[:, None, :, :].expand_as(
        predicted_position
    ).clone()
    radius_trajectory = []
    maximum_gap_trajectory = []
    mean_gap_error_trajectory = []
    for step in range(prediction_steps):
        planar_acceleration = velocity_kp * (
            candidate_velocity - predicted_velocity
        )
        acceleration_norm = planar_acceleration.norm(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        planar_acceleration = planar_acceleration * (
            maximum_planar_acceleration_mps2 / acceleration_norm
        ).clamp_max(1.0)
        predicted_velocity = predicted_velocity + low_level_dt_s * planar_acceleration
        predicted_position = predicted_position + low_level_dt_s * predicted_velocity
        predicted_relative = (
            predicted_position - target_trajectory[:, step, None, None, :]
        )
        radius_trajectory.append(predicted_relative.norm(dim=-1))
        predicted_angle = torch.sort(
            torch.remainder(
                torch.atan2(predicted_relative[..., 1], predicted_relative[..., 0]),
                2.0 * math.pi,
            ),
            dim=2,
        ).values
        predicted_gap = torch.cat(
            (
                predicted_angle[:, :, 1:] - predicted_angle[:, :, :-1],
                2.0 * math.pi
                - predicted_angle[:, :, -1:]
                + predicted_angle[:, :, :1],
            ),
            dim=2,
        )
        maximum_gap_trajectory.append(predicted_gap.max(dim=2).values)
        mean_gap_error_trajectory.append(
            (predicted_gap - desired_gap).abs().mean(dim=2)
        )
    predicted_radius = torch.stack(radius_trajectory, dim=2)
    predicted_maximum_gap = torch.stack(maximum_gap_trajectory, dim=2)
    predicted_mean_gap_error = torch.stack(mean_gap_error_trajectory, dim=2)

    radial_scale = 0.5 * (maximum_radius_m - minimum_radius_m)
    outer_violation = torch.relu(predicted_radius - maximum_radius_m) * (
        radial_constraints_active[:, None, None, None]
    )
    inner_violation = torch.relu(minimum_radius_m - predicted_radius) * (
        minimum_radius_active[:, None, None, None]
        * radial_constraints_active[:, None, None, None]
    )
    radial_violation = (outer_violation + inner_violation).mean(dim=3) / radial_scale
    maximum_gap_violation = torch.relu(
        predicted_maximum_gap - maximum_gap_rad
    ) / maximum_gap_slack
    mean_gap_violation = torch.relu(
        predicted_mean_gap_error - maximum_mean_gap_error_rad
    ) / maximum_mean_gap_error_rad
    total_violation = radial_violation + maximum_gap_violation + mean_gap_violation
    maximum_violation = total_violation.amax(dim=2)
    feasible = maximum_violation <= 1.0e-4
    correction_norm = (
        candidate_velocity - desired_velocity_xy[:, None, :, :]
    ).flatten(start_dim=2).norm(dim=2)
    normalized_correction = correction_norm / maximum_speed_mps
    selection_cost = torch.where(
        feasible,
        normalized_correction,
        1.0e6
        + 1.0e3 * maximum_violation
        + total_violation[:, :, -1]
        + 1.0e-3 * normalized_correction,
    )
    selected_index = selection_cost.argmin(dim=1)
    selected_index = torch.where(
        active, selected_index, torch.zeros_like(selected_index)
    )
    selected_velocity = torch.gather(
        candidate_velocity,
        1,
        selected_index[:, None, None, None].expand(-1, 1, agent_count, 2),
    ).squeeze(1)
    selected_radius = torch.gather(
        predicted_radius,
        1,
        selected_index[:, None, None, None].expand(
            -1, 1, prediction_steps, agent_count
        ),
    ).squeeze(1)
    selected_maximum_gap = torch.gather(
        predicted_maximum_gap,
        1,
        selected_index[:, None, None].expand(-1, 1, prediction_steps),
    ).squeeze(1)
    selected_mean_gap_error = torch.gather(
        predicted_mean_gap_error,
        1,
        selected_index[:, None, None].expand(-1, 1, prediction_steps),
    ).squeeze(1)
    selected_violation = torch.gather(
        maximum_violation, 1, selected_index[:, None]
    ).squeeze(1)
    selected_correction_norm = torch.gather(
        correction_norm, 1, selected_index[:, None]
    ).squeeze(1)
    nominal_constraint_count = (
        (radial_violation[:, 0].amax(dim=1) > 1.0e-6).to(torch.long)
        + (maximum_gap_violation[:, 0].amax(dim=1) > 1.0e-6).to(torch.long)
        + (mean_gap_violation[:, 0].amax(dim=1) > 1.0e-6).to(torch.long)
    )
    return PredictiveAngularGapProjectionResult(
        velocity_xy=selected_velocity,
        active=active.clone(),
        corrected=active & (selected_correction_norm > 1.0e-6),
        correction_norm_mps=selected_correction_norm,
        corrected_constraints=torch.where(
            active, nominal_constraint_count, torch.zeros_like(nominal_constraint_count)
        ),
        minimum_predicted_radius_m=selected_radius.amin(dim=(1, 2)),
        maximum_predicted_radius_m=selected_radius.amax(dim=(1, 2)),
        maximum_predicted_gap_rad=selected_maximum_gap.amax(dim=1),
        maximum_predicted_mean_gap_error_rad=selected_mean_gap_error.amax(dim=1),
        infeasible=active & (selected_violation > 1.0e-4),
    )


def project_predictive_target_separation_velocity(
    desired_velocity_xy: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    target_xy: torch.Tensor,
    target_velocity_xy: torch.Tensor,
    active: torch.Tensor,
    *,
    minimum_center_distance_m: float,
    prediction_horizon_s: float,
    command_response_ratio: float,
    maximum_speed_mps: float,
    projection_iterations: int = 2,
    restore_unsafe_buffer: bool = False,
) -> PredictiveTargetProjectionResult:
    """Prevent physical target contact without disturbing the capture ring."""
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if current_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("current velocity and pursuer positions must match")
    batch = pursuer_xy.shape[0]
    if target_xy.shape != (batch, 2) or target_velocity_xy.shape != (batch, 2):
        raise ValueError("target state shape mismatch")
    if active.shape != (batch,):
        raise ValueError("active shape mismatch")
    if isinstance(minimum_center_distance_m, torch.Tensor):
        if minimum_center_distance_m.shape != (batch,):
            raise ValueError("target safety distance tensor must be [batch]")
        if (minimum_center_distance_m <= 0.0).any():
            raise ValueError("target safety distance must be positive")
        safety_distance = minimum_center_distance_m[:, None].to(
            dtype=pursuer_xy.dtype, device=pursuer_xy.device
        )
    else:
        if minimum_center_distance_m <= 0.0:
            raise ValueError("target safety distance must be positive")
        safety_distance = torch.full(
            (batch, 1),
            float(minimum_center_distance_m),
            dtype=pursuer_xy.dtype,
            device=pursuer_xy.device,
        )
    if prediction_horizon_s <= 0.0:
        raise ValueError("target safety horizon must be positive")
    if not 0.0 < command_response_ratio <= 1.0:
        raise ValueError("command_response_ratio must be in (0, 1]")
    if maximum_speed_mps <= 0.0 or projection_iterations <= 0:
        raise ValueError("speed and projection iterations must be positive")

    projected = desired_velocity_xy.clone()
    corrected_constraints = torch.zeros(
        batch, dtype=torch.long, device=pursuer_xy.device
    )
    for _ in range(projection_iterations):
        realized = (
            command_response_ratio * projected
            + (1.0 - command_response_ratio) * current_velocity_xy
        )
        relative_position = pursuer_xy - target_xy[:, None, :]
        distance = relative_position.norm(dim=-1).clamp_min(1.0e-6)
        outward_unit = relative_position / distance[..., None]
        relative_velocity = realized - target_velocity_xy[:, None, :]
        closing_speed = -(relative_velocity * outward_unit).sum(dim=-1)
        distance_margin = distance - safety_distance
        allowed_closing_speed = (
            distance_margin / prediction_horizon_s
            if restore_unsafe_buffer
            else distance_margin.clamp_min(0.0) / prediction_horizon_s
        )
        violation = closing_speed - allowed_closing_speed
        repair = active[:, None] & (violation > 1.0e-6)
        correction = (
            violation.clamp_min(0.0)[..., None]
            / command_response_ratio
            * outward_unit
            * repair[..., None]
        )
        projected = projected + correction
        corrected_constraints += repair.sum(dim=1)
        speed = projected.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        projected = projected * (maximum_speed_mps / speed).clamp_max(1.0)

    realized = (
        command_response_ratio * projected
        + (1.0 - command_response_ratio) * current_velocity_xy
    )
    relative_position = pursuer_xy - target_xy[:, None, :]
    relative_velocity = realized - target_velocity_xy[:, None, :]
    predicted_relative = (
        relative_position + prediction_horizon_s * relative_velocity
    )
    predicted_distance = predicted_relative.norm(dim=-1)

    # The iterative half-space repair above linearizes around the current
    # pursuer--target ray.  A fast target can rotate the future relative ray
    # enough that the linearized iterations leave a large residual even when
    # the speed-bounded command problem is feasible.  In future-relative
    # coordinates the command contribution is an isotropic ball, so the
    # maximum attainable separation has a closed form.  For any residual,
    # command maximum speed along that ball centre's outward ray.  This is an
    # exact maximum-separation best effort: it satisfies every feasible target
    # constraint and exposes genuinely unavoidable ones instead of conflating
    # them with numerical projection failure.
    future_without_new_command = relative_position + prediction_horizon_s * (
        (1.0 - command_response_ratio) * current_velocity_xy
        - target_velocity_xy[:, None, :]
    )
    controllable_radius = (
        prediction_horizon_s * command_response_ratio * maximum_speed_mps
    )
    maximum_attainable_distance = (
        future_without_new_command.norm(dim=-1) + controllable_radius
    )
    residual = active[:, None] & restore_unsafe_buffer & (
        predicted_distance < safety_distance - 1.0e-4
    )
    fallback_direction_norm = future_without_new_command.norm(
        dim=-1, keepdim=True
    )
    fallback_direction = future_without_new_command / fallback_direction_norm.clamp_min(
        1.0e-6
    )
    current_outward = relative_position / relative_position.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    desired_outward = desired_velocity_xy / desired_velocity_xy.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    canonical_outward = torch.zeros_like(fallback_direction)
    canonical_outward[..., 0] = 1.0
    fallback_direction = torch.where(
        fallback_direction_norm > 1.0e-6,
        fallback_direction,
        torch.where(
            relative_position.norm(dim=-1, keepdim=True) > 1.0e-6,
            current_outward,
            torch.where(
                desired_velocity_xy.norm(dim=-1, keepdim=True) > 1.0e-6,
                desired_outward,
                canonical_outward,
            ),
        ),
    )
    maximum_separation_velocity = maximum_speed_mps * fallback_direction
    projected = torch.where(
        residual[..., None], maximum_separation_velocity, projected
    )
    maximum_separation_fallback_constraints = residual.sum(dim=1)
    corrected_constraints += maximum_separation_fallback_constraints

    realized = (
        command_response_ratio * projected
        + (1.0 - command_response_ratio) * current_velocity_xy
    )
    relative_velocity = realized - target_velocity_xy[:, None, :]
    predicted_distance = (
        relative_position + prediction_horizon_s * relative_velocity
    ).norm(dim=-1)
    minimum_predicted_distance = predicted_distance.min(dim=1).values
    unavoidable_constraint = active[:, None] & (
        maximum_attainable_distance < safety_distance - 1.0e-4
    )
    unavoidable_constraints = unavoidable_constraint.sum(dim=1)
    unavoidable = active & (unavoidable_constraints > 0)
    solver_residual = active[:, None] & (
        predicted_distance < safety_distance - 1.0e-4
    ) & ~unavoidable_constraint
    correction_norm = (projected - desired_velocity_xy).flatten(start_dim=1).norm(dim=1)
    return PredictiveTargetProjectionResult(
        velocity_xy=projected,
        active=active.clone(),
        corrected=active & (correction_norm > 1.0e-6),
        correction_norm_mps=correction_norm,
        corrected_constraints=corrected_constraints,
        maximum_separation_fallback_constraints=(
            maximum_separation_fallback_constraints
        ),
        minimum_predicted_center_distance_m=minimum_predicted_distance,
        minimum_maximum_attainable_center_distance_m=(
            maximum_attainable_distance.min(dim=1).values
        ),
        unavoidable=unavoidable,
        unavoidable_constraints=unavoidable_constraints,
        infeasible=active & solver_residual.any(dim=1),
    )


def project_predictive_static_obstacle_velocity(
    desired_velocity_xy: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radius_m: torch.Tensor,
    active: torch.Tensor,
    *,
    obstacle_velocity_xy: torch.Tensor | None = None,
    pursuer_radius_m: float,
    minimum_clearance_m: float,
    prediction_horizon_s: float,
    command_response_ratio: float,
    maximum_speed_mps: float,
    projection_iterations: int,
) -> PredictiveObstacleProjectionResult:
    """Project commands onto obstacle-relative radial safety half-spaces."""
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if current_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("current velocity and pursuer positions must match")
    if obstacle_xy.ndim != 3 or obstacle_xy.shape[-1] != 2:
        raise ValueError("obstacle_xy must have shape [batch, obstacle, 2]")
    if obstacle_radius_m.shape != obstacle_xy.shape[:2]:
        raise ValueError("obstacle radius shape mismatch")
    if obstacle_velocity_xy is None:
        obstacle_velocity_xy = torch.zeros_like(obstacle_xy)
    if obstacle_velocity_xy.shape != obstacle_xy.shape:
        raise ValueError("obstacle velocity shape mismatch")
    if obstacle_xy.shape[0] != pursuer_xy.shape[0]:
        raise ValueError("obstacle and pursuer batch dimensions must match")
    if active.shape != (pursuer_xy.shape[0],):
        raise ValueError("active shape mismatch")
    if pursuer_radius_m <= 0.0 or minimum_clearance_m <= 0.0:
        raise ValueError("vehicle radius and safety clearance must be positive")
    if prediction_horizon_s <= 0.0 or maximum_speed_mps <= 0.0:
        raise ValueError("prediction horizon and speed must be positive")
    if not 0.0 < command_response_ratio <= 1.0:
        raise ValueError("command_response_ratio must be in (0, 1]")
    if projection_iterations <= 0:
        raise ValueError("projection_iterations must be positive")

    projected = desired_velocity_xy.clone()
    correction_count = torch.zeros(
        pursuer_xy.shape[0], dtype=torch.long, device=pursuer_xy.device
    )
    for _ in range(projection_iterations):
        for obstacle_index in range(obstacle_xy.shape[1]):
            realized = (
                command_response_ratio * projected
                + (1.0 - command_response_ratio) * current_velocity_xy
            )
            relative_position = (
                pursuer_xy - obstacle_xy[:, obstacle_index, None, :]
            )
            distance = relative_position.norm(dim=-1).clamp_min(1.0e-6)
            outward_unit = relative_position / distance[..., None]
            relative_velocity = (
                realized
                - obstacle_velocity_xy[:, obstacle_index, None, :]
            )
            closing_speed = -(relative_velocity * outward_unit).sum(dim=-1)
            safe_center_distance = (
                obstacle_radius_m[:, obstacle_index, None]
                + pursuer_radius_m
                + minimum_clearance_m
            )
            allowed_closing_speed = (
                (distance - safe_center_distance).clamp_min(0.0)
                / prediction_horizon_s
            )
            violation = closing_speed - allowed_closing_speed
            repair = active[:, None] & (violation > 1.0e-6)
            correction = (
                violation.clamp_min(0.0)[..., None]
                / command_response_ratio
                * outward_unit
                * repair[..., None]
            )
            projected = projected + correction
            correction_count += repair.sum(dim=1)
        speed = projected.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        projected = projected * (maximum_speed_mps / speed).clamp_max(1.0)

    realized = (
        command_response_ratio * projected
        + (1.0 - command_response_ratio) * current_velocity_xy
    )
    relative = pursuer_xy[:, :, None, :] - obstacle_xy[:, None, :, :]
    relative_velocity = (
        realized[:, :, None, :] - obstacle_velocity_xy[:, None, :, :]
    )
    predicted_relative = relative + prediction_horizon_s * relative_velocity
    predicted_clearance = (
        predicted_relative.norm(dim=-1)
        - obstacle_radius_m[:, None, :]
        - pursuer_radius_m
    )
    distance = relative.norm(dim=-1).clamp_min(1.0e-6)
    outward_unit = relative / distance[..., None]
    closing_speed = -(relative_velocity * outward_unit).sum(dim=-1)
    safe_center_distance = (
        obstacle_radius_m[:, None, :]
        + pursuer_radius_m
        + minimum_clearance_m
    )
    allowed_closing_speed = (
        (distance - safe_center_distance).clamp_min(0.0) / prediction_horizon_s
    )
    residual = closing_speed - allowed_closing_speed
    correction_norm = (projected - desired_velocity_xy).flatten(start_dim=1).norm(dim=1)
    return PredictiveObstacleProjectionResult(
        velocity_xy=projected,
        active=active.clone(),
        corrected=active & (correction_norm > 1.0e-6),
        correction_norm_mps=correction_norm,
        corrected_constraints=correction_count,
        minimum_predicted_clearance_m=predicted_clearance.flatten(1).min(dim=1).values,
        infeasible=active & (residual.flatten(1).max(dim=1).values > 1.0e-4),
    )


def resolve_static_obstacle_tangent_velocity(
    desired_velocity_xy: torch.Tensor,
    current_velocity_xy: torch.Tensor,
    pursuer_xy: torch.Tensor,
    estimated_target_xy: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radius_m: torch.Tensor,
    active: torch.Tensor,
    previous_obstacle_index: torch.Tensor,
    previous_side: torch.Tensor,
    previous_blocked_steps: torch.Tensor,
    previous_hold_steps_remaining: torch.Tensor,
    *,
    pursuer_radius_m: float,
    minimum_clearance_m: float,
    prediction_horizon_s: float,
    command_response_ratio: float,
    maximum_speed_mps: float,
    activation_steps: int,
    hold_steps: int,
    tangent_speed_mps: float,
    outward_speed_mps: float,
    release_distance_m: float,
    target_progress_weight: float = 1.0,
    obstacle_clearance_weight: float = 0.25,
    teammate_clearance_weight: float = 0.20,
) -> StaticTangentBypassResult:
    """Resolve repeated radial obstacle blocking with a stable tangent command.

    A short persistence gate distinguishes a real obstacle deadlock from a
    one-step safety correction.  Once activated, two left/right tangent
    candidates are scored by target progress, predicted obstacle clearance,
    and predicted teammate clearance.  The selected side is retained for the
    same obstacle until the hysteresis window and release-distance tests allow
    it to clear, which prevents left/right chattering.
    """
    if desired_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("desired velocity and pursuer positions must match")
    if current_velocity_xy.shape != pursuer_xy.shape:
        raise ValueError("current velocity and pursuer positions must match")
    if pursuer_xy.ndim != 3 or pursuer_xy.shape[-1] != 2:
        raise ValueError("pursuer tensors must have shape [batch, agent, 2]")
    if estimated_target_xy.shape != (pursuer_xy.shape[0], 2):
        raise ValueError("estimated_target_xy shape mismatch")
    if obstacle_xy.ndim != 3 or obstacle_xy.shape[-1] != 2:
        raise ValueError("obstacle_xy must have shape [batch, obstacle, 2]")
    if obstacle_radius_m.shape != obstacle_xy.shape[:2]:
        raise ValueError("obstacle radius shape mismatch")
    state_shape = pursuer_xy.shape[:2]
    for name, value in (
        ("previous_obstacle_index", previous_obstacle_index),
        ("previous_side", previous_side),
        ("previous_blocked_steps", previous_blocked_steps),
        ("previous_hold_steps_remaining", previous_hold_steps_remaining),
    ):
        if value.shape != state_shape:
            raise ValueError(f"{name} shape mismatch")
    if active.shape != (pursuer_xy.shape[0],):
        raise ValueError("active shape mismatch")
    if pursuer_radius_m <= 0.0 or minimum_clearance_m <= 0.0:
        raise ValueError("vehicle radius and safety clearance must be positive")
    if prediction_horizon_s <= 0.0 or maximum_speed_mps <= 0.0:
        raise ValueError("prediction horizon and speed must be positive")
    if not 0.0 < command_response_ratio <= 1.0:
        raise ValueError("command_response_ratio must be in (0, 1]")
    if activation_steps <= 0 or hold_steps <= 0:
        raise ValueError("activation and hold steps must be positive")
    if tangent_speed_mps <= 0.0 or outward_speed_mps < 0.0:
        raise ValueError("tangent speed must be positive and outward speed nonnegative")
    if release_distance_m <= 0.0:
        raise ValueError("release distance must be positive")
    if min(
        target_progress_weight,
        obstacle_clearance_weight,
        teammate_clearance_weight,
    ) < 0.0:
        raise ValueError("candidate weights must be nonnegative")

    batch_size, agent_count, _ = pursuer_xy.shape
    obstacle_count = obstacle_xy.shape[1]
    relative = pursuer_xy[:, :, None, :] - obstacle_xy[:, None, :, :]
    distance = relative.norm(dim=-1).clamp_min(1.0e-6)
    outward_unit = relative / distance[..., None]
    realized_nominal = (
        command_response_ratio * desired_velocity_xy
        + (1.0 - command_response_ratio) * current_velocity_xy
    )
    closing_speed = -(
        realized_nominal[:, :, None, :] * outward_unit
    ).sum(dim=-1)
    safe_center_distance = (
        obstacle_radius_m[:, None, :]
        + pursuer_radius_m
        + minimum_clearance_m
    )
    allowed_closing_speed = (
        (distance - safe_center_distance).clamp_min(0.0)
        / prediction_horizon_s
    )
    violation = closing_speed - allowed_closing_speed
    maximum_violation, blocking_obstacle = violation.max(dim=2)
    blocking = active[:, None] & (maximum_violation > 1.0e-6)
    same_blocker = blocking_obstacle == previous_obstacle_index
    blocked_steps = torch.where(
        blocking,
        torch.where(
            same_blocker,
            previous_blocked_steps + 1,
            torch.ones_like(previous_blocked_steps),
        ),
        torch.zeros_like(previous_blocked_steps),
    )

    state_index = previous_obstacle_index.clamp(0, obstacle_count - 1)
    previous_distance = distance.gather(2, state_index[..., None]).squeeze(2)
    previous_safe_distance = safe_center_distance.expand(
        batch_size, agent_count, obstacle_count
    ).gather(2, state_index[..., None]).squeeze(2)
    previous_valid = previous_obstacle_index >= 0
    previous_near = (
        previous_valid
        & (previous_distance <= previous_safe_distance + release_distance_m)
    )
    holding = (
        active[:, None]
        & previous_near
        & (previous_side != 0)
        & (previous_hold_steps_remaining > 0)
    )
    newly_activated = blocking & (blocked_steps >= activation_steps) & ~holding
    bypass_active = holding | newly_activated
    selected_obstacle = torch.where(
        holding, previous_obstacle_index, blocking_obstacle
    )
    selected_index = selected_obstacle.clamp(0, obstacle_count - 1)
    gather_vector_index = selected_index[..., None, None].expand(-1, -1, 1, 2)
    selected_outward = outward_unit.gather(2, gather_vector_index).squeeze(2)

    left_tangent = torch.stack(
        (-selected_outward[..., 1], selected_outward[..., 0]), dim=-1
    )
    desired_outward_speed = (
        desired_velocity_xy * selected_outward
    ).sum(dim=-1).clamp(0.0, outward_speed_mps)
    outward_component = desired_outward_speed[..., None] * selected_outward
    left_candidate = outward_component + tangent_speed_mps * left_tangent
    right_candidate = outward_component - tangent_speed_mps * left_tangent
    candidates = torch.stack((left_candidate, right_candidate), dim=2)
    candidate_speed = candidates.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    candidates = candidates * (maximum_speed_mps / candidate_speed).clamp_max(1.0)

    target_direction = estimated_target_xy[:, None, :] - pursuer_xy
    target_direction = target_direction / target_direction.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    target_progress = (
        candidates * target_direction[:, :, None, :]
    ).sum(dim=-1) / maximum_speed_mps
    realized_candidates = (
        command_response_ratio * candidates
        + (1.0 - command_response_ratio) * current_velocity_xy[:, :, None, :]
    )
    predicted_position = (
        pursuer_xy[:, :, None, :]
        + prediction_horizon_s * realized_candidates
    )
    candidate_obstacle_clearance = (
        (
            predicted_position[:, :, :, None, :]
            - obstacle_xy[:, None, None, :, :]
        ).norm(dim=-1)
        - obstacle_radius_m[:, None, None, :]
        - pursuer_radius_m
    ).min(dim=3).values

    predicted_other_nominal = (
        pursuer_xy
        + prediction_horizon_s
        * (
            command_response_ratio * desired_velocity_xy
            + (1.0 - command_response_ratio) * current_velocity_xy
        )
    )
    teammate_relative = (
        predicted_position[:, :, :, None, :]
        - predicted_other_nominal[:, None, None, :, :]
    )
    teammate_distance = teammate_relative.norm(dim=-1)
    identity = torch.eye(
        agent_count, dtype=torch.bool, device=pursuer_xy.device
    )[None, :, None, :]
    teammate_distance = teammate_distance.masked_fill(identity, torch.inf)
    if agent_count > 1:
        candidate_teammate_clearance = teammate_distance.min(dim=3).values
    else:
        candidate_teammate_clearance = torch.full_like(
            candidate_obstacle_clearance, maximum_speed_mps
        )
    score = (
        target_progress_weight * target_progress
        + obstacle_clearance_weight
        * candidate_obstacle_clearance.clamp(-minimum_clearance_m, 5.0)
        / 5.0
        + teammate_clearance_weight
        * candidate_teammate_clearance.clamp(0.0, 10.0)
        / 10.0
    )
    # Deterministically break exact symmetry in alternating directions so a
    # group does not choose the same side around a perfectly radial obstacle.
    parity_prefers_left = (
        torch.arange(agent_count, device=pursuer_xy.device) % 2 == 0
    )[None, :]
    score[:, :, 0] += parity_prefers_left.float() * 1.0e-4
    score[:, :, 1] += (~parity_prefers_left).float() * 1.0e-4
    scored_side = torch.where(
        score[:, :, 0] >= score[:, :, 1],
        torch.ones_like(previous_side),
        -torch.ones_like(previous_side),
    )
    selected_side = torch.where(holding, previous_side, scored_side)
    selected_candidate = torch.where(
        (selected_side > 0)[..., None], candidates[:, :, 0], candidates[:, :, 1]
    )
    velocity = torch.where(
        bypass_active[..., None], selected_candidate, desired_velocity_xy
    )

    hold_steps_remaining = torch.where(
        newly_activated,
        torch.full_like(previous_hold_steps_remaining, hold_steps),
        torch.where(
            holding,
            (previous_hold_steps_remaining - 1).clamp_min(0),
            torch.zeros_like(previous_hold_steps_remaining),
        ),
    )
    output_obstacle_index = torch.where(
        bypass_active | blocking,
        torch.where(holding, previous_obstacle_index, blocking_obstacle),
        torch.full_like(previous_obstacle_index, -1),
    )
    output_side = torch.where(
        bypass_active, selected_side, torch.zeros_like(previous_side)
    )
    selected_clearance = torch.where(
        (selected_side > 0),
        candidate_obstacle_clearance[:, :, 0],
        candidate_obstacle_clearance[:, :, 1],
    )
    return StaticTangentBypassResult(
        velocity_xy=velocity,
        active=bypass_active,
        activated=newly_activated,
        obstacle_index=output_obstacle_index,
        side=output_side,
        blocked_steps=blocked_steps,
        hold_steps_remaining=hold_steps_remaining,
        minimum_candidate_clearance_m=torch.where(
            bypass_active,
            selected_clearance,
            torch.full_like(selected_clearance, torch.inf),
        ),
    )


def advance_orientation_gate(
    distance_to_target_m: torch.Tensor,
    visible_each: torch.Tensor,
    yaw_error_rad: torch.Tensor,
    staging_reached: torch.Tensor,
    alignment_steps: torch.Tensor,
    contraction_unlocked: torch.Tensor,
    *,
    staging_radius_m: float,
    yaw_tolerance_rad: float,
    required_alignment_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the persistent staging/alignment/contraction state machine."""
    if distance_to_target_m.shape != visible_each.shape:
        raise ValueError("distance and visibility shapes must match")
    if yaw_error_rad.shape != distance_to_target_m.shape:
        raise ValueError("yaw_error_rad shape mismatch")
    batch = distance_to_target_m.shape[0]
    for name, value in (
        ("staging_reached", staging_reached),
        ("alignment_steps", alignment_steps),
        ("contraction_unlocked", contraction_unlocked),
    ):
        if value.shape != (batch,):
            raise ValueError(f"{name} shape mismatch")
    if staging_radius_m <= 0.0 or yaw_tolerance_rad <= 0.0:
        raise ValueError("staging radius and yaw tolerance must be positive")
    if required_alignment_steps <= 0:
        raise ValueError("required_alignment_steps must be positive")
    maximum_distance = distance_to_target_m.max(dim=1).values
    reached_now = maximum_distance <= staging_radius_m
    new_staging_reached = staging_reached | reached_now
    aligned_now = (
        new_staging_reached
        & (maximum_distance <= staging_radius_m)
        & visible_each.all(dim=1)
        & (yaw_error_rad.abs().max(dim=1).values <= yaw_tolerance_rad)
    )
    new_alignment_steps = torch.where(
        contraction_unlocked,
        alignment_steps,
        torch.where(
            aligned_now,
            alignment_steps + 1,
            torch.zeros_like(alignment_steps),
        ),
    )
    new_unlocked = contraction_unlocked | (
        new_alignment_steps >= required_alignment_steps
    )
    return new_staging_reached, new_alignment_steps, new_unlocked, aligned_now


def advance_asynchronous_orientation_gate(
    distance_to_target_m: torch.Tensor,
    visible_each: torch.Tensor,
    yaw_error_rad: torch.Tensor,
    staging_reached_each: torch.Tensor,
    alignment_steps_each: torch.Tensor,
    contraction_unlocked: torch.Tensor,
    *,
    staging_radius_m: float,
    yaw_tolerance_rad: float,
    required_alignment_steps: int,
    require_current_team_ready_to_unlock: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Latch per-agent staging/alignment before team contraction.

    Unlike the legacy all-at-once gate, each pursuer can finish its staging and
    target-facing handoff at a different time.  Team contraction still unlocks
    only after every pursuer has independently satisfied the same distance,
    visibility, yaw and hold requirements.
    """
    if distance_to_target_m.shape != visible_each.shape:
        raise ValueError("distance and visibility shapes must match")
    if yaw_error_rad.shape != distance_to_target_m.shape:
        raise ValueError("yaw_error_rad shape mismatch")
    if staging_reached_each.shape != distance_to_target_m.shape:
        raise ValueError("staging_reached_each shape mismatch")
    if alignment_steps_each.shape != distance_to_target_m.shape:
        raise ValueError("alignment_steps_each shape mismatch")
    if contraction_unlocked.shape != (distance_to_target_m.shape[0],):
        raise ValueError("contraction_unlocked shape mismatch")
    if staging_radius_m <= 0.0 or yaw_tolerance_rad <= 0.0:
        raise ValueError("staging radius and yaw tolerance must be positive")
    if required_alignment_steps <= 0:
        raise ValueError("required_alignment_steps must be positive")

    inside_now = distance_to_target_m <= staging_radius_m
    new_staging_each = staging_reached_each | inside_now
    aligned_now_each = (
        inside_now
        & visible_each
        & (yaw_error_rad.abs() <= yaw_tolerance_rad)
    )
    already_aligned = alignment_steps_each >= required_alignment_steps
    new_alignment_each = torch.where(
        already_aligned | contraction_unlocked[:, None],
        alignment_steps_each,
        torch.where(
            aligned_now_each,
            alignment_steps_each + 1,
            torch.zeros_like(alignment_steps_each),
        ),
    ).clamp_max(required_alignment_steps)
    all_historically_aligned = (
        new_alignment_each >= required_alignment_steps
    ).all(dim=1)
    if require_current_team_ready_to_unlock:
        all_currently_ready = aligned_now_each.all(dim=1)
        new_unlocked = contraction_unlocked | (
            all_historically_aligned & all_currently_ready
        )
    else:
        new_unlocked = contraction_unlocked | all_historically_aligned
    team_staging_reached = new_staging_each.all(dim=1)
    team_aligned_now = aligned_now_each.all(dim=1)
    return (
        new_staging_each,
        new_alignment_each,
        team_staging_reached,
        new_unlocked,
        team_aligned_now,
    )


def native_team_reward(
    pursuer_position: torch.Tensor,
    target_position: torch.Tensor,
    pursuer_velocity: torch.Tensor,
    normalized_action: torch.Tensor,
    success: torch.Tensor,
    contact: torch.Tensor,
    agent_collision: torch.Tensor,
    obstacle_collision: torch.Tensor,
    boundary_invalid: torch.Tensor,
    height_invalid: torch.Tensor,
    team_visible: torch.Tensor,
    minimum_static_clearance_m: torch.Tensor,
    minimum_boundary_clearance_m: torch.Tensor,
    *,
    previous_team_distance_m: torch.Tensor | None = None,
    previous_angular_error_rad: torch.Tensor | None = None,
    desired_radius_m: float | torch.Tensor = 4.5,
    near_switch_m: float = 9.0,
    near_gate_width_m: float = 1.5,
    command_speed_mps: float = 4.0,
    policy_dt_s: float = 0.1,
    fixed_altitude_m: float = 3.0,
    safe_agent_center_distance_m: float = 2.5,
    safe_static_clearance_m: float = 1.5,
    safe_boundary_clearance_m: float = 5.0,
    safety_smoothing_m: float = 0.25,
    team_visibility_bonus: float = 0.01,
    angular_error_weight: float = 0.35,
    angular_progress_weight: float = 0.0,
) -> NativeRewardTerms:
    """Return normalized approach/formation/safety reward terms.

    Distance progress supplies an immediate sign for whether the last action
    improved the pursuit. Formation errors are gated on only near the target,
    following the distance-first/formation-second curriculum used in the
    reviewed pursuit literature.
    """
    desired_radius = torch.as_tensor(
        desired_radius_m, dtype=pursuer_position.dtype, device=pursuer_position.device
    )
    if desired_radius.ndim == 0:
        desired_radius_per_environment = desired_radius.expand(pursuer_position.shape[0])
    elif desired_radius.shape == (pursuer_position.shape[0],):
        desired_radius_per_environment = desired_radius
    else:
        raise ValueError("desired_radius_m must be scalar or have shape [batch]")
    if (desired_radius_per_environment <= 0.0).any() or near_gate_width_m <= 0.0:
        raise ValueError("reward radii and gate width must be positive")
    progress_scale = command_speed_mps * policy_dt_s
    if progress_scale <= 0.0:
        raise ValueError("command_speed_mps and policy_dt_s must be positive")
    if (
        team_visibility_bonus < 0.0
        or angular_error_weight < 0.0
        or angular_progress_weight < 0.0
    ):
        raise ValueError("reward weights must be nonnegative")

    pursuer_xy = pursuer_position[..., :2]
    target_xy = target_position[..., :2]
    distance = (pursuer_xy - target_xy[:, None, :]).norm(dim=-1)
    team_distance = distance.mean(dim=1)
    if previous_team_distance_m is None:
        previous_team_distance_m = team_distance.detach()
    if previous_team_distance_m.shape != team_distance.shape:
        raise ValueError("previous_team_distance_m shape mismatch")
    distance_progress = (
        (previous_team_distance_m - team_distance) / progress_scale
    ).clamp(-1.0, 1.0)
    near_gate = torch.sigmoid((near_switch_m - team_distance) / near_gate_width_m)

    radial_error = (
        distance - desired_radius_per_environment[:, None]
    ).abs().mean(dim=1)
    radial_error_normalized = radial_error / desired_radius_per_environment
    angular_error = circular_gap_error(pursuer_xy, target_xy)
    desired_gap = 2.0 * math.pi / pursuer_xy.shape[1]
    angular_error_normalized = angular_error / desired_gap
    if previous_angular_error_rad is None:
        angular_progress_normalized = torch.zeros_like(angular_error)
    else:
        if previous_angular_error_rad.shape != angular_error.shape:
            raise ValueError("previous_angular_error_rad shape mismatch")
        angular_progress_normalized = (
            (previous_angular_error_rad - angular_error) / desired_gap
        ).clamp(-1.0, 1.0)
    height_error = (pursuer_position[..., 2] - fixed_altitude_m).abs().mean(dim=1)
    vertical_speed = pursuer_velocity[..., 2].abs().mean(dim=1)

    pair_distances = []
    for left in range(pursuer_xy.shape[1]):
        for right in range(left + 1, pursuer_xy.shape[1]):
            pair_distances.append(
                (pursuer_xy[:, left] - pursuer_xy[:, right]).norm(dim=-1)
            )
    minimum_agent_distance = torch.stack(pair_distances, dim=1).min(dim=1).values
    agent_proximity = smooth_clearance_barrier(
        minimum_agent_distance,
        safe_agent_center_distance_m,
        safety_smoothing_m,
    )
    obstacle_proximity = smooth_clearance_barrier(
        minimum_static_clearance_m,
        safe_static_clearance_m,
        safety_smoothing_m,
    )
    boundary_proximity = smooth_clearance_barrier(
        minimum_boundary_clearance_m,
        safe_boundary_clearance_m,
        safety_smoothing_m,
    )
    action_energy = normalized_action.square().mean(dim=(1, 2))

    reward = (
        +1.00 * distance_progress
        -0.25 * near_gate * radial_error_normalized
        -angular_error_weight * near_gate * angular_error_normalized
        +angular_progress_weight * near_gate * angular_progress_normalized
        -0.10 * height_error
        -0.02 * vertical_speed
        -1.00 * agent_proximity
        -0.75 * obstacle_proximity
        -0.50 * boundary_proximity
        -0.01 * action_energy
        +team_visibility_bonus * team_visible.float()
        +5.00 * success.float()
        -5.00 * contact.float()
        -5.00 * (agent_collision | obstacle_collision).float()
        -5.00 * (boundary_invalid | height_invalid).float()
    )
    return NativeRewardTerms(
        reward=reward,
        team_distance_m=team_distance,
        distance_progress_normalized=distance_progress,
        near_gate=near_gate,
        radial_error_m=radial_error,
        radial_error_normalized=radial_error_normalized,
        angular_error_rad=angular_error,
        angular_error_normalized=angular_error_normalized,
        angular_progress_normalized=angular_progress_normalized,
        height_error_m=height_error,
        vertical_speed_mps=vertical_speed,
        agent_proximity_penalty=agent_proximity,
        obstacle_proximity_penalty=obstacle_proximity,
        boundary_proximity_penalty=boundary_proximity,
        action_energy=action_energy,
    )
