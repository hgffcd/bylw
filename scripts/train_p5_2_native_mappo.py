#!/usr/bin/env python3
"""Train dynamics-aware MAPPO directly in the guarded Isaac P5 task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--stage",
    choices=(
        "a0", "a1", "a2", "b", "c0", "c1", "c2", "c3", "c4", "c5",
        "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30", "c31", "c32", "c40", "c41", "c42", "c43", "c44", "c45", "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55", "c56", "c57", "c58", "c60", "c61", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79", "c80", "c81", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91",
    ),
    required=True,
)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--rollout_steps", type=int, default=64)
parser.add_argument("--eval_every", type=int, default=25)
parser.add_argument("--eval_episodes", type=int, default=32)
parser.add_argument("--checkpoint_every", type=int, default=25)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--resume", type=Path)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--physical_gpu_index", type=int, required=True)
parser.add_argument("--memory_limit_mb", type=int, default=12000)
parser.add_argument("--max_wall_hours", type=float, default=6.0)
parser.add_argument("--target_success", type=float, default=0.70)
parser.add_argument("--max_collision", type=float, default=0.05)
parser.add_argument("--required_passes", type=int, default=2)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--actor_learning_rate", type=float, default=None,
    help="Override actor LR only; critic uses learning_rate. Resume must match saved optimizer rates or reset it.")
parser.add_argument("--expected_initial_evaluation", type=Path, default=None,
    help="Require exact matched evaluation-summary metrics inside the trainer before the first PPO update")
parser.add_argument("--ppo_epochs", type=int, default=5)
parser.add_argument("--minibatches", type=int, default=8)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--episode_length_s", type=float, default=20.0)
parser.add_argument("--region_capture", action="store_true",
                    help="C97 explicit 5.5m/150deg first-hit region protocol")
parser.add_argument("--region_objective", action="store_true",
                    help="align reward and control with region enclosure; requires --region_capture")
parser.add_argument("--ppo_exact_logprob", action="store_true",
                    help="store pre-tanh samples for consistent PPO probability ratios")
parser.add_argument("--allow_capture_protocol_migration", action="store_true",
                    help="explicitly permit parent weights to enter another named task protocol")
parser.add_argument("--dual_observer_formation", action="store_true",
                    help="C98 measured observer-pair anchor and soft role regions")
parser.add_argument("--dual_observer_role_lock", action="store_true",
                    help="C99 keep the current observer pair until it is invalid")
parser.add_argument("--region_gap_reserve", action="store_true",
                    help="C100 start angular feedback at 135deg; capture stays 150deg")
parser.add_argument("--target_path_guard_extended", action="store_true",
    help="C106: extend only the final CV/CA target safety path from 0.5 s to 1.0 s")
parser.add_argument("--target_path_guard_margin", action="store_true",
    help="C107: maximize worst target distance among equally repairing admissible commands")
parser.add_argument("--target_path_guard_reserve", action="store_true",
    help="C109: add0.5m prediction reserve; actual target failure/capture buffer stays1.5m")
parser.add_argument("--target_buffer_failure", action="store_true", help="C108: actual target distance below 1.5m is a separate terminal safety failure")
parser.add_argument("--allow_target_buffer_failure_migration", action="store_true", help="Explicitly migrate the training termination/reward contract")
parser.add_argument("--frame_accurate_loss_counters", action="store_true", help="Count visibility history once per physical high-level frame and environment")
parser.add_argument("--allow_loss_counter_migration", action="store_true", help="Explicitly migrate the observation history clock")
parser.add_argument("--learning_diagnostics", action="store_true",
    help="C108 read-only active-sample PPO and slot action/command diagnostics")
parser.add_argument("--target_path_guard_acceleration", action="store_true",
    help="C105: retain CV and add bounded acceleration from consecutive visible target measurements")
parser.add_argument("--target_path_guard", action="store_true",
    help="C104 final joint guard using the 0.5 s acceleration-limited swept path")
parser.add_argument("--region_joint_lookahead", action="store_true",
                    help="C101 bounded joint radial/phase search with a 0.5s dynamics predictor")
parser.add_argument("--allow_controller_migration", action="store_true",
                    help="explicitly permit weights to enter/leave a named controller arm")
parser.add_argument(
    "--reset_optimizer_on_resume",
    action="store_true",
    help="load actor/critic weights but start a fresh optimizer for fine-tuning",
)
parser.add_argument(
    "--reset_actor_output_on_resume",
    action="store_true",
    help=(
        "zero the actor mean head and restore log_std=-0.5 after loading; "
        "required when C32 changes the physical action semantics"
    ),
)
parser.add_argument(
    "--evaluation_only",
    action="store_true",
    help="load --resume and write matched per-episode safety evidence without updating",
)
parser.add_argument(
    "--evaluation_seed_plan",
    type=Path,
    help="CSV containing an explicit unique episode_seed column for matched evaluation",
)
parser.add_argument(
    "--evaluation_trace",
    action="store_true",
    help="write per-control-step geometry/safety telemetry for evaluation seeds",
)
parser.add_argument(
    "--preentry_radial_priority",
    action="store_true",
    help=(
        "C93 development arm: when exact angular entry constraints are met, "
        "allocate the remaining equal-speed correction budget to radial "
        "annulus entry before tangential refinement"
    ),
)
parser.add_argument(
    "--c96_near_ring_reset_curriculum",
    action="store_true",
    help="C96 training-only 5.6--6.2 m reset; formal eval always uses the original reset",
)
parser.add_argument(
    "--c94_extended_episode_horizon",
    action="store_true",
    help=(
        "C94 preregistered arm: permit only stage=c91 with the explicit "
        "45 s episode horizon; the default C91 contract remains frozen at 30 s"
    ),
)
parser.add_argument(
    "--evaluation_capture_radius_m",
    type=float,
    choices=(5.0, 6.0),
    help=(
        "evaluation-only termination radius; use a separate 5 m replay because "
        "the normal 6 m terminal otherwise censors the strict criterion"
    ),
)
parser.add_argument(
    "--evaluation_sense_range_m",
    type=float,
    choices=(25.0, 30.0, 50.0, 100.0),
    help=(
        "evaluation-only sensor-range override; reset geometry remains tied "
        "to tracked_reset_reference_range_m for a paired visibility audit"
    ),
)
parser.add_argument(
    "--evaluation_tracked_reset_reference_range_m",
    type=float,
    choices=(12.0, 15.0, 20.0, 25.0),
    help=(
        "evaluation-only C40 curriculum diagnostic: change only the tracked "
        "initial pursuer ring while retaining the physical sensor range"
    ),
)
parser.add_argument(
    "--evaluation_evader_speed_mps",
    type=float,
    choices=(1.0, 1.5, 2.0, 4.0),
    help=(
        "evaluation-only single-variable evader speed override; 4.0 m/s is "
        "the strict equal-speed curriculum diagnostic"
    ),
)
parser.add_argument(
    "--evaluation_dynamic_obstacle_speed_mps",
    type=float,
    choices=(0.6, 1.0),
    help="evaluation-only single-variable dynamic-obstacle speed override",
)
parser.add_argument(
    "--evaluation_ignore_dynamic_obstacle_velocity",
    action="store_true",
    help=(
        "evaluation-only ablation: moving obstacles remain physical and observed, "
        "but the predictive obstacle projection treats their velocity as zero"
    ),
)
parser.add_argument(
    "--evaluation_slot_prediction_horizon_s",
    type=float,
    choices=(0.25, 0.5),
    help=(
        "evaluation-only CV lead for the moving slot centre; capture remains "
        "defined around the current physical target"
    ),
)
parser.add_argument(
    "--evaluation_loss_reacquisition",
    choices=("off", "last", "flank", "sector"),
    help=(
        "evaluation-only P5-L0/P5-L2/P5-L3 treatment: track loss/reacquisition "
        "events and optionally compose Last, occlusion-flank, or bounded-sector guidance"
    ),
)
parser.add_argument(
    "--evaluation_pairwise_recovery",
    choices=("legacy", "restorative"),
    help=(
        "evaluation-only P5-S1 ablation: retain the legacy pairwise safety "
        "buffer or actively restore an already-violated buffer"
    ),
)
parser.add_argument(
    "--evaluation_contraction_speed_mps",
    type=float,
    choices=(1.0, 1.5, 2.0),
    help=(
        "evaluation-only C28 time-budget ablation; overrides both the "
        "reference-radius contraction rate and relative inward-speed cap"
    ),
)
parser.add_argument(
    "--evaluation_orientation_staging_radius_m",
    type=float,
    choices=(12.0, 15.0),
    help=(
        "evaluation-only C40 time-budget repair: start the moving-role ring "
        "and its contraction before the legacy 9.5 m readiness boundary"
    ),
)
parser.add_argument(
    "--evaluation_hybrid_residual_scale",
    type=float,
    help=(
        "evaluation-only action-attribution override for the hybrid MAPPO "
        "Cartesian residual; zero replays the identical analytic supervisor"
    ),
)
parser.add_argument(
    "--evaluation_phase_gain",
    type=float,
    help="evaluation-only C31/C32 adjacent-gap phase gain override",
)
parser.add_argument(
    "--evaluation_phase_separation_control",
    action="store_true",
    help=(
        "evaluation-only C40-N4: add adjacent-gap analytic tangential control "
        "without C31's cyclic assignment or permanent role lock"
    ),
)
parser.add_argument(
    "--evaluation_latched_reference_contraction",
    action="store_true",
    help=(
        "evaluation-only C40-N6: once the legal team-ready gate unlocks, "
        "continue reference-ring contraction through later readiness loss"
    ),
)
parser.add_argument(
    "--evaluation_maximum_tangential_speed_mps",
    type=float,
    help="evaluation-only C31/C32 analytic tangential-speed cap override",
)
parser.add_argument(
    "--evaluation_maximum_radial_speed_mps",
    type=float,
    help="evaluation-only C36 analytic radial-component speed cap",
)
parser.add_argument(
    "--evaluation_phase_scheduled_radial_budget",
    action="store_true",
    help=(
        "evaluation-only C37 schedule from total-speed radial authority far "
        "away to --evaluation_maximum_radial_speed_mps inside the phase ring"
    ),
)
parser.add_argument(
    "--evaluation_blocker_phase_anchor_gain",
    type=float,
    help=(
        "evaluation-only C34 weak forward-blocker phase anchor gain; "
        "enables the anchor with its frozen 0.8 m/s cap"
    ),
)
parser.add_argument(
    "--evaluation_reachable_role_slot_lead",
    action="store_true",
    help=(
        "evaluation-only C38: map each reachable agent's capped CV intercept "
        "lead to its locked role slot"
    ),
)
parser.add_argument(
    "--evaluation_far_slot_intercept_route",
    action="store_true",
    help=(
        "evaluation-only C39: outside the phase window, fly directly at the "
        "bounded CV intercept of each locked moving role slot"
    ),
)
parser.add_argument(
    "--evaluation_lock_slot_assignment",
    action="store_true",
    help=(
        "evaluation-only C40-B: keep the first valid blocker/scout role "
        "assignment without enabling C31 phase control"
    ),
)
parser.add_argument(
    "--evaluation_loss_scout_blend",
    type=float,
    help="evaluation-only P5-L1 override for the Last-search scout blend",
)
parser.add_argument(
    "--evaluation_loss_tracker_blend",
    type=float,
    help="evaluation-only P5-L1 override for the Last-search tracker blend",
)
parser.add_argument(
    "--evaluation_loss_blend_growth",
    type=float,
    help="evaluation-only P5-L1 override for the loss-duration blend growth",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

for name in (
    "num_envs",
    "iterations",
    "rollout_steps",
    "eval_every",
    "eval_episodes",
    "checkpoint_every",
    "required_passes",
):
    if getattr(args_cli, name) <= 0:
        parser.error(f"--{name} must be positive")
if not 0.0 <= args_cli.target_success <= 1.0:
    parser.error("--target_success must be in [0, 1]")
if not 0.0 <= args_cli.max_collision <= 1.0:
    parser.error("--max_collision must be in [0, 1]")
if args_cli.max_wall_hours <= 0.0:
    parser.error("--max_wall_hours must be positive")
for name in ("learning_rate", "actor_learning_rate"):
    value = getattr(args_cli, name)
    if value is not None and (not math.isfinite(value) or value <= 0):
        parser.error(f"--{name} must be finite and positive")
if args_cli.evaluation_only and args_cli.resume is None:
    parser.error("--evaluation_only requires --resume")
if args_cli.evaluation_only and args_cli.expected_initial_evaluation is not None:
    parser.error("--expected_initial_evaluation is a training-only gate")
if args_cli.evaluation_trace and not args_cli.evaluation_only:
    parser.error("--evaluation_trace requires --evaluation_only")
if args_cli.reset_actor_output_on_resume and args_cli.resume is None:
    parser.error("--reset_actor_output_on_resume requires --resume")
if (
    args_cli.reset_actor_output_on_resume
    and not args_cli.evaluation_only
    and not args_cli.reset_optimizer_on_resume
):
    parser.error("actor output reset requires optimizer reset during training")
if (
    args_cli.stage in ("c32", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79")
    and not args_cli.evaluation_only
    and args_cli.resume is not None
    and not args_cli.reset_actor_output_on_resume
):
    parser.error("gain-residual training requires a neutral actor output reset")
if args_cli.evaluation_hybrid_residual_scale is not None:
    if not args_cli.evaluation_only:
        parser.error(
            "--evaluation_hybrid_residual_scale is only valid with "
            "--evaluation_only"
        )
    if not 0.0 <= args_cli.evaluation_hybrid_residual_scale <= 1.0:
        parser.error("--evaluation_hybrid_residual_scale must be in [0, 1]")
for name in (
    "evaluation_phase_gain",
    "evaluation_maximum_radial_speed_mps",
    "evaluation_maximum_tangential_speed_mps",
    "evaluation_blocker_phase_anchor_gain",
):
    value = getattr(args_cli, name)
    if value is not None:
        if not args_cli.evaluation_only:
            parser.error(f"--{name} is only valid with --evaluation_only")
        if value <= 0.0:
            parser.error(f"--{name} must be positive")
# An explicit seed plan is valid in both modes.  During training it is applied
# only inside evaluate_in_place(), which clears the forced seeds and restores
# the training seed before rollouts resume.
if args_cli.evaluation_phase_scheduled_radial_budget:
    if not args_cli.evaluation_only:
        parser.error(
            "--evaluation_phase_scheduled_radial_budget requires evaluation_only"
        )
    if args_cli.evaluation_maximum_radial_speed_mps is None:
        parser.error(
            "phase-scheduled radial budget requires a maximum radial speed"
        )
if args_cli.evaluation_phase_separation_control:
    if not args_cli.evaluation_only:
        parser.error("phase separation control requires evaluation_only")
    if args_cli.stage != "c40":
        parser.error("phase separation diagnostic is reserved for C40")
if args_cli.evaluation_latched_reference_contraction:
    if not args_cli.evaluation_only:
        parser.error("latched reference contraction requires evaluation_only")
    if args_cli.stage != "c40":
        parser.error("latched contraction diagnostic is reserved for C40")
if args_cli.evaluation_reachable_role_slot_lead and not args_cli.evaluation_only:
    parser.error("reachable role slot lead requires evaluation_only")
if args_cli.evaluation_far_slot_intercept_route and not args_cli.evaluation_only:
    parser.error("far slot intercept route requires evaluation_only")
if args_cli.evaluation_lock_slot_assignment and not args_cli.evaluation_only:
    parser.error("slot assignment lock requires evaluation_only")
if args_cli.evaluation_lock_slot_assignment and args_cli.stage != "c40":
    parser.error("slot assignment lock diagnostic is reserved for C40")
if args_cli.evaluation_capture_radius_m is not None and not args_cli.evaluation_only:
    parser.error("--evaluation_capture_radius_m is only valid with --evaluation_only")
if args_cli.evaluation_sense_range_m is not None and not args_cli.evaluation_only:
    parser.error("--evaluation_sense_range_m is only valid with --evaluation_only")
if (
    args_cli.evaluation_tracked_reset_reference_range_m is not None
    and not args_cli.evaluation_only
):
    parser.error(
        "--evaluation_tracked_reset_reference_range_m is only valid with "
        "--evaluation_only"
    )
if (
    args_cli.evaluation_tracked_reset_reference_range_m is not None
    and args_cli.stage != "c40"
):
    parser.error("tracked reset range curriculum diagnostic is reserved for C40")
if args_cli.evaluation_evader_speed_mps is not None and not args_cli.evaluation_only:
    parser.error("--evaluation_evader_speed_mps is only valid with --evaluation_only")
if (
    (
        args_cli.stage in ("c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")
        or args_cli.stage in ("c31", "c32", "c40", "c41", "c42", "c43", "c44", "c45", "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55", "c56", "c57", "c58", "c60", "c61", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79", "c80", "c81", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91")
    )
    and args_cli.evaluation_evader_speed_mps is not None
    and args_cli.evaluation_evader_speed_mps != 4.0
):
    parser.error(
        "C20--C30 equal-speed stages only permit the 4.0 m/s diagnostic; "
        "slower evaluation overrides violate their strict speed contract"
    )
if args_cli.stage in ("c25", "c26", "c27"):
    allowed_episode_lengths = (30.0, 40.0) if args_cli.evaluation_only else (30.0,)
    if not any(
        math.isclose(args_cli.episode_length_s, value)
        for value in allowed_episode_lengths
    ):
        parser.error(
            "C25/C26/C27 training freezes 30 s; evaluation-only permits "
            "30 or 40 s for the preregistered time-budget ablation"
        )
if args_cli.stage in ("c29", "c30"):
    required_episode_length = 30.0 if args_cli.evaluation_only else 40.0
    if not math.isclose(args_cli.episode_length_s, required_episode_length):
        parser.error(
            "C29/C30/C31 train only at 40 s and evaluate only at the frozen 30 s horizon"
        )
if args_cli.stage in ("c31", "c32"):
    required_episode_length = 30.0 if args_cli.evaluation_only else 40.0
    if not math.isclose(args_cli.episode_length_s, required_episode_length):
        parser.error(
            "C31/C32 train only at 40 s and evaluate only at the frozen 30 s horizon"
        )
if args_cli.stage == "c40" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C40 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c41" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C41 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c42" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C42 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c43" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C43 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c44" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C44 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c45" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C45 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c46" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C46 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c47" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C47 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c48" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C48 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c49" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C49 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c50" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C50 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c51" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C51 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c52" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C52 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c53" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C53 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c54" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C54 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c55" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C55 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c56" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C56 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c57" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C57 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c58" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C58 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c60" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C60 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c61" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C61 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c62" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C62 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c63" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C63 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c64" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C64 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c65" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C65 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c66" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C66 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c67" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C67 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c68" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C68 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c69" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C69 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c70" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C70 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c71" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C71 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c72" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C72 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c73" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C73 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c74" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C74 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c75" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C75 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c76" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C76 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c77" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C77 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c78" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C78 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c79" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C79 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c80" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C80 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c81" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C81 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c82" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C82 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c83" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C83 freezes both training and evaluation to the 30 s horizon")
if args_cli.stage == "c84" and not math.isclose(args_cli.episode_length_s, 30.0):
    parser.error("C84 freezes both training and evaluation to the 30 s horizon")
if args_cli.c96_near_ring_reset_curriculum and (
    args_cli.stage != "c91" or args_cli.evaluation_only
    or args_cli.c94_extended_episode_horizon or args_cli.preentry_radial_priority
    or args_cli.evaluation_seed_plan is None or args_cli.eval_episodes != 32
):
    parser.error("C96 requires c91 training, explicit fixed32, and no C93/C94 arm")
if args_cli.region_capture and (
    args_cli.stage != "c91" or args_cli.c96_near_ring_reset_curriculum
    or args_cli.c94_extended_episode_horizon or args_cli.preentry_radial_priority
    or args_cli.evaluation_capture_radius_m is not None
):
    parser.error("C97 requires C91 with the original reset/horizon and no other development arm")
if args_cli.region_objective and not args_cli.region_capture:
    parser.error("--region_objective requires --region_capture")
if args_cli.dual_observer_formation and not args_cli.region_objective:
    parser.error("--dual_observer_formation requires the aligned regional objective")
if args_cli.dual_observer_role_lock and not args_cli.dual_observer_formation:
    parser.error("--dual_observer_role_lock requires --dual_observer_formation")
if args_cli.region_gap_reserve and (not args_cli.region_objective or args_cli.dual_observer_formation):
    parser.error("C100 requires region objective and excludes dual-observer arms")
if args_cli.target_path_guard_extended and not args_cli.target_path_guard_acceleration:
    parser.error("--target_path_guard_extended requires --target_path_guard_acceleration")
if args_cli.target_path_guard_margin and not args_cli.target_path_guard_extended:
    parser.error("--target_path_guard_margin requires --target_path_guard_extended")
if args_cli.target_path_guard_reserve and not args_cli.target_path_guard_margin:
    parser.error("--target_path_guard_reserve requires --target_path_guard_margin")
if args_cli.learning_diagnostics and not (args_cli.stage == 'c91' and args_cli.region_objective and args_cli.target_path_guard_margin):
    parser.error("--learning_diagnostics requires the C107 c91 region/margin controller")
if args_cli.target_buffer_failure and not (args_cli.stage == 'c91' and args_cli.region_objective and args_cli.region_capture):
    parser.error("--target_buffer_failure requires c91 region capture and objective")
if args_cli.target_path_guard_acceleration and not args_cli.target_path_guard:
    parser.error("--target_path_guard_acceleration requires --target_path_guard")
if args_cli.target_path_guard and not (args_cli.region_objective and args_cli.region_joint_lookahead):
    parser.error("--target_path_guard requires --region_objective and --region_joint_lookahead")
if args_cli.region_joint_lookahead and (not args_cli.region_objective or args_cli.dual_observer_formation or args_cli.region_gap_reserve):
    parser.error("C101 requires region objective and excludes C98/C99/C100 arms")
if args_cli.region_capture and not args_cli.evaluation_only and (
    not args_cli.region_objective or not args_cli.ppo_exact_logprob
):
    parser.error("C97 training requires aligned objective and exact PPO log probabilities")
if args_cli.c94_extended_episode_horizon and (
    args_cli.stage != "c91"
    or not math.isclose(args_cli.episode_length_s, 45.0)
):
    parser.error(
        "--c94_extended_episode_horizon requires stage=c91 and "
        "--episode_length_s 45"
    )
if (
    args_cli.stage in ("c85", "c86", "c87", "c88", "c89", "c90", "c91")
    and not math.isclose(args_cli.episode_length_s, 30.0)
    and not args_cli.c94_extended_episode_horizon
):
    parser.error("C85--C91 freeze both training and evaluation to the 30 s horizon")
if (
    args_cli.evaluation_dynamic_obstacle_speed_mps is not None
    and not args_cli.evaluation_only
):
    parser.error(
        "--evaluation_dynamic_obstacle_speed_mps is only valid with --evaluation_only"
    )
if args_cli.evaluation_ignore_dynamic_obstacle_velocity and not args_cli.evaluation_only:
    parser.error(
        "--evaluation_ignore_dynamic_obstacle_velocity is only valid with "
        "--evaluation_only"
    )
if args_cli.evaluation_slot_prediction_horizon_s is not None and not args_cli.evaluation_only:
    parser.error(
        "--evaluation_slot_prediction_horizon_s is only valid with "
        "--evaluation_only"
    )
if args_cli.evaluation_loss_reacquisition is not None and not args_cli.evaluation_only:
    parser.error(
        "--evaluation_loss_reacquisition is only valid with --evaluation_only"
    )
if args_cli.evaluation_pairwise_recovery is not None and not args_cli.evaluation_only:
    parser.error(
        "--evaluation_pairwise_recovery is only valid with --evaluation_only"
    )
if (
    args_cli.evaluation_contraction_speed_mps is not None
    and not args_cli.evaluation_only
):
    parser.error(
        "--evaluation_contraction_speed_mps is only valid with --evaluation_only"
    )
if (
    args_cli.evaluation_orientation_staging_radius_m is not None
    and not args_cli.evaluation_only
):
    parser.error(
        "--evaluation_orientation_staging_radius_m is only valid with "
        "--evaluation_only"
    )
if (
    args_cli.evaluation_orientation_staging_radius_m is not None
    and args_cli.stage != "c40"
):
    parser.error(
        "--evaluation_orientation_staging_radius_m is reserved for C40"
    )
for name in (
    "evaluation_loss_scout_blend",
    "evaluation_loss_tracker_blend",
    "evaluation_loss_blend_growth",
):
    value = getattr(args_cli, name)
    if value is not None and not args_cli.evaluation_only:
        parser.error(f"--{name} is only valid with --evaluation_only")
    if value is not None and not 0.0 <= value <= 1.0:
        parser.error(f"--{name} must be in [0, 1]")
    if value is not None and args_cli.evaluation_loss_reacquisition != "last":
        parser.error(f"--{name} requires --evaluation_loss_reacquisition last")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np  # noqa: E402
import torch  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = REPO_ROOT / "isaac_p5" / "source" / "bylw_isaac_p5"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PACKAGE_SOURCE))
os.environ["BYLW_P5_ENABLE_ISAAC_RUNTIME"] = "1"

from algos.mappo import (  # noqa: E402
    MAPPO,
    RolloutBuffer,
    expand_mlp_actor_observation_state,
)
from bylw_isaac_p5.core.native_observations import (  # noqa: E402
    NATIVE_OBSERVATION_DIM,
    NATIVE_ROLE_PHASE_OBSERVATION_DIM,
    NATIVE_STATE_DIM,
)
from bylw_isaac_p5.core.native_training import native_curriculum_stage  # noqa: E402
from bylw_isaac_p5.core.region_capture import (  # noqa: E402
    REGION_PROTOCOL, REGION_REWARD, REGION_OBSERVATION_DIM, REGION_STATE_DIM,
    configure_region_capture, expand_region_critic_state,
)
from bylw_isaac_p5.core.evaluation_reset import standard_reset_evaluation  # noqa: E402
from bylw_isaac_p5.tasks.direct.bylw_pursuit.bylw_pursuit_env_cfg import (  # noqa: E402
    DENSE_STATIC_OBSTACLE_BOX_SIZES_XY_M,
    DENSE_STATIC_OBSTACLE_SHAPES,
    DENSE_STATIC_OBSTACLE_SPECS,
    MIXED_STATIC_OBSTACLE_BOX_SIZES_XY_M,
    MIXED_STATIC_OBSTACLE_SHAPES,
    MIXED_STATIC_OBSTACLE_SPECS,
    STATIC_OBSTACLE_BOX_SIZES_XY_M,
    STATIC_OBSTACLE_SHAPES,
    STATIC_OBSTACLE_SPECS,
)
from bylw_isaac_p5.tasks.direct.bylw_pursuit.bylw_pursuit_train_env import (  # noqa: E402
    BylwPursuitTrainEnv,
)
from bylw_isaac_p5.tasks.direct.bylw_pursuit.bylw_pursuit_train_env_cfg import (  # noqa: E402
    BylwPursuitTrainEnvCfg,
)

def gpu_snapshot(physical_index: int) -> dict[str, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical_index),
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    memory, utilization, temperature = (
        float(value.strip()) for value in result.stdout.strip().split(",")
    )
    return {
        "memory_used_mb": memory,
        "utilization_percent": utilization,
        "temperature_c": temperature,
    }


def resolve_inside_repo(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("training outputs must remain inside the bylw repository") from exc
    return resolved


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stack_observations(observations: dict[str, torch.Tensor], agents: list[str]) -> torch.Tensor:
    return torch.stack([observations[agent] for agent in agents], dim=1)


def action_dictionary(action: torch.Tensor, agents: list[str]) -> dict[str, torch.Tensor]:
    return {agent: action[:, index] for index, agent in enumerate(agents)}


def observation_contract(stage: str, *, region_capture: bool = False) -> tuple[int, str]:
    if region_capture:
        return REGION_OBSERVATION_DIM, "p5_2_native_64_region_time_v1"
    if stage in ("c27", "c29", "c30", "c40", "c41", "c42", "c43", "c44", "c45", "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55", "c56", "c57", "c58", "c60", "c61", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79", "c80", "c81", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91"):
        return (
            NATIVE_ROLE_PHASE_OBSERVATION_DIM,
            "p5_2_native_63_role_goal_phase_v1",
        )
    if stage in ("c31", "c32"):
        return (
            NATIVE_ROLE_PHASE_OBSERVATION_DIM,
            "p5_2_native_63_role_goal_phase_v1",
        )
    return NATIVE_OBSERVATION_DIM, "p5_2_native_53_sweep_angle_v2"


def action_semantics_contract(stage: str) -> list[str]:
    if stage in ("c80", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91"):
        return ["radial_gain_delta", "phase_gain_delta", "radial_velocity_residual"]
    if stage == "c81":
        return [
            "radial_gain_delta",
            "tangential_velocity_residual",
            "radial_velocity_residual",
        ]
    if stage in (
        "c32", "c62", "c63", "c64", "c65", "c66", "c67", "c68",
        "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76",
        "c77", "c78", "c79",
    ):
        return ["radial_gain_delta", "phase_gain_delta", "yaw_gain_delta"]
    return ["vx_world", "vy_world", "yaw_rate"]


def save_checkpoint(
    path: Path,
    agent: MAPPO,
    iteration: int,
    stage: str,
    best_success: float,
) -> None:
    curriculum = native_curriculum_stage(stage)
    observation_dim, observation_schema = observation_contract(stage, region_capture=args_cli.region_capture)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "optimizer": agent.opt.state_dict(),
            "iteration": int(iteration),
            "stage": stage,
            "capture_protocol": REGION_PROTOCOL if args_cli.region_capture else "legacy",
            "region_objective": args_cli.region_objective,
            "dual_observer_formation": getattr(args_cli, "dual_observer_formation", False),
            "dual_observer_role_lock": getattr(args_cli, "dual_observer_role_lock", False),
            "region_gap_reserve": getattr(args_cli, "region_gap_reserve", False),
            "region_joint_lookahead": getattr(args_cli, "region_joint_lookahead", False),
            "target_path_guard": getattr(args_cli, "target_path_guard", False),
            "target_path_guard_acceleration": getattr(args_cli, "target_path_guard_acceleration", False),
            "target_path_guard_extended": getattr(args_cli, "target_path_guard_extended", False),
            "target_path_guard_margin": getattr(args_cli, "target_path_guard_margin", False),
            "target_path_guard_reserve": getattr(args_cli, "target_path_guard_reserve", False),
            "actor_learning_rate": agent.opt.param_groups[0]["lr"],
            "critic_learning_rate": agent.opt.param_groups[1]["lr"],
            "target_buffer_failure": getattr(args_cli, "target_buffer_failure", False),
            "frame_accurate_loss_counters": getattr(args_cli, "frame_accurate_loss_counters", False),
            "ppo_logprob_schema": "pre_tanh_v1" if args_cli.ppo_exact_logprob else "legacy",
            "best_success": float(best_success),
            "observation_dim": observation_dim,
            "state_dim": int(agent.critic.v[0].in_features),
            "action_dim": 3,
            "action_semantics": (
                ["radial_gain_delta", "phase_gain_delta", "radial_velocity_residual"]
                if stage in ("c80", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91")
                else
                ["radial_gain_delta", "tangential_velocity_residual", "radial_velocity_residual"]
                if stage == "c81"
                else
                ["radial_gain_delta", "phase_gain_delta", "yaw_gain_delta"]
                if stage in ("c32", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79")
                else ["vx_world", "vy_world", "yaw_rate"]
            ),
            "reward_schema": (
                "region_v1_terminal_potential_target_buffer_failure_v1" if getattr(args_cli, "target_buffer_failure", False) else
                REGION_REWARD if args_cli.region_objective else
                "p5_2_native_reward_v6_strong_ring_violation_progress"
                if stage in ("c79", "c80", "c81", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91")
                else "p5_2_native_reward_v5_strong_ring_step"
                if stage in ("c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78")
                else "p5_2_native_reward_v4_observable_role_phase"
                if stage in ("c27", "c29", "c30", "c31", "c32", "c40", "c41", "c42", "c43", "c44", "c45", "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55", "c56", "c57", "c58", "c60", "c61", "c62")
                else
                "p5_2_native_reward_v3_slot_handoff"
                if stage in ("c9", "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30", "c31", "c32", "c40", "c41", "c42", "c43", "c44", "c45", "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55", "c56", "c57", "c58", "c60", "c61", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79", "c80", "c81", "c82", "c83", "c84", "c85")
                else "p5_2_native_reward_v2"
            ),
            "observation_schema": observation_schema,
            "training_capture_radius_m": 5.5 if args_cli.region_capture else curriculum.training_capture_radius_m,
            "fixed_evaluation_capture_radius_m": [6.0, 5.0],
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    agent: MAPPO,
    *,
    load_optimizer: bool = True,
    expected_action_semantics: list[str] | None = None,
    allow_action_semantics_mismatch: bool = False,
    expected_capture_protocol: str = "legacy",
    allow_capture_protocol_migration: bool = False,
    expected_dual_observer_formation: bool | None = None,
    expected_dual_observer_role_lock: bool | None = None,
    expected_region_gap_reserve: bool | None = None,
    expected_region_joint_lookahead: bool | None = None,
    expected_target_path_guard: bool | None = None,
    expected_target_path_guard_acceleration: bool | None = None,
    expected_target_path_guard_extended: bool | None = None,
    expected_target_path_guard_margin: bool | None = None,
    expected_target_path_guard_reserve: bool | None = None,
    expected_optimizer_learning_rates: tuple[float, float] | None = None,
    expected_target_buffer_failure: bool | None = None,
    expected_frame_accurate_loss_counters: bool | None = None,
    allow_loss_counter_migration: bool = False,
    allow_target_buffer_failure_migration: bool = False,
    allow_controller_migration: bool = False,
) -> int:
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    if (expected_target_path_guard_reserve is not None
            and bool(checkpoint.get("target_path_guard_reserve", False)) != expected_target_path_guard_reserve):
        if not allow_controller_migration:
            raise ValueError("controller target prediction reserve differs; explicit controller migration is required")
        if load_optimizer:
            raise ValueError("target prediction reserve migration requires optimizer reset")
    if load_optimizer and expected_optimizer_learning_rates is not None:
        saved_groups = checkpoint.get("optimizer", {}).get("param_groups", [])
        if tuple(group["lr"] for group in saved_groups) != tuple(expected_optimizer_learning_rates):
            raise ValueError("optimizer learning rates differ; pass matching rates or reset optimizer")
    if (expected_frame_accurate_loss_counters is not None
            and bool(checkpoint.get("frame_accurate_loss_counters", False)) != expected_frame_accurate_loss_counters):
        if not allow_loss_counter_migration:
            raise ValueError("loss counter observation clock differs; explicit migration is required")
        if load_optimizer:
            raise ValueError("loss counter migration requires optimizer reset")
    if (expected_target_buffer_failure is not None
            and bool(checkpoint.get("target_buffer_failure", False)) != expected_target_buffer_failure):
        if not allow_target_buffer_failure_migration:
            raise ValueError("target buffer failure contract differs; explicit migration is required")
        if load_optimizer:
            raise ValueError("target buffer failure migration requires optimizer reset")
    if (expected_target_path_guard_margin is not None
            and bool(checkpoint.get("target_path_guard_margin", False)) != expected_target_path_guard_margin
            and not allow_controller_migration):
        raise ValueError("controller target margin selection differs; explicit controller migration is required")
    if (expected_target_path_guard_extended is not None
            and bool(checkpoint.get("target_path_guard_extended", False)) != expected_target_path_guard_extended
            and not allow_controller_migration):
        raise ValueError("controller target path horizon differs; explicit controller migration is required")
    if (expected_target_path_guard_acceleration is not None
            and bool(checkpoint.get("target_path_guard_acceleration", False)) != expected_target_path_guard_acceleration
            and not allow_controller_migration):
        raise ValueError("controller target acceleration scenario differs; explicit controller migration is required")
    if (expected_target_path_guard is not None
            and bool(checkpoint.get("target_path_guard", False)) != expected_target_path_guard
            and not allow_controller_migration):
        raise ValueError("controller target path guard differs; explicit controller migration is required")
    if (expected_region_joint_lookahead is not None
            and bool(checkpoint.get("region_joint_lookahead", False)) != expected_region_joint_lookahead
            and not allow_controller_migration):
        raise ValueError("controller joint lookahead differs; explicit controller migration is required")
    if (expected_region_gap_reserve is not None
            and bool(checkpoint.get("region_gap_reserve", False)) != expected_region_gap_reserve
            and not allow_controller_migration):
        raise ValueError("controller gap reserve differs; explicit controller migration is required")
    if (expected_dual_observer_role_lock is not None
            and bool(checkpoint.get("dual_observer_role_lock", False)) != expected_dual_observer_role_lock
            and not allow_controller_migration):
        raise ValueError("controller role lock differs; explicit controller migration is required")
    if (expected_dual_observer_formation is not None
            and bool(checkpoint.get("dual_observer_formation", False)) != expected_dual_observer_formation
            and not allow_controller_migration):
        raise ValueError("controller differs; explicit controller migration is required")
    if (checkpoint.get("capture_protocol", "legacy") != expected_capture_protocol
            and not allow_capture_protocol_migration):
        raise ValueError("capture protocol differs; explicit migration is required")
    if (
        expected_action_semantics is not None
        and checkpoint.get("action_semantics") != expected_action_semantics
        and not allow_action_semantics_mismatch
    ):
        raise ValueError(
            "checkpoint action semantics differ; reset the actor output before migration"
        )
    expected = (
        checkpoint.get("observation_dim"),
        checkpoint.get("state_dim"),
        checkpoint.get("action_dim"),
    )
    actor_input_dim = int(agent.actor.body[0].in_features)
    critic_input_dim = int(agent.critic.v[0].in_features)
    exact_observation_contract = expected == (actor_input_dim, critic_input_dim, 3)
    expanding_role_phase_contract = (
        expected[0] == NATIVE_OBSERVATION_DIM
        and actor_input_dim == NATIVE_ROLE_PHASE_OBSERVATION_DIM
        and expected[1:] == (NATIVE_STATE_DIM, 3)
        and critic_input_dim == NATIVE_STATE_DIM
    )
    expanding_region_contract = (
        expected == (NATIVE_ROLE_PHASE_OBSERVATION_DIM, NATIVE_STATE_DIM, 3)
        and (actor_input_dim, critic_input_dim) == (REGION_OBSERVATION_DIM, REGION_STATE_DIM)
        and expected_capture_protocol == REGION_PROTOCOL
    )
    if not (exact_observation_contract or expanding_role_phase_contract or expanding_region_contract):
        raise ValueError(f"incompatible native checkpoint dimensions: {expected}")
    if checkpoint.get("reward_schema") not in {
        "p5_2_native_reward_v2",
        "p5_2_native_reward_v3_slot_handoff",
        "p5_2_native_reward_v4_observable_role_phase",
        "p5_2_native_reward_v5_strong_ring_step",
        "p5_2_native_reward_v6_strong_ring_violation_progress",
        "region_v1_terminal_potential",
        "region_v1_terminal_potential_target_buffer_failure_v1",
    }:
        raise ValueError("refusing an incompatible native reward checkpoint")
    allowed_schema = {
        NATIVE_OBSERVATION_DIM: "p5_2_native_53_sweep_angle_v2",
        NATIVE_ROLE_PHASE_OBSERVATION_DIM: "p5_2_native_63_role_goal_phase_v1",
        REGION_OBSERVATION_DIM: "p5_2_native_64_region_time_v1",
    }.get(expected[0])
    if checkpoint.get("observation_schema") != allowed_schema:
        raise ValueError("incompatible teammate observation ordering")
    actor_state = checkpoint["actor"]
    critic_state = checkpoint["critic"]
    if expanding_role_phase_contract or expanding_region_contract:
        if load_optimizer:
            raise ValueError("observation/state migration requires optimizer reset")
        actor_state = expand_mlp_actor_observation_state(actor_state, actor_input_dim)
        if expanding_region_contract:
            critic_state = expand_region_critic_state(critic_state, critic_input_dim)
    agent.actor.load_state_dict(actor_state)
    agent.critic.load_state_dict(critic_state)
    if load_optimizer and "optimizer" in checkpoint:
        agent.opt.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("iteration", 0))


def assert_expected_initial_evaluation(actual: dict, expected: dict) -> None:
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if not expected or mismatches:
        raise RuntimeError(f"matched initial evaluation differs; no PPO updates allowed: {mismatches}")


@torch.no_grad()
def deterministic_action(agent: MAPPO, observations: torch.Tensor) -> torch.Tensor:
    mean, _ = agent.actor(observations)
    return torch.tanh(mean)


@torch.no_grad()
@standard_reset_evaluation
def evaluate_in_place(
    env: BylwPursuitTrainEnv,
    agent: MAPPO,
    episodes: int,
    eval_seed: int,
    training_seed: int,
    episode_records: list[dict] | None = None,
    episode_seed_plan: list[int] | None = None,
    trace_records: list[dict] | None = None,
) -> dict[str, float]:
    agents = list(env.cfg.possible_agents)
    first = agents[0]
    env.cfg.seed = eval_seed
    effective_seed_plan = (
        list(episode_seed_plan)
        if episode_seed_plan is not None
        else [eval_seed + offset for offset in range(episodes)]
    )
    planned_seed_set = set(effective_seed_plan)
    if len(effective_seed_plan) != episodes:
        raise ValueError("episode seed plan length must equal evaluation episodes")
    if len(planned_seed_set) != episodes:
        raise ValueError("episode seed plan must be unique")
    if min(effective_seed_plan) < 0 or max(effective_seed_plan) >= 2**32:
        raise ValueError("evaluation episode seeds must be uint32-compatible")
    if episode_seed_plan is not None:
        if len(episode_seed_plan) != episodes:
            raise ValueError("episode seed plan length must equal evaluation episodes")
        if len(planned_seed_set) != episodes:
            raise ValueError("episode seed plan must be unique")
    env.set_forced_episode_seed_plan(effective_seed_plan)
    observations, _ = env.reset(seed=eval_seed)
    recorded_seeds: set[int] = set()
    completed = 0
    training_success = 0
    success = 0
    strict = 0
    collision = 0
    collision_agent = 0
    collision_agent_before_unlock = 0
    collision_agent_after_unlock = 0
    collision_static = 0
    collision_dynamic = 0
    contact = 0
    target_buffer_failures = 0
    invalid = 0
    boundary_invalid = 0
    height_invalid = 0
    timeout = 0
    orientation_unlocked = 0
    length_sum = 0
    all_visible_fraction_sum = 0.0
    staging_visible_fraction_sum = 0.0
    terminal_yaw_error_sum = 0.0
    terminal_staging_fraction_sum = 0.0
    terminal_alignment_fraction_sum = 0.0
    terminal_phase_ready = 0
    phase_ready_fraction_sum = 0.0
    terminal_intercept_reachable_fraction_sum = 0.0
    terminal_intercept_horizon_sum = 0.0
    intercept_blocker_reachable = 0
    first_unlock_step_sum = 0.0
    first_unlock_step_count = 0
    target_projection_episodes = 0
    target_projection_calls = 0
    target_projection_corrected_steps = 0
    target_projection_constraint_corrections = 0
    target_projection_maximum_separation_fallback_constraints = 0
    target_projection_unavoidable_steps = 0
    target_projection_unavoidable_constraints = 0
    target_projection_infeasible_steps = 0
    target_projection_correction_norm_sum = 0.0
    target_projection_correction_norm_max = 0.0
    target_projection_min_predicted_center_distance = math.inf
    target_projection_minimum_maximum_attainable_center_distance = math.inf
    safety_projection_episodes = 0
    safety_projection_calls = 0
    safety_projection_corrected_steps = 0
    safety_projection_pair_corrections = 0
    safety_projection_infeasible_steps = 0
    safety_projection_correction_norm_sum = 0.0
    safety_projection_correction_norm_max = 0.0
    safety_projection_min_predicted_center_distance = math.inf
    obstacle_projection_episodes = 0
    obstacle_projection_calls = 0
    obstacle_projection_corrected_steps = 0
    obstacle_projection_constraint_corrections = 0
    obstacle_projection_infeasible_steps = 0
    obstacle_projection_correction_norm_sum = 0.0
    obstacle_projection_correction_norm_max = 0.0
    obstacle_projection_min_predicted_clearance = math.inf
    boundary_projection_episodes = 0
    boundary_projection_calls = 0
    boundary_projection_corrected_steps = 0
    boundary_projection_constraint_corrections = 0
    boundary_projection_infeasible_steps = 0
    boundary_projection_correction_norm_sum = 0.0
    boundary_projection_correction_norm_max = 0.0
    boundary_projection_min_predicted_clearance = math.inf
    tangent_bypass_episodes = 0
    tangent_bypass_active_steps = 0
    tangent_bypass_active_agent_steps = 0
    tangent_bypass_activations = 0
    tangent_bypass_min_candidate_clearance = math.inf
    slot_guidance_episodes = 0
    slot_active_steps_sum = 0
    slot_handoff_count_sum = 0
    slot_terminal_error_sum = 0.0
    hybrid_supervisor_episodes = 0
    hybrid_traditional_steps_sum = 0
    hybrid_rl_steps_sum = 0
    hybrid_rl_active_episodes = 0
    policy_residual_active_steps_sum = 0
    policy_residual_norm_sum_mps = 0.0
    policy_residual_radial_support_sum_mps = 0.0
    policy_residual_radial_opposition_sum_mps = 0.0
    policy_residual_phase_support_sum_mps = 0.0
    policy_residual_phase_opposition_sum_mps = 0.0
    slot_speed_limit_active_agent_fraction_sum = 0.0
    slot_speed_limit_mean_scale_sum = 0.0
    slot_speed_limit_forward_loss_sum_mps = 0.0
    slot_requested_tangential_speed_sum_mps = 0.0
    slot_executed_tangential_speed_sum_mps = 0.0
    loss_episodes = 0
    long_loss_2s_episodes = 0
    maximum_lost_steps_sum = 0
    maximum_lost_steps_global = 0
    team_lost_steps = 0
    recapture_count_sum = 0
    loss_reacquisition_event_sum = 0
    loss_reacquisition_completed_sum = 0
    loss_reacquisition_active_step_sum = 0
    loss_reacquisition_eligible_episodes = 0
    loss_reacquisition_first_completed_episodes = 0
    loss_reacquisition_first_within_2s_episodes = 0
    loss_reacquisition_first_delay_step_sum = 0
    loss_reacquisition_first_delay_count = 0
    loss_reacquisition_flank_event_sum = 0
    loss_reacquisition_flank_episode_count = 0
    loss_reacquisition_flank_active_step_sum = 0
    loss_reacquisition_flank_fallback_step_sum = 0
    loss_reacquisition_flank_minimum_path_clearance = math.inf
    loss_reacquisition_flank_minimum_los_clearance = math.inf
    loss_reacquisition_sector_valid_step_sum = 0
    loss_reacquisition_sector_fallback_step_sum = 0
    loss_reacquisition_sector_episode_count = 0
    loss_reacquisition_sector_minimum_path_clearance = math.inf
    target_displacement_sum = 0.0
    terminal_target_speed_sum = 0.0
    evader_episode_speed_limit_sum = 0.0
    evader_episode_speed_limit_min = math.inf
    evader_episode_speed_limit_max = 0.0
    strong_ring_max_hold_steps_sum = 0
    strong_ring_max_hold_steps_global = 0
    dynamic_obstacle_displacement_sum = 0.0
    dynamic_obstacle_displacement_max = 0.0
    dynamic_obstacle_speed_sum = 0.0
    obstacle_layout_signatures: set[str] = set()
    loss_reason_steps = {
        "pure_range": 0,
        "pure_fov": 0,
        "pure_occlusion": 0,
        "mixed": 0,
    }
    episode_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    episode_all_visible_steps = torch.zeros_like(episode_lengths)
    episode_staging_steps = torch.zeros_like(episode_lengths)
    episode_staging_visible_steps = torch.zeros_like(episode_lengths)
    episode_phase_ready_steps = torch.zeros_like(episode_lengths)
    episode_orientation_unlocked = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    episode_first_unlock_step = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    episode_target_projection_calls = torch.zeros_like(episode_lengths)
    episode_target_projection_corrected_steps = torch.zeros_like(episode_lengths)
    episode_target_projection_constraint_corrections = torch.zeros_like(
        episode_lengths
    )
    episode_target_projection_maximum_separation_fallback_constraints = (
        torch.zeros_like(episode_lengths)
    )
    episode_target_projection_unavoidable_steps = torch.zeros_like(episode_lengths)
    episode_target_projection_unavoidable_constraints = torch.zeros_like(
        episode_lengths
    )
    episode_target_projection_infeasible_steps = torch.zeros_like(episode_lengths)
    episode_target_projection_correction_norm_sum = torch.zeros(
        env.num_envs, device=env.device
    )
    episode_target_projection_correction_norm_max = torch.zeros_like(
        episode_target_projection_correction_norm_sum
    )
    episode_target_projection_min_predicted_center_distance = torch.full_like(
        episode_target_projection_correction_norm_sum, torch.inf
    )
    episode_target_projection_minimum_maximum_attainable_center_distance = (
        torch.full_like(episode_target_projection_correction_norm_sum, torch.inf)
    )
    episode_safety_projection_calls = torch.zeros_like(episode_lengths)
    episode_safety_projection_corrected_steps = torch.zeros_like(episode_lengths)
    episode_safety_projection_pair_corrections = torch.zeros_like(episode_lengths)
    episode_safety_projection_infeasible_steps = torch.zeros_like(episode_lengths)
    episode_safety_projection_correction_norm_sum = torch.zeros(
        env.num_envs, device=env.device
    )
    episode_safety_projection_correction_norm_max = torch.zeros_like(
        episode_safety_projection_correction_norm_sum
    )
    episode_safety_projection_min_predicted_center_distance = torch.full_like(
        episode_safety_projection_correction_norm_sum, torch.inf
    )
    episode_obstacle_projection_calls = torch.zeros_like(episode_lengths)
    episode_obstacle_projection_corrected_steps = torch.zeros_like(episode_lengths)
    episode_obstacle_projection_constraint_corrections = torch.zeros_like(
        episode_lengths
    )
    episode_obstacle_projection_infeasible_steps = torch.zeros_like(episode_lengths)
    episode_obstacle_projection_correction_norm_sum = torch.zeros(
        env.num_envs, device=env.device
    )
    episode_obstacle_projection_correction_norm_max = torch.zeros_like(
        episode_obstacle_projection_correction_norm_sum
    )
    episode_obstacle_projection_min_predicted_clearance = torch.full_like(
        episode_obstacle_projection_correction_norm_sum, torch.inf
    )
    episode_boundary_projection_calls = torch.zeros_like(episode_lengths)
    episode_boundary_projection_corrected_steps = torch.zeros_like(episode_lengths)
    episode_boundary_projection_constraint_corrections = torch.zeros_like(
        episode_lengths
    )
    episode_boundary_projection_infeasible_steps = torch.zeros_like(episode_lengths)
    episode_boundary_projection_correction_norm_sum = torch.zeros(
        env.num_envs, device=env.device
    )
    episode_boundary_projection_correction_norm_max = torch.zeros_like(
        episode_boundary_projection_correction_norm_sum
    )
    episode_boundary_projection_min_predicted_clearance = torch.full_like(
        episode_boundary_projection_correction_norm_sum, torch.inf
    )
    episode_tangent_bypass_active_steps = torch.zeros_like(episode_lengths)
    episode_tangent_bypass_active_agent_steps = torch.zeros_like(episode_lengths)
    episode_tangent_bypass_activations = torch.zeros_like(episode_lengths)
    episode_tangent_bypass_min_candidate_clearance = torch.full(
        (env.num_envs,), torch.inf, device=env.device
    )
    episode_maximum_lost_steps = torch.zeros_like(episode_lengths)
    episode_team_lost_steps = torch.zeros_like(episode_lengths)
    episode_loss_reason_steps = torch.zeros(
        env.num_envs, 4, dtype=torch.long, device=env.device
    )
    episode_min_agent_clearance = torch.full(
        (env.num_envs,), torch.inf, device=env.device
    )
    episode_min_static_clearance = torch.full_like(
        episode_min_agent_clearance, torch.inf
    )
    episode_min_boundary_clearance = torch.full_like(
        episode_min_agent_clearance, torch.inf
    )
    episode_dynamic_obstacle_max_displacement = torch.zeros(
        env.num_envs, device=env.device
    )
    episode_dynamic_obstacle_max_speed = torch.zeros(
        env.num_envs, device=env.device
    )
    episode_strong_ring_max_hold_steps = torch.zeros_like(episode_lengths)
    clearance_sums = {"agent": 0.0, "static": 0.0, "boundary": 0.0}
    clearance_minima = {"agent": math.inf, "static": math.inf, "boundary": math.inf}
    radial_error_sum = 0.0
    angular_error_sum = 0.0
    maximum_steps = (math.ceil(episodes / env.num_envs) + 2) * (
        env.max_episode_length + 10
    )
    for _ in range(maximum_steps):
        stacked = stack_observations(observations, agents)
        action = deterministic_action(agent, stacked)
        observations, _, terminated, truncated, extras = env.step(
            action_dictionary(action, agents)
        )
        episode_lengths += 1
        p5 = extras[first]["p5"]
        if trace_records is not None:
            p5_training_step = extras[first]["p5_training"]
            for env_index in range(env.num_envs):
                current_seed = int(p5["episode_seed"][env_index].item())
                if current_seed not in planned_seed_set:
                    continue
                trace_records.append(
                    {
                        "episode_seed": current_seed,
                        **({"lost_steps": int(p5["lost_steps"][env_index].item()),
                            "recapture_count": int(p5["recapture_count"][env_index].item())}
                           if env.cfg.frame_accurate_loss_counters else {}),
                        **({"target_buffer_failure": int(p5["target_buffer_failure"][env_index].item())}
                           if env.cfg.target_buffer_failure_enabled else {}),
                        **{key: (json.dumps(value[env_index].detach().cpu().tolist())
                                 if value[env_index].ndim else value[env_index].item())
                           for key,value in getattr(env,'_learning_diagnostics',{}).items()},
                        **{key: (json.dumps(value[env_index].detach().cpu().tolist())
                                 if value[env_index].ndim else value[env_index].item())
                           for key, value in p5_training_step.items() if key.startswith("target_path_guard_")},
                        "joint_lookahead_active": int(p5_training_step["joint_lookahead_active"][env_index].item()),
                        "joint_lookahead_corrected": int(p5_training_step["joint_lookahead_corrected"][env_index].item()),
                        "joint_lookahead_correction_mps": float(p5_training_step["joint_lookahead_correction_mps"][env_index].item()),
                        "joint_lookahead_nominal_violation": float(p5_training_step["joint_lookahead_nominal_violation"][env_index].item()),
                        "joint_lookahead_selected_violation": float(p5_training_step["joint_lookahead_selected_violation"][env_index].item()),
                        "dual_observer_active": int(p5_training_step["dual_observer_active"][env_index].item()),
                        "dual_observer_pair_angle_rad": float(p5_training_step["dual_observer_pair_angle_rad"][env_index].item()),
                        "step": int(episode_lengths[env_index].item()),
                        "time_s": float(episode_lengths[env_index].item())
                        * env._task_spec.timing.high_level_dt,
                        "strong_ring_now": int(
                            p5["strong_ring_now"][env_index].item()
                        ),
                        "strong_ring_hold_steps": int(
                            p5["strong_ring_hold_steps"][env_index].item()
                        ),
                        "stabilization_active": int(
                            p5[
                                "strong_ring_confirmation_stabilization_active"
                            ][env_index].item()
                        ),
                        "slot_preentry_radial_priority_active": int(
                            p5["slot_preentry_radial_priority_active"][
                                env_index
                            ].item()
                        ),
                        "ring_projection_active": int(
                            p5["ring_projection_active"][env_index].item()
                        ),
                        "ring_projection_activation_latched": int(
                            p5["ring_projection_activation_latched"][env_index].item()
                        ),
                        "ring_projection_corrected": int(
                            p5["ring_projection_corrected"][env_index].item()
                        ),
                        "ring_projection_correction_mps": float(
                            p5["ring_projection_correction_norm_mps"][
                                env_index
                            ].item()
                        ),
                        "ring_projection_min_predicted_radius_m": float(
                            p5["ring_projection_min_predicted_radius_m"][
                                env_index
                            ].item()
                        ),
                        "ring_projection_max_predicted_radius_m": float(
                            p5["ring_projection_max_predicted_radius_m"][
                                env_index
                            ].item()
                        ),
                        "ring_projection_infeasible": int(
                            p5["ring_projection_infeasible"][env_index].item()
                        ),
                        "angular_gap_projection_active": int(
                            p5["angular_gap_projection_active"][env_index].item()
                        ),
                        "angular_gap_projection_corrected": int(
                            p5["angular_gap_projection_corrected"][env_index].item()
                        ),
                        "angular_gap_projection_correction_mps": float(
                            p5[
                                "angular_gap_projection_correction_norm_mps"
                            ][env_index].item()
                        ),
                        "angular_gap_projection_max_predicted_gap_rad": float(
                            p5[
                                "angular_gap_projection_max_predicted_gap_rad"
                            ][env_index].item()
                        ),
                        "angular_gap_projection_max_predicted_mean_gap_error_rad": float(
                            p5[
                                "angular_gap_projection_max_predicted_mean_gap_error_rad"
                            ][env_index].item()
                        ),
                        "angular_gap_projection_infeasible": int(
                            p5["angular_gap_projection_infeasible"][env_index].item()
                        ),
                        "preentry_braking_active": int(
                            p5["preentry_braking_active"][env_index].item()
                        ),
                        "preentry_braking_corrected": int(
                            p5["preentry_braking_corrected"][env_index].item()
                        ),
                        "preentry_braking_correction_mps": float(
                            p5["preentry_braking_correction_norm_mps"][
                                env_index
                            ].item()
                        ),
                        "preentry_braking_limited_agents": int(
                            p5["preentry_braking_limited_agents"][env_index].item()
                        ),
                        "preentry_braking_minimum_stopping_margin_m": float(
                            p5["preentry_braking_minimum_stopping_margin_m"][
                                env_index
                            ].item()
                        ),
                        "preentry_braking_infeasible": int(
                            p5["preentry_braking_infeasible"][env_index].item()
                        ),
                        "minimum_radius_m": float(
                            p5["strong_ring_minimum_distance_m"][env_index].item()
                        ),
                        "maximum_radius_m": float(
                            p5["strong_ring_maximum_distance_m"][env_index].item()
                        ),
                        "maximum_gap_rad": float(
                            p5["training_capture_max_gap_rad"][env_index].item()
                        ),
                        "mean_gap_error_rad": float(
                            p5["strong_ring_mean_gap_error_rad"][env_index].item()
                        ),
                        "pursuer_target_distances_m": json.dumps(
                            p5["pursuer_target_distance_m"][env_index]
                            .detach()
                            .cpu()
                            .tolist(),
                            separators=(",", ":"),
                        ),
                        "pursuer_position_xy_m": json.dumps(
                            p5["pursuer_position_xy_m"][env_index]
                            .detach()
                            .cpu()
                            .tolist(),
                            separators=(",", ":"),
                        ),
                        "pursuer_velocity_xy_mps": json.dumps(
                            p5["pursuer_velocity_xy_mps"][env_index]
                            .detach()
                            .cpu()
                            .tolist(),
                            separators=(",", ":"),
                        ),
                        "commanded_velocity_xy_mps": json.dumps(
                            p5["commanded_velocity_xy_mps"][env_index]
                            .detach()
                            .cpu()
                            .tolist(),
                            separators=(",", ":"),
                        ),
                        "target_position_xy_m": json.dumps(
                            p5["target_position_xy_m"][env_index]
                            .detach()
                            .cpu()
                            .tolist(),
                            separators=(",", ":"),
                        ),
                        "target_velocity_xy_mps": json.dumps(
                            p5["target_velocity_xy_mps"][env_index]
                            .detach()
                            .cpu()
                            .tolist(),
                            separators=(",", ":"),
                        ),
                        "target_acceleration_xy_mps2": json.dumps(
                            p5["target_acceleration_xy_mps2"][env_index]
                            .detach()
                            .cpu()
                            .tolist(),
                            separators=(",", ":"),
                        ),
                        "all_visible": int(
                            p5["orientation_all_visible"][env_index].item()
                        ),
                        "visibility_status_code": int(
                            p5["visibility_status_code"][env_index].item()
                        ),
                        "orientation_unlocked": int(
                            p5["orientation_contraction_unlocked"][env_index].item()
                        ),
                        "orientation_phase_ready": int(
                            p5["orientation_phase_ready"][env_index].item()
                        ),
                        "reference_radius_m": float(
                            p5["orientation_reference_radius_m"][env_index].item()
                        ),
                        "reference_contraction_tracking_ready": int(
                            p5["reference_contraction_tracking_ready"][
                                env_index
                            ].item()
                        ),
                        "reference_contraction_maximum_radius_error_m": float(
                            p5["reference_contraction_maximum_radius_error_m"][
                                env_index
                            ].item()
                        ),
                        "reference_contraction_maximum_relative_radial_speed_mps": float(
                            p5[
                                "reference_contraction_maximum_relative_radial_speed_mps"
                            ][env_index].item()
                        ),
                        "radial_error_m": float(
                            p5_training_step["radial_error_m"][env_index].item()
                        ),
                        "angular_error_rad": float(
                            p5_training_step["angular_error_rad"][env_index].item()
                        ),
                        "target_speed_mps": float(
                            env._target_velocity[env_index, :2].norm().item()
                        ),
                        "target_acceleration_mps2": float(
                            p5["ring_target_acceleration_mps2"][env_index].item()
                        ),
                        "intercept_horizon_s": float(
                            p5["intercept_horizon_s"][env_index].item()
                        ),
                        "safety_correction_mps": float(
                            p5["safety_projection_correction_norm_mps"][env_index].item()
                        ),
                        "target_correction_mps": float(
                            p5["target_projection_correction_norm_mps"][env_index].item()
                        ),
                        "obstacle_correction_mps": float(
                            p5["obstacle_projection_correction_norm_mps"][env_index].item()
                        ),
                        "boundary_correction_mps": float(
                            p5["boundary_projection_correction_norm_mps"][env_index].item()
                        ),
                        "minimum_agent_clearance_m": float(
                            p5["min_agent_clearance_m"][env_index].item()
                        ),
                        "minimum_static_clearance_m": float(
                            p5["min_static_clearance_m"][env_index].item()
                        ),
                    }
                )
        episode_strong_ring_max_hold_steps = torch.maximum(
            episode_strong_ring_max_hold_steps,
            p5["strong_ring_hold_steps"],
        )
        all_visible = p5["orientation_all_visible"]
        staging_reached = p5["orientation_staging_reached"]
        episode_all_visible_steps += all_visible.to(torch.long)
        episode_staging_steps += staging_reached.to(torch.long)
        episode_staging_visible_steps += (
            staging_reached & all_visible
        ).to(torch.long)
        episode_phase_ready_steps += p5["orientation_phase_ready"].to(torch.long)
        unlocked_now = p5["orientation_contraction_unlocked"]
        first_unlock_now = unlocked_now & (episode_first_unlock_step < 0)
        episode_first_unlock_step[first_unlock_now] = episode_lengths[
            first_unlock_now
        ]
        episode_orientation_unlocked |= unlocked_now
        target_projection_active = p5["target_projection_active"]
        target_projection_corrected = p5["target_projection_corrected"]
        target_projection_correction_norm = p5[
            "target_projection_correction_norm_mps"
        ]
        episode_target_projection_calls += target_projection_active.long()
        episode_target_projection_corrected_steps += (
            target_projection_corrected.long()
        )
        episode_target_projection_constraint_corrections += p5[
            "target_projection_constraint_corrections"
        ].long()
        episode_target_projection_maximum_separation_fallback_constraints += p5[
            "target_projection_maximum_separation_fallback_constraints"
        ].long()
        episode_target_projection_unavoidable_steps += p5[
            "target_projection_unavoidable"
        ].long()
        episode_target_projection_unavoidable_constraints += p5[
            "target_projection_unavoidable_constraints"
        ].long()
        episode_target_projection_infeasible_steps += p5[
            "target_projection_infeasible"
        ].long()
        episode_target_projection_correction_norm_sum += (
            target_projection_correction_norm
        )
        episode_target_projection_correction_norm_max = torch.maximum(
            episode_target_projection_correction_norm_max,
            target_projection_correction_norm,
        )
        target_predicted_distance = torch.where(
            target_projection_active,
            p5["target_projection_min_predicted_center_distance_m"],
            torch.full_like(target_projection_correction_norm, torch.inf),
        )
        episode_target_projection_min_predicted_center_distance = torch.minimum(
            episode_target_projection_min_predicted_center_distance,
            target_predicted_distance,
        )
        target_maximum_attainable_distance = torch.where(
            target_projection_active,
            p5[
                "target_projection_minimum_maximum_attainable_center_distance_m"
            ],
            torch.full_like(target_projection_correction_norm, torch.inf),
        )
        episode_target_projection_minimum_maximum_attainable_center_distance = (
            torch.minimum(
                episode_target_projection_minimum_maximum_attainable_center_distance,
                target_maximum_attainable_distance,
            )
        )
        projection_active = p5["safety_projection_active"]
        projection_corrected = p5["safety_projection_corrected"]
        projection_correction_norm = p5["safety_projection_correction_norm_mps"]
        episode_safety_projection_calls += projection_active.long()
        episode_safety_projection_corrected_steps += projection_corrected.long()
        episode_safety_projection_pair_corrections += p5[
            "safety_projection_pair_corrections"
        ].long()
        episode_safety_projection_infeasible_steps += p5[
            "safety_projection_infeasible"
        ].long()
        episode_safety_projection_correction_norm_sum += projection_correction_norm
        episode_safety_projection_correction_norm_max = torch.maximum(
            episode_safety_projection_correction_norm_max,
            projection_correction_norm,
        )
        predicted_distance = torch.where(
            projection_active,
            p5["safety_projection_min_predicted_center_distance_m"],
            torch.full_like(projection_correction_norm, torch.inf),
        )
        episode_safety_projection_min_predicted_center_distance = torch.minimum(
            episode_safety_projection_min_predicted_center_distance,
            predicted_distance,
        )
        obstacle_projection_active = p5["obstacle_projection_active"]
        obstacle_projection_corrected = p5["obstacle_projection_corrected"]
        obstacle_projection_correction_norm = p5[
            "obstacle_projection_correction_norm_mps"
        ]
        episode_obstacle_projection_calls += obstacle_projection_active.long()
        episode_obstacle_projection_corrected_steps += (
            obstacle_projection_corrected.long()
        )
        episode_obstacle_projection_constraint_corrections += p5[
            "obstacle_projection_constraint_corrections"
        ].long()
        episode_obstacle_projection_infeasible_steps += p5[
            "obstacle_projection_infeasible"
        ].long()
        episode_obstacle_projection_correction_norm_sum += (
            obstacle_projection_correction_norm
        )
        episode_obstacle_projection_correction_norm_max = torch.maximum(
            episode_obstacle_projection_correction_norm_max,
            obstacle_projection_correction_norm,
        )
        obstacle_predicted_clearance = torch.where(
            obstacle_projection_active,
            p5["obstacle_projection_min_predicted_clearance_m"],
            torch.full_like(obstacle_projection_correction_norm, torch.inf),
        )
        episode_obstacle_projection_min_predicted_clearance = torch.minimum(
            episode_obstacle_projection_min_predicted_clearance,
            obstacle_predicted_clearance,
        )
        boundary_projection_active = p5["boundary_projection_active"]
        boundary_projection_corrected = p5["boundary_projection_corrected"]
        boundary_projection_correction_norm = p5[
            "boundary_projection_correction_norm_mps"
        ]
        episode_boundary_projection_calls += boundary_projection_active.long()
        episode_boundary_projection_corrected_steps += (
            boundary_projection_corrected.long()
        )
        episode_boundary_projection_constraint_corrections += p5[
            "boundary_projection_constraint_corrections"
        ].long()
        episode_boundary_projection_infeasible_steps += p5[
            "boundary_projection_infeasible"
        ].long()
        episode_boundary_projection_correction_norm_sum += (
            boundary_projection_correction_norm
        )
        episode_boundary_projection_correction_norm_max = torch.maximum(
            episode_boundary_projection_correction_norm_max,
            boundary_projection_correction_norm,
        )
        boundary_predicted_clearance = torch.where(
            boundary_projection_active,
            p5["boundary_projection_min_predicted_clearance_m"],
            torch.full_like(boundary_projection_correction_norm, torch.inf),
        )
        episode_boundary_projection_min_predicted_clearance = torch.minimum(
            episode_boundary_projection_min_predicted_clearance,
            boundary_predicted_clearance,
        )
        tangent_active_agents = p5["tangent_bypass_active_agent_count"].long()
        tangent_activations = p5["tangent_bypass_activation_count"].long()
        episode_tangent_bypass_active_steps += (tangent_active_agents > 0).long()
        episode_tangent_bypass_active_agent_steps += tangent_active_agents
        episode_tangent_bypass_activations += tangent_activations
        tangent_candidate_clearance = torch.where(
            tangent_active_agents > 0,
            p5["tangent_bypass_min_candidate_clearance_m"],
            torch.full_like(
                p5["tangent_bypass_min_candidate_clearance_m"], torch.inf
            ),
        )
        episode_tangent_bypass_min_candidate_clearance = torch.minimum(
            episode_tangent_bypass_min_candidate_clearance,
            tangent_candidate_clearance,
        )
        current_lost_steps = p5["lost_steps"]
        episode_maximum_lost_steps = torch.maximum(
            episode_maximum_lost_steps, current_lost_steps
        )
        team_lost_now = ~p5["team_visible"]
        episode_team_lost_steps += team_lost_now.long()
        visibility_status = p5["visibility_status_code"]
        for status_code in range(1, 5):
            episode_loss_reason_steps[:, status_code - 1] += (
                team_lost_now & (visibility_status == status_code)
            ).long()
        episode_min_agent_clearance = torch.minimum(
            episode_min_agent_clearance, p5["min_agent_clearance_m"]
        )
        episode_min_static_clearance = torch.minimum(
            episode_min_static_clearance, p5["min_static_clearance_m"]
        )
        episode_min_boundary_clearance = torch.minimum(
            episode_min_boundary_clearance,
            p5["min_pursuer_boundary_clearance_m"],
        )
        episode_dynamic_obstacle_max_displacement = torch.maximum(
            episode_dynamic_obstacle_max_displacement,
            p5["dynamic_obstacle_max_displacement_m"],
        )
        episode_dynamic_obstacle_max_speed = torch.maximum(
            episode_dynamic_obstacle_max_speed,
            p5["dynamic_obstacle_max_speed_mps"],
        )
        done = terminated[first] | truncated[first]
        if not done.any():
            continue
        training = extras[first]["p5_training"]
        for env_index in done.nonzero(as_tuple=False).flatten().tolist():
            if completed >= episodes:
                break
            current_episode_seed = int(p5["episode_seed"][env_index].item())
            if (
                current_episode_seed not in planned_seed_set
                or current_episode_seed in recorded_seeds
            ):
                continue
            training_success += int(p5["success_training"][env_index].item())
            success += int(p5["success_6m"][env_index].item())
            strict += int(p5["success_5m"][env_index].item())
            collided_agent = bool(p5["collision_agent"][env_index].item())
            collided_static = bool(p5["collision_static"][env_index].item())
            collided_dynamic = bool(p5["collision_dynamic"][env_index].item())
            collision += int(collided_agent or collided_static or collided_dynamic)
            collision_agent += int(collided_agent)
            collided_after_unlock = bool(
                collided_agent
                and episode_orientation_unlocked[env_index].item()
            )
            collision_agent_after_unlock += int(collided_after_unlock)
            collision_agent_before_unlock += int(
                collided_agent and not collided_after_unlock
            )
            collision_static += int(collided_static)
            collision_dynamic += int(collided_dynamic)
            contact += int(p5["contact_0p3m"][env_index].item())
            if env.cfg.target_buffer_failure_enabled:
                target_buffer_failures += int(p5["target_buffer_failure"][env_index].item())
            invalid += int(
                (
                    p5["boundary_invalid"][env_index]
                    | p5["height_invalid"][env_index]
                ).item()
            )
            boundary_invalid += int(p5["boundary_invalid"][env_index].item())
            height_invalid += int(p5["height_invalid"][env_index].item())
            timeout += int(p5["timeout"][env_index].item())
            episode_length = int(episode_lengths[env_index].item())
            length_sum += episode_length
            orientation_unlocked += int(
                episode_orientation_unlocked[env_index].item()
            )
            first_unlock_step = int(episode_first_unlock_step[env_index].item())
            if first_unlock_step >= 0:
                first_unlock_step_sum += first_unlock_step
                first_unlock_step_count += 1
            target_projection_call_count = int(
                episode_target_projection_calls[env_index].item()
            )
            target_projection_corrected_count = int(
                episode_target_projection_corrected_steps[env_index].item()
            )
            target_constraint_correction_count = int(
                episode_target_projection_constraint_corrections[env_index].item()
            )
            target_maximum_separation_fallback_count = int(
                episode_target_projection_maximum_separation_fallback_constraints[
                    env_index
                ].item()
            )
            target_projection_unavoidable_count = int(
                episode_target_projection_unavoidable_steps[env_index].item()
            )
            target_unavoidable_constraint_count = int(
                episode_target_projection_unavoidable_constraints[env_index].item()
            )
            target_projection_infeasible_count = int(
                episode_target_projection_infeasible_steps[env_index].item()
            )
            target_projection_correction_sum = float(
                episode_target_projection_correction_norm_sum[env_index].item()
            )
            target_projection_correction_max = float(
                episode_target_projection_correction_norm_max[env_index].item()
            )
            target_projection_min_distance = float(
                episode_target_projection_min_predicted_center_distance[
                    env_index
                ].item()
            )
            target_projection_minimum_maximum_attainable_distance = float(
                episode_target_projection_minimum_maximum_attainable_center_distance[
                    env_index
                ].item()
            )
            projection_calls = int(
                episode_safety_projection_calls[env_index].item()
            )
            projection_corrected_steps = int(
                episode_safety_projection_corrected_steps[env_index].item()
            )
            projection_pair_corrections = int(
                episode_safety_projection_pair_corrections[env_index].item()
            )
            projection_infeasible_steps = int(
                episode_safety_projection_infeasible_steps[env_index].item()
            )
            projection_correction_sum = float(
                episode_safety_projection_correction_norm_sum[env_index].item()
            )
            projection_correction_max = float(
                episode_safety_projection_correction_norm_max[env_index].item()
            )
            projection_min_predicted_distance = float(
                episode_safety_projection_min_predicted_center_distance[env_index].item()
            )
            obstacle_projection_call_count = int(
                episode_obstacle_projection_calls[env_index].item()
            )
            obstacle_projection_corrected_count = int(
                episode_obstacle_projection_corrected_steps[env_index].item()
            )
            obstacle_constraint_correction_count = int(
                episode_obstacle_projection_constraint_corrections[env_index].item()
            )
            obstacle_projection_infeasible_count = int(
                episode_obstacle_projection_infeasible_steps[env_index].item()
            )
            obstacle_projection_correction_sum = float(
                episode_obstacle_projection_correction_norm_sum[env_index].item()
            )
            obstacle_projection_correction_max = float(
                episode_obstacle_projection_correction_norm_max[env_index].item()
            )
            obstacle_projection_min_clearance = float(
                episode_obstacle_projection_min_predicted_clearance[env_index].item()
            )
            boundary_projection_call_count = int(
                episode_boundary_projection_calls[env_index].item()
            )
            boundary_projection_corrected_count = int(
                episode_boundary_projection_corrected_steps[env_index].item()
            )
            boundary_constraint_correction_count = int(
                episode_boundary_projection_constraint_corrections[
                    env_index
                ].item()
            )
            boundary_projection_infeasible_count = int(
                episode_boundary_projection_infeasible_steps[env_index].item()
            )
            boundary_projection_correction_sum = float(
                episode_boundary_projection_correction_norm_sum[env_index].item()
            )
            boundary_projection_correction_max = float(
                episode_boundary_projection_correction_norm_max[env_index].item()
            )
            boundary_projection_min_clearance = float(
                episode_boundary_projection_min_predicted_clearance[
                    env_index
                ].item()
            )
            tangent_active_step_count = int(
                episode_tangent_bypass_active_steps[env_index].item()
            )
            tangent_active_agent_step_count = int(
                episode_tangent_bypass_active_agent_steps[env_index].item()
            )
            tangent_activation_count = int(
                episode_tangent_bypass_activations[env_index].item()
            )
            tangent_min_candidate_clearance = float(
                episode_tangent_bypass_min_candidate_clearance[env_index].item()
            )
            slot_active_steps = int(
                p5["slot_active_steps"][env_index].item()
            )
            slot_handoff_count = int(
                p5["slot_handoff_count"][env_index].item()
            )
            slot_terminal_error = float(
                p5["slot_mean_error_m"][env_index].item()
            )
            hybrid_supervisor_enabled = bool(
                p5["hybrid_phase_supervisor_enabled"][env_index].item()
            )
            hybrid_traditional_steps = int(
                p5["hybrid_traditional_steps"][env_index].item()
            )
            hybrid_rl_steps = int(
                p5["hybrid_rl_steps"][env_index].item()
            )
            policy_residual_active_steps = int(
                p5["slot_policy_residual_active_steps"][env_index].item()
            )
            policy_residual_norm_sum = float(
                p5["slot_policy_residual_norm_sum_mps"][env_index].item()
            )
            policy_residual_radial_support_sum = float(
                p5["slot_policy_residual_radial_support_sum_mps"][
                    env_index
                ].item()
            )
            policy_residual_radial_opposition_sum = float(
                p5["slot_policy_residual_radial_opposition_sum_mps"][
                    env_index
                ].item()
            )
            policy_residual_phase_support_sum = float(
                p5["slot_policy_residual_phase_support_sum_mps"][
                    env_index
                ].item()
            )
            policy_residual_phase_opposition_sum = float(
                p5["slot_policy_residual_phase_opposition_sum_mps"][
                    env_index
                ].item()
            )
            speed_limit_active_agent_fraction_sum = float(
                p5["slot_speed_limit_active_agent_fraction_sum"][
                    env_index
                ].item()
            )
            speed_limit_mean_scale_sum = float(
                p5["slot_speed_limit_mean_scale_sum"][env_index].item()
            )
            speed_limit_forward_loss_sum = float(
                p5["slot_speed_limit_forward_loss_sum_mps"][
                    env_index
                ].item()
            )
            requested_tangential_speed_sum = float(
                p5["slot_requested_tangential_speed_sum_mps"][
                    env_index
                ].item()
            )
            executed_tangential_speed_sum = float(
                p5["slot_executed_tangential_speed_sum_mps"][
                    env_index
                ].item()
            )
            slot_speed_limit_active_agent_fraction_sum += (
                speed_limit_active_agent_fraction_sum
            )
            slot_speed_limit_mean_scale_sum += speed_limit_mean_scale_sum
            slot_speed_limit_forward_loss_sum_mps += speed_limit_forward_loss_sum
            slot_requested_tangential_speed_sum_mps += (
                requested_tangential_speed_sum
            )
            slot_executed_tangential_speed_sum_mps += (
                executed_tangential_speed_sum
            )
            if hybrid_supervisor_enabled:
                hybrid_supervisor_episodes += 1
                hybrid_traditional_steps_sum += hybrid_traditional_steps
                hybrid_rl_steps_sum += hybrid_rl_steps
                hybrid_rl_active_episodes += int(hybrid_rl_steps > 0)
                policy_residual_active_steps_sum += policy_residual_active_steps
                policy_residual_norm_sum_mps += policy_residual_norm_sum
                policy_residual_radial_support_sum_mps += (
                    policy_residual_radial_support_sum
                )
                policy_residual_radial_opposition_sum_mps += (
                    policy_residual_radial_opposition_sum
                )
                policy_residual_phase_support_sum_mps += (
                    policy_residual_phase_support_sum
                )
                policy_residual_phase_opposition_sum_mps += (
                    policy_residual_phase_opposition_sum
                )
            maximum_lost_steps = int(
                episode_maximum_lost_steps[env_index].item()
            )
            episode_lost_steps = int(
                episode_team_lost_steps[env_index].item()
            )
            reason_steps = [
                int(value)
                for value in episode_loss_reason_steps[env_index].tolist()
            ]
            recaptures = int(p5["recapture_count"][env_index].item())
            loss_reacquisition_events = int(
                p5["loss_reacquisition_events"][env_index].item()
            )
            loss_reacquisition_completed = int(
                p5["loss_reacquisition_completed"][env_index].item()
            )
            loss_reacquisition_active_steps = int(
                p5["loss_reacquisition_active_steps"][env_index].item()
            )
            loss_reacquisition_first_completed = bool(
                p5["loss_reacquisition_first_completed"][env_index].item()
            )
            loss_reacquisition_first_delay_steps = int(
                p5["loss_reacquisition_first_delay_steps"][env_index].item()
            )
            loss_reacquisition_max_segment_steps = int(
                p5["loss_reacquisition_max_segment_steps"][env_index].item()
            )
            loss_reacquisition_scout_index = int(
                p5["loss_reacquisition_scout_index"][env_index].item()
            )
            loss_reacquisition_flank_events = int(
                p5["loss_reacquisition_flank_events"][env_index].item()
            )
            loss_reacquisition_flank_active_steps = int(
                p5["loss_reacquisition_flank_active_steps"][env_index].item()
            )
            loss_reacquisition_flank_fallback_steps = int(
                p5["loss_reacquisition_flank_fallback_steps"][env_index].item()
            )
            loss_reacquisition_flank_blocker_index = int(
                p5["loss_reacquisition_flank_blocker_index"][env_index].item()
            )
            loss_reacquisition_flank_path_clearance = float(
                p5["loss_reacquisition_flank_minimum_path_clearance_m"][
                    env_index
                ].item()
            )
            loss_reacquisition_flank_los_clearance = float(
                p5["loss_reacquisition_flank_minimum_los_clearance_m"][
                    env_index
                ].item()
            )
            if not math.isfinite(loss_reacquisition_flank_path_clearance):
                loss_reacquisition_flank_path_clearance = -1.0
            if not math.isfinite(loss_reacquisition_flank_los_clearance):
                loss_reacquisition_flank_los_clearance = -1.0
            loss_reacquisition_sector_valid_steps = int(
                p5["loss_reacquisition_sector_valid_steps"][env_index].item()
            )
            loss_reacquisition_sector_fallback_steps = int(
                p5["loss_reacquisition_sector_fallback_steps"][env_index].item()
            )
            loss_reacquisition_sector_no_path_candidate_steps = int(
                p5["loss_reacquisition_sector_no_path_candidate_steps"][
                    env_index
                ].item()
            )
            loss_reacquisition_sector_no_boundary_candidate_steps = int(
                p5["loss_reacquisition_sector_no_boundary_candidate_steps"][
                    env_index
                ].item()
            )
            loss_reacquisition_sector_phase_index = int(
                p5["loss_reacquisition_sector_phase_index"][env_index].item()
            )
            loss_reacquisition_sector_path_clearance = float(
                p5["loss_reacquisition_sector_minimum_path_clearance_m"][
                    env_index
                ].item()
            )
            if not math.isfinite(loss_reacquisition_sector_path_clearance):
                loss_reacquisition_sector_path_clearance = -1.0
            target_displacement = float(
                p5["target_episode_displacement_m"][env_index].item()
            )
            terminal_target_speed = float(
                p5["target_velocity_mps"][env_index, :2].norm().item()
            )
            evader_episode_speed_limit = float(
                p5["evader_episode_speed_limit_mps"][env_index].item()
            )
            strong_ring_max_hold_steps = int(
                episode_strong_ring_max_hold_steps[env_index].item()
            )
            dynamic_obstacle_displacement = float(
                episode_dynamic_obstacle_max_displacement[env_index].item()
            )
            dynamic_obstacle_speed = float(
                episode_dynamic_obstacle_max_speed[env_index].item()
            )
            obstacle_layout_json = json.dumps(
                [
                    [round(float(value), 5) for value in point]
                    for point in p5["obstacle_xy_local_m"][env_index]
                    .detach()
                    .cpu()
                    .tolist()
                ],
                separators=(",", ":"),
            )
            obstacle_layout_signatures.add(
                hashlib.sha256(obstacle_layout_json.encode("utf-8")).hexdigest()
            )
            target_displacement_sum += target_displacement
            terminal_target_speed_sum += terminal_target_speed
            evader_episode_speed_limit_sum += evader_episode_speed_limit
            evader_episode_speed_limit_min = min(
                evader_episode_speed_limit_min, evader_episode_speed_limit
            )
            evader_episode_speed_limit_max = max(
                evader_episode_speed_limit_max, evader_episode_speed_limit
            )
            strong_ring_max_hold_steps_sum += strong_ring_max_hold_steps
            strong_ring_max_hold_steps_global = max(
                strong_ring_max_hold_steps_global, strong_ring_max_hold_steps
            )
            dynamic_obstacle_displacement_sum += dynamic_obstacle_displacement
            dynamic_obstacle_displacement_max = max(
                dynamic_obstacle_displacement_max,
                dynamic_obstacle_displacement,
            )
            dynamic_obstacle_speed_sum += dynamic_obstacle_speed
            loss_episodes += int(maximum_lost_steps > 0)
            long_loss_2s_episodes += int(maximum_lost_steps >= 20)
            maximum_lost_steps_sum += maximum_lost_steps
            maximum_lost_steps_global = max(
                maximum_lost_steps_global, maximum_lost_steps
            )
            team_lost_steps += episode_lost_steps
            recapture_count_sum += recaptures
            loss_reacquisition_event_sum += loss_reacquisition_events
            loss_reacquisition_completed_sum += loss_reacquisition_completed
            loss_reacquisition_active_step_sum += loss_reacquisition_active_steps
            loss_reacquisition_eligible_episodes += int(
                loss_reacquisition_events > 0
            )
            loss_reacquisition_first_completed_episodes += int(
                loss_reacquisition_first_completed
            )
            loss_reacquisition_first_within_2s_episodes += int(
                loss_reacquisition_first_completed
                and 0 <= loss_reacquisition_first_delay_steps <= 20
            )
            if loss_reacquisition_first_completed:
                loss_reacquisition_first_delay_step_sum += (
                    loss_reacquisition_first_delay_steps
                )
                loss_reacquisition_first_delay_count += 1
            loss_reacquisition_flank_event_sum += loss_reacquisition_flank_events
            loss_reacquisition_flank_episode_count += int(
                loss_reacquisition_flank_events > 0
            )
            loss_reacquisition_flank_active_step_sum += (
                loss_reacquisition_flank_active_steps
            )
            loss_reacquisition_flank_fallback_step_sum += (
                loss_reacquisition_flank_fallback_steps
            )
            if loss_reacquisition_flank_path_clearance >= 0.0:
                loss_reacquisition_flank_minimum_path_clearance = min(
                    loss_reacquisition_flank_minimum_path_clearance,
                    loss_reacquisition_flank_path_clearance,
                )
            if loss_reacquisition_flank_los_clearance >= 0.0:
                loss_reacquisition_flank_minimum_los_clearance = min(
                    loss_reacquisition_flank_minimum_los_clearance,
                    loss_reacquisition_flank_los_clearance,
                )
            loss_reacquisition_sector_valid_step_sum += (
                loss_reacquisition_sector_valid_steps
            )
            loss_reacquisition_sector_fallback_step_sum += (
                loss_reacquisition_sector_fallback_steps
            )
            loss_reacquisition_sector_episode_count += int(
                loss_reacquisition_sector_valid_steps > 0
            )
            if loss_reacquisition_sector_path_clearance >= 0.0:
                loss_reacquisition_sector_minimum_path_clearance = min(
                    loss_reacquisition_sector_minimum_path_clearance,
                    loss_reacquisition_sector_path_clearance,
                )
            for name, value in zip(loss_reason_steps, reason_steps):
                loss_reason_steps[name] += value
            target_projection_episodes += int(target_projection_call_count > 0)
            target_projection_calls += target_projection_call_count
            target_projection_corrected_steps += target_projection_corrected_count
            target_projection_constraint_corrections += (
                target_constraint_correction_count
            )
            target_projection_maximum_separation_fallback_constraints += (
                target_maximum_separation_fallback_count
            )
            target_projection_unavoidable_steps += (
                target_projection_unavoidable_count
            )
            target_projection_unavoidable_constraints += (
                target_unavoidable_constraint_count
            )
            target_projection_infeasible_steps += target_projection_infeasible_count
            target_projection_correction_norm_sum += target_projection_correction_sum
            target_projection_correction_norm_max = max(
                target_projection_correction_norm_max,
                target_projection_correction_max,
            )
            target_projection_min_predicted_center_distance = min(
                target_projection_min_predicted_center_distance,
                target_projection_min_distance,
            )
            target_projection_minimum_maximum_attainable_center_distance = min(
                target_projection_minimum_maximum_attainable_center_distance,
                target_projection_minimum_maximum_attainable_distance,
            )
            safety_projection_episodes += int(projection_calls > 0)
            safety_projection_calls += projection_calls
            safety_projection_corrected_steps += projection_corrected_steps
            safety_projection_pair_corrections += projection_pair_corrections
            safety_projection_infeasible_steps += projection_infeasible_steps
            safety_projection_correction_norm_sum += projection_correction_sum
            safety_projection_correction_norm_max = max(
                safety_projection_correction_norm_max, projection_correction_max
            )
            safety_projection_min_predicted_center_distance = min(
                safety_projection_min_predicted_center_distance,
                projection_min_predicted_distance,
            )
            obstacle_projection_episodes += int(
                obstacle_projection_call_count > 0
            )
            obstacle_projection_calls += obstacle_projection_call_count
            obstacle_projection_corrected_steps += (
                obstacle_projection_corrected_count
            )
            obstacle_projection_constraint_corrections += (
                obstacle_constraint_correction_count
            )
            obstacle_projection_infeasible_steps += (
                obstacle_projection_infeasible_count
            )
            obstacle_projection_correction_norm_sum += (
                obstacle_projection_correction_sum
            )
            obstacle_projection_correction_norm_max = max(
                obstacle_projection_correction_norm_max,
                obstacle_projection_correction_max,
            )
            obstacle_projection_min_predicted_clearance = min(
                obstacle_projection_min_predicted_clearance,
                obstacle_projection_min_clearance,
            )
            boundary_projection_episodes += int(
                boundary_projection_call_count > 0
            )
            boundary_projection_calls += boundary_projection_call_count
            boundary_projection_corrected_steps += (
                boundary_projection_corrected_count
            )
            boundary_projection_constraint_corrections += (
                boundary_constraint_correction_count
            )
            boundary_projection_infeasible_steps += (
                boundary_projection_infeasible_count
            )
            boundary_projection_correction_norm_sum += (
                boundary_projection_correction_sum
            )
            boundary_projection_correction_norm_max = max(
                boundary_projection_correction_norm_max,
                boundary_projection_correction_max,
            )
            boundary_projection_min_predicted_clearance = min(
                boundary_projection_min_predicted_clearance,
                boundary_projection_min_clearance,
            )
            tangent_bypass_episodes += int(tangent_active_step_count > 0)
            tangent_bypass_active_steps += tangent_active_step_count
            tangent_bypass_active_agent_steps += tangent_active_agent_step_count
            tangent_bypass_activations += tangent_activation_count
            tangent_bypass_min_candidate_clearance = min(
                tangent_bypass_min_candidate_clearance,
                tangent_min_candidate_clearance,
            )
            slot_guidance_episodes += int(slot_active_steps > 0)
            slot_active_steps_sum += slot_active_steps
            slot_handoff_count_sum += slot_handoff_count
            slot_terminal_error_sum += slot_terminal_error
            all_visible_fraction = (
                float(episode_all_visible_steps[env_index].item()) / episode_length
            )
            staging_steps = int(episode_staging_steps[env_index].item())
            staging_visible_fraction = (
                float(episode_staging_visible_steps[env_index].item()) / staging_steps
                if staging_steps > 0
                else 0.0
            )
            all_visible_fraction_sum += all_visible_fraction
            staging_visible_fraction_sum += staging_visible_fraction
            terminal_yaw_error = float(
                p5["orientation_yaw_error_max_rad"][env_index].item()
            )
            terminal_yaw_error_sum += terminal_yaw_error
            terminal_staging_fraction = float(
                p5["orientation_staging_fraction"][env_index].item()
            )
            terminal_alignment_fraction = float(
                p5["orientation_alignment_fraction"][env_index].item()
            )
            terminal_phase_ready_now = bool(
                p5["orientation_phase_ready"][env_index].item()
            )
            phase_ready_fraction = float(
                episode_phase_ready_steps[env_index].item()
            ) / episode_length
            terminal_intercept_reachable_fraction = float(
                p5["intercept_reachable_fraction"][env_index].item()
            )
            terminal_intercept_horizon = float(
                p5["intercept_horizon_s"][env_index].item()
            )
            terminal_staging_fraction_sum += terminal_staging_fraction
            terminal_alignment_fraction_sum += terminal_alignment_fraction
            terminal_phase_ready += int(terminal_phase_ready_now)
            phase_ready_fraction_sum += phase_ready_fraction
            terminal_intercept_reachable_fraction_sum += (
                terminal_intercept_reachable_fraction
            )
            terminal_intercept_horizon_sum += terminal_intercept_horizon
            intercept_blocker_reachable += int(
                p5["intercept_blocker_reachable"][env_index].item()
            )
            radial_error_sum += float(training["radial_error_m"][env_index].item())
            angular_error_sum += float(training["angular_error_rad"][env_index].item())
            minimum_clearances = {
                "agent": float(episode_min_agent_clearance[env_index].item()),
                "static": float(episode_min_static_clearance[env_index].item()),
                "boundary": float(episode_min_boundary_clearance[env_index].item()),
            }
            for name, value in minimum_clearances.items():
                clearance_sums[name] += value
                clearance_minima[name] = min(clearance_minima[name], value)
            if episode_records is not None:
                episode_records.append(
                    {
                        "episode_index": completed,
                        **({"target_buffer_failure": int(p5["target_buffer_failure"][env_index].item())}
                           if env.cfg.target_buffer_failure_enabled else {}),
                        "joint_lookahead_active_steps": int(training["joint_lookahead_active_steps"][env_index].item()),
                        "joint_lookahead_corrected_steps": int(training["joint_lookahead_corrected_steps"][env_index].item()),
                        "dual_observer_active_steps": int(training["dual_observer_active_steps"][env_index].item()),
                        "dual_observer_role_switches": int(training["dual_observer_role_switches"][env_index].item()),
                        "episode_seed": int(p5["episode_seed"][env_index].item()),
                        "episode_steps": int(episode_lengths[env_index].item()),
                        "capture_protocol": REGION_PROTOCOL if env.cfg.region_capture_enabled else "legacy",
                        "success_training": int(
                            p5["success_training"][env_index].item()
                        ),
                        "success_6m": int(p5["success_6m"][env_index].item()),
                        "success_5m": int(p5["success_5m"][env_index].item()),
                        "strong_ring_enabled": int(
                            p5["strong_ring_enabled"][env_index].item()
                        ),
                        "strong_ring_success": int(
                            p5["strong_ring_success"][env_index].item()
                        ),
                        "maximum_strong_ring_hold_steps": (
                            strong_ring_max_hold_steps
                        ),
                        "terminal_strong_ring_hold_steps": int(
                            p5["strong_ring_hold_steps"][env_index].item()
                        ),
                        "terminal_strong_ring_now": int(
                            p5["strong_ring_now"][env_index].item()
                        ),
                        "terminal_strong_ring_minimum_distance_m": float(
                            p5["strong_ring_minimum_distance_m"][env_index].item()
                        ),
                        "terminal_strong_ring_maximum_distance_m": float(
                            p5["strong_ring_maximum_distance_m"][env_index].item()
                        ),
                        "terminal_strong_ring_mean_gap_error_rad": float(
                            p5["strong_ring_mean_gap_error_rad"][env_index].item()
                        ),
                        "terminal_training_capture_inside_count": int(
                            p5["training_capture_inside_count"][env_index].item()
                        ),
                        "terminal_training_capture_max_gap_rad": float(
                            p5["training_capture_max_gap_rad"][env_index].item()
                        ),
                        "terminal_pursuer_target_distances_m": json.dumps(
                            [
                                float(value)
                                for value in p5["pursuer_target_distance_m"][
                                    env_index
                                ]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            separators=(",", ":"),
                        ),
                        "collision_agent": int(collided_agent),
                        "collision_agent_before_unlock": int(
                            collided_agent and not collided_after_unlock
                        ),
                        "collision_agent_after_unlock": int(
                            collided_after_unlock
                        ),
                        "collision_static": int(collided_static),
                        "collision_dynamic": int(collided_dynamic),
                        "contact_0p3m": int(
                            p5["contact_0p3m"][env_index].item()
                        ),
                        "boundary_invalid": int(
                            p5["boundary_invalid"][env_index].item()
                        ),
                        "height_invalid": int(
                            p5["height_invalid"][env_index].item()
                        ),
                        "timeout": int(p5["timeout"][env_index].item()),
                        "minimum_agent_clearance_m": minimum_clearances["agent"],
                        "minimum_static_clearance_m": minimum_clearances["static"],
                        "minimum_boundary_clearance_m": minimum_clearances[
                            "boundary"
                        ],
                        "terminal_radial_error_m": float(
                            training["radial_error_m"][env_index].item()
                        ),
                        "terminal_angular_error_rad": float(
                            training["angular_error_rad"][env_index].item()
                        ),
                        "hybrid_phase_supervisor_enabled": int(
                            hybrid_supervisor_enabled
                        ),
                        "hybrid_traditional_steps": hybrid_traditional_steps,
                        "hybrid_rl_steps": hybrid_rl_steps,
                        "hybrid_rl_step_fraction": (
                            hybrid_rl_steps
                            / max(
                                hybrid_traditional_steps + hybrid_rl_steps,
                                1,
                            )
                        ),
                        "policy_residual_active_steps": (
                            policy_residual_active_steps
                        ),
                        "mean_policy_residual_norm_mps": (
                            policy_residual_norm_sum
                            / max(policy_residual_active_steps, 1)
                        ),
                        "mean_policy_residual_radial_support_mps": (
                            policy_residual_radial_support_sum
                            / max(policy_residual_active_steps, 1)
                        ),
                        "mean_policy_residual_radial_opposition_mps": (
                            policy_residual_radial_opposition_sum
                            / max(policy_residual_active_steps, 1)
                        ),
                        "mean_policy_residual_phase_support_mps": (
                            policy_residual_phase_support_sum
                            / max(policy_residual_active_steps, 1)
                        ),
                        "mean_policy_residual_phase_opposition_mps": (
                            policy_residual_phase_opposition_sum
                            / max(policy_residual_active_steps, 1)
                        ),
                        "mean_slot_speed_limit_active_agent_fraction": (
                            speed_limit_active_agent_fraction_sum
                            / max(slot_active_steps, 1)
                        ),
                        "mean_slot_speed_limit_scale": (
                            speed_limit_mean_scale_sum
                            / max(slot_active_steps, 1)
                        ),
                        "mean_slot_speed_limit_forward_loss_mps": (
                            speed_limit_forward_loss_sum
                            / max(slot_active_steps, 1)
                        ),
                        "mean_slot_requested_tangential_speed_mps": (
                            requested_tangential_speed_sum
                            / max(slot_active_steps, 1)
                        ),
                        "mean_slot_executed_tangential_speed_mps": (
                            executed_tangential_speed_sum
                            / max(slot_active_steps, 1)
                        ),
                        "orientation_unlocked": int(
                            episode_orientation_unlocked[env_index].item()
                        ),
                        "first_orientation_unlock_step": first_unlock_step,
                        "all_visible_step_fraction": all_visible_fraction,
                        "staging_all_visible_step_fraction": (
                            staging_visible_fraction
                        ),
                        "terminal_orientation_yaw_error_max_rad": (
                            terminal_yaw_error
                        ),
                        "terminal_orientation_reference_radius_m": float(
                            p5["orientation_reference_radius_m"][env_index].item()
                        ),
                        "terminal_orientation_alignment_steps": int(
                            p5["orientation_alignment_steps"][env_index].item()
                        ),
                        "terminal_orientation_staging_fraction": (
                            terminal_staging_fraction
                        ),
                        "terminal_orientation_alignment_fraction": (
                            terminal_alignment_fraction
                        ),
                        "terminal_orientation_phase_ready": int(
                            terminal_phase_ready_now
                        ),
                        "orientation_phase_ready_step_fraction": (
                            phase_ready_fraction
                        ),
                        "terminal_intercept_reachable_fraction": (
                            terminal_intercept_reachable_fraction
                        ),
                        "terminal_intercept_horizon_s": (
                            terminal_intercept_horizon
                        ),
                        "terminal_intercept_mean_gap_m": float(
                            p5["intercept_mean_gap_m"][env_index].item()
                        ),
                        "terminal_intercept_blocker_agent_index": int(
                            p5["intercept_blocker_agent_index"][env_index].item()
                        ),
                        "terminal_intercept_blocker_reachable": int(
                            p5["intercept_blocker_reachable"][env_index].item()
                        ),
                        "terminal_boundary_projection_corrected": int(
                            p5["boundary_projection_corrected"][env_index].item()
                        ),
                        "terminal_boundary_projection_correction_norm_mps": float(
                            p5["boundary_projection_correction_norm_mps"][
                                env_index
                            ].item()
                        ),
                        "terminal_boundary_projection_constraint_corrections": int(
                            p5["boundary_projection_constraint_corrections"][
                                env_index
                            ].item()
                        ),
                        "terminal_boundary_projection_min_predicted_clearance_m": float(
                            p5["boundary_projection_min_predicted_clearance_m"][
                                env_index
                            ].item()
                        ),
                        "terminal_boundary_projection_infeasible": int(
                            p5["boundary_projection_infeasible"][env_index].item()
                        ),
                        "boundary_projection_enabled": int(
                            p5["boundary_projection_enabled"][env_index].item()
                        ),
                        "boundary_projection_call_steps": (
                            boundary_projection_call_count
                        ),
                        "boundary_projection_corrected_steps": (
                            boundary_projection_corrected_count
                        ),
                        "boundary_projection_constraint_corrections": (
                            boundary_constraint_correction_count
                        ),
                        "boundary_projection_infeasible_steps": (
                            boundary_projection_infeasible_count
                        ),
                        "boundary_projection_correction_norm_sum_mps": (
                            boundary_projection_correction_sum
                        ),
                        "boundary_projection_correction_norm_max_mps": (
                            boundary_projection_correction_max
                        ),
                        "boundary_projection_min_predicted_clearance_m": (
                            boundary_projection_min_clearance
                        ),
                        "maximum_lost_steps": maximum_lost_steps,
                        "maximum_lost_seconds": maximum_lost_steps * 0.1,
                        "team_lost_step_count": episode_lost_steps,
                        "pure_range_lost_steps": reason_steps[0],
                        "pure_fov_lost_steps": reason_steps[1],
                        "pure_occlusion_lost_steps": reason_steps[2],
                        "mixed_loss_steps": reason_steps[3],
                        "recapture_count": recaptures,
                        "loss_reacquisition_guidance_enabled": int(
                            p5["loss_reacquisition_guidance_enabled"][
                                env_index
                            ].item()
                        ),
                        "loss_reacquisition_event_count": (
                            loss_reacquisition_events
                        ),
                        "loss_reacquisition_completed_count": (
                            loss_reacquisition_completed
                        ),
                        "loss_reacquisition_active_steps": (
                            loss_reacquisition_active_steps
                        ),
                        "loss_reacquisition_first_completed": int(
                            loss_reacquisition_first_completed
                        ),
                        "loss_reacquisition_first_delay_steps": (
                            loss_reacquisition_first_delay_steps
                        ),
                        "loss_reacquisition_first_delay_s": (
                            loss_reacquisition_first_delay_steps * 0.1
                            if loss_reacquisition_first_delay_steps >= 0
                            else -1.0
                        ),
                        "loss_reacquisition_first_within_2s": int(
                            loss_reacquisition_first_completed
                            and 0 <= loss_reacquisition_first_delay_steps <= 20
                        ),
                        "loss_reacquisition_max_segment_steps": (
                            loss_reacquisition_max_segment_steps
                        ),
                        "loss_reacquisition_terminal_scout_index": (
                            loss_reacquisition_scout_index
                        ),
                        "loss_reacquisition_flank_enabled": int(
                            p5["loss_reacquisition_flank_enabled"][
                                env_index
                            ].item()
                        ),
                        "loss_reacquisition_flank_event_count": (
                            loss_reacquisition_flank_events
                        ),
                        "loss_reacquisition_flank_active_steps": (
                            loss_reacquisition_flank_active_steps
                        ),
                        "loss_reacquisition_flank_fallback_steps": (
                            loss_reacquisition_flank_fallback_steps
                        ),
                        "loss_reacquisition_terminal_flank_blocker_index": (
                            loss_reacquisition_flank_blocker_index
                        ),
                        "loss_reacquisition_minimum_flank_path_clearance_m": (
                            loss_reacquisition_flank_path_clearance
                        ),
                        "loss_reacquisition_minimum_flank_los_clearance_m": (
                            loss_reacquisition_flank_los_clearance
                        ),
                        "loss_reacquisition_sector_enabled": int(
                            p5["loss_reacquisition_sector_enabled"][
                                env_index
                            ].item()
                        ),
                        "loss_reacquisition_sector_valid_steps": (
                            loss_reacquisition_sector_valid_steps
                        ),
                        "loss_reacquisition_sector_fallback_steps": (
                            loss_reacquisition_sector_fallback_steps
                        ),
                        "loss_reacquisition_sector_no_path_candidate_steps": (
                            loss_reacquisition_sector_no_path_candidate_steps
                        ),
                        "loss_reacquisition_sector_no_boundary_candidate_steps": (
                            loss_reacquisition_sector_no_boundary_candidate_steps
                        ),
                        "loss_reacquisition_terminal_sector_phase_index": (
                            loss_reacquisition_sector_phase_index
                        ),
                        "loss_reacquisition_minimum_sector_path_clearance_m": (
                            loss_reacquisition_sector_path_clearance
                        ),
                        "target_episode_displacement_m": target_displacement,
                        "terminal_target_speed_mps": terminal_target_speed,
                        "evader_episode_speed_limit_mps": (
                            evader_episode_speed_limit
                        ),
                        "maximum_dynamic_obstacle_displacement_m": (
                            dynamic_obstacle_displacement
                        ),
                        "maximum_dynamic_obstacle_speed_mps": (
                            dynamic_obstacle_speed
                        ),
                        "terminal_target_xy_local_m": json.dumps(
                            [
                                float(value)
                                for value in p5["target_position_local_m"][
                                    env_index, :2
                                ]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            separators=(",", ":"),
                        ),
                        "obstacle_layout_xy_local_m": obstacle_layout_json,
                        "target_projection_enabled": int(
                            p5["target_projection_enabled"][env_index].item()
                        ),
                        "target_projection_call_steps": (
                            target_projection_call_count
                        ),
                        "target_projection_corrected_steps": (
                            target_projection_corrected_count
                        ),
                        "target_projection_constraint_corrections": (
                            target_constraint_correction_count
                        ),
                        "target_projection_maximum_separation_fallback_constraints": (
                            target_maximum_separation_fallback_count
                        ),
                        "target_projection_unavoidable_steps": (
                            target_projection_unavoidable_count
                        ),
                        "target_projection_unavoidable_constraints": (
                            target_unavoidable_constraint_count
                        ),
                        "target_projection_infeasible_steps": (
                            target_projection_infeasible_count
                        ),
                        "target_projection_correction_norm_sum_mps": (
                            target_projection_correction_sum
                        ),
                        "target_projection_correction_norm_max_mps": (
                            target_projection_correction_max
                        ),
                        "target_projection_min_predicted_center_distance_m": (
                            target_projection_min_distance
                        ),
                        "target_projection_minimum_maximum_attainable_center_distance_m": (
                            target_projection_minimum_maximum_attainable_distance
                        ),
                        "safety_projection_enabled": int(
                            p5["safety_projection_enabled"][env_index].item()
                        ),
                        "safety_projection_restorative_enabled": int(
                            p5["safety_projection_restorative_enabled"][
                                env_index
                            ].item()
                        ),
                        "safety_projection_call_steps": projection_calls,
                        "safety_projection_corrected_steps": (
                            projection_corrected_steps
                        ),
                        "safety_projection_pair_corrections": (
                            projection_pair_corrections
                        ),
                        "safety_projection_infeasible_steps": (
                            projection_infeasible_steps
                        ),
                        "safety_projection_correction_norm_sum_mps": (
                            projection_correction_sum
                        ),
                        "safety_projection_correction_norm_max_mps": (
                            projection_correction_max
                        ),
                        "safety_projection_min_predicted_center_distance_m": (
                            projection_min_predicted_distance
                        ),
                        "obstacle_projection_enabled": int(
                            p5["obstacle_projection_enabled"][env_index].item()
                        ),
                        "obstacle_projection_call_steps": (
                            obstacle_projection_call_count
                        ),
                        "obstacle_projection_corrected_steps": (
                            obstacle_projection_corrected_count
                        ),
                        "obstacle_projection_constraint_corrections": (
                            obstacle_constraint_correction_count
                        ),
                        "obstacle_projection_infeasible_steps": (
                            obstacle_projection_infeasible_count
                        ),
                        "obstacle_projection_correction_norm_sum_mps": (
                            obstacle_projection_correction_sum
                        ),
                        "obstacle_projection_correction_norm_max_mps": (
                            obstacle_projection_correction_max
                        ),
                        "obstacle_projection_min_predicted_clearance_m": (
                            obstacle_projection_min_clearance
                        ),
                        "tangent_bypass_enabled": int(
                            p5["tangent_bypass_enabled"][env_index].item()
                        ),
                        "tangent_bypass_active_steps": tangent_active_step_count,
                        "tangent_bypass_active_agent_steps": (
                            tangent_active_agent_step_count
                        ),
                        "tangent_bypass_activations": tangent_activation_count,
                        "tangent_bypass_min_candidate_clearance_m": (
                            tangent_min_candidate_clearance
                        ),
                        "slot_guidance_enabled": int(
                            p5["slot_guidance_enabled"][env_index].item()
                        ),
                        "slot_guidance_active_steps": slot_active_steps,
                        "slot_handoff_count": slot_handoff_count,
                        "terminal_slot_error_m": slot_terminal_error,
                        "terminal_slot_assignment": json.dumps(
                            [
                                int(value)
                                for value in p5["slot_assignment"][env_index]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            separators=(",", ":"),
                        ),
                    }
                )
            completed += 1
            recorded_seeds.add(current_episode_seed)
        episode_lengths[done] = 0
        episode_min_agent_clearance[done] = torch.inf
        episode_min_static_clearance[done] = torch.inf
        episode_min_boundary_clearance[done] = torch.inf
        episode_dynamic_obstacle_max_displacement[done] = 0.0
        episode_dynamic_obstacle_max_speed[done] = 0.0
        episode_strong_ring_max_hold_steps[done] = 0
        episode_all_visible_steps[done] = 0
        episode_staging_steps[done] = 0
        episode_staging_visible_steps[done] = 0
        episode_phase_ready_steps[done] = 0
        episode_orientation_unlocked[done] = False
        episode_first_unlock_step[done] = -1
        episode_target_projection_calls[done] = 0
        episode_target_projection_corrected_steps[done] = 0
        episode_target_projection_constraint_corrections[done] = 0
        episode_target_projection_maximum_separation_fallback_constraints[
            done
        ] = 0
        episode_target_projection_unavoidable_steps[done] = 0
        episode_target_projection_unavoidable_constraints[done] = 0
        episode_target_projection_infeasible_steps[done] = 0
        episode_target_projection_correction_norm_sum[done] = 0.0
        episode_target_projection_correction_norm_max[done] = 0.0
        episode_target_projection_min_predicted_center_distance[done] = torch.inf
        episode_target_projection_minimum_maximum_attainable_center_distance[
            done
        ] = torch.inf
        episode_safety_projection_calls[done] = 0
        episode_safety_projection_corrected_steps[done] = 0
        episode_safety_projection_pair_corrections[done] = 0
        episode_safety_projection_infeasible_steps[done] = 0
        episode_safety_projection_correction_norm_sum[done] = 0.0
        episode_safety_projection_correction_norm_max[done] = 0.0
        episode_safety_projection_min_predicted_center_distance[done] = torch.inf
        episode_obstacle_projection_calls[done] = 0
        episode_obstacle_projection_corrected_steps[done] = 0
        episode_obstacle_projection_constraint_corrections[done] = 0
        episode_obstacle_projection_infeasible_steps[done] = 0
        episode_obstacle_projection_correction_norm_sum[done] = 0.0
        episode_obstacle_projection_correction_norm_max[done] = 0.0
        episode_obstacle_projection_min_predicted_clearance[done] = torch.inf
        episode_boundary_projection_calls[done] = 0
        episode_boundary_projection_corrected_steps[done] = 0
        episode_boundary_projection_constraint_corrections[done] = 0
        episode_boundary_projection_infeasible_steps[done] = 0
        episode_boundary_projection_correction_norm_sum[done] = 0.0
        episode_boundary_projection_correction_norm_max[done] = 0.0
        episode_boundary_projection_min_predicted_clearance[done] = torch.inf
        episode_tangent_bypass_active_steps[done] = 0
        episode_tangent_bypass_active_agent_steps[done] = 0
        episode_tangent_bypass_activations[done] = 0
        episode_tangent_bypass_min_candidate_clearance[done] = torch.inf
        episode_maximum_lost_steps[done] = 0
        episode_team_lost_steps[done] = 0
        episode_loss_reason_steps[done] = 0
        if completed >= episodes:
            break
    if completed != episodes:
        raise RuntimeError(f"evaluation completed {completed}/{episodes} episodes")
    if recorded_seeds != planned_seed_set:
        raise RuntimeError("evaluation did not complete the exact forced seed plan")
    return {
        "episodes": completed,
        **({"target_buffer_failure_rate": target_buffer_failures / completed}
           if env.cfg.target_buffer_failure_enabled else {}),
        "region_capture_enabled": bool(env.cfg.region_capture_enabled),
        "success_rate_region": training_success / completed if env.cfg.region_capture_enabled else 0.0,
        "success_rate_training": training_success / completed,
        "success_rate_6m": success / completed,
        "success_rate_5m": strict / completed,
        "collision_rate": collision / completed,
        "collision_agent_rate": collision_agent / completed,
        "collision_agent_before_unlock_rate": (
            collision_agent_before_unlock / completed
        ),
        "collision_agent_after_unlock_rate": (
            collision_agent_after_unlock / completed
        ),
        "collision_static_rate": collision_static / completed,
        "collision_dynamic_rate": collision_dynamic / completed,
        "contact_rate": contact / completed,
        "invalid_rate": invalid / completed,
        "boundary_invalid_rate": boundary_invalid / completed,
        "height_invalid_rate": height_invalid / completed,
        "timeout_rate": timeout / completed,
        "orientation_unlock_rate": orientation_unlocked / completed,
        "terminal_orientation_staging_fraction": (
            terminal_staging_fraction_sum / completed
        ),
        "terminal_orientation_alignment_fraction": (
            terminal_alignment_fraction_sum / completed
        ),
        "terminal_orientation_phase_ready_rate": (
            terminal_phase_ready / completed
        ),
        "mean_orientation_phase_ready_step_fraction": (
            phase_ready_fraction_sum / completed
        ),
        "terminal_intercept_reachable_fraction": (
            terminal_intercept_reachable_fraction_sum / completed
        ),
        "mean_terminal_intercept_horizon_s": (
            terminal_intercept_horizon_sum / completed
        ),
        "intercept_blocker_reachable_rate": (
            intercept_blocker_reachable / completed
        ),
        "mean_all_visible_step_fraction": all_visible_fraction_sum / completed,
        "mean_staging_all_visible_step_fraction": (
            staging_visible_fraction_sum / completed
        ),
        "loss_episode_rate": loss_episodes / completed,
        "long_loss_2s_episode_rate": long_loss_2s_episodes / completed,
        "mean_maximum_lost_steps": maximum_lost_steps_sum / completed,
        "maximum_lost_steps": maximum_lost_steps_global,
        "team_lost_step_fraction": team_lost_steps / max(length_sum, 1),
        "mean_recapture_count": recapture_count_sum / completed,
        "loss_reacquisition_guidance_enabled": bool(
            env.cfg.loss_reacquisition_guidance_enabled
        ),
        "loss_reacquisition_eligible_episode_rate": (
            loss_reacquisition_eligible_episodes / completed
        ),
        "loss_reacquisition_first_completed_episode_rate": (
            loss_reacquisition_first_completed_episodes / completed
        ),
        "loss_reacquisition_first_within_2s_episode_rate": (
            loss_reacquisition_first_within_2s_episodes / completed
        ),
        "loss_reacquisition_event_completion_rate": (
            loss_reacquisition_completed_sum / loss_reacquisition_event_sum
            if loss_reacquisition_event_sum
            else 0.0
        ),
        "mean_loss_reacquisition_active_steps_per_episode": (
            loss_reacquisition_active_step_sum / completed
        ),
        "mean_first_loss_reacquisition_delay_steps": (
            loss_reacquisition_first_delay_step_sum
            / loss_reacquisition_first_delay_count
            if loss_reacquisition_first_delay_count
            else -1.0
        ),
        "loss_reacquisition_flank_enabled": bool(
            env.cfg.loss_reacquisition_occlusion_flank_enabled
        ),
        "loss_reacquisition_flank_episode_rate": (
            loss_reacquisition_flank_episode_count / completed
        ),
        "loss_reacquisition_flank_event_coverage_rate": (
            loss_reacquisition_flank_event_sum / loss_reacquisition_event_sum
            if loss_reacquisition_event_sum
            else 0.0
        ),
        "mean_loss_reacquisition_flank_active_steps_per_episode": (
            loss_reacquisition_flank_active_step_sum / completed
        ),
        "mean_loss_reacquisition_flank_fallback_steps_per_episode": (
            loss_reacquisition_flank_fallback_step_sum / completed
        ),
        "minimum_loss_reacquisition_flank_path_clearance_m": (
            loss_reacquisition_flank_minimum_path_clearance
            if math.isfinite(loss_reacquisition_flank_minimum_path_clearance)
            else -1.0
        ),
        "minimum_loss_reacquisition_flank_los_clearance_m": (
            loss_reacquisition_flank_minimum_los_clearance
            if math.isfinite(loss_reacquisition_flank_minimum_los_clearance)
            else -1.0
        ),
        "loss_reacquisition_sector_enabled": bool(
            env.cfg.loss_reacquisition_reachable_sector_enabled
        ),
        "loss_reacquisition_sector_episode_rate": (
            loss_reacquisition_sector_episode_count / completed
        ),
        "loss_reacquisition_sector_step_coverage_rate": (
            loss_reacquisition_sector_valid_step_sum
            / (
                loss_reacquisition_sector_valid_step_sum
                + loss_reacquisition_sector_fallback_step_sum
            )
            if (
                loss_reacquisition_sector_valid_step_sum
                + loss_reacquisition_sector_fallback_step_sum
            )
            else 0.0
        ),
        "mean_loss_reacquisition_sector_valid_steps_per_episode": (
            loss_reacquisition_sector_valid_step_sum / completed
        ),
        "mean_loss_reacquisition_sector_fallback_steps_per_episode": (
            loss_reacquisition_sector_fallback_step_sum / completed
        ),
        "minimum_loss_reacquisition_sector_path_clearance_m": (
            loss_reacquisition_sector_minimum_path_clearance
            if math.isfinite(loss_reacquisition_sector_minimum_path_clearance)
            else -1.0
        ),
        "mean_target_episode_displacement_m": (
            target_displacement_sum / completed
        ),
        "mean_terminal_target_speed_mps": (
            terminal_target_speed_sum / completed
        ),
        "minimum_evader_episode_speed_limit_mps": (
            evader_episode_speed_limit_min
        ),
        "mean_evader_episode_speed_limit_mps": (
            evader_episode_speed_limit_sum / completed
        ),
        "maximum_evader_episode_speed_limit_mps": (
            evader_episode_speed_limit_max
        ),
        "mean_maximum_strong_ring_hold_steps": (
            strong_ring_max_hold_steps_sum / completed
        ),
        "maximum_strong_ring_hold_steps": strong_ring_max_hold_steps_global,
        "mean_maximum_dynamic_obstacle_displacement_m": (
            dynamic_obstacle_displacement_sum / completed
        ),
        "maximum_dynamic_obstacle_displacement_m": (
            dynamic_obstacle_displacement_max
        ),
        "mean_maximum_dynamic_obstacle_speed_mps": (
            dynamic_obstacle_speed_sum / completed
        ),
        "unique_obstacle_layout_count": len(obstacle_layout_signatures),
        "pure_range_lost_step_fraction": (
            loss_reason_steps["pure_range"] / max(team_lost_steps, 1)
        ),
        "pure_fov_lost_step_fraction": (
            loss_reason_steps["pure_fov"] / max(team_lost_steps, 1)
        ),
        "pure_occlusion_lost_step_fraction": (
            loss_reason_steps["pure_occlusion"] / max(team_lost_steps, 1)
        ),
        "mixed_loss_step_fraction": (
            loss_reason_steps["mixed"] / max(team_lost_steps, 1)
        ),
        "terminal_orientation_yaw_error_max_rad": (
            terminal_yaw_error_sum / completed
        ),
        "mean_first_orientation_unlock_step": (
            first_unlock_step_sum / first_unlock_step_count
            if first_unlock_step_count
            else -1.0
        ),
        "mean_episode_steps": length_sum / completed,
        "terminal_radial_error_m": radial_error_sum / completed,
        "terminal_angular_error_rad": angular_error_sum / completed,
        "hybrid_supervisor_episode_rate": (
            hybrid_supervisor_episodes / completed
        ),
        "mean_hybrid_traditional_steps_per_episode": (
            hybrid_traditional_steps_sum / completed
        ),
        "mean_hybrid_rl_steps_per_episode": hybrid_rl_steps_sum / completed,
        "hybrid_rl_active_episode_rate": (
            hybrid_rl_active_episodes / completed
        ),
        "hybrid_rl_step_fraction": (
            hybrid_rl_steps_sum
            / max(hybrid_traditional_steps_sum + hybrid_rl_steps_sum, 1)
        ),
        "policy_residual_active_steps": policy_residual_active_steps_sum,
        "mean_policy_residual_norm_mps": (
            policy_residual_norm_sum_mps
            / max(policy_residual_active_steps_sum, 1)
        ),
        "mean_policy_residual_radial_support_mps": (
            policy_residual_radial_support_sum_mps
            / max(policy_residual_active_steps_sum, 1)
        ),
        "mean_policy_residual_radial_opposition_mps": (
            policy_residual_radial_opposition_sum_mps
            / max(policy_residual_active_steps_sum, 1)
        ),
        "mean_policy_residual_phase_support_mps": (
            policy_residual_phase_support_sum_mps
            / max(policy_residual_active_steps_sum, 1)
        ),
        "mean_policy_residual_phase_opposition_mps": (
            policy_residual_phase_opposition_sum_mps
            / max(policy_residual_active_steps_sum, 1)
        ),
        "mean_slot_speed_limit_active_agent_fraction": (
            slot_speed_limit_active_agent_fraction_sum
            / max(slot_active_steps_sum, 1)
        ),
        "mean_slot_speed_limit_scale": (
            slot_speed_limit_mean_scale_sum / max(slot_active_steps_sum, 1)
        ),
        "mean_slot_speed_limit_forward_loss_mps": (
            slot_speed_limit_forward_loss_sum_mps
            / max(slot_active_steps_sum, 1)
        ),
        "mean_slot_requested_tangential_speed_mps": (
            slot_requested_tangential_speed_sum_mps
            / max(slot_active_steps_sum, 1)
        ),
        "mean_slot_executed_tangential_speed_mps": (
            slot_executed_tangential_speed_sum_mps
            / max(slot_active_steps_sum, 1)
        ),
        "mean_minimum_agent_clearance_m": clearance_sums["agent"] / completed,
        "minimum_agent_clearance_m": clearance_minima["agent"],
        "mean_minimum_static_clearance_m": clearance_sums["static"] / completed,
        "minimum_static_clearance_m": clearance_minima["static"],
        "mean_minimum_boundary_clearance_m": (
            clearance_sums["boundary"] / completed
        ),
        "minimum_boundary_clearance_m": clearance_minima["boundary"],
        "target_projection_episode_rate": target_projection_episodes / completed,
        "mean_target_projection_calls_per_episode": (
            target_projection_calls / completed
        ),
        "target_projection_corrected_step_rate": (
            target_projection_corrected_steps / target_projection_calls
            if target_projection_calls
            else 0.0
        ),
        "mean_target_projection_constraint_corrections_per_call": (
            target_projection_constraint_corrections / target_projection_calls
            if target_projection_calls
            else 0.0
        ),
        "mean_target_projection_maximum_separation_fallback_constraints_per_call": (
            target_projection_maximum_separation_fallback_constraints
            / target_projection_calls
            if target_projection_calls
            else 0.0
        ),
        "target_projection_unavoidable_step_rate": (
            target_projection_unavoidable_steps / target_projection_calls
            if target_projection_calls
            else 0.0
        ),
        "mean_target_projection_unavoidable_constraints_per_call": (
            target_projection_unavoidable_constraints / target_projection_calls
            if target_projection_calls
            else 0.0
        ),
        "target_projection_infeasible_step_rate": (
            target_projection_infeasible_steps / target_projection_calls
            if target_projection_calls
            else 0.0
        ),
        "mean_target_projection_correction_norm_mps": (
            target_projection_correction_norm_sum
            / target_projection_corrected_steps
            if target_projection_corrected_steps
            else 0.0
        ),
        "maximum_target_projection_correction_norm_mps": (
            target_projection_correction_norm_max
        ),
        "minimum_target_projection_predicted_center_distance_m": (
            target_projection_min_predicted_center_distance
            if target_projection_calls
            else -1.0
        ),
        "minimum_target_projection_maximum_attainable_center_distance_m": (
            target_projection_minimum_maximum_attainable_center_distance
            if target_projection_calls
            else -1.0
        ),
        "safety_projection_restorative_enabled": bool(
            env.cfg.predictive_agent_restore_unsafe_buffer
        ),
        "safety_projection_episode_rate": safety_projection_episodes / completed,
        "mean_safety_projection_calls_per_episode": (
            safety_projection_calls / completed
        ),
        "safety_projection_corrected_step_rate": (
            safety_projection_corrected_steps / safety_projection_calls
            if safety_projection_calls
            else 0.0
        ),
        "mean_safety_projection_pair_corrections_per_call": (
            safety_projection_pair_corrections / safety_projection_calls
            if safety_projection_calls
            else 0.0
        ),
        "safety_projection_infeasible_step_rate": (
            safety_projection_infeasible_steps / safety_projection_calls
            if safety_projection_calls
            else 0.0
        ),
        "mean_safety_projection_correction_norm_mps": (
            safety_projection_correction_norm_sum
            / safety_projection_corrected_steps
            if safety_projection_corrected_steps
            else 0.0
        ),
        "maximum_safety_projection_correction_norm_mps": (
            safety_projection_correction_norm_max
        ),
        "minimum_safety_projection_predicted_center_distance_m": (
            safety_projection_min_predicted_center_distance
            if safety_projection_calls
            else -1.0
        ),
        "obstacle_projection_episode_rate": (
            obstacle_projection_episodes / completed
        ),
        "mean_obstacle_projection_calls_per_episode": (
            obstacle_projection_calls / completed
        ),
        "obstacle_projection_corrected_step_rate": (
            obstacle_projection_corrected_steps / obstacle_projection_calls
            if obstacle_projection_calls
            else 0.0
        ),
        "mean_obstacle_projection_constraint_corrections_per_call": (
            obstacle_projection_constraint_corrections / obstacle_projection_calls
            if obstacle_projection_calls
            else 0.0
        ),
        "obstacle_projection_infeasible_step_rate": (
            obstacle_projection_infeasible_steps / obstacle_projection_calls
            if obstacle_projection_calls
            else 0.0
        ),
        "mean_obstacle_projection_correction_norm_mps": (
            obstacle_projection_correction_norm_sum
            / obstacle_projection_corrected_steps
            if obstacle_projection_corrected_steps
            else 0.0
        ),
        "maximum_obstacle_projection_correction_norm_mps": (
            obstacle_projection_correction_norm_max
        ),
        "minimum_obstacle_projection_predicted_clearance_m": (
            obstacle_projection_min_predicted_clearance
            if obstacle_projection_calls
            else -1.0
        ),
        "boundary_projection_episode_rate": (
            boundary_projection_episodes / completed
        ),
        "mean_boundary_projection_calls_per_episode": (
            boundary_projection_calls / completed
        ),
        "boundary_projection_corrected_step_rate": (
            boundary_projection_corrected_steps / boundary_projection_calls
            if boundary_projection_calls
            else 0.0
        ),
        "mean_boundary_projection_constraint_corrections_per_call": (
            boundary_projection_constraint_corrections
            / boundary_projection_calls
            if boundary_projection_calls
            else 0.0
        ),
        "boundary_projection_infeasible_step_rate": (
            boundary_projection_infeasible_steps / boundary_projection_calls
            if boundary_projection_calls
            else 0.0
        ),
        "mean_boundary_projection_correction_norm_mps": (
            boundary_projection_correction_norm_sum
            / boundary_projection_corrected_steps
            if boundary_projection_corrected_steps
            else 0.0
        ),
        "maximum_boundary_projection_correction_norm_mps": (
            boundary_projection_correction_norm_max
        ),
        "minimum_boundary_projection_predicted_clearance_m": (
            boundary_projection_min_predicted_clearance
            if boundary_projection_calls
            else -1.0
        ),
        "tangent_bypass_episode_rate": tangent_bypass_episodes / completed,
        "mean_tangent_bypass_active_steps_per_episode": (
            tangent_bypass_active_steps / completed
        ),
        "mean_tangent_bypass_active_agent_steps_per_episode": (
            tangent_bypass_active_agent_steps / completed
        ),
        "mean_tangent_bypass_activations_per_episode": (
            tangent_bypass_activations / completed
        ),
        "minimum_tangent_bypass_candidate_clearance_m": (
            tangent_bypass_min_candidate_clearance
            if tangent_bypass_episodes
            else -1.0
        ),
        "slot_guidance_episode_rate": slot_guidance_episodes / completed,
        "mean_slot_guidance_active_steps_per_episode": (
            slot_active_steps_sum / completed
        ),
        "mean_slot_handoffs_per_episode": slot_handoff_count_sum / completed,
        "terminal_slot_error_m": slot_terminal_error_sum / completed,
    }


def configure_stage(cfg: BylwPursuitTrainEnvCfg, stage: str) -> None:
    if stage == "c91":
        configure_stage(cfg, "c88")
        cfg.loss_reacquisition_scout_minimum_selection_distance_m = (
            cfg.training_near_switch_m
        )
        cfg.loss_reacquisition_tracker_hold_enabled = True
        return
    if stage == "c90":
        configure_stage(cfg, "c89")
        cfg.predictive_target_last_seen_reachable_tube_enabled = True
        return
    if stage == "c89":
        configure_stage(cfg, "c88")
        cfg.predictive_target_last_seen_cv_during_loss_enabled = True
        cfg.predictive_target_last_seen_cv_maximum_loss_time_s = 2.0
        return
    if stage == "c88":
        configure_stage(cfg, "c86")
        cfg.predictive_ring_preentry_braking_enabled = True
        cfg.predictive_ring_preentry_braking_activation_radius_m = (
            cfg.training_near_switch_m
        )
        return
    if stage == "c87":
        configure_stage(cfg, "c86")
        cfg.predictive_ring_preentry_braking_enabled = True
        cfg.predictive_ring_preentry_braking_activation_radius_m = (
            cfg.training_near_switch_m
        )
        cfg.predictive_ring_angular_gap_preentry_enabled = True
        cfg.predictive_ring_angular_gap_preentry_activation_radius_m = (
            cfg.training_near_switch_m
        )
        return
    if stage == "c86":
        configure_stage(cfg, "c80")
        cfg.predictive_ring_angular_gap_projection_enabled = True
        return
    if stage == "c85":
        configure_stage(cfg, "c84")
        low_level_response_fraction = 1.0 - (
            1.0 - cfg.predictive_ring_low_level_velocity_kp * cfg.sim.dt
        ) ** cfg.decimation
        graph_laplacian_eigenvalue = float(len(cfg.possible_agents))
        cfg.moving_target_slot_phase_angular_velocity_damping_gain_s = (
            2.0
            * (1.0 - low_level_response_fraction)
            / (low_level_response_fraction * graph_laplacian_eigenvalue)
        )
        return
    if stage == "c84":
        configure_stage(cfg, "c83")
        cfg.moving_target_slot_phase_angular_velocity_damping_confirmation_only = True
        return
    if stage == "c83":
        configure_stage(cfg, "c80")
        agent_count = len(cfg.possible_agents)
        graph_laplacian_eigenvalue = float(agent_count)
        low_level_velocity_kp = cfg.predictive_ring_low_level_velocity_kp
        critical_ratio = math.sqrt(
            2.0
            * cfg.moving_target_slot_phase_gain
            * graph_laplacian_eigenvalue
            / low_level_velocity_kp
        )
        cfg.moving_target_slot_phase_angular_velocity_damping_gain_s = (
            2.0
            * max(critical_ratio - 1.0, 0.0)
            / graph_laplacian_eigenvalue
        )
        return
    if stage == "c82":
        configure_stage(cfg, "c80")
        cfg.strong_ring_confirmation_maximum_phase_gain_enabled = True
        return
    if stage == "c81":
        configure_stage(cfg, "c80")
        cfg.moving_target_slot_tangential_velocity_residual_control_enabled = True
        policy_dt_s = cfg.sim.dt * cfg.decimation
        confirmation_time_s = cfg.strong_ring_required_hold_steps * policy_dt_s
        reference_radius_m = 0.5 * (
            cfg.strong_ring_minimum_radius_m
            + cfg.strong_ring_maximum_radius_m
        )
        cfg.moving_target_slot_tangential_velocity_residual_scale_mps = (
            reference_radius_m
            * cfg.strong_ring_maximum_mean_gap_error_rad
            / confirmation_time_s
        )
        return
    if stage == "c80":
        configure_stage(cfg, "c79")
        cfg.moving_target_slot_radial_velocity_residual_control_enabled = True
        policy_dt_s = cfg.sim.dt * cfg.decimation
        confirmation_time_s = cfg.strong_ring_required_hold_steps * policy_dt_s
        cfg.moving_target_slot_radial_velocity_residual_scale_mps = (
            0.5
            * (
                cfg.strong_ring_maximum_radius_m
                - cfg.strong_ring_minimum_radius_m
            )
            / confirmation_time_s
        )
        return
    if stage == "c79":
        configure_stage(cfg, "c78")
        cfg.strong_ring_violation_progress_reward_weight = 0.5
        return
    if stage == "c78":
        configure_stage(cfg, "c77")
        cfg.predictive_ring_preentry_velocity_sync_enabled = True
        return
    if stage == "c77":
        configure_stage(cfg, "c76")
        policy_dt_s = cfg.sim.dt * cfg.decimation
        effective_response_s = (
            policy_dt_s / cfg.predictive_ring_command_response_ratio
        )
        cfg.predictive_ring_activation_maximum_gap_margin_rad = (
            2.0
            * cfg.training_command_speed_mps
            * effective_response_s
            / cfg.strong_ring_minimum_radius_m
        )
        return
    if stage == "c76":
        configure_stage(cfg, "c75")
        cfg.orientation_reference_contraction_tracking_gate_enabled = False
        return
    if stage == "c75":
        configure_stage(cfg, "c74")
        cfg.moving_target_slot_radial_velocity_damping_gain = (
            math.sqrt(2.0 * cfg.moving_target_slot_radial_gain) - 1.0
        )
        return
    if stage == "c74":
        configure_stage(cfg, "c73")
        cfg.moving_target_slot_feedforward_priority_projection_enabled = True
        return
    if stage == "c73":
        configure_stage(cfg, "c72")
        cfg.moving_target_slot_full_feedforward_above_strong_ring = True
        return
    if stage == "c72":
        configure_stage(cfg, "c71")
        cfg.orientation_reference_anchor_to_mean_radius_on_unlock = True
        return
    if stage == "c71":
        configure_stage(cfg, "c69")
        cfg.orientation_reference_contraction_tracking_gate_enabled = True
        return
    if stage == "c70":
        configure_stage(cfg, "c69")
        cfg.predictive_ring_preentry_velocity_sync_enabled = True
        return
    if stage == "c69":
        configure_stage(cfg, "c68")
        cfg.predictive_ring_common_acceleration_feedforward_after_strong_entry = True
        return
    if stage == "c68":
        configure_stage(cfg, "c67")
        cfg.predictive_ring_dynamics_consistent_enabled = True
        return
    if stage == "c67":
        configure_stage(cfg, "c66")
        cfg.predictive_ring_minimum_radius_after_strong_entry = True
        return
    if stage == "c66":
        configure_stage(cfg, "c65")
        cfg.predictive_ring_enforce_minimum_radius = False
        return
    if stage == "c65":
        configure_stage(cfg, "c64")
        cfg.predictive_ring_response_trajectory_substeps = 5
        return
    if stage == "c64":
        configure_stage(cfg, "c63")
        cfg.evader_minimum_episode_speed_mps = 4.0
        return
    if stage == "c63":
        configure_stage(cfg, "c62")
        cfg.strong_ring_step_reward_weight = 0.5
        return
    if stage == "c62":
        configure_stage(cfg, "c58")
        cfg.moving_target_slot_gain_residual_control_enabled = True
        cfg.moving_target_slot_radial_gain_residual_scale = 0.25
        cfg.moving_target_slot_phase_gain_residual_scale = 0.25
        cfg.orientation_yaw_gain_residual_scale = 0.25
        return
    if stage == "c61":
        configure_stage(cfg, "c56")
        cfg.moving_target_slot_feedforward_scale = 1.0
        cfg.moving_target_slot_feedforward_priority_projection_enabled = True
        return
    if stage == "c60":
        configure_stage(cfg, "c56")
        policy_dt_s = cfg.sim.dt * cfg.decimation
        effective_response_s = (
            policy_dt_s / cfg.predictive_ring_command_response_ratio
        )
        cfg.predictive_ring_activation_margin_m = (
            cfg.training_command_speed_mps * effective_response_s
        )
        return
    if stage == "c58":
        configure_stage(cfg, "c55")
        cfg.evader_minimum_episode_speed_mps = 1.0
        return
    if stage == "c57":
        configure_stage(cfg, "c55")
        cfg.evader_minimum_episode_speed_mps = 2.0
        return
    if stage == "c56":
        configure_stage(cfg, "c55")
        policy_dt_s = cfg.sim.dt * cfg.decimation
        effective_response_s = (
            policy_dt_s / cfg.predictive_ring_command_response_ratio
        )
        cfg.predictive_ring_activation_maximum_gap_margin_rad = (
            2.0
            * cfg.training_command_speed_mps
            * effective_response_s
            / cfg.strong_ring_minimum_radius_m
        )
        return
    if stage == "c55":
        configure_stage(cfg, "c54")
        cfg.predictive_ring_target_acceleration_enabled = True
        policy_dt_s = cfg.sim.dt * cfg.decimation
        cfg.predictive_ring_target_acceleration_substeps = round(
            cfg.predictive_ring_prediction_horizon_s / policy_dt_s
        )
        return
    if stage == "c54":
        configure_stage(cfg, "c53")
        # Five 0.1 s yaw-alignment steps are required before contraction.  An
        # equal-speed 4 m/s evader can move 2 m in that interval, so keep the
        # legal 8 m reset but allow readiness inside an 8 + 2 = 10 m region.
        cfg.orientation_staging_radius_m = 10.0
        return
    if stage == "c53":
        configure_stage(cfg, "c52")
        cfg.tracked_reset_reference_range_m = 8.0
        cfg.orientation_staging_radius_m = 8.0
        return
    if stage == "c52":
        configure_stage(cfg, "c51")
        cfg.predictive_ring_activation_requires_inside_inner_boundary = True
        policy_dt_s = cfg.sim.dt * cfg.decimation
        cfg.predictive_ring_activation_maximum_gap_margin_rad = (
            2.0
            * cfg.training_command_speed_mps
            * policy_dt_s
            / cfg.strong_ring_minimum_radius_m
        )
        return
    if stage == "c51":
        configure_stage(cfg, "c49")
        cfg.predictive_ring_activation_requires_maximum_gap_eligibility = True
        return
    if stage == "c50":
        configure_stage(cfg, "c49")
        cfg.predictive_ring_activation_requires_strong_ring_entry = True
        return
    if stage == "c49":
        configure_stage(cfg, "c48")
        cfg.predictive_ring_activation_latched_after_measured_entry = True
        return
    if stage == "c48":
        configure_stage(cfg, "c47")
        # The 0.5 m band is wider than the maximum 0.4 m travelled in one
        # 0.1 s policy step at 4 m/s.  This lets the predictive annulus take
        # over immediately before a strict-ring boundary crossing, while
        # preventing C47's premature activation during far-field approach.
        cfg.predictive_ring_activation_margin_m = 0.5
        return
    if stage == "c47":
        configure_stage(cfg, "c41")
        cfg.predictive_ring_stabilization_enabled = True
        cfg.predictive_ring_prediction_horizon_s = 0.5
        cfg.predictive_ring_command_response_ratio = 0.5
        cfg.predictive_ring_projection_iterations = 4
        return
    if stage == "c46":
        configure_stage(cfg, "c41")
        cfg.strong_ring_confirmation_stabilization_enabled = True
        cfg.strong_ring_confirmation_suppress_policy_residual = False
        return
    if stage == "c45":
        configure_stage(cfg, "c41")
        cfg.strong_ring_confirmation_stabilization_enabled = True
        cfg.strong_ring_confirmation_suppress_policy_residual = True
        return
    if stage == "c44":
        configure_stage(cfg, "c41")
        cfg.reachable_intercept_release_during_contraction = True
        cfg.reachable_intercept_lead_fade_start_radius_m = (
            cfg.training_near_switch_m
        )
        cfg.reachable_intercept_lead_zero_radius_m = 5.5
        return
    if stage == "c43":
        # The C41 near-ring reset is already reachable.  Keep the persistent
        # blocker role assignment, but centre every final ring slot on the
        # live target so the physical success criterion and control target
        # use the same origin.
        configure_stage(cfg, "c41")
        cfg.reachable_intercept_blocker_only_slot_lead = False
        return
    if stage == "c42":
        configure_stage(cfg, "c41")
        cfg.moving_target_slot_feedforward_priority_projection_enabled = True
        return
    if stage == "c41":
        # Freeze the C40-N6 controller that first cleared the equal-capability
        # static-obstacle zero-training gate.  This is a near-ring curriculum,
        # not the final 100 m reset-range generalization claim.
        configure_stage(cfg, "c40")
        cfg.tracked_reset_reference_range_m = 12.0
        cfg.orientation_staging_radius_m = 12.0
        cfg.moving_target_slot_phase_separation_control_enabled = True
        cfg.moving_target_slot_phase_activation_margin_m = 6.0
        cfg.orientation_reference_contraction_latched_after_unlock = True
        return
    if stage == "c40":
        # Keep C30's hierarchical search/approach/MAPPO-ring supervisor and
        # strict strong-ring/safety contract.  Only obstacle motion is removed.
        configure_stage(cfg, "c30")
        cfg.dynamic_obstacle_indices = ()
        cfg.evader_speed_mps = 4.0
        cfg.evader_minimum_episode_speed_mps = 4.0
        cfg.evader_allow_stop_mode = False
        cfg.evader_maintain_episode_speed = True
        cfg.pursuer_max_planar_accel_mps2 = 4.0
        cfg.evader_max_planar_accel_mps2 = 4.0
        return
    if stage == "c32":
        # Preserve the entire C31 task/safety contract.  Learning may only
        # scale positive radial and adjacent-gap gains by +/-25%; it cannot
        # inject a Cartesian direction or a learned yaw residual.
        configure_stage(cfg, "c31")
        cfg.moving_target_slot_gain_residual_control_enabled = True
        cfg.moving_target_slot_radial_gain_residual_scale = 0.25
        cfg.moving_target_slot_phase_gain_residual_scale = 0.25
        cfg.orientation_yaw_gain_residual_scale = 0.25
        return
    if stage == "c31":
        # Inherit the frozen C30 task, dynamics, hybrid supervisor and safety
        # contract exactly; C31 changes only formation topology and phase order.
        configure_stage(cfg, "c30")
        cfg.moving_target_slot_cyclic_order_preserving_assignment = True
        cfg.moving_target_slot_lock_valid_assignment = True
        cfg.moving_target_slot_phase_separation_control_enabled = True
        cfg.moving_target_slot_phase_gain = 0.8
        cfg.moving_target_slot_radial_gain = 1.0
        cfg.moving_target_slot_maximum_tangential_speed_mps = 3.0
        cfg.moving_target_slot_phase_activation_margin_m = 6.0
        cfg.reachable_intercept_blocker_only_slot_lead = True
        cfg.moving_target_slot_feedforward_scale = 1.0
        cfg.hybrid_rl_activation_on_staging = True
        cfg.orientation_phase_gate_enabled = True
        cfg.orientation_phase_maximum_gap_rad = 5.0 * math.pi / 6.0
        cfg.orientation_phase_maximum_mean_gap_error_rad = math.pi / 9.0
        cfg.moving_target_slot_error_reward_weight = 0.0
        cfg.moving_target_slot_progress_reward_weight = 0.0
        return
    curriculum = native_curriculum_stage(stage)
    moving_target_stages = (
        "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30"
    )
    c9_method_stages = (
        "c9", "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30"
    )
    cfg.training_capture_radius_m = curriculum.training_capture_radius_m
    cfg.desired_training_radius_m = curriculum.desired_radius_m
    cfg.training_near_switch_m = curriculum.near_switch_m
    if curriculum.static_obstacles:
        cfg.obstacle_specs = STATIC_OBSTACLE_SPECS
        cfg.obstacle_shape_kinds = STATIC_OBSTACLE_SHAPES
        cfg.obstacle_box_sizes_xy_m = STATIC_OBSTACLE_BOX_SIZES_XY_M
    if stage == "c13":
        cfg.obstacle_specs = DENSE_STATIC_OBSTACLE_SPECS
        cfg.obstacle_shape_kinds = DENSE_STATIC_OBSTACLE_SHAPES
        cfg.obstacle_box_sizes_xy_m = DENSE_STATIC_OBSTACLE_BOX_SIZES_XY_M
    if stage in ("c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30"):
        cfg.obstacle_specs = MIXED_STATIC_OBSTACLE_SPECS
        cfg.obstacle_shape_kinds = MIXED_STATIC_OBSTACLE_SHAPES
        cfg.obstacle_box_sizes_xy_m = MIXED_STATIC_OBSTACLE_BOX_SIZES_XY_M
    if stage in ("c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30"):
        cfg.dynamic_obstacle_indices = (10, 11)
    cfg.evader_enabled = stage in moving_target_stages
    cfg.evader_speed_mps = (
        4.0
        if stage in ("c25", "c26", "c27", "c29", "c30")
        else 4.8
        if stage in ("c20", "c21", "c22", "c23", "c24")
        else 1.5
        if stage == "c17"
        else 1.0
        if stage in ("c8", "c9", "c10", "c11", "c12", "c13", "c14", "c16", "c19")
        else 2.0
        if stage in ("c6", "c7")
        else cfg.evader_speed_mps
    )
    if stage in ("c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30"):
        # Strict contract: every episode commands a non-stopping evader whose
        # sampled speed limit is at least the 4 m/s pursuer command limit.
        cfg.evader_minimum_episode_speed_mps = 4.0
        cfg.evader_allow_stop_mode = False
        cfg.evader_gap_escape_weight = 0.80
        cfg.evader_maintain_episode_speed = True
    cfg.evader_pursuer_repulsion_gain = (
        0.6
        if stage in moving_target_stages
        else cfg.evader_pursuer_repulsion_gain
    )
    cfg.randomize_obstacles_on_reset = stage in moving_target_stages
    cfg.sense_range_m = {
        "c10": 50.0,
        "c11": 30.0,
        "c12": 25.0,
    }.get(stage, 100.0)
    cfg.fov_half_rad = (
        math.pi / 3.0
        if stage in (
            "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9",
            "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30",
        )
        else math.pi
    )
    cfg.orientation_gate_enabled = stage in (
        "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9",
        "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30",
    )
    cfg.predictive_agent_projection_enabled = stage in (
        "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10",
        "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30",
    )
    cfg.predictive_agent_always_active = stage in (
        "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30"
    )
    cfg.predictive_static_projection_enabled = stage in (
        "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30"
    )
    cfg.static_tangent_bypass_enabled = stage in (
        "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30"
    )
    cfg.joint_safety_projection_sweeps = (
        2
        if stage in ("c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")
        else 1
    )
    cfg.moving_target_slot_guidance_enabled = stage in c9_method_stages
    cfg.predictive_target_projection_enabled = stage in c9_method_stages
    if stage == "c19":
        cfg.moving_target_slot_prediction_horizon_s = 0.5
    if stage == "c20":
        # At equal speed, waiting until the 14 m staging boundary is too late.
        # Start coordinated interception early and lead the slot centre by
        # 1.5 s while retaining a bounded learned residual.
        cfg.moving_target_slot_activation_radius_m = 30.0
        cfg.moving_target_slot_position_gain = 1.0
        cfg.moving_target_slot_policy_residual_scale = 0.25
        cfg.moving_target_slot_feedforward_scale = 0.75
        cfg.moving_target_slot_prediction_horizon_s = 1.5
        cfg.moving_target_slot_maximum_anchor_turn_rate_rps = 1.2
        cfg.strong_ring_termination_enabled = True
        # The C16 safety calibration assumed a much slower target.  Widen the
        # predictive buffers and convergence budget before any equal-speed
        # policy update; these remain execution-layer projections, not reward
        # shortcuts or altered collision definitions.
        cfg.predictive_agent_minimum_center_distance_m = 3.5
        cfg.predictive_agent_prediction_horizon_s = 1.5
        cfg.predictive_agent_projection_iterations = 8
        cfg.predictive_target_minimum_center_distance_m = 1.5
        cfg.predictive_target_prediction_horizon_s = 1.5
        cfg.predictive_target_projection_iterations = 4
        cfg.predictive_static_minimum_clearance_m = 1.5
        cfg.predictive_static_prediction_horizon_s = 1.5
        cfg.predictive_static_projection_iterations = 8
        cfg.joint_safety_projection_sweeps = 4
    if stage in ("c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30"):
        # C21/C22 keep the strict C20 speed, scene, capture and safety contract,
        # but repairs reachability before learning: begin at reset, select the
        # pursuer furthest ahead along target motion as a persistent blocker,
        # and place the common slot centre at its bounded CV intercept horizon.
        cfg.moving_target_slot_activation_radius_m = 100.0
        cfg.moving_target_slot_position_gain = 1.0
        cfg.moving_target_slot_policy_residual_scale = 0.20
        cfg.moving_target_slot_feedforward_scale = 0.75
        cfg.moving_target_slot_prediction_horizon_s = 0.0
        cfg.moving_target_slot_maximum_anchor_turn_rate_rps = 1.2
        cfg.reachable_intercept_guidance_enabled = True
        cfg.reachable_intercept_minimum_horizon_s = 0.25
        cfg.reachable_intercept_maximum_horizon_s = 8.0
        cfg.reachable_intercept_horizon_samples = 33
        # With the 4.8 m/s maximum target and 4.5 m forward slot, a 1.0 s
        # lead stays at 9.3 m from the current target and remains reachable by
        # the frozen 9.5 m staging gate.
        cfg.reachable_intercept_slot_horizon_cap_s = 1.0
        cfg.reachable_intercept_blocker_only_slot_lead = stage in ("c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30")
        cfg.reachable_intercept_blocker_role_lock_weight_m = 50.0
        cfg.asynchronous_orientation_gate_enabled = True
        cfg.asynchronous_orientation_progress_reward_weight = 0.20
        cfg.strong_ring_termination_enabled = True
        cfg.predictive_agent_minimum_center_distance_m = 3.5
        cfg.predictive_agent_prediction_horizon_s = 1.5
        cfg.predictive_agent_projection_iterations = 8
        cfg.predictive_target_minimum_center_distance_m = 1.5
        cfg.predictive_target_prediction_horizon_s = 1.5
        cfg.predictive_target_projection_iterations = 4
        cfg.predictive_static_minimum_clearance_m = 1.5
        cfg.predictive_static_prediction_horizon_s = 1.5
        cfg.predictive_static_projection_iterations = 8
        cfg.joint_safety_projection_sweeps = 4
        cfg.predictive_boundary_projection_enabled = stage in ("c23", "c24", "c25", "c26", "c27", "c29", "c30")
        cfg.orientation_target_relative_staging_enabled = stage in ("c24", "c25", "c26", "c27", "c29", "c30")
        cfg.asynchronous_require_current_team_ready = stage in ("c24", "c25", "c26", "c27", "c29", "c30")
        cfg.asynchronous_current_readiness_reward_enabled = stage in ("c24", "c25", "c26", "c27", "c29", "c30")
    if stage in ("c26", "c27", "c29", "c30"):
        # NavRL-inspired potential difference toward the already assigned
        # moving role goal. C27 inherits this C26 learning signal.
        cfg.moving_target_slot_progress_reward_weight = 0.50
    if stage in ("c27", "c29", "c30"):
        cfg.role_phase_observation_enabled = True
        cfg.moving_target_slot_prepare_assignment_for_observation = True
        cfg.moving_target_slot_progress_use_maximum_error = True
        cfg.predictive_agent_restore_unsafe_buffer = True
    if stage == "c30":
        # Route B: deterministic Last search and moving-slot formation own all
        # pre-encirclement actions.  MAPPO is a masked residual only while the
        # full team is currently visible, aligned, and contraction-ready.
        cfg.hybrid_phase_supervisor_enabled = True
        cfg.hybrid_encirclement_policy_residual_scale = 0.20
        cfg.hybrid_require_current_team_ready = True
        cfg.moving_target_slot_activation_radius_m = 1.0e6
        cfg.moving_target_slot_obstacle_aware_assignment = True
        cfg.moving_target_slot_assignment_path_clearance_m = 1.0
        cfg.moving_target_slot_assignment_obstacle_penalty = 30.0
        cfg.orientation_contraction_pause_without_team_ready = True
        cfg.orientation_policy_residual_scale = 0.0
        cfg.loss_reacquisition_guidance_enabled = True
        cfg.loss_reacquisition_scout_blend = 1.0
        cfg.loss_reacquisition_tracker_blend = 1.0
        cfg.loss_reacquisition_blend_growth = 0.0
        cfg.moving_target_slot_error_use_maximum_error = True
        cfg.predictive_target_restore_unsafe_buffer = True
        cfg.predictive_target_projection_last = True
        # Four alternating half-space/speed-ball projections left a 7.2 mm
        # residual on reused development seed 51015032, and eight reduced it
        # to 0.153 mm but remained 0.053 mm outside the strict feasibility
        # tolerance. Keep the frozen 1.5 m buffer and dynamics while converging
        # the C30 execution layer tightly before any policy update.
        cfg.predictive_target_projection_iterations = 16
    if stage in c9_method_stages:
        # At 0.5 m/s the 9.5 -> 4.5 m reference contraction consumes the
        # entire remaining 20 s episode after alignment.  C9 retains a large
        # margin below the 4 m/s command limit while leaving time to settle.
        cfg.orientation_reference_contraction_speed_mps = 1.0
        cfg.orientation_maximum_inward_speed_mps = 1.0
    if stage == "c2":
        cfg.predictive_agent_correction_reward_weight = 0.05
        cfg.predictive_agent_infeasible_reward_penalty = 0.10


def main() -> int:
    output_dir = resolve_inside_repo(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_gpu = gpu_snapshot(args_cli.physical_gpu_index)
    if initial_gpu["memory_used_mb"] > args_cli.memory_limit_mb:
        raise RuntimeError(
            f"GPU memory gate failed before launch: {initial_gpu['memory_used_mb']:.0f} MB"
        )

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    cfg = BylwPursuitTrainEnvCfg()
    configure_stage(cfg, args_cli.stage)
    if args_cli.region_capture:
        configure_region_capture(cfg, align_objective=args_cli.region_objective)
    cfg.dual_observer_formation_enabled = args_cli.dual_observer_formation
    cfg.dual_observer_role_lock_enabled = args_cli.dual_observer_role_lock
    cfg.region_gap_reserve_enabled = args_cli.region_gap_reserve
    cfg.region_joint_lookahead_enabled = args_cli.region_joint_lookahead
    cfg.target_path_guard_enabled = args_cli.target_path_guard
    cfg.target_path_guard_acceleration_enabled = args_cli.target_path_guard_acceleration
    cfg.target_path_guard_extended_enabled = args_cli.target_path_guard_extended
    cfg.target_path_guard_margin_enabled = args_cli.target_path_guard_margin
    cfg.target_path_guard_reserve_enabled = args_cli.target_path_guard_reserve
    cfg.learning_diagnostics_enabled = args_cli.learning_diagnostics
    cfg.target_buffer_failure_enabled = args_cli.target_buffer_failure
    cfg.frame_accurate_loss_counters = args_cli.frame_accurate_loss_counters
    cfg.near_ring_reset_curriculum_enabled = args_cli.c96_near_ring_reset_curriculum
    if args_cli.preentry_radial_priority:
        cfg.moving_target_slot_preentry_radial_priority_enabled = True
    if args_cli.stage in ("c40", "c41", "c42", "c43", "c44", "c45", "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55", "c56", "c57", "c58", "c60", "c61", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79", "c80", "c81", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91"):
        if cfg.dynamic_obstacle_indices:
            raise RuntimeError("C40/C41 require every physical obstacle to be static")
        if cfg.evader_speed_mps < cfg.training_command_speed_mps:
            raise RuntimeError("C40/C41 evader speed capability is below the pursuer")
        if (
            cfg.evader_max_planar_accel_mps2
            < cfg.pursuer_max_planar_accel_mps2
        ):
            raise RuntimeError(
                "C40/C41 evader acceleration capability is below the pursuer"
            )
    observation_dim, observation_schema = observation_contract(
        args_cli.stage, region_capture=args_cli.region_capture
    )
    state_dim = REGION_STATE_DIM if args_cli.region_capture else NATIVE_STATE_DIM
    cfg.state_space = state_dim
    cfg.observation_spaces = {
        agent: observation_dim for agent in cfg.possible_agents
    }
    if args_cli.evaluation_capture_radius_m is not None:
        cfg.training_capture_radius_m = args_cli.evaluation_capture_radius_m
    if args_cli.evaluation_sense_range_m is not None:
        cfg.sense_range_m = args_cli.evaluation_sense_range_m
    if args_cli.evaluation_tracked_reset_reference_range_m is not None:
        cfg.tracked_reset_reference_range_m = (
            args_cli.evaluation_tracked_reset_reference_range_m
        )
    if args_cli.evaluation_evader_speed_mps is not None:
        cfg.evader_speed_mps = args_cli.evaluation_evader_speed_mps
    if args_cli.evaluation_dynamic_obstacle_speed_mps is not None:
        cfg.dynamic_obstacle_speed_mps = (
            args_cli.evaluation_dynamic_obstacle_speed_mps
        )
    if args_cli.evaluation_ignore_dynamic_obstacle_velocity:
        cfg.predictive_static_use_obstacle_velocity = False
    if args_cli.evaluation_slot_prediction_horizon_s is not None:
        cfg.moving_target_slot_prediction_horizon_s = (
            args_cli.evaluation_slot_prediction_horizon_s
        )
    if args_cli.evaluation_loss_reacquisition is not None:
        cfg.loss_reacquisition_guidance_enabled = (
            args_cli.evaluation_loss_reacquisition in ("last", "flank", "sector")
        )
        cfg.loss_reacquisition_occlusion_flank_enabled = (
            args_cli.evaluation_loss_reacquisition == "flank"
        )
        cfg.loss_reacquisition_reachable_sector_enabled = (
            args_cli.evaluation_loss_reacquisition == "sector"
        )
        if args_cli.evaluation_loss_reacquisition in ("flank", "sector"):
            cfg.loss_reacquisition_scout_blend = 0.85
            cfg.loss_reacquisition_tracker_blend = 0.0
            cfg.loss_reacquisition_blend_growth = 0.0
    if args_cli.evaluation_pairwise_recovery is not None:
        cfg.predictive_agent_restore_unsafe_buffer = (
            args_cli.evaluation_pairwise_recovery == "restorative"
        )
    if args_cli.evaluation_contraction_speed_mps is not None:
        cfg.orientation_reference_contraction_speed_mps = (
            args_cli.evaluation_contraction_speed_mps
        )
        cfg.orientation_maximum_inward_speed_mps = (
            args_cli.evaluation_contraction_speed_mps
        )
    if args_cli.evaluation_orientation_staging_radius_m is not None:
        cfg.orientation_staging_radius_m = (
            args_cli.evaluation_orientation_staging_radius_m
        )
    if args_cli.evaluation_hybrid_residual_scale is not None:
        cfg.hybrid_encirclement_policy_residual_scale = (
            args_cli.evaluation_hybrid_residual_scale
        )
    if args_cli.evaluation_phase_gain is not None:
        cfg.moving_target_slot_phase_gain = args_cli.evaluation_phase_gain
    if args_cli.evaluation_phase_separation_control:
        cfg.moving_target_slot_phase_separation_control_enabled = True
        cfg.moving_target_slot_phase_activation_margin_m = 6.0
    if args_cli.evaluation_latched_reference_contraction:
        cfg.orientation_reference_contraction_latched_after_unlock = True
    if args_cli.evaluation_maximum_radial_speed_mps is not None:
        cfg.moving_target_slot_maximum_radial_speed_mps = (
            args_cli.evaluation_maximum_radial_speed_mps
        )
    if args_cli.evaluation_phase_scheduled_radial_budget:
        cfg.moving_target_slot_phase_scheduled_radial_speed_budget = True
    if args_cli.evaluation_maximum_tangential_speed_mps is not None:
        cfg.moving_target_slot_maximum_tangential_speed_mps = (
            args_cli.evaluation_maximum_tangential_speed_mps
        )
    if args_cli.evaluation_blocker_phase_anchor_gain is not None:
        cfg.moving_target_slot_blocker_phase_anchor_control_enabled = True
        cfg.moving_target_slot_blocker_phase_anchor_gain = (
            args_cli.evaluation_blocker_phase_anchor_gain
        )
    if args_cli.evaluation_reachable_role_slot_lead:
        cfg.reachable_intercept_reachable_role_slot_lead = True
    if args_cli.evaluation_far_slot_intercept_route:
        cfg.moving_target_slot_far_intercept_control_enabled = True
    if args_cli.evaluation_lock_slot_assignment:
        cfg.moving_target_slot_lock_valid_assignment = True
    if args_cli.evaluation_loss_scout_blend is not None:
        cfg.loss_reacquisition_scout_blend = args_cli.evaluation_loss_scout_blend
    if args_cli.evaluation_loss_tracker_blend is not None:
        cfg.loss_reacquisition_tracker_blend = args_cli.evaluation_loss_tracker_blend
    if args_cli.evaluation_loss_blend_growth is not None:
        cfg.loss_reacquisition_blend_growth = args_cli.evaluation_loss_blend_growth
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    cfg.episode_length_s = args_cli.episode_length_s

    evaluation_seed_plan = None
    evaluation_seed_plan_path = None
    if args_cli.evaluation_seed_plan is not None:
        evaluation_seed_plan_path = resolve_inside_repo(args_cli.evaluation_seed_plan)
        with evaluation_seed_plan_path.open(newline="", encoding="utf-8") as handle:
            evaluation_seed_plan = [
                int(row["episode_seed"]) for row in csv.DictReader(handle)
            ]
        if len(evaluation_seed_plan) != args_cli.eval_episodes:
            raise ValueError(
                "evaluation seed plan row count must equal --eval_episodes"
            )
        if len(set(evaluation_seed_plan)) != len(evaluation_seed_plan):
            raise ValueError("evaluation seed plan contains duplicate seeds")

    curriculum = native_curriculum_stage(args_cli.stage)
    resolved_config = {
        "schema": "p5_2_native_mappo_config_v2",
        "mode": "evaluation_only" if args_cli.evaluation_only else "training",
        "stage": args_cli.stage,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "iterations": args_cli.iterations,
        "rollout_steps": args_cli.rollout_steps,
        "eval_every": args_cli.eval_every,
        "eval_episodes": args_cli.eval_episodes,
        "target_success": args_cli.target_success,
        "max_collision": args_cli.max_collision,
        "required_passes": args_cli.required_passes,
        "hidden": args_cli.hidden,
        "learning_rate": args_cli.learning_rate,
        "actor_learning_rate": args_cli.actor_learning_rate or args_cli.learning_rate,
        "critic_learning_rate": args_cli.learning_rate,
        "expected_initial_evaluation": (str(args_cli.expected_initial_evaluation) if args_cli.expected_initial_evaluation else None),
        "ppo_epochs": args_cli.ppo_epochs,
        "minibatches": args_cli.minibatches,
        "entropy_coef": args_cli.entropy_coef,
        "preentry_radial_priority": args_cli.preentry_radial_priority,
        "episode_length_s": args_cli.episode_length_s,
        "action_semantics": (
            ["radial_gain_delta", "phase_gain_delta", "radial_velocity_residual"]
            if args_cli.stage in ("c80", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91")
            else
            ["radial_gain_delta", "tangential_velocity_residual", "radial_velocity_residual"]
            if args_cli.stage == "c81"
            else
            ["radial_gain_delta", "phase_gain_delta", "yaw_gain_delta"]
            if args_cli.stage in ("c32", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79")
            else ["vx_world", "vy_world", "yaw_rate"]
        ),
        "observation_dim": observation_dim,
        "observation_schema": observation_schema,
        "state_dim": state_dim,
        "action_dim": 3,
        "command_speed_limit_mps": cfg.training_command_speed_mps,
        "pursuer_max_planar_accel_mps2": cfg.pursuer_max_planar_accel_mps2,
        "evader_max_planar_accel_mps2": cfg.evader_max_planar_accel_mps2,
        "evader_capability_not_weaker_than_pursuer": bool(
            cfg.evader_speed_mps >= cfg.training_command_speed_mps
            and cfg.evader_max_planar_accel_mps2
            >= cfg.pursuer_max_planar_accel_mps2
        ),
        "training_capture_radius_m": cfg.training_capture_radius_m,
        "evaluation_termination_capture_radius_m": (
            cfg.training_capture_radius_m if args_cli.evaluation_only else None
        ),
        "strict_5m_requires_separate_replay": True,
        "fixed_evaluation_capture_radius_main_m": cfg.capture_radius_main_m,
        "fixed_evaluation_capture_radius_strict_m": cfg.capture_radius_strict_m,
        "desired_training_radius_m": cfg.desired_training_radius_m,
        "training_near_switch_m": cfg.training_near_switch_m,
        "static_obstacles": curriculum.static_obstacles,
        "physical_obstacle_count": len(cfg.obstacle_specs),
        "dynamic_obstacle_indices": list(cfg.dynamic_obstacle_indices),
        "dynamic_obstacle_speed_mps": cfg.dynamic_obstacle_speed_mps,
        "evaluation_dynamic_obstacle_speed_override_mps": (
            args_cli.evaluation_dynamic_obstacle_speed_mps
        ),
        "dynamic_obstacle_reversal_interval_s": (
            cfg.dynamic_obstacle_reversal_interval_s
        ),
        "physical_obstacle_kinds": list(cfg.obstacle_shape_kinds),
        "physical_obstacle_box_sizes_xy_m": [
            list(size_xy) for size_xy in cfg.obstacle_box_sizes_xy_m
        ],
        "obstacle_geometry_contract": (
            "exact_cylinder_or_box_physics; conservative_circumscribed_circle_"
            "perception_reset_and_safety"
        ),
        "randomize_obstacles_on_reset": cfg.randomize_obstacles_on_reset,
        "obstacle_randomization_seed_rule": (
            "episode_seed; fixed within episode"
            if cfg.randomize_obstacles_on_reset
            else "disabled"
        ),
        "evader_enabled": cfg.evader_enabled,
        "evader_model": (
            "CV/CT without STOP + largest-gap/obstacle/wall/pursuer APF"
            if args_cli.stage in ("c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30", "c31", "c32", "c40", "c41", "c42", "c43", "c44", "c45", "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55", "c56", "c57", "c58", "c60", "c61", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79", "c80", "c81", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91")
            else
            "CV/CT/STOP + obstacle/wall/pursuer APF"
            if cfg.evader_enabled
            else "stationary"
        ),
        "evader_speed_mps": cfg.evader_speed_mps,
        "evader_minimum_episode_speed_mps": (
            cfg.evader_minimum_episode_speed_mps
        ),
        "evader_allow_stop_mode": cfg.evader_allow_stop_mode,
        "evader_gap_escape_weight": cfg.evader_gap_escape_weight,
        "evader_maintain_episode_speed": cfg.evader_maintain_episode_speed,
        "evaluation_evader_speed_override_mps": (
            args_cli.evaluation_evader_speed_mps
        ),
        "sense_range_m": cfg.sense_range_m,
        "evaluation_sense_range_override_m": args_cli.evaluation_sense_range_m,
        "tracked_reset_reference_range_m": cfg.tracked_reset_reference_range_m,
        "c96_near_ring_reset_curriculum": cfg.near_ring_reset_curriculum_enabled,
        "capture_protocol": REGION_PROTOCOL if args_cli.region_capture else "legacy",
        "region_objective": args_cli.region_objective,
        "dual_observer_formation": args_cli.dual_observer_formation,
        "dual_observer_role_lock": args_cli.dual_observer_role_lock,
        "region_gap_reserve": args_cli.region_gap_reserve,
        "target_path_guard_acceleration": args_cli.target_path_guard_acceleration,
        "target_path_guard_extended": args_cli.target_path_guard_extended,
        "target_path_guard_margin": args_cli.target_path_guard_margin,
        "target_path_guard_reserve": args_cli.target_path_guard_reserve,
        "learning_diagnostics": args_cli.learning_diagnostics,
        "target_buffer_failure": args_cli.target_buffer_failure,
        "frame_accurate_loss_counters": args_cli.frame_accurate_loss_counters,
        "target_buffer_failure_contract": {"minimum_distance_m": 1.5, "sample_hz": 10,
            "comparison": "strict_less_than", "unsafe_terminal_reward": -5.,
            "contact_redefined": False, "enabled": args_cli.target_buffer_failure},
        "target_path_guard": args_cli.target_path_guard,
        "target_path_guard_contract": {
            "candidates": 125, "horizon_s": (1.0 if cfg.target_path_guard_extended_enabled else cfg.predictive_ring_prediction_horizon_s),
            "sample_dt_s": cfg.sim.dt, "activation": "team_visible",
            "actual_target_buffer_m": cfg.predictive_target_minimum_center_distance_m,
            "prediction_reserve_m": .5 if args_cli.target_path_guard_reserve else 0.,
            "prediction_target_threshold_m": cfg.predictive_target_minimum_center_distance_m + (.5 if args_cli.target_path_guard_reserve else 0.),
            "selected_infeasible_refers_to": "prediction_target_threshold_m",
            "target_prediction": ("cv_and_bounded_ca_from_consecutive_visible_measurements"
                                  if args_cli.target_path_guard_acceleration else "current_measured_cv"),
            "selection": ("per_constraint_path_safety_nonregression_then_target_buffer_repair_then_maximum_worst_target_distance_then_least_command_change"
                          if args_cli.target_path_guard_margin else
                          "per_constraint_path_safety_nonregression_then_target_buffer_repair_then_least_command_change"),
            "location": "after_all_original_safety_sweeps",
            "limitations": "candidate-set feasibility in a planar model; not a rigid-body safety certificate",
        } if args_cli.target_path_guard else None,
        "target_path_guard_source_sha256": sha256_file(PACKAGE_SOURCE / "bylw_isaac_p5/core/target_path_guard.py"),
        "region_joint_lookahead": args_cli.region_joint_lookahead,
        "region_joint_lookahead_contract": {
            "candidates": 125, "horizon_s": cfg.predictive_ring_prediction_horizon_s,
            "maximum_agent_correction_mps": cfg.moving_target_slot_radial_velocity_residual_scale_mps,
            "activation": "all_three_visible_and_unlocked", "target_prediction": "current_measured_cv",
            "selection": "terminal_radial_and_gap_pareto_with_sampled_safety_and_los_nonregression",
        } if args_cli.region_joint_lookahead else None,
        "region_control_maximum_gap_rad": cfg.region_capture_maximum_gap_rad - (math.pi/12 if args_cli.region_gap_reserve else 0.),
        "dual_observer_role_hysteresis_m": cfg.dual_observer_role_hysteresis_m,
        "dual_observer_phase_tolerance_rad": cfg.dual_observer_phase_tolerance_rad,
        "dual_observer_source_sha256": sha256_file(PACKAGE_SOURCE / "bylw_isaac_p5/core/dual_observer.py"),
        "region_reward_schema": ("region_v1_terminal_potential_target_buffer_failure_v1"
            if args_cli.target_buffer_failure else REGION_REWARD if args_cli.region_objective else None),
        "region_minimum_target_distance_m": cfg.region_capture_minimum_target_distance_m,
        "region_control_minimum_radius_m": cfg.region_control_minimum_radius_m if cfg.region_objective_enabled else None,
        "ppo_exact_logprob": args_cli.ppo_exact_logprob,
        "region_source_sha256": sha256_file(PACKAGE_SOURCE / "bylw_isaac_p5/core/region_capture.py"),
        "formal_evaluation_reset_curriculum_enabled": False,
        "c96_reset_radius_range_m": [5.6, 6.2],
        "c96_reset_angular_perturbation_deg": 10.0,
        "reset_sampler_sha256": sha256_file(
            PACKAGE_SOURCE / "bylw_isaac_p5" / "core" / "reset_sampler.py"
        ),
        "evaluation_reset_guard_sha256": sha256_file(
            PACKAGE_SOURCE / "bylw_isaac_p5" / "core" / "evaluation_reset.py"
        ),
        "evaluation_tracked_reset_reference_range_override_m": (
            args_cli.evaluation_tracked_reset_reference_range_m
        ),
        "sensor_ablation_changes_reset_geometry": False,
        "fov_half_rad": cfg.fov_half_rad,
        "orientation_gate_enabled": cfg.orientation_gate_enabled,
        "role_phase_observation_enabled": cfg.role_phase_observation_enabled,
        "role_phase_feature_order": (
            [
                "assigned_slot_dx",
                "assigned_slot_dy",
                "role_blocker",
                "role_left_scout",
                "role_right_scout",
                "phase_approach",
                "phase_align",
                "phase_contract",
                "normalized_reference_radius",
                "team_current_ready_fraction",
            ]
            if cfg.role_phase_observation_enabled
            else []
        ),
        "role_phase_checkpoint_migration": (
            "copy_first_53_actor_input_columns;zero_initialize_new_10;"
            "copy_remaining_actor_and_58d_critic;reset_optimizer"
            if cfg.role_phase_observation_enabled
            else None
        ),
        "orientation_staging_radius_m": cfg.orientation_staging_radius_m,
        "evaluation_orientation_staging_radius_override_m": (
            args_cli.evaluation_orientation_staging_radius_m
        ),
        "orientation_required_alignment_steps": (
            cfg.orientation_required_alignment_steps
        ),
        "orientation_yaw_tolerance_rad": cfg.orientation_yaw_tolerance_rad,
        "orientation_yaw_gain": cfg.orientation_yaw_gain,
        "orientation_policy_residual_scale": (
            cfg.orientation_policy_residual_scale
        ),
        "orientation_maximum_inward_speed_mps": (
            cfg.orientation_maximum_inward_speed_mps
        ),
        "evaluation_contraction_speed_override_mps": (
            args_cli.evaluation_contraction_speed_mps
        ),
        "orientation_target_relative_staging_enabled": (
            cfg.orientation_target_relative_staging_enabled
        ),
        "asynchronous_require_current_team_ready": (
            cfg.asynchronous_require_current_team_ready
        ),
        "asynchronous_current_readiness_reward_enabled": (
            cfg.asynchronous_current_readiness_reward_enabled
        ),
        "orientation_reference_contraction_speed_mps": (
            cfg.orientation_reference_contraction_speed_mps
        ),
        "orientation_reference_contraction_tracking_gate_enabled": (
            cfg.orientation_reference_contraction_tracking_gate_enabled
        ),
        "orientation_reference_anchor_to_mean_radius_on_unlock": (
            cfg.orientation_reference_anchor_to_mean_radius_on_unlock
        ),
        "orientation_contraction_pause_without_team_ready": (
            cfg.orientation_contraction_pause_without_team_ready
        ),
        "orientation_phase_gate_enabled": cfg.orientation_phase_gate_enabled,
        "orientation_phase_maximum_gap_rad": (
            cfg.orientation_phase_maximum_gap_rad
        ),
        "orientation_phase_maximum_mean_gap_error_rad": (
            cfg.orientation_phase_maximum_mean_gap_error_rad
        ),
        "predictive_ring_stabilization_enabled": (
            cfg.predictive_ring_stabilization_enabled
        ),
        "predictive_ring_prediction_horizon_s": (
            cfg.predictive_ring_prediction_horizon_s
        ),
        "predictive_ring_command_response_ratio": (
            cfg.predictive_ring_command_response_ratio
        ),
        "predictive_ring_projection_iterations": (
            cfg.predictive_ring_projection_iterations
        ),
        "predictive_ring_activation_margin_m": (
            cfg.predictive_ring_activation_margin_m
        ),
        "predictive_ring_activation_maximum_gap_margin_rad": (
            cfg.predictive_ring_activation_maximum_gap_margin_rad
        ),
        "predictive_ring_activation_requires_inside_inner_boundary": (
            cfg.predictive_ring_activation_requires_inside_inner_boundary
        ),
        "predictive_ring_activation_requires_maximum_gap_eligibility": (
            cfg.predictive_ring_activation_requires_maximum_gap_eligibility
        ),
        "predictive_ring_response_trajectory_substeps": (
            cfg.predictive_ring_response_trajectory_substeps
        ),
        "predictive_ring_enforce_minimum_radius": (
            cfg.predictive_ring_enforce_minimum_radius
        ),
        "predictive_ring_minimum_radius_after_strong_entry": (
            cfg.predictive_ring_minimum_radius_after_strong_entry
        ),
        "predictive_ring_dynamics_consistent_enabled": (
            cfg.predictive_ring_dynamics_consistent_enabled
        ),
        "predictive_ring_low_level_velocity_kp": (
            cfg.predictive_ring_low_level_velocity_kp
        ),
        "predictive_ring_radial_candidate_count": (
            cfg.predictive_ring_radial_candidate_count
        ),
        "predictive_ring_common_acceleration_feedforward_after_strong_entry": (
            cfg.predictive_ring_common_acceleration_feedforward_after_strong_entry
        ),
        "predictive_ring_preentry_velocity_sync_enabled": (
            cfg.predictive_ring_preentry_velocity_sync_enabled
        ),
        "predictive_ring_target_acceleration_enabled": (
            cfg.predictive_ring_target_acceleration_enabled
        ),
        "predictive_ring_target_acceleration_substeps": (
            cfg.predictive_ring_target_acceleration_substeps
        ),
        "predictive_ring_angular_gap_projection_enabled": (
            cfg.predictive_ring_angular_gap_projection_enabled
        ),
        "predictive_ring_angular_gap_candidate_count": (
            cfg.predictive_ring_angular_gap_candidate_count
        ),
        "predictive_ring_preentry_braking_enabled": (
            cfg.predictive_ring_preentry_braking_enabled
        ),
        "predictive_ring_preentry_braking_activation_radius_m": (
            cfg.predictive_ring_preentry_braking_activation_radius_m
        ),
        "predictive_ring_angular_gap_preentry_enabled": (
            cfg.predictive_ring_angular_gap_preentry_enabled
        ),
        "predictive_ring_angular_gap_preentry_activation_radius_m": (
            cfg.predictive_ring_angular_gap_preentry_activation_radius_m
        ),
        "hybrid_phase_supervisor_enabled": (
            cfg.hybrid_phase_supervisor_enabled
        ),
        "hybrid_encirclement_policy_residual_scale": (
            cfg.hybrid_encirclement_policy_residual_scale
        ),
        "evaluation_hybrid_residual_scale_override": (
            args_cli.evaluation_hybrid_residual_scale
        ),
        "evaluation_phase_gain_override": args_cli.evaluation_phase_gain,
        "evaluation_phase_separation_control_override": (
            args_cli.evaluation_phase_separation_control
        ),
        "evaluation_latched_reference_contraction_override": (
            args_cli.evaluation_latched_reference_contraction
        ),
        "evaluation_maximum_radial_speed_mps_override": (
            args_cli.evaluation_maximum_radial_speed_mps
        ),
        "evaluation_phase_scheduled_radial_budget_override": (
            args_cli.evaluation_phase_scheduled_radial_budget
        ),
        "evaluation_maximum_tangential_speed_mps_override": (
            args_cli.evaluation_maximum_tangential_speed_mps
        ),
        "evaluation_blocker_phase_anchor_gain_override": (
            args_cli.evaluation_blocker_phase_anchor_gain
        ),
        "evaluation_reachable_role_slot_lead_override": (
            args_cli.evaluation_reachable_role_slot_lead
        ),
        "evaluation_far_slot_intercept_route_override": (
            args_cli.evaluation_far_slot_intercept_route
        ),
        "evaluation_lock_slot_assignment_override": (
            args_cli.evaluation_lock_slot_assignment
        ),
        "hybrid_require_current_team_ready": (
            cfg.hybrid_require_current_team_ready
        ),
        "hybrid_rl_activation_on_staging": (
            cfg.hybrid_rl_activation_on_staging
        ),
        "asynchronous_orientation_gate_enabled": (
            cfg.asynchronous_orientation_gate_enabled
        ),
        "asynchronous_orientation_progress_reward_weight": (
            cfg.asynchronous_orientation_progress_reward_weight
        ),
        "strong_ring_termination_enabled": (
            cfg.strong_ring_termination_enabled
        ),
        "strong_ring_minimum_radius_m": cfg.strong_ring_minimum_radius_m,
        "strong_ring_maximum_radius_m": cfg.strong_ring_maximum_radius_m,
        "strong_ring_maximum_gap_rad": cfg.strong_ring_maximum_gap_rad,
        "strong_ring_maximum_mean_gap_error_rad": (
            cfg.strong_ring_maximum_mean_gap_error_rad
        ),
        "strong_ring_required_hold_steps": cfg.strong_ring_required_hold_steps,
        "strong_ring_require_all_visible": cfg.strong_ring_require_all_visible,
        "strong_ring_step_reward_weight": cfg.strong_ring_step_reward_weight,
        "strong_ring_violation_progress_reward_weight": (
            cfg.strong_ring_violation_progress_reward_weight
        ),
        "strong_ring_confirmation_maximum_phase_gain_enabled": (
            cfg.strong_ring_confirmation_maximum_phase_gain_enabled
        ),
        "moving_target_slot_guidance_enabled": (
            cfg.moving_target_slot_guidance_enabled
        ),
        "moving_target_slot_activation_radius_m": (
            cfg.moving_target_slot_activation_radius_m
        ),
        "moving_target_slot_position_gain": cfg.moving_target_slot_position_gain,
        "moving_target_slot_policy_residual_scale": (
            cfg.moving_target_slot_policy_residual_scale
        ),
        "moving_target_slot_feedforward_scale": (
            cfg.moving_target_slot_feedforward_scale
        ),
        "moving_target_slot_feedforward_priority_projection_enabled": (
            cfg.moving_target_slot_feedforward_priority_projection_enabled
        ),
        "moving_target_slot_preentry_radial_priority_enabled": (
            cfg.moving_target_slot_preentry_radial_priority_enabled
        ),
        "moving_target_slot_full_feedforward_above_strong_ring": (
            cfg.moving_target_slot_full_feedforward_above_strong_ring
        ),
        "moving_target_slot_prediction_horizon_s": (
            cfg.moving_target_slot_prediction_horizon_s
        ),
        "moving_target_slot_minimum_target_speed_mps": (
            cfg.moving_target_slot_minimum_target_speed_mps
        ),
        "moving_target_slot_maximum_anchor_turn_rate_rps": (
            cfg.moving_target_slot_maximum_anchor_turn_rate_rps
        ),
        "moving_target_slot_assignment_hysteresis_m": (
            cfg.moving_target_slot_assignment_hysteresis_m
        ),
        "moving_target_slot_error_use_maximum_error": (
            cfg.moving_target_slot_error_use_maximum_error
        ),
        "moving_target_slot_obstacle_aware_assignment": (
            cfg.moving_target_slot_obstacle_aware_assignment
        ),
        "moving_target_slot_assignment_path_clearance_m": (
            cfg.moving_target_slot_assignment_path_clearance_m
        ),
        "moving_target_slot_assignment_obstacle_penalty": (
            cfg.moving_target_slot_assignment_obstacle_penalty
        ),
        "moving_target_slot_cyclic_order_preserving_assignment": (
            cfg.moving_target_slot_cyclic_order_preserving_assignment
        ),
        "moving_target_slot_lock_valid_assignment": (
            cfg.moving_target_slot_lock_valid_assignment
        ),
        "moving_target_slot_phase_separation_control_enabled": (
            cfg.moving_target_slot_phase_separation_control_enabled
        ),
        "moving_target_slot_phase_gain": cfg.moving_target_slot_phase_gain,
        "moving_target_slot_phase_angular_velocity_damping_gain_s": (
            cfg.moving_target_slot_phase_angular_velocity_damping_gain_s
        ),
        "moving_target_slot_radial_gain": cfg.moving_target_slot_radial_gain,
        "moving_target_slot_radial_velocity_damping_gain": (
            cfg.moving_target_slot_radial_velocity_damping_gain
        ),
        "moving_target_slot_maximum_radial_speed_mps": (
            cfg.moving_target_slot_maximum_radial_speed_mps
        ),
        "moving_target_slot_phase_scheduled_radial_speed_budget": (
            cfg.moving_target_slot_phase_scheduled_radial_speed_budget
        ),
        "moving_target_slot_maximum_tangential_speed_mps": (
            cfg.moving_target_slot_maximum_tangential_speed_mps
        ),
        "moving_target_slot_phase_activation_margin_m": (
            cfg.moving_target_slot_phase_activation_margin_m
        ),
        "moving_target_slot_far_intercept_control_enabled": (
            cfg.moving_target_slot_far_intercept_control_enabled
        ),
        "moving_target_slot_blocker_phase_anchor_control_enabled": (
            cfg.moving_target_slot_blocker_phase_anchor_control_enabled
        ),
        "moving_target_slot_blocker_phase_anchor_gain": (
            cfg.moving_target_slot_blocker_phase_anchor_gain
        ),
        "moving_target_slot_blocker_phase_anchor_maximum_speed_mps": (
            cfg.moving_target_slot_blocker_phase_anchor_maximum_speed_mps
        ),
        "moving_target_slot_gain_residual_control_enabled": (
            cfg.moving_target_slot_gain_residual_control_enabled
        ),
        "moving_target_slot_radial_velocity_residual_control_enabled": (
            cfg.moving_target_slot_radial_velocity_residual_control_enabled
        ),
        "moving_target_slot_radial_velocity_residual_scale_mps": (
            cfg.moving_target_slot_radial_velocity_residual_scale_mps
        ),
        "moving_target_slot_tangential_velocity_residual_control_enabled": (
            cfg.moving_target_slot_tangential_velocity_residual_control_enabled
        ),
        "moving_target_slot_tangential_velocity_residual_scale_mps": (
            cfg.moving_target_slot_tangential_velocity_residual_scale_mps
        ),
        "moving_target_slot_radial_gain_residual_scale": (
            cfg.moving_target_slot_radial_gain_residual_scale
        ),
        "moving_target_slot_phase_gain_residual_scale": (
            cfg.moving_target_slot_phase_gain_residual_scale
        ),
        "orientation_yaw_gain_residual_scale": (
            cfg.orientation_yaw_gain_residual_scale
        ),
        "moving_target_slot_error_reward_weight": (
            cfg.moving_target_slot_error_reward_weight
        ),
        "moving_target_slot_progress_reward_weight": (
            cfg.moving_target_slot_progress_reward_weight
        ),
        "evaluation_loss_reacquisition": args_cli.evaluation_loss_reacquisition,
        "loss_reacquisition_guidance_enabled": (
            cfg.loss_reacquisition_guidance_enabled
        ),
        "loss_reacquisition_occlusion_flank_enabled": (
            cfg.loss_reacquisition_occlusion_flank_enabled
        ),
        "loss_reacquisition_start_lost_steps": (
            cfg.loss_reacquisition_start_lost_steps
        ),
        "loss_reacquisition_slot_radius_m": (
            cfg.loss_reacquisition_slot_radius_m
        ),
        "loss_reacquisition_scout_blend": cfg.loss_reacquisition_scout_blend,
        "loss_reacquisition_tracker_blend": cfg.loss_reacquisition_tracker_blend,
        "loss_reacquisition_blend_growth": (
            cfg.loss_reacquisition_blend_growth
        ),
        "loss_reacquisition_scout_speed_ratio": (
            cfg.loss_reacquisition_scout_speed_ratio
        ),
        "loss_reacquisition_tracker_speed_ratio": (
            cfg.loss_reacquisition_tracker_speed_ratio
        ),
        "loss_reacquisition_scout_minimum_selection_distance_m": (
            cfg.loss_reacquisition_scout_minimum_selection_distance_m
        ),
        "loss_reacquisition_tracker_hold_enabled": (
            cfg.loss_reacquisition_tracker_hold_enabled
        ),
        "loss_reacquisition_slot_arrival_m": (
            cfg.loss_reacquisition_slot_arrival_m
        ),
        "loss_reacquisition_flank_waypoint_margin_m": (
            cfg.loss_reacquisition_flank_waypoint_margin_m
        ),
        "loss_reacquisition_flank_los_margin_m": (
            cfg.loss_reacquisition_flank_los_margin_m
        ),
        "loss_reacquisition_flank_blocker_margin_m": (
            cfg.loss_reacquisition_flank_blocker_margin_m
        ),
        "loss_reacquisition_flank_boundary_clearance_m": (
            cfg.loss_reacquisition_flank_boundary_clearance_m
        ),
        "loss_reacquisition_flank_angular_samples": (
            cfg.loss_reacquisition_flank_angular_samples
        ),
        "loss_reacquisition_reachable_sector_enabled": (
            cfg.loss_reacquisition_reachable_sector_enabled
        ),
        "loss_reacquisition_sector_target_speed_bound_mps": (
            cfg.loss_reacquisition_sector_target_speed_bound_mps
        ),
        "loss_reacquisition_sector_velocity_heading_threshold_mps": (
            cfg.loss_reacquisition_sector_velocity_heading_threshold_mps
        ),
        "loss_reacquisition_sector_lookahead_s": (
            cfg.loss_reacquisition_sector_lookahead_s
        ),
        "loss_reacquisition_sector_minimum_radius_m": (
            cfg.loss_reacquisition_sector_minimum_radius_m
        ),
        "loss_reacquisition_sector_maximum_radius_m": (
            cfg.loss_reacquisition_sector_maximum_radius_m
        ),
        "loss_reacquisition_sector_initial_half_angle_rad": (
            cfg.loss_reacquisition_sector_initial_half_angle_rad
        ),
        "loss_reacquisition_sector_half_angle_growth_rps": (
            cfg.loss_reacquisition_sector_half_angle_growth_rps
        ),
        "loss_reacquisition_sector_phase_hold_steps": (
            cfg.loss_reacquisition_sector_phase_hold_steps
        ),
        "loss_reacquisition_sector_boundary_clearance_m": (
            cfg.loss_reacquisition_sector_boundary_clearance_m
        ),
        "loss_reacquisition_sector_detour_margin_m": (
            cfg.loss_reacquisition_sector_detour_margin_m
        ),
        "loss_reacquisition_sector_detour_maximum_extra_m": (
            cfg.loss_reacquisition_sector_detour_maximum_extra_m
        ),
        "loss_reacquisition_sector_detour_radial_samples": (
            cfg.loss_reacquisition_sector_detour_radial_samples
        ),
        "loss_reacquisition_sector_detour_angular_samples": (
            cfg.loss_reacquisition_sector_detour_angular_samples
        ),
        "reachable_intercept_guidance_enabled": (
            cfg.reachable_intercept_guidance_enabled
        ),
        "reachable_intercept_minimum_horizon_s": (
            cfg.reachable_intercept_minimum_horizon_s
        ),
        "reachable_intercept_maximum_horizon_s": (
            cfg.reachable_intercept_maximum_horizon_s
        ),
        "reachable_intercept_horizon_samples": (
            cfg.reachable_intercept_horizon_samples
        ),
        "reachable_intercept_slot_horizon_cap_s": (
            cfg.reachable_intercept_slot_horizon_cap_s
        ),
        "reachable_intercept_blocker_only_slot_lead": (
            cfg.reachable_intercept_blocker_only_slot_lead
        ),
        "reachable_intercept_reachable_role_slot_lead": (
            cfg.reachable_intercept_reachable_role_slot_lead
        ),
        "reachable_intercept_blocker_role_lock_weight_m": (
            cfg.reachable_intercept_blocker_role_lock_weight_m
        ),
        "moving_target_angular_error_reward_weight": (
            cfg.moving_target_angular_error_reward_weight
        ),
        "moving_target_angular_progress_reward_weight": (
            cfg.moving_target_angular_progress_reward_weight
        ),
        "predictive_target_projection_enabled": (
            cfg.predictive_target_projection_enabled
        ),
        "predictive_target_minimum_center_distance_m": (
            cfg.predictive_target_minimum_center_distance_m
        ),
        "predictive_target_prediction_horizon_s": (
            cfg.predictive_target_prediction_horizon_s
        ),
        "predictive_target_command_response_ratio": (
            cfg.predictive_target_command_response_ratio
        ),
        "predictive_target_projection_iterations": (
            cfg.predictive_target_projection_iterations
        ),
        "predictive_target_restore_unsafe_buffer": (
            cfg.predictive_target_restore_unsafe_buffer
        ),
        "predictive_target_projection_last": (
            cfg.predictive_target_projection_last
        ),
        "predictive_target_last_seen_cv_during_loss_enabled": (
            cfg.predictive_target_last_seen_cv_during_loss_enabled
        ),
        "predictive_target_last_seen_cv_maximum_loss_time_s": (
            cfg.predictive_target_last_seen_cv_maximum_loss_time_s
        ),
        "predictive_target_last_seen_reachable_tube_enabled": (
            cfg.predictive_target_last_seen_reachable_tube_enabled
        ),
        "predictive_agent_projection_enabled": (
            cfg.predictive_agent_projection_enabled
        ),
        "evaluation_pairwise_recovery": args_cli.evaluation_pairwise_recovery,
        "predictive_agent_restore_unsafe_buffer": (
            cfg.predictive_agent_restore_unsafe_buffer
        ),
        "predictive_agent_always_active": cfg.predictive_agent_always_active,
        "predictive_agent_minimum_center_distance_m": (
            cfg.predictive_agent_minimum_center_distance_m
        ),
        "predictive_agent_prediction_horizon_s": (
            cfg.predictive_agent_prediction_horizon_s
        ),
        "predictive_agent_command_response_ratio": (
            cfg.predictive_agent_command_response_ratio
        ),
        "predictive_agent_projection_iterations": (
            cfg.predictive_agent_projection_iterations
        ),
        "predictive_agent_correction_reward_weight": (
            cfg.predictive_agent_correction_reward_weight
        ),
        "predictive_agent_infeasible_reward_penalty": (
            cfg.predictive_agent_infeasible_reward_penalty
        ),
        "predictive_static_projection_enabled": (
            cfg.predictive_static_projection_enabled
        ),
        "predictive_static_use_obstacle_velocity": (
            cfg.predictive_static_use_obstacle_velocity
        ),
        "predictive_static_minimum_clearance_m": (
            cfg.predictive_static_minimum_clearance_m
        ),
        "predictive_static_prediction_horizon_s": (
            cfg.predictive_static_prediction_horizon_s
        ),
        "predictive_static_command_response_ratio": (
            cfg.predictive_static_command_response_ratio
        ),
        "predictive_static_projection_iterations": (
            cfg.predictive_static_projection_iterations
        ),
        "predictive_boundary_projection_enabled": (
            cfg.predictive_boundary_projection_enabled
        ),
        "predictive_boundary_minimum_clearance_m": (
            cfg.predictive_boundary_minimum_clearance_m
        ),
        "predictive_boundary_prediction_horizon_s": (
            cfg.predictive_boundary_prediction_horizon_s
        ),
        "predictive_boundary_command_response_ratio": (
            cfg.predictive_boundary_command_response_ratio
        ),
        "predictive_boundary_correction_reward_weight": (
            cfg.predictive_boundary_correction_reward_weight
        ),
        "predictive_boundary_infeasible_reward_penalty": (
            cfg.predictive_boundary_infeasible_reward_penalty
        ),
        "static_tangent_bypass_enabled": cfg.static_tangent_bypass_enabled,
        "static_tangent_activation_steps": cfg.static_tangent_activation_steps,
        "static_tangent_hold_steps": cfg.static_tangent_hold_steps,
        "static_tangent_speed_mps": cfg.static_tangent_speed_mps,
        "static_tangent_outward_speed_mps": (
            cfg.static_tangent_outward_speed_mps
        ),
        "static_tangent_release_distance_m": (
            cfg.static_tangent_release_distance_m
        ),
        "joint_safety_projection_sweeps": cfg.joint_safety_projection_sweeps,
        "reset_optimizer_on_resume": args_cli.reset_optimizer_on_resume,
        "reset_actor_output_on_resume": args_cli.reset_actor_output_on_resume,
        "c94_extended_episode_horizon": (
            args_cli.c94_extended_episode_horizon
        ),
        "evaluation_seed_plan": (
            str(evaluation_seed_plan_path.relative_to(REPO_ROOT))
            if evaluation_seed_plan_path
            else None
        ),
        "evaluation_seed_plan_sha256": (
            sha256_file(evaluation_seed_plan_path)
            if evaluation_seed_plan_path
            else None
        ),
        "evaluation_seed_rule": (
            "explicit_csv"
            if evaluation_seed_plan_path
            else "contiguous_eval_seed_plus_episode_index"
        ),
        "implicit_evaluation_seed_start": (
            None if evaluation_seed_plan_path else args_cli.seed + 50_000_000
        ),
        "implicit_evaluation_seed_stop_exclusive": (
            None
            if evaluation_seed_plan_path
            else args_cli.seed + 50_000_000 + args_cli.eval_episodes
        ),
        "resume": str(args_cli.resume.resolve()) if args_cli.resume else None,
        "device": args_cli.device,
        "physical_gpu_index": args_cli.physical_gpu_index,
        "initial_gpu": initial_gpu,
    }
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    wall_start = time.monotonic()
    env = BylwPursuitTrainEnv(cfg)
    observations, _ = env.reset(seed=args_cli.seed)
    if args_cli.c96_near_ring_reset_curriculum:
        initial_reset_audit = {
            "episode_seeds": env._episode_seeds.cpu().tolist(),
            "initial_radius_m": env._reset_initial_radius_m.cpu().tolist(),
            "attempts": env._curriculum_reset_attempts,
            "applied": env._curriculum_reset_applied,
            "initial_velocity_contract": "all zero, unchanged from C91",
            "orientation_and_hold_gates": "not prefilled",
        }
        (output_dir / "initial_reset_audit.json").write_text(
            json.dumps(initial_reset_audit, indent=2) + "\n", encoding="utf-8"
        )
    state = env.state()
    agents = list(cfg.possible_agents)
    first = agents[0]
    if stack_observations(observations, agents).shape != (
        args_cli.num_envs, 3, observation_dim
    ):
        raise RuntimeError("native observation shape drift")
    if state.shape != (args_cli.num_envs, state_dim):
        raise RuntimeError("native state shape drift")

    agent = MAPPO(
        obs_dim=observation_dim,
        state_dim=state_dim,
        act_dim=3,
        n_agent=3,
        hidden=args_cli.hidden,
        lr=args_cli.learning_rate,
        actor_lr=args_cli.actor_learning_rate,
        clip=0.2,
        epochs=args_cli.ppo_epochs,
        minibatch=args_cli.minibatches,
        vf_coef=0.5,
        ent_coef=args_cli.entropy_coef,
        max_grad_norm=0.5,
        device=args_cli.device,
        collect_diagnostics=args_cli.learning_diagnostics,
    )
    start_iteration = 0
    if args_cli.resume:
        resume_path = resolve_inside_repo(args_cli.resume)
        start_iteration = load_checkpoint(
            resume_path,
            agent,
            load_optimizer=(
                not args_cli.reset_optimizer_on_resume
                and not args_cli.evaluation_only
            ),
            expected_action_semantics=action_semantics_contract(args_cli.stage),
            allow_action_semantics_mismatch=(
                args_cli.reset_actor_output_on_resume
            ),
            expected_capture_protocol=REGION_PROTOCOL if args_cli.region_capture else "legacy",
            allow_capture_protocol_migration=args_cli.allow_capture_protocol_migration,
            expected_dual_observer_formation=args_cli.dual_observer_formation,
            expected_dual_observer_role_lock=args_cli.dual_observer_role_lock,
            expected_region_gap_reserve=args_cli.region_gap_reserve,
            expected_region_joint_lookahead=args_cli.region_joint_lookahead,
            expected_target_path_guard=args_cli.target_path_guard,
            expected_target_path_guard_acceleration=args_cli.target_path_guard_acceleration,
            expected_target_path_guard_extended=args_cli.target_path_guard_extended,
            expected_target_path_guard_margin=args_cli.target_path_guard_margin,
            expected_target_path_guard_reserve=args_cli.target_path_guard_reserve,
            expected_optimizer_learning_rates=(args_cli.actor_learning_rate or args_cli.learning_rate, args_cli.learning_rate),
            expected_target_buffer_failure=args_cli.target_buffer_failure,
            expected_frame_accurate_loss_counters=args_cli.frame_accurate_loss_counters,
            allow_loss_counter_migration=args_cli.allow_loss_counter_migration,
            allow_target_buffer_failure_migration=args_cli.allow_target_buffer_failure_migration,
            allow_controller_migration=args_cli.allow_controller_migration,
        )
        if args_cli.reset_actor_output_on_resume:
            with torch.no_grad():
                agent.actor.mu.weight.zero_()
                agent.actor.mu.bias.zero_()
                agent.actor.log_std.fill_(-0.5)
        print(f"[P5-2] resumed {resume_path} at iteration={start_iteration}", flush=True)

    train_csv = output_dir / "train_log.csv"
    eval_csv = output_dir / "eval_log.csv"
    with train_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            [
                "iteration",
                "environment_steps",
                "rolling_return",
                "rolling_success_training",
                "pi_loss",
                "v_loss",
                "entropy",
                "policy_active_fraction",
                "gpu_memory_mb",
                "wall_time_s",
            ]
        )
    with eval_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            [
                "iteration",
                "episodes",
                "success_rate_training",
                "success_rate_6m",
                "success_rate_5m",
                "collision_rate",
                "collision_agent_rate",
                "collision_agent_before_unlock_rate",
                "collision_agent_after_unlock_rate",
                "collision_static_rate",
                "collision_dynamic_rate",
                "contact_rate",
                "invalid_rate",
                "boundary_invalid_rate",
                "height_invalid_rate",
                "timeout_rate",
                "orientation_unlock_rate",
                "terminal_orientation_staging_fraction",
                "terminal_orientation_alignment_fraction",
                "terminal_intercept_reachable_fraction",
                "mean_terminal_intercept_horizon_s",
                "intercept_blocker_reachable_rate",
                "mean_all_visible_step_fraction",
                "mean_staging_all_visible_step_fraction",
                "loss_episode_rate",
                "long_loss_2s_episode_rate",
                "mean_maximum_lost_steps",
                "maximum_lost_steps",
                "team_lost_step_fraction",
                "mean_recapture_count",
                "loss_reacquisition_guidance_enabled",
                "loss_reacquisition_eligible_episode_rate",
                "loss_reacquisition_first_completed_episode_rate",
                "loss_reacquisition_first_within_2s_episode_rate",
                "loss_reacquisition_event_completion_rate",
                "mean_loss_reacquisition_active_steps_per_episode",
                "mean_first_loss_reacquisition_delay_steps",
                "mean_target_episode_displacement_m",
                "mean_terminal_target_speed_mps",
                "minimum_evader_episode_speed_limit_mps",
                "mean_evader_episode_speed_limit_mps",
                "maximum_evader_episode_speed_limit_mps",
                "mean_maximum_strong_ring_hold_steps",
                "maximum_strong_ring_hold_steps",
                "mean_maximum_dynamic_obstacle_displacement_m",
                "maximum_dynamic_obstacle_displacement_m",
                "mean_maximum_dynamic_obstacle_speed_mps",
                "unique_obstacle_layout_count",
                "pure_range_lost_step_fraction",
                "pure_fov_lost_step_fraction",
                "pure_occlusion_lost_step_fraction",
                "mixed_loss_step_fraction",
                "terminal_orientation_yaw_error_max_rad",
                "mean_first_orientation_unlock_step",
                "mean_episode_steps",
                "terminal_radial_error_m",
                "terminal_angular_error_rad",
                "mean_minimum_agent_clearance_m",
                "minimum_agent_clearance_m",
                "mean_minimum_static_clearance_m",
                "minimum_static_clearance_m",
                "mean_minimum_boundary_clearance_m",
                "minimum_boundary_clearance_m",
                "target_projection_episode_rate",
                "mean_target_projection_calls_per_episode",
                "target_projection_corrected_step_rate",
                "mean_target_projection_constraint_corrections_per_call",
                "mean_target_projection_maximum_separation_fallback_constraints_per_call",
                "target_projection_unavoidable_step_rate",
                "mean_target_projection_unavoidable_constraints_per_call",
                "target_projection_infeasible_step_rate",
                "mean_target_projection_correction_norm_mps",
                "maximum_target_projection_correction_norm_mps",
                "minimum_target_projection_predicted_center_distance_m",
                "minimum_target_projection_maximum_attainable_center_distance_m",
                "safety_projection_episode_rate",
                "mean_safety_projection_calls_per_episode",
                "safety_projection_corrected_step_rate",
                "mean_safety_projection_pair_corrections_per_call",
                "safety_projection_infeasible_step_rate",
                "mean_safety_projection_correction_norm_mps",
                "maximum_safety_projection_correction_norm_mps",
                "minimum_safety_projection_predicted_center_distance_m",
                "obstacle_projection_episode_rate",
                "mean_obstacle_projection_calls_per_episode",
                "obstacle_projection_corrected_step_rate",
                "mean_obstacle_projection_constraint_corrections_per_call",
                "obstacle_projection_infeasible_step_rate",
                "mean_obstacle_projection_correction_norm_mps",
                "maximum_obstacle_projection_correction_norm_mps",
                "minimum_obstacle_projection_predicted_clearance_m",
                "tangent_bypass_episode_rate",
                "mean_tangent_bypass_active_steps_per_episode",
                "mean_tangent_bypass_active_agent_steps_per_episode",
                "mean_tangent_bypass_activations_per_episode",
                "minimum_tangent_bypass_candidate_clearance_m",
                "slot_guidance_episode_rate",
                "mean_slot_guidance_active_steps_per_episode",
                "mean_slot_handoffs_per_episode",
                "terminal_slot_error_m",
            ]
        )

    evaluation_records = (
        [] if args_cli.evaluation_only or args_cli.c96_near_ring_reset_curriculum or args_cli.region_capture else None
    )
    evaluation_trace_records = (
        [] if args_cli.evaluation_only and args_cli.evaluation_trace else None
    )
    initial_eval = evaluate_in_place(
        env,
        agent,
        args_cli.eval_episodes,
        args_cli.seed + 50_000_000,
        args_cli.seed,
        episode_records=evaluation_records,
        episode_seed_plan=evaluation_seed_plan,
        trace_records=evaluation_trace_records,
    )
    evaluation_metric_names = tuple(initial_eval.keys())
    # Derive the CSV contract from the authoritative evaluator result.  The
    # metric set grows with new telemetry, so a duplicated static header can
    # silently shift every value into the wrong column.
    with eval_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["iteration", *evaluation_metric_names])
    observations = env.obs_dict
    state = env.state()
    with eval_csv.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([start_iteration, *initial_eval.values()])
    print(f"[P5-2] initial_eval={json.dumps(initial_eval, sort_keys=True)}", flush=True)
    if args_cli.expected_initial_evaluation is not None:
        expected_path = resolve_inside_repo(args_cli.expected_initial_evaluation)
        expected = json.loads(expected_path.read_text())["evaluation"]
        assert_expected_initial_evaluation(initial_eval, expected)
    if args_cli.region_capture and not args_cli.evaluation_only:
        if initial_eval.get("target_buffer_failure_rate", 0.) != 0.:
            raise RuntimeError("C108 initial target buffer safety gate failed; no PPO updates allowed")
        if any(initial_eval[key] != 0 for key in ("collision_rate", "contact_rate", "invalid_rate")):
            raise RuntimeError("C97 initial safety gate failed; no PPO updates allowed")
        with (output_dir / "baseline_evaluation_episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(evaluation_records[0]))
            writer.writeheader()
            writer.writerows(evaluation_records)
    if args_cli.c96_near_ring_reset_curriculum:
        expected = {
            "success_rate_training": 2 / 32, "success_rate_6m": 8 / 32,
            "success_rate_5m": 0.0, "collision_rate": 0.0,
            "contact_rate": 0.0, "invalid_rate": 0.0,
        }
        if any(initial_eval[key] != value for key, value in expected.items()):
            raise RuntimeError("C96 parent fixed32 baseline drift; no PPO update allowed")
        with (output_dir / "baseline_evaluation_episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(evaluation_records[0]))
            writer.writeheader()
            writer.writerows(evaluation_records)
        print("[C96] original-reset baseline reproduced; curriculum training unlocked", flush=True)

    if args_cli.evaluation_only:
        assert evaluation_records is not None
        episode_csv = output_dir / "evaluation_episodes.csv"
        with episode_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(evaluation_records[0].keys())
            )
            writer.writeheader()
            writer.writerows(evaluation_records)
        trace_csv = None
        if evaluation_trace_records:
            trace_csv = output_dir / "evaluation_trace.csv"
            with trace_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(evaluation_trace_records[0].keys())
                )
                writer.writeheader()
                writer.writerows(evaluation_trace_records)
        gate_passed = (
            initial_eval["success_rate_training"] >= args_cli.target_success
            and initial_eval["collision_rate"] <= args_cli.max_collision
            and initial_eval["contact_rate"] == 0.0
            and initial_eval["invalid_rate"] == 0.0
        )
        checkpoint_path = resolve_inside_repo(args_cli.resume)
        final_gpu = gpu_snapshot(args_cli.physical_gpu_index)
        summary = {
            "schema": "p5_2_native_checkpoint_evaluation_v2",
            "capture_protocol": REGION_PROTOCOL if args_cli.region_capture else "legacy",
            "region_objective": args_cli.region_objective,
            "auxiliary_metrics_censored_at_region_success": bool(args_cli.region_capture),
            "stage": args_cli.stage,
            "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "runtime_actor_output_reset": (
                args_cli.reset_actor_output_on_resume
            ),
            "evaluation_seed_base": args_cli.seed + 50_000_000,
            "evaluation_seed_plan": (
                str(evaluation_seed_plan_path.relative_to(REPO_ROOT))
                if evaluation_seed_plan_path
                else None
            ),
            "evaluation_seed_plan_sha256": (
                sha256_file(evaluation_seed_plan_path)
                if evaluation_seed_plan_path
                else None
            ),
            "evaluation_seed_rule": (
                "explicit_csv"
                if evaluation_seed_plan_path
                else "contiguous_eval_seed_plus_episode_index"
            ),
            "implicit_evaluation_seed_start": (
                None if evaluation_seed_plan_path else args_cli.seed + 50_000_000
            ),
            "implicit_evaluation_seed_stop_exclusive": (
                None
                if evaluation_seed_plan_path
                else args_cli.seed + 50_000_000 + args_cli.eval_episodes
            ),
            "difficulty_overrides": {
                "evader_speed_mps": args_cli.evaluation_evader_speed_mps,
                "dynamic_obstacle_speed_mps": (
                    args_cli.evaluation_dynamic_obstacle_speed_mps
                ),
                "sense_range_m": args_cli.evaluation_sense_range_m,
                "tracked_reset_reference_range_m": (
                    args_cli.evaluation_tracked_reset_reference_range_m
                ),
                "ignore_dynamic_obstacle_velocity": (
                    args_cli.evaluation_ignore_dynamic_obstacle_velocity
                ),
                "slot_prediction_horizon_s": (
                    args_cli.evaluation_slot_prediction_horizon_s
                ),
                "episode_length_s": args_cli.episode_length_s,
                "contraction_speed_mps": (
                    args_cli.evaluation_contraction_speed_mps
                ),
                "orientation_staging_radius_m": (
                    args_cli.evaluation_orientation_staging_radius_m
                ),
                "hybrid_residual_scale": (
                    args_cli.evaluation_hybrid_residual_scale
                ),
                "phase_gain": args_cli.evaluation_phase_gain,
                "phase_separation_control": (
                    args_cli.evaluation_phase_separation_control
                ),
                "latched_reference_contraction": (
                    args_cli.evaluation_latched_reference_contraction
                ),
                "maximum_radial_speed_mps": (
                    args_cli.evaluation_maximum_radial_speed_mps
                ),
                "phase_scheduled_radial_budget": (
                    args_cli.evaluation_phase_scheduled_radial_budget
                ),
                "maximum_tangential_speed_mps": (
                    args_cli.evaluation_maximum_tangential_speed_mps
                ),
                "blocker_phase_anchor_gain": (
                    args_cli.evaluation_blocker_phase_anchor_gain
                ),
                "reachable_role_slot_lead": (
                    args_cli.evaluation_reachable_role_slot_lead
                ),
                "far_slot_intercept_route": (
                    args_cli.evaluation_far_slot_intercept_route
                ),
                "lock_slot_assignment": (
                    args_cli.evaluation_lock_slot_assignment
                ),
            },
            "evaluation": initial_eval,
            "gate": {
                "target_success_rate": args_cli.target_success,
                "maximum_collision_rate": args_cli.max_collision,
                "require_zero_contact": True,
                "require_zero_invalid": True,
                "passed": gate_passed,
            },
            "decision": (
                "PASS_PREDICTIVE_SAFETY_GATE"
                if gate_passed
                and args_cli.stage in ("c1", "c2", "c3", "c4", "c5")
                else "PASS_G1_DEVELOPMENT_GATE"
                if gate_passed
                and (
                    args_cli.stage in (
                        "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30"
                    )
                    or args_cli.stage in ("c31", "c32", "c40", "c41", "c42", "c43", "c44", "c45", "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55", "c56", "c57", "c58", "c60", "c61", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79", "c80", "c81", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91")
                )
                else "PASS_C0_SAFETY_GATE"
                if gate_passed and args_cli.stage == "c0"
                else "PASS_TO_A1"
                if gate_passed
                else "HOLD_REPAIR_METHOD"
                if args_cli.stage in ("c20", "c21", "c22", "c23", "c24", "c25", "c26", "c27", "c29", "c30", "c31", "c32", "c40", "c41", "c42", "c43", "c44", "c45", "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55", "c56", "c57", "c58", "c60", "c61", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79", "c80", "c81", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91")
                else "HOLD_REPAIR_SAFETY"
            ),
            "artifacts": {
                "episodes_csv": str(episode_csv.relative_to(REPO_ROOT)),
                "aggregate_csv": str(eval_csv.relative_to(REPO_ROOT)),
                "trace_csv": (
                    str(trace_csv.relative_to(REPO_ROOT)) if trace_csv else None
                ),
            },
            "resource": {
                "physical_gpu_index": args_cli.physical_gpu_index,
                "initial_gpu": initial_gpu,
                "final_gpu": final_gpu,
                "wall_time_s": time.monotonic() - wall_start,
            },
            "provenance": {
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "trainer_sha256": sha256_file(Path(__file__)),
                "reward_source_sha256": sha256_file(
                    PACKAGE_SOURCE / "bylw_isaac_p5" / "core" / "native_training.py"
                ),
                "environment_source_sha256": sha256_file(
                    PACKAGE_SOURCE
                    / "bylw_isaac_p5"
                    / "tasks"
                    / "direct"
                    / "bylw_pursuit"
                    / "bylw_pursuit_train_env.py"
                ),
                "base_environment_source_sha256": sha256_file(
                    PACKAGE_SOURCE
                    / "bylw_isaac_p5"
                    / "tasks"
                    / "direct"
                    / "bylw_pursuit"
                    / "bylw_pursuit_env.py"
                ),
                "base_environment_cfg_sha256": sha256_file(
                    PACKAGE_SOURCE
                    / "bylw_isaac_p5"
                    / "tasks"
                    / "direct"
                    / "bylw_pursuit"
                    / "bylw_pursuit_env_cfg.py"
                ),
                "training_environment_cfg_sha256": sha256_file(
                    PACKAGE_SOURCE
                    / "bylw_isaac_p5"
                    / "tasks"
                    / "direct"
                    / "bylw_pursuit"
                    / "bylw_pursuit_train_env_cfg.py"
                ),
            },
        }
        summary_path = output_dir / "evaluation_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        env.close()
        return 0

    episode_return = torch.zeros(args_cli.num_envs, device=env.device)
    recent_returns: list[float] = []
    recent_success: list[float] = []
    best_success = initial_eval["success_rate_training"]
    best_fixed_success = initial_eval["success_rate_6m"]
    best_strict_success = initial_eval["success_rate_5m"]
    best_strong_ring_hold = initial_eval["maximum_strong_ring_hold_steps"]
    best_collision = initial_eval["collision_rate"]
    best_timeout = initial_eval["timeout_rate"]
    best_projection_infeasible = initial_eval[
        "safety_projection_infeasible_step_rate"
    ]
    best_projection_corrected = initial_eval[
        "safety_projection_corrected_step_rate"
    ]
    best_is_safe = (
        initial_eval["collision_rate"] <= args_cli.max_collision
        and initial_eval["contact_rate"] == 0.0
        and initial_eval["invalid_rate"] == 0.0
        and initial_eval.get("target_buffer_failure_rate", 0.) == 0.0
    )
    save_checkpoint(
        output_dir / "best.pt", agent, start_iteration, args_cli.stage, best_success
    )
    consecutive_passes = 0
    passed = False
    stop_reason = "iteration_budget"
    final_iteration = start_iteration
    peak_gpu_mb = initial_gpu["memory_used_mb"]
    evaluation_history = [{"iteration": start_iteration, **initial_eval}]
    policy_active_fractions: list[float] = []

    for local_iteration in range(1, args_cli.iterations + 1):
        iteration = start_iteration + local_iteration
        final_iteration = iteration
        if time.monotonic() - wall_start >= args_cli.max_wall_hours * 3600.0:
            stop_reason = "wall_time_budget"
            break
        buffer = RolloutBuffer(
            args_cli.rollout_steps,
            args_cli.num_envs,
            3,
            observation_dim,
            state_dim,
            3,
            agent.device,
        )
        learning_accumulator = [] if args_cli.learning_diagnostics else None
        for _ in range(args_cli.rollout_steps):
            observation_tensor = stack_observations(observations, agents)
            raw_action = None
            if args_cli.ppo_exact_logprob:
                action, log_probability, value, raw_action = agent.act(
                    observation_tensor, state, return_raw=True
                )
            else:
                action, log_probability, value = agent.act(observation_tensor, state)
            observations, rewards, terminated, truncated, extras = env.step(
                action_dictionary(action, agents)
            )
            reward = rewards[first]
            if args_cli.region_capture and not (
                torch.isfinite(observation_tensor).all() and torch.isfinite(state).all()
                and torch.isfinite(reward).all() and torch.isfinite(log_probability).all()
            ):
                raise RuntimeError("C97 non-finite rollout; stopping before any optimizer update")
            done = terminated[first] | truncated[first]
            episode_return += reward
            p5 = extras[first]["p5"]
            if learning_accumulator is not None:
                d = env._learning_diagnostics
                active = d['learning_diag_slot_active']
                def active_sum(value):
                    return float(value[active].sum()) if active.any() else 0.
                before = d['learning_diag_slot_effect_before_speed'].norm(dim=-1).mean(dim=-1)
                after = d['learning_diag_slot_effect_after_speed'].norm(dim=-1).mean(dim=-1)
                downstream = (d['learning_diag_final_velocity']-d['learning_diag_slot_velocity']).norm(dim=-1).mean(dim=-1)
                learning_accumulator.append(dict(active_count=int(active.sum()),before_speed_sum=active_sum(before),
                    after_speed_sum=active_sum(after),downstream_change_sum=active_sum(downstream),
                    completed_episodes=int(done.sum()),
                    target_buffer_failures=int(p5.get('target_buffer_failure', torch.zeros_like(done)).sum()),
                    target_buffer_failures_on_rl_steps=int((p5.get('target_buffer_failure', torch.zeros_like(done)) & p5['hybrid_rl_active']).sum())))
            for env_index in done.nonzero(as_tuple=False).flatten().tolist():
                recent_returns.append(float(episode_return[env_index].item()))
                recent_success.append(
                    float(p5["success_training"][env_index].item())
                )
            episode_return[done] = 0.0
            next_state = env.state()
            buffer.add(
                observation_tensor,
                state,
                action,
                log_probability,
                reward,
                done.float(),
                value,
                raw_action=raw_action,
                policy_mask=(
                    p5["hybrid_rl_active"].float()
                    if cfg.hybrid_phase_supervisor_enabled
                    else None
                ),
            )
            state = next_state
        with torch.no_grad():
            last_value = agent.critic(state).squeeze(-1)
        update = agent.update(buffer, last_value, gamma=0.99, lam=0.95)
        if args_cli.learning_diagnostics:
            count=sum(x['active_count'] for x in learning_accumulator)
            entry=dict(iteration=iteration,local_iteration=local_iteration,environment_steps_this_run=local_iteration*args_cli.rollout_steps*args_cli.num_envs,
                **update['diagnostics'],slot_active_environment_samples=count)
            for key in ('completed_episodes', 'target_buffer_failures', 'target_buffer_failures_on_rl_steps'):
                entry[key]=sum(x[key] for x in learning_accumulator)
            for key in ('before_speed_sum','after_speed_sum','downstream_change_sum'):
                entry[key.removesuffix('_sum')+'_mean_mps']=sum(x[key] for x in learning_accumulator)/count if count else None
            with (output_dir/'learning_diagnostics.jsonl').open('a') as handle:
                handle.write(json.dumps(entry,allow_nan=False)+'\n')
        policy_active_fractions.append(update["policy_active_fraction"])

        if local_iteration % 5 == 0 or local_iteration == 1:
            gpu = gpu_snapshot(args_cli.physical_gpu_index)
            peak_gpu_mb = max(peak_gpu_mb, gpu["memory_used_mb"])
            if gpu["memory_used_mb"] > args_cli.memory_limit_mb:
                stop_reason = "gpu_memory_limit"
                break
            rolling_return = float(np.mean(recent_returns[-100:])) if recent_returns else 0.0
            rolling_success = float(np.mean(recent_success[-100:])) if recent_success else 0.0
            with train_csv.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(
                    [
                        iteration,
                        iteration * args_cli.rollout_steps * args_cli.num_envs,
                        rolling_return,
                        rolling_success,
                        update["pi_loss"],
                        update["v_loss"],
                        update["entropy"],
                        update["policy_active_fraction"],
                        gpu["memory_used_mb"],
                        time.monotonic() - wall_start,
                    ]
                )
            print(
                f"[P5-2] stage={args_cli.stage} iter={iteration} "
                f"return={rolling_return:.3f} train_success={rolling_success:.3f} "
                f"pi={update['pi_loss']:.4f} value={update['v_loss']:.4f} "
                f"policy_active={update['policy_active_fraction']:.3f} "
                f"gpu_mb={gpu['memory_used_mb']:.0f}",
                flush=True,
            )

        if local_iteration % args_cli.checkpoint_every == 0:
            save_checkpoint(
                output_dir / "latest.pt", agent, iteration, args_cli.stage, best_success
            )

        if local_iteration % args_cli.eval_every == 0:
            c96_episode_records = [] if args_cli.c96_near_ring_reset_curriculum or args_cli.region_capture else None
            evaluation = evaluate_in_place(
                env,
                agent,
                args_cli.eval_episodes,
                args_cli.seed + 50_000_000,
                args_cli.seed,
                episode_seed_plan=evaluation_seed_plan,
                episode_records=c96_episode_records,
            )
            if c96_episode_records:
                with (output_dir / f"eval_episodes_iter{iteration}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(c96_episode_records[0]))
                    writer.writeheader()
                    writer.writerows(c96_episode_records)
            if tuple(evaluation.keys()) != evaluation_metric_names:
                raise RuntimeError(
                    "evaluation metric schema changed within one training run"
                )
            observations = env.obs_dict
            state = env.state()
            episode_return.zero_()
            evaluation_history.append({"iteration": iteration, **evaluation})
            with eval_csv.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow([iteration, *evaluation.values()])
            print(f"[P5-2] eval@{iteration}={json.dumps(evaluation, sort_keys=True)}", flush=True)
            score = (
                int(
                    evaluation["collision_rate"] <= args_cli.max_collision
                    and evaluation["contact_rate"] == 0.0
                    and evaluation["invalid_rate"] == 0.0
                    and evaluation.get("target_buffer_failure_rate", 0.) == 0.0
                ),
                evaluation["success_rate_training"],
                0 if args_cli.region_capture else evaluation["maximum_strong_ring_hold_steps"],
                evaluation["success_rate_6m"],
                evaluation["success_rate_5m"],
                -evaluation["timeout_rate"],
                -evaluation["collision_rate"],
                -evaluation["safety_projection_infeasible_step_rate"],
                -evaluation["safety_projection_corrected_step_rate"],
            )
            best_score = (
                int(best_is_safe),
                best_success,
                0 if args_cli.region_capture else best_strong_ring_hold,
                best_fixed_success,
                best_strict_success,
                -best_timeout,
                -best_collision,
                -best_projection_infeasible,
                -best_projection_corrected,
            )
            if score > best_score:
                best_success = evaluation["success_rate_training"]
                best_fixed_success = evaluation["success_rate_6m"]
                best_strict_success = evaluation["success_rate_5m"]
                best_strong_ring_hold = evaluation[
                    "maximum_strong_ring_hold_steps"
                ]
                best_collision = evaluation["collision_rate"]
                best_timeout = evaluation["timeout_rate"]
                best_projection_infeasible = evaluation[
                    "safety_projection_infeasible_step_rate"
                ]
                best_projection_corrected = evaluation[
                    "safety_projection_corrected_step_rate"
                ]
                best_is_safe = bool(score[0])
                save_checkpoint(
                    output_dir / "best.pt", agent, iteration, args_cli.stage, best_success
                )
            gate = (
                evaluation["success_rate_training"] >= args_cli.target_success
                and evaluation["collision_rate"] <= args_cli.max_collision
                and evaluation["invalid_rate"] == 0.0
                and evaluation["contact_rate"] == 0.0
                and evaluation.get("target_buffer_failure_rate", 0.) == 0.0
            )
            consecutive_passes = consecutive_passes + 1 if gate else 0
            if consecutive_passes >= args_cli.required_passes:
                passed = True
                stop_reason = "development_gate_passed"
                break

    save_checkpoint(
        output_dir / "final.pt", agent, final_iteration, args_cli.stage, best_success
    )
    if not (output_dir / "best.pt").exists():
        save_checkpoint(
            output_dir / "best.pt", agent, final_iteration, args_cli.stage, best_success
        )
    final_gpu = gpu_snapshot(args_cli.physical_gpu_index)
    peak_gpu_mb = max(peak_gpu_mb, final_gpu["memory_used_mb"])
    best_checkpoint_path = output_dir / "best.pt"
    final_checkpoint_path = output_dir / "final.pt"
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    role_phase_input_weight = (
        agent.actor.body[0].weight[:, NATIVE_OBSERVATION_DIM:]
        if args_cli.stage in ("c27", "c29", "c30", "c31", "c32", "c40", "c41", "c42", "c43", "c44", "c45", "c46", "c47", "c48", "c49", "c50", "c51", "c52", "c53", "c54", "c55", "c56", "c57", "c58", "c60", "c61", "c62", "c63", "c64", "c65", "c66", "c67", "c68", "c69", "c70", "c71", "c72", "c73", "c74", "c75", "c76", "c77", "c78", "c79", "c80", "c81", "c82", "c83", "c84", "c85", "c86", "c87", "c88", "c89", "c90", "c91")
        else None
    )
    summary = {
        "schema": "p5_2_native_mappo_summary_v2",
        "capture_protocol": REGION_PROTOCOL if args_cli.region_capture else "legacy",
        "region_objective": args_cli.region_objective,
        "ppo_logprob_schema": "pre_tanh_v1" if args_cli.ppo_exact_logprob else "legacy",
        "auxiliary_metrics_censored_at_region_success": bool(args_cli.region_capture),
        "stage": args_cli.stage,
        "passed": passed,
        "stop_reason": stop_reason,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "final_iteration": final_iteration,
        "training_capture_radius_m": cfg.training_capture_radius_m,
        "best_success_rate_training": best_success,
        "best_success_rate_6m": max(
            item["success_rate_6m"] for item in evaluation_history
        ),
        "best_orientation_unlock_rate": max(
            item["orientation_unlock_rate"] for item in evaluation_history
        ),
        "best_all_visible_step_fraction": max(
            item["mean_all_visible_step_fraction"] for item in evaluation_history
        ),
        "best_collision_rate": best_collision,
        "best_timeout_rate": best_timeout,
        "best_safety_projection_infeasible_step_rate": (
            best_projection_infeasible
        ),
        "best_safety_projection_corrected_step_rate": (
            best_projection_corrected
        ),
        "peak_gpu_memory_mb": peak_gpu_mb,
        "mean_policy_active_fraction": (
            float(np.mean(policy_active_fractions))
            if policy_active_fractions
            else 0.0
        ),
        "wall_time_s": time.monotonic() - wall_start,
        "curriculum_reset_attempts": env._curriculum_reset_attempts,
        "curriculum_reset_applied": env._curriculum_reset_applied,
        "completed_training_episodes": len(recent_success),
        "successful_training_episodes": sum(recent_success),
        "role_phase_input_weight_l2_norm": (
            float(role_phase_input_weight.detach().norm().item())
            if role_phase_input_weight is not None
            else None
        ),
        "role_phase_input_weight_nonzero_count": (
            int(torch.count_nonzero(role_phase_input_weight.detach()).item())
            if role_phase_input_weight is not None
            else None
        ),
        "actor_output_head_weight_l2_norm": float(
            agent.actor.mu.weight.detach().norm().item()
        ),
        "actor_output_head_weight_nonzero_count": int(
            torch.count_nonzero(agent.actor.mu.weight.detach()).item()
        ),
        "actor_output_head_bias_l2_norm": float(
            agent.actor.mu.bias.detach().norm().item()
        ),
        "evaluations": evaluation_history,
        "artifacts": {
            "best_checkpoint": str(best_checkpoint_path.relative_to(REPO_ROOT)),
            "best_checkpoint_sha256": sha256_file(best_checkpoint_path),
            "final_checkpoint": str(final_checkpoint_path.relative_to(REPO_ROOT)),
            "final_checkpoint_sha256": sha256_file(final_checkpoint_path),
            "train_log": str(train_csv.relative_to(REPO_ROOT)),
            "eval_log": str(eval_csv.relative_to(REPO_ROOT)),
        },
        "provenance": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_dirty": bool(git_status.strip()),
            "trainer_sha256": sha256_file(Path(__file__)),
            "reward_source_sha256": sha256_file(
                PACKAGE_SOURCE / "bylw_isaac_p5" / "core" / "native_training.py"
            ),
            "observation_source_sha256": sha256_file(
                PACKAGE_SOURCE / "bylw_isaac_p5" / "core" / "native_observations.py"
            ),
            "environment_source_sha256": sha256_file(
                PACKAGE_SOURCE
                / "bylw_isaac_p5"
                / "tasks"
                / "direct"
                / "bylw_pursuit"
                / "bylw_pursuit_train_env.py"
            ),
            "base_environment_source_sha256": sha256_file(
                PACKAGE_SOURCE
                / "bylw_isaac_p5"
                / "tasks"
                / "direct"
                / "bylw_pursuit"
                / "bylw_pursuit_env.py"
            ),
            "base_environment_cfg_sha256": sha256_file(
                PACKAGE_SOURCE
                / "bylw_isaac_p5"
                / "tasks"
                / "direct"
                / "bylw_pursuit"
                / "bylw_pursuit_env_cfg.py"
            ),
            "training_environment_cfg_sha256": sha256_file(
                PACKAGE_SOURCE
                / "bylw_isaac_p5"
                / "tasks"
                / "direct"
                / "bylw_pursuit"
                / "bylw_pursuit_train_env_cfg.py"
            ),
            "python": sys.version,
            "torch": torch.__version__,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    env.close()
    return 0 if stop_reason not in {"gpu_memory_limit"} else 2


def close_simulation_app(exit_code: int, timeout_s: float = 15.0) -> None:
    finished = threading.Event()

    def force_exit_after_timeout() -> None:
        if not finished.wait(timeout_s):
            print("[P5-2] Isaac shutdown timed out after flushing outputs.", flush=True)
            os._exit(exit_code)

    threading.Thread(target=force_exit_after_timeout, daemon=True).start()
    simulation_app.close(wait_for_replicator=False)
    finished.set()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        # Isaac shutdown can block long enough for the watchdog to use
        # os._exit.  Print the original Python failure before closing the app,
        # otherwise the watchdog hides the traceback and leaves only an
        # incomplete output directory.
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        close_simulation_app(exit_code)
    raise SystemExit(exit_code)
