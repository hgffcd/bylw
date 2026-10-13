"""C222 frozen-controller evaluation; unchanged C218 telemetry, with per-tick cumulative safety minima.

This runner is intentionally separate from C166--C170.  It uses the same
quadrotor assets, low-level controller and safety recorder, while the pilot
world and geometry judge expose an activity mask for 3/4/5 pursuers.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac_p5/source/bylw_isaac_p5"))
os.environ["BYLW_P5_ENABLE_ISAAC_RUNTIME"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--agents", type=int, choices=[3, 4, 5], required=True)
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--seed-base", type=int, default=17150000)
parser.add_argument("--scene", choices=["random_train", "random_test"], default="random_train")
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--scene-scale", type=float, default=1.6)
parser.add_argument("--obstacle-count", type=int, choices=[4, 6, 8], default=4)
parser.add_argument("--shape-profile", choices=["cylinder", "mixed"], default="cylinder")
parser.add_argument("--layout-profile", choices=["distributed", "route_dense"], default="distributed")
parser.add_argument("--failure-at", type=float, default=-1.0)
parser.add_argument("--failure-index", type=int, default=0)
parser.add_argument("--failure-times", type=str, default="")
parser.add_argument("--failure-indices", type=str, default="")
parser.add_argument("--reconfig", choices=["active_polar", "retained_slots", "dynamic", "greedy", "no_feedforward", "no_response_guard", "gated_dynamic", "hysteretic_dynamic", "predictive_hysteretic"], default="hysteretic_dynamic")
parser.add_argument("--reconfig-gain", type=float, default=0.8)
parser.add_argument("--reconfig-duration", type=float, default=1.0)
parser.add_argument("--gpu", type=int, default=2)
parser.add_argument("--initial-ring-screen", action="store_true")
parser.add_argument("--recovery", choices=["sector_bypass", "coverage_metric_sector", "radial_damped_sector"], default="none")
parser.add_argument("--map-protocol", choices=["legacy", "valid"], default="valid")
parser.add_argument("--opponent", choices=["local", "routed"], default="routed")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from gymnasium.spaces import Discrete

from experiments.c150_native_search.task import Task
from experiments.c169_comm_recovery.native_env import ResponseEnv
from experiments.c171_variable_scale.geometry import active_geometry
from experiments.c210_radial_damping.world import make_world
from experiments.c179_random_map_generalization.world import obstacle_shape_profile
from experiments.c175_multi_fault_reconfig.world import METHODS
from experiments.c180_random_map_variable.metrics import post_fault_metrics
from experiments.c180_random_map_variable.native_env import AuditEnv
from bylw_isaac_p5.tasks.direct.bylw_pursuit.bylw_pursuit_env_cfg import BylwPursuitEnvCfg


def rollout(env):
    world = env.pilot
    n = env.num_envs
    m = args.agents
    steps = round(args.duration * 10)
    if args.failure_times:
        failure_times = [float(x) for x in args.failure_times.split(',') if x.strip()]
        failure_indices = [int(x) for x in args.failure_indices.split(',') if x.strip()]
    elif args.failure_at >= 0:
        failure_times = [float(args.failure_at)]
        failure_indices = [int(args.failure_index)]
    else:
        failure_times, failure_indices = [], []
    if len(failure_times) != len(failure_indices):
        raise ValueError('failure times and indices must have equal length')
    schedule = sorted((round(t * 10), i) for t, i in zip(failure_times, failure_indices))
    failure_tick = schedule[-1][0] if schedule else 0
    active_history = []
    geometry_history = []
    radius_history = []
    gap_history = []
    p_history = []
    belief_history = []
    measured_history = []
    role_history = []
    age_history = []
    failure_history = []
    safety_minima_history = []
    target_history = []
    route_index_history = []
    route_done_history = []
    evader_command_history = []
    command_history = []
    recovery_history = []
    joint_history = []
    joint_applied_history = []
    blend_history = []
    bypass_history = []
    planner_inputs=[]
    target_height_history=[]
    pursuer_height_history=[]
    chase_history = []
    precision_applied_history = []
    precision_center_history = []
    failure_reached = np.zeros((n, len(schedule)), dtype=bool)
    first_failure_tick = np.full(n, -1, dtype=int)
    first_failure_age = np.full(n, np.nan, dtype=float)
    geometry_run = np.zeros(n, dtype=int)
    post_failure_capture5 = np.zeros(n, dtype=bool)
    action = np.zeros((n, m), dtype=int)
    next_fault = 0
    for tick in range(steps):
        while next_fault < len(schedule) and tick == schedule[next_fault][0]:
            _, failed_index = schedule[next_fault]
            if not world.active[:, failed_index].all():
                raise ValueError(f'failure index {failed_index} is already inactive')
            world.withdraw(failed_index, args.reconfig_gain, args.reconfig_duration)
            geometry_run[:] = 0
            failure_reached[:, next_fault] = True
            next_fault += 1
        actions = {
            agent: torch.as_tensor(action[:, index], device=env.device)
            for index, agent in enumerate(env.cfg.possible_agents)
        }
        with torch.inference_mode():
            env.step(actions)
        env.record_safety()
        safety_minima_history.append(env.minima.copy())
        env.observe_now()
        target_height_history.append(world.tz.copy())
        pursuer_height_history.append(world.z.copy())
        newly_failed = env.failures.any(1) & (first_failure_tick < 0)
        first_failure_tick[newly_failed] = tick
        first_failure_age[newly_failed] = world.age[newly_failed]
        valid, radii, gaps, count, _ = active_geometry(
            world.p, world.z, world.target, world.tz, world.active
        )
        valid &= count >= 3
        active_history.append(world.active.copy())
        geometry_history.append(valid.copy())
        radius_history.append(
            np.stack(
                [np.nanmin(np.where(world.active, radii, np.nan), axis=1),
                 np.nanmax(np.where(world.active, radii, np.nan), axis=1)],
                axis=1,
            )
        )
        gap_history.append(np.nanmax(gaps, axis=1))
        p_history.append(world.p.copy())
        belief_history.append(world.estimate.copy())
        measured_history.append(world.measured.copy())
        role_history.append(env.last_deployed_searcher.copy())
        age_history.append(world.age.copy())
        failure_history.append(env.failures.copy())
        target_history.append(world.target.copy())
        route_index_history.append(world.route_index.copy())
        route_done_history.append(world.route_done.copy())
        evader_command_history.append(world.last_evader_command.copy())
        command_history.append(env.last_command.copy())
        planner = world.recovery_planner
        planner_inputs.append({k: np.asarray(v).copy() for k,v in planner.audit_last.items()})
        bypass_history.append(planner.bypass_applied.copy() if hasattr(planner, 'bypass_applied') else np.zeros((n,m), bool))
        joint_history.append(planner.joint_on.copy() if hasattr(planner, 'joint_on') else np.zeros(n, bool))
        joint_applied_history.append(planner.joint_changed.copy() if hasattr(planner, 'joint_changed') else np.zeros(n, bool))
        blend_history.append(planner.blend_changed.copy() if hasattr(planner, 'blend_changed') else np.zeros(n, bool))
        recovery_history.append(planner.on.copy() if planner is not None else np.zeros(n, bool))
        chase_history.append(planner.chase_on.copy() if hasattr(planner, 'chase_on') else np.zeros((n,m), bool))
        precision_applied_history.append(planner.precision_changed.copy() if hasattr(planner, 'precision_changed') else np.zeros(n, bool))
        precision_center_history.append(planner.precision_center.copy() if hasattr(planner, 'precision_center') else np.full((n,2),np.nan))
        geometry_run = np.where(valid, geometry_run + 1, 0)
        if failure_tick >= 0 and tick >= failure_tick:
            post_failure_capture5 |= geometry_run >= 50
    geometry_history = np.asarray(geometry_history)
    safe = ~env.failures.any(1)
    metrics = post_fault_metrics(geometry_history, ~safe, failure_tick)
    return {
        **metrics,
        "target_z_history": np.asarray(target_height_history),
        "pursuer_z_history": np.asarray(pursuer_height_history),
        "safety_minima_history": np.asarray(safety_minima_history),
        "planner_inputs": {"input_"+k:np.asarray([row[k] for row in planner_inputs]) for k in planner_inputs[0]},
        "safe_episode": safe.astype(bool),
        "unsafe": (~safe).astype(bool),
            "geometry_fraction": geometry_history.mean(0),
            "geometry_history": geometry_history,
            "radius_history": np.asarray(radius_history),
            "gap_history": np.asarray(gap_history),
        "active_count_final": world.active.sum(1),
        "failure_reached": failure_reached,
        "first_failure_tick": first_failure_tick,
        "first_failure_age": first_failure_age,
        "active_history": np.asarray(active_history),
        "p_history": np.asarray(p_history),
        "belief_history": np.asarray(belief_history),
        "measured_history": np.asarray(measured_history),
        "role_history": np.asarray(role_history),
        "age_history": np.asarray(age_history),
        "failure_history": np.asarray(failure_history),
        "target_history": np.asarray(target_history),
        "route_index_history": np.asarray(route_index_history),
        "route_done_history": np.asarray(route_done_history),
        "evader_command_history": np.asarray(evader_command_history),
        "command_history": np.asarray(command_history),
        "recovery_history": np.asarray(recovery_history),
        "joint_history": np.asarray(joint_history),
        "joint_applied_history": np.asarray(joint_applied_history),
        "blend_history": np.asarray(blend_history),
        "bypass_history": np.asarray(bypass_history),
        "chase_history": np.asarray(chase_history),
        "precision_applied_history": np.asarray(precision_applied_history),
        "precision_center_history": np.asarray(precision_center_history),
    }


def main():
    if args.failure_times:
        times = [float(x) for x in args.failure_times.split(',') if x.strip()]
        indices = [int(x) for x in args.failure_indices.split(',') if x.strip()]
    elif args.failure_at >= 0:
        times, indices = [args.failure_at], [args.failure_index]
    else:
        times, indices = [], []
    if len(times) != len(indices):
        raise ValueError("failure times and indices must have equal length")
    if any(t < 0 or t > args.duration - 5 for t in times):
        raise ValueError("each failure must allow at least 5 post-fault seconds")
    if len(indices) >= args.agents - 2:
        raise ValueError("at least three active pursuers are required after all failures")
    if any(i < 0 or i >= args.agents for i in indices):
        raise ValueError("failure index is outside the configured team")
    if len(set(indices)) != len(indices):
        raise ValueError("failure indices must be distinct")
    if list(times) != sorted(times):
        raise ValueError("failure times must be nondecreasing")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(args.out)
    obstacle_shape_profile(args.obstacle_count, args.shape_profile)
    args.out.mkdir(parents=True, exist_ok=True)
    agents = [f"pursuer_{index}" for index in range(args.agents)]
    cfg = BylwPursuitEnvCfg()
    cfg.n_pursuers = args.agents
    cfg.min_capture_agents = 3
    cfg.allow_variable_scale = True
    cfg.possible_agents = agents
    cfg.action_spaces = {agent: 8 for agent in agents}
    # The C150 pilot feature adapter intentionally keeps its 34-D local actor
    # contract; the scale-aware geometry is evaluated outside the actor.
    cfg.observation_spaces = {agent: 34 for agent in agents}
    cfg.state_space = 34 * args.agents
    cfg.scene.num_envs = args.n
    cfg.sim.device = args.device
    cfg.seed = args.seed_base
    cfg.episode_length_s = 3600
    cfg.reset_layout = "tracked_ring"
    cfg.tracked_target_center_xy_m = (0.0, 0.0)
    cfg.tracked_target_spread_m = 5.0
    shape_kinds, local_radii, local_box_sizes = obstacle_shape_profile(
        args.obstacle_count, args.shape_profile
    )
    cfg.obstacle_specs = tuple(
        (x, y, float(radius) * args.scene_scale)
        for (x, y), radius in zip(
            [
                (-20.0, -20.0),
                *[
                    (float(10 + 3 * i), float(-20 + 4 * i))
                    for i in range(args.obstacle_count - 1)
                ],
            ],
            local_radii,
        )
    )
    cfg.obstacle_shape_kinds = shape_kinds
    cfg.obstacle_box_sizes_xy_m = tuple(
        (float(size_x) * args.scene_scale, float(size_y) * args.scene_scale)
        for size_x, size_y in local_box_sizes
    )
    cfg.randomize_obstacles_on_reset = True
    cfg.evader_speed_mps = 3.2
    cfg.evader_max_planar_accel_mps2 = 4.0
    cfg.pursuer_max_planar_accel_mps2 = 4.0
    env = AuditEnv(cfg)
    seeds = np.arange(args.seed_base, args.seed_base + args.n)

    def factory(ss):
        return make_world(
            args.scene,
            ss,
            Task(scene="corner", duration=args.duration),
            args.scene_scale,
            args.agents, method=args.reconfig, n_obstacles=args.obstacle_count,
            shape_profile=args.shape_profile,
            layout_profile=args.layout_profile,
            initial_ring_screen=args.initial_ring_screen,
            recovery=args.recovery, map_protocol=args.map_protocol, opponent=args.opponent,
        )

    env.world_factory = factory
    try:
        with torch.inference_mode():
            env.initialize_case(seeds)
        result = rollout(env)
        effective_duration = None if args.reconfig == "active_polar" else (.1 if args.reconfig == "greedy" else args.reconfig_duration)
        summary = {
            "schema": "c175_multi_fault_native_v1",
            "scene": args.scene,
            "agents": args.agents,
            "obstacle_count": args.obstacle_count,
            "shape_profile": args.shape_profile,
            "layout_profile": args.layout_profile,
            "initial_ring_screen": args.initial_ring_screen,
            "recovery": args.recovery,
            "map_protocol": args.map_protocol, "opponent": args.opponent,
            "failure_at": args.failure_at,
            "failure_index": args.failure_index,
            "failure_times": times,
            "failure_indices": indices,
            "reconfig": args.reconfig,
            "reconfig_gain": args.reconfig_gain,
            "reconfig_duration": effective_duration,
            "n": args.n,
            "seed_base": args.seed_base,
            "seeds": seeds.tolist(),
            "evaluation_only": True,
            "source_manifest": str(args.out.parent / (args.out.name + "_source") / "manifest.json"),
            "post_geometry_fraction_mean": float(result["post_geometry_fraction"].mean()),
            "max_post_hold_s_mean": float(result["max_post_hold_s"].mean()),
            "branch_metadata": env.pilot.branch_metadata(),
            "duration": args.duration,
            "safe_capture5": int(result["safe_capture5"].sum()),
            "safe_episode": int(result["safe_episode"].sum()),
            "unsafe": int(result["unsafe"].sum()),
            "safety_failure_counts": {
                name: int(env.failures[:, index].sum())
                for index, name in enumerate(
                    ["target", "pair", "obstacle", "target_obstacle", "height", "boundary", "finite"]
                )
            },
            "geometry_fraction_mean": float(result["geometry_fraction"].mean()),
            "geometry_fraction_per_env": result["geometry_fraction"].tolist(),
            "radius_min_final": float(result["radius_history"][-1, :, 0].mean()),
            "radius_max_final": float(result["radius_history"][-1, :, 1].mean()),
            "gap_max_final": float(result["gap_history"][-1].mean()),
            "active_count_final": result["active_count_final"].tolist(),
            "fault_semantics": ("sequential task withdrawal with live hover controller; active-only sensing/control/threat; physical retired bodies remain in pair-safety scoring" if times else "fixed three active UAVs throughout; no withdrawal; score from episode start"),
        }
        planner=env.pilot.recovery_planner
        if hasattr(planner, 'bypass_entries'):
            summary['branch_metadata'].update(bypass_entries=planner.bypass_entries.tolist(), bypass_calls=planner.bypass_calls.tolist(), bypass_no_plan=planner.bypass_no_plan.tolist())
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(
            args.out / "trace.npz",
            **result["planner_inputs"],
            target_z_history=result["target_z_history"],
            pursuer_z_history=result["pursuer_z_history"],
            safety_minima_history=result["safety_minima_history"],
            seeds=seeds,
            obstacles=env.pilot.obstacles,
            obstacle_radii=env.pilot.radii,
            map_rejections=env.pilot.map_rejections,
            ring_rejections=env.pilot.ring_rejections,
            initial_ring_sample_found=env.pilot.initial_ring_sample_found,
            post_geometry_fraction=result["post_geometry_fraction"],
            max_post_hold_s=result["max_post_hold_s"],
            post_capture_delay_s=result["post_capture_delay_s"],
            belief_history=result["belief_history"],
            measured_history=result["measured_history"],
            role_history=result["role_history"],
            age_history=result["age_history"],
            failure_history=result["failure_history"],
            geometry=result["geometry_fraction"],
            geometry_history=result["geometry_history"],
            radius_history=result["radius_history"],
            gap_history=result["gap_history"],
            active_history=result["active_history"],
            safe_capture5=result["safe_capture5"],
            unsafe=result["unsafe"],
            first_failure_tick=result["first_failure_tick"],
            first_failure_age=result["first_failure_age"],
            safety_failure_flags=env.failures,
            safety_minima=env.minima,
            p_history=result["p_history"],
            target_history=result["target_history"],
            route_index_history=result["route_index_history"],
            route_done_history=result["route_done_history"],
            evader_command_history=result["evader_command_history"],
            command_history=result["command_history"],
            recovery_history=result["recovery_history"],
            joint_history=result["joint_history"],
            joint_applied_history=result["joint_applied_history"],
            blend_history=result["blend_history"],
            bypass_history=result["bypass_history"],
            chase_history=result["chase_history"],
            precision_applied_history=result["precision_applied_history"],
            precision_center_history=result["precision_center_history"],
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    if args.out.exists():
        (args.out / "failure.txt").write_text(traceback.format_exc())
    raise
finally:
    app.close()
