"""Native Isaac 3-D same-seed comparison for four controller arms.

This runner evaluates only deterministic controller arms.  PPO training is
kept out of the final matrix because C161 showed no reproducible gain over the
zero-residual controller; learned policies can be evaluated separately after
the physical scenario matrix is complete.
"""

import argparse
import hashlib
import json
import os
import time
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
parser.add_argument("--n", type=int, default=16)
parser.add_argument("--seed-base", type=int, default=16200000)
parser.add_argument("--methods", nargs="+", default=["all_search_staged"])
parser.add_argument("--scenes", nargs="+", default=["corner", "corridor"])
parser.add_argument("--duration", type=float, default=60.0)
parser.add_argument("--scene-scale", type=float, default=1.6)
parser.add_argument("--vertical-mode", choices=("free", "fixed"), default="free")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from gymnasium.spaces import Discrete

from experiments.c150_native_search.task import Task, geometry3
from experiments.c167_vertical_ablation.common import snapshot, write_json
from experiments.c167_vertical_ablation.metrics import score
from experiments.c167_vertical_ablation.world import make_world
from experiments.c167_vertical_ablation.native_env import ResponseEnv
from bylw_isaac_p5.tasks.direct.bylw_pursuit.bylw_pursuit_env_cfg import BylwPursuitEnvCfg


METHODS = {"brake_predict", "brake_search", "staged_brake", "all_search_staged"}
SCENES = {"corner", "corridor"}


def rollout(env):
    world = env.pilot
    n = env.num_envs
    steps = round(world.task.duration * 10)
    keys = [
        "time",
        "p",
        "z",
        "target",
        "target_z",
        "geometry",
        "mask",
        "age",
        "searcher",
        "unsafe",
        "failures",
        "command",
        "altitude_command",
        "action",
        "estimate",
        "yaw",
        "velocity",
        "target_velocity",
        "route_done",
        "branch",
        "branch_reached",
        "branch_switch_tick",
        "branch_switch_hidden",
        "safety_minima",
        "peak_speed",
        "raw_command",
        "guard_delta",
        "phase_mix",
        "deployed_searcher",
        "stage_entries",
    ]
    trace = {key: [] for key in keys}
    action = np.zeros((n, 3), dtype=int)
    for tick in range(steps):
        actions = {
            agent: torch.as_tensor(action[:, i], device=env.device)
            for i, agent in enumerate(env.cfg.possible_agents)
        }
        with torch.inference_mode():
            env.step(actions)
        env.record_safety()
        obs = env.observe_now()
        geometry, _, _ = geometry3(world.p, world.z, world.target, world.tz)
        values = [
            world.time,
            world.p.copy(),
            world.z.copy(),
            world.target.copy(),
            world.tz.copy(),
            geometry,
            obs["mask"].copy(),
            obs["age"].copy(),
            obs["searcher"].copy(),
            env.failures.any(1).copy(),
            env.failures.copy(),
            env.last_command.copy(),
            env.last_altitude.copy(),
            action.copy(),
            obs["target"].copy(),
            world.yaw.copy(),
            world.pv.copy(),
            world.tv.copy(),
            world.route_done.copy(),
            world.branch.copy(),
            world.branch_reached.copy(),
            world.branch_switch_tick.copy(),
            world.branch_switch_hidden.copy(),
            env.minima.copy(),
            env.peak_speed.copy(),
            env.last_raw_command.copy(),
            env.last_guard_delta.copy(),
            env.last_phase_mix.copy(),
            env.last_deployed_searcher.copy(),
            env.last_stage_entries.copy(),
        ]
        for key, value in zip(keys, values):
            trace[key].append(value)
        if tick % 200 == 199:
            print("PROGRESS", tick + 1, flush=True)
    return {key: np.asarray(value) for key, value in trace.items()}


def main():
    if not args.methods or not set(args.methods) <= METHODS:
        raise ValueError(f"registered methods are {sorted(METHODS)}")
    if not args.scenes or not set(args.scenes) <= SCENES:
        raise ValueError(f"registered scenes are {sorted(SCENES)}")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        schema="c167_vertical_ablation_v1",
        n_per_method=args.n,
        methods=args.methods,
        scenes=args.scenes,
        seed_base=args.seed_base,
        duration=args.duration,
        scene_scale=args.scene_scale,
        vertical_mode=args.vertical_mode,
        physics_hz=100,
        control_hz=10,
        policy_hz=5,
        success="radii 4..8 m, spread <=2 m, gaps 90..150 deg, dz <=1 m, visible hold 5 s, whole-episode safe",
        sensing="natural 20 m, horizontal FOV 90 deg, vertical FOV 60 deg, cylinder LOS, shared noisy observations",
        loss_definition="passive sensor event: team mask is false because range/FOV/occlusion removes all legal measurements",
        corner_branch="C154 common waypoint [0,4] followed by two hidden continuations",
        corridor_branch="same hidden branch with a flank cylinder at local [5.5,0]; initial visibility and clearance checked",
        controller_note="same-seed comparison of brake_predict, brake_search, staged_brake and all_search_staged",
    )
    snapshot(args.out, metadata)

    torch.set_num_threads(1)
    cfg = BylwPursuitEnvCfg()
    cfg.scene.num_envs = args.n * len(args.methods)
    cfg.sim.device = args.device
    cfg.seed = args.seed_base
    cfg.episode_length_s = 3600
    cfg.observation_spaces = {agent: 34 for agent in cfg.possible_agents}
    cfg.state_space = 102
    cfg.action_spaces = {agent: Discrete(8) for agent in cfg.possible_agents}
    cfg.obstacle_specs = tuple(
        (x, y, radius * args.scene_scale)
        for x, y, radius in [(-20.0, -20.0, 3.0), (-20.0, 20.0, 1.2), (20.0, -20.0, 1.2), (20.0, 20.0, 1.2)]
    )
    cfg.obstacle_shape_kinds = ("cylinder",) * 4
    cfg.obstacle_box_sizes_xy_m = ((0.0, 0.0),) * 4
    cfg.randomize_obstacles_on_reset = True
    cfg.evader_speed_mps = 3.2
    cfg.evader_max_planar_accel_mps2 = 4.0
    cfg.pursuer_max_planar_accel_mps2 = 4.0
    cfg.reset_layout = "tracked_ring"
    cfg.tracked_target_center_xy_m = (0.0, 0.0)
    cfg.tracked_target_spread_m = 5.0
    env = ResponseEnv(cfg)
    env.vertical_mode = args.vertical_mode
    start = time.monotonic()
    env.action_rng = torch.Generator(device="cpu").manual_seed(args.seed_base + 100000000)
    rows = []
    try:
        for scene_index, scene in enumerate(args.scenes):
            seeds = np.arange(args.seed_base + scene_index * args.n, args.seed_base + (scene_index + 1) * args.n)
            all_seeds = np.tile(seeds, len(args.methods))

            def factory(ss, scene=scene):
                task = Task(scene=scene, duration=args.duration)
                world = make_world(scene, ss, task, args.scene_scale)
                for method_index, method in enumerate(args.methods):
                    sl = slice(method_index * args.n, (method_index + 1) * args.n)
                    if method == "all_search_staged":
                        world.role_modes[sl] = 2
                        world.braking[sl] = True
                        world.staged[sl] = True
                    else:
                        world.configure(sl, method)
                return world

            env.world_factory = factory
            with torch.inference_mode():
                env.initialize_case(all_seeds)
            trace = rollout(env)
            for method_index, method in enumerate(args.methods):
                sl = slice(method_index * args.n, (method_index + 1) * args.n)
                data = {key: value if key == "time" else value[:, sl] for key, value in trace.items()}
                episodes, stats = score(data, seeds)
                data_branch_hidden = float(data["branch_switch_hidden"][-1].mean())
                row = dict(
                    scene=scene,
                    method=method,
                    branch_switch_hidden_fraction=data_branch_hidden,
                    **stats,
                )
                stem = f"{scene}__{method}"
                np.savez_compressed(
                    args.out / f"{stem}.npz",
                    seeds=seeds,
                    obstacles=env.pilot.obstacles[sl],
                    obstacle_radii=env.pilot.radii[sl],
                    **data,
                )
                write_json(
                    args.out / f"{stem}.json",
                    dict(summary=stats, episodes=episodes, branch_switch_hidden_fraction=data_branch_hidden),
                )
                rows.append(row)
                print("RESULT", json.dumps(row, ensure_ascii=False), flush=True)
            write_json(args.out / "summary.json", rows)
        write_json(
            args.out / "completion.json",
            dict(
                scenes=args.scenes,
                methods=args.methods,
                episodes=args.n * len(args.scenes) * len(args.methods),
                env_steps=round(args.duration * 10) * args.n * len(args.scenes) * len(args.methods),
                wall_seconds=time.monotonic() - start,
            ),
        )
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback

    error = traceback.format_exc()
    print(error, flush=True)
    if args.out.exists():
        (args.out / "failure.txt").write_text(error)
    raise
finally:
    app.close()
