"""Random obstacle maps for the Chapter-5 4-to-3 fault protocol."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments.c150_native_search.task import Task, visibility3
from experiments.c175_multi_fault_reconfig.world import CornerWorld as _CornerWorld
from experiments.c179_random_map_generalization.world import (
    RandomMapWorld,
    capture_ring_feasible,
    obstacle_shape_profile,
)


class RandomVariableCornerWorld(_CornerWorld):
    """C175 active-member semantics with C179 held-out map sampling."""

    scenario = "random"

    def __init__(
        self,
        seeds,
        task=None,
        scale=1.6,
        n_pursuers=4,
        split="train",
        n_obstacles=4,
        shape_profile="cylinder",
        layout_profile="distributed",
        initial_ring_screen=False,
        recovery="none",
    ):
        if int(n_obstacles) not in (4, 6, 8):
            raise ValueError("registered random-map obstacle counts are 4, 6, and 8")
        obstacle_shape_profile(n_obstacles, shape_profile)
        self.n_obstacles = int(n_obstacles)
        self.shape_profile = str(shape_profile)
        if layout_profile not in ("distributed", "route_dense"):
            raise ValueError("registered layout profiles are distributed and route_dense")
        self.layout_profile = str(layout_profile)
        self.initial_ring_screen = bool(initial_ring_screen)
        if recovery not in ("none", "routed", "routed_chase", "refined_fixed", "routed_precision", "routed_transit"):
            raise ValueError(recovery)
        self.recovery = recovery
        self.recovery_planner = None
        if task is None:
            task = Task(scene="corner")
        internal = replace(task, scene="corner")
        super().__init__(seeds, internal, scale, n_pursuers=n_pursuers)
        if self.obstacles.shape[1] != self.n_obstacles:
            self.obstacles = np.empty((len(seeds), self.n_obstacles, 2), dtype=float)
            self.radii = np.empty((len(seeds), self.n_obstacles), dtype=float)
        self.layout_split = str(split)
        self.map_rejections = np.zeros(len(seeds), dtype=int)
        self.ring_rejections = np.zeros(len(seeds), dtype=int)
        self.initial_ring_sample_found = np.zeros(len(seeds), dtype=bool)
        split_code = 0 if self.layout_split == "train" else 1
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 179, split_code]))
            theta = self._scene_rotation(i)
            for _attempt in range(200):
                local, local_radii = RandomMapWorld._sample_layout(
                    rng, self.n_obstacles, self.shape_profile, self.layout_profile
                )
                obstacles = self.target[i] + (local @ theta.T) * scale
                radii = local_radii * scale
                ring_found = capture_ring_feasible(obstacles, radii, self.target[i])
                if self.initial_ring_screen and not ring_found:
                    self.ring_rejections[i] += 1
                    continue
                mask = visibility3(
                    self.p[i : i + 1], self.z[i : i + 1], self.target[i : i + 1],
                    self.tz[i : i + 1], self.yaw[i : i + 1], obstacles[None],
                    radii[None], self.cfg,
                )
                target_clearance = (np.linalg.norm(self.target[i] - obstacles, axis=-1) - radii).min()
                pursuer_clearance = (
                    np.linalg.norm(self.p[i, :, None] - obstacles[None], axis=-1) - radii[None]
                ).min()
                if mask.any() and target_clearance > 0.5 and pursuer_clearance > 0.5:
                    self.obstacles[i] = obstacles
                    self.radii[i] = radii
                    self.map_rejections[i] = _attempt
                    self.initial_ring_sample_found[i] = ring_found
                    break
            else:
                raise RuntimeError(f"random variable map sampler exhausted for seed {seed}")
        self._validate_initial_layout()

    def commands(self, o, actions=None):
        baseline, yaw, altitude = super().commands(o, actions)
        if self.recovery == "none":
            return baseline, yaw, altitude
        from experiments.c184_route_recovery.planner import RecoveryPlanner
        from experiments.c145_feasibility_reset.planar import limit
        from experiments.c151_response_guard.guard import response_guard
        if self.recovery_planner is None:
            planner_class = RecoveryPlanner
            if self.recovery == "routed_chase":
                from experiments.c185_chase_recovery.planner import ChaseRecoveryPlanner
                planner_class = ChaseRecoveryPlanner
            elif self.recovery in ("refined_fixed", "routed_precision"):
                from experiments.c186_precision_recovery.planner import PrecisionRecoveryPlanner
                planner_class = PrecisionRecoveryPlanner
            elif self.recovery == "routed_transit":
                from experiments.c187_transit_recovery.planner import TransitRecoveryPlanner
                planner_class = TransitRecoveryPlanner
            self.recovery_planner = planner_class(len(self.seeds), self.n_pursuers)
        proposal, changed = self.recovery_planner.propose(o, baseline, self.time)
        if not changed.any():
            return baseline, yaw, altitude
        view, obs, ids = self._compact(o)
        raw = view.safety_filter(limit(proposal[:, ids], self.cfg.pursuer_speed), obs)
        if self.recovery in ("routed_precision", "routed_transit") and self.recovery_planner.precision_changed.any():
            from experiments.c186_precision_recovery.projection import precision_filter
            selected = self.recovery_planner.precision_changed
            precise_raw = precision_filter(view, limit(proposal[:, ids], self.cfg.pursuer_speed), obs)
            raw[selected] = precise_raw[selected]
        guarded_obs = obs
        if view.departed_p.shape[1]:
            guarded_obs = dict(obs,
                obstacles=np.concatenate([obs['obstacles'], view.departed_p], axis=1),
                radii=np.concatenate([obs['radii'], np.full(view.departed_p.shape[:2], .6)], axis=1))
        guarded = response_guard(raw, guarded_obs, self.cfg.pursuer_speed, target_buffer=3.)
        for local, agent in enumerate(ids):
            baseline[changed, agent] = guarded[changed, local]
            self.raw_command[changed, agent] = raw[changed, local]
        self.guard_delta = baseline - self.raw_command
        return baseline, yaw, altitude

    def _validate_initial_layout(self):
        mask = visibility3(self.p, self.z, self.target, self.tz, self.yaw, self.obstacles, self.radii, self.cfg)
        target_clearance = (np.linalg.norm(self.target[:, None] - self.obstacles, axis=-1) - self.radii).min()
        pursuer_clearance = (
            np.linalg.norm(self.p[:, :, None] - self.obstacles[:, None], axis=-1) - self.radii[:, None]
        ).min()
        if not mask.any(1).all():
            raise ValueError("random variable map lacks an initial legal team observation")
        if target_clearance <= 0.5 or pursuer_clearance <= 0.5:
            raise ValueError("random variable map violates initial clearance")

    def branch_metadata(self):
        meta = super().branch_metadata()
        meta.update(
            scenario=f"random_{self.layout_split}",
            obstacle_count=self.n_obstacles,
            shape_profile=self.shape_profile,
            layout_profile=self.layout_profile,
            initial_ring_screen=self.initial_ring_screen,
            initial_ring_sample_found_fraction=float(self.initial_ring_sample_found.mean()),
            ring_rejections_mean=float(self.ring_rejections.mean()),
            ring_check_scope="sampled regular triangle at initial target position only; no reachability or future-position guarantee",
            map_rejections_mean=float(self.map_rejections.mean()),
            obstacle_layout=(
                "one_approach_blocker_plus_three_seeded_distractors"
                if self.n_obstacles == 4
                else "one_approach_blocker_plus_one_continuation_blocker_plus_seeded_distractors"
            ),
            map_split=self.layout_split,
            passive_visibility_trigger=True,
            recovery=self.recovery,
            recovery_entries=(self.recovery_planner.entries.tolist() if self.recovery_planner else []),
            recovery_plan_calls=(self.recovery_planner.plan_calls.tolist() if self.recovery_planner else []),
            recovery_no_plan=(self.recovery_planner.no_plan.tolist() if self.recovery_planner else []),
            chase_entries=(self.recovery_planner.chase_entries.tolist()
                           if hasattr(self.recovery_planner, 'chase_entries') else []),
            precision_entries=(self.recovery_planner.precision_entries.tolist()
                               if hasattr(self.recovery_planner, 'precision_entries') else []),
            precision_plan_calls=(self.recovery_planner.precision_calls.tolist()
                                  if hasattr(self.recovery_planner, 'precision_calls') else []),
            precision_no_plan=(self.recovery_planner.precision_no_plan.tolist()
                               if hasattr(self.recovery_planner, 'precision_no_plan') else []),
            adaptive_obstacle_reserve=self.recovery in ("routed_precision", "routed_transit"),
            separate_transit_reserve=self.recovery == "routed_transit",
        )
        return meta


def make_world(
    scene,
    seeds,
    task=None,
    scale=1.6,
    n_pursuers=4,
    method="active_polar",
    n_obstacles=4,
    shape_profile="cylinder",
    layout_profile="distributed",
    initial_ring_screen=False,
    recovery="none",
):
    if scene not in ("random_train", "random_test"):
        raise ValueError(f"unregistered C180 scene: {scene}")
    task = Task(scene="corner") if task is None else replace(task, scene="corner")
    split = "train" if scene == "random_train" else "test"
    world = RandomVariableCornerWorld(
        seeds,
        task,
        scale,
        n_pursuers=n_pursuers,
        split=split,
        n_obstacles=n_obstacles,
        shape_profile=shape_profile,
        layout_profile=layout_profile,
        initial_ring_screen=initial_ring_screen,
        recovery=recovery,
    )
    world.configure(slice(None), "staged_brake")
    # C180 evaluates the proven active-member baseline first; the fault and
    # active-slot semantics are identical to C175.
    world.method = method
    return world
