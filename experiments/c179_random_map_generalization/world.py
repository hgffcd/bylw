"""Randomized obstacle layouts for a held-out-map Isaac evaluation.

The route, sensor model, and controller remain those audited in C162.  Only
the obstacle layout is sampled from a deterministic seed-specific generator.
This keeps the experiment a map-generalization test rather than a new hidden
state or a new escape policy.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments.c150_native_search.task import Task, visibility3
from experiments.c160_all_search.world import HiddenBranchWorld as _CornerWorld


def obstacle_shape_profile(n_obstacles: int, shape_profile: str = "cylinder"):
    """Return actor kinds, local safety envelopes, and local box sizes.

    The circular envelope is intentionally conservative for boxes: line of
    sight, analytic avoidance, and Isaac collision checks use the same
    circumscribed radius.  This lets the mixed-shape experiment isolate shape
    generalization without introducing an unmodelled collision footprint.
    """

    n_obstacles = int(n_obstacles)
    if n_obstacles not in (4, 6, 8):
        raise ValueError("registered random-map obstacle counts are 4, 6, and 8")
    if shape_profile not in ("cylinder", "mixed"):
        raise ValueError("registered shape profiles are cylinder and mixed")
    if shape_profile == "cylinder":
        return ("cylinder",) * n_obstacles, np.array(
            [3.0] + [1.2] * (n_obstacles - 1), dtype=float
        ), ((0.0, 0.0),) * n_obstacles

    # Local dimensions are scaled by the same 1.6 factor as the scene.  The
    # alternating boxes include a square and two wall-like rectangles.
    local_box_sizes = {
        1: (2.4, 4.0),
        3: (2.4, 2.4),
        5: (1.5, 5.0),
        7: (2.0, 3.5),
    }
    kinds = []
    radii = []
    boxes = []
    for index in range(n_obstacles):
        if index == 0 or index not in local_box_sizes:
            kinds.append("cylinder")
            radii.append(3.0 if index == 0 else 1.2)
            boxes.append((0.0, 0.0))
        else:
            size_x, size_y = local_box_sizes[index]
            kinds.append("box")
            radii.append(0.5 * float(np.hypot(size_x, size_y)))
            boxes.append((size_x, size_y))
    return tuple(kinds), np.asarray(radii, dtype=float), tuple(boxes)


def capture_ring_feasible(obstacles, radii, target=None, clearance=0.5):
    """Find a sampled regular triangle clear of obstacle envelopes at reset.

    A positive result is a static witness for the initial target position.
    A negative result means no sampled regular triangle was found, not a
    proof that all permitted (possibly irregular) formations are impossible.
    Neither outcome certifies reachability or feasibility at future positions.
    Used as a diagnostic by default; optional filtering defines another map
    distribution and must be declared in the experiment metadata.
    """
    obstacles = np.asarray(obstacles, dtype=float)
    radii = np.asarray(radii, dtype=float)
    center = np.zeros(2, dtype=float) if target is None else np.asarray(target, dtype=float)
    for radius in np.linspace(4.5, 7.5, 13):
        for phase in np.arange(72) * (2.0 * np.pi / 72.0):
            angles = phase + np.arange(3) * (2.0 * np.pi / 3.0)
            points = center + radius * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
            if np.all(
                np.linalg.norm(points[:, None] - obstacles[None], axis=-1)
                - radii[None]
                > clearance
            ):
                return True
    return False


class RandomMapWorld(_CornerWorld):
    """C154 hidden-branch route with a seed-specific obstacle map.

    Four cylinders remain the frozen C179 baseline.  The optional six/eight
    profiles are used by the obstacle-density follow-up and deliberately keep
    the same route, sensor model, and reset legality checks.
    """

    scenario = "random"

    def __init__(
        self,
        seeds,
        task=None,
        scale=1.0,
        n_pursuers=3,
        split="train",
        n_obstacles=4,
        shape_profile="cylinder",
        layout_profile="distributed",
        initial_ring_screen=False,
    ):
        if int(n_obstacles) not in (4, 6, 8):
            raise ValueError("registered random-map obstacle counts are 4, 6, and 8")
        obstacle_shape_profile(n_obstacles, shape_profile)
        if layout_profile not in ("distributed", "route_dense"):
            raise ValueError("registered layout profiles are distributed and route_dense")
        self.n_obstacles = int(n_obstacles)
        self.shape_profile = str(shape_profile)
        self.layout_profile = str(layout_profile)
        self.initial_ring_screen = bool(initial_ring_screen)
        if task is None:
            task = Task(scene="corner")
        # Keep the physically audited corner route and hidden branch.  The
        # split only changes the deterministic map sampler, never the policy.
        internal = replace(task, scene="corner")
        super().__init__(seeds, internal, scale, n_pursuers=n_pursuers)
        # SearchWorld allocates the legacy four-obstacle tensors.  Reallocate
        # them before installing a six/eight-obstacle sampled layout so the
        # analytic LOS and safety envelopes have the same cardinality as the
        # Isaac actors created by the runner.
        if self.obstacles.shape[1] != self.n_obstacles:
            self.obstacles = np.empty((len(seeds), self.n_obstacles, 2), dtype=float)
            self.radii = np.empty((len(seeds), self.n_obstacles), dtype=float)
        self.layout_split = str(split)
        self.map_rejections = np.zeros(len(seeds), dtype=int)
        self.ring_rejections = np.zeros(len(seeds), dtype=int)
        self.initial_ring_sample_found = np.zeros(len(seeds), dtype=bool)
        split_code = 0 if self.layout_split == "train" else 1
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), 179, split_code])
            )
            theta = self._scene_rotation(i)
            for _attempt in range(200):
                local, local_radii = self._sample_layout(
                    rng, self.n_obstacles, self.shape_profile, self.layout_profile
                )
                obstacles = self.target[i] + (local @ theta.T) * scale
                radii = local_radii * scale
                ring_found = capture_ring_feasible(obstacles, radii, self.target[i])
                if self.initial_ring_screen and not ring_found:
                    self.ring_rejections[i] += 1
                    continue
                mask = visibility3(
                    self.p[i : i + 1],
                    self.z[i : i + 1],
                    self.target[i : i + 1],
                    self.tz[i : i + 1],
                    self.yaw[i : i + 1],
                    obstacles[None],
                    radii[None],
                    self.cfg,
                )
                target_clearance = (
                    np.linalg.norm(self.target[i] - obstacles, axis=-1) - radii
                ).min()
                pursuer_clearance = (
                    np.linalg.norm(self.p[i, :, None] - obstacles[None], axis=-1)
                    - radii[None]
                ).min()
                if mask.any() and target_clearance > 0.5 and pursuer_clearance > 0.5:
                    self.obstacles[i] = obstacles
                    self.radii[i] = radii
                    self.map_rejections[i] = _attempt
                    self.initial_ring_sample_found[i] = ring_found
                    break
            else:
                raise RuntimeError(f"random map rejection sampler exhausted for seed {seed}")

        self._validate_initial_layout()

    @staticmethod
    def _sample_layout(
        rng, n_obstacles=4, shape_profile="cylinder", layout_profile="distributed"
    ):
        """Sample route-relevant blockers plus seeded distractors.

        The rejection constraints keep the initial state legal and leave the
        route/target physically reachable.  They do not inspect future sensor
        measurements or force a loss flag; visibility is still computed by
        the ordinary 3-D line-of-sight implementation.
        """
        if int(n_obstacles) not in (4, 6, 8):
            raise ValueError("registered random-map obstacle counts are 4, 6, and 8")
        if layout_profile not in ("distributed", "route_dense"):
            raise ValueError("registered layout profiles are distributed and route_dense")
        n_obstacles = int(n_obstacles)
        _, profile_radii, _ = obstacle_shape_profile(n_obstacles, shape_profile)
        # The initial pursuers approach from local x ~= -11.  Sampling the
        # primary cylinder on the upper side of that approach creates a range
        # of genuine occlusion geometries while retaining an initial sighting.
        for _ in range(200):
            x = rng.uniform(-5.2, -1.7)
            y = rng.uniform(1.7, 4.8)
            # Isaac creates one physical obstacle asset per slot.  Keep the
            # local radii fixed so the analytic LOS/safety model and the
            # physical cylinder have exactly the same envelope at scale 1.6.
            radius = float(profile_radii[0])
            if np.hypot(x, y) > radius + 0.9:
                break
        primary = np.array([x, y], dtype=float)

        # The frozen C179 baseline remains one blocker plus three remote
        # distractors.  Only the density follow-up adds a second route blocker.
        if n_obstacles == 4:
            distractors = []
            for _ in range(3):
                for _attempt in range(400):
                    if layout_profile == "route_dense":
                        # Keep the baseline approach blocker but place the
                        # remaining bodies around the shared/branch corridor.
                        candidate = np.array(
                            [rng.uniform(3.0, 16.0), rng.uniform(-2.0, 14.0)],
                            dtype=float,
                        )
                    else:
                        angle = rng.uniform(-np.pi, np.pi)
                        distance = rng.uniform(14.0, 24.0)
                        candidate = distance * np.array([np.cos(angle), np.sin(angle)])
                    candidate_radius = float(profile_radii[1 + len(distractors)])
                    route_clear = (
                        True
                        if layout_profile == "route_dense"
                        else candidate[0] > 7.0 or abs(candidate[1]) > 8.0
                    )
                    far_primary = np.linalg.norm(candidate - primary) > candidate_radius + radius + 1.5
                    far_previous = all(
                        np.linalg.norm(candidate - previous[0])
                        > candidate_radius + previous[1] + 1.5
                        for previous in distractors
                    )
                    if (
                        np.linalg.norm(candidate) > candidate_radius + 3.0
                        and route_clear
                        and far_primary
                        and far_previous
                    ):
                        distractors.append((candidate, candidate_radius))
                        break
                else:
                    raise RuntimeError("random map rejection sampler exhausted")
            local = np.vstack([primary, *(item[0] for item in distractors)])
            local_radii = np.array(
                [radius, *(item[1] for item in distractors)], dtype=float
            )
            return local, local_radii

        # A second blocker is placed near the post-occlusion continuation. It
        # makes the six/eight profiles genuinely multi-obstacle along the
        # route instead of merely adding far-away visual clutter.
        secondary = None
        for _attempt in range(400):
            candidate = np.array(
                [rng.uniform(3.0, 8.0), rng.uniform(4.5, 10.0)], dtype=float
            )
            candidate_radius = float(profile_radii[1])
            if (
                np.linalg.norm(candidate) > candidate_radius + 3.0
                and np.linalg.norm(candidate - primary) > candidate_radius + radius + 1.5
            ):
                secondary = (candidate, candidate_radius)
                break
        if secondary is None:
            raise RuntimeError("random map rejection sampler exhausted for secondary blocker")

        distractors = []
        for index in range(n_obstacles - 2):
            for _attempt in range(400):
                angle = rng.uniform(-np.pi, np.pi)
                # Keep the first two extra obstacles close enough to interact
                # with the 20 m sensor at scale 1.6. Remaining distractors
                # occupy a wider ring and test irrelevant-obstacle filtering.
                if layout_profile == "route_dense":
                    candidate = np.array(
                        [rng.uniform(3.0, 18.0), rng.uniform(-4.0, 15.0)],
                        dtype=float,
                    )
                else:
                    distance = rng.uniform(9.0, 13.0) if index < 2 else rng.uniform(14.0, 24.0)
                    candidate = distance * np.array([np.cos(angle), np.sin(angle)])
                candidate_radius = float(profile_radii[2 + index])
                # Keep distractors away from the target, initial approach and
                # the shared/continuation route corridor.
                route_clear = (
                    True
                    if layout_profile == "route_dense"
                    else candidate[0] > 7.0 or abs(candidate[1]) > 8.0
                )
                far_primary = np.linalg.norm(candidate - primary) > candidate_radius + radius + 1.5
                far_previous = all(
                    np.linalg.norm(candidate - previous[0])
                    > candidate_radius + previous[1] + 1.5
                    for previous in distractors
                )
                if (
                    np.linalg.norm(candidate) > candidate_radius + 3.0
                    and route_clear
                    and far_primary
                    and far_previous
                ):
                    distractors.append((candidate, candidate_radius))
                    break
            else:
                raise RuntimeError("random map rejection sampler exhausted")
        local = np.vstack([primary, secondary[0], *(item[0] for item in distractors)])
        local_radii = np.array(
            [radius, secondary[1], *(item[1] for item in distractors)], dtype=float
        )
        return local, local_radii

    def _validate_initial_layout(self):
        mask = visibility3(
            self.p,
            self.z,
            self.target,
            self.tz,
            self.yaw,
            self.obstacles,
            self.radii,
            self.cfg,
        )
        target_clearance = (
            np.linalg.norm(self.target[:, None] - self.obstacles, axis=-1) - self.radii
        ).min()
        pursuer_clearance = (
            np.linalg.norm(self.p[:, :, None] - self.obstacles[:, None], axis=-1)
            - self.radii[:, None]
        ).min()
        if not mask.any(1).all():
            raise ValueError("random map lacks an initial legal team observation")
        if target_clearance <= 0.5 or pursuer_clearance <= 0.5:
            raise ValueError("random map violates initial clearance")

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
        )
        return meta


def make_world(
    scene, seeds, task, scale, n_obstacles=4, shape_profile="cylinder",
    layout_profile="distributed", initial_ring_screen=False,
):
    if scene == "random_train":
        return RandomMapWorld(
            seeds, task, scale, split="train", n_obstacles=n_obstacles,
            shape_profile=shape_profile,
            layout_profile=layout_profile,
            initial_ring_screen=initial_ring_screen,
        )
    if scene == "random_test":
        return RandomMapWorld(
            seeds, task, scale, split="test", n_obstacles=n_obstacles,
            shape_profile=shape_profile,
            layout_profile=layout_profile,
            initial_ring_screen=initial_ring_screen,
        )
    raise ValueError(f"unregistered C179 scene: {scene}")
