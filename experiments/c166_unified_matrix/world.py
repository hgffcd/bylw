"""C162 final-matrix worlds.

The matrix keeps the registered C154 corner case and adds a second, scene
specific occlusion layout.  The corridor case is built from the same sensor,
route and dynamics model, but places a primary cylinder on the other flank of
the common approach.  It is not a scripted ``lost`` flag: visibility is still
computed by the 3-D line-of-sight test.
"""

from dataclasses import replace

import numpy as np

from experiments.c150_native_search.task import Task, visibility3
from experiments.c160_all_search.world import HiddenBranchWorld as _CornerWorld


class HiddenBranchWorld(_CornerWorld):
    """Registered C154 corner layout, exposed for the final matrix."""

    scenario = "corner"


class CorridorWorld(_CornerWorld):
    """Second occlusion layout with a flank cylinder on the common approach.

    ``SearchWorld`` only has registered route semantics for ``cover`` and
    ``corner``.  Internally we therefore use the cover route controller while
    exposing the experiment as ``corridor`` in the protocol and results.  The
    pursuers still receive only the same noisy visual measurements and public
    obstacle map.
    """

    scenario = "corridor"
    OBSTACLES_LOCAL = np.array(
        [[5.5, 0.0], [20.0, 20.0], [-20.0, 20.0], [20.0, -20.0]],
        dtype=float,
    )

    def __init__(self, seeds, task=None, scale=1.0):
        if task is None:
            task = Task(scene="corridor")
        # The parent route implementation is registered for cover/corner.
        internal = replace(task, scene="cover")
        super().__init__(seeds, internal, scale)
        for i in range(len(seeds)):
            theta = self._scene_rotation(i)
            self.obstacles[i] = self.target[i] + (self.OBSTACLES_LOCAL @ theta.T) * scale
        self.radii[:] = np.array([3.0, 1.2, 1.2, 1.2]) * scale
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
            raise ValueError("corridor layout lacks an initial legal team observation")
        if target_clearance <= 0.5 or pursuer_clearance <= 0.5:
            raise ValueError("corridor layout violates initial clearance")

    def branch_metadata(self):
        meta = super().branch_metadata()
        meta.update(
            scenario=self.scenario,
            obstacle_layout="flank_cylinder_local_5_5_0",
            passive_visibility_trigger=True,
        )
        return meta


def make_world(scene, seeds, task, scale):
    if scene == "corner":
        return HiddenBranchWorld(seeds, task, scale)
    if scene == "corridor":
        return CorridorWorld(seeds, task, scale)
    raise ValueError(f"unregistered C162 scene: {scene}")
