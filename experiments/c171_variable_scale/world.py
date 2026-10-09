"""P5-C worlds built on the audited C162 routes with variable team size."""

from __future__ import annotations

import numpy as np

from experiments.c150_native_search.task import Task
from experiments.c162_final_matrix.world import CorridorWorld as _CorridorWorld
from experiments.c162_final_matrix.world import HiddenBranchWorld as _CornerWorld


class VariableCornerWorld(_CornerWorld):
    def __init__(self, seeds, task, scale=1.6, n_pursuers=3):
        super().__init__(seeds, task, scale, n_pursuers=n_pursuers)


class VariableCorridorWorld(_CorridorWorld):
    def __init__(self, seeds, task, scale=1.6, n_pursuers=3):
        super().__init__(seeds, task, scale, n_pursuers=n_pursuers)


def make_world(scene, seeds, task=None, scale=1.6, n_pursuers=3):
    if task is None:
        task = Task(scene=scene)
    if scene == "corner":
        world = VariableCornerWorld(seeds, task, scale, n_pursuers)
    elif scene == "corridor":
        world = VariableCorridorWorld(seeds, task, scale, n_pursuers)
    else:
        raise ValueError(scene)
    # Match the audited low-level transition layer used by C166--C170.
    world.configure(slice(None), "staged_brake")
    # The nominal controller remains the audited C162/C169 low-level policy;
    # P5-C activates slot reallocation only when the fault event requests it.
    return world


def fail_one(world, index=0, reconfigure=True, gain=0.8, duration=3.0):
    """Disable one pursuer while retaining a feasible >=3-agent team."""

    active = world.active.copy()
    if active.shape[1] <= 3:
        raise ValueError("fail_one requires at least four configured pursuers")
    if not 0 <= index < active.shape[1]:
        raise IndexError(index)
    active[:, index] = False
    world.set_active(active)
    if reconfigure:
        # Dynamic slot reassignment is an explicit P5-C treatment.  The fixed
        # baseline keeps the audited polar commands after the same fault event.
        world.enable_slot_control(gain=gain, duration=duration)
    return active
