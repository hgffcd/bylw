"""C154 scene: two route continuations diverge while the team is occluded."""
import numpy as np
from experiments.c152_hidden_branch.world import HiddenBranchWorld as _C152HiddenBranchWorld


class HiddenBranchWorld(_C152HiddenBranchWorld):
    """A corrected hidden-branch world for the train/evaluate protocol.

    The shared waypoint is deliberately closer than C152's old 7 m waypoint.
    At scale 1.6 the route switch occurs inside the natural 20 m/90 deg loss
    window for the registered seeds, so the continuation is selected before
    the pursuers can use a new measurement.  The branch label and future route
    are never included in the observation.
    """
    # Both routes use the same occluding approach [0, 4].  They then separate.
    BRANCH_LOCAL = np.array([
        [[0., 4.], [8., 12.], [25., 12.]],
        [[0., 4.], [8.,  2.], [25.,  0.]],
    ])

    def __init__(self, seeds, task=None, scale=1., n_pursuers=3):
        if task is None:
            from experiments.c150_native_search.task import Task
            task = Task(scene='corner')
        super().__init__(seeds, task, scale, n_pursuers=n_pursuers)
        n = len(seeds)
        self.branch_switch_tick = np.full(n, -1, dtype=int)
        self.branch_switch_hidden = np.zeros(n, dtype=bool)

    def evader_command(self):
        before = self.route_index.copy()
        command = super().evader_command()
        switched = (before == 0) & (self.route_index >= 1)
        if switched.any():
            self.branch_switch_tick[switched] = int(round(self.time * 10))
            # This is the semantic audit: branch activation must precede the
            # next shared target measurement, rather than merely occur near it.
            self.branch_switch_hidden[switched] = ~self.measured[switched]
        self.branch_reached |= self.route_index >= 1
        return command

    def branch_metadata(self):
        return dict(
            branch_count=2,
            branch_fraction=float(self.branch.mean()),
            reached_fraction=float(self.branch_reached.mean()),
            switch_hidden_fraction=float(self.branch_switch_hidden.mean()),
            branch_labels_hidden_from_pursuers=True,
        )
