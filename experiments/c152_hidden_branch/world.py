import numpy as np
from experiments.c151_response_guard.world import ResponseWorld
from experiments.c150_native_search.task import Task


class HiddenBranchWorld(ResponseWorld):
    """ResponseWorld with a hidden post-occlusion route branch."""
    BRANCH_LOCAL = np.array([
        [[0., 7.], [8., 12.], [25., 12.]],
        [[0., 7.], [8.,  2.], [25.,  0.]],
    ])

    def __init__(self,seeds,task=Task(scene='corner'),scale=1.,n_pursuers=3):
        super().__init__(seeds,task,scale,n_pursuers=n_pursuers)
        self.branch=np.array([
            np.random.default_rng(np.random.SeedSequence([int(s),152])).integers(0,2)
            for s in seeds],dtype=int)
        self.branch_routes=self.routes.copy()
        for i,s in enumerate(seeds):
            theta=self._scene_rotation(i)
            center=self.target[i]
            local=self.BRANCH_LOCAL[self.branch[i]]
            self.branch_routes[i]=center+(local@theta.T)*scale
        # The initial approach is shared by both routes; only the hidden
        # continuation differs. This branch is not included in observations.
        self.routes=self.branch_routes.copy()
        self.route_index[:]=0;self.route_done[:]=False
        self.branch_reached=np.zeros(len(seeds),bool)

    def _scene_rotation(self,i):
        # Recover the same deterministic rotation used by SearchWorld. The
        # first target-centred route is enough to estimate it robustly.
        d=self.routes[i,0]-self.target[i]
        # local [0,7] points along local +y; columns of R map local to world.
        ey=d/np.maximum(np.linalg.norm(d),1e-9)
        ex=np.array([ey[1],-ey[0]])
        # SearchWorld maps local points by ``local @ rot.T``.  ``ex`` and
        # ``ey`` are the world columns of rot, so return rot itself here.
        return np.stack([ex,ey],axis=1)

    def evader_command(self):
        self.branch_reached|=self.route_index>=1
        return super().evader_command()

    def branch_metadata(self):
        return dict(branch_count=2,branch_fraction=float(self.branch.mean()),
                    reached_fraction=float(self.branch_reached.mean()),
                    branch_labels_hidden_from_pursuers=True)
