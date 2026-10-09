"""Candidate: catch up before angular redistribution during fast escape.

Uses shared current observations only. The caller must apply the unchanged
active-member safety projection and response guard. No new success criterion.
"""
import numpy as np
from experiments.c184_route_recovery.planner import RecoveryPlanner


class ChaseRecoveryPlanner(RecoveryPlanner):
    def __init__(self, batch, agents):
        super().__init__(batch, agents)
        self.chase_on=np.zeros((batch,agents),bool)
        self.chase_entries=np.zeros((batch,agents),int)
        self.chase_clock=None
        self.combined_output=None

    def propose(self,o,base,time):
        if self.chase_clock==time and self.combined_output is not None:
            return self.combined_output
        self.chase_clock=time
        result,changed=super().propose(o,base,time)
        out=result.copy();changed=changed.copy()
        active=np.asarray(o['active'],bool)
        eligible=(active.sum(1)==3)&np.asarray(o['measured'],bool)
        eligible &= np.linalg.norm(o['target_v'],axis=-1)>2.5
        if self.agents>3:eligible &= active.sum(1)<self.agents
        delta=o['p']-o['target'][:,None]
        radius=np.linalg.norm(delta,axis=-1)
        previous=self.chase_on.copy()
        self.chase_on &= active & eligible[:,None] & (radius>7.0)
        self.chase_on |= active & eligible[:,None] & (radius>8.5)
        self.chase_entries += self.chase_on & ~previous
        # The ordinary polar law spends speed on angular redistribution even
        # when an agent is far behind. Preserve the same radial gain/radius,
        # but restore radial approach before resuming the original angular law.
        u=o['target_v'][:,None]-.9*(radius-6.)[...,None]*delta/np.maximum(radius[...,None],1e-9)
        speed=np.linalg.norm(u,axis=-1)
        u*=np.minimum(1.,4./np.maximum(speed,1e-9))[...,None]
        out[self.chase_on]=u[self.chase_on]
        changed |= self.chase_on.any(1)
        self.combined_output=(out,changed)
        return self.combined_output
