"""C230 with a bounded, public-state bypass-to-formation command ramp.

Only successful clear-path exits are ramped for the existing 0.6 s replan
period. Loss/search, near-ring exits and the 6 s lease do not start a ramp.
The downstream unchanged response/safety layer owns executed commands.
"""
import numpy as np
from experiments.c230_fixed3_time_match.planner import TimeProgressPlanner

HANDOFF_SECONDS=.6

class SmoothHandoffPlanner(TimeProgressPlanner):
    def __init__(self,batch,agents):
        super().__init__(batch,agents)
        self.handoff_start=np.full((batch,agents),-np.inf)
        self.handoff_from=np.zeros((batch,agents,2))
        self.handoff_weight=np.zeros((batch,agents))
        self.handoff_entries=np.zeros((batch,agents),int)
        self.handoff_clock=None
        self.handoff_output=None

    def propose(self,o,base,time):
        if self.handoff_clock==time and self.handoff_output is not None:
            return self.handoff_output
        prior_on=self.bypass_on.copy()
        prior_applied=self.bypass_applied.copy()
        prior_start=self.bypass_start.copy()
        previous=None if self.handoff_output is None else self.handoff_output[0].copy()
        out,changed=super().propose(o,base,time)
        out,changed=out.copy(),changed.copy()
        eligible=(o['active'].sum(-1)==self.agents)&(self.agents==3)&o['measured']
        eligible &= o.get('phase_mix',np.zeros(len(base)))<=0
        eligible &= np.linalg.norm(o['target_v'],axis=-1)>1.
        radius=np.linalg.norm(o['p']-o['target'][:,None],axis=-1)
        exited=prior_on&~self.bypass_on&prior_applied&eligible[:,None]
        exited &= (radius>=8.5)&(time-prior_start<6.)
        if previous is not None:
            self.handoff_start[exited]=time
            self.handoff_from[exited]=previous[exited]
            self.handoff_entries[exited]+=1
        cancel=(~eligible[:,None])|self.bypass_on|(radius<8.5)
        self.handoff_start[cancel]=-np.inf
        weight=np.clip(1.-(time-self.handoff_start)/HANDOFF_SECONDS,0.,1.)
        self.handoff_weight=weight
        out=weight[...,None]*self.handoff_from+(1.-weight[...,None])*out
        applied=weight>0
        changed |= applied.any(-1)
        self.bypass_applied |= applied
        self.handoff_clock=time
        self.handoff_output=out,changed
        return self.handoff_output
