"""Keep C204 routing and C203 handover; replace only the far-core allocator."""
import numpy as np
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c203_phased_budget.world import PhasedBudgetWorld,team_weight
from experiments.c204_time_matched_route.world import make_world as previous_world
from experiments.c206_radial_priority.budget_core import RadialBudgetCore


class PhasedRadialWorld(PhasedBudgetWorld):
    def _core_commands(self,o,actions=None):
        departed=getattr(self,'departed_p',None)
        if departed is None or not departed.shape[1]:
            return AuditMixin._core_commands(self,o,actions)
        weight=team_weight(o)
        if (weight==1).all():return RadialBudgetCore._core_commands(self,o,actions)
        old=AuditMixin._core_commands(self,o,actions)
        if (weight==0).all():return old
        new=RadialBudgetCore._core_commands(self,o,actions)
        return old[0]+weight[:,None,None]*(new[0]-old[0]),old[1],old[2]


def make_world(*args,recovery='routed_transit',**kwargs):
    if recovery!='radial_priority_sector':return previous_world(*args,recovery=recovery,**kwargs)
    w=previous_world(*args,recovery='time_matched_sector',**kwargs)
    assert w.n_pursuers in (4,5)
    w.__class__=PhasedRadialWorld;w.requested_recovery=recovery
    return w
