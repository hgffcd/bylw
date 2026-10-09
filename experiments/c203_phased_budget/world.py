"""Team-wide smooth handover from feed-preserving pursuit to old formation.

The 8 m outer capture radius and 12 m far-member guard are existing scales.
Interpolation is after the same core safety filter; the inherited response
guard and route-proposal projection still execute normally. Not a certificate.
"""
import numpy as np
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c200_feedforward_budget.world import BudgetRouteWorld
from experiments.c202_postfault_budget.world import PostFaultBudgetWorld, make_world as previous_world


def team_weight(o):
    radius=np.linalg.norm(o['p']-o['target'][:,None],axis=-1)
    return np.where(o['measured'],np.clip((radius.max(-1)-8.)/4.,0.,1.),0.)


class PhasedBudgetWorld(PostFaultBudgetWorld):
    def _core_commands(self,o,actions=None):
        departed=getattr(self,'departed_p',None)
        if departed is None or not departed.shape[1]:
            return AuditMixin._core_commands(self,o,actions)
        weight=team_weight(o)
        if (weight==1).all():return BudgetRouteWorld._core_commands(self,o,actions)
        old=AuditMixin._core_commands(self,o,actions)
        if (weight==0).all():return old
        new=BudgetRouteWorld._core_commands(self,o,actions)
        command=old[0]+weight[:,None,None]*(new[0]-old[0])
        return command,old[1],old[2]


def make_world(*args,recovery='routed_transit',**kwargs):
    if recovery!='phased_budget_sector':return previous_world(*args,recovery=recovery,**kwargs)
    w=previous_world(*args,recovery='budget_sector',**kwargs)
    w.__class__=PhasedBudgetWorld;w.requested_recovery=recovery
    return w
