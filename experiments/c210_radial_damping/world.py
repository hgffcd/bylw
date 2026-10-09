"""C209 routing/handover with damped original formation core near the ring."""
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c208_metric_budget.budget_core import MetricBudgetCore
from experiments.c209_coverage_handover.controller import team_weight
from experiments.c209_coverage_handover.world import CoverageHandoverWorld,make_world as previous_world
from experiments.c210_radial_damping.formation_core import DampedFormationCore


class RadialDampedWorld(CoverageHandoverWorld):
    def _core_commands(self,o,actions=None):
        departed=getattr(self,'departed_p',None)
        if departed is None or not departed.shape[1]:
            return AuditMixin._core_commands(self,o,actions)
        weight=team_weight(o)
        if (weight==1).all():return MetricBudgetCore._core_commands(self,o,actions)
        old=DampedFormationCore._core_commands(self,o,actions)
        if (weight==0).all():return old
        new=MetricBudgetCore._core_commands(self,o,actions)
        return old[0]+weight[:,None,None]*(new[0]-old[0]),old[1],old[2]


def make_world(*args,recovery='routed_transit',**kwargs):
    if recovery!='radial_damped_sector':return previous_world(*args,recovery=recovery,**kwargs)
    w=previous_world(*args,recovery='coverage_metric_sector',**kwargs)
    w.__class__=RadialDampedWorld;w.requested_recovery=recovery
    return w
