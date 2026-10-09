"""C208 metric core and route; change only the geometric handover weight."""
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c208_metric_budget.world import PhasedMetricWorld,make_world as previous_world
from experiments.c208_metric_budget.budget_core import MetricBudgetCore
from experiments.c209_coverage_handover.controller import team_weight


class CoverageHandoverWorld(PhasedMetricWorld):
    def _core_commands(self,o,actions=None):
        departed=getattr(self,'departed_p',None)
        if departed is None or not departed.shape[1]:
            return AuditMixin._core_commands(self,o,actions)
        weight=team_weight(o)
        if (weight==1).all():return MetricBudgetCore._core_commands(self,o,actions)
        old=AuditMixin._core_commands(self,o,actions)
        if (weight==0).all():return old
        new=MetricBudgetCore._core_commands(self,o,actions)
        return old[0]+weight[:,None,None]*(new[0]-old[0]),old[1],old[2]


def make_world(*args,recovery='routed_transit',**kwargs):
    if recovery!='coverage_metric_sector':return previous_world(*args,recovery=recovery,**kwargs)
    w=previous_world(*args,recovery='metric_budget_sector',**kwargs)
    w.__class__=CoverageHandoverWorld;w.requested_recovery=recovery
    return w
