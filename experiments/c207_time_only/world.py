"""Component ablation: C195 core with the C204 time-matched route planner."""
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c203_phased_budget.world import PhasedBudgetWorld
from experiments.c206_radial_priority.world import make_world as previous_world


class TimeOnlyWorld(PhasedBudgetWorld):
    _core_commands=AuditMixin._core_commands


def make_world(*args,recovery='routed_transit',**kwargs):
    if recovery!='time_only_sector':return previous_world(*args,recovery=recovery,**kwargs)
    w=previous_world(*args,recovery='time_matched_sector',**kwargs)
    assert w.n_pursuers in (4,5)
    w.__class__=TimeOnlyWorld;w.requested_recovery=recovery
    return w
