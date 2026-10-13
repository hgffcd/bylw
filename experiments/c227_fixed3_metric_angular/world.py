"""Fixed-three angular-distance scaling with the same C200 feed allocator."""
from experiments.c225_search_removal.world import make_world as base_world
from experiments.c208_metric_budget.budget_core import MetricBudgetCore

def make_world(*args,control_variant='none',**kwargs):
    if control_variant not in ('none','metric'):raise ValueError(control_variant)
    w=base_world(*args,**kwargs)
    assert w.requested_recovery=='budget_transit' and not w.slot_control and w.active.all()
    if control_variant=='metric':w.__class__=MetricBudgetCore
    w.fixed3_metric=control_variant=='metric'
    return w
