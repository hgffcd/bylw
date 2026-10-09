"""C192 retains the frozen C191 valid-map/routed-opponent protocol."""
from experiments.c191_route_validity.world import make_world as base_world
from experiments.c192_bounded_bypass.planner import BoundedBypassPlanner


def make_world(*args,recovery='routed_transit',**kwargs):
    if recovery!='bounded_bypass':return base_world(*args,recovery=recovery,**kwargs)
    w=base_world(*args,recovery='routed_transit',**kwargs)
    w.recovery_planner=BoundedBypassPlanner(len(w.seeds),w.n_pursuers)
    w.requested_recovery='bounded_bypass'
    return w
