"""C193 retains the frozen C191 valid-map/routed-opponent protocol."""
from experiments.c191_route_validity.world import make_world as base_world
from experiments.c193_nearfield_guard.planner import GuardedBypassPlanner


def make_world(*args,recovery='routed_transit',**kwargs):
    if recovery!='guarded_bypass':return base_world(*args,recovery=recovery,**kwargs)
    w=base_world(*args,recovery='routed_transit',**kwargs)
    w.recovery_planner=GuardedBypassPlanner(len(w.seeds),w.n_pursuers)
    w.requested_recovery='guarded_bypass'
    return w
