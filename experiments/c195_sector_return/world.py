"""Frozen map/opponent and old controllers; isolated C195 candidate."""
from experiments.c193_nearfield_guard.world import make_world as base_world
from experiments.c195_sector_return.planner import SectorBypassPlanner


def make_world(*args,recovery='routed_transit',**kwargs):
    if recovery!='sector_bypass':return base_world(*args,recovery=recovery,**kwargs)
    w=base_world(*args,recovery='routed_transit',**kwargs)
    w.recovery_planner=SectorBypassPlanner(len(w.seeds),w.n_pursuers)
    w.requested_recovery='sector_bypass'
    return w
