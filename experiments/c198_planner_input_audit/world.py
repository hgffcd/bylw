"""Read-only copies of planner inputs; the original planner returns unchanged."""
import numpy as np
from experiments.c195_sector_return.world import make_world as base_world
from experiments.c187_transit_recovery.planner import TransitRecoveryPlanner


def make_world(*args,**kwargs):
    w=base_world(*args,**kwargs)
    if w.recovery_planner is None:
        assert w.recovery=='routed_transit'
        w.recovery_planner=TransitRecoveryPlanner(len(w.seeds),w.n_pursuers)
    planner=w.recovery_planner
    original=planner.propose
    def recorded(o,base,time):
        planner.audit_last={k:np.asarray(o[k]).copy() for k in ['p','v','z','vz','target','target_v','omega','measured','active','age','variance']}
        planner.audit_last['base']=base.copy()
        planner.audit_last['time']=np.asarray(time)
        result=original(o,base,time)
        planner.audit_last['proposal']=result[0].copy()
        planner.audit_last['changed']=result[1].copy()
        planner.audit_last['bypass_calls']=getattr(planner,'bypass_calls',np.zeros(o['active'].shape,int)).copy()
        return result
    planner.propose=recorded
    return w
