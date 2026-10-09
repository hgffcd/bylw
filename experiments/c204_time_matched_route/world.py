"""C203 team handover with a time-matched C195 bypass destination."""
import numpy as np
from experiments.c203_phased_budget.world import make_world as previous_world
from experiments.c204_time_matched_route.planner import TimeMatchedBypassPlanner


def make_world(*args,recovery='routed_transit',**kwargs):
    if recovery!='time_matched_sector':
        return previous_world(*args,recovery=recovery,**kwargs)
    w=previous_world(*args,recovery='phased_budget_sector',**kwargs)
    planner=TimeMatchedBypassPlanner(len(w.seeds),w.n_pursuers)
    w.recovery_planner=planner
    w.requested_recovery=recovery
    original=planner.propose
    # Same read-only input instrumentation and cached-query timing as C198.
    def recorded(o,base,time):
        planner.audit_last={k:np.asarray(o[k]).copy() for k in ['p','v','z','vz','target','target_v','omega','measured','active','age','variance']}
        planner.audit_last['base']=base.copy()
        planner.audit_last['time']=np.asarray(time)
        result=original(o,base,time)
        planner.audit_last['proposal']=result[0].copy()
        planner.audit_last['changed']=result[1].copy()
        planner.audit_last['bypass_calls']=planner.bypass_calls.copy()
        return result
    planner.propose=recorded
    return w
