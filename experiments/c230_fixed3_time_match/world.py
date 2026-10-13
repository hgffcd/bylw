"""Keep C200 fixed-three core; candidate enables progress-based obstacle bypass."""
import numpy as np
from experiments.c227_fixed3_metric_angular.world import make_world as base_world
from experiments.c229_fixed3_progress_bypass.planner import ProgressBypassPlanner
from experiments.c230_fixed3_time_match.planner import TimeProgressPlanner

def make_world(*args,control_variant='none',**kwargs):
    if control_variant not in ('none','progress','matched'):raise ValueError(control_variant)
    w=base_world(*args,control_variant='none',**kwargs)
    w.fixed3_progress=control_variant!='none'
    if not w.fixed3_progress:return w
    planner=(TimeProgressPlanner if control_variant=='matched' else ProgressBypassPlanner)(len(w.seeds),w.n_pursuers)
    original=planner.propose
    def recorded(o,base,time):
        planner.audit_last={k:np.asarray(o[k]).copy() for k in ['p','v','z','vz','target','target_v','omega','measured','active','age','variance']}
        planner.audit_last['base']=base.copy();planner.audit_last['time']=np.asarray(time)
        result=original(o,base,time)
        planner.audit_last['proposal']=result[0].copy();planner.audit_last['changed']=result[1].copy()
        planner.audit_last['bypass_calls']=planner.bypass_calls.copy()
        return result
    planner.propose=recorded;w.recovery_planner=planner
    return w
