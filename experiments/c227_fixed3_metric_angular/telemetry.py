"""Read-only public command inputs and pre-safety angular-scaling effect."""
import numpy as np
from experiments.c224_fixed3_search_factorial.telemetry import snapshot as old_snapshot,reset_snapshot
from experiments.c200_feedforward_budget.controller import budget_polar
from experiments.c208_metric_budget.controller import metric_budget_polar

def snapshot(env):
    values=old_snapshot(env);w=env.pilot;o=env.last_obs
    acc=o['omega'][:,None]*np.stack([-o['target_v'][:,1],o['target_v'][:,0]],-1)
    args=(o['p'],o['target'],o['target_v'],w.control_radii,w.cfg,w.compensations[:,None,None]*acc[:,None])
    old=budget_polar(*args);new=metric_budget_polar(*args)
    active=(o['searcher']>=0)&(w.role_modes>0)
    branch_weight=np.where(active,np.where(w.staged,1.-o['phase_mix'],0.),1.)
    weight=np.where(o['measured'],branch_weight,0.)
    if not w.fixed3_metric:weight=np.zeros_like(weight)
    values['cmd_metric_nominal_delta']=weight[:,None,None]*(new-old)
    values['cmd_metric_weight']=weight.copy()
    for key in ['p','v','target','target_v','omega','searcher']:values['cmd_public_'+key]=o[key].copy()
    return values
