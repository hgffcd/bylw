"""Explicit fixed-three search modes; existing C200/C198 control is unchanged."""
from experiments.c200_feedforward_budget.world import make_world as base_world


def make_world(*args,search_mode='single',**kwargs):
    if search_mode not in ('single','all'):
        raise ValueError(search_mode)
    w=base_world(*args,**kwargs)
    assert w.n_pursuers==3 and w.method=='active_polar'
    assert w.requested_recovery in ('routed_transit','budget_transit')
    w.role_modes[:]=1 if search_mode=='single' else 2
    assert w.braking.all() and w.staged.all()
    return w
