"""Remove active search role only; retain prediction, staging and safety."""
from experiments.c224_fixed3_search_factorial.world import make_world as base_world

def make_world(*args,search_mode='predict',**kwargs):
    if search_mode not in ('predict','single','all'):raise ValueError(search_mode)
    w=base_world(*args,search_mode='single' if search_mode=='predict' else search_mode,**kwargs)
    if search_mode=='predict':w.role_modes[:]=0
    assert not w.slot_control and w.braking.all() and w.staged.all()
    return w
