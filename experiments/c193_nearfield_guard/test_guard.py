import numpy as np
from experiments.c192_bounded_bypass.planner import BoundedBypassPlanner
from experiments.c192_bounded_bypass.test_bypass import observation
from experiments.c193_nearfield_guard.planner import GuardedBypassPlanner
from experiments.c191_route_validity.world import make_world as old_world
from experiments.c193_nearfield_guard.world import make_world


def test_near_formation_stall_does_not_start_a_large_bypass():
    o=observation();o['target'][0]=[5.,0.]
    base=np.ones((1,5,2))*2.;base[0,0]=0.
    old=BoundedBypassPlanner(1,5);new=GuardedBypassPlanner(1,5)
    for tick in range(10):
        before,old_changed=old.propose(o,base,tick*.1)
        after,changed=new.propose(o,base,tick*.1)
    assert old.bypass_entries.sum()>0
    assert new.bypass_entries.sum()==0 and not changed.any()
    np.testing.assert_array_equal(after,base)


def test_far_stall_routing_and_handover_remain_exactly_c192():
    o=observation();base=np.ones((1,5,2))*2.;base[0,0]=0.
    planners=[BoundedBypassPlanner(1,5),GuardedBypassPlanner(1,5)]
    for tick in range(80):
        if tick==70:o['measured'][:]=False
        outputs=[p.propose(o,base,tick*.1) for p in planners]
        for a,b in zip(*outputs):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(planners[0].bypass_applied,planners[1].bypass_applied)
    assert planners[1].bypass_entries.sum()>0 and not planners[1].bypass_on.any()


def test_world_and_opponent_are_unchanged():
    args=dict(n_pursuers=4,n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')
    a=old_world('random_test',[19120001],recovery='routed_transit',**args)
    b=make_world('random_test',[19120001],recovery='guarded_bypass',**args)
    for name in ['obstacles','radii','target','routes','p','active']:
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    np.testing.assert_array_equal(a.evader_command(),b.evader_command())
