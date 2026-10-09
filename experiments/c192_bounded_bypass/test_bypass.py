import numpy as np
from experiments.c184_route_recovery.planner import visible_edges
from experiments.c192_bounded_bypass.planner import bypass_path, BoundedBypassPlanner
from experiments.c191_route_validity.world import make_world as old_world
from experiments.c192_bounded_bypass.world import make_world


def observation():
    return dict(p=np.array([[[-5.,0.],[10.,6.],[4.,0.],[10.,-6.],[-20.,-20.]]]),
        active=np.array([[1,1,1,1,0]],bool),measured=np.array([True]),
        target=np.array([[10.,0.]]),target_v=np.array([[2.,0.]]),v=np.zeros((1,5,2)),
        obstacles=np.array([[[-2.,0.]]]),radii=np.array([[2.5]]))


def test_only_stalled_member_changes_then_loss_hands_back_to_search():
    o=observation();base=np.ones((1,5,2))*2.;base[0,0]=0.
    planner=BoundedBypassPlanner(1,5)
    for t in [0.,.1,.2,.3,.4,.5]:out,changed=planner.propose(o,base,t)
    assert changed[0] and np.linalg.norm(out[0,0])>0.
    np.testing.assert_array_equal(out[0,1:],base[0,1:])
    assert np.linalg.norm(out[0,0])<=4.+1e-8
    o['measured'][:]=False
    out,changed=planner.propose(o,base,.6)
    assert not changed.any() and not planner.bypass_on.any()
    np.testing.assert_array_equal(out,base)


def test_lease_expires_even_when_original_stall_persists():
    o=observation();base=np.ones((1,5,2))*2.;base[0,0]=0.
    planner=BoundedBypassPlanner(1,5)
    for tick in range(70):out,changed=planner.propose(o,base,tick*.1)
    assert planner.bypass_entries.sum()==1
    assert not planner.bypass_on.any() and not changed.any()


def test_route_goes_around_blocker_without_crossing_envelopes():
    start=np.array([-8.,0.]);center=np.array([10.,0.]);obs=np.array([[0.,0.]]);radii=np.array([2.])
    path=bypass_path(start,center,np.array([2.,0.]),obs,radii)
    assert path is not None and len(path)>1
    chain=np.r_[start[None],path]
    for a,b in zip(chain[:-1],chain[1:]):
        assert visible_edges(a[None],b[None],np.r_[obs,center[None]],np.r_[radii+1.25,3.2])[0,0]


def test_world_preserves_c191_map_opponent_and_withdrawal_model():
    settings=dict(n_pursuers=5,n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')
    a=old_world('random_test',[19121000],recovery='routed_transit',**settings)
    b=make_world('random_test',[19121000],recovery='bounded_bypass',**settings)
    for field in ['obstacles','radii','p','target','routes','active']:
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    np.testing.assert_array_equal(a.evader_command(),b.evader_command())
