import numpy as np
from experiments.c184_route_recovery.planner import VisibilityRouter, visible_edges, OFFSETS, RecoveryPlanner
from experiments.c180_random_map_variable.world import make_world


def test_route_goes_around_blocked_straight_segment():
    c=np.array([[0.,0.]]);r=np.array([2.])
    router=VisibilityRouter(c,r,[[5.,0.]])
    distance,paths=router.paths_from([-5.,0.])
    assert np.isfinite(distance[0]) and distance[0]>10.
    path=np.r_[[[-5.,0.]],paths[0]]
    assert len(path)>2
    assert all(visible_edges(a[None],b[None],c,r)[0,0] for a,b in zip(path[:-1],path[1:]))


def test_start_inside_inflation_escapes_outwards_only():
    router=VisibilityRouter([[0.,0.]],[2.],[[5.,0.]])
    distance,paths=router.paths_from([-1.8,0.])
    assert np.isfinite(distance[0])
    assert paths[0][0,0]<=-1.8


def test_destination_inside_obstacle_is_unreachable():
    distance,paths=VisibilityRouter([[0.,0.]],[2.],[[0.,0.]]).paths_from([-5.,0.])
    assert not np.isfinite(distance[0]) and paths[0] is None


def test_sampled_formations_remain_inside_original_geometry_bounds():
    radius=np.linalg.norm(OFFSETS,axis=-1)
    angles=np.sort(np.arctan2(OFFSETS[...,1],OFFSETS[...,0]),axis=-1)
    gaps=np.diff(np.c_[angles,angles[:,:1]+2*np.pi],axis=-1)
    assert radius.min()>4 and radius.max()<8
    assert np.ptp(radius,axis=-1).max()<2
    assert gaps.min()>np.pi/2 and gaps.max()<5*np.pi/6


def test_routed_mode_matches_baseline_before_fault_and_without_trigger():
    base=make_world('random_train',[18042000,18042001],n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')
    routed=make_world('random_train',[18042000,18042001],n_obstacles=6,shape_profile='mixed',layout_profile='route_dense',recovery='routed')
    np.testing.assert_array_equal(base.obstacles,routed.obstacles)
    for before in (True,False):
        if not before:
            base.withdraw(0);routed.withdraw(0)
        a=base.commands(base.observe());b=routed.commands(routed.observe())
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    assert routed.recovery_planner.entries.sum()==0


def test_loss_suspends_recovery_without_reading_hidden_state():
    w=make_world('random_train',[18042000],n_obstacles=6)
    o=w.observe();w.withdraw(0);o=w.observe()
    planner=RecoveryPlanner(1,4);planner.on[:]=True
    o=dict(o, measured=np.array([False]),target=np.array([[999.,999.]]))
    base=np.zeros((1,4,2));out,changed=planner.propose(o,base,1.)
    np.testing.assert_array_equal(out,base)
    assert not changed.any() and not planner.on.any()


def test_fast_target_returns_control_to_baseline_pursuit():
    w=make_world('random_train',[18042000],n_obstacles=6)
    w.withdraw(0)
    o=dict(w.observe(), measured=np.array([True]), target_v=np.array([[3.2,0.]]))
    planner=RecoveryPlanner(1,4)
    planner.on[:]=True
    base=np.arange(8,dtype=float).reshape(1,4,2)
    out,changed=planner.propose(o,base,1.)
    np.testing.assert_array_equal(out,base)
    assert not changed.any() and not planner.on.any()
    assert planner.plan_calls.sum()==0
