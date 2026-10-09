from types import SimpleNamespace
import numpy as np
import pytest

from experiments.c145_feasibility_reset.planar import Config, limit, wrap
from experiments.c179_random_map_generalization.world import obstacle_shape_profile
from experiments.c180_random_map_variable.world import make_world as old_world
from experiments.c191_route_validity.sampling import sample_valid, validity
from experiments.c191_route_validity.world import make_world
from experiments.c191_route_validity.opponent import RouteGuide, remaining_cost


@pytest.mark.parametrize('count', [4,6,8])
def test_all_pairs_including_second_obstacle_and_both_branch_goals_are_legal(count):
    for seed in range(8):
        o,r,*_=sample_valid(np.random.default_rng(seed),count,'mixed','route_dense',1.6)
        assert validity(o,r)[0]
        gap=np.linalg.norm(o[:,None]-o[None],axis=-1)-r[:,None]-r[None]
        assert gap[np.triu_indices(count,1)].min()>=1.5-1e-8


def test_legacy_reproduction_and_deterministic_new_map_across_methods():
    settings=dict(n_pursuers=5,n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')
    old=old_world('random_test',[18912018],**settings)
    legacy=make_world('random_test',[18912018],map_protocol='legacy',opponent='local',**settings)
    np.testing.assert_array_equal(old.obstacles,legacy.obstacles)
    np.testing.assert_array_equal(old.evader_command(),legacy.evader_command())
    worlds=[make_world('random_test',[19120000],recovery=r,**settings) for r in ('routed_transit','moving_joint_v2')]
    np.testing.assert_array_equal(worlds[0].obstacles,worlds[1].obstacles)
    for w in worlds:
        assert not w.initial_ring_screen and w.observe()['measured'].all()
        assert (np.linalg.norm(w.routes[:,:,None]-w.obstacles[:,None],axis=-1)-w.radii[:,None]).min()>=1.4-1e-8


def test_route_progress_can_reward_detour_away_from_final_goal():
    start=np.array([0.,0.]);path=np.array([[0.,3.],[4.,3.],[4.,0.]])
    points=np.array([[0.,0.],[0.,1.],[0.,2.]])
    assert np.diff(np.linalg.norm(points-path[-1],axis=1)).min()>0
    assert np.diff(remaining_cost(points,start,path)).max()<0


def test_bounded_preview_follows_detour_without_stalling_or_crossing_obstacle():
    w=SimpleNamespace(seeds=[0],obstacles=np.array([[[0.,0.]]]),radii=np.array([[2.]]),
        routes=np.array([[[5.,0.],[25.,0.],[40.,0.]]]),target=np.array([[-6.,0.]]),tv=np.zeros((1,2)),
        heading=np.zeros(1),route_index=np.zeros(1,int),route_done=np.zeros(1,bool),
        branch_switch_tick=np.full(1,-1),branch_switch_hidden=np.zeros(1,bool),branch_reached=np.zeros(1,bool),
        measured=np.ones(1,bool),p=np.array([[[-15.,-2.],[-15.,0.],[-15.,2.]]]),active=np.ones((1,3),bool),
        cfg=Config(),time=0.)
    guide=RouteGuide(w);minimum=float('inf')
    for tick in range(160):
        w.time=tick*.1;heading=w.heading.copy();command=guide.command(w)
        assert np.linalg.norm(command,axis=1).max()<=3.2+1e-8
        if not w.route_done.any():assert abs(wrap(w.heading-heading)).max()<=.08+1e-8
        for _ in range(5):
            w.tv+=.02*limit(2*(command-w.tv),4.);w.target+=.02*w.tv
            minimum=min(minimum,float(np.linalg.norm(w.target[0])-2))
    assert minimum>=.5
    assert w.route_index[0]>=1 and w.target[0,0]>5
    assert guide.no_plan[0]==0
