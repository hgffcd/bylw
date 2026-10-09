import json
from pathlib import Path
import numpy as np

from experiments.c180_random_map_variable.world import make_world
from experiments.c184_route_recovery.planner import visible_edges,plan_formation
from experiments.c186_precision_recovery.planner import refine_formation,PrecisionRecoveryPlanner
from experiments.c186_precision_recovery.projection import obstacle_extra_reserve,precision_filter


def test_refinement_finds_routed_legal_slots_in_tight_regression_observation():
    data=json.loads((Path(__file__).parent/'tight_observation.json').read_text())
    p,c,obs,rr=[np.array(data[k]) for k in ('p','center','obstacles','radii')]
    assert plan_formation(p,c,obs,rr) is None
    plan=refine_formation(p,c,obs,rr,data['n_static'])
    assert plan is not None
    goals=c+plan['offsets'];radius=np.linalg.norm(plan['offsets'],axis=-1)
    angle=np.sort(np.arctan2(plan['offsets'][:,1],plan['offsets'][:,0]))
    gap=np.rad2deg(np.diff(np.r_[angle,angle[0]+2*np.pi]))
    assert radius.min()>4 and radius.max()<8 and np.ptp(radius)<2
    assert gap.min()>90 and gap.max()<150
    envelope=rr+np.r_[np.full(6,.8),1.1]
    assert (np.linalg.norm(goals[:,None]-obs,axis=-1)-envelope).min()>=-1e-5
    for start,path in zip(p,plan['paths']):
        chain=np.r_[start[None],path]
        for a,b in zip(chain[:-1],chain[1:]):
            assert visible_edges(a[None],b[None],np.r_[obs,c[None]],np.r_[envelope,3.8])[0,0]


def test_extra_reserve_grows_with_closing_speed_and_keeps_fast_limit():
    speed=np.linspace(0,4,100)
    extra=obstacle_extra_reserve(speed)
    assert extra[0]==.15 and extra[-1]==.8
    assert np.diff(extra).min()>=0
    np.testing.assert_array_equal(obstacle_extra_reserve(np.array([-1.,-2.])),[.15,.15])


def test_non_obstacle_constraints_are_exactly_the_original_projection():
    w=make_world('random_train',[18042001],n_obstacles=6);w.withdraw(0)
    view,o,ids=w._compact(w.observe())
    o=dict(o,obstacles=np.zeros((1,0,2)),radii=np.zeros((1,0)))
    command=np.array([[[3.,-1.],[2.,1.],[-1.,-4.]]])
    np.testing.assert_array_equal(view.safety_filter(command,o),precision_filter(view,command,o))


def test_untriggered_worlds_match_c184_commands_exactly():
    worlds=[make_world('random_train',[18042000,18042001],n_obstacles=6,shape_profile='mixed',
                      layout_profile='route_dense',recovery=r)
            for r in ('routed','refined_fixed','routed_precision')]
    for failed in (False,True):
        if failed:
            for w in worlds:w.withdraw(0)
        commands=[w.commands(w.observe()) for w in worlds]
        for candidate in commands[1:]:
            for a,b in zip(commands[0],candidate):np.testing.assert_array_equal(a,b)


def test_loss_or_moving_target_disables_precision():
    w=make_world('random_train',[18042000],n_obstacles=6);w.withdraw(0)
    for seen,velocity in ((False,[0.,0.]),(True,[3.2,0.])):
        planner=PrecisionRecoveryPlanner(1,4);planner.precision_on[:]=True
        o=dict(w.observe(),measured=np.array([seen]),target_v=np.array([velocity]))
        base=np.zeros((1,4,2));u,changed=planner.propose(o,base,1.)
        assert not planner.precision_on.any() and not planner.precision_changed.any()
        assert not np.isfinite(planner.precision_center).any()
        np.testing.assert_array_equal(u,base)


def test_observed_center_reduces_static_jitter_without_changing_commands():
    w=make_world('random_train',[18042000],n_obstacles=6);w.withdraw(0)
    planner=PrecisionRecoveryPlanner(1,4);o=dict(w.observe(),measured=np.array([True]),target_v=np.zeros((1,2)))
    base=np.full((1,4,2),2.);filtered=[]
    for tick in range(100):
        o['target']=np.array([[.12*(-1)**tick,0.]])
        command,changed=planner.propose(o,base,tick*.1)
        assert not changed.any()
        np.testing.assert_array_equal(command,base)
        filtered.append(planner.precision_center[0,0])
    assert np.std(filtered[-50:])<.12*.3


def test_velocity_prediction_avoids_lag_for_slow_constant_motion():
    w=make_world('random_train',[18042000],n_obstacles=6);w.withdraw(0)
    planner=PrecisionRecoveryPlanner(1,4);o=dict(w.observe(),measured=np.array([True]),target_v=np.array([[.4,0.]]))
    for tick in range(30):
        o['target']=np.array([[.04*tick,0.]])
        planner.propose(o,np.full((1,4,2),2.),tick*.1)
        np.testing.assert_allclose(planner.precision_center,o['target'],atol=1e-10)
