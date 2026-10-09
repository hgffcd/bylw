import numpy as np
import pytest
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c208_metric_budget.budget_core import MetricBudgetCore
from experiments.c209_coverage_handover.world import CoverageHandoverWorld
from experiments.c210_radial_damping.world import make_world,RadialDampedWorld
from experiments.c210_radial_damping.formation_core import DampedFormationCore
from experiments.c209_coverage_handover.controller import team_weight


def make(agents,arm):
    return make_world('random_test',[20151003],n_pursuers=agents,recovery=arm,method='active_polar',
        n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')


@pytest.mark.parametrize('agents',[4,5])
def test_before_fault_map_and_commands_identical(agents):
    a=make(agents,'coverage_metric_sector');b=make(agents,'radial_damped_sector')
    for key in ['p','target','z','active','obstacles','radii','routes']:
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    for x,y in zip(a.commands(a.observe()),b.commands(b.observe())):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(a.evader_command(),b.evader_command())


@pytest.mark.parametrize('retired',[[1],[1,3]])
def test_compaction_hidden_core_and_retired_outputs(retired):
    w=make(5,'radial_damped_sector')
    for j in retired:w.withdraw(j)
    view,o,_=w._compact(w.observe());assert isinstance(view,RadialDampedWorld)
    assert o['p'].shape[1]==5-len(retired)
    np.testing.assert_array_equal(o['p'],w.p[:,w.active[0]])
    o=dict(o,measured=np.zeros(1,bool),searcher=np.zeros(1,int),age=np.ones(1))
    for x,y in zip(view._core_commands(o),AuditMixin._core_commands(view,o)):np.testing.assert_array_equal(x,y)
    w.time=.1
    assert (w.commands(w.observe())[0][:,retired]==0).all()


@pytest.mark.parametrize('radius,span',[(7.,90),(10.,90),(13.,90),(13.,270)])
def test_core_dispatch_with_coverage_and_speed_bound(radius,span):
    w=make(5,'radial_damped_sector');w.withdraw(1);view,o,_=w._compact(w.observe())
    angles=np.deg2rad(np.linspace(0,span,4))
    o=dict(o,p=o['target'][:,None]+radius*np.stack([np.cos(angles),np.sin(angles)],-1)[None],
           measured=np.ones(1,bool),searcher=np.full(1,-1,int),target_v=np.array([[3.2,0.]]),omega=np.zeros(1))
    old=DampedFormationCore._core_commands(view,o);new=MetricBudgetCore._core_commands(view,o);actual=view._core_commands(o)
    weight=team_weight(o)
    np.testing.assert_allclose(actual[0],old[0]+weight[:,None,None]*(new[0]-old[0]),atol=1e-12)
    for j in [1,2]:np.testing.assert_array_equal(actual[j],old[j])
    assert np.isfinite(actual[0]).all() and (np.linalg.norm(actual[0],axis=-1)<=4+1e-8).all()


@pytest.mark.parametrize('radius',[13.,7.])
def test_far_or_matching_relative_velocity_exact_parent(radius):
    w=make(5,'radial_damped_sector');w.withdraw(1);view,o,_=w._compact(w.observe())
    angles=np.deg2rad([0,30,60,90]);tv=np.array([[2.,.1]])
    o=dict(o,p=o['target'][:,None]+radius*np.stack([np.cos(angles),np.sin(angles)],-1)[None],
        v=np.broadcast_to(tv[:,None],o['v'].shape).copy() if radius<12 else o['v'],
        measured=np.ones(1,bool),searcher=np.full(1,-1,int),target_v=tv,omega=np.zeros(1))
    for x,y in zip(view._core_commands(o),CoverageHandoverWorld._core_commands(view,o)):
        np.testing.assert_array_equal(x,y)
