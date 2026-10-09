import numpy as np
import pytest
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c208_metric_budget.world import make_world,PhasedMetricWorld
from experiments.c208_metric_budget.budget_core import MetricBudgetCore


def make(agents,arm):
    return make_world('random_test',[20151003],n_pursuers=agents,recovery=arm,method='active_polar',
        n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')


@pytest.mark.parametrize('agents',[4,5])
def test_before_fault_map_and_commands_identical(agents):
    a=make(agents,'time_matched_sector');b=make(agents,'metric_budget_sector')
    for key in ['p','target','z','active','obstacles','radii','routes']:
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    for x,y in zip(a.commands(a.observe()),b.commands(b.observe())):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(a.evader_command(),b.evader_command())


@pytest.mark.parametrize('retired',[[1],[1,3]])
def test_compaction_hidden_core_and_retired_outputs(retired):
    w=make(5,'metric_budget_sector')
    for j in retired:w.withdraw(j)
    view,o,_=w._compact(w.observe());assert isinstance(view,PhasedMetricWorld)
    o=dict(o,measured=np.zeros(1,bool),searcher=np.zeros(1,int),age=np.ones(1))
    for x,y in zip(view._core_commands(o),AuditMixin._core_commands(view,o)):np.testing.assert_array_equal(x,y)
    w.time=.1
    assert (w.commands(w.observe())[0][:,retired]==0).all()


@pytest.mark.parametrize('radius',[7.,10.,13.])
def test_core_dispatch_and_bounded_commands(radius):
    w=make(5,'metric_budget_sector');w.withdraw(1);view,o,_=w._compact(w.observe())
    d=o['p']-o['target'][:,None]
    o=dict(o,p=o['target'][:,None]+radius*d/np.linalg.norm(d,axis=-1)[...,None],
           measured=np.ones(1,bool),searcher=np.full(1,-1,int),target_v=np.array([[3.2,0.]]),omega=np.zeros(1))
    old=AuditMixin._core_commands(view,o);new=MetricBudgetCore._core_commands(view,o);actual=view._core_commands(o)
    weight=np.clip((radius-8)/4,0,1)
    np.testing.assert_allclose(actual[0],old[0]+weight*(new[0]-old[0]),atol=1e-12)
    for j in [1,2]:np.testing.assert_array_equal(actual[j],old[j])
    assert np.isfinite(actual[0]).all() and (np.linalg.norm(actual[0],axis=-1)<=4+1e-8).all()
