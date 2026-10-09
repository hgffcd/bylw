import numpy as np
import pytest
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c204_time_matched_route.planner import TimeMatchedBypassPlanner
from experiments.c207_time_only.world import make_world,TimeOnlyWorld


def make(agents,arm):
    return make_world('random_test',[20151003],n_pursuers=agents,recovery=arm,method='active_polar',
        n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')


@pytest.mark.parametrize('agents',[4,5])
def test_before_fault_map_opponent_and_commands_equal(agents):
    a=make(agents,'sector_bypass');b=make(agents,'time_only_sector')
    for key in ['p','target','z','active','obstacles','radii','routes']:
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    for x,y in zip(a.commands(a.observe()),b.commands(b.observe())):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(a.evader_command(),b.evader_command())
    assert isinstance(b.recovery_planner,TimeMatchedBypassPlanner)


@pytest.mark.parametrize('agents,retired',[(4,[1]),(5,[1]),(5,[1,3])])
@pytest.mark.parametrize('observed',[True,False])
def test_compacted_core_exactly_matches_c195(agents,retired,observed):
    a=make(agents,'sector_bypass');b=make(agents,'time_only_sector')
    for w in [a,b]:
        for j in retired:w.withdraw(j)
    old,o,_=a._compact(a.observe());new,_,_=b._compact(b.observe())
    assert isinstance(new,TimeOnlyWorld)
    o=dict(o,measured=np.array([observed]),searcher=np.array([-1 if observed else 0]),age=np.array([0. if observed else 2.]))
    for x,y in zip(old._core_commands(o),new._core_commands(o)):np.testing.assert_array_equal(x,y)
    assert TimeOnlyWorld._core_commands is AuditMixin._core_commands
    b.time=.1
    command=b.commands(b.observe())[0]
    assert (command[:,retired]==0).all()
    assert np.isfinite(command).all() and (np.linalg.norm(command,axis=-1)<=4+1e-8).all()
