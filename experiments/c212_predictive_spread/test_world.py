import numpy as np
import pytest
from experiments.c212_predictive_spread.world import make_world, PredictiveSpreadWorld
from experiments.c210_radial_damping.world import RadialDampedWorld
from experiments.c210_radial_damping.formation_core import DampedFormationCore
from experiments.c212_predictive_spread.formation_core import PredictiveFormationCore


def make(agents,arm):
    return make_world('random_test',[20151003],n_pursuers=agents,recovery=arm,method='active_polar',
        n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')


@pytest.mark.parametrize('agents',[4,5])
def test_prefault_identical_to_direct_parent(agents):
    a=make(agents,'radial_damped_sector');b=make(agents,'predictive_spread_sector')
    for key in ['p','target','z','active','obstacles','radii','routes']:
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    for t in [0,.1,.1,.2]:
        a.time=b.time=t
        for x,y in zip(a.commands(a.observe()),b.commands(b.observe())):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(a.evader_command(),b.evader_command())


@pytest.mark.parametrize('retired',[[1],[1,3]])
def test_acceleration_history_compacts_active_ids_and_repeated_queries(retired):
    w=make(5,'predictive_spread_sector');o=w.observe();w.commands(o)
    assert not w._response_history['valid']
    w.time=.1;v=o['v']+np.arange(5)[None,:,None]*.1
    w.commands(dict(o,v=v));a=w._response_history['acceleration'].copy()
    w.commands(dict(o,v=v));np.testing.assert_array_equal(w._response_history['acceleration'],a)
    for j in retired:w.withdraw(j)
    obs=dict(w.observe(),own_acceleration=a,acceleration_valid=np.ones(1,bool))
    view,compact,ids=w._compact(obs);assert isinstance(view,PredictiveSpreadWorld)
    np.testing.assert_array_equal(compact['own_acceleration'],a[:,ids])
    assert compact['p'].shape==compact['own_acceleration'].shape
    assert (w.commands(w.observe())[0][:,retired]==0).all()
    w.time=.4;w.commands(w.observe());assert not w._response_history['valid']


@pytest.mark.parametrize('disabled',['hidden','search','far','no_history'])
def test_formation_core_exact_parent_when_inactive(disabled):
    w=make(4,'predictive_spread_sector');w.withdraw(0);view,o,_=w._compact(w.observe())
    angles=np.arange(3)*2*np.pi/3;er=np.stack([np.cos(angles),np.sin(angles)],-1)[None]
    o=dict(o,p=o['target'][:,None]+(13 if disabled=='far' else 6)*er,
        measured=np.ones(1,bool),searcher=np.full(1,-1),own_acceleration=er*2,acceleration_valid=np.ones(1,bool))
    if disabled=='hidden':o['measured'][:]=False
    if disabled=='search':o['searcher'][:]=0
    if disabled=='no_history':o['acceleration_valid'][:]=False
    for x,y in zip(PredictiveFormationCore._core_commands(view,o),DampedFormationCore._core_commands(view,o)):
        np.testing.assert_array_equal(x,y)


def test_recorded_acceleration_is_causal_and_same_tick_stable():
    w=make(4,'predictive_spread_sector');w.commands(w.observe())
    first=w.recovery_planner.audit_last
    assert not first['acceleration_valid'].any()
    w.time=.1;w.commands(w.observe())
    current=w.recovery_planner.audit_last
    assert current['acceleration_valid'].all()
    assert current['own_acceleration'].shape==(1,4,2)
    np.testing.assert_array_equal(current['own_acceleration'],np.zeros((1,4,2)))
