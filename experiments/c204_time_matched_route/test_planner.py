import numpy as np
from experiments.c184_route_recovery.planner import visible_edges
from experiments.c195_sector_return.planner import sector_candidates
from experiments.c203_phased_budget.world import make_world as old_world
from experiments.c204_time_matched_route.world import make_world
from experiments.c204_time_matched_route.planner import HORIZONS,time_matched_plan


def positions(radius=20.):
    a=np.arange(3)*2*np.pi/3
    return radius*np.c_[np.cos(a),np.sin(a)]


def test_open_interception_deadline_replaces_too_short_lead():
    p=positions();center=np.zeros(2);velocity=np.array([3.2,0.])
    result=time_matched_plan(p,0,center,velocity,np.empty((0,2)),np.empty(0))
    assert result['deadline_met'] and result['horizon']>1.2
    assert result['eta']<=result['horizon']
    # In the balanced triangle, zero offset is the sole non-increasing option.
    for h in HORIZONS[HORIZONS<result['horizon']-1e-8]:
        assert np.linalg.norm(center+h*velocity+np.array([6.,0.])-p[0])/4>h
    np.testing.assert_allclose(result['goal'],center+result['horizon']*velocity+[6.,0.],atol=1e-12)


def test_out_of_horizon_case_is_explicitly_not_reachable():
    p=positions(45.)
    result=time_matched_plan(p,0,np.zeros(2),np.array([-3.2,0.]),np.empty((0,2)),np.empty(0))
    assert result is not None and not result['deadline_met']
    assert result['eta']>result['horizon'] and result['horizon']<=6.


def test_routed_segments_and_sector_offsets_remain_legal():
    p=positions();center=np.zeros(2);velocity=np.array([0.,3.2])
    obs=np.array([[12.,2.],[8.,8.]])
    radii=np.array([2.,1.5])
    result=time_matched_plan(p,0,center,velocity,obs,radii)
    assert result is not None
    path=np.r_[p[0][None],result['path']]
    for a,b in zip(path[:-1],path[1:]):
        assert visible_edges(a[None],b[None],np.r_[obs,center[None]],np.r_[radii+1.25,3.2])[0,0]
    offsets,_=sector_candidates(p,0,center)
    assert np.min(abs(offsets-result['offset']))<1e-10


def test_unseen_and_before_withdrawal_commands_unchanged_and_logged():
    kw=dict(n_pursuers=5,method='active_polar',n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')
    a=old_world('random_test',[20151003],recovery='phased_budget_sector',**kw)
    b=make_world('random_test',[20151003],recovery='time_matched_sector',**kw)
    for x,y in zip(a.commands(a.observe()),b.commands(b.observe())):np.testing.assert_array_equal(x,y)
    for w in [a,b]:w.withdraw(1)
    oa=a.observe();ob=b.observe()
    for o in [oa,ob]:o['measured'][:]=False
    for x,y in zip(a.commands(oa),b.commands(ob)):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(b.recovery_planner.audit_last['p'],ob['p'])
    assert not b.recovery_planner.bypass_applied.any()
