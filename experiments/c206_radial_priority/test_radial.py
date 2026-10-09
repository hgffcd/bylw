import numpy as np
import pytest
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c200_feedforward_budget.controller import allocate,budget_polar
from experiments.c204_time_matched_route.world import make_world as old_world
from experiments.c206_radial_priority.controller import radial_project,radial_priority_polar
from experiments.c206_radial_priority.budget_core import RadialBudgetCore
from experiments.c206_radial_priority.world import make_world,PhasedRadialWorld


def test_large_tangential_request_does_not_scale_radial_correction():
    feed=np.array([3.2,0.]);q=np.array([-.9,12.]);er=np.array([1.,0.])
    old,_=allocate(feed,q);new=radial_project(feed+q,er)
    assert abs(old[0]-(feed+q)[0])>.5
    np.testing.assert_allclose(new[0],2.3,atol=1e-12)
    assert np.linalg.norm(new)<=4+1e-12


def test_unconstrained_requests_and_zero_basis_are_well_defined():
    rng=np.random.default_rng(206);angle=rng.uniform(-np.pi,np.pi,500)
    er=np.c_[np.cos(angle),np.sin(angle)];nominal=rng.uniform(-2,2,(500,2))
    np.testing.assert_allclose(radial_project(nominal,er),nominal,atol=1e-12)
    np.testing.assert_allclose(radial_project([6.,8.],[0.,0.]),[2.4,3.2])


def test_speed_bound_and_radial_optimum_against_feasible_samples():
    rng=np.random.default_rng(20601);angle=rng.uniform(-np.pi,np.pi,300)
    er=np.c_[np.cos(angle),np.sin(angle)];nominal=rng.normal(size=(300,2))*10
    output=radial_project(nominal,er)
    assert (np.linalg.norm(output,axis=-1)<=4+1e-12).all()
    # Independent feasible directions cannot have less radial tracking error.
    angle=np.linspace(-np.pi,np.pi,2001);bank=4*np.c_[np.cos(angle),np.sin(angle)]
    desired=np.sum(nominal*er,axis=-1)
    best_sample=np.min(abs(bank@er.T-desired),axis=0)
    actual=abs(np.sum(output*er,axis=-1)-desired)
    assert (actual<=best_sample+1e-10).all()


def test_rotation_equivariance():
    a=.81;rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    nominal=np.array([8.,-3.]);er=np.array([.6,.8])
    np.testing.assert_allclose(radial_project(nominal@rot.T,er@rot.T),radial_project(nominal,er)@rot.T,atol=1e-12)


def make(agents,arm):
    return make_world('random_test',[20151003],n_pursuers=agents,recovery=arm,method='active_polar',
        n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')


@pytest.mark.parametrize('agents',[4,5])
def test_before_withdrawal_map_opponent_and_commands_are_identical(agents):
    a=make(agents,'time_matched_sector');b=make(agents,'radial_priority_sector')
    for key in ['p','target','z','active','obstacles','radii','routes']:
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    for x,y in zip(a.commands(a.observe()),b.commands(b.observe())):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(a.evader_command(),b.evader_command())


@pytest.mark.parametrize('retired',[[1],[1,3]])
def test_compaction_search_and_retired_commands_are_preserved(retired):
    w=make(5,'radial_priority_sector')
    for j in retired:w.withdraw(j)
    view,o,_=w._compact(w.observe());assert isinstance(view,PhasedRadialWorld)
    assert view.departed_p.shape==(1,len(retired),2)
    o=dict(o,measured=np.zeros(1,bool),searcher=np.zeros(1,int),age=np.ones(1))
    for x,y in zip(view._core_commands(o),AuditMixin._core_commands(view,o)):np.testing.assert_array_equal(x,y)
    w.time=.1
    assert (w.commands(w.observe())[0][:,retired]==0.).all()


@pytest.mark.parametrize('radius',[7.,10.,13.])
def test_old_near_core_blend_and_new_far_core_dispatch(radius):
    w=make(5,'radial_priority_sector');w.withdraw(1);view,o,_=w._compact(w.observe())
    d=o['p']-o['target'][:,None]
    o=dict(o,p=o['target'][:,None]+radius*d/np.linalg.norm(d,axis=-1)[...,None],
           measured=np.ones(1,bool),searcher=np.full(1,-1,int),target_v=np.array([[3.2,0.]]),omega=np.zeros(1))
    actual=view._core_commands(o);old=AuditMixin._core_commands(view,o);new=RadialBudgetCore._core_commands(view,o)
    weight=np.clip((radius-8.)/4.,0,1)
    np.testing.assert_allclose(actual[0],old[0]+weight*(new[0]-old[0]),atol=1e-12)
    for i in [1,2]:np.testing.assert_array_equal(actual[i],old[i])
    assert np.isfinite(actual[0]).all() and (np.linalg.norm(actual[0],axis=-1)<=4+1e-8).all()


def test_feasible_nominal_polar_request_matches_uniform_allocator():
    w=make(4,'radial_priority_sector');w.withdraw(1);view,o,_=w._compact(w.observe())
    angle=np.arange(3)*2*np.pi/3;p=o['target'][:,None]+6.*np.c_[np.cos(angle),np.sin(angle)][None]
    args=(p,o['target'],np.array([[2.,0.]]),view.control_radii,view.cfg,np.zeros((1,1,2)))
    np.testing.assert_allclose(radial_priority_polar(*args),budget_polar(*args),atol=1e-12)
