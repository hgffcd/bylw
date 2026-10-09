import numpy as np
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c200_feedforward_budget.world import BudgetRouteWorld
from experiments.c202_postfault_budget.test_postfault import world
from experiments.c203_phased_budget.world import make_world,team_weight,PhasedBudgetWorld


def view_and_input(radius):
    w=make_world('random_test',[20151003],n_pursuers=5,recovery='phased_budget_sector',method='active_polar',
        n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')
    w.withdraw(1);view,o,_=w._compact(w.observe())
    delta=o['p']-o['target'][:,None]
    p=o['target'][:,None]+radius*delta/np.linalg.norm(delta,axis=-1)[...,None]
    return view,dict(o,p=p,measured=np.ones(1,bool),searcher=np.full(1,-1,int))


def test_before_fault_identical_to_combined_parent():
    a=world(5,'budget_sector')
    b=make_world('random_test',[20151003],n_pursuers=5,recovery='phased_budget_sector',method='active_polar',
        n_obstacles=6,shape_profile='mixed',layout_profile='route_dense')
    for x,y in zip(a.commands(a.observe()),b.commands(b.observe())):np.testing.assert_array_equal(x,y)


def test_near_ring_returns_exact_old_core_and_far_returns_exact_budget():
    for radius,expected in [(7.,AuditMixin),(13.,BudgetRouteWorld)]:
        view,o=view_and_input(radius)
        assert isinstance(view,PhasedBudgetWorld)
        for x,y in zip(view._core_commands(o),expected._core_commands(view,o)):
            np.testing.assert_array_equal(x,y)


def test_same_team_blend_with_unchanged_yaw_altitude_and_speed_bound():
    view,o=view_and_input(10.)
    old=AuditMixin._core_commands(view,o);new=BudgetRouteWorld._core_commands(view,o)
    actual=view._core_commands(o)
    np.testing.assert_allclose(team_weight(o),.5,atol=1e-12)
    np.testing.assert_allclose(actual[0],.5*(old[0]+new[0]),atol=1e-12)
    for i in [1,2]:np.testing.assert_array_equal(actual[i],old[i])
    assert (np.linalg.norm(actual[0],axis=-1)<=4+1e-8).all()


def test_hidden_state_cannot_activate_new_budget():
    view,o=view_and_input(20.);o=dict(o,measured=np.zeros(1,bool))
    assert (team_weight(o)==0).all()
    for x,y in zip(view._core_commands(o),AuditMixin._core_commands(view,o)):
        np.testing.assert_array_equal(x,y)
