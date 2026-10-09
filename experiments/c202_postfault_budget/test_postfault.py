import numpy as np
import pytest
from experiments.c175_multi_fault_reconfig.world import AuditMixin
from experiments.c200_feedforward_budget.world import BudgetRouteWorld
from experiments.c195_sector_return.planner import SectorBypassPlanner
from experiments.c202_postfault_budget.world import make_world, PostFaultBudgetWorld


def world(agents, arm):
    return make_world('random_test', [20151003], n_pursuers=agents, recovery=arm,
        method='active_polar', n_obstacles=6, shape_profile='mixed', layout_profile='route_dense')


@pytest.mark.parametrize('agents', [4, 5])
@pytest.mark.parametrize('parent,arm', [('routed_transit','postfault_budget'), ('sector_bypass','budget_sector')])
def test_before_withdrawal_and_initial_specs_identical(agents, parent, arm):
    a, b = world(agents, parent), world(agents, arm)
    for key in ['p','z','target','obstacles','radii','routes','yaw','active']:
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    oa, ob = a.observe(), b.observe()
    for x, y in zip(a.commands(oa), b.commands(ob)):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a.evader_command(), b.evader_command())


@pytest.mark.parametrize('retired', [[1], [1,3]])
def test_budget_dispatch_survives_active_compaction(retired):
    w=world(5,'budget_sector')
    for i in retired: w.withdraw(i)
    o=w.observe(); view, obs, ids=w._compact(o)
    assert isinstance(view,PostFaultBudgetWorld)
    assert isinstance(w.recovery_planner,SectorBypassPlanner)
    assert len(ids)==5-len(retired) and view.departed_p.shape==(1,len(retired),2)
    obs=dict(obs,measured=np.ones(1,bool))
    actual=view._core_commands(obs)
    expected=BudgetRouteWorld._core_commands(view,obs)
    for x,y in zip(actual,expected): np.testing.assert_array_equal(x,y)
    u,_,_=w.commands(o)
    assert (u[:,retired]==0).all()


def test_no_new_control_when_unobserved():
    w=world(5,'postfault_budget'); w.withdraw(1)
    view,obs,_=w._compact(w.observe())
    obs=dict(obs,measured=np.zeros(1,bool),searcher=np.zeros(1,int),age=np.ones(1))
    for x,y in zip(view._core_commands(obs),AuditMixin._core_commands(view,obs)):
        np.testing.assert_array_equal(x,y)


def test_observed_budget_changes_saturated_command_without_hidden_state():
    w=world(5,'postfault_budget'); w.withdraw(1)
    view,obs,_=w._compact(w.observe())
    obs=dict(obs,measured=np.ones(1,bool),searcher=np.full(1,-1,int),
             target_v=np.array([[3.2,0.]]),omega=np.zeros(1))
    # Compare core proposals with the same public input and safety treatment.
    a=view._core_commands(obs)[0]; b=AuditMixin._core_commands(view,obs)[0]
    assert not np.array_equal(a,b)
    assert (np.linalg.norm(a,axis=-1)<=4+1e-8).all()
