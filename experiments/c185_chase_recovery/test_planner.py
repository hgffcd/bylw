import numpy as np
from experiments.c180_random_map_variable.world import make_world
from experiments.c185_chase_recovery.planner import ChaseRecoveryPlanner


def observation():
    w=make_world('random_train',[18042001],n_obstacles=6)
    w.withdraw(0)
    o=w.observe()
    return dict(o, measured=np.array([True]),target=np.array([[0.,0.]]),
                target_v=np.array([[3.2,0.]]),p=np.array([[[0.,10.],[-15.,0.],[5.,3.],[5.,-3.]]]))


def test_far_agent_catches_up_without_touching_other_commands():
    o=observation();p=ChaseRecoveryPlanner(1,4);base=np.full((1,4,2),.1)
    u,changed=p.propose(o,base,0.)
    assert changed[0] and p.chase_on[0,1]
    np.testing.assert_allclose(u[0,1],[4.,0.])
    np.testing.assert_array_equal(u[0,[0,2,3]],base[0,[0,2,3]])


def test_hysteresis_and_same_timestamp_cache():
    o=observation();p=ChaseRecoveryPlanner(1,4);base=np.zeros((1,4,2))
    p.propose(o,base,0.);p.propose(o,base,0.)
    assert p.chase_entries[0,1]==1
    o['p'][0,1]=[-8.,0.];p.propose(o,base,.1)
    assert p.chase_on[0,1]
    o['p'][0,1]=[-6.9,0.];u,changed=p.propose(o,base,.2)
    assert not p.chase_on.any() and not changed.any()
    np.testing.assert_array_equal(u,base)


def test_loss_disables_chase_and_preserves_search_commands():
    o=observation();p=ChaseRecoveryPlanner(1,4);base=np.full((1,4,2),.1)
    p.propose(o,base,0.)
    o['measured'][:]=False
    u,changed=p.propose(o,base,.1)
    assert not changed.any() and not p.chase_on.any()
    np.testing.assert_array_equal(u,base)
