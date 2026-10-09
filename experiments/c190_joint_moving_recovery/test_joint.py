import numpy as np
import pytest

from experiments.c184_route_recovery.planner import visible_edges
from experiments.c190_joint_moving_recovery.planner import plan_joint, JointMovingPlanner
from experiments.c190_joint_moving_recovery.world import make_world


@pytest.mark.parametrize('n', [3, 4])
def test_joint_ring_and_every_routed_segment_clear_obstacles(n):
    angles = np.arange(n)*2*np.pi/n
    p = 11*np.c_[np.cos(angles), np.sin(angles)]
    obs = np.array([[8., 0.], [0., 8.]])
    radii = np.array([1., 1.])
    plan = plan_joint(p, np.zeros(2), obs, radii)
    assert plan is not None and any(len(path)>1 for path in plan['paths'])
    offsets = plan['offsets']
    rr = np.linalg.norm(offsets, axis=1)
    aa = np.sort(np.arctan2(offsets[:, 1], offsets[:, 0]))
    gaps = np.diff(np.r_[aa, aa[0]+2*np.pi])
    assert rr.min()>=4 and rr.max()<=8 and np.ptp(rr)<=2
    np.testing.assert_allclose(gaps, 2*np.pi/n, atol=1e-7)
    centers, envelope = np.r_[obs, [[0., 0.]]], np.r_[radii+1.4, 3.4]
    for start, path in zip(p, plan['paths']):
        chain = np.r_[start[None], path]
        for a, b in zip(chain[:-1], chain[1:]):
            assert visible_edges(a[None], b[None], centers, envelope)[0, 0]


@pytest.mark.parametrize('candidate', ['moving_joint', 'moving_joint_v2'])
def test_no_withdrawal_does_not_change_original_commands_or_map(candidate):
    worlds = [make_world('random_test', [18912000, 18912001], n_pursuers=5,
        n_obstacles=6, shape_profile='mixed', layout_profile='route_dense', recovery=r)
        for r in ('routed_transit', candidate)]
    np.testing.assert_array_equal(worlds[0].obstacles, worlds[1].obstacles)
    for tick in range(8):
        for w in worlds:
            w.time = tick*.1
        a, b = [w.commands(w.observe()) for w in worlds]
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


def test_loss_discards_recovery_delta_and_returns_search_commands():
    w = make_world('random_test', [18912000], n_pursuers=5)
    w.withdraw(0)
    o = dict(w.observe(), measured=np.array([False]), target_v=np.array([[3., 0.]]))
    planner = JointMovingPlanner(1, 5)
    planner.smooth_delta[:] = 2
    planner.joint_on[:] = True
    base = np.ones((1, 5, 2)); base[:, 0] = 0
    result, changed = planner.propose(o, base, .1)
    np.testing.assert_array_equal(result, base)
    assert not changed.any() and not planner.joint_on.any()
