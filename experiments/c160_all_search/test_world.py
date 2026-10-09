import copy
import unittest
import numpy as np
from experiments.c150_native_search.task import Task
from experiments.c151_response_guard.world import ResponseWorld
from experiments.c160_all_search.world import HiddenBranchWorld


class C154WorldTest(unittest.TestCase):
    def _set_branch(self, w, label):
        w.branch[:] = label
        w.branch_routes = w.routes.copy()
        for i in range(len(w.seeds)):
            theta = w._scene_rotation(i)
            w.branch_routes[i] = w.target[i] + (w.BRANCH_LOCAL[label] @ theta.T) * w.scale
        w.routes = w.branch_routes.copy()

    def test_shared_waypoint_is_external_and_scaled(self):
        seed = 16600001; scale = 1.6
        got = HiddenBranchWorld([seed], Task(scene='corner'), scale)
        base = ResponseWorld([seed], Task(scene='corner'), scale)
        d = base.routes[0, 0] - base.target[0]
        ey = d / np.linalg.norm(d); ex = np.array([ey[1], -ey[0]])
        rot = np.stack([ex, ey], axis=1)
        expected = base.target[0] + (np.array([0., 4.]) @ rot.T) * scale
        np.testing.assert_allclose(got.branch_routes[0, 0], expected, atol=1e-7)

    def test_different_continuations_are_observation_equivalent_before_switch(self):
        a = HiddenBranchWorld([16600001], Task(scene='corner'), 1.6)
        b = HiddenBranchWorld([16600001], Task(scene='corner'), 1.6)
        self._set_branch(a, 0); self._set_branch(b, 1)
        for tick in range(24):
            for x, y in ((a, b),):
                x.time = round(tick * .1, 8); x.cached = None
                y.time = x.time; y.cached = None
                ox, oy = x.observe(), y.observe()
                np.testing.assert_allclose(ox['p'], oy['p'], atol=1e-9)
                np.testing.assert_equal(ox['measured'], oy['measured'])
                np.testing.assert_allclose(x.commands(ox)[0], y.commands(oy)[0], atol=1e-8)
                x.step(ox); y.step(oy)
            if a.route_index[0] >= 1: break
        self.assertEqual(int(a.route_index[0]), 0)
        self.assertEqual(int(b.route_index[0]), 0)

    def test_registered_switch_is_inside_team_loss(self):
        seeds = np.arange(16600001, 16600009)
        w = HiddenBranchWorld(seeds, Task(scene='corner'), 1.6)
        measured = []; switches = []
        for tick in range(70):
            w.time = round(tick * .1, 8); w.cached = None
            o = w.observe(); measured.append(o['measured'].copy())
            before = w.route_index.copy(); w.step(o)
            switches.append((before == 0) & (w.route_index >= 1))
        self.assertTrue(np.all(w.route_index >= 1))
        self.assertGreaterEqual(float(w.branch_switch_hidden.mean()), .75)
        self.assertTrue(np.any(np.asarray(switches)))


if __name__ == '__main__': unittest.main()
