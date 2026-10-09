import unittest

import numpy as np

from experiments.c150_native_search.task import Task
from experiments.c171_variable_scale.geometry import active_geometry
from experiments.c171_variable_scale.world import fail_one, make_world


class VariableWorldTest(unittest.TestCase):
    def test_four_agent_world_and_failure_mask(self):
        seeds = np.arange(17110000, 17110004)
        world = make_world("corner", seeds, Task(scene="corner", duration=8), 1.6, 4)
        obs = world.observe()
        self.assertEqual(world.p.shape, (len(seeds), 4, 2))
        self.assertEqual(obs["mask"].shape, (len(seeds), 4))
        fail_one(world, 1)
        obs = world.observe()
        self.assertFalse(obs["mask"][:, 1].any())
        self.assertTrue(world.active[:, [0, 2, 3]].all())

    def test_five_agent_world_has_scale_aware_geometry(self):
        seeds = np.arange(17110010, 17110014)
        world = make_world("corridor", seeds, Task(scene="corridor", duration=8), 1.6, 5)
        valid, _, _, count, _ = active_geometry(
            world.p, world.z, world.target, world.tz, world.active
        )
        self.assertEqual(world.p.shape[1], 5)
        self.assertTrue(np.all(count == 5))
        self.assertEqual(valid.shape, (len(seeds),))


if __name__ == "__main__":
    unittest.main()
