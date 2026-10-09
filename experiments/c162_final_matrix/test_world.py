import unittest

import numpy as np

from experiments.c150_native_search.task import Task, visibility3
from experiments.c162_final_matrix.world import CorridorWorld, HiddenBranchWorld


class C162WorldTest(unittest.TestCase):
    def test_corner_keeps_registered_hidden_branch(self):
        seeds = np.arange(16200001, 16200009)
        world = HiddenBranchWorld(seeds, Task(scene="corner", duration=30), 1.6)
        for _ in range(300):
            obs = world.observe()
            world.step(obs)
        self.assertEqual(float(world.branch_reached.mean()), 1.0)
        self.assertGreaterEqual(float(world.branch_switch_hidden.mean()), 0.75)

    def test_corridor_initial_measurement_and_clearance(self):
        seeds = np.arange(16200001, 16200017)
        world = CorridorWorld(seeds, Task(scene="corridor", duration=40), 1.6)
        mask = visibility3(
            world.p,
            world.z,
            world.target,
            world.tz,
            world.yaw,
            world.obstacles,
            world.radii,
            world.cfg,
        )
        self.assertTrue(mask.any(1).all())
        self.assertGreater(
            float(
                (
                    np.linalg.norm(world.target[:, None] - world.obstacles, axis=-1)
                    - world.radii
                ).min()
            ),
            0.5,
        )

    def test_corridor_switch_is_hidden_and_passive(self):
        seeds = np.arange(16200001, 16200017)
        world = CorridorWorld(seeds, Task(scene="corridor", duration=40), 1.6)
        measured = []
        for _ in range(400):
            obs = world.observe()
            measured.append(obs["measured"].copy())
            world.step(obs)
        self.assertEqual(float(world.branch_reached.mean()), 1.0)
        self.assertEqual(float(world.branch_switch_hidden.mean()), 1.0)
        lost = ~np.asarray(measured)
        self.assertGreaterEqual(int((lost.sum(0) >= 2).sum()), 8)


if __name__ == "__main__":
    unittest.main()

