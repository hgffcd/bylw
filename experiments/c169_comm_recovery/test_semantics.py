import unittest

import numpy as np

from experiments.c150_native_search.task import Task
from experiments.c169_comm_recovery.world import make_world


class CommunicationInterruptionSemanticsTest(unittest.TestCase):
    """The registered outage is temporary information loss, not UAV exit."""

    def test_episode_and_all_three_agents_continue_through_fault(self):
        seeds = np.arange(16930000, 16930004)
        world = make_world(
            "corner",
            seeds,
            Task(scene="corner", duration=6),
            1.6,
            "recovery_share",
            1.0,
            2.0,
        )
        times = []
        active = []
        masks = []
        roles = []
        for _ in range(36):
            obs = world.observe()
            times.append(world.time)
            active.append(world.comm_fault_active)
            masks.append(obs["mask"].copy())
            roles.append(world.role_modes.copy())
            world.step(obs)

        # The episode clock advances through the full 2 s window.
        self.assertAlmostEqual(times[30] - times[10], 2.0, places=6)
        self.assertTrue(all(active[10:30]))
        self.assertFalse(active[30])
        # Shared measurements are suppressed, while all three agents switch
        # to the registered recovery/search mode; no agent is removed.
        self.assertFalse(np.asarray(masks[10:30]).any())
        self.assertTrue(np.all(np.asarray(roles[10:30]) == 2))
        self.assertTrue(np.all(np.asarray(roles[30]) == 1))
        self.assertEqual(world.p.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
