import unittest

import numpy as np

from .geometry import active_geometry, uniform_slots


def ring(count, radius=6.0, phase=0.0):
    a = phase + np.arange(count) * 2 * np.pi / count
    return np.stack([radius * np.cos(a), radius * np.sin(a)], axis=-1)


class ActiveGeometryTest(unittest.TestCase):
    def test_three_active_preserves_existing_criterion(self):
        p = ring(3)[None]
        valid, _, gaps, count, _ = active_geometry(
            p,
            np.full((1, 3), 3.0),
            np.zeros((1, 2)),
            np.array([3.0]),
            np.ones((1, 3), dtype=bool),
        )
        self.assertTrue(valid[0])
        self.assertEqual(count[0], 3)
        np.testing.assert_allclose(gaps[0, :3], 2 * np.pi / 3)

    def test_four_to_three_reconfiguration_excludes_failed_member(self):
        p = np.zeros((1, 4, 2), dtype=float)
        p[0, [0, 2, 3]] = ring(3)
        active = np.array([[True, False, True, True]])
        valid, _, _, count, _ = active_geometry(
            p,
            np.full((1, 4), 3.0),
            np.zeros((1, 2)),
            np.array([3.0]),
            active,
        )
        self.assertTrue(valid[0])
        self.assertEqual(count[0], 3)
        slots = uniform_slots(active)
        self.assertTrue(np.isnan(slots[0, 1]))
        np.testing.assert_allclose(np.diff(slots[0, [0, 2, 3]]), 2 * np.pi / 3)

    def test_less_than_three_active_is_not_closed_encirclement(self):
        p = ring(2)[None]
        valid, _, _, count, _ = active_geometry(
            p,
            np.full((1, 2), 3.0),
            np.zeros((1, 2)),
            np.array([3.0]),
            np.ones((1, 2), dtype=bool),
        )
        self.assertFalse(valid[0])
        self.assertEqual(count[0], 2)

    def test_four_agent_gap_tolerance_is_scale_dependent(self):
        p = ring(4)[None]
        p[0, 1] = np.array([6.0, 0.0])  # duplicate angle creates a bad gap
        valid, _, _, _, _ = active_geometry(
            p,
            np.full((1, 4), 3.0),
            np.zeros((1, 2)),
            np.array([3.0]),
            np.ones((1, 4), dtype=bool),
        )
        self.assertFalse(valid[0])


if __name__ == "__main__":
    unittest.main()
