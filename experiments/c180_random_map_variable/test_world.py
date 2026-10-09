import numpy as np

from experiments.c150_native_search.task import Task
from experiments.c180_random_map_variable.world import RandomVariableCornerWorld


def test_random_variable_world_supports_four_to_three_state():
    seeds = np.arange(18000000, 18000008)
    world = RandomVariableCornerWorld(seeds, Task(scene="corner"), scale=1.6, n_pursuers=4, split="test")
    assert world.p.shape == (8, 4, 2)
    assert world.active.all()
    assert world.sensor()[0].any(axis=1).all()
    world.observe()
    active = world.active.copy()
    active[:, 2] = False
    world.set_active(active)
    assert world.active.sum(axis=1).tolist() == [3] * 8


def test_random_variable_world_density_profiles_match_radii():
    seeds = np.arange(18001000, 18001004)
    for count in (6, 8):
        world = RandomVariableCornerWorld(
            seeds,
            Task(scene="corner"),
            scale=1.6,
            n_pursuers=4,
            split="test",
            n_obstacles=count,
        )
        assert world.obstacles.shape == (4, count, 2)
        assert np.allclose(world.radii[:, 0], 4.8)
        assert np.allclose(world.radii[:, 1:], 1.92)


def test_random_variable_world_supports_mixed_shapes():
    seeds = np.arange(18002000, 18002004)
    world = RandomVariableCornerWorld(
        seeds,
        Task(scene="corner"),
        scale=1.6,
        n_pursuers=4,
        split="test",
        n_obstacles=6,
        shape_profile="mixed",
    )
    assert world.obstacles.shape == (4, 6, 2)
    assert world.branch_metadata()["shape_profile"] == "mixed"
    assert np.allclose(world.radii[:, 0], 4.8)
    assert np.all(world.radii[:, 1] > 1.92)


def test_default_variable_sampler_preserves_maps_when_ring_witness_missing(monkeypatch):
    import experiments.c180_random_map_variable.world as module
    seeds = [18047000, 18047001]
    original = RandomVariableCornerWorld(seeds, scale=1.6, n_obstacles=6)
    monkeypatch.setattr(module, "capture_ring_feasible", lambda *args: False)
    diagnostic = RandomVariableCornerWorld(seeds, scale=1.6, n_obstacles=6)
    np.testing.assert_array_equal(diagnostic.obstacles, original.obstacles)
    assert not diagnostic.initial_ring_sample_found.any()
    assert diagnostic.ring_rejections.sum() == 0
    assert diagnostic.branch_metadata()["initial_ring_screen"] is False
