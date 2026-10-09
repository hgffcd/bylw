import numpy as np

from experiments.c150_native_search.task import Task
from experiments.c179_random_map_generalization.world import RandomMapWorld, capture_ring_feasible


def test_random_train_and_test_maps_are_legal_and_distinct():
    seeds = np.arange(17900000, 17900032)
    train = RandomMapWorld(seeds, Task(scene="corner"), scale=1.6, split="train")
    test = RandomMapWorld(seeds, Task(scene="corner"), scale=1.6, split="test")
    assert train.obstacles.shape == (32, 4, 2)
    assert test.obstacles.shape == (32, 4, 2)
    assert np.isfinite(train.obstacles).all()
    assert np.isfinite(test.obstacles).all()
    assert not np.allclose(train.obstacles, test.obstacles)
    assert train.branch_metadata()["map_split"] == "train"
    assert test.branch_metadata()["map_split"] == "test"


def test_random_maps_keep_initial_team_observation_and_clearance():
    seeds = np.arange(17910000, 17910016)
    world = RandomMapWorld(seeds, Task(scene="corner"), scale=1.6, split="test")
    mask = world.sensor()[0]
    assert mask.any(axis=1).all()
    target_clearance = (
        np.linalg.norm(world.target[:, None] - world.obstacles, axis=-1) - world.radii
    ).min()
    pursuer_clearance = (
        np.linalg.norm(world.p[:, :, None] - world.obstacles[:, None], axis=-1)
        - world.radii[:, None]
    ).min()
    assert target_clearance > 0.5
    assert pursuer_clearance > 0.5


def test_density_profiles_keep_analytic_layout_and_radius_contract():
    seeds = np.arange(17920000, 17920004)
    for count in (6, 8):
        world = RandomMapWorld(
            seeds, Task(scene="corner"), scale=1.6, split="test", n_obstacles=count
        )
        assert world.obstacles.shape == (4, count, 2)
        assert world.radii.shape == (4, count)
        assert np.allclose(world.radii[:, 0], 4.8)
        assert np.allclose(world.radii[:, 1:], 1.92)
        assert world.branch_metadata()["obstacle_count"] == count


def test_mixed_shape_profile_keeps_conservative_box_envelopes():
    seeds = np.arange(17925000, 17925004)
    world = RandomMapWorld(
        seeds,
        Task(scene="corner"),
        scale=1.6,
        split="test",
        n_obstacles=6,
        shape_profile="mixed",
    )
    kinds, local_radii, boxes = __import__(
        "experiments.c179_random_map_generalization.world",
        fromlist=["obstacle_shape_profile"],
    ).obstacle_shape_profile(6, "mixed")
    assert kinds == ("cylinder", "box", "cylinder", "box", "cylinder", "box")
    assert world.branch_metadata()["shape_profile"] == "mixed"
    assert np.allclose(world.radii[:, 0], local_radii[0] * 1.6)
    for radius, (size_x, size_y) in zip(world.radii[0], boxes):
        if size_x > 0.0:
            assert radius + 1.0e-6 >= 0.5 * np.hypot(size_x * 1.6, size_y * 1.6)


def test_route_dense_profile_increases_route_relevant_obstacles():
    seeds = np.arange(17926000, 17926008)
    dense = RandomMapWorld(
        seeds,
        Task(scene="corner"),
        scale=1.6,
        split="test",
        n_obstacles=6,
        shape_profile="mixed",
        layout_profile="route_dense",
        initial_ring_screen=True,
    )
    dense_count = (
        np.linalg.norm(dense.obstacles - dense.target[:, None, :], axis=-1) < 20.0
    ).sum(axis=1).mean()
    assert dense.branch_metadata()["layout_profile"] == "route_dense"
    assert dense_count >= 3.0
    assert all(
        capture_ring_feasible(obstacles, radii, target)
        for obstacles, radii, target in zip(dense.obstacles, dense.radii, dense.target)
    )


def test_default_sampler_does_not_discard_maps_without_sampled_triangle(monkeypatch):
    import experiments.c179_random_map_generalization.world as module
    seeds = [17926000, 17926001]
    original = RandomMapWorld(seeds, Task(scene="corner"), scale=1.6, n_obstacles=6)
    monkeypatch.setattr(module, "capture_ring_feasible", lambda *args: False)
    diagnostic = RandomMapWorld(seeds, Task(scene="corner"), scale=1.6, n_obstacles=6)
    np.testing.assert_array_equal(diagnostic.obstacles, original.obstacles)
    assert not diagnostic.initial_ring_sample_found.any()
    assert diagnostic.ring_rejections.sum() == 0
    assert diagnostic.branch_metadata()["initial_ring_screen"] is False
