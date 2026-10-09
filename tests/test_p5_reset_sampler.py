"""CPU gates for the opt-in P5-1R tracked-ring reset."""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_SOURCE = PROJECT_ROOT / "isaac_p5" / "source" / "bylw_isaac_p5"
sys.path.insert(0, str(EXTENSION_SOURCE))

from bylw_isaac_p5.core.reset_sampler import (
    sample_tracked_ring_reset,
    segment_circle_intersects,
)


OBSTACLE_XY = np.asarray(
    [
        (-18.0, -15.0),
        (-18.0, 15.0),
        (-4.0, -18.0),
        (-4.0, 18.0),
        (10.0, -15.0),
        (10.0, 15.0),
        (25.0, -12.0),
        (25.0, 12.0),
    ]
)
OBSTACLE_RADIUS = np.asarray([3.0, 3.0, 4.0, 4.0, 3.5, 3.5, 3.0, 3.0])


def test_tracked_ring_reset_is_seed_deterministic_and_seed_sensitive():
    first = sample_tracked_ring_reset(20262001, OBSTACLE_XY, OBSTACLE_RADIUS)
    repeated = sample_tracked_ring_reset(20262001, OBSTACLE_XY, OBSTACLE_RADIUS)
    different = sample_tracked_ring_reset(20262002, OBSTACLE_XY, OBSTACLE_RADIUS)

    assert np.array_equal(first.target_xy_m, repeated.target_xy_m)
    assert np.array_equal(first.pursuer_xy_m, repeated.pursuer_xy_m)
    assert not np.array_equal(first.target_xy_m, different.target_xy_m)


@pytest.mark.parametrize("seed", range(20262001, 20262033))
def test_tracked_ring_reset_satisfies_visibility_and_clearance_contract(seed: int):
    geometry = sample_tracked_ring_reset(seed, OBSTACLE_XY, OBSTACLE_RADIUS)
    target = geometry.target_xy_m
    pursuers = geometry.pursuer_xy_m

    assert target.shape == (2,)
    assert pursuers.shape == (3, 2)
    assert geometry.pursuer_yaw_rad.shape == (3,)
    assert np.all(np.abs(target) <= 49.0)
    assert np.all(np.abs(pursuers) <= 49.0)

    distances = np.linalg.norm(pursuers - target[None, :], axis=1)
    assert np.all(distances > 6.0)
    assert np.all(distances <= 25.0 + 1.0e-10)
    for index in range(3):
        bearing = math.atan2(
            target[1] - pursuers[index, 1],
            target[0] - pursuers[index, 0],
        )
        assert geometry.pursuer_yaw_rad[index] == pytest.approx(bearing)
        assert not any(
            segment_circle_intersects(
                pursuers[index], target, center, obstacle_radius
            )
            for center, obstacle_radius in zip(OBSTACLE_XY, OBSTACLE_RADIUS)
        )
        assert all(
            np.linalg.norm(pursuers[index] - center)
            > obstacle_radius + 1.0
            for center, obstacle_radius in zip(OBSTACLE_XY, OBSTACLE_RADIUS)
        )

    pair_distances = [
        np.linalg.norm(pursuers[left] - pursuers[right])
        for left in range(3)
        for right in range(left + 1, 3)
    ]
    assert min(pair_distances) >= 2.5
    assert geometry.maximum_angular_gap_rad <= math.pi + 1.0e-10


def test_tracked_ring_reset_rejects_invalid_contract():
    with pytest.raises(ValueError, match="sense_range_m"):
        sample_tracked_ring_reset(
            1,
            OBSTACLE_XY,
            OBSTACLE_RADIUS,
            sense_range_m=6.0,
            capture_radius_m=6.0,
        )


def test_p5_1r_source_keeps_legacy_default_and_dense_physical_telemetry():
    task_dir = (
        EXTENSION_SOURCE
        / "bylw_isaac_p5"
        / "tasks"
        / "direct"
        / "bylw_pursuit"
    )
    cfg_path = task_dir / "bylw_pursuit_env_cfg.py"
    env_path = task_dir / "bylw_pursuit_env.py"
    runner_path = PROJECT_ROOT / "scripts" / "run_p5_1_zero_shot.py"
    pipeline_path = PROJECT_ROOT / "scripts" / "run_p5_1_pipeline.py"
    summarizer_path = PROJECT_ROOT / "scripts" / "summarize_p5_1_zero_shot.py"
    cfg_source = cfg_path.read_text(encoding="utf-8")
    env_source = env_path.read_text(encoding="utf-8")
    runner_source = runner_path.read_text(encoding="utf-8")
    pipeline_source = pipeline_path.read_text(encoding="utf-8")
    summarizer_source = summarizer_path.read_text(encoding="utf-8")
    ast.parse(cfg_source, filename=str(cfg_path))
    ast.parse(env_source, filename=str(env_path))
    ast.parse(runner_source, filename=str(runner_path))
    ast.parse(pipeline_source, filename=str(pipeline_path))
    ast.parse(summarizer_source, filename=str(summarizer_path))

    assert 'reset_layout = "legacy_fixed_cluster"' in cfg_source
    assert "dense_physical_telemetry = False" in cfg_source
    assert 'cfg.reset_layout == "tracked_ring"' in env_source
    assert "tracked_reset_reference_range_m = None" in cfg_source
    assert "self.cfg.tracked_reset_reference_range_m" in env_source
    assert "if self.cfg.dense_physical_telemetry:" in env_source
    for field_name in (
        "pursuer_position_local_m",
        "pursuer_velocity_mps",
        "pursuer_angular_velocity_rps",
        "applied_thrust_b_n",
        "applied_moment_b_nm",
        "desired_body_rate_rps",
    ):
        assert f'"{field_name}"' in env_source
        assert f'p5["{field_name}"]' in runner_source
    assert "--replay_pairs" in runner_source
    assert '"formal P5-1 is frozen at base=20261001, 32 envs, legacy reset"' in (
        pipeline_source
    )
    assert "P5_1R_DIAGNOSTIC_COMPLETE_REVIEW_NO_AUTO_TRAINING" in (
        summarizer_source
    )
