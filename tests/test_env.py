# -*- coding: utf-8 -*-
"""环境层单元测试(纯numpy, 云端/本机均可运行): pytest tests -q"""
import math
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.pursuit_lost_env import EnvConfig, PursuitLostEnv
from utils.geometry import segment_circle_intersect, sorted_gap_max, wrap_angle


def test_geometry():
    assert segment_circle_intersect([0, 0], [10, 0], [5, 0.5], 1.0)
    assert not segment_circle_intersect([0, 0], [10, 0], [5, 3.0], 1.0)
    assert abs(wrap_angle(3 * math.pi) - math.pi) < 1e-9 or abs(wrap_angle(3 * math.pi) + math.pi) < 1e-9
    pts = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)
    assert abs(sorted_gap_max(pts, np.zeros(2)) - math.pi / 2) < 1e-6


def test_reset_shapes():
    env = PursuitLostEnv(EnvConfig(seed=0))
    obs, state = env.reset()
    assert obs.shape == (env.n, env.obs_dim)
    assert state.shape == (env.state_dim,)
    assert np.isfinite(obs).all() and np.isfinite(state).all()


def test_random_rollout_finite():
    env = PursuitLostEnv(EnvConfig(seed=1, n_dynamic_obs=2))
    obs, state = env.reset()
    total = 0.0
    for _ in range(300):
        act = np.random.uniform(-1, 1, (env.n, 2))
        obs, state, rew, done, info = env.step(act)
        assert np.isfinite(obs).all() and np.isfinite(rew).all()
        total += rew[0]
        if done:
            obs, state = env.reset()
    assert np.isfinite(total)


def test_visibility_and_lost_counter():
    cfg = EnvConfig(seed=2, sense_range=5.0, fov_half=0.5, n_static_obs=6)
    env = PursuitLostEnv(cfg)
    env.reset()
    seen_lost = False
    for _ in range(200):
        _, _, _, done, info = env.step(np.zeros((env.n, 2)))
        if not info["visible"]:
            seen_lost = True
            assert info["t_lost"] >= 1
        if done:
            env.reset()
    assert seen_lost, "视场收窄后应出现失观事件"


def test_obs_modes():
    for mode in ("oracle", "zero", "last"):
        env = PursuitLostEnv(EnvConfig(seed=3, obs_mode=mode))
        obs, _ = env.reset()
        for _ in range(50):
            obs, _, _, done, _ = env.step(np.random.uniform(-1, 1, (env.n, 2)))
            assert np.isfinite(obs).all()
            if done:
                env.reset()


def test_team_target_sharing_broadcasts_current_measurement_without_changing_local_visibility():
    env = PursuitLostEnv(EnvConfig(
        seed=20260821,
        obs_mode="last",
        n_static_obs=0,
        n_dynamic_obs=0,
        team_target_sharing=False,
    ))
    env.p_pos = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    env.visible_each = np.array([True, False, False])
    env.target_measurement_pos = np.array([60.0, 55.0])
    env.last_seen_pos = np.array([45.0, 45.0])
    env.t_vel = np.array([2.0, -4.0])
    env.t_lost = 0

    legacy = env._target_block(1)
    assert legacy[4] == -1.0
    assert np.allclose(legacy[2:4], 0.0)
    assert np.allclose(legacy[:2], [0.25, 0.25])

    env.cfg.team_target_sharing = True
    shared = env._target_block(1)
    assert shared[4] == 1.0
    assert np.allclose(shared[:2], [0.40, 0.35])
    assert np.allclose(shared[2:4], [0.25, -0.50])
    assert np.array_equal(env.visible_each, [True, False, False])


def test_capture_success_geometry():
    env = PursuitLostEnv(EnvConfig(seed=4, n_pursuers=3))
    env.reset()
    env.t_pos = np.array([50.0, 50.0])
    r = env.cfg.capture_radius * 0.8
    env.p_pos = np.array([[50 + r, 50], [50 - r * 0.5, 50 + r * 0.87], [50 - r * 0.5, 50 - r * 0.87]])
    assert env._capture_success()
    env.p_pos = np.array([[50 + r, 50], [50 + r * 0.9, 50 + 1], [50 + r * 0.9, 50 - 1]])
    assert not env._capture_success()  # 同侧聚集, 角度缺口过大


def test_reward_target_radius_is_independent_from_capture_boundary():
    env = PursuitLostEnv(EnvConfig(
        seed=20260819,
        n_static_obs=0,
        n_dynamic_obs=0,
        capture_radius=6.0,
        reward_target_radius=5.0,
        reward_target_reference="truth",
    ))
    env.t_pos = np.array([50.0, 50.0])
    angles = np.arange(3) * 2.0 * math.pi / 3.0

    env.p_pos = env.t_pos + 5.0 * np.column_stack(
        (np.cos(angles), np.sin(angles)))
    assert env._capture_success()
    assert env._potential() == pytest.approx(0.0, abs=1e-12)

    env.p_pos = env.t_pos + 5.5 * np.column_stack(
        (np.cos(angles), np.sin(angles)))
    assert env._capture_success()
    assert env._potential() == pytest.approx(-0.5, abs=1e-12)


def test_reward_target_radius_default_and_validation():
    legacy = PursuitLostEnv(EnvConfig(
        seed=20260820, capture_radius=6.0,
        reward_target_reference="truth"))
    assert legacy._reward_ring_radius() == 6.0
    with pytest.raises(ValueError, match="reward_target_radius"):
        PursuitLostEnv(EnvConfig(reward_target_radius=0.0))


def test_safety_layer_numpy():
    from safety.cbf_qp import CBFSafetyLayer
    layer = CBFSafetyLayer()
    pos = np.zeros(2)
    # 正前方近距障碍, 名义动作全速前冲 -> 应被修正
    u, corrected = layer.project(np.array([1.0, 0.0]), pos, 0.0, 4.0,
                                 neighbors=[], obstacles=[(np.array([2.5, 0.0]), 1.0, np.zeros(2))])
    assert corrected
    assert u[0] < 1.0  # 加速度被压低
    # 无威胁 -> 不修正
    u2, c2 = layer.project(np.array([0.5, 0.1]), pos, 0.0, 2.0, [], [])
    assert not c2 and np.allclose(u2, [0.5, 0.1])


def test_j3_defaults_preserve_legacy_path_bitwise():
    env = PursuitLostEnv(EnvConfig(
        seed=123, n_static_obs=8, n_dynamic_obs=2))
    obs, state = env.reset(seed=123)
    actions = np.array([[0.2, -0.1], [0.0, 0.3], [-0.4, 0.5]])
    obs2, state2, reward, _, _ = env.step(actions)
    payload = b"".join(
        np.asarray(value).tobytes()
        for value in (
            env.obs_pos, env.obs_r, env.obs_vel, obs, state,
            obs2, state2, reward, env.p_pos, env.t_pos,
        )
    )
    assert hashlib.sha256(payload).hexdigest() == (
        "a4407715d15b9f14a36458e092c35618e"
        "ec995334714a6d06182a50c2c49ec34"
    )


def test_j3_structured_layouts_are_valid_and_distinct():
    snapshots = []
    for layout in ("corridor", "cluster", "narrow_passage"):
        env = PursuitLostEnv(EnvConfig(
            seed=31,
            obstacle_layout=layout,
            n_static_obs=8,
            n_dynamic_obs=2,
        ))
        assert env.obs_pos.shape == (10, 2)
        assert env.obs_r.shape == (10,)
        assert np.allclose(env.obs_vel[:8], 0.0)
        assert np.linalg.norm(env.obs_vel[8:], axis=1).min() > 0
        for i in range(len(env.obs_pos)):
            for j in range(i + 1, len(env.obs_pos)):
                assert np.linalg.norm(
                    env.obs_pos[i] - env.obs_pos[j]
                ) > env.obs_r[i] + env.obs_r[j]
        snapshots.append(env.obs_pos[:8].copy())
    assert not np.array_equal(snapshots[0], snapshots[1])
    assert not np.array_equal(snapshots[1], snapshots[2])


def test_j3_target_noise_does_not_perturb_true_dynamics_rng():
    clean = PursuitLostEnv(EnvConfig(
        seed=37, obs_mode="oracle", target_position_noise_std_m=0.0))
    noisy = PursuitLostEnv(EnvConfig(
        seed=37, obs_mode="oracle", target_position_noise_std_m=1.0))
    clean_obs, _ = clean.reset(seed=37)
    noisy_obs, _ = noisy.reset(seed=37)
    assert np.array_equal(clean.obs_pos, noisy.obs_pos)
    assert np.array_equal(clean.p_pos, noisy.p_pos)
    assert np.array_equal(clean.t_pos, noisy.t_pos)
    assert not np.array_equal(clean_obs, noisy_obs)
    actions = np.zeros((clean.n, 2))
    for _ in range(10):
        clean_obs, *_ = clean.step(actions)
        noisy_obs, *_ = noisy.step(actions)
        assert np.array_equal(clean.p_pos, noisy.p_pos)
        assert np.array_equal(clean.t_pos, noisy.t_pos)
        assert np.array_equal(clean.obs_pos, noisy.obs_pos)
    assert not np.array_equal(clean_obs, noisy_obs)


def test_measurement_blackout_is_opt_in_and_causally_overrides_visibility():
    base = dict(
        seed=20260817,
        n_static_obs=0,
        n_dynamic_obs=0,
        sense_range=200.0,
        fov_half=math.pi,
        visual_occlusion=False,
        require_initial_visibility=True,
        visibility_diagnostics=True,
        measurement_blackout_start_step=2,
        measurement_blackout_duration_steps=3,
    )
    env = PursuitLostEnv(EnvConfig(**base))
    env.reset(seed=20260817)
    assert env.visible_each.any()
    statuses = []
    for _ in range(6):
        _, _, _, _, info = env.step(np.zeros((env.n, 2)))
        statuses.append((
            info["visible"], info["measurement_blackout_active"],
            env.team_visibility_status,
        ))
    assert statuses[0] == (True, False, "visible")
    assert statuses[1:4] == [
        (False, True, "measurement_blackout"),
        (False, True, "measurement_blackout"),
        (False, True, "measurement_blackout"),
    ]
    assert statuses[4] == (True, False, "visible")
    assert env.stats["loss_steps_measurement_blackout"] == 3
    assert env.stats["loss_events_measurement_blackout"] == 1
    assert env.stats["agent_loss_checks_measurement_blackout"] == 9


def test_truth_reward_reference_prevents_visibility_source_switches():
    env = PursuitLostEnv(EnvConfig(
        seed=20260818,
        n_static_obs=0,
        n_dynamic_obs=0,
        sense_range=200.0,
        fov_half=math.pi,
        visual_occlusion=False,
        require_initial_visibility=True,
        measurement_blackout_start_step=1,
        measurement_blackout_duration_steps=2,
        reward_target_reference="truth",
    ))
    env.reset(seed=20260818)
    for _ in range(4):
        _, _, _, _, info = env.step(np.zeros((env.n, 2)))
        assert info["reward_target_reference"] == "truth"
        assert info["reward_components"]["reference_switch"] == 0
