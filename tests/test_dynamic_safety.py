# -*- coding: utf-8 -*-
import numpy as np

from envs.pursuit_lost_env import EnvConfig, PursuitLostEnv
from safety.cbf_qp import CBFSafetyLayer
from safety.factory import build_safety_layer
from train_baseline import apply_safety


def _controlled_env():
    cfg = EnvConfig(
        seed=17,
        n_static_obs=0,
        n_dynamic_obs=1,
        dyn_obs_speed=2.0,
        collision_terminates=True,
    )
    env = PursuitLostEnv(cfg)
    env.p_pos = np.array([[20.0, 50.0], [20.0, 20.0], [20.0, 80.0]])
    env.p_v = np.array([8.0, 0.0, 0.0])
    env.p_psi = np.zeros(3)
    env.obs_pos = np.array([[32.0, 50.0]])
    env.obs_r = np.array([2.0])
    env.obs_vel = np.array([[-2.0, 0.0]])
    return env


def test_safety_factory_keeps_legacy_default():
    cfg = EnvConfig(seed=2)
    assert isinstance(build_safety_layer(cfg), CBFSafetyLayer)
    assert isinstance(build_safety_layer(cfg, "legacy"), CBFSafetyLayer)


def test_collision_source_classification_is_additive():
    env = _controlled_env()
    env.obs_pos = np.array([[20.0, 50.0]])
    env.obs_vel = np.zeros((1, 2))
    _, _, _, done, info = env.step(np.zeros((3, 2)))
    assert done
    assert info["collision"]
    assert info["collision_dynamic"]
    assert not info["collision_static"]
    assert not info["collision_agent"]
    assert info["stats"]["collision_dynamic_steps"] == 1


def test_static_and_agent_collision_sources_are_separate():
    cfg = EnvConfig(
        seed=23,
        n_static_obs=1,
        n_dynamic_obs=0,
        collision_terminates=True,
    )
    static_env = PursuitLostEnv(cfg)
    static_env.p_pos = np.array([
        [20.0, 50.0], [20.0, 20.0], [20.0, 80.0]])
    static_env.p_v[:] = 0.0
    static_env.obs_pos = np.array([[20.0, 50.0]])
    static_env.obs_r = np.array([2.0])
    static_env.obs_vel = np.zeros((1, 2))
    _, _, _, _, static_info = static_env.step(np.zeros((3, 2)))
    assert static_info["collision_static"]
    assert not static_info["collision_dynamic"]
    assert not static_info["collision_agent"]

    agent_env = PursuitLostEnv(EnvConfig(
        seed=29,
        n_static_obs=0,
        n_dynamic_obs=0,
        collision_terminates=True,
    ))
    agent_env.p_pos = np.array([
        [20.0, 50.0], [20.5, 50.0], [20.0, 80.0]])
    agent_env.p_v[:] = 0.0
    _, _, _, _, agent_info = agent_env.step(np.zeros((3, 2)))
    assert agent_info["collision_agent"]
    assert not agent_info["collision_static"]
    assert not agent_info["collision_dynamic"]


def test_dynamic_v2_predicts_and_avoids_head_on_obstacle():
    env = _controlled_env()
    layer = build_safety_layer(
        env.cfg,
        "dynamic_v2",
        {"horizon_s": 1.5, "safety_buffer": 0.5},
    )
    nominal = np.zeros((env.n, 2))
    executed, diagnostics = apply_safety(
        env, nominal, layer, return_diagnostics=True)
    assert diagnostics["corrected"]
    assert diagnostics["nominal_min_clearance"] < 0
    assert diagnostics["output_min_clearance"] >= -1.0e-6
    assert diagnostics["output_next_step_physical_clearance"] >= 0.0
    assert np.isfinite(executed).all()
    assert (np.abs(executed) <= 1.0).all()
    for _ in range(5):
        _, _, _, done, info = env.step(executed)
        assert not info["collision"]
        assert not done


def test_dynamic_v2_no_threat_is_identity():
    env = _controlled_env()
    env.obs_pos = np.array([[90.0, 50.0]])
    env.obs_vel = np.zeros((1, 2))
    layer = build_safety_layer(env.cfg, "dynamic_v2")
    nominal = np.array([[0.2, 0.1], [0.0, -0.2], [-0.1, 0.0]])
    executed, diagnostics = apply_safety(
        env, nominal, layer, return_diagnostics=True)
    assert np.allclose(executed, nominal)
    assert not diagnostics["corrected"]
    assert layer.summary()["qp_attempts"] == 0
    assert layer.summary()["output_next_step_physical_unsafe"] == 0


def test_dynamic_v2_emergency_prioritizes_executed_step_safety(monkeypatch):
    env = _controlled_env()
    layer = build_safety_layer(
        env.cfg,
        "dynamic_v2",
        {
            "horizon_s": 1.5,
            "safety_buffer": 0.5,
            "emergency_ranking": "safe_prefix",
        },
    )
    candidates = np.zeros((3, env.n, 2), dtype=float)
    candidates[:, 0, 0] = [-1.0, 0.0, 1.0]
    monkeypatch.setattr(
        layer, "_fallback_candidates", lambda _n: candidates)
    monkeypatch.setattr(
        layer,
        "_batch_clearance_metrics",
        lambda *_args: (
            np.array([0.5, -0.2, 0.4]),   # horizon-buffer score
            np.array([-0.1, 0.0, 0.1]),   # next-buffer safety
            np.array([0.4, 0.5, 0.3]),    # next-physical safety
            np.array([0, 3, 5]),           # safe physical prefix steps
        ),
    )
    action, _, tier = layer._emergency_action(
        env,
        np.zeros((env.n, 2), dtype=float),
        layer._predict_obstacles(env),
        None,
    )
    assert tier == "next_buffer_safe"
    assert np.array_equal(action, candidates[2])


def test_dynamic_v2_default_retains_formal_revision_2_2():
    env = _controlled_env()
    layer = build_safety_layer(env.cfg, "dynamic_v2")
    assert layer.cfg.emergency_ranking == "horizon"
    assert layer.summary()["safety_revision"] == "2.2"
