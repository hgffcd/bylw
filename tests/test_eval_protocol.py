import csv

import numpy as np

from envs.pursuit_lost_env import EnvConfig
from scripts.eval_policy import (
    _capture_sensitivity_hits,
    _instantaneous_clearances,
    effective_training_eval_fields,
    managed_env_mismatches,
    run_block,
    write_mode_stats_csv,
)
from algos.networks import build_gaussian_actor
from envs.pursuit_lost_env import PursuitLostEnv
from scripts.summarize_e2_formal import read_mode_stats


def _managed_run(evader_type="mode_switch"):
    return {
        "env": {
            "obs_mode": "belief",
            "evader_type": evader_type,
            "sense_range": 200.0,
            "fov_half": 3.14159,
        },
        "train": {
            "anneal": {
                "fields": {
                    "sense_range": [200.0, 25.0],
                    "fov_half": [3.14159, 1.0472],
                }
            }
        },
        "eval_env": {"collision_terminates": True},
    }


def test_effective_training_eval_fields_applies_final_anneal_and_eval_override():
    fields = effective_training_eval_fields(_managed_run())
    assert fields["sense_range"] == 25.0
    assert fields["fov_half"] == 1.0472
    assert fields["collision_terminates"] is True


def test_managed_env_protocol_detects_wrong_evader_but_ignores_obs_mode():
    run_cfg = _managed_run()
    correct = vars(EnvConfig(
        obs_mode="zero",
        evader_type="mode_switch",
        sense_range=25.0,
        fov_half=1.0472,
        collision_terminates=True,
    ))
    assert managed_env_mismatches(run_cfg, correct) == []
    wrong = dict(correct, evader_type="apf")
    mismatches = managed_env_mismatches(run_cfg, wrong)
    assert [item[0] for item in mismatches] == ["evader_type"]


def test_write_mode_stats_csv_preserves_episode_trace(tmp_path):
    out = tmp_path / "mode.csv"
    rows = [{
        "eval_seed": 2,
        "episode": 7,
        "env_seed": 120007,
        "success": 1,
        "collision": 0,
        "long_lost": 0,
        "ep_len": 91,
        "max_lost": 12,
        "evader_mode_switches": 3,
        "evader_mode_seq": "V20|T40|S20|V11",
    }]
    write_mode_stats_csv(out, rows)
    with out.open(encoding="utf-8", newline="") as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == 1
    assert saved[0]["evader_mode_switches"] == "3"
    assert saved[0]["evader_mode_seq"] == "V20|T40|S20|V11"


def test_read_mode_stats_aggregates_occupancy_and_completed_dwells(tmp_path):
    out = tmp_path / "mode.csv"
    rows = [
        {
            "eval_seed": 0, "episode": 0, "env_seed": 100000,
            "success": 1, "collision": 0, "long_lost": 0,
            "ep_len": 60, "max_lost": 10, "evader_mode_switches": 2,
            "evader_mode_seq": "V20|T20|S20",
        },
        {
            "eval_seed": 0, "episode": 1, "env_seed": 100001,
            "success": 0, "collision": 0, "long_lost": 1,
            "ep_len": 50, "max_lost": 40, "evader_mode_switches": 1,
            "evader_mode_seq": "S30|V20",
        },
    ]
    write_mode_stats_csv(out, rows)
    stats = read_mode_stats(out, expected_rows=2)
    assert stats["mode_switches_mean"] == 1.5
    assert stats["mode_switches_per_100_steps"] == 300 / 110
    assert stats["episodes_all_three_modes"] == 1
    assert stats["cv_occupancy"] == 40 / 110
    assert stats["ct_occupancy"] == 20 / 110
    assert stats["stop_occupancy"] == 50 / 110
    assert stats["cv_completed_dwell_steps_mean"] == 20
    assert stats["stop_completed_dwell_steps_mean"] == 30


def test_capture_sensitivity_separates_interception_from_encirclement():
    env = PursuitLostEnv(EnvConfig(
        seed=8, n_static_obs=0, n_dynamic_obs=0))
    env.t_pos[:] = [50.0, 50.0]
    env.p_pos[:] = [
        [54.0, 50.0],
        [47.5, 54.330127],
        [47.5, 45.669873],
    ]
    hits = _capture_sensitivity_hits(env)
    assert hits["capture_r4_k1_step"]
    assert not hits["capture_r4_k2_step"]
    assert hits["capture_r5_k3_step"]
    assert hits["capture_r6_k3_step"]

    env.p_pos[:] = [[54.0, 50.0], [54.5, 50.5], [55.0, 49.5]]
    hits = _capture_sensitivity_hits(env)
    assert hits["capture_r5_k2_step"]
    assert not hits["capture_r5_k3_step"]


def test_instantaneous_clearances_use_environment_collision_geometry():
    env = PursuitLostEnv(EnvConfig(
        seed=9, n_static_obs=1, n_dynamic_obs=1, pursuer_radius=0.5))
    env.p_pos[:] = [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]
    env.obs_pos = np.asarray([[10.0, 0.0], [0.0, 10.0]], float)
    env.obs_r = np.asarray([2.0, 3.0], float)
    env.obs_vel = np.zeros((2, 2), float)
    values = _instantaneous_clearances(env)
    assert values["min_agent_center_distance"] == 3.0
    assert values["min_agent_clearance"] == 2.0
    assert values["min_static_obstacle_clearance"] == 4.5
    assert values["min_dynamic_obstacle_clearance"] == 2.5
    assert values["min_obstacle_clearance"] == 2.5


def test_step_telemetry_contains_safety_and_physical_collision_audit():
    fields = {
        "n_static_obs": 0,
        "n_dynamic_obs": 0,
        "obs_mode": "zero",
        "sense_range": 200.0,
        "fov_half": 3.14159265,
        "capture_radius": 0.1,
        "max_steps": 1,
        "lost_cap_steps": 10,
    }
    probe = PursuitLostEnv(EnvConfig(**fields))
    actor = build_gaussian_actor(
        probe.obs_dim, probe.act_dim, hidden=16, n_agent=probe.n)
    result = run_block(
        actor,
        fields,
        episodes=1,
        seed_base=910,
        use_cbf=True,
        collect=True,
        safety_version="dynamic_v2",
        safety_params={"horizon_s": 1.5, "safety_buffer": 0.5},
        collect_steps=True,
    )
    step = result["steps"][0]
    assert step["safety_called"] == 1
    assert step["safety_calls_delta"] == 1
    assert "safety_solver_status" in step
    assert "min_agent_clearance" in step
    assert "collision_agent" in step
    episode = result["safety_episodes"][0]
    assert "min_agent_center_distance" in episode
    assert "min_static_obstacle_clearance" in episode
