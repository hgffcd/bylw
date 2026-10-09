# -*- coding: utf-8 -*-
"""ModeSwitchEvader 单元测试 (论文A S-A4): pytest tests -q"""
import hashlib
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.evaders import (MODE_CT, MODE_CV, MODE_STOP, ModeSwitchEvader,
                          apf_obstacle_wall_force, apf_pursuer_force)
from envs.pursuit_lost_env import EnvConfig, PursuitLostEnv

# 改动前基线哈希 (2026-07-27 生成, numpy 2.2.6, 本机):
# 同 seed 下 reset(seed) 后, 固定动作序列(rng 123)跑 50 步,
# 每步记录 [t_pos, t_vel] 的 float64 轨迹字节 sha1.
_APF_BASELINE = {
    7: "cfad12ae1617cfc05a67219907504f07d040685f",
    11: "3640d028966bd20739143f0737febf600015c218",
}

_MODE_SWITCH_BASELINES = {
    # The first digest is the frozen Windows/libm result from the original
    # workstation.  The second is the verified Linux/glibc result from the
    # server migration bundle.  Both remain exact, bitwise regression guards.
    13: {
        "16a506f7bc339ba5977d6a5009889212dc5d6389c45a4f43325bda89999b1130",
        "381a77ccb08c95e0d88a9cf617884a61728591a86da737635281349c46522f0e",
    },
    42: {
        "f7c0e3460c58109f6191e4ff3c41a8e4ed2b81af9c8e9c5fb5c89fe25676a2b9",
        "eabfe43062a1ee27ae394ec533775110f659085e60bdd6f90e9814802a0a9cff",
    },
}


def _apf_traj_hash(env_seed: int, act_seed: int = 123, steps: int = 50) -> str:
    env = PursuitLostEnv(EnvConfig(seed=env_seed))
    env.reset(seed=env_seed)
    arng = np.random.default_rng(act_seed)
    traj = []
    for _ in range(steps):
        act = arng.uniform(-1, 1, (env.n, 2))
        _, _, _, done, _ = env.step(act)
        traj.append(np.concatenate([env.t_pos, env.t_vel]))
        if done:
            break
    return hashlib.sha1(np.asarray(traj, np.float64).tobytes()).hexdigest()


def test_default_apf_bitwise_unchanged():
    """默认 evader_type='apf' 的目标轨迹与接入 dispatch 前逐位一致."""
    for env_seed, expect in _APF_BASELINE.items():
        assert _apf_traj_hash(env_seed) == expect, \
            f"seed={env_seed}: 默认 APF 逃逸者轨迹与改动前基线不一致"


def test_frozen_mode_switch_bitwise_unchanged():
    """新增反应式分支不得改变冻结ModeSwitch轨迹。"""
    for env_seed, expected in _MODE_SWITCH_BASELINES.items():
        env = _make_ms_env(env_seed)
        env.reset(seed=env_seed)
        arng = np.random.default_rng(707)
        traj = []
        for _ in range(120):
            env.step(arng.uniform(-1, 1, (env.n, 2)))
            traj.append(np.concatenate([env.t_pos, env.t_vel]))
        got = hashlib.sha256(
            np.asarray(traj, np.float64).tobytes()).hexdigest()
        assert got in expected, f"seed={env_seed}: 冻结ModeSwitch轨迹变化"


def test_mode_dynamics_unit():
    """三模式意图动力学: CV 保向, CT 定转率, STOP 指数衰减到 0 (无障碍空场)."""
    no_obs = (np.zeros((0, 2)), np.zeros(0))
    pos = np.array([50.0, 50.0])
    # CV: 航向不变, 速度=v_max
    ev = ModeSwitchEvader(dt=0.1, v_max=7.2, arena_size=100.0, seed=0)
    ev.mode, ev._dwell_left, ev.heading = MODE_CV, 10 ** 6, 0.3
    h0 = ev.heading
    for _ in range(20):
        _, v = ev.step(pos, *no_obs)
    assert ev.heading == h0
    assert abs(np.linalg.norm(v) - 7.2) < 1e-9
    assert abs(math.atan2(v[1], v[0]) - h0) < 1e-9
    # CT: 每步转 ct_dir * turn_rate * dt
    ev = ModeSwitchEvader(dt=0.1, v_max=7.2, arena_size=100.0, turn_rate=0.3, seed=1)
    ev.mode, ev._dwell_left, ev.heading, ev.ct_dir = MODE_CT, 10 ** 6, 0.0, 1.0
    for k in range(30):
        ev.step(pos, *no_obs)
    assert abs(ev.heading - 30 * 0.3 * 0.1) < 1e-9
    # STOP: 速度指数衰减到 0
    ev = ModeSwitchEvader(dt=0.1, v_max=7.2, arena_size=100.0, seed=2)
    ev.mode, ev._dwell_left = MODE_STOP, 10 ** 6
    speeds = []
    for _ in range(40):
        _, v = ev.step(pos, *no_obs)
        speeds.append(np.linalg.norm(v))
    assert all(b <= a + 1e-12 for a, b in zip(speeds, speeds[1:])), "急停速度应单调不增"
    assert speeds[-1] < 1e-3, "4 s 内应衰减到 ~0"
    # 空场处 APF 修正力应为 0
    assert np.allclose(apf_obstacle_wall_force(pos, *no_obs, 100.0), 0.0)


def test_reactive_mode_switch_repels_pursuer():
    """STOP意图为零时，ReactiveModeSwitch仍应沿远离追捕者方向应急移动。"""
    pos = np.array([50.0, 50.0])
    pursuers = np.array([[45.0, 50.0]])
    no_obs = (np.zeros((0, 2)), np.zeros(0))
    force = apf_pursuer_force(pos, pursuers, reaction_range=30.0)
    assert force[0] > 0 and abs(force[1]) < 1e-12

    plain = ModeSwitchEvader(
        dt=0.1, v_max=4.8, arena_size=100.0, seed=1)
    plain.mode, plain._dwell_left, plain.speed = MODE_STOP, 10 ** 6, 0.0
    _, v_plain = plain.step(pos, *no_obs, pursuer_pos=pursuers)
    assert np.allclose(v_plain, 0.0)

    reactive = ModeSwitchEvader(
        dt=0.1, v_max=4.8, arena_size=100.0, seed=1,
        pursuer_repulsion_gain=1.0, pursuer_reaction_range=30.0)
    reactive.mode, reactive._dwell_left, reactive.speed = MODE_STOP, 10 ** 6, 0.0
    _, v_reactive = reactive.step(pos, *no_obs, pursuer_pos=pursuers)
    assert v_reactive[0] > 0 and abs(v_reactive[1]) < 1e-12


def test_reactive_mode_switch_env_is_finite_and_reports_modes():
    cfg = EnvConfig(
        seed=17, evader_type="reactive_mode_switch",
        evader_pursuer_repulsion_gain=1.0,
        evader_pursuer_reaction_range=30.0,
        lost_cap_steps=10 ** 6, collision_terminates=False)
    env = PursuitLostEnv(cfg)
    obs, state = env.reset(seed=17)
    assert np.isfinite(obs).all() and np.isfinite(state).all()
    for _ in range(80):
        obs, state, rew, _, info = env.step(np.zeros((env.n, 2)))
        assert np.isfinite(obs).all() and np.isfinite(state).all()
        assert np.isfinite(rew).all()
    assert "evader_mode_seq" in info["stats"]


def _make_ms_env(seed: int) -> PursuitLostEnv:
    cfg = EnvConfig(seed=seed, evader_type="mode_switch",
                    lost_cap_steps=10 ** 6, collision_terminates=False)
    return PursuitLostEnv(cfg)


def test_mode_switch_rollout():
    """mode_switch 回合: 三模式都出现/停留时长在配置范围/切换统计正确/
    不穿障碍不出界/观测奖励全有限."""
    env = _make_ms_env(seed=42)
    env.reset(seed=42)
    arng = np.random.default_rng(7)
    info = None
    for _ in range(450):
        obs, state, rew, done, info = env.step(arng.uniform(-1, 1, (env.n, 2)))
        assert np.isfinite(obs).all() and np.isfinite(state).all() \
            and np.isfinite(rew).all()
        # 不出界(env 已 clip 到 [0.5, L-0.5])
        assert (env.t_pos >= 0.5 - 1e-9).all() \
            and (env.t_pos <= env.cfg.arena_size - 0.5 + 1e-9).all()
        # 目标中心不进入任何障碍圆
        if len(env.obs_pos):
            d = np.linalg.norm(env.obs_pos - env.t_pos, axis=1) - env.obs_r
            assert d.min() > 0.0, f"目标穿障碍: 侵入 {d.min():.3f} m"
        if done:
            break
    # 三模式都出现
    counts = env._mode_evader.mode_step_counts()
    assert all(v > 0 for v in counts.values()), f"三模式未全出现: {counts}"
    # 停留时长在配置范围内 (dwell (2,4) s, dt 0.1 -> [20, 40] 步; 末段可被截断)
    seq = info["stats"]["evader_mode_seq"]
    segs = re.findall(r"([VTS])(\d+)", seq)
    assert len(segs) >= 4
    for _, s in segs[:-1]:
        assert 20 <= int(s) <= 40, f"完整段停留 {s} 步超出 [20,40]: {seq}"
    assert int(segs[-1][1]) <= 40
    # 切换统计正确: 切换次数 = 段数 - 1
    assert info["stats"]["evader_mode_switches"] == len(segs) - 1
    # 模式字符合法且相邻段模式不同(默认转移矩阵无自环)
    for (m1, _), (m2, _) in zip(segs, segs[1:]):
        assert m1 != m2


def test_mode_switch_reset_reinit():
    """reset 后逃逸者重新初始化: 统计清零、序列重开; 同 seed 复位轨迹可复现."""
    env = _make_ms_env(seed=5)
    env.reset(seed=5)
    for _ in range(60):
        env.step(np.zeros((env.n, 2)))
    assert env.stats["evader_mode_switches"] >= 1
    env.reset(seed=99)
    _, _, _, _, info = env.step(np.zeros((env.n, 2)))
    assert info["stats"]["evader_mode_switches"] == 0
    assert len(re.findall(r"[VTS]\d+", info["stats"]["evader_mode_seq"])) == 1
    # 同 seed 全复位 -> 逃逸者派生种子相同 -> 轨迹可复现
    def _roll(s):
        e = _make_ms_env(seed=s)
        e.reset(seed=s)
        tr = []
        for _ in range(80):
            e.step(np.zeros((e.n, 2)))
            tr.append(e.t_pos.copy())
        return np.asarray(tr)
    assert np.array_equal(_roll(13), _roll(13))
